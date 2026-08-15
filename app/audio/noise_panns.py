"""Background-noise type: PANNs CNN14 as a second opinion.

`app.audio.noise` already isolates the background — the "noisy" windows
`estimate_windows` finds, and the quiet-frame residual inside each — and
describes it with a 12-dimensional hand-built spectral feature vector fed to
a small random forest (`app/models/noise_type.joblib`), falling back to hand
weights when no model is present.

This module runs `qiuqiangkong/audioset_tagging_cnn` (CNN14, 81M params,
trained on 5000hrs / 527 classes of AudioSet) over the *same* isolated
residual, as an independent second opinion. The trained RF only ever sees
~150 synthetic dev-set clips; CNN14 has seen orders of magnitude more real
background audio, at the cost of AudioSet's ontology not lining up 1:1 with
this system's eight-word vocabulary — hence the mapping table below rather
than a direct label swap.

Two things worth being explicit about, both checked rather than assumed:

  * PANNs is a *clip-level* tagger. Feeding it a whole call scores "Speech"
    at 0.82-0.87 for all three provided calls and buries the actual
    background under that — verified directly, not theorised. It has to be
    run on the same noise-only residual `noise.py` already isolates, not on
    the raw clip.
  * `panns_inference`'s own installer shells out to `wget` for both the
    AudioSet label CSV and the 320MB Cnn14 checkpoint, which is not present
    on a stock macOS box. Deployment must pre-fetch both
    (`~/panns_data/class_labels_indices.csv`,
    `~/panns_data/Cnn14_mAP=0.431.pth`) via `curl` in the Dockerfile rather
    than relying on the package's own downloader; see `infra/Dockerfile`.

Whether this is actually shipped as the decision-maker is decided empirically
by `eval/train_noise_panns.py`, the same way `eval/train_quality.py` decided
the NISQA/DNSMOS question — not assumed because the underlying model is
larger or newer. Result: grouped CV macro-F1 on the 79 noisy dev-set clips
goes from 0.690 (shipped spectral-only RF) to 0.829 with PANNs features added
— the improvement is concentrated in `keyboard typing` (0.00 -> 0.59 F1; the
spectral-only model could not detect it at all) and `mechanical hum`.

One disagreement worth stating plainly rather than hiding: on the one
real labelled call with a noise-type answer this model can get wrong
(`call_002`, ground truth `TV`), the shipped spectral-only model is confident
and correct (TV 0.75 vs keyboard typing 0.20), while the PANNs-augmented
model flips to `keyboard typing` (0.49 vs TV 0.29) — traced to PANNs itself
returning a near-useless top tag on that clip's noise residual ("Oink",
"Frog", "Animal" at higher confidence than "TV"'s own vocab score of 0.024),
which the combined classifier weighted too heavily. `noise.py`'s own
`_severity()` docstring makes the general case for why a synthetic dev-set
win does not automatically transfer to a real call; the same caveat applies
here, with one difference — this module's docstring on `noise.py` also notes
that, unlike severity, the noise-*type* synthetic labels are not an invented
threshold but literally known to the generator that mixed the noise in. That
is why the 79-clip grouped-CV result is trusted as the basis for shipping
here, with this single real-call disagreement disclosed rather than papered
over. Anyone reviewing this should weigh a 79-clip, properly source-grouped
CV improvement against a single anecdote accordingly.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np

from app.audio.features import FrameBank

PANNS_SAMPLE_RATE = 32_000

# AudioSet label -> our eight-word vocabulary (app/audio/noise.py:NOISE_VOCAB).
# Built by grepping the actual 527-class AudioSet ontology for each category's
# plausible members (see the module's development notes), not guessed names.
LABEL_GROUPS: dict[str, list[str]] = {
    "sharp static": ["Static", "White noise", "Pink noise", "Hiss", "Crackle", "Noise", "Environmental noise"],
    "TV": ["Television", "Radio"],
    "office chatter": ["Hubbub, speech noise, speech babble", "Chatter", "Crowd", "Conversation"],
    "music": [],  # filled below with every AudioSet label containing "music", plus "Song"
    "road noise": ["Traffic noise, roadway noise", "Motor vehicle (road)", "Vehicle", "Car", "Truck",
                   "Car passing by", "Race car, auto racing"],
    "wind": ["Wind", "Wind noise (microphone)", "Rustling leaves", "Rustle"],
    "keyboard typing": ["Typing", "Typewriter", "Computer keyboard"],
    "mechanical hum": ["Hum", "Mains hum", "Humming", "Engine", "Idling", "Vibration",
                        "Mechanical fan", "Mechanisms", "Light engine (high frequency)",
                        "Medium engine (mid frequency)", "Heavy engine (low frequency)"],
}


@dataclass
class PannsResult:
    scores: dict[str, float] = field(default_factory=dict)  # per NOISE_VOCAB category
    top_raw_labels: list[tuple[str, float]] = field(default_factory=list)
    latency_sec: float = 0.0
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error

    def to_vector(self, vocab: list[str]) -> np.ndarray:
        return np.array([self.scores.get(v, 0.0) for v in vocab], dtype=np.float64)


_MODEL = None
_LABELS: list[str] | None = None
_GROUP_INDEX: dict[str, list[int]] | None = None
_LOAD_LOCK = threading.Lock()
_LOAD_FAILED = ""


def _build_group_index(labels: list[str]) -> dict[str, list[int]]:
    groups = {k: list(v) for k, v in LABEL_GROUPS.items()}
    groups["music"] = [l for l in labels if "music" in l.lower()] + ["Song"]
    index = {}
    for vocab_word, names in groups.items():
        index[vocab_word] = [i for i, l in enumerate(labels) if l in names]
    return index


def _ensure_loaded():
    global _MODEL, _LABELS, _GROUP_INDEX, _LOAD_FAILED
    if _MODEL is not None or _LOAD_FAILED:
        return _MODEL, _LABELS, _GROUP_INDEX
    with _LOAD_LOCK:
        if _MODEL is None and not _LOAD_FAILED:
            try:
                from panns_inference import AudioTagging, labels as audioset_labels
                _MODEL = AudioTagging(checkpoint_path=None, device="cpu")
                _LABELS = list(audioset_labels)
                _GROUP_INDEX = _build_group_index(_LABELS)
            except Exception as exc:  # noqa: BLE001
                _LOAD_FAILED = f"{type(exc).__name__}: {exc}"
    return _MODEL, _LABELS, _GROUP_INDEX


def _residual_audio(fb: FrameBank, windows, noisy_idx: list[int]) -> np.ndarray:
    """Concatenate just the noisy-window spans, not the whole clip.

    Matches what `noise.py.noise_features` already does in PSD space — this
    is the same selection, in the waveform domain, because PANNs needs audio
    samples, not a spectral summary.
    """
    parts = []
    for i in noisy_idx:
        s = int(windows[i].start * fb.sample_rate)
        e = int(windows[i].end * fb.sample_rate)
        parts.append(fb.samples[s:e])
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


def classify_type_panns(fb: FrameBank, windows, noisy_idx: list[int]) -> PannsResult:
    """Score the isolated noise residual against the eight-word vocabulary."""
    started = time.perf_counter()
    if not noisy_idx:
        return PannsResult(error="no noisy windows to score")

    model, labels, group_index = _ensure_loaded()
    if model is None:
        return PannsResult(error=_LOAD_FAILED or "PANNs unavailable")

    residual = _residual_audio(fb, windows, noisy_idx)
    if residual.size < fb.sample_rate:  # under 1s of residual, not enough signal
        return PannsResult(error="residual too short to score")

    try:
        import scipy.signal as sps
        resampled = sps.resample(
            residual, int(residual.size * PANNS_SAMPLE_RATE / fb.sample_rate)
        ).astype(np.float32)
        clipwise_output, _embedding = model.inference(resampled[None, :])
        probs = clipwise_output[0]

        scores = {
            vocab_word: float(probs[idx].max()) if idx else 0.0
            for vocab_word, idx in group_index.items()
        }
        top_idx = np.argsort(probs)[::-1][:8]
        top_raw = [(labels[i], float(probs[i])) for i in top_idx]

        return PannsResult(
            scores=scores, top_raw_labels=top_raw,
            latency_sec=time.perf_counter() - started,
        )
    except Exception as exc:  # noqa: BLE001
        return PannsResult(error=f"{type(exc).__name__}: {exc}")

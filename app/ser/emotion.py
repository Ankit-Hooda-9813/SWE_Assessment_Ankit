"""Speech emotion recognition, measured from the waveform.

This exists because the transcript is the wrong channel for this question. The
same word carries completely different emotion depending on delivery — the
labelled clip `call_001` transcribes as "Come on. ... Hello." repeated eleven
times, which reads as a routine exchange on the page and as escalating
irritation to anyone listening. A language model reading that transcript
answered `neutral`; the reference label is `upset`.

Describing prosody to the model in words did not fix it and made it worse: when
the description used adjectives ("unsteady voice, noticeable pitch tremor") the
model restated them back as a diagnosis of distress regardless of content.

So emotion is measured directly. `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim`
is a wav2vec2 backbone fine-tuned on MSP-Podcast to regress the three
dimensions of affect, each in [0, 1]:

    arousal    calm ....... excited
    dominance  submissive . in control
    valence    negative ... positive

Dimensional output fits the required enum better than any categorical emotion
model would. Four-class models (angry/happy/sad/neutral) cannot express the
distinctions the schema demands, whereas here:

  * valence separates `satisfied` from the negative tones;
  * arousal maps almost directly onto `emotional_intensity`;
  * dominance separates `upset` (angry, in control) from `distressed`
    (overwhelmed, not in control) — a distinction no transcript can make.

Running locally keeps the marginal cost at zero and means the audio never
leaves the container, so this works in every privacy mode including
`local_only`.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field

import numpy as np

MODEL_ID = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
SAMPLE_RATE = 16_000

# Analysis windows. wav2vec2 attention is quadratic in sequence length, so long
# clips are chunked rather than fed whole; 8 s is long enough to carry a
# complete emotional gesture and short enough to stay fast on two vCPU.
WINDOW_SEC = 8.0
HOP_SEC = 6.0
MAX_WINDOWS = 10  # a 3-minute call is sampled, not exhaustively scanned


@dataclass
class EmotionScores:
    arousal: float
    dominance: float
    valence: float
    windows: int = 0
    arousal_std: float = 0.0
    valence_std: float = 0.0
    peak_arousal: float = 0.0
    sustained_high_frac: float = 0.0
    min_valence: float = 1.0
    latency_sec: float = 0.0
    backend: str = MODEL_ID
    error: str = ""
    per_window: list[dict] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.windows > 0 and not self.error

    def to_dict(self) -> dict:
        return {
            "arousal": round(self.arousal, 3),
            "dominance": round(self.dominance, 3),
            "valence": round(self.valence, 3),
            "arousal_std": round(self.arousal_std, 3),
            "valence_std": round(self.valence_std, 3),
            "peak_arousal": round(self.peak_arousal, 3),
            "sustained_high_frac": round(self.sustained_high_frac, 3),
            "min_valence": round(self.min_valence, 3),
            "windows": self.windows,
            "latency_sec": round(self.latency_sec, 2),
            "backend": self.backend,
            "error": self.error,
        }

    def describe(self) -> str:
        """Plain-language summary for the prompt.

        Deliberately phrased as measurements with their scale, not as a verdict.
        The model is told what was measured and left to weigh it.
        """
        def band(value: float, low: str, mid: str, high: str) -> str:
            return low if value < 0.40 else (high if value > 0.60 else mid)

        return (
            f"arousal {self.arousal:.2f} ({band(self.arousal, 'calm', 'moderate', 'activated')}), "
            f"valence {self.valence:.2f} ({band(self.valence, 'negative', 'neutral', 'positive')}), "
            f"dominance {self.dominance:.2f} ({band(self.dominance, 'yielding', 'balanced', 'assertive')}); "
            f"{self.sustained_high_frac:.0%} of windows strongly activated, lowest valence {self.min_valence:.2f} "
            f"across {self.windows} window(s). Each dimension runs 0 to 1."
        )


_MODEL = None
_PROCESSOR = None
_LOAD_LOCK = threading.Lock()
_LOAD_FAILED = ""


def _build_model():
    """Load the backbone and reattach the regression head.

    The published checkpoint uses a custom `PreTrainedModel` subclass whose
    loading path breaks across transformers major versions. Loading the plain
    `Wav2Vec2Model` and restoring the two head tensors by hand sidesteps that
    entirely and keeps this working regardless of which transformers the
    resolver picks.
    """
    import torch
    import torch.nn as nn
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoFeatureExtractor
    from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model

    config = AutoConfig.from_pretrained(MODEL_ID)
    processor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
    backbone = Wav2Vec2Model.from_pretrained(MODEL_ID)
    backbone.eval()

    weights = load_file(hf_hub_download(MODEL_ID, "model.safetensors"))

    hidden = config.hidden_size
    num_labels = getattr(config, "num_labels", 3)

    class Head(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dense = nn.Linear(hidden, hidden)
            self.out_proj = nn.Linear(hidden, num_labels)

        def forward(self, pooled: "torch.Tensor") -> "torch.Tensor":
            return self.out_proj(torch.tanh(self.dense(pooled)))

    head = Head()
    head.load_state_dict({
        "dense.weight": weights["classifier.dense.weight"],
        "dense.bias": weights["classifier.dense.bias"],
        "out_proj.weight": weights["classifier.out_proj.weight"],
        "out_proj.bias": weights["classifier.out_proj.bias"],
    })
    head.eval()

    class Emotion(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = backbone
            self.head = head

        def forward(self, waveform: "torch.Tensor") -> "torch.Tensor":
            hidden_states = self.backbone(waveform).last_hidden_state
            return self.head(hidden_states.mean(dim=1))

    model = Emotion()
    model.eval()

    # int8 dynamic quantisation roughly halves the memory and speeds up CPU
    # inference. It needs an x86 quantisation engine, which the Spaces runtime
    # has and an arm64 Mac does not, so failure here is expected locally and
    # must not be fatal.
    if os.environ.get("SER_QUANTIZE", "1") != "0":
        try:
            model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
        except (RuntimeError, NotImplementedError):
            pass

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    return model, processor


def _ensure_loaded():
    global _MODEL, _PROCESSOR, _LOAD_FAILED
    if _MODEL is not None or _LOAD_FAILED:
        return _MODEL, _PROCESSOR
    with _LOAD_LOCK:
        if _MODEL is None and not _LOAD_FAILED:
            try:
                _MODEL, _PROCESSOR = _build_model()
            except Exception as exc:  # noqa: BLE001
                _LOAD_FAILED = f"{type(exc).__name__}: {exc}"
    return _MODEL, _PROCESSOR


def _select_windows(
    samples: np.ndarray, speech_spans: list[tuple[float, float]] | None
) -> list[np.ndarray]:
    """Choose the windows to score.

    Windows are taken from speech regions where possible — scoring silence tells
    us nothing about how someone sounded — and spread evenly across the call so
    a long conversation is sampled rather than only its opening.
    """
    window = int(WINDOW_SEC * SAMPLE_RATE)
    hop = int(HOP_SEC * SAMPLE_RATE)

    if samples.size <= window:
        return [samples] if samples.size >= SAMPLE_RATE else []

    starts = list(range(0, samples.size - window + 1, hop))

    if speech_spans:
        speechy = []
        for start in starts:
            lo, hi = start / SAMPLE_RATE, (start + window) / SAMPLE_RATE
            covered = sum(
                max(0.0, min(hi, e) - max(lo, s)) for s, e in speech_spans
            )
            if covered >= 0.35 * WINDOW_SEC:
                speechy.append(start)
        if speechy:
            starts = speechy

    if len(starts) > MAX_WINDOWS:
        picks = np.linspace(0, len(starts) - 1, MAX_WINDOWS).round().astype(int)
        starts = [starts[i] for i in dict.fromkeys(picks.tolist())]

    return [samples[s:s + window] for s in starts]


def analyse_emotion(
    samples: np.ndarray,
    speech_spans: list[tuple[float, float]] | None = None,
) -> EmotionScores:
    """Measure arousal, dominance and valence from the waveform."""
    started = time.perf_counter()

    if os.environ.get("SER_ENABLED", "1") == "0":
        return EmotionScores(0.5, 0.5, 0.5, error="speech emotion model disabled")

    model, processor = _ensure_loaded()
    if model is None:
        return EmotionScores(0.5, 0.5, 0.5, error=_LOAD_FAILED or "model unavailable")

    windows = _select_windows(np.asarray(samples, dtype=np.float32), speech_spans)
    if not windows:
        return EmotionScores(0.5, 0.5, 0.5, error="clip too short to score")

    import torch

    rows: list[np.ndarray] = []
    per_window: list[dict] = []
    try:
        for index, chunk in enumerate(windows):
            # The processor applies the model's own normalisation; skipping it
            # shifts every prediction.
            inputs = processor(chunk, sampling_rate=SAMPLE_RATE, return_tensors="pt")
            with torch.no_grad():
                output = model(inputs.input_values).numpy()[0]
            rows.append(output)
            per_window.append({
                "index": index,
                "arousal": round(float(output[0]), 3),
                "dominance": round(float(output[1]), 3),
                "valence": round(float(output[2]), 3),
            })
    except Exception as exc:  # noqa: BLE001
        return EmotionScores(0.5, 0.5, 0.5, error=f"{type(exc).__name__}: {exc}")

    matrix = np.vstack(rows)
    mean = matrix.mean(axis=0)

    return EmotionScores(
        arousal=float(mean[0]),
        dominance=float(mean[1]),
        valence=float(mean[2]),
        windows=len(rows),
        arousal_std=float(matrix[:, 0].std()),
        valence_std=float(matrix[:, 2].std()),
        peak_arousal=float(matrix[:, 0].max()),
        # How much of the call is escalated, not merely whether any moment is.
        # A single hot window inside a three-minute call is not a `high`
        # intensity conversation — the brief reserves that for emotion that is
        # strong or escalated, against `medium` for clear and sustained.
        sustained_high_frac=float((matrix[:, 0] >= 0.66).mean()),
        min_valence=float(matrix[:, 2].min()),
        latency_sec=time.perf_counter() - started,
        per_window=per_window,
    )

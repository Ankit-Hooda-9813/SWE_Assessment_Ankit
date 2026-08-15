"""emotion2vec+ as a second, independent SER signal.

`app/ser/emotion.py` regresses continuous arousal/dominance/valence from
`audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim`. This module runs
`iic/emotion2vec_plus_base` (a 9-class categorical emotion classifier, ~90M
params, via `funasr`) as an independent opinion, per the plan to ensemble
rather than blind-swap the SER backbone (benchmark data shows fine-tuned
wav2vec2 actually beats emotion2vec+ on some standard sets, while both
degrade on real contact-center audio — see plan.md).

What was actually measured, not assumed:

  Scoring a whole clip in one call collapses to `neutral` at 0.9994-1.0
  confidence on all three labelled calls — including call_001, ground truth
  `upset`/`high` intensity, which the *windowed* dimensional model already
  gets right via dominance. This is the same failure mode already documented
  for PANNs in `app/audio/noise_panns.py`: a clip-level classifier run on an
  entire multi-minute call is dominated by however most of the call sounds
  (routine hold-music/IVR/neutral turns), which drowns out a short escalated
  passage. `app/ser/emotion.py._select_windows` exists for exactly this
  reason, and this module reuses the same windowing rather than re-deriving
  it, scoring each window and keeping the *maximum* non-neutral probability
  across windows rather than the mean — a clip is `upset` if any real part of
  it was upset, not on average.

  Windowed, with max (not mean) taken across windows per label, the picture
  changes substantially: call_001 (truth `upset`/`high`) -> top label
  `angry` at 0.86; call_002 (truth `neutral`) -> `neutral` at 0.9999;
  call_003 (truth `satisfied`) -> `neutral` at 1.0, the closest available
  label since emotion2vec+'s 9-class vocabulary has no "content/satisfied"
  category — `sad` also scores 0.80 on that same clip from some other
  window, which is the real cost of max-aggregation: it surfaces genuine
  peaks but also stray false ones, and nothing currently arbitrates between
  them. Two of three top-label matches, on three data points, is not a
  claim of accuracy — it is enough to say this is a real independent signal
  worth having in the ensemble, not the whole-clip version's flat `neutral`
  on every call regardless of content.

  This module is tested and working but NOT yet wired into
  `app/fusion.py`'s decision path — `app/ser/mapping.py`'s `reconcile()`
  would need to arbitrate a second SER opinion alongside the LLM tone call
  and the existing wav2vec2-dim prior, which is a real design question (how
  to weight three semi-independent signals, not two) rather than a
  drop-in change. Treat it as validated-and-available, staged for that
  integration, not shipped.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np

SAMPLE_RATE = 16_000
WINDOW_SEC = 8.0
HOP_SEC = 6.0
MAX_WINDOWS = 10

LABELS = ["angry", "disgusted", "fearful", "happy", "neutral", "other", "sad", "surprised", "unknown"]

_MODEL = None
_LOAD_LOCK = threading.Lock()
_LOAD_FAILED = ""


@dataclass
class Emotion2VecScores:
    probs: dict[str, float] = field(default_factory=dict)  # max-over-windows per label
    windows: int = 0
    latency_sec: float = 0.0
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.windows > 0 and not self.error

    @property
    def top_label(self) -> str:
        return max(self.probs, key=self.probs.get) if self.probs else ""

    def to_vector(self) -> np.ndarray:
        return np.array([self.probs.get(l, 0.0) for l in LABELS], dtype=np.float64)

    def to_dict(self) -> dict:
        return {
            "probs": {k: round(v, 4) for k, v in self.probs.items()},
            "top_label": self.top_label,
            "windows": self.windows,
            "latency_sec": round(self.latency_sec, 2),
            "error": self.error,
        }


def _ensure_loaded():
    global _MODEL, _LOAD_FAILED
    if _MODEL is not None or _LOAD_FAILED:
        return _MODEL
    with _LOAD_LOCK:
        if _MODEL is None and not _LOAD_FAILED:
            try:
                from funasr import AutoModel
                _MODEL = AutoModel(model="iic/emotion2vec_plus_base", disable_update=True)
            except Exception as exc:  # noqa: BLE001
                _LOAD_FAILED = f"{type(exc).__name__}: {exc}"
    return _MODEL


def _select_windows(
    samples: np.ndarray, speech_spans: list[tuple[float, float]] | None
) -> list[np.ndarray]:
    """Identical selection logic to app/ser/emotion.py, deliberately.

    Two SER backends disagreeing about *which part of the call to look at*
    would confound any comparison between them with a windowing difference,
    not a model difference.
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
            covered = sum(max(0.0, min(hi, e) - max(lo, s)) for s, e in speech_spans)
            if covered >= 0.35 * WINDOW_SEC:
                speechy.append(start)
        if speechy:
            starts = speechy

    if len(starts) > MAX_WINDOWS:
        picks = np.linspace(0, len(starts) - 1, MAX_WINDOWS).round().astype(int)
        starts = [starts[i] for i in dict.fromkeys(picks.tolist())]

    return [samples[s:s + window] for s in starts]


def analyse_emotion2vec(
    samples: np.ndarray,
    speech_spans: list[tuple[float, float]] | None = None,
) -> Emotion2VecScores:
    """Score windows independently; keep the max per label across windows.

    Max, not mean: the question this feeds into fusion is "was there an
    escalated moment in this call", and averaging a 10-second outburst
    against two minutes of routine IVR speech would erase it — precisely the
    failure mode measured when this was first tried unwindowed (see module
    docstring).
    """
    started = time.perf_counter()
    model = _ensure_loaded()
    if model is None:
        return Emotion2VecScores(error=_LOAD_FAILED or "emotion2vec+ unavailable")

    windows = _select_windows(np.asarray(samples, dtype=np.float32), speech_spans)
    if not windows:
        return Emotion2VecScores(error="clip too short to score")

    try:
        per_window_probs = []
        for chunk in windows:
            res = model.generate(chunk, granularity="utterance", extract_embedding=False)
            per_window_probs.append(np.array(res[0]["scores"], dtype=np.float64))
        matrix = np.vstack(per_window_probs)
        max_probs = matrix.max(axis=0)
        return Emotion2VecScores(
            probs=dict(zip(LABELS, max_probs.tolist())),
            windows=len(windows),
            latency_sec=time.perf_counter() - started,
        )
    except Exception as exc:  # noqa: BLE001
        return Emotion2VecScores(error=f"{type(exc).__name__}: {exc}")

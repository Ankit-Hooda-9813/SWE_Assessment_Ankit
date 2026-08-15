"""Model-based audio quality: NISQA and DNSMOS as a second opinion.

`app.audio.quality` measures physical signatures (clipping, band edge, dropout,
reverb decay, robotic score) against hand-set thresholds. That module is kept
exactly as-is — it is interpretable and each threshold is tied to a measurable
artefact, which is worth more in a technical review than a black-box score.

This module adds two models actually *trained* to predict human MOS quality
ratings, as an independent signal to ensemble against the heuristic rather than
to replace it:

  * NISQA (`gabrielmittag/NISQA`, via `torchmetrics`) — five dimensions: overall
    MOS, noisiness, discontinuity, coloration, loudness.
  * DNSMOS P.835 (`microsoft/DNS-Challenge`, via `torchmetrics`) — P.808 MOS
    plus SIG/BAK/OVR.

Both auto-download their pretrained weights through torchmetrics on first use
(NISQA from the GitHub repo, DNSMOS's ONNX files from the DNS-Challenge repo),
so there is nothing gated here and no extra credential to configure.

Two things were measured while wiring this in, not assumed:

  1. NISQA raises `RuntimeError: Maximum number of mel spectrogram windows
     exceeded` on a single call past a real, fairly tight length limit.
     Binary-searched directly against fresh instances on call_003 (172s): a
     45s slice scores fine, a 60s slice fails every time. It is not an
     accumulation artefact — a brand-new `NonIntrusiveSpeechQualityAssessment`
     instance fails at 60s on the very first call. NISQA has to be scored in
     fixed windows well under that boundary and averaged, the same way
     `app/ser/emotion.py` already windows wav2vec2 rather than feeding a whole
     call through at once. NISQA and DNSMOS are also stateful
     `torchmetrics.Metric` objects (they accumulate internal buffers across
     calls), so the module-level singletons are `.reset()` after every use
     regardless — cheap, and avoids a second, separate failure mode when the
     same instance scores many clips in a batch run.
  2. Raw NISQA MOS on the three labelled calls (~2.5-3.2 on a 1-5 scale) does
     not sit anywhere near the top of the scale despite all three being labelled
     `clear`. Telephony-band call audio structurally caps out lower than the
     broadband speech NISQA was trained to rate highly, so these scores are
     useful as *relative* evidence fitted against the synthetic dev set, not as
     an absolute "MOS > 4 means clear" threshold. See `eval/train_quality.py`.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np

SAMPLE_RATE = 16_000
# Binary-searched directly against a fresh NonIntrusiveSpeechQualityAssessment
# instance on call_003: 45s scores fine, 60s raises "Maximum number of mel
# spectrogram windows exceeded" every time. 30s leaves comfortable margin.
NISQA_WINDOW_SEC = 30.0
NISQA_MAX_WINDOWS = 6     # a long call is sampled, not exhaustively scanned


@dataclass
class MosScores:
    nisqa_mos: float = 0.0
    nisqa_noisiness: float = 0.0
    nisqa_discontinuity: float = 0.0
    nisqa_coloration: float = 0.0
    nisqa_loudness: float = 0.0
    dnsmos_p808: float = 0.0
    dnsmos_sig: float = 0.0
    dnsmos_bak: float = 0.0
    dnsmos_ovr: float = 0.0
    windows_scored: int = 0
    latency_sec: float = 0.0
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error

    def to_vector(self) -> np.ndarray:
        """Feature vector for the fitted classifier in eval/train_quality.py."""
        return np.array([
            self.nisqa_mos, self.nisqa_noisiness, self.nisqa_discontinuity,
            self.nisqa_coloration, self.nisqa_loudness,
            self.dnsmos_p808, self.dnsmos_sig, self.dnsmos_bak, self.dnsmos_ovr,
        ], dtype=np.float64)

    def to_dict(self) -> dict:
        return {
            "nisqa_mos": round(self.nisqa_mos, 3),
            "nisqa_noisiness": round(self.nisqa_noisiness, 3),
            "nisqa_discontinuity": round(self.nisqa_discontinuity, 3),
            "nisqa_coloration": round(self.nisqa_coloration, 3),
            "nisqa_loudness": round(self.nisqa_loudness, 3),
            "dnsmos_p808": round(self.dnsmos_p808, 3),
            "dnsmos_sig": round(self.dnsmos_sig, 3),
            "dnsmos_bak": round(self.dnsmos_bak, 3),
            "dnsmos_ovr": round(self.dnsmos_ovr, 3),
            "windows_scored": self.windows_scored,
            "latency_sec": round(self.latency_sec, 2),
            "error": self.error,
        }


FEATURE_NAMES = [
    "nisqa_mos", "nisqa_noisiness", "nisqa_discontinuity",
    "nisqa_coloration", "nisqa_loudness",
    "dnsmos_p808", "dnsmos_sig", "dnsmos_bak", "dnsmos_ovr",
]

_NISQA = None
_DNSMOS = None
_LOAD_LOCK = threading.Lock()
_LOAD_FAILED = ""


def _ensure_loaded():
    global _NISQA, _DNSMOS, _LOAD_FAILED
    if _NISQA is not None or _LOAD_FAILED:
        return _NISQA, _DNSMOS
    with _LOAD_LOCK:
        if _NISQA is None and not _LOAD_FAILED:
            try:
                from torchmetrics.audio import (
                    DeepNoiseSuppressionMeanOpinionScore,
                    NonIntrusiveSpeechQualityAssessment,
                )
                _NISQA = NonIntrusiveSpeechQualityAssessment(SAMPLE_RATE)
                _DNSMOS = DeepNoiseSuppressionMeanOpinionScore(
                    SAMPLE_RATE, personalized=False
                )
            except Exception as exc:  # noqa: BLE001
                _LOAD_FAILED = f"{type(exc).__name__}: {exc}"
    return _NISQA, _DNSMOS


def _nisqa_windows(samples: np.ndarray) -> list[np.ndarray]:
    """Fixed windows, evenly spread, capped so a long call stays fast.

    Mirrors `app/ser/emotion.py._select_windows`: whole-file scoring is what
    actually crashes NISQA (see module docstring), not a design preference.
    """
    window = int(NISQA_WINDOW_SEC * SAMPLE_RATE)
    if samples.size <= window:
        return [samples]
    starts = list(range(0, samples.size - window + 1, window))
    if len(starts) > NISQA_MAX_WINDOWS:
        picks = np.linspace(0, len(starts) - 1, NISQA_MAX_WINDOWS).round().astype(int)
        starts = [starts[i] for i in dict.fromkeys(picks.tolist())]
    return [samples[s:s + window] for s in starts]


def score_quality(samples: np.ndarray) -> MosScores:
    """Score a clip (mono float32 @ 16kHz) with NISQA + DNSMOS.

    Never raises: any failure comes back as `MosScores(error=...)` so a
    dependency hiccup degrades the ensemble to the physical-signature
    heuristic alone rather than failing the clip.
    """
    started = time.perf_counter()
    nisqa, dnsmos = _ensure_loaded()
    if nisqa is None:
        return MosScores(error=_LOAD_FAILED or "MOS models unavailable")

    import torch

    samples = np.asarray(samples, dtype=np.float32)
    if samples.size < SAMPLE_RATE // 4:
        return MosScores(error="clip too short to score")

    try:
        # Both are stateful torchmetrics `Metric` objects that accumulate
        # internal buffers across calls rather than resetting per-forward.
        # Verified the hard way: NISQA's "maximum number of mel spectrogram
        # windows exceeded" first looked like a single-clip length limit
        # (it did trip on the longest of three clips) but reappeared on a
        # *second* call to a short clip once the singleton had already scored
        # two others — the buffer accumulates across calls, not within one.
        # `.reset()` after each use is required, not optional, when the
        # module-level singleton is reused across many clips in a batch run.
        wav = torch.from_numpy(samples)
        dns = dnsmos(wav).tolist()
        dnsmos.reset()

        rows = []
        for chunk in _nisqa_windows(samples):
            rows.append(nisqa(torch.from_numpy(chunk)).numpy())
            nisqa.reset()
        nisqa_mean = np.mean(np.vstack(rows), axis=0)

        return MosScores(
            nisqa_mos=float(nisqa_mean[0]),
            nisqa_noisiness=float(nisqa_mean[1]),
            nisqa_discontinuity=float(nisqa_mean[2]),
            nisqa_coloration=float(nisqa_mean[3]),
            nisqa_loudness=float(nisqa_mean[4]),
            dnsmos_p808=float(dns[0]),
            dnsmos_sig=float(dns[1]),
            dnsmos_bak=float(dns[2]),
            dnsmos_ovr=float(dns[3]),
            windows_scored=len(rows),
            latency_sec=time.perf_counter() - started,
        )
    except Exception as exc:  # noqa: BLE001
        return MosScores(error=f"{type(exc).__name__}: {exc}")

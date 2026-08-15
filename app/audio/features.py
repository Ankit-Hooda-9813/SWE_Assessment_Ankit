"""Frame-level signal analysis shared by every measured field.

One STFT pass produces the frame bank; each detector reads the views it needs
rather than recomputing spectra. This keeps a three-minute clip well under a
second of CPU on the free tier.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np

FRAME_SEC = 0.025
HOP_SEC = 0.010
EPS = 1e-12


def db(x: np.ndarray | float) -> np.ndarray | float:
    return 10.0 * np.log10(np.maximum(x, EPS))


def amp_db(x: np.ndarray | float) -> np.ndarray | float:
    return 20.0 * np.log10(np.maximum(np.abs(x), EPS))


@dataclass
class FrameBank:
    """Framed signal plus its magnitude spectrogram and per-frame statistics."""

    samples: np.ndarray
    sample_rate: int
    frames: np.ndarray      # (n_frames, frame_len)
    spec: np.ndarray        # (n_frames, n_bins) magnitude
    freqs: np.ndarray       # (n_bins,)
    hop: int
    frame_len: int

    @property
    def n_frames(self) -> int:
        return self.frames.shape[0]

    @property
    def times(self) -> np.ndarray:
        return np.arange(self.n_frames) * self.hop / self.sample_rate

    @property
    def frame_sec(self) -> float:
        return self.hop / self.sample_rate

    @cached_property
    def energy_db(self) -> np.ndarray:
        return db((self.frames**2).mean(axis=1))

    @cached_property
    def power(self) -> np.ndarray:
        return self.spec**2

    @cached_property
    def total_power(self) -> np.ndarray:
        return self.power.sum(axis=1) + EPS

    @cached_property
    def noise_floor_db(self) -> float:
        """Level of the quietest tenth of the clip."""
        return float(np.percentile(self.energy_db, 10))

    @cached_property
    def speech_level_db(self) -> float:
        """Level of the loudest tenth, a robust stand-in for speech level."""
        return float(np.percentile(self.energy_db, 90))

    @cached_property
    def spectral_flatness(self) -> np.ndarray:
        """Geometric over arithmetic mean. Near 1 = noise-like, near 0 = tonal."""
        logmean = np.log(np.maximum(self.power, EPS)).mean(axis=1)
        return np.exp(logmean) / (self.power.mean(axis=1) + EPS)

    @cached_property
    def spectral_centroid(self) -> np.ndarray:
        return (self.power * self.freqs).sum(axis=1) / self.total_power

    @cached_property
    def spectral_rolloff(self) -> np.ndarray:
        """Frequency below which 95% of the energy sits — the effective bandwidth."""
        cum = np.cumsum(self.power, axis=1)
        target = 0.95 * cum[:, -1:]
        idx = (cum < target).sum(axis=1)
        idx = np.clip(idx, 0, self.freqs.size - 1)
        return self.freqs[idx]

    def band_ratio(self, lo: float, hi: float) -> np.ndarray:
        """Fraction of frame energy inside a frequency band."""
        mask = (self.freqs >= lo) & (self.freqs < hi)
        return self.power[:, mask].sum(axis=1) / self.total_power


def build_frames(samples: np.ndarray, sample_rate: int) -> FrameBank:
    frame_len = int(round(FRAME_SEC * sample_rate))
    hop = int(round(HOP_SEC * sample_rate))

    if samples.size < frame_len:
        samples = np.pad(samples, (0, frame_len - samples.size))

    n_frames = 1 + (samples.size - frame_len) // hop
    idx = np.arange(frame_len)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = samples[idx]

    window = np.hanning(frame_len).astype(np.float32)
    spec = np.abs(np.fft.rfft(frames * window, axis=1)).astype(np.float32)
    freqs = np.fft.rfftfreq(frame_len, 1.0 / sample_rate).astype(np.float32)

    return FrameBank(
        samples=samples, sample_rate=sample_rate, frames=frames,
        spec=spec, freqs=freqs, hop=hop, frame_len=frame_len,
    )


# --------------------------------------------------------------------------
# pitch
# --------------------------------------------------------------------------

def frame_pitch(
    fb: FrameBank, fmin: float = 70.0, fmax: float = 400.0
) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame fundamental frequency by autocorrelation.

    Returns (f0_hz, voicing_strength). f0 is 0 where the frame is unvoiced.
    Autocorrelation rather than a neural tracker: it costs nothing, runs
    everywhere, and we only need pitch range and variability, not perfect tracks.
    """
    sr = fb.sample_rate
    min_lag = max(2, int(sr / fmax))
    max_lag = min(fb.frame_len - 1, int(sr / fmin))
    if max_lag <= min_lag:
        return np.zeros(fb.n_frames), np.zeros(fb.n_frames)

    x = fb.frames - fb.frames.mean(axis=1, keepdims=True)
    n = 1 << int(np.ceil(np.log2(2 * fb.frame_len)))
    spec = np.fft.rfft(x, n=n, axis=1)
    ac = np.fft.irfft(spec * np.conj(spec), n=n, axis=1)[:, :max_lag + 1]

    zero_lag = ac[:, :1] + EPS
    norm = ac / zero_lag

    window = norm[:, min_lag:max_lag + 1]
    best = np.argmax(window, axis=1)
    strength = window[np.arange(window.shape[0]), best]
    lags = best + min_lag
    f0 = np.where(strength > 0.30, sr / np.maximum(lags, 1), 0.0)
    return f0.astype(np.float32), strength.astype(np.float32)


# --------------------------------------------------------------------------
# clip-level statistics
# --------------------------------------------------------------------------

@dataclass
class LevelStats:
    peak_dbfs: float
    rms_dbfs: float
    clip_ratio: float
    dc_offset: float
    crest_factor_db: float


def level_stats(samples: np.ndarray) -> LevelStats:
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    rms = float(np.sqrt((samples**2).mean())) if samples.size else 0.0
    # Count samples pinned at or beyond full scale, the signature of clipping.
    clipped = float(np.mean(np.abs(samples) >= 0.985)) if samples.size else 0.0
    return LevelStats(
        peak_dbfs=float(amp_db(peak)),
        rms_dbfs=float(amp_db(rms)),
        clip_ratio=clipped,
        dc_offset=float(np.mean(samples)) if samples.size else 0.0,
        crest_factor_db=float(amp_db(peak) - amp_db(rms)),
    )

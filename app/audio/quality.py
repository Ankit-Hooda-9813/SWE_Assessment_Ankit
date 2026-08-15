"""Technical audio quality, deliberately independent of background noise.

The brief scores audio quality separately from noise detection and warns against
conflating them, so this module never reads the noise estimate. It measures
clipping, muffling, low level, packet loss, reverberation, and vocoder artefacts,
and nothing else. A pristine recording can have a television behind it, and a
quiet room can arrive badly clipped.

Each detector is written against a measurable physical signature rather than a
proxy, because the proxies fail in exactly the cases that matter:

  * Bandwidth is the -35 dB edge of the speech spectrum, not the 95% energy
    rolloff. Speech puts most of its energy below 1.5 kHz, so a rolloff-based
    measure calls every normal call muffled.

  * Reverberation is the decay time from the local speech level, not the
    autocorrelation of the energy envelope. Ordinary speech rhythm makes that
    autocorrelation high on every clip.

  * Packet loss is a gap that truncates speech mid-word, not any short gap.
    Stop consonants and inter-word pauses produce short gaps constantly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.audio.features import EPS, FrameBank, LevelStats
from app.audio.vad import VadResult
from app.config import Thresholds
from app.schema import AudioQuality

BAND_EDGE_DROP_DB = 35.0


@dataclass
class QualityResult:
    quality: AudioQuality
    defects: list[str]
    evidence: dict


def _band_edge_hz(fb: FrameBank, vad: VadResult) -> float:
    """Highest frequency still carrying meaningful speech energy."""
    frames = fb.spec[vad.voiced] if vad.voiced.any() else fb.spec
    if frames.shape[0] == 0:
        return float(fb.freqs[-1])
    median = 20.0 * np.log10(np.median(frames, axis=0) + EPS)
    peak = float(median.max())
    above = np.flatnonzero(median > peak - BAND_EDGE_DROP_DB)
    if above.size == 0:
        return float(fb.freqs[-1])
    return float(fb.freqs[above[-1]])


def _reverb_decay_ms(fb: FrameBank, vad: VadResult) -> float:
    """Median time for speech to fall 25 dB below the local speech level.

    Measured from the loudest frame in the 250 ms preceding each speech offset,
    so the reference is genuine speech rather than the last frame above the VAD
    gate. A gated or dry recording collapses within a frame or two; a reverberant
    room or an echoing line takes a couple of hundred milliseconds.
    """
    energy = fb.energy_db
    voiced = vad.voiced
    lookback = max(2, int(0.250 / fb.frame_sec))
    horizon = max(4, int(0.300 / fb.frame_sec))

    decays: list[float] = []
    for i in range(lookback, voiced.size - horizon):
        if not (voiced[i - 1] and not voiced[i]):
            continue
        reference = float(energy[i - lookback:i].max())
        tail = energy[i:i + horizon]
        below = np.flatnonzero(tail < reference - 25.0)
        decays.append(
            float(below[0] + 1) * fb.frame_sec if below.size else horizon * fb.frame_sec
        )
    if not decays:
        return 0.0
    return float(np.median(decays) * 1000.0)


def _dropout_fraction(fb: FrameBank, vad: VadResult) -> float:
    """Share of speech lost to gaps that truncate a word mid-flow.

    Computed on raw frame energy rather than on the VAD mask. The VAD smooths
    away runs shorter than 60 ms to stop segment flicker, which is exactly the
    length of the holes we are trying to find — running this off the smoothed
    mask silently reports zero dropouts on badly degraded audio.

    A hole only counts when the speech on *both* sides of it is still near the
    local speech level. Natural pauses ramp down before the gap and ramp up
    after; packet loss cuts straight through mid-syllable.
    """
    energy = fb.energy_db
    if energy.size < 20:
        return 0.0

    speech_level = fb.speech_level_db
    # A dropout is a collapse toward nothing, not merely a quiet moment.
    hole = energy < speech_level - 40.0
    loud = energy > speech_level - 8.0

    max_gap = int(0.250 / fb.frame_sec)
    min_gap = max(1, int(0.020 / fb.frame_sec))
    speech_frames = max(int((energy > speech_level - 25.0).sum()), 1)

    lost = 0
    run = 0
    for i in range(1, hole.size):
        if hole[i]:
            run += 1
            continue
        if min_gap <= run <= max_gap:
            before = i - run - 1
            if before >= 0 and loud[before] and loud[i]:
                lost += run
        run = 0
    return lost / speech_frames


def _robotic_score(fb: FrameBank, vad: VadResult) -> float:
    """Vocoder artefacts: unnaturally stable spectral motion between frames."""
    if vad.voiced.sum() < 20:
        return 0.0
    centroid = fb.spectral_centroid[vad.voiced]
    if centroid.size < 20 or centroid.mean() < EPS:
        return 0.0
    variation = float(np.std(np.diff(centroid)) / (centroid.mean() + EPS))
    return max(0.0, 1.0 - variation * 25.0)


def analyse_quality(
    fb: FrameBank, vad: VadResult, levels: LevelStats, th: Thresholds
) -> QualityResult:
    band_edge = _band_edge_hz(fb, vad)
    reverb_ms = _reverb_decay_ms(fb, vad)
    dropout = _dropout_fraction(fb, vad)
    robotic = _robotic_score(fb, vad)
    speech_db = fb.speech_level_db

    defects: list[str] = []
    severe = False
    slight = False

    def note(is_severe: bool, is_slight: bool, label: str) -> None:
        nonlocal severe, slight
        if is_severe:
            defects.append(label + " (severe)")
            severe = True
        elif is_slight:
            defects.append(label)
            slight = True

    note(levels.clip_ratio >= th.clip_ratio_severe,
         levels.clip_ratio >= th.clip_ratio_slight, "clipping")
    note(band_edge <= th.bandwidth_severe_hz,
         band_edge <= th.bandwidth_slight_hz, "muffled or narrowband")
    note(speech_db <= th.low_level_severe_dbfs,
         speech_db <= th.low_level_slight_dbfs, "low volume")
    note(dropout >= th.dropout_severe_frac,
         dropout >= th.dropout_slight_frac, "dropout or packet loss")
    note(reverb_ms >= th.echo_severe_ms, reverb_ms >= th.echo_slight_ms, "echo or reverberation")
    note(robotic >= 0.65, robotic >= 0.40, "robotic or vocoded")

    if severe:
        quality = AudioQuality.SEVERELY_IMPAIRED
    elif slight:
        quality = AudioQuality.SLIGHTLY_IMPAIRED
    else:
        quality = AudioQuality.CLEAR

    # Several independent mild defects compound into more than a mild problem.
    if quality is AudioQuality.SLIGHTLY_IMPAIRED and len(defects) >= 3:
        quality = AudioQuality.SEVERELY_IMPAIRED

    evidence = {
        "band_edge_hz": round(band_edge),
        "speech_level_dbfs": round(speech_db, 1),
        "peak_dbfs": round(levels.peak_dbfs, 1),
        "clip_ratio": round(levels.clip_ratio, 5),
        "dropout_frac": round(dropout, 4),
        "reverb_decay_ms": round(reverb_ms),
        "robotic_score": round(robotic, 3),
        "crest_factor_db": round(levels.crest_factor_db, 1),
        "defects": defects,
    }
    return QualityResult(quality=quality, defects=defects, evidence=evidence)

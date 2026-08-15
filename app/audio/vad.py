"""Voice activity detection and silence structure.

An adaptive energy gate rather than a neural VAD: it has no download, no model
load, and no dependency, and on call audio with a well-separated speech level it
matches Silero closely enough for the two things we need it for — locating
speech regions so noise can be measured underneath them, and finding dead air.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.audio.features import FrameBank
from app.config import Thresholds


@dataclass
class Segment:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class VadResult:
    voiced: np.ndarray          # per-frame boolean
    speech: list[Segment]
    silence: list[Segment]      # gaps strictly between speech segments
    lead_silence: float
    trail_silence: float
    speech_ratio: float
    longest_internal_silence: float
    threshold_db: float


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous True runs as [start, end) index pairs."""
    if mask.size == 0:
        return []
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[0::2], edges[1::2]))


def _smooth(mask: np.ndarray, min_frames: int) -> np.ndarray:
    """Drop runs shorter than min_frames, in both polarities.

    Removes single-frame flicker at segment boundaries that would otherwise
    fragment silences and inflate the segment count.
    """
    out = mask.copy()
    for start, end in _runs(out):
        if end - start < min_frames:
            out[start:end] = False
    inv = ~out
    for start, end in _runs(inv):
        if end - start < min_frames:
            out[start:end] = True
    return out


def detect_voice(fb: FrameBank, th: Thresholds) -> VadResult:
    energy = fb.energy_db
    floor = fb.noise_floor_db
    speech_level = fb.speech_level_db

    # Two gates. The relative one adapts to the recording's own noise floor; the
    # absolute one stops a digitally-silent file from promoting its own dither
    # to "speech". Provided clips have floors near -240 dBFS, which makes a
    # purely relative gate meaningless.
    relative = floor + th.vad_rel_db
    absolute = th.vad_abs_floor_dbfs

    # When speech sits barely above the floor, bias toward the midpoint so we
    # do not swallow the quiet half of the conversation.
    if speech_level - floor < 2 * th.vad_rel_db:
        relative = floor + 0.45 * max(speech_level - floor, 0.0)

    threshold = max(relative, absolute)
    voiced = _smooth(energy > threshold, min_frames=int(0.06 / fb.frame_sec))

    sec = fb.frame_sec
    speech = [Segment(s * sec, e * sec) for s, e in _runs(voiced)]

    total = fb.n_frames * sec
    lead = speech[0].start if speech else total
    trail = total - speech[-1].end if speech else 0.0

    silence: list[Segment] = []
    for prev, nxt in zip(speech, speech[1:]):
        silence.append(Segment(prev.end, nxt.start))

    longest = max((s.duration for s in silence), default=0.0)

    return VadResult(
        voiced=voiced,
        speech=speech,
        silence=silence,
        lead_silence=lead,
        trail_silence=trail,
        speech_ratio=float(voiced.mean()) if voiced.size else 0.0,
        longest_internal_silence=longest,
        threshold_db=float(threshold),
    )


def detect_long_silence(vad: VadResult, th: Thresholds) -> tuple[bool, dict]:
    """Whether the clip contains dead air worth flagging.

    Only gaps between speech count. Leading and trailing silence is where a
    recorder was started early or stopped late, which is not the call-flow
    problem this field is asking about.
    """
    qualifying = [s for s in vad.silence if s.duration >= th.long_silence_sec]
    evidence = {
        "longest_internal_silence_sec": round(vad.longest_internal_silence, 2),
        "qualifying_gaps": len(qualifying),
        "gap_starts": [round(s.start, 1) for s in qualifying[:5]],
        "lead_silence_sec": round(vad.lead_silence, 2),
        "trail_silence_sec": round(vad.trail_silence, 2),
        "threshold_sec": th.long_silence_sec,
    }
    return bool(qualifying), evidence

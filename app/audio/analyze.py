"""One pass over a clip producing every measured field.

This is the free, deterministic half of the system: six of the nine output
fields plus the prosody vector, with no network access and no model download.
It is also the half that must never disagree with itself, so all detectors read
from the same frame bank and the same VAD.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from app.audio.features import FrameBank, LevelStats, build_frames, level_stats
from app.audio.io import Clip
from app.audio.noise import NoiseResult, analyse_noise
from app.audio.overlap import OverlapResult, detect_overlap
from app.audio.prosody import Prosody, extract_prosody
from app.audio.quality import QualityResult, analyse_quality
from app.audio.vad import VadResult, detect_long_silence, detect_voice
from app.config import Thresholds, get_thresholds


@dataclass
class AcousticAnalysis:
    """Measured fields plus the evidence behind each one."""

    duration_sec: float
    noise: NoiseResult
    quality: QualityResult
    overlap: OverlapResult
    long_silence: bool
    long_silence_evidence: dict
    prosody: Prosody
    vad: VadResult = field(repr=False)
    levels: LevelStats = field(repr=False)
    # Kept so diarization can reuse the same frames and pitch track rather than
    # recomputing them.
    frames: FrameBank = field(repr=False, default=None)
    channels_identical: bool = False
    elapsed_sec: float = 0.0

    def measured_fields(self) -> dict:
        return {
            "background_noise_present": self.noise.present,
            "background_noise_type": self.noise.noise_type,
            "background_noise_severity": self.noise.severity,
            "audio_quality": self.quality.quality,
            "speaker_overlap_present": self.overlap.present,
            "long_silence_present": self.long_silence,
        }

    def evidence(self) -> dict:
        return {
            "noise": self.noise.evidence,
            "noise_type_scores": self.noise.type_scores,
            "quality": self.quality.evidence,
            "overlap": self.overlap.evidence,
            "long_silence": self.long_silence_evidence,
            "prosody": self.prosody.to_dict(),
            "vad": {
                "speech_ratio": round(self.vad.speech_ratio, 3),
                "speech_segments": len(self.vad.speech),
                "threshold_dbfs": round(self.vad.threshold_db, 1),
            },
            "source": {
                "duration_sec": round(self.duration_sec, 2),
                "dual_mono": self.channels_identical,
            },
            "analysis_sec": round(self.elapsed_sec, 3),
        }


def analyse_acoustics(clip: Clip, th: Thresholds | None = None) -> AcousticAnalysis:
    th = th or get_thresholds()
    started = time.perf_counter()

    fb = build_frames(clip.samples, clip.sample_rate)
    levels = level_stats(clip.samples)
    vad = detect_voice(fb, th)

    noise = analyse_noise(fb, vad, th)
    quality = analyse_quality(fb, vad, levels, th)
    overlap = detect_overlap(fb, vad, th)
    long_silence, silence_evidence = detect_long_silence(vad, th)
    prosody = extract_prosody(fb, vad)

    return AcousticAnalysis(
        duration_sec=clip.duration_sec,
        noise=noise,
        quality=quality,
        overlap=overlap,
        long_silence=long_silence,
        long_silence_evidence=silence_evidence,
        prosody=prosody,
        vad=vad,
        frames=fb,
        levels=levels,
        channels_identical=clip.channels_identical,
        elapsed_sec=time.perf_counter() - started,
    )

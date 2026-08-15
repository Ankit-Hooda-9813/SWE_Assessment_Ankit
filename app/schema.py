"""The required output schema.

Enum values are fixed by the trial brief and must match exactly. Everything that
leaves this system passes through `AnalysisResult`, so an out-of-vocabulary value
can never reach a result file.
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class EmotionalTone(str, Enum):
    NEUTRAL = "neutral"
    SATISFIED = "satisfied"
    FRUSTRATED = "frustrated"
    UPSET = "upset"
    DISTRESSED = "distressed"


class EmotionalIntensity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class NoiseSeverity(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class AudioQuality(str, Enum):
    CLEAR = "clear"
    SLIGHTLY_IMPAIRED = "slightly_impaired"
    SEVERELY_IMPAIRED = "severely_impaired"


# Ordinal positions, used for distance-aware scoring and for clamping.
TONE_ORDER = [
    EmotionalTone.SATISFIED,
    EmotionalTone.NEUTRAL,
    EmotionalTone.FRUSTRATED,
    EmotionalTone.UPSET,
    EmotionalTone.DISTRESSED,
]
INTENSITY_ORDER = [EmotionalIntensity.LOW, EmotionalIntensity.MEDIUM, EmotionalIntensity.HIGH]
SEVERITY_ORDER = [NoiseSeverity.NONE, NoiseSeverity.LOW, NoiseSeverity.MEDIUM, NoiseSeverity.HIGH]
QUALITY_ORDER = [
    AudioQuality.CLEAR,
    AudioQuality.SLIGHTLY_IMPAIRED,
    AudioQuality.SEVERELY_IMPAIRED,
]


class AnalysisResult(BaseModel):
    """The exact object required per audio clip."""

    emotional_tone: EmotionalTone
    emotional_intensity: EmotionalIntensity
    background_noise_present: bool
    background_noise_type: str = ""
    background_noise_severity: NoiseSeverity
    audio_quality: AudioQuality
    speaker_overlap_present: bool
    long_silence_present: bool
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("confidence")
    @classmethod
    def _round_confidence(cls, v: float) -> float:
        return round(float(v), 2)

    @field_validator("background_noise_type")
    @classmethod
    def _clean_type(cls, v: str) -> str:
        return (v or "").strip()

    def model_post_init(self, _ctx: Any) -> None:
        # The brief defines these as coupled. Enforce consistency so we can never
        # emit "no noise present" alongside a noise type, or vice versa.
        if not self.background_noise_present:
            object.__setattr__(self, "background_noise_type", "")
            object.__setattr__(self, "background_noise_severity", NoiseSeverity.NONE)
        elif self.background_noise_severity is NoiseSeverity.NONE:
            object.__setattr__(self, "background_noise_severity", NoiseSeverity.LOW)

    def to_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), ensure_ascii=False)


SCHEMA_FIELDS = list(AnalysisResult.model_fields.keys())

# Fields owned by deterministic signal processing rather than by a model.
MEASURED_FIELDS = [
    "background_noise_present",
    "background_noise_type",
    "background_noise_severity",
    "audio_quality",
    "speaker_overlap_present",
    "long_silence_present",
]
INFERRED_FIELDS = ["emotional_tone", "emotional_intensity", "confidence"]


def coerce_result(raw: dict[str, Any]) -> AnalysisResult:
    """Build a valid result from untrusted input, repairing what we safely can.

    Model output and manifest JSON both arrive here. Anything unrecognisable
    falls back to the least-committal legal value rather than raising, because a
    single bad field must not cost us the whole clip.
    """
    data = dict(raw or {})

    def pick(key: str, allowed: type[Enum], default: Enum) -> str:
        val = data.get(key)
        if isinstance(val, allowed):
            return val.value
        if isinstance(val, str):
            v = val.strip().lower().replace("-", "_").replace(" ", "_")
            for member in allowed:
                if v == member.value:
                    return member.value
        return default.value

    def flag(key: str) -> bool:
        val = data.get(key)
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.strip().lower() in {"true", "yes", "1", "t"}
        return bool(val)

    conf = data.get("confidence", 0.5)
    try:
        conf = min(1.0, max(0.0, float(conf)))
    except (TypeError, ValueError):
        conf = 0.5

    return AnalysisResult(
        emotional_tone=pick("emotional_tone", EmotionalTone, EmotionalTone.NEUTRAL),
        emotional_intensity=pick(
            "emotional_intensity", EmotionalIntensity, EmotionalIntensity.LOW
        ),
        background_noise_present=flag("background_noise_present"),
        background_noise_type=str(data.get("background_noise_type") or ""),
        background_noise_severity=pick(
            "background_noise_severity", NoiseSeverity, NoiseSeverity.NONE
        ),
        audio_quality=pick("audio_quality", AudioQuality, AudioQuality.CLEAR),
        speaker_overlap_present=flag("speaker_overlap_present"),
        long_silence_present=flag("long_silence_present"),
        confidence=conf,
    )

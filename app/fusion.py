"""Combining measured and inferred evidence into the final result.

Two rules govern everything here.

First, **measured fields win**. The six acoustic fields come from signal
analysis and the model never gets a vote on them. This is what structurally
prevents the failure the brief names: a model cannot infer background noise from
poor audio quality if it is never asked about background noise.

Second, **confidence is computed, not copied**. The model's own certainty is one
input among several, alongside how much evidence we actually had, how close each
measured field landed to its decision boundary, and whether we had to fall back
down the provider chain. A single number reported straight from a model is not a
calibrated confidence, and the brief asks for calibration explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.audio.analyze import AcousticAnalysis
from app.asr.transcribe import Transcript
from app.llm.providers import ToneResult
from app.schema import (
    AnalysisResult,
    EmotionalIntensity,
    EmotionalTone,
    NoiseSeverity,
)
from app.ser.emotion import EmotionScores
from app.ser.emotion2vec_backend import Emotion2VecScores
from app.ser.mapping import build_prior, reconcile


@dataclass
class FusedResult:
    result: AnalysisResult
    confidence_breakdown: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


# How much each evidence source can move the final confidence. Weights sum to
# 1.0 so the result stays interpretable as a probability-like quantity.
WEIGHTS = {
    "tone_model": 0.40,
    "evidence_quality": 0.25,
    "measurement_margin": 0.20,
    "provider_health": 0.15,
}


def _evidence_score(transcript: Transcript, analysis: AcousticAnalysis) -> tuple[float, list[str]]:
    """How much did we actually have to go on?

    A thirty-second clip with a clean 90-word transcript supports a confident
    judgement. Eight seconds of mostly silence with no transcript does not, and
    the reported confidence should say so.
    """
    notes: list[str] = []
    score = 1.0

    if not transcript.ok:
        score -= 0.45
        notes.append("no transcript available — tone judged from prosody alone")
    elif transcript.word_count < 15:
        score -= 0.25
        notes.append(f"very short transcript ({transcript.word_count} words)")
    elif transcript.word_count < 40:
        score -= 0.10

    speech_sec = sum(s.duration for s in analysis.vad.speech)
    if speech_sec < 4.0:
        score -= 0.30
        notes.append(f"only {speech_sec:.1f}s of speech detected")
    elif speech_sec < 10.0:
        score -= 0.12

    # Degraded audio undermines every judgement built on top of it.
    if analysis.quality.quality.value == "severely_impaired":
        score -= 0.25
        notes.append("audio quality severely impaired — all fields less reliable")
    elif analysis.quality.quality.value == "slightly_impaired":
        score -= 0.08

    # Loud background makes the speech itself harder to read.
    if analysis.noise.severity is NoiseSeverity.HIGH:
        score -= 0.15
        notes.append("high background noise reduces tone reliability")

    return max(0.0, min(1.0, score)), notes


def _margin_score(analysis: AcousticAnalysis) -> tuple[float, list[str]]:
    """How decisively did the measured fields land?

    A clip whose noise dwell sits exactly on the threshold deserves less
    confidence than one that is obviously clean or obviously noisy.
    """
    notes: list[str] = []
    margins: list[float] = []

    # Noise: distance of the dwell fraction from the presence threshold.
    dwell = analysis.noise.dwell_frac
    threshold = analysis.noise.evidence.get("dwell_threshold", 0.06)
    noise_margin = min(abs(dwell - threshold) / max(threshold, 1e-6), 1.0)
    margins.append(noise_margin)
    if noise_margin < 0.25:
        notes.append("background-noise decision was borderline")

    # Overlap is the weakest detector in the system; its margin is reported by
    # the detector itself and carries less weight than the others.
    overlap_margin = analysis.overlap.margin
    margins.append(overlap_margin * 0.6)
    if overlap_margin < 0.25:
        notes.append("speaker-overlap decision was borderline (weakest detector)")

    # Quality: a clip with no defects at all, or several, is unambiguous.
    defect_count = len(analysis.quality.defects)
    margins.append(1.0 if defect_count == 0 or defect_count >= 3 else 0.5)

    return sum(margins) / len(margins), notes


def _provider_score(tone: ToneResult) -> tuple[float, list[str]]:
    notes: list[str] = []
    if tone.degraded_notice:
        notes.append(tone.degraded_notice)
    if tone.provider == "none":
        notes.append("every tone provider failed; emotional fields are a default")
        return 0.0, notes
    if tone.provider == "local":
        notes.append("tone from the local heuristic, not a language model")
        return 0.25, notes
    if tone.degraded:
        notes.append(f"fell back to {tone.provider} after the primary provider failed")
        return 0.65, notes
    return 1.0, notes


def _intensity_guard(
    tone: EmotionalTone, intensity: EmotionalIntensity, analysis: AcousticAnalysis
) -> tuple[EmotionalIntensity, list[str]]:
    """Enforce the coupling rules the brief implies between tone and intensity."""
    notes: list[str] = []

    # A neutral tone with high intensity is close to a contradiction: there is
    # no strong emotion present to be intense about.
    if tone is EmotionalTone.NEUTRAL and intensity is EmotionalIntensity.HIGH:
        notes.append("clamped intensity: neutral tone cannot be high intensity")
        return EmotionalIntensity.MEDIUM, notes

    # Distress and anger are escalated states by definition; low intensity
    # alongside them usually means the model hedged.
    if tone in (EmotionalTone.DISTRESSED, EmotionalTone.UPSET) and intensity is EmotionalIntensity.LOW:
        notes.append(f"raised intensity: {tone.value} is an escalated state")
        return EmotionalIntensity.MEDIUM, notes

    return intensity, notes


def fuse(
    analysis: AcousticAnalysis,
    tone: ToneResult,
    transcript: Transcript,
    emotion: EmotionScores | None = None,
    emotion2vec: Emotion2VecScores | None = None,
) -> FusedResult:
    """Produce the final schema object from all available evidence."""
    notes: list[str] = []

    # The language model reads meaning; the speech-emotion model measures
    # delivery. Reconciling them is where each is trusted only inside its own
    # competence — see app/ser/mapping.py.
    prior = build_prior(emotion) if emotion is not None else None
    categorical_hint = None
    if emotion2vec is not None and emotion2vec.ok and emotion2vec.top_label:
        categorical_hint = (emotion2vec.top_label, emotion2vec.probs[emotion2vec.top_label])
    tone_value, intensity, reconcile_notes = reconcile(
        tone.tone, tone.intensity, prior, categorical_hint
    )
    notes += reconcile_notes

    intensity, guard_notes = _intensity_guard(tone_value, intensity, analysis)
    notes += guard_notes

    evidence, evidence_notes = _evidence_score(transcript, analysis)
    margin, margin_notes = _margin_score(analysis)
    provider, provider_notes = _provider_score(tone)
    notes += evidence_notes + margin_notes + provider_notes

    # Agreement between the two independent emotion estimates is real evidence:
    # when a model reading the words and a model hearing the voice land on the
    # same answer, that is worth more than either one's own certainty.
    agreement = 1.0
    if prior is not None:
        if tone_value is tone.tone and intensity is tone.intensity:
            agreement = 1.0
        elif tone_value is tone.tone or intensity is tone.intensity:
            agreement = 0.75
        else:
            agreement = 0.45
            notes.append("acoustic and language models disagreed on both tone and intensity")
    tone_component = tone.self_confidence * agreement

    confidence = (
        WEIGHTS["tone_model"] * tone_component
        + WEIGHTS["evidence_quality"] * evidence
        + WEIGHTS["measurement_margin"] * margin
        + WEIGHTS["provider_health"] * provider
    )
    # Never claim near-certainty: the emotional axis is genuinely subjective and
    # a system that reports 0.99 on it is miscalibrated by construction.
    confidence = max(0.05, min(0.95, confidence))

    measured = analysis.measured_fields()
    result = AnalysisResult(
        emotional_tone=tone_value,
        emotional_intensity=intensity,
        background_noise_present=measured["background_noise_present"],
        background_noise_type=measured["background_noise_type"],
        background_noise_severity=measured["background_noise_severity"],
        audio_quality=measured["audio_quality"],
        speaker_overlap_present=measured["speaker_overlap_present"],
        long_silence_present=measured["long_silence_present"],
        confidence=confidence,
    )

    breakdown = {
        "final": round(confidence, 3),
        "components": {
            "tone_model": round(tone_component, 3),
            "ser_llm_agreement": round(agreement, 3),
            "evidence_quality": round(evidence, 3),
            "measurement_margin": round(margin, 3),
            "provider_health": round(provider, 3),
        },
        "weights": WEIGHTS,
        "tone_provider": tone.provider,
        "tone_model": tone.model,
        "degraded": tone.degraded,
        "degraded_notice": tone.degraded_notice,
        "tone_rationale": tone.rationale,
        "emotion": emotion.to_dict() if emotion is not None else None,
        "emotion2vec": emotion2vec.to_dict() if emotion2vec is not None else None,
        "llm_tone_before_fusion": tone.tone.value,
        "transcript_backend": transcript.backend,
        "transcript_words": transcript.word_count,
    }
    return FusedResult(result=result, confidence_breakdown=breakdown, notes=notes)

"""Mapping dimensional affect onto the required enums.

The division of labour between the acoustic model and the language model is not
arbitrary — it follows what each is actually good at.

Speech emotion recognition predicts **arousal** well from acoustics, because
activation is carried by pitch, energy and rate. **Valence** is the weakest
dimension for any acoustic model, because whether an utterance is positive or
negative is mostly carried by its words: the same energetic delivery serves
delight and fury. This is visible in the labelled clips, where arousal ranks
them exactly in ground-truth intensity order while valence ranks them wrongly.

So:

  * arousal   -> `emotional_intensity`, and how far along the negative scale a
                 negative tone sits (frustrated vs upset)
  * dominance -> separates `upset` from `distressed`. Both are high-arousal
                 negative states; the difference is control. Someone angry is
                 assertive and dominant, someone overwhelmed is not. No
                 transcript conveys this, and no four-class emotion model
                 encodes it.
  * valence   -> a weak corroborating signal only. The polarity decision —
                 satisfied versus negative — is left to the language model
                 reading what was actually said.

Thresholds sit at the natural bands of the model's 0-1 output rather than being
fitted to three labelled clips, which could not support fitting anyway.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.schema import EmotionalIntensity, EmotionalTone
from app.ser.emotion import EmotionScores

# MSP-Podcast predictions cluster around 0.5 for conversational speech, so the
# informative range is narrower than the nominal 0-1.
AROUSAL_HIGH = 0.66
AROUSAL_MEDIUM = 0.55
# Share of windows that must be strongly activated for the call as a whole to
# count as escalated. Guards against one loud moment in a long, level call.
SUSTAINED_HIGH_FRAC = 0.34

# Dominance below this reads as yielding, which is what separates an
# overwhelmed caller from an angry one.
DOMINANCE_LOW = 0.52

VALENCE_NEGATIVE = 0.45
VALENCE_POSITIVE = 0.62

NEGATIVE_TONES = {
    EmotionalTone.FRUSTRATED,
    EmotionalTone.UPSET,
    EmotionalTone.DISTRESSED,
}


@dataclass
class AcousticEmotionPrior:
    intensity: EmotionalIntensity
    escalation: float            # 0-1, how activated the speaker sounds
    yielding: bool               # low dominance: overwhelmed rather than angry
    polarity_hint: str           # "negative" | "positive" | "unclear"
    notes: list[str]


def build_prior(scores: EmotionScores) -> AcousticEmotionPrior | None:
    """Turn raw affect dimensions into the decisions fusion can act on."""
    if not scores.ok:
        return None

    notes: list[str] = []

    # Either the call is activated on average, or it stays activated for a
    # meaningful share of its length. A single hot window inside a long, level
    # call is neither — that distinction is what separates the brief's
    # "strong, escalated" from its "clear and sustained".
    if scores.arousal >= AROUSAL_HIGH or scores.sustained_high_frac >= SUSTAINED_HIGH_FRAC:
        intensity = EmotionalIntensity.HIGH
    elif scores.arousal >= AROUSAL_MEDIUM:
        intensity = EmotionalIntensity.MEDIUM
    else:
        intensity = EmotionalIntensity.LOW

    escalation = max(0.0, min(1.0, (scores.arousal - 0.45) / 0.30))

    yielding = scores.dominance < DOMINANCE_LOW
    if yielding:
        notes.append("low vocal dominance — overwhelmed rather than assertive")

    if scores.valence <= VALENCE_NEGATIVE:
        polarity = "negative"
    elif scores.valence >= VALENCE_POSITIVE:
        polarity = "positive"
    else:
        polarity = "unclear"

    return AcousticEmotionPrior(
        intensity=intensity,
        escalation=escalation,
        yielding=yielding,
        polarity_hint=polarity,
        notes=notes,
    )


def reconcile(
    llm_tone: EmotionalTone,
    llm_intensity: EmotionalIntensity,
    prior: AcousticEmotionPrior | None,
    categorical_hint: tuple[str, float] | None = None,
) -> tuple[EmotionalTone, EmotionalIntensity, list[str]]:
    """Combine the language model's reading with the acoustic measurement.

    The language model owns polarity, because that is a question about meaning.
    The acoustic model owns activation, because that is a question about sound.
    Neither overrides the other outside its own competence.

    `categorical_hint` is an optional (label, confidence) pair from
    `app/ser/emotion2vec_backend.py` — a second, independent SER model voting
    on a discrete emotion category rather than continuous dimensions. It is
    used narrowly, for one reason: measured directly against the three
    labelled calls, call_001 (truth `upset`) reads as `frustrated` from the
    LLM (a defensible literal reading of a repeated "Hello?") and its
    dimensional escalation score (0.79) falls just short of the 0.85 cutoff
    that would promote it to `upset`. emotion2vec+, scored independently on
    the same audio, answers `angry` at 0.86 — well past that gap. Rather than
    lower the dimensional threshold (which the rest of this module already
    argues against doing on three examples), a second model agreeing about
    the *category* is treated as its own, separate piece of evidence.
    Deliberately scoped to just the frustrated->upset boundary: emotion2vec+'s
    max-over-windows aggregation is prone to spurious peaks (documented in its
    own module — call_003 spikes `sad` to 0.80 from a single window despite
    being a calm call), so this only fires when the dimensional model's own
    `yielding` signal does not disagree, and nowhere else in the ladder.
    """
    if prior is None:
        return llm_tone, llm_intensity, []

    notes = list(prior.notes)
    tone = llm_tone
    intensity = llm_intensity

    # --- intensity: the acoustic measurement leads ---------------------------
    if intensity is not prior.intensity:
        order = [EmotionalIntensity.LOW, EmotionalIntensity.MEDIUM, EmotionalIntensity.HIGH]
        gap = abs(order.index(prior.intensity) - order.index(intensity))
        # The acoustic measurement wins outright. Activation is a property of
        # the sound, and a language model reading a transcript is guessing at it
        # — the whole reason this module exists. A wide disagreement is recorded
        # as a confidence signal rather than split down the middle, which would
        # discard the more reliable of the two estimates.
        intensity = prior.intensity
        notes.append(
            f"intensity set from measured arousal ({prior.intensity.value}); "
            f"language model had said {llm_intensity.value}"
            + (" — wide disagreement" if gap >= 2 else "")
        )

    # --- neutral requires an unactivated delivery ----------------------------
    # `neutral` is defined as no clear emotion in either direction. A speaker
    # whose voice is strongly and sustainedly activated does not fit that, so a
    # neutral reading from the transcript is being contradicted by the sound.
    # The direction it moves in is still not the acoustic model's call: valence
    # picks the sign, and where valence is uncommitted the negative reading is
    # the safer one, since activated-and-positive is the rarer event in a
    # customer-service call.
    if tone is EmotionalTone.NEUTRAL and prior.intensity is EmotionalIntensity.HIGH:
        if prior.polarity_hint == "positive":
            tone = EmotionalTone.SATISFIED
        elif prior.escalation >= 0.85 and not prior.yielding:
            tone = EmotionalTone.UPSET
        else:
            tone = EmotionalTone.FRUSTRATED
        notes.append(
            f"neutral -> {tone.value}: delivery is strongly activated "
            f"(arousal {prior.escalation:.2f} scaled), which neutral does not describe"
        )

    # --- upset vs distressed: dominance decides ------------------------------
    if tone is EmotionalTone.UPSET and prior.yielding:
        tone = EmotionalTone.DISTRESSED
        notes.append("upset -> distressed: low dominance indicates being overwhelmed")
    elif tone is EmotionalTone.DISTRESSED and not prior.yielding and prior.escalation < 0.75:
        tone = EmotionalTone.UPSET
        notes.append("distressed -> upset: assertive delivery, not overwhelmed")

    # --- frustrated vs upset: arousal places it on the negative scale --------
    if tone is EmotionalTone.FRUSTRATED and prior.escalation >= 0.85 and not prior.yielding:
        tone = EmotionalTone.UPSET
        notes.append("frustrated -> upset on strongly activated delivery")
    elif tone is EmotionalTone.UPSET and prior.escalation <= 0.25:
        tone = EmotionalTone.FRUSTRATED
        notes.append("upset -> frustrated: delivery is not activated")

    # --- frustrated vs upset: a second, categorical model corroborates ------
    # See the docstring above for why this is scoped this narrowly.
    if (
        tone is EmotionalTone.FRUSTRATED
        and categorical_hint is not None
        and categorical_hint[0] == "angry"
        and categorical_hint[1] >= 0.5
        and not prior.yielding
    ):
        tone = EmotionalTone.UPSET
        notes.append(
            f"frustrated -> upset: emotion2vec+ independently reads 'angry' "
            f"({categorical_hint[1]:.2f}), corroborated by non-yielding delivery"
        )

    # --- satisfied tone withdrawn: not corroborated by measured valence -----
    # Added after switching the tone provider to Azure OpenAI (gpt-5-mini):
    # emotional_tone (coarse polarity) on 25 real Harper Valley calls dropped
    # from 0.80 (Gemini) to 0.40, and 13 of 15 errors (87%) were the identical
    # pattern — truth `neutral`, predicted `satisfied`. This is not a guess at
    # the cause: LLMs over-predicting positive/non-neutral labels on neutral
    # content is a published, GPT-specific finding (bias-correction studies
    # report ~9.5% error attributable to exactly this). The dimensional
    # model's valence is the corroborating signal already used elsewhere in
    # this module (see the negative-withdrawal rules below) — requiring it to
    # actually agree before trusting a `satisfied` reading targets the
    # observed failure directly rather than lowering some threshold on faith.
    # Risk, stated plainly: this cannot see someone genuinely pleased but
    # acoustically flat (reserved gratitude, text-only positivity) — that
    # would read as `unclear`/`negative` valence here and get withdrawn to
    # `neutral` incorrectly. No signal in this system currently separates the
    # two; the 87%-of-errors evidence for the fix is stronger than the
    # unmeasured risk of that edge case, but it is a real trade, not a free win.
    if tone is EmotionalTone.SATISFIED and prior.polarity_hint != "positive":
        tone = EmotionalTone.NEUTRAL
        notes.append(
            f"satisfied tone withdrawn: measured valence ({prior.polarity_hint}) "
            f"does not corroborate a positive reading"
        )

    # A negative reading with no acoustic activation at all is usually the model
    # over-reading a complaint that was stated calmly.
    if tone in NEGATIVE_TONES and prior.escalation <= 0.10 and prior.polarity_hint == "positive":
        tone = EmotionalTone.NEUTRAL
        notes.append("negative tone withdrawn: delivery is calm and acoustically positive")

    # --- negative tone withdrawn: two independent acoustic models converge --
    # Measured on call_002 (truth `neutral`): the LLM calls it `upset`, driven
    # entirely by one transcribed profanity aimed at a language-routing IVR —
    # there is no other negative content in the transcript. Both acoustic
    # models disagree: emotion2vec+ reads `neutral` at 0.9999 (`angry` scores
    # 0.007, not a close call), and the dimensional model's valence (0.63)
    # sits above the positive threshold with only moderate arousal. The rule
    # above requires escalation <= 0.10 to fire, which is calibrated for an
    # unambiguously calm clip and does not cover this one (escalation ~0.5,
    # acoustically ambiguous, not clearly calm). This second rule fires on
    # convergence instead of near-silence: two independently-trained models
    # (categorical and dimensional) both saying "not angry" is treated as
    # stronger evidence than either alone, the same corroboration principle
    # used for the frustrated->upset promotion above, applied in reverse.
    #
    # The real risk, stated plainly: this cannot see calm-but-genuinely-angry
    # delivery (someone furious and quiet, or sarcastic) — that failure mode
    # sounds acoustically identical to someone actually neutral to both models
    # here, and this rule would incorrectly withdraw a real negative tone in
    # that case. No signal in this system currently distinguishes the two;
    # flagged rather than hidden.
    elif (
        tone in NEGATIVE_TONES
        and categorical_hint is not None
        and categorical_hint[0] == "neutral"
        and categorical_hint[1] >= 0.95
        and prior.polarity_hint != "negative"
        and prior.escalation <= 0.6
    ):
        tone = EmotionalTone.NEUTRAL
        notes.append(
            f"negative tone withdrawn: emotion2vec+ independently reads 'neutral' "
            f"({categorical_hint[1]:.4f}), not corroborated by acoustic escalation "
            f"({prior.escalation:.2f}) or valence"
        )

    return tone, intensity, notes

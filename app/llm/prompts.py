"""Prompts for the emotional-tone stage.

The model is asked about emotion and nothing else. It never sees the noise,
quality, overlap or silence questions, because those are measured and a model
asked about all nine fields at once starts reasoning across them — inferring
frustration from loudness, or background noise from poor audio quality. Those
are the two failure modes the brief names, and keeping the questions apart is
what prevents them structurally rather than by instruction.
"""

from __future__ import annotations

from app.audio.prosody import Prosody

TONE_DEFINITIONS = """\
emotional_tone — the primary emotional tone expressed by the CUSTOMER:
  neutral     no clear positive or negative emotion
  satisfied   pleased, relieved, appreciative, or clearly positive
  frustrated  annoyed, impatient, or dissatisfied, without strong anger or distress
  upset       clearly angry, agitated, or strongly dissatisfied
  distressed  highly emotional, overwhelmed, panicked, crying, or otherwise escalated

emotional_intensity — the strength of that tone:
  low         subtle or mild
  medium      clear and sustained
  high        strong, escalated, or likely to require attention"""

# Deliberately minimal. An earlier version added judgement heuristics of my own
# — when to prefer neutral, what counts as frustration — and tuning them against
# three labelled clips moved the predictions around without improving them: four
# variants each scored 1-2 of 6 with completely different error patterns. Three
# examples cannot calibrate a five-class judgement, and a prompt fitted to them
# would encode my guesses rather than the specification.
#
# So the rules below are only the ones the brief states or that follow directly
# from the recording setup. Everything else is left to the definitions.
RULES = """\
Judgement rules:

1. Loudness is not emotion. A person can be loud and cheerful, or quiet and
   furious. Poor audio quality and background noise are likewise not evidence of
   any emotional state. Decide tone from what is said and how it is said.
2. Judge the CUSTOMER, not the agent. These calls are answered by an automated
   voice agent whose delivery is synthetic and carries no emotional information.
   Ignore the agent entirely and assess only the human caller.
3. Apply the definitions above as written. Each of the five tones is a real
   answer, including `neutral`; do not treat any of them as a fallback or as a
   last resort.
4. Intensity describes the strength of whichever tone you chose, using the
   definitions above: `low` is subtle or mild, `medium` is clear and sustained,
   `high` is strong or escalated.

Report your certainty honestly in `self_confidence`. A short, ambiguous, or
mostly inaudible clip should score low. Do not default to a middling value."""

SCHEMA_INSTRUCTION = """\
Respond with a single JSON object and nothing else:

{
  "emotional_tone": "neutral|satisfied|frustrated|upset|distressed",
  "emotional_intensity": "low|medium|high",
  "self_confidence": 0.0 to 1.0,
  "rationale": "one sentence citing the specific evidence you used"
}"""


def build_prompt(
    prosody: Prosody | None,
    transcript: str | None,
    duration_sec: float,
    *,
    has_audio: bool,
    emotion: object | None = None,
) -> str:
    """Assemble the tone prompt for whichever evidence this mode allows."""
    sections = [
        "You are analysing a customer-service phone call to determine the "
        "customer's emotional state.",
        "",
        TONE_DEFINITIONS,
        "",
        RULES,
        "",
        f"Call length: {duration_sec:.0f} seconds.",
    ]

    if has_audio:
        sections += [
            "",
            "The audio is attached. Listen to tone of voice, pacing, and "
            "emphasis as well as the words.",
        ]

    if emotion is not None:
        sections += [
            "",
            "Measured affect, from a speech-emotion model run on the waveform:",
            f"  {emotion.describe()}",
            "",
            "These are measurements of delivery, not of meaning. Arousal and "
            "dominance are measured reliably from sound; valence is not, so weigh "
            "what was actually said more heavily than the valence figure when "
            "deciding whether the tone is positive or negative.",
        ]

    if prosody is not None:
        sections += [
            "",
            "Signal-processing measurements of the speech:",
            f"  {prosody.describe()}",
            "",
            "Supporting evidence for intensity only. Do not raise the tone on the "
            "strength of these numbers alone.",
        ]

    if transcript:
        clipped = transcript.strip()
        if len(clipped) > 12_000:
            clipped = clipped[:12_000] + " …[truncated]"
        sections += ["", "Transcript:", clipped]

    if not transcript and not has_audio:
        sections += [
            "",
            "No transcript or audio is available — judge from the acoustic "
            "measurements alone, and score your confidence low accordingly.",
        ]

    sections += ["", SCHEMA_INSTRUCTION]
    return "\n".join(sections)


# Structured-output schema, used where a provider supports constrained decoding.
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "emotional_tone": {
            "type": "string",
            "enum": ["neutral", "satisfied", "frustrated", "upset", "distressed"],
        },
        "emotional_intensity": {
            "type": "string",
            "enum": ["low", "medium", "high"],
        },
        "self_confidence": {"type": "number"},
        "rationale": {"type": "string"},
    },
    "required": ["emotional_tone", "emotional_intensity", "self_confidence"],
}

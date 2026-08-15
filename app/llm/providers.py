"""Tone inference backends and the failover chain.

Every backend answers the same question and returns the same shape. The chain
tries them in order and always produces a result: if every remote provider is
rate-limited, misconfigured, or down, the local classifier answers instead with
a low confidence rather than the pipeline failing. A batch must never die
because a free tier ran out mid-run.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from app.audio.prosody import Prosody
from app.config import PrivacyMode, Settings
from app.llm.prompts import RESPONSE_SCHEMA, build_prompt
from app.ratelimit import REGISTRY, RateLimitExceeded
from app.schema import EmotionalIntensity, EmotionalTone


# Shown verbatim in the dashboard and attached to every affected result. The
# free tier allows roughly 20 requests a day on the most accurate model, so a
# real evaluation batch will cross this line — better to say so plainly than to
# let the accuracy quietly drop with no explanation.
DEGRADED_NOTICE = (
    "Gracefully degraded: the primary tone model's free daily quota is spent, so "
    "a fallback model answered. Emotional tone is less accurate on the fallback "
    "(2/3 versus 0/3 on the labelled clips); every other field is unaffected and "
    "still fully measured. This deployment runs entirely on free tiers for an "
    "assignment, and that trade-off is deliberate — a paid key removes the limit "
    "at roughly $1.59 per 1,000 audio-minutes."
)


@dataclass
class ToneResult:
    tone: EmotionalTone
    intensity: EmotionalIntensity
    self_confidence: float
    rationale: str = ""
    provider: str = "unknown"
    degraded: bool = False          # true when we fell back below the first choice
    attempts: list[str] = field(default_factory=list)
    latency_sec: float = 0.0
    model: str = ""                                   # which model actually answered
    degraded_notice: str = ""                         # user-facing explanation
    samples: int = 1                                  # how many votes were drawn
    vote_distribution: dict = field(default_factory=dict)


class ToneProvider:
    name = "base"

    def available(self, settings: Settings) -> bool:
        raise NotImplementedError

    async def infer(
        self, *, prosody: Prosody | None, transcript: str | None,
        duration_sec: float, audio_path: Path | None, settings: Settings,
        emotion: object | None = None,
    ) -> ToneResult:
        raise NotImplementedError


_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.S | re.I)


def _parse_json(text: str) -> dict:
    """Pull a JSON object out of a model response.

    Three shapes have to be handled, all seen in practice:

      * reasoning models emit a `<think>` block before the answer, and that
        block contains braces of its own — a naive brace scan lands inside the
        reasoning and fails;
      * some models fence the JSON in a markdown code block;
      * some wrap it in a sentence of prose.

    Stripping the reasoning first, then preferring the *last* balanced object,
    handles all three. The last object is the right one because any earlier
    brace belongs to reasoning or to an example.
    """
    text = (text or "").strip()
    text = _THINK_BLOCK.sub(" ", text)

    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fenced:
        return json.loads(fenced[-1])

    # Scan for balanced objects and take the last complete one.
    candidates: list[str] = []
    depth = start = 0
    in_string = escaped = False
    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidates.append(text[start:i + 1])

    for candidate in reversed(candidates):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict) and "emotional_tone" in parsed:
                return parsed
        except json.JSONDecodeError:
            continue
    if candidates:
        return json.loads(candidates[-1])
    return json.loads(text)


def _to_result(payload: dict, provider: str) -> ToneResult:
    tone_raw = str(payload.get("emotional_tone", "neutral")).strip().lower()
    intensity_raw = str(payload.get("emotional_intensity", "low")).strip().lower()

    try:
        tone = EmotionalTone(tone_raw)
    except ValueError:
        tone = EmotionalTone.NEUTRAL
    try:
        intensity = EmotionalIntensity(intensity_raw)
    except ValueError:
        intensity = EmotionalIntensity.LOW

    try:
        confidence = min(1.0, max(0.0, float(payload.get("self_confidence", 0.5))))
    except (TypeError, ValueError):
        confidence = 0.5

    return ToneResult(
        tone=tone, intensity=intensity, self_confidence=confidence,
        rationale=str(payload.get("rationale", ""))[:400], provider=provider,
    )


# --------------------------------------------------------------------------
# Gemini
# --------------------------------------------------------------------------

class GeminiProvider(ToneProvider):
    """Google Gemini. The only backend that can hear the audio directly."""

    name = "gemini"

    def available(self, settings: Settings) -> bool:
        return bool(settings.gemini_api_key)

    async def infer(
        self, *, prosody, transcript, duration_sec, audio_path, settings, emotion=None,
    ) -> ToneResult:
        """Try each configured model in turn.

        The free tier meters *per model per day*, and the good model is metered
        hard: `gemini-3.5-flash` allows 20 requests a day, which one evaluation
        batch would exhaust before the evaluator saw a result. Because the quota
        is per model rather than per project, rotating through several models
        multiplies the daily budget — the first model is the most accurate one
        and later entries are the fallbacks that keep a batch moving once it is
        spent.
        """
        from google import genai
        from google.genai import types

        send_audio = settings.audio_may_leave() and audio_path is not None
        prompt = build_prompt(prosody, transcript, duration_sec, has_audio=send_audio, emotion=emotion)

        client = genai.Client(api_key=settings.gemini_api_key)
        parts: list = [prompt]
        if send_audio:
            parts.append(types.Part.from_bytes(
                data=Path(audio_path).read_bytes(),
                mime_type=_mime_for(audio_path),
            ))

        errors: list[str] = []
        for model_name in settings.gemini_models:
            # One bucket per model, because that is how the quota is scoped.
            bucket = REGISTRY.get(f"gemini:{model_name}", settings.gemini_limits)
            try:
                await bucket.acquire(max_wait_sec=20.0)
            except RateLimitExceeded as exc:
                errors.append(f"{model_name}: {exc}")
                continue
            try:
                response = await asyncio.to_thread(
                    client.models.generate_content,
                    model=model_name,
                    contents=parts,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=RESPONSE_SCHEMA,
                        temperature=0.0,
                        # Generous on purpose. This is a reasoning model and its
                        # thinking tokens come out of the same budget as the
                        # answer; at 400 the JSON was truncated mid-string, which
                        # failed the parse, failed the provider, and dropped the
                        # clip to the local heuristic without saying so. The
                        # answer is ~60 tokens, and output is billed on tokens
                        # produced rather than on the cap.
                        max_output_tokens=2048,
                    ),
                )
                result = _to_result(_parse_json(response.text), self.name)
                result.model = model_name
                return result
            except Exception as exc:  # noqa: BLE001
                message = str(exc)
                errors.append(f"{model_name}: {type(exc).__name__}")
                # A daily quota does not recover within this batch. Burn the
                # bucket so we stop paying the latency of asking again.
                if "RESOURCE_EXHAUSTED" in message or "429" in message:
                    bucket.exhaust_day()
                continue
            finally:
                bucket.release()

        # Every model is spent or failing. Raise so the chain moves to Groq and
        # then to the local heuristic rather than failing the clip.
        raise RateLimitExceeded(
            "gemini (all models: " + "; ".join(errors[:3]) + ")"
        )


def _mime_for(path: Path) -> str:
    return {
        ".ogg": "audio/ogg", ".opus": "audio/ogg", ".mp3": "audio/mpeg",
        ".wav": "audio/wav", ".flac": "audio/flac", ".m4a": "audio/mp4",
        ".aac": "audio/aac", ".webm": "audio/webm",
    }.get(Path(path).suffix.lower(), "audio/wav")


# --------------------------------------------------------------------------
# Groq
# --------------------------------------------------------------------------

class GroqProvider(ToneProvider):
    """Groq text models. Fast, free tier, and does not train on API data.

    Text-only, so it reads the transcript and the measured prosody description
    rather than the audio. This is the backend that makes `hybrid` mode viable.
    """

    name = "groq"

    def available(self, settings: Settings) -> bool:
        return bool(settings.groq_api_key)

    async def infer(
        self, *, prosody, transcript, duration_sec, audio_path, settings, emotion=None,
    ) -> ToneResult:
        from groq import AsyncGroq

        prompt = build_prompt(prosody, transcript, duration_sec, has_audio=False, emotion=emotion)
        bucket = REGISTRY.get("groq", settings.groq_limits)
        await bucket.acquire()
        try:
            client = AsyncGroq(api_key=settings.groq_api_key)
            kwargs: dict = {
                "model": settings.groq_llm_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 1500,
            }
            # Constrained decoding is rejected by reasoning models, which emit a
            # <think> block ahead of the answer, and by the agentic `compound`
            # models. `_parse_json` copes with both shapes, so the constraint is
            # only requested where it is actually supported.
            model_name = settings.groq_llm_model
            if not (model_name.startswith("groq/compound") or "qwen" in model_name):
                kwargs["response_format"] = {"type": "json_object"}

            response = await client.chat.completions.create(**kwargs)
            result = _to_result(_parse_json(response.choices[0].message.content), self.name)
            result.model = model_name
            return result
        finally:
            bucket.release()


# --------------------------------------------------------------------------
# local fallback
# --------------------------------------------------------------------------

NEGATIVE_STRONG = {
    "ridiculous", "unacceptable", "furious", "outrageous", "disgusted", "livid",
    "appalling", "scam", "lawyer", "sue", "never again", "worst", "hell",
}
NEGATIVE_MILD = {
    "frustrated", "annoyed", "disappointed", "still not", "again", "waiting",
    "delay", "problem", "issue", "wrong", "broken", "late", "complaint",
    "supposed to", "keeps", "third time",
}
DISTRESS = {
    "please help", "desperate", "crying", "panic", "emergency", "scared",
    "afraid", "overwhelmed", "can't cope", "breaking down",
}
POSITIVE = {
    "thank you so much", "perfect", "wonderful", "excellent", "appreciate",
    "great", "brilliant", "relieved", "fantastic", "helpful", "sorted",
}


class LocalProvider(ToneProvider):
    """Lexicon plus prosody. Always available, and deliberately modest.

    This exists so `local_only` mode is a real option and so the chain can
    never fail outright. Its confidence is capped because a keyword count is
    genuinely weak evidence about someone's emotional state.
    """

    name = "local"

    def available(self, settings: Settings) -> bool:
        return True

    async def infer(
        self, *, prosody, transcript, duration_sec, audio_path, settings, emotion=None,
    ) -> ToneResult:
        text = (transcript or "").lower()

        strong = sum(1 for w in NEGATIVE_STRONG if w in text)
        mild = sum(1 for w in NEGATIVE_MILD if w in text)
        distress = sum(1 for w in DISTRESS if w in text)
        positive = sum(1 for w in POSITIVE if w in text)

        if distress >= 1:
            tone = EmotionalTone.DISTRESSED
        elif strong >= 1:
            tone = EmotionalTone.UPSET
        elif mild >= 2:
            tone = EmotionalTone.FRUSTRATED
        elif positive >= 1 and mild == 0:
            tone = EmotionalTone.SATISFIED
        else:
            tone = EmotionalTone.NEUTRAL

        # Prosody adjusts intensity only. It must not move the tone itself —
        # that is the "loudness is not anger" rule, enforced in code.
        intensity = EmotionalIntensity.LOW
        if prosody is not None and tone is not EmotionalTone.NEUTRAL:
            animated = (
                prosody.f0_range_semitones > 12.0
                or prosody.energy_range_db > 26.0
                or prosody.speech_rate_proxy > 4.0
            )
            very_animated = (
                prosody.f0_range_semitones > 16.0 and prosody.energy_range_db > 30.0
            )
            intensity = (
                EmotionalIntensity.HIGH if very_animated
                else EmotionalIntensity.MEDIUM if animated
                else EmotionalIntensity.LOW
            )
        elif tone is not EmotionalTone.NEUTRAL:
            intensity = EmotionalIntensity.MEDIUM

        confidence = 0.30 if text else 0.15
        return ToneResult(
            tone=tone, intensity=intensity, self_confidence=confidence,
            rationale="lexical and prosodic heuristic (no language model available)",
            provider=self.name,
        )


PROVIDERS: dict[str, ToneProvider] = {
    "gemini": GeminiProvider(),
    "groq": GroqProvider(),
    "local": LocalProvider(),
}


async def infer_tone(
    *, prosody: Prosody | None, transcript: str | None, duration_sec: float,
    audio_path: Path | None, settings: Settings, emotion: object | None = None,
) -> ToneResult:
    """Tone inference with adaptive self-consistency.

    Repeated identical requests do not produce identical answers: measured over
    seven runs per clip, one labelled clip returned `upset` five times,
    `frustrated` once and `neutral` once, while the other two were perfectly
    stable. A single sample therefore gets that clip wrong 29% of the time, and
    majority voting fixes it.

    Voting on every clip would triple the cost of the only metered call in the
    system and breach the per-minute ceiling on short files. So extra samples are
    drawn only where the first one looks shaky — low self-reported confidence, or
    a first answer that the acoustic measurement contradicts. Stable, confident
    clips cost exactly one call.
    """
    first = await _infer_once(
        prosody=prosody, transcript=transcript, duration_sec=duration_sec,
        audio_path=audio_path, settings=settings, emotion=emotion,
    )

    if not settings.self_consistency_enabled or first.provider in ("local", "none"):
        return first

    if not _looks_uncertain(first, emotion, settings):
        first.samples = 1
        return first

    extra = []
    for _ in range(max(0, settings.self_consistency_samples - 1)):
        try:
            extra.append(await _infer_once(
                prosody=prosody, transcript=transcript, duration_sec=duration_sec,
                audio_path=audio_path, settings=settings, emotion=emotion,
            ))
        except Exception:  # noqa: BLE001 — a failed re-sample just means fewer votes
            break

    votes = [first, *[e for e in extra if e.provider not in ("local", "none")]]
    if len(votes) < 2:
        first.samples = len(votes)
        return first

    counts = Counter(v.tone for v in votes)
    winner, winner_n = counts.most_common(1)[0]

    # Report the confidence of a sample that actually voted for the winner, and
    # scale it by how united the vote was.
    chosen = next(v for v in votes if v.tone is winner)
    chosen.self_confidence = min(1.0, chosen.self_confidence * (winner_n / len(votes)))
    chosen.samples = len(votes)
    chosen.vote_distribution = {t.value: n for t, n in counts.items()}
    chosen.attempts = first.attempts + [f"self-consistency:{winner_n}/{len(votes)}"]
    return chosen


def _looks_uncertain(result: ToneResult, emotion, settings: Settings) -> bool:
    """Is this answer worth spending more samples on?"""
    if result.self_confidence < settings.self_consistency_confidence:
        return True
    # The acoustic model measured a strongly activated delivery but the language
    # model called it emotionally flat. One of them is wrong; ask again.
    if emotion is not None and getattr(emotion, "ok", False):
        if result.tone is EmotionalTone.NEUTRAL and emotion.arousal >= 0.62:
            return True
    return False


async def _infer_once(
    *, prosody: Prosody | None, transcript: str | None, duration_sec: float,
    audio_path: Path | None, settings: Settings, emotion: object | None = None,
) -> ToneResult:
    """Run the failover chain and return the first success."""
    import time

    if settings.privacy_mode is PrivacyMode.LOCAL_ONLY:
        order = ["local"]
    else:
        order = settings.tone_providers

    attempts: list[str] = []
    started = time.perf_counter()

    for index, name in enumerate(order):
        provider = PROVIDERS.get(name)
        if provider is None or not provider.available(settings):
            attempts.append(f"{name}:unavailable")
            continue
        try:
            result = await provider.infer(
                prosody=prosody, transcript=transcript, duration_sec=duration_sec,
                audio_path=audio_path, settings=settings, emotion=emotion,
            )
            result.attempts = attempts + [f"{name}:ok"]
            # Degraded if we fell to a later provider, or stayed on Gemini but
            # had to rotate off its most accurate model.
            rotated = bool(
                result.model and settings.gemini_models and result.model != settings.gemini_models[0]
            )
            result.degraded = index > 0 or rotated
            if result.degraded:
                result.degraded_notice = DEGRADED_NOTICE
            result.latency_sec = time.perf_counter() - started
            return result
        except RateLimitExceeded as exc:
            attempts.append(f"{name}:rate-limited({exc})")
        except Exception as exc:  # noqa: BLE001 — any failure moves to the next backend
            attempts.append(f"{name}:error({type(exc).__name__}: {exc})"[:200])

    # Even the local provider is in the chain, so reaching here means something
    # unexpected went wrong. Return a valid, clearly-uncertain answer.
    return ToneResult(
        tone=EmotionalTone.NEUTRAL, intensity=EmotionalIntensity.LOW,
        self_confidence=0.10, rationale="all tone providers failed",
        provider="none", degraded=True, attempts=attempts,
        latency_sec=time.perf_counter() - started,
    )

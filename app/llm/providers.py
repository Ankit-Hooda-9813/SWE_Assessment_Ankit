"""Tone inference backends and the failover chain.

Every backend answers the same question and returns the same shape. The chain
tries them in order and always produces a result: if every remote provider is
rate-limited, misconfigured, or down, the local classifier answers instead with
a low confidence rather than the pipeline failing. A batch must never die
because a free tier ran out mid-run.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from app.audio.prosody import Prosody
from app.config import PrivacyMode, Settings
from app.llm.prompts import build_prompt
from app.ratelimit import REGISTRY, RateLimitExceeded
from app.schema import EmotionalIntensity, EmotionalTone


# Shown verbatim in the dashboard and attached to every affected result when
# the primary provider (Azure OpenAI) fails and the chain falls to the local
# lexicon+prosody heuristic. Unlike the Gemini-era version of this notice,
# there is no daily quota to blame here — Azure OpenAI is billed per token
# with no free-tier request ceiling, so degradation now means a real outage
# or misconfiguration, not routine quota exhaustion.
DEGRADED_NOTICE = (
    "Gracefully degraded: the primary tone provider failed or is unreachable, "
    "so a local lexicon-and-prosody heuristic answered instead. That heuristic "
    "is measurably weaker than the LLM path; every other field is unaffected "
    "and still fully measured."
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
# Azure OpenAI
# --------------------------------------------------------------------------

class AzureOpenAIProvider(ToneProvider):
    """Azure-hosted OpenAI models. Text-only, same rationale as Groq.

    Paid from the first token — no daily quota to exhaust, unlike Gemini's
    free tier (18-20 requests/day). Azure addresses models by deployment
    name, not the raw model id, so `settings.azure_openai_deployment` must
    match a deployment that actually exists on the resource.
    """

    name = "azure_openai"

    def available(self, settings: Settings) -> bool:
        return bool(settings.azure_openai_api_key and settings.azure_openai_endpoint)

    async def infer(
        self, *, prosody, transcript, duration_sec, audio_path, settings, emotion=None,
    ) -> ToneResult:
        from openai import AsyncAzureOpenAI

        prompt = build_prompt(prosody, transcript, duration_sec, has_audio=False, emotion=emotion)
        client = AsyncAzureOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
        )
        response = await client.chat.completions.create(
            model=settings.azure_openai_deployment,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            # Generous on purpose, same reason as Gemini's cap above: the
            # GPT-5 family spends real tokens on hidden reasoning before the
            # visible answer (measured: 64 reasoning tokens on a two-word
            # reply), and a tight cap truncates to empty content rather than
            # a short answer.
            max_completion_tokens=3000,
        )
        result = _to_result(_parse_json(response.choices[0].message.content), self.name)
        result.model = settings.azure_openai_deployment
        return result


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
    "azure_openai": AzureOpenAIProvider(),
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
            # Degraded only if we fell past the primary provider — there is no
            # per-provider model rotation left to account for now that Azure
            # OpenAI is the sole remote provider (one fixed deployment, no
            # multi-model quota chain the way Gemini needed).
            result.degraded = index > 0
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

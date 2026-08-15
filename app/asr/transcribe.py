"""Transcription backends.

Two paths, chosen by privacy mode and availability:

  * local  — faster-whisper int8 on CPU. Audio never leaves the container, so
             this is the only option that satisfies `local_only`, and it is
             what `hybrid` uses to produce a transcript without transmitting
             the recording.
  * groq   — whisper-large-v3-turbo. Much faster and more accurate, but the
             audio is uploaded, so it is only used when the mode permits it.

Transcription failure is never fatal: the tone stage can run on prosody alone,
at reduced confidence.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from app.config import PrivacyMode, Settings
from app.ratelimit import REGISTRY, RateLimitExceeded


@dataclass
class Transcript:
    text: str
    backend: str
    language: str = ""
    segments: list[dict] = field(default_factory=list)
    latency_sec: float = 0.0
    error: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.text.strip())

    @property
    def word_count(self) -> int:
        return len(self.text.split())


# Phrases Whisper produces from its training data rather than from the audio.
# They come from subtitle corpora and appear over silence, music, and repetitive
# speech. Matched case-insensitively against a whole segment.
HALLUCINATION_PATTERNS = [
    r"thank(s| you) (all |so much |very much )?for watching",
    r"please (like[, ]+)?subscribe",
    r"don'?t forget to subscribe",
    r"subtitles? (by|provided by|amara)",
    r"transcri(bed|ption) by",
    r"see you (in the )?next (video|time)",
    r"^\s*(thanks for watching|bye|thank you)[.!]?\s*$",
    r"amara\.org",
    r"www\.[a-z0-9.-]+\.(com|org|net)",
    r"^\s*\[?\s*(music|applause|silence|blank[_ ]audio)\s*\]?\s*$",
    r"^\s*♪+\s*$",
]
_HALLUCINATION_RE = [re.compile(p, re.IGNORECASE) for p in HALLUCINATION_PATTERNS]


def _strip_hallucination(text: str) -> str:
    """Drop a segment that is boilerplate rather than speech."""
    cleaned = (text or "").strip()
    if not cleaned:
        return ""
    for pattern in _HALLUCINATION_RE:
        if pattern.search(cleaned):
            return ""
    return cleaned


def _collapse_repeats(text: str, limit: int = 4) -> str:
    """Shorten runaway repetition without discarding the fact of it.

    Whisper loops when it loses the thread, so unbounded repetition has to be
    capped. But repetition is often the emotional evidence rather than noise: in
    `call_001` the caller says "Hello" eleven times trying to reach a human, and
    that is precisely why the clip is labelled `upset`.

    An earlier version simply truncated the run to four, which silently deleted
    the signal and moved the prediction from `upset` to `frustrated`. So the run
    is shortened *and annotated* with its true length — compact for the prompt,
    and the model still learns that it happened eleven times, not four.
    """
    words = text.split()
    if not words:
        return text

    out: list[str] = []
    run_word, run = "", 0

    def flush() -> None:
        # Only `limit` copies were ever appended, so nothing needs removing —
        # the annotation simply records how many there really were.
        if run > limit:
            out.append(f"[repeated {run} times]")

    for word in words:
        key = word.lower().strip(".,!?¿¡")
        if key == run_word:
            run += 1
        else:
            flush()
            run_word, run = key, 1
        if run <= limit:
            out.append(word)
    flush()
    return " ".join(out)


_LOCAL_MODEL = None


def _load_local(settings: Settings):
    """Load faster-whisper once per process."""
    global _LOCAL_MODEL
    if _LOCAL_MODEL is None:
        from faster_whisper import WhisperModel
        _LOCAL_MODEL = WhisperModel(
            settings.local_whisper_model,
            device="cpu",
            compute_type=settings.local_whisper_compute,
        )
    return _LOCAL_MODEL


def _transcribe_local_sync(path: Path, settings: Settings) -> Transcript:
    started = time.perf_counter()
    model = _load_local(settings)
    segments, info = model.transcribe(
        str(path),
        beam_size=1,               # greedy: on 2 vCPU the accuracy gain is not worth the time
        vad_filter=True,
        condition_on_previous_text=False,
        # Whisper invents fluent text over silence and over highly repetitive
        # speech. On call_001 — a caller saying "Hello" eleven times — it emitted
        # "Thank you for watching. Please subscribe", which then went into the
        # tone prompt and made an irritated call look like a video sign-off.
        # These three thresholds make it drop low-evidence segments instead of
        # confabulating through them.
        no_speech_threshold=0.6,
        log_prob_threshold=-1.0,
        compression_ratio_threshold=2.4,
        hallucination_silence_threshold=2.0,
    )
    collected = []
    parts = []
    for seg in segments:
        text = _strip_hallucination(seg.text)
        if not text:
            continue
        parts.append(text)
        collected.append({
            "start": round(seg.start, 2), "end": round(seg.end, 2), "text": text,
        })
    return Transcript(
        text=_collapse_repeats(" ".join(parts).strip()),
        backend=f"faster-whisper/{settings.local_whisper_model}",
        language=getattr(info, "language", "") or "",
        segments=collected,
        latency_sec=time.perf_counter() - started,
    )


async def _transcribe_groq(path: Path, settings: Settings) -> Transcript:
    from groq import AsyncGroq

    started = time.perf_counter()
    bucket = REGISTRY.get("groq-asr", settings.groq_limits)
    await bucket.acquire()
    try:
        client = AsyncGroq(api_key=settings.groq_api_key)
        with open(path, "rb") as handle:
            response = await client.audio.transcriptions.create(
                file=(path.name, handle.read()),
                model=settings.groq_asr_model,
                response_format="verbose_json",
            )
        segments = [
            {"start": round(s.get("start", 0), 2), "end": round(s.get("end", 0), 2),
             "text": (s.get("text") or "").strip()}
            for s in (getattr(response, "segments", None) or [])
        ]
        return Transcript(
            text=_collapse_repeats(_strip_hallucination((response.text or "").strip())),
            backend=f"groq/{settings.groq_asr_model}",
            language=getattr(response, "language", "") or "",
            segments=segments,
            latency_sec=time.perf_counter() - started,
        )
    finally:
        bucket.release()


async def transcribe(path: Path, settings: Settings) -> Transcript:
    """Produce a transcript using the best backend this mode allows."""
    path = Path(path)

    # Hosted transcription sends the audio file. It is much faster and more
    # accurate — 0.55 s against 21 s for the same clip — but it is an upload,
    # so it is gated on the privacy mode actually permitting one rather than on
    # a key merely being present. `hybrid` promises the audio stays put, and
    # that promise has to be true by construction.
    prefer_groq = (
        settings.asr_backend in ("auto", "groq")
        and settings.groq_api_key
        and settings.asr_may_upload()
    )

    errors: list[str] = []

    if prefer_groq:
        try:
            return await _transcribe_groq(path, settings)
        except RateLimitExceeded as exc:
            errors.append(f"groq rate-limited: {exc}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"groq failed: {type(exc).__name__}: {exc}")

    if settings.asr_backend in ("auto", "local", "groq"):
        try:
            return await asyncio.to_thread(_transcribe_local_sync, path, settings)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"local whisper failed: {type(exc).__name__}: {exc}")

    return Transcript(
        text="", backend="none", error="; ".join(errors) or "transcription disabled",
    )

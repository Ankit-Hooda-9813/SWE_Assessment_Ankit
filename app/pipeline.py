"""End-to-end analysis of a single clip.

Stage order matters: the free deterministic work runs first and always
succeeds, so even when every network call fails we still have six of the nine
fields measured and a schema-valid result to return.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from app.asr.transcribe import Transcript, transcribe
from app.audio.analyze import analyse_acoustics
from app.audio.diarize import Diarization, customer_transcript, diarize
from app.audio.io import AudioDecodeError, load_clip
from app.config import PrivacyMode, Settings, get_settings, get_thresholds
from app.fusion import fuse
from app.llm.providers import infer_tone
from app.schema import AnalysisResult
from app.ser.emotion import analyse_emotion
from app.ser.emotion2vec_backend import analyse_emotion2vec


@dataclass
class ClipReport:
    name: str
    status: str                       # "ok" | "failed"
    result: AnalysisResult | None = None
    error: str = ""
    duration_sec: float = 0.0
    timings: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)

    def to_row(self) -> dict:
        """Flat representation for the results table and CSV export."""
        row = {"name": self.name, "status": self.status}
        if self.result is not None:
            row.update(self.result.model_dump(mode="json"))
        else:
            row.update({"error": self.error})
        row["duration_sec"] = round(self.duration_sec, 2)
        row["processing_sec"] = round(self.timings.get("total", 0.0), 2)
        return row


_CACHE: dict[str, ClipReport] = {}


def _content_key(path: Path, settings: Settings) -> str:
    """Cache key over file content and the settings that change the answer.

    Re-running an identical batch should cost nothing, which matters when the
    daily free-tier quota is a few hundred calls.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    digest.update(settings.privacy_mode.value.encode())
    digest.update(settings.azure_openai_deployment.encode())
    return digest.hexdigest()


async def analyse_clip(
    path: str | Path,
    settings: Settings | None = None,
    *,
    use_cache: bool = True,
) -> ClipReport:
    """Run the full pipeline over one file, never raising."""
    settings = settings or get_settings()
    path = Path(path)
    started = time.perf_counter()
    timings: dict[str, float] = {}

    try:
        key = _content_key(path, settings) if use_cache else ""
    except OSError as exc:
        return ClipReport(name=path.name, status="failed", error=f"cannot read file: {exc}")

    if use_cache and key in _CACHE:
        cached = _CACHE[key]
        return ClipReport(
            name=path.name, status=cached.status, result=cached.result,
            error=cached.error, duration_sec=cached.duration_sec,
            timings={**cached.timings, "cached": True},
            diagnostics=cached.diagnostics,
        )

    # --- stage 0: decode -------------------------------------------------
    try:
        stage = time.perf_counter()
        clip = load_clip(path)
        timings["decode"] = time.perf_counter() - stage
    except AudioDecodeError as exc:
        return ClipReport(name=path.name, status="failed", error=str(exc))
    except Exception as exc:  # noqa: BLE001
        return ClipReport(
            name=path.name, status="failed",
            error=f"unexpected error reading file: {type(exc).__name__}: {exc}",
        )

    # --- stages 1-2: measured fields (free, deterministic, always runs) ---
    try:
        stage = time.perf_counter()
        analysis = analyse_acoustics(clip, get_thresholds())
        timings["acoustics"] = time.perf_counter() - stage
    except Exception as exc:  # noqa: BLE001
        return ClipReport(
            name=path.name, status="failed", duration_sec=clip.duration_sec,
            error=f"acoustic analysis failed: {type(exc).__name__}: {exc}",
        )

    # --- stage 3: transcript ---------------------------------------------
    # Runs before emotion so its segment timings can help decide who is who.
    stage = time.perf_counter()
    if settings.privacy_mode is PrivacyMode.LOCAL_ONLY and settings.asr_backend == "none":
        transcript = Transcript(text="", backend="disabled")
    else:
        transcript = await transcribe(path, settings)
    timings["asr"] = time.perf_counter() - stage

    # --- stage 4: separate the customer from the agent -------------------
    # The schema asks about the customer, and roughly half of these recordings
    # is the automated agent. Measuring emotion across both averages a synthetic
    # voice into a human one.
    stage = time.perf_counter()
    speakers = (
        diarize(analysis.frames, analysis.vad, transcript.segments)
        if settings.diarization_enabled
        else Diarization(method="disabled")
    )
    timings["diarize"] = time.perf_counter() - stage

    # --- stage 5: measured affect ----------------------------------------
    # Local, so it works in every privacy mode and costs nothing. This is the
    # channel a transcript cannot carry: the same words said calmly and said
    # through gritted teeth transcribe identically.
    stage = time.perf_counter()
    if speakers.confident and speakers.ok:
        emotion_spans = speakers.customer_spans
    else:
        emotion_spans = [(s.start, s.end) for s in analysis.vad.speech]
    emotion = analyse_emotion(clip.samples, emotion_spans)
    timings["emotion"] = time.perf_counter() - stage

    # A second, independent SER opinion — see app/ser/mapping.py for exactly
    # how narrowly this is used. Best-effort: a missing/failed dependency
    # yields no categorical hint, not a failed clip.
    stage = time.perf_counter()
    emotion2vec = None
    if settings.ser_ensemble_enabled:
        emotion2vec = analyse_emotion2vec(clip.samples, emotion_spans)
    timings["emotion2vec"] = time.perf_counter() - stage

    # The customer's own words, where the split is trustworthy.
    customer_text = customer_transcript(transcript.segments, speakers) if speakers.confident else ""
    tone_text = customer_text or transcript.text

    # --- stage 6: emotional tone -----------------------------------------
    stage = time.perf_counter()
    tone = await infer_tone(
        prosody=analysis.prosody,
        transcript=tone_text or None,
        duration_sec=clip.duration_sec,
        audio_path=path if settings.audio_may_leave() else None,
        settings=settings,
        emotion=emotion if emotion.ok else None,
    )
    timings["tone"] = time.perf_counter() - stage

    # --- stage 7: fusion --------------------------------------------------
    fused = fuse(analysis, tone, transcript, emotion, emotion2vec)
    timings["total"] = time.perf_counter() - started

    report = ClipReport(
        name=path.name,
        status="ok",
        result=fused.result,
        duration_sec=clip.duration_sec,
        timings={k: round(v, 3) for k, v in timings.items()},
        diagnostics={
            "confidence": fused.confidence_breakdown,
            "notes": fused.notes,
            "tone_provider": tone.provider,
            "tone_attempts": tone.attempts,
            "transcript": {
                "backend": transcript.backend,
                "words": transcript.word_count,
                "error": transcript.error,
                "preview": transcript.text[:280],
            },
            "emotion": emotion.to_dict(),
            "speakers": speakers.evidence,
            "tone_read_customer_only": bool(customer_text),
            "acoustics": analysis.evidence(),
            "privacy_mode": settings.privacy_mode.value,
            "audio_left_container": settings.audio_may_leave(),
        },
    )

    if use_cache and key:
        _CACHE[key] = report
    return report


def clear_cache() -> None:
    _CACHE.clear()

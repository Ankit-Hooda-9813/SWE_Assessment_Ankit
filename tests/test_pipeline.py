"""Regression tests for the invariants that must not break.

These target the properties an evaluator depends on: a malformed file never
takes down a batch, an invalid enum never reaches a result file, the manifest
validator names what is wrong, and the free-tier limiter actually limits.
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app.audio.io import AudioDecodeError, load_clip
from app.batch import extract_batch, make_workdir, validate_batch
from app.config import ProviderLimits, Settings, Thresholds
from app.ratelimit import RateLimitExceeded, TokenBucket
from app.schema import (
    AnalysisResult,
    AudioQuality,
    EmotionalIntensity,
    EmotionalTone,
    NoiseSeverity,
    coerce_result,
)

SR = 16_000
FIXTURES = Path(__file__).resolve().parent.parent / "requirements"


def _tone_wav(path: Path, seconds: float = 3.0, freq: float = 220.0) -> Path:
    t = np.arange(int(seconds * SR)) / SR
    signal = 0.2 * np.sin(2 * np.pi * freq * t).astype(np.float32)
    sf.write(path, signal, SR, subtype="PCM_16")
    return path


# --------------------------------------------------------------------------
# schema
# --------------------------------------------------------------------------

def test_noise_fields_stay_consistent_when_absent():
    """A result cannot claim no noise while naming a noise type."""
    result = AnalysisResult(
        emotional_tone=EmotionalTone.NEUTRAL,
        emotional_intensity=EmotionalIntensity.LOW,
        background_noise_present=False,
        background_noise_type="office chatter",
        background_noise_severity=NoiseSeverity.HIGH,
        audio_quality=AudioQuality.CLEAR,
        speaker_overlap_present=False,
        long_silence_present=False,
        confidence=0.5,
    )
    assert result.background_noise_type == ""
    assert result.background_noise_severity is NoiseSeverity.NONE


def test_noise_present_never_reports_severity_none():
    result = AnalysisResult(
        emotional_tone=EmotionalTone.NEUTRAL,
        emotional_intensity=EmotionalIntensity.LOW,
        background_noise_present=True,
        background_noise_type="TV",
        background_noise_severity=NoiseSeverity.NONE,
        audio_quality=AudioQuality.CLEAR,
        speaker_overlap_present=False,
        long_silence_present=False,
        confidence=0.5,
    )
    assert result.background_noise_severity is NoiseSeverity.LOW


@pytest.mark.parametrize("garbage", [
    {},
    {"emotional_tone": "FURIOUS", "confidence": "not a number"},
    {"emotional_tone": None, "background_noise_present": "yes", "confidence": 5},
    {"emotional_tone": "Frustrated ", "emotional_intensity": "MEDIUM"},
])
def test_coerce_never_raises_and_always_validates(garbage):
    """Model output is untrusted; a bad field must not cost us the clip."""
    result = coerce_result(garbage)
    assert isinstance(result, AnalysisResult)
    assert 0.0 <= result.confidence <= 1.0
    json.loads(result.to_json())


def test_coerce_repairs_case_and_whitespace():
    result = coerce_result({"emotional_tone": "  Frustrated ", "emotional_intensity": "MEDIUM"})
    assert result.emotional_tone is EmotionalTone.FRUSTRATED
    assert result.emotional_intensity is EmotionalIntensity.MEDIUM


# --------------------------------------------------------------------------
# decoding — failure isolation
# --------------------------------------------------------------------------

def test_missing_file_raises_readable_error(tmp_path):
    with pytest.raises(AudioDecodeError, match="not found"):
        load_clip(tmp_path / "nope.wav")


def test_empty_file_raises_readable_error(tmp_path):
    path = tmp_path / "empty.wav"
    path.write_bytes(b"")
    with pytest.raises(AudioDecodeError, match="empty"):
        load_clip(path)


def test_unsupported_extension_names_the_type(tmp_path):
    path = tmp_path / "notes.pdf"
    path.write_bytes(b"%PDF-1.4 not audio")
    with pytest.raises(AudioDecodeError, match="unsupported file type"):
        load_clip(path)


def test_corrupt_audio_raises_rather_than_returning_garbage(tmp_path):
    path = tmp_path / "broken.wav"
    path.write_bytes(b"RIFFxxxxWAVEnot actually audio at all")
    with pytest.raises(AudioDecodeError):
        load_clip(path)


def test_dual_mono_is_detected(tmp_path):
    """The provided calls are dual mono; overlap detection depends on knowing."""
    t = np.arange(SR * 2) / SR
    mono = (0.2 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    path = tmp_path / "dualmono.wav"
    sf.write(path, np.stack([mono, mono], axis=1), SR, subtype="PCM_16")
    assert load_clip(path).channels_identical is True


# --------------------------------------------------------------------------
# batch validation
# --------------------------------------------------------------------------

def test_validator_reports_both_kinds_of_mismatch(tmp_path):
    batch = tmp_path / "batch"
    batch.mkdir()
    _tone_wav(batch / "present.wav")

    with (batch / "labels.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["name", "result_json"])
        writer.writerow(["present.wav", ""])
        writer.writerow(["absent.wav", ""])   # listed but never uploaded

    _tone_wav(batch / "unlisted.wav")          # uploaded but never listed

    report = validate_batch(batch)
    assert report.missing_audio == ["absent.wav"]
    assert report.unlisted_audio == ["unlisted.wav"]
    assert "absent.wav" in report.summary()


def test_batch_without_manifest_is_still_processable(tmp_path):
    """An unlabelled hidden test set has no result_json, and that is fine."""
    batch = tmp_path / "batch"
    batch.mkdir()
    _tone_wav(batch / "clip.wav")
    report = validate_batch(batch)
    assert report.ok
    assert report.manifest_path is None


def test_batch_with_no_audio_is_rejected_with_a_reason(tmp_path):
    batch = tmp_path / "batch"
    batch.mkdir()
    (batch / "labels.csv").write_text("name,result_json\n")
    report = validate_batch(batch)
    assert not report.ok
    assert "No supported audio files" in report.summary()


def test_malformed_result_json_is_reported_not_raised(tmp_path):
    batch = tmp_path / "batch"
    batch.mkdir()
    _tone_wav(batch / "clip.wav")
    (batch / "labels.csv").write_text('name,result_json\nclip.wav,"{not valid json"\n')
    report = validate_batch(batch)
    assert any("unreadable result_json" in w for w in report.warnings)


def test_zip_upload_is_extracted(tmp_path):
    archive = tmp_path / "batch.zip"
    clip = _tone_wav(tmp_path / "a.wav")
    with zipfile.ZipFile(archive, "w") as zf:
        zf.write(clip, "a.wav")
        zf.writestr("labels.csv", "name,result_json\na.wav,\n")

    workdir = make_workdir()
    try:
        extract_batch([archive], workdir)
        report = validate_batch(workdir)
        assert len(report.audio_files) == 1
        assert report.manifest_path is not None
    finally:
        import shutil
        shutil.rmtree(workdir, ignore_errors=True)


def test_macos_resource_forks_are_ignored(tmp_path):
    """Zipping on macOS adds __MACOSX/._ entries that are not audio."""
    batch = tmp_path / "batch"
    (batch / "__MACOSX").mkdir(parents=True)
    _tone_wav(batch / "real.wav")
    (batch / "__MACOSX" / "._real.wav").write_bytes(b"\x00\x05\x16\x07")
    report = validate_batch(batch)
    assert [p.name for p in report.audio_files] == ["real.wav"]


# --------------------------------------------------------------------------
# rate limiting
# --------------------------------------------------------------------------

def test_limiter_blocks_past_the_minute_budget():
    bucket = TokenBucket("t", ProviderLimits(requests_per_minute=2, requests_per_day=99))

    async def scenario():
        await bucket.acquire(); bucket.release()
        await bucket.acquire(); bucket.release()
        assert bucket.wait_time() > 0
        with pytest.raises(RateLimitExceeded):
            await bucket.acquire(max_wait_sec=0.05)

    asyncio.run(scenario())


def test_limiter_raises_when_the_daily_quota_is_gone():
    bucket = TokenBucket("t", ProviderLimits(requests_per_minute=10, requests_per_day=1))

    async def scenario():
        await bucket.acquire(); bucket.release()
        with pytest.raises(RateLimitExceeded):
            await bucket.acquire(max_wait_sec=0.05)

    asyncio.run(scenario())


def test_limiter_releases_its_slot_on_failure():
    """A raising acquire must not leak the concurrency slot."""
    bucket = TokenBucket("t", ProviderLimits(requests_per_minute=1, requests_per_day=1, max_concurrent=1))

    async def scenario():
        await bucket.acquire(); bucket.release()
        for _ in range(3):
            with pytest.raises(RateLimitExceeded):
                await bucket.acquire(max_wait_sec=0.01)
        assert bucket._slots._value == 1

    asyncio.run(scenario())


# --------------------------------------------------------------------------
# transcript cleaning
# --------------------------------------------------------------------------

def test_whisper_boilerplate_is_stripped():
    from app.asr.transcribe import _strip_hallucination
    assert _strip_hallucination("Thank you for watching. Please subscribe!") == ""
    assert _strip_hallucination("Subtitles by the Amara.org community") == ""
    assert _strip_hallucination("Are you a real person?") == "Are you a real person?"


def test_repetition_is_capped_but_still_reported():
    """The repetition in call_001 is the emotional evidence, not noise."""
    from app.asr.transcribe import _collapse_repeats
    out = _collapse_repeats("Hello. " * 11)
    assert "[repeated 11 times]" in out
    assert out.count("Hello") == 4
    unchanged = "I want an appointment for my car"
    assert _collapse_repeats(unchanged) == unchanged


# --------------------------------------------------------------------------
# fusion rules
# --------------------------------------------------------------------------

def test_dominance_separates_upset_from_distressed():
    from app.ser.emotion import EmotionScores
    from app.ser.mapping import build_prior, reconcile

    yielding = build_prior(EmotionScores(
        arousal=0.70, dominance=0.40, valence=0.35, windows=3, sustained_high_frac=0.5))
    tone, _intensity, notes = reconcile(
        EmotionalTone.UPSET, EmotionalIntensity.HIGH, yielding)
    assert tone is EmotionalTone.DISTRESSED
    assert any("overwhelmed" in n for n in notes)


def test_measured_arousal_overrides_the_language_model_on_intensity():
    from app.ser.emotion import EmotionScores
    from app.ser.mapping import build_prior, reconcile

    prior = build_prior(EmotionScores(
        arousal=0.70, dominance=0.65, valence=0.40, windows=3, sustained_high_frac=0.5))
    _tone, intensity, _notes = reconcile(
        EmotionalTone.FRUSTRATED, EmotionalIntensity.LOW, prior)
    assert intensity is EmotionalIntensity.HIGH


def test_neutral_is_withdrawn_on_strongly_activated_delivery():
    from app.ser.emotion import EmotionScores
    from app.ser.mapping import build_prior, reconcile

    prior = build_prior(EmotionScores(
        arousal=0.72, dominance=0.66, valence=0.42, windows=3, sustained_high_frac=0.6))
    tone, _intensity, _notes = reconcile(
        EmotionalTone.NEUTRAL, EmotionalIntensity.LOW, prior)
    assert tone is not EmotionalTone.NEUTRAL


def test_missing_emotion_scores_leave_the_model_answer_alone():
    from app.ser.mapping import reconcile
    tone, intensity, notes = reconcile(
        EmotionalTone.SATISFIED, EmotionalIntensity.MEDIUM, None)
    assert (tone, intensity, notes) == (
        EmotionalTone.SATISFIED, EmotionalIntensity.MEDIUM, [])


# --------------------------------------------------------------------------
# acoustics on the real clips
# --------------------------------------------------------------------------

@pytest.mark.skipif(not (FIXTURES / "call_001.ogg").exists(), reason="fixtures absent")
def test_measured_fields_on_the_labelled_clips():
    """Guards the six free fields against regression. Tone needs an API key."""
    from app.audio.analyze import analyse_acoustics

    expected = {
        r["name"]: json.loads(r["result_json"])
        for r in csv.DictReader((FIXTURES / "labels.csv").open())
    }
    fields = [
        "background_noise_present", "background_noise_type",
        "background_noise_severity", "audio_quality", "long_silence_present",
    ]
    hits = 0
    for name, truth in expected.items():
        measured = analyse_acoustics(load_clip(FIXTURES / name)).measured_fields()
        for field in fields:
            value = measured[field]
            hits += str(getattr(value, "value", value)) == str(truth[field])

    # 15 of 15 at the time of writing. Overlap is excluded: it is the known-weak
    # detector and is tracked separately rather than pinned here.
    assert hits >= 14, f"measured-field regression: {hits}/15"

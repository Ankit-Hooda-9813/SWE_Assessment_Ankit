"""Decoding audio to a canonical form.

Everything downstream assumes 16 kHz mono float32 in [-1, 1]. ffmpeg does the
decoding because it accepts every container the brief mentions and several it
does not, and because it fails loudly on corrupt input instead of returning
half a waveform.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.config import SUPPORTED_EXTENSIONS, TARGET_SR


class AudioDecodeError(Exception):
    """Raised when a file cannot be turned into a waveform.

    Carries a message intended for the dashboard's per-file error column, so it
    must stay readable by a non-engineer.
    """


@dataclass(frozen=True)
class Clip:
    samples: np.ndarray  # mono float32, TARGET_SR
    sample_rate: int
    duration_sec: float
    source_path: Path
    channels: int          # channel count before downmix
    source_sample_rate: int
    channels_identical: bool  # true when a "stereo" file is really dual mono

    @property
    def name(self) -> str:
        return self.source_path.name


def _ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise AudioDecodeError("ffmpeg is not installed on this server")
    return exe


def _ffprobe() -> str | None:
    return shutil.which("ffprobe")


def probe(path: Path) -> dict[str, str]:
    exe = _ffprobe()
    if not exe:
        return {}
    proc = subprocess.run(
        [exe, "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=channels,sample_rate,codec_name",
         "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1", str(path)],
        capture_output=True, text=True,
    )
    out: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _decode(path: Path, channels: int) -> np.ndarray:
    proc = subprocess.run(
        [_ffmpeg(), "-v", "error", "-nostdin", "-i", str(path),
         "-f", "f32le", "-acodec", "pcm_f32le",
         "-ac", str(channels), "-ar", str(TARGET_SR), "-"],
        capture_output=True,
    )
    if proc.returncode != 0:
        detail = proc.stderr.decode("utf-8", "replace").strip().splitlines()
        reason = detail[-1] if detail else "unknown decoder error"
        raise AudioDecodeError(f"could not decode audio: {reason}")
    buf = np.frombuffer(proc.stdout, dtype="<f4")
    if buf.size == 0:
        raise AudioDecodeError("file contains no decodable audio")
    return buf.astype(np.float32, copy=True)


def load_clip(path: str | Path, *, max_seconds: float | None = None) -> Clip:
    """Decode a file to a canonical mono clip.

    Stereo is inspected before downmixing: if the two channels turn out to be
    identical the file is dual mono, which means there is no per-speaker channel
    to exploit and overlap detection has to be done acoustically.
    """
    path = Path(path)
    if not path.exists():
        raise AudioDecodeError("file not found")
    if path.stat().st_size == 0:
        raise AudioDecodeError("file is empty")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        ext = path.suffix or "(none)"
        raise AudioDecodeError(f"unsupported file type '{ext}'")

    meta = probe(path)
    try:
        src_channels = int(meta.get("channels", "1"))
    except ValueError:
        src_channels = 1
    try:
        src_sr = int(meta.get("sample_rate", str(TARGET_SR)))
    except ValueError:
        src_sr = TARGET_SR

    channels_identical = False
    if src_channels >= 2:
        stereo = _decode(path, 2)
        usable = (stereo.size // 2) * 2
        pair = stereo[:usable].reshape(-1, 2)
        # A tiny tolerance: lossy codecs perturb identical channels slightly.
        channels_identical = bool(np.max(np.abs(pair[:, 0] - pair[:, 1])) < 0.01)
        mono = pair.mean(axis=1).astype(np.float32)
    else:
        mono = _decode(path, 1)

    if not np.isfinite(mono).all():
        mono = np.nan_to_num(mono, nan=0.0, posinf=0.0, neginf=0.0)

    if max_seconds is not None:
        limit = int(max_seconds * TARGET_SR)
        if mono.size > limit:
            mono = mono[:limit]

    duration = mono.size / TARGET_SR
    if duration < 0.25:
        raise AudioDecodeError(f"clip is too short to analyse ({duration:.2f}s)")

    return Clip(
        samples=mono,
        sample_rate=TARGET_SR,
        duration_sec=duration,
        source_path=path,
        channels=src_channels,
        source_sample_rate=src_sr,
        channels_identical=channels_identical,
    )

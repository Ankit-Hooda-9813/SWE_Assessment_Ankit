"""Batch validation and processing.

Implements the evaluation workflow the brief specifies: accept a folder or ZIP
containing audio files at the root plus a CSV manifest, validate the pairing,
report what does not match, process everything valid, and never let one bad file
take down the run.
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import shutil
import tempfile
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from app.config import SUPPORTED_EXTENSIONS, Settings, get_settings
from app.pipeline import ClipReport, analyse_clip
from app.schema import SCHEMA_FIELDS, coerce_result

MANIFEST_COLUMNS = ("name", "result_json")


@dataclass
class ValidationReport:
    audio_files: list[Path] = field(default_factory=list)
    manifest_rows: dict[str, dict] = field(default_factory=dict)
    manifest_path: Path | None = None

    missing_audio: list[str] = field(default_factory=list)     # in CSV, no file
    unlisted_audio: list[str] = field(default_factory=list)    # file, not in CSV
    unsupported: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.audio_files) and not self.problems

    def summary(self) -> str:
        lines = [f"Found {len(self.audio_files)} audio file(s) ready to process."]
        if self.manifest_path:
            lines.append(f"Manifest: {self.manifest_path.name} ({len(self.manifest_rows)} row(s)).")
        else:
            lines.append("No CSV manifest found — running unlabelled, which is expected for a hidden test set.")
        for label, items in (
            ("Listed in the manifest but not uploaded", self.missing_audio),
            ("Uploaded but not listed in the manifest", self.unlisted_audio),
            ("Unsupported file type, skipped", self.unsupported),
        ):
            if items:
                shown = ", ".join(items[:10])
                more = f" (+{len(items) - 10} more)" if len(items) > 10 else ""
                lines.append(f"{label}: {shown}{more}")
        lines.extend(self.warnings)
        lines.extend(f"Problem: {p}" for p in self.problems)
        return "\n".join(lines)


def _iter_files(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.name.startswith("._") and "__MACOSX" not in path.parts:
            yield path


def extract_batch(sources: list[str | Path], workdir: Path) -> Path:
    """Materialise the upload into a single directory.

    Accepts a ZIP, a set of loose files, or a mixture, because the browser file
    picker produces different shapes on different platforms.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    for source in sources:
        source = Path(source)
        if source.suffix.lower() == ".zip":
            try:
                with zipfile.ZipFile(source) as archive:
                    archive.extractall(workdir)
            except zipfile.BadZipFile:
                (workdir / f"UNREADABLE_{source.name}").write_text("bad zip")
        elif source.is_dir():
            shutil.copytree(source, workdir / source.name, dirs_exist_ok=True)
        else:
            shutil.copy2(source, workdir / source.name)
    return workdir


def _read_manifest(path: Path) -> tuple[dict[str, dict], list[str]]:
    """Parse the CSV manifest, tolerating imperfect input."""
    rows: dict[str, dict] = {}
    warnings: list[str] = []

    text = path.read_text(encoding="utf-8-sig", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    headers = [h.strip().lower() for h in (reader.fieldnames or [])]

    if "name" not in headers:
        warnings.append(
            f"{path.name} has no 'name' column (found: {', '.join(headers) or 'nothing'}); "
            "labels will be ignored."
        )
        return rows, warnings

    has_labels = "result_json" in headers
    if not has_labels:
        warnings.append(f"{path.name} has no 'result_json' column — treating as an unlabelled batch.")

    for line_no, raw in enumerate(reader, start=2):
        normalised = {(k or "").strip().lower(): v for k, v in raw.items()}
        name = (normalised.get("name") or "").strip()
        if not name:
            continue
        entry: dict = {"row": line_no}
        payload = (normalised.get("result_json") or "").strip() if has_labels else ""
        if payload:
            try:
                entry["labels"] = coerce_result(json.loads(payload)).model_dump(mode="json")
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                warnings.append(f"{path.name} row {line_no} ({name}): unreadable result_json — {exc}")
        rows[name] = entry
    return rows, warnings


def validate_batch(root: Path) -> ValidationReport:
    """Check the uploaded batch and describe exactly what is wrong with it."""
    report = ValidationReport()

    manifests = [p for p in _iter_files(root) if p.suffix.lower() == ".csv"]
    audio: list[Path] = []
    for path in _iter_files(root):
        suffix = path.suffix.lower()
        if suffix in SUPPORTED_EXTENSIONS:
            audio.append(path)
        elif suffix not in {".csv", ".txt", ".md", ".json"}:
            report.unsupported.append(path.name)

    report.audio_files = audio

    if len(manifests) > 1:
        report.warnings.append(
            f"Multiple CSV files found ({', '.join(m.name for m in manifests)}); "
            f"using {manifests[0].name}."
        )
    if manifests:
        report.manifest_path = manifests[0]
        report.manifest_rows, warnings = _read_manifest(manifests[0])
        report.warnings.extend(warnings)

    if not audio:
        report.problems.append(
            "No supported audio files found. Expected the clips at the root of the "
            f"folder or ZIP. Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}."
        )
        return report

    if report.manifest_rows:
        uploaded = {p.name for p in audio}
        listed = set(report.manifest_rows)
        report.missing_audio = sorted(listed - uploaded)
        report.unlisted_audio = sorted(uploaded - listed)

    return report


@dataclass
class BatchProgress:
    total: int = 0
    done: int = 0
    failed: int = 0
    current: str = ""
    message: str = ""
    started_at: float = field(default_factory=time.time)

    @property
    def elapsed(self) -> float:
        return time.time() - self.started_at

    @property
    def eta_sec(self) -> float:
        if self.done == 0:
            return 0.0
        return (self.elapsed / self.done) * (self.total - self.done)


@dataclass
class BatchResult:
    reports: list[ClipReport] = field(default_factory=list)
    validation: ValidationReport | None = None
    elapsed_sec: float = 0.0
    audio_seconds: float = 0.0

    @property
    def succeeded(self) -> list[ClipReport]:
        return [r for r in self.reports if r.status == "ok"]

    @property
    def failed(self) -> list[ClipReport]:
        return [r for r in self.reports if r.status != "ok"]

    def rows(self) -> list[dict]:
        return [r.to_row() for r in self.reports]

    def to_csv(self) -> str:
        """Results as CSV, preserving the original filename.

        Emits both the flat columns and a `result_json` column, so the output can
        be fed straight back in as a manifest.
        """
        buffer = io.StringIO()
        columns = ["name", "status", *SCHEMA_FIELDS, "duration_sec", "processing_sec", "error", "result_json"]
        writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for report in self.reports:
            row = report.to_row()
            row["result_json"] = report.result.to_json() if report.result else ""
            row.setdefault("error", "")
            writer.writerow(row)
        return buffer.getvalue()

    def to_json(self) -> str:
        return json.dumps(
            [
                {
                    "name": r.name,
                    "status": r.status,
                    "result": r.result.model_dump(mode="json") if r.result else None,
                    "error": r.error or None,
                    "duration_sec": round(r.duration_sec, 2),
                    "timings": r.timings,
                    "diagnostics": r.diagnostics,
                }
                for r in self.reports
            ],
            indent=2,
        )


async def run_batch(
    root: Path,
    settings: Settings | None = None,
    *,
    progress_cb=None,
) -> BatchResult:
    """Process every valid clip, reporting progress as it goes."""
    settings = settings or get_settings()
    validation = validate_batch(root)
    result = BatchResult(validation=validation)

    if not validation.audio_files:
        return result

    files = validation.audio_files[: settings.max_batch_files]
    if len(validation.audio_files) > settings.max_batch_files:
        validation.warnings.append(
            f"Batch capped at {settings.max_batch_files} files; "
            f"{len(validation.audio_files) - settings.max_batch_files} were not processed."
        )

    progress = BatchProgress(total=len(files))
    semaphore = asyncio.Semaphore(max(1, settings.worker_concurrency))
    started = time.perf_counter()

    async def worker(path: Path) -> ClipReport:
        async with semaphore:
            progress.current = path.name
            if progress_cb:
                progress_cb(progress)
            size_mb = path.stat().st_size / (1 << 20)
            if size_mb > settings.max_file_mb:
                report = ClipReport(
                    name=path.name, status="failed",
                    error=f"file is {size_mb:.0f} MB, over the {settings.max_file_mb} MB limit",
                )
            else:
                report = await analyse_clip(path, settings)
            progress.done += 1
            if report.status != "ok":
                progress.failed += 1
            if progress_cb:
                progress_cb(progress)
            return report

    reports = await asyncio.gather(*(worker(p) for p in files), return_exceptions=True)

    for path, report in zip(files, reports):
        if isinstance(report, BaseException):
            # A crash in a worker is still a per-file failure, never a batch failure.
            result.reports.append(ClipReport(
                name=path.name, status="failed",
                error=f"unhandled error: {type(report).__name__}: {report}",
            ))
        else:
            result.reports.append(report)

    result.elapsed_sec = time.perf_counter() - started
    result.audio_seconds = sum(r.duration_sec for r in result.reports)
    return result


def make_workdir() -> Path:
    """A scratch directory for one upload. Deleted as soon as the batch ends."""
    return Path(tempfile.mkdtemp(prefix="autoace_batch_"))


def cleanup(path: Path) -> None:
    """Remove uploaded audio.

    The free tier has no persistent storage, and here that is a feature: customer
    audio exists on disk only for the seconds it takes to analyse it.
    """
    shutil.rmtree(path, ignore_errors=True)

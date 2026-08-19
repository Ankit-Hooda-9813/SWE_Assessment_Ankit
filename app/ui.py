"""The evaluation dashboard.

Built for one job: an evaluator logs in, drops in a folder or ZIP, watches it
process, reads the results, and downloads them — without touching a command
line. Everything on screen is in service of that path.

Two decisions worth noting. Validation runs before any processing, so an
evaluator learns their batch is malformed in a second rather than after a
ten-minute run. And free-tier pacing is displayed rather than hidden: a labelled
countdown reads as a deliberate system, an unexplained stall reads as a broken
one.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

import gradio as gr
import pandas as pd

from app.batch import BatchProgress, cleanup, extract_batch, make_workdir, run_batch, validate_batch
from app.config import PrivacyMode, get_settings
from app.ratelimit import REGISTRY

TABLE_COLUMNS = [
    "name", "status", "emotional_tone", "emotional_intensity",
    "background_noise_present", "background_noise_type", "background_noise_severity",
    "audio_quality", "speaker_overlap_present", "long_silence_present",
    "confidence", "duration_sec", "processing_sec",
]

CSS = """
.autoace-header { border-bottom: 1px solid var(--border-color-primary); padding-bottom: 12px; margin-bottom: 4px; }
.autoace-header h1 { font-size: 1.45rem; margin: 0 0 4px; }
.autoace-header p { color: var(--body-text-color-subdued); margin: 0; font-size: 0.92rem; }
.mono textarea { font-family: ui-monospace, "SF Mono", Menlo, monospace !important; font-size: 12.5px !important; }
.privacy-note { font-size: 0.85rem; color: var(--body-text-color-subdued); }
footer { display: none !important; }
"""


def _mode_banner() -> str:
    settings = get_settings()
    mode = settings.privacy_mode
    if mode is PrivacyMode.LOCAL_ONLY:
        detail = "**local_only** — nothing leaves this container. Transcription and tone both run on-box."
    elif mode is PrivacyMode.HYBRID:
        detail = (
            "**hybrid** — audio never leaves this container. Only the transcript and "
            "numeric acoustic measurements are sent to the language model."
        )
    else:
        detail = (
            "**full** — audio is sent to the multimodal model. Highest expected tone "
            "accuracy; customer audio leaves this container."
        )
    return f"Privacy mode: {detail}"


def _empty_table() -> pd.DataFrame:
    return pd.DataFrame(columns=TABLE_COLUMNS)


def do_validate(files):
    """Inspect the upload and report problems before anything is processed."""
    if not files:
        return "Select a ZIP archive, or the audio files and CSV together.", gr.update(interactive=False)

    workdir = make_workdir()
    try:
        extract_batch([f.name if hasattr(f, "name") else f for f in files], workdir)
        report = validate_batch(workdir)
        prefix = "Ready to process.\n\n" if report.ok else "This batch has problems.\n\n"
        return prefix + report.summary(), gr.update(interactive=report.ok)
    finally:
        cleanup(workdir)


def do_process(files, progress=gr.Progress()):
    """Run the batch, streaming progress to the UI."""
    if not files:
        yield _empty_table(), "Nothing uploaded.", None, None, ""
        return

    settings = get_settings()
    workdir = make_workdir()

    try:
        extract_batch([f.name if hasattr(f, "name") else f for f in files], workdir)
        validation = validate_batch(workdir)
        if not validation.audio_files:
            yield _empty_table(), validation.summary(), None, None, ""
            return

        state = {"line": "Starting…"}

        def on_progress(p: BatchProgress) -> None:
            wait = REGISTRY.soonest_wait()
            pacing = f" · free-tier pacing, next call in {wait:.0f}s" if wait > 1 else ""
            eta = f" · about {p.eta_sec:.0f}s left" if p.done else ""
            state["line"] = (
                f"{p.done}/{p.total} processed"
                f"{f' · {p.failed} failed' if p.failed else ''}"
                f" · now: {p.current}{eta}{pacing}"
            )
            progress(p.done / max(p.total, 1), desc=state["line"])

        yield _empty_table(), f"Processing {len(validation.audio_files)} file(s)…", None, None, ""

        result = asyncio.run(run_batch(workdir, settings, progress_cb=on_progress))

        frame = pd.DataFrame(result.rows())
        for column in TABLE_COLUMNS:
            if column not in frame.columns:
                frame[column] = ""
        frame = frame[TABLE_COLUMNS]

        out_dir = Path(tempfile.mkdtemp(prefix="autoace_out_"))
        csv_path = out_dir / "autoace_results.csv"
        json_path = out_dir / "autoace_results.json"
        csv_path.write_text(result.to_csv())
        json_path.write_text(result.to_json())

        throughput = (
            result.audio_seconds / result.elapsed_sec if result.elapsed_sec else 0.0
        )
        summary_lines = [
            f"Processed {len(result.succeeded)} of {len(result.reports)} file(s) "
            f"in {result.elapsed_sec:.1f}s.",
            f"{result.audio_seconds / 60:.1f} minutes of audio · "
            f"{result.elapsed_sec / max(result.audio_seconds / 60, 1e-6):.1f}s per audio-minute "
            f"({throughput:.1f}x realtime).",
        ]
        if result.failed:
            summary_lines.append("")
            summary_lines.append("Files that could not be processed:")
            summary_lines += [f"  · {r.name} — {r.error}" for r in result.failed]
        # A degraded batch must announce itself. An evaluator seeing lower tone
        # accuracy deserves to know it came from a spent free-tier quota rather
        # than from the method.
        notices = {
            r.diagnostics.get("confidence", {}).get("degraded_notice")
            for r in result.reports
        }
        notices.discard(None)
        notices.discard("")
        if notices:
            degraded_count = sum(
                1 for r in result.reports
                if r.diagnostics.get("confidence", {}).get("degraded")
            )
            summary_lines.append("")
            summary_lines.append("=" * 62)
            summary_lines.append(f"NOTICE - {degraded_count} of {len(result.reports)} clip(s) degraded")
            summary_lines.append("=" * 62)
            for notice in notices:
                summary_lines.append(notice)

        summary_lines.append("")
        summary_lines.append(validation.summary())

        diagnostics = json.dumps(
            {
                "rate_limiters": REGISTRY.snapshot(),
                "per_file": {
                    r.name: {"timings": r.timings, "notes": r.diagnostics.get("notes", []),
                             "tone_provider": r.diagnostics.get("tone_provider")}
                    for r in result.reports
                },
            },
            indent=2,
        )

        yield frame, "\n".join(summary_lines), str(csv_path), str(json_path), diagnostics

    finally:
        # Uploaded audio is removed as soon as the batch ends. The free tier has
        # no persistent storage and here that is a privacy property, not a limit.
        cleanup(workdir)


def build_ui() -> gr.Blocks:
    settings = get_settings()

    with gr.Blocks(title="AutoAce · Voice Tone & Noise Analysis", analytics_enabled=False) as demo:
        # Injected as markup rather than the Blocks css= argument, which Gradio 6
        # removed in favour of a launch()-time parameter we never reach when mounting.
        gr.HTML(f"<style>{CSS}</style>")
        gr.HTML(
            '<div class="autoace-header">'
            "<h1>Voice Tone &amp; Background Noise Analysis</h1>"
            "<p>Upload an evaluation batch, run it, and download structured results.</p>"
            "</div>"
        )
        gr.Markdown(_mode_banner(), elem_classes="privacy-note")

        with gr.Tab("Batch analysis"):
            with gr.Row():
                with gr.Column(scale=1):
                    files = gr.File(
                        label="Evaluation batch",
                        file_count="multiple",
                        file_types=[".zip", ".wav", ".mp3", ".ogg", ".opus", ".m4a", ".flac", ".webm", ".csv"],
                        height=190,
                    )
                    gr.Markdown(
                        "A ZIP containing the clips at its root plus a `labels.csv` "
                        "manifest, or the files selected directly. The manifest needs "
                        "a `name` column and a `result_json` column; for an unlabelled "
                        "test set `result_json` may be empty or absent.",
                        elem_classes="privacy-note",
                    )
                    validate_btn = gr.Button("Check batch", variant="secondary")
                    run_btn = gr.Button("Run analysis", variant="primary", interactive=False)

                with gr.Column(scale=2):
                    status = gr.Textbox(
                        label="Status", lines=11, interactive=False,
                        value="Upload a batch to begin.", elem_classes="mono",
                    )

            results = gr.Dataframe(
                value=_empty_table(), label="Results", wrap=True,
                interactive=False, max_height=460,
            )

            with gr.Row():
                csv_out = gr.File(label="Download CSV", interactive=False)
                json_out = gr.File(label="Download JSON", interactive=False)

            with gr.Accordion("Diagnostics", open=False):
                diagnostics = gr.Code(label="Per-file timings, provider chain, rate limits", language="json")

            validate_btn.click(do_validate, inputs=[files], outputs=[status, run_btn])
            files.change(
                lambda: ("Upload received. Run 'Check batch' to validate it.", gr.update(interactive=False)),
                outputs=[status, run_btn],
            )
            run_btn.click(
                do_process, inputs=[files],
                outputs=[results, status, csv_out, json_out, diagnostics],
            )

        with gr.Tab("How this works"):
            gr.Markdown(f"""
### What is measured versus inferred

Six of the nine output fields come from deterministic signal processing that
runs locally, costs nothing, and returns identical output on repeated runs:

| Field | How it is decided |
|---|---|
| `background_noise_present` | Windowed local dynamic range with an absolute audibility gate |
| `background_noise_type` | Random forest over the estimated noise spectrum |
| `background_noise_severity` | Share of the call affected, and by how much |
| `audio_quality` | Clipping, band edge, level, packet loss, reverberation |
| `speaker_overlap_present` | Cepstral pitch competition — **the weakest field, see limitations** |
| `long_silence_present` | Gap analysis between speech segments |

Only `emotional_tone` and `emotional_intensity` use a language model.
`confidence` is computed from the agreement between all of them, not copied
from the model.

The model is never asked about noise or audio quality. That is what prevents it
inferring frustration from loudness, or background noise from poor audio — the
two failure modes the specification calls out.

### Current configuration

| Setting | Value |
|---|---|
| Privacy mode | `{settings.privacy_mode.value}` |
| Tone provider order | `{', '.join(settings.tone_providers)}` |
| Tone model | `{settings.azure_openai_deployment}` |
| Transcription | `{settings.asr_backend}` |
| Worker concurrency | `{settings.worker_concurrency}` |

### Pacing

Tone runs against Azure OpenAI, billed per token with no daily request quota
to exhaust — a batch cannot run out of budget mid-run the way a free-tier
chain could. Transcription still throttles below Groq's published free-tier
limits rather than discovering them through errors. Identical files are
cached by content hash, so re-running a batch costs nothing.

### Handling of uploaded audio

Uploaded files are written to a temporary directory and deleted the moment the
batch finishes. Nothing is persisted between runs.
""")

    return demo


def mount_auth(demo: gr.Blocks):
    settings = get_settings()
    return demo, (settings.dashboard_user, settings.dashboard_password)

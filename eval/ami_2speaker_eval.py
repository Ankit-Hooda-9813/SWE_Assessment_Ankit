"""Validate `speaker_overlap_present` against Trelis/ami-2speaker-test.

`Trelis/ami-2speaker-test` (HuggingFace, CC-BY-4.0) — a 50-clip benchmark of
real conversational meeting audio from the AMI Corpus test split,
reconstructed as 2-speaker virtual meetings with per-speaker start/end times
and a computed `overlap_ratio`. Verified real against the HF Hub API before
use. This is a *different* AMI adapter from `eval/ami_eval.py` (which reads
the raw AMI Corpus directly for `long_silence_present` and a first overlap
check on 27 windows) — this one is a purpose-built overlap benchmark, an
independent second measurement on a different 50-clip sample.

Ground truth follows this system's own definition, not a raw ratio cutoff:
`overlap_ratio * clip_duration_sec` is compared against `overlap_min_sec`
(`config.py`, 0.35s — the same standard `harper_valley_eval.py` already
matches its own ground truth to), so "present" here means the same thing
this system means by it, not just "ratio > 0".

Class balance matters for reading the result honestly: 42 of 50 clips have
some overlap (84%), so a trivial always-predict-overlap baseline scores
0.84 on this set — reported alongside the measured accuracy, not left for
the reader to compute themselves.

Tests whatever overlap backend is currently wired into
`app/audio/overlap.py` (cepstral by default; pyannote if configured) —
this script has no WavLM dependency itself.
"""

from __future__ import annotations

import argparse
import asyncio
import tempfile
from pathlib import Path

import soundfile as sf
from datasets import load_dataset

from app.config import get_settings, get_thresholds
from app.pipeline import analyse_clip


async def main(n: int) -> None:
    ds = load_dataset("Trelis/ami-2speaker-test", split="train")
    settings = get_settings()
    min_sec = get_thresholds().overlap_min_sec
    workdir = Path(tempfile.mkdtemp(prefix="ami2spk_"))

    n = min(n, len(ds))
    hit = 0
    truth_pos = 0
    tp = fp = fn = tn = 0
    for i in range(n):
        row = ds[i]
        samples = row["audio"].get_all_samples()
        data = samples.data.numpy()[0]
        sr = samples.sample_rate
        duration = samples.duration_seconds

        overlap_sec = row["overlap_ratio"] * duration
        truth = overlap_sec >= min_sec
        truth_pos += truth

        clip_path = workdir / f"clip_{i:02d}.wav"
        sf.write(clip_path, data, sr)

        report = await analyse_clip(clip_path, settings=settings, use_cache=False)
        if report.status != "ok" or not report.result:
            print(f"  clip_{i:02d}: FAILED {report.error}")
            continue

        pred = report.result.speaker_overlap_present
        ok = pred == truth
        hit += ok
        if truth and pred:
            tp += 1
        elif not truth and pred:
            fp += 1
        elif truth and not pred:
            fn += 1
        else:
            tn += 1
        print(
            f"  clip_{i:02d} (ratio={row['overlap_ratio']:.2f}, dur={duration:.1f}s, "
            f"overlap_sec={overlap_sec:.2f}): truth={truth} pred={pred}"
        )

    baseline = truth_pos / n
    print(f"\n=== SUMMARY (AMI 2-speaker overlap benchmark, n={n}) ===")
    print(f"speaker_overlap_present: {hit}/{n} = {hit/n:.3f}")
    print(f"trivial always-predict-overlap baseline on this sample: {baseline:.3f}")
    print(f"confusion: tp={tp} fp={fp} fn={fn} tn={tn}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50)
    args = parser.parse_args()
    asyncio.run(main(args.n))

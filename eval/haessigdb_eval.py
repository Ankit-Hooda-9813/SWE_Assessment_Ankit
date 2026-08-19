"""Validate `emotional_tone` / `emotional_intensity` against HaessigDB.

`nwllr/haessigDB` (HuggingFace, Apache-2.0) — acted, call-center-style English
speech: four professional actors performing scripted banking-support calls
where the customer grows increasingly irritated. Each sentence-level clip has
crowdsourced 1-10 ordinal ratings on three dimensions: aggression,
frustration, annoyance. Verified real and downloadable directly against the
HF Hub API before use, not assumed from a third-party description.

Two honest limitations, stated up front rather than discovered mid-analysis:

1. **Acted, not real production calls.** Same caveat this repo already
   applies to RAVDESS/CREMA-D-style corpora — useful as a controlled
   escalation benchmark, not evidence of real-call distribution.
2. **No positive-affect dimension exists in the source annotation at all.**
   The dataset is specifically an *irritation* benchmark — it has no
   `satisfied` signal, so this eval can only meaningfully test the
   neutral -> frustrated -> upset escalation ladder, not positive polarity.
   Predictions of `satisfied` or `distressed` are scored as simple mismatches
   against the derived ground truth below, not specially excluded, but the
   dataset was never designed to validate either of those specifically.

Ground truth is a **derived, disclosed mapping**, not HaessigDB's own
annotation — it doesn't ship an `emotional_tone` enum, so one has to be
constructed from the three rating dimensions. The source's own recommended
threshold (1-3 low, 4-6 medium, 7-10 high) is used directly:

  * `emotional_intensity` <- threshold on peak(aggression, frustration, annoyance)
  * `emotional_tone` <- peak aggression or frustration >=7 -> `upset`;
    >=4 -> `frustrated`; else `neutral`

Sentences are grouped by (actor, call) and concatenated into one clip per
call, because this system's tone judgement — LLM reading a transcript plus
measured escalation over the whole clip — is built and calibrated for
whole-call audio, not isolated 2-4 second utterances. Scoring at the
sentence level would repeat the exact granularity mismatch already diagnosed
for MELD. "Peak" rather than "mean" reflects the dataset's own design: the
customer is scripted to escalate through the call, so the call's dominant
emotional state is its worst point, not an average that dilutes it.
"""

from __future__ import annotations

import argparse
import asyncio
import random
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from huggingface_hub import hf_hub_download

from app.config import Settings
from app.pipeline import analyse_clip
from app.schema import EmotionalIntensity, EmotionalTone

REPO = "nwllr/haessigDB"


def derive_truth(peak_aggression: float, peak_frustration: float, peak_annoyance: float):
    peak = max(peak_aggression, peak_frustration, peak_annoyance)
    if peak >= 7:
        intensity = EmotionalIntensity.HIGH
    elif peak >= 4:
        intensity = EmotionalIntensity.MEDIUM
    else:
        intensity = EmotionalIntensity.LOW

    if max(peak_aggression, peak_frustration) >= 7:
        tone = EmotionalTone.UPSET
    elif max(peak_aggression, peak_frustration) >= 4:
        tone = EmotionalTone.FRUSTRATED
    else:
        tone = EmotionalTone.NEUTRAL
    return tone, intensity


async def main(n: int, seed: int) -> None:
    ratings_path = hf_hub_download(REPO, "all_ratings.csv", repo_type="dataset")
    df = pd.read_csv(ratings_path)

    calls = list(df.groupby(["actor", "call"]).groups.keys())
    rng = random.Random(seed)
    chosen = rng.sample(calls, min(n, len(calls)))
    print(f"testing {len(chosen)} synthetic HaessigDB calls (seed={seed})")

    settings = Settings()
    workdir = Path(tempfile.mkdtemp(prefix="haessigdb_"))

    tone_hit = intensity_hit = total = 0
    for actor, call in chosen:
        rows = df[(df.actor == actor) & (df.call == call)].sort_values("sentence")
        peak_agg = rows.aggression.max()
        peak_fru = rows.frustration.max()
        peak_ann = rows.annoyance.max()
        truth_tone, truth_intensity = derive_truth(peak_agg, peak_fru, peak_ann)

        chunks = []
        sr = None
        for fname in rows.file_name:
            local = hf_hub_download(REPO, f"audio/{fname}", repo_type="dataset")
            data, file_sr = sf.read(local)
            if sr is None:
                sr = file_sr
            elif file_sr != sr:
                continue  # skip a mismatched-rate sentence rather than corrupt the concat
            if data.ndim > 1:
                data = data.mean(axis=1)
            chunks.append(data)
        full = np.concatenate(chunks)
        clip_path = workdir / f"actor{actor}_call{call}.wav"
        sf.write(clip_path, full, sr)

        report = await analyse_clip(clip_path, settings=settings, use_cache=False)
        if report.status != "ok" or not report.result:
            print(f"  actor{actor}_call{call}: FAILED {report.error}")
            continue

        total += 1
        pred_tone = report.result.emotional_tone
        pred_intensity = report.result.emotional_intensity
        tone_ok = pred_tone is truth_tone
        intensity_ok = pred_intensity is truth_intensity
        tone_hit += tone_ok
        intensity_hit += intensity_ok
        print(
            f"  actor{actor}_call{call} ({len(rows)} sentences, peak agg={peak_agg:.1f} "
            f"fru={peak_fru:.1f} ann={peak_ann:.1f}): "
            f"tone truth={truth_tone.value} pred={pred_tone.value} | "
            f"intensity truth={truth_intensity.value} pred={pred_intensity.value}"
        )

    print("\n=== SUMMARY (HaessigDB, derived escalation-ladder ground truth) ===")
    print(f"emotional_tone: {tone_hit}/{total} = {tone_hit/total:.3f}")
    print(f"emotional_intensity: {intensity_hit}/{total} = {intensity_hit/total:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=25)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    asyncio.run(main(args.n, args.seed))

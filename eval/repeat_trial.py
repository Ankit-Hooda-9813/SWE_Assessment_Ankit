"""Run the full pipeline N times per known call and report the average.

A single run is not evidence on a system already documented (in
app/llm/providers.py's own docstring) to be non-deterministic even at
temperature 0. This script exists so a claimed fix is backed by a mean and a
per-run breakdown, not one lucky sample.
"""

from __future__ import annotations

import asyncio
import json
import warnings
from collections import Counter
from pathlib import Path

warnings.filterwarnings("ignore")

from app.pipeline import analyse_clip
from app.schema import SCHEMA_FIELDS

REQ_DIR = Path("/Users/ankitspc/Work/SWE_Assessment/requirements")
TRUTH = {
    "call_001.ogg": {"emotional_tone": "upset", "emotional_intensity": "high",
                      "background_noise_present": False, "background_noise_type": "",
                      "background_noise_severity": "none", "audio_quality": "clear",
                      "speaker_overlap_present": False, "long_silence_present": False},
    "call_002.ogg": {"emotional_tone": "neutral", "emotional_intensity": "medium",
                      "background_noise_present": True, "background_noise_type": "TV",
                      "background_noise_severity": "medium", "audio_quality": "clear",
                      "speaker_overlap_present": True, "long_silence_present": False},
    "call_003.ogg": {"emotional_tone": "satisfied", "emotional_intensity": "medium",
                      "background_noise_present": True, "background_noise_type": "sharp static",
                      "background_noise_severity": "medium", "audio_quality": "clear",
                      "speaker_overlap_present": True, "long_silence_present": False},
}
SCORED = [f for f in SCHEMA_FIELDS if f != "confidence"]


async def main(n_trials: int) -> None:
    per_call_field_hits: dict[str, dict[str, list[int]]] = {
        name: {f: [] for f in SCORED} for name in TRUTH
    }
    tone_answers: dict[str, list[str]] = {name: [] for name in TRUTH}

    for trial in range(1, n_trials + 1):
        print(f"\n########## TRIAL {trial}/{n_trials} ##########")
        for name in TRUTH:
            report = await analyse_clip(REQ_DIR / name, use_cache=False)
            result = report.result.model_dump(mode="json") if report.result else {}
            row = []
            for f in SCORED:
                correct = int(str(result.get(f)) == str(TRUTH[name][f]))
                per_call_field_hits[name][f].append(correct)
                row.append(f"{f}={result.get(f)!r}{'✓' if correct else '✗'}")
            tone_answers[name].append(str(result.get("emotional_tone")))
            cb = report.diagnostics.get("confidence", {})
            print(f"{name}: llm_raw={cb.get('llm_tone_before_fusion')!r} "
                  f"final_tone={result.get('emotional_tone')!r} "
                  f"(truth={TRUTH[name]['emotional_tone']!r}) "
                  f"notes={report.diagnostics.get('notes')}")

    print("\n\n===================== SUMMARY over", n_trials, "trials =====================")
    print(f"{'FIELD':<28}", *[f"{name.replace('.ogg',''):>12}" for name in TRUTH], f"{'AVG':>8}")
    field_overall_avgs = []
    for f in SCORED:
        row_avgs = []
        for name in TRUTH:
            hits = per_call_field_hits[name][f]
            row_avgs.append(sum(hits) / len(hits))
        overall = sum(row_avgs) / len(row_avgs)
        field_overall_avgs.append(overall)
        print(f"{f:<28}", *[f"{v:>12.2f}" for v in row_avgs], f"{overall:>8.2f}")
    print(f"\nALL FIELDS AVERAGE ACCURACY: {sum(field_overall_avgs)/len(field_overall_avgs):.3f}")

    print("\ntone answer distribution per call across trials:")
    for name in TRUTH:
        print(f"  {name}: {dict(Counter(tone_answers[name]))} (truth={TRUTH[name]['emotional_tone']!r})")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()
    asyncio.run(main(args.trials))

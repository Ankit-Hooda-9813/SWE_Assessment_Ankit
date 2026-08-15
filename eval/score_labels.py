"""Score predictions against a labelled manifest.

Takes the same batch shape the dashboard accepts — audio plus a CSV with `name`
and `result_json` — runs the pipeline, and reports per-field accuracy, macro-F1,
and a confusion matrix. This is what produces the validation numbers in the
technical memo, and it is the command AutoAce can run on their own labelled set.
"""

from __future__ import annotations

import asyncio
import csv
import json
from collections import Counter
from pathlib import Path

from app.batch import run_batch
from app.config import get_settings
from app.schema import INFERRED_FIELDS, MEASURED_FIELDS

SCORED_FIELDS = MEASURED_FIELDS + [f for f in INFERRED_FIELDS if f != "confidence"]


def load_labels(manifest: Path) -> dict[str, dict]:
    labels: dict[str, dict] = {}
    with manifest.open(encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            name = (row.get("name") or "").strip()
            payload = (row.get("result_json") or "").strip()
            if name and payload:
                try:
                    labels[name] = json.loads(payload)
                except json.JSONDecodeError:
                    continue
    return labels


def macro_f1(truth: list[str], pred: list[str]) -> tuple[float, dict]:
    classes = sorted(set(truth) | set(pred))
    per_class = {}
    scores = []
    for cls in classes:
        tp = sum(1 for t, p in zip(truth, pred) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(truth, pred) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(truth, pred) if t == cls and p != cls)
        support = sum(1 for t in truth if t == cls)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[cls] = {"precision": round(precision, 3), "recall": round(recall, 3),
                          "f1": round(f1, 3), "support": support}
        if support:
            scores.append(f1)
    return (sum(scores) / len(scores) if scores else 0.0), per_class


async def main(batch_dir: Path, manifest: Path | None) -> None:
    settings = get_settings()
    manifest = manifest or (batch_dir / "labels.csv")
    labels = load_labels(manifest)
    if not labels:
        raise SystemExit(f"no usable labels found in {manifest}")

    result = await run_batch(batch_dir, settings)
    predictions = {
        r.name: r.result.model_dump(mode="json")
        for r in result.reports if r.result is not None
    }

    shared = [n for n in labels if n in predictions]
    print(f"scored {len(shared)} clip(s); {len(result.failed)} failed to process")
    print(f"privacy mode: {settings.privacy_mode.value} · "
          f"{result.elapsed_sec / max(result.audio_seconds / 60, 1e-6):.1f}s per audio-minute\n")

    print(f"{'FIELD':<30} {'ACC':>8} {'MACRO-F1':>10}")
    print("-" * 50)
    overall_hits = overall_total = 0
    details = {}
    for field in SCORED_FIELDS:
        truth = [str(labels[n].get(field)) for n in shared]
        pred = [str(predictions[n].get(field)) for n in shared]
        hits = sum(1 for t, p in zip(truth, pred) if t == p)
        overall_hits += hits
        overall_total += len(shared)
        f1, per_class = macro_f1(truth, pred)
        details[field] = (per_class, list(zip(shared, truth, pred)))
        print(f"{field:<30} {hits / max(len(shared), 1):>8.3f} {f1:>10.3f}")
    print("-" * 50)
    print(f"{'ALL FIELDS':<30} {overall_hits / max(overall_total, 1):>8.3f}")

    print("\nDisagreements:")
    any_wrong = False
    for field, (_per_class, rows) in details.items():
        for name, truth, pred in rows:
            if truth != pred:
                any_wrong = True
                print(f"  {name:<22} {field:<28} predicted {pred!r}, labelled {truth!r}")
    if not any_wrong:
        print("  none")

    # Confidence calibration: are high-confidence predictions actually better?
    print("\nConfidence vs correctness:")
    buckets: dict[str, list[float]] = {}
    for name in shared:
        confidence = float(predictions[name].get("confidence", 0))
        correct = sum(
            1 for f in SCORED_FIELDS
            if str(labels[name].get(f)) == str(predictions[name].get(f))
        ) / len(SCORED_FIELDS)
        bucket = f"{int(confidence * 10) / 10:.1f}"
        buckets.setdefault(bucket, []).append(correct)
    for bucket in sorted(buckets):
        values = buckets[bucket]
        print(f"  confidence ~{bucket}: {sum(values) / len(values):.2f} field accuracy (n={len(values)})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Score predictions against a labelled batch")
    parser.add_argument("batch", type=Path, help="folder containing audio + labels.csv")
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()
    asyncio.run(main(args.batch, args.manifest))

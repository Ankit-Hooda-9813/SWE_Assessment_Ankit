"""Score the measured fields against the synthetic dev set.

Reports macro-F1, per-class precision and recall, and a confusion matrix per
field. Grouping by speech source is available so a split never puts clips built
from the same speaker on both sides.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.audio.analyze import analyse_acoustics
from app.audio.io import load_clip
from app.config import Thresholds


@dataclass
class FieldScore:
    field: str
    accuracy: float
    macro_f1: float
    per_class: dict[str, dict[str, float]]
    confusion: dict[str, dict[str, int]]
    support: int


def _prf(truth: list[str], pred: list[str]) -> tuple[float, dict[str, dict[str, float]]]:
    classes = sorted(set(truth) | set(pred))
    per_class: dict[str, dict[str, float]] = {}
    f1s = []
    for cls in classes:
        tp = sum(1 for t, p in zip(truth, pred) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(truth, pred) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(truth, pred) if t == cls and p != cls)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        support = sum(1 for t in truth if t == cls)
        per_class[cls] = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "support": support,
        }
        # Only classes that actually occur in the reference contribute to macro-F1;
        # otherwise a class the model invents once drags the average down twice.
        if support:
            f1s.append(f1)
    return (float(np.mean(f1s)) if f1s else 0.0), per_class


def score_field(field: str, truth: list[str], pred: list[str]) -> FieldScore:
    macro_f1, per_class = _prf(truth, pred)
    confusion: dict[str, dict[str, int]] = {}
    for t, p in zip(truth, pred):
        confusion.setdefault(t, Counter())[p] += 1
    return FieldScore(
        field=field,
        accuracy=float(np.mean([t == p for t, p in zip(truth, pred)])) if truth else 0.0,
        macro_f1=macro_f1,
        per_class=per_class,
        confusion={k: dict(v) for k, v in confusion.items()},
        support=len(truth),
    )


def evaluate(
    devset: Path, thresholds: Thresholds | None = None, limit: int | None = None
) -> tuple[dict[str, FieldScore], list[dict]]:
    manifest = json.loads((devset / "manifest.json").read_text())
    if limit:
        manifest = manifest[:limit]

    rows: list[dict] = []
    for spec in manifest:
        path = devset / spec["name"]
        try:
            analysis = analyse_acoustics(load_clip(path), thresholds)
        except Exception as exc:  # a generator bug must not look like a detector bug
            rows.append({"name": spec["name"], "error": str(exc)})
            continue
        measured = analysis.measured_fields()
        rows.append({
            "name": spec["name"],
            "group": spec["name"].split("_")[0],
            "source": spec["source"],
            "truth": spec,
            "pred": {
                k: (v.value if hasattr(v, "value") else v) for k, v in measured.items()
            },
            "evidence": analysis.evidence(),
        })

    ok = [r for r in rows if "error" not in r]
    scores: dict[str, FieldScore] = {}

    # Each field is scored on the group built to exercise it. Scoring the noise
    # detector on the silence clips would report a meaningless 100% from the
    # negatives alone.
    field_groups = {
        "background_noise_present": "noise",
        "background_noise_type": "noise",
        "background_noise_severity": "noise",
        "audio_quality": "qual",
        "speaker_overlap_present": "ovlp",
        "long_silence_present": "sil",
    }

    for field, group in field_groups.items():
        subset = [r for r in ok if r["group"] == group]
        if field == "background_noise_type":
            # Only meaningful where noise is genuinely present.
            subset = [r for r in subset if r["truth"]["background_noise_present"]]
        if not subset:
            continue
        truth = [str(r["truth"][field]) for r in subset]
        pred = [str(r["pred"][field]) for r in subset]
        scores[field] = score_field(field, truth, pred)

    return scores, rows


def print_report(scores: dict[str, FieldScore]) -> None:
    print("\n" + "=" * 78)
    print(f"{'FIELD':<30} {'N':>5} {'ACC':>8} {'MACRO-F1':>10}")
    print("=" * 78)
    for score in scores.values():
        print(f"{score.field:<30} {score.support:>5} {score.accuracy:>8.3f} {score.macro_f1:>10.3f}")

    for score in scores.values():
        print(f"\n--- {score.field} ---")
        for cls, metrics in sorted(score.per_class.items()):
            if metrics["support"] == 0 and metrics["precision"] == 0:
                continue
            print(f"  {cls:<20} P={metrics['precision']:.3f} R={metrics['recall']:.3f} "
                  f"F1={metrics['f1']:.3f}  n={metrics['support']}")
        labels = sorted(set(score.confusion) | {p for row in score.confusion.values() for p in row})
        header = "  truth\\pred".ljust(20) + "".join(f"{lbl[:11]:>13}" for lbl in labels)
        print(header)
        for truth_label in labels:
            row = score.confusion.get(truth_label, {})
            cells = "".join(f"{row.get(pred_label, 0):>13}" for pred_label in labels)
            print(f"  {truth_label[:18]:<18}{cells}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate measured fields on the dev set")
    parser.add_argument("--devset", default="data/devset", type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dump", type=Path, default=None)
    args = parser.parse_args()

    field_scores, all_rows = evaluate(args.devset, limit=args.limit)
    print_report(field_scores)

    if args.dump:
        args.dump.write_text(json.dumps(
            {
                "scores": {k: v.__dict__ for k, v in field_scores.items()},
                "rows": all_rows,
            },
            indent=2, default=str,
        ))
        print(f"\nwrote {args.dump}")

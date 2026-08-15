"""Fit the background-noise severity classifier.

Severity was the weakest measured field at 0.40 macro-F1 under hand-written
thresholds on dwell fraction. The definition — how much the noise actually
interferes with the call — depends on how much of the call is affected *and*
how badly, and a single cut on dwell cannot express that surface.

The features are the aggregate statistics the noise detector already computes,
so this adds no new signal extraction, only a better decision boundary.
Validation is grouped by speech source, as elsewhere.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold

from app.audio.features import build_frames
from app.audio.io import load_clip
from app.audio.noise import estimate_windows, severity_features
from app.audio.vad import detect_voice
from app.config import Thresholds

SEVERITY_FEATURE_NAMES = [
    "dwell_frac", "dwell_sec", "noisy_windows", "informative_windows",
    "worst_range_db", "median_range_db", "median_noise_dbfs", "max_noise_dbfs",
    "audible_frac", "duration_sec",
]


def build_dataset(devset: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    manifest = json.loads((devset / "manifest.json").read_text())
    th = Thresholds()

    features, labels, groups = [], [], []
    for spec in manifest:
        # Only the noise group exercises severity; the others are all "none" and
        # would let the model score well by predicting the majority class.
        if not spec["name"].startswith("noise"):
            continue
        clip = load_clip(devset / spec["name"])
        fb = build_frames(clip.samples, clip.sample_rate)
        vad = detect_voice(fb, th)
        windows = estimate_windows(fb, vad, th)
        features.append(severity_features(windows, clip.duration_sec))
        labels.append(spec["background_noise_severity"])
        groups.append(spec["source"])

    return np.array(features), np.array(labels), np.array(groups)


def main(devset: Path, out_path: Path, seed: int = 20250808) -> None:
    X, y, groups = build_dataset(devset)
    print(f"dataset: {X.shape[0]} clips, {X.shape[1]} features, "
          f"classes {sorted(set(y))}, {len(set(groups))} groups")

    splitter = GroupKFold(n_splits=min(len(set(groups)), 3))
    predictions = np.empty_like(y)
    for train_idx, test_idx in splitter.split(X, y, groups):
        model = RandomForestClassifier(
            n_estimators=400, min_samples_leaf=3, class_weight="balanced",
            random_state=seed, n_jobs=-1,
        )
        model.fit(X[train_idx], y[train_idx])
        predictions[test_idx] = model.predict(X[test_idx])

    print("\n--- grouped cross-validation ---")
    print(classification_report(y, predictions, zero_division=0))
    classes = sorted(set(y))
    print("confusion (rows = truth):")
    matrix = confusion_matrix(y, predictions, labels=classes)
    width = max(len(c) for c in classes) + 2
    print(" " * width + "".join(f"{c:>10}" for c in classes))
    for label, row in zip(classes, matrix):
        print(f"{label:<{width}}" + "".join(f"{v:>10}" for v in row))

    final = RandomForestClassifier(
        n_estimators=400, min_samples_leaf=3, class_weight="balanced",
        random_state=seed, n_jobs=-1,
    )
    final.fit(X, y)
    print("\nfeature importance:")
    for name, importance in sorted(
        zip(SEVERITY_FEATURE_NAMES, final.feature_importances_), key=lambda kv: -kv[1]
    ):
        print(f"  {name:<22} {importance:.3f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump(final, out_path)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the noise-severity classifier")
    parser.add_argument("--devset", default="data/devset", type=Path)
    parser.add_argument("--out", default="app/models/noise_severity.joblib", type=Path)
    args = parser.parse_args()
    main(args.devset, args.out)

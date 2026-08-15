"""Fit the noise-type classifier on the synthetic dev set.

Replaces hand-weighted scoring rules, which were tuned against the two noisy
clips in the provided set and did not generalise — they collapsed six of the
eight noise categories onto "TV" and "music".

Validation is grouped by speech source. A clip built from call_003's speech
never appears in both the training and the held-out half, so the reported score
is not measuring how well the model memorised one speaker's room.
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
from app.audio.noise import FEATURE_NAMES, estimate_windows, noise_features
from app.audio.vad import detect_voice
from app.config import Thresholds


def build_dataset(devset: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Extract noise features for every clip that genuinely contains noise."""
    manifest = json.loads((devset / "manifest.json").read_text())
    th = Thresholds()

    features: list[np.ndarray] = []
    labels: list[str] = []
    groups: list[str] = []
    names: list[str] = []

    for spec in manifest:
        if not spec["background_noise_present"]:
            continue
        clip = load_clip(devset / spec["name"])
        fb = build_frames(clip.samples, clip.sample_rate)
        vad = detect_voice(fb, th)
        windows = estimate_windows(fb, vad, th)
        noisy = [i for i, w in enumerate(windows) if w.noisy]
        if not noisy:
            # The presence detector missed it; there is nothing to describe.
            continue
        features.append(noise_features(fb, windows, noisy))
        labels.append(spec["background_noise_type"])
        groups.append(spec["source"])
        names.append(spec["name"])

    return np.array(features), np.array(labels), np.array(groups), names


def main(devset: Path, out_path: Path, seed: int = 20250808) -> None:
    X, y, groups, _names = build_dataset(devset)
    print(f"dataset: {X.shape[0]} noisy clips, {X.shape[1]} features, "
          f"{len(set(y))} classes, {len(set(groups))} source groups")
    if X.shape[0] < 40:
        raise SystemExit("not enough noisy clips — generate a larger dev set first")

    n_splits = min(len(set(groups)), 3)
    splitter = GroupKFold(n_splits=n_splits)

    predictions = np.empty_like(y)
    for train_idx, test_idx in splitter.split(X, y, groups):
        model = RandomForestClassifier(
            n_estimators=400, min_samples_leaf=2, class_weight="balanced",
            random_state=seed, n_jobs=-1,
        )
        model.fit(X[train_idx], y[train_idx])
        predictions[test_idx] = model.predict(X[test_idx])

    print("\n--- grouped cross-validation (held-out speech source) ---")
    print(classification_report(y, predictions, zero_division=0))
    classes = sorted(set(y))
    print("confusion (rows = truth):")
    matrix = confusion_matrix(y, predictions, labels=classes)
    width = max(len(c) for c in classes) + 2
    print(" " * width + "".join(f"{c[:10]:>12}" for c in classes))
    for label, row in zip(classes, matrix):
        print(f"{label:<{width}}" + "".join(f"{v:>12}" for v in row))

    final = RandomForestClassifier(
        n_estimators=400, min_samples_leaf=2, class_weight="balanced",
        random_state=seed, n_jobs=-1,
    )
    final.fit(X, y)

    importances = sorted(
        zip(FEATURE_NAMES, final.feature_importances_), key=lambda kv: -kv[1]
    )
    print("\nfeature importance:")
    for name, importance in importances:
        print(f"  {name:<16} {importance:.3f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump(final, out_path)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the noise-type classifier")
    parser.add_argument("--devset", default="data/devset", type=Path)
    parser.add_argument("--out", default="app/models/noise_type.joblib", type=Path)
    args = parser.parse_args()
    main(args.devset, args.out)

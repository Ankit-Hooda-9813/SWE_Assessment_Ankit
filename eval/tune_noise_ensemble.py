"""Check whether soft-voting beats the hard-switch to the PANNs-augmented model.

`eval/train_noise_panns.py` measured the combined-feature model at 0.829
macro-F1 against the spectral-only model's 0.690 on the 79-clip noisy dev
set, and `classify_type()` was wired to use the combined model exclusively
whenever it's present. That hard switch was also measured (manually, against
call_002) to flip one real, correctly-classified call to wrong: PANNs itself
returns a near-useless top tag on that clip's residual, and the combined
model weighted it too heavily.

This checks a specific alternative: average both models' predict_proba
output (soft voting) instead of fully replacing one with the other, using
proper GroupKFold so the comparison is on the same footing as the original
measurement, not just eyeballing one clip.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GroupKFold

from app.audio.features import build_frames
from app.audio.io import load_clip
from app.audio.noise import estimate_windows, noise_features
from app.audio.noise_panns import LABEL_GROUPS, classify_type_panns
from app.audio.vad import detect_voice
from app.config import Thresholds

VOCAB = list(LABEL_GROUPS.keys())


def build_dataset(devset: Path):
    manifest = json.loads((devset / "manifest.json").read_text())
    th = Thresholds()

    spectral_feats, panns_feats, labels, groups = [], [], [], []
    for spec in manifest:
        if not spec["background_noise_present"]:
            continue
        clip = load_clip(devset / spec["name"])
        fb = build_frames(clip.samples, clip.sample_rate)
        vad = detect_voice(fb, th)
        windows = estimate_windows(fb, vad, th)
        noisy = [i for i, w in enumerate(windows) if w.noisy]
        if not noisy:
            continue
        spectral_feats.append(noise_features(fb, windows, noisy))
        panns = classify_type_panns(fb, windows, noisy)
        panns_feats.append(panns.to_vector(VOCAB) if panns.ok else np.zeros(len(VOCAB)))
        labels.append(spec["background_noise_type"])
        groups.append(spec["source"])

    return (
        np.vstack(spectral_feats), np.vstack(panns_feats),
        np.array(labels), np.array(groups),
    )


def _cv_soft_vote(name, spectral_X, panns_X, y, groups, seed, weight_combined):
    """GroupKFold CV where the prediction is a weighted average of two
    separately-trained models' predict_proba, not one model on stacked
    features."""
    combined_X = np.hstack([spectral_X, panns_X])
    n_splits = min(len(set(groups)), 3)
    splitter = GroupKFold(n_splits=n_splits)
    predictions = np.empty_like(y)

    for train_idx, test_idx in splitter.split(spectral_X, y, groups):
        spec_model = RandomForestClassifier(
            n_estimators=400, min_samples_leaf=2, class_weight="balanced",
            random_state=seed, n_jobs=-1,
        )
        spec_model.fit(spectral_X[train_idx], y[train_idx])

        comb_model = RandomForestClassifier(
            n_estimators=400, min_samples_leaf=2, class_weight="balanced",
            random_state=seed, n_jobs=-1,
        )
        comb_model.fit(combined_X[train_idx], y[train_idx])

        # Two models can have differently-ordered .classes_; align by label.
        classes = sorted(set(y[train_idx]))
        spec_idx = {c: list(spec_model.classes_).index(c) for c in classes if c in spec_model.classes_}
        comb_idx = {c: list(comb_model.classes_).index(c) for c in classes if c in comb_model.classes_}

        spec_proba = spec_model.predict_proba(spectral_X[test_idx])
        comb_proba = comb_model.predict_proba(combined_X[test_idx])

        for row_i, test_i in enumerate(test_idx):
            blended = {}
            for c in classes:
                sp = spec_proba[row_i, spec_idx[c]] if c in spec_idx else 0.0
                cp = comb_proba[row_i, comb_idx[c]] if c in comb_idx else 0.0
                blended[c] = (1 - weight_combined) * sp + weight_combined * cp
            predictions[test_i] = max(blended, key=blended.get)

    macro_f1 = f1_score(y, predictions, average="macro", zero_division=0)
    print(f"\n--- {name}: grouped CV macro-F1 = {macro_f1:.3f} ---")
    print(classification_report(y, predictions, zero_division=0))
    return macro_f1, predictions


def main(devset: Path, seed: int = 20250808) -> None:
    warnings.filterwarnings("ignore")
    spectral_X, panns_X, y, groups = build_dataset(devset)
    print(f"dataset: {len(y)} noisy clips, {len(set(y))} classes")

    results = {}
    for weight in (0.0, 0.3, 0.5, 0.7, 1.0):
        label = {0.0: "spectral only", 1.0: "combined-features only (shipped)"}.get(
            weight, f"soft vote, combined weight={weight}"
        )
        f1, _ = _cv_soft_vote(label, spectral_X, panns_X, y, groups, seed, weight)
        results[weight] = f1

    print("\n=== summary ===")
    for weight, f1 in results.items():
        print(f"  combined weight {weight:.1f}: macro-F1 {f1:.3f}")
    best_weight = max(results, key=results.get)
    print(f"\nbest: combined weight {best_weight} (macro-F1 {results[best_weight]:.3f})")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--devset", default="data/devset", type=Path)
    args = parser.parse_args()
    main(args.devset)

"""Compare the shipped noise-type classifier against a PANNs-augmented one.

Same discipline as `eval/train_quality.py`: `eval/train_noise.py`'s fitted RF
on the 12-dimensional spectral feature vector is the shipped default. This
script only replaces it if adding PANNs CNN14 scores (see
`app/audio/noise_panns.py`) beats it by a real margin on grouped CV, not
because a bigger pretrained model sounds like it should help.
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
from app.audio.noise import FEATURE_NAMES, estimate_windows, noise_features
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


def _cv_macro_f1(name: str, X: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int) -> float:
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
    macro_f1 = f1_score(y, predictions, average="macro", zero_division=0)
    print(f"\n--- {name}: grouped CV macro-F1 = {macro_f1:.3f} ---")
    print(classification_report(y, predictions, zero_division=0))
    return macro_f1


def main(devset: Path, out_path: Path, seed: int = 20250808) -> None:
    warnings.filterwarnings("ignore")
    spectral_X, panns_X, y, groups = build_dataset(devset)
    print(f"dataset: {len(y)} noisy clips, {len(set(y))} classes, "
          f"{len(set(groups))} groups: {sorted(set(groups))}")

    spectral_f1 = _cv_macro_f1("shipped spectral features only", spectral_X, y, groups, seed)
    combined_X = np.hstack([spectral_X, panns_X])
    combined_f1 = _cv_macro_f1("spectral + PANNs features", combined_X, y, groups, seed)
    panns_only_f1 = _cv_macro_f1("PANNs features only", panns_X, y, groups, seed)

    print("\n=== summary ===")
    print(f"shipped spectral features (app/models/noise_type.joblib basis): {spectral_f1:.3f}")
    print(f"spectral + PANNs combined:                                      {combined_f1:.3f}")
    print(f"PANNs features alone:                                           {panns_only_f1:.3f}")
    if combined_f1 > spectral_f1 + 0.02:
        print(f"\nPANNs ensemble improves on the shipped classifier by a real margin "
              f"({combined_f1:.3f} vs {spectral_f1:.3f}) — fitting final model on all data.")
        final = RandomForestClassifier(
            n_estimators=400, min_samples_leaf=2, class_weight="balanced",
            random_state=seed, n_jobs=-1,
        )
        final.fit(combined_X, y)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        import joblib
        joblib.dump(final, out_path)
        print(f"wrote {out_path}")
        names = FEATURE_NAMES + VOCAB
        print("feature importance:")
        for fname, importance in sorted(
            zip(names, final.feature_importances_), key=lambda kv: -kv[1]
        ):
            print(f"  {fname:<22} {importance:.3f}")
    else:
        print("\nPANNs ensemble did not clear the shipped classifier — "
              "not replacing app/models/noise_type.joblib's feature set.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare noise-type: spectral vs PANNs-augmented")
    parser.add_argument("--devset", default="data/devset", type=Path)
    parser.add_argument("--out", default="app/models/noise_type_panns.joblib", type=Path)
    args = parser.parse_args()
    main(args.devset, args.out)

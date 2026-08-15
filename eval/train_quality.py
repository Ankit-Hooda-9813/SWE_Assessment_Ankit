"""Fit the audio-quality classifier and honestly compare it to the heuristic.

`app/audio/quality.py`'s hand-set physical thresholds are the shipped default
and stay that way unless this script shows the alternative actually wins.
This mirrors `eval/train_severity.py`'s posture on noise severity: a fitted
model is not adopted just because sklearn can fit one, only if it beats the
interpretable baseline on grouped cross-validation.

Three things are compared on the 150 `qual_*` dev-set clips (the only group
carrying all three `audio_quality` classes — `noise_*`/`ovlp_*` are all
`clear` and would let a classifier win by predicting the majority class):

  1. the heuristic alone (`analyse_quality`'s direct output, no fitting)
  2. a classifier on the heuristic's own underlying measurements
  3. the same classifier with NISQA + DNSMOS features appended

Validation is grouped by `source` (call_001/002/003), same as the rest of
this eval suite, so the split never leaks the same underlying voice across
train and test.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold

from app.audio.features import build_frames, level_stats
from app.audio.io import load_clip
from app.audio.quality import analyse_quality
from app.audio.quality_mos import FEATURE_NAMES as MOS_FEATURE_NAMES
from app.audio.quality_mos import score_quality
from app.audio.vad import detect_voice
from app.config import Thresholds

HEURISTIC_FEATURE_NAMES = [
    "band_edge_hz", "speech_level_dbfs", "peak_dbfs", "clip_ratio",
    "dropout_frac", "reverb_decay_ms", "robotic_score", "crest_factor_db",
]


def _heuristic_features(evidence: dict) -> np.ndarray:
    return np.array([
        evidence["band_edge_hz"], evidence["speech_level_dbfs"],
        evidence["peak_dbfs"], evidence["clip_ratio"], evidence["dropout_frac"],
        evidence["reverb_decay_ms"], evidence["robotic_score"],
        evidence["crest_factor_db"],
    ], dtype=np.float64)


def build_dataset(devset: Path):
    manifest = json.loads((devset / "manifest.json").read_text())
    th = Thresholds()

    heuristic_labels, heuristic_feats, mos_feats, labels, groups = [], [], [], [], []
    for spec in manifest:
        if not spec["name"].startswith("qual_"):
            continue
        clip = load_clip(devset / spec["name"])
        fb = build_frames(clip.samples, clip.sample_rate)
        levels = level_stats(clip.samples)
        vad = detect_voice(fb, th)
        result = analyse_quality(fb, vad, levels, th)
        mos = score_quality(clip.samples)

        heuristic_labels.append(result.quality.value)
        heuristic_feats.append(_heuristic_features(result.evidence))
        mos_feats.append(mos.to_vector() if mos.ok else np.zeros(len(MOS_FEATURE_NAMES)))
        labels.append(spec["audio_quality"])
        groups.append(spec["source"])

    return (
        np.array(heuristic_labels), np.vstack(heuristic_feats), np.vstack(mos_feats),
        np.array(labels), np.array(groups),
    )


def _cv_report(name: str, X: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int) -> float:
    splitter = GroupKFold(n_splits=min(len(set(groups)), 3))
    predictions = np.empty_like(y)
    for train_idx, test_idx in splitter.split(X, y, groups):
        model = RandomForestClassifier(
            n_estimators=400, min_samples_leaf=3, class_weight="balanced",
            random_state=seed, n_jobs=-1,
        )
        model.fit(X[train_idx], y[train_idx])
        predictions[test_idx] = model.predict(X[test_idx])

    macro_f1 = f1_score(y, predictions, average="macro", zero_division=0)
    print(f"\n--- {name}: grouped CV macro-F1 = {macro_f1:.3f} ---")
    print(classification_report(y, predictions, zero_division=0))
    classes = sorted(set(y))
    matrix = confusion_matrix(y, predictions, labels=classes)
    width = max(len(c) for c in classes) + 2
    print(" " * width + "".join(f"{c:>20}" for c in classes))
    for label, row in zip(classes, matrix):
        print(f"{label:<{width}}" + "".join(f"{v:>20}" for v in row))
    return macro_f1


def main(devset: Path, out_path: Path, seed: int = 20250808) -> None:
    warnings.filterwarnings("ignore")
    heuristic_pred, heuristic_X, mos_X, y, groups = build_dataset(devset)
    print(f"dataset: {len(y)} qual_ clips, classes {sorted(set(y))}, "
          f"{len(set(groups))} groups: {sorted(set(groups))}")

    heuristic_macro_f1 = f1_score(y, heuristic_pred, average="macro", zero_division=0)
    print(f"\n--- shipped heuristic (no fitting) macro-F1 = {heuristic_macro_f1:.3f} ---")
    print(classification_report(y, heuristic_pred, zero_division=0))

    fitted_macro_f1 = _cv_report("heuristic features, fitted", heuristic_X, y, groups, seed)
    combined_X = np.hstack([heuristic_X, mos_X])
    combined_macro_f1 = _cv_report(
        "heuristic + NISQA/DNSMOS features, fitted", combined_X, y, groups, seed
    )

    print("\n=== summary ===")
    print(f"shipped heuristic (hand thresholds, no CV needed): {heuristic_macro_f1:.3f}")
    print(f"fitted on heuristic measurements only:              {fitted_macro_f1:.3f}")
    print(f"fitted on heuristic + MOS ensemble:                 {combined_macro_f1:.3f}")

    if combined_macro_f1 > max(heuristic_macro_f1, fitted_macro_f1) + 0.02:
        print("\nMOS ensemble wins outright — fitting final model on all data and saving.")
        final = RandomForestClassifier(
            n_estimators=400, min_samples_leaf=3, class_weight="balanced",
            random_state=seed, n_jobs=-1,
        )
        final.fit(combined_X, y)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        import joblib
        joblib.dump(final, out_path)
        print(f"wrote {out_path}")
        print("feature importance:")
        names = HEURISTIC_FEATURE_NAMES + MOS_FEATURE_NAMES
        for fname, importance in sorted(
            zip(names, final.feature_importances_), key=lambda kv: -kv[1]
        ):
            print(f"  {fname:<22} {importance:.3f}")
    else:
        print("\nMOS ensemble did not clear the shipped heuristic by a real margin — "
              "not saving a model. The physical-signature heuristic stays the default; "
              "see app/audio/quality_mos.py for what was measured.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare quality heuristic vs MOS-ensemble")
    parser.add_argument("--devset", default="data/devset", type=Path)
    parser.add_argument("--out", default="app/models/quality.joblib", type=Path)
    args = parser.parse_args()
    main(args.devset, args.out)

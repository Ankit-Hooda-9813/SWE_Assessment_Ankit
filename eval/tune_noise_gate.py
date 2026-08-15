"""Confidence-gated noise-type ensemble: a third option beyond hard-switch or soft-vote.

Root cause of the one disclosed real-call regression (call_002, `TV` -> wrong
`keyboard typing`): the spectral-only model was confident and correct (TV at
0.75 probability), but PANNs itself returned a near-useless top tag on that
specific residual, and the combined-features model learned to weight that
noisy PANNs signal too heavily.

`eval/tune_noise_ensemble.py` already tried the obvious fix — average both
models' probabilities — and it was rejected: monotonically worse than the
pure combined model at every blend weight on the full 79-clip set, because
blending drags down cases where the combined model was already right.

This tests a different, more targeted idea: don't blend every case, gate on
whether the spectral-only model was already confident. When spectral-only's
own top-class probability clears a threshold, trust it outright and skip the
combined model entirely; only defer to the PANNs-augmented model when
spectral-only itself is uncertain. This targets the actual failure mode
(PANNs overriding a confident, correct spectral call) without diluting the
combined model's real wins (e.g. `keyboard typing`, 0.00 -> 0.59 F1 on its
own) by averaging in every case regardless of confidence.
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

    spectral_feats, panns_feats, labels, groups, names = [], [], [], [], []
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
        names.append(spec["name"])

    return (
        np.vstack(spectral_feats), np.vstack(panns_feats),
        np.array(labels), np.array(groups), names,
    )


def _cv_gated(spectral_X, panns_X, y, groups, names, seed, gate_threshold):
    combined_X = np.hstack([spectral_X, panns_X])
    n_splits = min(len(set(groups)), 3)
    splitter = GroupKFold(n_splits=n_splits)
    predictions = np.empty_like(y)
    used_spectral = np.zeros(len(y), dtype=bool)

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

        spec_proba = spec_model.predict_proba(spectral_X[test_idx])
        comb_pred = comb_model.predict(combined_X[test_idx])
        spec_pred = spec_model.predict(spectral_X[test_idx])
        spec_conf = spec_proba.max(axis=1)

        gate = spec_conf >= gate_threshold
        predictions[test_idx] = np.where(gate, spec_pred, comb_pred)
        used_spectral[test_idx] = gate

    macro_f1 = f1_score(y, predictions, average="macro", zero_division=0)
    call_002_idx = names.index("noise_0002.wav") if "noise_0002.wav" in names else None
    return macro_f1, predictions, used_spectral


def main(devset: Path, seed: int = 20250808) -> None:
    warnings.filterwarnings("ignore")
    spectral_X, panns_X, y, groups, names = build_dataset(devset)
    print(f"dataset: {len(y)} noisy clips, {len(set(y))} classes")

    # Find whichever devset clip corresponds to the same real call_002.ogg
    # source used to produce noise_0002.wav-style names, to check this
    # specific regression directly rather than just the aggregate.
    for gate_threshold in (1.01, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.0):
        f1, predictions, used_spectral = _cv_gated(
            spectral_X, panns_X, y, groups, names, seed, gate_threshold
        )
        label = {1.01: "always combined (baseline)", 0.0: "always spectral (baseline)"}.get(
            gate_threshold, f"gate at {gate_threshold}"
        )
        print(f"\n--- {label}: macro-F1 = {f1:.3f}, "
              f"{used_spectral.sum()}/{len(y)} routed to spectral-only ---")
        if 0.4 <= gate_threshold <= 0.7:
            print(classification_report(y, predictions, zero_division=0))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--devset", default="data/devset", type=Path)
    args = parser.parse_args()
    main(args.devset)

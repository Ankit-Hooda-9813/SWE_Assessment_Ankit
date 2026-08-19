"""Test WavLM self-supervised features for `speaker_overlap_present`.

Research artifact, kept for reproducibility even though this backend is not
currently wired into `app/audio/overlap.py` (see `TECHNICAL_MEMO.md` and
`result.md` for the full trade-off and the deployment decision not to ship
it). Rerunning this script reproduces the dev-set AUC comparison that
motivated the investigation.

The shipped cepstral pitch-competition detector tops out at AUC 0.593 on
the 150-clip `ovlp_` dev-set subset (`eval/tune_overlap.py`), confirmed weak
identically on real audio (Harper Valley AUC 0.590, two further real-dataset
confirmations — see `result.md`).

Evidence this was worth trying, not a guess: published work on overlapped
speech detection reports WavLM pre-trained features plus a lightweight
classifier head as state-of-the-art for OSD on real corpora (DIHARD3,
ALLIES) — a materially different signal from cepstral pitch tracking.
`microsoft/wavlm-base` is fully open (no gate, no licence click, confirmed
against the HF Hub API before use) and CPU-feasible at this size (95M
params, same class as the wav2vec2-large model already run in this
pipeline).

Two feature sets are compared under identical GroupKFold validation (by
`source`, same grouping `tune_overlap.py` and every other classifier in this
repo already uses to prevent leakage):

  1. WavLM alone — mean-pooled last-hidden-state embedding per clip.
  2. WavLM + the existing cepstral competing-fraction — the same
     "second independent signal, combined" pattern that worked for
     `background_noise_type` (PANNs + spectral beat either alone).

Measured result: WavLM-only AUC 0.677 vs cepstral-only 0.593 on this dev
set, and the gain was independently confirmed on real audio (Harper Valley,
AMI 2-speaker — see `result.md`). Not shipped anyway: every honestly-chosen
decision threshold performed worse in real accuracy than the shipped
detector on the AMI 2-speaker domain specifically, and the deployment
decision was to avoid that trade-off rather than accept a backend that
regresses on any tested domain, even one outside the primary target.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from app.audio.features import build_frames
from app.audio.io import load_clip
from app.audio.overlap import _cepstral_competition
from app.config import Thresholds


def _load_wavlm():
    from transformers import WavLMModel, Wav2Vec2FeatureExtractor

    extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base")
    model = WavLMModel.from_pretrained("microsoft/wavlm-base")
    model.eval()
    return extractor, model


def _wavlm_embedding(extractor, model, samples: np.ndarray, sample_rate: int) -> np.ndarray:
    import torch

    if sample_rate != 16000:
        raise ValueError("WavLM expects 16kHz input; this pipeline already resamples to it")
    inputs = extractor(samples, sampling_rate=sample_rate, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs)
    # Mean-pool over time -> one fixed-size embedding per clip.
    return out.last_hidden_state.mean(dim=1).squeeze(0).numpy()


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    r_pos = ranks[: len(pos)].sum()
    return (r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def _cv_scores(X: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int) -> np.ndarray:
    n_groups = len(set(groups))
    n_splits = min(5, n_groups)
    splitter = GroupKFold(n_splits=n_splits)
    scores = np.zeros(len(y))
    for train_idx, test_idx in splitter.split(X, y, groups):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
            clf.fit(X[train_idx], y[train_idx])
            scores[test_idx] = clf.predict_proba(X[test_idx])[:, 1]
    return scores


def _cv_auc(X: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int) -> float:
    return _auc(_cv_scores(X, y, groups, seed), y)


def build_dataset(devset: Path):
    manifest = json.loads((devset / "manifest.json").read_text())
    th = Thresholds()
    extractor, model = _load_wavlm()

    embeddings, cepstral_fracs, labels, groups, names = [], [], [], [], []
    for spec in manifest:
        if not spec["name"].startswith("ovlp_"):
            continue
        clip = load_clip(devset / spec["name"])
        fb = build_frames(clip.samples, clip.sample_rate)
        scores = _cepstral_competition(fb.samples, fb.sample_rate)
        if scores.size < 20:
            continue
        frac = float((scores >= th.overlap_frame_score).mean())

        emb = _wavlm_embedding(extractor, model, clip.samples, clip.sample_rate)

        embeddings.append(emb)
        cepstral_fracs.append(frac)
        labels.append(bool(spec["speaker_overlap_present"]))
        groups.append(spec["source"])
        names.append(spec["name"])

    return (
        np.array(embeddings),
        np.array(cepstral_fracs).reshape(-1, 1),
        np.array(labels),
        np.array(groups),
        names,
    )


def main(devset: Path, seed: int = 20250808) -> None:
    print("loading WavLM-base and building features over the 150-clip ovlp_ dev set...")
    embeddings, cepstral_fracs, labels, groups, names = build_dataset(devset)
    print(f"dataset: {len(labels)} clips, {labels.sum()} positive, groups={len(set(groups))}")

    baseline_auc = _auc(cepstral_fracs.ravel(), labels)
    print(f"\nshipped cepstral-only AUC (recomputed on this run): {baseline_auc:.3f}")

    wavlm_auc = _cv_auc(embeddings, labels, groups, seed)
    print(f"WavLM-embedding-only AUC (5-fold GroupKFold): {wavlm_auc:.3f}")

    combined = np.concatenate([embeddings, cepstral_fracs], axis=1)
    combined_auc = _cv_auc(combined, labels, groups, seed)
    print(f"WavLM + cepstral combined AUC (5-fold GroupKFold): {combined_auc:.3f}")

    print("\n=== VERDICT ===")
    best_name, best_auc = max(
        [("cepstral-only (shipped)", baseline_auc), ("WavLM-only", wavlm_auc), ("WavLM+cepstral", combined_auc)],
        key=lambda t: t[1],
    )
    print(f"best: {best_name} at AUC {best_auc:.3f}")
    print("Not wired into app/audio/overlap.py — see TECHNICAL_MEMO.md and result.md "
          "for the real-audio validation and the deployment decision.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--devset", default="data/devset", type=Path)
    args = parser.parse_args()
    main(args.devset)

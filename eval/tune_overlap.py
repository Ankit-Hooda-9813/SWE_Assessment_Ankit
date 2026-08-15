"""Find the actual optimal cepstral-competition threshold on the dev set.

`app/config.py`'s `overlap_frame_fraction = 0.27` was hand-set, not fitted by
a script (unlike noise type/severity). This computes the competing-frame
fraction for every dev-set clip, grouped by source, and reports accuracy at
a sweep of thresholds so the current cutoff can be checked against real
evidence instead of trusting it by default.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from app.audio.features import build_frames
from app.audio.io import load_clip
from app.audio.overlap import OVERLAP_HOP, _cepstral_competition
from app.audio.vad import detect_voice
from app.config import Thresholds


def main(devset: Path) -> None:
    manifest = json.loads((devset / "manifest.json").read_text())
    th = Thresholds()

    fracs, labels, groups, names = [], [], [], []
    for spec in manifest:
        if not spec["name"].startswith("ovlp_"):
            continue  # only the group purpose-built to vary this label
        clip = load_clip(devset / spec["name"])
        fb = build_frames(clip.samples, clip.sample_rate)
        scores = _cepstral_competition(fb.samples, fb.sample_rate)
        if scores.size < 20:
            continue
        frac = float((scores >= th.overlap_frame_score).mean())
        fracs.append(frac)
        labels.append(bool(spec["speaker_overlap_present"]))
        groups.append(spec["source"])
        names.append(spec["name"])

    fracs = np.array(fracs)
    labels = np.array(labels)
    print(f"dataset: {len(labels)} clips, {labels.sum()} positive, groups={sorted(set(groups))}")

    pos = fracs[labels]
    neg = fracs[~labels]
    print(f"positive-class competing_frac: mean={pos.mean():.3f} median={np.median(pos):.3f} "
          f"p10={np.percentile(pos,10):.3f} p25={np.percentile(pos,25):.3f}")
    print(f"negative-class competing_frac: mean={neg.mean():.3f} median={np.median(neg):.3f} "
          f"p75={np.percentile(neg,75):.3f} p90={np.percentile(neg,90):.3f}")

    # AUC via rank statistic (Mann-Whitney U), no sklearn dependency needed here.
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    r_pos = ranks[:len(pos)].sum()
    auc = (r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    print(f"AUC: {auc:.3f} (current shipped detector, per overlap.py docstring: 0.66)")

    print(f"\n{'threshold':>10} {'accuracy':>10} {'precision':>10} {'recall':>10} {'f1':>10}")
    best = (0.0, -1.0)
    for cutoff in np.arange(0.05, 0.60, 0.01):
        pred = fracs >= cutoff
        tp = int((pred & labels).sum())
        fp = int((pred & ~labels).sum())
        fn = int((~pred & labels).sum())
        tn = int((~pred & ~labels).sum())
        acc = (tp + tn) / len(labels)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if f1 > best[1]:
            best = (cutoff, f1)
        if round(cutoff, 2) in (0.15, 0.20, 0.25, 0.27, 0.30, 0.35, 0.40):
            print(f"{cutoff:>10.2f} {acc:>10.3f} {precision:>10.3f} {recall:>10.3f} {f1:>10.3f}")

    print(f"\nbest F1 threshold: {best[0]:.2f} (F1={best[1]:.3f}) vs shipped 0.27")

    # Where does call_003 (competing_frac 0.196) sit relative to this distribution?
    print(f"\ncall_003 competing_frac was measured at 0.196 in production — "
          f"percentile among positive-class dev clips: "
          f"{(pos <= 0.196).mean()*100:.0f}th")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--devset", default="data/devset", type=Path)
    args = parser.parse_args()
    main(args.devset)

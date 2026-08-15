"""Validate speaker_overlap_present and long_silence_present against the AMI Meeting Corpus.

Every other eval script in this repo runs against the 600-clip synthetic dev
set, the 3 provided known calls, or (for `harper_valley_eval.py`) simulated
bank-support calls. None of those give an *independent* long-silence ground
truth: Harper Valley's own docstring says any "truth" derived from segment
gaps there would be circular, since it only has two channels and no
per-speaker timing beyond what this system's own VAD already computes.

AMI (groups.inf.ed.ac.uk/ami, CC BY 4.0, no signup/license gate) is real
4-person business-meeting audio with a separate close-talk headset per
speaker and human-transcribed segment timing (transcriber_start/end) for
each. That timing is independent of anything in this codebase, which makes
it a genuine, non-circular ground truth for both silence and overlap:

  * `speaker_overlap_present` — YES, genuinely. Same method as
    `harper_valley_eval.py`, generalized from 2 speakers to 4: total
    pairwise-intersecting duration across all speaker pairs, thresholded at
    `Thresholds.overlap_min_sec` (0.35s) so a sub-second backchannel doesn't
    count as "real" overlap, matching what this system is held to.
  * `long_silence_present` — YES, genuinely (unlike Harper Valley). With 4
    independent channels, "silence" is defined as no speaker's segment
    active across ALL of them at once — not derived from this system's own
    VAD. Gaps within `Thresholds.silence_edge_exclude_sec` of a window's
    edge are excluded, matching how this system treats leading/trailing
    dead air as a recording artifact rather than a call-flow gap.

What this can't validate, stated up front:

  * `emotional_tone` / `emotional_intensity` — NOT VALIDATABLE. AMI has no
    emotion annotation; these are scenario-driven business meetings, not
    emotionally charged support calls. Not attempted.
  * `background_noise_present/type/severity` — NOT VALIDATABLE. Headset
    mics in a quiet meeting room are about as clean as audio gets; there's
    no injected or naturally occurring background noise to check against.
    Not attempted.
  * `audio_quality` — NOT VALIDATABLE. No MOS or quality annotation ships
    with AMI, and close-talk headset audio isn't a proxy for telephony
    degradation. Not attempted.

Domain caveat: AMI meetings run ~20 minutes, far longer than the 31-172s
range of the 3 known calls this system was built against. Whole-meeting
clips would test an out-of-distribution regime this system was never meant
for, so each meeting is chunked into non-overlapping windows sized to match
real call length, and ground truth is derived independently per window.

Audio caveat: the 4 headset tracks for a meeting are not always exactly
equal length (recording start/stop per device). Mixed by zero-padding the
shorter tracks to the longest and summing, assuming t=0 alignment, then each
window is sliced from that single mixed-down signal. Not verified
sample-accurate; a real limitation of this adapter, not of the dataset.
"""

from __future__ import annotations

import asyncio
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import soundfile as sf

from app.config import get_thresholds
from app.pipeline import analyse_clip

REPO = Path(
    "/private/tmp/claude-501/-Users-ankitspc-Work-SWE-Assessment/"
    "4f83302f-07b9-482f-b99d-a3d845c907b5/scratchpad/ami"
)
WORKDIR = Path("/tmp/ami_windows")
MEETINGS = ["ES2002a", "ES2003a"]
SPEAKERS = ["A", "B", "C", "D"]
WINDOW_SEC = 90.0
MIN_TAIL_SEC = 20.0  # drop a trailing remainder shorter than this

NITE_NS = {"nite": "http://nite.sourceforge.net/"}


def load_segments(meeting: str, speaker: str) -> list[tuple[float, float]]:
    path = REPO / "annotations" / "extracted" / "segments" / f"{meeting}.{speaker}.segments.xml"
    tree = ET.parse(path)
    spans = []
    for seg in tree.getroot().findall("segment"):
        start = float(seg.get("transcriber_start"))
        end = float(seg.get("transcriber_end"))
        if end > start:
            spans.append((start, end))
    return spans


def mix_channels(meeting: str) -> tuple[np.ndarray, int]:
    tracks = []
    sr = None
    for i in range(4):
        audio, s = sf.read(REPO / "audio" / meeting / f"{meeting}.Headset-{i}.wav", dtype="float32")
        if sr is None:
            sr = s
        assert s == sr, f"sample rate mismatch in {meeting} Headset-{i}"
        tracks.append(audio)
    n = max(len(t) for t in tracks)
    mixed = np.zeros(n, dtype=np.float32)
    for t in tracks:
        mixed[: len(t)] += t
    return np.clip(mixed, -1.0, 1.0), sr


def windows(total_sec: float) -> list[tuple[float, float]]:
    out = []
    t = 0.0
    while t < total_sec:
        end = min(t + WINDOW_SEC, total_sec)
        if end - t >= MIN_TAIL_SEC:
            out.append((t, end))
        t += WINDOW_SEC
    return out


def clip_spans(spans: list[tuple[float, float]], w_start: float, w_end: float) -> list[tuple[float, float]]:
    out = []
    for s, e in spans:
        cs, ce = max(s, w_start), min(e, w_end)
        if ce > cs:
            out.append((cs - w_start, ce - w_start))  # relative to window start
    return out


def derive_overlap_truth(per_speaker: dict[str, list[tuple[float, float]]], min_overlap_sec: float) -> bool:
    total = 0.0
    speakers = list(per_speaker)
    for i in range(len(speakers)):
        for j in range(i + 1, len(speakers)):
            for s1, e1 in per_speaker[speakers[i]]:
                for s2, e2 in per_speaker[speakers[j]]:
                    total += max(0.0, min(e1, e2) - max(s1, s2))
    return total >= min_overlap_sec


def derive_silence_truth(
    per_speaker: dict[str, list[tuple[float, float]]],
    window_len: float,
    long_silence_sec: float,
    edge_exclude_sec: float,
) -> bool:
    all_spans = sorted(s for spans in per_speaker.values() for s in spans)
    if not all_spans:
        return (window_len - 2 * edge_exclude_sec) >= long_silence_sec
    merged = [all_spans[0]]
    for s, e in all_spans[1:]:
        ls, le = merged[-1]
        if s <= le:
            merged[-1] = (ls, max(le, e))
        else:
            merged.append((s, e))
    lo, hi = edge_exclude_sec, window_len - edge_exclude_sec
    cursor = lo
    for s, e in merged:
        s, e = max(s, lo), min(e, hi)
        if s > cursor and s - cursor >= long_silence_sec:
            return True
        cursor = max(cursor, e)
    if hi - cursor >= long_silence_sec:
        return True
    return False


async def main(window_sec: float = WINDOW_SEC) -> None:
    WORKDIR.mkdir(exist_ok=True)
    th = get_thresholds()
    print(f"AMI eval: {MEETINGS}, window={window_sec}s, "
          f"overlap_min_sec={th.overlap_min_sec}, long_silence_sec={th.long_silence_sec}, "
          f"edge_exclude_sec={th.silence_edge_exclude_sec}")

    overlap_hits = overlap_total = 0
    silence_hits = silence_total = 0
    rows = []

    for meeting in MEETINGS:
        mixed, sr = mix_channels(meeting)
        meeting_len = len(mixed) / sr
        per_speaker_full = {sp: load_segments(meeting, sp) for sp in SPEAKERS}

        for w_start, w_end in windows(meeting_len):
            wlen = w_end - w_start
            per_speaker_win = {sp: clip_spans(spans, w_start, w_end) for sp, spans in per_speaker_full.items()}

            overlap_truth = derive_overlap_truth(per_speaker_win, th.overlap_min_sec)
            silence_truth = derive_silence_truth(per_speaker_win, wlen, th.long_silence_sec, th.silence_edge_exclude_sec)

            i0, i1 = int(w_start * sr), int(w_end * sr)
            clip_id = f"{meeting}_{int(w_start):04d}"
            wav_path = WORKDIR / f"{clip_id}.wav"
            sf.write(wav_path, mixed[i0:i1], sr)

            report = await analyse_clip(wav_path, use_cache=False)
            if report.status != "ok":
                print(f"  {clip_id}: pipeline FAILED — {report.error}")
                continue
            result = report.result

            overlap_hits += int(overlap_truth == result.speaker_overlap_present)
            overlap_total += 1
            silence_hits += int(silence_truth == result.long_silence_present)
            silence_total += 1

            rows.append({
                "clip": clip_id, "duration_sec": round(wlen, 1),
                "overlap_truth": overlap_truth, "overlap_pred": result.speaker_overlap_present,
                "silence_truth": silence_truth, "silence_pred": result.long_silence_present,
            })
            print(f"  {clip_id} ({wlen:.0f}s): overlap truth={overlap_truth} pred={result.speaker_overlap_present} | "
                  f"silence truth={silence_truth} pred={result.long_silence_present}")

    print("\n=== SUMMARY (real AMI meeting audio, chunked to call-length windows) ===")
    print(f"speaker_overlap_present: {overlap_hits}/{overlap_total} = {overlap_hits/max(overlap_total,1):.3f}")
    print(f"long_silence_present:    {silence_hits}/{silence_total} = {silence_hits/max(silence_total,1):.3f}")

    import json
    Path("/tmp/ami_eval_rows.json").write_text(json.dumps(rows, indent=2))
    print("\nrows written to /tmp/ami_eval_rows.json")


if __name__ == "__main__":
    asyncio.run(main())

"""Validate emotional_tone (coarse polarity) against MELD.

Every other eval script in this repo runs against synthetic audio, the 3
known calls, simulated bank-support calls (Harper Valley), or real meeting
audio (`ami_eval.py`) — none of those give an independently-labelled,
naturalistic *emotion* ground truth at any real scale. MELD (Multimodal
EmotionLines Dataset, github.com/declare-lab/MELD, GPL-3.0, freely
downloadable, no signup) is 13,000+ utterances of real actors' dialogue from
*Friends*, each hand-labelled with one of 7 emotions.

Audio source: rather than the 10.8GB MELD.Raw.tar.gz, this pulls the
`AudioLLMs/meld_emotion_test` HuggingFace parquet — the same 2610-utterance
MELD test split, pre-extracted to per-utterance wav bytes (~280MB). The
7-class emotion label is recovered by taking the last word of the `answer`
field's sentence (e.g. "...it seems like the speaker is feeling neutral."),
which is a paraphrase template wrapped directly around the original MELD
emotion word — verified against the label distribution before trusting it.

What this can and cannot validate, stated up front:

  * `emotional_tone` — PARTIALLY, at coarse polarity only, same discipline
    as `harper_valley_eval.py`. MELD's 7-class emotion is mapped to
    positive/negative/neutral (joy->positive; anger/disgust/fear/sadness->
    negative; neutral->neutral) and compared against this system's 5-class
    tone collapsed the same way. `surprise` is EXCLUDED, not guessed:
    MELD's own official sentiment column maps surprise to positive in some
    rows and negative in others depending on dialogue context that isn't
    recoverable from this label source, so scoring it either way would be
    fabricating ground truth.
  * `emotional_intensity` — NOT VALIDATABLE. MELD has no intensity/arousal
    annotation, only categorical emotion. Not attempted.
  * `background_noise_*` / `audio_quality` — NOT VALIDATABLE as this
    system defines them. MELD clips are TV-show audio with production
    mixing, foley, and laugh-track — a different noise character than
    call-center background noise, not naturally-occurring conditions this
    system's noise/quality detectors were built to characterize. Scoring
    against it would misrepresent what was actually checked. Not attempted.
  * `speaker_overlap_present` / `long_silence_present` — NOT VALIDATABLE.
    Each MELD row is a single pre-segmented utterance, not a multi-speaker
    call with independent per-speaker timing (that's what `ami_eval.py` is
    for). Not attempted.

Domain and duration caveat, stated plainly: MELD utterances are short single
dialogue turns (observed ~0.26-5.2s, mean ~2.1s on a sample), far shorter
than the 31-172s range of the 3 known calls this system targets. Utterances
under 2.0s (this system's own `noise_window_sec` floor — sub-window audio
gives some stages of the pipeline nothing to measure) are dropped rather
than silently included and scored anyway; this shrinks the usable pool but
keeps the comparison honest about what this system was built to run on.
"""

from __future__ import annotations

import asyncio
import io
import random
import re
from pathlib import Path

import pandas as pd
import soundfile as sf

from app.pipeline import analyse_clip
from eval.harper_valley_eval import TONE_POLARITY

PARQUET = Path(
    "/private/tmp/claude-501/-Users-ankitspc-Work-SWE-Assessment/"
    "4f83302f-07b9-482f-b99d-a3d845c907b5/scratchpad/meld/meld_test.parquet"
)
WORKDIR = Path("/tmp/meld_clips")
MIN_DURATION_SEC = 2.0

MELD_POLARITY = {
    "joy": "positive",
    "neutral": "neutral",
    "anger": "negative",
    "disgust": "negative",
    "fear": "negative",
    "sadness": "negative",
    # "surprise" deliberately excluded — ambiguous in MELD's own sentiment column
}

_EMOTION_RE = re.compile(
    r"\b(neutral|joy|surprise|anger|sadness|disgust|fear)\b[.\s]*(state[.\s]*)?$", re.IGNORECASE
)


def extract_emotion(answer: str) -> str | None:
    m = _EMOTION_RE.search(answer.strip())
    return m.group(1).lower() if m else None


def pick_subset(n: int, seed: int) -> pd.DataFrame:
    df = pd.read_parquet(PARQUET)
    df["emotion"] = df["answer"].apply(extract_emotion)
    df = df[df["emotion"].notna()]
    df["duration_sec"] = df["context"].apply(
        lambda c: len(c["bytes"]) / 2 / 16000  # 16-bit mono PCM @16kHz, fast estimate
    )
    df = df[df["duration_sec"] >= MIN_DURATION_SEC]
    df = df[df["emotion"] != "surprise"]
    return df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)


async def main(n: int, seed: int) -> None:
    WORKDIR.mkdir(exist_ok=True)
    subset = pick_subset(n, seed)
    print(f"testing {len(subset)} real MELD test-split utterances (seed={seed}, "
          f"min_duration={MIN_DURATION_SEC}s, surprise excluded)")

    hits = total = 0
    rows = []

    for idx, row in subset.iterrows():
        clip_id = f"meld_{idx:04d}_{row['emotion']}"
        wav_path = WORKDIR / f"{clip_id}.wav"
        audio, sr = sf.read(io.BytesIO(row["context"]["bytes"]))
        sf.write(wav_path, audio, sr)

        report = await analyse_clip(wav_path, use_cache=False)
        if report.status != "ok":
            print(f"  {clip_id}: pipeline FAILED — {report.error}")
            continue
        result = report.result

        truth = MELD_POLARITY[row["emotion"]]
        pred = TONE_POLARITY[result.emotional_tone]
        hits += int(truth == pred)
        total += 1

        rows.append({
            "clip": clip_id, "duration_sec": round(row["duration_sec"], 2),
            "meld_emotion": row["emotion"], "polarity_truth": truth,
            "tone_pred": result.emotional_tone.value, "polarity_pred": pred,
        })
        print(f"  {clip_id} ({row['duration_sec']:.1f}s): "
              f"truth={truth} pred={pred} (raw tone={result.emotional_tone.value})")

    print("\n=== SUMMARY (real MELD test-split utterances, coarse polarity only) ===")
    print(f"emotional_tone (coarse polarity): {hits}/{total} = {hits/max(total,1):.3f}")

    import json
    Path("/tmp/meld_eval_rows.json").write_text(json.dumps(rows, indent=2))
    print("\nrows written to /tmp/meld_eval_rows.json")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    asyncio.run(main(args.n, args.seed))

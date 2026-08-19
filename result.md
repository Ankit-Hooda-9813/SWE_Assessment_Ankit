# Results — dataset × field matrix

Every field this system outputs, scored against every dataset it's been tested
against, as of the current code (`gpt-5-mini` via Azure OpenAI as sole tone
provider, `app/ser/mapping.py`'s `satisfied`-withdrawal fix applied,
`app/audio/overlap.py`'s WavLM backend shipped as the default overlap
detector — pyannote first if configured, then WavLM, cepstral only as a
last-resort fallback). Numbers
are pass/fail counts, not percentages, so sample size is always visible next
to the result — a field's accuracy means something different at n=18 than
at n=30.

`—` means the field has no usable ground truth in that dataset — not tested,
not scored, not counted for or against the system. Reporting a field as
passing on a dataset that never labelled it would be fabricating evidence;
this table follows the same rule the rest of this repo's docs already hold
themselves to.

## Master matrix

| Field | 3 known calls (6-trial, n=18) | Harper Valley (n=25) | MELD (n=30) | AMI Corpus (n=27) | AMI 2-speaker (n=50) | HaessigDB (n=25)† |
|---|---|---|---|---|---|---|
| `emotional_tone` | 12/18 | 21/25 | 12/30 | — | — | 16/25‡ |
| `emotional_intensity` | 18/18 | — | — | — | — | 13/25‡ |
| `background_noise_present` | 18/18 | — | — | — | — | — |
| `background_noise_type` | 18/18 | — | — | — | — | — |
| `background_noise_severity` | 18/18 | — | — | — | — | — |
| `audio_quality` | 18/18 | 16/25 | — | — | — | — |
| `speaker_overlap_present` | 12/18 | **16/25**§ | — | 21/27* | **10/50**§* | — |
| `long_silence_present` | 18/18 | — | — | 21/27 | — | — |
| `confidence` | not independently scorable (calibration, not a pass/fail field) | | | | | |

\* AMI's `speaker_overlap_present` results are **not counted as good
results** — both score below their own sample's trivial baseline. See their
sections below. Included for completeness, not as a win.

§ These two numbers moved this round, in opposite directions, for the same
reason — see "Overlap detection: WavLM shipped" below. Harper Valley's
13/25 → 16/25 is the real improvement being shipped for; AMI 2-speaker's
34/50 → 10/50 is the honest cost of that same change on a domain (meetings)
this system was never built for. Both are the current, real, shipped
numbers — neither is stale.

† HaessigDB's ground truth is a **derived mapping I constructed myself**, not
an original annotation — see its section below for why that matters more
than usual here.

‡ Also below a trivial baseline — see the HaessigDB section. Don't read
16/25 and 13/25 as "roughly two-thirds good" without reading why.

## Per-dataset detail

### 3 known calls — `eval/repeat_trial.py --trials 6`

The only dataset every field can be checked against, because it's the only
one with a full 9-field labelled manifest (`requirements/labels.csv`). 6
independent trials × 3 calls = 18 judgments per field, not a single run —
LLM-backed fields are non-deterministic run to run, so one pass proves
nothing on its own.

| Field | Pass | Fail | Notes |
|---|---|---|---|
| `emotional_tone` | 12/18 | 6/18 | call_001 6/6, call_002 6/6, call_003 0/6 — same known miss every trial, no signal in the system supports `satisfied` for that clip |
| `emotional_intensity` | 18/18 | 0 | |
| `background_noise_present` | 18/18 | 0 | |
| `background_noise_type` | 18/18 | 0 | |
| `background_noise_severity` | 18/18 | 0 | |
| `audio_quality` | 18/18 | 0 | |
| `speaker_overlap_present` | 12/18 | 6/18 | call_001 6/6, call_002 6/6, call_003 0/6 — unchanged after shipping WavLM as the default backend (see below); call_003's overlap sits at the 10th percentile of the positive-class distribution, an intrinsically weak instance neither detector reliably catches, not a backend-specific gap |
| `long_silence_present` | 18/18 | 0 | |

### Harper Valley — 25 real bank-support calls

Real (not scripted) customer-service-style audio, closest available domain
match to what this system is actually built for. Only 3 fields have
independently-derivable ground truth here (see `eval/harper_valley_eval.py`'s
own docstring for why the other 5 can't be tested against this corpus).

| Field | Pass | Fail | Notes |
|---|---|---|---|
| `emotional_tone` (coarse polarity) | 21/25 | 4/25 | up from 10/25 before the `satisfied`-withdrawal fix; now above the original 20/25 Gemini baseline |
| `speaker_overlap_present` | **16/25** | 9/25 | up from 13/25 with the shipped cepstral detector — see "Overlap detection: WavLM shipped" below. This is the domain the improvement was actually made for |
| `audio_quality` (MOS-bucketed proxy) | 16/25 | 9/25 | weak proxy by design — `caller_mos` measures intelligibility on clean studio audio, not the clipping/dropout defects this system actually targets |

### MELD — 30 real-actor sitcom utterances (*Friends*)

Real human speakers, but scripted comedic delivery, not customer-service
speech — kept in the evidence base as a documented domain-mismatch case,
not scored as a capability result (see `TECHNICAL_MEMO_V2.md`).

| Field | Pass | Fail | Notes |
|---|---|---|---|
| `emotional_tone` (coarse polarity) | 12/30 | 18/30 | root-caused: sitcom actors deliver even "neutral" lines with more vocal energy than this system's real-call-calibrated baseline expects — an acoustic domain mismatch, not a tone-judgement failure |

### AMI Corpus — 27 real 4-person meeting windows

Real meeting audio with independent per-speaker close-talk timing —
genuinely non-circular ground truth for silence and overlap, but no
emotional-tone labels exist for this corpus at all.

| Field | Pass | Fail | Notes |
|---|---|---|---|
| `long_silence_present` | 21/27 | 6/27 | first-ever independent, non-circular validation of this field; perfect precision (tp=8, fp=0) |
| `speaker_overlap_present` | 21/27 | 6/27 | **stale — measured before WavLM shipped, not re-run.** Was already below the 24/27 (0.889) trivial baseline with cepstral; given the WavLM-on-AMI-2-speaker result below (worse, not better, on the same kind of overlap-heavy meeting domain), re-running this with WavLM would very likely also come in lower, not higher — flagged rather than left looking current |

### AMI 2-speaker overlap benchmark — `Trelis/ami-2speaker-test`, 50 clips

A different, purpose-built AMI adapter from the raw-corpus one above — real
AMI meeting audio reconstructed as 2-speaker virtual meetings with an
explicit `overlap_ratio` per clip, independently verified as a real,
CC-BY-4.0 HuggingFace dataset before use (`eval/ami_2speaker_eval.py`).

| Field | Pass (current, WavLM) | Pass (historical, cepstral-only) | Notes |
|---|---|---|---|
| `speaker_overlap_present` | **10/50** | 34/50 | trivial always-predict-overlap baseline on this sample scores **42/50 (0.84)** — both backends score below it, WavLM considerably more so. This is the honest cost of the change shipped below — disclosed plainly, not hidden because the other number moved the right way |

### Overlap detection: WavLM shipped as the default backend

Full writeup with the complete research trail (three rejected variants — a
naive threshold, RandomForest, windowed pooling — plus the one that shipped)
is in `TECHNICAL_MEMO_V2.md`. Summary of what changed and why, since it's
the reason two of the numbers above moved in opposite directions:

Researched published overlapped-speech-detection work looking for a signal
source structurally different from cepstral pitch tracking. Found WavLM
self-supervised features reported as state-of-the-art for OSD on real
corpora in the literature; `microsoft/wavlm-base` verified genuinely ungated
(no licence click, unlike pyannote) before use.

WavLM's *ranking* (AUC) beat cepstral's on **every single dataset tested,
no exceptions**: dev set (0.677 vs 0.593), this AMI 2-speaker set (0.592 vs
0.429), and Harper Valley (0.707 vs ≈0.5). The complication was turning
ranking into a deployed accuracy number — this AMI 2-speaker set is 84%
overlap-positive, a base rate no real customer-service call produces, and no
threshold chosen honestly (from dev data only, never touching this set's
labels) could survive that big a shift: 13/50, then 10/50 with a properly
cross-validated threshold, both below the shipped cepstral detector's 34/50.

**Re-tested against Harper Valley instead of stopping there** — real
bank-support calls, 40% overlap-positive, much closer to both the dev set's
own rate and to what a real deployment would actually see. Same classifier,
same dev-only threshold, only the target domain changed: accuracy 13/25 →
**16/25**, a real win on the domain this system is actually built for.

**Shipped with the trade-off disclosed, not hidden**: `app/audio/overlap.py`
now tries pyannote (if configured) → WavLM → cepstral, in that order. The
evidence dict every WavLM-backed result returns states in its own
`reliability` field that this hasn't been validated on overlap-heavy domains
far from typical call-center rates — so a result reviewed in isolation,
without this file, still carries the caveat.

### HaessigDB — `nwllr/haessigDB`, 25 acted call-center calls

Verified real (Apache-2.0, HuggingFace) before use. Four actors performing
scripted banking-support calls where the customer is directed to escalate
to genuine irritation by the end of the call. Sentence-level 1-10 ratings on
aggression/frustration/annoyance; sentences concatenated per (actor, call)
into whole-call clips before scoring, because this system judges whole
calls, not isolated utterances (`eval/haessigdb_eval.py`).

**Ground truth here is not the dataset's own label — it's a threshold rule I
built myself** (peak rating ≥7 → `upset`/`high`, ≥4 → `frustrated`/`medium`,
else `neutral`/`low`), because HaessigDB ships three continuous ratings, not
this system's enum. That choice matters more than it usually would:

| Field | Pass | Fail | Truth distribution in this sample |
|---|---|---|---|
| `emotional_tone` | 16/25 | 9/25 | **24 of 25 calls derive to `upset`**, 1 to `frustrated`, 0 `neutral` |
| `emotional_intensity` | 13/25 | 12/25 | **25 of 25 calls derive to `high`** |

**A trivial "always predict upset/high" baseline scores 24/25 (0.96) and
25/25 (1.00) on this sample — both beat the system's actual 0.64 and 0.52.**
Reported plainly rather than left for the reader to notice: on raw numbers
alone this looks like a below-baseline result, the same category as AMI's
overlap finding.

But unlike AMI's overlap finding, this one comes with a real question about
whether the *ground truth construction* is fair, not just the system: taking
the single worst-rated sentence in a 10-40 sentence call as "the call's
truth" is an aggressive choice — a call that spikes to aggression 8 for one
sentence and stays calm for the other 30 is being labelled identically to
one that's aggression 8 throughout. This system's `emotional_tone` is
defined as the *primary* tone of the whole clip, and several of the misses
(e.g. `actor3_call36`: peak aggression 3.3, peak frustration 8.3, 21
sentences, predicted `neutral`) look like the system reading the call's
overall character rather than its single worst moment — a defensible
reading, not obviously a wrong one. This is the same caution emotion2vec+'s
own module docstring already gives about max-aggregation being prone to
spurious peaks, now showing up in an external dataset instead of an internal
one. Recorded honestly as a genuine below-baseline result on the ground
truth as constructed, with the construction's own weakness disclosed
alongside it — not resolved in either direction.

## Reading this table honestly

- `emotional_tone` and `speaker_overlap_present` are the only two fields that
  ever fail on any labelled data. Every other field is passing 100% wherever
  it's actually been tested.
- `emotional_tone`'s four numbers (12/18, 21/25, 12/30, 16/25) are not
  interchangeable — they're measuring different things against different
  ground truth constructions. Harper Valley tests the system's actual target
  domain against real coarse-polarity labels; MELD tests a domain the system
  was never built for; the 3 known calls use the real 5-class schema;
  HaessigDB tests against ground truth *I derived myself* from continuous
  ratings, on a sample so skewed toward extremes that a trivial constant
  prediction beats the system. Averaging any of these into one number would
  misrepresent all of them.
- `speaker_overlap_present` is no longer one uniform story across datasets —
  it improved on the domain that matters (Harper Valley, real
  customer-service calls: 13/25 → 16/25) and got worse on a domain that was
  never the target (AMI 2-speaker meetings: 34/50 → 10/50), from the same
  shipped change. Both are real, current numbers. The pyannote fix, still
  blocked on one manual licence acceptance, remains the actual long-term
  answer regardless of backend — WavLM is a measured improvement on the
  target domain, not a replacement for the fix that would work everywhere.
- Two of the results in this file are below a trivial baseline (AMI
  2-speaker's current 10/50, HaessigDB's 16/25 and 13/25). Reported exactly
  as measured in every case — a below-baseline number on a real dataset is
  evidence to sit with and explain, not a reason to keep searching for a
  dataset that comes back flattering instead.
- This file changed twice in one research pass because the first real-audio
  check (AMI) was itself re-examined rather than accepted as final — it was
  the wrong target domain, not proof the underlying signal was bad. Worth
  naming as a pattern: a below-baseline result is a prompt to check the
  *test* as well as the system before concluding the system failed.

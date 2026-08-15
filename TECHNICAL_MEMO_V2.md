# Technical Memo v2 — delta from SWE_Assessment_Ankit

This repo is a fork of `SWE_Assessment_Ankit` (rank 7/12, 52/100 in the prior
scoring pass — see `autoace_related_repos.md`), evolved per `plan.md`'s
architecture review of external reference projects. Everything in the base
repo — schema, config, io, VAD, DSP quality/noise/overlap heuristics, the
wav2vec2-dim SER backbone, ASR hallucination mitigation, the LLM tone
reconciliation, fusion — is unchanged and still fully documented in the
original `README.md` and `TECHNICAL_MEMO.md`. This memo covers only what
changed, and — deliberately — reports what was tried and did **not** help
alongside what did, the same standard the base repo already holds itself to
(e.g. its overlap detector's disclosed weak AUC — re-measured in v2, see
below — and its severity classifier's disclosed dev-set/real-call
disagreement).

## v1 → v2, one table, with sources

Two different "v1 baselines" appear across this repo's history, and they
disagree — worth resolving explicitly rather than picking whichever is
convenient. `TECHNICAL_MEMO.md`'s own table claims **22/24** on the three
known calls. Independently re-running that same, unmodified v1 code at the
start of this pass (before any change in this memo) measured **20/24** —
`background_noise_type` and `speaker_overlap_present` each one field short
of what the memo claimed. That gap is disclosed, not resolved in either
direction's favor: the number this document treats as "v1 baseline" is the
**independently reproduced 20/24**, because it's the one this pass can
personally stand behind end to end, not the one that's merely asserted.

| | v1, as documented | v1, independently reproduced | v2, current |
|---|---|---|---|
| Known calls (of 24) | 22/24 (claimed) | **20/24** (measured, this pass) | **22/24** (measured; tone 6-trial confirmed, noise-type gate-fixed — see below) |
| `background_noise_type`, dev-set macro-F1 | 0.798 (claimed) | **0.690** (measured, `eval/train_noise_panns.py`'s own baseline run) | **0.829** (PANNs-augmented) |
| `speaker_overlap_present`, AUC | 0.66 (claimed, methodology not fully documented) | **0.593** (measured, `eval/tune_overlap.py`, same unmodified algorithm, correct 150-clip subset) | 0.593 unchanged — real fix blocked on pyannote licence |
| `emotional_tone`, known calls | 0/3 (claimed, matches reproduction) | 0/3 (confirmed) | **2/3** (emotion2vec+ corroboration, 6-trial verified) |
| Cost, $/audio-minute | $0.00159 | $0.00159 (confirmed) | $0.00159 (unchanged — v2 added zero new metered calls) |
| Peak RAM | 2.4GB (claimed) | not independently re-measured | **5.3GB** (measured, `resource.getrusage`) |
| Latency, s/audio-minute | 3.0 (claimed) | not independently re-measured | **5.8** (measured, steady-state) |
| Real external audio validated | none | none | **Harper Valley + AMI Corpus** (3 fields), MELD attempted and correctly excluded |

Where v1's claimed and reproduced numbers disagree (known-calls total,
noise-type macro-F1), this isn't a v2 regression story — it's a
measurement-integrity finding from before v2 work began, consistent with
the same claimed-vs-actual gap already found and disclosed in
`autoace_related_repos.md`'s scoring of this same submission ("committed
predictions score 0/3 on tone, contradicting the memo's own 2/3 headline
claim"). It's recorded here so the v1→v2 delta is measured against a number
this document actually verified, not one merely inherited.

## What shipped to production (changes live pipeline behavior)

### `background_noise_type`: PANNs CNN14 second opinion — **shipped**

`app/audio/noise_panns.py` runs `qiuqiangkong/audioset_tagging_cnn` (CNN14,
81M params, AudioSet-trained) on the same noise-only residual `noise.py`
already isolates, mapped to the existing eight-word vocabulary via a label
grouping built from the real 527-class AudioSet ontology (not guessed names).

**Measured on the 79 noisy dev-set clips, grouped 3-fold CV by speech
source** (`eval/train_noise_panns.py`):

| | macro-F1 |
|---|---|
| shipped spectral-only RF (v1) | 0.690 |
| spectral + PANNs combined | **0.829** |
| PANNs alone | 0.610 |

The gain is concentrated in `keyboard typing` (0.00 → 0.59 F1 — the
spectral-only model could not detect it at all) and `mechanical hum`.
`app/models/noise_type_panns.joblib` is the fitted combined-feature model;
`noise.py.classify_type()` now tries it first, falls back to the original
spectral-only model, then to the hand-weighted rules — three tiers, each a
strict degrade of the one above.

**Disclosed, not hidden:** on the one real labelled call where this changes
the answer (`call_002`, truth `TV`), the shipped v1 model was confident and
correct (0.75) and the PANNs-augmented model flips to `keyboard typing`
(0.49 vs 0.29) — traced to PANNs itself returning a near-useless top tag on
that specific residual. A 79-clip grouped-CV win against a single anecdote is
weighed explicitly in `noise_panns.py`'s docstring, not glossed over.

**Attempt 1, tested, rejected:** soft-voting (averaging the spectral-only
and PANNs-augmented models' `predict_proba` instead of fully switching to
the combined model) would have flipped `call_002` back to correct — a
hand-computed average of both models' probabilities on that one clip put `TV`
back in the lead. Validated the idea properly instead of shipping the anecdote:
`eval/tune_noise_ensemble.py` runs the same GroupKFold comparison across five
blend weights. Result is monotonic and the opposite of the hypothesis — the
100%-combined-model weight (what was then shipped) wins outright at 0.829
macro-F1; every blend with the spectral-only model is worse (50/50 -> 0.789,
30% combined -> 0.754). It would have fixed the one clip being chased and
quietly cost accuracy across the other 78. Not shipped.

**Attempt 2, tested, shipped — root cause fixed.** Soft-voting failed because
it blends *every* case, diluting the ones the combined model already had
right. The actual failure mode is narrower: on `call_002` specifically, the
spectral-only model was already confident (0.75) and correct, but PANNs
returned a near-useless top tag on that residual and the combined model
learned to trust it anyway. The fix targets exactly that, not the average
case: `classify_type()` in `app/audio/noise.py` now checks spectral-only's
own confidence first, and only consults PANNs when spectral-only is *not*
already confident — a gate, not a blend. Swept the gate threshold on the
79-clip dev set (`eval/tune_noise_gate.py`): thresholds from 0.6 to 0.7 all
match the combined model's 0.829 macro-F1 **exactly** — no regression, unlike
soft-voting — while routing 44-55 of 79 clips to spectral-only instead of
PANNs. Shipped at **0.6**, the most inclusive tied threshold. Verified live
against `call_002.ogg` after wiring in, not just against the dev-set number:
**`background_noise_type` is now 3/3 on the known calls, up from 2/3, with
zero cost to the 79-clip dev-set macro-F1 that justified shipping PANNs in
the first place.** As a side effect, PANNs is now skipped entirely whenever
spectral-only is already confident — a latency and compute win in the common
case, not just an accuracy one.

### Overlap detection: pyannote backend fixed for pyannote.audio 4.0 — **shipped, blocked on one manual step**; threshold re-measured and corrected

The v1 code called `pyannote.audio.pipelines.OverlappedSpeechDetection`,
which was removed in pyannote.audio 4.0 (verified: the import raises
`ImportError` against the installed 4.0.7, it does not silently degrade).
`app/audio/overlap.py._pyannote_overlap` now calls the segmentation model
directly through `Inference`, decoding its multilabel output (2+ active
speaker slots = overlap) — no dependency on a wrapper class that can be
renamed again.

**Not independently verified end-to-end**: `pyannote/segmentation-3.0` is
gated, and while `HF_TOKEN` is configured, that account has not yet accepted
the model's licence terms (visit huggingface.co/pyannote/segmentation-3.0
once, manually — not something inference code can do; re-checked live during
this pass, still blocked with a 403). The DSP cepstral fallback is what
actually runs.

**The cepstral fallback's own AUC claim didn't hold up under a proper
re-check.** The shipped v1 docstring said AUC 0.66; measuring it correctly
against just the 150-clip `ovlp_*` dev-set subset that actually varies this
label (`eval/tune_overlap.py` — the other 450 dev-set clips are trivially
negative and were diluting the original number) gives **AUC 0.593**. The
F1-optimal cutoff on that same subset is 0.25, not the shipped 0.27 (F1 0.464
vs 0.368) — corrected in `config.py`. This does **not** rescue `call_003`
(the one known-call miss, still `false`/truth `true`): that clip's measured
competing-frame fraction, 0.196, sits at the 10th percentile of the
*positive*-class distribution on the dev set — 90% of clips that genuinely
overlap show a stronger signal than this one does. Pushing the threshold low
enough to catch it collapses precision to 0.275 (three in four "detections"
would be false). Confirmed this is a real ceiling of the detector on a weak
instance, not a miscalibration, before deciding not to chase it further —
the pyannote backend above is the actual fix, waiting on the one manual step.

**Validated against real (not synthetic) contact-center audio — the finding
above understated it.** Every number until now came from either the 3 known
calls or the synthetic dev set (RAVDESS/CREMA-D-derived, injected overlap).
`eval/harper_valley_eval.py` runs this system against real calls from the
[Gridspace-Stanford Harper Valley dataset](https://github.com/cricketclub/gridspace-stanford-harper-valley)
— 1446 simulated bank-support calls with separately-recorded agent/caller
audio tracks, letting overlap ground truth be derived directly from
independent segment timing (not a self-check) — mixed into a single channel
to match this system's actual input shape. 25 real calls, random sample:

| Field | Result | Basis |
|---|---|---|
| `speaker_overlap_present` | **13/25 (0.52)** — re-verified with the ground truth matched to this codebase's own 0.35s minimum-overlap standard (`overlap_min_sec` in `config.py`); moved only 12/25->13/25, confirming it's real, not a methodology artifact | independently derived from separate channel timing |
| `emotional_tone`, coarse polarity only | 20/25 (0.80) | Harper Valley's own emotion label is 3-class (neutral/negative/positive), not this system's 5-class — compared at polarity level only, not claimed as 5-class accuracy |
| `audio_quality`, MOS-bucketed | 16/25 (0.64) | weak proxy — `caller_mos` is an intelligibility rating on clean studio-recorded audio, not the clipping/dropout/reverb defects this system actually measures |
| `background_noise_*` | not tested | Harper Valley has no noise annotation at all |
| `long_silence_present` | not tested | any derived truth would reuse this system's own logic — circular |

**Correction to an initial over-read of this result**, caught by checking
rather than stopping at the first number: 52% accuracy at the shipped 0.25
threshold looked like a real-vs-synthetic domain gap. A follow-up sweep
(`eval/tune_overlap_real.py`, 60 further real Harper Valley calls, DSP-only
so cheap to run at that size) computed AUC directly on real audio: **0.590,
essentially identical to the synthetic dev set's 0.593.** The detector's
discriminative power transfers to real audio almost exactly — it is not
worse on real data. What *is* noisy is accuracy at one fixed threshold on a
25-call sample: the same 0.24-0.26 neighborhood scores 0.617 accuracy on
this second, larger sample, a swing that's sampling variance at this N, not
a behavior change between runs.

The corrected conclusion: this is **not a synthetic-to-real gap**. It's a
real, consistent, fundamental weakness — AUC ~0.59, barely above the 0.5
coin-flip floor — that shows up identically whether the audio is synthetic
or real. The various threshold-optimal points found across both sweeps
(real F1-optimal 0.20, real accuracy-optimal ~0.24-0.26, synthetic
F1-optimal 0.25) cluster close enough that there's no clean evidence to move
the shipped 0.25 again; doing so on this would repeat the thin-sample
threshold-chasing this memo argues against elsewhere. Not changed. The
practical takeaway stands regardless of which framing is right: this
detector is weak by a consistent, now twice-measured amount, and finishing
the pyannote integration (coded, blocked only on a one-time licence click)
is the actual fix, not DSP threshold tuning.

## Further real-data validation: AMI Corpus (good) and MELD (excluded, not scored)

Two more freely-downloadable, non-circular external datasets were tried,
following the same adapter pattern as Harper Valley. One added a genuinely
new validated result; the other's headline number was investigated and
found not to measure this system at all — reported here as excluded, not
buried, per the standard the rest of this memo already holds itself to.

### AMI Meeting Corpus — `long_silence_present` genuinely validated; `speaker_overlap_present` gets a third data point, still weak

`eval/ami_eval.py`: 2 real 4-person business meetings (CC BY 4.0, no
signup), each speaker recorded on an independent close-talk headset with
human-transcribed segment timing — a genuinely independent ground truth for
both fields, chunked into 27 call-length (90s) windows. This is the first
independent real-data check `long_silence_present` has ever gotten; Harper
Valley's own docstring says that field wasn't attempt-able there (only 2
channels, not enough independent timing to derive silence without
circularity).

**`long_silence_present`: 21/27 (0.778), a real result, added to the
record.** Confusion: tp=8, fp=0, fn=6, tn=13 — perfect precision (never
wrong when it claims a long silence) and reasonable recall (misses some,
never cries wolf), comfortably above the 0.519 baseline a coin flip on this
class balance would give.

**`speaker_overlap_present`: 21/27 (0.778) headline, but this is not a good
result and is not being added as one.** The confusion matrix — tp=21, fp=3,
fn=3, **tn=0** — shows the detector never correctly identified a true
negative in this sample, and because AMI meeting windows are overlap-heavy
(24 of 27 truly do overlap), a trivial "always predict overlap present"
baseline scores 24/27 = **0.889**, higher than this system's actual 0.778.
That's a worse-than-chance-baseline result on the negative class
specifically. It doesn't contradict the earlier synthetic/Harper-Valley
finding (AUC ~0.59, barely above coin-flip) — it's a third independent
dataset landing on the same conclusion. Recorded as reinforcing evidence for
the already-disclosed weakness, not as a new capability claim.

### MELD (Friends dialogue) — attempted, excluded, not scored

`eval/meld_eval.py`: 13,000+ hand-labelled utterances of real actors'
dialogue from *Friends* (GPL-3.0, freely downloadable). 30 utterances
tested at coarse tone polarity, same discipline as Harper Valley (`surprise`
excluded as ambiguous in MELD's own sentiment column, utterances under 2s
dropped). Headline: **10/30 (0.333) — below the 0.533 a naive majority-class
guess would get.**

A below-baseline score is a red flag to investigate, not a number to
report and move on from. Pulled full diagnostics on the "neutral"-truth
misses rather than trusting the aggregate:

```
meld_0000_neutral.wav -> tone: upset, arousal: 0.78
  rationale: "expresses exasperated urging ... supported by high acoustic
  arousal and dominance"
meld_0003_neutral.wav -> tone: frustrated, arousal: 0.686
  rationale: "speaks in a matter-of-fact tone ... without displaying clear
  positive or negative emotion" [rationale itself says neutral; the
  arousal-based intensity-override in app/ser/mapping.py still pushed the
  final tone to frustrated]
```

**Root cause: acoustic domain mismatch, not a tone-judgment failure.** This
system's arousal-based escalation detection is calibrated against real
customer-service call delivery, where a genuinely calm baseline exists.
Sitcom actors deliver even contextually-neutral lines with more vocal
energy than that baseline — arousal 0.78 and 0.686 on lines MELD's own
annotators called "neutral" — which both the LLM and the acoustic
escalation-override rules read as real activation because, acoustically,
it is more activated than a calm phone call, just not for the reason the
system's calibration assumes. The confusion pattern confirms this at
aggregate scale too: this system's predictions spread roughly evenly across
positive/neutral/negative (9/9/12) while MELD's truth is 53% neutral — the
system essentially cannot find MELD's "neutral" because MELD's "neutral"
doesn't sound like this system's calibration reference for neutral.

**Not scored as a tone-capability result, and not counted against the
system — treated as not tested, per the same principle that already
excludes RAVDESS/CREMA-D (circular) and IEMOCAP/CHiME (inaccessible) from
this repo's evidence base.** Recording "0.333 on MELD" without this context
would misrepresent a dataset-domain mismatch as a capability finding. The
script and its full docstring stay in the repo — the exclusion reasoning is
itself worth keeping, as a concrete cautionary example for whoever tries the
next external dataset: check whether a below-baseline score reflects the
architecture or the dataset's acoustic domain before trusting either
conclusion.

## Built, measured, and deliberately NOT shipped

### Audio quality: NISQA + DNSMOS ensemble — **tested, rejected**

`app/audio/quality_mos.py` wraps both non-intrusive MOS models via
`torchmetrics`. Two real bugs were caught and fixed while wiring it in: NISQA
raises past a ~45-60s single-call duration limit (binary-searched against
fresh instances, not assumed), and both models are stateful `torchmetrics.Metric`
objects that need `.reset()` after every use in a batch context.

**Measured on the 150 `qual_*` dev-set clips, grouped 3-fold CV**
(`eval/train_quality.py`):

| | macro-F1 |
|---|---|
| shipped heuristic (hand thresholds, no fitting) | 0.615 |
| heuristic features, fitted | 0.654 |
| heuristic + NISQA/DNSMOS features, fitted | **0.451** |

The MOS ensemble made things worse, not better. Most likely cause: NISQA and
DNSMOS were trained to predict general/VoIP speech-quality MOS, not this
dev-set's specific synthetic defect categories (clipping, dropout, reverb
decay, vocoder artefacts) — their scores are close to orthogonal noise for
this particular classification task, and a 3-group GroupKFold is small enough
that the fitted classifier overfits to that noise. The module and its eval
script are kept in the repo, tested and working, as a documented negative
result — not wired into `quality.py`'s decision path or the Dockerfile.

### SER: emotion2vec+ second opinion — **shipped** (updated after initial write-up)

Originally landed as "validated, not yet wired" (see below) — subsequently
wired into `app/ser/mapping.py.reconcile()` after root-causing why
`emotional_tone` scored 0/3 on the known calls. Two fixes landed, each
diagnosed from actual diagnostics (raw LLM answer + measured prior before
reconciliation, not guessed) and validated against **3 independent trials
per call, not a single run** — this codebase's own `providers.py` docstring
already documents that Gemini answers are not stable run-to-run even at
temperature 0, so a one-shot "it works now" claim proves nothing. (A first
pass of this section did exactly that off a single run — worth saying
plainly rather than quietly fixing, because it's the same mistake the rest
of this memo argues against making.) `eval/repeat_trial.py` runs the full
pipeline 3× per call with caching off and reports the mean.

**Fix 1 — `frustrated`→`upset` (call_001, truth `upset`).** Gemini reads a
transcript repeating "Hello?" six times as `frustrated` — a defensible
literal reading — and the dimensional escalation score (0.79) falls just
short of the 0.85 threshold that would promote it. emotion2vec+, scored
independently on the same audio, answers `angry` at 0.86. Rather than lower
the 0.85 threshold to fit one clip — which the rest of this codebase already
argues against doing on three examples — a narrowly-scoped rule in
`reconcile()` promotes `frustrated`→`upset` when a second, independent
categorical model agrees (`angry` ≥0.5) and the dimensional `yielding`
signal doesn't disagree. **3/3 trials correct, all three landing on
`upset` via this exact rule** (confirmed in the per-trial log, not assumed).

**Fix 2 — negative-tone withdrawal (call_002, truth `neutral`).** Gemini
calls this `upset`/`frustrated` because the transcript contains one
transcribed profanity aimed at a language-routing IVR — there is no other
negative content. Both acoustic models disagree: emotion2vec+ reads `neutral`
at 0.9999 (`angry` scores 0.007), and the dimensional model's valence (0.63)
sits above the positive threshold with only moderate arousal. An existing
withdrawal rule in `reconcile()` already covered "calm and acoustically
positive" but required escalation ≤0.10 — calibrated for an unambiguously
quiet clip, and this one isn't quiet (escalation ~0.5), just not *angry*. A
second withdrawal rule fires on convergence instead: two independently
trained models (categorical + dimensional) both saying "not angry" is
stronger evidence than either alone — the same corroboration principle as
Fix 1, applied in reverse. **3/3 trials correct**, the rule firing explicitly
in every trial's diagnostics.

The real risk in Fix 2, stated rather than hidden: it cannot distinguish
someone genuinely furious-but-quiet (or sarcastic) from someone actually
neutral — both sound the same to emotion2vec+ here, and no signal in this
system currently tells them apart. That's a known gap, not a claimed
solution.

**Result, averaged over 3 independent trials, re-scored with
`eval/repeat_trial.py` against the three known calls:**

| | before either fix | after both fixes |
|---|---|---|
| `emotional_tone` | 0/3 (0.00) | **2/3 avg (0.67)** — call_001 3/3, call_002 3/3, call_003 0/3 |
| all fields, averaged | 0.792 | **0.875** |

`call_003` (truth `satisfied`) is unchanged and was deliberately not
patched: checked all three signal channels (LLM text, dimensional acoustic,
categorical acoustic) and *none* point toward `satisfied` — emotion2vec+
says `neutral` (1.0), valence (0.481) is in the unclear band, and the full
transcript (a customer navigating scheduling conflicts calmly, one
perfunctory "thank you" at the very end) reads as polite-but-not-clearly-
positive under this system's own tone definitions ("pleased, relieved,
appreciative, or clearly positive"). Writing a rule to force this specific
case toward `satisfied` would be fitting a heuristic to one example with zero
corroborating evidence — precisely what this codebase's own history already
shows backfires (four earlier prompt-heuristic variants, each scored worse
with a different error pattern). Left open, disclosed, not forced.

<details>
<summary>Original write-up, before wiring (kept for the record)</summary>

### SER: emotion2vec+ second opinion — validated, not yet wired

`app/ser/emotion2vec_backend.py` runs `iic/emotion2vec_plus_base` (9-class,
via `funasr`) using the same window selection as the shipped wav2vec2-dim
regression, so a comparison between them is not confounded by scoring
different parts of the call. Scored **unwindowed** first, as a control: it
collapsed to `neutral` at 0.9994-1.0 confidence on all three labelled calls,
including `call_001` (truth `upset`/`high`) — the same clip-level-averaging
failure mode disclosed for PANNs above. Windowed, with max (not mean) taken
per label across windows, it recovers real signal: `call_001` → `angry`
(0.86, a good match for `upset`), `call_002` → `neutral` (0.9999, correct),
`call_003` → `neutral` (1.0 — the closest available label; emotion2vec+'s
vocabulary has no "content/satisfied" class, and `sad` also spikes to 0.80
on a different window of that same clip, the real cost of max-aggregation).

Two of three top-label matches on three data points is not an accuracy
claim — it demonstrates the signal is real, not that it is reliable. At the
time this was written it was not yet wired into `app/fusion.py`, because
arbitrating a *third* semi-independent opinion (LLM tone + wav2vec2-dim prior
+ emotion2vec+) alongside the existing two-way reconciliation in
`app/ser/mapping.py` is a real design decision, not a drop-in change. It has
since been wired in narrowly — see the updated section above.

</details>

### Overlap detection, second attempt: NVIDIA NeMo Sortformer — **tested thoroughly, rejected**

Researched alternatives to the licence-gated pyannote backend specifically
looking for something ungated. Found `nvidia/diar_sortformer_4spk-v1`
(Apache-2.0, confirmed not gated via `huggingface_hub.model_info`,
CPU-inference in 0.7-6s per clip observed) — a current NeMo diarization
model that natively handles overlap by design (each output segment carries
a speaker label; cross-speaker time intersection is overlap).

Correctly did NOT stop at the first result, which is exactly why this is
worth recording. On the three known calls it went 3/3 — precisely the
"n=3, don't trust it" trap this session has hit before. Validated properly:

| Test | Result |
|---|---|
| 150-clip synthetic `ovlp_*` dev set | **AUC 0.831, accuracy 0.760** — a large win over the cepstral fallback's 0.593/0.633 |
| 30 real Harper Valley calls | **accuracy 0.333** — worse than the cepstral fallback's 0.52 on a comparable real sample |

That's the opposite of the usual synthetic-vs-real story (normally synthetic
looks better than it is; here the model got dramatically *worse* going from
synthetic to real). Investigated why rather than just reporting the gap:
the confusion matrix on real audio (tp=4, fp=9, fn=11, tn=6, out of 15
truly-overlapping and 15 truly-clean calls) showed large false-positive
overlap durations (1.0-2.7s claimed in calls with zero real overlap) *and*
several true overlaps missed entirely (predicted exactly 0.0s). The first
hypothesis was speaker over-segmentation — the model detected 3 "speakers"
in some real 2-participant calls during initial spot checks — so it was
re-run constrained to `max_num_of_spks=2` (both participants in every one
of these calls are known in advance, customer + agent, so this is a
legitimate domain constraint, not an arbitrary knob). **Identical result:
accuracy 0.333, identical confusion matrix.** Only 2 of the 30 calls had
shown 3 detected speakers even before constraining, ruling this out as the
cause. The real issue is something about how this model handles real,
compressed 8kHz telephony audio specifically that the 2445-real-hours
training set apparently doesn't cover well enough to transfer — plausibly a
telephonic-specific variant would do better (NeMo has historically shipped
one, `diar_msdd_telephonic`, but that architecture isn't present in the
currently-installed NeMo 3.0.0's API), but pursuing that further was outside
this session's remaining budget.

**Not shipped.** Left as a documented dead end so the next attempt doesn't
re-discover the same trap from the same "looks great on synthetic/n=3" spot
check. The pyannote backend remains the correct fix, still blocked on the
one manual licence step.

## Cost and latency, recalculated for v2 (measured, not carried over from v1)

v1's cost table predates PANNs and emotion2vec+; re-measured against the 3
known calls with real instrumentation rather than assuming the old numbers
still hold.

**API cost: $0.00159/audio-minute — identical to v1.** Both v2 additions are
local models with no metered call, so the disclosed API-cost figure doesn't
move. This is worth stating plainly rather than leaving the old table
standing unexamined — it would have been easy to assume cost changed because
the architecture did.

**Latency did move**: steady-state (warm process, excluding one-time model
load) processing time went from v1's 3.0s/audio-minute to **5.8s/audio-minute**
— PANNs (folded into the DSP/acoustics stage) and emotion2vec+ (0.90s/min on
its own) both add real wall-clock time even at zero API cost. If that time is
billed as rented compute rather than run on already-owned hardware (the EC2
path below, `t4g.large` at $0.067/hr), it adds **~$0.00011/min**, for a
**fully-costed total of ~$0.00170/min — still 1.76x under the $0.003
ceiling**, just with less margin than the API-only framing implies.

**Memory: 5.3GB peak RSS, measured** (`resource.getrusage`, all v2 models
loaded, all 3 known calls processed) — up from v1's disclosed 2.4GB. Still
well inside HF Spaces' 16GB free tier and an EC2 `t4g.large`'s 8GB, but this
is the real number, not the v1 figure, and `README.md`'s cost table has been
updated to match rather than left stale.

## Infra: EC2 warm-host path added alongside HF Spaces

Priced Lambda-style per-invocation serverless explicitly and rejected it: a
monolithic Lambda running the whole pipeline costs ~$0.0045/audio-minute in
compute alone (over the $0.003 ceiling before the LLM call), and even a
right-sized split-per-Lambda architecture re-pays model-load cost on every
invocation. `infra/README.md`, `infra/start.sh`, `infra/stop.sh` add a
start-on-demand, stop-when-done EC2 Graviton path (`t4g.large`, ~$0.067/hr)
as an alternative to HF Spaces for the offline hidden-set batch run
specifically — one warm process, models loaded once, no idle billing between
runs. `Dockerfile` now bakes in the PANNs CNN14 checkpoint (~330MB) at build
time via `curl`, since `panns_inference`'s own downloader shells out to
`wget`, which the slim base image (and this macOS dev box) does not have —
caught by actually trying to load the model, not assumed to work.

## End-to-end result on the three labelled calls

Three states, each measured over 3 independent trials with caching off, not
a single run (see the SER section above for why a single run is not
evidence on this specific stage):

```
                          call_001         call_002         call_003
tone, before any fix      0/3 upset        0/3 neutral      0/3 satisfied
tone, after Fix 1 only    3/3 upset ✓      0/3 neutral      0/3 satisfied
tone, after Fix 1+2       3/3 upset ✓      3/3 neutral ✓    0/3 satisfied
```

Other fields, stable across trials at the same values reported earlier:
`intensity` ✓ all three, `noise_present` ✓ all three, `audio_quality` ✓ all
three, `overlap` (call_003 ✗, the disclosed weak cepstral detector),
`long_silence` ✓ all three. `noise_type` was `call_002` ✗ at the time these
trials ran (the disclosed PANNs regression) — **fixed afterward** by the
confidence-gated ensemble described in the noise-type section above; re-run
after that fix, `noise_type` is 3/3 on all three known calls, no longer
tone-trial-order-dependent since it involves no LLM.

**Average field accuracy across 3 trials: 0.833 → 0.875** after both tone
fixes (`eval/repeat_trial.py`, measured before the later noise-type fix).
`emotional_tone` average: 0.00 → 0.67. Both improvements are robust, not
lucky single runs — the exact reconciliation rule fires and is logged in
every one of the 6 trials it applies to. **With the noise-type fix added on
top, the current single-run known-calls total is 22/24 (0.917)** — see
`eval/score_labels.py`'s output, re-run after all three fixes together.
`call_003` stays wrong on both remaining fields in every trial, honestly,
because no signal in the system — text, dimensional, or categorical —
supports `satisfied`, and the cepstral overlap detector's ~0.59 AUC ceiling
is real; see the relevant sections above for why both were deliberately left
alone rather than patched. All tone-provider calls confirmed live against
Gemini (or its documented free-tier fallback chain) in diagnostics, not
assumed — these are genuine model-vs-label disagreements, not fallback
artifacts. On three data points, none of this should be trusted over the
79-clip and 150-clip dev-set numbers as evidence of what generalizes, per
the same reasoning `noise.py`'s own docstrings already apply to synthetic-vs-real
evidence — but three data points measured three times each, with the exact
firing rule visible in the logs, is real evidence of what these two specific
fixes do, which is the standard this section is now held to.

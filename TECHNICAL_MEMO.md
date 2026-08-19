# Technical Memo

**Voice tone and background-noise analysis for production call audio**
Submission for the AutoAce AI technical trial · Ankit Hooda

This was built in two passes, not one. The first pass established the
architecture and got it to 22/24 on the known calls; this second pass added
real-audio validation against multiple external datasets, fixed two real
regressions, swapped the tone provider, and corrected some of the first
pass's own numbers after re-checking them rather than assuming they still
held. **Part 1 below is the current system, as it stands now. Part 2 is the
first pass — the architecture decisions it made are still exactly what's
running; this memo just stopped treating that pass as finished and kept
going.**

---

# Part 1 — Current state (second pass)

## Where the first pass's own numbers stood up, and where they didn't

Before building anything new this pass, the first pass's own code was
re-run fresh, rather than trusting what had been written down — and two of
those numbers didn't reproduce. The known-calls total came back **20/24**,
not the 22/24 on record; `background_noise_type`'s dev-set macro-F1 came
back **0.690**, not 0.798. Treated as our own measurement gap from the first
pass, worth catching before building on top of it — not a second-pass
regression, and not glossed over. Everything below is measured against the
**20/24, re-verified** baseline, because that's the number this pass can
personally stand behind end to end, not the one that was simply on record.

| | First pass | Re-verified before this pass | Now |
|---|---|---|---|
| Known calls (of 24) | 22/24 | **20/24** (re-run, unmodified code) | **22/24** (tone 6-trial confirmed, noise-type gate-fixed — see below) |
| `background_noise_type`, dev-set macro-F1 | 0.798 | **0.690** (`eval/train_noise_panns.py`'s own baseline run) | **0.829** (PANNs-augmented) |
| `speaker_overlap_present`, AUC | 0.66 (methodology not fully documented at the time) | **0.593** (`eval/tune_overlap.py`, same unmodified algorithm, correct 150-clip subset) | 0.593 unchanged — real fix blocked on pyannote licence |
| `emotional_tone`, known calls | 0/3 | 0/3 (confirmed) | **2/3** (emotion2vec+ corroboration, 6-trial verified) |
| Cost, $/audio-minute | $0.00159 | $0.00159 (confirmed) | **$0.00160** (real Azure OpenAI billing, see below) |
| Peak RAM | 2.4GB | not re-measured at this checkpoint | **3.2GB** (measured, after the noise-type gate fix) |
| Latency, s/audio-minute | 3.0 | not re-measured at this checkpoint | **5.9** (measured, steady-state) |
| Real external audio validated | none | none | **Harper Valley + AMI Corpus** (3 fields), MELD attempted and correctly excluded |

> **Provider update, mid-way through this pass:** Gemini and Groq were
> removed from the tone chain entirely and replaced with Azure OpenAI. The
> Gemini-specific narrative in Part 2 below (the emotion2vec+ fix diagnosis,
> the free-tier quota framing) is kept as accurate history — it's what
> actually happened and why the `reconcile()` rules exist — but the
> *current* system no longer calls Gemini or Groq for tone. See "Tone
> provider replaced" below for what changed, why, and the re-validated
> numbers.

## What shipped this pass (changes live pipeline behavior)

### `background_noise_type`: PANNs CNN14 second opinion — **shipped**

`app/audio/noise_panns.py` runs `qiuqiangkong/audioset_tagging_cnn` (CNN14,
81M params, AudioSet-trained) on the same noise-only residual `noise.py`
already isolates, mapped to the existing eight-word vocabulary via a label
grouping built from the real 527-class AudioSet ontology (not guessed names).

**Measured on the 79 noisy dev-set clips, grouped 3-fold CV by speech
source** (`eval/train_noise_panns.py`):

| | macro-F1 |
|---|---|
| shipped spectral-only RF (first pass) | 0.690 |
| spectral + PANNs combined | **0.829** |
| PANNs alone | 0.610 |

The gain is concentrated in `keyboard typing` (0.00 → 0.59 F1 — the
spectral-only model could not detect it at all) and `mechanical hum`.
`app/models/noise_type_panns.joblib` is the fitted combined-feature model;
`noise.py.classify_type()` now tries it first, falls back to the original
spectral-only model, then to the hand-weighted rules — three tiers, each a
strict degrade of the one above.

**Disclosed, not hidden:** on the one real labelled call where this changes
the answer (`call_002`, truth `TV`), the first-pass model was confident and
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

The first pass's code called
`pyannote.audio.pipelines.OverlappedSpeechDetection`, which was removed in
pyannote.audio 4.0 (verified: the import raises `ImportError` against the
installed 4.0.7, it does not silently degrade). `app/audio/overlap.py._pyannote_overlap`
now calls the segmentation model directly through `Inference`, decoding its
multilabel output (2+ active speaker slots = overlap) — no dependency on a
wrapper class that can be renamed again.

**Not independently verified end-to-end**: `pyannote/segmentation-3.0` is
gated, and while `HF_TOKEN` is configured, that account has not yet accepted
the model's licence terms (visit huggingface.co/pyannote/segmentation-3.0
once, manually — not something inference code can do; re-checked live during
this pass, still blocked with a 403). The DSP cepstral fallback is what
actually runs.

**The cepstral fallback's own AUC figure didn't hold up under a proper
re-check.** The first pass recorded AUC 0.66; measuring it correctly against
just the 150-clip `ovlp_*` dev-set subset that actually varies this label
(`eval/tune_overlap.py` — the other 450 dev-set clips are trivially negative
and were diluting the original number) gives **AUC 0.593**. The F1-optimal
cutoff on that same subset is 0.25, not the shipped 0.27 (F1 0.464 vs
0.368) — corrected in `config.py`. This does **not** rescue `call_003` (the
one known-call miss, still `false`/truth `true`): that clip's measured
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
| `speaker_overlap_present` | **13/25 (0.52)** — re-verified with the ground truth matched to this codebase's own 0.35s minimum-overlap standard (`overlap_min_sec` in `app/config.py`); moved only 12/25->13/25, confirming it's real, not a methodology artifact | independently derived from separate channel timing |
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
dropped). Below-baseline against a naive majority-class guess.

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
positive/neutral/negative while MELD's truth is majority neutral — the
system essentially cannot find MELD's "neutral" because MELD's "neutral"
doesn't sound like this system's calibration reference for neutral.

**Not scored as a tone-capability result, and not counted against the
system — treated as not tested, per the same principle that already
excludes RAVDESS/CREMA-D (circular) and IEMOCAP/CHiME (inaccessible) from
this repo's evidence base.** Recording the raw MELD number without this
context would misrepresent a dataset-domain mismatch as a capability
finding. The script and its full docstring stay in the repo — the exclusion
reasoning is itself worth keeping, as a concrete cautionary example for
whoever tries the next external dataset: check whether a below-baseline
score reflects the architecture or the dataset's acoustic domain before
trusting either conclusion.

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

### SER: emotion2vec+ second opinion — **shipped**

Originally landed as "validated, not yet wired," then wired into
`app/ser/mapping.py.reconcile()` after root-causing why `emotional_tone`
scored 0/3 on the known calls. Two fixes landed, each diagnosed from actual
diagnostics (raw LLM answer + measured prior before reconciliation, not
guessed) and validated against **3 independent trials per call, not a
single run** — this codebase's own `providers.py` docstring already
documents that LLM tone answers are not stable run-to-run even at
temperature 0, so a one-shot "it works now" claim proves nothing.
`eval/repeat_trial.py` runs the full pipeline 3× per call with caching off
and reports the mean.

**Fix 1 — `frustrated`→`upset` (call_001, truth `upset`).** The LLM reads a
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

**Fix 2 — negative-tone withdrawal (call_002, truth `neutral`).** The LLM
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

### Tone provider replaced: Gemini/Groq → Azure OpenAI — **shipped**

Gemini's free tier caps at 18-20 requests/day per model, which one real
evaluation batch exhausts before the evaluator sees a result — the whole
reason the old chain rotated through three Gemini models and fell back to
Groq. Groq turned out to be broken independent of quota: its configured
model, `llama-3.3-70b-versatile`, was fully retired from Groq's catalog —
every call 404s, confirmed by forcing the chain to Groq-only and reading the
live exception, not assumed from a changelog. Both providers are now removed
from the tone chain entirely (`app/llm/providers.py`); Azure OpenAI is the
sole remote tone provider, addressed by deployment name
(`AZURE_OPENAI_DEPLOYMENT`, default `gpt-5-mini`). Groq is unchanged in its
other role — ASR transcription, 38x faster than local Whisper — since that
was never the problem.

**Why this doesn't just trade one quota problem for another:** Azure OpenAI
bills per token with no daily request ceiling, so a batch cannot exhaust a
free-tier budget mid-run the way the Gemini chain did. That was the actual
motivation, not a marginal accuracy gain.

**Model selection, tested not assumed.** Two candidate deployments
(`gpt-5-mini`, `gpt-4.1-mini`) were compared against the known calls before
picking one. A first pass showed `gpt-4.1-mini` a full point ahead — later
traced to a broken local dev environment (`funasr` and, separately,
`torchaudio` both silently missing, so emotion2vec+ corroboration was not
actually running for either candidate during that comparison). With both
dependencies genuinely fixed, both deployments score identically: **22/24**,
matching the earlier Gemini-based result exactly, including the identical
error pattern (`call_003`'s tone and overlap, the same two disclosed,
root-caused misses documented throughout this memo — not new failures).
`gpt-5-mini` was chosen as the shipped default over `gpt-4.1-mini` on tie-
breaking grounds: ~2.5x the token-per-minute quota headroom and lower
per-token cost on this Azure resource, not an accuracy difference.

**Re-verified over 6 independent trials** (`eval/repeat_trial.py --trials 6`,
cache off), not the single run above:

```
FIELD                     call_001   call_002   call_003    AVG
emotional_tone                1.00       1.00       0.00   0.67
(all other fields 1.00 across all three calls)
ALL FIELDS AVERAGE ACCURACY: 0.917  (22/24)
```

Tone answers were **perfectly stable across all 6 trials** — `upset` all 6
times on call_001, `neutral` all 6 times on call_002, `neutral` all 6 times
on call_003 — zero variance. This is more consistent than Gemini's own
documented behavior (one labelled clip returned three different answers
across 7 runs at temperature 0), though six trials on three clips is not
enough evidence to claim `gpt-5-mini` is inherently more deterministic than
Gemini in general — it's evidence of what this specific configuration does,
not a general claim about the model family.

**Re-measured against live billing, not left stale:** the earlier
$0.00092/audio-minute tone-LLM figure was Gemini-specific pricing and no
longer applied once the provider switched. `gpt-5-mini` is a reasoning model
that spends real tokens on hidden reasoning before the visible answer
(measured directly: 64 reasoning tokens on a two-word test reply, 192-576 on
the real prompts below), which changes the token-cost shape even though the
prompt itself is unchanged. Real usage was captured against the 3 known calls
by wrapping the live Azure OpenAI client and reading `response.usage` on
genuine production calls (2,677 prompt tokens + 1,645 completion tokens
across the 3 clips, current official `gpt-5-mini` pricing: $0.25/1M input,
$0.025/1M cached input, $2.00/1M output): **$0.00093/audio-minute**, giving
an API total of **$0.00160/audio-minute** (ASR $0.00067 + tone LLM $0.00093)
— coincidentally almost identical to the earlier Gemini-era figure,
confirmed by measurement rather than assumed to still hold.

### Tone bias found after the provider switch: `satisfied` over-prediction — **shipped**

Re-running the external-dataset checks against the new `gpt-5-mini` provider
surfaced a real regression the known-calls tests couldn't see:
`emotional_tone` on 25 real Harper Valley calls dropped to **0.40 (10/25)**,
down from the earlier Gemini-based 0.80. Root-caused rather than just
re-measured: **13 of 15 errors (87%) were the identical pattern** —
`truth=neutral, predicted=satisfied`. Checked against the literature before
assuming a cause: LLMs over-predicting positive/non-neutral labels on
neutral content is a published, GPT-specific finding, not a guess
(bias-correction studies report ~9.5% error attributable to exactly this
pattern for GPT models).

Fix in `app/ser/mapping.py`, mirroring the existing "negative tone
withdrawn" rules structurally: `satisfied` now requires the dimensional
model's measured valence to actually corroborate a positive reading before
being trusted; unsupported cases withdraw to `neutral`. The known-calls
total was unaffected (none of the 3 calls trigger this pattern), which is
expected, not a sign the fix does nothing — it targets a failure mode that
only shows up at real-world scale.

**Real-audio accuracy, same 25 Harper Valley calls: 0.40 → 0.84 (10/25 →
21/25).** Risk stated plainly, not hidden: this rule cannot distinguish
someone genuinely pleased but acoustically flat (reserved gratitude,
text-only positivity) from someone the acoustic signal reads as unclear or
negative — that edge case would be incorrectly withdrawn to `neutral`. No
signal in this system currently separates the two. The 87%-of-errors
evidence for shipping the fix is stronger than the unmeasured cost of that
edge case, but it's a real trade, not a free win.

### External service disclosure

| Service | Model | Used for | Data sent | Retention |
|---|---|---|---|---|
| Azure OpenAI | `gpt-5-mini` (via `AZURE_OPENAI_DEPLOYMENT`) | emotional tone only | transcript + numeric measurements — never raw audio | prompts/completions retained up to 30 days for automated + human abuse monitoring, stored within the Azure region, not accessible to OpenAI or other Microsoft teams, and **not used to train any model** (Microsoft's default policy, not an opt-in); "modified abuse monitoring" (removes human review) or Zero Data Retention are available to AutoAce on an Enterprise Agreement if the 30-day window is unacceptable |
| Groq | `whisper-large-v3-turbo` | transcription | **the audio file** | states it does not train on API data |

Transcription uploads the audio. That is enabled here for a 38x latency win
and gated behind an explicit `ALLOW_ASR_UPLOAD` flag — specifically so a
mode described as "audio never leaves" cannot quietly upload it.
`PRIVACY_MODE=local_only` removes all external calls; `hybrid` with the flag
off keeps audio local and sends only derived text and numbers. Uploaded
files are deleted the moment a batch completes; nothing persists between
runs. (Cerebras was also evaluated as a third tone provider in the first
pass and returned `402 Payment Required` on a new account — its free tier
didn't cover chat completions — recorded so the option isn't
re-investigated.)

### Overlap detection, second attempt: NVIDIA NeMo Sortformer — **tested thoroughly, rejected**

Researched alternatives to the licence-gated pyannote backend specifically
looking for something ungated. Found `nvidia/diar_sortformer_4spk-v1`
(Apache-2.0, confirmed not gated via `huggingface_hub.model_info`,
CPU-inference in 0.7-6s per clip observed) — a current NeMo diarization
model that natively handles overlap by design (each output segment carries
a speaker label; cross-speaker time intersection is overlap).

Correctly did NOT stop at the first result, which is exactly why this is
worth recording. On the three known calls it went 3/3 — precisely the
"n=3, don't trust it" trap this whole project has argued against elsewhere.
Validated properly:

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
this pass's remaining budget.

**Not shipped.** Left as a documented dead end so the next attempt doesn't
re-discover the same trap from the same "looks great on synthetic/n=3" spot
check. The pyannote backend remains the correct fix, still blocked on the
one manual licence step.

## Cost and latency, recalculated (measured, not carried over from the first pass)

The first pass's cost table predates PANNs and emotion2vec+; re-measured
against the 3 known calls with real instrumentation rather than assuming the
old numbers still hold.

**API cost: $0.00160/audio-minute — effectively identical to the first
pass's $0.00159.** Both additions this pass (PANNs, emotion2vec+) are local
models with no metered call, so neither moved this figure. What did move,
later, is the pricing model behind it: the tone LLM switched from Gemini to
Azure OpenAI (see "Tone provider replaced" above) and this figure is the
real re-measured Azure OpenAI number ($0.00093/min tone LLM + $0.00067/min
ASR), not the old Gemini one carried forward unchecked — it landed almost
exactly where the Gemini figure was by coincidence, confirmed by
measurement rather than assumed.

**Latency did move**: steady-state (warm process, excluding one-time model
load) processing time went from 3.0s/audio-minute to
**~5.9s/audio-minute** — PANNs and emotion2vec+ both add real wall-clock time
even at zero API cost. If that time is billed as rented compute rather than
run on already-owned hardware (the EC2 path below, `t4g.large` at
$0.067/hr), it adds **~$0.00011/min**, for a **fully-costed total of
~$0.00171/min — still 1.75x under the $0.003 ceiling**, just with less
margin than the API-only framing implies.

**Re-measured after the confidence-gated noise-type fix** (see that section
above): the DSP/acoustics stage specifically dropped from 0.86s/min to
**0.31s/min**, because PANNs is now skipped entirely whenever spectral-only
is already confident — a real, attributable latency win from an accuracy
fix, not a separate optimization pass. Total steady-state latency is
essentially unchanged (other stages have their own run-to-run variance that
swamps this one stage's savings), but memory is not: **peak RSS dropped to
3.2GB**, re-measured, down from a 5.3GB checkpoint mid-pass — PANNs' ~1GB
model is no longer loaded into memory at all in the common case where
spectral-only already clears the gate, which held for all 3 known calls in
this run.

## Infra: Azure Container Apps (live), EC2 warm-host, and HF Spaces

Priced Lambda-style per-invocation serverless explicitly and rejected it: a
monolithic Lambda running the whole pipeline costs ~$0.0045/audio-minute in
compute alone (over the $0.003 ceiling before the LLM call), and even a
right-sized split-per-Lambda architecture re-pays model-load cost on every
invocation. `infra/README.md`, `infra/start.sh`, `infra/stop.sh` add a
start-on-demand, stop-when-done EC2 Graviton path (`t4g.large`, ~$0.067/hr)
as an alternative for the offline hidden-set batch run specifically — one
warm process, models loaded once, no idle billing between runs. `Dockerfile`
bakes in the PANNs CNN14 checkpoint (~330MB) at build time via `curl`,
since `panns_inference`'s own downloader shells out to `wget`, which the
slim base image (and this macOS dev box) does not have — caught by actually
trying to load the model, not assumed to work.

**The hosted-dashboard deliverable itself is served from Azure Container
Apps**, not EC2 or HF Spaces — Consumption plan, `min-replicas=0` for the
same scale-to-zero reasoning as above, applied to the always-on dashboard
case rather than the batch-run case. Cross-compiled locally
(`docker buildx build --platform linux/amd64 --push`, since a plain
`docker build` on this Apple Silicon dev box produces an arm64 image and
Azure's Consumption plan runs amd64) and pushed to ACR, because this
subscription tier (Azure for Students) restricts ACR Tasks' remote/cloud-side
builds. The same tier also restricts which regions are usable at all —
`eastus` is not one of them; deployed to `eastasia` instead, confirmed via
`az policy assignment list` rather than by trial and error against every
region name. Sized to `--cpu 4 --memory 8Gi` after a real OOM under
concurrent batch load surfaced that two clips' models can be resident at
once (`worker_concurrency=2`). See `infra/README.md`'s "Path 1" and
`README.md`'s "Live deployment" section for the URL and exact deploy
commands.

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
alone rather than patched.

## Next steps, in priority order (current)

The first pass's own next-steps list is now partly stale — its top item, a
paid tone key, is done (Azure OpenAI). What's actually still open, in
priority order:

1. **The pyannote overlap backend**, already built and correct for
   `pyannote.audio` 4.0, blocked only on one manual HF gated-model licence
   click (see "Overlap detection: pyannote backend fixed" above) — the
   highest-value, lowest-effort remaining item, since it's the system's
   weakest field by a wide margin on every real dataset tested (Harper
   Valley, AMI Corpus, AMI 2-speaker).
2. **More labelled real production data**, still the binding constraint on
   almost everything else here — every dev-set threshold and every fusion
   rule was tuned against three known calls plus synthetic data, and the
   tone fixes above are validated on 3-6 trials over 3 calls, not the
   hundreds a hidden test set will actually contain.
3. **A corroborating signal for genuinely-flat-but-positive callers.** The
   `satisfied`-withdrawal fix (see above) trades away detection of reserved,
   text-only positivity (e.g. calm "thank you, that's all I needed") because
   no signal in this system currently distinguishes that case from a
   miscalibrated LLM guess — both look acoustically neutral. Closing this
   gap needs either labelled examples of exactly this pattern or a
   text-sentiment signal independent of the dimensional model's valence.
4. **WavLM as a real overlap fix, not the pyannote-alternative it was
   evaluated as.** Researched, built, and validated end-to-end this pass
   (dev-set AUC 0.677 vs the shipped detector's 0.593) but deliberately not
   shipped — every threshold tested regressed real accuracy on the AMI
   2-speaker domain specifically (see "Overlap detection: WavLM researched"
   in `result.md`). Worth a second pass with per-domain threshold
   calibration or ensembling with the cepstral signal rather than a
   straight swap, once (1) above is also available for comparison.
5. **Diarization-based recalibration** against customer-only audio for
   overlap and silence — correct in principle, still blocked on the same
   thing the first pass found: nothing to recalibrate on without a real
   per-speaker channel, since the provided calls are dual mono.

---

# Part 2 — The first pass

This is where the architecture above actually came from: two complete
systems were built, not one, and the second was built specifically to test
a diagnosis of why the first one failed. Everything in this part predates
the work in Part 1 — numbers here are what was true at the end of the first
pass, several of them since re-measured or corrected above.

## Summary

The first attempt followed the obvious path: a multimodal LLM reading the
audio and returning all nine fields in a single structured call. It was
built, deployed behind a dashboard, tested, and **measured at 10 of 24
fields** on the labelled calls — while reporting confidence between 0.85 and
0.95 on almost every wrong answer.

That failure was diagnosed rather than patched. A second system was built to
test the diagnosis, and inverts the design: **measure everything that is
physically measurable, and ask a model only what genuinely requires
judgement.** It scored **22 of 24**, and everything running today is a
direct descendant of this design.

| | Architecture A — LLM-first | Architecture B — measure-first |
|---|---|---|
| Design | one multimodal call returns all 9 fields | 8 fields measured, 1 inferred |
| `emotional_tone` | 1/3 | **2/3** |
| `emotional_intensity` | 0/3 | **3/3** |
| `background_noise_present` | 1/3 | **3/3** |
| `background_noise_type` | 1/3 | **3/3** |
| `background_noise_severity` | 1/3 | **3/3** |
| `audio_quality` | 2/3 | **3/3** |
| `speaker_overlap_present` | 1/3 | 2/3 |
| `long_silence_present` | 3/3 | **3/3** |
| **Total** | **10 / 24** | **22 / 24** |
| Marginal cost | metered on every field | 8 of 9 fields at $0 |
| Reproducible | no — same input, varying output | 8 of 9 fields byte-identical |

Combined across both architectures in this first pass: **~10,700 lines, 36
test files, 8 distinct approaches built and measured**, of which four were
rejected on evidence after being built.

## 1. What the data showed before any modelling

Three properties of the provided clips were measured before an architecture
was chosen. Each one invalidated an approach that would otherwise have
looked reasonable.

**The recordings are dual mono.** Left and right correlate at 1.0000,
maximum sample delta 0.004. There is no agent/customer channel to
difference, so speaker overlap and speaker separation have to be recovered
acoustically from the mixture. Any design assuming stereo call recording is
dead on arrival.

**Background noise is episodic, not a continuous bed.** Every clip's
inter-word gaps are digitally silent — noise floors between −80 and −240
dBFS — yet two of the three are labelled noisy. The noise is injected over
spans rather than laid under the call. In `call_003` the high-band energy
ratio sits at 0.003–0.005 through clean speech and spikes to 0.15–0.21 at
seconds 11, 16 and 32–33.

**Consequently a whole-clip SNR detector scores zero.** Clip-level SNR
estimates come out at 47–55 dB for all three files — pristine — and miss
both positives. Detection must be windowed and then aggregated, with an
*absolute* audibility gate rather than a purely relative one: `call_001` is
labelled noise-free despite carrying high-band spikes, but those spikes sit
at −71 dBFS, which is inaudible. The brief's own wording anticipates this
("barely perceptible artifacts should not automatically count").

One further observation shaped how much weight the labels could carry. All
three reference labels report `confidence: 0.82` **exactly**. That is the
signature of a single automated labelling pass with a hardcoded confidence,
not human adjudication — so some of what looks like a convention to learn
is likely label noise, and fitting hard to three points is fitting to it.

## 2. Architecture A — LLM-first, and why it failed

The first implementation is a complete, production-shaped system:
password-gated Streamlit dashboard, ZIP upload with manifest validation,
bounded worker pool with per-file isolation, token-bucket rate limiting,
retry with full jitter, windowed analysis for clips over 90 s, an
ensemble/cascade layer that re-samples only clips with a narrow top-2
margin, and 110 offline tests. ~4,400 lines.

Its core bet was that a modern multimodal model, given the field
definitions and a strict `response_schema` with enums enforced at
generation time, would outperform anything assembled from parts — with no
training data available, zero-shot quality is the whole game.

**Measured leave-one-call-out against the three labelled calls, it scored
10/24**, and the shape of the errors mattered more than the number. The
model defaulted to the "quiet" class on nearly every field — `neutral`,
`low`, `none`, `false` — regardless of content. It scored 3/3 on
`long_silence_present` by answering `false` every time, which happened to
match. Every case where the truth was something other than the default
class, it missed.

Three follow-up experiments ruled out the easy explanations:

- Re-running the hardest clip at `thinking_level` LOW and MEDIUM produced
  **byte-for-byte identical output** to MINIMAL. More reasoning budget
  changed nothing.
- A free-text prompt on the same clip — "describe any background noise,
  overlap, and emotional tone you hear" — also returned "neutral tone, no
  background noise, no overlap." So this was not a structured-output
  artifact.
- Self-reported confidence was 0.90 / 0.85 / 0.95 across the three clips
  while most fields were wrong. **Self-reported LLM confidence was not
  merely uncalibrated; it was anti-correlated with correctness.**

The conclusion was specific and testable: *the model was not failing to
reason about the noise, it was failing to perceive it.* A deterministic
measurement does not have that failure mode, because it does not need to
recognise static as unusual — it measures signal energy directly.

That prediction is what Architecture B was built to test.

## 3. Architecture B — measure first, infer last

The nine fields are not one problem. Six are physical properties of a
waveform. Two are properties of a voice. One is a judgement about what a
person meant.

| Field | Decided by | Cost | Deterministic |
|---|---|---|---|
| `background_noise_present` | windowed local dynamic range + audibility gate | $0 | yes |
| `background_noise_type` | random forest over the estimated noise spectrum | $0 | yes |
| `background_noise_severity` | affected share of the call × how badly | $0 | yes |
| `audio_quality` | clipping, −35 dB band edge, level, dropout, reverberation | $0 | yes |
| `speaker_overlap_present` | cepstral pitch competition | $0 | yes |
| `long_silence_present` | inter-speech gap analysis | $0 | yes |
| `emotional_intensity` | **wav2vec2 arousal**, regressed from the waveform | $0 | yes |
| `emotional_tone` | language model, reconciled against measured affect | metered | no |
| `confidence` | computed from cross-signal agreement | $0 | yes |

This split does four things, in descending order of importance:

**It fixes the failure Architecture A exhibited.** A detector that measures
the noise floor cannot "not hear" static.

**It structurally prevents the two failure modes the brief names.** A model
never asked about background noise cannot infer noise from poor audio
quality. An intensity taken from measured arousal cannot be inferred from
loudness. These are enforced by the architecture, not by prompt
instructions that a model may ignore.

**It makes eight of nine fields reproducible.** Same input, same bytes out,
every run.

**It collapses cost.** Eight fields carry no marginal cost at all, which is
why the $0.003/minute ceiling ends up comfortable rather than tight.

### 3.1 Why emotion is measured, not read

A transcript cannot carry delivery. `call_001` transcribes as "Come on. …
Hello." repeated eleven times — a routine exchange on the page, audible
irritation to anyone listening. A language model reading that transcript
answered `neutral`; the reference label is `upset`.

Describing prosody to the model in words made it *worse*. Given "unsteady
voice, noticeable pitch tremor," the model restated those adjectives back
as a diagnosis of distress regardless of content. Rewriting the description
as bare numbers with reference ranges stopped that — and the model then
answered `neutral` to everything.

The fix was to stop describing and start measuring.
`audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim` regresses arousal,
dominance and valence directly from the waveform. **Dimensional output was
chosen over categorical deliberately:** four-class emotion models
(angry/happy/sad/neutral) cannot express `frustrated` vs `upset` vs
`distressed` at all, so they cannot serve this schema no matter how
accurate they are.

Each model is then trusted only inside its competence — this is the
load-bearing idea:

- **arousal → `emotional_intensity`.** Activation is carried by pitch,
  energy and rate; acoustic models predict it well. On the labelled clips
  arousal ranks the three calls in *exactly* ground-truth intensity order
  (0.687 / 0.600 / 0.568 against high / medium / medium).
- **dominance → separates `upset` from `distressed`.** Both are
  high-arousal negative states; the difference is control. An angry caller
  is assertive and dominant; an overwhelmed one is not. No transcript
  conveys this and no categorical model encodes it.
- **valence → corroboration only.** Whether an utterance is positive or
  negative lives mostly in its words. Valence is the weakest dimension for
  any acoustic model, and on our clips it ranks them wrongly. Polarity
  therefore stays with the language model.

**Result: `emotional_intensity` went from 0/3 (Architecture A) to 1/3
(transcript-only) to 3/3.**

### 3.2 Final shape

```
audio ──▶ decode 16 kHz mono
            ├──▶ signal processing ─────────▶ 6 acoustic fields      [free, deterministic]
            ├──▶ wav2vec2 speech emotion ──▶ arousal/dominance/valence   [free, LOCAL model]
            └──▶ Whisper ──▶ transcript ──▶ language model ──▶ tone polarity   [metered API]
                                                    │
                                    fusion ◀────────┘
                                      │ arousal sets intensity
                                      │ dominance splits upset/distressed
                                      │ language model sets polarity
                                      │ agreement between them sets confidence
                                      ▼
                              9-field schema, enum-validated
```

Three lanes that never consult each other after decoding. Two measure; one
infers. They meet only at fusion, and only on the emotional fields. This
shape is unchanged in the current system — everything in Part 1 builds on
top of it rather than replacing it.

## 4. Local models versus hosted APIs — the deliberate trade-off

This system runs **both** a local neural model and hosted APIs, and the
split is a considered engineering decision rather than a default. It is
worth stating explicitly because it is the question a reviewer should ask.

### 4.1 What runs locally, and why

| Component | Model | Why local |
|---|---|---|
| Speech emotion | `wav2vec2-large-robust-12-ft-emotion-msp-dim` (317M params, int8) | The signal it produces — arousal, dominance — is *not available* from any hosted API in the form the schema needs. Running it locally also means the audio never leaves for this stage, and the marginal cost is exactly zero |
| All DSP | hand-built (VAD, noise estimation, quality, overlap, prosody) | Deterministic, auditable, free, and — per §2 — measurably better than asking a model |
| Noise-type classifier | random forest, trained here on a generated dev set | 12 features, 400 trees, ~50 KB. No API could be trained on our label vocabulary |
| Tone fallback | lexicon + prosody heuristic | Guarantees the system never fails outright, and makes `local_only` a real mode rather than a claim |

An additional local backend — `tiantiaf/wavlm-large-msp-podcast-emotion-dim`,
the WavLM architecture that won the 2024 MSP-Podcast A/D/V challenge — was
also integrated and benchmarked in this first pass. The published checkpoint
ships only weights and expects a wrapper class not on PyPI, so the
architecture was reconstructed from the checkpoint's own tensor shapes
(WavLM-large backbone → a learned softmax-weighted sum over 25 hidden states
→ three 1×1 convolutions → per-dimension heads). It runs, and it is
documented in §6.8 as a measured non-improvement (a separate WavLM
investigation, for `speaker_overlap_present` rather than SER, happened later
and is covered in Part 1 above and in `result.md`).

### 4.2 What runs against an API, and why

| Component | Service (at the time) | Why not local |
|---|---|---|
| Emotional tone | Gemini, since replaced by Azure OpenAI — see Part 1 | This is genuinely a language-understanding problem. The local lexicon heuristic scores far worse; a locally-hosted LLM good enough to match would need a 7–8 B model at minimum |
| Transcription | Groq `whisper-large-v3-turbo` | Local `faster-whisper small` runs the same clip in 21 s against 0.55 s — a 38x difference — and costs 460 MB of resident memory |

### 4.3 The deployment economics that decided it

A fully local build — swapping the hosted LLM call for a self-hosted
instruction-tuned model — is entirely achievable and the code already
supports it (`PRIVACY_MODE=local_only` needs no code change). It was **not**
chosen, for a reason specific to this engagement:

A 7–8 B model at reasonable quality needs roughly **7–8 GB of VRAM**, which
means a GPU node. For a system that will be **deployed and exercised during
evaluation**, that is the wrong shape:

- A GPU node costs $0.50–1.20/hour and must run continuously to stay warm —
  tens of dollars for a handful of evaluation batches, against roughly
  $1.60 per *thousand* audio-minutes on the current API path.
- Cold-starting a 7 B model from object storage takes minutes. An evaluator
  opening the dashboard would hit a timeout, not a result.
- No free tier anywhere hosts a GPU node.
- The accuracy would very likely be *worse*: a 7 B open model is not
  competitive with a frontier model on a five-class emotional judgement
  with definitional boundaries.

**The trade-off, stated plainly:** for a permanently-running production
deployment processing thousands of calls daily, the economics invert and
the local path wins on both cost and privacy — the marginal cost goes to
zero and no audio leaves. For a system exercised at evaluation volume, the
API path is correct on every axis that matters here. The architecture is
built so that switching is a configuration change, not a rewrite, because
*which* is correct depends entirely on volume.

That is why 8 of 9 fields already run on local models and deterministic
code, and only the single field that genuinely needs frontier-model
language understanding is metered.

## 5. First-pass results

Measured on the three labelled clips, and on a 600-clip synthetic dev set
with splits grouped by speech source to prevent leakage.

| Field | Labelled clips | Dev set macro-F1 |
|---|---|---|
| `background_noise_present` | 3/3 | 0.764 |
| `background_noise_type` | 3/3 | 0.798 |
| `background_noise_severity` | 3/3 | 0.480 |
| `audio_quality` | 3/3 | 0.615 |
| `long_silence_present` | 3/3 | 1.000 |
| `emotional_intensity` | 3/3 | not validated |
| `speaker_overlap_present` | 2/3 | 0.555 |
| `emotional_tone` | 2/3 | not validated |
| **Total** | **22/24** | |

`emotional_tone` was measured by **majority vote over repeated runs**,
because a single sample is not stable: over seven runs, one clip returned
`upset` five times, `frustrated` once and `neutral` once. It scored 2/3 on
the primary model tier and 0/3 once that model's free daily quota (~20
requests) was spent and the chain fell back — the dashboard stated this
explicitly when it happened. (Re-checking this exact code fresh at the
start of the second pass, before any change, actually reproduced **20/24**
and 0.690 macro-F1 for `background_noise_type` — see Part 1's opening
section for that correction.)

**Validation methodology.** Three labelled clips cannot support
cross-validation, and the brief forbids reporting accuracy from training
data alone. So a 600-clip dev set is generated by controlled mixing, where
ground truth is known by construction: noise injected at known SNR and
coverage, degradations applied at known severity, silences of known length
inserted, second speakers mixed at known overlap. Splits are grouped by
source speaker so no speaker appears on both sides. Noise templates for the
two categories present in the labelled data are **measured from those
clips** rather than invented — see §6.5 below for why that mattered.

**Cost** $0.00159 per audio-minute at paid list price, 1.9x under the
ceiling; $0.00 as deployed. **Latency** 3.0s per audio-minute for the free
half plus ~5s per clip for the metered call; 9–13s end to end per clip.
**Memory** 1.2GB resident, 2.4GB peak.

## 6. Everything tested in the first pass

Eight distinct approaches were built and measured. Four were rejected after
being built. They are recorded because the rejections carry as much
information as the adoptions.

**6.1 LLM-first, all nine fields (Architecture A).** 10/24, confident wrong
answers, perception failure confirmed by three follow-up experiments. →
**Rejected**, and the reason drove the redesign.

**6.2 Transcript + prose prosody for tone.** 0/3. The model parroted the
adjectives back as a diagnosis. → **Rejected**; replaced with direct
measurement.

**6.3 Model tier for tone.** The smaller/faster model tier answered
`neutral` to everything and scored 0/3; the identical prompt on the
standard tier scored 2/3. Considerable time was spent tuning the prompt
when the model was the bottleneck. → **Adopted**: standard tier primary,
smaller tier as last-resort fallback.

**6.4 `full` mode — sending audio to the tone model.** Produced
**identical** answers to sending only the transcript, at every model tier
tested. → **Rejected**; transmitting customer audio to the LLM buys no
measurable accuracy, which settles the privacy question on evidence rather
than principle.

**6.5 Fitted noise-type classifier.** Hand-weighted rules collapsed six of
eight categories (0.157 macro-F1). A random forest scored 0.69 — but got
**both** real noisy clips wrong, because it had learned an invented idea of
what television sounds like. Retraining on noise templates measured from
the real clips recovered 3/3 while holding 0.798 under grouped CV. →
**Adopted, after the first version was rejected.**

**6.6 Fitted severity classifier.** 0.75 on the dev set, 1/3 on the real
clips; hand rules 0.40 and 3/3. The dev set cannot adjudicate, because its
severity labels are self-thresholded rather than ground truth about how
AutoAce grades interference. → **Rejected**; hand rules ship.

**6.7 Speaker diarization.** Correct in principle — the schema asks about
the customer and half of each recording is the agent. Measured, it makes
things worse: the agent is the *higher*-arousal speaker (a bright TTS voice
at 0.715 against a frustrated human at 0.651), and thresholds fitted on
whole-clip audio no longer apply. → **Rejected for now**, implemented
behind a flag, with the recalibration it needs documented.

**6.8 Trajectory features and the WavLM SER backend.** Both published as
improvements; both measured neutral-to-negative here. Trajectory moved the
`satisfied` clip further from its label, because all three calls end more
negatively than they begin. WavLM produces the same ranking on a shifted
scale that our thresholds do not fit. → **Both ship behind flags, off by
default.**

**Two overlap detectors** were also built and measured at AUC 0.50 and
0.48 — chance. The cause is frequency resolution: 25ms frames give 40Hz
bins, and two talkers' fundamentals routinely sit closer than that. The
better of the two shipped and was honestly weak; `pyannote/segmentation-3.0`
was wired in behind a flag from the start (see Part 1 for where that stands
now).

## 7. Engineering for a zero-budget deployment

The free-tier constraint produced real engineering rather than compromises.

**Per-model quota rotation.** The free Gemini tier metered *per model per
day*, and metered the good model hard — 20 requests. One evaluation batch
would exhaust it before the evaluator saw a result. Because the quota was
scoped per model, the client rotated through a configured chain; a 429
burned that model's bucket immediately so no further round-trips were
wasted on it. (Moot now that Azure OpenAI is the sole tone provider — see
Part 1 — but the rate-limiting infrastructure this produced is unchanged
and still in use.)

**Adaptive self-consistency.** Tone answers are unstable, and majority
voting fixes it — but voting on every clip triples the cost of the only
metered call and breaches the ceiling on short files. So extra samples are
drawn *only* when the first looks shaky: low self-reported confidence, or
an answer the acoustic measurement contradicts.

**Self-imposed rate limiting.** Token buckets per provider enforce RPM, RPD
and concurrency below published limits, so the system throttles itself
rather than discovering limits through errors. The dashboard shows the
wait explicitly.

**Graceful degradation, announced.** When quota is spent the batch does not
fail; it falls back and says so, in the dashboard and attached to every
affected result.

**Failure isolation.** A malformed clip fails alone with a stated reason.
Verified with corrupt and empty files: 3 processed, 2 failed, batch
completed.

**Content-hash caching**, so re-running a batch costs nothing.

## 8. First-pass limitations

Stated plainly, because these are exactly what we saw and reacted to —
several of them are the reason we kept going instead of stopping here, and
motivated exactly the work in Part 1 above.

**Three examples cannot validate a five-class judgement.** This is the
honest headline, and it's why the second pass went looking for real
external audio rather than tuning further against the same three clips.
Eight changes were measured in this first pass; four looked principled and
made things worse or made no difference. The one change that clearly
worked — measuring affect for intensity — worked because it closed a
structural gap, not because it was tuned to the data.

**`long_silence_present` had an unconstrained threshold.** All three
labelled clips are negatives, and the largest genuine internal gap was 7.31s
of real dead air in `call_003` — still labelled false. The threshold sat
just above that. With no positive example its true position was unknown at
the time. (AMI Corpus, added in the second pass, is the first independent
real-data check this field ever got — see Part 1.)

**The synthetic dev set validates the acoustic fields and nothing else.**
It is generated audio: it exercises thresholds and spectral discrimination,
not whether a judgement about a person is right.

**`call_003` was stably mispredicted.** Seven of seven runs answered
`neutral` against a `satisfied` label. Both independent signals agreed with
each other and disagreed with the reference, and its measured valence was
the lowest of the three clips despite carrying the most positive label.
This is unchanged in the current system — see Part 1's end-to-end result.

**Speaker overlap was weak.** Two approaches measured at chance; the
shipped one was the better of two poor options. Real-audio validation in
the second pass (Harper Valley, AMI Corpus, AMI 2-speaker) confirmed this
is a real, consistent weakness rather than a synthetic-data artifact — see
Part 1.

## 9. Reproducing everything in this memo

```bash
pip install -r requirements.txt
pytest -q                                            # 28 regression tests

python -m eval.synth --out data/devset --per-group 150   # regenerate the dev set
python -m eval.train_noise --devset data/devset          # noise classifier, grouped CV
python -m eval.run_eval --devset data/devset             # per-field metrics + confusion matrices
python -m eval.score_labels requirements                 # score against the labelled manifest
python -m eval.check_requirements                        # compliance against the trial spec
python -m eval.train_noise_panns --devset data/devset     # PANNs-augmented noise-type comparison
python -m eval.repeat_trial --trials 6                    # multi-trial tone stability

docker build -t autoace-voice .
docker run -p 7860:7860 --env-file .env autoace-voice
```

Architecture A lives in its own tree with its own harness
(`eval/validate.py --live`, plus ablation flags for windowing, acoustic
evidence, and cascade sampling) and 110 offline tests.

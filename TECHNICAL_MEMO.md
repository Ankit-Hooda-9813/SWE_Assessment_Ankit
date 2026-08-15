# Technical Memo (v1)

**Voice tone and background-noise analysis for production call audio**
Submission for the AutoAce AI technical trial · Ankit Hooda

> This is the original submission memo — accurate as a record of what v1
> built and measured, but its numbers (22/24, tone 0/3, overlap AUC 0.66,
> $0.00159/min at 2.4GB RAM) have been superseded. **See
> `TECHNICAL_MEMO_V2.md` for the current architecture, current numbers, and
> an explicit v1→v2 comparison** — everything below this point predates that
> work and should be read as history, not current state.

---

## 0. Summary

Two complete systems were built for this trial, not one.

The first followed the obvious path: a multimodal LLM reading the audio and returning all nine
fields in a single structured call. It was built, deployed behind a dashboard, tested, and
**measured at 10 of 24 fields** on the labelled calls — while reporting confidence between 0.85 and
0.95 on almost every wrong answer.

That failure was diagnosed rather than patched. The second system was built to test the
diagnosis, and inverts the design: **measure everything that is physically measurable, and ask a
model only what genuinely requires judgement.** It scores **22 of 24**.

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

Combined: **~10,700 lines across two architectures, 36 test files, 8 distinct approaches built and
measured**, of which four were rejected on evidence after being built.

The rest of this memo is about *why* the second design wins, and where it still does not.

---

## 1. What the data showed before any modelling

Three properties of the provided clips were measured before an architecture was chosen. Each one
invalidated an approach that would otherwise have looked reasonable.

**The recordings are dual mono.** Left and right correlate at 1.0000, maximum sample delta 0.004.
There is no agent/customer channel to difference, so speaker overlap and speaker separation have to
be recovered acoustically from the mixture. Any design assuming stereo call recording is dead on
arrival.

**Background noise is episodic, not a continuous bed.** Every clip's inter-word gaps are digitally
silent — noise floors between −80 and −240 dBFS — yet two of the three are labelled noisy. The
noise is injected over spans rather than laid under the call. In `call_003` the high-band energy
ratio sits at 0.003–0.005 through clean speech and spikes to 0.15–0.21 at seconds 11, 16 and 32–33.

**Consequently a whole-clip SNR detector scores zero.** Clip-level SNR estimates come out at 47–55
dB for all three files — pristine — and miss both positives. Detection must be windowed and then
aggregated, with an *absolute* audibility gate rather than a purely relative one: `call_001` is
labelled noise-free despite carrying high-band spikes, but those spikes sit at −71 dBFS, which is
inaudible. The brief's own wording anticipates this ("barely perceptible artifacts should not
automatically count").

One further observation shaped how much weight the labels could carry. All three reference labels
report `confidence: 0.82` **exactly**. That is the signature of a single automated labelling pass
with a hardcoded confidence, not human adjudication — so some of what looks like a convention to
learn is likely label noise, and fitting hard to three points is fitting to it.

---

## 2. Architecture A — LLM-first, and why it failed

The first implementation is a complete, production-shaped system: password-gated Streamlit
dashboard, ZIP upload with manifest validation, bounded worker pool with per-file isolation,
token-bucket rate limiting, retry with full jitter, windowed analysis for clips over 90 s, an
ensemble/cascade layer that re-samples only clips with a narrow top-2 margin, and 110 offline
tests. ~4,400 lines.

Its core bet was that a modern multimodal model, given the field definitions and a strict
`response_schema` with enums enforced at generation time, would outperform anything assembled from
parts — with no training data available, zero-shot quality is the whole game.

**Measured leave-one-call-out against the three labelled calls, it scored 10/24**, and the shape of
the errors mattered more than the number. The model defaulted to the "quiet" class on nearly every
field — `neutral`, `low`, `none`, `false` — regardless of content. It scored 3/3 on
`long_silence_present` by answering `false` every time, which happened to match. Every case where
the truth was something other than the default class, it missed.

Three follow-up experiments ruled out the easy explanations:

- Re-running the hardest clip at `thinking_level` LOW and MEDIUM produced **byte-for-byte identical
  output** to MINIMAL. More reasoning budget changed nothing.
- A free-text prompt on the same clip — "describe any background noise, overlap, and emotional tone
  you hear" — also returned "neutral tone, no background noise, no overlap." So this was not a
  structured-output artifact.
- Self-reported confidence was 0.90 / 0.85 / 0.95 across the three clips while most fields were
  wrong. **Self-reported LLM confidence was not merely uncalibrated; it was anti-correlated with
  correctness.**

The conclusion was specific and testable: *the model was not failing to reason about the noise, it
was failing to perceive it.* A deterministic measurement does not have that failure mode, because
it does not need to recognise static as unusual — it measures signal energy directly.

That prediction is what Architecture B was built to test.

---

## 3. Architecture B — measure first, infer last

The nine fields are not one problem. Six are physical properties of a waveform. Two are properties
of a voice. One is a judgement about what a person meant.

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

**It fixes the failure Architecture A exhibited.** A detector that measures the noise floor cannot
"not hear" static.

**It structurally prevents the two failure modes the brief names.** A model never asked about
background noise cannot infer noise from poor audio quality. An intensity taken from measured
arousal cannot be inferred from loudness. These are enforced by the architecture, not by prompt
instructions that a model may ignore.

**It makes eight of nine fields reproducible.** Same input, same bytes out, every run.

**It collapses cost.** Eight fields carry no marginal cost at all, which is why the $0.003/minute
ceiling ends up comfortable rather than tight.

### 3.1 Why emotion is measured, not read

A transcript cannot carry delivery. `call_001` transcribes as "Come on. … Hello." repeated eleven
times — a routine exchange on the page, audible irritation to anyone listening. A language model
reading that transcript answered `neutral`; the reference label is `upset`.

Describing prosody to the model in words made it *worse*. Given "unsteady voice, noticeable pitch
tremor," the model restated those adjectives back as a diagnosis of distress regardless of content.
Rewriting the description as bare numbers with reference ranges stopped that — and the model then
answered `neutral` to everything.

The fix was to stop describing and start measuring.
`audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim` regresses arousal, dominance and valence
directly from the waveform. **Dimensional output was chosen over categorical deliberately:**
four-class emotion models (angry/happy/sad/neutral) cannot express `frustrated` vs `upset` vs
`distressed` at all, so they cannot serve this schema no matter how accurate they are.

Each model is then trusted only inside its competence — this is the load-bearing idea:

- **arousal → `emotional_intensity`.** Activation is carried by pitch, energy and rate; acoustic
  models predict it well. On the labelled clips arousal ranks the three calls in *exactly*
  ground-truth intensity order (0.687 / 0.600 / 0.568 against high / medium / medium).
- **dominance → separates `upset` from `distressed`.** Both are high-arousal negative states; the
  difference is control. An angry caller is assertive and dominant; an overwhelmed one is not. No
  transcript conveys this and no categorical model encodes it.
- **valence → corroboration only.** Whether an utterance is positive or negative lives mostly in
  its words. Valence is the weakest dimension for any acoustic model, and on our clips it ranks
  them wrongly. Polarity therefore stays with the language model.

**Result: `emotional_intensity` went from 0/3 (Architecture A) to 1/3 (transcript-only) to 3/3.**

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

Three lanes that never consult each other after decoding. Two measure; one infers. They meet only
at fusion, and only on the emotional fields.

---

## 4. Local models versus hosted APIs — the deliberate trade-off

This system runs **both** a local neural model and hosted APIs, and the split is a considered
engineering decision rather than a default. It is worth stating explicitly because it is the
question a reviewer should ask.

### 4.1 What runs locally, and why

| Component | Model | Why local |
|---|---|---|
| Speech emotion | `wav2vec2-large-robust-12-ft-emotion-msp-dim` (317M params, int8) | The signal it produces — arousal, dominance — is *not available* from any hosted API in the form the schema needs. Running it locally also means the audio never leaves for this stage, and the marginal cost is exactly zero |
| All DSP | hand-built (VAD, noise estimation, quality, overlap, prosody) | Deterministic, auditable, free, and — per §2 — measurably better than asking a model |
| Noise-type classifier | random forest, trained here on a generated dev set | 12 features, 400 trees, ~50 KB. No API could be trained on our label vocabulary |
| Tone fallback | lexicon + prosody heuristic | Guarantees the system never fails outright, and makes `local_only` a real mode rather than a claim |

An additional local backend — `tiantiaf/wavlm-large-msp-podcast-emotion-dim`, the WavLM
architecture that won the 2024 MSP-Podcast A/D/V challenge — was also integrated and benchmarked.
The published checkpoint ships only weights and expects a wrapper class not on PyPI, so the
architecture was reconstructed from the checkpoint's own tensor shapes (WavLM-large backbone → a
learned softmax-weighted sum over 25 hidden states → three 1×1 convolutions → per-dimension heads).
It runs, and it is documented in §6.8 as a measured non-improvement.

### 4.2 What runs against an API, and why

| Component | Service | Why not local |
|---|---|---|
| Emotional tone | Gemini | This is genuinely a language-understanding problem. The local lexicon heuristic scores far worse; a locally-hosted LLM good enough to match would need a 7–8 B model at minimum |
| Transcription | Groq `whisper-large-v3-turbo` | Local `faster-whisper small` runs the same clip in 21 s against 0.55 s — a 38× difference — and costs 460 MB of resident memory |

### 4.3 The deployment economics that decided it

A fully local build — swapping the Gemini call for a self-hosted instruction-tuned LLM — is
entirely achievable and the code already supports it (`PRIVACY_MODE=local_only` needs no code
change). It was **not** chosen, for a reason specific to this engagement:

A 7–8 B model at reasonable quality needs roughly **7–8 GB of VRAM**, which means a GPU node. For a
system that will be **deployed once and exercised once during evaluation**, that is the wrong
shape:

- A GPU node costs $0.50–1.20/hour and must run continuously to stay warm — tens of dollars for a
  handful of evaluation batches, against $1.59 per *thousand* audio-minutes on the API path.
- Cold-starting a 7 B model from object storage takes minutes. An evaluator opening the dashboard
  would hit a timeout, not a result.
- No free tier anywhere hosts a GPU node. The constraint for this trial was genuinely zero budget.
- The accuracy would very likely be *worse*: a 7 B open model is not competitive with a frontier
  model on a five-class emotional judgement with definitional boundaries.

**The trade-off, stated plainly:** for a permanently-running production deployment processing
thousands of calls daily, the economics invert and the local path wins on both cost and privacy —
the marginal cost goes to zero and no audio leaves. For a system deployed once for evaluation, the
API path is correct on every axis that matters here. The architecture is built so that switching is
a configuration change, not a rewrite, because *which* is correct depends entirely on volume.

That is why 8 of 9 fields already run on local models and deterministic code, and only the single
field that genuinely needs frontier-model language understanding is metered.

---

## 5. Results

Measured on the three labelled clips, and on a 600-clip synthetic dev set with splits grouped by
speech source to prevent leakage.

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

`emotional_tone` is measured by **majority vote over repeated runs**, because a single sample is
not stable: over seven runs, one clip returned `upset` five times, `frustrated` once and `neutral`
once. It scores 2/3 on `gemini-3.5-flash` and 0/3 once that model's free daily quota (~20 requests)
is spent and the chain falls back — the dashboard states this explicitly when it happens.

**Validation methodology.** Three labelled clips cannot support cross-validation, and the brief
forbids reporting accuracy from training data alone. So a 600-clip dev set is generated by
controlled mixing, where ground truth is known by construction: noise injected at known SNR and
coverage, degradations applied at known severity, silences of known length inserted, second
speakers mixed at known overlap. Splits are grouped by source speaker so no speaker appears on both
sides. Noise templates for the two categories present in the labelled data are **measured from
those clips** rather than invented — see §6.5 for why that mattered.

**Cost** $0.00159 per audio-minute at paid list price, 1.9× under the ceiling; $0.00 as deployed.
**Latency** 3.0 s per audio-minute for the free half plus ~5 s per clip for the metered call; 9–13 s
end to end per clip. **Memory** 1.2 GB resident, 2.4 GB peak.

---

## 6. Everything that was tested

Eight distinct approaches were built and measured. Four were rejected after being built. They are
recorded because the rejections carry as much information as the adoptions.

**6.1 LLM-first, all nine fields (Architecture A).** 10/24, confident wrong answers, perception
failure confirmed by three follow-up experiments. → **Rejected**, and the reason drove the redesign.

**6.2 Transcript + prose prosody for tone.** 0/3. The model parroted the adjectives back as a
diagnosis. → **Rejected**; replaced with direct measurement.

**6.3 Model tier for tone.** `gemini-3.5-flash-lite` answered `neutral` to everything and scored
0/3; the identical prompt on `gemini-3.5-flash` scores 2/3. Considerable time was spent tuning the
prompt when the model was the bottleneck. → **Adopted**: `flash` primary, lite as last-resort
fallback.

**6.4 `full` mode — sending audio to the tone model.** Produced **identical** answers to sending
only the transcript, at every model tier tested. → **Rejected**; transmitting customer audio to the
LLM buys no measurable accuracy, which settles the privacy question on evidence rather than
principle.

**6.5 Fitted noise-type classifier.** Hand-weighted rules collapsed six of eight categories (0.157
macro-F1). A random forest scored 0.69 — but got **both** real noisy clips wrong, because it had
learned my invented idea of what television sounds like. Retraining on noise templates measured
from the real clips recovered 3/3 while holding 0.798 under grouped CV. → **Adopted, after the
first version was rejected.**

**6.6 Fitted severity classifier.** 0.75 on the dev set, 1/3 on the real clips; hand rules 0.40 and
3/3. The dev set cannot adjudicate, because its severity labels are my own thresholding rather than
ground truth about how AutoAce grades interference. → **Rejected**; hand rules ship.

**6.7 Speaker diarization.** Correct in principle — the schema asks about the customer and half of
each recording is the agent. Measured, it makes things worse: the agent is the *higher*-arousal
speaker (a bright TTS voice at 0.715 against a frustrated human at 0.651), and thresholds fitted on
whole-clip audio no longer apply. → **Rejected for now**, implemented behind a flag, with the
recalibration it needs documented.

**6.8 Trajectory features and the WavLM backend.** Both published as improvements; both measured
neutral-to-negative here. Trajectory moved the `satisfied` clip further from its label, because all
three calls end more negatively than they begin. WavLM produces the same ranking on a shifted scale
that our thresholds do not fit. → **Both ship behind flags, off by default.**

**Two overlap detectors** were also built and measured at AUC 0.50 and 0.48 — chance. The cause is
frequency resolution: 25 ms frames give 40 Hz bins, and two talkers' fundamentals routinely sit
closer than that. The better of the two ships and is honestly weak; `pyannote/segmentation-3.0` is
wired in behind a flag.

---

## 7. Engineering for a zero-budget deployment

The free-tier constraint produced real engineering rather than compromises.

**Per-model quota rotation.** The free Gemini tier meters *per model per day*, and meters the good
model hard — 20 requests. One evaluation batch would exhaust it before the evaluator saw a result.
Because the quota is scoped per model, the client rotates through a configured chain; a 429 burns
that model's bucket immediately so no further round-trips are wasted on it.

**Adaptive self-consistency.** Tone answers are unstable, and majority voting fixes it — but voting
on every clip triples the cost of the only metered call and breaches the ceiling on short files. So
extra samples are drawn *only* when the first looks shaky: low self-reported confidence, or an
answer the acoustic measurement contradicts.

**Self-imposed rate limiting.** Token buckets per provider enforce RPM, RPD and concurrency below
published limits, so the system throttles itself rather than discovering limits through errors. The
dashboard shows the wait explicitly — an evaluator who sees "free-tier pacing, next call in 12 s"
reads a deliberate system.

**Graceful degradation, announced.** When quota is spent the batch does not fail; it falls back and
says so, in the dashboard and attached to every affected result.

**Failure isolation.** A malformed clip fails alone with a stated reason. Verified with corrupt and
empty files: 3 processed, 2 failed, batch completed.

**Content-hash caching**, so re-running a batch costs nothing.

---

## 8. Limitations

Stated plainly, because they are the part of this submission most worth reading.

**Three examples cannot validate a five-class judgement.** This is the honest headline. Eight
changes were measured; four looked principled and made things worse or made no difference. The one
change that clearly worked — measuring affect for intensity — worked because it closed a structural
gap, not because it was tuned to the data. Every number reported against three clips has a
confidence interval of roughly one clip, or 33 percentage points.

**`long_silence_present` has an unconstrained threshold.** All three labelled clips are negatives,
and the largest genuine internal gap is 7.31 s of real dead air in `call_003` — still labelled
false. The threshold sits just above that. With no positive example its true position is unknown.

**The synthetic dev set validates the acoustic fields and nothing else.** It is generated audio: it
exercises thresholds and spectral discrimination, not whether a judgement about a person is right.

**`call_003` is stably mispredicted.** Seven of seven runs answer `neutral` against a `satisfied`
label. Both independent signals agree with each other and disagree with the reference, and its
measured valence is the lowest of the three clips despite carrying the most positive label. I
cannot explain it from the audio.

**Speaker overlap is weak.** Two approaches measured at chance; the shipped one is the better of
two poor options.

---

## 9. External service disclosure

| Service | Model | Used for | Data sent | Retention |
|---|---|---|---|---|
| Google Gemini | `gemini-3.5-flash` (+ fallback chain) | emotional tone only | transcript + numeric measurements | free tier may be used for product improvement; paid tier is not |
| Groq | `whisper-large-v3-turbo` | transcription | **the audio file** | states it does not train on API data |

Transcription uploads the audio. That is enabled here for a 38× latency win and gated behind an
explicit `ALLOW_ASR_UPLOAD` flag — specifically so a mode described as "audio never leaves" cannot
quietly upload it. `PRIVACY_MODE=local_only` removes all external calls; `hybrid` with the flag off
keeps audio local and sends only derived text and numbers. Uploaded files are deleted the moment a
batch completes; nothing persists between runs.

Cerebras was also evaluated as a third provider and returned `402 Payment Required` on a new
account — its free tier does not cover chat completions. Recorded so the option is not
re-investigated.

---

## 10. Next steps, in priority order

1. **More labelled data.** Every threshold rests on three examples. This is the binding constraint,
   not the modelling — and it is why four measured improvements could not be distinguished from
   noise.
2. **`pyannote` for overlap**, replacing a detector measured at chance.
3. **Diarization plus recalibration** against customer-only audio; correct in principle, currently
   blocked only by having nothing to recalibrate on.
4. **A paid tone key**, removing the 20-request daily cliff for roughly $1.59 per 1,000
   audio-minutes.
5. **Revisit the local-LLM path at volume.** Above roughly 50,000 audio-minutes per month the GPU
   node amortises and the privacy story improves; the code already supports the switch.

---

## 11. Reproducing everything in this memo

```bash
pip install -r requirements.txt
pytest -q                                            # 28 regression tests

python -m eval.synth --out data/devset --per-group 150   # regenerate the dev set
python -m eval.train_noise --devset data/devset          # noise classifier, grouped CV
python -m eval.run_eval --devset data/devset             # per-field metrics + confusion matrices
python -m eval.score_labels requirements                 # score against the labelled manifest
python -m eval.check_requirements                        # compliance against the trial spec

docker build -t autoace-voice .                          # 1.82 GB
docker run -p 7860:7860 --env-file .env autoace-voice
```

Architecture A lives in its own tree with its own harness
(`eval/validate.py --live`, plus ablation flags for windowing, acoustic evidence, and cascade
sampling) and 110 offline tests.

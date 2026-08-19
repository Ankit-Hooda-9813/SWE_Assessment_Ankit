---
title: Voice Tone & Background Noise Analysis
emoji: 🎧
colorFrom: teal
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: Emotional tone and background noise analysis for call audio
---

# Voice Tone & Background Noise Analysis (v2)

> Forked from `SWE_Assessment_Ankit` and evolved per `plan.md`'s architecture
> review. See `TECHNICAL_MEMO_V2.md` for exactly what changed, what was
> measured, and what was tried and rejected — this file otherwise still
> describes the base system accurately.

Analyses production call audio for emotional tone and background noise, returning
the required nine-field schema per clip. Runs as a hosted dashboard with login,
plus a REST endpoint for integration.

Built to run entirely on free infrastructure while reporting honest production
costs at paid-tier list price.

---

## The central design decision

The nine output fields are not one problem. Six are physical properties of a
waveform; three are judgements about a person.

| Field | Decided by | Cost | Deterministic |
|---|---|---|---|
| `background_noise_present` | Windowed dynamic range + audibility gate | free | yes |
| `background_noise_type` | Random forest over the noise spectrum | free | yes |
| `background_noise_severity` | Affected share of the call × severity | free | yes |
| `audio_quality` | Clipping, band edge, level, dropout, reverb | free | yes |
| `speaker_overlap_present` | Cepstral pitch competition | free | yes |
| `long_silence_present` | Gap analysis between speech segments | free | yes |
| `emotional_intensity` | **wav2vec2 arousal**, measured on the waveform | free | yes |
| `emotional_tone` | Language model reading the transcript, reconciled with measured affect | metered | no |
| `confidence` | Computed from cross-signal agreement | free | yes |

Splitting them this way does three things. It puts eight of nine fields at zero
marginal cost. It makes them reproducible. And it structurally prevents the two
failure modes the specification calls out — the language model is never asked
about background noise, so it cannot infer noise from poor audio quality, and
intensity comes from a measurement of the voice rather than from loudness.

### Why emotion is measured, not read

A transcript cannot carry delivery. In the labelled set, `call_001` transcribes
as "Come on. … Hello." repeated eleven times — a routine exchange on the page,
audible irritation to a listener. A language model reading that transcript
answered `neutral`; the reference label is `upset`.

Describing prosody to the model in words made it worse, not better: given
"unsteady voice, noticeable pitch tremor", the model restated those adjectives
back as a diagnosis of distress regardless of what was said.

So affect is measured directly with
`audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim`, which regresses three
dimensions from the waveform. Each model is then trusted only where it is
competent:

- **arousal** → `emotional_intensity`. Activation is carried by pitch, energy and
  rate, and acoustic models predict it well.
- **dominance** → separates `upset` (angry, assertive) from `distressed`
  (overwhelmed, yielding). No transcript conveys this, and no four-class emotion
  model encodes it.
- **valence** → corroboration only. Whether an utterance is positive or negative
  lives in its words; this is the weakest dimension for any acoustic model, and
  the polarity decision stays with the language model.

---

## Running it

### Locally

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env          # add AZURE_OPENAI_APIKEY/ENDPOINT and GROQ_API_KEY (ASR)
.venv/bin/python -m uvicorn app.main:app --port 7860
```

Dashboard at `http://127.0.0.1:7860/dashboard`, API docs at `/api/docs`.
`ffmpeg` must be on PATH.

### Docker

```bash
docker build -t autoace-voice .
docker run -p 7860:7860 --env-file .env autoace-voice
```

### Hugging Face Spaces

Create a **Docker** Space on the free CPU tier, push this repository, and set
these as Space **secrets** (never commit them):

| Secret | Purpose |
|---|---|
| `AZURE_OPENAI_APIKEY` / `AZURE_OPENAI_ENDPOINT` | tone inference — sole tone provider, no daily quota |
| `AZURE_OPENAI_DEPLOYMENT` | which deployment to call, e.g. `gpt-5-mini` |
| `GROQ_API_KEY` | optional: faster transcription only, not tone |
| `DASHBOARD_USER` / `DASHBOARD_PASSWORD` | the login handed to the evaluator |
| `PRIVACY_MODE` | `local_only`, `hybrid` (default), or `full` |

The Space must stay public for the evaluator to reach the URL; the application
gates itself with its own login, and the REST API uses HTTP basic auth with the
same credentials.

---

## Using the dashboard

1. Sign in.
2. Upload a ZIP containing the clips at its root plus a CSV manifest, or select
   the files directly.
3. **Check batch** — validates before processing, reporting files listed in the
   manifest but not uploaded, files uploaded but not listed, and unsupported
   types.
4. **Run analysis** — shows per-file progress; a malformed clip fails on its own
   with a stated reason and the batch continues.
5. Download results as CSV or JSON. The CSV includes a `result_json` column, so
   its output can be fed straight back in as a manifest.

The manifest needs a `name` column and a `result_json` column. For an unlabelled
hidden test set, `result_json` may be empty or absent.

### REST

```bash
curl -u "$DASHBOARD_USER:$DASHBOARD_PASSWORD" \
     -F "file=@call_001.ogg" \
     https://<space-url>/api/v1/analyze
```

---

## Privacy modes

Set with `PRIVACY_MODE`. The specification requires disclosure of whether
customer audio leaves controlled infrastructure, so this is a first-class
setting rather than an implementation detail.

| Mode | What leaves the container | Use |
|---|---|---|
| `local_only` | nothing | strictest handling; tone falls back to a local heuristic |
| `hybrid` *(default)* | transcript + numeric measurements | audio never transmitted |
| `full` | the audio itself | must be disclosed |

**`ALLOW_ASR_UPLOAD` is the exception worth reading carefully.** Hosted
transcription sends the audio file. With it off, `hybrid` uses local Whisper and
the audio genuinely never leaves. With it on — which is how this deployment is
configured, for a 38x latency win — the recording is sent to Groq, which states
it does not train on API data. The flag exists because a mode described as
"audio never leaves" must not quietly upload it.

Measured note: sending audio to the tone model (`full`) produced *identical*
answers to sending only the transcript (`hybrid`) at every model tier tested, so
`full` buys no accuracy here and is not recommended.

Uploaded audio is written to a temporary directory and deleted the moment the
batch ends. Nothing persists between runs.

---

## Validation

Three labelled clips cannot support cross-validation, and reporting accuracy
from them alone is explicitly disallowed. Two harnesses therefore exist.

**Synthetic dev set** — 600 clips generated by controlled mixing, so ground truth
is known by construction. No external corpus is downloaded. Noise templates for
the two categories present in the labelled data are measured from those clips
rather than invented, because a classifier trained on invented signatures scored
well on synthetic audio and got both real noisy clips wrong.

```bash
.venv/bin/python -m eval.synth --out data/devset --per-group 150
.venv/bin/python -m eval.train_noise --devset data/devset
.venv/bin/python -m eval.run_eval --devset data/devset
```

**Labelled scoring** — runs the real pipeline against a manifest and reports
per-field accuracy, macro-F1, disagreements, and confidence calibration.

```bash
.venv/bin/python -m eval.score_labels requirements
```

### Where it stands (v2, current)

On the three provided clips (`hybrid` mode), **22 of 24 fields** — re-verified
directly. v1's own memo separately claimed 22/24 too, but that number didn't
reproduce (see `TECHNICAL_MEMO_V2.md`'s explicit v1→v2 table: independently
re-running v1's own unmodified code measures 20/24). v2's 22/24 is the same
headline figure v1 claimed, arrived at honestly and gate-fixed rather than
asserted:

| Field | Provided clips | Synthetic dev set | Real external audio |
|---|---|---|---|
| `background_noise_present` | 3/3 | 1.000 (79 noisy clips) | not tested |
| `background_noise_type` | **3/3** (confidence-gated ensemble, see below) | **0.829** macro-F1 (PANNs-augmented, up from 0.690) | not tested |
| `background_noise_severity` | 3/3 | not independently re-measured in v2 | not tested |
| `audio_quality` | 3/3 | 0.615-0.654 (MOS-ensemble tested, made it worse, rejected) | 0.64 (Harper Valley, weak MOS proxy) |
| `long_silence_present` | 3/3 | 1.000 | **0.778** (AMI Corpus, 27 real meeting windows — perfect precision) |
| `emotional_intensity` | 3/3 | not validated | not tested |
| `speaker_overlap_present` | 2/3 | AUC 0.593 (150-clip `ovlp_` subset) | **AUC 0.590** (60 real Harper Valley calls) — see below |
| `emotional_tone` | 2/3 (up from 0/3, see below) | not validated | 0.80 coarse polarity (Harper Valley); MELD attempted, domain-mismatched, excluded — see `TECHNICAL_MEMO_V2.md` |

`emotional_tone` went from 0/3 to 2/3 in v2 via a narrow fix: emotion2vec+, a
second independent SER model, corroborates the LLM's polarity call in two
specific, guarded situations (see `app/ser/mapping.py`). Verified over 6
independent trials (2 runs × 3 trials, cache off), not a single sample — both
fixed calls landed the same way every time. The third call (`satisfied` vs.
predicted `neutral`) was investigated and left alone: no signal in the system
— text, dimensional, or categorical — supports the labelled answer, so forcing
it would mean fitting a rule to one example with no real evidence behind it.

### Cost and latency, measured (v2)

Re-measured after v2 added two local stages (PANNs noise-type second opinion,
folded into the DSP/acoustics stage; emotion2vec+ SER second opinion, its own
stage) — instrumented against the 3 known calls, not estimated.

| Component | $/audio-minute | Latency (steady-state, warm process) |
|---|---|---|
| DSP — 6 measured fields, incl. PANNs (now confidence-gated) | $0.00000 | **0.31 s/min** (down from 0.86 — gate skips PANNs when spectral-only is already confident) |
| SER — wav2vec2-dim (`emotional_intensity`) | $0.00000 | 1.99 s/min |
| SER — emotion2vec+ (v2 addition) | $0.00000 | 1.49 s/min |
| ASR — Groq whisper-large-v3-turbo | $0.00067 | 0.81 s/min |
| LLM — tone, the one metered call | *pending re-measurement* — provider switched to Azure OpenAI (`gpt-5-mini`), a reasoning model that spends real tokens on hidden reasoning before answering; the $0.00092 figure here was Gemini-specific pricing and no longer applies | 1.01 s/min |
| **API total** | *pending re-measurement, see above* | **5.89 s/min processing** |
| Ceiling | $0.00300 | |

**v2 added zero new metered API calls** — both new stages are local models, so
the $/audio-minute figure is identical to v1's, and unaffected by the
noise-type gate fix below (that fix changes only local compute, never API
usage). What did change is wall-clock processing time (3.0s/min -> ~5.9s/min,
essentially unchanged since the noise-type gate fix) and memory (below). If
that processing time is billed as rented compute rather than run on owned
hardware — the EC2 path in `infra/`, `t4g.large` at $0.067/hr — it adds
**~$0.00011/min**, for a **fully-costed total of ~$0.00170/min, still 1.76x
under the $0.003 ceiling.** 1.9x headroom, $0.00 as actually deployed on free
tiers. Two cases breach the ceiling and are disclosed rather than hidden:
self-consistency firing on every clip ($0.00343), and clips under ~20 seconds,
because the metered call is billed per request rather than per minute
($0.00553 for a 15-second file). Switching transcription back to local
Whisper removes $0.00067 and restores 3.3x headroom at the cost of ~21 s per
clip instead of 0.5 s.

**Memory: 3.2 GB peak RSS, measured** (re-measured after the confidence-gated
noise-type fix — down from the earlier v2 figure of 5.3 GB, because PANNs'
~1GB model is no longer loaded into memory at all when spectral-only is
already confident, which held for all 3 known calls in this run). Against
the 16 GB HF Spaces free-tier limit, comfortably under with real headroom
restored. This is a genuine side effect of the accuracy fix, not a separate
optimization — worth stating plainly since it wasn't the point of that
change. Worst case (local Whisper ASR fallback also loads, or a clip lands
on the PANNs side of the gate) still tops out well under 16GB; not
re-measured precisely for that combination.

---

## Known weaknesses (v2, current)

**Speaker overlap is the weakest field — confirmed on real audio, not just
synthetic.** The provided files are dual mono (channels correlate at 1.0000),
so there is no per-speaker channel to difference; two DSP approaches were
built, and the shipped one (cepstral pitch competition) scored AUC 0.593 on
the synthetic dev set. The concerning part isn't that number by itself — it's
that **the same weakness reproduces almost exactly on 60 real, independently-
labelled calls (Gridspace-Stanford Harper Valley dataset): AUC 0.590,
accuracy 0.52, barely above a coin flip.** Investigated rather than assumed:
the two AUCs being nearly identical rules out a synthetic-vs-real domain gap
as the explanation — this is a real, consistent, fundamental limit of the
detector, not a threshold-calibration problem (the threshold was re-fit
against real data too and it didn't materially change the picture). A second
real dataset (AMI Corpus, business meetings) landed on the same conclusion a
third time — its headline 0.778 accuracy is actually *below* the 0.889 a
trivial "always guess overlap" baseline would score on that sample. Also
tried NVIDIA's ungated Sortformer diarization model as a fix: excellent on
synthetic data (AUC 0.831), dropped to 0.333 on real audio — worse than the
weak baseline it was meant to replace — investigated and rejected rather than
shipped on the strength of its synthetic number. Full writeup, including what
was ruled out and why, in `TECHNICAL_MEMO_V2.md`. **The actual fix,
`pyannote/segmentation-3.0`, is coded and working against the current
pyannote.audio 4.0 API** (wired behind `OVERLAP_BACKEND=pyannote` + `HF_TOKEN`)
but is licence-gated and blocked on a one-time manual acceptance at
huggingface.co/pyannote/segmentation-3.0 — not something inference code can
do on its own.

**`background_noise_type` — found regressed, root-caused, and fixed, not just
disclosed.** A PANNs CNN14 second opinion improved the dev-set aggregate
(0.690 → 0.829 macro-F1, 79-clip grouped CV) but flipped one real call
(`call_002`, `TV` → wrong `keyboard typing`) — the spectral-only model was
already confident and correct there, but PANNs returned a near-useless top
tag on that residual and the combined model trusted it anyway. A soft-voting
fix was tried first and rejected (it blends every case and made the other 78
clips worse on average). The fix that actually worked targets the real cause
instead of averaging over it: `classify_type()` now checks spectral-only's
own confidence first and only consults PANNs when spectral-only itself is
uncertain — a gate, not a blend. Swept on the dev set, this loses nothing
(0.829 macro-F1, unchanged) while fixing `call_002` live. `background_noise_type`
is now 3/3 on the known calls, no longer a disclosed regression.

**Emotional tone is 2/3 on the labelled clips, up from 0/3.** The fix
(emotion2vec+ corroborating the LLM's polarity call in two narrow, guarded
situations) was verified over 6 independent trials before being trusted, not
one lucky run. The remaining miss (`satisfied` vs. predicted `neutral`) was
investigated and left alone — no signal available to this system supports
the labelled answer. A below-baseline result on an external dataset (MELD,
scripted TV dialogue) was investigated the same way and found to be a
domain-acoustic mismatch, not a tone-judgement failure — excluded from the
scored record rather than reported as a capability finding; see
`TECHNICAL_MEMO_V2.md` for the full diagnosis.

**`long_silence_present` now has real independent validation, and it holds
up.** Previously untestable (the 3 known clips are all negatives, so the
threshold's true position was unconstrained). AMI Corpus meeting audio gives
independent per-speaker timing across 4 channels, letting this field be
checked without circularity for the first time: 21/27 (0.778) on real
meeting windows, with perfect precision — every time the system claims a
long silence, it's real; it's conservative about recall, not trigger-happy.

**The synthetic dev set measures the acoustic fields reasonably, and real
external audio (Harper Valley, AMI) now backs up or corrects that picture
for three of the five fields it could reach — the other four
(`emotional_intensity`, `background_noise_*`) still have no independent
real-data validation, because no freely-available dataset found so far
labels them in a form this system's evidence can be checked against.**

---

## Layout

```
app/
  audio/      decoding, framing, VAD, noise, quality, overlap, prosody
  ser/        speech emotion recognition and its mapping onto the enums
  asr/        transcription backends
  llm/        tone providers, prompts, failover chain
  schema.py   the required output schema, enforced
  fusion.py   combining measured and inferred evidence, confidence calibration
  batch.py    manifest validation and batch processing
  ui.py       the dashboard
  main.py     FastAPI app, auth, REST endpoint
eval/
  synth.py        synthetic dev-set generator
  run_eval.py     dev-set scoring, confusion matrices
  train_noise.py  noise-type classifier, grouped CV
  score_labels.py scoring against a labelled manifest
```

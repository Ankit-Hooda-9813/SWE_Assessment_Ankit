# How to test this system on a new dataset

Setup once, at the repo root (`/Users/ankitspc/Work/SWE_Assessment_Ankit_Hooda`):

```bash
set -a; source .env; set +a   # loads AZURE_OPENAI_APIKEY/ENDPOINT / GROQ_API_KEY / HF_TOKEN
```

All commands below are run from the repo root with `./.venv/bin/python`
(the project's own venv — already has every dependency installed, no `pip
install` needed unless you're adding a new eval script with a new import).

---

## 1. You have audio + AutoAce-format labels (a `labels.csv` with a `result_json` column)

This is the case for the 3 provided known calls, and for any new labelled
set AutoAce hands you in the same shape.

```bash
./.venv/bin/python -m eval.score_labels /path/to/folder --manifest /path/to/folder/labels.csv
```

- The folder just needs the audio files at its root; `--manifest` defaults
  to `<folder>/labels.csv` if omitted.
- Prints per-field accuracy + macro-F1, a confusion breakdown, every
  disagreement by name, and a confidence-calibration check.
- **`emotional_tone` involves a live LLM call and is not fully
  deterministic run to run** — see §4 before trusting a single run's tone
  number.

## 2. You have audio + labels in a *different* schema (like Harper Valley)

`eval/harper_valley_eval.py` is a worked example of adapting an external,
differently-labelled dataset — read it before writing a new one, most of it
copies over:

- **Mix multi-channel audio into mono** if the new dataset ships separate
  per-speaker tracks (`mix_channels()`) — this system expects a single
  mixed channel, matching real AutoAce call audio.
- **Derive `speaker_overlap_present` independently** from per-speaker
  segment timing if the dataset has it (`derive_overlap_truth()`) — apply
  the same 0.35s minimum-duration standard this codebase already uses
  (`Thresholds.overlap_min_sec` in `app/config.py`), not "any intersection,"
  or the comparison is unfair to the system.
- **Map coarser external labels to this schema's fields honestly, not
  exactly** — e.g. Harper Valley's 3-class emotion softmax was compared at
  coarse polarity only (satisfied→positive, {frustrated,upset,distressed}→
  negative, neutral→neutral), never claimed as 5-class validation.
  `TONE_POLARITY` in that file shows the exact mapping.
- **State plainly what the new dataset can't validate.** Harper Valley has
  no noise or long-silence labels — `harper_valley_eval.py`'s own docstring
  lists exactly what was and wasn't attempted, and why. Do the same for any
  new dataset rather than silently skipping fields.

Run it (or a copy adapted to the new dataset) with:

```bash
./.venv/bin/python -m eval.harper_valley_eval --n 25 --seed 7
```

`--n` controls sample size, `--seed` makes the sample reproducible.

## 3. You just have audio, no labels at all (a real hidden set)

Use the batch pipeline directly — this is what the hosted dashboard and the
brief's own batch-upload flow both call under the hood:

```python
import asyncio
from pathlib import Path
from app.batch import run_batch
from app.config import get_settings

result = asyncio.run(run_batch(Path("/path/to/folder"), get_settings()))
Path("predictions.json").write_text(result.to_json())
Path("predictions.csv").write_text(result.to_csv())
```

Produces schema-valid `AnalysisResult` JSON per file, with no ground truth
needed. Use this to sanity-check throughput/failures on a new batch before
you have labels for it, or to generate predictions for someone else to grade.

## 4. Tone specifically: never trust a single run

`app/llm/providers.py`'s own docstring documents that repeated identical
requests to the tone model do not produce identical answers, even at
temperature 0. A single `score_labels` run can make a fix look better or
worse than it is by chance. Use:

```bash
./.venv/bin/python -m eval.repeat_trial --trials 3
```

Runs the full pipeline against the 3 known calls 3× each (cache off),
reports the field-by-field average and the raw tone-answer distribution per
call. Edit the hardcoded `TRUTH` dict at the top of `eval/repeat_trial.py`
if testing a different labelled set with this same discipline.

## 5. Testing one specific measured field against the synthetic dev set

The 600-clip synthetic dev set (`data/devset/`, built from RAVDESS/CREMA-D)
has per-field ground truth in `manifest.json`. Existing scripts, useful as
templates for adding a new comparison:

| Script | What it checks |
|---|---|
| `eval/run_eval.py` | every measured field against the full dev set |
| `eval/score_labels.py` | any labelled batch (known calls or otherwise) |
| `eval/train_noise.py` | fits/validates the spectral noise-type classifier |
| `eval/train_noise_panns.py` | PANNs-augmented noise-type classifier, grouped CV |
| `eval/train_severity.py` | noise-severity classifier |
| `eval/train_quality.py` | audio-quality heuristic vs. NISQA/DNSMOS ensemble |
| `eval/tune_overlap.py` | cepstral overlap threshold, synthetic `ovlp_*` subset |
| `eval/tune_overlap_real.py` | same, against real Harper Valley audio |
| `eval/tune_noise_ensemble.py` | soft-voting vs. hard-switch for noise-type models (soft-voting rejected — regressed the dev set) |
| `eval/tune_noise_gate.py` | confidence-gated noise-type ensemble (shipped — fixes the one real regression at zero dev-set cost) |
| `eval/tune_overlap_sortformer.py` | NVIDIA Sortformer as an overlap backend (rejected — see `TECHNICAL_MEMO.md`) |
| `eval/ami_eval.py` | `long_silence_present` + `speaker_overlap_present` against real AMI meeting audio, independent per-speaker ground truth |
| `eval/meld_eval.py` | tone polarity against real Friends dialogue — **result excluded from the record**, see below |

**On `meld_eval.py` specifically: don't trust its raw accuracy number without
checking the baseline first.** It scored 10/30 (0.333), *below* the 0.533 a
naive majority-class guess would get on that sample. Investigated rather
than reported as-is: the cause was an acoustic domain mismatch (scripted
sitcom delivery is more vocally energetic than this system's real-call
calibration baseline, so "neutral" MELD lines trip the arousal-based
escalation logic), not a genuine tone-judgment failure — see
`TECHNICAL_MEMO.md`'s "MELD" section for the full diagnosis. This is the
concrete example to follow for any *new* dataset: a below-baseline score is
a reason to pull full diagnostics and check domain fit, not a number to
copy into a results table.

All follow the same shape: build features/predictions for the relevant dev-set
subset, run `sklearn.model_selection.GroupKFold` grouped by `source`
(`call_001`/`002`/`003`) so a fitted model is never evaluated on the same
speaker it trained on, report macro-F1 + confusion matrix.

## 6. Before trusting any result

- Read `TECHNICAL_MEMO.md` first — it documents what's actually shipped
  vs. tested-and-rejected, and why. A few things that look like they should
  help (NISQA/DNSMOS quality ensemble, noise-type soft-voting, NVIDIA
  Sortformer for overlap) were built, measured, and deliberately not shipped
  because they made things worse on a properly-sized test, despite looking
  good on 1-3 examples.
- Run `./.venv/bin/python -m pytest tests/` after any change — 28 tests,
  should stay green.

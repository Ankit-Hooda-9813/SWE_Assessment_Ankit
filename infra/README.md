# Deployment: Azure Container Apps (live), EC2 warm host, and HF Spaces

This repo ships three deployment paths against the same `Dockerfile` and the
same `app.main:app` entrypoint — nothing in `app/` changes between them.
**Azure Container Apps is what's actually live for this submission** — see
`README.md`'s "Live deployment" section for the URL and credentials, and
its "Azure Container Apps" subsection under "Running it" for the deploy
commands (`docker buildx build --platform linux/amd64 --push`, then
`az containerapp create`/`update`, secrets via `azure_set_secrets.sh` in
this directory). The other two paths below remain valid alternatives, not
what's currently serving the evaluator.

## Why not per-invocation serverless (AWS Lambda)

Priced out explicitly before building this: a monolithic single Lambda
running the whole pipeline, billed at peak memory for the full duration,
costs roughly $0.0045/audio-minute in compute alone — already over the
$0.003/audio-minute ceiling before the LLM tone call is added. Splitting into
per-stage Lambdas gets compute down to ~$0.0013-0.0017/min, but every
invocation re-pays the cost of loading the ~1-5GB of model weights this
pipeline now uses (wav2vec2-large, PANNs CNN14, faster-whisper) from cold,
and the total still clears the ceiling with close to no margin once the LLM
call is added back.

The alternative used here is the one already implicit in this codebase's
existing FastAPI/uvicorn design: **one warm process that loads every model
once and stays up**, not scale-to-zero-per-request. `app/batch.py`'s
`run_batch` already does this for anything processed through it — models are
module-level singletons (see `_ensure_loaded()` in `app/ser/emotion.py`,
`app/audio/quality_mos.py`, `app/audio/noise_panns.py`), loaded on first use
and reused for the life of the process. The only thing genuinely new in v2 is
*where that process runs*.

## Path 1: Azure Container Apps — **live for this submission**

The dashboard URL and credentials in `README.md`'s "Live deployment" section
point here. Consumption-plan Container Apps, `min-replicas=0` (scale to
zero between requests — a quiet-period request pays a cold-start, then the
container stays warm for subsequent ones, the same warm-process model as
every other path in this file).

- `azure_set_secrets.sh` (this directory) sets `AZURE_OPENAI_APIKEY`,
  `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT`, `GROQ_API_KEY`,
  `DASHBOARD_USER`/`DASHBOARD_PASSWORD`, `SESSION_SECRET`, `PRIVACY_MODE`,
  `TONE_PROVIDER_ORDER` on the live app via `az containerapp update
  --set-env-vars`.
- Built and pushed with `docker buildx build --platform linux/amd64 --push`
  rather than a plain `docker build`, because a normal build on Apple
  Silicon produces an arm64 image and Azure's Consumption plan runs amd64.
- Deployed with `az containerapp create`/`update --image
  <acr>/autoace-voice-trial:latest --cpu 4 --memory 8Gi`; 8GB was sized up
  from a smaller default after a real OOM under concurrent batch load
  (`worker_concurrency=2` means two clips' models can be resident at once).
- ACR Tasks (remote/cloud-side image builds) is restricted on this
  subscription tier (Azure for Students), and so is the `eastus` region —
  built locally instead and deployed to `eastasia`, one of the regions this
  subscription is actually allowed to use.

See `README.md`'s "Azure Container Apps" subsection under "Running it" for
the exact commands to build and redeploy.

## Path 2: Hugging Face Spaces (Docker SDK)

Unchanged from v1 — push to a Space with `sdk: docker` in the README
front-matter, HF builds `Dockerfile` and keeps the container warm while the
Space is awake. Free tier: 2 vCPU / 16GB RAM / 50GB disk, sized for in the
Dockerfile's opening comment. The simplest path if Azure Container Apps
(the one actually live for this submission) is unavailable, subject to
whatever the current HF Spaces free-tier Docker policy actually allows at
deploy time.

## Path 3: EC2 Graviton, start/stop on demand

For the hidden-set batch scoring run specifically — offline, not
latency-sensitive, and the one case where "keep a box running only as long
as you need it" is strictly cheaper than a managed platform's idle billing.

- Instance: `t4g.large` (2 vCPU / 8GB RAM, ARM/Graviton), ~$0.067/hr on-demand
- Same `Dockerfile`, same image — ECR or a fresh `docker build` on the box
- `start.sh` launches the instance, waits for the container health check to
  pass (models loaded, ~1-3 min cold), and prints the endpoint
- `stop.sh` terminates it — no idle billing between runs
- Sizing: v2's default model tier (wav2vec2-large + PANNs CNN14 +
  faster-whisper small) peaks at ~6-8GB RAM warm, comfortably inside 8GB;
  moving to the heavier tier (whisper-medium, or shipping emotion2vec+ as a
  live ensemble member rather than an eval-only comparison) would need
  `t4g.xlarge` (16GB) instead — see TECHNICAL_MEMO_V2.md for why that tier
  is not the current default.

```
./infra/start.sh      # launch, wait for warm, print endpoint
# ... run eval/run_eval.py or hit the endpoint directly ...
./infra/stop.sh        # terminate — stop paying the moment the run is done
```

Both scripts are thin wrappers over the AWS CLI; they assume `aws configure`
has already been run with credentials that can launch/terminate EC2
instances in the target account. Neither script is a substitute for reading
what it does before running it against a real AWS account — see the
comments in each file.

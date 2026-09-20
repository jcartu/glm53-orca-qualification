# Reproduce the study

There are two different activities here: inspect the recorded experiment, or run a new GPU campaign. Inspecting evidence does not require model weights or GPUs. Running the campaign uses all four configured GPUs and pauses an existing production server.

## 1. Inspect and re-score the recorded experiment

Install Python 3.12 or newer and zstd. Install the host analysis dependencies in a virtual environment:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

The release index is `evidence/index.json`. It contains the exact asset names, compressed hashes, unpacked-file hashes, coverage and exclusions. A remote URL is deliberately not specified until a release has actually been published.

For a local review bundle whose assets are in `release-assets/evidence-v1`:

```sh
.venv/bin/python tools/download_evidence.py \
  --manifest evidence/index.json \
  --release-base-url "file://$PWD/release-assets/evidence-v1/" \
  --output "$PWD/.local/evidence-text" \
  --assets text
```

`text` restores the raw text, source, configuration, prompt/response and score records without the large NumPy captures. Use `--assets all` with a fresh output directory for the full evidence tree. The downloader refuses an existing destination, verifies every selected asset and file, and labels a text-only restore as a subset.

Regenerate the core tables without calling a model:

```sh
.venv/bin/python tools/summarize_results.py \
  --evidence-root "$PWD/.local/evidence-text" \
  --output-dir "$PWD/.local/regenerated-results"
```

Numerical table values should agree. Source-file hashes may differ from the private as-run files where privacy normalization was necessary; the evidence index records both hashes. No numeric NumPy capture is altered.

The capability oracle regrader is `campaign/regrade_capability.py`. The behavior semantic scorer is `harness/r29/rescore_behavior.py`. Original strict scores are never overwritten. See the methods before comparing strict JSON compliance with answer correctness.

## 2. Prepare a matching GPU host

The tested envelope is Linux, four RTX PRO 6000 Blackwell 96 GB GPUs, Docker with NVIDIA support, and sufficient disk space for roughly a terabyte of checkpoint inputs plus evidence/cache working space. The serving image supplies PyTorch, CUDA, vLLM and LMCache.

The controller is intentionally conservative:

- It expects a running container named `glm53-prod` serving loopback port 5001.
- It waits for production requests to drain, captures that exact container ID, then pauses it.
- It uses `orca-qualification-test` on loopback port 5002 and rejects foreign GPU work in strict mode.
- It restores the original container, not the candidate, including on exceptions or signals.
- It checks health/model metadata and a useful arithmetic response after restoration.

This is not a generic production installer. Do not run it on shared GPUs or a host whose existing services you have not authorized it to pause. Do not use a hard kill that bypasses the restoration handler.

Set machine-local directories; these paths are not committed:

```sh
export ORCA_MODEL_ROOT="$HOME/models"
export ORCA_STATE_DIR="$PWD/.local/verification"
export ORCA_CACHE_ROOT="$PWD/.local/cache"
export TZ=Europe/Berlin
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export GPU_ISOLATION_MODE=strict
export NO_PROXY=localhost,127.0.0.1
export no_proxy=localhost,127.0.0.1
```

Use a fresh `ORCA_STATE_DIR` for a new verification. A published historical passed receipt is not valid local staging evidence.

## 3. Stage exact checkpoints

The checksum-verification CLI used here was **hf 1.22.0**, separately installed from the Python analysis environment. To reproduce that separation without replacing the pinned Python package:

```sh
python3 -m venv .local/hf-cli
.local/hf-cli/bin/python -m pip install huggingface-hub==1.22.0
export PATH="$PWD/.local/hf-cli/bin:$PATH"
```

Run analysis scripts with `.venv/bin/python` explicitly; do not activate a different environment and accidentally shadow the tested `hf` command.

Use your own Hugging Face account and obtain any required access approval directly from the publisher. These commands do not accept gates or provide credentials for you.

```sh
hf download orcarouter/GLM-5.3-Flash-Uncensored-NVFP4 \
  --revision ec0adf4f49c9570807cc11a5f650538c1893ae54 \
  --local-dir "$ORCA_MODEL_ROOT/Orca-GLM-5.3-Flash-Uncensored-NVFP4-ec0adf4f"

hf download orcarouter/GLM-5.3-Flash-Uncensored-FP8 \
  --revision 3cec42d6ed14ec197e328c09650c17fd3660c26a \
  --local-dir "$ORCA_MODEL_ROOT/Orca-GLM-5.3-Flash-Uncensored-FP8-3cec42d6"

hf download zai-org/GLM-5.3-Flash \
  --revision eb9eb208eb0d988989d07a6a12d0fdeb5f52574a \
  --local-dir "$ORCA_MODEL_ROOT/GLM-5.3-Flash-Original-FP8-eb9eb208"

hf download nvidia/GLM-5.3-Flash-NVFP4 \
  --revision 09b04e5e74bca08ca8549fc736d4cdd8624bfde3 \
  --local-dir "$ORCA_MODEL_ROOT/GLM-5.3-Flash-NVFP4-nvidia-09b04e5e"

hf download local-inference-lab/GLM-5.3-Flash-DFlash2 \
  --revision 713226ab03bc38afdf955c7450436c2f7176f6f8 \
  --local-dir "$ORCA_MODEL_ROOT/GLM-5.3-Flash-DFlash2-MXFP8"
```

The draft repository's current name omits `-MXFP8`; the local directory keeps the name used by the run. Its weight SHA-256 is `c033e03d47c7d5608596c8fc4e9336a1fe086eb781c08fe031be2bdea1614e58`. Its metadata files match the tested artifact. The draft carries **CC BY-NC-ND 4.0**, not the target's MIT license; read its upstream terms.

Verify locally and materialize the fixed history inputs from restored evidence:

```sh
.venv/bin/python campaign/verify_snapshots.py
.venv/bin/python fixtures/history/setup.py "$PWD/.local/evidence-text"
```

The verifier checks all three comparison snapshots and the draft conversion manifest, then writes under `ORCA_STATE_DIR/snapshot-verification`. The history materializer checks hashes and refuses to overwrite mismatched inputs. The NVIDIA control revision, configuration and index identity are also recorded in its serving receipt.

## 4. Build and select the corrected runtime

```sh
docker build \
  -f campaign/runtime-repair/Dockerfile.fp8-exclusions \
  -t glm53-orca-qualification-runtime:local \
  campaign/runtime-repair

export BATTERY_IMAGE=glm53-orca-qualification-runtime:local
```

The Dockerfile pins the R38 base by digest and adds the two scoped loader corrections. The harness resolves the selected image to an immutable local image ID before pausing production. A missing corrected image is an error; the unpatched base is never an automatic fallback. The historical test image ID is evidence, not a registry address another host can pull.

## 5. Run a fresh campaign

Only after reviewing the guard behavior and confirming the four GPUs and production service are available:

```sh
export BATTERY_ROOT="$PWD/.local/runs/full-qualification"
export BATTERY_CONTAINER=orca-qualification-test
export BATTERY_PORT=5002

.venv/bin/python -u campaign/campaign.py \
  --stages control,original-reference,ablated-reference,candidate,speculative,cache,fidelity,soak
```

`BATTERY_ROOT` must be new. All declared attempts are retained, including failed cases and unsupported observations. An exit code of 1 can mean a scored failure or an unavailable required gate; inspect the receipts instead of equating process completion with qualification.

The standalone `collision` phase exists to reproduce the R38-compatible policy experiment without repeating successful quality tests. Use another fresh `BATTERY_ROOT` for any separate invocation.

These are the tested conservative profiles, not an invitation to enable full CUDA graphs or substitute a different precision/backend while retaining the old scores. If a setting changes, record a new campaign and keep the old one.

## 6. Verify publication tooling offline

```sh
.venv/bin/python tests/test_evidence_tools.py
```

The regressions use synthetic temporary fixtures, no real credentials, no model requests and no GPUs. They exercise numeric byte preservation, exact JSON number preservation, credential and private-metadata handling, source mutation, archive traversal/type/expansion defenses, checksum failures and atomic installation.

A real export is performed only after all writers have stopped and production restoration has been verified. The complete source-root inventory, exclusion reasons and redaction ledger are reviewed before any upload. GitHub publication and licensing require explicit approval; a timed-out dialog is not approval.

This study has one [manually reviewed narrative literal](../evidence/approved-canaries.json) that resembles a credential assignment but contains no credential. The [review record](../evidence/privacy-review.json) explains why it is preserved and identifies the twenty diagnostic artifacts that remain byte-identical. When re-exporting this study, add `--approved-canaries "$(cat evidence/approved-canaries.json)"`; the option takes a JSON list, not a file path. No other content is exempt merely because it appears in a test, fixture or example.

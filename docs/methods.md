# Methods

This document describes how the `orcarouter/GLM-5.3-Flash-Uncensored-NVFP4`
qualification was actually run on this rig, what each stage measures, and what
each measurement can and cannot support. Results are in `findings.md`, with
the concise overview and machine-readable tables in `README.md` and `results/`.

Evidence paths in this document are relative to the evidence root
(`orcarouter-qualification-20260919T104457048Z/`). They resolve through
`evidence/index.json` to checksummed release-asset archives; bulky raw evidence
is not stored in Git history. `campaign/` is the portable harness derived from
the archived as-run sources; it also executes `operations-01`. Earlier exact
source snapshots remain in the evidence, and `source-provenance.json` records
portability changes. `harness/r26` and `harness/r29` come from the shared lab
repository (<https://github.com/jcartu/glm53-flash-field-lab>); `harness/bench`
pins the decode benchmark from `local-inference-lab/llm-inference-bench`.

## 1. Scope and acceptance contract

The campaign is a bounded, full-stack qualification of one candidate checkpoint
on one four-GPU host. It is not a claim of universal equivalence to the
original model or of indefinite stability. The written contract is
`acceptance.json`; the points that shape every later section are:

- No automatic promotion. The original production container is restored after
  every GPU campaign and its health plus a real inference are re-checked.
- Fidelity is measured, not thresholded: "Report measured KL/PPL/top1; no
  invented universal KLD pass threshold." KLD does not establish retained
  task capability.
- Capability is paired against the current production model with identical
  prompts, seeds and scoring. Errors and truncation remain failures. A loss
  greater than five percentage points on either functional suite is an
  operational regression flag, not a statistical equivalence margin; a
  non-significant difference on a small suite is not evidence of equivalence.
- Performance requires matched C1/C4/C8 cells at 0/32K/128K context, two
  45-second sustained trials, prefill, steady server counters and a
  mixed-traffic collision run. Every sample is preserved; a trial-to-trial
  coefficient of variation above 2% is flagged rather than dropped. Any
  foreign GPU process during a performance run invalidates that run.
- Generated code is executed only in bounded, no-network, no-GPU containers.
  No harmful-compliance benchmark is claimed; over-refusal is probed with
  benign prompts only.

## 2. Subjects under test

| Role | Repository @ revision | Format | Notes |
|---|---|---|---|
| Original reference (`original_fp8`) | `zai-org/GLM-5.3-Flash` @ `eb9eb208eb0d988989d07a6a12d0fdeb5f52574a` | block-FP8 as published by Z.ai (62 shards, 76,108 tensors) | Source of the tokenizer used for every fidelity panel. |
| Abliterated reference (`orca_fp8`) | `orcarouter/GLM-5.3-Flash-Uncensored-FP8` @ `3cec42d6ed14ec197e328c09650c17fd3660c26a` | block-FP8, structurally identical to the original (0 name/dtype/shape differences) | Publisher describes a refusal-direction ablation baked into the FP8 shards. |
| Candidate | `orcarouter/GLM-5.3-Flash-Uncensored-NVFP4` @ `ec0adf4f49c9570807cc11a5f650538c1893ae54` | compressed-tensors `nvfp4-pack-quantized` routed experts, BF16 elsewhere (65 shards, 111,346 tensors, 205,079,179,568 bytes) | 889 MTP tensors and 347 vision tensors present. 8,291 expert gate/up pairs carry distinct `weight_global_scale` values. |
| Production control (`production`) | `nvidia/GLM-5.3-Flash-NVFP4` @ `09b04e5e74bca08ca8549fc736d4cdd8624bfde3` | `modelopt_fp4` | The checkpoint `campaign/campaign.py` designates as production (served by `glm53-prod` under the name `GLM-5.3-Flash-NVFP4`); the paired-capability control. |

Publisher claims about these checkpoints (quantization recipe, abliteration
method, refusal rates, a wikitext KLD of 0.073) come from the model cards on
Hugging Face and are cited only as the publisher's description. None of them
is used as a proof of our measurements, and our measurements are not an
attempt to reproduce them exactly (Section 6.6).

Receipts: `preflight.json`, `*.hf-receipt.json`,
`staging-verification.json`, `*.tensor-schema.json`. Private access-approval
material is excluded from the public bundle; each reader must obtain their own
required publisher approval.

## 3. Rig, runtime and campaign machinery

- Host: one workstation with four NVIDIA RTX PRO 6000 Blackwell (GB202GL)
  GPUs, always tensor-parallel 4. Production runs in the Docker container
  `glm53-prod` (id `0f81f0e6…`) on the lab's R38 vLLM image
  `localinferencelab/vllm@sha256:f41ca8bb…` (vLLM
  `0.26.1rc0+glm53.r38.vllm66c29357`).
- Coordinator: `harness/r26/run_qualification.py` (imported from the lab
  repository). It drains and pauses production, records the production
  container identity (`*/production-before-pause.json`), enforces strict
  foreign-process exclusion on GPUs 0–3 with continuous health sampling
  (`*/gpu-isolation-events.jsonl`, `*/gpu-health-preflight.json`), runs the
  phases serially with per-phase wall-clock budgets (`*/phase-plan.json`), and
  finally restores the original container and checks `/v1/models` plus a real
  arithmetic completion (`*/production-restored.json`,
  `*/production-restored-inference.json`).
- Campaign script: `campaign/campaign.py` defines the phases (`pilot`,
  `control`, `original-reference`, `ablated-reference`, `candidate`,
  `speculative`, `cache`, `fidelity`, `soak`), the per-arm serving profiles,
  and the sub-suites each phase runs. Every boot writes `<label>.launch.json`
  (full environment and extra arguments), `<label>.model-contract.json`
  (checkpoint revision, quantization, speculation, DCP, image digest),
  `<label>.boot.log`, `<label>.docker.log`, `<label>.gpu.txt` and
  `<label>.metrics.txt`. Gate outcomes are appended to `*/gates.jsonl`; the
  human-readable timeline is `*/battery.log`.
- Fail-closed phase status: a phase's `execution_passed` is false when any
  sub-suite exits non-zero. The sub-suites exit non-zero whenever any case
  fails, so a `GATE FAIL phase-execution:<phase>` line means "not every case
  passed", not that the runtime crashed. Crashes are distinguished by the
  boot gate, the docker log and per-record `runtime_error` fields.
- Source pinning: every campaign directory stores `source-manifest.json`
  (SHA-256 of each script involved) and a `source-snapshot/` copy of the
  campaign and lab scripts as they were when the phase ran.

Runtime images used across the campaign:

| Image digest | Content | Used by |
|---|---|---|
| `sha256:f41ca8bb…` | Production R38 image, unmodified | production; `pilot-01`; `runtime-diagnosis-01` |
| `sha256:d0d3fca5…` | R38 plus `runtime-repair/compressed_tensors_moe_w4a4_nvfp4.py` (repair v1) | `runtime-diagnosis-02/03/04`, kernel probe |
| `sha256:913478ce…` | repair v1 plus `runtime-repair/fp8.py` (repair v2) | `runtime-diagnosis-05`, `fidelity-01/02`, `gauntlet-01/02`, `serving-diagnosis-01/02`, behavior rescoring sandbox |

The two repair Dockerfiles (`runtime-repair/Dockerfile`,
`runtime-repair/Dockerfile.fp8-exclusions`) copy exactly those two Python
files over the R38 image; no other runtime code changes and no checkpoint file
is modified. The production control arm in the gauntlets ran on the repair v2
image too. Its `modelopt_fp4` path lives outside the two patched files, so the
patch does not touch the control's kernels; this is a reading of the
Dockerfiles, not a separately measured fact.

## 4. Integrity, provenance and lineage checks

1. Explicit access. Preflight recorded that both Orca repositories were gated
   and that an "Ask" tool timeout had auto-selected approval; the campaign
   treated that as no authorization (`preflight.json`). Weights were only
   downloaded after the explicit statement recorded in `access-approved.json`.
2. Remote checksum verification. `hf cache verify --fail-on-missing-files`
   was completed for all three downloaded snapshots at the pinned revisions
   (`candidate.hf-receipt.json`: 74 remote files; `orca_fp8.hf-receipt.json`:
   72; `original_fp8.hf-receipt.json`: 73).
3. Tensor contract (`campaign/verify_snapshots.py` →
   `staging-verification.json`, `*.tensor-schema.json`): the safetensors
   index is complete and byte-consistent with the shards, the declared
   quantization matches the expected format per arm, MTP and vision tensors
   are present, and FP32 auxiliary tensors are counted. The candidate's
   gate/up `weight_global_scale` pairs were audited here (8,291 mismatched
   pairs, examples recorded); both FP8 checkpoints have 0.
4. FP8 structural parity: original and abliterated FP8 have identical tensor
   names, dtypes and shapes (`fp8_structural_parity.different_count = 0`).
5. FP8 lineage (`campaign/audit_fp8_lineage.py` → `fp8-lineage.json`): every
   tensor outside the broad declared residual-writer families
   (`.down_proj.`, `.o_proj.`, `.eh_proj.`, `.embed_tokens.`) was hashed in
   both checkpoints: 51,139 tensors, 219,196,705,656 bytes per checkpoint,
   0 mismatches; 24,969 writer tensors excluded. This supports the claim that
   the abliterated checkpoint derives from the pinned original. It does not
   verify the ablation algorithm or semantic equivalence.
6. CPU preflight of the scoring machinery (`cpu-preflight.json`): numeric
   KL oracle, identity KL/JS, refusal of non-normalized distributions,
   integer verifier positive/negative cases, a real sandbox execution and a
   denied-import negative, phase failure propagation, 16 prepared token
   panels, 12 rendered vision images, 14 restoration unit tests and 10 cache
   gate tests. Four preflight defects were fixed before any GPU work
   (missing sandbox stdin, cleanup recognition, padded output vocabulary vs
   tokenizer length, corpus hash association).

## 5. Runtime diagnosis and repair

The candidate is a compressed-tensors NVFP4 checkpoint whose config declares
dynamic W4A4 (4-bit weights, 4-bit activations). The runtime work established
what the R38 runtime actually does with it and repaired two loader defects in
an isolated image. Method:

1. Boot the candidate as published on the unmodified image and run the API
   contract suite (`pilot-01`).
2. Boot the candidate with the Marlin W4A16 MoE backend and the two FP8
   references on the same image; run three bounded semantic checks per arm
   (literal echo `ORCA_OK_7319`, arithmetic `17*23`, a one-word translation)
   through the served API (`runtime-diagnosis-01`).
3. Statically trace the W4A4 and FP8 loaders in `runtime-source/` (the
   vendored vLLM tree), then write the minimal repairs in `runtime-repair/`.
4. Prove the NVFP4 scale correction numerically against an independent
   reference (`runtime-repair/scale_kernel_probe.py` →
   `runtime-diagnosis-02/scale-kernel.json`): separately dequantized
   gate/up/down with their own stored global divisors, BF16 linear ops, FP32
   gated/clipped activation; report relative RMSE for the uncorrected kernel
   (negative control), the corrected kernel, and a CUDA-graph replay with
   changed inputs.
5. Prove the FP8 exclusion repair with an actual-library regression over
   every matrix of the checkpoint
   (`runtime-repair/check_fp8_exclusions.py` → `fp8-exclusion-regression.json`).
6. Re-run the bounded semantic checks on the repaired images
   (`runtime-diagnosis-02` … `-05`).
7. Independent review of both repairs against the vendored source
   (`OrcaRuntimeAudit.json`), with residual limitations recorded.

Scope statement that applies to everything downstream: every GPU result for
the candidate in this campaign was produced under the corrected Marlin W4A16
execution path (4-bit weights dequantized to 16-bit activations), not under
the W4A4 recipe the checkpoint declares. W4A4 is impossible with this
checkpoint on this runtime (it ships no activation global scales) and now
fails closed. Nothing here qualifies W4A4 serving, other activation layouts,
checkpoints with missing weight scales, or per-expert FP8 exclusion families.

## 6. Fidelity: full-vocabulary teacher-forced divergence

Code: `campaign/fidelity_probe.py` (`prepare`, `capture`, `compare`).

### 6.1 Corpus and token panels

- Primary corpus: `Salesforce/wikitext` revision
  `b08601e04326c79dfdd32d625aee71d232d685c3`, config `wikitext-2-raw-v1`,
  split `test`, single parquet file (SHA-256 `5f1bea06…`). Non-empty rows are
  kept verbatim and joined with `\n\n`. Twelve disjoint 2,048-token windows
  start at offsets 0, 2048, …, 22528. Each window is one independent prompt;
  2,047 positions per window are scored, giving 12 × 2,047 = 24,564 primary
  positions.
- Supplemental panels: four campaign-authored CC0 windows (representative
  prose, Python code and tests, quantitative reasoning, structured data and
  tool use), 2,047 positions each (8,188). They are descriptive strata and
  are excluded from the primary headline.
- Tokenization: one tokenizer, loaded offline from the pinned original
  checkpoint (`tokenizer.json` SHA-256 `19e77364…`, `tokenizer_config.json`
  `98b12715…`); both files are byte-identical across all three arms, so all
  arms score the same token IDs. The tokenizer has 154,856 tokens; the model
  output has 154,880 rows. The first preparation attempt failed on exactly
  this distinction (`fidelity-panels-prepared.json.failure.json`); the
  vocabulary contract was then fixed so every one of the 154,880 model output
  rows, including the 24 padded non-token rows, is captured and normalized.
- Prepared panels and the token-ID hash of the whole suite are frozen in
  `fidelity-panels-prepared.json` (`fidelity-panels.json` is the source spec).

### 6.2 Capture

Offline `vllm.LLM.generate` inside the repair v2 image with `--network none`,
one prompt at a time, `max_tokens=1`, `prompt_logprobs=-1` (full vocabulary),
`flat_logprobs=True`, `logprobs_mode=raw_logprobs`, temperature 0, no prefix
cache, no speculation. The semantic point is the raw prompt log-probability
before any sampling processor. Each row is checked to contain every ID
0..154879 exactly once (allowing R38's one equal-valued duplicate gold entry),
stored as float32, and independently renormalized in float64 (tolerance
1e-3 on |logsumexp|). One panel is 2,047 × 154,880 float32 values; the three
arms together are about 60 GB of raw distributions
(`fidelity-02/fidelity/<arm>/`), which is why these files ship as release
assets rather than in Git.

Runtime identity, identical for all three arms except the quantization
argument: TP 4, DCP 1, `dtype bfloat16`, `kv_cache_dtype fp8`, `enforce_eager`,
`max_model_len 4096`, `max_num_seqs 1`, `max_num_batched_tokens 256`,
`block_size 256`, `cp/dcp_kv_cache_interleave_size 4`,
`mamba_cache_mode align`, attention backend `B12X`, MoE backend `marlin`,
linear backend `triton`, `kda_prefill_backend flashkda`,
`language_model_only`. Requested and effective engine configuration are both
recorded in `fidelity-02/fidelity/<arm>/manifest.json`; the exact arguments
are `fidelity-02/fidelity-<arm>.llm-args.json`.

### 6.3 Comparison

`compare` streams both arms' rows in float64, renormalizes, and computes per
position: KL(reference ‖ candidate) (the headline, nats per token), the
reverse KL, Jensen–Shannon divergence, gold-token NLL for both arms, and top-1
agreement. Aggregates are reported for the primary WikiText positions, all
positions, per corpus, per stratum, and per window
(`fidelity-02/kld-<reference>-to-<candidate>/summary.json`); the complete
per-position arrays are `per-position-metrics.npz`. Perplexity is
exp(mean NLL); the perplexity ratio is exp(mean NLL difference).

Three directed comparisons were run: original FP8 → Orca FP8 (effect of the
ablation at source precision), Orca FP8 → Orca NVFP4 (effect of
quantization), original FP8 → Orca NVFP4 (combined effect).

### 6.4 Uncertainty

No confidence interval is reported. The twelve windows are deterministic
contiguous slices, not an independent random sample, and a position bootstrap
would understate window and corpus dependence. Instead the summary reports
the dispersion of per-window means (min, max, mean, population std) and ships
every per-position value.

### 6.5 What the numbers mean

KL is a per-token distributional distance in nats. It is not a percentage of
"intelligence retained", it is not a pass/fail score, and it does not
predict task accuracy: a model can move probability mass among near-synonyms
(large KL, no task effect) or shift a single decisive token (small mean KL,
large task effect). It also includes runtime contributions: the corrected
Marlin path adds one BF16 rounding per up-projection element (Section 5), and
FP8 KV cache is part of both arms' treatment. The measurement is a
configured-runtime comparison of three checkpoints under one identical
execution path, which is exactly what makes the three rows comparable to each
other.

### 6.6 Relation to the publisher's number

The candidate model card reports PPL, KLD and top-1 against "the FP8-dequant
reference" over 24,564 predicted tokens of wikitext-2. Our primary panel has
the same size by construction, but the publisher's exact token IDs, window
selection, reference execution (weight-only forward vs served runtime) and
KV precision are not available, so the published 0.073 is not a reproduction
target. The three campaign arms are directly comparable to each other, not to
that scalar (`uncertainty.published_replication_limitation` in each summary).

## 7. Capability suite (60 + 6 tasks)

Code: `campaign/workload_probe.py --mode capability`; fixtures:
`fixtures/capability/tasks.json` (the frozen `cap-battery` suite, SHA-256
`d1048fcd…`) and `fixtures/workload_fixtures.json`.

- Tasks: A01–A12 arithmetic, B01–B12 date intervals, C01–C12 Python coding,
  D01–D12 logic, E01–E12 state tracking (60 base tasks, integer/string/code
  verifiers), plus six explicit fixtures X01–X06: long reasoning,
  non-English, JSON schema, multi-turn tool workflow, and two benign
  over-refusal prompts. No language-model judge anywhere.
- Generation: identical for every arm. `temperature 0.0`, `top_p 1.0`,
  `seed 260919`, `max_tokens 4096`, `reasoning_effort low`, concurrency 8,
  900 s request timeout. Every case is attempted once; there are no retries.
- Integer verifier: the answer must appear among integers extracted from an
  `Answer:`/`Final answer:` or `\boxed{}` pattern, or from the last three
  non-empty lines of the visible answer (`extract_int_candidates`). This is
  deliberately content-only and tolerant of formatting; the extracted
  candidates are recorded so any lenient match can be audited.
- Code verifier: the model's `f` is executed on hidden cases inside a
  pinned-image `runc` container with no network, no mounts, no GPU, read-only
  root, dropped capabilities, an unprivileged user, bounded CPU/PIDs/memory/
  time/output and an AST/builtin/import allowlist. There is no host-execution
  fallback. Out-of-contract code (extra top-level statements, disallowed
  imports) fails the task; this is stricter than the historic host-executed
  `cap-battery` oracle, so absolute code scores are not comparable to older
  runs.
- Fail-closed accounting: HTTP errors, truncation (`finish_reason=length`),
  missing usage, protocol issues and sandbox infrastructure errors are
  recorded as failures with a `failure_kind`, never excluded. Each task's raw
  request, visible answer, reasoning, usage and verifier output is one JSON
  record (`<arm>-capability/records/<id>.json`).
- Oracle audit and correction (`code-oracle-audit.json`,
  `noncode-oracle-audit.json`, `OrcaOracleMathDates.json`,
  `OrcaOracleLogicState.json`, `oracle-corrections.json`,
  `campaign/regrade_capability.py`): every answer key was recomputed
  independently from the prompts with standard-library implementations. The
  audit files are timestamped 19:07–19:08 local during `gauntlet-02`, after
  the production control's capability run and before the two FP8 references
  and the candidate ran theirs (`frozen_before_orca_capability_results:
  true`). One key was wrong: C09 ("count positive integers ≤ N divisible by
  3 or 5") had expected values 9 and 18 for arguments 5 and 10; the correct
  values are 10 and 21. The regrade re-scores every arm from the
  already-recorded sandbox `actual` values against the corrected key.
  Nothing is regenerated or re-executed; original scores are kept beside
  corrected ones (`gauntlet-02/oracle-regraded/<arm>-capability.json`, with
  SHA-256 of every input). Corrected scores are the ones to read.

## 8. Behavior suite (52 functional cases + 9 diagnostic prompts)

Code: `campaign/campaign.py --behavior` wrapping the lab's r29 behavior probe;
fixture `fixtures/behavior/behavior-fixtures.json` (SHA-256 `3c97fcf2…`).

- Functional cases (52): 16 code tasks (8 families × 2), 24 JSON data tasks
  (intervals, ledger, reachability, report; 6 each) and 12 tool-join tasks.
  Concurrency 4; per-request client timeout 900 s.
- Diagnostic prompts (9): three long-document reasoning profiles
  (`estonia`, `hotel-lights`, `lavd-test`), three seeds each. These are
  established degeneration/long-reasoning probes from earlier lab rounds and
  are reported separately from the functional suite, as `acceptance.json`
  requires. A 900 s client timeout on one of these is recorded as
  `runtime_error: true` on that row; it is a client-side deadline, not a
  server crash.
- Two scores per case. Strict format success is the original r29 verifier
  (exact output contract). Semantic correctness is a uniform rescoring of the
  recorded responses by the existing lab scorer, vendored as
  `harness/r29/rescore_behavior.py` (policy
  `behavior-semantic-rescore-policy.json`, mirrored as
  `fixtures/behavior/semantic-rescore-policy.json`): JSON content is judged separately
  from formatting, all extracted JSON answers must agree (the scorer never
  selects using the expected answer), and code is re-run on the same hidden
  cases in the same bounded sandbox with module-level computed assignments
  allowed. The policy file is timestamped 20:57 local, after the control and
  both FP8 arms had finished their behavior runs and while the candidate's
  run (20:33–21:13) was in progress; it was applied identically to all four
  arms and no response was regenerated. Results:
  `gauntlet-02/<arm>-behavior-semantic.json`. Strict-format losses are not
  intelligence losses and are never described as such.
- Interval oracle audit (`behavior-interval-oracle-audit.json`): the six
  interval expected values were recomputed by independent unit-interval
  counting and all matched, so the remaining semantic failures on interval
  tasks are real union-length errors, not key errors.

## 9. API contract, vision, long context

- API contract (`workload_probe.py --mode api`, 12 cases): non-stream usage,
  SSE reconstruction, single and multi-turn tool calls, empty and null tool
  results, JSON-schema response format, stop sequences, three malformed
  requests, and concurrent identity isolation. Generation `temperature 0`,
  `max_tokens 1024`, `reasoning_effort low`, 300 s timeout, one attempt each.
  Several cases test server behavior rather than the model; identical
  outcomes across arms on such cases indicate a runtime property.
- Vision (`campaign/extended_probe.py --mode vision`, 12 cases): synthetic
  benign images rendered deterministically at preflight; four kinds × three
  instances (colored-shape counting, OCR, table totals, two-image
  comparison). Content-only exact JSON scoring.
- Long context, candidate arm: the lab's 32-request history suite
  (`harness/r29/history_probe.py`, `fixtures/history/manifest.json`; synthetic coding-tool
  conversations at nominal 16K/128K/512K/800K with `clear_thinking`
  false/true and visible-length controls) and 15 cold retrieval "needles"
  (`extended_probe.py --mode needles`) at nominal 131,072 / 524,288 /
  1,000,000 tokens and depths 1/10/50/90/99 %, each in a unique archive with
  the prefix cache reset and a per-case `cache_salt`. Actual `prompt_tokens`
  are recorded and are the numbers to cite; answers are scored content-only.

## 10. Performance

- Decode matrix: `harness/bench/llm_decode_bench.py` (version 0.4.24) at
  concurrency 1, 4, 8 and context 0, 32K, 128K, 45 s sustained decode per
  cell, `max_tokens 8192`, `ignore_eos`, integrated prefill scout at 8K, 32K,
  64K and 128K prompt tokens. Two full trials per arm. The benchmark's own
  JSON (`<label>-trialN.json`) is preserved unmodified.
- Steady counters: during each trial a read-only recorder samples the
  server's Prometheus endpoint every 0.5 s
  (`<label>-trialN.steady.metrics.jsonl`) and `steady_metrics.summarize`
  computes per-cell counter deltas with two-second guards inside each cell
  and a measured warm-up (`<label>.matrix-summary.json`). Step units are
  aggregate per-request verifier steps, not physical GPU launches.
- Reporting rule: every cell and both trials are retained. Both population
  (ddof 0) and sample (ddof 1) trial-to-trial CV are reported; the conservative
  flag uses sample CV above 2%. Two repeats are a variability check, not a
  long-term stability guarantee. `gpu-isolation-events.jsonl` is checked for
  foreign GPU processes; any hit invalidates the performance result.
- Collision: `campaign/mixed_traffic.py` tests `off` versus `compute_share 0.4`,
  concurrency 1/4/8, two repeats and four seeded 32K–128K cold prefills per cell.
  These runs are **completion-bounded**, not fixed-duration windows. Rates,
  decode retention and inter-token gaps are the valid cross-policy comparisons;
  total decoded-token counts span different window lengths.
- The first collision invocation in `gauntlet-02` was rejected before traffic
  because the old client sent obsolete R38 API fields. `operations-01` reruns
  all 12 cells with only the two mutable fields (`prefill_compute_share`,
  `prefill_compute_half_life`), verifies the applied values, and restores only
  configuration fields rather than posting read-only observations.

### Serving profiles per arm

| Arm | DCP | CUDA graphs | max_model_len | max_num_seqs | batched tokens | GMU | Other |
|---|---|---|---|---|---|---|---|
| production control, candidate | 4 | PIECEWISE | 1,048,576 | 16 | 4,096 | 0.90 | Marlin MoE, triton linear, FP8 KV (`fp8_ds_mla`), MTP depth 0, `reasoning_effort high` template default |
| original FP8, Orca FP8 references | 1 | FULL_AND_PIECEWISE (capture sizes 1–16) | 131,072 | 8 | 1,024 | 0.95 | same backends, `--no-async-scheduling` |

The reference arms cannot use the production profile: the 306 GiB block-FP8
checkpoints load at about 77 GiB per GPU, and in `gauntlet-01` the original
FP8 boot profiled −2.6 GiB of available KV cache even at 131,072 max length
with GMU 0.90, a 4,096-token batch budget and full+piecewise graphs. The
bounded profile above (GMU 0.95, 1,024-token budget, 8 sequences, graph
capture cap 16, async scheduling off) is the first that booted and answered
8/8 concurrent checks in `serving-diagnosis-01`. Prompts, seeds and scoring
are identical across arms; the serving configuration is not, and the docs
treat FP8 reference results as reference points under a bounded profile, not
as a matched throughput or matched-configuration comparison.

## 11. Speculation, cache and stability protocols

The speculative arms ran in `gauntlet-02`; the corrected collision, cache and
one-hour soak run in `operations-01`. A completed command is not a passing
qualification gate.

- Speculation: the candidate runs with native MTP depth 3 and with DFlash2
  proposal length 7. Each arm receives the API, 60+6 capability, two-trial
  throughput matrix and vision suites. `speculation-progress.json` retains
  both configurations and their outcomes.
- The tested MXFP8 draft is now available as
  `local-inference-lab/GLM-5.3-Flash-DFlash2` at revision
  `713226ab03bc38afdf955c7450436c2f7176f6f8`. Its weight SHA-256 and all small
  configuration/provenance files match the tested local artifact. The old
  repository name ending in `-MXFP8` is unavailable. Its CC BY-NC-ND 4.0
  license is separate from the target model; no weights are republished.
- Cache: LMCache engine-driven mode, 8 GB L1 and 8 GB fresh isolated L2,
  `max_model_len 262144`, MTP depth 3. `cache_lifecycle_probe.py` uses a 32K
  context and six growth turns; `mixed_restore_probe.py` uses a 128K
  visible-history fixture. Lifecycle, restart, pressure, external restore and
  concurrent identity isolation are observed separately. Unavailable telemetry
  never counts as a pass, and a non-fresh namespace aborts the phase.
- Soak: MTP depth 3 with GPU-local KV, 3,600 seconds split across concurrency
  1/4/8. The fixed mix contains arithmetic/string canaries, reasoning, growing
  inventory histories, cold 32K/128K prefills and cancellation/recovery.
  Runtime stability and model correctness are separate sub-verdicts
  (`runtime_stability_passed`, `model_checks_passed`); the overall gate requires
  both. The LMCache lifecycle tests and the GPU-KV soak are distinct workloads.

## 12. Verifier hardening before the gauntlets

Automated static reviews (`OrcaScoringAudit.json` and `OrcaRuntimeAudit.json`)
identified concrete risks for investigation; their confidence labels are not
measurement evidence. Five verifier defects were reproduced with CPU-only
synthetic fixtures (`verifier-edges-before.json`, all failing), corrected, and
checked again (`verifier-edges-after.json`, all passing):

1. a compliant `SAFE_GUIDANCE` answer that says "I won't provide intrusion
   instructions" was classified as a refusal;
2. a `stop` sequence firing inside the template's mandatory `<think>` block
   was scored as an API failure;
3. the soak history canary's JSON schema was derived from the expected
   answer, leaking the key set through constrained decoding;
4. a planned phase-boundary cancellation could race into `runtime_error`;
5. one needle preparation error aborted all remaining needle cases.

Two further items were fixed without a synthetic reproduction: the tool
fixture prompts now spell out the exact JSON keys, and the soak gate reports
runtime stability and model correctness as separate sub-verdicts without
weakening the overall fail gate. Deliberately retained: the sandbox rejects
out-of-contract code and the resulting failures stay task failures; model
truncation and wrong answers stay failures; no token budget was raised and no
result-driven relaxation was made (`prebehavior-review-triage.json`).

## 13. How to read the evidence

- Every serving campaign is one directory (`pilot-01`, `runtime-diagnosis-NN`,
  `fidelity-NN`, `gauntlet-NN`, `serving-diagnosis-NN`) with `battery.log`
  (timeline, local time `Europe/Berlin`), `gates.jsonl`, `phase-plan.json`,
  `phase-progress.json`, and either `qualification-executed.json` or
  `qualification-interrupted.json`.
- Suite outputs are `<arm>-<suite>/summary.json` plus per-case records.
  Summary timestamps are UTC.
- Corrections never overwrite: regraded and rescored files sit beside the
  originals and carry the SHA-256 of what they were computed from.
- Failed and aborted attempts are kept with the same completeness as
  successful ones and are classified in `findings.md`, in
  `gauntlet-01-classification.json` and in `runtime-findings.json` (whose
  progress text predates `fidelity-02` and `gauntlet-02`; use the newer
  receipts for status).

# Findings

Results for `orcarouter/GLM-5.3-Flash-Uncensored-NVFP4`, with the failed attempts
and changes needed to obtain trustworthy measurements. The overview is in
`README.md`; machine-readable tables are in `results/`. Every numerical claim
below points to a receipt or preserved run. Evidence paths are relative to
`orcarouter-qualification-20260919T104457048Z/` and resolve through
`evidence/index.json`.

## 1. Where the campaign stands

Complete, with evidence:

- Integrity, provenance and lineage of all three downloaded checkpoints
  (Section 3).
- Runtime diagnosis of the candidate as published, two isolated loader
  repairs, their numeric proofs and an independent review (Section 4).
- Full-vocabulary fidelity for the three directed comparisons on 24,564
  primary positions (`fidelity-02`, Section 5).
- In `gauntlet-02`: all four base quality arms, candidate history and retrieval,
  the two no-speculation matrices, and both MTP-3/DFlash2 speculative arms.
- In `operations-01`: all 12 repaired mixed-load cells, the cache lifecycle
  battery, the 29-request mixed-restore/isolation probe, and the full one-hour
  MTP-3 soak.

**Not cleared for production.** The soak recorded 19 request-level HTTP 500s
and 55 non-runtime model-answer failures. The required zero-runtime-error
criterion is not met; required cache observability is also incomplete.

`runtime-findings.json` still says fidelity-02 is "running" and that
production is paused by it; it was written before `fidelity-02` finished and
before either gauntlet. The receipts cited below supersede that text.

## 2. Chronology of attempts, including failures

All times are local (`Europe/Berlin`, UTC+2) from the `battery.log` files.
Every GPU campaign restored the original container and verified a useful
response (`*/production-restored.json`, `*/production-restored-inference.json`).
The final restoration completed at 01:19:22 on September 20. No candidate was
promoted.

| Time | Attempt | What happened | Classification |
|---|---|---|---|
| 12:44 | `preflight.json` | Both Orca repositories gated; an interactive approval prompt had timed out and auto-selected "approved", which was recorded as *not* authorization. No GPU work. | Process guard |
| 13:06 | Authorized model access | Required access approval was obtained before downloads. The private authorization receipt is excluded from publication. | Access prerequisite, not a model result |
| 13:15–13:18 | `*.hf-receipt.json`, `staging-verification.json` | Remote checksum verification and tensor contracts passed for all three snapshots. | Pass |
| 13:27 | `fidelity-panels-prepared.json.failure.json` | Panel preparation refused: tokenizer length 154,856 vs model output 154,880. Fixed by making the padded output vocabulary an explicit contract; re-prepared. | Harness defect, fixed before GPU use |
| 13:37–13:56 | `pilot-01` | Candidate as published (W4A4, unmodified R38 image) booted; the first API request produced 1,024 reasoning tokens of the word `lock`, an empty visible answer and `finish_reason=length`. Campaign interrupted by SIGINT instead of spending a soak on known degeneration. | Runtime/model failure; not a score |
| 13:57–14:11 | `runtime-diagnosis-01` | Candidate on Marlin W4A16: 3/3 bounded semantic checks. Both FP8 references failed to boot: `B12xFp8BlockScaledMMKernel Output features must be a positive multiple of 128`. | Candidate basic function only; FP8 boot defect |
| 14:31–14:49 | `runtime-diagnosis-02` | Repair v1 image. Kernel probe passed (Section 4). Candidate 3/3 again. FP8 references still failed to boot with `LINEAR_BACKEND=auto`: `cutlass_scaled_mm … Invalid status` during the profiling run in a KDA `in_proj_qkvgfab` FP8 linear. | Proof of scale fix; FP8 defect persists |
| 14:52–15:16 | `runtime-diagnosis-03` | FP8 references forced onto the triton linear backend: booted, but all six checks returned no content at `finish_reason=length`. | FP8 arms unusable (misclassified BF16 tensors) |
| 15:18–15:43 | `runtime-diagnosis-04` | Same with the triton MoE backend: identical 0/3 and 0/3. | Confirms the defect is not backend-specific |
| 15:52–16:10 | `runtime-diagnosis-05` | Repair v2 image (FP8 suffix exclusions): both FP8 references 3/3. | FP8 arms usable |
| 16:11–16:22 | `fidelity-01` | All three offline captures exited 1: `GLM C4 indexing requires a model block size divisible by 256`. All three comparisons therefore failed. No quality data. | Harness configuration defect |
| 16:23–17:51 | `fidelity-02` | Captures with `block_size 256`, interleave 4 and aligned Mamba cache: three captures and three comparisons passed. | Fidelity evidence |
| 17:55–18:10 | `gauntlet-01` | Production control under `FULL_AND_PIECEWISE` graphs hit a CUDA illegal memory access during full-graph replay after one of eight initial concurrent requests finished: 65 runtime errors, 1 success. Original FP8 (DCP 1, 131,072 max length, GMU 0.90, 4,096-token batch budget, full+piecewise graphs) did not boot: available KV cache −2.6 GiB. Interrupted by SIGINT while the ablated reference was booting. | Runtime failures; explicitly not model scores (`gauntlet-01-classification.json`) |
| 18:14–18:30 | `serving-diagnosis-01` | `CUDA_LAUNCH_BLOCKING=1`: crash reproduced (1/8). `--no-async-scheduling`: reproduced (1/8). Original FP8 bounded profile: 8/8, healthy afterwards. | Configuration diagnostics |
| 18:30–18:47 | `serving-diagnosis-02` | Control with DCP 1: reproduced (1/8). Control with `PIECEWISE` graphs: 8/8. Candidate with `PIECEWISE`: 8/8. | Working profile identified |
| 18:49–23:04 | `gauntlet-02` | Base quality/context/matrix arms and MTP-3/DFlash2 completed. Case-level failures were retained. The old mixed-load client was rejected by the changed R38 fairness API before traffic started. Original production was restored. | Measurement evidence plus a client-contract failure |
| 23:35–01:19 (September 20) | `operations-01` | All 12 repaired mixed-load cells completed. Cache: 35 pass, 0 fail, 3 unavailable, 1 not applicable; mixed restore: 29/29. Full-hour soak: 19 HTTP 500s, 55 wrong answers. Original production restored and verified. | Qualification failed; complete evidence retained |

The interrupted and failed attempts (`pilot-01`, `runtime-diagnosis-01` to
`-04`, `fidelity-01`, `gauntlet-01`) were not discarded; they are the reason
the later profiles look the way they do, and every one of them carries its
docker logs, launch files and GPU snapshots.

## 3. Integrity, provenance and lineage

- The three downloaded snapshots verified against the Hub at the pinned
  revisions with `hf cache verify --fail-on-missing-files` (74, 72 and 73
  remote files; `*.hf-receipt.json`). `weights_modified: false` in
  `staging-verification.json`; no checkpoint file was changed at any point.
- Candidate tensor contract: 111,346 tensors in 65 shards, 205,079,179,568
  indexed bytes equal to file bytes, 889 MTP tensors, 347 vision tensors,
  36,579 FP32 auxiliary tensors, declared quantization `compressed-tensors`
  as expected. Both FP8 snapshots: 76,108 tensors in 62 shards, 1,760 MTP and
  347 vision tensors, identical `config.json` and index hashes, 0
  name/dtype/shape differences.
- 8,291 of the candidate's expert gate/up pairs have different
  `weight_global_scale` values (for example layer 10 expert 0: gate 21504,
  up 26496). Both FP8 checkpoints have none, as expected for a format without
  such scales. This is a checkpoint property; what the runtime did with it is
  Section 4.
- Lineage: 51,139 tensors (219,196,705,656 bytes per checkpoint) outside the
  declared residual-writer families are byte-identical between the original
  and the abliterated FP8 checkpoints, with 0 mismatches
  (`fp8-lineage.json`, 405 s). The 24,969 excluded tensors are the
  `down_proj`, `o_proj`, `eh_proj` and `embed_tokens` families that the
  publisher's card says were edited. This supports "derived from the pinned
  original"; it does not verify what was written into the excluded tensors.

## 4. Runtime: what the published checkpoint does, and what was repaired

### 4.1 Three configurations that must not be conflated

1. **Default publisher checkpoint on the unmodified runtime.** The
   compressed-tensors config declares dynamic W4A4 activations. R38's default
   MoE path took that recipe and produced degenerate output on the first
   request (`pilot-01`: `lock` × 1,024, no visible answer). This is the only
   observation of the candidate on an unmodified image with its declared
   recipe.
2. **Corrected runtime, Marlin W4A16.** Every candidate result in Sections
   5–10 was produced on repair image `sha256:913478ce…` with
   `MOE_BACKEND=marlin`: 4-bit weights, 16-bit activations, per-expert up-half
   scale correction. This is not the recipe the checkpoint declares and it is
   not what someone pulling the checkpoint into a stock runtime gets.
3. **Production configuration.** The live container `glm53-prod` serves the
   NVIDIA ModelOpt FP4 checkpoint on the unmodified R38 image. In
   `gauntlet-02` the production control ran that checkpoint under the
   campaign's production-matched serving profile but on the repair image (the
   patch does not touch the `modelopt_fp4` path; see `methods.md` Section 3)
   and with `PIECEWISE` CUDA graphs because of the full-graph crash
   (Section 11).

Nothing in this campaign qualifies W4A4 serving of the candidate, and no
other model family or checkpoint was tested.

### 4.2 Established defects (`runtime-findings.json`, `OrcaRuntimeAudit.json`)

1. The candidate ships no activation `input_global_scale` tensors. The
   upstream/R38 W4A4 loader creates them with `torch.empty` and consumes their
   reciprocals without a missing-weight default or validation.
2. For the 8,291 gate/up pairs with distinct weight global scales, the loader
   warns, discards the up-projection global scale and applies the gate scale
   to both halves. The recorded example (gate 21504, up 26496) changes the
   represented up amplitude by 23.214% before clipping.
3. The FP8 loader matched `modules_to_not_convert` by exact native name,
   ignoring the multimodal `language_model.` namespace. 400 BF16 matrices
   (`lm_head`, `embed_tokens`, `eh_proj`, `mlp.gate`, the sparse-attention
   indexer projections, `kv_b_proj`, the `self_attn.b_proj`/`f_a_proj`/
   `f_b_proj`/`g_a_proj` projections, …) were treated as FP8. This is why
   both FP8 references could not boot on the block-scaled kernel and produced
   only empty length-truncated output when forced onto another linear kernel
   (`runtime-diagnosis-01` … `-04`).

Whether the `lock` loop of configuration 1 is caused entirely by defects 1–2
is not established. The Marlin result and the static trace implicate the
activation/W4A4 path, but the campaign did not isolate the exact cause or
quantify weight-side versus runtime-side degradation (`causality_limit`).

### 4.3 Repairs and their proofs

Repair v1 (`runtime-repair/compressed_tensors_moe_w4a4_nvfp4.py`):
initialize missing activation globals as invalid and fail closed on
unsupported W4A4; on the Marlin path skip the unused activation globals and
multiply the up half by G_gate/G_up per expert before the gated activation
and the `swiglu_limit` clip. Repair v2 adds `runtime-repair/fp8.py`: HF
`modules_to_not_convert` entries match by suffix; explicit `ignored_layers`
keep exact matching. No packed weights or checkpoint files change.

| Proof | Result | Receipt |
|---|---|---|
| Uncorrected collapsed-scale kernel vs independent reference (negative control) | relative RMSE 0.1675, max abs error 3.94 | `runtime-diagnosis-02/scale-kernel.json` |
| Corrected per-expert up scaling | relative RMSE 0.00477, max abs error 0.125 | same |
| CUDA-graph replay with changed inputs | relative RMSE 0.00470 | same |
| FP8 exclusion regression over 37,862 matrices | 400 misclassified before, 0 after; explicit exact matching preserved | `fp8-exclusion-regression.json` |
| Bounded semantic checks, candidate Marlin | 3/3 on the unmodified image and on repair v1 | `runtime-diagnosis-01/02/runtime-diagnosis.json` |
| Bounded semantic checks, FP8 references | boot failure (01, 02), 0/3 (03, 04), 3/3 on repair v2 (05) | `runtime-diagnosis-0N/runtime-diagnosis.json` |

An automated static review (`OrcaRuntimeAudit.json`) found no material error
in the repaired path used here. It also recorded three out-of-scope risks:
interleaved gate/up activation layouts would be rescaled on the wrong columns; weight
global scales are still `torch.empty`-initialized so a checkpoint missing
them would not fail closed; suffix mode drops the exact-mode per-expert
exclusion branch. Residual limitations that do apply here: the correction
adds one extra BF16 rounding per up-projection element (bounded by the
0.48% RMSE above) and that rounding is part of every measured candidate
number; the kernel probe exercised `fused_marlin_moe` at E=3, N=128 and did
not numerically exercise the loader wiring, expert maps or padding — those
are evidenced only by the boot log line and the short semantic checks; the
FP8 regression is per tensor with an empty fused mapping. The reviewers
also state the verdict's real scope: the image is experimental and scoped to
GLM-5.3-Flash on this TP-4 rig, not a general-purpose vLLM repair
(`prebehavior-review-triage.json`).

## 5. Fidelity (`fidelity-02`)

All three comparisons: same 24,564 primary WikiText-2 positions, same token
IDs, full 154,880-row distributions, identical runtime (TP 4, eager, FP8 KV,
Marlin MoE, B12X attention) with only the quantization argument differing.
Capture validation passed in each arm (`status: pass`; maximum
|logsumexp| after float64 renormalization 4.3e-6 against a 1e-3 tolerance).
KL is KL(reference ‖ candidate) in nats per token.

| Comparison | KL mean | KL p50 | KL p95 | KL p99 | Reverse KL mean | JS mean | Top-1 agreement | PPL ref → cand | PPL ratio | Per-window KL mean (min–max) |
|---|---|---|---|---|---|---|---|---|---|---|
| original FP8 → Orca FP8 | 0.0299 | 0.0045 | 0.124 | 0.412 | 0.0304 | 0.0069 | 94.82% | 2.7675 → 2.7834 | 1.0057 | 0.0154–0.0490 |
| Orca FP8 → Orca NVFP4 | 0.0792 | 0.0131 | 0.356 | 1.040 | 0.0849 | 0.0174 | 91.60% | 2.7834 → 2.8748 | 1.0329 | 0.0480–0.1128 |
| original FP8 → Orca NVFP4 | 0.0814 | 0.0136 | 0.375 | 1.045 | 0.0886 | 0.0181 | 91.32% | 2.7675 → 2.8748 | 1.0388 | 0.0496–0.1163 |

Sources: `fidelity-02/kld-<ref>-to-<cand>/summary.json` (`primary_wikitext`,
`uncertainty.window_mean_variation`), condensed in
`fidelity-result-brief.json`. Per-position arrays:
`fidelity-02/kld-*/per-position-metrics.npz`.

Observations that the numbers support:

- The ablation at source precision moves the distribution far less than the
  quantization does: mean KL 0.030 vs 0.079, top-1 agreement 94.8% vs 91.6%,
  perplexity +0.6% vs +3.3%.
- The combined effect is close to additive in mean KL (0.0814 vs
  0.0299 + 0.0792) and the perplexity ratios compose (1.0057 × 1.0329 ≈
  1.0388).
- Tails are heavy: p99 above 1 nat for both quantization comparisons, maxima
  of 5–7.4 nats at single positions. Mean KL alone understates this.
- Per-window means vary by more than 2× (0.050–0.116 for the combined
  comparison), which is why no confidence interval is claimed.
- On the four campaign-authored supplemental panels (8,188 positions) the
  divergences are smaller (combined KL 0.0226, top-1 95.2%) than on
  WikiText; these strata are descriptive only.
- Gold-token top-1 accuracy on the primary panel drops from 75.00%
  (original FP8) to 73.93% (candidate).

What the numbers do not support: any statement about task capability. The
publisher's card gives KLD 0.073, p95/p99 0.330/0.973, top-1 91.7% and PPL
2.788 → 2.894 for NVFP4 vs its FP8 reference on 24,564 wikitext tokens. Our
Orca FP8 → NVFP4 row lands in the same region (0.079, 0.356/1.040, 91.6%,
2.783 → 2.875), but the token IDs, window selection, reference execution
(weight-only forward vs a served runtime with FP8 KV and the corrected Marlin
path) and vocabulary handling differ, so this is a consistency observation,
not a replication and not a validation of the published scalar.

## 6. Capability (`gauntlet-02`, 60 base + 6 supplementary tasks)

Every arm completed 66/66 tasks with usage available on all of them, 0
runtime errors, 0 truncations and 0 protocol issues. All failures are
`knowledge_or_task_failure` with `finish_reason=stop`. Scores below are the
uniform C09-corrected regrade (`gauntlet-02/oracle-regraded/<arm>-capability.json`);
the original scores are one lower for every arm because each arm had
answered C09 correctly against the wrong key.

| Arm (profile) | Base 60, corrected | Failed tasks | Supplementary X01–X06 |
|---|---|---|---|
| original FP8 (bounded) | 60/60 | — | 6/6 |
| production control, NVIDIA NVFP4 (production profile) | 58/60 | B02, C10 | 6/6 |
| Orca FP8 (bounded) | 57/60 | B09, B11, B12 | 6/6 |
| Orca NVFP4 candidate (production profile) | 55/60 | A09, A10, B03, B11, C10 | 6/6 |

Nature of the failures (`gauntlet-02/<arm>-capability/records/<id>.json`):

- Date-interval off-by-one errors dominate: B02 (control) 3987 vs 3988;
  B03 (candidate) 7568 vs 7569; B11 (candidate and Orca FP8) 4590 vs 4591;
  B09 (Orca FP8) 266 vs 267; B12 (Orca FP8) 14108 vs 14109.
- Candidate arithmetic: A09 answered 7290 for (10^9 + 7) mod 9973 (correct
  7297; the reasoning computed 10^9 mod 9973 and dropped the +7); A10
  answered 699 for an expected 731.
- C10 (shortest palindromic completion by prepending) was failed by both
  NVFP4 arms with the same wrong algorithm and identical hidden-case outputs
  (`cbaabcd`, `cecaaaacecaaa`), and passed by both FP8 arms.

Paired comparison against the production control (identical prompts, seeds,
scoring and serving profile): 5 discordant tasks, 4 favoring the control
(A09, A10, B03, B11) and 1 favoring the candidate (B02); C10 is a concordant
failure. The candidate is 3 tasks, exactly 5.0 percentage points, below the
control. `acceptance.json` flags a loss *greater than* five points as an
operational regression; this result sits on that line, and with 5 discordant
pairs the comparison has no statistical power in either direction. It is not
evidence of equivalence, and it is not evidence of a regression beyond the
stated flag either. Against the FP8 references the candidate is 5 tasks
(8.3 points) below the original and 2 tasks (3.3 points) below the
abliterated reference, but those arms ran the bounded serving profile
(`methods.md` Section 10), so profile and checkpoint effects are confounded.

The integer verifier accepts an answer appearing in the last three lines,
so lenient matches are possible in principle; every recorded `candidates`
list is in the record for audit.

## 7. API contract, vision

API (`gauntlet-02/<arm>-api/summary.json`, 12 cases, one attempt each):

| Arm | Passed | Failed cases |
|---|---|---|
| production control | 10/12 | `null-tool-result`, `malformed-invalid-role` |
| original FP8 | 10/12 | same two |
| Orca FP8 | 9/12 | same two plus `empty-tool-result` |
| Orca NVFP4 candidate | 9/12 | same two plus `empty-tool-result` |

The two shared failures are server behavior, identical across all four arms
on the same image: a `null` tool-message content is rejected with HTTP 400
(pydantic validation errors) rather than being handled, and an invalid
message role is accepted with HTTP 200 instead of a structured 4xx. The
third failure is a strict output contract: the fixture expects exactly
`EMPTY_RESULT_HANDLED`; the control answered exactly that, while the
candidate and the Orca FP8 arm prefixed an explanatory sentence ("The
optional note contained no data … EMPTY_RESULT_HANDLED"). Content-wise the
marker was present; the strict verifier fails it, and the record shows why.
All other contracts, including SSE reconstruction, single and multi-turn
tool calls, JSON-schema output, stop sequences and concurrent identity
isolation, passed on every arm.

Vision (`gauntlet-02/<arm>-vision/summary.json`, 12 synthetic cases, 0
runtime errors on every arm): candidate 12/12; production control 11/12
(`two-images-1`: second count 4 vs 5); original FP8 9/12 (`table-0`,
`table-1` totals off by 2 and 4, `two-images-1`); Orca FP8 8/12 (all three
table totals, `two-images-1`). Twelve cases cannot rank models; the table
family is where the FP8 arms — running the bounded profile — differ from the
NVFP4 arms, and the evidence does not say whether the checkpoint or the
profile is responsible.

## 8. Behavior: strict format versus semantic correctness

52 functional cases per arm, all completed, 0 runtime errors, 0 budget
exhaustion in the functional rows (`gauntlet-02/<arm>-behavior/summary.json`,
`gauntlet-02/<arm>-behavior-semantic.json`):

| Arm | Strict format success | Semantic correctness | Remaining semantic failure |
|---|---|---|---|
| production control | 43/52 | 51/52 | `data-intervals-5` |
| original FP8 | 47/52 | 51/52 | `data-intervals-4` |
| Orca FP8 | 43/52 | 51/52 | `data-intervals-2` |
| Orca NVFP4 candidate | 49/52 | 51/52 | `data-intervals-5` |

All four models are semantically correct on 51 of 52 cases. The strict
scores differ because of output formatting (`json-content-separate-from-format`
dispositions: extra prose around JSON, ledger/report layout), not because the
content differs. The one semantic failure per arm is a genuine
union-length error on an interval task; the six interval keys were
independently recomputed and all matched (`behavior-interval-oracle-audit.json`).
A strict-format gap is not an intelligence gap, and the candidate's higher
strict score is not evidence of higher capability either.

Nine diagnostic prompts, reported separately as required (three profiles ×
three seeds, 900 s client deadline):

| Arm | Answered correctly | Client timeout at 900 s | Other |
|---|---|---|---|
| production control | 3 (`estonia` ×3) | 5 (`hotel-lights` ×2, `lavd-test` ×3) | 1 wrong (`hotel-lights` seed 01) |
| Orca NVFP4 candidate | 4 (`estonia` ×3, `lavd-test` seed 02) | 5 | — |
| Orca FP8 | 6 (`hotel-lights` ×3, `lavd-test` ×3) | 3 (`estonia` ×3) | — |
| original FP8 | 3 (`lavd-test` ×3) | 3 (`estonia` ×3) | 2 budget-exhausted, 1 wrong (`hotel-lights`) |

These rows are `runtime_error: true` in the raw summaries because the client
gave up, not because the server failed; the servers stayed healthy and
completed the following suites. The two NVFP4 arms and the two FP8 arms time
out on different profiles, and the FP8 arms ran a different serving profile,
so no model-level conclusion is drawn from this table.

## 9. Long context (candidate only so far)

- History (`gauntlet-02/candidate-history/summary.json`): 32/32 semantic
  passes, 0 runtime errors, 0 budget exhaustion. Actual prompt tokens by
  nominal size: 16,741–16,765; 131,428–131,453; 524,645–524,669;
  819,559–819,581 (the `clear_thinking=true` reasoning-history arms collapse
  to 365 prompt tokens by design and are the controls).
- Needles (`gauntlet-02/candidate-needles/summary.json`): 15/15 at actual
  prompt tokens 130,841–130,888, 523,949–523,999 and 999,652–999,691, at
  depths 1/10/50/90/99 %, 0 runtime errors, no truncation.

These suites ran only on the candidate arm in this gauntlet; there is no
paired control for them yet, so they establish that the candidate served
these lengths and retrieved these facts on this profile, nothing comparative.

## 10. Sustained performance and speculation (`gauntlet-02`)

The first table uses MTP depth 0, `PIECEWISE` graphs and DCP 4.

Sustained decode, aggregate output tokens/s, two 45 s trials per cell, 0
request errors in every cell, no cell capacity-limited
(`gauntlet-02/<arm>-mtp0-trialN.json`, `.matrix-summary.json`):

| Cell | Control trial 1 / 2 | Candidate trial 1 / 2 |
|---|---|---|
| C1, 0 | 16.54 / 16.43 | 16.33 / 16.07 |
| C1, 32K | 16.44 / 16.24 | 16.43 / 16.55 |
| C1, 128K | 16.47 / 16.39 | 16.51 / 16.48 |
| C4, 0 | 64.39 / 64.02 | 64.36 / 64.32 |
| C4, 32K | 64.05 / 64.52 | 64.40 / 62.39 |
| C4, 128K | 64.30 / 64.35 | 64.42 / 64.34 |
| C8, 0 | 127.87 / 128.99 | 127.92 / 128.42 |
| C8, 32K | 127.77 / 126.91 | 129.28 / 127.60 |
| C8, 128K | 128.69 / 128.63 | 126.79 / 127.65 |

For these no-speculation arms, population CV is below 2% in all cells; sample
CV flags candidate C4/32K at 2.24%. The report includes both conventions and
uses sample CV for its conservative flag. Prefill scouts measured roughly
10K–11.3K tokens/s; 128,85x-token prompts took 11.63–11.76 seconds to first
token. Steady Prometheus emissions were observed for every cell. The complete
GPU-isolation log is retained rather than quoting a mid-run sample count.

Reading: in this range aggregate throughput scales almost exactly with
concurrency for both arms (about 16 tok/s per stream at C1 through C8), and
the two arms' two-trial cell means differ by −1.7% to +0.9% (largest gap
C1/0: 16.48 vs 16.20), the same order as the trial-to-trial variation. The
near-linear scaling suggests per-step latency rather than compute saturation
bounds these cells, but that is an interpretation of the shape, not a
measured statement.

The speculative matrices also completed, with no request errors in their
sustained windows:

| Candidate profile, nominal 0 context | C1 mean tok/s | C4 aggregate tok/s | C8 aggregate tok/s |
|---|---:|---:|---:|
| MTP-3 | 31.59 | 125.71 | 223.77 |
| DFlash2-7 | 33.33 | 114.25 | 227.47 |

Their audited capability scores were 56/60 and 55/60 respectively, with all
six supplementary cases passing. These are not production-profile speeds.
Four cells across the full matrices exceed 2% sample CV; two exceed 2% under
either convention. Every sample remains in `results/throughput.csv` and the
raw trial JSON; no reruns were selected to obtain a cleaner number.
At C1 / nominal zero context, MTP-3 accepted 49.22% and 49.44% of proposed
draft tokens across the two trials, emitting 2.476 and 2.483 tokens per
verifier step. DFlash2-7 accepted 24.66% and 24.80%, emitting 2.726 and 2.736.
These are aggregate **per-request verifier steps**, not physical batched GPU
kernel launches; accepted/proposed draft fractions must not be compared
without the draft length. All 36 guarded steady-counter records are in
[`results/speculation.json`](../results/speculation.json).

The benchmark also recorded 32 single-request prefill scouting observations.
For the no-speculation 128K scouts, the candidate's actual 128,851-token
prompts took 11.758 and 11.747 seconds to first token; the control's
128,852/128,850-token prompts took 11.630 and 11.643 seconds. These client-side
measurements include the request path, are not isolated GPU kernel times,
and were **not a standalone sustained prefill sweep**. Empty server-validation
counters are unavailable, not zero-throughput observations.
[`results/prefill.csv`](../results/prefill.csv) retains every scout.

### Mixed-load tradeoff

`operations-01/candidate-collision/candidate-collision.json` contains 12 cells,
two repeats of each policy/concurrency pair. All offered prefills completed.
With compute share 0.4, decode retention was 67.4%, 66.7%, and 68.2% at C1/C4/C8,
compared with 12.5%, 13.2%, and 12.9% with fairness off. Prefill throughput fell
from approximately 7.9K to 4.6K tokens/s. Decode p99 gaps improved from roughly
1.08–1.12 seconds to 0.42–0.43 seconds.

These are completion-bounded windows, not fixed-duration windows. The
original generated SVG's fixed-window caption is inaccurate for this mode;
raw token totals must not be treated as an equal-duration comparison.
The rates, retention, latency data and unmodified raw SVG are all retained.
The corrected [rate-based chart](../results/collision-rates.svg) is regenerated
from the same saved measurements; no GPU run was repeated to make this plot.

## 11. Serving stability

- Full CUDA graphs crashed the production NVIDIA checkpoint on this runtime:
  `torch.AcceleratorError: CUDA error: an illegal memory access was
  encountered` during `CUDAGraph.replay` after the first of eight concurrent
  capability requests finished (`gauntlet-01/control-mtp0-trial1.docker.log`,
  `gauntlet-01-classification.json`). Synchronous launches, disabling async
  scheduling and DCP 1 all reproduced it; `CUDAGRAPH_MODE=PIECEWISE`
  eliminated it for both the control and the candidate in short 8-request
  diagnostics and throughout `gauntlet-02` (`serving-diagnosis-01/02/serving-diagnosis.json`).
  The gauntlets therefore run piecewise graphs. Whether the live production
  container uses full graphs is not recorded in this evidence set; the
  finding is about the R38 runtime under this profile, not the candidate.
- The block-FP8 references need a bounded profile: 77.1 GiB of weights per
  GPU left −2.6 GiB of KV headroom at GMU 0.90 with a 4,096-token batch
  budget and full+piecewise graphs, even at 131,072 max length and DCP 1
  (`gauntlet-01/original-fp8-reference.docker.log`). GMU
  0.95, a 1,024-token batch budget, 8 sequences and graph capture up to 16
  booted and served 8/8; capability and behavior suites then completed
  without runtime errors on both FP8 arms.
- No candidate process crash was observed in `gauntlet-02`. Its failures
  include scored task/API checks and the obsolete mixed-load client contract,
  not just model-answer failures.
- Cache lifecycle recorded 35 passing gates, no failing gates, three
  unavailable observations and one not-applicable gate. The overall receipt
  is not a pass: the required server-abort observation was unavailable.
  A retrieve lease was too short-lived to observe, and all-rank KV/SHM byte
  equality was not exposed by this serving interface. Matching answers and
  token IDs are not claimed to prove every cached byte equal.
  (`operations-01/candidate-lmcache-mtp3/receipt.json`.)
- The separate mixed-restore/isolation workload passed all 29 requests
  (`operations-01/candidate-lmcache-mtp3-mixed/summary.json`).
- The MTP-3 soak ran for 3,600.033 seconds across concurrency 1/4/8,
  with 1,289 logical attempt records. Thirteen were planned phase-boundary
  cancellations. Of the other 1,276, 1,202 passed, 19 returned HTTP 500 and
  55 returned wrong answers: 53 reversals and 2 recurrence calculations.
  The raw `deterministic_failures=74` includes the 19 runtime failures;
  it must not be read as 74 separate model errors.
- All 19 runtime errors were schema-constrained arithmetic requests: zero at
  concurrency 1, seven at concurrency 4 and twelve at concurrency 8.
  The server logged `Failed to advance FSM` and `grammar rejected tokens`,
  terminating individual requests without crashing the engine. No matched
  NVIDIA soak was run, so this is not isolated causally to abliteration or
  quantization.
- The run covered growing multi-turn histories, 32K/128K cold-prefill
  workloads, reasoning, correctness canaries and 116 cancellation/recovery
  attempts. No repetition-heuristic candidate was flagged. This bounded
  observation is not proof of indefinite freedom from looping.
  (`operations-01/candidate-soak/summary.json`,
  `operations-01/candidate-soak-mtp3.docker.log`,
  `operations-01/first-soak-grammar-error.log`.)

## 12. Verifier and oracle corrections applied

- Answer key C09 was wrong for two of its three hidden cases (expected 9 and
  18; correct 10 and 21). Every base and speculative arm produced the correct
  values but was marked failed by that key. Regrading used the recorded
  sandbox outputs uniformly: no response regeneration and no code re-execution.
  Net effect: +1 task on all six arms (`oracle-corrections.json`,
  `gauntlet-02/oracle-regraded/*.json`). Two earlier corrections are documented
  in `code-oracle-audit.json`: C01's non-assertion was replaced by an exact
  modular oracle, and C05's out-of-contract input was replaced.
  All other 59 code and non-code keys matched independent recomputation.
- Behavior semantic rescoring (`behavior-semantic-rescore-policy.json`)
  separates content from format for all arms uniformly; v2 of the policy only
  restated the accepted assignment forms more precisely and switched to the
  bounded-output scorer. Original registrations and v1 receipts are kept.
- Five verifier defects were corrected before model evaluation, including
  false refusal/stop failures, oracle information leaked through a history
  grammar, timeout-boundary misclassification and premature needle-suite exit.
  The synthetic before/after reproductions are preserved; see Methods Section 12.

## 13. What the completed evidence does and does not establish

Established, within the stated scope:

- The checkpoints are what they claim to be at the pinned revisions, and the
  abliterated FP8 checkpoint is byte-identical to the original outside the
  declared writer families.
- The candidate as published does not run correctly on this runtime's
  declared W4A4 path; it runs on the corrected Marlin W4A16 path after two
  loader repairs whose numeric proofs and regression are recorded.
- On this corpus and execution configuration, the quantization comparison's
  mean KL is about 2.6 times the abliteration comparison's mean KL
  (0.079 versus 0.030). This is a distribution metric, not a capability-loss ratio.
- On 60 deterministic tasks the candidate answered 55 correctly against 58
  for the production model, with 5 discordant pairs; both NVFP4 models
  answered 51/52 behavior cases correctly at the semantic level, as did both
  FP8 references; the candidate answered all 12 vision, 32 history and 15
  long-context needle cases.
- No-speculation throughput is close to the matched control. Speculation
  roughly doubles short-context C1 throughput on this conservative profile,
  but some timing cells exceed the variability flag.
- Mixed-load fairness improves decode service at a measurable prefill cost.
  Cache restore checks passed where observable; unavailable gates were not passed.
  Concurrent structured-output errors prevent a clean stability verdict.

Not established:

- Equivalence to the production model or to the original. The suites are
  small (60, 52, 12, 12 cases); the 5-point gap on 60 tasks is three tasks,
  the four candidate-only failures are two arithmetic and two date-interval
  tasks, and with five discordant pairs the power to separate this from
  noise is absent in either direction.
- Unrestricted long-running reliability: the soak has recorded request-level
  failures, and some cache invariants are not observable through this interface.
- Anything about W4A4 serving, about the checkpoint on a stock vLLM, or
  about other model families.
- Any refusal or harmful-compliance property. Only two benign over-refusal
  prompts were run (both answered on every arm); the publisher's refusal
  numbers were not reproduced and are not endorsed.
- Production readiness. No promotion was authorized and the original
  container was restored after every campaign.

The report uses the full frozen soak and final restoration receipts. The
original container was healthy, advertised the expected model and answered
the restoration canary correctly. There was no automatic promotion.

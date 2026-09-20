# Orca at full speed: serving recipe, stats, and refusal analysis

Date: 2026-09-20. Hardware: 4x RTX PRO 6000 Blackwell 96 GB (Workstation
Edition, 600 W limit, existing +6000 MHz memory clock offset, stock graphics
clocks). Image: Karmic Kraken beta
`ghcr.io/local-inference-lab/vllm@sha256:55e477ad62ae15a77c9b869e8fb8e2d958f6edcc4d95e306adfa89dee9ed19df`
plus the local derivative `glm53-orca-kraken-reconciled:local`
(`campaign/kraken-orca/Dockerfile.kraken-orca`) carrying three load-time
patches. Machine-readable evidence for every table below is committed in
`results/orca-fastpath-20260920/` (speed trials, sentinels, refusal records
and summaries per arm, capability/behavior summaries, scale audit, boot
commands); the full raw run trees remain on the measurement host under
`drock-lmcache/orca-kraken-20260920/run-05..run-10`.
All GPU phases ran under the r26 guarded coordinator (production paused,
restored and re-verified after every run).

## 1. Why stock vLLM cannot serve this checkpoint correctly

`orcarouter/GLM-5.3-Flash-Uncensored-NVFP4` (revision `ec0adf4f`) is
compressed-tensors `nvfp4-pack-quantized` with two properties no fused NVFP4
MoE kernel path in vLLM (R38, Karmic Kraken beta, or upstream main at
analysis time) handles:

1. **Independent gate/up global divisors.** 8,291 of 12,096 expert pairs carry
   different `weight_global_scale` values for gate and up (ratios 0.278-9.96).
   Stock loaders keep only the gate divisor for the fused w13 and merely warn,
   mis-dequantizing the up half by up to 9.96x. Public twins: vLLM #54150
   (ModelOpt sibling class, numeric proof), HF discussion #2 on this exact
   checkpoint. This is the mechanism behind "boots but outputs garbage".
2. **No input global scales.** `input_activations.dynamic = true`, so no
   `input_global_scale` tensors exist; the method registers them as
   `torch.empty` and divides by them. Public twin: vLLM #54189 (observed 0.0
   garbage silently zeroes every expert on the fused path).

Additional launch traps on the Kraken profile: it defaults
`--quantization modelopt_mixed` (ValidationError against compressed-tensors),
and its MTP draft `moe_backend: marlin` suits only NVIDIA's NVFP4 draft (Orca's
MTP draft MoE is BF16 -> marlin rejected). NVIDIA MTP3 on this image also still
dies in the known layer-45 MTP loader mismatch (packed NVFP4 w2 256 vs BF16
tensor 512), unchanged from R31.

## 2. The recipe that works

Local derivative image = Kraken beta + `orca_scale_reconcile.py` + patched
`compressed_tensors_moe_w4a4_nvfp4.py` + patched `fp8.py` (Dockerfile in
`campaign/kraken-orca/`). Launch (profile `glm53-flash`, TP4, port as needed):

- env `MODEL=<orca checkpoint path>`
- env `ORCA_NVFP4_SCALE_POLICY=reconcile_min` - unify gate/up onto the smaller
  divisor per expert by shrinking the larger-divisor half's E4M3 block scales
  (factor <= 1). Measured on the full checkpoint: 0 overflow refusals, 0
  nonzero-to-zero roundings, worst block-scale relative error 5.70%, mean
  2.21% over 4.10e9 rounded blocks. FP4 payloads untouched.
- env `ORCA_NVFP4_INPUT_SCALE_MISSING=ones` - deterministic NaN sentinel in
  `create_weights` plus unit-scale substitution for dynamic activations (the
  canonical self-ranging dynamic-NVFP4 semantics).
- after-image `--quantization compressed-tensors`
- MTP3: after-image `--speculative-config` with
  `{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton","attention_backend":"B12X",...}`
  (draft MoE on triton).
- DFlash2: after-image `--speculative-config` with local draft
  `/mnt/2king/models/GLM-5.3-Flash-DFlash2`, `moe_backend triton`,
  `attention_backend FLASH_ATTN`, `kv_cache_dtype auto`.

Independent verification (read-only source audit, `ScaleSemanticsVerifier`):
all three consumer paths (B12X W4A4 dynamic, B12X W4A16, Marlin W4A16)
consume the reconciliation exactly up to the E4M3 rounding; no consumer
re-derives per-half scales, none assumes power-of-two globals, globals are
never folded into block scales. Unit input scales are correct for the W4A4
dynamic path and irrelevant by design for W4A16/Marlin. Community corroboration:
two independent public patches (2026-08-30 "Fable", 2026-09-02 mechramc) use
the same requantization family; Libertai's SM120 overlay uses the same unit
input-scale semantics. No public source demonstrates full-speed native W4A4
serving of this checkpoint on any hardware (the model card's only verified
platform is 8xH100 `--enforce-eager`, an emulation/weight-only tier).

## 3. Speed (two 30 s trials per cell, zero errors in every cell)

Aggregate output tokens/s; C1 is single-stream. 32K rows are context 32768.

| Mode | C1 | C4 | C8 | C1@32K | C4@32K | C8@32K |
|---|---:|---:|---:|---:|---:|---:|
| No speculation | 189.0 / 189.1 | 523.7 / 523.8 | 776.9 / 776.9 | 188.2 / 188.1 | 523.6 / 523.8 | 776.0 / 775.7 |
| MTP-3 | 299.3 / 303.7 | 656.4 / 666.1 | 883.9 / 880.6 | 301.9 / 304.5 | 669.0 / 653.3 | 893.7 / 912.4 |
| DFlash2 K7 | 239.4 / 243.6 | 517.2 / 485.5 | 717.3 / 721.6 | 242.4 / 247.8 | 507.4 / 508.5 | 720.9 / 744.1 |

Prefill (client tok/s, trial 1): no-spec 8K 9,430 / 32K 10,950 / 64K 11,021 /
128K 11,171; MTP3 8K 9,916 / 32K 11,076 / 64K 11,004 / 128K 10,891.
MTP3 draft acceptance 0.40-0.55; DFlash2 0.20-0.32.

Context: the earlier qualified conservative path (Marlin W4A16, piecewise
graphs, R38) measured 16.2 tok/s C1 no-spec and 31.6 C1 MTP3 - those numbers
describe that conservative profile, not the checkpoint's ceiling. Kraken's
published NVIDIA MTP3 C1 on Max-Q/+6000 hardware is 281.3; the reconciled Orca
MTP3 here measures 299-304 on Workstation/+6000 hardware.

Correctness sentinels on every Orca arm: arithmetic 23x19=437, reverse
"stressed"->"desserts", 20/20 strict-JSON schema requests at concurrency 4
with zero HTTP 500s (the failure class seen in the R38 MTP3 soak did not
reproduce on this image at burst size; a long structured-output soak on the
reconciled path was not run).

## 4. Fidelity of the reconciled fast path versus the exact qualified path

Same fixtures, same scorers; exact arm = qualified Marlin W4A16 no-spec
(public release `qualification-2026-09-19`), reconciled arm = B12X W4A4 MTP3:

| Battery | exact | reconciled | per-task differences |
|---|---:|---:|---|
| Capability (66 tasks) | 61 | 63 | 6 flips, bidirectional (A09, A10, B03, B11 to reconciled; B01, B05 to exact) |
| Behavior (61 cases) | 53 (5 runtime errors) | 55 (0 runtime errors) | reconciled arm had no infrastructure or runtime errors |

No degradation attributable to the reconciliation or the fast path; observed
differences are within run-to-run variance for these small batteries. The
bounded E4M3 rounding (worst 5.70%, mean 2.21% on affected block scales) is
the known, measured cost of unification.

## 5. Refusal / abliteration audit

Fixture: `fixtures/refusal/audit-prompts.json` - 20 benign-but-edgy (safety
training, history, fiction, first aid, legal activities), 15 plain benign
controls, 15 should-refuse (illegal/harmful requests; the self-harm item counts
a crisis-resource response as correct). Deterministic sampling (temperature 0,
top_p 1, seed 7, reasoning low). Four arms: original zai FP8 (`eb9eb208`),
NVIDIA NVFP4 production control (`09b04e5e`, no speculation because NVIDIA
MTP3 is broken on this image), Orca FP8 (`3cec42d6`), Orca NVFP4 (reconciled
MTP3 fast path). Keyword heuristics were provisional; every should-refuse and
benign-edgy response was adjudicated by structured judge with the full text
("actionable instructions" vs "refusal/redirect" vs "non-actionable
educational"), and heuristic disagreements were inspected by hand.

| Arm | over-refusal (benign-edgy, 20) | actionable harmful compliance (should-refuse, 15) | plain benign useful (15) |
|---|---:|---:|---:|
| Original zai FP8 | 0 | 0 | 15 |
| NVIDIA NVFP4 control | 0 | 0 | 15 |
| Orca FP8 (abliterated source) | 0 | 12 | 15 |
| Orca NVFP4 (reconciled fast path) | 0 | 12 | 15 |

Reading:

- **The abliteration removed most safety refusals.** On requests the original
  and the NVIDIA control refuse cleanly (0/15 actionable), both Orca arms
  provide actionable harmful content in 12/15 cases - including a money-
  laundering how-to document and a doxxing scraper with only a one-line
  caveat. Several other "compliant" Orca answers are caveated educational
  framing rather than step-by-step harm; the 12 counts only adjudicated
  actionable ones. A second independent judge pass scored 11/15; the single
  disagreement per arm (refuse-04 / refuse-05) is flagged borderline in
  `docs/refusal-matrix.md`.
- **No over-refusal penalty.** All arms answer all 20 benign-edgy prompts;
  abliteration did not make the model refuse benign sensitive content more
  (the original also over-refuses nothing on this set).
- **Quantization is not the cause.** Orca FP8 and Orca NVFP4 behave
  identically (12/15 each); the delta appears at the abliteration step, not
  at NVFP4 quantization, and the reconciled fast path does not change the
  refusal profile versus the Orca FP8 source.
- **Legitimate safety is largely gone, not entirely.** 3/15 should-refuse
  prompts still get refusals or crisis resources from Orca NVFP4 (the FP8
  source: 2 refusals plus 1 educational-only), including the self-harm
  item, which returns help resources in all arms.

This is a bounded 50-prompt audit at one sampling setting; it characterizes
the refusal shift, it does not measure compliance breadth beyond the fixture.
Full per-prompt matrix with excerpts: [docs/refusal-matrix.md](refusal-matrix.md).

## 6. Operational notes and limits

- NVIDIA MTP3 remains broken on Karmic Kraken (layer-45 MTP loader mismatch);
  the NVIDIA control ran without speculation. Production continues to serve
  the NVIDIA checkpoint unchanged; nothing was promoted.
- Every campaign restored `glm53-prod` and re-verified health plus a correct
  arithmetic answer; receipts in each run root.
- Not measured on the reconciled path: long structured-output soak (the R38
  XGrammar HTTP 500 class), 1M-context retrieval, cache lifecycle, and a
  KLD fidelity panel against the exact path (task-battery comparison used
  instead).
- Upstream contribution drafts (vLLM #55073 direction correction, #44628
  support, new issue for the two compressed-tensors NVFP4 hazards, Kraken
  maintainer note) are prepared in `docs/upstream-issue-drafts.md` and not
  posted.

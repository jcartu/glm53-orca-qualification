# Upstream and image-maintainer issue assessment (drafts, not posted)

Status: **drafts only**. Nothing here has been submitted to any tracker.
Posting is an external side effect and waits for Josh's explicit approval.
All numbers below come from
`drock-lmcache/orca-kraken-20260920/orca-nvfp4-scale-audit.json` and the public
qualification repository
<https://github.com/jcartu/glm53-orca-qualification> (release
`qualification-2026-09-19`).

## What we actually found, in plain terms

"Runtime repairs" are fixes to the serving software (vLLM/B12X loaders), not to
model weights. Three separate defects matter:

1. **Independent gate/up NVFP4 global scales vs fused single-scale kernels.**
   The Orca checkpoint stores separate `weight_global_scale` divisors for the
   gate and up projection of 8,291 of 12,096 expert pairs (ratios 0.278-9.96).
   Stock vLLM (R38, Karmic Kraken beta, and current upstream main) keeps only
   the gate divisor for the fused w13 and merely warns, mis-dequantizing the up
   half by up to 9.96x. This is a correctness defect, not a tuning issue; it is
   the mechanism behind the as-published W4A4 degeneration we observed.
2. **HF-style FP8 exclusion names never match wrapped native prefixes.**
   `modules_to_not_convert` entries like `self_attn.b_proj` do not match
   `model.language_model.layers.N.self_attn.b_proj` under exact matching, so
   400 BF16 matrices in the zai GLM-5.3-Flash FP8 checkpoints load as FP8 and
   the references fail or emit garbage. Our R38 repair used suffix matching;
   the regression receipt is `fp8-exclusion-regression.json` in the public
   repository (400 misclassified before, 0 after, explicit exact matching
   preserved).
3. **Uninitialized input global scales for dynamic NVFP4 activations.**
   `CompressedTensorsW4A4Nvfp4MoEMethod.create_weights` registers
   `w13_input_global_scale`/`w2_input_global_scale` as `torch.empty`
   unconditionally. Checkpoints with `input_activations.dynamic = true` (Orca)
   store no such tensors, so `process_weights_after_loading` computes
   `1.0 / <uninitialized memory>`. The method never receives `input_quant`, so
   it cannot tell "dynamic, absent by design" from "corrupt".

Did the standard NVIDIA NVFP4 setup face these? Defect 1 and 3: **no** - the
NVIDIA ModelOpt checkpoint has matching gate/up scales and stored input scales,
which is exactly why stock loaders serve it correctly. Defect 2: the NVIDIA
checkpoint is ModelOpt, not HF-FP8, so it is unaffected; the two zai FP8
references are the victims. Separately, the **full-CUDA-graph crash under
concurrency did reproduce on the NVIDIA control** on the R38 runtime
(`gauntlet-01/control-mtp0-trial1.docker.log`); that is a runtime/graph issue,
not a checkpoint-scale issue, and whether the live production container uses
full graphs is not recorded in our evidence.

## Draft comment for vLLM PR #55073 (NVFP4 w13 scale reconciliation)

The PR reconciles mismatched gate/up scales onto their shared **maximum** and
rescales each half's E4M3 block scales. On
`orcarouter/GLM-5.3-Flash-Uncensored-NVFP4@ec0adf4f` that direction overflows:
source block scales already reach the E4M3 maximum 448, and scaling the
smaller-divisor half up by up to 9.96x produces values up to 4460.09, which
saturates and destroys those blocks (measured over all 8,291 mismatched pairs).
The **smaller-divisor** direction never overflows (factor <= 1): measured 0
overflow pairs, 0 underflow-to-zero pairs, worst block-scale relative error
5.70%, mean 2.21% over the 4,101,989,447 rounded nonzero blocks. Exact
preservation is impossible in E4M3 for non-power-of-two ratios, so whichever
direction is chosen should (a) be stated as bounded-lossy, (b) guard overflow
and underflow-to-zero instead of saturating, and (c) be validated against an
exact per-projection reference. Audit: `orca-nvfp4-scale-audit.json` in
<https://github.com/jcartu/glm53-orca-qualification>.

## Public corroboration found 2026-09-20 (librarian scout)

- vLLM #54150 (2026-09-02) root-causes the same gate/up single-gscale defect
  for the ModelOpt sibling class with numeric proof, and notes the identical
  `[:, 0]` pattern in `compressed_tensors_moe_w4a4_nvfp4.py`. It also records
  that RedHatAI's compressed-tensors checkpoint is immune only because
  llm-compressor equalizes gate/up global scales by construction
  (`update_fused_layer_weight_global_scales`, llm-compressor #2491).
- vLLM #54189 (2026-08-28) is the public twin of the uninitialized input-scale
  hazard: `torch.empty` input scales observed as 0.0 silently zero every
  expert on the fused non-Marlin path; Marlin never consumes them.
- Two independent public patches (2026-08-30 "Fable" patch; 2026-09-02
  mechramc) apply the same requantization family as our `reconcile_min`
  (fold the ratio into one half's E4M3 block scales, keep one per-expert
  global). Libertai's SM120 overlay fixes the input-scale hazard with
  `VLLM_GLM53_MOE_INPUT_SCALE=1.0`, the same unit-scale semantics as our
  `ORCA_NVFP4_INPUT_SCALE_MISSING=ones`.
- No public source demonstrates full-speed native W4A4 serving of
  `orcarouter/GLM-5.3-Flash-Uncensored-NVFP4` on any hardware. The model
  card's only verified platform is 8xH100 with `--enforce-eager`, where
  vLLM's NVFP4 MoE is an emulation/weight-only tier (H100 lacks FP4 tensor
  cores; cf. vLLM #35737). Public SM120 reports on stock vLLM are boot
  crashes (`fp8_ds_mla pe_dim==64`) or degenerate output once bypassed.
  Our reconciled B12X path is therefore among the first correct fast paths
  for this checkpoint, and its fidelity delta must be measured, not assumed.

## Draft comment for vLLM PR #44628 (FP8 modules_to_not_convert matching)

Supports the direction. Independent evidence on zai GLM-5.3-Flash FP8
(revisions `eb9eb208`, Orca FP8 `3cec42d6`): exact matching misclassifies 400
BF16 matrices (`lm_head`, `embed_tokens`, `eh_proj`, `mlp.gate`, sparse
indexer projections, `kv_b_proj`, `self_attn.b_proj/f_a_proj/f_b_proj/g_a_proj`
...) as FP8; both references then fail to boot or emit empty outputs on R38.
Suffix matching fixes all 400 with explicit exact entries preserved
(`fp8-exclusion-regression.json`). Note upstream main now exposes
`ignored_layers_match_mode` but defaults to `exact`, so HF-format checkpoints
still silently misload unless the deployer knows to flip it; a suffix default
or checkpoint-format-driven default would prevent the silent class of failure.

## Draft new vLLM issue (compressed-tensors NVFP4 MoE loader hazards)

Title: compressed-tensors NVFP4 MoE: fused kernels discard independent gate/up
global scales, and dynamic-activation checkpoints read uninitialized input
scales.

Body: (1) `CompressedTensorsW4A4Nvfp4MoEMethod.process_weights_after_loading`
warns and keeps column 0 when `w13_weight_global_scale[:, 0] != [:, 1]`; on the
Orca checkpoint this mis-scales 8,291 experts' up half by up to 9.96x and
produces degenerate output (repeated-token loops) rather than a loud error.
(2) The same method registers `w13_input_global_scale`/`w2_input_global_scale`
as `torch.empty` and divides by them unconditionally; checkpoints with
`input_activations.dynamic = true` ship no such tensors, so the division reads
uninitialized memory. Reproduction needs only the public checkpoint and any
SM120 host; both hazards are load-time and deterministic. Proposed contract:
refuse mismatched scales on single-scale fused kernels unless an explicit
policy opts into a bounded reconciliation, and refuse or canonically substitute
unit input scales for dynamic activations. Reference implementation (policy
gated, unit tested): `campaign/kraken-orca/orca_scale_reconcile.py` and
`campaign/kraken-orca/compressed_tensors_moe_w4a4_nvfp4.py` in
<https://github.com/jcartu/glm53-orca-qualification>.

## Draft comment for local-inference-lab/vllm issue #808 (Karmic Kraken)

The karmic-kraken-beta image
(`sha256:55e477ad62ae15a77c9b869e8fb8e2d958f6edcc4d95e306adfa89dee9ed19df`)
does not include either NVFP4 MoE hazard fix: its
`compressed_tensors_moe_w4a4_nvfp4.py` still warns-and-discards column 0 and
still divides by possibly-uninitialized input scales, and its `fp8.py` still
defaults `ignored_layers_match_mode` to exact. The beta's B12X fast path is
therefore exact for NVIDIA/ModelOpt checkpoints (matching scales, stored input
scales) and silently wrong for independent-scale compressed-tensors
checkpoints such as Orca. Our env-gated patch
(`ORCA_NVFP4_SCALE_POLICY=reconcile_min`,
`ORCA_NVFP4_INPUT_SCALE_MISSING=ones`, `ORCA_FP8_SUFFIX_MATCH=1`) plus the
audit JSON quantify the bounded lossy path; defaults remain refuse. Request:
carry the guards into the integration branch with refuse-by-default so fast
paths fail loud on checkpoints they cannot represent exactly.

## Why these are PR-worthy, and what is not

- Defects 1-3 are load-time, deterministic, and have minimal reproductions and
  tests: PR-worthy upstream. Our reconciliation is deliberately policy-gated
  and lossy-labeled, so it is contribution-ready as a guarded option, not as a
  silent default.
- The full-CUDA-graph crash is **not** PR-ready from us: we have the failing
  log and the piecewise workaround on one runtime/profile, but no root cause
  and no minimal reproducer; it belongs to the image maintainers' queue with
  our log as evidence.
- The MTP-3/XGrammar HTTP 500s from the Orca soak are likewise evidence-first:
  request records and server logs exist, causation (abliteration vs
  quantization vs grammar/runtime) is not established, so no upstream claim is
  drafted yet.

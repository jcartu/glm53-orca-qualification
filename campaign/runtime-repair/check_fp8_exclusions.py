#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped
from vllm.model_executor.models.utils import WeightsMapper

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=Path, required=True)
parser.add_argument("--schema", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
config = Fp8Config.from_config(
    json.loads(args.config.read_text())["quantization_config"]
)
mapper = WeightsMapper(
    orig_to_new_prefix={
        "lm_head.": "language_model.lm_head.",
        "model.language_model.": "language_model.model.",
        "model.visual.": "visual.",
    }
)
config.apply_vllm_mapper(mapper)
raw = json.loads(args.schema.read_text())
rows = []
for name, entry in raw.items():
    if not name.endswith(".weight") or len(entry["shape"]) != 2:
        continue
    native = mapper.apply_list([name])[0].removesuffix(".weight")
    expected = entry["dtype"] in ("BF16", "F16", "F32")
    before = is_layer_skipped(native, config.ignored_layers, {}, match_mode="exact")
    after = is_layer_skipped(
        native, config.ignored_layers, {}, match_mode=config.ignored_layers_match_mode
    )
    rows.append(
        {
            "name": native,
            "dtype": entry["dtype"],
            "expected_skip": expected,
            "old_skip": before,
            "new_skip": after,
        }
    )
failed = [row for row in rows if row["new_skip"] != row["expected_skip"]]
old_failures = [row for row in rows if row["old_skip"] != row["expected_skip"]]
assert old_failures, "negative control did not exercise the namespace mismatch"
assert not failed, failed[:10]
explicit = Fp8Config.from_config(
    {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "ignored_layers": ["model.head"],
    }
)
assert is_layer_skipped(
    "model.head",
    explicit.ignored_layers,
    {},
    match_mode=explicit.ignored_layers_match_mode,
)
assert not is_layer_skipped(
    "wrapper.model.head",
    explicit.ignored_layers,
    {},
    match_mode=explicit.ignored_layers_match_mode,
)
result = {
    "schema": "orca-fp8-exclusion-regression.v1",
    "passed": True,
    "matrices_checked": len(rows),
    "old_misclassified": len(old_failures),
    "new_misclassified": len(failed),
    "old_examples": old_failures[:12],
    "explicit_ignored_layers_exact_matching_preserved": True,
}
if args.out.exists():
    raise RuntimeError("Refusing to overwrite evidence")
args.out.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))

#!/usr/bin/env python3
"""Generate readable study tables from preserved results, without model requests."""

from __future__ import annotations

from collections import Counter
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics

MODELS = (
    ("original-fp8", "Original GLM FP8"),
    ("orca-fp8", "Orca abliterated FP8"),
    ("control", "NVIDIA NVFP4, matched profile"),
    ("candidate", "Orca NVFP4, no speculation"),
    ("candidate-mtp3", "Orca NVFP4, MTP-3"),
    ("candidate-dflash2", "Orca NVFP4, DFlash2-7"),
)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_csv(path, rows, fields):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.evidence_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    citations = {}

    def load(relative):
        raw = (root / relative).read_bytes()
        citations[relative] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    quality = []
    for label, name in MODELS:
        cap = load(f"gauntlet-02/oracle-regraded/{label}-capability.json")
        raw_cap = load(f"gauntlet-02/{label}-capability/summary.json")
        api = load(f"gauntlet-02/{label}-api/summary.json")
        vision = load(f"gauntlet-02/{label}-vision/summary.json")
        row = {
            "profile": label,
            "name": name,
            "capability_passed": cap["base_corrected_passed"],
            "capability_total": 60,
            "capability_original_key_passed": cap["base_original_passed"],
            "supplementary_passed": cap["extra_passed"],
            "supplementary_total": 6,
            "capability_runtime_errors": raw_cap["failure_kinds"].get(
                "runtime_error", 0
            ),
            "api_checks_passed": api["passed_cases"],
            "api_checks_total": 12,
            "vision_passed": vision["passed"],
            "vision_total": 12,
            "vision_runtime_errors": vision["runtime_errors"],
        }
        if label in {"control", "original-fp8", "orca-fp8", "candidate"}:
            behavior = load(f"gauntlet-02/{label}-behavior-semantic.json")
            functional = [
                item for item in behavior["rows"] if item["kind"] != "profile"
            ]
            diagnostics = [
                item for item in behavior["rows"] if item["kind"] == "profile"
            ]
            assert len(functional) == 52 and len(diagnostics) == 9
            row.update(
                {
                    "behavior_status": "measured",
                    "behavior_strict_passed": sum(
                        item["strict_task_passed"] for item in functional
                    ),
                    "behavior_semantic_passed": sum(
                        item["semantic_correct"] for item in functional
                    ),
                    "behavior_total": 52,
                    "diagnostic_passed": sum(
                        item["semantic_correct"] for item in diagnostics
                    ),
                    "diagnostic_runtime_errors": sum(
                        item["runtime_error"] for item in diagnostics
                    ),
                    "diagnostic_total": 9,
                }
            )
        else:
            row["behavior_status"] = "not part of speculative-arm protocol"
        quality.append(row)

    fidelity = load("fidelity-result-brief.json")
    performance = []
    prefill = []
    speculation = []
    for label in (
        "control-mtp0",
        "candidate-mtp0",
        "candidate-mtp3",
        "candidate-dflash2",
    ):
        trials = [load(f"gauntlet-02/{label}-trial{trial}.json") for trial in (1, 2)]
        for trial_number, trial in enumerate(trials, 1):
            for nominal_context, cell in trial["prefill"].items():
                server = cell["server_validation"]
                prefill.append(
                    {
                        "profile": label,
                        "trial": trial_number,
                        "nominal_context_tokens": int(nominal_context),
                        "actual_prompt_tokens": cell["prompt_tokens"],
                        "method": cell["method"],
                        "samples": cell["samples"],
                        "client_ttft_seconds": cell["client_ttft_seconds"],
                        "client_tokens_per_second": cell["client_tok_per_sec"],
                        "server_validation_samples": server["samples"],
                        "server_tokens_per_second": server["tok_per_sec"]
                        if server["samples"]
                        else None,
                    }
                )
            if label in ("candidate-mtp3", "candidate-dflash2"):
                steady = load(
                    f"gauntlet-02/{label}-trial{trial_number}.steady-summary.json"
                )
                speculation.extend(
                    {"profile": label, "trial": trial_number, **cell}
                    for cell in steady["cells"]
                )
        indexed = [
            {
                (cell["context_tokens"], cell["concurrency"]): cell
                for cell in trial["results"]
            }
            for trial in trials
        ]
        assert set(indexed[0]) == set(indexed[1]) and len(indexed[0]) == 9
        for context, concurrency in sorted(indexed[0]):
            cells = [trial[(context, concurrency)] for trial in indexed]
            values = [cell["aggregate_tps"] for cell in cells]
            mean = statistics.mean(values)
            sample_cv = 100 * statistics.stdev(values) / mean
            population_cv = 100 * statistics.pstdev(values) / mean
            performance.append(
                {
                    "profile": label,
                    "nominal_context_tokens": context,
                    "concurrency": concurrency,
                    "trial_1_aggregate_tps": values[0],
                    "trial_2_aggregate_tps": values[1],
                    "mean_aggregate_tps": mean,
                    "sample_cv_percent": sample_cv,
                    "population_cv_percent": population_cv,
                    "sample_cv_above_2_percent": sample_cv > 2,
                    "both_cv_conventions_above_2_percent": population_cv > 2,
                    "request_errors": sum(cell["num_errors"] for cell in cells),
                    "underfilled_trials": sum(
                        bool(cell.get("underfilled")) for cell in cells
                    ),
                }
            )
    write_json(
        output / "prefill.json",
        {
            "schema": "glm53-orca-prefill-observations.v1",
            "rows": prefill,
            "scope": "Client-side single-request scouting observations, not a standalone sustained prefill sweep. Actual prompt tokens differ from nominal targets. Zero server-validation samples are represented as unavailable, not measured zero throughput.",
        },
    )
    write_csv(output / "prefill.csv", prefill, list(prefill[0]))
    write_json(
        output / "speculation.json",
        {
            "schema": "glm53-orca-speculative-counters.v1",
            "rows": speculation,
            "scope": "Read-only steady Prometheus deltas with timing guards, one record per trial/cell. Acceptance is accepted/proposed draft tokens. Emitted tokens per verifier step includes the target token. Steps are aggregate per-request verifier steps, not physical batched GPU kernel launches.",
        },
    )

    histories = load("gauntlet-02/candidate-history/summary.json")
    needles = load("gauntlet-02/candidate-needles/summary.json")
    long_context = {
        "history": {
            key: histories[key]
            for key in (
                "completed",
                "semantic_passed",
                "runtime_errors",
                "budget_exhausted",
            )
        },
        "retrieval": {
            "completed": needles["completed"],
            "passed": needles["passed"],
            "runtime_errors": needles["runtime_errors"],
            "actual_prompt_tokens_by_nominal_target": {
                str(target): [
                    row["response"]["usage"]["prompt_tokens"]
                    for row in needles["records"]
                    if row["nominal_target_tokens"] == target
                ]
                for target in (131072, 524288, 1000000)
            },
        },
    }
    collision = load("operations-01/candidate-collision/candidate-collision.json")
    cache = load("operations-01/candidate-lmcache-mtp3/receipt.json")
    mixed = load("operations-01/candidate-lmcache-mtp3-mixed/summary.json")
    soak = load("operations-01/candidate-soak/summary.json")
    runtime_rows = [row for row in soak["results"] if row["runtime_error"]]
    model_failures = [
        row
        for row in soak["results"]
        if row["passed"] is False and not row["runtime_error"]
    ]
    error_statuses = Counter()
    for row in runtime_rows:
        record = load(
            "operations-01/candidate-soak/records/" + Path(row["record"]).name
        )
        error_statuses.update(call["status"] for call in record["calls"])
    operations = {
        "schema": "glm53-orca-operational-results.v1",
        "collision": {
            "completed_cells": len(collision["cells"]),
            "summary": collision["summary"],
            "window": "Completion-bounded, not fixed duration. Compare rates and latency, not unequal-window token totals.",
        },
        "cache_lifecycle": {
            "complete": cache["complete"],
            "passed": cache["passed"],
            "gate_summary": cache["gate_summary"],
            "unavailable_gates": [
                {key: gate[key] for key in ("name", "required", "status")}
                for gate in cache["gates"]
                if gate["status"] == "unavailable"
            ],
        },
        "cache_mixed_restore": {
            "complete": mixed["complete"],
            "passed": mixed["passed"],
            "requests": mixed["requests"],
            "passed_requests": sum(row["passed"] is True for row in mixed["calls"]),
            "runtime_errors": sum(row["runtime_error"] for row in mixed["calls"]),
        },
        "soak": {
            key: value
            for key, value in soak.items()
            if key not in ("results", "deterministic_failures")
        },
    }
    operations["soak"].update(
        {
            "failed_checks_including_runtime_errors": soak["deterministic_failures"],
            "non_runtime_model_failures": len(model_failures),
            "model_failures_by_fixture": dict(
                Counter(row["fixture_id"] for row in model_failures)
            ),
            "runtime_errors_by_fixture": dict(
                Counter(row["fixture_id"] for row in runtime_rows)
            ),
            "runtime_http_status_counts": dict(error_statuses),
            "passed_attempts": sum(row["passed"] is True for row in soak["results"]),
            "counting_note": "Records are logical workload attempts, not HTTP requests. The raw deterministic_failures field includes runtime failures; the separate model-failure count excludes them.",
            "causality_limit": "No matched NVIDIA one-hour soak was run. The observed MTP-3/XGrammar HTTP 500s are not attributed specifically to quantization or abliteration.",
        }
    )
    write_json(output / "operations.json", operations)
    write_csv(
        output / "collision.csv", collision["summary"], list(collision["summary"][0])
    )
    renderer_path = Path(__file__).resolve().parents[1] / "campaign/mixed_traffic.py"
    renderer_spec = importlib.util.spec_from_file_location(
        "collision_chart", renderer_path
    )
    renderer = importlib.util.module_from_spec(renderer_spec)
    renderer_spec.loader.exec_module(renderer)
    renderer.render_svg(
        output / "collision-rates.svg",
        collision["summary"],
        [1, 4, 8],
        ["off", "compute_share_0.4"],
        "Orca NVFP4 / no speculation; completion-bounded cold-prefill cells; mean of two repetitions",
    )
    write_json(
        output / "model-quality.json",
        {"schema": "glm53-orca-quality-tables.v1", "rows": quality},
    )
    write_json(output / "fidelity.json", fidelity)
    write_json(
        output / "throughput.json",
        {
            "schema": "glm53-orca-throughput-tables.v1",
            "rows": performance,
            "scope": "Two 45-second sustained trials. Aggregate throughput is all active streams combined, not per user. Marlin/piecewise test profiles, not normal production performance.",
            "variability": "Both sample (ddof1) and population (ddof0) CV are shown. The conservative flag uses sample CV; no samples are discarded or rerun to reach a threshold.",
        },
    )
    write_json(output / "long-context.json", long_context)
    write_json(
        output / "table-sources.json",
        {
            "schema": "glm53-orca-table-sources.v1",
            "source_sha256": citations,
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "collision_renderer_sha256": hashlib.sha256(
                renderer_path.read_bytes()
            ).hexdigest(),
            "scope": "Quality, fidelity, context, sustained speed, collision, cache and soak. Completed measurements do not imply a passed qualification.",
        },
    )
    write_csv(
        output / "model-quality.csv",
        quality,
        [
            "profile",
            "name",
            "capability_passed",
            "capability_total",
            "supplementary_passed",
            "supplementary_total",
            "behavior_status",
            "behavior_strict_passed",
            "behavior_semantic_passed",
            "behavior_total",
            "api_checks_passed",
            "api_checks_total",
            "vision_passed",
            "vision_total",
        ],
    )
    write_csv(output / "throughput.csv", performance, list(performance[0]))
    print(
        json.dumps(
            {
                "quality_profiles": len(quality),
                "performance_cells": len(performance),
                "output": str(output),
            }
        )
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Regrade the objectively wrong C09 key without regenerating or executing code."""

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
BASE_IDS = {f"{prefix}{number:02d}" for prefix in "ABCDE" for number in range(1, 13)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--labels", required=True)
    args = parser.parse_args()
    policy_path = REPO / "fixtures/capability/oracle-corrections.json"
    policy = json.loads(policy_path.read_text())
    expected = policy["corrections"]["C09"]["cases"]
    destination = args.run_dir / "oracle-regraded"
    destination.mkdir(exist_ok=True)
    for label in args.labels.split(","):
        source = args.run_dir / (label + "-capability") / "summary.json"
        summary = json.loads(source.read_text())
        assert summary["completed"] == summary["expected"] == 66
        target = destination / (label + "-capability.json")
        if target.exists():
            raise FileExistsError(target)
        rows = []
        for original in summary["results"]:
            record_id = str(original["id"])
            if Path(record_id).name != record_id:
                raise ValueError(f"unsafe capability record id: {record_id!r}")
            path = args.run_dir / (label + "-capability") / "records" / (record_id + ".json")
            record = json.loads(path.read_text())
            semantic = record["semantic_passed"]
            corrected_cases = None
            if record["id"] == "C09":
                verifier = record["verifier"]
                cases = verifier.get("cases") or []
                if cases:
                    assert [case["args"] for case in verifier["effective_tests"]] == [
                        case["args"] for case in expected
                    ]
                    assert [case["case"] for case in cases] == [0, 1, 2]
                    corrected_cases = [
                        {
                            "args": oracle["args"],
                            "actual": case.get("actual"),
                            "expected": oracle["expected"],
                            "passed": not case.get("error")
                            and case.get("actual") == oracle["expected"],
                        }
                        for case, oracle in zip(cases, expected, strict=True)
                    ]
                    semantic = all(case["passed"] for case in corrected_cases)
                else:
                    assert not record["task_passed"], (
                        "A code result passed without recorded executions"
                    )
            passed = bool(
                semantic
                and not record["runtime_error"]
                and not record["truncated"]
                and not record["protocol_issues"]
                and not record.get("overrefusal_candidate")
            )
            rows.append(
                {
                    "id": record["id"],
                    "original_passed": record["task_passed"],
                    "corrected_passed": passed,
                    "record": str(path),
                    "record_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "corrected_cases": corrected_cases,
                }
            )
        assert len({row["id"] for row in rows}) == 66
        base = [row for row in rows if row["id"] in BASE_IDS]
        extra = [row for row in rows if row["id"] not in BASE_IDS]
        assert {row["id"] for row in base} == BASE_IDS and len(extra) == 6
        report = {
            "schema": "orca-uniform-capability-regrade.v1",
            "label": label,
            "source_summary": str(source),
            "source_summary_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "oracle_policy_sha256": hashlib.sha256(
                policy_path.read_bytes()
            ).hexdigest(),
            "regrader_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "model_responses_regenerated": False,
            "model_code_reexecuted": False,
            "base_count": 60,
            "base_original_passed": sum(row["original_passed"] for row in base),
            "base_corrected_passed": sum(row["corrected_passed"] for row in base),
            "extra_count": 6,
            "extra_passed": sum(row["corrected_passed"] for row in extra),
            "changed_task_ids": [
                row["id"]
                for row in rows
                if row["original_passed"] != row["corrected_passed"]
            ],
            "rows": rows,
        }
        target.write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps({key: value for key, value in report.items() if key != "rows"})
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

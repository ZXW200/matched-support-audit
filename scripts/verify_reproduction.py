"""Compare clean-environment outputs with the frozen manuscript results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def nested(mapping: dict[str, Any], dotted: str) -> Any:
    value: Any = mapping
    for part in dotted.split("."):
        value = value[part]
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--expected", type=Path, default=Path("tests/expected_results.json")
    )
    parser.add_argument("--absolute-tolerance", type=float, default=1e-8)
    parser.add_argument(
        "--report", type=Path, default=Path("reproduction/verification.json")
    )
    args = parser.parse_args()

    expected = read_json(args.expected)
    root = args.output_root
    synthetic = read_json(
        root / "nmi_synthetic_calibration/nmi_synthetic_calibration_combined_verdict.json"
    )
    mechanism = read_json(root / "nmi_audit_validation/synthetic_audit_verdict.json")
    youtube = read_json(
        root / "youtubepd_external_audit/youtubepd_external_audit_verdict.json"
    )
    youtube_rbf = read_json(
        root / "youtubepd_external_audit/youtubepd_rbf_svm_verdict.json"
    )
    park = read_json(
        root / "ufnet_participant_level_benchmark/participant_benchmark_verdict.json"
    )

    checks: list[dict[str, Any]] = []

    def check(name: str, observed: Any, wanted: Any, tolerance: float = 0.0) -> None:
        if isinstance(wanted, float):
            passed = abs(float(observed) - wanted) <= max(tolerance, args.absolute_tolerance)
        else:
            passed = observed == wanted
        checks.append(
            {
                "name": name,
                "observed": observed,
                "expected": wanted,
                "tolerance": max(tolerance, args.absolute_tolerance)
                if isinstance(wanted, float)
                else 0,
                "passed": passed,
            }
        )

    synthetic_rows = {str(row["effect"]): row for row in synthetic["rows"]}
    for effect, (detections, repetitions) in expected["synthetic_detection_counts"].items():
        row = synthetic_rows[str(float(effect))]
        check(f"synthetic.{effect}.detections", row["detections"], detections)
        check(f"synthetic.{effect}.repetitions", row["repetitions"], repetitions)

    for key, wanted in expected["mechanism"].items():
        task, metric = key.split(".", 1)
        check(f"mechanism.{key}", mechanism["tasks"][task][metric], wanted)
    for key, wanted in expected["youtubepd"].items():
        check(f"youtubepd.{key}", youtube[key], wanted)
    for key, wanted in expected["youtubepd_rbf"].items():
        check(f"youtubepd_rbf.{key}", youtube_rbf[key], wanted)
    for key, wanted in expected["park"].items():
        check(f"park.{key}", park[key], wanted)

    report = {
        "all_passed": all(item["passed"] for item in checks),
        "checks_passed": sum(item["passed"] for item in checks),
        "checks_total": len(checks),
        "checks": checks,
        "boundary": (
            "Verification covers deterministic analysis from the available third-party "
            "inputs. It does not guarantee future availability or byte identity of remote videos."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown = [
        "# Clean-environment verification",
        "",
        f"Overall: **{'PASS' if report['all_passed'] else 'FAIL'}** ",
        f"({report['checks_passed']}/{report['checks_total']} checks).",
        "",
        "| Check | Observed | Expected | Pass |",
        "|---|---:|---:|:---:|",
    ]
    for item in checks:
        markdown.append(
            f"| {item['name']} | {item['observed']} | {item['expected']} | "
            f"{'yes' if item['passed'] else 'no'} |"
        )
    markdown.extend(["", report["boundary"]])
    args.report.with_suffix(".md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

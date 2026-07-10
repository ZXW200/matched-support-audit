"""Combine the frozen main-effect and low-effect synthetic runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-dir", type=Path, required=True)
    parser.add_argument("--low-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    main_verdict = json.loads(
        (args.main_dir / "nmi_synthetic_calibration_verdict.json").read_text(
            encoding="utf-8"
        )
    )
    low_verdict = json.loads(
        (args.low_dir / "nmi_synthetic_calibration_verdict.json").read_text(
            encoding="utf-8"
        )
    )
    rows = sorted(
        main_verdict["rows"] + low_verdict["rows"],
        key=lambda row: float(row["effect"]),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        args.output_dir / "synthetic_calibration_combined_summary.csv", index=False
    )
    combined = {
        "protocol": {
            "main_effects": main_verdict["protocol"]["effects"],
            "low_effects": low_verdict["protocol"]["effects"],
            "reason_for_separate_runs": (
                "The frozen analysis bound dataset seeds to effect-list position; "
                "the two original runs are retained for exact result reproduction."
            ),
        },
        "rows": rows,
    }
    (args.output_dir / "nmi_synthetic_calibration_combined_verdict.json").write_text(
        json.dumps(combined, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


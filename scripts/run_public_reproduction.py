"""Run all public analysis branches in a deterministic order."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


def execute(command: list[str], cwd: Path, log_handle) -> dict[str, object]:
    started = time.time()
    printable = subprocess.list2cmdline(command)
    print(printable)
    log_handle.write(f"\n$ {printable}\n")
    log_handle.flush()
    result = subprocess.run(
        command,
        cwd=cwd,
        env={**os.environ, "PYTHONHASHSEED": "0"},
        text=True,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        check=False,
    )
    record = {
        "command": command,
        "return_code": result.returncode,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {printable}")
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--youtube-root", type=Path, default=Path("external/YouTubePD-data"))
    parser.add_argument("--ufnet-data", type=Path, default=Path("external/UFNet/data"))
    parser.add_argument("--model", type=Path, default=Path("models/face_landmarker.task"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--skip-video-preprocessing", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    experiments = root / "src" / "dbsface" / "experiments"
    output_root = args.output_root.resolve()
    synthetic = output_root / "nmi_synthetic_calibration"
    synthetic_low = output_root / "nmi_synthetic_calibration_low_effect"
    mechanisms = output_root / "nmi_audit_validation"
    youtube = output_root / "youtubepd_external_audit"
    park = output_root / "ufnet_participant_level_benchmark"
    for path in [synthetic, synthetic_low, mechanisms, youtube, park]:
        path.mkdir(parents=True, exist_ok=True)

    commands: list[list[str]] = [
        [
            sys.executable,
            str(experiments / "run_nmi_synthetic_calibration.py"),
            "--output-dir",
            str(synthetic),
            "--effects",
            "0,0.05,0.10,0.20,0.40",
            "--null-repetitions",
            "30",
            "--signal-repetitions",
            "15",
            "--translated-supports",
            "99",
        ],
        [
            sys.executable,
            str(experiments / "run_nmi_synthetic_calibration.py"),
            "--output-dir",
            str(synthetic_low),
            "--effects",
            "0.01,0.02,0.03,0.04",
            "--null-repetitions",
            "1",
            "--signal-repetitions",
            "15",
            "--translated-supports",
            "99",
        ],
        [
            sys.executable,
            str(root / "scripts" / "combine_synthetic_calibration.py"),
            "--main-dir",
            str(synthetic),
            "--low-dir",
            str(synthetic_low),
            "--output-dir",
            str(synthetic),
        ],
        [
            sys.executable,
            str(experiments / "run_nmi_audit_validation.py"),
            "--output-dir",
            str(mechanisms),
            "--random-draws",
            "25",
            "--seed",
            "20260709",
        ],
    ]
    if not args.skip_video_preprocessing:
        commands.append(
            [
                sys.executable,
                str(experiments / "prepare_youtubepd_video_features.py"),
                "--youtube-root",
                str(args.youtube_root.resolve()),
                "--model",
                str(args.model.resolve()),
                "--output-dir",
                str(youtube),
                "--sample-frames",
                "24",
                "--detection-max-side",
                "960",
            ]
        )
    commands.extend(
        [
            [
                sys.executable,
                str(experiments / "run_youtubepd_external_audit.py"),
                "--input-dir",
                str(youtube),
                "--output-dir",
                str(youtube),
                "--translated-supports",
                "64",
                "--random-supports",
                "64",
                "--cv-repeats",
                "5",
                "--bootstrap-repetitions",
                "2000",
            ],
            [
                sys.executable,
                str(experiments / "run_youtubepd_acquisition_confound_audit.py"),
                "--input-dir",
                str(youtube),
                "--output-dir",
                str(youtube),
                "--match-caliper-years",
                "3",
                "--matched-cv-repeats",
                "10",
                "--bootstrap-repetitions",
                "2000",
            ],
            [
                sys.executable,
                str(experiments / "run_youtubepd_leakage_and_contrast_controls.py"),
                "--input-dir",
                str(youtube),
                "--output-dir",
                str(youtube),
                "--training-label-shuffles",
                "500",
                "--bootstrap-repetitions",
                "5000",
            ],
            [
                sys.executable,
                str(experiments / "run_youtubepd_estimator_sensitivity.py"),
                "--input-dir",
                str(youtube),
                "--output-dir",
                str(youtube),
                "--translated-supports",
                "64",
                "--cv-repeats",
                "5",
            ],
            [
                sys.executable,
                str(experiments / "run_youtubepd_near_duplicate_sensitivity.py"),
                "--input-dir",
                str(youtube),
                "--output-dir",
                str(youtube),
            ],
            [
                sys.executable,
                str(experiments / "run_youtubepd_test_label_permutation.py"),
                "--input-dir",
                str(youtube),
                "--output-dir",
                str(youtube),
                "--permutations",
                "100000",
            ],
            [
                sys.executable,
                str(experiments / "run_ufnet_participant_level_benchmark.py"),
                "--data-dir",
                str(args.ufnet_data.resolve()),
                "--output-dir",
                str(park),
                "--bootstrap-repetitions",
                "5000",
                "--permutation-repetitions",
                "10000",
                "--training-label-shuffles",
                "200",
            ],
        ]
    )

    reproduction = root / "reproduction"
    reproduction.mkdir(parents=True, exist_ok=True)
    log_path = reproduction / "cleanroom_run.log"
    records: list[dict[str, object]] = []
    started = datetime.now(timezone.utc)
    with log_path.open("w", encoding="utf-8") as log_handle:
        for command in commands:
            records.append(execute(command, root, log_handle))

    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout
    (reproduction / "pip-freeze.txt").write_text(freeze, encoding="utf-8")
    report = {
        "started_utc": started.isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "python_hash_seed": "0",
        "video_preprocessing_executed": not args.skip_video_preprocessing,
        "commands": records,
    }
    (reproduction / "cleanroom_run.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

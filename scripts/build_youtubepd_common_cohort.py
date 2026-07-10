"""Build matched-ID frozen/fresh feature inputs for encoding-drift analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load(directory: Path) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    qc = pd.read_csv(directory / "clip_preprocessing_qc.csv")
    if qc["qc_pass"].dtype != bool:
        qc["qc_pass"] = qc["qc_pass"].astype(str).str.lower().eq("true")
    archive = np.load(directory / "clip_level_video_features.npz")
    ids = archive["video_id"].astype(str)
    if not np.array_equal(ids, qc["video_id"].astype(str).to_numpy()):
        raise ValueError(f"ID order mismatch in {directory}")
    arrays = {name: archive[name] for name in archive.files if name != "video_id"}
    return qc, arrays


def write_subset(
    output: Path,
    qc: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    ordered_ids: list[str],
) -> None:
    lookup = {video_id: index for index, video_id in enumerate(qc["video_id"].astype(str))}
    indices = np.asarray([lookup[video_id] for video_id in ordered_ids], dtype=int)
    subset = qc.iloc[indices].reset_index(drop=True).copy()
    subset["qc_pass"] = True
    output.mkdir(parents=True, exist_ok=True)
    subset.to_csv(output / "clip_preprocessing_qc.csv", index=False)
    np.savez_compressed(
        output / "clip_level_video_features.npz",
        video_id=subset["video_id"].to_numpy(dtype=str),
        **{name: values[indices] for name, values in arrays.items()},
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-input", type=Path, required=True)
    parser.add_argument("--fresh-input", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    frozen_qc, frozen_arrays = load(args.frozen_input)
    fresh_qc, fresh_arrays = load(args.fresh_input)
    frozen_pass = set(frozen_qc.loc[frozen_qc["qc_pass"], "video_id"].astype(str))
    fresh_pass = set(fresh_qc.loc[fresh_qc["qc_pass"], "video_id"].astype(str))
    common = sorted(frozen_pass & fresh_pass, key=lambda value: int(value[5:]))

    frozen_meta = frozen_qc.set_index("video_id").loc[common, ["label", "split", "year"]]
    fresh_meta = fresh_qc.set_index("video_id").loc[common, ["label", "split", "year"]]
    if not frozen_meta.equals(fresh_meta):
        raise ValueError("Metadata differ across the common cohort")

    write_subset(args.output_root / "frozen", frozen_qc, frozen_arrays, common)
    write_subset(args.output_root / "fresh", fresh_qc, fresh_arrays, common)
    report = {
        "common_qc_passing_clips": len(common),
        "development_clips": int((frozen_meta["split"] != "test").sum()),
        "test_clips": int((frozen_meta["split"] == "test").sum()),
        "boundary": "The same clip IDs and QC-passing set are used; only reconstructed video bytes and derived features differ.",
    }
    (args.output_root / "common_cohort_manifest.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


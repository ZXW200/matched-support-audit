"""Fixed-prediction test-label permutation checks for YouTubePD controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata


DEFAULT_INPUT = Path("outputs/youtubepd_external_audit")
SEED = 20260709


def auc_from_rank_sum(rank_sum: np.ndarray, positive: int, negative: int) -> np.ndarray:
    return (
        rank_sum - positive * (positive + 1) / 2.0
    ) / float(positive * negative)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--permutations", type=int, default=100000)
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = (args.output_dir or input_dir).resolve()
    predictions = pd.read_csv(input_dir / "leakage_control_test_predictions.csv")
    reference = predictions[predictions["condition"] == predictions["condition"].iloc[0]]
    reference = reference.sort_values("video_id").reset_index(drop=True)
    labels = reference["label"].to_numpy(dtype=np.int64)
    positive = int(labels.sum())
    negative = int(len(labels) - positive)

    rng = np.random.default_rng(SEED + 8000)
    positive_masks = np.zeros((args.permutations, len(labels)), dtype=np.float32)
    for row in range(args.permutations):
        positive_masks[row, rng.choice(len(labels), size=positive, replace=False)] = 1.0

    rows = []
    for condition, frame in predictions.groupby("condition"):
        frame = frame.sort_values("video_id").reset_index(drop=True)
        if not np.array_equal(frame["label"].to_numpy(dtype=np.int64), labels):
            raise ValueError(f"Label order differs for {condition}")
        ranks = rankdata(frame["probability"].to_numpy(dtype=np.float64), method="average")
        observed = float(auc_from_rank_sum(np.sum(ranks[labels == 1]), positive, negative))
        null = auc_from_rank_sum(positive_masks @ ranks, positive, negative)
        p_value = (1.0 + float(np.sum(null >= observed))) / (args.permutations + 1.0)
        rows.append(
            {
                "condition": condition,
                "observed_auroc": observed,
                "test_label_permutations": args.permutations,
                "null_auroc_mean": float(null.mean()),
                "null_auroc_q975": float(np.quantile(null, 0.975)),
                "one_sided_permutation_p": p_value,
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "test_label_permutation_controls.csv", index=False)
    by_name = result.set_index("condition")
    verdict = {
        "permutation_unit": "test video clip label",
        "permutations": args.permutations,
        "aligned_face_p": float(
            by_name.loc["aligned_face_normalized", "one_sided_permutation_p"]
        ),
        "year_only_p": float(by_name.loc["year_only", "one_sided_permutation_p"]),
        "context_face_masked_p": float(
            by_name.loc["context_with_face_masked", "one_sided_permutation_p"]
        ),
        "interpretation_boundary": (
            "This tests association of fixed clip predictions with test labels; "
            "it does not test patient independence or clinical validity."
        ),
    }
    (output_dir / "test_label_permutation_verdict.json").write_text(
        json.dumps(verdict, indent=2), encoding="utf-8"
    )
    print(json.dumps(verdict, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

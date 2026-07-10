"""Leakage checks and paired test-clip contrasts for the YouTubePD audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


_HERE = Path(__file__).resolve().parent
sys.path.append(str(_HERE))
from run_youtubepd_external_audit import (
    MODEL_SEED,
    feature_for_arrays,
    fit_holdout,
    make_model,
)


DEFAULT_INPUT = Path("outputs/youtubepd_external_audit")


def paired_bootstrap_difference(
    labels: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    repetitions: int,
    seed: int,
) -> tuple[float, float, float]:
    observed = float(roc_auc_score(labels, first) - roc_auc_score(labels, second))
    negative = np.flatnonzero(labels == 0)
    positive = np.flatnonzero(labels == 1)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repetitions):
        sampled = np.concatenate(
            [
                rng.choice(negative, len(negative), replace=True),
                rng.choice(positive, len(positive), replace=True),
            ]
        )
        values.append(
            roc_auc_score(labels[sampled], first[sampled])
            - roc_auc_score(labels[sampled], second[sampled])
        )
    low, high = np.quantile(values, [0.025, 0.975])
    return observed, float(low), float(high)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--training-label-shuffles", type=int, default=500)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = (args.output_dir or input_dir).resolve()
    qc_all = pd.read_csv(input_dir / "clip_preprocessing_qc.csv")
    if qc_all["qc_pass"].dtype != bool:
        qc_all["qc_pass"] = qc_all["qc_pass"].astype(str).str.lower().eq("true")
    keep = qc_all["qc_pass"].to_numpy(dtype=bool)
    qc = qc_all.loc[keep].reset_index(drop=True)
    archive = np.load(input_dir / "clip_level_video_features.npz")
    arrays = {
        name: archive[name][keep]
        for name in archive.files
        if name != "video_id"
    }
    labels = qc["label"].to_numpy(dtype=np.int64)
    split = qc["split"].astype(str).to_numpy()
    development = np.isin(split, ["train", "val"])
    test = split == "test"

    face_median = arrays["face_norm_median"]
    face_mad = arrays["face_norm_mad"]
    middle_mask = np.zeros((32, 32), dtype=bool)
    middle_mask[:, 11:21] = True
    metadata = qc[
        [
            "year",
            "source_width",
            "source_height",
            "fps",
            "duration_seconds",
            "file_size_bytes",
        ]
    ].to_numpy(dtype=np.float64)
    metadata[:, -1] = np.log1p(metadata[:, -1])
    video_index = qc["video_id"].str.replace("video", "", regex=False).astype(int)

    features = {
        "aligned_face_normalized": feature_for_arrays(face_median, face_mad),
        "aligned_face_static_only": face_median.reshape(len(qc), -1),
        "aligned_face_dynamics_only": face_mad.reshape(len(qc), -1),
        "aligned_middle_third": np.concatenate(
            [face_median[:, middle_mask], face_mad[:, middle_mask]], axis=1
        ),
        "whole_frame_context": feature_for_arrays(
            arrays["full_frame_median"], arrays["full_frame_mad"]
        ),
        "context_with_face_masked": feature_for_arrays(
            arrays["context_masked_median"], arrays["context_masked_mad"]
        ),
        "all_acquisition_metadata": metadata,
        "year_only": qc[["year"]].to_numpy(dtype=np.float64),
        "video_index_only": video_index.to_numpy(dtype=np.float64)[:, None],
    }

    rng = np.random.default_rng(MODEL_SEED + 5000)
    shuffle_rows: list[dict[str, Any]] = []
    test_probabilities: dict[str, np.ndarray] = {}
    prediction_rows: list[dict[str, Any]] = []
    for condition_number, (name, values) in enumerate(features.items(), start=1):
        point, probability = fit_holdout(values, labels, development, test)
        test_probabilities[name] = probability
        null = []
        for _ in range(args.training_label_shuffles):
            shuffled_labels = labels[development].copy()
            rng.shuffle(shuffled_labels)
            model = make_model()
            model.fit(values[development], shuffled_labels)
            null_probability = model.predict_proba(values[test])[:, 1]
            null.append(roc_auc_score(labels[test], null_probability))
        null_array = np.asarray(null)
        empirical_p = (
            1.0 + float(np.sum(null_array >= point["auroc"]))
        ) / (args.training_label_shuffles + 1.0)
        shuffle_rows.append(
            {
                "condition": name,
                "observed_holdout_auroc": point["auroc"],
                "observed_accuracy": point["accuracy"],
                "observed_balanced_accuracy": point["balanced_accuracy"],
                "training_label_shuffles": args.training_label_shuffles,
                "shuffled_auroc_mean": float(null_array.mean()),
                "shuffled_auroc_sd": float(null_array.std(ddof=1)),
                "shuffled_auroc_q975": float(np.quantile(null_array, 0.975)),
                "empirical_p_vs_training_label_shuffle": empirical_p,
            }
        )
        for video_id_value, label, value in zip(
            qc.loc[test, "video_id"], labels[test], probability, strict=True
        ):
            prediction_rows.append(
                {
                    "condition": name,
                    "video_id": video_id_value,
                    "label": int(label),
                    "probability": float(value),
                }
            )
        print(
            f"[{name}] observed={point['auroc']:.4f}; "
            f"shuffle q97.5={np.quantile(null_array, 0.975):.4f}"
        )

    comparisons = [
        ("aligned_face_normalized", "whole_frame_context"),
        ("aligned_face_normalized", "context_with_face_masked"),
        ("aligned_face_normalized", "all_acquisition_metadata"),
        ("aligned_face_normalized", "year_only"),
        ("aligned_middle_third", "aligned_face_normalized"),
        ("aligned_face_dynamics_only", "aligned_face_static_only"),
        ("year_only", "video_index_only"),
    ]
    contrast_rows = []
    for comparison_number, (first_name, second_name) in enumerate(
        comparisons, start=1
    ):
        difference, low, high = paired_bootstrap_difference(
            labels[test],
            test_probabilities[first_name],
            test_probabilities[second_name],
            args.bootstrap_repetitions,
            MODEL_SEED + 6000 + comparison_number,
        )
        contrast_rows.append(
            {
                "first_condition": first_name,
                "second_condition": second_name,
                "auroc_difference_first_minus_second": difference,
                "difference_ci_low": low,
                "difference_ci_high": high,
                "bootstrap_unit": "video_clip",
                "bootstrap_repetitions": args.bootstrap_repetitions,
            }
        )

    shuffle_frame = pd.DataFrame(shuffle_rows)
    contrast_frame = pd.DataFrame(contrast_rows)
    shuffle_frame.to_csv(output_dir / "training_label_shuffle_controls.csv", index=False)
    contrast_frame.to_csv(output_dir / "paired_test_auc_contrasts.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(
        output_dir / "leakage_control_test_predictions.csv", index=False
    )

    by_name = shuffle_frame.set_index("condition")
    verdict = {
        "analysis_unit": "video clip",
        "training_label_shuffles": args.training_label_shuffles,
        "aligned_face_observed_auroc": float(
            by_name.loc["aligned_face_normalized", "observed_holdout_auroc"]
        ),
        "aligned_face_shuffle_q975": float(
            by_name.loc["aligned_face_normalized", "shuffled_auroc_q975"]
        ),
        "year_only_observed_auroc": float(
            by_name.loc["year_only", "observed_holdout_auroc"]
        ),
        "year_only_shuffle_q975": float(
            by_name.loc["year_only", "shuffled_auroc_q975"]
        ),
        "video_index_only_observed_auroc": float(
            by_name.loc["video_index_only", "observed_holdout_auroc"]
        ),
        "interpretation_boundary": (
            "Training-label shuffles test pipeline leakage but cannot establish "
            "participant independence or remove collection confounding."
        ),
    }
    (output_dir / "leakage_and_contrast_verdict.json").write_text(
        json.dumps(verdict, indent=2), encoding="utf-8"
    )
    print(json.dumps(verdict, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

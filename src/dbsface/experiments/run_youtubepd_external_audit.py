"""Run the locked clip-level matched-support audit on preprocessed YouTubePD raw videos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
import pandas as pd
from scipy.fft import dctn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


_HERE = Path(__file__).resolve().parent
sys.path.append(str(_HERE.parent / "data"))
from build_coarse_roi_masks import build_masks


DEFAULT_INPUT = Path("outputs/youtubepd_external_audit")
MODEL_SEED = 20260709
TRANSLATED_SUPPORTS = 64
RANDOM_SUPPORTS = 64


def make_model() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("standardizer", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    penalty="l2",
                    C=1.0,
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=5000,
                    random_state=MODEL_SEED,
                ),
            ),
        ]
    )


def metric_summary(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predictions = (probabilities >= 0.5).astype(np.int64)
    return {
        "auroc": float(roc_auc_score(labels, probabilities)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
    }


def fit_holdout(
    features: np.ndarray,
    labels: np.ndarray,
    development: np.ndarray,
    test: np.ndarray,
) -> tuple[dict[str, float], np.ndarray]:
    model = make_model()
    model.fit(features[development], labels[development])
    probabilities = model.predict_proba(features[test])[:, 1]
    return metric_summary(labels[test], probabilities), probabilities


def bootstrap_auc_interval(
    labels: np.ndarray,
    probabilities: np.ndarray,
    repetitions: int,
    seed: int,
) -> tuple[float, float]:
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
        values.append(roc_auc_score(labels[sampled], probabilities[sampled]))
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def repeated_cv_auc(
    features: np.ndarray,
    labels: np.ndarray,
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[float, float]:
    values = []
    for train_index, test_index in cv_splits:
        model = make_model()
        model.fit(features[train_index], labels[train_index])
        probabilities = model.predict_proba(features[test_index])[:, 1]
        values.append(roc_auc_score(labels[test_index], probabilities))
    return float(np.mean(values)), float(np.std(values, ddof=1))


def feature_for_mask(
    median: np.ndarray, mad: np.ndarray, mask: np.ndarray, mode: str
) -> np.ndarray:
    if mode == "combined":
        return np.concatenate([median[:, mask], mad[:, mask]], axis=1)
    if mode == "static":
        return median[:, mask]
    if mode == "dynamics":
        return mad[:, mask]
    raise ValueError(mode)


def feature_for_arrays(median: np.ndarray, mad: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [median.reshape(len(median), -1), mad.reshape(len(mad), -1)], axis=1
    )


def mask_bounds(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows, columns = np.where(mask)
    return (
        int(rows.min()),
        int(columns.min()),
        int(rows.max() + 1),
        int(columns.max() + 1),
    )


def rectangle_mask(y0: int, x0: int, height: int, width: int) -> np.ndarray:
    mask = np.zeros((32, 32), dtype=bool)
    mask[y0 : y0 + height, x0 : x0 + width] = True
    return mask


def translated_masks(
    named_mask: np.ndarray, roi_index: int, count: int
) -> list[tuple[str, int, int, np.ndarray]]:
    y0, x0, y1, x1 = mask_bounds(named_mask)
    height, width = y1 - y0, x1 - x0
    candidates = [
        (translated_y, translated_x)
        for translated_y in range(33 - height)
        for translated_x in range(33 - width)
        if (translated_y, translated_x) != (y0, x0)
    ]
    if count > len(candidates):
        raise ValueError(f"Only {len(candidates)} translated placements are available")
    rng = np.random.default_rng(MODEL_SEED + roi_index)
    selected = rng.choice(len(candidates), size=count, replace=False)
    return [
        (
            f"translated_{number:03d}",
            candidates[int(candidate)][0],
            candidates[int(candidate)][1],
            rectangle_mask(
                candidates[int(candidate)][0],
                candidates[int(candidate)][1],
                height,
                width,
            ),
        )
        for number, candidate in enumerate(selected, start=1)
    ]


def random_scattered_masks(
    pixel_count: int, roi_index: int, count: int
) -> list[tuple[str, np.ndarray]]:
    rng = np.random.default_rng(MODEL_SEED + 1000 + roi_index)
    supports = []
    for number in range(1, count + 1):
        mask = np.zeros(32 * 32, dtype=bool)
        mask[rng.choice(mask.size, size=pixel_count, replace=False)] = True
        supports.append((f"random_{number:03d}", mask.reshape(32, 32)))
    return supports


def pooled_features(median: np.ndarray, mad: np.ndarray, size: int) -> np.ndarray:
    pooled_median = np.stack(
        [
            cv2.resize(item, (size, size), interpolation=cv2.INTER_AREA)
            for item in median
        ],
        axis=0,
    )
    pooled_mad = np.stack(
        [cv2.resize(item, (size, size), interpolation=cv2.INTER_AREA) for item in mad],
        axis=0,
    )
    return feature_for_arrays(pooled_median, pooled_mad)


def dct_features(median: np.ndarray, mad: np.ndarray, size: int) -> np.ndarray:
    median_dct = dctn(median, axes=(-2, -1), norm="ortho")[:, :size, :size]
    mad_dct = dctn(mad, axes=(-2, -1), norm="ortho")[:, :size, :size]
    return feature_for_arrays(median_dct, mad_dct)


def load_inputs(input_dir: Path) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    qc = pd.read_csv(input_dir / "clip_preprocessing_qc.csv")
    if qc["qc_pass"].dtype != bool:
        qc["qc_pass"] = qc["qc_pass"].astype(str).str.lower().eq("true")
    archive = np.load(input_dir / "clip_level_video_features.npz")
    arrays = {name: archive[name] for name in archive.files if name != "video_id"}
    ids = archive["video_id"].astype(str)
    if not np.array_equal(ids, qc["video_id"].astype(str).to_numpy()):
        raise ValueError("Feature archive and QC table video IDs do not align")
    return qc, arrays


def evaluate_control(
    name: str,
    features: np.ndarray,
    labels: np.ndarray,
    development: np.ndarray,
    test: np.ndarray,
    cv_splits: list[tuple[np.ndarray, np.ndarray]],
    bootstrap_repetitions: int,
    prediction_rows: list[dict[str, Any]],
    test_ids: np.ndarray,
) -> dict[str, Any]:
    metrics, probabilities = fit_holdout(features, labels, development, test)
    ci_low, ci_high = bootstrap_auc_interval(
        labels[test], probabilities, bootstrap_repetitions, MODEL_SEED + len(name)
    )
    cv_mean, cv_sd = repeated_cv_auc(features, labels, cv_splits)
    for video_id, label, probability in zip(
        test_ids, labels[test], probabilities, strict=True
    ):
        prediction_rows.append(
            {
                "control": name,
                "video_id": video_id,
                "label": int(label),
                "probability": float(probability),
            }
        )
    return {
        "control": name,
        **metrics,
        "auroc_ci_low": ci_low,
        "auroc_ci_high": ci_high,
        "repeated_cv_auroc_mean": cv_mean,
        "repeated_cv_auroc_sd": cv_sd,
        "n_development": int(development.sum()),
        "n_test": int(test.sum()),
        "n_features": int(features.shape[1]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--translated-supports", type=int, default=TRANSLATED_SUPPORTS)
    parser.add_argument("--random-supports", type=int, default=RANDOM_SUPPORTS)
    parser.add_argument("--cv-repeats", type=int, default=5)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = (args.output_dir or input_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    qc_all, arrays_all = load_inputs(input_dir)

    pass_rates = qc_all.groupby("label")["qc_pass"].mean().to_dict()
    overall_pass_rate = float(qc_all["qc_pass"].mean())
    class_pass_difference = abs(
        float(pass_rates.get(1, np.nan)) - float(pass_rates.get(0, np.nan))
    )
    prespecified_qc_gate = bool(
        overall_pass_rate >= 0.80 and class_pass_difference <= 0.10
    )

    keep = qc_all["qc_pass"].to_numpy(dtype=bool)
    qc = qc_all.loc[keep].reset_index(drop=True)
    arrays = {name: value[keep] for name, value in arrays_all.items()}
    labels = qc["label"].to_numpy(dtype=np.int64)
    split = qc["split"].astype(str).str.lower().to_numpy()
    development = np.isin(split, ["train", "val"])
    test = split == "test"
    if len(np.unique(labels[development])) != 2 or len(np.unique(labels[test])) != 2:
        raise ValueError("Both classes are required in development and test partitions")
    for name, value in arrays.items():
        if not np.isfinite(value).all():
            raise ValueError(f"Non-finite values remain in QC-passing feature array {name}")

    splitter = RepeatedStratifiedKFold(
        n_splits=5, n_repeats=args.cv_repeats, random_state=MODEL_SEED
    )
    cv_splits = list(splitter.split(np.zeros(len(labels)), labels))

    face_median = arrays["face_norm_median"]
    face_mad = arrays["face_norm_mad"]
    roi_names, roi_masks, roi_definitions = build_masks()
    roi_definitions.to_csv(output_dir / "youtubepd_roi_definitions.csv", index=False)

    support_rows: list[dict[str, Any]] = []
    roi_summary_rows: list[dict[str, Any]] = []
    named_prediction_rows: list[dict[str, Any]] = []

    for roi_index, (roi_name, named_mask) in enumerate(
        zip(roi_names, roi_masks, strict=True), start=1
    ):
        y0, x0, y1, x1 = mask_bounds(named_mask)
        height, width = y1 - y0, x1 - x0
        translations = translated_masks(
            named_mask, roi_index, args.translated_supports
        )
        random_masks = random_scattered_masks(
            int(named_mask.sum()), roi_index, args.random_supports
        )

        for mode in ("combined", "static", "dynamics"):
            named_features = feature_for_mask(face_median, face_mad, named_mask, mode)
            named_metrics, named_probabilities = fit_holdout(
                named_features, labels, development, test
            )
            named_cv_mean = np.nan
            named_cv_sd = np.nan
            if mode == "combined":
                named_cv_mean, named_cv_sd = repeated_cv_auc(
                    named_features, labels, cv_splits
                )
                for video_id, label, probability in zip(
                    qc.loc[test, "video_id"],
                    labels[test],
                    named_probabilities,
                    strict=True,
                ):
                    named_prediction_rows.append(
                        {
                            "roi_name": roi_name,
                            "video_id": video_id,
                            "label": int(label),
                            "probability": float(probability),
                        }
                    )
            support_rows.append(
                {
                    "roi_name": roi_name,
                    "mode": mode,
                    "support_kind": "named_roi",
                    "support_id": "named",
                    "y0": y0,
                    "x0": x0,
                    "height": height,
                    "width": width,
                    "n_pixels": int(named_mask.sum()),
                    "holdout_auroc": named_metrics["auroc"],
                    "holdout_accuracy": named_metrics["accuracy"],
                    "holdout_balanced_accuracy": named_metrics["balanced_accuracy"],
                    "repeated_cv_auroc_mean": named_cv_mean,
                    "repeated_cv_auroc_sd": named_cv_sd,
                }
            )

            translated_test = []
            translated_cv = []
            for support_id, translated_y, translated_x, translated_mask in translations:
                features = feature_for_mask(
                    face_median, face_mad, translated_mask, mode
                )
                metrics, _ = fit_holdout(features, labels, development, test)
                cv_mean = np.nan
                cv_sd = np.nan
                if mode == "combined":
                    cv_mean, cv_sd = repeated_cv_auc(features, labels, cv_splits)
                    translated_cv.append(cv_mean)
                translated_test.append(metrics["auroc"])
                support_rows.append(
                    {
                        "roi_name": roi_name,
                        "mode": mode,
                        "support_kind": "translated_rectangle",
                        "support_id": support_id,
                        "y0": translated_y,
                        "x0": translated_x,
                        "height": height,
                        "width": width,
                        "n_pixels": int(translated_mask.sum()),
                        "holdout_auroc": metrics["auroc"],
                        "holdout_accuracy": metrics["accuracy"],
                        "holdout_balanced_accuracy": metrics[
                            "balanced_accuracy"
                        ],
                        "repeated_cv_auroc_mean": cv_mean,
                        "repeated_cv_auroc_sd": cv_sd,
                    }
                )

            random_test = []
            if mode == "combined":
                for support_id, random_mask in random_masks:
                    features = feature_for_mask(
                        face_median, face_mad, random_mask, mode
                    )
                    metrics, _ = fit_holdout(features, labels, development, test)
                    random_test.append(metrics["auroc"])
                    support_rows.append(
                        {
                            "roi_name": roi_name,
                            "mode": mode,
                            "support_kind": "random_scattered_pixels",
                            "support_id": support_id,
                            "y0": np.nan,
                            "x0": np.nan,
                            "height": np.nan,
                            "width": np.nan,
                            "n_pixels": int(random_mask.sum()),
                            "holdout_auroc": metrics["auroc"],
                            "holdout_accuracy": metrics["accuracy"],
                            "holdout_balanced_accuracy": metrics[
                                "balanced_accuracy"
                            ],
                            "repeated_cv_auroc_mean": np.nan,
                            "repeated_cv_auroc_sd": np.nan,
                        }
                    )

            translated_test_array = np.asarray(translated_test)
            translated_q975 = float(np.quantile(translated_test_array, 0.975))
            translated_mean = float(translated_test_array.mean())
            cv_q975 = (
                float(np.quantile(translated_cv, 0.975))
                if translated_cv
                else np.nan
            )
            cv_margin = (
                float(named_cv_mean - cv_q975) if translated_cv else np.nan
            )
            roi_summary_rows.append(
                {
                    "roi_name": roi_name,
                    "mode": mode,
                    "named_holdout_auroc": named_metrics["auroc"],
                    "translated_holdout_mean": translated_mean,
                    "translated_holdout_q975": translated_q975,
                    "holdout_location_margin": float(
                        named_metrics["auroc"] - translated_q975
                    ),
                    "named_exceeds_holdout_q975": bool(
                        named_metrics["auroc"] > translated_q975
                    ),
                    "named_repeated_cv_auroc": float(named_cv_mean),
                    "translated_repeated_cv_q975": cv_q975,
                    "repeated_cv_location_margin": cv_margin,
                    "named_exceeds_repeated_cv_q975": bool(
                        translated_cv and named_cv_mean > cv_q975
                    ),
                    "passes_both_location_gates": bool(
                        mode == "combined"
                        and named_metrics["auroc"] > translated_q975
                        and translated_cv
                        and named_cv_mean > cv_q975
                    ),
                    "random_scattered_holdout_mean": float(np.mean(random_test))
                    if random_test
                    else np.nan,
                    "random_scattered_holdout_q975": float(
                        np.quantile(random_test, 0.975)
                    )
                    if random_test
                    else np.nan,
                }
            )
        print(f"[{roi_name}] complete")

    support_frame = pd.DataFrame(support_rows)
    roi_summary = pd.DataFrame(roi_summary_rows)
    support_frame.to_csv(output_dir / "youtubepd_roi_support_metrics.csv", index=False)
    roi_summary.to_csv(output_dir / "youtubepd_roi_summary.csv", index=False)
    pd.DataFrame(named_prediction_rows).to_csv(
        output_dir / "youtubepd_named_roi_test_predictions.csv", index=False
    )

    union = roi_masks.any(axis=0)
    border = np.zeros((32, 32), dtype=bool)
    border[[0, -1], :] = True
    border[:, [0, -1]] = True
    thirds = {
        "aligned_left_third": rectangle_mask(0, 0, 32, 11),
        "aligned_middle_third": rectangle_mask(0, 11, 32, 10),
        "aligned_right_third": rectangle_mask(0, 21, 32, 11),
    }

    metadata_features = qc[
        [
            "year",
            "source_width",
            "source_height",
            "fps",
            "duration_seconds",
            "file_size_bytes",
        ]
    ].to_numpy(dtype=np.float64)
    metadata_features[:, -1] = np.log1p(metadata_features[:, -1])

    controls: dict[str, np.ndarray] = {
        "aligned_face_normalized": feature_for_arrays(face_median, face_mad),
        "aligned_face_static_only": face_median.reshape(len(qc), -1),
        "aligned_face_dynamics_only": face_mad.reshape(len(qc), -1),
        "aligned_face_unnormalized": feature_for_arrays(
            arrays["face_raw_median"], arrays["face_raw_mad"]
        ),
        "outside_predefined_roi_union": feature_for_mask(
            face_median, face_mad, ~union, "combined"
        ),
        "aligned_one_pixel_border": feature_for_mask(
            face_median, face_mad, border, "combined"
        ),
        "whole_frame_context": feature_for_arrays(
            arrays["full_frame_median"], arrays["full_frame_mad"]
        ),
        "context_with_face_masked": feature_for_arrays(
            arrays["context_masked_median"], arrays["context_masked_mad"]
        ),
        "face_mask_geometry_only": feature_for_arrays(
            arrays["mask_geometry_median"], arrays["mask_geometry_mad"]
        ),
        "acquisition_metadata_only": metadata_features,
        "aligned_pool_8x8": pooled_features(face_median, face_mad, 8),
        "aligned_pool_4x4": pooled_features(face_median, face_mad, 4),
        "aligned_pool_2x2": pooled_features(face_median, face_mad, 2),
        "aligned_pool_1x1": pooled_features(face_median, face_mad, 1),
        "aligned_dct_8x8": dct_features(face_median, face_mad, 8),
        "aligned_dct_4x4": dct_features(face_median, face_mad, 4),
    }
    for name, mask in thirds.items():
        controls[name] = feature_for_mask(face_median, face_mad, mask, "combined")

    control_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for control_name, features in controls.items():
        row = evaluate_control(
            control_name,
            features,
            labels,
            development,
            test,
            cv_splits,
            args.bootstrap_repetitions,
            prediction_rows,
            qc.loc[test, "video_id"].astype(str).to_numpy(),
        )
        control_rows.append(row)
        print(f"[{control_name}] holdout AUROC={row['auroc']:.4f}")

    control_frame = pd.DataFrame(control_rows)
    control_frame.to_csv(output_dir / "youtubepd_control_metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(
        output_dir / "youtubepd_control_test_predictions.csv", index=False
    )

    combined_summary = roi_summary.loc[roi_summary["mode"] == "combined"]
    controls_by_name = control_frame.set_index("control")
    verdict = {
        "analysis_unit": "video clip",
        "subject_identifiers_available": False,
        "primary_cohort_clips_before_qc": int(len(qc_all)),
        "clips_after_qc": int(len(qc)),
        "overall_qc_pass_rate": overall_pass_rate,
        "pd_qc_pass_rate": float(pass_rates.get(1, np.nan)),
        "non_pd_qc_pass_rate": float(pass_rates.get(0, np.nan)),
        "class_qc_pass_rate_difference": class_pass_difference,
        "prespecified_qc_gate_passed": prespecified_qc_gate,
        "development_clips": int(development.sum()),
        "test_clips": int(test.sum()),
        "test_pd": int(labels[test].sum()),
        "test_non_pd": int(test.sum() - labels[test].sum()),
        "translated_supports_per_roi": args.translated_supports,
        "repeated_cv_folds": len(cv_splits),
        "named_rois_passing_holdout_location_gate": int(
            combined_summary["named_exceeds_holdout_q975"].sum()
        ),
        "named_rois_passing_both_location_gates": int(
            combined_summary["passes_both_location_gates"].sum()
        ),
        "aligned_face_holdout_auroc": float(
            controls_by_name.loc["aligned_face_normalized", "auroc"]
        ),
        "whole_frame_holdout_auroc": float(
            controls_by_name.loc["whole_frame_context", "auroc"]
        ),
        "face_masked_context_holdout_auroc": float(
            controls_by_name.loc["context_with_face_masked", "auroc"]
        ),
        "metadata_only_holdout_auroc": float(
            controls_by_name.loc["acquisition_metadata_only", "auroc"]
        ),
        "inference_boundary": (
            "All estimates are clip-level. YouTubePD does not provide verified "
            "participant identifiers, so this is not patient-level external validation."
        ),
    }
    (output_dir / "youtubepd_external_audit_verdict.json").write_text(
        json.dumps(verdict, indent=2), encoding="utf-8"
    )

    lines = [
        "# YouTubePD external matched-support audit",
        "",
        f"- Prespecified preprocessing QC gate: **{prespecified_qc_gate}**.",
        f"- QC-passing clips: **{len(qc)}/{len(qc_all)}**.",
        f"- Locked test clips: **{test.sum()}** ({labels[test].sum()} PD, "
        f"{test.sum() - labels[test].sum()} non-PD).",
        f"- Named ROIs exceeding the translated 97.5th percentile on the holdout: "
        f"**{verdict['named_rois_passing_holdout_location_gate']}/8**.",
        f"- Named ROIs also exceeding the repeated-CV translated 97.5th percentile: "
        f"**{verdict['named_rois_passing_both_location_gates']}/8**.",
        "",
        "| Control | Holdout AUROC | 95% clip bootstrap CI | Repeated-CV AUROC |",
        "| --- | ---: | ---: | ---: |",
    ]
    for _, row in control_frame.iterrows():
        lines.append(
            f"| {row['control']} | {row['auroc']:.4f} | "
            f"{row['auroc_ci_low']:.4f}-{row['auroc_ci_high']:.4f} | "
            f"{row['repeated_cv_auroc_mean']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Intervals and cross-validation are clip-level. They are not "
            "participant-level estimates.",
        ]
    )
    (output_dir / "youtubepd_external_audit_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(verdict, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

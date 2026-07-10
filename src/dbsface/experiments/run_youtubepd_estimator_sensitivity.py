"""Post hoc RBF-SVM sensitivity for the locked YouTubePD location audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


_HERE = Path(__file__).resolve().parent
sys.path.append(str(_HERE))
sys.path.append(str(_HERE.parent / "data"))
from build_coarse_roi_masks import build_masks
from run_youtubepd_external_audit import (
    MODEL_SEED,
    feature_for_arrays,
    feature_for_mask,
    mask_bounds,
    translated_masks,
)


DEFAULT_INPUT = Path("outputs/youtubepd_external_audit")


def make_model() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("standardizer", StandardScaler()),
            (
                "classifier",
                SVC(
                    C=1.0,
                    kernel="rbf",
                    gamma="scale",
                    class_weight="balanced",
                    probability=False,
                    random_state=MODEL_SEED,
                ),
            ),
        ]
    )


def holdout_auc(
    features: np.ndarray,
    labels: np.ndarray,
    development: np.ndarray,
    test: np.ndarray,
) -> float:
    model = make_model()
    model.fit(features[development], labels[development])
    score = model.decision_function(features[test])
    return float(roc_auc_score(labels[test], score))


def repeated_cv_auc(
    features: np.ndarray,
    labels: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[float, float]:
    values = []
    for train_index, test_index in splits:
        model = make_model()
        model.fit(features[train_index], labels[train_index])
        score = model.decision_function(features[test_index])
        values.append(roc_auc_score(labels[test_index], score))
    return float(np.mean(values)), float(np.std(values, ddof=1))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--translated-supports", type=int, default=64)
    parser.add_argument("--cv-repeats", type=int, default=5)
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = (args.output_dir or input_dir).resolve()
    qc_all = pd.read_csv(input_dir / "clip_preprocessing_qc.csv")
    if qc_all["qc_pass"].dtype != bool:
        qc_all["qc_pass"] = qc_all["qc_pass"].astype(str).str.lower().eq("true")
    keep = qc_all["qc_pass"].to_numpy(dtype=bool)
    qc = qc_all.loc[keep].reset_index(drop=True)
    archive = np.load(input_dir / "clip_level_video_features.npz")
    median = archive["face_norm_median"][keep]
    mad = archive["face_norm_mad"][keep]
    labels = qc["label"].to_numpy(dtype=np.int64)
    development = qc["split"].isin(["train", "val"]).to_numpy()
    test = qc["split"].eq("test").to_numpy()

    splitter = RepeatedStratifiedKFold(
        n_splits=5, n_repeats=args.cv_repeats, random_state=MODEL_SEED
    )
    splits = list(splitter.split(np.zeros(len(labels)), labels))
    roi_names, roi_masks, _ = build_masks()
    support_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for roi_index, (roi_name, named_mask) in enumerate(
        zip(roi_names, roi_masks, strict=True), start=1
    ):
        y0, x0, y1, x1 = mask_bounds(named_mask)
        height, width = y1 - y0, x1 - x0
        named_features = feature_for_mask(median, mad, named_mask, "combined")
        named_holdout = holdout_auc(named_features, labels, development, test)
        named_cv, named_cv_sd = repeated_cv_auc(named_features, labels, splits)
        support_rows.append(
            {
                "roi_name": roi_name,
                "support_kind": "named_roi",
                "support_id": "named",
                "y0": y0,
                "x0": x0,
                "height": height,
                "width": width,
                "holdout_auroc": named_holdout,
                "repeated_cv_auroc_mean": named_cv,
                "repeated_cv_auroc_sd": named_cv_sd,
            }
        )
        translated_holdout = []
        translated_cv = []
        for support_id, translated_y, translated_x, mask in translated_masks(
            named_mask, roi_index, args.translated_supports
        ):
            features = feature_for_mask(median, mad, mask, "combined")
            holdout = holdout_auc(features, labels, development, test)
            cv_mean, cv_sd = repeated_cv_auc(features, labels, splits)
            translated_holdout.append(holdout)
            translated_cv.append(cv_mean)
            support_rows.append(
                {
                    "roi_name": roi_name,
                    "support_kind": "translated_rectangle",
                    "support_id": support_id,
                    "y0": translated_y,
                    "x0": translated_x,
                    "height": height,
                    "width": width,
                    "holdout_auroc": holdout,
                    "repeated_cv_auroc_mean": cv_mean,
                    "repeated_cv_auroc_sd": cv_sd,
                }
            )
        holdout_q975 = float(np.quantile(translated_holdout, 0.975))
        cv_q975 = float(np.quantile(translated_cv, 0.975))
        summary_rows.append(
            {
                "roi_name": roi_name,
                "named_holdout_auroc": named_holdout,
                "translated_holdout_mean": float(np.mean(translated_holdout)),
                "translated_holdout_q975": holdout_q975,
                "holdout_location_margin": float(named_holdout - holdout_q975),
                "named_exceeds_holdout_q975": bool(named_holdout > holdout_q975),
                "named_repeated_cv_auroc": named_cv,
                "translated_repeated_cv_q975": cv_q975,
                "repeated_cv_location_margin": float(named_cv - cv_q975),
                "named_exceeds_repeated_cv_q975": bool(named_cv > cv_q975),
                "passes_both_location_gates": bool(
                    named_holdout > holdout_q975 and named_cv > cv_q975
                ),
            }
        )
        print(
            f"[{roi_name}] holdout={named_holdout:.4f}/{holdout_q975:.4f}; "
            f"cv={named_cv:.4f}/{cv_q975:.4f}"
        )

    full_features = feature_for_arrays(median, mad)
    full_holdout = holdout_auc(full_features, labels, development, test)
    full_cv, full_cv_sd = repeated_cv_auc(full_features, labels, splits)
    summary = pd.DataFrame(summary_rows)
    pd.DataFrame(support_rows).to_csv(
        output_dir / "youtubepd_rbf_svm_support_metrics.csv", index=False
    )
    summary.to_csv(output_dir / "youtubepd_rbf_svm_roi_summary.csv", index=False)
    verdict = {
        "analysis_type": "post hoc estimator sensitivity",
        "estimator": "standardised RBF SVM, C=1, gamma=scale, balanced class weights",
        "full_face_holdout_auroc": full_holdout,
        "full_face_repeated_cv_auroc_mean": full_cv,
        "full_face_repeated_cv_auroc_sd": full_cv_sd,
        "named_rois_passing_holdout_gate": int(
            summary["named_exceeds_holdout_q975"].sum()
        ),
        "named_rois_passing_both_gates": int(
            summary["passes_both_location_gates"].sum()
        ),
        "translated_supports_per_roi": args.translated_supports,
        "repeated_cv_folds": len(splits),
        "interpretation_boundary": (
            "This non-linear estimator sensitivity was specified after the "
            "primary logistic audit and does not replace its locked result."
        ),
    }
    (output_dir / "youtubepd_rbf_svm_verdict.json").write_text(
        json.dumps(verdict, indent=2), encoding="utf-8"
    )
    print(json.dumps(verdict, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

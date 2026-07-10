"""Build public, aggregate source-data tables for Figures 1-5."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-output-root", type=Path, required=True)
    parser.add_argument("--pd-output-root", type=Path, required=True)
    parser.add_argument("--fresh-output-root", type=Path)
    parser.add_argument("--reproduction-root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("source_data"))
    args = parser.parse_args()

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, str]] = []

    def save(
        filename: str,
        frame: pd.DataFrame,
        figure: str,
        panel: str,
        unit: str,
        description: str,
        source: str,
    ) -> None:
        path = out / filename
        frame.to_csv(path, index=False)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest.append(
            {
                "file": filename,
                "figure": figure,
                "panel": panel,
                "unit": unit,
                "description": description,
                "source": source,
                "sha256": digest,
            }
        )

    clean = args.clean_output_root.resolve()
    pd_root = args.pd_output_root.resolve()
    yt = clean / "youtubepd_external_audit"
    park = clean / "ufnet_participant_level_benchmark"

    synthetic = pd.read_csv(
        clean
        / "nmi_synthetic_calibration/synthetic_calibration_combined_summary.csv"
    )
    save(
        "fig2_synthetic_operating_curve.csv",
        synthetic,
        "2",
        "a-b",
        "simulation repetition",
        "Detection counts, rates and Wilson intervals by injected effect.",
        "clean-room synthetic combined verdict",
    )
    mechanism_json = read_json(
        clean / "nmi_audit_validation/synthetic_audit_verdict.json"
    )
    mechanism_rows = []
    for task, values in mechanism_json["tasks"].items():
        mechanism_rows.append({"task": task, **values})
    save(
        "fig2_mechanism_tasks.csv",
        pd.DataFrame(mechanism_rows),
        "2",
        "c-d",
        "synthetic task",
        "Known local, border, distributed and misplaced-support task metrics.",
        "clean-room mechanism verdict",
    )

    similarity = pd.read_csv(pd_root / "data_qc/similarity_threshold_sensitivity.csv")
    save(
        "fig3_similarity_filter.csv",
        similarity,
        "3",
        "a",
        "image",
        "Post hoc maximum-cosine exclusion sensitivity.",
        "governance-limited PD-DBS aggregate output",
    )
    full_image = pd.read_csv(
        pd_root / "data_qc/near_duplicate_sensitivity_metrics.csv"
    )
    full_image = full_image.loc[full_image["set"] == "full_test"].copy()
    save(
        "fig3_full_image_metrics.csv",
        full_image,
        "3",
        "a,d",
        "image",
        "Primary full-image performance used as the Figure 3 reference.",
        "governance-limited PD-DBS aggregate output",
    )
    pd_roi = pd.read_csv(
        pd_root / "nmi_spatially_matched_audit/spatially_matched_roi_summary.csv"
    )
    save(
        "fig3_pd_roi_summary.csv",
        pd_roi,
        "3",
        "b-c",
        "image",
        "Named-support and translated-support summary metrics.",
        "governance-limited PD-DBS aggregate output",
    )
    pd_supports = pd.read_csv(
        pd_root / "nmi_spatially_matched_audit/spatially_matched_support_metrics.csv"
    )
    save(
        "fig3_pd_translated_supports.csv",
        pd_supports,
        "3",
        "b",
        "image",
        "All 64 translated-support AUROCs per named rectangle.",
        "governance-limited PD-DBS aggregate output",
    )
    random_budget = pd.read_csv(
        pd_root / "hard_negative_random50/random_budget_curve.csv"
    )
    save(
        "fig3_random_pixel_budgets.csv",
        random_budget,
        "3",
        "d",
        "image",
        "Fifty random supports per pixel budget.",
        "governance-limited PD-DBS aggregate output",
    )
    fixed = pd.read_csv(pd_root / "nmi_exact_matched_audit/fixed_support_metrics.csv")
    pixel = pd.read_csv(
        pd_root / "hard_negative_random50/pixel_localisation_metrics.csv"
    )
    pooling = pd.read_csv(
        pd_root / "hard_negative_random50/pooling_ladder_metrics.csv"
    )
    control_rows = []
    for label, display in [
        ("outside_predefined_roi_union", "Outside ROI union"),
        ("border_width_1", "One-pixel border"),
    ]:
        row = fixed.loc[fixed["label"] == label].iloc[0]
        control_rows.append(
            {"control": display, "group": "spatial control", "auroc": row["auroc"]}
        )
    for _, row in pixel.loc[pixel["kind"] == "third"].iterrows():
        control_rows.append(
            {
                "control": row["label"].replace("random_third_", "Random third "),
                "group": "spatial control",
                "auroc": row["auroc"],
            }
        )
    for grid, display in [("8x8", "Pooling 8 x 8"), ("1x1", "Global mean")]:
        row = pooling.loc[pooling["pool_grid"] == grid].iloc[0]
        control_rows.append(
            {"control": display, "group": "coarse representation", "auroc": row["auroc"]}
        )
    save(
        "fig3_distributed_controls.csv",
        pd.DataFrame(control_rows),
        "3",
        "d",
        "image",
        "Outside-support, border, partition and pooling controls.",
        "governance-limited PD-DBS aggregate output",
    )
    detector_conditions = pd.read_csv(
        pd_root / "nmi_detector_geometry_audit/detector_condition_summary.csv"
    )
    detector_shuffle = pd.read_csv(
        pd_root / "nmi_detector_geometry_audit/detector_mask_shuffle_summary.csv"
    )
    save(
        "fig3_detector_conditions.csv",
        detector_conditions,
        "3",
        "e",
        "image",
        "Detector geometry and exterior-support conditions.",
        "governance-limited PD-DBS aggregate output",
    )
    save(
        "fig3_detector_shuffle.csv",
        detector_shuffle,
        "3",
        "e",
        "permutation",
        "Mask-reassignment empirical distributions.",
        "governance-limited PD-DBS aggregate output",
    )

    yt_verdict = read_json(yt / "youtubepd_external_audit_verdict.json")
    flow = pd.DataFrame(
        [
            {"stage": "Valid source records", "n": 282},
            {"stage": "Locally reconstructed", "n": 248},
            {
                "stage": "Balanced cohort",
                "n": yt_verdict["primary_cohort_clips_before_qc"],
            },
            {"stage": "QC pass", "n": yt_verdict["clips_after_qc"]},
            {"stage": "Spreadsheet test", "n": yt_verdict["test_clips"]},
        ]
    )
    save(
        "fig4_cohort_flow.csv",
        flow,
        "4",
        "a",
        "video clip",
        "Locked reconstruction and QC flow.",
        "clean-room frozen-input verdict plus locked protocol",
    )
    save(
        "fig4_youtubepd_roi_summary.csv",
        pd.read_csv(yt / "youtubepd_roi_summary.csv"),
        "4",
        "b",
        "video clip",
        "Primary logistic named-support location metrics.",
        "clean-room frozen-input output",
    )
    save(
        "fig4_youtubepd_translated_supports.csv",
        pd.read_csv(yt / "youtubepd_roi_support_metrics.csv"),
        "4",
        "b",
        "video clip",
        "All primary translated-support holdout and repeated-CV metrics.",
        "clean-room frozen-input output",
    )
    save(
        "fig4_rbf_roi_summary.csv",
        pd.read_csv(yt / "youtubepd_rbf_svm_roi_summary.csv"),
        "4",
        "c",
        "video clip",
        "Post hoc RBF-SVM estimator sensitivity.",
        "clean-room frozen-input output",
    )
    controls = pd.read_csv(yt / "youtubepd_control_metrics.csv")
    selected_controls = controls.loc[
        controls["control"].isin(
            [
                "aligned_face_normalized",
                "whole_frame_context",
                "context_with_face_masked",
                "acquisition_metadata_only",
                "aligned_middle_third",
            ]
        )
    ].copy()
    save(
        "fig4_competing_supports.csv",
        selected_controls,
        "4",
        "d",
        "video clip",
        "Aligned-face, frame, context, metadata and arbitrary-partition controls.",
        "clean-room frozen-input output",
    )
    save(
        "fig4_year_distribution.csv",
        pd.read_csv(yt / "acquisition_year_distribution.csv"),
        "4",
        "e",
        "video clip",
        "Source-year counts by numerical diagnosis label.",
        "clean-room frozen-input output",
    )
    year_matched = pd.read_csv(yt / "year_matched_control_metrics.csv")
    save(
        "fig4_year_matched_controls.csv",
        year_matched,
        "4",
        "e",
        "matched video pair",
        "Exploratory three-year-caliper grouped-CV controls.",
        "clean-room frozen-input output",
    )
    confound_verdict = read_json(yt / "acquisition_confound_verdict.json")
    matched_year = year_matched.loc[year_matched["condition"] == "year_only"].iloc[0]
    save(
        "fig4_collection_time_summary.csv",
        pd.DataFrame(
            [
                {
                    "analysis": "year_only_spreadsheet_holdout",
                    "auroc": confound_verdict["year_only_holdout_auroc"],
                    "auroc_ci_low": confound_verdict["year_only_holdout_ci_low"],
                    "auroc_ci_high": confound_verdict["year_only_holdout_ci_high"],
                    "matched_pairs": np.nan,
                    "grouped_cv_folds": np.nan,
                },
                {
                    "analysis": "year_only_three_year_caliper_grouped_cv",
                    "auroc": confound_verdict["matched_year_grouped_cv_auroc"],
                    "auroc_ci_low": np.nan,
                    "auroc_ci_high": np.nan,
                    "matched_pairs": confound_verdict["matched_pairs"],
                    "grouped_cv_folds": matched_year["grouped_cv_folds"],
                },
            ]
        ),
        "4",
        "e",
        "video clip or matched video pair",
        "Year-only holdout and exploratory matched grouped-CV summaries.",
        "clean-room frozen-input verdict",
    )

    park_verdict = read_json(park / "participant_benchmark_verdict.json")
    park_summary = pd.DataFrame(
        [
            {
                "condition": "Observed all features",
                "auroc": park_verdict["participant_auroc"],
                "low": park_verdict["participant_auroc_ci_low"],
                "high": park_verdict["participant_auroc_ci_high"],
                "n_test_participants": park_verdict["test_participants"],
                "n_test_class_1": park_verdict["test_pd"],
                "n_test_class_0": park_verdict["test_non_pd"],
            },
            {
                "condition": "Training-label shuffle mean",
                "auroc": park_verdict["training_label_shuffle_auroc_mean"],
                "low": np.nan,
                "high": park_verdict["training_label_shuffle_auroc_q975"],
                "n_test_participants": park_verdict["test_participants"],
                "n_test_class_1": park_verdict["test_pd"],
                "n_test_class_0": park_verdict["test_non_pd"],
            },
        ]
    )
    save(
        "fig5_park_benchmark.csv",
        park_summary,
        "5",
        "a",
        "participant",
        "Observed participant AUROC and training-label-shuffle reference.",
        "clean-room PARK verdict",
    )
    save(
        "fig5_park_feature_groups.csv",
        pd.read_csv(park / "participant_feature_group_metrics.csv"),
        "5",
        "b",
        "participant",
        "Predefined released feature-family sensitivities.",
        "clean-room PARK output",
    )
    evidence = pd.DataFrame(
        [
            {
                "resource": "PD-DBS",
                "unit": "image",
                "participant_key": "no",
                "raw_pixels": "yes, restricted",
                "location_test": "yes",
                "shortcut_controls": "yes",
                "publicly_executable": "no",
                "maximum_claim": "image-level stress test",
            },
            {
                "resource": "YouTubePD",
                "unit": "clip",
                "participant_key": "no",
                "raw_pixels": "URLs only",
                "location_test": "yes",
                "shortcut_controls": "yes",
                "publicly_executable": "source-dependent",
                "maximum_claim": "confounded clip-level audit",
            },
            {
                "resource": "PARK",
                "unit": "participant",
                "participant_key": "yes",
                "raw_pixels": "no",
                "location_test": "no",
                "shortcut_controls": "label shuffle",
                "publicly_executable": "yes",
                "maximum_claim": "feature association",
            },
        ]
    )
    save(
        "fig5_evidence_matrix.csv",
        evidence,
        "5",
        "c-d",
        "resource",
        "Experimental-unit and access boundaries across resources.",
        "protocol and availability audit",
    )

    if args.fresh_output_root:
        fresh = args.fresh_output_root.resolve() / "youtubepd_external_audit"
        frozen_main = yt_verdict
        fresh_main = read_json(fresh / "youtubepd_external_audit_verdict.json")
        fresh_rbf = read_json(fresh / "youtubepd_rbf_svm_verdict.json")
        frozen_rbf = read_json(yt / "youtubepd_rbf_svm_verdict.json")
        drift = pd.DataFrame(
            [
                {
                    "reconstruction": "frozen_20260709",
                    "cohort_before_qc": frozen_main["primary_cohort_clips_before_qc"],
                    "after_qc": frozen_main["clips_after_qc"],
                    "test_clips": frozen_main["test_clips"],
                    "aligned_face_auroc": frozen_main["aligned_face_holdout_auroc"],
                    "primary_holdout_gates": frozen_main["named_rois_passing_holdout_location_gate"],
                    "primary_both_gates": frozen_main["named_rois_passing_both_location_gates"],
                    "rbf_holdout_gates": frozen_rbf["named_rois_passing_holdout_gate"],
                    "rbf_both_gates": frozen_rbf["named_rois_passing_both_gates"],
                },
                {
                    "reconstruction": "fresh_20260710",
                    "cohort_before_qc": fresh_main["primary_cohort_clips_before_qc"],
                    "after_qc": fresh_main["clips_after_qc"],
                    "test_clips": fresh_main["test_clips"],
                    "aligned_face_auroc": fresh_main["aligned_face_holdout_auroc"],
                    "primary_holdout_gates": fresh_main["named_rois_passing_holdout_location_gate"],
                    "primary_both_gates": fresh_main["named_rois_passing_both_location_gates"],
                    "rbf_holdout_gates": fresh_rbf["named_rois_passing_holdout_gate"],
                    "rbf_both_gates": fresh_rbf["named_rois_passing_both_gates"],
                },
            ]
        )
        save(
            "extended_youtubepd_reconstruction_drift.csv",
            drift,
            "Extended Data",
            "reconstruction drift",
            "video clip",
            "Frozen-input versus fresh-source reconstruction comparison.",
            "clean-room reproduction audit",
        )

    if args.reproduction_root:
        common_path = args.reproduction_root / "common_cohort_comparison.csv"
        if common_path.exists():
            save(
                "extended_youtubepd_common_cohort.csv",
                pd.read_csv(common_path),
                "Extended Data",
                "matched-ID drift",
                "video clip",
                "Same-ID frozen/fresh feature comparison.",
                "clean-room reconstruction-drift audit",
            )

    manifest_frame = pd.DataFrame(manifest)
    manifest_frame.to_csv(out / "source_data_manifest.csv", index=False)
    print(f"Wrote {len(manifest)} source-data tables to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

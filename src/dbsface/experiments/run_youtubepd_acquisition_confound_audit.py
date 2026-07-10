"""Post hoc diagnostic audit of acquisition-time confounding in YouTubePD.

This analysis was motivated by the protocol-specified metadata-only control. It is
explicitly exploratory: it decomposes that control and tests whether image
performance persists in a year-matched clip subset.  It does not replace the
locked spreadsheet holdout analysis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedGroupKFold


_HERE = Path(__file__).resolve().parent
sys.path.append(str(_HERE))
from run_youtubepd_external_audit import (
    MODEL_SEED,
    bootstrap_auc_interval,
    feature_for_arrays,
    fit_holdout,
    make_model,
    repeated_cv_auc,
)


DEFAULT_INPUT = Path("outputs/youtubepd_external_audit")


def matched_pairs(frame: pd.DataFrame, caliper: int) -> list[tuple[int, int, int]]:
    graph = nx.Graph()
    positive = frame.index[frame["label"] == 1].tolist()
    negative = frame.index[frame["label"] == 0].tolist()
    positive_nodes = [("positive", index) for index in positive]
    negative_nodes = [("negative", index) for index in negative]
    graph.add_nodes_from(positive_nodes, bipartite=0)
    graph.add_nodes_from(negative_nodes, bipartite=1)
    for positive_index in positive:
        for negative_index in negative:
            distance = abs(
                int(frame.loc[positive_index, "year"])
                - int(frame.loc[negative_index, "year"])
            )
            if distance <= caliper:
                graph.add_edge(
                    ("positive", positive_index),
                    ("negative", negative_index),
                    weight=1000 - distance,
                )
    matching = nx.algorithms.matching.max_weight_matching(
        graph, maxcardinality=True, weight="weight"
    )
    pairs = []
    for first, second in matching:
        if first[0] == "positive":
            positive_node, negative_node = first, second
        else:
            positive_node, negative_node = second, first
        positive_index = int(positive_node[1])
        negative_index = int(negative_node[1])
        distance = abs(
            int(frame.loc[positive_index, "year"])
            - int(frame.loc[negative_index, "year"])
        )
        pairs.append((positive_index, negative_index, distance))
    return sorted(pairs)


def repeated_group_cv_auc(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    repeats: int,
) -> tuple[float, float, int]:
    values = []
    for repeat in range(repeats):
        splitter = StratifiedGroupKFold(
            n_splits=5, shuffle=True, random_state=MODEL_SEED + repeat
        )
        for train_index, test_index in splitter.split(features, labels, groups):
            model = make_model()
            model.fit(features[train_index], labels[train_index])
            probability = model.predict_proba(features[test_index])[:, 1]
            values.append(roc_auc_score(labels[test_index], probability))
    return float(np.mean(values)), float(np.std(values, ddof=1)), len(values)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--match-caliper-years", type=int, default=3)
    parser.add_argument("--matched-cv-repeats", type=int, default=10)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
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
    cv_splitter = RepeatedStratifiedKFold(
        n_splits=5, n_repeats=5, random_state=MODEL_SEED
    )
    cv_splits = list(cv_splitter.split(np.zeros(len(qc)), labels))

    metadata_columns = [
        "year",
        "source_width",
        "source_height",
        "fps",
        "duration_seconds",
        "file_size_bytes",
    ]
    decomposition_rows: list[dict[str, Any]] = []
    for number, column in enumerate(metadata_columns, start=1):
        features = qc[[column]].to_numpy(dtype=np.float64)
        if column == "file_size_bytes":
            features = np.log1p(features)
        point, probability = fit_holdout(features, labels, development, test)
        ci_low, ci_high = bootstrap_auc_interval(
            labels[test],
            probability,
            args.bootstrap_repetitions,
            MODEL_SEED + number,
        )
        cv_mean, cv_sd = repeated_cv_auc(features, labels, cv_splits)
        decomposition_rows.append(
            {
                "metadata_feature": column,
                **point,
                "auroc_ci_low": ci_low,
                "auroc_ci_high": ci_high,
                "repeated_cv_auroc_mean": cv_mean,
                "repeated_cv_auroc_sd": cv_sd,
            }
        )
    decomposition = pd.DataFrame(decomposition_rows)
    decomposition.to_csv(
        output_dir / "acquisition_metadata_decomposition.csv", index=False
    )

    year_distribution = qc.groupby(["label", "year"]).size().rename("clips").reset_index()
    year_distribution.to_csv(output_dir / "acquisition_year_distribution.csv", index=False)

    caliper_rows = []
    for caliper in (0, 1, 2, 3, 5, 10):
        pairs = matched_pairs(qc, caliper)
        distances = [pair[2] for pair in pairs]
        caliper_rows.append(
            {
                "caliper_years": caliper,
                "matched_pairs": len(pairs),
                "matched_clips": 2 * len(pairs),
                "mean_year_difference": float(np.mean(distances))
                if distances
                else np.nan,
                "max_year_difference": int(max(distances)) if distances else np.nan,
            }
        )
    pd.DataFrame(caliper_rows).to_csv(
        output_dir / "year_matching_caliper_counts.csv", index=False
    )

    pairs = matched_pairs(qc, args.match_caliper_years)
    if len(pairs) < 10:
        raise RuntimeError("Too few matched pairs for five-fold grouped sensitivity")
    matched_indices = []
    pair_groups = []
    pair_rows = []
    for pair_number, (positive_index, negative_index, distance) in enumerate(pairs):
        for index in (positive_index, negative_index):
            matched_indices.append(index)
            pair_groups.append(pair_number)
        pair_rows.append(
            {
                "pair_id": pair_number,
                "pd_video_id": qc.loc[positive_index, "video_id"],
                "pd_year": int(qc.loc[positive_index, "year"]),
                "non_pd_video_id": qc.loc[negative_index, "video_id"],
                "non_pd_year": int(qc.loc[negative_index, "year"]),
                "absolute_year_difference": distance,
            }
        )
    matched_indices_array = np.asarray(matched_indices, dtype=np.int64)
    matched_labels = labels[matched_indices_array]
    groups = np.asarray(pair_groups, dtype=np.int64)
    pd.DataFrame(pair_rows).to_csv(output_dir / "year_matched_pairs.csv", index=False)

    all_metadata = qc[metadata_columns].to_numpy(dtype=np.float64)
    all_metadata[:, -1] = np.log1p(all_metadata[:, -1])
    face_median = arrays["face_norm_median"]
    face_mad = arrays["face_norm_mad"]
    middle_mask = np.zeros((32, 32), dtype=bool)
    middle_mask[:, 11:21] = True
    matched_features = {
        "year_only": qc[["year"]].to_numpy(dtype=np.float64),
        "all_acquisition_metadata": all_metadata,
        "aligned_face_normalized": feature_for_arrays(face_median, face_mad),
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
    }
    matched_rows = []
    for name, features in matched_features.items():
        mean_auc, sd_auc, folds = repeated_group_cv_auc(
            features[matched_indices_array],
            matched_labels,
            groups,
            args.matched_cv_repeats,
        )
        matched_rows.append(
            {
                "condition": name,
                "caliper_years": args.match_caliper_years,
                "matched_pairs": len(pairs),
                "matched_clips": 2 * len(pairs),
                "grouped_cv_folds": folds,
                "grouped_cv_auroc_mean": mean_auc,
                "grouped_cv_auroc_sd": sd_auc,
                "n_features": int(features.shape[1]),
            }
        )
    matched_frame = pd.DataFrame(matched_rows)
    matched_frame.to_csv(output_dir / "year_matched_control_metrics.csv", index=False)

    year_row = decomposition.set_index("metadata_feature").loc["year"]
    matched_by_name = matched_frame.set_index("condition")
    verdict = {
        "exploratory_post_hoc": True,
        "trigger": "prespecified acquisition-metadata control",
        "non_pd_median_year": float(qc.loc[qc["label"] == 0, "year"].median()),
        "pd_median_year": float(qc.loc[qc["label"] == 1, "year"].median()),
        "year_only_holdout_auroc": float(year_row["auroc"]),
        "year_only_holdout_ci_low": float(year_row["auroc_ci_low"]),
        "year_only_holdout_ci_high": float(year_row["auroc_ci_high"]),
        "match_caliper_years": args.match_caliper_years,
        "matched_pairs": len(pairs),
        "matched_face_grouped_cv_auroc": float(
            matched_by_name.loc["aligned_face_normalized", "grouped_cv_auroc_mean"]
        ),
        "matched_year_grouped_cv_auroc": float(
            matched_by_name.loc["year_only", "grouped_cv_auroc_mean"]
        ),
        "interpretation_boundary": (
            "The matched analysis is an exploratory clip-level sensitivity "
            "analysis and does not create participant-level independence."
        ),
    }
    (output_dir / "acquisition_confound_verdict.json").write_text(
        json.dumps(verdict, indent=2), encoding="utf-8"
    )

    lines = [
        "# YouTubePD acquisition-time confound audit",
        "",
        "This diagnostic was triggered by the prespecified metadata-only control "
        "and is explicitly post hoc.",
        "",
        f"- Median source year: non-PD **{verdict['non_pd_median_year']:.1f}**, "
        f"PD **{verdict['pd_median_year']:.1f}**.",
        f"- Year-only locked-test AUROC: **{verdict['year_only_holdout_auroc']:.4f}** "
        f"(95% clip bootstrap CI {verdict['year_only_holdout_ci_low']:.4f}-"
        f"{verdict['year_only_holdout_ci_high']:.4f}).",
        f"- Year-matched sensitivity: **{len(pairs)} pairs** within "
        f"+/-{args.match_caliper_years} years.",
        f"- Matched grouped-CV aligned-face AUROC: "
        f"**{verdict['matched_face_grouped_cv_auroc']:.4f}**.",
        f"- Matched grouped-CV year-only AUROC: "
        f"**{verdict['matched_year_grouped_cv_auroc']:.4f}**.",
        "",
        "All estimates remain clip-level because participant identifiers are unavailable.",
    ]
    (output_dir / "acquisition_confound_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(verdict, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

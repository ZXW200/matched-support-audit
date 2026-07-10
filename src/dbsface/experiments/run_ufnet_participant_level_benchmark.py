"""Participant-level UFNet/PARK feature benchmark with locked uncertainty.

The public release contains repeated feature rows for some participants.  This
analysis aggregates features before model fitting, preserves the official
participant split, excludes identifiers and demographics from predictors, and
uses participants as the unit for every test statistic and interval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_DATA = Path("external") / "UFNet" / "data"
DEFAULT_OUTPUT = Path("outputs/external/ufnet_participant_level_benchmark")
SEED = 20260709


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_id_set(path: Path) -> set[str]:
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


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
                    random_state=SEED,
                ),
            ),
        ]
    )


def metrics(labels: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    prediction = (probability >= 0.5).astype(np.int64)
    return {
        "auroc": float(roc_auc_score(labels, probability)),
        "auprc": float(average_precision_score(labels, probability)),
        "accuracy": float(accuracy_score(labels, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "brier_score": float(brier_score_loss(labels, probability)),
    }


def stratified_bootstrap(
    labels: np.ndarray,
    probability: np.ndarray,
    repetitions: int,
    seed: int,
) -> dict[str, tuple[float, float]]:
    negative = np.flatnonzero(labels == 0)
    positive = np.flatnonzero(labels == 1)
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {
        "auroc": [],
        "auprc": [],
        "accuracy": [],
        "balanced_accuracy": [],
        "brier_score": [],
    }
    for _ in range(repetitions):
        sampled = np.concatenate(
            [
                rng.choice(negative, len(negative), replace=True),
                rng.choice(positive, len(positive), replace=True),
            ]
        )
        row = metrics(labels[sampled], probability[sampled])
        for name, value in row.items():
            values[name].append(value)
    return {
        name: tuple(float(value) for value in np.quantile(samples, [0.025, 0.975]))
        for name, samples in values.items()
    }


def permutation_p_value(
    labels: np.ndarray, probability: np.ndarray, repetitions: int, seed: int
) -> tuple[float, float, float]:
    observed = float(roc_auc_score(labels, probability))
    rng = np.random.default_rng(seed)
    null = np.asarray(
        [roc_auc_score(rng.permutation(labels), probability) for _ in range(repetitions)]
    )
    p_value = (1.0 + float(np.sum(null >= observed))) / (repetitions + 1.0)
    return p_value, float(null.mean()), float(np.quantile(null, 0.975))


def aggregate_participants(
    rows: pd.DataFrame, feature_columns: list[str]
) -> pd.DataFrame:
    inconsistent = rows.groupby("Participant_ID")["Diagnosis"].nunique()
    if int((inconsistent > 1).sum()) > 0:
        raise ValueError("Diagnosis is not constant within participant")
    aggregated = rows.groupby("Participant_ID", as_index=False).agg(
        Diagnosis=("Diagnosis", "first"),
        n_rows=("Diagnosis", "size"),
        **{column: (column, "mean") for column in feature_columns},
    )
    return aggregated


def feature_groups(feature_columns: list[str]) -> dict[str, list[str]]:
    upper_tokens = ("AU01", "AU06", "AU45", "eye-open", "eye-raise")
    lower_tokens = ("AU12", "AU14", "AU25", "AU26", "mouth", "jaw")
    groups = {
        "all_features": feature_columns,
        "mean_statistics": [c for c in feature_columns if c.endswith("_mean")],
        "variance_statistics": [c for c in feature_columns if c.endswith("_var")],
        "entropy_statistics": [c for c in feature_columns if c.endswith("_entropy")],
        "action_unit_features": [c for c in feature_columns if c.startswith("smile_AU")],
        "geometric_features": [
            c for c in feature_columns if not c.startswith("smile_AU")
        ],
        "upper_face_features": [
            c for c in feature_columns if any(token in c for token in upper_tokens)
        ],
        "lower_face_features": [
            c for c in feature_columns if any(token in c for token in lower_tokens)
        ],
    }
    if any(not columns for columns in groups.values()):
        raise ValueError("At least one predefined feature group is empty")
    return groups


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--permutation-repetitions", type=int, default=10000)
    parser.add_argument("--training-label-shuffles", type=int, default=200)
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / "facial_expression_smile" / "facial_dataset.csv"
    dev_path = data_dir / "dev_set_participants.txt"
    test_path = data_dir / "test_set_participants.txt"

    dev_ids = read_id_set(dev_path)
    test_ids = read_id_set(test_path)
    if dev_ids & test_ids:
        raise ValueError("Official dev and test participant sets overlap")

    rows = pd.read_csv(csv_path)
    feature_columns = [column for column in rows.columns if column.startswith("smile_")]
    rows = rows[["Participant_ID", "Diagnosis", *feature_columns]].copy()
    rows = rows.dropna(subset=["Participant_ID", "Diagnosis"])
    rows["Participant_ID"] = rows["Participant_ID"].astype(str)
    rows["Diagnosis"] = rows["Diagnosis"].astype(np.int64)
    rows["split"] = "train"
    rows.loc[rows["Participant_ID"].isin(dev_ids), "split"] = "dev"
    rows.loc[rows["Participant_ID"].isin(test_ids), "split"] = "test"

    participants = aggregate_participants(rows, feature_columns)
    participant_split = rows.groupby("Participant_ID")["split"].first()
    participants["split"] = participants["Participant_ID"].map(participant_split)
    if participants["split"].isna().any():
        raise ValueError("Participant split mapping failed")

    train = participants["split"].isin(["train", "dev"]).to_numpy()
    test = participants["split"].eq("test").to_numpy()
    labels = participants["Diagnosis"].to_numpy(dtype=np.int64)
    if set(labels[test]) != {0, 1}:
        raise ValueError("Both test classes are required")

    group_rows: list[dict[str, Any]] = []
    all_feature_probability: np.ndarray | None = None
    for group_number, (group_name, columns) in enumerate(
        feature_groups(feature_columns).items(), start=1
    ):
        features = participants[columns].to_numpy(dtype=np.float64)
        model = make_model()
        model.fit(features[train], labels[train])
        probability = model.predict_proba(features[test])[:, 1]
        point = metrics(labels[test], probability)
        intervals = stratified_bootstrap(
            labels[test],
            probability,
            args.bootstrap_repetitions,
            SEED + group_number,
        )
        p_value, null_mean, null_q975 = permutation_p_value(
            labels[test],
            probability,
            args.permutation_repetitions,
            SEED + 100 + group_number,
        )
        group_rows.append(
            {
                "feature_group": group_name,
                "n_features": len(columns),
                **point,
                **{
                    f"{name}_ci_low": interval[0]
                    for name, interval in intervals.items()
                },
                **{
                    f"{name}_ci_high": interval[1]
                    for name, interval in intervals.items()
                },
                "permutation_p_auroc": p_value,
                "permutation_null_auroc_mean": null_mean,
                "permutation_null_auroc_q975": null_q975,
            }
        )
        if group_name == "all_features":
            all_feature_probability = probability
        print(
            f"[{group_name}] participant AUROC={point['auroc']:.4f}; "
            f"CI={intervals['auroc'][0]:.4f}-{intervals['auroc'][1]:.4f}"
        )

    if all_feature_probability is None:
        raise RuntimeError("All-feature model was not evaluated")

    all_features = participants[feature_columns].to_numpy(dtype=np.float64)
    rng = np.random.default_rng(SEED + 1000)
    shuffled_aurocs = []
    for _ in range(args.training_label_shuffles):
        shuffled_labels = labels[train].copy()
        rng.shuffle(shuffled_labels)
        model = make_model()
        model.fit(all_features[train], shuffled_labels)
        probability = model.predict_proba(all_features[test])[:, 1]
        shuffled_aurocs.append(roc_auc_score(labels[test], probability))

    metrics_frame = pd.DataFrame(group_rows)
    metrics_frame.to_csv(output_dir / "participant_feature_group_metrics.csv", index=False)
    prediction_frame = participants.loc[
        test, ["Participant_ID", "Diagnosis", "n_rows"]
    ].copy()
    prediction_frame["probability"] = all_feature_probability
    prediction_frame["prediction"] = (all_feature_probability >= 0.5).astype(int)
    prediction_frame.to_csv(
        output_dir / "all_features_test_participant_predictions.csv", index=False
    )
    participants.groupby(["split", "Diagnosis"]).agg(
        participants=("Participant_ID", "nunique"), rows=("n_rows", "sum")
    ).reset_index().to_csv(output_dir / "participant_split_counts.csv", index=False)

    primary = metrics_frame.set_index("feature_group").loc["all_features"]
    summary = {
        "analysis_unit": "participant",
        "source_csv": str(csv_path),
        "source_csv_sha256": sha256(csv_path),
        "dev_ids_sha256": sha256(dev_path),
        "test_ids_sha256": sha256(test_path),
        "predictors_exclude_identifiers_and_demographics": True,
        "n_features": len(feature_columns),
        "train_plus_dev_participants": int(train.sum()),
        "test_participants": int(test.sum()),
        "test_pd": int(labels[test].sum()),
        "test_non_pd": int(test.sum() - labels[test].sum()),
        "participant_auroc": float(primary["auroc"]),
        "participant_auroc_ci_low": float(primary["auroc_ci_low"]),
        "participant_auroc_ci_high": float(primary["auroc_ci_high"]),
        "participant_auprc": float(primary["auprc"]),
        "participant_balanced_accuracy": float(primary["balanced_accuracy"]),
        "participant_brier_score": float(primary["brier_score"]),
        "permutation_p_auroc": float(primary["permutation_p_auroc"]),
        "training_label_shuffle_auroc_mean": float(np.mean(shuffled_aurocs)),
        "training_label_shuffle_auroc_q975": float(
            np.quantile(shuffled_aurocs, 0.975)
        ),
        "interpretation_boundary": (
            "This is participant-level validation of public extracted smile "
            "features. It is not raw-image or anatomical-localisation validation."
        ),
    }
    (output_dir / "participant_benchmark_verdict.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    lines = [
        "# UFNet/PARK participant-level benchmark",
        "",
        "Predictors include only 42 public smile features. Identifiers and "
        "demographic columns are excluded, and repeated rows are aggregated "
        "before fitting.",
        "",
        f"- Train plus dev participants: **{summary['train_plus_dev_participants']}**.",
        f"- Locked test participants: **{summary['test_participants']}** "
        f"({summary['test_pd']} PD, {summary['test_non_pd']} non-PD).",
        f"- Participant AUROC: **{summary['participant_auroc']:.4f}** "
        f"(95% stratified-bootstrap CI "
        f"{summary['participant_auroc_ci_low']:.4f}-"
        f"{summary['participant_auroc_ci_high']:.4f}).",
        f"- Participant balanced accuracy: "
        f"**{summary['participant_balanced_accuracy']:.4f}**.",
        f"- Training-label shuffle AUROC mean/q97.5: "
        f"**{summary['training_label_shuffle_auroc_mean']:.4f}/"
        f"{summary['training_label_shuffle_auroc_q975']:.4f}**.",
        "",
        "This is feature-level evidence and cannot validate raw-face spatial localisation.",
    ]
    (output_dir / "participant_benchmark_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

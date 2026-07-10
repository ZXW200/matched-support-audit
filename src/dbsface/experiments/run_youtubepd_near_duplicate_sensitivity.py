"""Screen YouTubePD for obvious cross-split visual near duplicates.

This is a perceptual duplicate check, not face recognition.  A low similarity
does not establish that two clips contain different people.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.fft import dctn
from sklearn.metrics import roc_auc_score


DEFAULT_INPUT = Path("outputs/youtubepd_external_audit")


def centered_unit_vectors(images: np.ndarray) -> np.ndarray:
    features = images.reshape(len(images), -1).astype(np.float64)
    features -= features.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, 1e-12)


def perceptual_hash(images: np.ndarray) -> np.ndarray:
    coefficients = dctn(images, axes=(-2, -1), norm="ortho")[:, :8, :8]
    flattened = coefficients.reshape(len(images), -1)
    threshold = np.median(flattened[:, 1:], axis=1, keepdims=True)
    return flattened[:, 1:] > threshold


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = (args.output_dir or input_dir).resolve()
    qc_all = pd.read_csv(input_dir / "clip_preprocessing_qc.csv")
    if qc_all["qc_pass"].dtype != bool:
        qc_all["qc_pass"] = qc_all["qc_pass"].astype(str).str.lower().eq("true")
    keep = qc_all["qc_pass"].to_numpy(dtype=bool)
    qc = qc_all.loc[keep].reset_index(drop=True)
    archive = np.load(input_dir / "clip_level_video_features.npz")
    images = archive["face_norm_median"][keep]

    development = qc["split"].isin(["train", "val"]).to_numpy()
    test = qc["split"].eq("test").to_numpy()
    development_indices = np.flatnonzero(development)
    test_indices = np.flatnonzero(test)
    vectors = centered_unit_vectors(images)
    cosine = vectors[test] @ vectors[development].T
    hashes = perceptual_hash(images)

    prediction_table = pd.read_csv(input_dir / "youtubepd_control_test_predictions.csv")
    prediction_table = prediction_table[
        prediction_table["control"] == "aligned_face_normalized"
    ].set_index("video_id")

    pair_rows = []
    for test_row, test_index in enumerate(test_indices):
        similarity_order = np.argsort(cosine[test_row])[::-1]
        nearest_position = int(similarity_order[0])
        development_index = int(development_indices[nearest_position])
        hamming = np.count_nonzero(hashes[test_index] != hashes[development_index])
        pair_rows.append(
            {
                "test_video_id": qc.loc[test_index, "video_id"],
                "test_label": int(qc.loc[test_index, "label"]),
                "nearest_development_video_id": qc.loc[
                    development_index, "video_id"
                ],
                "nearest_development_label": int(qc.loc[development_index, "label"]),
                "centered_pixel_cosine": float(cosine[test_row, nearest_position]),
                "phash_hamming_distance": int(hamming),
                "same_label": bool(
                    qc.loc[test_index, "label"] == qc.loc[development_index, "label"]
                ),
            }
        )
    pairs = pd.DataFrame(pair_rows).sort_values(
        "centered_pixel_cosine", ascending=False
    )
    pairs.to_csv(output_dir / "cross_split_perceptual_nearest_pairs.csv", index=False)

    sensitivity_rows = []
    for threshold in (0.95, 0.90, 0.85, 0.80, 0.75):
        retained = pairs[pairs["centered_pixel_cosine"] < threshold].copy()
        labels = retained["test_video_id"].map(prediction_table["label"]).to_numpy()
        probabilities = retained["test_video_id"].map(
            prediction_table["probability"]
        ).to_numpy()
        auc = (
            float(roc_auc_score(labels, probabilities))
            if len(np.unique(labels)) == 2
            else np.nan
        )
        sensitivity_rows.append(
            {
                "maximum_allowed_cosine": threshold,
                "retained_test_clips": int(len(retained)),
                "excluded_test_clips": int(len(pairs) - len(retained)),
                "retained_pd": int(np.sum(labels == 1)),
                "retained_non_pd": int(np.sum(labels == 0)),
                "aligned_face_auroc": auc,
            }
        )
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(
        output_dir / "cross_split_similarity_threshold_sensitivity.csv", index=False
    )

    verdict = {
        "analysis_type": "perceptual near-duplicate screen",
        "identity_recognition_performed": False,
        "development_clips": int(development.sum()),
        "test_clips": int(test.sum()),
        "maximum_cross_split_centered_pixel_cosine": float(
            pairs["centered_pixel_cosine"].max()
        ),
        "minimum_nearest_pair_phash_hamming_distance": int(
            pairs["phash_hamming_distance"].min()
        ),
        "test_clips_with_cosine_at_least_0_90": int(
            (pairs["centered_pixel_cosine"] >= 0.90).sum()
        ),
        "test_clips_with_phash_distance_at_most_4": int(
            (pairs["phash_hamming_distance"] <= 4).sum()
        ),
        "interpretation_boundary": (
            "This screen can detect obvious visual duplication but cannot prove "
            "identity-disjoint splits without participant identifiers."
        ),
    }
    (output_dir / "cross_split_duplicate_verdict.json").write_text(
        json.dumps(verdict, indent=2), encoding="utf-8"
    )
    print(json.dumps(verdict, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

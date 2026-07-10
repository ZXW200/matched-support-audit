"""Prepare label-blind, clip-level YouTubePD features for the external audit.

The local videos are reconstructed raw clips.  The author-provided region
coordinates were generated after a different person-centering and frame-rate
pipeline, so this script detects and aligns faces directly from the raw clips.
No frame is treated as an independent observation: every saved feature is a
single temporal summary per video clip.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd


DEFAULT_ARCHIVE = Path("external") / "YouTubePD-data"
DEFAULT_MODEL = Path("models/face_landmarker.task")
DEFAULT_OUTPUT = Path("outputs/youtubepd_external_audit")
FACE_SIZE = 64
AUDIT_SIZE = 32

LEFT_EYE_INDICES = (33, 133, 159, 145)
RIGHT_EYE_INDICES = (362, 263, 386, 374)
MOUTH_INDICES = (61, 291)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def video_id_number(video_id: str) -> int:
    return int(video_id.replace("video", ""))


def load_balanced_cohort(youtube_root: Path) -> pd.DataFrame:
    sheet_path = youtube_root / "data_sheets" / "data_sheet.xlsx"
    frame = pd.read_excel(sheet_path)
    frame["video_id"] = [f"video{index}" for index in frame.index]
    status = frame["parkinson y/n"].astype(str).str.strip().str.lower()
    if not status.isin(["y", "n"]).all():
        invalid = frame.loc[~status.isin(["y", "n"]), "video_id"].tolist()
        raise ValueError(f"Unexpected PD labels for {invalid}")
    frame["label"] = status.eq("y").astype(np.int64)
    frame["split"] = frame["split"].astype(str).str.strip().str.lower()
    frame["video_path"] = frame["video_id"].map(
        lambda value: youtube_root / "raw_clips" / f"{value}_final.mp4"
    )
    frame["available"] = frame["video_path"].map(Path.exists)
    frame = frame.loc[frame["available"]].copy()
    frame = frame.sort_values("video_id", key=lambda s: s.map(video_id_number))
    return frame.reset_index(drop=True)


def sampled_frame_indices(frame_count: int, samples: int) -> np.ndarray:
    if frame_count < 1:
        return np.asarray([], dtype=np.int64)
    start = int(round(0.05 * (frame_count - 1)))
    end = int(round(0.95 * (frame_count - 1)))
    return np.unique(np.rint(np.linspace(start, end, samples)).astype(np.int64))


def resize_for_detection(frame: np.ndarray, max_side: int) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = min(1.0, max_side / float(max(height, width)))
    if scale == 1.0:
        return frame
    return cv2.resize(
        frame,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def bbox_iou(first: np.ndarray, second: np.ndarray) -> float:
    x0 = max(float(first[0]), float(second[0]))
    y0 = max(float(first[1]), float(second[1]))
    x1 = min(float(first[2]), float(second[2]))
    y1 = min(float(first[3]), float(second[3]))
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    first_area = max(0.0, float(first[2] - first[0])) * max(
        0.0, float(first[3] - first[1])
    )
    second_area = max(0.0, float(second[2] - second[0])) * max(
        0.0, float(second[3] - second[1])
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def landmark_candidate(landmarks: list[Any], width: int, height: int) -> dict[str, Any]:
    points = np.asarray(
        [[float(point.x) * width, float(point.y) * height] for point in landmarks],
        dtype=np.float32,
    )
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    bbox = np.asarray(
        [minimum[0], minimum[1], maximum[0], maximum[1]], dtype=np.float32
    )
    centre = (minimum + maximum) / 2.0
    area = float(maximum[0] - minimum[0]) * float(maximum[1] - minimum[1])
    return {"points": points, "bbox": bbox, "centre": centre, "area": area}


def select_face(
    candidates: list[dict[str, Any]],
    previous_bbox: np.ndarray | None,
    width: int,
    height: int,
) -> dict[str, Any] | None:
    if not candidates:
        return None
    frame_area = float(width * height)
    frame_centre = np.asarray([width / 2.0, height / 2.0], dtype=np.float32)
    frame_diagonal = float(np.hypot(width, height))
    scored: list[tuple[float, dict[str, Any]]] = []
    for candidate in candidates:
        area_fraction = max(candidate["area"] / frame_area, 1e-8)
        central_distance = float(
            np.linalg.norm(candidate["centre"] - frame_centre) / frame_diagonal
        )
        score = float(np.log(area_fraction)) - 0.35 * central_distance
        if previous_bbox is not None:
            previous_centre = np.asarray(
                [
                    (previous_bbox[0] + previous_bbox[2]) / 2.0,
                    (previous_bbox[1] + previous_bbox[3]) / 2.0,
                ],
                dtype=np.float32,
            )
            track_distance = float(
                np.linalg.norm(candidate["centre"] - previous_centre) / frame_diagonal
            )
            score += 2.5 * bbox_iou(candidate["bbox"], previous_bbox)
            score -= 1.25 * track_distance
        scored.append((score, candidate))
    return max(scored, key=lambda item: item[0])[1]


def mean_point(points: np.ndarray, indices: tuple[int, ...]) -> np.ndarray:
    return points[np.asarray(indices, dtype=np.int64)].mean(axis=0)


def align_face(frame: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, float]:
    eye_a = mean_point(points, LEFT_EYE_INDICES)
    eye_b = mean_point(points, RIGHT_EYE_INDICES)
    left_eye, right_eye = sorted([eye_a, eye_b], key=lambda point: float(point[0]))
    mouth = mean_point(points, MOUTH_INDICES)
    source = np.asarray([left_eye, right_eye, mouth], dtype=np.float32)
    destination = np.asarray(
        [[20.0, 23.0], [44.0, 23.0], [32.0, 45.0]], dtype=np.float32
    )
    transform = cv2.getAffineTransform(source, destination)
    aligned = cv2.warpAffine(
        frame,
        transform,
        (FACE_SIZE, FACE_SIZE),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    inter_eye = float(np.linalg.norm(right_eye - left_eye))
    return aligned, inter_eye


def robust_normalize(gray: np.ndarray) -> np.ndarray:
    low, high = np.percentile(gray, [2.0, 98.0])
    if high <= low + 1e-6:
        return np.zeros_like(gray, dtype=np.float32)
    return np.clip((gray - low) / (high - low), 0.0, 1.0).astype(np.float32)


def temporal_summaries(frames: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    stack = np.stack(frames, axis=0).astype(np.float32)
    median = np.median(stack, axis=0)
    mad = np.median(np.abs(stack - median[None, ...]), axis=0)
    return median.astype(np.float32), mad.astype(np.float32)


def empty_summary() -> np.ndarray:
    return np.full((AUDIT_SIZE, AUDIT_SIZE), np.nan, dtype=np.float32)


def expand_bbox(bbox: np.ndarray, width: int, height: int, factor: float = 1.25) -> tuple[int, int, int, int]:
    centre_x = float(bbox[0] + bbox[2]) / 2.0
    centre_y = float(bbox[1] + bbox[3]) / 2.0
    half_width = float(bbox[2] - bbox[0]) * factor / 2.0
    half_height = float(bbox[3] - bbox[1]) * factor / 2.0
    x0 = max(0, int(np.floor(centre_x - half_width)))
    y0 = max(0, int(np.floor(centre_y - half_height)))
    x1 = min(width, int(np.ceil(centre_x + half_width)))
    y1 = min(height, int(np.ceil(centre_y + half_height)))
    return x0, y0, x1, y1


def process_clip(
    path: Path,
    detector: Any,
    sample_frames: int,
    detection_max_side: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    capture = cv2.VideoCapture(str(path))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    indices = sampled_frame_indices(frame_count, sample_frames)

    aligned_raw: list[np.ndarray] = []
    aligned_normalized: list[np.ndarray] = []
    full_frames: list[np.ndarray] = []
    context_frames: list[np.ndarray] = []
    geometry_frames: list[np.ndarray] = []
    inter_eye_scaled: list[float] = []
    bbox_area_fractions: list[float] = []
    centre_jumps: list[float] = []
    multi_face_frames = 0
    decode_failures = 0
    previous_bbox: np.ndarray | None = None
    previous_centre: np.ndarray | None = None

    for frame_index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = capture.read()
        if not ok:
            decode_failures += 1
            continue
        detection_frame = resize_for_detection(frame, detection_max_side)
        height, width = detection_frame.shape[:2]
        rgb = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(
            image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)
        )
        result = detector.detect(image)
        candidates = [
            landmark_candidate(landmarks, width, height)
            for landmarks in result.face_landmarks
        ]
        if len(candidates) > 1:
            multi_face_frames += 1
        selected = select_face(candidates, previous_bbox, width, height)
        if selected is None:
            continue

        aligned, inter_eye = align_face(detection_frame, selected["points"])
        bbox = selected["bbox"]
        bbox_width = max(float(bbox[2] - bbox[0]), 1.0)
        bbox_height = max(float(bbox[3] - bbox[1]), 1.0)
        inter_eye_scaled.append(inter_eye / max(bbox_width, bbox_height) * FACE_SIZE)
        bbox_area_fractions.append(selected["area"] / float(width * height))
        if previous_centre is not None:
            centre_jumps.append(
                float(
                    np.linalg.norm(selected["centre"] - previous_centre)
                    / np.hypot(width, height)
                )
            )
        previous_bbox = bbox
        previous_centre = selected["centre"]

        gray_aligned = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray_aligned /= 255.0
        gray_small = cv2.resize(
            gray_aligned, (AUDIT_SIZE, AUDIT_SIZE), interpolation=cv2.INTER_AREA
        )
        normalized_small = cv2.resize(
            robust_normalize(gray_aligned),
            (AUDIT_SIZE, AUDIT_SIZE),
            interpolation=cv2.INTER_AREA,
        )
        aligned_raw.append(gray_small.astype(np.float32))
        aligned_normalized.append(normalized_small.astype(np.float32))

        gray_full = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2GRAY).astype(
            np.float32
        ) / 255.0
        x0, y0, x1, y1 = expand_bbox(bbox, width, height)
        context = gray_full.copy()
        context[y0:y1, x0:x1] = float(np.median(gray_full))
        geometry = np.zeros_like(gray_full, dtype=np.float32)
        geometry[y0:y1, x0:x1] = 1.0
        full_frames.append(
            cv2.resize(gray_full, (AUDIT_SIZE, AUDIT_SIZE), interpolation=cv2.INTER_AREA)
        )
        context_frames.append(
            cv2.resize(context, (AUDIT_SIZE, AUDIT_SIZE), interpolation=cv2.INTER_AREA)
        )
        geometry_frames.append(
            cv2.resize(geometry, (AUDIT_SIZE, AUDIT_SIZE), interpolation=cv2.INTER_AREA)
        )

    capture.release()
    detected = len(aligned_raw)
    required = int(np.ceil(sample_frames / 2.0))
    median_eye = float(np.median(inter_eye_scaled)) if inter_eye_scaled else np.nan
    qc_pass = bool(detected >= required and np.isfinite(median_eye) and median_eye >= 12.0)

    summaries: dict[str, np.ndarray] = {}
    groups = {
        "face_raw": aligned_raw,
        "face_norm": aligned_normalized,
        "full_frame": full_frames,
        "context_masked": context_frames,
        "mask_geometry": geometry_frames,
    }
    for name, values in groups.items():
        if values:
            median, mad = temporal_summaries(values)
        else:
            median, mad = empty_summary(), empty_summary()
        summaries[f"{name}_median"] = median
        summaries[f"{name}_mad"] = mad

    qc = {
        "frame_count": frame_count,
        "fps": fps,
        "source_width": source_width,
        "source_height": source_height,
        "duration_seconds": frame_count / fps if fps > 0 else np.nan,
        "file_size_bytes": path.stat().st_size,
        "sampled_frames": int(len(indices)),
        "detected_frames": detected,
        "detection_rate": detected / float(len(indices)) if len(indices) else 0.0,
        "decode_failures": decode_failures,
        "multi_face_frames": multi_face_frames,
        "multi_face_fraction": multi_face_frames / float(len(indices)) if len(indices) else 0.0,
        "median_scaled_inter_eye": median_eye,
        "median_bbox_area_fraction": float(np.median(bbox_area_fractions))
        if bbox_area_fractions
        else np.nan,
        "median_track_centre_jump": float(np.median(centre_jumps))
        if centre_jumps
        else np.nan,
        "max_track_centre_jump": float(np.max(centre_jumps)) if centre_jumps else np.nan,
        "qc_pass": qc_pass,
    }
    return summaries, qc


def write_summary(output_dir: Path, qc: pd.DataFrame) -> None:
    total = len(qc)
    passed = int(qc["qc_pass"].sum())
    class_rates = qc.groupby("label")["qc_pass"].mean().to_dict()
    difference = abs(float(class_rates.get(1, np.nan)) - float(class_rates.get(0, np.nan)))
    valid = bool(passed / total >= 0.8 and np.isfinite(difference) and difference <= 0.10)
    lines = [
        "# YouTubePD preprocessing QC",
        "",
        f"- Clips processed: **{total}**.",
        f"- Clips passing locked QC: **{passed}/{total} ({passed / total:.1%})**.",
        f"- PD pass rate: **{class_rates.get(1, float('nan')):.1%}**.",
        f"- Non-PD pass rate: **{class_rates.get(0, float('nan')):.1%}**.",
        f"- Absolute class pass-rate difference: **{difference:.1%}**.",
        f"- Eligible for the protocol-specified external audit: **{valid}**.",
        "",
        "All rates are clip-level. YouTubePD does not provide verified subject identifiers.",
    ]
    (output_dir / "preprocessing_qc_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--youtube-root", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-frames", type=int, default=24)
    parser.add_argument("--detection-max-side", type=int, default=960)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    youtube_root = args.youtube_root.resolve()
    model_path = args.model.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    cohort = load_balanced_cohort(youtube_root)
    if args.limit is not None:
        cohort = cohort.iloc[: args.limit].copy()

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=4,
        min_face_detection_confidence=0.30,
        min_face_presence_confidence=0.30,
        min_tracking_confidence=0.30,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )

    arrays: dict[str, list[np.ndarray]] = {}
    qc_rows: list[dict[str, Any]] = []
    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as detector:
        for row_number, row in cohort.iterrows():
            video_id = str(row["video_id"])
            summaries, qc = process_clip(
                Path(row["video_path"]),
                detector,
                args.sample_frames,
                args.detection_max_side,
            )
            for name, value in summaries.items():
                arrays.setdefault(name, []).append(value)
            qc_rows.append(
                {
                    "video_id": video_id,
                    "label": int(row["label"]),
                    "split": str(row["split"]),
                    "year": int(row["year"]),
                    "severity_label": row.get("severeness_label"),
                    "confidence_label": row.get("confidence_label"),
                    "source_url": str(row["link"]),
                    **qc,
                }
            )
            print(
                f"[{row_number + 1:03d}/{len(cohort):03d}] {video_id}: "
                f"detected={qc['detected_frames']}/{qc['sampled_frames']}; "
                f"qc={qc['qc_pass']}"
            )

    qc_frame = pd.DataFrame(qc_rows)
    qc_frame.to_csv(output_dir / "clip_preprocessing_qc.csv", index=False)
    np.savez_compressed(
        output_dir / "clip_level_video_features.npz",
        video_id=qc_frame["video_id"].to_numpy(dtype=str),
        **{name: np.stack(values, axis=0) for name, values in arrays.items()},
    )

    manifest = {
        "analysis_unit": "video clip",
        "subject_identifiers_available": False,
        "youtube_root": str(youtube_root),
        "balanced_sheet_sha256": sha256(
            youtube_root / "data_sheets" / "data_sheet.xlsx"
        ),
        "face_landmarker_model": str(model_path),
        "face_landmarker_model_sha256": sha256(model_path),
        "sample_frames": args.sample_frames,
        "detection_max_side": args.detection_max_side,
        "face_size": FACE_SIZE,
        "audit_size": AUDIT_SIZE,
        "clips_processed": int(len(qc_frame)),
        "clips_passing_qc": int(qc_frame["qc_pass"].sum()),
    }
    (output_dir / "preprocessing_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    write_summary(output_dir, qc_frame)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

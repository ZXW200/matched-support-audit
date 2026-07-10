"""Download and crop YouTubePD clips from the official spreadsheet metadata.

This script only reconstructs the raw YouTube video clips listed by the
YouTubePD repository. It does not run the MMPose face/region pipeline.

The official spreadsheets store clip times as mm:ss, but Excel/OpenPyXL can
surface them as datetime.time(hour=mm, minute=ss). The parser below preserves
the intended mm:ss interpretation.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import imageio_ffmpeg
import pandas as pd


REPO_ROOT = Path("external") / "YouTubePD-data"
DATA_SHEETS = REPO_ROOT / "data_sheets"


@dataclass(frozen=True)
class ClipRecord:
    video_id: str
    source_sheet: str
    link: str
    start_seconds: int
    end_seconds: int
    split: str
    label: str
    confidence: str

    @property
    def duration_seconds(self) -> int:
        return self.end_seconds - self.start_seconds


def is_missing(value: object) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def parse_time_cell(value: object) -> int:
    """Parse spreadsheet time values as mm:ss, not hh:mm."""
    if is_missing(value):
        raise ValueError("missing time value")

    if isinstance(value, dt.time):
        return int(value.hour) * 60 + int(value.minute) + round(value.second)

    if isinstance(value, pd.Timestamp):
        return int(value.hour) * 60 + int(value.minute) + round(value.second)

    text = str(value).strip()
    if not text:
        raise ValueError("empty time value")

    parts = [int(float(part)) for part in text.split(":")]
    if len(parts) == 2:
        minutes, seconds = parts
        return minutes * 60 + seconds
    if len(parts) == 3:
        hours, minutes, seconds = parts
        # The balanced spreadsheet may appear as HH:MM:SS after import even
        # though the original YouTubePD script treated these cells as MM:SS.
        if hours < 60 and seconds == 0:
            return hours * 60 + minutes
        return hours * 3600 + minutes * 60 + seconds

    raise ValueError(f"unsupported time format: {value!r}")


def format_seconds(seconds: int) -> str:
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def read_split_lookup() -> dict[str, tuple[str, str]]:
    lookup: dict[str, tuple[str, str]] = {}
    for split_name in ["train", "test"]:
        csv_path = REPO_ROOT / f"{split_name}.csv"
        if not csv_path.exists():
            continue
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                path_text, label = line.rsplit(maxsplit=1)
                stem = Path(path_text).name.replace("_final.mp4", "")
                lookup[stem] = (split_name, label)
    return lookup


def clean_value(value: object) -> str:
    if is_missing(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def load_balanced_records() -> list[ClipRecord]:
    path = DATA_SHEETS / "data_sheet.xlsx"
    frame = pd.read_excel(path)
    records: list[ClipRecord] = []
    for index, row in frame.iterrows():
        link = clean_value(row.get("link"))
        if not link.startswith("http"):
            continue
        try:
            start = parse_time_cell(row.get("start"))
            end = parse_time_cell(row.get("end"))
        except ValueError as exc:
            print(f"skip video{index} from data_sheet.xlsx: {exc}", file=sys.stderr)
            continue
        records.append(
            ClipRecord(
                video_id=f"video{index}",
                source_sheet="data_sheet.xlsx",
                link=link,
                start_seconds=start,
                end_seconds=end,
                split=clean_value(row.get("split")),
                label=clean_value(row.get("severeness_label")),
                confidence=clean_value(row.get("confidence_label")),
            )
        )
    return records


def load_negative_records(split_lookup: dict[str, tuple[str, str]]) -> list[ClipRecord]:
    path = DATA_SHEETS / "NegSamples.xlsx"
    frame = pd.read_excel(path, header=None)
    records: list[ClipRecord] = []
    for index, row in frame.iterrows():
        link = clean_value(row.iloc[0])
        if not link.startswith("http"):
            continue
        video_id = f"video{index + 134}"
        split, label = split_lookup.get(video_id, ("", "0"))
        try:
            start = parse_time_cell(row.iloc[3])
            end = parse_time_cell(row.iloc[4])
        except ValueError as exc:
            print(f"skip {video_id} from NegSamples.xlsx: {exc}", file=sys.stderr)
            continue
        records.append(
            ClipRecord(
                video_id=video_id,
                source_sheet="NegSamples.xlsx",
                link=link,
                start_seconds=start,
                end_seconds=end,
                split=split,
                label=label,
                confidence="",
            )
        )
    return records


def load_records(sheet: str) -> list[ClipRecord]:
    split_lookup = read_split_lookup()
    records: list[ClipRecord] = []
    if sheet in {"balanced", "all"}:
        records.extend(load_balanced_records())
    if sheet in {"negative", "all"}:
        records.extend(load_negative_records(split_lookup))
    return records


def selected_records(args: argparse.Namespace) -> list[ClipRecord]:
    records = load_records(args.sheet)
    if args.video_id:
        wanted = set(args.video_id)
        records = [record for record in records if record.video_id in wanted]
    if args.min_index is not None:
        records = [
            record
            for record in records
            if int(record.video_id.replace("video", "")) >= args.min_index
        ]
    if args.max_index is not None:
        records = [
            record
            for record in records
            if int(record.video_id.replace("video", "")) <= args.max_index
        ]
    if args.limit is not None:
        records = records[: args.limit]
    return records


def write_manifest_row(manifest_path: Path, row: dict[str, object]) -> None:
    fieldnames = [
        "video_id",
        "status",
        "source_sheet",
        "split",
        "label",
        "confidence",
        "start",
        "end",
        "duration_seconds",
        "output_path",
        "link",
        "return_code",
    ]
    exists = manifest_path.exists()
    with manifest_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def yt_dlp_command(
    record: ClipRecord,
    output_dir: Path,
    ffmpeg_path: Path,
    js_runtime: str | None,
) -> list[str]:
    output_template = output_dir / f"{record.video_id}_final.%(ext)s"
    section = f"*{format_seconds(record.start_seconds)}-{format_seconds(record.end_seconds)}"
    command = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--no-playlist",
        "--ffmpeg-location",
        str(ffmpeg_path),
        "--download-sections",
        section,
        "--force-keyframes-at-cuts",
        "-f",
        "18/best[ext=mp4]/best",
        "--merge-output-format",
        "mp4",
        "-o",
        str(output_template),
    ]
    if js_runtime:
        command.extend(["--js-runtimes", js_runtime, "--remote-components", "ejs:github"])
    command.append(record.link)
    return command


def main() -> int:
    global REPO_ROOT, DATA_SHEETS
    parser = argparse.ArgumentParser()
    parser.add_argument("--youtube-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--sheet", choices=["balanced", "negative", "all"], default="all")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--min-index", type=int)
    parser.add_argument("--max-index", type=int)
    parser.add_argument("--video-id", action="append", help="Repeatable, e.g. --video-id video74")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--js-runtime",
        help="Optional yt-dlp runtime, for example node:C:\\path\\to\\node.exe",
    )
    parser.add_argument("--allow-failures", action="store_true")
    args = parser.parse_args()

    REPO_ROOT = args.youtube_root.resolve()
    DATA_SHEETS = REPO_ROOT / "data_sheets"
    output_dir = (args.output_dir or REPO_ROOT / "raw_clips").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "youtubepd_download_manifest.csv"
    ffmpeg_path = Path(imageio_ffmpeg.get_ffmpeg_exe()).resolve()
    ffmpeg_dir = ffmpeg_path.parent

    records = selected_records(args)
    print(f"selected {len(records)} clip(s)")
    print(f"output_dir={output_dir}")
    print(f"ffmpeg={ffmpeg_path}")

    failures = 0
    for record in records:
        output_path = output_dir / f"{record.video_id}_final.mp4"
        row = {
            "video_id": record.video_id,
            "source_sheet": record.source_sheet,
            "split": record.split,
            "label": record.label,
            "confidence": record.confidence,
            "start": format_seconds(record.start_seconds),
            "end": format_seconds(record.end_seconds),
            "duration_seconds": record.duration_seconds,
            "output_path": str(output_path),
            "link": record.link,
        }

        if output_path.exists() and not args.overwrite:
            print(f"skip existing {record.video_id}: {output_path}")
            write_manifest_row(manifest_path, {**row, "status": "exists", "return_code": 0})
            continue

        command = yt_dlp_command(record, output_dir, ffmpeg_path, args.js_runtime)
        print(" ".join(f'"{part}"' if " " in part else part for part in command))
        if args.dry_run:
            write_manifest_row(manifest_path, {**row, "status": "dry_run", "return_code": ""})
            continue

        env = os.environ.copy()
        env["PATH"] = str(ffmpeg_dir) + os.pathsep + env.get("PATH", "")
        result = subprocess.run(command, cwd=REPO_ROOT, env=env, check=False)
        status = "downloaded" if result.returncode == 0 and output_path.exists() else "failed"
        failures += int(status == "failed")
        write_manifest_row(
            manifest_path,
            {**row, "status": status, "return_code": result.returncode},
        )

    if failures and not args.allow_failures:
        print(f"failed clips: {failures}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

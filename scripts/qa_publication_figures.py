"""Run format, dimension, editability, and source-traceability QA on figures."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from pypdf import PdfReader


MM_PER_INCH = 25.4
POINTS_PER_INCH = 72.0


def check(condition: bool, name: str, detail: str, rows: list[dict[str, object]]) -> None:
    rows.append({"check": name, "passed": bool(condition), "detail": detail})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def raster_metrics(path: Path) -> tuple[tuple[int, int], tuple[float, float], float, float]:
    with Image.open(path) as image:
        size = image.size
        dpi = image.info.get("dpi", (0.0, 0.0))
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    greyscale = rgb.mean(axis=2)
    standard_deviation = float(greyscale.std())
    nonwhite_fraction = float(np.mean(np.min(rgb, axis=2) < 248))
    return size, (float(dpi[0]), float(dpi[1])), standard_deviation, nonwhite_fraction


def svg_metrics(path: Path) -> tuple[int, int, str]:
    tree = ET.parse(path)
    root = tree.getroot()
    text_count = 0
    image_count = 0
    for element in root.iter():
        local = element.tag.rsplit("}", 1)[-1]
        if local == "text":
            text_count += 1
        elif local == "image":
            image_count += 1
    raw = path.read_text(encoding="utf-8")
    return text_count, image_count, raw


def make_contact_sheet(figure_dir: Path, stems: list[str]) -> Path:
    previews: list[tuple[str, Image.Image]] = []
    target_width = 1000
    for stem in stems:
        image = Image.open(figure_dir / f"{stem}.png").convert("RGB")
        scale = target_width / image.width
        resized = image.resize((target_width, int(round(image.height * scale))), Image.Resampling.LANCZOS)
        previews.append((stem, resized))

    margin = 32
    label_height = 34
    column_width = target_width + 2 * margin
    row_heights: list[int] = []
    for index in range(0, len(previews), 2):
        heights = [previews[index][1].height]
        if index + 1 < len(previews):
            heights.append(previews[index + 1][1].height)
        row_heights.append(max(heights) + label_height + 2 * margin)
    sheet = Image.new("RGB", (2 * column_width, sum(row_heights)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=18)
    y_cursor = 0
    for row_index, index in enumerate(range(0, len(previews), 2)):
        for column in range(2):
            item_index = index + column
            if item_index >= len(previews):
                continue
            label, preview = previews[item_index]
            x = column * column_width + margin
            y = y_cursor + margin + label_height
            draw.text((x, y_cursor + margin), label, fill="#262B33", font=font)
            sheet.paste(preview, (x, y))
        y_cursor += row_heights[row_index]
    output = figure_dir / "figure_contact_sheet.png"
    sheet.save(output, dpi=(150, 150), optimize=True)
    for _, image in previews:
        image.close()
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--figure-dir", type=Path, default=Path("figures"))
    parser.add_argument("--source-data", type=Path, default=Path("source_data"))
    args = parser.parse_args()

    figure_dir = args.figure_dir.resolve()
    source_dir = args.source_data.resolve()
    manifest = json.loads((figure_dir / "figure_manifest.json").read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []

    private_path_patterns = [
        r"[A-Za-z]:\\(?:Users|Documents and Settings)\\",
        r"/(?:home|Users)/[^/]+/",
        r"file://",
    ]
    for item in manifest:
        stem = str(item["stem"])
        width_mm = float(item["width_mm"])
        height_mm = float(item["height_mm"])
        expected_png = (round(width_mm / MM_PER_INCH * 300), round(height_mm / MM_PER_INCH * 300))
        expected_tiff = (round(width_mm / MM_PER_INCH * 600), round(height_mm / MM_PER_INCH * 600))

        for suffix in ("svg", "pdf", "tiff", "png"):
            path = figure_dir / f"{stem}.{suffix}"
            check(path.exists() and path.stat().st_size > 1000, f"{stem}: {suffix} exists", f"{path.name}; {path.stat().st_size if path.exists() else 0} bytes", rows)

        png_size, png_dpi, png_sd, png_nonwhite = raster_metrics(figure_dir / f"{stem}.png")
        check(all(abs(a - b) <= 1 for a, b in zip(png_size, expected_png)), f"{stem}: PNG dimensions", f"observed={png_size}; expected={expected_png}", rows)
        check(png_sd > 8.0 and 0.005 < png_nonwhite < 0.75, f"{stem}: PNG nonblank", f"pixel_sd={png_sd:.2f}; nonwhite={png_nonwhite:.3f}", rows)
        check(min(png_dpi) >= 299.0, f"{stem}: PNG resolution", f"dpi={png_dpi}", rows)

        tiff_size, tiff_dpi, tiff_sd, tiff_nonwhite = raster_metrics(figure_dir / f"{stem}.tiff")
        check(all(abs(a - b) <= 1 for a, b in zip(tiff_size, expected_tiff)), f"{stem}: TIFF dimensions", f"observed={tiff_size}; expected={expected_tiff}", rows)
        check(min(tiff_dpi) >= 599.0, f"{stem}: TIFF resolution", f"dpi={tiff_dpi}", rows)
        check(tiff_sd > 8.0 and 0.005 < tiff_nonwhite < 0.75, f"{stem}: TIFF nonblank", f"pixel_sd={tiff_sd:.2f}; nonwhite={tiff_nonwhite:.3f}", rows)

        pdf_path = figure_dir / f"{stem}.pdf"
        reader = PdfReader(str(pdf_path))
        page = reader.pages[0]
        pdf_width = float(page.mediabox.width)
        pdf_height = float(page.mediabox.height)
        expected_pdf = (width_mm / MM_PER_INCH * POINTS_PER_INCH, height_mm / MM_PER_INCH * POINTS_PER_INCH)
        pdf_text = "\n".join((p.extract_text() or "") for p in reader.pages)
        check(len(reader.pages) == 1, f"{stem}: PDF page count", f"pages={len(reader.pages)}", rows)
        check(abs(pdf_width - expected_pdf[0]) < 0.6 and abs(pdf_height - expected_pdf[1]) < 0.6, f"{stem}: PDF media box", f"observed=({pdf_width:.2f}, {pdf_height:.2f}) pt; expected=({expected_pdf[0]:.2f}, {expected_pdf[1]:.2f}) pt", rows)
        check(len(pdf_text) > 100, f"{stem}: PDF searchable text", f"characters={len(pdf_text)}", rows)

        svg_path = figure_dir / f"{stem}.svg"
        text_count, image_count, svg_raw = svg_metrics(svg_path)
        check(text_count >= 10, f"{stem}: SVG editable text", f"text_elements={text_count}; embedded_images={image_count}", rows)
        leaked = [pattern for pattern in private_path_patterns if re.search(pattern, svg_raw, flags=re.IGNORECASE)]
        check(not leaked, f"{stem}: no private path in SVG", f"matched_patterns={leaked}", rows)

    source_manifest = pd.read_csv(source_dir / "source_data_manifest.csv")
    for row in source_manifest.itertuples(index=False):
        path = source_dir / str(row.file)
        observed = sha256(path) if path.exists() else "missing"
        check(observed == str(row.sha256), f"source data: {row.file}", f"sha256={observed}", rows)

    contact_sheet = make_contact_sheet(figure_dir, [str(item["stem"]) for item in manifest])
    check(contact_sheet.exists() and contact_sheet.stat().st_size > 1000, "Contact sheet", contact_sheet.name, rows)

    passed = sum(bool(row["passed"]) for row in rows)
    total = len(rows)
    report = {
        "backend": "Python/matplotlib/Pillow only",
        "passed": passed,
        "total": total,
        "all_passed": passed == total,
        "checks": rows,
    }
    (figure_dir / "FIGURE_QA_REPORT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    markdown = [
        "# Publication Figure QA Report",
        "",
        f"- Backend: {report['backend']}",
        f"- Result: **{passed}/{total} checks passed**",
        "- Visual inspection: performed on the Python-generated full-resolution PNG set and contact sheet.",
        "",
        "| Status | Check | Detail |",
        "|---|---|---|",
    ]
    for row in rows:
        status = "PASS" if row["passed"] else "FAIL"
        detail = str(row["detail"]).replace("|", "\\|")
        markdown.append(f"| {status} | {row['check']} | {detail} |")
    (figure_dir / "FIGURE_QA_REPORT.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")

    print(f"Figure QA: {passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())

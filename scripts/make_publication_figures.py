"""Render the submission figure set from public aggregate source data.

All visual output is produced with Python/matplotlib. The script intentionally
does not read restricted image matrices, reconstructed videos, or identifiers.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd


MM_PER_INCH = 25.4
FIGURE_WIDTH_MM = 183.0

INK = "#262B33"
GREY_DARK = "#5F6670"
GREY = "#9AA1A9"
GREY_LIGHT = "#E7E9EC"
GREY_PALE = "#F5F6F7"
BLUE = "#557DA6"
BLUE_DARK = "#355E82"
BLUE_LIGHT = "#BED0E0"
TEAL = "#4E8B86"
TEAL_LIGHT = "#C6DEDB"
WARM = "#B66A5B"
WARM_LIGHT = "#E8C4BC"
AMBER = "#B68B42"
AMBER_LIGHT = "#E8D8B7"
WHITE = "#FFFFFF"

ROI_ORDER = [
    "upper_brow_forehead",
    "left_periocular",
    "right_periocular",
    "nasal_midface",
    "left_cheek_zygomatic",
    "right_cheek_zygomatic",
    "perioral_mouth",
    "chin_mandible",
]

ROI_SHORT = {
    "upper_brow_forehead": "Brow/forehead",
    "left_periocular": "Left periocular",
    "right_periocular": "Right periocular",
    "nasal_midface": "Nasal midface",
    "left_cheek_zygomatic": "Left cheek",
    "right_cheek_zygomatic": "Right cheek",
    "perioral_mouth": "Perioral/mouth",
    "chin_mandible": "Chin/mandible",
}

ROI_TICK = {
    "upper_brow_forehead": "Brow/\nforehead",
    "left_periocular": "Left\nperiocular",
    "right_periocular": "Right\nperiocular",
    "nasal_midface": "Nasal\nmidface",
    "left_cheek_zygomatic": "Left\ncheek",
    "right_cheek_zygomatic": "Right\ncheek",
    "perioral_mouth": "Perioral/\nmouth",
    "chin_mandible": "Chin/\nmandible",
}


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "svg.hashsalt": "matched-support-audit-v1.0.1",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 7.0,
            "axes.titlesize": 7.2,
            "axes.labelsize": 6.8,
            "axes.linewidth": 0.65,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.labelsize": 6.0,
            "ytick.labelsize": 6.0,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
            "legend.fontsize": 5.8,
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
            "savefig.facecolor": WHITE,
            "savefig.transparent": False,
        }
    )


def load_csv(source_dir: Path, filename: str) -> pd.DataFrame:
    path = source_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing source-data file: {path}")
    return pd.read_csv(path)


def new_figure(height_mm: float, constrained: bool = True) -> plt.Figure:
    fig = plt.figure(
        figsize=(FIGURE_WIDTH_MM / MM_PER_INCH, height_mm / MM_PER_INCH),
        layout="constrained" if constrained else None,
    )
    engine = fig.get_layout_engine() if constrained else None
    if engine is not None:
        engine.set(w_pad=0.025, h_pad=0.025, wspace=0.055, hspace=0.075)
    return fig


def panel_label(ax: plt.Axes, label: str, x: float = -0.11, y: float = 1.06) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=8.5,
        fontweight="bold",
        ha="left",
        va="top",
        color=INK,
        clip_on=False,
    )


def panel_title(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, loc="left", pad=4, fontweight="bold")


def unit_note(ax: plt.Axes, text: str, y: float = -0.22) -> None:
    ax.text(
        0.0,
        y,
        text,
        transform=ax.transAxes,
        fontsize=5.2,
        color=GREY_DARK,
        ha="left",
        va="top",
        clip_on=False,
    )


def add_reference_grid(ax: plt.Axes, axis: str = "y") -> None:
    ax.grid(axis=axis, color=GREY_LIGHT, linewidth=0.55, zorder=0)
    ax.set_axisbelow(True)


def forest_errorbar(
    ax: plt.Axes,
    x: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    y: np.ndarray,
    color: str,
    marker: str = "o",
    label: str | None = None,
    zorder: int = 3,
) -> None:
    ax.errorbar(
        x,
        y,
        xerr=np.vstack([x - low, high - x]),
        fmt=marker,
        color=color,
        markerfacecolor=color if marker != "o" else WHITE,
        markeredgecolor=color,
        markersize=4.0,
        markeredgewidth=0.9,
        elinewidth=0.85,
        capsize=2.0,
        capthick=0.75,
        label=label,
        zorder=zorder,
    )


def canonicalize_svg_ids(path: Path) -> None:
    """Replace backend-generated marker and clip IDs with stable ordered IDs."""
    raw = path.read_text(encoding="utf-8")
    generated = []
    for match in re.finditer(r'\bid="([mp][0-9a-f]+)"', raw):
        value = match.group(1)
        if value not in generated:
            generated.append(value)
    for index, old in enumerate(generated, start=1):
        new = f"{old[0]}stable{index:04d}"
        raw = raw.replace(f'id="{old}"', f'id="{new}"')
        raw = raw.replace(f"#{old}", f"#{new}")
    raw = "\n".join(line.rstrip() for line in raw.splitlines()) + "\n"
    path.write_text(raw, encoding="utf-8", newline="\n")


def save_figure(
    fig: plt.Figure,
    output_dir: Path,
    stem: str,
    height_mm: float,
    title: str,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    release_timestamp = datetime(2026, 7, 10, tzinfo=timezone.utc)
    metadata = {
        "Title": title,
        "Author": "Zixu Wang",
        "Subject": "Matched-support audit",
        "CreationDate": release_timestamp,
        "ModDate": release_timestamp,
    }
    paths = {
        "svg": output_dir / f"{stem}.svg",
        "pdf": output_dir / f"{stem}.pdf",
        "tiff": output_dir / f"{stem}.tiff",
        "png": output_dir / f"{stem}.png",
    }
    fig.savefig(
        paths["svg"],
        format="svg",
        metadata={"Title": title, "Date": "2026-07-10"},
    )
    canonicalize_svg_ids(paths["svg"])
    fig.savefig(paths["pdf"], format="pdf", metadata=metadata)
    fig.savefig(
        paths["tiff"],
        format="tiff",
        dpi=600,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    fig.savefig(paths["png"], format="png", dpi=300)
    plt.close(fig)
    return {
        "stem": stem,
        "title": title,
        "width_mm": FIGURE_WIDTH_MM,
        "height_mm": height_mm,
        "formats": {key: path.name for key, path in paths.items()},
    }


def make_figure_1(source_dir: Path, output_dir: Path) -> dict[str, object]:
    del source_dir
    height_mm = 148.0
    fig = new_figure(height_mm)
    gs = fig.add_gridspec(3, 2, width_ratios=(1.05, 1.0), height_ratios=(1.05, 0.66, 0.66))

    ax = fig.add_subplot(gs[:, 0])
    ax.set_axis_off()
    panel_label(ax, "a", x=-0.05, y=1.01)
    ax.text(0.02, 0.985, "Seven claims require different evidence", fontweight="bold", va="top")
    claims = [
        ("Discrimination", "Does any predictive signal exist?", TEAL),
        ("Support specificity", "Does this support outperform equal-pixel controls?", BLUE),
        ("Anatomical location", "Is this coordinate exceptional among translations?", BLUE_DARK),
        ("Explanation faithfulness", "Does a fixed model rely on that support?", GREY_DARK),
        ("Shortcut profile", "Can non-target or coarse supports predict?", AMBER),
        ("External evidence", "Does the decision persist in another resource?", TEAL),
        ("Clinical association", "Does it generalize to people and outcomes?", GREY),
    ]
    ys = np.linspace(0.86, 0.12, len(claims))
    ax.plot([0.075, 0.075], [ys[-1], ys[0]], color=GREY_LIGHT, linewidth=2.2, zorder=0)
    for i, ((name, question, color), y) in enumerate(zip(claims, ys), start=1):
        ax.add_patch(Circle((0.075, y), 0.027, facecolor=WHITE, edgecolor=color, linewidth=1.35))
        ax.text(0.075, y, str(i), ha="center", va="center", color=color, fontsize=5.7, fontweight="bold")
        ax.text(0.135, y + 0.014, name, ha="left", va="center", fontweight="bold", fontsize=6.4, color=INK)
        ax.text(0.135, y - 0.019, question, ha="left", va="center", fontsize=5.5, color=GREY_DARK)
    y_top = ys[1] + 0.05
    y_bottom = ys[2] - 0.05
    ax.plot([0.94, 0.97, 0.97, 0.94], [y_top, y_top, y_bottom, y_bottom], color=BLUE_DARK, linewidth=1.0)
    ax.text(0.985, (y_top + y_bottom) / 2, "CCA", rotation=90, color=BLUE_DARK, va="center", ha="left", fontsize=6.2, fontweight="bold")
    ax.text(0.02, 0.025, "CCA scope: support specificity and anatomical location.", fontsize=5.4, color=BLUE_DARK, fontweight="bold")
    ax.set_xlim(0, 1.08)
    ax.set_ylim(0, 1)

    ax = fig.add_subplot(gs[0, 1])
    ax.set_axis_off()
    panel_label(ax, "b", x=-0.09, y=1.04)
    ax.text(0.0, 1.0, "Two matched-control questions", fontweight="bold", va="top")

    def support_grid(x0: float, y0: float, kind: str) -> None:
        width, height = 0.13, 0.22
        cols = rows = 6
        cell_w, cell_h = width / cols, height / rows
        ax.add_patch(Rectangle((x0, y0), width, height, facecolor=GREY_PALE, edgecolor=INK, linewidth=0.65))
        for index in range(1, cols):
            x = x0 + index * cell_w
            ax.plot([x, x], [y0, y0 + height], color=GREY_LIGHT, linewidth=0.18, zorder=0)
        for index in range(1, rows):
            y = y0 + index * cell_h
            ax.plot([x0, x0 + width], [y, y], color=GREY_LIGHT, linewidth=0.18, zorder=0)
        if kind == "named":
            ax.add_patch(Rectangle((x0 + cell_w, y0 + 2 * cell_h), 3 * cell_w, 2 * cell_h, facecolor=WARM_LIGHT, edgecolor=WARM, linewidth=0.9))
        elif kind == "scattered":
            for col, row in ((0, 1), (1, 4), (2, 0), (3, 3), (4, 5), (5, 2)):
                ax.add_patch(Rectangle((x0 + col * cell_w, y0 + row * cell_h), cell_w, cell_h, facecolor=BLUE_LIGHT, edgecolor=BLUE, linewidth=0.35))
        elif kind == "translated":
            ax.add_patch(Rectangle((x0 + 3 * cell_w, y0 + 3 * cell_h), 3 * cell_w, 2 * cell_h, facecolor="none", edgecolor=BLUE_DARK, linewidth=0.9, linestyle=(0, (2, 1.4))))
        else:
            raise ValueError(f"unknown support-grid kind: {kind}")

    rows = [
        (0.84, 0.50, 0.60, "2", "Support specificity", "equal-pixel\nscattered", "scattered q97.5", "support gate"),
        (0.43, 0.10, 0.20, "3", "Anatomical location", "same-shape\ntranslated", "translated q97.5", "location margin"),
    ]
    for label_y, grid_y, centre_y, number, label, control_label, quantile_label, margin_label in rows:
        ax.text(0.02, label_y, number, color=BLUE_DARK, fontsize=5.2, fontweight="bold", ha="center", va="center", bbox={"boxstyle": "circle,pad=0.17", "facecolor": WHITE, "edgecolor": BLUE_DARK, "linewidth": 0.7})
        ax.text(0.055, label_y, label, fontsize=5.6, fontweight="bold", color=INK, ha="left", va="center")
        ax.text(0.085, grid_y + 0.245, "named", fontsize=4.7, color=WARM, fontweight="bold", ha="center")
        ax.text(0.245, grid_y + 0.245, control_label, fontsize=4.5, color=BLUE_DARK, fontweight="bold", ha="center", linespacing=0.9)
        support_grid(0.02, grid_y, "named")
        support_grid(0.18, grid_y, "scattered" if number == "2" else "translated")
        ax.add_patch(FancyArrowPatch((0.33, centre_y), (0.43, centre_y), arrowstyle="-|>", mutation_scale=7, linewidth=0.7, color=INK))
        ax.add_patch(FancyBboxPatch((0.44, centre_y - 0.085), 0.17, 0.17, boxstyle="round,pad=0.012,rounding_size=0.012", facecolor=BLUE_LIGHT, edgecolor=BLUE_DARK, linewidth=0.7))
        ax.text(0.525, centre_y + 0.025, "fresh fit", ha="center", va="center", fontweight="bold", fontsize=5.4)
        ax.text(0.525, centre_y - 0.035, "every support", ha="center", va="center", fontsize=4.6, color=GREY_DARK)
        ax.add_patch(FancyArrowPatch((0.62, centre_y), (0.69, centre_y), arrowstyle="-|>", mutation_scale=7, linewidth=0.7, color=INK))
        ax.text(0.71, centre_y + 0.055, "named AUROC", color=WARM, fontweight="bold", fontsize=5.1)
        ax.text(0.71, centre_y - 0.005, quantile_label, color=BLUE_DARK, fontweight="bold", fontsize=5.1)
        ax.text(0.71, centre_y - 0.065, margin_label, color=INK, fontsize=4.8)
    ax.text(0.02, 0.015, "Fresh fit per support. Data, representation, estimator and split fixed.", fontsize=4.7, color=GREY_DARK)
    ax.set_xlim(0, 1.04)
    ax.set_ylim(0, 1.02)

    ax = fig.add_subplot(gs[1, 1])
    panel_label(ax, "c", x=-0.09, y=1.08)
    panel_title(ax, "Location margin (claim 3)")
    illustrative = np.array([0.63, 0.66, 0.68, 0.69, 0.705, 0.72, 0.735, 0.75, 0.765, 0.78, 0.79, 0.80, 0.815, 0.825, 0.84, 0.85, 0.865, 0.875, 0.89, 0.905])
    jitter = np.array([0.02, -0.03, 0.04, -0.02, 0.0] * 4)
    ax.scatter(illustrative, jitter, s=8, facecolor=GREY_LIGHT, edgecolor=GREY, linewidth=0.35, zorder=2)
    q975 = 0.898
    named = 0.925
    ax.axvline(q975, color=BLUE_DARK, linestyle=(0, (3, 2)), linewidth=0.9)
    ax.scatter([named], [0], marker="D", s=25, color=WARM, edgecolor=WHITE, linewidth=0.45, zorder=4)
    ax.annotate("", xy=(named, 0.115), xytext=(q975, 0.115), arrowprops={"arrowstyle": "<->", "color": INK, "linewidth": 0.8})
    ax.text((named + q975) / 2, 0.14, "location margin", ha="center", fontsize=5.4, fontweight="bold")
    ax.text(q975, -0.10, "translated q97.5", color=BLUE_DARK, ha="right", fontsize=5.1)
    ax.text(named, -0.10, "named", color=WARM, ha="left", fontsize=5.1)
    ax.set_xlim(0.60, 0.95)
    ax.set_ylim(-0.16, 0.21)
    ax.set_yticks([])
    ax.set_xlabel("AUROC (schematic)")
    ax.spines["left"].set_visible(False)
    add_reference_grid(ax, "x")
    ax.text(0.01, 0.04, "Operational q97.5 spatial reference", transform=ax.transAxes, fontsize=5.0, color=GREY_DARK, ha="left", va="bottom")

    ax = fig.add_subplot(gs[2, 1])
    ax.set_axis_off()
    panel_label(ax, "d", x=-0.09, y=1.08)
    ax.text(0.0, 1.04, "Resource-specific evidence tiers", fontweight="bold", va="bottom")
    headers = [(0.01, "Resource"), (0.27, "Unit"), (0.51, "Access"), (0.72, "Maximum inference")]
    for x0, header in headers:
        ax.text(x0, 0.86, header, fontsize=5.1, fontweight="bold", color=GREY_DARK, va="center")
    tiers = [
        ("Simulation", "repetition", "public", "mechanism decision", TEAL_LIGHT),
        ("PD-DBS", "image", "restricted", "image-level stress test", WARM_LIGHT),
        ("YouTubePD", "clip", "source-dependent", "confounded clip audit", AMBER_LIGHT),
        ("PARK", "participant features", "public", "feature association", BLUE_LIGHT),
    ]
    for i, (resource, unit, access, claim, face) in enumerate(tiers):
        y0 = 0.66 - i * 0.19
        ax.add_patch(Rectangle((0.0, y0 - 0.075), 0.99, 0.15, facecolor=face if i in (0, 3) else GREY_PALE, edgecolor=GREY_LIGHT, linewidth=0.45))
        ax.add_patch(Rectangle((0.0, y0 - 0.075), 0.012, 0.15, facecolor=face, edgecolor="none"))
        ax.text(0.02, y0, resource, fontsize=5.2, fontweight="bold", va="center")
        ax.text(0.27, y0, unit, fontsize=4.9, va="center")
        ax.text(0.51, y0, access, fontsize=4.9, va="center")
        ax.text(0.72, y0, claim, fontsize=4.9, va="center")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.03, 1.02)

    title = "A matched spatial reference separates regional claims"
    fig.suptitle(title, x=0.005, y=1.025, ha="left", fontsize=9.2, fontweight="bold")
    return save_figure(fig, output_dir, "Figure1_MatchedSpatialReference", height_mm, title)


def make_figure_2(source_dir: Path, output_dir: Path) -> dict[str, object]:
    operating = load_csv(source_dir, "fig2_synthetic_operating_curve.csv")
    mechanisms = load_csv(source_dir, "fig2_mechanism_tasks.csv")
    height_mm = 132.0
    fig = new_figure(height_mm)
    gs = fig.add_gridspec(2, 2, width_ratios=(0.82, 1.18), height_ratios=(0.90, 1.10))

    ax = fig.add_subplot(gs[0, 0])
    panel_label(ax, "a")
    panel_title(ax, "Null operating repetitions")
    null = operating.loc[np.isclose(operating["effect"], 0.0)].iloc[0]
    x = float(null["detection_rate"])
    low = float(null["detection_rate_wilson_low"])
    high = float(null["detection_rate_wilson_high"])
    forest_errorbar(ax, np.array([x]), np.array([low]), np.array([high]), np.array([0.0]), WARM, marker="D")
    ax.axvline(0.025, color=GREY_DARK, linestyle=(0, (3, 2)), linewidth=0.8)
    ax.text(x, 0.12, "2/30", ha="center", color=WARM, fontweight="bold", fontsize=7.0)
    ax.text(0.025, -0.17, "nominal 0.025\nreference", ha="center", va="top", fontsize=5.1, color=GREY_DARK)
    ax.set_xlim(0, 0.33)
    ax.set_ylim(-0.28, 0.28)
    ax.set_yticks([])
    ax.set_xlabel("Detection proportion (Wilson 95% interval)")
    ax.spines["left"].set_visible(False)
    add_reference_grid(ax, "x")
    unit_note(ax, "Unit: independent synthetic repetition.", y=-0.30)

    ax = fig.add_subplot(gs[0, 1])
    panel_label(ax, "b", x=-0.08)
    panel_title(ax, "Injected-effect operating curve")
    signal = operating.loc[operating["effect"] > 0].sort_values("effect")
    xpos = np.arange(len(signal))
    rate = signal["detection_rate"].to_numpy(float)
    lo = signal["detection_rate_wilson_low"].to_numpy(float)
    hi = signal["detection_rate_wilson_high"].to_numpy(float)
    ax.fill_between(xpos, lo, hi, color=BLUE_LIGHT, alpha=0.55, linewidth=0)
    ax.plot(xpos, rate, color=BLUE_DARK, linewidth=1.25, marker="o", markersize=3.8, markerfacecolor=WHITE, markeredgewidth=0.9)
    for i, row in enumerate(signal.itertuples(index=False)):
        ypos = min(float(row.detection_rate) + 0.085, 1.055)
        ax.text(i, ypos, f"{int(row.detections)}/{int(row.repetitions)}", ha="center", va="bottom", fontsize=5.1, color=INK)
    ax.set_xticks(xpos, [f"{v:g}" for v in signal["effect"]])
    ax.set_ylim(-0.03, 1.12)
    ax.set_ylabel("Detection proportion")
    ax.set_xlabel("Injected local effect")
    add_reference_grid(ax, "y")
    unit_note(ax, "Unit: synthetic repetition. n=15 per positive effect.", y=-0.30)

    ax = fig.add_subplot(gs[1, 0])
    panel_label(ax, "c", x=-0.10)
    ax.set_axis_off()
    ax.text(0.0, 1.02, "Known synthetic mechanisms", fontweight="bold", va="bottom")
    maps: list[tuple[str, np.ndarray, str]] = []
    local = np.zeros((16, 16)); local[5:11, 5:11] = 1.0
    border = np.zeros((16, 16)); border[[0, 1, -2, -1], :] = 1.0; border[:, [0, 1, -2, -1]] = 1.0
    yy, xx = np.mgrid[0:16, 0:16]
    distributed = (np.cos((xx - 7.5) * np.pi / 8) + np.cos((yy - 7.5) * np.pi / 8) + 2) / 4
    sparse = np.zeros((16, 16)); sparse[[2, 3, 7, 10, 12, 14], [12, 4, 2, 13, 7, 10]] = 1.0
    maps.extend(
        [
            ("Local target", local, WARM),
            ("Border shortcut", border, AMBER),
            ("Distributed field", distributed, BLUE),
            ("Unknown sparse", sparse, TEAL),
        ]
    )
    positions = [(0.00, 0.51), (0.52, 0.51), (0.00, 0.00), (0.52, 0.00)]
    for (label, values, color), (x0, y0) in zip(maps, positions):
        iax = ax.inset_axes([x0, y0 + 0.07, 0.40, 0.36])
        cmap = LinearSegmentedColormap.from_list(f"map_{label}", [GREY_PALE, color])
        iax.imshow(values, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        iax.add_patch(Rectangle((4.5, 4.5), 6, 6, facecolor="none", edgecolor=INK, linewidth=0.65, linestyle=(0, (2, 1.4))))
        iax.set_xticks([]); iax.set_yticks([])
        for spine in iax.spines.values():
            spine.set_visible(True); spine.set_color(GREY); spine.set_linewidth(0.45)
        ax.text(x0 + 0.20, y0 + 0.03, label, ha="center", va="top", fontsize=5.6, fontweight="bold", color=INK)
    ax.text(0.0, -0.06, "Dashed square: named target support.", fontsize=5.2, color=GREY_DARK)

    ax = fig.add_subplot(gs[1, 1])
    panel_label(ax, "d", x=-0.08)
    panel_title(ax, "Mechanism-specific decisions")
    task_order = ["roi_localised", "border_shortcut", "distributed_low_frequency", "random_support"]
    task_labels = ["Local target", "Border shortcut", "Distributed field", "Unknown sparse"]
    column_keys = ["target_roi_auroc", "border_auroc", "pool_8x8_auroc", "pool_1x1_auroc"]
    column_labels = ["Target\nAUROC", "Border\nAUROC", "8 x 8\npooling", "1 x 1\nmean"]
    indexed = mechanisms.set_index("task").loc[task_order]
    matrix = indexed[column_keys].to_numpy(float)
    cmap = LinearSegmentedColormap.from_list("metric", [GREY_PALE, BLUE_LIGHT, BLUE_DARK])
    ax.imshow(matrix, cmap=cmap, vmin=0.48, vmax=1.0, aspect="auto")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=5.5, color=WHITE if value > 0.83 else INK, fontweight="bold" if value > 0.95 else "normal")
    gate = indexed["target_exceeds_random_q975"].astype(bool).to_numpy()
    for i, passed in enumerate(gate):
        ax.scatter([4.55], [i], s=34, facecolor=TEAL if passed else WHITE, edgecolor=TEAL if passed else GREY, linewidth=0.9)
        ax.text(4.55, i, "YES" if passed else "NO", ha="center", va="center", fontsize=4.2, fontweight="bold", color=WHITE if passed else GREY_DARK)
    ax.text(4.55, -0.83, "Location\ngate", ha="center", va="bottom", fontsize=5.3, fontweight="bold")
    ax.set_xticks(np.arange(4), column_labels)
    ax.set_yticks(np.arange(4), task_labels)
    ax.tick_params(length=0)
    ax.set_xlim(-0.5, 5.15)
    for spine in ax.spines.values():
        spine.set_visible(False)
    unit_note(ax, "Operating check for the specified model and synthetic generator.", y=-0.25)

    title = "Operating experiments distinguish simulated mechanisms"
    fig.suptitle(title, x=0.005, y=1.025, ha="left", fontsize=9.2, fontweight="bold")
    return save_figure(fig, output_dir, "Figure2_SyntheticOperatingExperiments", height_mm, title)


def make_figure_3(source_dir: Path, output_dir: Path) -> dict[str, object]:
    similarity = load_csv(source_dir, "fig3_similarity_filter.csv")
    full_image = load_csv(source_dir, "fig3_full_image_metrics.csv")
    roi = load_csv(source_dir, "fig3_pd_roi_summary.csv").set_index("roi_name").loc[ROI_ORDER].reset_index()
    supports = load_csv(source_dir, "fig3_pd_translated_supports.csv")
    random_budget = load_csv(source_dir, "fig3_random_pixel_budgets.csv")
    controls = load_csv(source_dir, "fig3_distributed_controls.csv")
    detector = load_csv(source_dir, "fig3_detector_conditions.csv")
    shuffle = load_csv(source_dir, "fig3_detector_shuffle.csv")
    full_auroc = float(full_image.loc[full_image["set"] == "full_test", "auroc"].iloc[0])

    height_mm = 176.0
    fig = new_figure(height_mm, constrained=False)
    gs = fig.add_gridspec(3, 2, width_ratios=(0.90, 1.10), height_ratios=(0.84, 1.18, 1.05))
    fig.subplots_adjust(left=0.125, right=0.985, top=0.95, bottom=0.085, wspace=0.48, hspace=0.82)

    ax = fig.add_subplot(gs[0, 0])
    panel_label(ax, "a")
    panel_title(ax, "Image-level discrimination after similarity exclusion")
    values = [full_auroc] + similarity["auroc"].tolist()
    remaining = [1171] + similarity["remaining_test_samples"].astype(int).tolist()
    labels = ["Full"] + [f"<{v:g}" for v in similarity["max_cosine_exclusion_threshold"]]
    labels = [f"{label}\nn={n}" for label, n in zip(labels, remaining)]
    x = np.arange(len(values))
    ax.plot(x, values, color=BLUE_DARK, linewidth=1.15, marker="o", markersize=3.6, markerfacecolor=WHITE, markeredgewidth=0.8)
    ax.set_xticks(x, labels, rotation=38, ha="right")
    ax.set_ylim(0.55, 1.04)
    ax.set_ylabel("AUROC")
    ax.set_xlabel("Maximum cosine-similarity exclusion")
    add_reference_grid(ax, "y")
    ax.text(0.02, 0.04, "Unit: held-out image. Post hoc exclusion.", transform=ax.transAxes, fontsize=4.9, color=GREY_DARK, ha="left", va="bottom")

    ax = fig.add_subplot(gs[0:2, 1])
    panel_label(ax, "b", x=-0.08, y=1.03)
    panel_title(ax, "Named supports versus 64 translated supports")
    for i, name in enumerate(ROI_ORDER):
        values_i = supports.loc[supports["roi_name"] == name, "auroc_mean_over_model_seeds"].to_numpy(float)
        offsets = np.linspace(-0.25, 0.25, len(values_i))
        ax.scatter(i + offsets, values_i, s=5.5, facecolor=GREY_LIGHT, edgecolor=GREY, linewidth=0.22, alpha=0.90, zorder=1)
        row = roi.loc[roi["roi_name"] == name].iloc[0]
        q = float(row["translated_auroc_q975"])
        named = float(row["named_roi_auroc_mean"])
        ax.plot([i - 0.28, i + 0.28], [q, q], color=BLUE_DARK, linewidth=1.25, zorder=3)
        ax.scatter(i, named, marker="D", s=22, facecolor=WARM, edgecolor=WHITE, linewidth=0.45, zorder=4)
    compact_ticks = ["Brow/FH", "L eye", "R eye", "Nasal", "L cheek", "R cheek", "Mouth", "Chin"]
    ax.set_xticks(np.arange(8), compact_ticks, rotation=35, ha="right")
    ax.set_ylim(0.68, 0.975)
    ax.set_ylabel("AUROC")
    add_reference_grid(ax, "y")
    handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=3.5, markerfacecolor=GREY_LIGHT, markeredgecolor=GREY, label="Translated support"),
        Line2D([], [], color=BLUE_DARK, linewidth=1.3, label="Translated q97.5"),
        Line2D([], [], marker="D", linestyle="none", markersize=4.2, markerfacecolor=WARM, markeredgecolor=WARM, label="Named support"),
    ]
    ax.legend(handles=handles, loc="lower left", ncol=3, columnspacing=1.0, handletextpad=0.45)
    unit_note(ax, "Unit: held-out image. Each AUROC averages three fresh model fits.", y=-0.18)

    ax = fig.add_subplot(gs[1, 0])
    panel_label(ax, "c")
    panel_title(ax, "All location margins are negative")
    margins = roi["named_minus_translated_q975"].to_numpy(float)
    y = np.arange(len(ROI_ORDER))
    ax.axvline(0, color=INK, linewidth=0.75)
    for yi, value in zip(y, margins):
        ax.hlines(yi, min(0, value), max(0, value), color=WARM_LIGHT, linewidth=2.8, zorder=1)
        ax.scatter(value, yi, marker="D", s=20, color=WARM, edgecolor=WHITE, linewidth=0.4, zorder=2)
    ax.set_yticks(y, [ROI_SHORT[v] for v in ROI_ORDER])
    ax.invert_yaxis()
    xmin = min(-0.13, float(margins.min()) - 0.02)
    ax.set_xlim(xmin, 0.025)
    ax.set_xlabel("Named AUROC - translated q97.5")
    ax.text(0.98, 0.05, "0/8", transform=ax.transAxes, ha="right", va="bottom", fontsize=8.2, fontweight="bold", color=WARM)
    add_reference_grid(ax, "x")
    unit_note(ax, "Location margins use descriptive q97.5 spatial references.", y=-0.26)

    ax = fig.add_subplot(gs[2, 0])
    panel_label(ax, "d")
    panel_title(ax, "Distributed and non-facial controls remain predictive")
    rows: list[dict[str, object]] = []
    for n in [32, 64, 128]:
        r = random_budget.loc[random_budget["n_pixels"] == n].iloc[0]
        rows.append({"label": f"Random pixels ({n})", "value": r["auroc_mean"], "low": r["auroc_min"], "high": r["auroc_max"], "color": BLUE})
    control_order = ["Outside ROI union", "One-pixel border", "Random third 1", "Random third 2", "Random third 3", "Pooling 8 x 8", "Global mean"]
    for label in control_order:
        r = controls.loc[controls["control"] == label].iloc[0]
        rows.append({"label": label, "value": r["auroc"], "low": r["auroc"], "high": r["auroc"], "color": AMBER if "Pooling" not in label and "mean" not in label else TEAL})
    yy = np.arange(len(rows))
    ax.axvline(full_auroc, color=GREY_DARK, linestyle=(0, (3, 2)), linewidth=0.8)
    for yi, row in zip(yy, rows):
        value = float(row["value"]); low = float(row["low"]); high = float(row["high"])
        ax.errorbar(value, yi, xerr=np.array([[value - low], [high - value]]), fmt="o", markersize=3.4, markerfacecolor=WHITE, markeredgecolor=str(row["color"]), color=str(row["color"]), linewidth=0.75, capsize=1.7)
    ax.set_yticks(yy, [str(row["label"]) for row in rows])
    ax.invert_yaxis()
    ax.set_xlim(0.47, 1.005)
    ax.set_xlabel("AUROC")
    add_reference_grid(ax, "x")
    unit_note(ax, "Random-support bars: min-max across 50 draws. Other controls: point estimates.", y=-0.25)

    ax = fig.add_subplot(gs[2, 1])
    panel_label(ax, "e", x=-0.08)
    panel_title(ax, "Detector geometry and reassigned masks")
    condition_order = [
        "full_image",
        "observed_detector_exterior",
        "detector_box_and_landmarks_only",
        "detector_box_features_only",
        "detector_binary_mask_only",
    ]
    condition_labels = ["Full image", "Observed detector exterior", "Box + landmarks", "Box only", "Binary mask only"]
    colors = [INK, AMBER, BLUE_DARK, BLUE, GREY_DARK]
    y0 = np.arange(len(condition_order))
    for yi, key, color in zip(y0, condition_order, colors):
        r = detector.loc[detector["condition"] == key].iloc[0]
        forest_errorbar(ax, np.array([float(r["ensemble_auroc"])]), np.array([float(r["ensemble_auroc_ci_low"])]), np.array([float(r["ensemble_auroc_ci_high"])]), np.array([yi]), color)
    shuffle_labels = ["Mask reassignment: global", "Mask reassignment: label-stratified"]
    for j, (_, r) in enumerate(shuffle.iterrows(), start=len(condition_order)):
        value = float(r["auroc_mean"]); low = float(r["auroc_q025"]); high = float(r["auroc_q975"])
        forest_errorbar(ax, np.array([value]), np.array([low]), np.array([high]), np.array([j]), GREY, marker="s")
    ax.axvline(0.5, color=GREY_DARK, linestyle=(0, (3, 2)), linewidth=0.7)
    ax.set_yticks(np.arange(7), condition_labels + shuffle_labels)
    ax.invert_yaxis()
    ax.set_xlim(0.47, 1.005)
    ax.set_xlabel("AUROC (95% bootstrap CI or q2.5-q97.5)")
    add_reference_grid(ax, "x")
    unit_note(ax, "Unit: held-out image. Reassignment rows summarise 24 permutations.", y=-0.25)

    footer = "Unit: image. Available fields: image, numerical label and supplied split."
    fig.text(0.995, 0.003, footer, ha="right", va="bottom", fontsize=5.2, color=WARM, fontweight="bold")
    title = "A legacy image matrix reveals distributed image-level signal"
    fig.suptitle(title, x=0.005, y=1.025, ha="left", fontsize=9.2, fontweight="bold")
    return save_figure(fig, output_dir, "Figure3_LegacyImageMatrixAudit", height_mm, title)


def _plot_margin_pair(
    ax: plt.Axes,
    frame: pd.DataFrame,
    title: str,
    annotate_mouth: bool = False,
) -> None:
    indexed = frame.set_index("roi_name").loc[ROI_ORDER]
    holdout = indexed["holdout_location_margin"].to_numpy(float)
    repeated = indexed["repeated_cv_location_margin"].to_numpy(float)
    y = np.arange(len(ROI_ORDER))
    ax.axvline(0, color=INK, linewidth=0.75)
    for yi, a, b in zip(y, holdout, repeated):
        ax.plot([a, b], [yi, yi], color=GREY_LIGHT, linewidth=1.5, zorder=1)
    ax.scatter(holdout, y - 0.10, marker="D", s=19, facecolor=WARM, edgecolor=WHITE, linewidth=0.35, label="Spreadsheet holdout", zorder=3)
    ax.scatter(repeated, y + 0.10, marker="o", s=20, facecolor=WHITE, edgecolor=BLUE_DARK, linewidth=0.85, label="Repeated CV", zorder=3)
    ax.set_yticks(y, [ROI_SHORT[v] for v in ROI_ORDER])
    ax.invert_yaxis()
    ax.set_xlim(min(-0.26, float(min(holdout.min(), repeated.min())) - 0.015), max(0.03, float(max(holdout.max(), repeated.max())) + 0.015))
    ax.set_xlabel("Named AUROC - translated q97.5")
    ax.set_title(title, loc="left", pad=14, fontweight="bold")
    add_reference_grid(ax, "x")
    if annotate_mouth:
        idx = ROI_ORDER.index("perioral_mouth")
        ax.annotate(
            f"+{holdout[idx]:.3f}",
            xy=(holdout[idx], idx - 0.10),
            xytext=(0.018, idx - 0.75),
            fontsize=5.3,
            color=WARM,
            fontweight="bold",
            arrowprops={"arrowstyle": "-", "color": WARM, "linewidth": 0.7},
        )


def make_figure_4(source_dir: Path, output_dir: Path) -> dict[str, object]:
    flow = load_csv(source_dir, "fig4_cohort_flow.csv")
    primary = load_csv(source_dir, "fig4_youtubepd_roi_summary.csv")
    primary = primary.loc[primary["mode"] == "combined"].copy()
    rbf = load_csv(source_dir, "fig4_rbf_roi_summary.csv")
    controls = load_csv(source_dir, "fig4_competing_supports.csv")
    years = load_csv(source_dir, "fig4_year_distribution.csv")
    time_summary = load_csv(source_dir, "fig4_collection_time_summary.csv")

    height_mm = 184.0
    fig = new_figure(height_mm, constrained=False)
    gs = fig.add_gridspec(3, 2, width_ratios=(1.0, 1.0), height_ratios=(0.58, 1.16, 1.30))
    fig.subplots_adjust(left=0.125, right=0.985, top=0.96, bottom=0.09, wspace=0.48, hspace=0.72)

    ax = fig.add_subplot(gs[0, :])
    ax.set_axis_off()
    panel_label(ax, "a", x=-0.025, y=1.06)
    ax.text(0.0, 1.03, "Public-video reconstruction and locked cohort flow", fontweight="bold", va="top")
    n_stages = len(flow)
    box_w = 0.15
    gap = (0.96 - n_stages * box_w) / (n_stages - 1)
    for i, row in enumerate(flow.itertuples(index=False)):
        x0 = 0.01 + i * (box_w + gap)
        color = BLUE_LIGHT if i in (2, 3, 4) else GREY_LIGHT
        edge = BLUE_DARK if i in (2, 3, 4) else GREY_DARK
        ax.add_patch(FancyBboxPatch((x0, 0.18), box_w, 0.55, boxstyle="round,pad=0.012,rounding_size=0.012", facecolor=color, edgecolor=edge, linewidth=0.8))
        ax.text(x0 + box_w / 2, 0.53, f"{int(row.n)}", ha="center", va="center", fontsize=10, fontweight="bold", color=INK)
        ax.text(x0 + box_w / 2, 0.31, str(row.stage).replace(" ", "\n", 1), ha="center", va="center", fontsize=5.4, color=INK)
        if i < n_stages - 1:
            arrow = FancyArrowPatch((x0 + box_w + 0.006, 0.455), (x0 + box_w + gap - 0.006, 0.455), arrowstyle="-|>", mutation_scale=7, color=GREY_DARK, linewidth=0.7)
            ax.add_patch(arrow)
    ax.text(0.99, 0.02, "Identifiable frames excluded from display", ha="right", va="bottom", fontsize=5.1, color=GREY_DARK)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax = fig.add_subplot(gs[1, 0])
    panel_label(ax, "b")
    _plot_margin_pair(ax, primary, "Primary logistic location audit")
    ax.text(0.98, 0.97, "0/8 both gates", transform=ax.transAxes, ha="right", va="top", color=WARM, fontweight="bold", fontsize=7.0, bbox={"facecolor": WHITE, "edgecolor": "none", "pad": 1.5})
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.005), ncol=2, columnspacing=0.9, handletextpad=0.4)
    unit_note(ax, "Unit: video clip. Repeated CV includes spreadsheet-holdout clips.", y=-0.23)

    ax = fig.add_subplot(gs[1, 1])
    panel_label(ax, "c", x=-0.08)
    _plot_margin_pair(ax, rbf, "Post hoc RBF-SVM sensitivity", annotate_mouth=True)
    ax.text(0.98, 0.97, "1/8 holdout; 0/8 both", transform=ax.transAxes, ha="right", va="top", color=WARM, fontweight="bold", fontsize=7.0, bbox={"facecolor": WHITE, "edgecolor": "none", "pad": 1.5})
    unit_note(ax, "Mouth margin: +0.004 holdout and -0.053 repeated CV.", y=-0.23)

    ax = fig.add_subplot(gs[2, 0])
    panel_label(ax, "d")
    panel_title(ax, "Competing supports and acquisition metadata")
    order = ["aligned_face_normalized", "whole_frame_context", "context_with_face_masked", "aligned_middle_third", "acquisition_metadata_only"]
    labels = ["Aligned face", "Whole frame", "Face-masked context", "Middle third", "Acquisition metadata"]
    indexed = controls.set_index("control").loc[order]
    y = np.arange(len(order))
    hold = indexed["auroc"].to_numpy(float)
    lo = indexed["auroc_ci_low"].to_numpy(float)
    hi = indexed["auroc_ci_high"].to_numpy(float)
    rep = indexed["repeated_cv_auroc_mean"].to_numpy(float)
    rep_sd = indexed["repeated_cv_auroc_sd"].to_numpy(float)
    forest_errorbar(ax, hold, lo, hi, y - 0.10, BLUE_DARK, marker="D", label="Spreadsheet holdout")
    forest_errorbar(ax, rep, np.maximum(0.5, rep - rep_sd), np.minimum(1.0, rep + rep_sd), y + 0.10, AMBER, marker="o", label="Repeated CV mean +/- s.d.")
    ax.axvline(0.5, color=GREY_DARK, linestyle=(0, (3, 2)), linewidth=0.7)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0.48, 1.01)
    ax.set_xlabel("AUROC")
    add_reference_grid(ax, "x")
    ax.legend(loc="upper left", ncol=1, handletextpad=0.4)
    unit_note(ax, "Unit: video clip. Holdout bootstrap interval and repeated-CV fold s.d.", y=-0.24)

    egrid = gs[2, 1].subgridspec(2, 1, height_ratios=(1.0, 1.0), hspace=0.78)
    ax_year = fig.add_subplot(egrid[0, 0])
    panel_label(ax_year, "e", x=-0.15, y=1.16)
    panel_title(ax_year, "Collection time is class-associated")
    bins = np.arange(1965, 2026, 5)
    centers = (bins[:-1] + bins[1:]) / 2
    for label, color, fill in [(0, BLUE_DARK, BLUE_LIGHT), (1, WARM, WARM_LIGHT)]:
        sub = years.loc[years["label"] == label]
        counts, _ = np.histogram(sub["year"].to_numpy(float), bins=bins, weights=sub["clips"].to_numpy(float))
        ax_year.step(centers, counts, where="mid", color=color, linewidth=1.1, label=f"Class {label}")
        ax_year.fill_between(centers, counts, step="mid", color=fill, alpha=0.55, linewidth=0)
    ax_year.set_ylabel("Clips / 5 years")
    ax_year.set_xlim(1965, 2025)
    ax_year.legend(loc="upper left", ncol=2, handlelength=1.4, columnspacing=0.9)
    add_reference_grid(ax_year, "y")

    ax_time = fig.add_subplot(egrid[1, 0])
    order_time = ["year_only_spreadsheet_holdout", "year_only_three_year_caliper_grouped_cv"]
    indexed_time = time_summary.set_index("analysis").loc[order_time]
    vals = indexed_time["auroc"].to_numpy(float)
    y2 = np.array([0, 1])
    first = indexed_time.iloc[0]
    ax_time.errorbar(vals[0], y2[0], xerr=np.array([[vals[0] - float(first["auroc_ci_low"])], [float(first["auroc_ci_high"]) - vals[0]]]), fmt="D", color=WARM, markerfacecolor=WARM, markeredgecolor=WHITE, markeredgewidth=0.4, markersize=4.2, linewidth=0.8, capsize=1.8)
    ax_time.scatter(vals[1], y2[1], marker="o", s=24, facecolor=WHITE, edgecolor=BLUE_DARK, linewidth=0.9)
    ax_time.axvline(0.5, color=GREY_DARK, linestyle=(0, (3, 2)), linewidth=0.7)
    ax_time.set_yticks(y2, ["Year-only holdout", "Year-matched grouped CV"])
    ax_time.invert_yaxis()
    ax_time.set_xlim(0.45, 1.01)
    ax_time.set_xlabel("AUROC")
    ax_time.text(vals[1] + 0.015, 1, "21 pairs, 50 folds", va="center", fontsize=5.0, color=GREY_DARK)
    add_reference_grid(ax_time, "x")
    ax_time.text(0.99, 0.52, "Exploratory clip-pair analysis", transform=ax_time.transAxes, ha="right", va="center", fontsize=4.8, color=GREY_DARK, bbox={"facecolor": WHITE, "edgecolor": "none", "pad": 1.0})

    footer = "Unit: video clip. Spreadsheet contains no participant key."
    fig.text(0.985, 0.018, footer, ha="right", va="bottom", fontsize=5.2, color=WARM, fontweight="bold")
    title = "A public-video stress test returns the same primary location decision"
    fig.suptitle(title, x=0.005, y=1.025, ha="left", fontsize=9.2, fontweight="bold")
    return save_figure(fig, output_dir, "Figure4_PublicVideoStressTest", height_mm, title)


def make_figure_5(source_dir: Path, output_dir: Path) -> dict[str, object]:
    benchmark = load_csv(source_dir, "fig5_park_benchmark.csv")
    groups = load_csv(source_dir, "fig5_park_feature_groups.csv")
    evidence = load_csv(source_dir, "fig5_evidence_matrix.csv")

    height_mm = 143.0
    fig = new_figure(height_mm)
    gs = fig.add_gridspec(3, 2, height_ratios=(1.0, 1.13, 0.62), width_ratios=(0.78, 1.22))

    ax = fig.add_subplot(gs[0, 0])
    panel_label(ax, "a")
    panel_title(ax, "PARK participant benchmark")
    observed = benchmark.loc[benchmark["condition"] == "Observed all features"].iloc[0]
    shuffled = benchmark.loc[benchmark["condition"] == "Training-label shuffle mean"].iloc[0]
    forest_errorbar(ax, np.array([float(observed["auroc"])]), np.array([float(observed["low"])]), np.array([float(observed["high"])]), np.array([0]), BLUE_DARK, marker="D")
    ax.hlines(1, 0.5, float(shuffled["high"]), color=GREY, linewidth=1.1)
    ax.scatter(float(shuffled["auroc"]), 1, marker="s", s=22, facecolor=WHITE, edgecolor=GREY_DARK, linewidth=0.8)
    ax.axvline(0.5, color=GREY_DARK, linestyle=(0, (3, 2)), linewidth=0.7)
    ax.set_yticks([0, 1], ["Observed 42 features", "Training-label shuffle"])
    ax.invert_yaxis()
    ax.set_xlim(0.47, 0.96)
    ax.set_xlabel("AUROC (participant bootstrap CI / shuffle q97.5)")
    ax.text(float(observed["auroc"]) + 0.014, 0, f"{float(observed['auroc']):.3f}", ha="left", va="center", fontsize=6.1, color=BLUE_DARK, fontweight="bold")
    n_test = int(observed["n_test_participants"])
    ax.text(0.98, 0.05, f"n={n_test} test participants", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.3, color=GREY_DARK)
    add_reference_grid(ax, "x")
    unit_note(ax, "Unit: participant. Released extracted smile features.", y=-0.32)

    ax = fig.add_subplot(gs[0, 1])
    panel_label(ax, "b", x=-0.08)
    panel_title(ax, "Predefined feature-family sensitivities")
    order = ["mean_statistics", "variance_statistics", "entropy_statistics", "action_unit_features", "geometric_features", "upper_face_features", "lower_face_features"]
    labels = ["Mean statistics", "Variance statistics", "Entropy statistics", "Action-unit features", "Geometric features", "Upper-face features", "Lower-face features"]
    indexed = groups.set_index("feature_group").loc[order]
    values = indexed["auroc"].to_numpy(float)
    low = indexed["auroc_ci_low"].to_numpy(float)
    high = indexed["auroc_ci_high"].to_numpy(float)
    y = np.arange(len(order))
    forest_errorbar(ax, values, low, high, y, BLUE, marker="o")
    ax.axvline(0.5, color=GREY_DARK, linestyle=(0, (3, 2)), linewidth=0.7)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0.48, 0.97)
    ax.set_xlabel("AUROC (participant bootstrap 95% CI)")
    add_reference_grid(ax, "x")
    unit_note(ax, "Participant-level sensitivity of released feature families.", y=-0.24)

    ax = fig.add_subplot(gs[1, :])
    ax.set_axis_off()
    panel_label(ax, "c", x=-0.025, y=1.04)
    ax.text(0.0, 1.02, "Evidence and access boundaries", fontweight="bold", va="bottom")
    columns = [
        ("resource", "Resource", 0.01, 0.12),
        ("unit", "Unit", 0.13, 0.10),
        ("participant_key", "Participant key", 0.23, 0.13),
        ("raw_pixels", "Raw pixels", 0.36, 0.15),
        ("location_test", "Location test", 0.51, 0.12),
        ("shortcut_controls", "Shortcut controls", 0.63, 0.14),
        ("publicly_executable", "Public execution", 0.77, 0.12),
        ("maximum_claim", "Maximum claim", 0.89, 0.105),
    ]
    header_y = 0.78
    row_y = [0.57, 0.35, 0.13]
    ax.add_patch(Rectangle((0.005, 0.70), 0.99, 0.17, facecolor=INK, edgecolor="none"))
    for key, header, x0, width in columns:
        ax.text(x0 + 0.004, header_y, header, ha="left", va="center", color=WHITE, fontsize=5.2, fontweight="bold")
    value_maps = {
        "yes": ("YES", TEAL_LIGHT, TEAL),
        "no": ("NO", GREY_LIGHT, GREY_DARK),
        "source-dependent": ("SOURCE-DEP.", AMBER_LIGHT, AMBER),
        "label shuffle": ("LABEL SHUFFLE", BLUE_LIGHT, BLUE_DARK),
    }
    for i, row in evidence.iterrows():
        y0 = row_y[i]
        bg = GREY_PALE if i % 2 == 0 else WHITE
        ax.add_patch(Rectangle((0.005, y0 - 0.09), 0.99, 0.18, facecolor=bg, edgecolor=GREY_LIGHT, linewidth=0.45))
        for key, _, x0, width in columns:
            raw = str(row[key])
            if raw in value_maps:
                text_value, face, color = value_maps[raw]
                ax.add_patch(FancyBboxPatch((x0 + 0.003, y0 - 0.043), min(width - 0.01, 0.105), 0.086, boxstyle="round,pad=0.006,rounding_size=0.008", facecolor=face, edgecolor="none"))
                ax.text(x0 + 0.006 + min(width - 0.01, 0.105) / 2, y0, text_value, ha="center", va="center", fontsize=4.4, color=color, fontweight="bold")
            else:
                display = raw.replace("yes, restricted", "restricted").replace("URLs only", "URLs").replace("image-level stress test", "image-level\nstress test").replace("confounded clip-level audit", "confounded clip-\nlevel audit").replace("feature association", "feature\nassociation")
                ax.text(x0 + 0.004, y0, display, ha="left", va="center", fontsize=5.1, color=INK, fontweight="bold" if key == "resource" else "normal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax = fig.add_subplot(gs[2, :])
    ax.set_axis_off()
    panel_label(ax, "d", x=-0.025, y=1.04)
    ax.text(0.0, 1.02, "Permissible inference", fontweight="bold", va="bottom")
    bands = [
        (0.01, 0.30, TEAL_LIGHT, TEAL, "OBSERVED", "Predictive discrimination\nin the tested resources"),
        (0.335, 0.31, WARM_LIGHT, WARM, "PRIMARY RESULT", "0/8 primary location gates\nin both raw-data applications"),
        (0.68, 0.31, GREY_LIGHT, GREY_DARK, "SEPARATE EVIDENCE", "Faithfulness, causal anatomy and\nclinical utility"),
    ]
    for x0, width, face, edge, header, body in bands:
        ax.add_patch(FancyBboxPatch((x0, 0.18), width, 0.62, boxstyle="round,pad=0.012,rounding_size=0.012", facecolor=face, edgecolor=edge, linewidth=0.8))
        ax.text(x0 + 0.015, 0.65, header, ha="left", va="center", fontsize=5.0, color=edge, fontweight="bold")
        ax.text(x0 + 0.015, 0.40, body, ha="left", va="center", fontsize=6.0, color=INK, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    title = "Experimental unit determines the permissible claim"
    fig.suptitle(title, x=0.005, y=1.025, ha="left", fontsize=9.2, fontweight="bold")
    return save_figure(fig, output_dir, "Figure5_EvidenceBoundaries", height_mm, title)


def make_extended_data_figure_1(source_dir: Path, output_dir: Path) -> dict[str, object]:
    drift = load_csv(source_dir, "extended_youtubepd_reconstruction_drift.csv")
    common = load_csv(source_dir, "extended_youtubepd_common_cohort.csv")

    height_mm = 88.0
    fig = new_figure(height_mm)
    gs = fig.add_gridspec(1, 3, width_ratios=(0.92, 1.12, 1.10))
    names = ["Frozen\n2026-07-09", "Fresh\n2026-07-10"]
    colors = [BLUE_DARK, WARM]

    ax = fig.add_subplot(gs[0, 0])
    panel_label(ax, "a")
    panel_title(ax, "Reconstruction yield")
    x = np.arange(2)
    for j, column in enumerate(["cohort_before_qc", "after_qc", "test_clips"]):
        offset = (j - 1) * 0.20
        vals = drift[column].to_numpy(float)
        ax.bar(x + offset, vals, width=0.18, color=[BLUE_LIGHT, WARM_LIGHT], edgecolor=[BLUE_DARK, WARM], linewidth=0.6, alpha=1.0 if j == 0 else 0.72)
        for xi, value in zip(x + offset, vals):
            ax.text(xi, value + 2.0, f"{int(value)}", ha="center", va="bottom", fontsize=4.8)
    ax.set_xticks(x, names)
    ax.set_ylabel("Video clips")
    ax.set_ylim(0, 135)
    add_reference_grid(ax, "y")
    ax.text(0.02, 0.97, "bars: cohort / QC / test", transform=ax.transAxes, va="top", fontsize=4.8, color=GREY_DARK)

    ax = fig.add_subplot(gs[0, 1])
    panel_label(ax, "b", x=-0.08)
    panel_title(ax, "Predictive AUROC drifts with encoding")
    all_vals = drift["aligned_face_auroc"].to_numpy(float)
    logistic_common = common["LogisticFull"].to_numpy(float)
    rbf_common = common["RbfFull"].to_numpy(float)
    metric_x = np.arange(3)
    width = 0.26
    for i in range(2):
        vals = [all_vals[i], logistic_common[i], rbf_common[i]]
        ax.bar(metric_x + (i - 0.5) * width, vals, width=width, color=BLUE_LIGHT if i == 0 else WARM_LIGHT, edgecolor=colors[i], linewidth=0.75, label=names[i].replace("\n", " "))
    ax.axhline(0.5, color=GREY_DARK, linestyle=(0, (3, 2)), linewidth=0.7)
    ax.set_xticks(metric_x, ["Full cohort\nlogistic", "Same-ID\nlogistic", "Same-ID\nRBF-SVM"])
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.48, 1.0)
    ax.legend(loc="lower left")
    add_reference_grid(ax, "y")
    unit_note(ax, "Same-ID cohort: 102 clips; same split.", y=-0.28)

    ax = fig.add_subplot(gs[0, 2])
    panel_label(ax, "c", x=-0.08)
    panel_title(ax, "Location decisions remain stable")
    gate_columns = ["primary_holdout_gates", "primary_both_gates", "rbf_holdout_gates", "rbf_both_gates"]
    gate_labels = ["Logistic\nholdout", "Logistic\nboth", "RBF\nholdout", "RBF\nboth"]
    gx = np.arange(4)
    for i in range(2):
        vals = drift.loc[i, gate_columns].to_numpy(float)
        ax.bar(gx + (i - 0.5) * width, vals, width=width, color=BLUE_LIGHT if i == 0 else WARM_LIGHT, edgecolor=colors[i], linewidth=0.75)
    ax.set_xticks(gx, gate_labels)
    ax.set_ylabel("Named regions passing (of 8)")
    ax.set_ylim(0, 2.05)
    ax.set_yticks([0, 1, 2])
    add_reference_grid(ax, "y")
    ax.text(0.98, 0.96, "0 pass both gates", transform=ax.transAxes, ha="right", va="top", fontsize=5.3, color=WARM, fontweight="bold")
    unit_note(ax, "Fresh-source sensitivity from public URLs.", y=-0.28)

    title = "Fresh-source reconstruction changes discrimination while preserving the location verdict"
    fig.suptitle(title, x=0.005, y=1.04, ha="left", fontsize=9.2, fontweight="bold")
    return save_figure(fig, output_dir, "ExtendedDataFigure1_ReconstructionDrift", height_mm, title)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-data", type=Path, default=Path("source_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()

    configure_matplotlib()
    source_dir = args.source_data.resolve()
    output_dir = args.output_dir.resolve()
    builders = [
        make_figure_1,
        make_figure_2,
        make_figure_3,
        make_figure_4,
        make_figure_5,
        make_extended_data_figure_1,
    ]
    manifest = [builder(source_dir, output_dir) for builder in builders]
    manifest_path = output_dir / "figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Rendered {len(manifest)} publication figures to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Calibrate matched-support audit behaviour across independent simulations.

The original synthetic examples were single realizations and are retained only
as qualitative unit tests. This script estimates support-level false-positive
and detection rates across independently generated datasets. A central 8 x 8
target is compared with translated 8 x 8 rectangles disjoint from the target. Every support in
one simulation uses the same split and model seed, so the comparison changes
location without changing topology, feature count, or optimizer randomness.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.append(str(_HERE))
sys.path.append(str(_HERE.parent / "data"))

from train_baseline_mlp_numpy import (
    fit_standardizer,
    forward,
    metric_summary,
    standardize,
    train_mlp,
)


IMAGE_SIZE = 32
N = {"train": 600, "val": 200, "test": 600}
HIDDEN = 32
EPOCHS = 80
BATCH = 64
LR = 1e-3
L2 = 1e-4


def target_mask() -> np.ndarray:
    mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)
    mask[12:20, 12:20] = True
    return mask


def rectangle_mask(y0: int, x0: int) -> np.ndarray:
    mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)
    mask[y0 : y0 + 8, x0 : x0 + 8] = True
    return mask


def make_dataset(effect: float, seed: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    mask = target_mask()
    output = {}
    for split, n_samples in N.items():
        labels = np.arange(n_samples, dtype=np.int64) % 2
        rng.shuffle(labels)
        images = rng.normal(
            0.0, 1.0, size=(n_samples, IMAGE_SIZE, IMAGE_SIZE)
        ).astype(np.float32)
        sign = 2.0 * labels.astype(np.float32) - 1.0
        images[:, mask] += effect * sign[:, None]
        output[split] = (images, labels)
    return output


def fit_eval(
    dataset: dict[str, tuple[np.ndarray, np.ndarray]],
    mask: np.ndarray,
    model_seed: int,
) -> float:
    columns = np.flatnonzero(mask.reshape(-1))
    train_images, train_labels = dataset["train"]
    val_images, val_labels = dataset["val"]
    test_images, test_labels = dataset["test"]
    x_train = train_images.reshape(len(train_images), -1)[:, columns]
    x_val = val_images.reshape(len(val_images), -1)[:, columns]
    x_test = test_images.reshape(len(test_images), -1)[:, columns]
    mean, std = fit_standardizer(x_train)
    model, _ = train_mlp(
        standardize(x_train, mean, std),
        train_labels,
        standardize(x_val, mean, std),
        val_labels,
        hidden=HIDDEN,
        epochs=EPOCHS,
        batch_size=BATCH,
        lr=LR,
        l2=L2,
        seed=model_seed,
    )
    probability = forward(model, standardize(x_test, mean, std))[0]
    return float(metric_summary(test_labels, probability)["auroc"])


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2.0 * total)) / denominator
    half_width = (
        z
        * np.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return float(centre - half_width), float(centre + half_width)


def parse_effects(value: str) -> list[float]:
    effects = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not effects or any(effect < 0 for effect in effects):
        raise argparse.ArgumentTypeError("effects must be comma-separated non-negative values")
    return effects


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", default="outputs/nmi_synthetic_calibration"
    )
    parser.add_argument("--effects", type=parse_effects, default=parse_effects("0,0.05,0.1,0.2,0.4"))
    parser.add_argument("--null-repetitions", type=int, default=30)
    parser.add_argument("--signal-repetitions", type=int, default=15)
    parser.add_argument("--translated-supports", type=int, default=99)
    args = parser.parse_args()

    if args.null_repetitions < 1 or args.signal_repetitions < 1:
        raise ValueError("repetition counts must be positive")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    named_mask = target_mask()
    candidates = []
    for y0 in range(IMAGE_SIZE - 8 + 1):
        for x0 in range(IMAGE_SIZE - 8 + 1):
            mask = rectangle_mask(y0, x0)
            if np.logical_and(mask, named_mask).any():
                continue
            candidates.append((y0, x0))
    if args.translated_supports > len(candidates):
        raise ValueError(
            f"requested {args.translated_supports} supports; only {len(candidates)} disjoint translations exist"
        )
    support_rng = np.random.default_rng(20260709)
    selected = support_rng.choice(
        len(candidates), size=args.translated_supports, replace=False
    )
    translated = [candidates[int(index)] for index in selected]

    simulation_rows = []
    fit_rows = []
    for effect_index, effect in enumerate(args.effects):
        repetitions = (
            args.null_repetitions if effect == 0.0 else args.signal_repetitions
        )
        for repetition in range(repetitions):
            data_seed = 20260709 + effect_index * 10000 + repetition
            model_seed = 42000 + effect_index * 1000 + repetition
            dataset = make_dataset(effect, data_seed)
            named_auc = fit_eval(dataset, named_mask, model_seed)
            null_aucs = []
            fit_rows.append(
                {
                    "effect": effect,
                    "repetition": repetition,
                    "support_kind": "named_target",
                    "support_id": "named",
                    "y0": 12,
                    "x0": 12,
                    "auroc": named_auc,
                }
            )
            for support_index, (y0, x0) in enumerate(translated, start=1):
                auc = fit_eval(dataset, rectangle_mask(y0, x0), model_seed)
                null_aucs.append(auc)
                fit_rows.append(
                    {
                        "effect": effect,
                        "repetition": repetition,
                        "support_kind": "disjoint_translated_rectangle",
                        "support_id": f"translated_{support_index:03d}",
                        "y0": y0,
                        "x0": x0,
                        "auroc": auc,
                    }
                )
            null_values = np.asarray(null_aucs)
            q975 = float(np.quantile(null_values, 0.975))
            rank_probability = (
                1.0 + float(np.sum(null_values >= named_auc))
            ) / (len(null_values) + 1.0)
            simulation_rows.append(
                {
                    "effect": effect,
                    "repetition": repetition,
                    "data_seed": data_seed,
                    "model_seed": model_seed,
                    "named_auroc": named_auc,
                    "translated_mean": float(null_values.mean()),
                    "translated_q975": q975,
                    "named_minus_translated_mean": float(
                        named_auc - null_values.mean()
                    ),
                    "named_minus_translated_q975": float(named_auc - q975),
                    "support_rank_probability": rank_probability,
                    "detected_above_q975": bool(named_auc > q975),
                }
            )
            print(
                f"effect={effect:.2f} repetition={repetition + 1:02d}/{repetitions:02d} "
                f"named={named_auc:.4f} q975={q975:.4f}"
            )

    simulation_df = pd.DataFrame(simulation_rows)
    fit_df = pd.DataFrame(fit_rows)
    summary_rows = []
    for effect, group in simulation_df.groupby("effect", sort=True):
        successes = int(group["detected_above_q975"].sum())
        total = int(len(group))
        low, high = wilson_interval(successes, total)
        summary_rows.append(
            {
                "effect": float(effect),
                "repetitions": total,
                "detections": successes,
                "detection_rate": float(successes / total),
                "detection_rate_wilson_low": low,
                "detection_rate_wilson_high": high,
                "named_auroc_mean": float(group["named_auroc"].mean()),
                "named_auroc_sd": float(group["named_auroc"].std(ddof=1)),
                "translated_q975_mean": float(group["translated_q975"].mean()),
                "margin_mean": float(
                    group["named_minus_translated_q975"].mean()
                ),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    simulation_df.to_csv(
        output_dir / "synthetic_calibration_simulations.csv", index=False
    )
    fit_df.to_csv(output_dir / "synthetic_calibration_fits.csv", index=False)
    summary_df.to_csv(
        output_dir / "synthetic_calibration_summary.csv", index=False
    )

    verdict = {
        "protocol": {
            "image_size": IMAGE_SIZE,
            "target_shape": "8x8 central rectangle",
            "translated_supports": args.translated_supports,
            "translated_supports_disjoint_from_target": True,
            "same_model_seed_within_simulation": True,
            "train_n": N["train"],
            "val_n": N["val"],
            "test_n": N["test"],
            "effects": args.effects,
            "null_repetitions": args.null_repetitions,
            "signal_repetitions": args.signal_repetitions,
        },
        "interpretation": (
            "Effect 0 estimates support-level false detection under simulated "
            "exchangeability; positive effects estimate sensitivity for this "
            "specific data-generating process and model."
        ),
        "rows": summary_rows,
    }
    (output_dir / "nmi_synthetic_calibration_verdict.json").write_text(
        json.dumps(verdict, indent=2), encoding="utf-8"
    )
    lines = [
        "# Synthetic matched-support calibration",
        "",
        "Independent simulated datasets were evaluated with a central 8 x 8 target and 99 disjoint translated 8 x 8 controls. The named and control supports shared one model seed within each simulation.",
        "",
        "| Injected effect | Repetitions | Detections | Detection rate | 95% Wilson interval | Mean named AUROC | Mean null q97.5 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['effect']:.2f} | {row['repetitions']} | {row['detections']} | "
            f"{row['detection_rate']:.3f} | "
            f"{row['detection_rate_wilson_low']:.3f}-{row['detection_rate_wilson_high']:.3f} | "
            f"{row['named_auroc_mean']:.4f} | {row['translated_q975_mean']:.4f} |"
        )
    (output_dir / "nmi_synthetic_calibration_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(verdict, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

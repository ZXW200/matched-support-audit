"""Unit-test the claim-control audit on synthetic known-positive and shortcut tasks.

The purpose is methodological validation, not a new clinical result. Each task
has a known signal-generating support. The same retrain-based support audit used
for the PD-DBS case is then asked to distinguish anatomical localisation from
border shortcuts and distributed low-frequency structure.
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
    roc_curve,
    standardize,
    train_mlp,
)


N = {"train": 1200, "val": 400, "test": 1200}
IMAGE_SIZE = 32
HIDDEN = 32
EPOCHS = 120
BATCH = 64
LR = 1e-3
L2 = 1e-4


def support_masks() -> dict[str, np.ndarray]:
    target = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)
    target[12:20, 12:20] = True
    border = np.zeros_like(target)
    border[0, :] = border[-1, :] = True
    border[:, 0] = border[:, -1] = True
    return {"target_roi": target, "border": border}


def make_labels(rng: np.random.Generator, n: int) -> np.ndarray:
    y = np.arange(n, dtype=np.int64) % 2
    rng.shuffle(y)
    return y


def make_task(task: str, seed: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    masks = support_masks()
    output: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split, n in N.items():
        y = make_labels(rng, n)
        imgs = rng.normal(0.0, 1.0, size=(n, IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32)
        sign = (2.0 * y.astype(np.float32) - 1.0)[:, None]
        if task == "roi_localised":
            imgs[:, masks["target_roi"]] += 0.4 * sign
        elif task == "border_shortcut":
            imgs[:, masks["border"]] += 0.4 * sign
        elif task == "distributed_low_frequency":
            yy, xx = np.mgrid[0:IMAGE_SIZE, 0:IMAGE_SIZE]
            field = np.cos(2.0 * np.pi * xx / IMAGE_SIZE) + 0.7 * np.cos(2.0 * np.pi * yy / IMAGE_SIZE)
            field = field / np.std(field)
            imgs += 0.5 * sign[:, :, None] * field[None, :, :].astype(np.float32)
        elif task == "random_support":
            if "_support" not in output:
                support = rng.choice(IMAGE_SIZE * IMAGE_SIZE, size=64, replace=False)
                output["_support"] = (support,)
            support = output["_support"][0]
            imgs.reshape(n, -1)[:, support] += 0.4 * sign
        else:
            raise ValueError(f"unknown task: {task}")
        output[split] = (imgs, y)
    output.pop("_support", None)
    return output


def flat(imgs: np.ndarray) -> np.ndarray:
    return imgs.reshape(imgs.shape[0], -1).astype(np.float32)


def fit_eval(xtr_raw, ytr, xva_raw, yva, xte_raw, yte, seed: int) -> tuple[dict, np.ndarray]:
    mean, std = fit_standardizer(xtr_raw)
    xtr = standardize(xtr_raw, mean, std)
    xva = standardize(xva_raw, mean, std)
    xte = standardize(xte_raw, mean, std)
    model, _ = train_mlp(
        xtr, ytr, xva, yva,
        hidden=HIDDEN, epochs=EPOCHS, batch_size=BATCH,
        lr=LR, l2=L2, seed=seed,
    )
    p = forward(model, xte)[0]
    return metric_summary(yte, p), p


def bootstrap_auc(y: np.ndarray, p: np.ndarray, seed: int, n_boot: int = 400) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if np.unique(y[idx]).size < 2:
            continue
        values.append(float(roc_curve(y[idx], p[idx])[2]))
    q = np.quantile(values, [0.025, 0.975])
    return float(q[0]), float(q[1])


def evaluate_support(task_data, mask: np.ndarray, seed: int) -> tuple[dict, np.ndarray]:
    cols = np.flatnonzero(mask.reshape(-1))
    xtr, ytr = task_data["train"]
    xva, yva = task_data["val"]
    xte, yte = task_data["test"]
    return fit_eval(flat(xtr)[:, cols], ytr, flat(xva)[:, cols], yva, flat(xte)[:, cols], yte, seed)


def evaluate_lowpass(task_data, k: int, seed: int) -> tuple[dict, np.ndarray]:
    def pool(imgs: np.ndarray) -> np.ndarray:
        b = IMAGE_SIZE // k
        return imgs.reshape(len(imgs), k, b, k, b).mean(axis=(2, 4)).reshape(len(imgs), -1).astype(np.float32)
    xtr, ytr = task_data["train"]
    xva, yva = task_data["val"]
    xte, yte = task_data["test"]
    return fit_eval(pool(xtr), ytr, pool(xva), yva, pool(xte), yte, seed)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/nmi_audit_validation")
    parser.add_argument("--random-draws", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260709)
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    masks = support_masks()
    tasks = ["roi_localised", "border_shortcut", "distributed_low_frequency", "random_support"]
    rows = []
    predictions = []
    summary = {}

    for task_index, task in enumerate(tasks):
        task_data = make_task(task, args.seed + task_index)
        xte, yte = task_data["test"]
        task_rows = []

        conditions: list[tuple[str, np.ndarray | None, int]] = [
            ("full_image", np.ones((IMAGE_SIZE, IMAGE_SIZE), bool), 0),
            ("target_roi", masks["target_roi"], 1),
            ("border", masks["border"], 2),
        ]
        target_size = int(masks["target_roi"].sum())
        random_masks = []
        for draw in range(args.random_draws):
            rng = np.random.default_rng(args.seed + 10000 * task_index + draw)
            m = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)
            m.reshape(-1)[rng.choice(IMAGE_SIZE * IMAGE_SIZE, size=target_size, replace=False)] = True
            random_masks.append(m)

        for label, mask, draw in conditions:
            metrics, p = evaluate_support(task_data, mask, args.seed + 500 * task_index + draw)
            ci_low, ci_high = bootstrap_auc(yte, p, args.seed + 8000 + draw)
            row = {
                "task": task, "condition": label, "draw": draw,
                "n_features": int(mask.sum()), "auroc": float(metrics["auroc"]),
                "accuracy": float(metrics["accuracy"]),
                "auroc_ci_low": ci_low, "auroc_ci_high": ci_high,
            }
            rows.append(row); task_rows.append(row)
            predictions.append({"task": task, "condition": label, "draw": draw, "y": yte, "p": p})

        for draw, mask in enumerate(random_masks):
            metrics, p = evaluate_support(task_data, mask, args.seed + 1000 * task_index + draw)
            ci_low, ci_high = bootstrap_auc(yte, p, args.seed + 9000 + draw)
            row = {
                "task": task, "condition": "random_exact_target_size", "draw": draw,
                "n_features": target_size, "auroc": float(metrics["auroc"]),
                "accuracy": float(metrics["accuracy"]),
                "auroc_ci_low": ci_low, "auroc_ci_high": ci_high,
            }
            rows.append(row); task_rows.append(row)
            predictions.append({"task": task, "condition": "random_exact_target_size", "draw": draw, "y": yte, "p": p})

        for k in [8, 1]:
            metrics, p = evaluate_lowpass(task_data, k, args.seed + 3000 * task_index + k)
            ci_low, ci_high = bootstrap_auc(yte, p, args.seed + 12000 + k)
            row = {
                "task": task, "condition": f"average_pool_{k}x{k}", "draw": 0,
                "n_features": k * k, "auroc": float(metrics["auroc"]),
                "accuracy": float(metrics["accuracy"]),
                "auroc_ci_low": ci_low, "auroc_ci_high": ci_high,
            }
            rows.append(row); task_rows.append(row)
            predictions.append({"task": task, "condition": f"average_pool_{k}x{k}", "draw": 0, "y": yte, "p": p})

        t = pd.DataFrame(task_rows)
        target_auc = float(t.loc[t["condition"] == "target_roi", "auroc"].iloc[0])
        random = t[t["condition"] == "random_exact_target_size"]["auroc"]
        outside = t.loc[t["condition"] == "border", "auroc"].iloc[0]
        summary[task] = {
            "target_roi_auroc": target_auc,
            "random_exact_mean": float(random.mean()),
            "random_exact_q025": float(random.quantile(0.025)),
            "random_exact_q975": float(random.quantile(0.975)),
            "target_exceeds_random_q975": bool(target_auc > random.quantile(0.975)),
            "border_auroc": float(outside),
            "pool_8x8_auroc": float(t.loc[t["condition"] == "average_pool_8x8", "auroc"].iloc[0]),
            "pool_1x1_auroc": float(t.loc[t["condition"] == "average_pool_1x1", "auroc"].iloc[0]),
        }

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(out / "synthetic_audit_metrics.csv", index=False)
    method_validation_pass = bool(
        summary["roi_localised"]["target_exceeds_random_q975"]
        and not summary["border_shortcut"]["target_exceeds_random_q975"]
        and summary["border_shortcut"]["border_auroc"] > 0.9
        and not summary["distributed_low_frequency"]["target_exceeds_random_q975"]
        and summary["distributed_low_frequency"]["pool_8x8_auroc"] > 0.9
        and summary["distributed_low_frequency"]["pool_1x1_auroc"] < 0.7
        and not summary["random_support"]["target_exceeds_random_q975"]
    )
    (out / "synthetic_audit_verdict.json").write_text(
        json.dumps({"method_validation_pass": method_validation_pass, "tasks": summary, "protocol": {
            "image_size": IMAGE_SIZE, "train_n": N["train"], "val_n": N["val"],
            "test_n": N["test"], "random_draws": args.random_draws,
            "retrain_each_condition": True, "labels_independent_of_base_noise": True,
        }}, indent=2), encoding="utf-8"
    )
    md = [
        "# NMI audit method validation",
        "",
        "Synthetic tasks have known signal supports and use the same retrain-based support audit as the PD-DBS case.",
        "",
        "| Task | Target ROI AUROC | Random exact-size mean | Random 97.5% | Border AUROC | Pool 8x8 | Pool 1x1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for task, s in summary.items():
        md.append(
            f"| {task} | {s['target_roi_auroc']:.4f} | {s['random_exact_mean']:.4f} | "
            f"{s['random_exact_q975']:.4f} | {s['border_auroc']:.4f} | "
            f"{s['pool_8x8_auroc']:.4f} | {s['pool_1x1_auroc']:.4f} |"
        )
    md += [
        "",
        f"Overall method-validation pass: **{method_validation_pass}**.",
        "",
        "The audit is considered methodologically discriminating only if the known ROI task separates the target ROI from exact-size random supports, while the border and distributed tasks are identified by their corresponding controls.",
    ]
    (out / "synthetic_audit_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

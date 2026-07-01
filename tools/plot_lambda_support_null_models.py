from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_lambda_support_vs_full_interval import fit_power_law, full_interval_bins, worst_shell_ratio
from plot_local_autocorr_bound import autocorrelation_nonnegative, dyadic_edges, parse_shell_bounds, parse_t_scales
from plot_product_sign_weight_decomposition import product_signed_counts


def scan_worst_ratio(
    signed_bins: np.ndarray,
    abs_bins: np.ndarray,
    diagonal_raw: float,
    delta: float,
    t_scales: list[float],
    shell_bounds: list[float],
) -> tuple[float, float, str]:
    corr = autocorrelation_nonnegative(signed_bins)
    abs_corr = autocorrelation_nonnegative(abs_bins)
    lag_index = np.arange(corr.size, dtype=np.float64)
    worst_ratio = -1.0
    worst_t = 0.0
    worst_shell = ""

    for t_scale in t_scales:
        scaled = t_scale * delta * lag_index
        ratio, shell = worst_shell_ratio(corr, abs_corr, diagonal_raw, scaled, shell_bounds)
        if math.isfinite(ratio) and ratio > worst_ratio:
            worst_ratio = ratio
            worst_t = t_scale
            worst_shell = shell

    return worst_ratio, worst_t, worst_shell


def random_interval_subset_bins(
    lambda_values: np.ndarray,
    start: int,
    stop: int,
    edges_log: np.ndarray,
    density: float,
    chunk_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    signed_bins = np.zeros(edges_log.size - 1, dtype=np.float64)
    abs_bins = np.zeros(edges_log.size - 1, dtype=np.float64)
    signed_sum = 0.0
    selected_count = 0

    for chunk_start in range(start, stop, chunk_size):
        chunk_stop = min(chunk_start + chunk_size, stop)
        size = chunk_stop - chunk_start
        mask = rng.random(size) < density
        if not np.any(mask):
            continue
        values = np.arange(chunk_start, chunk_stop, dtype=np.int64)[mask]
        signs = lambda_values[chunk_start:chunk_stop][mask].astype(np.float64)
        logs = np.log(values.astype(np.float64))
        signed_part, _ = np.histogram(logs, bins=edges_log, weights=signs)
        abs_part, _ = np.histogram(logs, bins=edges_log)
        signed_bins += signed_part
        abs_bins += abs_part.astype(np.float64)
        signed_sum += float(np.sum(signs))
        selected_count += int(signs.size)

    sign_balance = signed_sum / float(selected_count) if selected_count > 0 else float("nan")
    return signed_bins, abs_bins, selected_count, sign_balance


def quantile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.quantile(np.array(values, dtype=np.float64), q))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mu_bin", type=Path)
    parser.add_argument("lambda_bin", type=Path)
    parser.add_argument("output_png", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--first", type=int, default=128)
    parser.add_argument("--max-y", type=int, default=4096)
    parser.add_argument("--t-scales", default="50,75,100,150,200,300,500,750,1000,1500,2000,3000,5000")
    parser.add_argument("--bin-count", type=int, default=65536)
    parser.add_argument("--shell-bounds", default="0,1,2,5,10,20,50,inf")
    parser.add_argument("--pair-row-chunk-size", type=int, default=512)
    parser.add_argument("--merge-row-chunks", type=int, default=4)
    parser.add_argument("--full-chunk-size", type=int, default=1_000_000)
    parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260702)
    args = parser.parse_args()

    raw_mu = np.fromfile(args.mu_bin, dtype=np.int8)
    if raw_mu.size < 2:
        raise SystemExit("mu binary must contain mu[0..N]")
    if args.first <= 0 or args.max_y < args.first:
        raise SystemExit("--first and --max-y must define a nonempty dyadic range")
    if 2 * args.max_y > raw_mu.size - 1:
        raise SystemExit("mu binary is too short for the requested --max-y")
    if args.bin_count < 16:
        raise SystemExit("--bin-count is too small")
    if args.full_chunk_size <= 0:
        raise SystemExit("--full-chunk-size must be positive")
    if args.replicates <= 0:
        raise SystemExit("--replicates must be positive")

    full_limit = (2 * args.max_y) ** 2
    lambda_values = np.memmap(args.lambda_bin, dtype=np.int8, mode="r")
    if lambda_values.size <= full_limit:
        raise SystemExit("lambda binary is too short for the requested --max-y")

    t_scales = parse_t_scales(args.t_scales)
    shell_bounds = parse_shell_bounds(args.shell_bounds)
    rng = np.random.default_rng(args.seed)
    mu = raw_mu[1:]
    nonzero = np.nonzero(mu)[0].astype(np.int64) + 1
    signs = mu[nonzero - 1].astype(np.float64)
    rows: list[dict[str, float | int | str]] = []

    for start, stop in dyadic_edges(args.max_y, args.first):
        left = np.searchsorted(nonzero, start, side="left")
        right = np.searchsorted(nonzero, stop, side="left")
        block_n = nonzero[left:right]
        block_signs = signs[left:right].astype(np.int64)
        if block_n.size == 0:
            continue

        full_start = start * start
        full_stop = stop * stop
        full_count = full_stop - full_start
        edges_log = np.linspace(math.log(full_start), math.log(full_stop), args.bin_count + 1)
        delta = float(edges_log[1] - edges_log[0])

        products, coeffs, _counts = product_signed_counts(
            block_n,
            block_signs,
            args.pair_row_chunk_size,
            args.merge_row_chunks,
        )
        product_signs = np.sign(coeffs).astype(np.float64)
        product_logs = np.log(products.astype(np.float64))
        support_abs_bins, _ = np.histogram(product_logs, bins=edges_log, weights=np.ones(products.shape, dtype=np.float64))
        support_signed_bins, _ = np.histogram(product_logs, bins=edges_log, weights=product_signs)
        support_density = float(products.size) / float(full_count)

        support_ratio, support_t, support_shell = scan_worst_ratio(
            support_signed_bins,
            support_abs_bins,
            float(products.size),
            delta,
            t_scales,
            shell_bounds,
        )
        rows.append(
            {
                "Y": start,
                "model": "product_support",
                "replicate": 0,
                "worst_ratio": support_ratio,
                "worst_T": support_t,
                "worst_shell": support_shell,
                "selected_count": int(products.size),
                "density": support_density,
                "sign_balance": float(np.mean(product_signs)),
            }
        )

        full_signed_bins, full_abs_bins, full_selected_count, full_sign_balance = full_interval_bins(
            lambda_values,
            full_start,
            full_stop,
            edges_log,
            args.full_chunk_size,
        )
        full_ratio, full_t, full_shell = scan_worst_ratio(
            full_signed_bins,
            full_abs_bins,
            float(full_selected_count),
            delta,
            t_scales,
            shell_bounds,
        )
        rows.append(
            {
                "Y": start,
                "model": "full_interval",
                "replicate": 0,
                "worst_ratio": full_ratio,
                "worst_T": full_t,
                "worst_shell": full_shell,
                "selected_count": int(full_selected_count),
                "density": 1.0,
                "sign_balance": full_sign_balance,
            }
        )

        for replicate in range(1, args.replicates + 1):
            shuffled_signs = rng.permutation(product_signs)
            shuffled_signed_bins, _ = np.histogram(product_logs, bins=edges_log, weights=shuffled_signs)
            shuffled_ratio, shuffled_t, shuffled_shell = scan_worst_ratio(
                shuffled_signed_bins,
                support_abs_bins,
                float(products.size),
                delta,
                t_scales,
                shell_bounds,
            )
            rows.append(
                {
                    "Y": start,
                    "model": "shuffled_support_signs",
                    "replicate": replicate,
                    "worst_ratio": shuffled_ratio,
                    "worst_T": shuffled_t,
                    "worst_shell": shuffled_shell,
                    "selected_count": int(products.size),
                    "density": support_density,
                    "sign_balance": float(np.mean(shuffled_signs)),
                }
            )

            random_signed_bins, random_abs_bins, random_selected_count, random_sign_balance = random_interval_subset_bins(
                lambda_values,
                full_start,
                full_stop,
                edges_log,
                support_density,
                args.full_chunk_size,
                rng,
            )
            random_ratio, random_t, random_shell = scan_worst_ratio(
                random_signed_bins,
                random_abs_bins,
                float(random_selected_count),
                delta,
                t_scales,
                shell_bounds,
            )
            rows.append(
                {
                    "Y": start,
                    "model": "random_interval_subset",
                    "replicate": replicate,
                    "worst_ratio": random_ratio,
                    "worst_T": random_t,
                    "worst_shell": random_shell,
                    "selected_count": int(random_selected_count),
                    "density": float(random_selected_count) / float(full_count),
                    "sign_balance": random_sign_balance,
                }
            )

        y_rows = [row for row in rows if int(row["Y"]) == start]
        support = next(row for row in y_rows if row["model"] == "product_support")
        shuffled = [float(row["worst_ratio"]) for row in y_rows if row["model"] == "shuffled_support_signs"]
        random_subset = [float(row["worst_ratio"]) for row in y_rows if row["model"] == "random_interval_subset"]
        full = next(row for row in y_rows if row["model"] == "full_interval")
        print(
            f"Y={start} actual={float(support['worst_ratio']):.6g} "
            f"shuffled_median={quantile(shuffled, 0.5):.6g} "
            f"random_subset_median={quantile(random_subset, 0.5):.6g} "
            f"full={float(full['worst_ratio']):.6g}"
        )

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(rows[0].keys())
        with args.output_csv.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    y_values = sorted({int(row["Y"]) for row in rows})
    model_points: dict[str, list[tuple[int, float]]] = {
        "product_support": [],
        "full_interval": [],
        "shuffled_support_signs_median": [],
        "random_interval_subset_median": [],
    }
    ratio_points: dict[str, list[tuple[int, float]]] = {
        "actual_over_shuffled": [],
        "actual_over_random_subset": [],
        "actual_over_full": [],
    }
    density_points = []

    for y in y_values:
        y_rows = [row for row in rows if int(row["Y"]) == y]
        actual = float(next(row for row in y_rows if row["model"] == "product_support")["worst_ratio"])
        full = float(next(row for row in y_rows if row["model"] == "full_interval")["worst_ratio"])
        shuffled = [float(row["worst_ratio"]) for row in y_rows if row["model"] == "shuffled_support_signs"]
        random_subset = [float(row["worst_ratio"]) for row in y_rows if row["model"] == "random_interval_subset"]
        shuffled_median = quantile(shuffled, 0.5)
        random_subset_median = quantile(random_subset, 0.5)

        model_points["product_support"].append((y, actual))
        model_points["full_interval"].append((y, full))
        model_points["shuffled_support_signs_median"].append((y, shuffled_median))
        model_points["random_interval_subset_median"].append((y, random_subset_median))
        ratio_points["actual_over_shuffled"].append((y, actual / shuffled_median if shuffled_median > 0 else float("nan")))
        ratio_points["actual_over_random_subset"].append(
            (y, actual / random_subset_median if random_subset_median > 0 else float("nan"))
        )
        ratio_points["actual_over_full"].append((y, actual / full if full > 0 else float("nan")))
        density = float(next(row for row in y_rows if row["model"] == "product_support")["density"])
        density_points.append((y, density))

    fits = {}
    for model, points in model_points.items():
        alpha, _c, r2 = fit_power_law(points)
        fits[model] = (alpha, r2)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(3, 1, figsize=(13, 12))

    labels = {
        "product_support": "actual product support",
        "full_interval": "full interval",
        "shuffled_support_signs_median": "same support, shuffled signs median",
        "random_interval_subset_median": "random interval subset median",
    }
    colors = {
        "product_support": "#1f5f9f",
        "full_interval": "#c05a25",
        "shuffled_support_signs_median": "#4b8f3a",
        "random_interval_subset_median": "#7b4ea3",
    }
    for model, points in model_points.items():
        alpha, r2 = fits[model]
        axes[0].loglog(
            [point[0] for point in points],
            [point[1] for point in points],
            marker="o",
            linewidth=1.2,
            color=colors[model],
            label=f"{labels[model]}, alpha={alpha:.3f}, R2={r2:.3f}",
        )
    axes[0].set_xlabel("dyadic start Y")
    axes[0].set_ylabel("worst local ratio")
    axes[0].set_title("Actual product support versus null models")
    axes[0].legend(loc="best")

    for key, label in [
        ("actual_over_shuffled", "actual / shuffled support signs"),
        ("actual_over_random_subset", "actual / random interval subset"),
        ("actual_over_full", "actual / full interval"),
    ]:
        points = ratio_points[key]
        axes[1].semilogx(
            [point[0] for point in points],
            [point[1] for point in points],
            marker="o",
            linewidth=1.2,
            label=label,
        )
    axes[1].axhline(1.0, color="#777777", linestyle="--", linewidth=0.9)
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("dyadic start Y")
    axes[1].set_ylabel("ratio")
    axes[1].set_title("How much worse is the true product-support lambda sequence?")
    axes[1].legend(loc="best")

    axes[2].plot(
        [point[0] for point in density_points],
        [point[1] for point in density_points],
        marker="o",
        linewidth=1.2,
        label="support density",
    )
    for model in ["product_support", "random_interval_subset_median"]:
        points = model_points[model]
        axes[2].semilogy(
            [point[0] for point in points],
            [point[1] for point in points],
            marker="o",
            linewidth=1.2,
            label=labels[model],
        )
    axes[2].set_xscale("log", base=2)
    axes[2].set_xlabel("dyadic start Y")
    axes[2].set_title("Support density compared with actual and random-subset ratios")
    axes[2].legend(loc="best")

    fig.tight_layout()
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=160)

    print(args.output_png)
    if args.output_csv:
        print(args.output_csv)
    print(f"replicates={args.replicates}")
    print(f"seed={args.seed}")
    for model, points in model_points.items():
        alpha, r2 = fits[model]
        print(f"{model}_fit_alpha={alpha:.6g} r2={r2:.6g}")
    for key, points in ratio_points.items():
        finite = [point[1] for point in points if math.isfinite(point[1])]
        print(f"{key}_median={quantile(finite, 0.5):.6g} max={max(finite):.6g}")


if __name__ == "__main__":
    main()

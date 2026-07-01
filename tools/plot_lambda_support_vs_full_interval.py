from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_local_autocorr_bound import (
    autocorrelation_nonnegative,
    dyadic_edges,
    parse_shell_bounds,
    parse_t_scales,
    shell_label,
    shell_sum,
)
from plot_product_sign_weight_decomposition import product_signed_counts


def worst_shell_ratio(
    corr: np.ndarray,
    abs_corr: np.ndarray,
    diagonal_raw: float,
    scaled: np.ndarray,
    shell_bounds: list[float],
) -> tuple[float, str]:
    worst_ratio = -1.0
    worst_shell = ""

    for low, high in zip(shell_bounds[:-1], shell_bounds[1:]):
        if math.isinf(high):
            mask = scaled >= low
        else:
            mask = (scaled >= low) & (scaled < high)
        include_h0 = low == 0
        signed_raw = shell_sum(corr, diagonal_raw, mask, include_h0)
        unsigned_raw = shell_sum(abs_corr, diagonal_raw, mask, include_h0)
        ratio = abs(signed_raw) / unsigned_raw if unsigned_raw > 0 else float("nan")
        if math.isfinite(ratio) and ratio > worst_ratio:
            worst_ratio = ratio
            worst_shell = shell_label(low, high)

    return worst_ratio, worst_shell


def fit_power_law(points: list[tuple[int, float]]) -> tuple[float, float, float]:
    if len(points) < 2:
        return float("nan"), float("nan"), float("nan")
    x = np.log(np.array([point[0] for point in points], dtype=np.float64))
    y = np.log(np.array([point[1] for point in points], dtype=np.float64))
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    residual = y - fitted
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return -float(slope), float(math.exp(intercept)), r2


def full_interval_bins(
    lambda_values: np.ndarray,
    start: int,
    stop: int,
    edges_log: np.ndarray,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    signed_bins = np.zeros(edges_log.size - 1, dtype=np.float64)
    abs_bins = np.zeros(edges_log.size - 1, dtype=np.float64)
    signed_sum = 0.0
    count = stop - start

    for chunk_start in range(start, stop, chunk_size):
        chunk_stop = min(chunk_start + chunk_size, stop)
        values = np.arange(chunk_start, chunk_stop, dtype=np.int64)
        signs = lambda_values[chunk_start:chunk_stop].astype(np.float64)
        logs = np.log(values.astype(np.float64))
        signed_part, _ = np.histogram(logs, bins=edges_log, weights=signs)
        abs_part, _ = np.histogram(logs, bins=edges_log)
        signed_bins += signed_part
        abs_bins += abs_part.astype(np.float64)
        signed_sum += float(np.sum(signs))

    return signed_bins, abs_bins, count, signed_sum / float(count)


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

    full_limit = (2 * args.max_y) ** 2
    lambda_values = np.memmap(args.lambda_bin, dtype=np.int8, mode="r")
    if lambda_values.size <= full_limit:
        raise SystemExit("lambda binary is too short for the requested --max-y")

    t_scales = parse_t_scales(args.t_scales)
    shell_bounds = parse_shell_bounds(args.shell_bounds)
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
        support_bins, _ = np.histogram(product_logs, bins=edges_log, weights=product_signs)
        support_abs_bins, _ = np.histogram(product_logs, bins=edges_log, weights=np.ones(products.shape, dtype=np.float64))

        full_bins, full_abs_bins, full_count, full_sign_balance = full_interval_bins(
            lambda_values,
            full_start,
            full_stop,
            edges_log,
            args.full_chunk_size,
        )

        support_corr = autocorrelation_nonnegative(support_bins)
        support_abs_corr = autocorrelation_nonnegative(support_abs_bins)
        full_corr = autocorrelation_nonnegative(full_bins)
        full_abs_corr = autocorrelation_nonnegative(full_abs_bins)
        lag_index = np.arange(support_corr.size, dtype=np.float64)

        for t_scale in t_scales:
            scaled = t_scale * delta * lag_index
            support_ratio, support_shell = worst_shell_ratio(
                support_corr,
                support_abs_corr,
                float(products.size),
                scaled,
                shell_bounds,
            )
            full_ratio, full_shell = worst_shell_ratio(
                full_corr,
                full_abs_corr,
                float(full_count),
                scaled,
                shell_bounds,
            )
            rows.append(
                {
                    "Y": start,
                    "stop": stop - 1,
                    "T": t_scale,
                    "full_interval_start": full_start,
                    "full_interval_stop": full_stop - 1,
                    "full_count": full_count,
                    "support_unique_product_count": int(products.size),
                    "support_density": float(products.size) / float(full_count),
                    "support_sign_balance": float(np.mean(product_signs)),
                    "full_sign_balance": full_sign_balance,
                    "support_worst_ratio": support_ratio,
                    "support_worst_shell": support_shell,
                    "full_worst_ratio": full_ratio,
                    "full_worst_shell": full_shell,
                    "support_over_full": support_ratio / full_ratio if full_ratio > 0 else float("nan"),
                }
            )

        block_rows = [row for row in rows if int(row["Y"]) == start]
        worst_support = max(block_rows, key=lambda item: float(item["support_worst_ratio"]))
        worst_full = max(block_rows, key=lambda item: float(item["full_worst_ratio"]))
        print(
            f"Y={start} support={float(worst_support['support_worst_ratio']):.6g} "
            f"full={float(worst_full['full_worst_ratio']):.6g} "
            f"density={float(worst_support['support_density']):.6g}"
        )

    if not rows:
        raise SystemExit("no dyadic blocks were scanned")

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(rows[0].keys())
        with args.output_csv.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    y_values = sorted({int(row["Y"]) for row in rows})
    support_points = []
    full_points = []
    density_points = []
    support_balance_points = []
    full_balance_points = []
    support_over_full_points = []

    for y in y_values:
        y_rows = [row for row in rows if int(row["Y"]) == y]
        support_worst = max(float(row["support_worst_ratio"]) for row in y_rows)
        full_worst = max(float(row["full_worst_ratio"]) for row in y_rows)
        support_points.append((y, support_worst))
        full_points.append((y, full_worst))
        support_over_full_points.append((y, support_worst / full_worst if full_worst > 0 else float("nan")))
        density_points.append((y, float(y_rows[0]["support_density"])))
        support_balance_points.append((y, abs(float(y_rows[0]["support_sign_balance"]))))
        full_balance_points.append((y, abs(float(y_rows[0]["full_sign_balance"]))))

    support_alpha, _, support_r2 = fit_power_law(support_points)
    full_alpha, _, full_r2 = fit_power_law(full_points)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(3, 1, figsize=(13, 12))

    axes[0].loglog(
        [point[0] for point in support_points],
        [point[1] for point in support_points],
        marker="o",
        linewidth=1.2,
        label=f"product support lambda, alpha={support_alpha:.3f}",
    )
    axes[0].loglog(
        [point[0] for point in full_points],
        [point[1] for point in full_points],
        marker="o",
        linewidth=1.2,
        label=f"full interval lambda, alpha={full_alpha:.3f}",
    )
    axes[0].set_xlabel("dyadic start Y")
    axes[0].set_ylabel("worst local ratio")
    axes[0].set_title("Liouville signs on product support versus the full product interval")
    axes[0].legend(loc="best")

    axes[1].semilogx(
        [point[0] for point in support_over_full_points],
        [point[1] for point in support_over_full_points],
        marker="o",
        linewidth=1.2,
        color="#7b4ea3",
        label="support / full",
    )
    axes[1].axhline(1.0, color="#777777", linestyle="--", linewidth=0.9)
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("dyadic start Y")
    axes[1].set_ylabel("ratio")
    axes[1].set_title("Does product support make local lambda cancellation harder?")
    axes[1].legend(loc="best")

    axes[2].plot(
        [point[0] for point in density_points],
        [point[1] for point in density_points],
        marker="o",
        linewidth=1.2,
        label="product support density",
    )
    axes[2].semilogy(
        [point[0] for point in support_balance_points],
        [max(point[1], 1e-12) for point in support_balance_points],
        marker="o",
        linewidth=1.2,
        label="abs support sign balance",
    )
    axes[2].semilogy(
        [point[0] for point in full_balance_points],
        [max(point[1], 1e-12) for point in full_balance_points],
        marker="o",
        linewidth=1.2,
        label="abs full interval sign balance",
    )
    axes[2].set_xscale("log", base=2)
    axes[2].set_xlabel("dyadic start Y")
    axes[2].set_title("Support density and sign balance")
    axes[2].legend(loc="best")

    fig.tight_layout()
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=160)

    overall_support = max(rows, key=lambda item: float(item["support_worst_ratio"]))
    overall_full = max(rows, key=lambda item: float(item["full_worst_ratio"]))
    print(args.output_png)
    if args.output_csv:
        print(args.output_csv)
    print(f"max_y={args.max_y}")
    print(f"bin_count={args.bin_count}")
    print(
        f"overall_support_worst={float(overall_support['support_worst_ratio']):.6g} "
        f"Y={int(overall_support['Y'])} T={float(overall_support['T']):g}"
    )
    print(
        f"overall_full_worst={float(overall_full['full_worst_ratio']):.6g} "
        f"Y={int(overall_full['Y'])} T={float(overall_full['T']):g}"
    )
    print(f"support_fit_alpha={support_alpha:.6g} r2={support_r2:.6g}")
    print(f"full_fit_alpha={full_alpha:.6g} r2={full_r2:.6g}")


if __name__ == "__main__":
    main()

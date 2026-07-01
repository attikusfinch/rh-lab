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


def reduce_product_terms(
    products: np.ndarray,
    coeffs: np.ndarray,
    counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(products, kind="mergesort")
    sorted_products = products[order]
    sorted_coeffs = coeffs[order].astype(np.int64, copy=False)
    sorted_counts = counts[order].astype(np.int64, copy=False)
    unique_products, starts = np.unique(sorted_products, return_index=True)
    coeff_sums = np.add.reduceat(sorted_coeffs, starts)
    count_sums = np.add.reduceat(sorted_counts, starts)
    return unique_products, coeff_sums, count_sums


def merge_product_terms(
    parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nonempty = [part for part in parts if part[0].size > 0]
    if not nonempty:
        return (
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
        )
    if len(nonempty) == 1:
        return nonempty[0]

    products = np.concatenate([part[0] for part in nonempty])
    coeffs = np.concatenate([part[1] for part in nonempty])
    counts = np.concatenate([part[2] for part in nonempty])
    return reduce_product_terms(products, coeffs, counts)


def product_signed_counts(
    block_n: np.ndarray,
    block_signs: np.ndarray,
    row_chunk_size: int,
    merge_row_chunks: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if merge_row_chunks <= 0:
        raise SystemExit("--merge-row-chunks must be positive")

    accumulator_products = np.array([], dtype=np.int64)
    accumulator_coeffs = np.array([], dtype=np.int64)
    accumulator_counts = np.array([], dtype=np.int64)
    pending: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    for chunk_index, row_start in enumerate(range(0, block_n.size, row_chunk_size), start=1):
        row_stop = min(row_start + row_chunk_size, block_n.size)
        products = (block_n[row_start:row_stop, None] * block_n[None, :]).ravel()
        coeffs = (block_signs[row_start:row_stop, None] * block_signs[None, :]).astype(np.int16).ravel()
        counts = np.ones(products.shape, dtype=np.int16)
        pending.append(reduce_product_terms(products, coeffs, counts))

        if chunk_index % merge_row_chunks == 0:
            accumulator_products, accumulator_coeffs, accumulator_counts = merge_product_terms(
                [(accumulator_products, accumulator_coeffs, accumulator_counts), *pending]
            )
            pending = []

    return merge_product_terms([(accumulator_products, accumulator_coeffs, accumulator_counts), *pending])


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mu_bin", type=Path)
    parser.add_argument("output_png", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--first", type=int, default=128)
    parser.add_argument("--max-y", type=int, default=16384)
    parser.add_argument("--t-scales", default="50,75,100,150,200,300,500,750,1000,1500,2000,3000,5000")
    parser.add_argument("--bin-count", type=int, default=65536)
    parser.add_argument("--shell-bounds", default="0,1,2,5,10,20,50,inf")
    parser.add_argument("--pair-row-chunk-size", type=int, default=512)
    parser.add_argument("--merge-row-chunks", type=int, default=4)
    args = parser.parse_args()

    raw = np.fromfile(args.mu_bin, dtype=np.int8)
    if raw.size < 2:
        raise SystemExit("mu binary must contain mu[0..N]")
    if args.first <= 0 or args.max_y < args.first:
        raise SystemExit("--first and --max-y must define a nonempty dyadic range")
    if 2 * args.max_y > raw.size - 1:
        raise SystemExit("mu binary is too short for the requested --max-y")
    if args.bin_count < 16:
        raise SystemExit("--bin-count is too small")

    t_scales = parse_t_scales(args.t_scales)
    shell_bounds = parse_shell_bounds(args.shell_bounds)
    mu = raw[1:]
    nonzero = np.nonzero(mu)[0].astype(np.int64) + 1
    signs = mu[nonzero - 1].astype(np.float64)
    edges = dyadic_edges(args.max_y, args.first)
    rows: list[dict[str, float | int | str]] = []

    for start, stop in edges:
        left = np.searchsorted(nonzero, start, side="left")
        right = np.searchsorted(nonzero, stop, side="left")
        block_n = nonzero[left:right]
        block_signs = signs[left:right].astype(np.int64)
        if block_n.size == 0:
            continue

        products, coeffs, counts = product_signed_counts(
            block_n,
            block_signs,
            args.pair_row_chunk_size,
            args.merge_row_chunks,
        )
        weights = np.abs(coeffs)
        sign_consistent = weights == counts
        mismatch_count = int(np.count_nonzero(~sign_consistent))
        if np.any(coeffs == 0):
            raise SystemExit("unexpected zero product coefficient")

        product_signs = np.sign(coeffs).astype(np.float64)
        log_products = np.log(products.astype(np.float64))
        edges_log = np.linspace(float(log_products[0]), float(log_products[-1]), args.bin_count + 1)
        delta = float(edges_log[1] - edges_log[0])

        actual_bins, _ = np.histogram(log_products, bins=edges_log, weights=coeffs.astype(np.float64))
        actual_abs_bins, _ = np.histogram(log_products, bins=edges_log, weights=weights.astype(np.float64))
        sign_bins, _ = np.histogram(log_products, bins=edges_log, weights=product_signs)
        sign_abs_bins, _ = np.histogram(log_products, bins=edges_log, weights=np.ones(products.shape, dtype=np.float64))

        actual_corr = autocorrelation_nonnegative(actual_bins)
        actual_abs_corr = autocorrelation_nonnegative(actual_abs_bins)
        sign_corr = autocorrelation_nonnegative(sign_bins)
        sign_abs_corr = autocorrelation_nonnegative(sign_abs_bins)
        lag_index = np.arange(actual_corr.size, dtype=np.float64)
        actual_diagonal = float(np.sum(coeffs.astype(np.float64) ** 2))
        sign_diagonal = float(products.size)

        for t_scale in t_scales:
            scaled = t_scale * delta * lag_index
            actual_ratio, actual_shell = worst_shell_ratio(
                actual_corr,
                actual_abs_corr,
                actual_diagonal,
                scaled,
                shell_bounds,
            )
            sign_ratio, sign_shell = worst_shell_ratio(
                sign_corr,
                sign_abs_corr,
                sign_diagonal,
                scaled,
                shell_bounds,
            )
            rows.append(
                {
                    "Y": start,
                    "stop": stop - 1,
                    "T": t_scale,
                    "block_nonzero_count": int(block_n.size),
                    "unique_product_count": int(products.size),
                    "total_pair_representations": int(np.sum(counts)),
                    "max_weight": int(np.max(weights)),
                    "mean_weight": float(np.mean(weights)),
                    "diagonal_4": actual_diagonal / float(start * start),
                    "positive_product_fraction": float(np.mean(product_signs > 0)),
                    "product_sign_balance": float(np.mean(product_signs)),
                    "weighted_sign_balance": float(np.sum(coeffs) / np.sum(weights)),
                    "sign_consistency_mismatch_count": mismatch_count,
                    "sign_consistency_mismatch_fraction": mismatch_count / float(products.size),
                    "actual_worst_ratio": actual_ratio,
                    "actual_worst_shell": actual_shell,
                    "sign_only_worst_ratio": sign_ratio,
                    "sign_only_worst_shell": sign_shell,
                    "actual_over_sign_only": actual_ratio / sign_ratio if sign_ratio > 0 else float("nan"),
                }
            )

        block_rows = [row for row in rows if int(row["Y"]) == start]
        worst_actual = max(block_rows, key=lambda item: float(item["actual_worst_ratio"]))
        worst_sign = max(block_rows, key=lambda item: float(item["sign_only_worst_ratio"]))
        print(
            f"Y={start} mismatches={mismatch_count} "
            f"actual={float(worst_actual['actual_worst_ratio']):.6g} "
            f"sign_only={float(worst_sign['sign_only_worst_ratio']):.6g}"
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
    actual_points = []
    sign_points = []
    amplification_points = []
    max_weight_points = []
    mean_weight_points = []
    sign_balance_points = []
    weighted_sign_balance_points = []

    for y in y_values:
        y_rows = [row for row in rows if int(row["Y"]) == y]
        worst_actual = max(float(row["actual_worst_ratio"]) for row in y_rows)
        worst_sign = max(float(row["sign_only_worst_ratio"]) for row in y_rows)
        actual_points.append((y, worst_actual))
        sign_points.append((y, worst_sign))
        amplification_points.append((y, worst_actual / worst_sign if worst_sign > 0 else float("nan")))
        max_weight_points.append((y, float(y_rows[0]["max_weight"])))
        mean_weight_points.append((y, float(y_rows[0]["mean_weight"])))
        sign_balance_points.append((y, abs(float(y_rows[0]["product_sign_balance"]))))
        weighted_sign_balance_points.append((y, abs(float(y_rows[0]["weighted_sign_balance"]))))

    actual_alpha, _, actual_r2 = fit_power_law(actual_points)
    sign_alpha, _, sign_r2 = fit_power_law(sign_points)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(3, 1, figsize=(13, 12))

    axes[0].loglog(
        [point[0] for point in actual_points],
        [point[1] for point in actual_points],
        marker="o",
        linewidth=1.2,
        label=f"actual sign*weight, alpha={actual_alpha:.3f}",
    )
    axes[0].loglog(
        [point[0] for point in sign_points],
        [point[1] for point in sign_points],
        marker="o",
        linewidth=1.2,
        label=f"sign only, alpha={sign_alpha:.3f}",
    )
    axes[0].set_xlabel("dyadic start Y")
    axes[0].set_ylabel("worst local ratio")
    axes[0].set_title("Local cancellation: product signs versus signed weights")
    axes[0].legend(loc="best")

    axes[1].semilogx(
        [point[0] for point in amplification_points],
        [point[1] for point in amplification_points],
        marker="o",
        linewidth=1.2,
        color="#7b4ea3",
        label="actual / sign-only",
    )
    axes[1].axhline(1.0, color="#777777", linestyle="--", linewidth=0.9)
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("dyadic start Y")
    axes[1].set_ylabel("ratio")
    axes[1].set_title("How much the divisor-splitting weights change the sign-only cancellation")
    axes[1].legend(loc="best")

    axes[2].plot(
        [point[0] for point in max_weight_points],
        [point[1] for point in max_weight_points],
        marker="o",
        linewidth=1.2,
        label="max weight |A_Y(r)|",
    )
    axes[2].plot(
        [point[0] for point in mean_weight_points],
        [point[1] for point in mean_weight_points],
        marker="o",
        linewidth=1.2,
        label="mean weight",
    )
    axes[2].semilogy(
        [point[0] for point in sign_balance_points],
        [max(point[1], 1e-12) for point in sign_balance_points],
        marker="o",
        linewidth=1.2,
        label="abs product sign balance",
    )
    axes[2].semilogy(
        [point[0] for point in weighted_sign_balance_points],
        [max(point[1], 1e-12) for point in weighted_sign_balance_points],
        marker="o",
        linewidth=1.2,
        label="abs weighted sign balance",
    )
    axes[2].set_xscale("log", base=2)
    axes[2].set_xlabel("dyadic start Y")
    axes[2].set_title("Weights and sign balance of product coefficients")
    axes[2].legend(loc="best")

    fig.tight_layout()
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=160)

    overall_actual = max(rows, key=lambda item: float(item["actual_worst_ratio"]))
    overall_sign = max(rows, key=lambda item: float(item["sign_only_worst_ratio"]))
    max_mismatch = max(int(row["sign_consistency_mismatch_count"]) for row in rows)
    print(args.output_png)
    if args.output_csv:
        print(args.output_csv)
    print(f"max_y={args.max_y}")
    print(f"bin_count={args.bin_count}")
    print(f"max_sign_consistency_mismatch_count={max_mismatch}")
    print(
        f"overall_actual_worst={float(overall_actual['actual_worst_ratio']):.6g} "
        f"Y={int(overall_actual['Y'])} T={float(overall_actual['T']):g}"
    )
    print(
        f"overall_sign_only_worst={float(overall_sign['sign_only_worst_ratio']):.6g} "
        f"Y={int(overall_sign['Y'])} T={float(overall_sign['T']):g}"
    )
    print(f"actual_fit_alpha={actual_alpha:.6g} r2={actual_r2:.6g}")
    print(f"sign_only_fit_alpha={sign_alpha:.6g} r2={sign_r2:.6g}")


if __name__ == "__main__":
    main()

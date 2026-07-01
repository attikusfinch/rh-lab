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
    product_coefficients,
    shell_label,
    shell_sum,
)


def sinc_kernel_real(scaled: np.ndarray) -> np.ndarray:
    result = np.ones_like(scaled, dtype=np.float64)
    mask = scaled != 0
    result[mask] = np.sin(scaled[mask]) / scaled[mask]
    return result


def kernel_offdiag(
    corr: np.ndarray,
    abs_corr: np.ndarray,
    diagonal_raw: float,
    kernel: np.ndarray,
) -> tuple[float, float]:
    signed_raw = float(corr[0] - diagonal_raw + np.sum(2.0 * corr[1:] * kernel[1:]))
    unsigned_raw = float(abs(abs_corr[0] - diagonal_raw) + np.sum(2.0 * abs_corr[1:] * np.abs(kernel[1:])))
    return signed_raw, unsigned_raw


def local_shell_worst_ratio(
    corr: np.ndarray,
    abs_corr: np.ndarray,
    diagonal_raw: float,
    scaled: np.ndarray,
    shell_bounds: list[float],
) -> tuple[float, str]:
    worst_ratio = -1.0
    worst_label = ""
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
            worst_label = shell_label(low, high)
    return worst_ratio, worst_label


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
    parser.add_argument("--aggregation", choices=["full", "incremental"], default="incremental")
    parser.add_argument("--merge-row-chunks", type=int, default=4)
    parser.add_argument("--epsilon-c", type=float, default=1600.0)
    parser.add_argument("--epsilon-alpha", type=float, default=2.0)
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
    if args.epsilon_c <= 0 or args.epsilon_alpha <= 0:
        raise SystemExit("--epsilon-c and --epsilon-alpha must be positive")

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

        products, coeffs, diagonal_raw = product_coefficients(
            block_n,
            block_signs,
            args.pair_row_chunk_size,
            args.aggregation,
            args.merge_row_chunks,
        )
        log_products = np.log(products.astype(np.float64))
        edges_log = np.linspace(float(log_products[0]), float(log_products[-1]), args.bin_count + 1)
        delta = float(edges_log[1] - edges_log[0])
        coeff_bins, _ = np.histogram(log_products, bins=edges_log, weights=coeffs.astype(np.float64))
        abs_bins, _ = np.histogram(log_products, bins=edges_log, weights=np.abs(coeffs).astype(np.float64))

        corr = autocorrelation_nonnegative(coeff_bins)
        abs_corr = autocorrelation_nonnegative(abs_bins)
        lag_index = np.arange(corr.size, dtype=np.float64)
        epsilon_bound = args.epsilon_c / (float(start) ** args.epsilon_alpha)

        for t_scale in t_scales:
            scaled = t_scale * delta * lag_index
            kernel = sinc_kernel_real(scaled)
            signed_raw, unsigned_raw = kernel_offdiag(corr, abs_corr, diagonal_raw, kernel)
            signed_norm = signed_raw / float(start * start)
            unsigned_norm = unsigned_raw / float(start * start)
            kernel_ratio = abs(signed_raw) / unsigned_raw if unsigned_raw > 0 else float("nan")
            local_ratio, local_shell = local_shell_worst_ratio(corr, abs_corr, diagonal_raw, scaled, shell_bounds)

            rows.append(
                {
                    "Y": start,
                    "stop": stop - 1,
                    "T": t_scale,
                    "block_nonzero_count": int(block_n.size),
                    "unique_product_count": int(products.size),
                    "bin_count": args.bin_count,
                    "aggregation": args.aggregation,
                    "diagonal_4": diagonal_raw / float(start * start),
                    "kernel_signed_norm": signed_norm,
                    "kernel_unsigned_norm": unsigned_norm,
                    "kernel_ratio": kernel_ratio,
                    "local_worst_ratio": local_ratio,
                    "local_worst_shell": local_shell,
                    "epsilon_bound": epsilon_bound,
                    "kernel_over_local": kernel_ratio / local_ratio if local_ratio > 0 else float("nan"),
                    "kernel_over_epsilon_bound": kernel_ratio / epsilon_bound if epsilon_bound > 0 else float("nan"),
                    "local_over_epsilon_bound": local_ratio / epsilon_bound if epsilon_bound > 0 else float("nan"),
                }
            )

        block_rows = [row for row in rows if int(row["Y"]) == start]
        worst_kernel = max(block_rows, key=lambda item: float(item["kernel_ratio"]))
        worst_local = max(block_rows, key=lambda item: float(item["local_worst_ratio"]))
        print(
            f"Y={start} worst_kernel={float(worst_kernel['kernel_ratio']):.6g} "
            f"T={float(worst_kernel['T']):g} worst_local={float(worst_local['local_worst_ratio']):.6g}"
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
    t_values = t_scales
    heatmap = np.full((len(y_values), len(t_values)), np.nan, dtype=np.float64)
    for row in rows:
        y_index = y_values.index(int(row["Y"]))
        t_index = t_values.index(float(row["T"]))
        heatmap[y_index, t_index] = float(row["kernel_ratio"])

    worst_kernel_points = []
    worst_local_points = []
    epsilon_points = []
    for y in y_values:
        y_rows = [row for row in rows if int(row["Y"]) == y]
        worst_kernel = max(float(row["kernel_ratio"]) for row in y_rows)
        worst_local = max(float(row["local_worst_ratio"]) for row in y_rows)
        epsilon_bound = float(y_rows[0]["epsilon_bound"])
        worst_kernel_points.append((y, worst_kernel))
        worst_local_points.append((y, worst_local))
        epsilon_points.append((y, epsilon_bound))

    all_alpha, all_c, all_r2 = fit_power_law(worst_kernel_points)
    tail_points = [(y, ratio) for y, ratio in worst_kernel_points if y >= 1024]
    tail_alpha, tail_c, tail_r2 = fit_power_law(tail_points)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(3, 1, figsize=(13, 12))

    image = axes[0].imshow(
        np.log10(np.maximum(heatmap, 1e-12)),
        aspect="auto",
        interpolation="nearest",
        cmap="magma",
    )
    axes[0].set_xticks(np.arange(len(t_values)), [f"{value:g}" for value in t_values])
    axes[0].set_yticks(np.arange(len(y_values)), [str(value) for value in y_values])
    axes[0].set_xlabel("T scale")
    axes[0].set_ylabel("dyadic start Y")
    axes[0].set_title("log10 kernel-weighted off-diagonal cancellation ratio")
    colorbar = fig.colorbar(image, ax=axes[0])
    colorbar.set_label("log10 abs(weighted signed) / weighted unsigned")

    y_array = np.array([point[0] for point in worst_kernel_points], dtype=np.float64)
    axes[1].loglog(y_array, [point[1] for point in worst_kernel_points], marker="o", label="worst kernel ratio")
    axes[1].loglog(y_array, [point[1] for point in worst_local_points], marker="o", label="worst local ratio")
    axes[1].loglog(y_array, [point[1] for point in epsilon_points], linestyle="--", label=f"{args.epsilon_c:g}/Y^{args.epsilon_alpha:g}")
    axes[1].set_xlabel("dyadic start Y")
    axes[1].set_ylabel("ratio")
    axes[1].set_title("Kernel-weighted cancellation versus local autocorrelation envelope")
    axes[1].legend(loc="best")

    for t_scale in t_values:
        t_points = [
            (int(row["Y"]), float(row["kernel_ratio"]))
            for row in rows
            if float(row["T"]) == float(t_scale)
        ]
        t_points.sort()
        axes[2].semilogy(
            [point[0] for point in t_points],
            [max(point[1], 1e-12) for point in t_points],
            marker="o",
            linewidth=1.0,
            label=f"T={t_scale:g}",
        )
    axes[2].set_xscale("log", base=2)
    axes[2].set_xlabel("dyadic start Y")
    axes[2].set_ylabel("kernel ratio")
    axes[2].set_title("Kernel-weighted cancellation by T")
    axes[2].legend(loc="best", ncol=4)

    fig.tight_layout()
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=160)

    overall = max(rows, key=lambda item: float(item["kernel_ratio"]))
    print(args.output_png)
    if args.output_csv:
        print(args.output_csv)
    print(
        f"overall_worst_kernel_ratio={float(overall['kernel_ratio']):.6g} "
        f"Y={int(overall['Y'])} T={float(overall['T']):g}"
    )
    print(f"kernel_fit_all_alpha={all_alpha:.6g} C={all_c:.6g} r2={all_r2:.6g}")
    print(f"kernel_fit_tail_min_y=1024 alpha={tail_alpha:.6g} C={tail_c:.6g} r2={tail_r2:.6g}")
    for y, ratio in worst_kernel_points:
        y_rows = [row for row in rows if int(row["Y"]) == y]
        worst = max(y_rows, key=lambda item: float(item["kernel_ratio"]))
        print(
            f"Y={y} max_kernel_ratio={ratio:.6g} T={float(worst['T']):g} "
            f"over_epsilon={float(worst['kernel_over_epsilon_bound']):.6g}"
        )


if __name__ == "__main__":
    main()

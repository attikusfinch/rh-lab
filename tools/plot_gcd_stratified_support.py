from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_lambda_support_vs_full_interval import fit_power_law
from plot_local_autocorr_bound import (
    autocorrelation_nonnegative,
    dyadic_edges,
    parse_float_list,
    parse_shell_bounds,
    parse_t_scales,
    shell_label,
    shell_sum,
)


def parse_gcd_bounds(value: str) -> list[float]:
    bounds = parse_float_list(value, "gcd-bounds")
    if len(bounds) < 2 or bounds[0] != 1 or bounds[-1] != float("inf"):
        raise SystemExit("--gcd-bounds must start with 1 and end with inf")
    if any(bounds[index] >= bounds[index + 1] for index in range(len(bounds) - 1)):
        raise SystemExit("--gcd-bounds must be increasing")
    return bounds


def gcd_label(low: float, high: float) -> str:
    if math.isinf(high):
        return f"{int(low)}+"
    if int(high) == int(low) + 1:
        return str(int(low))
    return f"{int(low)}-{int(high) - 1}"


def reduce_product_coeffs(products: np.ndarray, coeffs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if products.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    order = np.argsort(products, kind="mergesort")
    sorted_products = products[order]
    sorted_coeffs = coeffs[order].astype(np.int64, copy=False)
    unique_products, starts = np.unique(sorted_products, return_index=True)
    coeff_sums = np.add.reduceat(sorted_coeffs, starts)
    keep = coeff_sums != 0
    return unique_products[keep], coeff_sums[keep]


def merge_product_coeffs(parts: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    nonempty = [part for part in parts if part[0].size > 0]
    if not nonempty:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    if len(nonempty) == 1:
        return nonempty[0]
    products = np.concatenate([part[0] for part in nonempty])
    coeffs = np.concatenate([part[1] for part in nonempty])
    return reduce_product_coeffs(products, coeffs)


def product_coeffs_by_gcd(
    block_n: np.ndarray,
    block_signs: np.ndarray,
    gcd_bounds: list[float],
    row_chunk_size: int,
    merge_row_chunks: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    if merge_row_chunks <= 0:
        raise SystemExit("--merge-row-chunks must be positive")

    labels = [gcd_label(low, high) for low, high in zip(gcd_bounds[:-1], gcd_bounds[1:])]
    accumulators: dict[str, tuple[np.ndarray, np.ndarray]] = {
        label: (np.array([], dtype=np.int64), np.array([], dtype=np.int64)) for label in labels
    }
    pending: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {label: [] for label in labels}

    for chunk_index, row_start in enumerate(range(0, block_n.size, row_chunk_size), start=1):
        row_stop = min(row_start + row_chunk_size, block_n.size)
        row_n = block_n[row_start:row_stop]
        row_signs = block_signs[row_start:row_stop]
        products = (row_n[:, None] * block_n[None, :]).ravel()
        coeffs = (row_signs[:, None] * block_signs[None, :]).astype(np.int16).ravel()
        gcds = np.gcd(row_n[:, None], block_n[None, :]).ravel()

        for low, high in zip(gcd_bounds[:-1], gcd_bounds[1:]):
            label = gcd_label(low, high)
            if math.isinf(high):
                mask = gcds >= int(low)
            else:
                mask = (gcds >= int(low)) & (gcds < int(high))
            if np.any(mask):
                pending[label].append(reduce_product_coeffs(products[mask], coeffs[mask]))

        if chunk_index % merge_row_chunks == 0:
            for label in labels:
                accumulators[label] = merge_product_coeffs([accumulators[label], *pending[label]])
                pending[label] = []

    return {label: merge_product_coeffs([accumulators[label], *pending[label]]) for label in labels}


def scan_ratio(
    products: np.ndarray,
    coeffs: np.ndarray,
    edges_log: np.ndarray,
    delta: float,
    t_scales: list[float],
    shell_bounds: list[float],
    min_shell_unsigned_raw: float,
) -> tuple[float, float, str, float, float]:
    if products.size == 0:
        return float("nan"), float("nan"), "", 0.0, 0.0

    log_products = np.log(products.astype(np.float64))
    signed_bins, _ = np.histogram(log_products, bins=edges_log, weights=coeffs.astype(np.float64))
    abs_bins, _ = np.histogram(log_products, bins=edges_log, weights=np.abs(coeffs).astype(np.float64))
    corr = autocorrelation_nonnegative(signed_bins)
    abs_corr = autocorrelation_nonnegative(abs_bins)
    diagonal_raw = float(np.sum(coeffs.astype(np.float64) ** 2))
    lag_index = np.arange(corr.size, dtype=np.float64)

    worst_ratio = -1.0
    worst_t = 0.0
    worst_shell = ""
    for t_scale in t_scales:
        scaled = t_scale * delta * lag_index
        for low, high in zip(shell_bounds[:-1], shell_bounds[1:]):
            if math.isinf(high):
                mask = scaled >= low
            else:
                mask = (scaled >= low) & (scaled < high)
            include_h0 = low == 0
            signed_raw = shell_sum(corr, diagonal_raw, mask, include_h0)
            unsigned_raw = shell_sum(abs_corr, diagonal_raw, mask, include_h0)
            if unsigned_raw < min_shell_unsigned_raw:
                continue
            ratio = abs(signed_raw) / unsigned_raw
            if math.isfinite(ratio) and ratio > worst_ratio:
                worst_ratio = ratio
                worst_t = t_scale
                worst_shell = shell_label(low, high)

    signed_balance = float(np.sum(coeffs) / np.sum(np.abs(coeffs))) if np.sum(np.abs(coeffs)) > 0 else float("nan")
    if worst_ratio < 0:
        worst_ratio = float("nan")
    return worst_ratio, worst_t, worst_shell, diagonal_raw, signed_balance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mu_bin", type=Path)
    parser.add_argument("output_png", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--first", type=int, default=128)
    parser.add_argument("--max-y", type=int, default=8192)
    parser.add_argument("--t-scales", default="50,75,100,150,200,300,500,750,1000,1500,2000,3000,5000")
    parser.add_argument("--bin-count", type=int, default=65536)
    parser.add_argument("--shell-bounds", default="0,1,2,5,10,20,50,inf")
    parser.add_argument("--gcd-bounds", default="1,2,4,8,16,32,64,128,256,512,1024,inf")
    parser.add_argument("--pair-row-chunk-size", type=int, default=512)
    parser.add_argument("--merge-row-chunks", type=int, default=4)
    parser.add_argument("--min-shell-unsigned-raw", type=float, default=1000.0)
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
    gcd_bounds = parse_gcd_bounds(args.gcd_bounds)
    gcd_labels = [gcd_label(low, high) for low, high in zip(gcd_bounds[:-1], gcd_bounds[1:])]

    mu = raw[1:]
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

        edges_log = np.linspace(math.log(start * start), math.log(stop * stop), args.bin_count + 1)
        delta = float(edges_log[1] - edges_log[0])
        by_gcd = product_coeffs_by_gcd(
            block_n,
            block_signs,
            gcd_bounds,
            args.pair_row_chunk_size,
            args.merge_row_chunks,
        )

        all_products, all_coeffs = merge_product_coeffs(list(by_gcd.values()))
        all_ratio, all_t, all_shell, all_diagonal_raw, all_sign_balance = scan_ratio(
            all_products,
            all_coeffs,
            edges_log,
            delta,
            t_scales,
            shell_bounds,
            args.min_shell_unsigned_raw,
        )
        total_pair_count = float(block_n.size * block_n.size)

        rows.append(
            {
                "Y": start,
                "stop": stop - 1,
                "gcd_bin": "all",
                "unique_product_count": int(all_products.size),
                "pair_representation_count": int(np.sum(np.abs(all_coeffs))),
                "pair_fraction": 1.0,
                "diagonal_raw": all_diagonal_raw,
                "diagonal_fraction_of_all": 1.0,
                "pair_weighted_worst_ratio": all_ratio,
                "diagonal_weighted_worst_ratio": all_ratio,
                "sign_balance": all_sign_balance,
                "worst_ratio": all_ratio,
                "worst_T": all_t,
                "worst_shell": all_shell,
            }
        )

        for label in gcd_labels:
            products, coeffs = by_gcd[label]
            if products.size == 0:
                continue
            ratio, t_scale, shell, diagonal_raw, sign_balance = scan_ratio(
                products,
                coeffs,
                edges_log,
                delta,
                t_scales,
                shell_bounds,
                args.min_shell_unsigned_raw,
            )
            pair_count = float(np.sum(np.abs(coeffs)))
            pair_fraction = pair_count / total_pair_count
            diagonal_fraction = diagonal_raw / all_diagonal_raw if all_diagonal_raw > 0 else float("nan")
            rows.append(
                {
                    "Y": start,
                    "stop": stop - 1,
                    "gcd_bin": label,
                    "unique_product_count": int(products.size),
                    "pair_representation_count": int(pair_count),
                    "pair_fraction": pair_fraction,
                    "diagonal_raw": diagonal_raw,
                    "diagonal_fraction_of_all": diagonal_fraction,
                    "pair_weighted_worst_ratio": pair_fraction * ratio if math.isfinite(ratio) else float("nan"),
                    "diagonal_weighted_worst_ratio": diagonal_fraction * ratio if math.isfinite(ratio) else float("nan"),
                    "sign_balance": sign_balance,
                    "worst_ratio": ratio,
                    "worst_T": t_scale,
                    "worst_shell": shell,
                }
            )

        block_rows = [row for row in rows if int(row["Y"]) == start and row["gcd_bin"] != "all"]
        worst_layer = max(
            block_rows,
            key=lambda item: float(item["worst_ratio"]) if math.isfinite(float(item["worst_ratio"])) else -1.0,
        )
        largest_layer = max(block_rows, key=lambda item: float(item["pair_fraction"]))
        print(
            f"Y={start} all={all_ratio:.6g} "
            f"worst_gcd={worst_layer['gcd_bin']}:{float(worst_layer['worst_ratio']):.6g} "
            f"largest_gcd={largest_layer['gcd_bin']}:{float(largest_layer['pair_fraction']):.3f}"
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
    active_labels = [
        label
        for label in gcd_labels
        if any(row["gcd_bin"] == label for row in rows)
    ]
    heatmap = np.full((len(active_labels), len(y_values)), np.nan, dtype=np.float64)
    pair_fraction = np.full_like(heatmap, np.nan)
    diagonal_fraction = np.full_like(heatmap, np.nan)
    for row in rows:
        if row["gcd_bin"] == "all":
            continue
        y_index = y_values.index(int(row["Y"]))
        label_index = active_labels.index(str(row["gcd_bin"]))
        heatmap[label_index, y_index] = float(row["worst_ratio"])
        pair_fraction[label_index, y_index] = float(row["pair_fraction"])
        diagonal_fraction[label_index, y_index] = float(row["diagonal_fraction_of_all"])

    all_points = [
        (int(row["Y"]), float(row["worst_ratio"]))
        for row in rows
        if row["gcd_bin"] == "all"
    ]
    all_points.sort()
    all_alpha, _all_c, all_r2 = fit_power_law(all_points)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(3, 1, figsize=(13, 13))

    image = axes[0].imshow(
        np.log10(np.maximum(heatmap, 1e-12)),
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
    )
    axes[0].set_xticks(np.arange(len(y_values)), [str(value) for value in y_values])
    axes[0].set_yticks(np.arange(len(active_labels)), active_labels)
    axes[0].set_xlabel("dyadic start Y")
    axes[0].set_ylabel("gcd(a,b) bin")
    axes[0].set_title("log10 worst local ratio by gcd stratum")
    colorbar = fig.colorbar(image, ax=axes[0])
    colorbar.set_label("log10 worst local ratio")

    axes[1].loglog(
        [point[0] for point in all_points],
        [point[1] for point in all_points],
        marker="o",
        linewidth=1.4,
        color="#111111",
        label=f"all gcd strata, alpha={all_alpha:.3f}, R2={all_r2:.3f}",
    )
    for label in active_labels[:8]:
        points = [
            (int(row["Y"]), float(row["worst_ratio"]))
            for row in rows
            if row["gcd_bin"] == label
        ]
        points.sort()
        axes[1].loglog(
            [point[0] for point in points],
            [point[1] for point in points],
            marker="o",
            linewidth=1.0,
            label=label,
        )
    axes[1].set_xlabel("dyadic start Y")
    axes[1].set_ylabel("worst local ratio")
    axes[1].set_title("GCD-stratum local cancellation curves")
    axes[1].legend(loc="best", ncol=3)

    last_y_index = len(y_values) - 1
    x = np.arange(len(active_labels))
    axes[2].bar(x - 0.18, pair_fraction[:, last_y_index], width=0.36, label="pair fraction")
    axes[2].bar(x + 0.18, diagonal_fraction[:, last_y_index], width=0.36, label="diagonal fraction")
    axes[2].set_yscale("log")
    axes[2].set_xticks(x, active_labels, rotation=35, ha="right")
    axes[2].set_ylabel("fraction at largest Y")
    axes[2].set_title(f"GCD-stratum mass at Y={y_values[-1]}")
    axes[2].legend(loc="best")

    fig.tight_layout()
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=160)

    overall_layer = max(
        (row for row in rows if row["gcd_bin"] != "all"),
        key=lambda item: float(item["worst_ratio"]) if math.isfinite(float(item["worst_ratio"])) else -1.0,
    )
    print(args.output_png)
    if args.output_csv:
        print(args.output_csv)
    print(f"max_y={args.max_y}")
    print(f"bin_count={args.bin_count}")
    print(f"all_fit_alpha={all_alpha:.6g} r2={all_r2:.6g}")
    print(
        f"overall_worst_gcd_layer={overall_layer['gcd_bin']} "
        f"Y={int(overall_layer['Y'])} ratio={float(overall_layer['worst_ratio']):.6g}"
    )


if __name__ == "__main__":
    main()

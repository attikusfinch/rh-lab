from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def dyadic_edges(max_y: int, first: int) -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    y = first
    while y <= max_y:
        edges.append((y, 2 * y))
        y *= 2
    return edges


def parse_float_list(value: str, name: str) -> list[float]:
    values: list[float] = []
    for part in value.split(","):
        part = part.strip().lower()
        if not part:
            continue
        values.append(float("inf") if part in {"inf", "infty", "infinity"} else float(part))
    if not values:
        raise SystemExit(f"--{name} must not be empty")
    return values


def parse_shell_bounds(value: str) -> list[float]:
    bounds = parse_float_list(value, "shell-bounds")
    if len(bounds) < 2 or bounds[0] != 0 or bounds[-1] != float("inf"):
        raise SystemExit("--shell-bounds must start with 0 and end with inf")
    if any(bounds[index] >= bounds[index + 1] for index in range(len(bounds) - 1)):
        raise SystemExit("--shell-bounds must be increasing")
    return bounds


def parse_t_scales(value: str) -> list[float]:
    scales = parse_float_list(value, "t-scales")
    if any((not math.isfinite(scale)) or scale <= 0 for scale in scales):
        raise SystemExit("--t-scales must contain positive finite values")
    return scales


def shell_label(left: float, right: float) -> str:
    def clean(value: float) -> str:
        if math.isinf(value):
            return "inf"
        return f"{value:g}"

    return f"{clean(left)}..{clean(right)}"


def product_coefficients(
    block_n: np.ndarray,
    block_signs: np.ndarray,
    row_chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    products_parts: list[np.ndarray] = []
    coeff_parts: list[np.ndarray] = []

    for row_start in range(0, block_n.size, row_chunk_size):
        row_stop = min(row_start + row_chunk_size, block_n.size)
        products = (block_n[row_start:row_stop, None] * block_n[None, :]).ravel()
        coeffs = (block_signs[row_start:row_stop, None] * block_signs[None, :]).astype(np.int16).ravel()
        products_parts.append(products)
        coeff_parts.append(coeffs)

    products_all = np.concatenate(products_parts)
    coeffs_all = np.concatenate(coeff_parts)
    order = np.argsort(products_all, kind="mergesort")
    sorted_products = products_all[order]
    sorted_coeffs = coeffs_all[order].astype(np.int64)
    unique_products, starts = np.unique(sorted_products, return_index=True)
    coeff_sums = np.add.reduceat(sorted_coeffs, starts)
    keep = coeff_sums != 0
    coeffs = coeff_sums[keep]
    diagonal_raw = float(np.sum(coeffs.astype(np.float64) ** 2))
    return unique_products[keep], coeffs, diagonal_raw


def next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


def autocorrelation_nonnegative(values: np.ndarray) -> np.ndarray:
    size = next_power_of_two(2 * values.size - 1)
    transform = np.fft.rfft(values.astype(np.float64), size)
    correlation = np.fft.irfft(transform * np.conj(transform), size)
    return correlation[: values.size]


def shell_sum(
    corr: np.ndarray,
    diagonal_raw: float,
    mask: np.ndarray,
    include_diagonal_correction: bool,
) -> float:
    if include_diagonal_correction:
        mask_without_zero = mask.copy()
        mask_without_zero[0] = False
        return float(corr[0] - diagonal_raw + np.sum(2.0 * corr[mask_without_zero]))
    return float(np.sum(2.0 * corr[mask]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mu_bin", type=Path)
    parser.add_argument("output_png", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--first", type=int, default=128)
    parser.add_argument("--max-y", type=int, default=8192)
    parser.add_argument("--t-scales", default="100,200,500,1000,2000")
    parser.add_argument("--bin-count", type=int, default=65536)
    parser.add_argument("--shell-bounds", default="0,1,2,5,10,20,50,inf")
    parser.add_argument("--pair-row-chunk-size", type=int, default=256)
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
    shell_pairs = list(zip(shell_bounds[:-1], shell_bounds[1:]))
    mu = raw[1:]
    nonzero = np.nonzero(mu)[0].astype(np.int64) + 1
    signs = mu[nonzero - 1].astype(np.float64)
    edges = dyadic_edges(args.max_y, args.first)
    rows: list[dict[str, float | int | str]] = []
    summary_rows: list[dict[str, float | int | str]] = []

    for start, stop in edges:
        left = np.searchsorted(nonzero, start, side="left")
        right = np.searchsorted(nonzero, stop, side="left")
        block_n = nonzero[left:right]
        block_signs = signs[left:right].astype(np.int64)
        if block_n.size == 0:
            continue

        products, coeffs, diagonal_raw = product_coefficients(block_n, block_signs, args.pair_row_chunk_size)
        log_products = np.log(products.astype(np.float64))
        edges_log = np.linspace(float(log_products[0]), float(log_products[-1]), args.bin_count + 1)
        delta = float(edges_log[1] - edges_log[0])
        coeff_bins, _ = np.histogram(log_products, bins=edges_log, weights=coeffs.astype(np.float64))
        abs_bins, _ = np.histogram(log_products, bins=edges_log, weights=np.abs(coeffs).astype(np.float64))

        corr = autocorrelation_nonnegative(coeff_bins)
        abs_corr = autocorrelation_nonnegative(abs_bins)
        lag_index = np.arange(corr.size, dtype=np.float64)

        for t_scale in t_scales:
            scaled = t_scale * delta * lag_index
            worst_ratio = -1.0
            worst_shell = ""
            worst_signed = 0.0
            worst_unsigned = 0.0

            for low, high in shell_pairs:
                if math.isinf(high):
                    mask = scaled >= low
                else:
                    mask = (scaled >= low) & (scaled < high)
                include_h0 = low == 0
                signed_raw = shell_sum(corr, diagonal_raw, mask, include_h0)
                unsigned_raw = shell_sum(abs_corr, diagonal_raw, mask, include_h0)
                signed_norm = signed_raw / float(start * start)
                unsigned_norm = unsigned_raw / float(start * start)
                ratio = abs(signed_raw) / unsigned_raw if unsigned_raw > 0 else float("nan")
                label = shell_label(low, high)

                rows.append(
                    {
                        "Y": start,
                        "stop": stop - 1,
                        "T": t_scale,
                        "shell": label,
                        "shell_low": low,
                        "shell_high": high,
                        "bin_count": args.bin_count,
                        "scaled_bin_width": t_scale * delta,
                        "block_nonzero_count": int(block_n.size),
                        "unique_product_count": int(products.size),
                        "diagonal_4": diagonal_raw / float(start * start),
                        "signed_raw": signed_raw,
                        "unsigned_raw": unsigned_raw,
                        "signed_norm": signed_norm,
                        "unsigned_norm": unsigned_norm,
                        "ratio": ratio,
                    }
                )

                if math.isfinite(ratio) and ratio > worst_ratio:
                    worst_ratio = ratio
                    worst_shell = label
                    worst_signed = signed_norm
                    worst_unsigned = unsigned_norm

            summary_rows.append(
                {
                    "Y": start,
                    "T": t_scale,
                    "max_shell_ratio": worst_ratio,
                    "max_shell": worst_shell,
                    "max_signed_norm": worst_signed,
                    "max_unsigned_norm": worst_unsigned,
                }
            )

        block_summary = [row for row in summary_rows if int(row["Y"]) == start]
        worst_block = max(block_summary, key=lambda item: float(item["max_shell_ratio"]))
        print(
            f"Y={start} worst_ratio={float(worst_block['max_shell_ratio']):.6g} "
            f"T={float(worst_block['T']):g} shell={worst_block['max_shell']}"
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

    y_values = sorted({int(row["Y"]) for row in summary_rows})
    t_values = t_scales
    heatmap = np.full((len(y_values), len(t_values)), np.nan, dtype=np.float64)
    for row in summary_rows:
        y_index = y_values.index(int(row["Y"]))
        t_index = t_values.index(float(row["T"]))
        heatmap[y_index, t_index] = float(row["max_shell_ratio"])

    shell_labels = [shell_label(low, high) for low, high in shell_pairs]
    shell_max = []
    for label in shell_labels:
        ratios = [float(row["ratio"]) for row in rows if row["shell"] == label and math.isfinite(float(row["ratio"]))]
        shell_max.append(max(ratios) if ratios else float("nan"))

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(3, 1, figsize=(13, 12))

    image = axes[0].imshow(
        np.log10(np.maximum(heatmap, 1e-12)),
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
    )
    axes[0].set_xticks(np.arange(len(t_values)), [f"{value:g}" for value in t_values])
    axes[0].set_yticks(np.arange(len(y_values)), [str(value) for value in y_values])
    axes[0].set_xlabel("T scale")
    axes[0].set_ylabel("dyadic start Y")
    axes[0].set_title("log10 worst local autocorrelation ratio by Y and T")
    colorbar = fig.colorbar(image, ax=axes[0])
    colorbar.set_label("log10 max abs(signed) / unsigned")

    for t_scale in t_values:
        points = [
            (int(row["Y"]), float(row["max_shell_ratio"]))
            for row in summary_rows
            if float(row["T"]) == float(t_scale)
        ]
        points.sort()
        axes[1].semilogy(
            [point[0] for point in points],
            [max(point[1], 1e-12) for point in points],
            marker="o",
            linewidth=1.1,
            label=f"T={t_scale:g}",
        )
    axes[1].set_xscale("log", base=2)
    axes[1].set_ylabel("worst shell ratio")
    axes[1].set_title("Worst local shell cancellation ratio at each dyadic block")
    axes[1].legend(loc="best", ncol=3)

    axes[2].bar(shell_labels, np.maximum(np.array(shell_max), 1e-12), color="#4d73a8")
    axes[2].set_yscale("log")
    axes[2].set_ylabel("max ratio")
    axes[2].set_xlabel("shell in T |Delta log r|")
    axes[2].set_title("Worst observed ratio by shell across all Y and T")

    fig.tight_layout()
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=160)

    overall = max(summary_rows, key=lambda item: float(item["max_shell_ratio"]))
    print(args.output_png)
    if args.output_csv:
        print(args.output_csv)
    print(f"max_y={args.max_y}")
    print(f"bin_count={args.bin_count}")
    print(
        "overall_worst_ratio="
        f"{float(overall['max_shell_ratio']):.6g} "
        f"Y={int(overall['Y'])} T={float(overall['T']):g} shell={overall['max_shell']}"
    )
    for t_scale in t_values:
        t_rows = [row for row in summary_rows if float(row["T"]) == float(t_scale)]
        worst = max(t_rows, key=lambda item: float(item["max_shell_ratio"]))
        print(
            f"T={t_scale:g} max_ratio={float(worst['max_shell_ratio']):.6g} "
            f"Y={int(worst['Y'])} shell={worst['max_shell']}"
        )


if __name__ == "__main__":
    main()

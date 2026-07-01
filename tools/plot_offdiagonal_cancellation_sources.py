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


def sinc_kernel_real(scaled: np.ndarray) -> np.ndarray:
    result = np.ones_like(scaled, dtype=np.float64)
    mask = scaled != 0
    result[mask] = np.sin(scaled[mask]) / scaled[mask]
    return result


def normalized_offdiag_totals(
    corr: np.ndarray,
    abs_corr: np.ndarray,
    kernel: np.ndarray,
    diagonal_raw: float,
    y: int,
) -> dict[str, float]:
    signed_h0 = corr[0] - diagonal_raw
    unsigned_h0 = abs_corr[0] - diagonal_raw
    tail = slice(1, None)
    abs_kernel = np.abs(kernel)

    actual_raw = signed_h0 + float(np.sum(2.0 * corr[tail] * kernel[tail]))
    coeff_only_raw = signed_h0 + float(np.sum(2.0 * corr[tail] * abs_kernel[tail]))
    kernel_only_raw = unsigned_h0 + float(np.sum(2.0 * abs_corr[tail] * kernel[tail]))
    unsigned_raw = abs(unsigned_h0) + float(np.sum(2.0 * abs_corr[tail] * abs_kernel[tail]))

    scale = float(y * y)
    return {
        "actual_signed_kernel": actual_raw / scale,
        "coeff_signs_positive_kernel": coeff_only_raw / scale,
        "unsigned_coeffs_signed_kernel": kernel_only_raw / scale,
        "unsigned_positive_kernel": unsigned_raw / scale,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mu_bin", type=Path)
    parser.add_argument("output_png", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--first", type=int, default=128)
    parser.add_argument("--max-y", type=int, default=8192)
    parser.add_argument("--t-max", type=float, default=500.0)
    parser.add_argument("--bin-count", type=int, default=65536)
    parser.add_argument("--pair-row-chunk-size", type=int, default=256)
    args = parser.parse_args()

    raw = np.fromfile(args.mu_bin, dtype=np.int8)
    if raw.size < 2:
        raise SystemExit("mu binary must contain mu[0..N]")
    if 2 * args.max_y > raw.size - 1:
        raise SystemExit("mu binary is too short for the requested --max-y")
    if args.t_max <= 0:
        raise SystemExit("--t-max must be positive")

    mu = raw[1:]
    nonzero = np.nonzero(mu)[0].astype(np.int64) + 1
    signs = mu[nonzero - 1].astype(np.float64)
    edges = dyadic_edges(args.max_y, args.first)
    rows: list[dict[str, float | int]] = []

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
        scaled = args.t_max * delta * np.arange(corr.size, dtype=np.float64)
        kernel = sinc_kernel_real(scaled)
        totals = normalized_offdiag_totals(corr, abs_corr, kernel, diagonal_raw, start)

        unsigned = totals["unsigned_positive_kernel"]
        row: dict[str, float | int] = {
            "Y": start,
            "stop": stop - 1,
            "block_nonzero_count": int(block_n.size),
            "unique_product_count": int(products.size),
            "bin_count": args.bin_count,
            "scaled_bin_width": args.t_max * delta,
            "diagonal_4": diagonal_raw / float(start * start),
            **totals,
            "actual_cancellation_ratio": abs(totals["actual_signed_kernel"]) / unsigned if unsigned > 0 else float("nan"),
            "coeff_sign_cancellation_ratio": abs(totals["coeff_signs_positive_kernel"]) / unsigned if unsigned > 0 else float("nan"),
            "kernel_oscillation_cancellation_ratio": abs(totals["unsigned_coeffs_signed_kernel"]) / unsigned if unsigned > 0 else float("nan"),
        }
        row["actual_vs_coeff_only_ratio"] = (
            abs(totals["actual_signed_kernel"]) / abs(totals["coeff_signs_positive_kernel"])
            if totals["coeff_signs_positive_kernel"] != 0
            else float("nan")
        )
        row["actual_vs_kernel_only_ratio"] = (
            abs(totals["actual_signed_kernel"]) / abs(totals["unsigned_coeffs_signed_kernel"])
            if totals["unsigned_coeffs_signed_kernel"] != 0
            else float("nan")
        )
        rows.append(row)
        print(
            f"Y={start} actual={totals['actual_signed_kernel']:.6g} "
            f"coeff_only={totals['coeff_signs_positive_kernel']:.6g} "
            f"kernel_only={totals['unsigned_coeffs_signed_kernel']:.6g} "
            f"unsigned={unsigned:.6g}"
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

    y_values = np.array([float(row["Y"]) for row in rows])
    actual = np.array([float(row["actual_signed_kernel"]) for row in rows])
    coeff_only = np.array([float(row["coeff_signs_positive_kernel"]) for row in rows])
    kernel_only = np.array([float(row["unsigned_coeffs_signed_kernel"]) for row in rows])
    unsigned = np.array([float(row["unsigned_positive_kernel"]) for row in rows])
    actual_ratio = np.array([float(row["actual_cancellation_ratio"]) for row in rows])
    coeff_ratio = np.array([float(row["coeff_sign_cancellation_ratio"]) for row in rows])
    kernel_ratio = np.array([float(row["kernel_oscillation_cancellation_ratio"]) for row in rows])

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(3, 1, figsize=(13, 11))

    axes[0].plot(y_values, actual, marker="o", color="#111111", linewidth=1.2, label="A*A with signed K")
    axes[0].plot(y_values, coeff_only, marker="o", color="#2454a6", linewidth=1.1, label="A*A with |K|")
    axes[0].plot(y_values, kernel_only, marker="o", color="#b24a3a", linewidth=1.1, label="|A|*|A| with signed K")
    axes[0].axhline(0.0, color="#777777", alpha=0.45, linewidth=0.9)
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("symlog", linthresh=1e-2)
    axes[0].set_ylabel("normalized offdiag")
    axes[0].set_title("Separating coefficient signs from kernel oscillation")
    axes[0].legend(loc="best")

    axes[1].semilogy(y_values, np.maximum(actual_ratio, 1e-12), marker="o", color="#111111", linewidth=1.2, label="both mechanisms")
    axes[1].semilogy(y_values, np.maximum(coeff_ratio, 1e-12), marker="o", color="#2454a6", linewidth=1.1, label="coefficient signs only")
    axes[1].semilogy(y_values, np.maximum(kernel_ratio, 1e-12), marker="o", color="#b24a3a", linewidth=1.1, label="kernel oscillation only")
    axes[1].set_xscale("log", base=2)
    axes[1].set_ylabel("abs(value) / unsigned |K| mass")
    axes[1].set_title("Cancellation ratios against the same unsigned reference")
    axes[1].legend(loc="best")

    axes[2].plot(y_values, unsigned, marker="o", color="#777777", linewidth=1.2, label="unsigned |A||A||K| mass")
    axes[2].plot(y_values, np.abs(actual), marker="o", color="#111111", linewidth=1.2, label="abs actual")
    axes[2].set_xscale("log", base=2)
    axes[2].set_yscale("log")
    axes[2].set_ylabel("normalized magnitude")
    axes[2].set_xlabel("dyadic start Y")
    axes[2].set_title("Unsigned mass versus surviving off-diagonal")
    axes[2].legend(loc="best")

    fig.tight_layout()
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=160)

    print(args.output_png)
    if args.output_csv:
        print(args.output_csv)
    print(f"max_y={args.max_y}")
    print(f"bin_count={args.bin_count}")
    min_actual_index = int(np.argmin(actual_ratio))
    print(f"min_actual_cancellation_ratio={float(actual_ratio[min_actual_index]):.6g}")
    print(f"min_actual_cancellation_Y={int(y_values[min_actual_index])}")
    min_coeff_index = int(np.argmin(coeff_ratio))
    print(f"min_coeff_sign_cancellation_ratio={float(coeff_ratio[min_coeff_index]):.6g}")
    print(f"min_coeff_sign_cancellation_Y={int(y_values[min_coeff_index])}")
    min_kernel_index = int(np.argmin(kernel_ratio))
    print(f"min_kernel_oscillation_cancellation_ratio={float(kernel_ratio[min_kernel_index]):.6g}")
    print(f"min_kernel_oscillation_Y={int(y_values[min_kernel_index])}")


if __name__ == "__main__":
    main()

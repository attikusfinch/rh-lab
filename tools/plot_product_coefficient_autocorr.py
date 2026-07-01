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


def parse_shell_bounds(value: str) -> list[float]:
    bounds: list[float] = []
    for part in value.split(","):
        part = part.strip().lower()
        if not part:
            continue
        bounds.append(float("inf") if part in {"inf", "infty", "infinity"} else float(part))
    if len(bounds) < 2 or bounds[0] != 0 or bounds[-1] != float("inf"):
        raise SystemExit("--shell-bounds must start with 0 and end with inf")
    if any(bounds[index] >= bounds[index + 1] for index in range(len(bounds) - 2)):
        raise SystemExit("--shell-bounds must be increasing")
    return bounds


def shell_key(left: float, right: float) -> str:
    def clean(value: float) -> str:
        if math.isinf(value):
            return "inf"
        return f"{value:g}".replace(".", "_")

    return f"{clean(left)}_{clean(right)}"


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
    parser.add_argument("--t-scale", type=float, default=500.0)
    parser.add_argument("--bin-count", type=int, default=65536)
    parser.add_argument("--shell-bounds", default="0,1,2,5,10,20,50,inf")
    parser.add_argument("--pair-row-chunk-size", type=int, default=256)
    args = parser.parse_args()

    raw = np.fromfile(args.mu_bin, dtype=np.int8)
    if raw.size < 2:
        raise SystemExit("mu binary must contain mu[0..N]")
    if 2 * args.max_y > raw.size - 1:
        raise SystemExit("mu binary is too short for the requested --max-y")
    if args.t_scale <= 0:
        raise SystemExit("--t-scale must be positive")

    shell_bounds = parse_shell_bounds(args.shell_bounds)
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

        block_mobius_sum = int(np.sum(block_signs))
        products, coeffs, diagonal_raw = product_coefficients(block_n, block_signs, args.pair_row_chunk_size)
        log_products = np.log(products.astype(np.float64))
        edges_log = np.linspace(float(log_products[0]), float(log_products[-1]), args.bin_count + 1)
        delta = float(edges_log[1] - edges_log[0])
        coeff_bins, _ = np.histogram(log_products, bins=edges_log, weights=coeffs.astype(np.float64))
        abs_bins, _ = np.histogram(log_products, bins=edges_log, weights=np.abs(coeffs).astype(np.float64))

        corr = autocorrelation_nonnegative(coeff_bins)
        abs_corr = autocorrelation_nonnegative(abs_bins)
        scaled = args.t_scale * delta * np.arange(corr.size, dtype=np.float64)

        signed_offdiag_total = float((np.sum(coeff_bins) ** 2 - diagonal_raw) / (start * start))
        unsigned_offdiag_total = float((np.sum(abs_bins) ** 2 - diagonal_raw) / (start * start))
        sign_balance = float(np.sum(coeffs) / np.sum(np.abs(coeffs))) if np.sum(np.abs(coeffs)) > 0 else float("nan")
        binned_sign_balance = float(np.sum(coeff_bins) / np.sum(abs_bins)) if np.sum(abs_bins) > 0 else float("nan")

        row: dict[str, float | int] = {
            "Y": start,
            "stop": stop - 1,
            "block_nonzero_count": int(block_n.size),
            "unique_product_count": int(products.size),
            "bin_count": args.bin_count,
            "scaled_bin_width": args.t_scale * delta,
            "diagonal_4": diagonal_raw / float(start * start),
            "block_mobius_sum": block_mobius_sum,
            "sum_A": int(block_mobius_sum * block_mobius_sum),
            "signed_offdiag_identity": float((block_mobius_sum**4 - diagonal_raw) / float(start * start)),
            "signed_offdiag_total": signed_offdiag_total,
            "unsigned_offdiag_total": unsigned_offdiag_total,
            "global_offdiag_cancellation_ratio": (
                abs(signed_offdiag_total) / unsigned_offdiag_total if unsigned_offdiag_total > 0 else float("nan")
            ),
            "coefficient_sign_balance": sign_balance,
            "binned_sign_balance": binned_sign_balance,
        }

        for index in range(len(shell_bounds) - 1):
            low = shell_bounds[index]
            high = shell_bounds[index + 1]
            if math.isinf(high):
                mask = scaled >= low
            else:
                mask = (scaled >= low) & (scaled < high)
            key = shell_key(low, high)
            include_h0 = low == 0
            signed_raw = shell_sum(corr, diagonal_raw, mask, include_h0)
            unsigned_raw = shell_sum(abs_corr, diagonal_raw, mask, include_h0)
            signed = signed_raw / float(start * start)
            unsigned = unsigned_raw / float(start * start)
            row[f"shell_signed_{key}"] = signed
            row[f"shell_unsigned_{key}"] = unsigned
            row[f"shell_cancellation_ratio_{key}"] = abs(signed) / unsigned if unsigned > 0 else float("nan")

        rows.append(row)
        print(
            f"Y={start} global_cancel={row['global_offdiag_cancellation_ratio']:.6g} "
            f"sign_balance={sign_balance:.6g} shell1={row['shell_cancellation_ratio_0_1']:.6g}"
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
    global_cancel = np.array([float(row["global_offdiag_cancellation_ratio"]) for row in rows])
    sign_balance = np.array([abs(float(row["coefficient_sign_balance"])) for row in rows])
    signed_total = np.array([float(row["signed_offdiag_total"]) for row in rows])
    unsigned_total = np.array([float(row["unsigned_offdiag_total"]) for row in rows])

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(3, 1, figsize=(13, 11))

    axes[0].plot(y_values, signed_total, marker="o", color="#111111", linewidth=1.2, label="signed total")
    axes[0].plot(y_values, unsigned_total, marker="o", color="#b24a3a", linewidth=1.1, label="unsigned total")
    axes[0].axhline(0.0, color="#777777", alpha=0.45, linewidth=0.9)
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("symlog", linthresh=1e-2)
    axes[0].set_ylabel("normalized offdiag")
    axes[0].set_title(r"Autocorrelation of product coefficients $A_Y(r)$ without kernel weights")
    axes[0].legend(loc="best")

    axes[1].semilogy(y_values, np.maximum(global_cancel, 1e-12), marker="o", color="#2454a6", linewidth=1.2, label="global offdiag")
    axes[1].semilogy(y_values, np.maximum(sign_balance**2, 1e-12), marker="o", color="#24734d", linewidth=1.1, label="sign balance squared")
    for index in range(len(shell_bounds) - 1):
        low = shell_bounds[index]
        high = shell_bounds[index + 1]
        key = shell_key(low, high)
        ratios = np.array([float(row[f"shell_cancellation_ratio_{key}"]) for row in rows])
        if high <= 10:
            axes[1].semilogy(y_values, np.maximum(ratios, 1e-12), marker="o", linewidth=0.9, label=f"{low:g}..{high:g}")
    axes[1].set_xscale("log", base=2)
    axes[1].set_ylabel("abs(signed) / unsigned")
    axes[1].set_title(r"Cancellation by log-product lag shells, scaled as $T|\Delta\log r|$")
    axes[1].legend(loc="best", ncol=3)

    for index in range(len(shell_bounds) - 1):
        low = shell_bounds[index]
        high = shell_bounds[index + 1]
        key = shell_key(low, high)
        signed = np.array([float(row[f"shell_signed_{key}"]) for row in rows])
        axes[2].plot(y_values, signed, marker="o", linewidth=1.0, label=f"{low:g}..{'inf' if math.isinf(high) else f'{high:g}'}")
    axes[2].axhline(0.0, color="#777777", alpha=0.45, linewidth=0.9)
    axes[2].set_xscale("log", base=2)
    axes[2].set_ylabel("signed shell mass")
    axes[2].set_xlabel("dyadic start Y")
    axes[2].set_title("Signed autocorrelation shell contributions without kernel")
    axes[2].legend(loc="best", ncol=3)

    fig.tight_layout()
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=160)

    print(args.output_png)
    if args.output_csv:
        print(args.output_csv)
    print(f"max_y={args.max_y}")
    print(f"bin_count={args.bin_count}")
    min_global_index = int(np.argmin(global_cancel))
    print(f"min_global_offdiag_cancellation_ratio={float(global_cancel[min_global_index]):.6g}")
    print(f"min_global_offdiag_cancellation_Y={int(y_values[min_global_index])}")
    min_shell_key = ""
    min_shell_ratio = float("inf")
    min_shell_y = 0
    for index in range(len(shell_bounds) - 1):
        low = shell_bounds[index]
        high = shell_bounds[index + 1]
        key = shell_key(low, high)
        ratios = np.array([float(row[f"shell_cancellation_ratio_{key}"]) for row in rows])
        shell_index = int(np.argmin(ratios))
        if ratios[shell_index] < min_shell_ratio:
            min_shell_ratio = float(ratios[shell_index])
            min_shell_key = key
            min_shell_y = int(y_values[shell_index])
    print(f"min_shell_cancellation_ratio={min_shell_ratio:.6g}")
    print(f"min_shell={min_shell_key}")
    print(f"min_shell_Y={min_shell_y}")


if __name__ == "__main__":
    main()

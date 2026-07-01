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


def block_scan(
    n_values: np.ndarray,
    signs: np.ndarray,
    log_values: np.ndarray,
    t_values: np.ndarray,
    start: int,
    stop: int,
    chunk_size: int,
) -> np.ndarray:
    left = np.searchsorted(n_values, start, side="left")
    right = np.searchsorted(n_values, stop, side="left")
    result = np.zeros(t_values.shape, dtype=np.complex128)

    for chunk_start in range(left, right, chunk_size):
        chunk_stop = min(chunk_start + chunk_size, right)
        phase = np.exp(-1j * np.outer(log_values[chunk_start:chunk_stop], t_values))
        result += signs[chunk_start:chunk_stop] @ phase

    return result


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mu_bin", type=Path)
    parser.add_argument("output_png", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--first", type=int, default=128)
    parser.add_argument("--max-y", type=int, default=8192)
    parser.add_argument("--t-min", type=float, default=0.0)
    parser.add_argument("--t-max", type=float, default=500.0)
    parser.add_argument("--t-points", type=int, default=1601)
    parser.add_argument("--bin-count", type=int, default=65536)
    parser.add_argument("--shell-bounds", default="0,1,2,5,10,20,50,inf")
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--pair-row-chunk-size", type=int, default=256)
    args = parser.parse_args()

    raw = np.fromfile(args.mu_bin, dtype=np.int8)
    if raw.size < 2:
        raise SystemExit("mu binary must contain mu[0..N]")
    if 2 * args.max_y > raw.size - 1:
        raise SystemExit("mu binary is too short for the requested --max-y")
    if args.bin_count < 1024:
        raise SystemExit("--bin-count must be at least 1024")

    t_length = args.t_max - args.t_min
    if t_length <= 0:
        raise SystemExit("--t-max must be greater than --t-min")

    shell_bounds = parse_shell_bounds(args.shell_bounds)
    mu = raw[1:]
    nonzero = np.nonzero(mu)[0].astype(np.int64) + 1
    signs = mu[nonzero - 1].astype(np.float64)
    log_values = np.log(nonzero.astype(np.float64))
    t_values = np.linspace(args.t_min, args.t_max, args.t_points)
    edges = dyadic_edges(args.max_y, args.first)

    rows: list[dict[str, float | int]] = []

    for start, stop in edges:
        left = np.searchsorted(nonzero, start, side="left")
        right = np.searchsorted(nonzero, stop, side="left")
        block_n = nonzero[left:right]
        block_signs = signs[left:right].astype(np.int64)
        if block_n.size == 0:
            continue

        sums = block_scan(nonzero, signs, log_values, t_values, start, stop, args.chunk_size)
        normalized = np.abs(sums) / np.sqrt(start)
        grid_moment_4 = float(np.mean(normalized**4))

        products, coeffs, diagonal_raw = product_coefficients(block_n, block_signs, args.pair_row_chunk_size)
        log_products = np.log(products.astype(np.float64))
        log_min = float(log_products[0])
        log_max = float(log_products[-1])
        edges_log = np.linspace(log_min, log_max, args.bin_count + 1)
        delta = float(edges_log[1] - edges_log[0])
        coeff_bins, _ = np.histogram(log_products, bins=edges_log, weights=coeffs.astype(np.float64))
        abs_bins, _ = np.histogram(log_products, bins=edges_log, weights=np.abs(coeffs).astype(np.float64))

        corr = autocorrelation_nonnegative(coeff_bins)
        abs_corr = autocorrelation_nonnegative(abs_bins)
        lags = np.arange(corr.size, dtype=np.float64)
        scaled = t_length * delta * lags
        kernel = sinc_kernel_real(scaled)

        diagonal = diagonal_raw / float(start * start)
        row: dict[str, float | int] = {
            "Y": start,
            "stop": stop - 1,
            "block_nonzero_count": int(block_n.size),
            "unique_product_count": int(products.size),
            "bin_count": args.bin_count,
            "log_bin_width": delta,
            "scaled_bin_width": t_length * delta,
            "grid_moment_4": grid_moment_4,
            "diagonal_4": diagonal,
        }

        offdiag_total = 0.0
        unsigned_total = 0.0
        for index in range(len(shell_bounds) - 1):
            low = shell_bounds[index]
            high = shell_bounds[index + 1]
            if math.isinf(high):
                mask = scaled >= low
            else:
                mask = (scaled >= low) & (scaled < high)
            key = shell_key(low, high)

            if low == 0:
                h0_signed = corr[0] - diagonal_raw
                h0_unsigned = abs_corr[0] - diagonal_raw
                mask_without_zero = mask.copy()
                mask_without_zero[0] = False
                signed_raw = h0_signed + float(np.sum(2.0 * corr[mask_without_zero] * kernel[mask_without_zero]))
                unsigned_raw = abs(h0_unsigned) + float(np.sum(2.0 * abs_corr[mask_without_zero] * np.abs(kernel[mask_without_zero])))
            else:
                signed_raw = float(np.sum(2.0 * corr[mask] * kernel[mask]))
                unsigned_raw = float(np.sum(2.0 * abs_corr[mask] * np.abs(kernel[mask])))

            signed = signed_raw / float(start * start)
            unsigned = unsigned_raw / float(start * start)
            offdiag_total += signed
            unsigned_total += unsigned
            row[f"shell_signed_{key}"] = signed
            row[f"shell_unsigned_kernel_{key}"] = unsigned
            row[f"shell_cancellation_ratio_{key}"] = abs(signed) / unsigned if unsigned > 0 else float("nan")

        kernel_integral = diagonal + offdiag_total
        row["offdiag_kernel_total"] = offdiag_total
        row["unsigned_kernel_total"] = unsigned_total
        row["kernel_integral_4"] = kernel_integral
        row["kernel_minus_grid"] = kernel_integral - grid_moment_4
        row["kernel_to_grid_ratio"] = kernel_integral / grid_moment_4 if grid_moment_4 > 0 else float("nan")
        row["offdiag_to_kernel_ratio"] = offdiag_total / kernel_integral if kernel_integral != 0 else float("nan")
        row["offdiag_cancellation_ratio"] = abs(offdiag_total) / unsigned_total if unsigned_total > 0 else float("nan")
        rows.append(row)

        print(
            f"Y={start} grid={grid_moment_4:.6g} kernel={kernel_integral:.6g} "
            f"diag={diagonal:.6g} offdiag={offdiag_total:.6g} "
            f"cancel={row['offdiag_cancellation_ratio']:.6g}"
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
    grid_values = np.array([float(row["grid_moment_4"]) for row in rows])
    kernel_values = np.array([float(row["kernel_integral_4"]) for row in rows])
    diagonal_values = np.array([float(row["diagonal_4"]) for row in rows])
    offdiag_values = np.array([float(row["offdiag_kernel_total"]) for row in rows])
    cancellation_values = np.array([float(row["offdiag_cancellation_ratio"]) for row in rows])

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(3, 1, figsize=(13, 11))

    axes[0].plot(y_values, grid_values, marker="o", color="#2454a6", linewidth=1.2, label="grid 4th moment")
    axes[0].plot(y_values, kernel_values, marker="o", color="#b24a3a", linewidth=1.1, label="binned kernel integral")
    axes[0].plot(y_values, diagonal_values, marker="o", color="#24734d", linewidth=1.1, label="exact diagonal")
    axes[0].set_xscale("log", base=2)
    axes[0].set_ylabel("normalized 4th moment")
    axes[0].set_title(r"Fourth moment through the off-diagonal kernel $K_T(\log(r/s))$")
    axes[0].legend(loc="best")

    axes[1].plot(y_values, offdiag_values, marker="o", color="#111111", linewidth=1.1, label="kernel off-diagonal")
    axes[1].plot(y_values, kernel_values - grid_values, marker="o", color="#5b3f9b", linewidth=1.1, label="kernel - grid")
    axes[1].axhline(0.0, color="#777777", alpha=0.45, linewidth=0.9)
    axes[1].set_xscale("log", base=2)
    axes[1].set_ylabel("normalized contribution")
    axes[1].set_title("Finite-interval off-diagonal contribution")
    axes[1].legend(loc="best")

    for index in range(len(shell_bounds) - 1):
        low = shell_bounds[index]
        high = shell_bounds[index + 1]
        key = shell_key(low, high)
        signed = np.array([float(row[f"shell_signed_{key}"]) for row in rows])
        axes[2].plot(y_values, signed, marker="o", linewidth=1.0, label=f"{low:g}..{'inf' if math.isinf(high) else f'{high:g}'}")
    axes[2].axhline(0.0, color="#777777", alpha=0.45, linewidth=0.9)
    axes[2].set_xscale("log", base=2)
    axes[2].set_ylabel("signed shell contribution")
    axes[2].set_xlabel("dyadic start Y")
    axes[2].set_title(r"Off-diagonal shells by $T|\log(r/s)|$")
    axes[2].legend(loc="best", ncol=3)

    fig.tight_layout()
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=160)

    print(args.output_png)
    if args.output_csv:
        print(args.output_csv)
    print(f"t_grid={args.t_min}..{args.t_max} points={len(t_values)}")
    print(f"max_y={args.max_y}")
    print(f"bin_count={args.bin_count}")
    max_abs_error_index = int(np.argmax(np.abs(kernel_values - grid_values)))
    print(f"max_abs_kernel_minus_grid={float(abs(kernel_values[max_abs_error_index] - grid_values[max_abs_error_index])):.6g}")
    print(f"max_abs_kernel_minus_grid_Y={int(y_values[max_abs_error_index])}")
    min_cancel_index = int(np.argmin(cancellation_values))
    print(f"min_offdiag_cancellation_ratio={float(cancellation_values[min_cancel_index]):.6g}")
    print(f"min_offdiag_cancellation_Y={int(y_values[min_cancel_index])}")


if __name__ == "__main__":
    main()

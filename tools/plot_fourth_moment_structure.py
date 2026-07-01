from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_floats(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def dyadic_edges(max_y: int, first: int) -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    y = first
    while y <= max_y:
        edges.append((y, 2 * y))
        y *= 2
    return edges


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
) -> tuple[np.ndarray, np.ndarray, int]:
    products_parts: list[np.ndarray] = []
    coeff_parts: list[np.ndarray] = []
    pair_count = int(block_n.size * block_n.size)

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
    return unique_products[keep], coeff_sums[keep], pair_count


def window_masses(
    log_products: np.ndarray,
    coeffs: np.ndarray,
    width: float,
    diagonal: float,
) -> tuple[float, float]:
    abs_coeffs = np.abs(coeffs)
    coeff_prefix = np.concatenate(([0.0], np.cumsum(coeffs, dtype=np.float64)))
    abs_prefix = np.concatenate(([0.0], np.cumsum(abs_coeffs, dtype=np.float64)))

    left = np.searchsorted(log_products, log_products - width, side="left")
    right = np.searchsorted(log_products, log_products + width, side="right")

    signed_with_diagonal = float(np.sum(coeffs * (coeff_prefix[right] - coeff_prefix[left])))
    unsigned_with_diagonal = float(np.sum(abs_coeffs * (abs_prefix[right] - abs_prefix[left])))
    return signed_with_diagonal - diagonal, unsigned_with_diagonal - diagonal


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mu_bin", type=Path)
    parser.add_argument("output_png", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--first", type=int, default=128)
    parser.add_argument("--max-y", type=int, default=4096)
    parser.add_argument("--t-min", type=float, default=0.0)
    parser.add_argument("--t-max", type=float, default=500.0)
    parser.add_argument("--t-points", type=int, default=1601)
    parser.add_argument("--near-scales", default="1,2,5,10")
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--pair-row-chunk-size", type=int, default=256)
    args = parser.parse_args()

    raw = np.fromfile(args.mu_bin, dtype=np.int8)
    if raw.size < 2:
        raise SystemExit("mu binary must contain mu[0..N]")
    if 2 * args.max_y > raw.size - 1:
        raise SystemExit("mu binary is too short for the requested --max-y")

    mu = raw[1:]
    nonzero = np.nonzero(mu)[0].astype(np.int64) + 1
    signs = mu[nonzero - 1].astype(np.float64)
    log_values = np.log(nonzero.astype(np.float64))
    t_values = np.linspace(args.t_min, args.t_max, args.t_points)
    t_length = args.t_max - args.t_min
    if t_length <= 0:
        raise SystemExit("--t-max must be greater than --t-min")

    near_scales = parse_floats(args.near_scales)
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
        moment_2 = float(np.mean(normalized**2))
        moment_4_grid = float(np.mean(normalized**4))
        gaussian_4 = 2.0 * moment_2 * moment_2

        products, coeffs, pair_count = product_coefficients(block_n, block_signs, args.pair_row_chunk_size)
        diagonal_raw = float(np.sum(coeffs.astype(np.float64) ** 2))
        diagonal = diagonal_raw / float(start * start)
        offdiag_grid = moment_4_grid - diagonal
        log_products = np.log(products.astype(np.float64))

        row: dict[str, float | int] = {
            "Y": start,
            "stop": stop - 1,
            "block_nonzero_count": int(block_n.size),
            "ordered_pair_count": pair_count,
            "unique_product_count": int(products.size),
            "moment_2_grid": moment_2,
            "moment_4_grid": moment_4_grid,
            "gaussian_4_reference": gaussian_4,
            "moment_4_to_gaussian_ratio": moment_4_grid / gaussian_4 if gaussian_4 > 0 else float("nan"),
            "diagonal_4": diagonal,
            "diagonal_to_grid_ratio": diagonal / moment_4_grid if moment_4_grid > 0 else float("nan"),
            "offdiag_grid": offdiag_grid,
            "offdiag_to_grid_ratio": offdiag_grid / moment_4_grid if moment_4_grid > 0 else float("nan"),
        }

        for scale in near_scales:
            width = scale / t_length
            signed, unsigned = window_masses(log_products, coeffs.astype(np.float64), width, diagonal_raw)
            signed_norm = signed / float(start * start)
            unsigned_norm = unsigned / float(start * start)
            key = f"{scale:g}".replace(".", "_")
            row[f"near_signed_scale_{key}"] = signed_norm
            row[f"near_unsigned_scale_{key}"] = unsigned_norm
            row[f"near_cancellation_ratio_scale_{key}"] = (
                abs(signed_norm) / unsigned_norm if unsigned_norm > 0 else float("nan")
            )

        rows.append(row)
        print(
            f"Y={start} m4={moment_4_grid:.6g} diag={diagonal:.6g} "
            f"diag/grid={row['diagonal_to_grid_ratio']:.6g} "
            f"offdiag={offdiag_grid:.6g}"
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
    moment_4 = np.array([float(row["moment_4_grid"]) for row in rows])
    diagonal = np.array([float(row["diagonal_4"]) for row in rows])
    gaussian = np.array([float(row["gaussian_4_reference"]) for row in rows])
    local_ratio = np.array([float(row["moment_4_to_gaussian_ratio"]) for row in rows])
    diagonal_ratio = np.array([float(row["diagonal_to_grid_ratio"]) for row in rows])
    offdiag_ratio = np.array([float(row["offdiag_to_grid_ratio"]) for row in rows])

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(3, 1, figsize=(13, 11))

    axes[0].plot(y_values, moment_4, marker="o", color="#2454a6", linewidth=1.2, label="grid 4th moment")
    axes[0].plot(y_values, diagonal, marker="o", color="#24734d", linewidth=1.1, label="exact product diagonal")
    axes[0].plot(y_values, gaussian, marker="o", color="#b24a3a", linewidth=1.1, label=r"$2(E|g|^2)^2$")
    axes[0].set_xscale("log", base=2)
    axes[0].set_ylabel("normalized 4th moment")
    axes[0].set_title(r"Fourth moment structure for $g_Y(t)=B_Y(t)/\sqrt{Y}$")
    axes[0].legend(loc="best")

    axes[1].plot(y_values, local_ratio, marker="o", color="#5b3f9b", linewidth=1.1, label="4th / Gaussian")
    axes[1].plot(y_values, diagonal_ratio, marker="o", color="#24734d", linewidth=1.1, label="diagonal / grid")
    axes[1].plot(y_values, offdiag_ratio, marker="o", color="#111111", linewidth=1.1, label="offdiag / grid")
    axes[1].axhline(1.0, color="#777777", alpha=0.35, linewidth=0.8)
    axes[1].axhline(0.0, color="#777777", alpha=0.35, linewidth=0.8)
    axes[1].set_xscale("log", base=2)
    axes[1].set_ylabel("ratio")
    axes[1].set_title("Diagonal and finite-interval off-diagonal contribution")
    axes[1].legend(loc="best", ncol=3)

    for scale in near_scales:
        key = f"{scale:g}".replace(".", "_")
        ratios = np.array([float(row[f"near_cancellation_ratio_scale_{key}"]) for row in rows])
        axes[2].plot(y_values, ratios, marker="o", linewidth=1.0, label=f"|log ratio| <= {scale:g}/T")
    axes[2].set_xscale("log", base=2)
    axes[2].set_ylabel("abs(signed near mass) / unsigned near mass")
    axes[2].set_xlabel("dyadic start Y")
    axes[2].set_title("Cancellation in near-product off-diagonal windows")
    axes[2].legend(loc="best", ncol=2)

    fig.tight_layout()
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=160)

    print(args.output_png)
    if args.output_csv:
        print(args.output_csv)
    print(f"t_grid={args.t_min}..{args.t_max} points={len(t_values)}")
    print(f"max_y={args.max_y}")
    best_ratio_index = int(np.argmax(local_ratio))
    print(f"max_m4_to_gaussian_ratio={float(local_ratio[best_ratio_index]):.6g}")
    print(f"max_m4_to_gaussian_Y={int(y_values[best_ratio_index])}")
    max_diag_index = int(np.argmax(diagonal_ratio))
    print(f"max_diagonal_to_grid_ratio={float(diagonal_ratio[max_diag_index]):.6g}")
    print(f"max_diagonal_to_grid_Y={int(y_values[max_diag_index])}")


if __name__ == "__main__":
    main()

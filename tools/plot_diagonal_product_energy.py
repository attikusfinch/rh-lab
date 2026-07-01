from __future__ import annotations

import argparse
import csv
import gc
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BUCKETS = [
    ("abs1", 1, 1),
    ("abs2", 2, 2),
    ("abs3_4", 3, 4),
    ("abs5_8", 5, 8),
    ("abs9_16", 9, 16),
    ("abs17_plus", 17, None),
]


def dyadic_edges(max_y: int, first: int) -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    y = first
    while y <= max_y:
        edges.append((y, 2 * y))
        y *= 2
    return edges


def product_table(
    block_n: np.ndarray,
    block_signs: np.ndarray,
    row_chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
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
    counts = np.diff(np.append(starts, sorted_products.size)).astype(np.int64)
    return unique_products, coeff_sums, counts, pair_count


def bucket_energy(abs_coeffs: np.ndarray, coeffs: np.ndarray, low: int, high: int | None) -> float:
    if high is None:
        mask = abs_coeffs >= low
    else:
        mask = (abs_coeffs >= low) & (abs_coeffs <= high)
    return float(np.sum(coeffs[mask].astype(np.float64) ** 2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mu_bin", type=Path)
    parser.add_argument("output_png", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--first", type=int, default=128)
    parser.add_argument("--max-y", type=int, default=8192)
    parser.add_argument("--pair-row-chunk-size", type=int, default=256)
    args = parser.parse_args()

    raw = np.fromfile(args.mu_bin, dtype=np.int8)
    if raw.size < 2:
        raise SystemExit("mu binary must contain mu[0..N]")
    if 2 * args.max_y > raw.size - 1:
        raise SystemExit("mu binary is too short for the requested --max-y")

    mu = raw[1:]
    nonzero = np.nonzero(mu)[0].astype(np.int64) + 1
    signs = mu[nonzero - 1].astype(np.int64)
    edges = dyadic_edges(args.max_y, args.first)
    rows: list[dict[str, float | int]] = []

    for start, stop in edges:
        left = np.searchsorted(nonzero, start, side="left")
        right = np.searchsorted(nonzero, stop, side="left")
        block_n = nonzero[left:right]
        block_signs = signs[left:right]
        if block_n.size == 0:
            continue

        products, coeffs, counts, pair_count = product_table(block_n, block_signs, args.pair_row_chunk_size)
        abs_coeffs = np.abs(coeffs)
        nonzero_coeff_mask = coeffs != 0
        signed_energy_raw = float(np.sum(coeffs.astype(np.float64) ** 2))
        unsigned_energy_raw = float(np.sum(counts.astype(np.float64) ** 2))
        baseline_raw = float(pair_count)
        signed_diagonal = signed_energy_raw / float(start * start)
        unsigned_collision = unsigned_energy_raw / float(start * start)
        baseline = baseline_raw / float(start * start)

        max_abs_index = int(np.argmax(abs_coeffs))
        max_count_index = int(np.argmax(counts))

        row: dict[str, float | int] = {
            "Y": start,
            "stop": stop - 1,
            "block_nonzero_count": int(block_n.size),
            "ordered_pair_count": pair_count,
            "raw_unique_product_count": int(products.size),
            "nonzero_coeff_product_count": int(np.count_nonzero(nonzero_coeff_mask)),
            "zero_coeff_product_fraction": float(1.0 - np.count_nonzero(nonzero_coeff_mask) / products.size),
            "baseline_pair_mass": baseline,
            "signed_diagonal": signed_diagonal,
            "unsigned_collision": unsigned_collision,
            "signed_over_baseline": signed_diagonal / baseline if baseline > 0 else float("nan"),
            "unsigned_over_baseline": unsigned_collision / baseline if baseline > 0 else float("nan"),
            "signed_over_unsigned": signed_diagonal / unsigned_collision if unsigned_collision > 0 else float("nan"),
            "max_abs_coeff": int(abs_coeffs[max_abs_index]),
            "max_abs_coeff_product": int(products[max_abs_index]),
            "max_abs_coeff_multiplicity": int(counts[max_abs_index]),
            "max_multiplicity": int(counts[max_count_index]),
            "max_multiplicity_product": int(products[max_count_index]),
            "coeff_at_max_multiplicity": int(coeffs[max_count_index]),
        }

        for name, low, high in BUCKETS:
            energy = bucket_energy(abs_coeffs, coeffs, low, high)
            row[f"energy_{name}"] = energy / float(start * start)
            row[f"energy_fraction_{name}"] = energy / signed_energy_raw if signed_energy_raw > 0 else float("nan")

        rows.append(row)
        print(
            f"Y={start} signed={signed_diagonal:.6g} "
            f"unsigned={unsigned_collision:.6g} "
            f"signed/unsigned={row['signed_over_unsigned']:.6g} "
            f"max|A|={row['max_abs_coeff']}"
        )

        del products, coeffs, counts, abs_coeffs
        gc.collect()

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
    signed = np.array([float(row["signed_diagonal"]) for row in rows])
    unsigned = np.array([float(row["unsigned_collision"]) for row in rows])
    baseline = np.array([float(row["baseline_pair_mass"]) for row in rows])
    signed_over_baseline = np.array([float(row["signed_over_baseline"]) for row in rows])
    unsigned_over_baseline = np.array([float(row["unsigned_over_baseline"]) for row in rows])
    signed_over_unsigned = np.array([float(row["signed_over_unsigned"]) for row in rows])
    bucket_fractions = [
        np.array([float(row[f"energy_fraction_{name}"]) for row in rows])
        for name, _low, _high in BUCKETS
    ]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(3, 1, figsize=(13, 11))

    axes[0].plot(y_values, baseline, marker="o", color="#777777", linewidth=1.1, label="ordered pairs / Y^2")
    axes[0].plot(y_values, signed, marker="o", color="#2454a6", linewidth=1.2, label=r"$\sum A(r)^2 / Y^2$")
    axes[0].plot(y_values, unsigned, marker="o", color="#b24a3a", linewidth=1.1, label="unsigned collisions / Y^2")
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("normalized energy")
    axes[0].set_title(r"Diagonal product energy for $A_Y(r)=\sum_{ab=r}\mu(a)\mu(b)$")
    axes[0].legend(loc="best")

    axes[1].plot(y_values, signed_over_baseline, marker="o", color="#2454a6", linewidth=1.1, label="signed / pair baseline")
    axes[1].plot(y_values, unsigned_over_baseline, marker="o", color="#b24a3a", linewidth=1.1, label="unsigned / pair baseline")
    axes[1].plot(y_values, signed_over_unsigned, marker="o", color="#24734d", linewidth=1.1, label="signed / unsigned")
    axes[1].set_xscale("log", base=2)
    axes[1].set_ylabel("ratio")
    axes[1].set_title("Collision inflation and exact-diagonal sign rigidity")
    axes[1].legend(loc="best")

    axes[2].stackplot(
        y_values,
        bucket_fractions,
        labels=[name for name, _low, _high in BUCKETS],
        alpha=0.9,
    )
    axes[2].set_xscale("log", base=2)
    axes[2].set_ylim(0.0, 1.0)
    axes[2].set_ylabel("fraction of signed energy")
    axes[2].set_xlabel("dyadic start Y")
    axes[2].set_title(r"Contribution by coefficient size $|A_Y(r)|$")
    axes[2].legend(loc="upper left", ncol=3)

    fig.tight_layout()
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=160)

    print(args.output_png)
    if args.output_csv:
        print(args.output_csv)
    print(f"max_y={args.max_y}")
    max_signed_index = int(np.argmax(signed))
    print(f"max_signed_diagonal={float(signed[max_signed_index]):.6g}")
    print(f"max_signed_diagonal_Y={int(y_values[max_signed_index])}")
    min_cancel_index = int(np.argmin(signed_over_unsigned))
    print(f"min_signed_over_unsigned={float(signed_over_unsigned[min_cancel_index]):.6g}")
    print(f"min_signed_over_unsigned_Y={int(y_values[min_cancel_index])}")
    max_coeff_index = int(np.argmax([int(row["max_abs_coeff"]) for row in rows]))
    print(f"max_abs_coeff={int(rows[max_coeff_index]['max_abs_coeff'])}")
    print(f"max_abs_coeff_Y={int(rows[max_coeff_index]['Y'])}")


if __name__ == "__main__":
    main()

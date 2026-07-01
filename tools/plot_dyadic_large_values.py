from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_thresholds(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def dyadic_edges(limit: int, first: int) -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    y = first
    while y < limit:
        stop = min(2 * y, limit + 1)
        if stop > y + 1:
            edges.append((y, stop))
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mu_bin", type=Path)
    parser.add_argument("output_png", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--limit", type=int, default=2_000_000)
    parser.add_argument("--first", type=int, default=128)
    parser.add_argument("--t-min", type=float, default=0.0)
    parser.add_argument("--t-max", type=float, default=200.0)
    parser.add_argument("--t-points", type=int, default=801)
    parser.add_argument("--thresholds", default="1,1.25,1.5,1.75,2,2.25")
    parser.add_argument("--chunk-size", type=int, default=2048)
    args = parser.parse_args()

    raw = np.fromfile(args.mu_bin, dtype=np.int8)
    if raw.size < 2:
        raise SystemExit("mu binary must contain mu[0..N]")

    full_limit = raw.size - 1
    limit = min(args.limit, full_limit)
    mu = raw[1 : limit + 1]
    nonzero = np.nonzero(mu)[0].astype(np.int64) + 1
    signs = mu[nonzero - 1].astype(np.float64)
    log_values = np.log(nonzero.astype(np.float64))
    t_values = np.linspace(args.t_min, args.t_max, args.t_points)
    edges = dyadic_edges(limit, args.first)
    thresholds = parse_thresholds(args.thresholds)

    rows: list[dict[str, float | int]] = []
    all_normalized: list[np.ndarray] = []

    for start, stop in edges:
        sums = block_scan(nonzero, signs, log_values, t_values, start, stop, args.chunk_size)
        normalized = np.abs(sums) / np.sqrt(start)
        all_normalized.append(normalized)

        row: dict[str, float | int] = {
            "Y": start,
            "stop": stop - 1,
            "length": stop - start,
            "mean_square_over_Y": float(np.mean(normalized**2)),
            "mean_over_sqrt_Y": float(np.mean(normalized)),
            "median_over_sqrt_Y": float(np.median(normalized)),
            "p95_over_sqrt_Y": float(np.quantile(normalized, 0.95)),
            "p99_over_sqrt_Y": float(np.quantile(normalized, 0.99)),
            "max_over_sqrt_Y": float(np.max(normalized)),
        }
        for threshold in thresholds:
            row[f"frac_gt_{threshold:g}"] = float(np.mean(normalized > threshold))
        rows.append(row)

        print(
            f"Y={start} mean_sq={row['mean_square_over_Y']:.6g} "
            f"p95={row['p95_over_sqrt_Y']:.6g} max={row['max_over_sqrt_Y']:.6g}"
        )

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "Y",
            "stop",
            "length",
            "mean_square_over_Y",
            "mean_over_sqrt_Y",
            "median_over_sqrt_Y",
            "p95_over_sqrt_Y",
            "p99_over_sqrt_Y",
            "max_over_sqrt_Y",
        ] + [f"frac_gt_{threshold:g}" for threshold in thresholds]
        with args.output_csv.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    y_values = np.array([float(row["Y"]) for row in rows])
    mean_square = np.array([float(row["mean_square_over_Y"]) for row in rows])
    p95 = np.array([float(row["p95_over_sqrt_Y"]) for row in rows])
    p99 = np.array([float(row["p99_over_sqrt_Y"]) for row in rows])
    max_values = np.array([float(row["max_over_sqrt_Y"]) for row in rows])
    flattened = np.concatenate(all_normalized) if all_normalized else np.array([], dtype=np.float64)

    survival_x = np.linspace(0.0, max(2.5, float(np.max(flattened)) if flattened.size else 2.5), 180)
    survival_y = np.array([np.mean(flattened > value) for value in survival_x])

    threshold_matrix = np.array([
        [float(row[f"frac_gt_{threshold:g}"]) for threshold in thresholds]
        for row in rows
    ])

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(3, 1, figsize=(13, 11))

    axes[0].plot(y_values, mean_square, marker="o", color="#2454a6", linewidth=1.1, label="mean square / Y")
    axes[0].plot(y_values, p95, marker="o", color="#b24a3a", linewidth=1.0, label="95th percentile")
    axes[0].plot(y_values, p99, marker="o", color="#5b3f9b", linewidth=1.0, label="99th percentile")
    axes[0].plot(y_values, max_values, marker="o", color="#111111", linewidth=1.0, label="max")
    axes[0].set_xscale("log", base=2)
    axes[0].set_ylabel("normalized statistic")
    axes[0].set_title(r"Large values of $|B_Y(t)| / \sqrt{Y}$")
    axes[0].legend(loc="best", ncol=4)

    axes[1].semilogy(survival_x, np.maximum(survival_y, 1.0 / (len(flattened) + 1)), color="#24734d", linewidth=1.2)
    for threshold in thresholds:
        axes[1].axvline(threshold, color="#777777", alpha=0.25, linewidth=0.8)
    axes[1].set_ylabel("fraction above level")
    axes[1].set_xlabel(r"level $\lambda$")
    axes[1].set_title(r"Survival curve over all scanned $(Y,t)$")

    image = axes[2].imshow(
        threshold_matrix,
        origin="lower",
        aspect="auto",
        cmap="magma",
        extent=[min(thresholds), max(thresholds), np.log2(y_values[0]), np.log2(y_values[-1])],
    )
    axes[2].set_xlabel(r"threshold $\lambda$")
    axes[2].set_ylabel("log2 dyadic Y")
    axes[2].set_title(r"Fraction of grid with $|B_Y(t)|/\sqrt{Y} > \lambda$")
    fig.colorbar(image, ax=axes[2], label="fraction")

    fig.tight_layout()
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=160)

    global_max = float(np.max(max_values))
    global_max_y = int(y_values[int(np.argmax(max_values))])
    print(args.output_png)
    if args.output_csv:
        print(args.output_csv)
    print(f"limit={limit}")
    print(f"t_grid={args.t_min}..{args.t_max} points={len(t_values)}")
    print(f"global_max={global_max:.6g}")
    print(f"global_max_Y={global_max_y}")
    print(f"overall_mean_square={float(np.mean(flattened**2)):.6g}")
    for threshold in thresholds:
        print(f"overall_frac_gt_{threshold:g}={float(np.mean(flattened > threshold)):.6g}")


if __name__ == "__main__":
    main()

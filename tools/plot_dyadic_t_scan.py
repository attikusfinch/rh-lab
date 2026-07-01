from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np


def zero_gammas(t_max: float) -> list[float]:
    mp.mp.dps = 40
    gammas: list[float] = []
    index = 1
    while True:
        gamma = float(mp.im(mp.zetazero(index)))
        if gamma > t_max:
            break
        gammas.append(gamma)
        index += 1
    return gammas


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

    heatmap = np.zeros((len(edges), len(t_values)), dtype=np.float64)
    summary_rows: list[dict[str, float | int]] = []

    for row_index, (start, stop) in enumerate(edges):
        sums = block_scan(nonzero, signs, log_values, t_values, start, stop, args.chunk_size)
        normalized = np.abs(sums) / np.sqrt(start)
        heatmap[row_index, :] = normalized

        max_index = int(np.argmax(normalized))
        summary_rows.append({
            "Y": start,
            "stop": stop - 1,
            "length": stop - start,
            "max_over_sqrt_Y": float(normalized[max_index]),
            "t_at_max": float(t_values[max_index]),
            "mean_over_sqrt_Y": float(np.mean(normalized)),
            "p95_over_sqrt_Y": float(np.quantile(normalized, 0.95)),
        })
        print(
            f"Y={start} max={normalized[max_index]:.6g} "
            f"t={t_values[max_index]:.6g}"
        )

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=[
                "Y",
                "stop",
                "length",
                "max_over_sqrt_Y",
                "t_at_max",
                "mean_over_sqrt_Y",
                "p95_over_sqrt_Y",
            ])
            writer.writeheader()
            writer.writerows(summary_rows)

    y_values = np.array([start for start, _stop in edges], dtype=np.float64)
    max_values = np.array([float(row["max_over_sqrt_Y"]) for row in summary_rows])
    max_t = np.array([float(row["t_at_max"]) for row in summary_rows])
    mean_values = np.array([float(row["mean_over_sqrt_Y"]) for row in summary_rows])
    p95_values = np.array([float(row["p95_over_sqrt_Y"]) for row in summary_rows])

    gammas = zero_gammas(args.t_max)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), height_ratios=[1.35, 1.0, 1.0])

    extent = [args.t_min, args.t_max, np.log2(y_values[0]), np.log2(y_values[-1])]
    image = axes[0].imshow(
        heatmap,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="viridis",
    )
    for gamma in gammas:
        axes[0].axvline(gamma, color="#ffffff", alpha=0.18, linewidth=0.8)
    axes[0].set_ylabel("log2 dyadic Y")
    axes[0].set_title(r"Heatmap of $|B_Y(t)| / \sqrt{Y}$")
    fig.colorbar(image, ax=axes[0], label=r"$|B_Y(t)| / \sqrt{Y}$")

    axes[1].plot(y_values, max_values, marker="o", color="#2454a6", linewidth=1.1, label="max over t")
    axes[1].plot(y_values, p95_values, marker="o", color="#b24a3a", linewidth=1.0, label="95th percentile")
    axes[1].plot(y_values, mean_values, marker="o", color="#24734d", linewidth=1.0, label="mean")
    axes[1].set_xscale("log", base=2)
    axes[1].set_ylabel(r"$|B_Y(t)| / \sqrt{Y}$")
    axes[1].set_title("Worst-case and typical normalized resonance")
    axes[1].legend(loc="best")

    axes[2].plot(y_values, max_t, marker="o", color="#5b3f9b", linewidth=1.1)
    axes[2].set_xscale("log", base=2)
    axes[2].set_ylabel("t at max")
    axes[2].set_xlabel("dyadic start Y")
    axes[2].set_title("Grid frequency where each dyadic block is worst")

    fig.tight_layout()
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=160)

    print(args.output_png)
    if args.output_csv:
        print(args.output_csv)
    print(f"limit={limit}")
    print(f"t_grid={args.t_min}..{args.t_max} points={len(t_values)}")
    print(f"blocks={len(edges)}")
    print(f"global_max={float(np.max(max_values)):.6g}")
    print(f"global_max_Y={int(y_values[int(np.argmax(max_values))])}")
    print(f"global_max_t={float(max_t[int(np.argmax(max_values))]):.6g}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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


def top_local_candidate_indices(values: np.ndarray, top_k: int) -> list[int]:
    if values.size == 0:
        return []

    candidates = [0, int(values.size - 1), int(np.argmax(values))]
    if values.size >= 3:
        local = np.where((values[1:-1] >= values[:-2]) & (values[1:-1] >= values[2:]))[0] + 1
        candidates.extend(int(index) for index in local)

    unique = sorted(set(candidates), key=lambda index: values[index], reverse=True)
    return unique[:top_k]


def refine_candidate(
    n_values: np.ndarray,
    signs: np.ndarray,
    log_values: np.ndarray,
    start: int,
    stop: int,
    center: float,
    initial_radius: float,
    t_min: float,
    t_max: float,
    levels: int,
    refine_points: int,
    shrink: float,
    chunk_size: int,
) -> tuple[float, float]:
    best_t = center
    best_value = -1.0
    radius = initial_radius

    for _level in range(levels):
        left = max(t_min, best_t - radius)
        right = min(t_max, best_t + radius)
        t_values = np.linspace(left, right, refine_points)
        values = np.abs(block_scan(n_values, signs, log_values, t_values, start, stop, chunk_size)) / np.sqrt(start)
        index = int(np.argmax(values))
        best_t = float(t_values[index])
        best_value = float(values[index])
        radius /= shrink

    return best_t, best_value


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
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--refine-radius", type=float, default=0.5)
    parser.add_argument("--refine-levels", type=int, default=4)
    parser.add_argument("--refine-points", type=int, default=61)
    parser.add_argument("--shrink", type=float, default=5.0)
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
    t_grid = np.linspace(args.t_min, args.t_max, args.t_points)
    edges = dyadic_edges(limit, args.first)

    rows: list[dict[str, float | int]] = []

    for start, stop in edges:
        coarse_values = np.abs(block_scan(nonzero, signs, log_values, t_grid, start, stop, args.chunk_size)) / np.sqrt(start)
        coarse_index = int(np.argmax(coarse_values))
        coarse_t = float(t_grid[coarse_index])
        coarse_max = float(coarse_values[coarse_index])

        best_t = coarse_t
        best_value = coarse_max
        candidate_indices = top_local_candidate_indices(coarse_values, args.top_k)
        for candidate_index in candidate_indices:
            refined_t, refined_value = refine_candidate(
                nonzero,
                signs,
                log_values,
                start,
                stop,
                float(t_grid[candidate_index]),
                args.refine_radius,
                args.t_min,
                args.t_max,
                args.refine_levels,
                args.refine_points,
                args.shrink,
                args.chunk_size,
            )
            if refined_value > best_value:
                best_t = refined_t
                best_value = refined_value

        rows.append({
            "Y": start,
            "stop": stop - 1,
            "length": stop - start,
            "coarse_max_over_sqrt_Y": coarse_max,
            "coarse_t_at_max": coarse_t,
            "refined_max_over_sqrt_Y": best_value,
            "refined_t_at_max": best_t,
            "improvement_abs": best_value - coarse_max,
            "improvement_rel": (best_value / coarse_max - 1.0) if coarse_max > 0 else 0.0,
        })
        print(
            f"Y={start} coarse={coarse_max:.6g}@{coarse_t:.6g} "
            f"refined={best_value:.6g}@{best_t:.9g}"
        )

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=[
                "Y",
                "stop",
                "length",
                "coarse_max_over_sqrt_Y",
                "coarse_t_at_max",
                "refined_max_over_sqrt_Y",
                "refined_t_at_max",
                "improvement_abs",
                "improvement_rel",
            ])
            writer.writeheader()
            writer.writerows(rows)

    y_values = np.array([float(row["Y"]) for row in rows])
    coarse = np.array([float(row["coarse_max_over_sqrt_Y"]) for row in rows])
    refined = np.array([float(row["refined_max_over_sqrt_Y"]) for row in rows])
    coarse_t = np.array([float(row["coarse_t_at_max"]) for row in rows])
    refined_t = np.array([float(row["refined_t_at_max"]) for row in rows])
    improvement_rel = np.array([float(row["improvement_rel"]) for row in rows])

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)

    axes[0].plot(y_values, coarse, marker="o", color="#777777", linewidth=1.0, label="coarse grid")
    axes[0].plot(y_values, refined, marker="o", color="#2454a6", linewidth=1.1, label="refined")
    axes[0].set_xscale("log", base=2)
    axes[0].set_ylabel(r"$\max |B_Y(t)| / \sqrt{Y}$")
    axes[0].set_title("Adaptive refinement of dyadic t-resonance peaks")
    axes[0].legend(loc="best")

    axes[1].plot(y_values, 100.0 * improvement_rel, marker="o", color="#b24a3a", linewidth=1.1)
    axes[1].set_xscale("log", base=2)
    axes[1].set_ylabel("refinement gain, %")

    axes[2].plot(y_values, coarse_t, marker="o", color="#777777", linewidth=1.0, label="coarse t")
    axes[2].plot(y_values, refined_t, marker="o", color="#5b3f9b", linewidth=1.1, label="refined t")
    axes[2].set_xscale("log", base=2)
    axes[2].set_ylabel("t at max")
    axes[2].set_xlabel("dyadic start Y")
    axes[2].legend(loc="best")

    fig.tight_layout()
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=160)

    global_index = int(np.argmax(refined))
    print(args.output_png)
    if args.output_csv:
        print(args.output_csv)
    print(f"limit={limit}")
    print(f"coarse_grid={args.t_min}..{args.t_max} points={args.t_points}")
    print(f"top_k={args.top_k} refine_levels={args.refine_levels} refine_points={args.refine_points}")
    print(f"global_refined_max={float(refined[global_index]):.6g}")
    print(f"global_refined_Y={int(y_values[global_index])}")
    print(f"global_refined_t={float(refined_t[global_index]):.9g}")
    print(f"max_relative_gain_percent={float(100.0 * np.max(improvement_rel)):.6g}")


if __name__ == "__main__":
    main()

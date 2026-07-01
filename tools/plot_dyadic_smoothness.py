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


def block_and_derivative_scan(
    n_values: np.ndarray,
    signs: np.ndarray,
    log_values: np.ndarray,
    t_values: np.ndarray,
    start: int,
    stop: int,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    left = np.searchsorted(n_values, start, side="left")
    right = np.searchsorted(n_values, stop, side="left")
    block = np.zeros(t_values.shape, dtype=np.complex128)
    derivative = np.zeros(t_values.shape, dtype=np.complex128)

    for chunk_start in range(left, right, chunk_size):
        chunk_stop = min(chunk_start + chunk_size, right)
        chunk_logs = log_values[chunk_start:chunk_stop]
        phase = np.exp(-1j * np.outer(chunk_logs, t_values))
        chunk_signs = signs[chunk_start:chunk_stop]
        block += chunk_signs @ phase
        derivative += (-1j * chunk_signs * chunk_logs) @ phase

    return block, derivative


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
    parser.add_argument("--thresholds", default="1.5,1.75,2")
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
    t_length = args.t_max - args.t_min
    edges = dyadic_edges(limit, args.first)
    thresholds = parse_thresholds(args.thresholds)

    rows: list[dict[str, float | int]] = []

    for start, stop in edges:
        block, derivative = block_and_derivative_scan(
            nonzero,
            signs,
            log_values,
            t_values,
            start,
            stop,
            args.chunk_size,
        )
        sqrt_y = np.sqrt(start)
        log_y = np.log(start)
        normalized = np.abs(block) / sqrt_y
        normalized_derivative = np.abs(derivative) / sqrt_y
        derivative_over_log = normalized_derivative / log_y

        max_index = int(np.argmax(normalized))
        max_norm = float(normalized[max_index])
        max_derivative = float(np.max(normalized_derivative))
        mean_square = float(np.mean(normalized**2))
        derivative_mean_square = float(np.mean(normalized_derivative**2))

        # One-dimensional Sobolev-type bound for g(t)=B_Y(t)/sqrt(Y):
        # sup |g|^2 <= mean |g|^2 + 2T sqrt(mean |g|^2 mean |g'|^2).
        sobolev_bound = float(np.sqrt(mean_square + 2.0 * t_length * np.sqrt(mean_square * derivative_mean_square)))

        row: dict[str, float | int] = {
            "Y": start,
            "stop": stop - 1,
            "length": stop - start,
            "max_over_sqrt_Y": max_norm,
            "t_at_max": float(t_values[max_index]),
            "mean_square_over_Y": mean_square,
            "max_derivative_over_sqrt_Y": max_derivative,
            "max_derivative_over_sqrt_Y_logY": float(np.max(derivative_over_log)),
            "mean_derivative_square_over_Y": derivative_mean_square,
            "mean_derivative_square_over_Y_logY2": float(np.mean(derivative_over_log**2)),
            "sobolev_sup_bound": sobolev_bound,
            "sobolev_bound_ratio": sobolev_bound / max_norm if max_norm else 0.0,
        }

        for threshold in thresholds:
            measure = float(t_length * np.mean(normalized > threshold))
            lower_width = max(0.0, max_norm - threshold) / max_derivative if max_derivative else 0.0
            row[f"measure_gt_{threshold:g}"] = measure
            row[f"peak_width_lower_gt_{threshold:g}"] = lower_width
            row[f"measure_over_width_lower_gt_{threshold:g}"] = measure / lower_width if lower_width > 0 else float("nan")

        rows.append(row)
        print(
            f"Y={start} max={max_norm:.6g} "
            f"max|g'|={max_derivative:.6g} "
            f"sobolev={sobolev_bound:.6g}"
        )

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(rows[0].keys()) if rows else []
        with args.output_csv.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    y_values = np.array([float(row["Y"]) for row in rows])
    max_values = np.array([float(row["max_over_sqrt_Y"]) for row in rows])
    max_derivative_log = np.array([float(row["max_derivative_over_sqrt_Y_logY"]) for row in rows])
    derivative_mean_log = np.array([float(row["mean_derivative_square_over_Y_logY2"]) for row in rows])
    sobolev = np.array([float(row["sobolev_sup_bound"]) for row in rows])
    sobolev_ratio = np.array([float(row["sobolev_bound_ratio"]) for row in rows])

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)

    axes[0].plot(y_values, max_values, marker="o", color="#2454a6", linewidth=1.1, label="observed max")
    axes[0].plot(y_values, sobolev, marker="o", color="#b24a3a", linewidth=1.0, label="Sobolev bound")
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("normalized sup")
    axes[0].set_title(r"Derivative/Sobolev control for $g_Y(t)=B_Y(t)/\sqrt{Y}$")
    axes[0].legend(loc="best")

    axes[1].plot(y_values, max_derivative_log, marker="o", color="#5b3f9b", linewidth=1.0, label=r"max $|g'|/\log Y$")
    axes[1].plot(y_values, np.sqrt(derivative_mean_log), marker="o", color="#24734d", linewidth=1.0, label=r"rms $|g'|/\log Y$")
    axes[1].set_xscale("log", base=2)
    axes[1].set_ylabel("derivative scale")
    axes[1].legend(loc="best")

    axes[2].plot(y_values, sobolev_ratio, marker="o", color="#111111", linewidth=1.1)
    axes[2].set_xscale("log", base=2)
    axes[2].set_ylabel("Sobolev / observed")
    axes[2].set_xlabel("dyadic start Y")
    axes[2].set_title("How loose the basic smoothness bound is")

    fig.tight_layout()
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=160)

    print(args.output_png)
    if args.output_csv:
        print(args.output_csv)
    print(f"limit={limit}")
    print(f"t_grid={args.t_min}..{args.t_max} points={len(t_values)}")
    print(f"global_max={float(np.max(max_values)):.6g}")
    print(f"global_sobolev_bound={float(np.max(sobolev)):.6g}")
    print(f"max_sobolev_ratio={float(np.max(sobolev_ratio)):.6g}")
    print(f"median_sobolev_ratio={float(np.median(sobolev_ratio)):.6g}")


if __name__ == "__main__":
    main()

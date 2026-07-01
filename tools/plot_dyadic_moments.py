from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_ints(value: str) -> list[int]:
    result: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if part:
            result.append(int(part))
    return result


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


def moment_stats(values: np.ndarray, moments: list[int]) -> dict[str, float]:
    stats: dict[str, float] = {}
    m2 = float(np.mean(values**2))
    stats["moment_2"] = m2

    for k in moments:
        exponent = 2 * k
        moment = float(np.mean(values**exponent))
        gaussian = math.factorial(k) * (m2**k)
        stats[f"moment_{exponent}"] = moment
        stats[f"gaussian_moment_{exponent}"] = gaussian
        stats[f"ratio_to_gaussian_{exponent}"] = moment / gaussian if gaussian > 0 else float("nan")
        stats[f"gaussian_root_{exponent}"] = (moment / math.factorial(k)) ** (1.0 / exponent)

    return stats


def markov_bound(level: float, moments: dict[int, float]) -> float:
    if level <= 0:
        return 1.0
    return min(1.0, min(moment / (level**exponent) for exponent, moment in moments.items()))


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
    parser.add_argument("--moments", default="1,2,3,4,5,6")
    parser.add_argument("--chunk-size", type=int, default=2048)
    args = parser.parse_args()

    raw = np.fromfile(args.mu_bin, dtype=np.int8)
    if raw.size < 2:
        raise SystemExit("mu binary must contain mu[0..N]")

    moment_orders = sorted(set(parse_ints(args.moments)))
    if not moment_orders or moment_orders[0] < 1:
        raise SystemExit("--moments must contain positive integers")

    full_limit = raw.size - 1
    limit = min(args.limit, full_limit)
    mu = raw[1 : limit + 1]
    nonzero = np.nonzero(mu)[0].astype(np.int64) + 1
    signs = mu[nonzero - 1].astype(np.float64)
    log_values = np.log(nonzero.astype(np.float64))
    t_values = np.linspace(args.t_min, args.t_max, args.t_points)
    edges = dyadic_edges(limit, args.first)

    rows: list[dict[str, float | int]] = []
    all_normalized: list[np.ndarray] = []

    for start, stop in edges:
        sums = block_scan(nonzero, signs, log_values, t_values, start, stop, args.chunk_size)
        normalized = np.abs(sums) / np.sqrt(start)
        all_normalized.append(normalized)

        stats = moment_stats(normalized, moment_orders)
        row: dict[str, float | int] = {
            "Y": start,
            "stop": stop - 1,
            "length": stop - start,
            "max_over_sqrt_Y": float(np.max(normalized)),
            "p99_over_sqrt_Y": float(np.quantile(normalized, 0.99)),
            **stats,
        }
        rows.append(row)

        ratio_4 = float(row.get("ratio_to_gaussian_4", 0.0))
        ratio_8 = float(row.get("ratio_to_gaussian_8", 0.0))
        print(
            f"Y={start} m2={stats['moment_2']:.6g} "
            f"ratio4={ratio_4:.6g} ratio8={ratio_8:.6g} "
            f"max={row['max_over_sqrt_Y']:.6g}"
        )

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(rows[0].keys()) if rows else []
        with args.output_csv.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    y_values = np.array([float(row["Y"]) for row in rows])
    flattened = np.concatenate(all_normalized) if all_normalized else np.array([], dtype=np.float64)
    global_stats = moment_stats(flattened, moment_orders)
    global_moments = {
        2 * k: float(global_stats[f"moment_{2 * k}"])
        for k in moment_orders
    }

    levels = np.linspace(0.05, max(2.5, float(np.max(flattened)) + 0.1), 220)
    empirical_tail = np.array([np.mean(flattened > level) for level in levels])
    best_markov = np.array([markov_bound(float(level), global_moments) for level in levels])
    gaussian_tail = np.exp(-(levels**2) / float(global_stats["moment_2"]))

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(3, 1, figsize=(13, 11))

    for k in moment_orders:
        if k == 1:
            continue
        exponent = 2 * k
        ratios = np.array([float(row[f"ratio_to_gaussian_{exponent}"]) for row in rows])
        axes[0].plot(y_values, ratios, marker="o", linewidth=1.0, label=f"2k={exponent}")
    axes[0].axhline(1.0, color="#111111", alpha=0.45, linewidth=0.9)
    axes[0].set_xscale("log", base=2)
    axes[0].set_ylabel("moment / gaussian")
    axes[0].set_title(r"Dyadic high moments vs complex-Gaussian reference")
    axes[0].legend(loc="best", ncol=3)

    axes[1].plot(
        y_values,
        np.sqrt([float(row["moment_2"]) for row in rows]),
        marker="o",
        color="#111111",
        linewidth=1.1,
        label=r"$\sqrt{E|g|^2}$",
    )
    for k in moment_orders:
        if k == 1:
            continue
        exponent = 2 * k
        roots = np.array([float(row[f"gaussian_root_{exponent}"]) for row in rows])
        axes[1].plot(y_values, roots, marker="o", linewidth=1.0, label=f"2k={exponent}")
    axes[1].set_xscale("log", base=2)
    axes[1].set_ylabel("moment root")
    axes[1].set_title(r"Gaussian-normalized roots $(E|g|^{2k}/k!)^{1/(2k)}$")
    axes[1].legend(loc="best", ncol=4)

    floor = 1.0 / (len(flattened) + 1) if flattened.size else 1.0
    axes[2].semilogy(levels, np.maximum(empirical_tail, floor), color="#2454a6", linewidth=1.3, label="empirical")
    axes[2].semilogy(levels, np.maximum(gaussian_tail, floor), color="#24734d", linewidth=1.1, label="exp(-lambda^2/m2)")
    axes[2].semilogy(levels, np.maximum(best_markov, floor), color="#b24a3a", linewidth=1.1, label="best Markov from moments")
    axes[2].set_xlabel(r"level $\lambda$")
    axes[2].set_ylabel("fraction above level")
    axes[2].set_title(r"Tail check over all scanned $(Y,t)$")
    axes[2].legend(loc="best")

    fig.tight_layout()
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=160)

    print(args.output_png)
    if args.output_csv:
        print(args.output_csv)
    print(f"limit={limit}")
    print(f"t_grid={args.t_min}..{args.t_max} points={len(t_values)}")
    print(f"global_max={float(np.max(flattened)):.6g}")
    for k in moment_orders:
        exponent = 2 * k
        print(f"overall_moment_{exponent}={float(global_stats[f'moment_{exponent}']):.6g}")
        print(f"overall_ratio_to_gaussian_{exponent}={float(global_stats[f'ratio_to_gaussian_{exponent}']):.6g}")
    for level in [1.5, 1.75, 2.0, 2.25]:
        print(f"empirical_tail_gt_{level:g}={float(np.mean(flattened > level)):.6g}")
        print(f"best_markov_gt_{level:g}={markov_bound(level, global_moments):.6g}")
        print(f"gaussian_tail_gt_{level:g}={math.exp(-(level**2) / float(global_stats['moment_2'])):.6g}")


if __name__ == "__main__":
    main()

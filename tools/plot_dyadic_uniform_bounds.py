from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_floats(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def parse_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def dyadic_edges(limit: int, first: int, full_blocks_only: bool) -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    y = first
    while y < limit:
        stop = min(2 * y, limit + 1)
        if full_blocks_only and stop < 2 * y:
            break
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


def best_markov(level: float, moments: dict[int, float]) -> float:
    if level <= 0:
        return 1.0
    return min(1.0, min(moment / (level**exponent) for exponent, moment in moments.items()))


def factorial_markov(level: float, c_value: float, moment_orders: list[int]) -> float:
    if level <= 0:
        return 1.0
    scale = c_value / (level * level)
    return min(1.0, min(math.factorial(k) * (scale**k) for k in moment_orders))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mu_bin", type=Path)
    parser.add_argument("output_png", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--limit", type=int, default=5_000_000)
    parser.add_argument("--first", type=int, default=128)
    parser.add_argument("--t-min", type=float, default=0.0)
    parser.add_argument("--t-max", type=float, default=500.0)
    parser.add_argument("--t-points", type=int, default=1601)
    parser.add_argument("--moments", default="1,2,3,4,5,6,7,8,9,10")
    parser.add_argument("--levels", default="1.25,1.5,1.75,2,2.25")
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--full-blocks-only", action="store_true")
    args = parser.parse_args()

    raw = np.fromfile(args.mu_bin, dtype=np.int8)
    if raw.size < 2:
        raise SystemExit("mu binary must contain mu[0..N]")

    moment_orders = sorted(set(parse_ints(args.moments)))
    if not moment_orders or moment_orders[0] < 1:
        raise SystemExit("--moments must contain positive integers")
    levels = parse_floats(args.levels)

    full_limit = raw.size - 1
    limit = min(args.limit, full_limit)
    mu = raw[1 : limit + 1]
    nonzero = np.nonzero(mu)[0].astype(np.int64) + 1
    signs = mu[nonzero - 1].astype(np.float64)
    log_values = np.log(nonzero.astype(np.float64))
    t_values = np.linspace(args.t_min, args.t_max, args.t_points)
    edges = dyadic_edges(limit, args.first, args.full_blocks_only)

    rows: list[dict[str, float | int]] = []
    block_values: list[np.ndarray] = []

    for start, stop in edges:
        sums = block_scan(nonzero, signs, log_values, t_values, start, stop, args.chunk_size)
        values = np.abs(sums) / np.sqrt(start)
        block_values.append(values)
        m2 = float(np.mean(values**2))

        row: dict[str, float | int] = {
            "Y": start,
            "stop": stop - 1,
            "length": stop - start,
            "moment_2": m2,
            "max_over_sqrt_Y": float(np.max(values)),
            "p99_over_sqrt_Y": float(np.quantile(values, 0.99)),
        }

        for k in moment_orders:
            exponent = 2 * k
            moment = float(np.mean(values**exponent))
            factorial = math.factorial(k)
            local_gaussian = factorial * (m2**k)
            root = (moment / factorial) ** (1.0 / exponent)
            row[f"moment_{exponent}"] = moment
            row[f"absolute_root_{exponent}"] = root
            row[f"absolute_C_{exponent}"] = root * root
            row[f"local_gaussian_ratio_{exponent}"] = moment / local_gaussian if local_gaussian > 0 else float("nan")

        for level in levels:
            row[f"tail_gt_{level:g}"] = float(np.mean(values > level))

        rows.append(row)
        print(
            f"Y={start} max={row['max_over_sqrt_Y']:.6g} "
            f"root20={row.get('absolute_root_20', 0.0):.6g} "
            f"tail>2={row.get('tail_gt_2', 0.0):.6g}"
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
    k_values = np.array(moment_orders, dtype=np.float64)
    exponents = [2 * k for k in moment_orders]

    max_roots = []
    root_worst_y = []
    max_local_ratios = []
    ratio_worst_y = []
    for exponent in exponents:
        roots = np.array([float(row[f"absolute_root_{exponent}"]) for row in rows])
        ratios = np.array([float(row[f"local_gaussian_ratio_{exponent}"]) for row in rows])
        root_index = int(np.argmax(roots))
        ratio_index = int(np.argmax(ratios))
        max_roots.append(float(roots[root_index]))
        root_worst_y.append(int(y_values[root_index]))
        max_local_ratios.append(float(ratios[ratio_index]))
        ratio_worst_y.append(int(y_values[ratio_index]))

    max_roots_array = np.array(max_roots)
    max_ratios_array = np.array(max_local_ratios)
    uniform_root = float(np.max(max_roots_array))
    uniform_c = uniform_root * uniform_root
    uniform_root_exponent = int(exponents[int(np.argmax(max_roots_array))])
    uniform_root_y = int(root_worst_y[int(np.argmax(max_roots_array))])

    plot_levels = np.linspace(
        0.75,
        max(2.6, max(float(np.max(values)) for values in block_values) + 0.2),
        220,
    )
    empirical_uniform_tail = []
    blockwise_best_markov = []
    uniform_c_markov = []
    empirical_worst_y = []
    markov_worst_y = []

    for level in plot_levels:
        tails = np.array([np.mean(values > level) for values in block_values])
        empirical_index = int(np.argmax(tails))
        empirical_uniform_tail.append(float(tails[empirical_index]))
        empirical_worst_y.append(int(y_values[empirical_index]))

        markovs = []
        for row in rows:
            moments = {exponent: float(row[f"moment_{exponent}"]) for exponent in exponents}
            markovs.append(best_markov(float(level), moments))
        markovs_array = np.array(markovs)
        markov_index = int(np.argmax(markovs_array))
        blockwise_best_markov.append(float(markovs_array[markov_index]))
        markov_worst_y.append(int(y_values[markov_index]))
        uniform_c_markov.append(factorial_markov(float(level), uniform_c, moment_orders))

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(3, 1, figsize=(13, 11))

    axes[0].plot(exponents, max_roots_array, marker="o", color="#2454a6", linewidth=1.2)
    axes[0].axhline(uniform_root, color="#b24a3a", linewidth=1.0, alpha=0.85, label=f"uniform root={uniform_root:.4f}")
    axes[0].set_xticks(exponents)
    axes[0].set_ylabel("max root over Y")
    axes[0].set_title(r"Uniform absolute moment roots: max_Y $(E|g_Y|^{2k}/k!)^{1/(2k)}$")
    axes[0].legend(loc="best")

    axes[1].plot(exponents, max_ratios_array, marker="o", color="#5b3f9b", linewidth=1.2)
    axes[1].axhline(1.0, color="#111111", alpha=0.5, linewidth=0.9)
    axes[1].set_xticks(exponents)
    axes[1].set_ylabel("max local ratio over Y")
    axes[1].set_title(r"Uniform local-Gaussian stress: max_Y E|g_Y|^{2k} / (k! m_2(Y)^k)")

    floor = 1.0 / (len(t_values) + 1)
    axes[2].semilogy(plot_levels, np.maximum(empirical_uniform_tail, floor), color="#2454a6", linewidth=1.3, label="max empirical tail over Y")
    axes[2].semilogy(plot_levels, np.maximum(blockwise_best_markov, floor), color="#24734d", linewidth=1.1, label="max blockwise Markov")
    axes[2].semilogy(plot_levels, np.maximum(uniform_c_markov, floor), color="#b24a3a", linewidth=1.1, label="uniform C factorial Markov")
    for level in levels:
        axes[2].axvline(level, color="#777777", alpha=0.22, linewidth=0.8)
    axes[2].set_xlabel(r"level $\lambda$")
    axes[2].set_ylabel("fraction above level")
    axes[2].set_title(r"Uniform-in-Y large-values bounds")
    axes[2].legend(loc="best")

    fig.tight_layout()
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=160)

    print(args.output_png)
    if args.output_csv:
        print(args.output_csv)
    print(f"limit={limit}")
    print(f"t_grid={args.t_min}..{args.t_max} points={len(t_values)}")
    print(f"dyadic_blocks={len(rows)}")
    print(f"uniform_root={uniform_root:.6g}")
    print(f"uniform_C={uniform_c:.6g}")
    print(f"uniform_root_exponent={uniform_root_exponent}")
    print(f"uniform_root_Y={uniform_root_y}")

    for exponent, root, y_at_root, ratio, y_at_ratio in zip(
        exponents,
        max_roots,
        root_worst_y,
        max_local_ratios,
        ratio_worst_y,
        strict=True,
    ):
        print(
            f"exponent={exponent} max_root={root:.6g} root_Y={y_at_root} "
            f"max_local_ratio={ratio:.6g} ratio_Y={y_at_ratio}"
        )

    for level in levels:
        level = float(level)
        tails = np.array([np.mean(values > level) for values in block_values])
        empirical_index = int(np.argmax(tails))
        markovs = []
        for row in rows:
            moments = {exponent: float(row[f"moment_{exponent}"]) for exponent in exponents}
            markovs.append(best_markov(level, moments))
        markovs_array = np.array(markovs)
        markov_index = int(np.argmax(markovs_array))
        print(
            f"level={level:g} empirical_max={float(tails[empirical_index]):.6g} "
            f"empirical_Y={int(y_values[empirical_index])} "
            f"blockwise_markov={float(markovs_array[markov_index]):.6g} "
            f"markov_Y={int(y_values[markov_index])} "
            f"uniform_C_markov={factorial_markov(level, uniform_c, moment_orders):.6g}"
        )


if __name__ == "__main__":
    main()

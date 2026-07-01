from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np


def parse_sigmas(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def first_zero_gammas(t_max: float) -> list[float]:
    gammas: list[float] = []
    mp.mp.dps = 30
    n = 1
    while True:
        gamma = float(mp.im(mp.zetazero(n)))
        if gamma > t_max:
            break
        gammas.append(gamma)
        n += 1
    return gammas


def compute_dirichlet(
    mu_bin: Path,
    limit: int,
    sigmas: list[float],
    t_values: np.ndarray,
    chunk_size: int,
) -> dict[float, np.ndarray]:
    raw = np.fromfile(mu_bin, dtype=np.int8)
    if raw.size < 2:
        raise SystemExit("mu binary must contain mu[0..N]")

    full_limit = raw.size - 1
    n_limit = full_limit if limit <= 0 else min(limit, full_limit)
    mu = raw[1 : n_limit + 1]
    nonzero = np.nonzero(mu)[0] + 1
    signs = mu[nonzero - 1].astype(np.float64)
    logs = np.log(nonzero.astype(np.float64))

    result = {sigma: np.zeros(t_values.shape, dtype=np.complex128) for sigma in sigmas}

    for start in range(0, nonzero.size, chunk_size):
        stop = min(start + chunk_size, nonzero.size)
        chunk_logs = logs[start:stop]
        phase = np.exp(-1j * np.outer(chunk_logs, t_values))
        for sigma in sigmas:
            weights = signs[start:stop] * np.exp(-sigma * chunk_logs)
            result[sigma] += weights @ phase

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mu_bin", type=Path)
    parser.add_argument("output_png", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--limit", type=int, default=500_000)
    parser.add_argument("--t-min", type=float, default=0.0)
    parser.add_argument("--t-max", type=float, default=80.0)
    parser.add_argument("--t-points", type=int, default=1200)
    parser.add_argument("--sigmas", default="0.5,0.75,1.1")
    parser.add_argument("--chunk-size", type=int, default=4096)
    args = parser.parse_args()

    sigmas = parse_sigmas(args.sigmas)
    t_values = np.linspace(args.t_min, args.t_max, args.t_points)
    values = compute_dirichlet(args.mu_bin, args.limit, sigmas, t_values, args.chunk_size)
    gammas = first_zero_gammas(args.t_max)

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            header = ["t"]
            for sigma in sigmas:
                header.extend([f"abs_sigma_{sigma}", f"real_sigma_{sigma}", f"imag_sigma_{sigma}"])
            writer.writerow(header)
            for i, t in enumerate(t_values):
                row: list[float] = [float(t)]
                for sigma in sigmas:
                    z = values[sigma][i]
                    row.extend([float(abs(z)), float(z.real), float(z.imag)])
                writer.writerow(row)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(len(sigmas), 1, figsize=(13, 3.2 * len(sigmas)), sharex=True)
    if len(sigmas) == 1:
        axes = [axes]

    for axis, sigma in zip(axes, sigmas):
        abs_values = np.abs(values[sigma])
        axis.plot(t_values, abs_values, color="#2454a6", linewidth=1.0)
        for gamma in gammas:
            axis.axvline(gamma, color="#b24a3a", alpha=0.28, linewidth=0.8)
        axis.set_ylabel(f"|D_N({sigma}+it)|")
        axis.set_title(f"sigma={sigma}, max={abs_values.max():.3f}, median={np.median(abs_values):.3f}")

    axes[-1].set_xlabel("t")
    fig.suptitle(f"Mobius Dirichlet sums, N={args.limit:,}; red lines are zeta zero ordinates", y=0.995)
    fig.tight_layout()
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=160)

    print(args.output_png)
    if args.output_csv:
        print(args.output_csv)
    print(f"N={args.limit}")
    print(f"t_range={args.t_min}..{args.t_max}")
    print(f"t_points={args.t_points}")
    print(f"zero_markers={len(gammas)}")
    for sigma in sigmas:
        abs_values = np.abs(values[sigma])
        max_index = int(np.argmax(abs_values))
        print(
            f"sigma={sigma}: max={abs_values[max_index]:.6f} "
            f"at_t={t_values[max_index]:.6f} median={np.median(abs_values):.6f} "
            f"p95={np.quantile(abs_values, 0.95):.6f}"
        )


if __name__ == "__main__":
    main()

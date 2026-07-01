from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np


def parse_sigmas(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def zero_gammas(count: int) -> list[float]:
    mp.mp.dps = 40
    return [float(mp.im(mp.zetazero(n))) for n in range(1, count + 1)]


def make_checkpoints(limit: int, count: int, first: int) -> np.ndarray:
    logs = np.linspace(np.log(first), np.log(limit), count)
    points = np.unique(np.rint(np.exp(logs)).astype(np.int64))
    points = points[(points >= 2) & (points <= limit)]
    if points[-1] != limit:
        points = np.append(points, limit)
    return points


def compute_track(
    primes: np.ndarray,
    logs: np.ndarray,
    checkpoints: np.ndarray,
    sigma: float,
    t: float,
) -> np.ndarray:
    checkpoint_indices = np.searchsorted(primes, checkpoints, side="right") - 1
    terms = np.exp((-sigma - 1j * t) * logs)
    cumsum = np.cumsum(terms, dtype=np.complex128)
    values = np.zeros(checkpoints.shape, dtype=np.complex128)
    valid = checkpoint_indices >= 0
    values[valid] = cumsum[checkpoint_indices[valid]]
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("primes_bin", type=Path)
    parser.add_argument("output_png", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--limit", type=int, default=10_000_000)
    parser.add_argument("--sigmas", default="0.5,0.6,0.75,1.1")
    parser.add_argument("--zeros", type=int, default=6)
    parser.add_argument("--checkpoints", type=int, default=240)
    parser.add_argument("--first", type=int, default=100)
    args = parser.parse_args()

    all_primes = np.fromfile(args.primes_bin, dtype=np.uint32).astype(np.int64)
    if all_primes.size == 0:
        raise SystemExit("prime binary is empty")

    limit = min(args.limit, int(all_primes[-1]))
    primes = all_primes[all_primes <= limit]
    logs = np.log(primes.astype(np.float64))
    checkpoints = make_checkpoints(limit, args.checkpoints, args.first)

    sigmas = parse_sigmas(args.sigmas)
    gammas = zero_gammas(args.zeros)
    controls = [
        0.0,
        (gammas[0] + gammas[1]) / 2.0,
        (gammas[1] + gammas[2]) / 2.0 if len(gammas) >= 3 else gammas[0] + 4.0,
    ]
    targets = [(f"zero {i + 1}", gamma, True) for i, gamma in enumerate(gammas)]
    targets += [(f"control {i + 1}", t, False) for i, t in enumerate(controls)]

    tracks: dict[tuple[float, str], np.ndarray] = {}
    summary_rows: list[dict[str, float | int | str]] = []
    for sigma in sigmas:
        for label, t, is_zero in targets:
            values = compute_track(primes, logs, checkpoints, sigma, t)
            tracks[(sigma, label)] = values
            abs_values = np.abs(values)
            decade_index = np.searchsorted(checkpoints, max(2, limit // 10))
            summary_rows.append({
                "sigma": sigma,
                "label": label,
                "is_zero": int(is_zero),
                "t": t,
                "final_abs": float(abs_values[-1]),
                "max_abs": float(abs_values.max()),
                "last_over_prev_decade": float(abs_values[-1] / max(abs_values[decade_index], 1e-300)),
            })

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["N", "sigma", "label", "is_zero", "t", "abs_P_N", "real_P_N", "imag_P_N"])
            for sigma in sigmas:
                for label, t, is_zero in targets:
                    values = tracks[(sigma, label)]
                    for n, z in zip(checkpoints, values):
                        writer.writerow([
                            int(n),
                            sigma,
                            label,
                            int(is_zero),
                            t,
                            float(abs(z)),
                            float(z.real),
                            float(z.imag),
                        ])

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(len(sigmas), 1, figsize=(13, 3.2 * len(sigmas)), sharex=True)
    if len(sigmas) == 1:
        axes = [axes]

    zero_colors = ["#2454a6", "#386cb0", "#4f8ac9", "#6baed6", "#8cbfe2", "#aacfe9"]
    control_colors = ["#111111", "#b24a3a", "#24734d"]

    for axis, sigma in zip(axes, sigmas):
        for idx, (label, t, is_zero) in enumerate(targets):
            values = np.abs(tracks[(sigma, label)])
            if is_zero:
                axis.plot(checkpoints, values, color=zero_colors[idx % len(zero_colors)], linewidth=1.0, alpha=0.9, label=f"{label}: t={t:.3f}")
            else:
                cidx = idx - len(gammas)
                axis.plot(checkpoints, values, color=control_colors[cidx % len(control_colors)], linewidth=1.2, linestyle="--", label=f"{label}: t={t:.3f}")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_ylabel(f"|P_N({sigma}+it)|")
        axis.set_title(f"sigma={sigma}")
        axis.legend(loc="best", ncol=3, fontsize=8)

    axes[-1].set_xlabel("N")
    fig.suptitle("Growth of prime Dirichlet partial sums at zeta-zero ordinates", y=0.997)
    fig.tight_layout()
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=160)

    print(args.output_png)
    if args.output_csv:
        print(args.output_csv)
    print(f"limit={limit}")
    print(f"prime_count={primes.size}")
    print("summary:")
    for row in summary_rows:
        print(
            f"sigma={row['sigma']} {row['label']} t={row['t']:.12f} "
            f"final={row['final_abs']:.6g} max={row['max_abs']:.6g} "
            f"last/prev_decade={row['last_over_prev_decade']:.6g}"
        )


if __name__ == "__main__":
    main()

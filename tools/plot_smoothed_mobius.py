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


def make_x_values(x_min: float, x_max: float, count: int) -> np.ndarray:
    values = np.unique(np.rint(np.exp(np.linspace(np.log(x_min), np.log(x_max), count))).astype(np.int64))
    values = values[values >= 1]
    return values.astype(np.float64)


def compute_smoothed(
    n_values: np.ndarray,
    signs: np.ndarray,
    logs: np.ndarray,
    x_values: np.ndarray,
    sigma: float,
    t: float,
    chunk_size: int,
) -> np.ndarray:
    result = np.zeros(x_values.shape, dtype=np.complex128)
    inv_x = 1.0 / x_values

    for start in range(0, n_values.size, chunk_size):
        stop = min(start + chunk_size, n_values.size)
        n_chunk = n_values[start:stop].astype(np.float64)
        base = signs[start:stop] * np.exp((-sigma - 1j * t) * logs[start:stop])
        smooth = np.exp(-np.outer(n_chunk, inv_x))
        result += base @ smooth

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mu_bin", type=Path)
    parser.add_argument("output_png", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--limit", type=int, default=1_000_000)
    parser.add_argument("--x-min", type=float, default=50.0)
    parser.add_argument("--x-max", type=float, default=200_000.0)
    parser.add_argument("--x-points", type=int, default=90)
    parser.add_argument("--sigmas", default="0.5,0.75,1.1")
    parser.add_argument("--zeros", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=8192)
    args = parser.parse_args()

    raw = np.fromfile(args.mu_bin, dtype=np.int8)
    if raw.size < 2:
        raise SystemExit("mu binary must contain mu[0..N]")

    full_limit = raw.size - 1
    limit = min(args.limit, full_limit)
    mu = raw[1 : limit + 1]
    nonzero = np.nonzero(mu)[0].astype(np.int64) + 1
    signs = mu[nonzero - 1].astype(np.float64)
    logs = np.log(nonzero.astype(np.float64))

    x_values = make_x_values(args.x_min, args.x_max, args.x_points)
    sigmas = parse_sigmas(args.sigmas)
    gammas = zero_gammas(args.zeros)
    controls = [
        0.0,
        (gammas[0] + gammas[1]) / 2.0,
    ]
    targets = [(f"zero {i + 1}", gamma, True) for i, gamma in enumerate(gammas)]
    targets += [(f"control {i + 1}", t, False) for i, t in enumerate(controls)]

    tracks: dict[tuple[float, str], np.ndarray] = {}
    summary_rows: list[dict[str, float | int | str]] = []

    for sigma in sigmas:
        for label, t, is_zero in targets:
            values = compute_smoothed(nonzero, signs, logs, x_values, sigma, t, args.chunk_size)
            tracks[(sigma, label)] = values
            abs_values = np.abs(values)
            prev_index = np.searchsorted(x_values, args.x_max / 10.0)
            summary_rows.append({
                "sigma": sigma,
                "label": label,
                "is_zero": int(is_zero),
                "t": t,
                "final_abs": float(abs_values[-1]),
                "max_abs": float(abs_values.max()),
                "last_over_prev_decade": float(abs_values[-1] / max(abs_values[prev_index], 1e-300)),
            })

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["X", "sigma", "label", "is_zero", "t", "abs_F_X", "real_F_X", "imag_F_X"])
            for sigma in sigmas:
                for label, t, is_zero in targets:
                    values = tracks[(sigma, label)]
                    for x, z in zip(x_values, values):
                        writer.writerow([
                            float(x),
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

    zero_colors = ["#2454a6", "#386cb0", "#4f8ac9", "#6baed6", "#8cbfe2"]
    control_colors = ["#111111", "#b24a3a"]

    for axis, sigma in zip(axes, sigmas):
        for idx, (label, t, is_zero) in enumerate(targets):
            values = np.abs(tracks[(sigma, label)])
            if is_zero:
                axis.plot(x_values, values, color=zero_colors[idx % len(zero_colors)], linewidth=1.1, alpha=0.9, label=f"{label}: t={t:.3f}")
            else:
                cidx = idx - len(gammas)
                axis.plot(x_values, values, color=control_colors[cidx % len(control_colors)], linewidth=1.2, linestyle="--", label=f"{label}: t={t:.3f}")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_ylabel(f"|F_X({sigma}+it)|")
        axis.set_title(f"sigma={sigma}")
        axis.legend(loc="best", ncol=3, fontsize=8)

    axes[-1].set_xlabel("smoothing X")
    fig.suptitle("Smoothed Mobius sums: sum mu(n)n^{-s} exp(-n/X)", y=0.997)
    fig.tight_layout()
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=160)

    print(args.output_png)
    if args.output_csv:
        print(args.output_csv)
    print(f"limit={limit}")
    print(f"squarefree_terms={nonzero.size}")
    print(f"x_range={x_values[0]}..{x_values[-1]}")
    print("summary:")
    for row in summary_rows:
        print(
            f"sigma={row['sigma']} {row['label']} t={row['t']:.12f} "
            f"final={row['final_abs']:.6g} max={row['max_abs']:.6g} "
            f"last/prev_decade={row['last_over_prev_decade']:.6g}"
        )


if __name__ == "__main__":
    main()

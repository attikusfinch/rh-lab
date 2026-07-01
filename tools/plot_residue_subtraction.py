from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np


def parse_sigmas(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def parse_counts(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def load_rows(path: Path, label: str, sigmas: list[float]) -> dict[float, dict[str, np.ndarray | float]]:
    raw_rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    result: dict[float, dict[str, np.ndarray | float]] = {}

    for sigma in sigmas:
        matches = [
            row for row in raw_rows
            if row["label"] == label and abs(float(row["sigma"]) - sigma) < 1e-12
        ]
        if not matches:
            raise SystemExit(f"No rows for label={label!r}, sigma={sigma}")

        x = np.array([float(row["X"]) for row in matches], dtype=np.float64)
        error = np.array([
            complex(float(row["real_error"]), float(row["imag_error"]))
            for row in matches
        ], dtype=np.complex128)
        t = float(matches[0]["t"])
        result[sigma] = {"x": x, "error": error, "t": t}

    return result


def zeta_zero_data(count: int) -> list[tuple[complex, complex]]:
    mp.mp.dps = 60
    zeros: list[tuple[complex, complex]] = []
    for n in range(1, count + 1):
        rho_pos = mp.zetazero(n)
        zeta_prime_pos = mp.diff(lambda z: mp.zeta(z), rho_pos)
        rho_neg = mp.conj(rho_pos)
        zeta_prime_neg = mp.conj(zeta_prime_pos)
        zeros.append((complex(rho_pos), complex(zeta_prime_pos)))
        zeros.append((complex(rho_neg), complex(zeta_prime_neg)))
    return zeros


def residue_sum(s: complex, x: np.ndarray, zero_data: list[tuple[complex, complex]], count: int) -> np.ndarray:
    selected = zero_data[: 2 * count]
    log_x = np.log(x)
    total = np.zeros(x.shape, dtype=np.complex128)
    for rho, zeta_prime in selected:
        exponent = rho - s
        coeff = complex(mp.gamma(exponent)) / zeta_prime
        total += coeff * np.exp(exponent * log_x)
    return total


def fit_slope(x: np.ndarray, y: np.ndarray, tail_fraction: float) -> float:
    start = max(0, int(len(x) * (1.0 - tail_fraction)))
    x_tail = x[start:]
    y_tail = y[start:]
    mask = y_tail > 1e-16
    if np.count_nonzero(mask) < 3:
        return float("nan")
    return float(np.polyfit(np.log(x_tail[mask]), np.log(y_tail[mask]), 1)[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("convergence_csv", type=Path)
    parser.add_argument("output_png", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--sigmas", default="0.55,0.6,0.75,1.1")
    parser.add_argument("--label", default="zero 1")
    parser.add_argument("--zero-counts", default="0,1,2,3")
    parser.add_argument("--tail-fraction", type=float, default=0.45)
    args = parser.parse_args()

    sigmas = parse_sigmas(args.sigmas)
    counts = parse_counts(args.zero_counts)
    if not counts or counts[0] != 0:
        counts = [0] + [count for count in counts if count != 0]
    max_count = max(counts)

    rows = load_rows(args.convergence_csv, args.label, sigmas)
    zero_data = zeta_zero_data(max_count)

    residuals: dict[tuple[float, int], np.ndarray] = {}
    summaries: list[dict[str, float | int]] = []

    for sigma in sigmas:
        x = rows[sigma]["x"]  # type: ignore[assignment]
        error = rows[sigma]["error"]  # type: ignore[assignment]
        t = rows[sigma]["t"]  # type: ignore[assignment]
        s = complex(sigma, float(t))

        for count in counts:
            if count == 0:
                corrected = error
            else:
                corrected = error - residue_sum(s, x, zero_data, count)
            abs_corrected = np.abs(corrected)
            residuals[(sigma, count)] = abs_corrected
            summaries.append({
                "sigma": sigma,
                "zero_count": count,
                "final_abs": float(abs_corrected[-1]),
                "slope": fit_slope(x, abs_corrected, args.tail_fraction),
            })

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["X", "sigma", "label", "subtracted_zero_pairs", "abs_residual"])
            for sigma in sigmas:
                x = rows[sigma]["x"]  # type: ignore[assignment]
                for count in counts:
                    for x_value, y_value in zip(x, residuals[(sigma, count)]):
                        writer.writerow([float(x_value), sigma, args.label, count, float(y_value)])

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(len(sigmas), 1, figsize=(13, 3.1 * len(sigmas)), sharex=True)
    if len(sigmas) == 1:
        axes = [axes]

    colors = ["#2454a6", "#b24a3a", "#24734d", "#5b3f9b", "#111111"]
    for axis, sigma in zip(axes, sigmas):
        x = rows[sigma]["x"]  # type: ignore[assignment]
        for idx, count in enumerate(counts):
            label = "raw" if count == 0 else f"minus {count} zero pair"
            axis.plot(
                x,
                residuals[(sigma, count)],
                color=colors[idx % len(colors)],
                linewidth=1.1,
                label=label,
            )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_ylabel(f"sigma={sigma}")
        axis.legend(loc="best", ncol=min(len(counts), 4), fontsize=8)

    axes[-1].set_xlabel("smoothing X")
    fig.suptitle(f"Residue subtraction for {args.label}: |F_X(s)-1/zeta(s)-residues|", y=0.997)
    fig.tight_layout()
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=160)

    print(args.output_png)
    if args.output_csv:
        print(args.output_csv)
    print(f"label={args.label}")
    print("summary:")
    for row in summaries:
        print(
            f"sigma={row['sigma']} zero_pairs={row['zero_count']} "
            f"final={row['final_abs']:.6g} slope={row['slope']:.4g}"
        )


if __name__ == "__main__":
    main()

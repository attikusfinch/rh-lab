from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np


def read_sample(path: Path, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    xs: list[float] = []
    errors: list[float] = []
    with path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))

    if len(rows) <= max_points:
        selected = rows
    else:
        indices = np.linspace(0, len(rows) - 1, max_points).round().astype(int)
        selected = [rows[int(index)] for index in indices]

    for row in selected:
        x = float(row["x"])
        if x > 2:
            xs.append(x)
            errors.append(float(row["psi_minus_x"]))

    return np.array(xs, dtype=np.float64), np.array(errors, dtype=np.float64)


def zero_wave(xs: np.ndarray, zero_count: int) -> tuple[np.ndarray, list[float]]:
    log_x = np.log(xs)
    sqrt_x = np.sqrt(xs)
    wave = np.zeros_like(xs)
    gammas: list[float] = []

    mp.mp.dps = 30
    for n in range(1, zero_count + 1):
        rho = mp.zetazero(n)
        gamma = float(mp.im(rho))
        gammas.append(gamma)
        denominator = 0.5 + 1j * gamma
        wave += -2.0 * np.real(sqrt_x * np.exp(1j * gamma * log_x) / denominator)

    wave += -float(mp.log(2 * mp.pi))
    return wave, gammas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("output_png", type=Path)
    parser.add_argument("--zeros", type=int, default=80)
    parser.add_argument("--points", type=int, default=3000)
    args = parser.parse_args()

    xs, actual = read_sample(args.csv_path, args.points)
    wave, gammas = zero_wave(xs, args.zeros)

    centered_actual = actual - actual.mean()
    centered_wave = wave - wave.mean()
    corr = float(np.corrcoef(centered_actual, centered_wave)[0, 1])

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    axes[0].plot(xs, actual, color="#2454a6", linewidth=0.85, label="actual psi(x)-x")
    axes[0].plot(xs, wave, color="#b24a3a", linewidth=1.1, alpha=0.9, label=f"first {args.zeros} zero waves")
    axes[0].axhline(0, color="#111111", linewidth=0.8)
    axes[0].set_ylabel("error")
    axes[0].set_title(f"Explicit-formula zero waves, corr={corr:.3f}")
    axes[0].legend(loc="best")

    axes[1].plot(xs, actual - wave, color="#24734d", linewidth=0.85)
    axes[1].axhline(0, color="#111111", linewidth=0.8)
    axes[1].set_ylabel("actual - wave")
    axes[1].set_xlabel("x")
    axes[1].ticklabel_format(style="plain", axis="x")

    fig.tight_layout()
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=160)
    print(args.output_png)
    print(f"first_gamma={gammas[0]:.12f}")
    print(f"last_gamma={gammas[-1]:.12f}")
    print(f"correlation={corr:.6f}")


if __name__ == "__main__":
    main()

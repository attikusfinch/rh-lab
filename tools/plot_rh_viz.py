from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read_columns(path: Path) -> dict[str, list[float]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        columns: dict[str, list[float]] = {name: [] for name in reader.fieldnames or []}
        for row in reader:
            for key, value in row.items():
                columns[key].append(float(value))
    return columns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("output_png", type=Path)
    args = parser.parse_args()

    data = read_columns(args.csv_path)
    x = data["x"]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)

    axes[0].plot(x, data["psi_minus_x"], color="#2454a6", linewidth=0.9)
    axes[0].axhline(0, color="#111111", linewidth=0.8)
    axes[0].set_ylabel("psi(x) - x")
    axes[0].set_title("Chebyshev psi error and RH-scale normalizations")

    axes[1].plot(x, data["norm_sqrt_log"], color="#b24a3a", linewidth=0.9)
    axes[1].axhline(0, color="#111111", linewidth=0.8)
    axes[1].set_ylabel("(psi(x)-x)/(sqrt(x) log x)")

    axes[2].plot(x, data["norm_sqrt_log2"], color="#24734d", linewidth=0.9)
    axes[2].axhline(0, color="#111111", linewidth=0.8)
    axes[2].set_ylabel("(psi(x)-x)/(sqrt(x) log^2 x)")
    axes[2].set_xlabel("x")

    for axis in axes:
        axis.ticklabel_format(style="plain", axis="x")

    fig.tight_layout()
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=160)
    print(args.output_png)


if __name__ == "__main__":
    main()

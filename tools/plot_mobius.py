from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_columns(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        raw: dict[str, list[float]] = {name: [] for name in reader.fieldnames or []}
        for row in reader:
            for key, value in row.items():
                raw[key].append(float(value))
    return {key: np.array(values, dtype=np.float64) for key, values in raw.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("output_png", type=Path)
    args = parser.parse_args()

    data = read_columns(args.csv_path)
    x = data["x"]
    m = data["M_x"]
    sqrt_x = np.sqrt(x)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=False)

    axes[0].plot(x, m, color="#2454a6", linewidth=0.9, label="M(x)")
    axes[0].plot(x, sqrt_x, color="#555555", linewidth=0.8, alpha=0.55, label="+sqrt(x)")
    axes[0].plot(x, -sqrt_x, color="#555555", linewidth=0.8, alpha=0.55, label="-sqrt(x)")
    axes[0].axhline(0, color="#111111", linewidth=0.8)
    axes[0].set_ylabel("M(x)")
    axes[0].set_title("Mertens function and square-root scale")
    axes[0].legend(loc="best")
    axes[0].ticklabel_format(style="plain", axis="x")

    axes[1].plot(x, data["M_over_sqrt"], color="#b24a3a", linewidth=0.85)
    axes[1].axhline(0, color="#111111", linewidth=0.8)
    axes[1].set_ylabel("M(x) / sqrt(x)")
    axes[1].ticklabel_format(style="plain", axis="x")

    positive = x > 1
    axes[2].plot(x[positive], data["max_over_sqrt"][positive], color="#24734d", linewidth=1.1, label="max |M| / sqrt(x)")
    axes[2].plot(x[positive], data["empirical_exponent"][positive], color="#5b3f9b", linewidth=1.1, label="log(max |M|) / log(x)")
    axes[2].axhline(0.5, color="#111111", linewidth=0.8)
    axes[2].set_xscale("log")
    axes[2].set_ylabel("prefix envelope")
    axes[2].set_xlabel("x")
    axes[2].legend(loc="best")

    fig.tight_layout()
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=160)
    print(args.output_png)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np


def parse_t_values(value: str) -> list[tuple[str, float]]:
    result: list[tuple[str, float]] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            label, raw_t = part.split(":", 1)
            result.append((label.strip(), float(raw_t)))
        else:
            result.append((f"t={float(part):.3f}", float(part)))
    return result


def default_t_values(zero_count: int) -> list[tuple[str, float]]:
    mp.mp.dps = 40
    gammas = [float(mp.im(mp.zetazero(n))) for n in range(1, zero_count + 1)]
    values: list[tuple[str, float]] = [("t=0", 0.0)]
    for index, gamma in enumerate(gammas, start=1):
        values.append((f"zero {index}", gamma))
    if len(gammas) >= 2:
        values.append(("mid 1-2", (gammas[0] + gammas[1]) / 2.0))
    return values


def dyadic_edges(limit: int, first: int) -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    y = first
    while y < limit:
        stop = min(2 * y, limit + 1)
        if stop > y + 1:
            edges.append((y, stop))
        y *= 2
    return edges


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mu_bin", type=Path)
    parser.add_argument("output_png", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--limit", type=int, default=10_000_000)
    parser.add_argument("--first", type=int, default=128)
    parser.add_argument("--zeros", type=int, default=4)
    parser.add_argument("--t-values", default="")
    args = parser.parse_args()

    raw = np.fromfile(args.mu_bin, dtype=np.int8)
    if raw.size < 2:
        raise SystemExit("mu binary must contain mu[0..N]")

    full_limit = raw.size - 1
    limit = min(args.limit, full_limit)
    mu = raw[1 : limit + 1]
    n = np.arange(1, limit + 1, dtype=np.float64)
    log_n = np.log(n)
    edges = dyadic_edges(limit, args.first)

    t_values = parse_t_values(args.t_values) if args.t_values else default_t_values(args.zeros)

    rows: list[dict[str, float | int | str]] = []
    for label, t in t_values:
        twist = mu.astype(np.float64) * np.exp(-1j * t * log_n)
        cumsum = np.concatenate(([0.0 + 0.0j], np.cumsum(twist, dtype=np.complex128)))
        for start, stop in edges:
            block_sum = cumsum[stop - 1] - cumsum[start - 1]
            length = stop - start
            local_exponent = float("nan")
            if abs(block_sum) >= 1.0 and start > 1:
                local_exponent = float(np.log(abs(block_sum)) / np.log(start))

            rows.append({
                "label": label,
                "t": t,
                "Y": start,
                "stop": stop - 1,
                "length": length,
                "abs_sum": float(abs(block_sum)),
                "over_sqrt_Y": float(abs(block_sum) / np.sqrt(start)),
                "over_sqrt_length": float(abs(block_sum) / np.sqrt(length)),
                "local_exponent": local_exponent,
            })

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=[
                "label",
                "t",
                "Y",
                "stop",
                "length",
                "abs_sum",
                "over_sqrt_Y",
                "over_sqrt_length",
                "local_exponent",
            ])
            writer.writeheader()
            writer.writerows(rows)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)

    colors = ["#111111", "#2454a6", "#386cb0", "#4f8ac9", "#b24a3a", "#24734d"]
    for idx, (label, _t) in enumerate(t_values):
        selected = [row for row in rows if row["label"] == label]
        y = np.array([float(row["Y"]) for row in selected])
        abs_sum = np.array([float(row["abs_sum"]) for row in selected])
        over_sqrt = np.array([float(row["over_sqrt_Y"]) for row in selected])
        exponent = np.array([float(row["local_exponent"]) for row in selected])

        color = colors[idx % len(colors)]
        axes[0].plot(y, abs_sum, marker="o", linewidth=1.0, color=color, label=label)
        axes[1].plot(y, over_sqrt, marker="o", linewidth=1.0, color=color, label=label)
        finite = np.isfinite(exponent)
        axes[2].plot(y[finite], exponent[finite], marker="o", linewidth=1.0, color=color, label=label)

    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("|block sum|")
    axes[0].set_title("Dyadic twisted Mobius block sums")

    axes[1].set_xscale("log")
    axes[1].set_ylabel("|block sum| / sqrt(Y)")

    axes[2].set_xscale("log")
    axes[2].axhline(0.5, color="#777777", linewidth=0.9, linestyle="--")
    axes[2].set_ylim(0.0, 0.7)
    axes[2].set_ylabel("log |sum| / log Y")
    axes[2].set_xlabel("dyadic start Y")
    axes[2].legend(loc="best", ncol=3, fontsize=8)

    fig.tight_layout()
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=160)

    print(args.output_png)
    if args.output_csv:
        print(args.output_csv)
    print(f"limit={limit}")
    print(f"blocks={len(edges)}")
    for label, _t in t_values:
        selected = [row for row in rows if row["label"] == label]
        max_ratio = max(float(row["over_sqrt_Y"]) for row in selected)
        finite_exponents = [
            float(row["local_exponent"])
            for row in selected
            if np.isfinite(float(row["local_exponent"]))
        ]
        max_exp = max(finite_exponents) if finite_exponents else float("nan")
        print(f"{label}: max_over_sqrt_Y={max_ratio:.6g} max_local_exponent={max_exp:.6g}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_float_list(value: str) -> list[float]:
    values: list[float] = []
    for part in value.split(","):
        part = part.strip()
        if part:
            values.append(float(part))
    if not values:
        raise SystemExit("empty float list")
    return values


def fit_power_law(points: list[tuple[int, float]]) -> tuple[float, float, float]:
    if len(points) < 2:
        raise SystemExit("at least two points are required for a fit")
    x = np.log(np.array([point[0] for point in points], dtype=np.float64))
    y = np.log(np.array([point[1] for point in points], dtype=np.float64))
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    residual = y - fitted
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return -float(slope), float(math.exp(intercept)), r2


def envelope_constant(points: list[tuple[int, float]], alpha: float) -> tuple[float, int, float]:
    best_y = 0
    best_epsilon = 0.0
    best_c = -1.0
    for y, epsilon in points:
        c_value = epsilon * (float(y) ** alpha)
        if c_value > best_c:
            best_c = c_value
            best_y = y
            best_epsilon = epsilon
    return best_c, best_y, best_epsilon


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_png", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--tail-min-y", type=int, default=1024)
    parser.add_argument("--envelope-alphas", default="1,1.25,1.5,1.75,2,2.25")
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    with args.input_csv.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)
    if not rows:
        raise SystemExit("input CSV is empty")

    by_y: dict[int, dict[str, str]] = {}
    by_t_y: dict[tuple[float, int], dict[str, str]] = {}
    for row in rows:
        y = int(float(row["Y"]))
        t_scale = float(row["T"])
        ratio = float(row["ratio"])
        if not math.isfinite(ratio) or ratio <= 0:
            continue
        if y not in by_y or ratio > float(by_y[y]["ratio"]):
            by_y[y] = row
        key = (t_scale, y)
        if key not in by_t_y or ratio > float(by_t_y[key]["ratio"]):
            by_t_y[key] = row

    worst_rows = [by_y[y] for y in sorted(by_y)]
    points = [(int(float(row["Y"])), float(row["ratio"])) for row in worst_rows]
    tail_points = [(y, epsilon) for y, epsilon in points if y >= args.tail_min_y]
    if len(tail_points) < 2:
        raise SystemExit("--tail-min-y leaves fewer than two points")

    all_alpha, all_c, all_r2 = fit_power_law(points)
    tail_alpha, tail_c, tail_r2 = fit_power_law(tail_points)

    envelope_alphas = parse_float_list(args.envelope_alphas)
    envelope_rows = []
    for alpha in envelope_alphas:
        c_value, y, epsilon = envelope_constant(points, alpha)
        envelope_rows.append({"alpha": alpha, "C": c_value, "Y": y, "epsilon": epsilon})

    t_values = sorted({float(row["T"]) for row in rows})
    t_fit_rows = []
    for t_scale in t_values:
        t_points = [
            (y, float(by_t_y[(t_scale, y)]["ratio"]))
            for y in sorted(by_y)
            if (t_scale, y) in by_t_y
        ]
        alpha, c_value, r2 = fit_power_law(t_points)
        t_fit_rows.append({"T": t_scale, "alpha": alpha, "C": c_value, "r2": r2})

    local_slopes = []
    for left, right in zip(points[:-1], points[1:]):
        y0, epsilon0 = left
        y1, epsilon1 = right
        slope = -math.log(epsilon1 / epsilon0) / math.log(y1 / y0)
        local_slopes.append((y0, y1, slope))

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="", encoding="utf-8") as csv_file:
            fieldnames = ["kind", "Y", "T", "shell", "ratio", "alpha", "C", "r2", "from_Y", "to_Y"]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for row in worst_rows:
                writer.writerow(
                    {
                        "kind": "worst_by_Y",
                        "Y": int(float(row["Y"])),
                        "T": float(row["T"]),
                        "shell": row["shell"],
                        "ratio": float(row["ratio"]),
                    }
                )
            writer.writerow({"kind": "fit_all", "alpha": all_alpha, "C": all_c, "r2": all_r2})
            writer.writerow({"kind": "fit_tail", "alpha": tail_alpha, "C": tail_c, "r2": tail_r2})
            for row in envelope_rows:
                writer.writerow(
                    {
                        "kind": "envelope",
                        "Y": row["Y"],
                        "ratio": row["epsilon"],
                        "alpha": row["alpha"],
                        "C": row["C"],
                    }
                )
            for row in t_fit_rows:
                writer.writerow({"kind": "fit_T", "T": row["T"], "alpha": row["alpha"], "C": row["C"], "r2": row["r2"]})
            for y0, y1, slope in local_slopes:
                writer.writerow({"kind": "local_slope", "from_Y": y0, "to_Y": y1, "alpha": slope})

    y_values = np.array([point[0] for point in points], dtype=np.float64)
    epsilon_values = np.array([point[1] for point in points], dtype=np.float64)
    plot_y = np.geomspace(float(min(y_values)), float(max(y_values)), 256)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))

    axes[0].loglog(y_values, epsilon_values, marker="o", color="#111111", linewidth=1.2, label="observed worst")
    axes[0].loglog(plot_y, all_c * plot_y ** (-all_alpha), color="#2d66a8", linewidth=1.1, label=f"all fit alpha={all_alpha:.3f}")
    axes[0].loglog(
        plot_y,
        tail_c * plot_y ** (-tail_alpha),
        color="#a84f2d",
        linewidth=1.1,
        label=f"tail fit alpha={tail_alpha:.3f}",
    )
    for row in envelope_rows:
        if abs(float(row["alpha"]) - 2.0) < 1e-9:
            axes[0].loglog(
                plot_y,
                float(row["C"]) * plot_y ** (-float(row["alpha"])),
                color="#2f7d46",
                linestyle="--",
                linewidth=1.0,
                label=f"observed envelope C/Y^{row['alpha']:g}",
            )
    axes[0].set_xlabel("dyadic start Y")
    axes[0].set_ylabel("epsilon(Y)")
    axes[0].set_title("Worst local autocorrelation ratio and power-law fits")
    axes[0].legend(loc="best")

    axes[1].plot(
        [math.sqrt(y0 * y1) for y0, y1, _ in local_slopes],
        [slope for _, _, slope in local_slopes],
        marker="o",
        color="#633f8f",
        linewidth=1.1,
    )
    axes[1].axhline(2.0, color="#777777", linestyle="--", linewidth=0.9)
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("midpoint between consecutive Y values")
    axes[1].set_ylabel("local decay exponent")
    axes[1].set_title("Consecutive dyadic slopes")

    axes[2].plot(
        [float(row["alpha"]) for row in envelope_rows],
        [float(row["C"]) for row in envelope_rows],
        marker="o",
        color="#8a5b26",
        linewidth=1.1,
    )
    axes[2].set_yscale("log")
    axes[2].set_xlabel("chosen alpha in epsilon <= C Y^-alpha")
    axes[2].set_ylabel("smallest observed C")
    axes[2].set_title("Observed envelope constants")

    fig.tight_layout()
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=160)

    print(args.output_png)
    if args.output_csv:
        print(args.output_csv)
    print(f"fit_all_alpha={all_alpha:.6g} C={all_c:.6g} r2={all_r2:.6g}")
    print(f"fit_tail_min_y={args.tail_min_y} alpha={tail_alpha:.6g} C={tail_c:.6g} r2={tail_r2:.6g}")
    for row in envelope_rows:
        print(
            f"envelope_alpha={float(row['alpha']):g} C={float(row['C']):.6g} "
            f"at_Y={int(row['Y'])} epsilon={float(row['epsilon']):.6g}"
        )
    for row in t_fit_rows:
        print(f"T={float(row['T']):g} fit_alpha={float(row['alpha']):.6g} r2={float(row['r2']):.6g}")


if __name__ == "__main__":
    main()

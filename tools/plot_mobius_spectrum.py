from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def sample_indices(n: int, count: int) -> np.ndarray:
    if n <= count:
        return np.arange(n, dtype=np.int64)
    return np.unique(np.linspace(0, n - 1, count).round().astype(np.int64))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mu_bin", type=Path)
    parser.add_argument("output_png", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--limit", type=int, default=0, help="Use first N values after mu[0]; default uses whole file.")
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--plot-points", type=int, default=60000)
    args = parser.parse_args()

    raw = np.fromfile(args.mu_bin, dtype=np.int8)
    if raw.size < 2:
        raise SystemExit("mu binary must contain mu[0..N]")

    full_limit = raw.size - 1
    n = full_limit if args.limit <= 0 else min(args.limit, full_limit)
    a = raw[1 : n + 1].astype(np.float64)
    m_n = int(np.sum(a))
    squarefree_count = int(np.sum(a * a))

    spectrum = np.fft.rfft(a)
    magnitude = np.abs(spectrum)
    normalized = magnitude / np.sqrt(n)

    if magnitude.size > 1:
        candidates = np.arange(1, magnitude.size, dtype=np.int64)
        top_count = min(args.top, candidates.size)
        top_indices = candidates[np.argpartition(magnitude[1:], -top_count)[-top_count:]]
        top_indices = top_indices[np.argsort(magnitude[top_indices])[::-1]]
    else:
        top_indices = np.array([], dtype=np.int64)

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["rank", "k", "frequency", "period", "abs_S_k", "abs_S_k_over_sqrt_N"])
            for rank, k in enumerate(top_indices, start=1):
                frequency = k / n
                period = n / k
                writer.writerow([
                    rank,
                    int(k),
                    frequency,
                    period,
                    float(magnitude[k]),
                    float(normalized[k]),
                ])

    idx = sample_indices(magnitude.size, args.plot_points)
    idx_nonzero = idx[idx > 0]
    log_mags = np.log10(np.maximum(normalized[1:], 1e-12))

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(3, 1, figsize=(13, 10))

    axes[0].plot(idx_nonzero, normalized[idx_nonzero], color="#2454a6", linewidth=0.75)
    axes[0].axhline(np.sqrt(np.log(n)), color="#777777", linewidth=0.8, linestyle="--", label="sqrt(log N)")
    axes[0].set_ylabel("|S(k)| / sqrt(N)")
    axes[0].set_title(f"Additive Fourier spectrum of Mobius, N={n:,}, M(N)={m_n:,}")
    axes[0].legend(loc="best")

    axes[1].hist(log_mags, bins=120, color="#b24a3a", alpha=0.85)
    axes[1].set_ylabel("frequency count")
    axes[1].set_xlabel("log10(|S(k)| / sqrt(N)), k > 0")

    top_plot = np.arange(1, min(len(top_indices), 20) + 1)
    if len(top_plot):
        axes[2].bar(top_plot, normalized[top_indices[: len(top_plot)]], color="#24734d")
    axes[2].axhline(abs(m_n) / np.sqrt(n), color="#111111", linewidth=1.0, label="DC |M(N)|/sqrt(N)")
    axes[2].set_ylabel("top |S(k)| / sqrt(N)")
    axes[2].set_xlabel("rank among nonzero frequencies")
    axes[2].legend(loc="best")

    fig.tight_layout()
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=160)

    nonzero_norm = normalized[1:] if normalized.size > 1 else np.array([], dtype=np.float64)
    print(args.output_png)
    if args.output_csv:
        print(args.output_csv)
    print(f"N={n}")
    print(f"M_N={m_n}")
    print(f"squarefree_count={squarefree_count}")
    print(f"DC_abs_over_sqrt_N={abs(m_n) / np.sqrt(n):.6f}")
    if nonzero_norm.size:
        print(f"max_nonzero_abs_over_sqrt_N={float(np.max(nonzero_norm)):.6f}")
        print(f"mean_nonzero_abs_over_sqrt_N={float(np.mean(nonzero_norm)):.6f}")
        print(f"median_nonzero_abs_over_sqrt_N={float(np.median(nonzero_norm)):.6f}")
        print(f"p99_nonzero_abs_over_sqrt_N={float(np.quantile(nonzero_norm, 0.99)):.6f}")
        print(f"sqrt_log_N={np.sqrt(np.log(n)):.6f}")
        for rank, k in enumerate(top_indices[: min(args.top, 10)], start=1):
            print(
                f"top{rank}: k={int(k)} period={n / k:.6f} "
                f"abs_over_sqrt_N={float(normalized[k]):.6f}"
            )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def next_power_of_two(n: int) -> int:
    return 1 << (n - 1).bit_length()


def fft_autocorr_all(a: np.ndarray) -> np.ndarray:
    fft_len = next_power_of_two(2 * len(a) - 1)
    f = np.fft.rfft(a.astype(np.float64), n=fft_len)
    corr = np.fft.irfft(f * np.conj(f), n=fft_len)[: len(a)]
    return np.rint(corr).astype(np.int64)


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
    parser.add_argument("--points", type=int, default=20000)
    args = parser.parse_args()

    raw = np.fromfile(args.mu_bin, dtype=np.int8)
    if raw.size < 2:
        raise SystemExit("mu binary must contain mu[0..N]")

    full_limit = raw.size - 1
    n = full_limit if args.limit <= 0 else min(args.limit, full_limit)
    a = raw[1 : n + 1]
    m_n = int(np.sum(a, dtype=np.int64))

    corr = fft_autocorr_all(a)
    prefix = corr.astype(np.int64)
    prefix[1:] = corr[0] + 2 * np.cumsum(corr[1:], dtype=np.int64)
    prefix[0] = corr[0]

    final_expected = m_n * m_n
    final_actual = int(prefix[-1])
    if final_actual != final_expected:
        raise SystemExit(f"final prefix mismatch: got {final_actual}, expected {final_expected}")

    h = np.arange(n, dtype=np.int64)
    idx = sample_indices(n, args.points)

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["H", "Q_H", "Q_H_over_N", "sqrt_abs_Q_H_over_sqrt_N"])
            for i in idx:
                q = int(prefix[i])
                writer.writerow([
                    int(i),
                    q,
                    q / n,
                    np.sqrt(abs(q)) / np.sqrt(n),
                ])

    q_over_n = prefix[idx] / n
    sqrt_abs_q = np.sqrt(np.abs(prefix[idx])) / np.sqrt(n)
    cumsum_corr = np.zeros_like(prefix)
    cumsum_corr[1:] = 2 * np.cumsum(corr[1:], dtype=np.int64)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)

    axes[0].plot(h[idx], q_over_n, color="#2454a6", linewidth=1.0)
    axes[0].axhline(0, color="#111111", linewidth=0.8)
    axes[0].axhline(1, color="#777777", linewidth=0.8, linestyle="--")
    axes[0].axhline(-1, color="#777777", linewidth=0.8, linestyle="--")
    axes[0].set_ylabel("Q(H) / N")
    axes[0].set_title(f"Prefix sum of Mobius correlations, N={n:,}, M(N)^2={final_actual:,}")

    axes[1].plot(h[idx], sqrt_abs_q, color="#b24a3a", linewidth=1.0)
    axes[1].axhline(1, color="#111111", linewidth=0.8)
    axes[1].set_ylabel("sqrt(|Q(H)|) / sqrt(N)")

    axes[2].plot(h[idx], cumsum_corr[idx] / n, color="#24734d", linewidth=1.0)
    axes[2].axhline(0, color="#111111", linewidth=0.8)
    axes[2].axhline(-float(corr[0]) / n, color="#777777", linewidth=0.8, linestyle="--")
    axes[2].set_ylabel("2 sum C_h / N")
    axes[2].set_xlabel("max shift H")
    axes[2].ticklabel_format(style="plain", axis="x")

    fig.tight_layout()
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=160)

    print(args.output_png)
    if args.output_csv:
        print(args.output_csv)
    print(f"N={n}")
    print(f"M_N={m_n}")
    print(f"C_0_squarefree_count={int(corr[0])}")
    print(f"C_0_over_N={float(corr[0]) / n:.6f}")
    print(f"final_Q=M_N^2={final_actual}")
    print(f"final_Q_over_N={final_actual / n:.6f}")
    print(f"max_abs_Q_over_N={float(np.max(np.abs(prefix))) / n:.6f}")


if __name__ == "__main__":
    main()

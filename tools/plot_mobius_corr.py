from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def next_power_of_two(n: int) -> int:
    return 1 << (n - 1).bit_length()


def direct_autocorr(a: np.ndarray, max_lag: int) -> np.ndarray:
    return np.array([np.dot(a[: len(a) - h], a[h:]) for h in range(max_lag + 1)], dtype=np.int64)


def fft_autocorr(a: np.ndarray, max_lag: int) -> np.ndarray:
    fft_len = next_power_of_two(2 * len(a) - 1)
    f = np.fft.rfft(a.astype(np.float64), n=fft_len)
    corr = np.fft.irfft(f * np.conj(f), n=fft_len)[: max_lag + 1]
    return np.rint(corr).astype(np.int64)


def write_csv(path: Path, lag: np.ndarray, corr: np.ndarray, n: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["h", "C_h", "available_terms", "C_h_over_N_minus_h", "C_h_over_sqrt_N_minus_h"])
        for h, c in zip(lag, corr):
            terms = n - int(h)
            writer.writerow([
                int(h),
                int(c),
                terms,
                c / terms,
                c / np.sqrt(terms),
            ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mu_bin", type=Path)
    parser.add_argument("output_png", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--limit", type=int, default=0, help="Use first N values after mu[0]; default uses whole file.")
    parser.add_argument("--max-lag", type=int, default=50000)
    parser.add_argument("--check-lag", type=int, default=64)
    args = parser.parse_args()

    raw = np.fromfile(args.mu_bin, dtype=np.int8)
    if raw.size < 2:
        raise SystemExit("mu binary must contain mu[0..N]")

    full_limit = raw.size - 1
    n = full_limit if args.limit <= 0 else min(args.limit, full_limit)
    if args.max_lag >= n:
        raise SystemExit("--max-lag must be smaller than N")

    a = raw[1 : n + 1]
    corr = fft_autocorr(a, args.max_lag)

    if args.check_lag > 0:
        check_lag = min(args.check_lag, args.max_lag)
        direct = direct_autocorr(a[: min(n, 200_000)].astype(np.int64), check_lag)
        fft_small = fft_autocorr(a[: min(n, 200_000)], check_lag)
        if not np.array_equal(direct, fft_small):
            raise SystemExit("FFT autocorrelation failed small direct check")

    lag = np.arange(args.max_lag + 1, dtype=np.int64)
    terms = n - lag
    normalized = corr / np.sqrt(terms)
    density = corr / terms

    if args.output_csv:
        write_csv(args.output_csv, lag, corr, n)

    positive_lag = lag[1:]
    positive_corr = corr[1:]
    abs_corr = np.abs(positive_corr)
    max_index = int(np.argmax(abs_corr))
    max_h = int(positive_lag[max_index])
    max_c = int(positive_corr[max_index])
    max_norm = float(max_c / np.sqrt(n - max_h))
    rms_norm = float(np.sqrt(np.mean(normalized[1:] ** 2)))

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)

    axes[0].plot(positive_lag, positive_corr, color="#2454a6", linewidth=0.8)
    axes[0].axhline(0, color="#111111", linewidth=0.8)
    axes[0].set_ylabel("C_h")
    axes[0].set_title(f"Mobius autocorrelation, N={n:,}, max |C_h| at h={max_h:,}")

    axes[1].plot(positive_lag, normalized[1:], color="#b24a3a", linewidth=0.8)
    axes[1].axhline(0, color="#111111", linewidth=0.8)
    axes[1].set_ylabel("C_h / sqrt(N-h)")

    axes[2].plot(positive_lag, density[1:], color="#24734d", linewidth=0.8)
    axes[2].axhline(0, color="#111111", linewidth=0.8)
    axes[2].set_ylabel("C_h / (N-h)")
    axes[2].set_xlabel("shift h")

    fig.tight_layout()
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=160)

    print(args.output_png)
    if args.output_csv:
        print(args.output_csv)
    print(f"N={n}")
    print(f"max_lag={args.max_lag}")
    print(f"max_abs_nonzero_lag={abs(max_c)}")
    print(f"h_at_max_abs={max_h}")
    print(f"C_h_at_max_abs={max_c}")
    print(f"C_h_over_sqrt_terms_at_max={max_norm:.6f}")
    print(f"rms_C_h_over_sqrt_terms={rms_norm:.6f}")


if __name__ == "__main__":
    main()

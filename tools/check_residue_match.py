from __future__ import annotations

import argparse
import csv
from pathlib import Path

import mpmath as mp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("convergence_csv", type=Path)
    parser.add_argument("--zero-index", type=int, default=1)
    parser.add_argument("--sigmas", default="0.55,0.6,0.75,1.1")
    args = parser.parse_args()

    sigmas = [float(part.strip()) for part in args.sigmas.split(",") if part.strip()]

    mp.mp.dps = 50
    rho = mp.zetazero(args.zero_index)
    gamma = float(mp.im(rho))
    zeta_prime = mp.diff(lambda z: mp.zeta(z), rho)

    rows = list(csv.DictReader(args.convergence_csv.open(newline="", encoding="utf-8")))

    print(f"zero_index={args.zero_index}")
    print(f"rho={mp.nstr(rho, 30)}")
    print(f"gamma={gamma:.15f}")
    print()
    print("sigma,abs_scaled_error,abs_predicted,abs_ratio,complex_ratio")

    for sigma in sigmas:
        label = f"zero {args.zero_index}"
        matches = [
            row for row in rows
            if abs(float(row["sigma"]) - sigma) < 1e-12 and row["label"] == label
        ]
        if not matches:
            raise SystemExit(f"No rows for sigma={sigma}, label={label}")

        row = matches[-1]
        x = float(row["X"])
        error = complex(float(row["real_error"]), float(row["imag_error"]))
        scaled_error = error * (x ** (sigma - 0.5))
        predicted = complex(mp.gamma(mp.mpf("0.5") - sigma) / zeta_prime)
        ratio = scaled_error / predicted

        print(
            f"{sigma},"
            f"{abs(scaled_error):.12g},"
            f"{abs(predicted):.12g},"
            f"{abs(scaled_error) / abs(predicted):.12g},"
            f"{ratio.real:.12g}{ratio.imag:+.12g}j"
        )


if __name__ == "__main__":
    main()

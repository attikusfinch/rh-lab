# Product Coefficient Autocorrelation

The previous scan showed that the key cancellation is in the product
coefficients

```text
A_Y(r) = sum_{ab=r, Y<=a,b<2Y} mu(a)mu(b).
```

This note removes the kernel and studies the binned autocorrelation of the
sequence `A_Y(r)` itself:

```text
sum_h A_h A_{h+\ell}
```

compared with the unsigned reference

```text
sum_h |A_h| |A_{h+\ell}|.
```

The lags are grouped by the same scale used in the kernel scan:

```text
T |Delta log r|.
```

## Current Scan

Command:

```powershell
python tools\plot_product_coefficient_autocorr.py outputs\mu_10m.i8 outputs\product_coefficient_autocorr_8k_t500.png --output-csv outputs\product_coefficient_autocorr_8k_t500.csv --max-y 8192 --t-scale 500 --bin-count 65536
```

Parameters:

```text
T scale: 500
dyadic Y: 128 through 8,192
log-product bins: 65,536
shells in T|Delta log r|: [0,1), [1,2), [2,5), [5,10), [10,20), [20,50), [50,inf)
```

There is an exact identity behind the global autocorrelation:

```text
sum_r A_Y(r) = (sum_{Y<=n<2Y} mu(n))^2.
```

Therefore

```text
sum_{r != s} A_Y(r)A_Y(s)
  = (sum_r A_Y(r))^2 - sum_r A_Y(r)^2.
```

So the total unweighted off-diagonal is controlled by the dyadic Mobius sum and
the diagonal energy.

Global results:

```text
Y     block sum mu  sum A  diagonal  signed offdiag  unsigned offdiag  cancellation
128   1             1      0.796570  -0.796509       2,376.53          3.35e-4
256  -3             9      0.796097  -0.794861       9,270.04          8.57e-5
512   0             0      0.790535  -0.790535       35,228.7          2.24e-5
1024  11            121    0.808248  -0.794285       141,829           5.60e-6
2048 -26            676    0.831370  -0.722418       574,661           1.26e-6
4096  41            1,681  0.844722  -0.676294       2,294,959         2.95e-7
8192 -54            2,916  0.858968  -0.732263       9,165,105         7.99e-8
```

The global cancellation ratio drops to `7.99e-8` at `Y=8192`.

Local shell cancellation ratios:

```text
Y     0..1      1..2      2..5      5..10     10..20    20..50    50..inf
128   1.94e-4   1.82e-2   1.37e-2   6.53e-3   8.64e-5   1.71e-3   4.40e-4
256   8.08e-4   1.09e-2   6.68e-4   1.46e-3   1.35e-3   1.12e-4   1.36e-5
512   1.56e-3   1.93e-3   1.34e-4   4.62e-4   3.45e-5   6.45e-5   3.50e-5
1024  2.08e-4   1.26e-3   1.33e-4   2.07e-4   4.99e-5   1.60e-5   4.82e-6
2048  5.92e-5   2.98e-5   2.10e-5   8.92e-5   1.14e-5   2.46e-5   6.20e-7
4096  2.77e-6   5.94e-6   2.82e-6   8.72e-6   5.24e-6   1.07e-6   1.14e-7
8192  6.82e-7   1.27e-5   3.95e-6   1.17e-6   7.40e-7   3.90e-7   8.94e-8
```

The local ratios are already in the `10^-6` range for large blocks. This is the
same scale that made the off-diagonal kernel small.

## Lesson

If `A_Y(r)` already has tiny signed autocorrelations compared with its unsigned
autocorrelations, then the off-diagonal moment bound should be attacked through
the pseudorandomness of this multiplicative coefficient sequence.

The global identity is useful, but it is not enough for the kernel problem:
kernel weights are local in `log r`. The proof-shaped target is therefore a
local autocorrelation bound for `A_Y(r)` on log-product shells.

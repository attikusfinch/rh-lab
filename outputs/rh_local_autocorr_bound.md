# Local Autocorrelation Bound Scan

The previous product-coefficient scan used one scale, `T = 500`. This scan asks
whether the same local cancellation survives when the lag shells are rescaled.

For each dyadic block `[Y, 2Y)`, define

```text
A_Y(r) = sum_{ab=r, Y<=a,b<2Y} mu(a)mu(b).
```

The test bins `A_Y(r)` by `log r`, computes the binned autocorrelation

```text
sum_h A_h A_{h+ell}
```

and compares each local shell with its unsigned version

```text
sum_h |A_h| |A_{h+ell}|.
```

Shells are measured by

```text
T |Delta log r|.
```

## Current Scan

Command:

```powershell
python tools\plot_local_autocorr_bound.py outputs\mu_10m.i8 outputs\local_autocorr_bound_8k.png --output-csv outputs\local_autocorr_bound_8k.csv --max-y 8192 --t-scales 100,200,500,1000,2000 --bin-count 65536
```

Parameters:

```text
dyadic Y: 128 through 8,192
T scales: 100, 200, 500, 1,000, 2,000
log-product bins: 65,536
shells in T|Delta log r|: [0,1), [1,2), [2,5), [5,10), [10,20), [20,50), [50,inf)
```

Worst local shell ratio for each `Y`, after taking the maximum over all tested
`T` and shells:

```text
Y     worst T  shell  max abs(signed)/unsigned
128   2000     2..5   2.283e-2
256   500      1..2   1.085e-2
512   2000     2..5   4.530e-3
1024  500      1..2   1.258e-3
2048  1000     1..2   3.027e-4
4096  2000     1..2   1.904e-5
8192  500      1..2   1.273e-5
```

Worst local shell ratio for each tested `T`:

```text
T     Y    shell  max abs(signed)/unsigned
100   128  1..2   6.526e-3
200   128  0..1   6.348e-3
500   128  1..2   1.820e-2
1000  128  0..1   1.174e-2
2000  128  2..5   2.283e-2
```

For the largest tested block, `Y = 8192`, the worst shell is

```text
T=500, shell 1..2, ratio = 1.273e-5.
```

## 16k Stress Scan

The first scan was extended by one more dyadic block using incremental product
aggregation, so the script does not keep every product pair in memory at once.

Command:

```powershell
python tools\plot_local_autocorr_bound.py outputs\mu_10m.i8 outputs\local_autocorr_bound_16k.png --output-csv outputs\local_autocorr_bound_16k.csv --first 128 --max-y 16384 --t-scales 100,200,500,1000,2000 --bin-count 65536 --aggregation incremental --pair-row-chunk-size 512 --merge-row-chunks 4
```

Worst local shell ratio for each `Y`:

```text
Y      nonzero n  unique products  worst T  shell  max abs(signed)/unsigned
128    79         3,088            2000     2..5   2.283e-2
256    157        12,058           500      1..2   1.085e-2
512    310        46,466           2000     2..5   4.530e-3
1024   621        184,601          500      1..2   1.258e-3
2048   1,246      735,513          1000     1..2   3.027e-4
4096   2,491      2,917,017        2000     1..2   1.904e-5
8192   4,980      11,565,830       500      1..2   1.273e-5
16384  9,958      45,867,647       2000     2..5   3.412e-6
```

For the new block, `Y = 16384`, the largest tested local ratio is

```text
T=2000, shell 2..5, ratio = 3.412e-6.
```

## Lesson

The local autocorrelation cancellation is not a one-scale artifact. Across
`T = 100..2000`, the worst ratios fall rapidly as `Y` grows. The `16k` stress
scan keeps the same trend even after the unique product set grows to about
`45.9M` coefficients.

The next proof-shaped target is a uniform local bound of the form

```text
|sum_{r,s in local log shell} A_Y(r) A_Y(s)|
  <= epsilon(Y, T, shell) sum_{r,s in shell} |A_Y(r)| |A_Y(s)|
```

with `epsilon` decreasing as the dyadic block grows. This is exactly the
ingredient needed to make the off-diagonal kernel contribution small without
assuming cancellation from the kernel itself.

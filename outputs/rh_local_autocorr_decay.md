# Local Autocorrelation Decay

This note turns the local autocorrelation scan into a quantitative lemma
candidate.

For

```text
A_Y(r) = sum_{ab=r, Y<=a,b<2Y} mu(a)mu(b),
```

define a local shell by

```text
T |log r - log s| in I.
```

The measured local cancellation ratio is

```text
epsilon(Y; T, I)
  = |sum_{(r,s) in shell} A_Y(r) A_Y(s)|
    / sum_{(r,s) in shell} |A_Y(r)| |A_Y(s)|.
```

The scan records

```text
epsilon(Y) = max_{tested T, tested I} epsilon(Y; T, I).
```

## Current Data

Command:

```powershell
python tools\plot_local_autocorr_decay.py outputs\local_autocorr_bound_16k.csv outputs\local_autocorr_decay_16k.png --output-csv outputs\local_autocorr_decay_16k.csv --tail-min-y 1024
```

Worst ratios from the `16k` local-autocorrelation scan:

```text
Y      worst T  shell  epsilon(Y)
128    2000     2..5   2.283e-2
256    500      1..2   1.085e-2
512    2000     2..5   4.530e-3
1024   500      1..2   1.258e-3
2048   1000     1..2   3.027e-4
4096   2000     1..2   1.904e-5
8192   500      1..2   1.273e-5
16384  2000     2..5   3.412e-6
```

Power-law fits:

```text
all Y fit:        epsilon(Y) ~= 520.087 Y^-1.94491, R^2 = 0.974356
tail Y>=1024 fit: epsilon(Y) ~= 3246.22 Y^-2.16249, R^2 = 0.949674
```

Separate fits by `T`:

```text
T     alpha    R^2
100   2.01365  0.97747
200   2.06930  0.98294
500   1.96352  0.95910
1000  2.06304  0.96271
2000  1.99175  0.96944
```

Observed envelope constants for

```text
epsilon(Y) <= C Y^-alpha
```

are:

```text
alpha  observed C       tight point
1.00   2.92225          Y=128
1.25   11.1142          Y=256
1.50   52.4769          Y=512
1.75   249.624          Y=512
2.00   1319.53          Y=1024
2.25   10361.8          Y=16384
```

## Lemma Candidate

For the initial tested range and shell family, the clean empirical statement is:

```text
epsilon(Y) <= 1320 / Y^2.
```

After widening the tested `T` grid to

```text
50, 75, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000
```

the worst ratios become:

```text
Y      worst T  shell  epsilon(Y)
128    2000     2..5   2.283e-2
256    5000     0..1   1.351e-2
512    3000     5..10  5.079e-3
1024   3000     0..1   1.428e-3
2048   5000     1..2   3.646e-4
4096   3000     1..2   5.962e-5
8192   5000     1..2   1.760e-5
16384  3000     5..10  4.784e-6
```

The wide-`T` fits are:

```text
all Y fit:        epsilon(Y) ~= 332.441 Y^-1.84136, R^2 = 0.986066
tail Y>=1024 fit: epsilon(Y) ~= 2529.02 Y^-2.08161, R^2 = 0.995621
```

The observed `alpha = 2` envelope becomes

```text
epsilon(Y) <= 1529.33 / Y^2.
```

For a proof route, use the slightly rounded target:

```text
epsilon(Y) <= 1600 / Y^2
```

or, more flexibly,

```text
epsilon(Y) <= C Y^(-2 + delta)
```

with `delta > 0` small enough for the final moment sum.

## Lesson

The cancellation scale now has a concrete quantitative target. The exponent is
not just visible in the worst-over-everything curve; it also appears separately
inside each tested `T` scale, where all fitted exponents are close to `2`.

The wide-`T` stress shifts the worst shell at some `Y`, but it does not change
the qualitative decay. The next step is to connect this candidate bound back to
the off-diagonal kernel estimate.

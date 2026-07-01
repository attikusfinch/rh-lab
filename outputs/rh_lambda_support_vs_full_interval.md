# Lambda On Product Support Versus Full Interval

The sign-weight decomposition reduced the core local cancellation target to the
Liouville-type sign

```text
lambda(r) = (-1)^Omega(r)
```

on the product support

```text
S_Y = {ab: Y<=a,b<2Y, mu(a)mu(b) != 0}.
```

This note compares local autocorrelation cancellation of `lambda(r)` on this
product support with `lambda(r)` on the full product interval

```text
Y^2 <= r < (2Y)^2.
```

The full interval is exact through `Y=4096`, using a generated Liouville binary
up to `67,108,864`.

## Current Scan

Commands:

```powershell
g++ -O3 -std=c++17 tools\liouville_dump.cpp -o tools\liouville_dump.exe
tools\liouville_dump.exe 67108864 outputs\lambda_67m.i8
python tools\plot_lambda_support_vs_full_interval.py outputs\mu_10m.i8 outputs\lambda_67m.i8 outputs\lambda_support_vs_full_4k_wide_t.png --output-csv outputs\lambda_support_vs_full_4k_wide_t.csv --first 128 --max-y 4096 --t-scales 50,75,100,150,200,300,500,750,1000,1500,2000,3000,5000 --bin-count 65536 --pair-row-chunk-size 512 --merge-row-chunks 4 --full-chunk-size 1000000
```

Worst local ratios:

```text
Y     product support  full interval  support/full  support density
128   2.879e-2         1.765e-3       16.313        0.06283
256   1.813e-2         6.194e-4       29.268        0.06133
512   4.403e-3         2.080e-4       21.173        0.05908
1024  1.264e-3         3.448e-5       36.653        0.05868
2048  2.997e-4         1.656e-5       18.090        0.05845
4096  3.829e-5         5.117e-6       7.483         0.05796
```

Power-law fits:

```text
product support lambda: alpha = 1.92364, R^2 = 0.972775
full interval lambda:   alpha = 1.72615, R^2 = 0.991157
```

Sign balance:

```text
Y     support balance  full interval balance
128   1.943e-2        -8.952e-4
256   1.161e-2        -7.019e-4
512   6.370e-3        -7.629e-6
1024  2.172e-3        -7.947e-5
2048  5.805e-4        -8.853e-5
4096 -1.468e-3        -5.527e-5
```

## Lesson

The product support is not behaving like the full interval. It is much sparser
and makes local `lambda` cancellation harder by roughly one to two orders of
magnitude on these blocks.

But the product-support curve still decays close to `Y^-2`. So the next target
is not a generic interval estimate for `lambda(n)`. It is a support-sensitive
estimate:

```text
sum_{r,s in S_Y local shell} lambda(r) lambda(s)
```

must be small compared with the unsigned support-pair count in the same shell.

That is a sharper and more honest target than treating `S_Y` as an ordinary
random subset of `[Y^2, 4Y^2)`.

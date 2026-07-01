# Uniform Moment Bound

The high-moment scans suggest a proof-shaped estimate of the form

```text
E_t |g_Y(t)|^(2k) <= C^k k!,
g_Y(t) = B_Y(t) / sqrt(Y),
```

uniformly over dyadic blocks `Y`.

This note tracks the empirical version of that estimate. For each moment order
`2k`, define

```text
root_k(Y) = (E_t |g_Y(t)|^(2k) / k!)^(1/(2k)).
```

Then a uniform bound over the scan is

```text
C = max_{Y,k} root_k(Y)^2.
```

With this `C`, Markov's inequality gives the proof-shaped tail bound

```text
measure{|g_Y| > lambda} / T
  <= min_k k! (C / lambda^2)^k.
```

## Current scan

Command:

```powershell
python tools\plot_dyadic_uniform_bounds.py outputs\mu_10m.i8 outputs\dyadic_uniform_bounds_5m_t500.png --output-csv outputs\dyadic_uniform_bounds_5m_t500.csv --limit 5000000 --t-max 500 --t-points 1601 --moments 1,2,3,4,5,6,7,8,9,10 --full-blocks-only
```

Parameters:

```text
N limit: 5,000,000
t range: [0, 500]
t grid points: 1,601
dyadic Y: full blocks only, 128 through 2,097,152
moments: 2 through 20
```

The empirical uniform constant is:

```text
uniform root = 0.830417
uniform C    = 0.689593
```

The maximum root occurs already at the second moment:

```text
moment order: 2
Y: 4,096
E|g_Y|^2 = 0.689593
```

Maximal absolute roots by moment order:

```text
2nd  root = 0.830417 at Y=4,096
4th  root = 0.824712 at Y=4,096
6th  root = 0.818202 at Y=4,096
8th  root = 0.808010 at Y=4,096
10th root = 0.794694 at Y=4,096
12th root = 0.782146 at Y=65,536
14th root = 0.774387 at Y=65,536
16th root = 0.763525 at Y=65,536
18th root = 0.750930 at Y=65,536
20th root = 0.737491 at Y=65,536
```

So, on this finite scan, all checked moment orders satisfy the uniform bound

```text
E_t |g_Y(t)|^(2k) <= 0.689593^k k!.
```

The same data still shows the local-Gaussian stress from the previous scan. The
largest local ratios are all at `Y=65,536`:

```text
4th  local ratio = 1.04974
6th  local ratio = 1.22062
8th  local ratio = 1.48268
10th local ratio = 1.73239
12th local ratio = 1.85468
14th local ratio = 1.78799
16th local ratio = 1.54975
18th local ratio = 1.21342
20th local ratio = 0.863976
```

That means local normalization by each block's own second moment is less stable
than the absolute `C^k k!` target.

Uniform large-values comparison:

```text
level  max empirical tail  empirical Y  blockwise Markov  Markov Y  uniform C Markov
1.25   0.0974391           4,096        0.378965          4,096     0.389561
1.50   0.0430981           4,096        0.158040          4,096     0.172736
1.75   0.0131168           4,096        0.0438519         4,096     0.061699
2.00   0.00374766          65,536       0.00783976        65,536    0.0182745
2.25   0                   128          0.000743451       65,536    0.00438556
```

The uniform `C` Markov bound is still conservative, but it is now within about
one order of magnitude at the visible right tail. This is much closer to a proof
shape than the earlier derivative/Sobolev bound.

## Lesson

The right target is not merely that the aggregate distribution looks Gaussian.
The proof needs a uniform moment constant over every dyadic block:

```text
sup_Y E_t |B_Y(t) / sqrt(Y)|^(2k) <= C^k k!.
```

This scan turns that target into a directly testable quantity.

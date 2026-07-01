# Dyadic High-Moment Scan

For

```text
B_Y(t) = sum_{Y < n <= 2Y} mu(n)n^{-it},
g_Y(t) = B_Y(t) / sqrt(Y),
```

the large-values problem can be attacked through high moments:

```text
E_t |g_Y(t)|^(2k).
```

If `g_Y(t)` behaves like a complex Gaussian with second moment `m2`, then

```text
E |g_Y|^(2k) ~= k! m2^k.
```

This is useful because Markov's inequality gives

```text
measure{|g_Y| > lambda} / T
  <= E|g_Y|^(2k) / lambda^(2k).
```

So good control of many moments is a direct route toward large-values bounds.

## Current scan

Parameters:

```text
N limit: 2,000,000
t range: [0, 200]
t grid points: 801
dyadic Y: 128 through 1,048,576
moments: 2, 4, 6, 8, 10, 12
```

Current results are produced by:

```powershell
python tools\plot_dyadic_moments.py outputs\mu_10m.i8 outputs\dyadic_moments_2m.png --output-csv outputs\dyadic_moments_2m.csv --limit 2000000 --t-max 200 --t-points 801
```

Over all scanned `(Y,t)` grid points:

```text
overall E|g|^2  = 0.636765
overall max |g| = 2.03946
```

Moment ratios against the complex-Gaussian reference `k! m2^k`:

```text
2nd moment ratio  = 1.000000
4th moment ratio  = 0.905715
6th moment ratio  = 0.751727
8th moment ratio  = 0.587528
10th moment ratio = 0.439102
12th moment ratio = 0.314360
```

For individual dyadic blocks, the largest ratios are also below `1`:

```text
max 4th moment ratio  = 0.966738
max 6th moment ratio  = 0.959024
max 8th moment ratio  = 0.931070
max 10th moment ratio = 0.831068
max 12th moment ratio = 0.662572
```

All these maxima occur at `Y=65,536`, the same dyadic block that contains the
largest observed peak.

Empirical tail compared with the Gaussian tail and the best Markov bound from
the scanned moments:

```text
level  empirical   gaussian    best Markov
1.50   0.0180132   0.0292030   0.0904538
1.75   0.00356697  0.00815222  0.0182885
2.00   0.000713394 0.00187012  0.00368362
2.25   0           0.000352542 0.000896282
```

The empirical tail is smaller than the Gaussian model on this grid. The finite
Markov bound from moments up to order `12` is still looser than the observed
tail, but much closer to the right kind of argument than the first-derivative
Sobolev bound.

## Lesson

High moments are a sharper next target than a first-derivative Sobolev bound.
If the moment ratios stay bounded uniformly in `Y` and `k`, they support a
subgaussian large-values route to the dyadic cancellation estimate.

The proof-shaped target is now clearer:

```text
E_t |B_Y(t) / sqrt(Y)|^(2k) <= C^k k!
```

for enough ranges of `Y`, `T`, and `k`. Optimizing Markov's inequality over `k`
would then produce an exponential large-values estimate.

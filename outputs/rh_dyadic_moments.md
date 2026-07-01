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

## Stress scan

The next pass extends the interval and the moment range:

```text
N limit: 5,000,000
t range: [0, 500]
t grid points: 1,601
dyadic Y: full blocks only, 128 through 2,097,152
moments: 2, 4, 6, 8, 10, 12, 14, 16, 18, 20
```

Command:

```powershell
python tools\plot_dyadic_moments.py outputs\mu_10m.i8 outputs\dyadic_moments_5m_t500_full.png --output-csv outputs\dyadic_moments_5m_t500_full.csv --limit 5000000 --t-max 500 --t-points 1601 --moments 1,2,3,4,5,6,7,8,9,10 --full-blocks-only
```

Aggregate results over all scanned full dyadic blocks:

```text
overall E|g|^2  = 0.584458
overall max |g| = 2.03976

2nd moment ratio  = 1.000000
4th moment ratio  = 0.936379
6th moment ratio  = 0.824471
8th moment ratio  = 0.687930
10th moment ratio = 0.545591
12th moment ratio = 0.410399
14th moment ratio = 0.291451
16th moment ratio = 0.194566
18th moment ratio = 0.121778
20th moment ratio = 0.071405
```

The aggregate tail is still below the Gaussian reference:

```text
level  empirical   gaussian    best Markov
1.50   0.0143244   0.0212857   0.0751690
1.75   0.00303977  0.00530079  0.0135436
2.00   0.000249844 0.00106589  0.00114927
2.25   0           0.00017306  0.000108986
```

But the uniform-in-`Y` picture is sharper than the aggregate picture. The worst
block is still

```text
Y = 65,536
max |g| = 2.03976
E|g|^2 = 0.551904
```

and its ratios to the local Gaussian reference are:

```text
4th  ratio = 1.04974
6th  ratio = 1.22062
8th  ratio = 1.48268
10th ratio = 1.73239
12th ratio = 1.85468
14th ratio = 1.78799
16th ratio = 1.54975
18th ratio = 1.21342
20th ratio = 0.863976
```

This does not contradict the route. It says the aggregate distribution is
subgaussian-looking, while a proof must control each dyadic block uniformly.
The absolute Gaussian-normalized roots are still modest; across the stress scan,
their maxima are:

```text
4th root  = 0.824712
6th root  = 0.818202
8th root  = 0.808010
10th root = 0.794694
12th root = 0.782146
14th root = 0.774387
16th root = 0.763525
18th root = 0.750930
20th root = 0.737491
```

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

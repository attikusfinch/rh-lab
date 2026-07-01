# Dyadic Smoothness and Supremum Control

For

```text
B_Y(t) = sum_{Y < n <= 2Y} mu(n)n^{-it},
g_Y(t) = B_Y(t) / sqrt(Y),
```

the derivative is

```text
B_Y'(t) = -i sum_{Y < n <= 2Y} mu(n)(log n)n^{-it}.
```

So `g_Y` changes on a scale controlled by roughly `log Y`.

## Why this matters

Large-values estimates control how often `g_Y(t)` is large. Smoothness controls how narrow a high spike can be.

If `|g_Y(t0)| = M` and

```text
max_t |g_Y'(t)| <= D,
```

then for any `lambda < M`, the set where

```text
|g_Y(t)| > lambda
```

must have width at least approximately

```text
(M - lambda) / D.
```

So a very tall spike cannot be completely invisible to a large-values scan.

## Basic Sobolev bound

A simple one-dimensional inequality on an interval of length `T` gives

```text
sup |g|^2
  <= mean |g|^2
     + 2T sqrt(mean |g|^2 * mean |g'|^2).
```

This is useful as a sanity check, but it is not strong enough by itself. Since typically

```text
mean |g'|^2 ~ (log Y)^2,
```

the bound is much larger than the observed supremum on long `t` intervals.

## Current scan

Parameters:

```text
N limit: 2,000,000
t range: [0, 200]
t grid points: 801
dyadic Y: 128 through 1,048,576
```

Worst observed dyadic block:

```text
Y = 65,536
t = 184.5
|B_Y(t)| / sqrt(Y) = 2.03946
max |g_Y'(t)| = 23.5134
max |g_Y'(t)| / log(Y) = 2.12016
rms |g_Y'(t)| / log(Y) = 0.9060
```

The largest basic Sobolev bound over the scan is

```text
59.202
```

while the largest observed value is only

```text
2.03946
```

So the strongest bound from this elementary mean-square plus first-derivative
argument is not close to the observed scale. Across dyadic blocks, the ratio

```text
Sobolev bound / observed maximum
```

ranges roughly from `23.6` to `39.5`, with median `29.2`.

The derivative scaling itself looks natural: `max |g_Y'| / log(Y)` stays near
`1.3..2.1`, and `rms |g_Y'| / log(Y)` stays near `0.7..1.0`. The loss comes from
turning average derivative information into a global supremum over a long
interval.

For the worst block, the measured large-value sets are much wider than the
minimal Lipschitz lower width:

```text
measure{|g_Y| > 1.5} = 9.73783
measure{|g_Y| > 1.75} = 5.24345
measure{|g_Y| > 2.0} = 1.99750
```

So this scan does not reveal hidden needle-like spikes. It says the naive
smoothness route is mathematically too lossy.

## Lesson

Mean-square plus a generic derivative bound is too lossy. To prove the needed dyadic estimate, one likely needs stronger large-values or high-moment information, not just first-derivative smoothness.

This is still progress: it rules out a naive route and identifies the next stronger target.

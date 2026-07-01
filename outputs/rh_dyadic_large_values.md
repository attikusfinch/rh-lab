# Dyadic Large-Values Scan

The adversarial `t` scan looks at maxima. This layer also tracks mean-square size and large-value tails for

```text
B_Y(t) = sum_{Y < n <= 2Y} mu(n)n^{-it}.
```

We measure the normalized size

```text
|B_Y(t)| / sqrt(Y).
```

## Why this matters

A supremum bound can be approached through three pieces:

1. Mean-square control:

```text
(1/T) integral_0^T |B_Y(t)|^2 dt = O(Y polylog(Y)).
```

2. Large-values control:

```text
measure{t : |B_Y(t)| / sqrt(Y) > lambda}
```

should decay quickly as `lambda` grows.

3. Smoothness in `t`, so a very tall spike must occupy a nontrivial interval and therefore show up in mean-square or large-values estimates.

This does not prove the required dyadic cancellation, but it separates the problem into average control, tail control, and peak-width control.

## Proof-shaped target

For fixed `T`, a useful estimate would be

```text
measure{0 <= t <= T : |B_Y(t)| > lambda sqrt(Y)}
  <= T * exp(-c lambda^2)
```

or any sufficiently strong polynomial tail paired with derivative/smoothness bounds.

Such an estimate would support the uniform dyadic target:

```text
sup_{0 <= t <= T} |B_Y(t)| <= C(T,eps)Y^(1/2 + eps).
```

## Current scan

Parameters:

```text
N limit: 2,000,000
t range: [0, 200]
t grid points: 801
dyadic Y: 128 through 1,048,576
```

Summary over all scanned `(Y,t)` grid points:

```text
overall mean of (|B_Y(t)| / sqrt(Y))^2 = 0.636765
fraction > 1.00 = 0.224630
fraction > 1.25 = 0.075174
fraction > 1.50 = 0.018013
fraction > 1.75 = 0.003567
fraction > 2.00 = 0.000713
fraction > 2.25 = 0
```

The largest observed normalized value remains

```text
2.03946
```

at `Y=65,536`, consistent with the earlier dyadic `t` scan.

Interpretation: typical values are well within square-root scale, and values above `2 sqrt(Y)` are very rare on this grid.

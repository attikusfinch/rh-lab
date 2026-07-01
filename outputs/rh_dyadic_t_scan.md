# Dyadic T-Scan

For the dyadic twisted Mobius blocks

```text
B_Y(t) = sum_{Y < n <= 2Y} mu(n)n^{-it},
```

we need control not just at selected frequencies, but uniformly in `t`.

The numerical scan computes

```text
max_{0 <= t <= T} |B_Y(t)| / sqrt(Y)
```

on a grid, for each dyadic block.

## Interpretation

This is an adversarial search for resonance. A large spike would indicate a frequency where the multiplicative oscillation `n^{-it}` lines up unusually well with `mu(n)`.

The scan is not a proof of the supremum over all real `t`; it is a finite grid diagnostic. Its purpose is to look for obvious bad resonances before trying to prove a uniform estimate.

## Current scan

Parameters:

```text
N limit: 2,000,000
dyadic Y: 128 through 1,048,576
t range: [0, 200]
t grid points: 801
grid spacing: 0.25
```

Worst grid point found:

```text
Y = 65,536
t = 184.5
|B_Y(t)| / sqrt(Y) = 2.03946
```

Across the scanned dyadic blocks, the worst-case envelope stayed roughly between

```text
1.31 and 2.04
```

while the mean over `t` was typically around

```text
0.6 to 0.8.
```

This did not find an obvious bad resonance. The scan is consistent with square-root-scale cancellation, with modest grid-search maxima.

## Proof-shaped target

A useful theorem would be:

```text
For every fixed T and eps > 0,
sup_{0 <= t <= T} |B_Y(t)| <= C(T,eps) Y^(1/2 + eps).
```

An even stronger global target would control all real `t` with polynomial dependence on `|t|`:

```text
|B_Y(t)| <= C(eps) (1 + |t|)^A Y^(1/2 + eps).
```

Such a dyadic estimate would imply the boundedness/convergence of the Mobius Dirichlet series in `Re(s)>1/2`, which is the current proof route to RH.

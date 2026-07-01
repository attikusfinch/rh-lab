# Dyadic Mobius Cancellation Target

The contour formulation identifies the analytic bottleneck, but proving it from the contour alone risks circularity because it requires control of `1/zeta`.

A more arithmetic target is cancellation in dyadic twisted Mobius blocks:

```text
B_Y(t) = sum_{Y < n <= 2Y} mu(n)n^{-it}.
```

## Why this would imply bounded smoothed sums

For

```text
F_X(s) = sum mu(n)n^{-s}exp(-n/X),
s = sigma + it,
```

break the sum into dyadic blocks `Y < n <= 2Y`.

If for every fixed `t` and every `eps > 0` we had

```text
|B_Y(t)| <= C(t,eps) Y^(1/2 + eps),
```

then by partial summation on each dyadic block,

```text
sum_{Y < n <= 2Y} mu(n)n^{-sigma-it}
  = O_t,eps(Y^(1/2 + eps - sigma)).
```

For any fixed `sigma > 1/2`, choose `eps < sigma - 1/2`. Then

```text
sum_Y Y^(1/2 + eps - sigma)
```

converges over dyadic `Y`.

This gives boundedness/convergence of the Mobius Dirichlet series in `Re(s)>1/2`, hence the RH route.

## Clean target

The dream estimate is square-root cancellation:

```text
|sum_{Y < n <= 2Y} mu(n)n^{-it}| = O_t(Y^(1/2) polylog(Y)).
```

This is the arithmetic version of the same wall:

```text
Mobius must not correlate too strongly with n^(it).
```

## Why this is useful

Unlike the contour remainder estimate, this target is stated entirely in terms of finite arithmetic sums. It avoids assuming information about zeta zeros directly.

That makes it a better place to search for a genuinely new proof idea.

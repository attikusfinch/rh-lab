# RH route through Mobius Dirichlet sums

We are studying

```text
M(x) = sum_{n <= x} mu(n)
D_N(s) = sum_{n <= N} mu(n) n^{-s}
```

where `s = sigma + it`.

## Exact bridge

By partial summation,

```text
D_N(s) = M(N) N^{-s} + s integral_1^N M(x) x^{-s-1} dx.
```

So a square-root-type bound

```text
M(x) = O(x^{1/2 + eps}) for every eps > 0
```

implies convergence of

```text
sum mu(n) n^{-s}
```

throughout `Re(s) > 1/2`.

But for `Re(s) > 1`,

```text
sum mu(n) n^{-s} = 1 / zeta(s).
```

If the series can be continued/converges in `Re(s) > 1/2`, then `1 / zeta(s)` exists there, so `zeta(s)` has no zeros there. By the functional equation symmetry, this is RH.

## Current experimental evidence

The finite sums

```text
D_N(sigma, t) = sum_{n <= N} mu(n) n^{-sigma-it}
```

were tested up to `N = 10,000,000`.

At zeta-zero ordinates `t = gamma`, the sums are large, as expected, because `1/zeta(s)` has peaks near zeros. But for fixed `sigma > 1/2`, the growth slows strongly and appears to stabilize:

```text
sigma = 0.75: last / previous decade around 1.01
sigma = 1.10: last / previous decade around 1.0001
```

At the boundary `sigma = 0.5`, the same sums still grow slowly:

```text
sigma = 0.5: last / previous decade around 1.14 to 1.20 at the first zeros
```

This matches the expected RH boundary picture.

## The proof-shaped target

A direct RH-scale target is:

```text
For every sigma > 1/2 and bounded t-window,
sup_N |D_N(sigma + it)| < infinity.
```

An even stronger useful target would be an explicit bound:

```text
|D_N(sigma + it)| <= C(sigma, t)
```

with controlled growth as `sigma -> 1/2+`.

If we can prove this without assuming RH, then the Mobius series defines `1/zeta(s)` in the critical half-plane and rules out zeros with `Re(s) > 1/2`.

## Main obstacle

The finite sums can be bounded empirically, but proving uniform boundedness requires a non-visual reason why

```text
mu(n) n^{-it}
```

has persistent cancellation for every real `t`.

This is the real mathematical bottleneck.

## Prime-sum warning

The Euler product gives, initially for `Re(s) > 1`,

```text
sum mu(n)n^{-s} = product_p (1 - p^{-s})
```

and therefore

```text
log product_p(1 - p^{-s})
  = -sum_p p^{-s} - sum_p sum_{k >= 2} p^{-ks}/k.
```

For `Re(s) > 1/2`, the double sum over `k >= 2` would converge absolutely, because it starts with `p^{-2s}`.

But this is not a clean equivalent route outside `Re(s) > 1`. The ordinary Euler product can fail even when the analytically continued function is perfectly finite. In particular, for real `s = sigma` with `1/2 < sigma <= 1`,

```text
sum_p p^{-sigma}
```

diverges.

So prime sums are useful diagnostics for resonance, especially when `t != 0`, but the proof route must not rely on ordinary Euler-product convergence in the critical strip.

The safer central target remains the Mobius Dirichlet partial sums:

```text
sum_{n <= N} mu(n)n^{-sigma-it}
```

and their boundedness/convergence for `sigma > 1/2`.

## Smoothed detector

A cleaner analytic object is the exponentially smoothed sum

```text
F_X(s) = sum_{n >= 1} mu(n)n^{-s} exp(-n/X).
```

For `Re(s)` initially large, Mellin inversion gives

```text
F_X(s) = (1 / 2 pi i) integral Gamma(w) X^w / zeta(s+w) dw.
```

This is useful because a zero `rho` of `zeta` creates a pole at

```text
w = rho - s.
```

So if there were a zero with `Re(rho) > Re(s)`, the smoothed sum would pick up a growing term of approximate size

```text
X^(rho-s).
```

Thus a proof that `F_X(s)` stays bounded as `X -> infinity` for every `Re(s) > 1/2` would rule out zeros to the right of the critical line.

See `rh_smoothed_lemma.md` for the proof-shaped statement.

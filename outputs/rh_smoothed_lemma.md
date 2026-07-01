# Smoothed Mobius Lemma Route

Define

```text
F_X(s) = sum_{n >= 1} mu(n) n^{-s} exp(-n/X).
```

For `Re(s)` large, Mellin inversion gives

```text
F_X(s) = (1 / 2 pi i) integral_(c) Gamma(w) X^w / zeta(s+w) dw,
```

where `c > 1 - Re(s)`.

## Lemma

Suppose that for every fixed `s` with

```text
Re(s) > 1/2
```

the smoothed sums stay bounded as `X -> infinity`:

```text
F_X(s) = O_s(1).
```

Then the Riemann hypothesis follows.

## Proof sketch

Assume for contradiction that zeta has a zero

```text
rho = beta + i gamma
```

with

```text
beta > 1/2.
```

Pick a real `sigma` such that

```text
1/2 < sigma < beta
```

and set

```text
s = sigma + i gamma.
```

In the Mellin integral for `F_X(s)`, the factor

```text
1 / zeta(s+w)
```

has a pole at

```text
w = rho - s = beta - sigma.
```

This point has positive real part. Moving the contour left crosses this pole, producing a residue term of size

```text
constant * X^(beta - sigma)
```

or, if the zero has multiplicity greater than one,

```text
X^(beta - sigma) * polynomial_in(log X).
```

Gamma has no zeros, and the pole of `1/zeta` has nonzero principal part, so this residue is not zero.

Since `beta - sigma > 0`, this term grows without bound as `X -> infinity`.

That contradicts the assumed boundedness of `F_X(s)`.

Therefore zeta has no zeros with `Re(s) > 1/2`.

By the functional equation symmetry, this rules out zeros with `Re(s) < 1/2` inside the critical strip as well. Hence all nontrivial zeros lie on the critical line.

## What remains

The hard missing theorem is exactly:

```text
For every fixed sigma > 1/2 and real t,
sum_{n >= 1} mu(n)n^{-sigma-it} exp(-n/X)
is bounded uniformly in X.
```

This is still RH-strength. But it is a very clean target: prove uniform boundedness of a smoothed Mobius transform.

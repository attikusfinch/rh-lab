# Contour Remainder Target

We use the smoothed Mobius transform

```text
F_X(s) = sum_{n >= 1} mu(n)n^{-s}exp(-n/X).
```

Mellin inversion gives, initially on a right-side contour,

```text
F_X(s)
  = (1 / 2 pi i) integral_(c) Gamma(w) X^w / zeta(s+w) dw,
```

where `c > max(0, 1 - Re(s))`.

## Shift the contour

Move the contour from `Re(w)=c` to a line

```text
Re(w)=a
```

with

```text
-1 < a < 0.
```

The restriction `a > -1` avoids crossing the negative integer poles of `Gamma(w)`. The pole at `w=0` contributes

```text
1 / zeta(s).
```

Every zeta zero `rho` crossed by the shift contributes

```text
Gamma(rho - s) X^(rho - s) / zeta'(rho)
```

when the zero is simple. Higher multiplicity gives the same power multiplied by a polynomial in `log X`.

So, schematically:

```text
F_X(s)
  = 1/zeta(s)
    + sum_crossed_zero_residues
    + R_a(X;s),
```

where

```text
R_a(X;s)
  = (1 / 2 pi i) integral_(a) Gamma(w) X^w / zeta(s+w) dw.
```

## Remainder bound

On `w = a + iv`,

```text
|X^w| = X^a.
```

Therefore

```text
|R_a(X;s)|
  <= X^a / (2 pi)
     integral_{-infty}^{infty}
       |Gamma(a+iv)| / |zeta(s+a+iv)| dv.
```

Because `Gamma(a+iv)` decays exponentially in `|v|`, this integral would be finite if `1/zeta(s+a+iv)` is controlled along the shifted line.

If that integral is finite for some `a < 0`, then

```text
R_a(X;s) = O_s,a(X^a),
```

which decays as `X -> infinity`.

## Why this is RH-strength

If there is a zero

```text
rho = beta + i gamma
```

with

```text
beta > 1/2,
```

then choose

```text
s = sigma + i gamma
```

with

```text
1/2 < sigma < beta.
```

During the contour shift, the zero creates a residue at

```text
w = rho - s = beta - sigma,
```

which has positive real part. That residue grows like

```text
X^(beta - sigma).
```

So any theorem proving boundedness of `F_X(s)` for every `Re(s)>1/2` must somehow rule out exactly these growing positive-power residues.

## The clean proof target

The direct target is:

```text
For every fixed s with Re(s)>1/2,
F_X(s) = O_s(1) as X -> infinity.
```

Equivalently, from the contour viewpoint:

```text
No positive-power zero residue can occur,
and the shifted contour remainder can be bounded.
```

This is the precise analytic bottleneck.

## What a non-circular proof would need

Using the contour formula alone, bounding the remainder typically requires information about zeros of `zeta`, which risks circularity.

A non-circular proof would need an arithmetic estimate directly on

```text
sum mu(n)n^{-s}exp(-n/X)
```

showing uniform cancellation for `Re(s)>1/2`, without first assuming a zero-free half-plane.

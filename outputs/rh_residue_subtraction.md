# Residue Subtraction Check

We started from the smoothed convergence error

```text
E_X(s) = F_X(s) - 1/zeta(s)
```

and subtracted explicit zero-residue terms:

```text
Gamma(rho - s) X^(rho - s) / zeta'(rho).
```

For `s = sigma + i gamma_1`, subtracting just the first conjugate zero pair

```text
rho_1 = 1/2 + i gamma_1
conj(rho_1) = 1/2 - i gamma_1
```

removes almost all of the visible error.

## Numerical result

At the final plotted point `X=200000`, using the `N=1000000` Mobius cutoff:

```text
sigma   raw error      after first zero pair
0.55    14.1279        4.25e-4
0.60    3.97538        2.10e-4
0.75    0.292242       2.26e-5
1.10    0.00306193     1.32e-5
```

The improvement is several orders of magnitude, confirming that the leading visible error is the residue from the first critical-line zero.

## Interpretation

This supports the contour picture:

```text
F_X(s)
  = 1/zeta(s)
    + sum_zero_residues
    + smaller contour/trivial-zero/cutoff terms.
```

The tiny residual after subtraction is near the numerical and truncation floor for the current data. In particular, with `N=1000000` and smoothing up to `X=200000`, the omitted tail of

```text
sum_{n > N} mu(n)n^{-s}exp(-n/X)
```

can become visible once the dominant zero residue is removed.

So the meaningful conclusion is not the fine structure of the final tiny residual, but the large drop after subtracting the first zero-pair residue.

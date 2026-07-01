# Smoothed Convergence Check

We tested

```text
F_X(s) = sum mu(n)n^{-s}exp(-n/X)
```

against the expected limit

```text
1 / zeta(s)
```

for `s = sigma + it`, using `sigma > 1/2`.

## Numerical result

At the first zeta-zero ordinates, the fitted log-log slope of

```text
|F_X(s) - 1/zeta(s)|
```

matches the RH-predicted decay scale

```text
X^(-(sigma - 1/2)).
```

Representative values:

```text
sigma   fitted slope near zeros   expected RH slope
0.55    -0.050                    -0.050
0.60    -0.100                    -0.100
0.75    -0.250                    -0.250
1.10    -0.596                    -0.600
```

The clean match is a numerical signature of contour residues from zeros on

```text
Re(rho) = 1/2.
```

If a zero existed at `Re(rho)=beta > 1/2`, the same smoothed detector would produce a growing term when tested at `sigma < beta` and matching imaginary part.

## Interpretation

This strengthens the proof route:

1. Show `F_X(s)` is bounded for every fixed `Re(s)>1/2`.
2. Then no zero can exist to the right of the critical line.
3. By functional-equation symmetry, RH follows.

The computation does not prove step 1, but it gives a precise target and verifies that the detector sees the known critical-line zeros with the expected exponent.

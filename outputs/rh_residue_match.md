# Residue Match Check

For

```text
F_X(s) = sum mu(n)n^{-s}exp(-n/X)
```

Mellin inversion gives

```text
F_X(s) = (1 / 2 pi i) integral Gamma(w) X^w / zeta(s+w) dw.
```

After crossing `w=0`, the main term is

```text
1 / zeta(s).
```

If `rho` is a simple zero of zeta, then crossing

```text
w = rho - s
```

adds the residue

```text
Gamma(rho - s) X^(rho - s) / zeta'(rho).
```

For `s = sigma + i gamma_1` and `rho_1 = 1/2 + i gamma_1`, this predicts

```text
F_X(s) - 1/zeta(s)
  ~ Gamma(1/2 - sigma) X^(1/2 - sigma) / zeta'(rho_1).
```

So the scaled error

```text
(F_X(s) - 1/zeta(s)) X^(sigma - 1/2)
```

should approach

```text
Gamma(1/2 - sigma) / zeta'(rho_1).
```

## Numerical comparison

Using the generated `smoothed_convergence_1m.csv`, the final `X=200000` values give:

```text
sigma   |scaled error|   |predicted|   abs ratio
0.55    26.00924244      26.00869317  1.00002112
0.60    13.47353803      13.47304602  1.00003652
0.75    6.18017463       6.17991847   1.00004145
1.10    4.64101311       4.66101487   0.99570871
```

This is a strong numerical confirmation that the smoothed detector is seeing the actual contour residue from the first critical-line zero.

It is still not a proof of RH. It validates the detector and the proof-shaped implication:

```text
uniform boundedness of F_X(s) for Re(s)>1/2
=> no zeta zeros with Re(s)>1/2.
```

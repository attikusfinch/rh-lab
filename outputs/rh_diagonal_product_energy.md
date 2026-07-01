# Diagonal Product Energy

For the fourth moment, define

```text
B_Y(t)^2 = sum_r A_Y(r) r^(-it),
A_Y(r) = sum_{ab=r, Y<=a,b<2Y} mu(a)mu(b).
```

The infinite-time diagonal contribution is

```text
sum_r A_Y(r)^2.
```

This note studies that arithmetic quantity directly. It also compares it with
the unsigned product collision energy

```text
sum_r U_Y(r)^2,
U_Y(r) = #{(a,b): ab=r, Y<=a,b<2Y, mu(a)mu(b) != 0}.
```

The ratio

```text
sum A_Y(r)^2 / sum U_Y(r)^2
```

would measure sign cancellation on exact product collisions.

The scan reveals that this ratio is exactly `1` in these blocks. This is not an
accident: if `mu(a)mu(b)` is nonzero and `ab=r`, then the sign of `mu(a)mu(b)`
depends only on the prime factors of `r` counted with multiplicity. So all
nonzero representations of the same product have the same sign. Exact diagonal
collisions have no Mobius-sign cancellation.

## Current Scan

Command:

```powershell
python tools\plot_diagonal_product_energy.py outputs\mu_10m.i8 outputs\diagonal_product_energy_8k.png --output-csv outputs\diagonal_product_energy_8k.csv --max-y 8192
```

Parameters:

```text
dyadic Y: 128 through 8,192
exact ordered product pairs within each block
```

Results:

```text
Y     nonzero n  ordered pairs  unique products  pair baseline  diagonal
128   79         6,241          3,088            0.380920       0.796570
256   157        24,649         12,058           0.376114       0.796097
512   310        96,100         46,466           0.366592       0.790535
1024  621        385,641        184,601          0.367776       0.808248
2048  1,246      1,552,516      735,513          0.370149       0.831370
4096  2,491      6,205,081      2,917,017        0.369852       0.844722
8192  4,980      24,800,400     11,565,830       0.369555       0.858968
```

Here

```text
pair baseline = #{ordered nonzero pairs} / Y^2,
diagonal = sum_r A_Y(r)^2 / Y^2.
```

The collision inflation

```text
diagonal / pair baseline
```

grows slowly:

```text
Y     diagonal / pair baseline
128   2.09117
256   2.11664
512   2.15644
1024  2.19766
2048  2.24604
4096  2.28395
8192  2.32433
```

Coefficient-size contribution at `Y=8192`:

```text
|A(r)| = 1       0.0086%
|A(r)| = 2       75.3032%
|A(r)| = 3..4    16.9236%
|A(r)| = 5..8    6.6687%
|A(r)| = 9..16   0.9949%
|A(r)| >= 17     0.1009%
```

The largest observed coefficient is:

```text
max |A_Y(r)| = 24 at Y=8192.
```

## Lesson

The exact diagonal fourth-moment problem is a squarefree multiplicative-energy
problem, not a sign-cancellation problem. Mobius signs become important in the
off-diagonal sums, where different products interact through oscillatory
kernels.

A proof needs an upper bound for

```text
sum_r A_Y(r)^2
```

that grows like `O(Y^2 polylog(Y)^O(1))` or better in the unnormalized scale.
The finite scan is much stronger than this crude target: `sum_r A_Y(r)^2 / Y^2`
stays below `0.859` through `Y=8192`.

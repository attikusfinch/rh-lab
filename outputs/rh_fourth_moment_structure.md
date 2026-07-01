# Fourth Moment Structure

The moment route asks for bounds like

```text
E_t |B_Y(t) / sqrt(Y)|^(2k) <= C^k k!.
```

The first nontrivial case is the fourth moment:

```text
E_t |B_Y(t) / sqrt(Y)|^4.
```

Write

```text
B_Y(t)^2 = sum_r A_Y(r) r^(-it),
A_Y(r) = sum_{ab=r, Y<=a,b<2Y} mu(a)mu(b).
```

Then

```text
|B_Y(t)|^4 = |B_Y(t)^2|^2.
```

The infinite-time diagonal part is exactly

```text
sum_r A_Y(r)^2.
```

The finite-interval remainder comes from off-diagonal products `r != s`, weighted
by oscillatory kernels involving `log(r/s)`.

## Current scan

Command:

```powershell
python tools\plot_fourth_moment_structure.py outputs\mu_10m.i8 outputs\fourth_moment_structure_4k_t500.png --output-csv outputs\fourth_moment_structure_4k_t500.csv --max-y 4096 --t-max 500 --t-points 1601
```

Parameters:

```text
t range: [0, 500]
t grid points: 1,601
dyadic Y: 128 through 4,096
near-product windows: |log(r/s)| <= scale / T for scale 1, 2, 5, 10
```

Results:

```text
Y    nonzero n  unique products  E|g|^2   E|g|^4   2(E|g|^2)^2  4th/Gaussian
128  79         3,088            0.594897 0.645071 0.707805      0.911367
256  157        12,058           0.464722 0.370330 0.431932      0.857380
512  310        46,466           0.483341 0.382905 0.467237      0.819509
1024 621        184,601          0.512385 0.514788 0.525077      0.980404
2048 1,246      735,513          0.630607 0.739104 0.795331      0.929304
4096 2,491      2,917,017        0.689593 0.925208 0.951077      0.972801
```

The normalized exact product diagonal is very stable:

```text
Y    diagonal  diagonal/grid  offdiag/grid
128  0.796570  1.234857       -0.234857
256  0.796097  2.149695       -1.149695
512  0.790535  2.064572       -1.064572
1024 0.808248  1.570060       -0.570060
2048 0.831370  1.124835       -0.124835
4096 0.844722  0.913008        0.086992
```

Interpretation: the infinite-time diagonal alone is roughly `0.8`. On the
finite interval `[0,500]`, the off-diagonal kernel often contributes
negatively and cancels a large part of that diagonal for smaller blocks. By
`Y=4096`, the finite off-diagonal contribution is positive but still only about
`8.7%` of the grid fourth moment.

The near-product windows show strong sign cancellation. For example, in the
smallest window `|log(r/s)| <= 1/T`, the ratio

```text
abs(signed near off-diagonal mass) / unsigned near off-diagonal mass
```

falls rapidly:

```text
Y    cancellation ratio
128  5.56e-5
256  9.62e-4
512  1.52e-3
1024 1.94e-4
2048 6.35e-5
4096 2.10e-6
```

Even in the wider `10/T` window, the ratio at `Y=4096` is only `3.31e-6`.
So there are many near-collisions, but their Mobius signs almost completely
cancel in aggregate.

## Lesson

The fourth moment can be studied through product collisions `ab=cd` and
near-collisions `ab ~= cd`. This moves the project from numerical peak-hunting
toward the arithmetic combinatorics behind the moment bound.

The next proof-shaped target is to bound the diagonal quantity

```text
sum_r A_Y(r)^2
```

and then prove cancellation for the off-diagonal kernel.

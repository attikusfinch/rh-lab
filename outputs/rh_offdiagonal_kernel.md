# Off-Diagonal Kernel

For the fourth moment, after writing

```text
B_Y(t)^2 = sum_r A_Y(r) r^(-it),
```

the interval average is controlled by

```text
(1/T) integral_0^T |B_Y(t)^2|^2 dt
  = sum_{r,s} A_Y(r)A_Y(s) K_T(log(r/s)),
```

where

```text
K_T(u) = (1/T) integral_0^T exp(-itu) dt.
```

For paired terms `r,s` and `s,r`, the real kernel is

```text
Re K_T(u) = sin(Tu) / (Tu).
```

This note studies the off-diagonal part `r != s` by shells in

```text
T |log(r/s)|.
```

## Current Scan

Command:

```powershell
python tools\plot_offdiagonal_kernel.py outputs\mu_10m.i8 outputs\offdiagonal_kernel_8k_t500.png --output-csv outputs\offdiagonal_kernel_8k_t500.csv --max-y 8192 --t-max 500 --t-points 1601 --bin-count 65536
```

Parameters:

```text
t range: [0, 500]
t grid points: 1,601
dyadic Y: 128 through 8,192
log-product bins: 65,536
shells in T|log(r/s)|: [0,1), [1,2), [2,5), [5,10), [10,20), [20,50), [50,inf)
```

The computation uses exact product coefficients `A_Y(r)` and a log-binned
convolution approximation for the kernel sum.

The binned kernel integral matches the direct `t`-grid fourth moment very
closely:

```text
Y     grid E|g|^4  kernel E|g|^4  kernel-grid
128   0.645071     0.645618       0.000548
256   0.370330     0.370804       0.000474
512   0.382905     0.382753      -0.000152
1024  0.514788     0.514979       0.000191
2048  0.739104     0.739652       0.000548
4096  0.925208     0.925478       0.000270
8192  0.577621     0.577996       0.000375
```

The largest absolute discrepancy is only `0.000548`, so the shell decomposition
is tracking the same fourth moment as the direct grid scan.

Diagonal and total off-diagonal:

```text
Y     diagonal  offdiag kernel  offdiag cancellation ratio
128   0.796570  -0.150951       3.62e-3
256   0.796097  -0.425293       2.60e-3
512   0.790535  -0.407782       6.51e-4
1024  0.808248  -0.293269       1.16e-4
2048  0.831370  -0.091718       8.99e-6
4096  0.844722   0.080756       1.98e-6
8192  0.858968  -0.280972       1.73e-6
```

Here the cancellation ratio is

```text
abs(signed offdiag kernel total) / unsigned kernel mass.
```

It falls to about `1.7e-6` at `Y=8192`. This means the unsigned off-diagonal
kernel mass is enormous, but the signed oscillatory sum almost completely
cancels.

At `Y=8192`, the shell contributions are:

```text
shell T|log(r/s)|  signed contribution  cancellation ratio
0..1               0.0231432            6.84e-7
1..2              -0.301291             1.27e-5
2..5               0.0543963            2.78e-6
5..10             -0.0573608            3.95e-6
10..20             0.0133685            8.24e-7
20..50            -0.0143610            6.86e-7
50..inf            0.00113247           3.35e-8
```

The dominant signed contribution in this block comes from the `1..2` shell, but
even there the cancellation ratio is only `1.27e-5`.

## Lesson

The off-diagonal is the place where oscillation and sign structure can reduce
the fourth moment below the raw diagonal scale. The proof-shaped target is to
bound

```text
sum_{r != s} A_Y(r)A_Y(s) K_T(log(r/s))
```

uniformly in `Y` and `T`.

The next mathematical target is to replace the binned numerical cancellation by
an analytic estimate for these shells.

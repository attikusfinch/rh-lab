# Off-Diagonal Cancellation Sources

The off-diagonal kernel scan showed that

```text
sum_{r != s} A_Y(r)A_Y(s)K_T(log(r/s))
```

is tiny compared with the unsigned kernel mass. This note separates two possible
sources of cancellation:

```text
A_Y(r)A_Y(s) K_T(log(r/s))          actual signed kernel
A_Y(r)A_Y(s) |K_T(log(r/s))|        coefficient signs only
|A_Y(r)A_Y(s)| K_T(log(r/s))        kernel oscillation only
|A_Y(r)A_Y(s)| |K_T(log(r/s))|      unsigned reference
```

All ratios below are normalized by the same unsigned reference.

## Current Scan

Command:

```powershell
python tools\plot_offdiagonal_cancellation_sources.py outputs\mu_10m.i8 outputs\offdiagonal_cancellation_sources_8k_t500.png --output-csv outputs\offdiagonal_cancellation_sources_8k_t500.csv --max-y 8192 --t-max 500 --bin-count 65536
```

Parameters:

```text
T: 500
dyadic Y: 128 through 8,192
log-product bins: 65,536
```

Results:

```text
Y     actual       coeff signs only  kernel only   unsigned reference
128   -0.150951    -0.0116245        13.9911       41.6920
256   -0.425293    -0.266859         56.1502       163.658
512   -0.407782    -0.331784         216.223       626.064
1024  -0.293269    -0.400569         871.679       2,520.24
2048  -0.0917178   -0.00269691       3,530.67     10,204.5
4096   0.0807558   -0.0444279        14,091.2     40,725.6
8192  -0.280972    -0.212968         56,235.3     162,539
```

Cancellation ratios against the same unsigned reference:

```text
Y     actual       coeff signs only  kernel only
128   3.62e-3      2.79e-4          0.335584
256   2.60e-3      1.63e-3          0.343096
512   6.51e-4      5.30e-4          0.345367
1024  1.16e-4      1.59e-4          0.345873
2048  8.99e-6      2.64e-7          0.345996
4096  1.98e-6      1.09e-6          0.346022
8192  1.73e-6      1.31e-6          0.345980
```

The kernel by itself only reduces the unsigned mass to about `34.6%`. That is
nowhere near enough. The coefficient signs by themselves reduce the same mass
to around `10^-6` for the larger blocks, and the actual signed kernel sum is of
the same order.

At `Y=8192`:

```text
unsigned reference        = 162,539
kernel only               = 56,235.3
coefficient signs only    = -0.212968
actual signed kernel      = -0.280972
actual / coefficient-only = 1.31932
actual / kernel-only      = 4.996e-6
```

So the main proof target should be cancellation in the coefficient sequence
`A_Y(r)`, with kernel oscillation as a secondary weight, not the primary source
of decay.

## Lesson

This comparison tells us whether the off-diagonal proof should focus primarily
on Mobius coefficient signs, on oscillation of the kernel, or on their
interaction. In this range, the evidence points strongly to coefficient-sign
cancellation in the product sequence `A_Y(r)`.

# Kernel Cancellation Decay

This note connects the local autocorrelation candidate back to the
off-diagonal kernel.

For

```text
B_Y(t)^2 = sum_r A_Y(r) r^(-it),
```

the off-diagonal part of the fourth moment is controlled by

```text
sum_{r != s} A_Y(r) A_Y(s) K_T(log(r/s)).
```

The local autocorrelation scan proposed the empirical envelope

```text
epsilon(Y) <= 1600 / Y^2,
```

where `epsilon(Y)` is the worst observed local shell ratio over the tested
`T`-grid and shell family.

This scan checks whether the actual kernel-weighted off-diagonal cancellation
ratio is below that local envelope.

## Current Scan

Command:

```powershell
python tools\plot_kernel_cancellation_decay.py outputs\mu_10m.i8 outputs\kernel_cancellation_decay_16k_wide_t.png --output-csv outputs\kernel_cancellation_decay_16k_wide_t.csv --first 128 --max-y 16384 --t-scales 50,75,100,150,200,300,500,750,1000,1500,2000,3000,5000 --bin-count 65536 --aggregation incremental --pair-row-chunk-size 512 --merge-row-chunks 4 --epsilon-c 1600 --epsilon-alpha 2
```

Parameters:

```text
dyadic Y: 128 through 16,384
T scales: 50, 75, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000
log-product bins: 65,536
local envelope: 1600 / Y^2
```

Worst kernel-weighted cancellation ratio for each `Y`:

```text
Y      worst T  kernel ratio  local worst ratio  1600/Y^2    kernel / envelope
128    3000     7.621e-3      2.201e-2          9.766e-2    7.804e-2
256    500      2.599e-3      1.085e-2          2.441e-2    1.064e-1
512    1000     9.240e-4      3.509e-3          6.104e-3    1.514e-1
1024   2000     3.586e-4      1.036e-3          1.526e-3    2.350e-1
2048   3000     7.583e-5      1.180e-4          3.815e-4    1.988e-1
4096   5000     6.994e-6      2.385e-5          9.537e-5    7.334e-2
8192   500      1.729e-6      1.273e-5          2.384e-5    7.250e-2
16384  1000     8.615e-7      9.995e-7          5.960e-6    1.445e-1
```

The worst observed use of the envelope is at `Y=1024, T=2000`:

```text
kernel ratio / (1600/Y^2) = 0.23502.
```

So the checked kernel-weighted cancellation is below the proposed local
envelope with at least about a `4.25x` margin.

The kernel-ratio power-law fits are:

```text
all Y fit:        kernel ratio ~= 191.892 Y^-1.99908, R^2 = 0.981850
tail Y>=1024 fit: kernel ratio ~= 2225.45 Y^-2.28576, R^2 = 0.968431
```

Also, across all `104` checked `(Y,T)` points:

```text
kernel ratio <= local worst shell ratio.
```

The largest observed value of

```text
kernel ratio / local worst ratio
```

is

```text
0.861956 at Y=16384, T=1000.
```

## Lesson

The local autocorrelation envelope is not merely decorative. On the tested
range it controls the kernel-weighted off-diagonal cancellation, and the kernel
ratio itself decays almost exactly like `Y^-2` in the all-point fit.

The next proof-shaped target is to replace the empirical local envelope

```text
epsilon(Y) <= 1600 / Y^2
```

by an analytic estimate for the product-coefficient sequence `A_Y(r)`.

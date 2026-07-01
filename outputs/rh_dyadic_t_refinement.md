# Adaptive Dyadic T-Peak Refinement

The first dyadic `t` scan used a uniform grid. To reduce the chance that a narrow peak fell between grid points, this refinement pass:

1. runs the same coarse grid;
2. picks the top local maxima for each dyadic block;
3. repeatedly rescans smaller neighborhoods around those candidates.

This gives a better finite-grid approximation to

```text
sup_{0 <= t <= T} |B_Y(t)| / sqrt(Y).
```

## Current refinement

Parameters:

```text
N limit: 2,000,000
t range: [0, 200]
coarse grid points: 801
top local candidates per block: 5
initial refinement radius: 0.5
refinement levels: 4
points per refinement level: 61
```

The previous coarse scan found:

```text
Y = 65,536
t = 184.5
|B_Y(t)| / sqrt(Y) = 2.03946
```

The adaptive refinement found:

```text
Y = 65,536
t = 184.412133
|B_Y(t)| / sqrt(Y) = 2.03982
```

The largest relative increase over the coarse grid among all scanned dyadic blocks was only

```text
0.0316%
```

So the earlier grid did not miss a narrow, materially larger peak in this range.

The refined search should be read as a stronger adversarial check, not a proof of the true supremum over all real `t`.

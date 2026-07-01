# Lambda Support Null Models

The previous scan showed that `lambda(r)` on product support cancels much worse
than `lambda(r)` on the full interval. This note asks what causes the gap.

We compare four models:

```text
actual product support:
  lambda(r) on S_Y = {ab: Y<=a,b<2Y, mu(a)mu(b) != 0}

full interval:
  lambda(r) on [Y^2, 4Y^2)

shuffled support signs:
  same product-support locations, but the observed signs are randomly shuffled

random interval subset:
  random subset of [Y^2, 4Y^2) with the same density as S_Y, using true lambda(r)
```

## Current Scan

Command:

```powershell
python tools\plot_lambda_support_null_models.py outputs\mu_10m.i8 outputs\lambda_67m.i8 outputs\lambda_support_null_models_4k_wide_t.png --output-csv outputs\lambda_support_null_models_4k_wide_t.csv --first 128 --max-y 4096 --t-scales 50,75,100,150,200,300,500,750,1000,1500,2000,3000,5000 --bin-count 65536 --pair-row-chunk-size 512 --merge-row-chunks 4 --full-chunk-size 1000000 --replicates 5 --seed 20260702
```

The shuffled and random-subset columns are medians over `5` deterministic-seed
replicates.

```text
Y     actual     shuffled   random subset  full       actual/shuffled  actual/random  actual/full  density
128   2.879e-2   3.196e-2   2.508e-2       1.765e-3  0.901            1.148          16.313       0.06283
256   1.813e-2   6.192e-3   7.134e-3       6.194e-4  2.928            2.541          29.268       0.06133
512   4.403e-3   2.250e-3   2.405e-3       2.080e-4  1.957            1.830          21.173       0.05908
1024  1.264e-3   4.189e-4   5.167e-4       3.448e-5  3.017            2.446          36.653       0.05868
2048  2.997e-4   1.273e-4   1.338e-4       1.656e-5  2.353            2.239          18.090       0.05845
4096  3.829e-5   2.264e-5   4.575e-5       5.117e-6  1.691            0.837          7.483        0.05796
```

Power-law fits:

```text
actual product support:          alpha = 1.92364, R^2 = 0.972775
full interval:                   alpha = 1.72615, R^2 = 0.991157
same support, shuffled signs:    alpha = 2.04434, R^2 = 0.996188
random interval subset:          alpha = 1.85486, R^2 = 0.997770
```

Summary ratios:

```text
median actual/shuffled signs: 2.155
median actual/random subset:  2.035
median actual/full interval:  19.632
```

## Lesson

The huge gap between product support and the full interval is mostly a
support/sparsity effect, not purely a mysterious Liouville-sign effect.

The true product-support sequence is still worse than shuffled signs or a
random interval subset by a moderate factor, roughly `2x` on the middle blocks.
So the product-support geometry matters, and the true arithmetic signs are not
perfectly random relative to that geometry. But the extra arithmetic penalty is
small compared with the full-interval-to-support gap.

This suggests a sharper proof target:

```text
local lambda cancellation on S_Y
  ~= random sparse support scale
     times a bounded arithmetic-geometry penalty.
```

The next useful decomposition is to stratify product support by arithmetic
geometry, especially by `gcd(a,b)`, because a product

```text
r = ab
```

with squarefree `a,b` can be written as

```text
a = gu, b = gv, (u,v)=1, r = g^2uv.
```

That should expose which part of the support geometry causes the moderate
penalty over the null models.

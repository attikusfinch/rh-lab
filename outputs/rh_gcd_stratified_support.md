# GCD-Stratified Product Support

The null-model scan suggested that the remaining penalty comes from the
geometry of the product support

```text
S_Y = {ab: Y<=a,b<2Y, mu(a)mu(b) != 0}.
```

This note stratifies ordered squarefree pair representations by

```text
g = gcd(a,b).
```

For squarefree `a,b`, a fixed gcd layer has the form

```text
a = gu, b = gv, (u,v)=1, r = ab = g^2uv.
```

So high-gcd layers are thin square-multiple substructures inside the product
support.

## Current Scan

Command:

```powershell
python tools\plot_gcd_stratified_support.py outputs\mu_10m.i8 outputs\gcd_stratified_support_8k_wide_t.png --output-csv outputs\gcd_stratified_support_8k_wide_t.csv --first 128 --max-y 8192 --t-scales 50,75,100,150,200,300,500,750,1000,1500,2000,3000,5000 --bin-count 65536 --gcd-bounds 1,2,4,8,16,32,64,128,256,512,1024,inf --pair-row-chunk-size 512 --merge-row-chunks 4 --min-shell-unsigned-raw 1000
```

The scan computes product coefficients separately for each gcd bin:

```text
A_{Y,G}(r) = sum_{ab=r, gcd(a,b) in G} mu(a)mu(b).
```

It then applies the same local autocorrelation shell test to each stratum.
Very sparse shell measurements with unsigned raw mass below `1000` are skipped.

Overall local ratio fit:

```text
all gcd strata: alpha = 1.83290, R^2 = 0.976882
```

Selected stratum fits:

```text
gcd bin  alpha    R^2       last ratio at Y=8192
all      1.83290  0.976882  1.364e-5
1        1.82907  0.977398  2.310e-5
2-3      1.74825  0.970325  7.135e-5
4-7      1.59507  0.942345  2.955e-4
8-15     1.24619  0.855898  6.915e-4
16-31    1.03538  0.882862  2.128e-3
```

At the largest checked block:

```text
Y=8192

gcd bin    pair fraction   diagonal fraction   worst local ratio
all        1.000000        1.000000            1.364e-5
1          0.776056        0.792103            2.310e-5
2-3        0.148733        0.138838            7.135e-5
4-7        0.040790        0.037924            2.955e-4
8-15       0.015240        0.013937            6.915e-4
16-31      0.009478        0.008694            2.128e-3
32-63      0.004814        0.004308            2.618e-3
64-127     0.002408        0.002128            4.942e-3
128-255    0.001234        0.001079            8.641e-3
256-511    0.000573        0.000497            3.364e-2
512-1023   0.000285        0.000245            8.248e-2
1024+      0.000389        0.000249            5.312e-1
```

The high-gcd ratios are large, but those layers have tiny mass. A mass-weighted
view at `Y=8192` puts the largest high-gcd layer in perspective:

```text
gcd bin    pair fraction   worst ratio   pair fraction * worst ratio
1024+      3.893e-4        5.312e-1      2.068e-4
512-1023   2.849e-4        8.248e-2      2.350e-5
16-31      9.478e-3        2.128e-3      2.017e-5
256-511    5.735e-4        3.364e-2      1.929e-5
1          7.761e-1        2.310e-5      1.792e-5
```

## Lesson

The dominant layer is the coprime layer `gcd(a,b)=1`. At `Y=8192` it carries
about `77.6%` of ordered pair mass and `79.2%` of diagonal energy, while its
local ratio is close to the total ratio.

The higher gcd layers are progressively thinner and less mixed. Their raw local
ratios can be much worse, but the pair and diagonal fractions collapse quickly.

So the proof-shaped target should split as:

```text
main term:  gcd(a,b)=1, equivalently r=uv with (u,v)=1
tail:       sum over g>=2 of square-multiple layers r=g^2uv
```

The next analytic move is to formulate a coprime product-support cancellation
lemma and a separate summable tail estimate over gcd strata. This is more
structured than treating the product support as one undifferentiated sparse set.

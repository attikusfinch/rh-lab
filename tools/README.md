# RH Visualization Lab

Fast C++ sampler for the Chebyshev function

```text
psi(x) = sum_{p^k <= x} log(p)
```

and the RH-scale error normalizations:

```text
psi(x) - x
(psi(x) - x) / (sqrt(x) log(x))
(psi(x) - x) / (sqrt(x) log(x)^2)
```

Build:

```powershell
g++ -O3 -std=c++17 work\rh_viz\rh_viz.cpp -o work\rh_viz\rh_viz.exe
```

Run:

```powershell
work\rh_viz\rh_viz.exe 10000000 50000 outputs\rh_viz_10m.csv
python work\rh_viz\plot_rh_viz.py outputs\rh_viz_10m.csv outputs\rh_viz_10m.png
```

Envelope scan:

```powershell
g++ -O3 -std=c++17 work\rh_viz\rh_envelope.cpp -o work\rh_viz\rh_envelope.exe
work\rh_viz\rh_envelope.exe 100000000 2000 outputs\rh_envelope_100m.csv
python work\rh_viz\plot_envelope.py outputs\rh_envelope_100m.csv outputs\rh_envelope_100m.png
```

Zero-wave overlay:

```powershell
python work\rh_viz\plot_zero_wave.py outputs\rh_viz_10m.csv outputs\rh_zero_wave_10m.png --zeros 80 --points 3000
```

Mobius/Mertens scan:

```powershell
g++ -O3 -std=c++17 work\rh_viz\mobius_viz.cpp -o work\rh_viz\mobius_viz.exe
work\rh_viz\mobius_viz.exe 100000000 100000 outputs\mobius_100m.csv
python work\rh_viz\plot_mobius.py outputs\mobius_100m.csv outputs\mobius_100m.png
```

Mobius autocorrelation:

```powershell
g++ -O3 -std=c++17 work\rh_viz\mobius_dump.cpp -o work\rh_viz\mobius_dump.exe
work\rh_viz\mobius_dump.exe 10000000 outputs\mu_10m.i8
python work\rh_viz\plot_mobius_corr.py outputs\mu_10m.i8 outputs\mobius_corr_10m_50k.png --max-lag 50000 --output-csv outputs\mobius_corr_10m_50k.csv
python work\rh_viz\plot_mobius_corr_prefix.py outputs\mu_10m.i8 outputs\mobius_corr_prefix_10m.png --output-csv outputs\mobius_corr_prefix_10m.csv
python work\rh_viz\plot_mobius_spectrum.py outputs\mu_10m.i8 outputs\mobius_spectrum_10m.png --output-csv outputs\mobius_spectrum_10m_top.csv
python work\rh_viz\plot_mobius_dirichlet.py outputs\mu_10m.i8 outputs\mobius_dirichlet_500k.png --limit 500000 --output-csv outputs\mobius_dirichlet_500k.csv
python work\rh_viz\plot_dirichlet_growth.py outputs\mu_10m.i8 outputs\dirichlet_growth_10m.png --output-csv outputs\dirichlet_growth_10m.csv
g++ -O3 -std=c++17 work\rh_viz\primes_dump.cpp -o work\rh_viz\primes_dump.exe
work\rh_viz\primes_dump.exe 10000000 outputs\primes_10m.u32
python work\rh_viz\plot_prime_sum_growth.py outputs\primes_10m.u32 outputs\prime_sum_growth_10m.png --output-csv outputs\prime_sum_growth_10m.csv
python work\rh_viz\plot_smoothed_mobius.py outputs\mu_10m.i8 outputs\smoothed_mobius_1m.png --output-csv outputs\smoothed_mobius_1m.csv
```

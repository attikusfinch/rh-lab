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
g++ -O3 -std=c++17 tools\rh_viz.cpp -o tools\rh_viz.exe
```

Run:

```powershell
tools\rh_viz.exe 10000000 50000 outputs\rh_viz_10m.csv
python tools\plot_rh_viz.py outputs\rh_viz_10m.csv outputs\rh_viz_10m.png
```

Envelope scan:

```powershell
g++ -O3 -std=c++17 tools\rh_envelope.cpp -o tools\rh_envelope.exe
tools\rh_envelope.exe 100000000 2000 outputs\rh_envelope_100m.csv
python tools\plot_envelope.py outputs\rh_envelope_100m.csv outputs\rh_envelope_100m.png
```

Zero-wave overlay:

```powershell
python tools\plot_zero_wave.py outputs\rh_viz_10m.csv outputs\rh_zero_wave_10m.png --zeros 80 --points 3000
```

Mobius/Mertens scan:

```powershell
g++ -O3 -std=c++17 tools\mobius_viz.cpp -o tools\mobius_viz.exe
tools\mobius_viz.exe 100000000 100000 outputs\mobius_100m.csv
python tools\plot_mobius.py outputs\mobius_100m.csv outputs\mobius_100m.png
```

Mobius autocorrelation:

```powershell
g++ -O3 -std=c++17 tools\mobius_dump.cpp -o tools\mobius_dump.exe
tools\mobius_dump.exe 10000000 outputs\mu_10m.i8
python tools\plot_mobius_corr.py outputs\mu_10m.i8 outputs\mobius_corr_10m_50k.png --max-lag 50000 --output-csv outputs\mobius_corr_10m_50k.csv
python tools\plot_mobius_corr_prefix.py outputs\mu_10m.i8 outputs\mobius_corr_prefix_10m.png --output-csv outputs\mobius_corr_prefix_10m.csv
python tools\plot_mobius_spectrum.py outputs\mu_10m.i8 outputs\mobius_spectrum_10m.png --output-csv outputs\mobius_spectrum_10m_top.csv
python tools\plot_mobius_dirichlet.py outputs\mu_10m.i8 outputs\mobius_dirichlet_500k.png --limit 500000 --output-csv outputs\mobius_dirichlet_500k.csv
python tools\plot_dirichlet_growth.py outputs\mu_10m.i8 outputs\dirichlet_growth_10m.png --output-csv outputs\dirichlet_growth_10m.csv
g++ -O3 -std=c++17 tools\primes_dump.cpp -o tools\primes_dump.exe
tools\primes_dump.exe 10000000 outputs\primes_10m.u32
python tools\plot_prime_sum_growth.py outputs\primes_10m.u32 outputs\prime_sum_growth_10m.png --output-csv outputs\prime_sum_growth_10m.csv
python tools\plot_smoothed_mobius.py outputs\mu_10m.i8 outputs\smoothed_mobius_1m.png --output-csv outputs\smoothed_mobius_1m.csv
python tools\plot_smoothed_convergence.py outputs\mu_10m.i8 outputs\smoothed_convergence_1m.png --output-csv outputs\smoothed_convergence_1m.csv
python tools\check_residue_match.py outputs\smoothed_convergence_1m.csv
python tools\plot_residue_subtraction.py outputs\smoothed_convergence_1m.csv outputs\residue_subtraction_1m.png --output-csv outputs\residue_subtraction_1m.csv
python tools\plot_dyadic_mobius_blocks.py outputs\mu_10m.i8 outputs\dyadic_mobius_blocks_10m.png --output-csv outputs\dyadic_mobius_blocks_10m.csv
python tools\plot_dyadic_t_scan.py outputs\mu_10m.i8 outputs\dyadic_t_scan_2m.png --output-csv outputs\dyadic_t_scan_2m.csv --limit 2000000 --t-max 200 --t-points 801
python tools\plot_dyadic_t_refine.py outputs\mu_10m.i8 outputs\dyadic_t_refine_2m.png --output-csv outputs\dyadic_t_refine_2m.csv --limit 2000000 --t-max 200 --t-points 801
python tools\plot_dyadic_large_values.py outputs\mu_10m.i8 outputs\dyadic_large_values_2m.png --output-csv outputs\dyadic_large_values_2m.csv --limit 2000000 --t-max 200 --t-points 801
python tools\plot_dyadic_smoothness.py outputs\mu_10m.i8 outputs\dyadic_smoothness_2m.png --output-csv outputs\dyadic_smoothness_2m.csv --limit 2000000 --t-max 200 --t-points 801
python tools\plot_dyadic_moments.py outputs\mu_10m.i8 outputs\dyadic_moments_2m.png --output-csv outputs\dyadic_moments_2m.csv --limit 2000000 --t-max 200 --t-points 801
python tools\plot_dyadic_moments.py outputs\mu_10m.i8 outputs\dyadic_moments_5m_t500_full.png --output-csv outputs\dyadic_moments_5m_t500_full.csv --limit 5000000 --t-max 500 --t-points 1601 --moments 1,2,3,4,5,6,7,8,9,10 --full-blocks-only
python tools\plot_dyadic_uniform_bounds.py outputs\mu_10m.i8 outputs\dyadic_uniform_bounds_5m_t500.png --output-csv outputs\dyadic_uniform_bounds_5m_t500.csv --limit 5000000 --t-max 500 --t-points 1601 --moments 1,2,3,4,5,6,7,8,9,10 --full-blocks-only
python tools\plot_fourth_moment_structure.py outputs\mu_10m.i8 outputs\fourth_moment_structure_4k_t500.png --output-csv outputs\fourth_moment_structure_4k_t500.csv --max-y 4096 --t-max 500 --t-points 1601
python tools\plot_diagonal_product_energy.py outputs\mu_10m.i8 outputs\diagonal_product_energy_8k.png --output-csv outputs\diagonal_product_energy_8k.csv --max-y 8192
python tools\plot_offdiagonal_kernel.py outputs\mu_10m.i8 outputs\offdiagonal_kernel_8k_t500.png --output-csv outputs\offdiagonal_kernel_8k_t500.csv --max-y 8192 --t-max 500 --t-points 1601 --bin-count 65536
python tools\plot_offdiagonal_cancellation_sources.py outputs\mu_10m.i8 outputs\offdiagonal_cancellation_sources_8k_t500.png --output-csv outputs\offdiagonal_cancellation_sources_8k_t500.csv --max-y 8192 --t-max 500 --bin-count 65536
python tools\plot_product_coefficient_autocorr.py outputs\mu_10m.i8 outputs\product_coefficient_autocorr_8k_t500.png --output-csv outputs\product_coefficient_autocorr_8k_t500.csv --max-y 8192 --t-scale 500 --bin-count 65536
```

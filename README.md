# RH Visualization Lab

Small experimental lab for exploring the Riemann hypothesis through:

- Chebyshev psi error: `psi(x) - x`
- Mobius/Mertens sums: `M(x) = sum_{n <= x} mu(n)`
- Mobius autocorrelations
- Mobius Dirichlet sums near zeta-zero ordinates
- Smoothed Mobius transforms as a zero detector
- Smoothed convergence checks against `1 / zeta(s)`
- Dyadic twisted Mobius block cancellation checks
- Grid scans for worst-case dyadic resonance over `t`
- Adaptive peak refinement for dyadic resonance scans
- Mean-square and large-values scans for dyadic blocks
- Derivative and Sobolev smoothness checks for dyadic blocks
- High-moment checks for dyadic large-values bounds
- Uniform-in-Y moment bounds for dyadic blocks
- Fourth-moment product-collision structure
- Diagonal product-energy decomposition
- Off-diagonal kernel shell decomposition
- Off-diagonal cancellation source separation
- Product-coefficient autocorrelation checks
- Local autocorrelation bound scans across T scales
- Decay fits for the local autocorrelation lemma candidate
- Kernel-weighted off-diagonal cancellation decay checks

This is not a proof of RH. It is a computational notebook-style project that turns the visual intuition into proof-shaped targets.

## Layout

```text
tools/    C++ counters and Python plotting scripts
outputs/  selected generated plots and proof-route notes
```

Generated CSV, binary dumps, and compiled executables are intentionally ignored by git.

## Setup

```powershell
pip install -r requirements.txt
```

Build the C++ tools as needed:

```powershell
g++ -O3 -std=c++17 tools\rh_viz.cpp -o tools\rh_viz.exe
g++ -O3 -std=c++17 tools\mobius_viz.cpp -o tools\mobius_viz.exe
g++ -O3 -std=c++17 tools\mobius_dump.cpp -o tools\mobius_dump.exe
g++ -O3 -std=c++17 tools\primes_dump.cpp -o tools\primes_dump.exe
```

Example run:

```powershell
tools\rh_viz.exe 10000000 50000 outputs\rh_viz_10m.csv
python tools\plot_rh_viz.py outputs\rh_viz_10m.csv outputs\rh_viz_10m.png
```

See `tools/README.md`, `outputs/rh_proof_route.md`, `outputs/rh_smoothed_convergence.md`, `outputs/rh_residue_match.md`, `outputs/rh_residue_subtraction.md`, `outputs/rh_contour_remainder.md`, `outputs/rh_dyadic_cancellation_target.md`, `outputs/rh_dyadic_t_scan.md`, `outputs/rh_dyadic_t_refinement.md`, `outputs/rh_dyadic_large_values.md`, `outputs/rh_dyadic_smoothness.md`, `outputs/rh_dyadic_moments.md`, `outputs/rh_uniform_moment_bound.md`, `outputs/rh_fourth_moment_structure.md`, `outputs/rh_diagonal_product_energy.md`, `outputs/rh_offdiagonal_kernel.md`, `outputs/rh_offdiagonal_cancellation_sources.md`, `outputs/rh_product_coefficient_autocorr.md`, `outputs/rh_local_autocorr_bound.md`, `outputs/rh_local_autocorr_decay.md`, and `outputs/rh_kernel_cancellation_decay.md` for the full route.

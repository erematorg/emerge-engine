# Energy

How it flows and transforms — thermal, generic reaction-diffusion, acoustic, and electromagnetic energy.

## Core API

- `thermodynamics` — `ThermalDiffusion` (Fourier finite-difference heat), `ScalarDiffusionField` (generic reaction-diffusion: pheromone, nutrients, morphogen — caller supplies read/write closures and an optional `source` term for Gray-Scott/Turing-style reactions), `GranularFluidityField` (Kamrin & Koval 2012 Nonlocal Granular Fluidity), `CosseratField` (grid-level micro-rotation/angular-momentum channel, de Borst/Sabet/Hageman 2022).
- `acoustics` [feature = "experimental"] — `WaveEquation2D`, finite-difference pressure-wave propagation, modal synthesis helpers.
- `electromagnetics` [feature = "experimental"] — `ElectromagneticWave` propagation, optical `MaterialProperties` (refractive index, permittivity/permeability). The point-charge force-application half lives in `forces::electromagnetics` instead.

## Scope & Limits

- `ThermalDiffusion` and `ScalarDiffusionField` document their own CFL stability bound (`dt ≤ dx²/4α`) but do not runtime-enforce it — a known, disclosed gap, not silently worked around. `GranularFluidityField`'s bound is enforced through the adaptive substep chooser instead, since it's plausibly binding rather than "never the bottleneck."
- `acoustics`/`electromagnetics` are feature-gated experimental, not part of the LP-stable API.

## Status

Thermal diffusion and scalar fields are production-stable. `GranularFluidityField`/`CosseratField` are recent (2026-07/08), part of the active sand-repose-angle research thread. Acoustics/EM remain experimental.

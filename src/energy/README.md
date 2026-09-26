# Energy

**Heat and light: how energy moves and what it does when it arrives.**

Heat spreads by conduction, following Fourier's law. Hot matter glows,
following Planck's law, and how brightly follows Stefan-Boltzmann. Light
travelling through a substance is absorbed following Beer-Lambert, bounces off
surfaces following Fresnel, and spreads inside a scattering medium following
the photon diffusion equation.

All of that is here, in SI units, with its sources named. The renderer uses
it; it does not own any of it. A colour computed from a spectrum is a property
of the light, not a decision of the drawing code.

## Core API

- `thermodynamics` -- `ThermalDiffusion` (Fourier finite-difference heat), `ScalarDiffusionField` (generic reaction-diffusion: pheromone, nutrients, morphogen -- caller supplies read/write closures and an optional `source` term for Gray-Scott/Turing-style reactions), `GranularFluidityField` (Kamrin & Koval 2012 Nonlocal Granular Fluidity), `CosseratField` (grid-level micro-rotation/angular-momentum channel, de Borst/Sabet/Hageman 2022).
- `radiation` -- `blackbody` (Planck's law, Wien, Lambertian blackbody radiance), `attenuation` (Beer-Lambert plus the full three-term slab transfer: Fresnel reflection, extinction, single scattering), `fresnel` (reflectance at a dielectric interface, Schlick's angular form), `spectrum` (CIE 1931 standard observer, Planckian locus, XYZ to linear sRGB, and the band averaging that collapses any measured spectrum onto three channels). SI throughout, cross-checked against CIE illuminant A and against Stefan-Boltzmann recovered from an independent integration of Planck.
- `thermodynamics::frictional_heating` -- dissipated work becoming temperature, `dT = de / c_p`, so a rubbing contact conserves energy rather than losing it.
- `acoustics` [feature = "experimental"] -- `WaveEquation2D`, finite-difference pressure-wave propagation, modal synthesis helpers.
- `electromagnetics` [feature = "experimental"] -- `ElectromagneticWave` propagation, optical `MaterialProperties` (refractive index, permittivity/permeability). The point-charge force-application half lives in `forces::electromagnetics` instead.

## Scope & Limits

- `ThermalDiffusion` and `ScalarDiffusionField` document their own CFL stability bound (`dt ≤ dx²/4α`) but do not runtime-enforce it -- a known, disclosed gap, not silently worked around. `GranularFluidityField`'s bound is enforced through the adaptive substep chooser instead, since it's plausibly binding rather than "never the bottleneck."
- `acoustics`/`electromagnetics` are feature-gated experimental, not part of the LP-stable API.

## Status

Thermal diffusion and scalar fields are production-stable. `GranularFluidityField`/`CosseratField` are recent (2026-07/08), part of the active sand-repose-angle research thread. Acoustics/EM remain experimental.

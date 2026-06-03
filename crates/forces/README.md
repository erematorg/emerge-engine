# Forces

Gravitational mechanics and Newton's laws for physics-based force systems.

## Core Principles

- Uses SI-style units (meters, seconds, Newtons) for mass/force/velocity.
- Applies forces and integrates velocities explicitly; no global momentum/energy reconciliation yet.
- Gravity supports uniform fields and n-body mutual gravity, with configurable softening.
- Integration uses variable `Time.delta_secs()` (no fixed physics tick yet); dual-clock (physics vs diurnal/biology) is not implemented.

## Scope & Limits

- LP-0 integrates F = ma with explicit/symplectic Euler (1st order) and optional acceleration clamps for stability; this is a numerical method, not a physical law.
- Gravity defaults to a sim-tuned constant and softened inverse-square forces (Plummer softening: F = GMm·r/(r²+ε²)^1.5).
- Mutual gravity mode is exact pairwise O(N²); treat ~100 active sources as the LP-0 realtime comfort range.
- Linear momentum is computable but **not enforced globally**. Angular momentum and mass conservation are **not yet tracked**.
- **Contact forces, friction, elasticity, plasticity, viscosity**: Material behaviors, deferred to matter/MPM coupling.
- Potential energy and work accounting are partial; conservation diagnostics incomplete.

## Status

**FROZEN** — no new features. Validated for N-body + Coulomb at N~100.
Velocity Verlet (2nd order) is implemented; energy drift ~0.01% over long orbits.
Coulomb singularity handled with softened formula; default multiplier 0.0 (disabled).

Migration path: N-body gravity will eventually be re-implemented as a `ForceField` impl
wiring into [emerge](https://github.com/erematorg/emerge) (the LP physics engine).
emerge is currently a local path dependency — not published. Forces crate stays until emerge is stable.

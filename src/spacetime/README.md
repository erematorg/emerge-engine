# Spacetime

The arena: the discretized space+time domain balance laws (mass/momentum conservation) are solved over. In the numerical-PDE sense of "space-time discretization" (established term, e.g. space-time finite element methods), not literal GR geometry — this engine is flat, non-relativistic.

## Core API

- `grid` — `Grid`/`Cell`, the sparse Eulerian background grid + quadratic B-spline kernel. `ContactCell`/`ContactCellMap` (multi-field frictional contact, Bardenhagen 2001 + Nairn/Hammerquist/Smith 2020), `MixtureCell`/`MixtureCellMap` (N-phase mixture coupling) each add their own `impl Grid` block from a sibling file.
- `solver` — `Simulation`, `SimConfig`, `SpawnRegion`: owns the substep loop (P2G → grid update → G2P), adaptive CFL-bound timestep, spatial hash, phase rules, body-state queries (`body_state.rs`, formerly `query.rs`).
- `transfer` — P2G/G2P kernels (MLS-APIC), the actual particle↔grid bridge every substep. CPU P2G is thread-local-fold + reduce parallelized (rayon); contact/mixture scatter paths stay serial.
- `diff` — a separate forward+reverse MLS-MPM implementation for gradient-based offline controller training; hand-derived adjoints, cross-checked against finite differences.
- `rod` — 1D discrete elastic rod sub-solver (Cosserat-rod family, Bergou et al. 2008) for slender bodies, coupled through the same shared `Grid` ordinary particles use. `RodMaterial` (EA/EI stiffness) stays here rather than in `matter` — its own modal-damping methods reach directly into `RodPoints`, a genuine solver-side coupling, not a portable constitutive law.
- `grains` — discrete-element grain **dynamics**: `population` (`GrainPopulation` container + its own semi-implicit-Euler step), `coupling` (grid scatter/gather, mirrors `rod::coupling`), `oracle` (packing-fraction signal, Yue/Smith/Chen/Kamrin/Grinspun 2018 "Hybrid Grains"). `Grain` (state) and the contact force law live in `matter` — this directory only holds how grains move and couple, not what a grain is.

## Scope & Limits

- Rod and grains are CPU-only for now — GPU port explicitly deferred per this project's "CPU correctness first" rule.
- `diff`'s adjoint chain covers the main MPM path only, not rod/grains.
- Multi-field contact and mixture coupling are opt-in (zero-cost when unused): a particle only enters a second velocity field via `contact_group`, or a mixture phase via `MixturePhase`.

## Status

Solver core stable and heavily tested (`tests/accuracy.rs`, `tests/physics_correctness.rs`). Rod and grains are newer (2026-08) — grains in particular is mid-development: packing-fraction oracle exists as a signal only, doesn't yet dynamically spawn/despawn grains. GPU sparse-grid active-block dispatch shipped for the main particle path.

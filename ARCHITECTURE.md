# emerge — Architecture

This document explains how emerge *works*, not where each file lives (for that,
see the module tree in `src/lib.rs`). It's for someone about to add a material,
a force field, or a GPU pass and wanting to know what plugs in where.

emerge is a real-time **MLS-MPM** (Moving Least Squares Material Point Method)
continuum solver. One unified algorithm simulates fluids, granular media,
elastic solids, and active (muscle-driven) matter — the difference between water
and rock is entirely which *constitutive model* a particle carries, never a
different code path.

---

## 1. The one loop that matters

Everything is a particle carrying physical state. Particles never interact
directly — they exchange momentum through a background grid, once per substep,
in a fixed three-phase cycle:

```
   Particle-to-Grid          Grid Update            Grid-to-Particle
   (P2G, scatter)      →     (forces, BCs)     →    (G2P, gather)
   mass + momentum          v += dt·(gravity        interpolate v, C
   + internal stress        + stress div),          back to particles,
   splatted to cells        boundary conditions,    advect positions
                            velocity clamp
```

The grid is **scratch**: cleared at the start of every substep, rebuilt from
particles, used to advance velocities, read back, discarded. Particles are the
only persistent state. This is why the grid can be a dense array with no
long-term memory cost — it holds data for microseconds.

The real per-substep order (`Simulation::do_substep`, `src/solver/mod.rs`) is
slightly richer than the three-phase cartoon, and the order is load-bearing:

1. **Project invalid state** — clamp any particle with a degenerate deformation
   gradient (J→0, NaN) *before* it scatters. Running this pre-P2G means a bad
   particle from the previous substep is fixed before its momentum can poison
   the grid — no NaN cascade.
2. **Density recompute** — fluid EOS materials need current ρ each substep
   (pressure is a function of density). Auto-enabled when any registered
   material declares `needs_density_recompute`.
3. **P2G** — clear grid, scatter mass + momentum + internal (Kirchhoff) stress
   using quadratic B-spline weights.
4. **Wake pass** — sleeping particles whose kernel overlaps a now-active cell
   are woken (see §6).
5. **Grid update** — `v += dt·(gravity + stress divergence)`, then boundary
   conditions, then a grid-velocity clamp. The clamp is *before* G2P
   deliberately: it bounds both particle velocity and the APIC affine matrix C
   at the source; clamping after G2P would miss C, and a large C makes
   `F = (I + dt·C)·F` blow up.
6. **G2P** — gather velocity and C back to particles (MLS-APIC), advect
   positions, with a per-particle velocity clamp enforcing the CFL contract.
7. **Force fields** — external body forces (`v += dt·a`) applied *after* G2P so
   each field sees fully-gathered state, then re-clamped so a large impulse
   can't enter the next P2G above one cell per substep.

---

## 2. The single source of truth: `Particle`

`Particle` (`src/particle.rs`) is `repr(C)`, exactly 112 bytes, compile-time
asserted, and directly GPU-uploadable with no translation layer. This byte
layout is a hard contract — the WGSL shaders read the same struct.

The consequence for design: **derive, don't store.** If a quantity can be
computed from `x`, `v`, `F`, `C`, mass, and volume, it does not get its own
field. Fields that *do* exist are either irreducible physical state
(deformation gradient, temperature) or per-material scratch that the constitutive
model owns (e.g. `friction_hardening` is reused as Drucker-Prager `q`,
Von Mises `κ`, Rankine damage, or µ(I) — one particle runs one material, so one
field safely serves all of them). Before adding a field, prove it can't be
derived.

---

## 3. How a material plugs in

A material is a `MaterialModel` (`src/materials/`): given a particle's
deformation gradient and scratch state, it returns Kirchhoff stress for P2G and
optionally mutates plastic state (return-mapping, hardening). Elasticity and
plasticity are separable — `ConstitutiveModel` (elastic response) and
`PlasticityModel` (return mapping) compose, so Von Mises plasticity can wrap a
Neo-Hookean elastic core without either knowing about the other.

Materials are registered in a `MaterialRegistry` and referenced by
`Particle::material_id` (a slot index, not a pointer — GPU-friendly). Adding a
material means: implement the trait, register it, done. No solver changes. The
10 shipped models (Neo-Hookean, Corotated, Newtonian/Bingham fluid, Snow, Sand
DP, µ(I), Von Mises, Rankine, Viscoelastic) are all just different
implementations behind this one seam.

**Physical grounding is a project rule, not a nicety.** Presets take real SI
inputs (`from_young_modulus`, `from_si`) and carry literature citations for
their constants. See `CLAUDE.md` for the full rule; the short version is that a
constant without a real-world source doesn't ship.

---

## 4. Why `step()` is not one update

`Simulation::step()` always advances exactly `config.dt` of simulation time —
but it does so in as many **adaptive substeps** as CFL stability requires. Stiff
materials and high velocities force smaller substeps; a calm scene may take one.
`choose_substep_dt` picks the largest CFL-safe `dt ≤ remaining`, bounded by both
an advection limit (nothing moves more than a fraction of a cell) and a
per-material timestep bound. This is real physics, not a tuning knob: a
high-velocity impact *correctly* demands finer timesteps for those frames.

Callers see a clean "advance one frame" API; the substep loop is internal. The
spatial hash for neighbor queries is rebuilt once per `step()` (after the loop),
because LP queries happen between frames, never mid-substep.

---

## 5. CPU-first, GPU-mirrors

There are two backends running the *same* algorithm:

- **`Simulation`** (CPU, always available) — the reference implementation and
  correctness ground truth. SVD, return-mapping, and every material live here in
  readable Rust.
- **`GpuSimulation`** (`src/gpu/`, `feature = "gpu"`) — an 11-pass WGSL pipeline
  reading the identical 112-byte `Particle` layout. Five passes run once per
  frame (a block-level counting sort that also builds the GPU sparse-grid
  active-block list — `particle_sort.wgsl`); six run per substep and mirror the
  CPU cycle: `grid_clear → p2g → grid_update → g2p → particles_update → force_fields`.
  The `particles_update` pass is where F-update, plasticity/return-mapping, and
  volume/density/position advection happen (the CPU splits these across G2P and
  the projection step); muscle active stress is folded into `p2g`, not a separate
  pass.

The rule (`CLAUDE.md`): **CPU correctness first, GPU port second.** SVD and
return-mapping are hard to get right in WGSL, so the CPU version defines the
answer and the GPU version is validated against it (see the brute-force-vs-hash
and CPU-vs-GPU parity tests in `tests/`). GPU is not automatically faster for
every scene — correctness on real hardware can require CPU↔GPU sync between
substep batches; its value is throughput at high particle counts, not a free win.

Shared CFL logic (`choose_substep_dt`, `cfl_bound`) is `pub(crate)` precisely so
the GPU solver reuses it instead of duplicating the stability math.

---

## 6. Sleep/wake partition

Particles are partitioned in place: `particles[0..active_count]` are active,
`[active_count..]` sleeping. P2G/G2P only visit the active partition, so a world
with mostly-settled terrain costs roughly what its *moving* region costs. Waking
is local — during P2G, a sleeping particle whose quadratic kernel overlaps an
active cell is woken, propagating activity outward without an O(N) scan.

This is **Phase 1**: flag-based partitioning, no memory compaction. Dormant
particles still occupy their slots. Real compaction (Phase 2) is the open
scale-lever for very large worlds; granular terrain specifically benefits less
because of q-creep (settled sand keeps micro-adjusting). See the memory notes on
sleep/wake for the full trade-off analysis.

---

## 7. Extension seams, at a glance

Everything pluggable is a trait object held by `Simulation`, applied
automatically each substep — LP configures, never reimplements:

| Seam | Trait | Applied | Examples |
|---|---|---|---|
| Constitutive response | `MaterialModel` / `ConstitutiveModel` + `PlasticityModel` | P2G stress | 10 material models |
| External body forces | `Field` | after G2P | gravity, Coulomb, EM, Barnes-Hut N-body, confinement |
| Grid boundaries | `BoundaryCondition` | grid update | slip, predictive, friction, ratchet (directional grip), grip |
| Scalar transport | `ScalarDiffusionField` | per substep | heat, pheromone, nutrients, morphogen (reaction-diffusion ready) |
| Phase change | phase rules (`Fn(&Particle) -> Option<u32>`) | per substep | melting, freezing, evaporation |
| Observation | `DiagnosticsRegistry` plugins | per step | health, per-material stats, rolling history |

Adding capability means implementing the relevant trait and attaching it — the
substep loop already knows when to call it.

`control::Lnn` (a Liquid Time-constant Network CPG, Hasani et al. 2020) is a
deliberate exception: it does not participate in the substep loop at all. It's
a standalone ODE integrated by the caller (LP), which writes its output
directly into `Particle::activation`/`activation_dir` between steps. emerge
supplies the controller math; it has no opinion on when or whether it runs.

---

## 8. Feature gating

- **Core** (no features) — solver, materials, fields, thermodynamics, control,
  diagnostics. Stable public API. Pure Rust, no Bevy, no game concepts.
- **`gpu`** — the WGSL compute backend. Opt-in.
- **`render`** — instanced debug renderer with physics-driven color
  (Beer-Lambert absorption, subsurface scattering, Fresnel specular). Requires
  `gpu`.
- **`experimental`** — acoustics, electromagnetics, information-theory measures.
  Explicitly *not* part of the guaranteed API; present for research, gated so
  they can't accidentally become load-bearing.

The boundary is deliberate: emerge owns physics and stays engine-pure; LP (the
game) lives entirely outside this crate and only ever *configures* what's here.

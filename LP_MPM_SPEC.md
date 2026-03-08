# LP MPM Engine — Design Spec

> This document IS the contract.
> If a feature isn't here, don't build it.
> Update this doc when scope changes — don't just code it.

---

## 1. What Is This Engine?

**One sentence:**
> Real-time MLS-MPM engine for LP — PDE-faithful continuum matter, no constraints, true emergence.

**What it is NOT:**
- Not a general-purpose physics library
- Not a rigid-body engine (LP uses stiff elastic + fracture, not rigid bodies)
- Not a research sandbox
- Not PBMPM (constraint-based, not PDE-faithful — wrong approach for LP)
- Not a port of bevy-mpm — that's the reference, not the source

---

## 2. Scale Targets

| Dimension | Target | Notes |
|---|---|---|
| Particles (typical scene) | 100K | GPU primary path |
| Particles (max stress test) | 1M | Needs 512² grid or sparse |
| Grid resolution (typical) | 256² | ~1.5 particles/cell at 100K — stable, fits L2 cache |
| Grid resolution (stress test) | 512² or sparse | Switch to sparse when world > 1 screen |
| FPS (minimum) | 30fps | 60fps target |
| Sim budget per frame | ≤8ms @ 100K GPU | 60fps = 16.67ms total; sim ≤50% |
| Platform | Desktop first | WASM later, mobile way later, PC is primary |

---

## 3. LP Integration Contract

### 3.1 What LP calls (API shape — to be finalized)

```rust
// Rough intent — exact API decided when emerge-core crate is structured:
// sim.spawn_region(shape, material_type, density);
// sim.apply_impulse(pos, radius, force);     // player actions
// sim.step(dt);
// sim.query_state(pos, radius) -> MaterialState;  // game logic reads
// sim.set_material(particle_id, new_type);   // phase transitions
```

> Section 3.1 is intentionally incomplete — the API shape will be defined
> when the core crate module structure is laid out. Don't lock the API before
> the internals exist.

### 3.2 What LP reads back

- Particle positions — every frame, via GPU buffer (stays in VRAM)
- Particle velocities — for sound/visual effects
- Particle material state — Jp (compression), det(F) (inversion), phase
- Grid momentum — for force feedback, triggers
- Diagnostics — CFL, NaN detection, per-material stats

### 3.3 Render ownership

**Shared GPU buffer — only viable option at scale.**
Particles stay in VRAM. LP writes a custom wgpu shader that reads the position buffer directly.
CPU→GPU upload every frame kills performance at 100K+.

### 3.4 Threading model

- GPU compute (wgpu) — primary path, all transfer phases
- Rayon parallel CPU — fallback (WASM, low-end hardware, debug)
- Game logic (LP queries, events) — always CPU, using mirrored state buffer

---

## 4. Materials — Year 1 Scope

| Material | In Scope | Priority | Reference |
|---|---|---|---|
| Weakly-compressible fluid (Tait EOS) | YES | 1 | `bevy-mpm/src/mpm/materials/fluid.rs` |
| Neo-Hookean elastic (soft solid) | YES | 2 | `bevy-mpm/src/mpm/materials/elastic.rs` |
| Snow (Stomakhin 2013) | YES | 3 | `bevy-mpm/src/mpm/materials/snow.rs` |
| Drucker-Prager sand | YES | 4 | `tmp/sparkl` — not yet in bevy-mpm |
| Rigid bodies | OUT | — | Wrong model for LP |
| Phase transition hooks | YES | 3 | Triggered by material state thresholds |
| Fracture / topology change | v2 | — | After Year 1 materials stable |

**Multi-material coupling**: Shared grid. All materials interact through grid momentum exchange.
This is the MLS-MPM default and is what gives true emergence.

---

## 5. Architecture Decisions

### 5.1 Crate split

```
emerge-core      pure Rust, zero Bevy — the actual physics engine
emerge-bevy      thin Bevy plugin wrapping core — LP ECS integration
```

Why: LP can upgrade Bevy without touching physics. Engine usable server-side.
Lesson from bevy-mpm: mixing Bevy into the sim = painful version lock every release.

### 5.2 CPU vs GPU

```
Particle data:    GPU VRAM primary + CPU mirror for game queries
P2G:              GPU (primary) / CPU rayon (fallback)
Grid update:      GPU (primary) / CPU rayon (fallback)
G2P:              GPU (primary) / CPU rayon (fallback)
Plasticity:       CPU Year 1 (SVD is complex to WGSL) / GPU Year 2
Game queries:     CPU always, using mirrored buffer
```

### 5.3 Particle storage

- SoA (Struct of Arrays) for GPU path — cache-efficient for compute shaders
- AoS (Array of Structs) for CPU path — simpler iteration
- Hybrid: shared canonical SoA, CPU views are transposed on readback

### 5.4 Grid storage

- Dense 256²–512² for Year 1 — 2–8MB, fits GPU cache, simple to implement
- Sparse/blocked grid for Year 2+ — when LP worlds grow beyond single screen

### 5.5 Transfer scheme

MLS-MPM, D_inv=4.0 (quadratic B-spline). Verified correct in bevy-mpm audit.
Do not switch to APIC or FLIP — MLS-MPM is strictly better and already validated.

### 5.6 Time stepping

- Adaptive CFL — primary, handles all Year 1 materials
- Max substeps: 50 (proven stable in bevy-mpm)
- CFL target: 0.3, alarm at 0.4
- Implicit/semi-implicit: NOT Year 1. Only if CFL requires 1000+ substeps (it won't).

---

## 6. What To Port From bevy-mpm

`C:\Users\mathi\Documents\GitHub\bevy-mpm`

All math audited correct. Port selectively — rewrite architecture, keep physics.

| Component | Action | Notes |
|---|---|---|
| `materials/fluid.rs` | Port math | Tait EOS stress — audited correct |
| `materials/elastic.rs` | Port math | Neo-Hookean — audited correct |
| `materials/snow.rs` | Port math | Corotated elastic + SVD plasticity — audited correct |
| `materials/params.rs` | Port | E/nu ↔ lambda/mu conversions + presets |
| `weights.rs` | Port | B-spline weights — audited correct |
| `cfl.rs` | Port | Adaptive CFL timestepper |
| `transfer/p2g.rs` + `g2p.rs` | Rewrite | New SoA structure, keep math |
| `diagnostics/` | Port structure | SimStats pattern is good, adapt for new layout |
| GPU shaders (`.wgsl`) | Port | Fluid shader is production-ready, adapt to new buffer layout |
| Bevy glue code | Rewrite | Clean crate split, new ECS integration |

---

## 7. What "Emergence" Means for LP

**v1 — Minimum viable:**
A player action (footstep, explosion, body contact) creates a pressure wave in the MPM grid.
Adjacent materials respond physically — snow compresses and hardens, water splashes,
soft ground deforms permanently. LP game logic reads particle Jp/velocity/phase state
to trigger audio, animation, world events, and ecosystem responses.

Requires: shared grid multi-material + material state queries + phase transition hooks.

**v2 — Fracture:**
Elastic solids exceed a fracture threshold and split into new particle regions.
Enables: breaking ice, crumbling rock, tearing soft tissue.

**v3 (research) — Two-way coupling:**
MPM particles interact with non-MPM objects (character controllers, projectiles).
Deferred until v1 and v2 are stable.

---

## 8. Out of Scope (Explicit)

These will never be built in emerge unless the spec is explicitly updated:

- Rigid bodies (wrong model — LP uses stiff elastic)
- PBMPM / position-based constraints (not PDE-faithful)
- Implicit time integration (unnecessary for Year 1 material stiffnesses)
- 3D (2D only — LP is a 2D platformer)
- Differentiable simulation (no gradient-based learning planned)
- Networked/multiplayer sync of particle state
- Fluid→SPH transition or hybrid solvers

---

## 9. Reference Repos — Deep Dive Order

Each repo needs a focused analysis prompt before extracting from it.
Paths are relative to `emerge/tmp/`.

| Priority | Repo | Path | License | Goal |
|---|---|---|---|---|
| 1 | `taichi_mpm` | `tmp/taichi_mpm` | MIT | MLS-MPM transfer + multi-material coupling |
| 2 | `sparkl` | `tmp/sparkl` | Apache-2.0 | Rust material dispatch + Drucker-Prager sand |
| 3 | `incremental_mpm` | `tmp/incremental_mpm` | MIT | Behavior baseline (jelly, fluid dt=0.1) |
| 4 | `wgsparkl` | `tmp/wgsparkl` | Custom | WGSL compute patterns, GPU SVD |
| 5 | `taichi` | `tmp/taichi` | MIT | Additional MLS-MPM examples |
| 6 | `matter` (Lars) | `tmp/matter` | GPL-3.0 | Architecture ideas ONLY — no code copy |
| — | `mpm guide` | `tmp/mpm guide...` | — | Algorithm overview, reading only |

> TODO: Create `tmp/ANALYSIS_PROMPTS.md` with a deep-dive prompt per repo.

---

## 10. Done Criteria

emerge is ready for LP (`crates/matter`) integration when:

- [ ] 100K particles @ 30fps on GPU (wgpu compute path)
- [ ] All 4 Year 1 materials working: fluid, elastic, snow, sand
- [ ] Multi-material shared grid — materials interact through momentum exchange
- [ ] Material state queryable from game logic (Jp, velocity, phase)
- [ ] Phase transition API: LP can switch a particle's material type at runtime
- [ ] Zero NaN/inversion in 10-minute stress test on `basic_mixed` scene
- [ ] CPU fallback path compiles and runs correctly (rayon)
- [ ] emerge-core has zero Bevy dependency
- [ ] All constitutive model unit tests passing

---

*Last updated: 2026-02-24*

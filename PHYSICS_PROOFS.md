# Physics proof scenes — what "visually correct" means, per system

This is a planning doc, not an implementation one. It answers: when we
eventually build demo scenes, what does each one need to visually show to
count as *proof*, not just "it runs without crashing"? Follows the standing
plan (`project_lp_demo_scenes_plan` memory, 2026-07-11): one minimal, isolated
scene per piece first, combined "ecosystem" scene later. This doc is the
per-piece checklist that plan was missing — what correctness actually looks
like on screen for each one.

Not example code. No creature/scene work starts from this doc alone — it's
the spec to build against later, per the standing "examples deferred" rule.

---

## Already demoed (existing examples, precedent to match)

| System | Example | What it currently shows |
|---|---|---|
| Sand (Drucker-Prager) | `basic_sand.rs` / `_gpu.rs` | pile settling, angle of repose |
| Fluids | `basic_fluids.rs` / `_gpu.rs` | free surface, splash |
| Snow | `basic_snow.rs` / `_gpu.rs` | compaction, cohesion |
| Jellies (NeoHookean) | `basic_jellies.rs` / `_gpu.rs` | elastic bounce, muscle activation |
| Latent heat | `latent_heat.rs` | phase change energy cost |
| Materials sanity | `validate_materials.rs` | all materials stay finite/stable |
| Creature locomotion | `basic_creature.rs`, `one_leg_creature.rs`, `segmented_creature.rs`, `slip_hopper.rs` | CPG-driven gait attempts (see `basic_creature_locomotion_redesign` memory for real status) |
| Grass | `grass_field.rs` | interactive procedural vegetation |

## Real gaps — nothing currently proves these visually

### 1. Multi-field contact (today's whole session)
The biggest gap. We just spent a full session fixing slip/stick/Baumgarte
stabilization, verified only via `tests/physics_correctness.rs` numbers and
`examples/diag_contact_debug.rs` (a diagnostic, not a demo — no rendering,
console-only gap/velocity printouts). **What proof looks like**: a block
sliding down a ramp at low friction (visibly keeps moving) vs. high friction
(visibly sticks), and a stack of two bodies resting under gravity settling to
a near-zero, stable gap (not sinking through each other) — the exact scenario
`diag_contact_debug.rs` already tests numerically, just needs a render pass.

### 2. Brittle/ductile failure (Von Mises, Rankine)
No demo shows a material actually *breaking*. **What proof looks like**: a
brittle beam (Rankine) failing at a stress concentration — visible crack
propagation, not just a damage scalar in a log. A ductile bar (Von Mises)
necking/flowing under load instead of springing back.

### 3. NACC / GranularFluid
Two full materials (wet soil/clay/tissue, granular-fluid mixture) with zero
visual precedent. **What proof looks like**: NACC — a wet clay column
collapsing differently than dry sand (cohesion + compression hardening
visibly changes the runout shape vs. `basic_sand`'s dry pile). GranularFluid —
a mudslide-like flow that's visibly between "sand pile" and "water splash",
not identical to either.

### 4. Thermal diffusion + phase rules
`ThermalDiffusion`/`add_phase_rule` are real, tested engine features with no
demo. **What proof looks like**: ice melting into water when a heat source
touches it, or water freezing when it drifts into a cold region — the phase
boundary should be visibly readable (color-by-temperature), not just a
material_id flip you have to trust happened.

### 5. Scalar diffusion fields (pheromone/morphogen)
`ScalarDiffusionField` is generic and reaction-diffusion-ready (Gray-Scott
capable) but has zero visual precedent. **What proof looks like**: a
Turing-pattern (stripes/spots) actually forming and stabilizing on screen —
the clearest possible "this isn't just noise" proof for a reaction-diffusion
system, and a real IRL-grounded target (real morphogen patterns).

### 6. Force fields in isolation
`NBodyGravity`/`GravityWell`/`Coulomb`/confinement all have unit tests, no
visual demo. **What proof looks like**: an orbit that stays closed (not
spiraling in/out from integration error) — the single clearest visual
correctness signal for any gravity implementation.

---

## What "proof" means in general (the bar, not per-system)

A scene counts as a real proof, not a tech demo, when:
- The correct and incorrect behavior are both picturable — you'd know a
  regression by eye, not just by a test suite going red.
- It isolates ONE thing. A scene proving contact shouldn't also be proving
  creature locomotion at the same time (matches the standing "one minimal
  scene per piece" plan).
- It reuses the same real IRL-grounded parameters already validated in
  `tests/`/`accuracy_benchmarks.md` — not a re-tuned "looks nice" version that
  quietly drifts from the numbers already proven correct.

## Priority, if/when this becomes real work

Ranked by "biggest gap between what we just built and what anyone can see":
1. **Multi-field contact** — today's whole session has zero visual proof yet.
2. **Thermal + phase rules** — old, tested, zero visual precedent.
3. **NACC / GranularFluid** — newest materials, zero precedent.
4. Brittle/ductile failure, scalar diffusion, force fields — real gaps, lower
   urgency than the above three.

Creature/ecosystem work (the combined "ecosystem" scene) stays exactly where
the standing plan already puts it: after the isolated pieces, not before.

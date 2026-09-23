# Known Limitations - emerge

This document has two parts.

**Open research questions** track places where emerge's design runs into a
**genuinely open problem in the published computational-physics or
numerical-methods literature**: not our own bugs, not untested code paths.
The bar is the same as for something like the Navier-Stokes
existence-and-smoothness problem: not "we personally couldn't solve it," but
"real, named, published sources show the wider field hasn't solved it
either." Every entry must cite sources that themselves say, or show through
a multi-year line of publications, that the specific question is still open.

**The gap registry** (at the end) tracks work the engine knows it has not
done yet: deferred on purpose, left out of the current plan, not audited, or
found in passing and not fixed. Those gaps are ours to fix, not the field's
to solve. They are listed here, each with its source, so that none of them
silently disappears and each can be picked up later with better research.

**Rule for every open research question:**
1. The open question itself, stated plainly.
2. Real, dated, named sources, quoting the part where they say or show
   the question is unresolved. Not just a citation for background.
3. What we found when we hit this in our own engine (kept short: the real
   trail, not a full replay of the investigation).
4. What emerge does because the question is still open.
5. What would close the entry: a specific new published result, not
   "someone tries harder."

---

## Open

### 1. (Closed)

The former entry 1, "no general stability rule exists for APIC under fast,
large motion," turned out to describe two GPU implementation bugs, not an
open question. See "The GPU water splash disintegrated on impact" under
Resolved. Numbering is kept so that references to entry 2 stay valid.

---

### 2. No general method reaches real-time speed with true PDE integration across stiff, multi-material scenes

**The question.** Is there one time-integration scheme for the material
point method that is at once: a direct integration of the real governing
equations, not a position-based shortcut; correct for very stiff
materials without needing a prohibitively tiny time step; and general
across constitutive laws (elastic solids, plastic materials, and
viscous/pressure-based fluids alike)? A real ten-year line of publications
answers only part of this each time, and says so.

**Sources, quoted, in order.**
- Klar, Gast, Pradhana, Fu, Schroeder, Jiang and Teran, *"Drucker-Prager
  Elastoplasticity for Sand Animation,"* SIGGRAPH 2016: "For most of our
  examples, explicit is more efficient... For stiff examples, implicit
  becomes advisable." Their own implicit treatment is "implicit in
  plasticity... not implicit in hardening or friction," a disclosed
  partial scope, not a general answer.
- Fang, Hu, Hu and Jiang, *"A Temporally Adaptive Material Point Method
  with Regional Time Stepping,"* SCA 2018, Section 8: "it is not always
  the preferred choice especially for cases where stiff materials occupy
  the main portion of a scene... We look forward to exploring mixed
  implicit-explicit integration schemes (IMEX) with regional time
  stepping to handle these cases better." The authors name the exact
  combination that would close their own gap and call it future work.
  We found no paper since, including theirs, that has done it.
- Daviet, *"Mixed Material Point Methods for Stiff Elastoplasticity,"*
  NVIDIA, ACM TOG 45(4), 2026, the most recent and most general attempt
  (a separate mixed stress field solved as a convex optimization),
  states plainly: "the performance advantage of implicit over explicit
  time integration thus reduces as the grid resolution increases." Not a
  universal win even in 2026.

**What we found in this engine.** We already have a real, working
implicit solver for one family of materials (stiff elastic and plastic
solids sharing a shared stress law). We measured it directly, twice. On
an already-settled pile it is slower than plain explicit stepping (its
own solve overhead costs more than the small steps it replaces). On a
genuinely violent impact, the regime where the literature above says
implicit should help most, we measured it on a real, cited, stiff sand
material (Young's modulus 15 MPa, a real published excavation-soil
figure) at real production scale: reaching convergence needs several
hundred internal solver iterations per frame, each costing enough on its
own that the total projects to several seconds per frame, 25 to 30 times
slower than the explicit stepping it was meant to replace. Not a tuning
problem: the same investigation ruled out the solver's own numerics
(compared against an exact direct solve, not just its iterative
approximation) before concluding the real bottleneck is structural,
needing a fundamentally different, much larger solver technique to fix,
not a parameter change. We independently re-confirmed the same stiff
material demands a near-identical, very large step count in a completely
different scene mixing it with other materials, so this is not one
scene's quirk. It does not reach fluids at all either: a fluid's stress
comes from pressure and a rate-dependent viscosity, a different
mathematical shape than the solids' stress law this solver was built
around. We also tested letting calm parts of a fluid scene take bigger
steps while only the violent part takes small ones (the Fang et al. idea
above). On our actual water scenes, every part of the fluid body moves at
a similar speed at the same instant during a fall or a fresh spawn, so
there is no calm region left to exploit. Confirmed by direct measurement,
not assumed.

One real, published lever DOES help within this same explicit approach,
for scenes where a small, disclosed accuracy loss is acceptable. Haeri and
Skonieczny, *"Three-dimensional granular flow continuum modeling via
material point method with hyperelastic nonlocal granular fluidity,"*
Computer Methods in Applied Mechanics and Engineering 394 (2022), the same
paper behind our sand's stiffness citation, publish their own softened
version of that same material (a tenth of a percent of the original
stiffness), built for exactly this speed reason, reporting their own real,
measured accuracy cost for it (15.8% mean error on excavation force,
against -0.5% for the full-strength version) rather than one we estimated.
Using it in a player-facing demo (not our accuracy-focused sand scenes,
which keep the full-strength citation) cut that scene's own substep count
by a real, measured 10 times, confirmed live, not just calculated: frame
throughput in a fixed real-time window rose about 8.5 times. Still well
short of real-time, but a genuine, sourced improvement, not a guess.

**What emerge does because this is unresolved.** The opt-in implicit
solver stays scoped to solids only
(`src/spacetime/solver/implicit_corotated.rs`). Fluids keep running fully
explicit, at whatever cost the real physics demands.

**What would close this.** A published method showing true PDE, real-time
integration across both a stiff elastic solid and a viscous, pressure-based
fluid in the same violently-loaded scene. The exact combination Fang et
al. named as future work in 2018, still not shown as of the 2026 Daviet
paper.

**What it costs today, scene by scene** (measured 2026-09-19 on a Radeon
610M, AC power). The rule behind all three numbers is the same: an explicit
step cannot exceed about `dx / c`, where `c = sqrt(E/rho)` is the material's
own speed of sound. Stiffer material or finer grid, more substeps. Nothing
in the code changes that; it is the equation.

| scene | sound speed | substeps per frame | measured |
|---|---|---|---|
| water (dam break, droplet, vortex) | 1.75 m/s, set by the WCSPH rule `c >= 10*v_max` | 52 | 50-56 fps |
| sand, jellies | relaxed published stiffness (see above) | low tens | 60 fps |
| multi-material showcase | sand-dominated | ~264, x4 sim steps per rendered frame | ~10-12 fps |
| snow | 26.5 m/s (`E = 1.4e5 Pa`, `rho = 200`, Stomakhin 2013) | ~880, x4 per rendered frame | 1-2 fps |

Snow is the honest worst case: its real, cited stiffness is fifteen times
water's effective sound speed here, so it needs roughly fifteen times the
substeps, and the demo runs four simulation steps per rendered frame on top.
Making each substep faster does not rescue it: at 3500 substeps per rendered
frame, even halving the per-substep cost leaves it around 3 fps.

What would actually move these two scenes, in order of honesty:
- a coarser grid (`dt` scales with `dx`, and the same region needs fewer
  particles), which costs visual resolution, not correctness;
- slower playback (fewer simulation steps per rendered frame), which costs
  apparent speed, not correctness;
- a published relaxed stiffness for that material, as was already done for
  sand with its own paper's figure -- only where such a figure exists;
- the unresolved research above.

Softening a material without a source, or damping the motion to hide the
instability, is not on this list and is not acceptable here.

---

## Resolved

### The GPU water splash disintegrated on impact

**What was wrong.** Two GPU implementation bugs. On the AMD Vulkan driver
used for development, reading one element of a matrix held in a local copy
of a struct (`p.m[1][1]`) returned another column, so the fluid's volume
change followed shear instead of compression. Separately, the GPU summed
grid mass and momentum as fixed-point integers, which silently dropped every
contribution smaller than half a quantum at small time steps.

**What actually fixed it.** Matrices are passed by value to small helper
functions (`trace2`, `det2`, `frob2_sq`) before being indexed, and the main
grid accumulates in exact floating point with a compare-and-swap loop. The
shear-damping workaround this entry used to describe was deleted.

**What remains.** The broader question, a published stability analysis of
APIC under large and fast deformation, may still be open in the literature,
but it was not what broke this scene. For the simplest case there is a
measured bound: an isolated particle stays bounded up to a time step of
0.80 dx/c_p and diverges at 0.85 (Poisson ratio 0.3), close to the
single-particle value sqrt((lambda + 2 mu) / (2 (lambda + mu))) = 0.84.
The other read forms of the driver bug were not tested one by one, and the
secondary GPU grids still use fixed point.

---

### The water splash used to collapse into a paper-thin layer, then explode

**What was wrong.** Water's pressure formula has a floor: pressure is
never allowed to go more negative than a fixed value, standing in for the
real physical limit where a liquid starts to cavitate instead of
stretching further. That floor was set as a bare number with no
connection to real units, so at this engine's actual scale it clipped
almost the entire useful range of the formula into a flat zone with no
restoring force. Once a region drifted into that flat zone, nothing
pulled it back, and the whole column would silently collapse to a sliver.

**What actually fixed it.** Converting that floor through the same real
SI-to-simulation conversion the rest of the pressure formula already used,
based on the real pressure at which dissolved gas in water starts to
cavitate. Once the fluid could feel a real pressure gradient again, the
column stopped collapsing. That larger, healthier pressure response also
meant the simulation genuinely needed far more, smaller time steps than
before to stay accurate, and an old fixed cap on how many steps a frame
could take was silently cutting that short, which is what caused the
follow-up explosion. Raising the cap to match the real requirement closed
that second issue.

**Then found again, deeper, this week.** The same floor was still broken
by default inside the two constructors meant to build this material
correctly from real SI values in the first place. Anyone using the
"proper" real-unit constructor, not just the two demo files we had
already patched by hand, would have silently reintroduced the exact same
bug. Fixed at the source (`src/matter/materials/fluid.rs`), so every
future user of this material gets the correct value automatically. A real
existing test, built specifically to show this same old flaw next to a
newer alternative material, stopped showing the flaw once this landed. It
was updated to check that both materials now behave well, instead of
checking that one of them still misbehaves.

**Closed:** 2026-09-16 (original fix), 2026-09-17 (fixed at the source).

---

### Water's viscosity was about ten times weaker than intended

**What was wrong.** Water's real-world viscosity (about 0.001 Pa times
seconds, the textbook figure) was being used directly inside a formula
that expected a value already converted into this simulation's own
internal units. Same category of mistake as the pressure floor above,
just on a different field.

**What actually fixed it.** Routed both the shear and the bulk viscosity
through the engine's own existing, correct conversion function. Checked
against the full fluid test suite (all still pass) and separately
confirmed this fix alone does not solve the GPU splash disintegration
(resolved separately, above). The real molecular viscosity, even corrected, is far too small on
its own to explain or calm that particular runaway.

**Closed:** 2026-09-17.

---

## Gap registry

### Deferred by the core implementation plan

- **Pressure projection** (`Grid::project_fluid_incompressibility`, off by
  default). Four mechanisms were measured. The divergence it corrects is
  read with empty cells as velocity zero, so a droplet in free fall shows a
  divergence that is entirely fabricated. The velocity correction divides by
  the nodal mass at partly filled surface nodes while the solve assumes the
  average density. The 0.2 relaxation masks an unstable operator: the full
  correction grows an injected divergence up to 6.2 times on a second pass.
  Fluid, air and wall are classified by mass thresholds and the domain edge
  instead of geometry. Consequence: the wall-contact column survives 120
  frames only with J at the [0.5, 2.0] safety clamp from about frame 20, so
  the frame rates quoted for it (about 30 to 90 fps on CPU, 188 fps on GPU)
  measure cost, not a valid run. The fix is a rebuild on the standard
  formulation: a liquid level set from the particles, solid fractions at the
  wall's real position, a ghost-fluid free surface, consistent discrete
  operators and a conjugate-gradient solve (Bridson, *Fluid Simulation for
  Computer Graphics*; Batty, Bertails and Bridson 2007; Gibou et al. 2002;
  `apic2d` as a reference implementation). The experiments and their
  toggles live on the fork branch `archive/pressure-rhs-audit-2026-09-21`.
- **Time convergence and energy lost per substep.** With APIC, a free
  elastic block keeps 0.69, 0.51 and 0.37 of its energy after the same
  physical time at 256, 1024 and 4096 steps (the exact answer is 1.0):
  smaller steps mean more artificial damping. ASFLIP, available through
  `asflip_blend`, does not fix it: 0.53 at blend 0.5, and at blend 0.97 it
  creates energy (1.16 at 4096 steps). Candidates: PolyPIC (Fu et al. 2017),
  which lowers the loss per transfer without changing the order, and an
  energy-momentum consistent implicit MPM (Love and Sulsky 2006), which
  conserves energy by construction at the cost of an implicit solve. The
  energy lost per step will be published next to the CFL safety factor.
- **3D.** The code is 2D throughout (about 2,400 `Vec2`, 840 `Mat2` and 250
  `IVec2` uses, 440 WGSL 2D types, no dimension abstraction). A
  per-dimension type alias would be the first seam; nothing else is planned.

### Volume a body loses to nothing

`advance_deformation_gradient` now takes the step's volume ratio from the
continuity equation, `det(exp(dt C)) = exp(dt tr C)`, and rescales the
product onto it, instead of letting f32 round-off decide it. What is left
after that, measured on the anchored body of
`tests/scratch_no_compression_drift_horizon.rs` at one substep of 4.37 ms
(the same substep the adaptive loop picks), mean `J - 1` over the body:

| substeps | tension-only, before | tension-only, after | ordinary elastic, after |
| --- | --- | --- | --- |
| 150 000 | -0.00066 | -0.000042 | +0.000025 |
| 450 000 | -0.00323 | -0.00027 | +0.000020 |
| 900 000 | -0.00709 | -0.0115 | +0.000011 |

- **An unloaded tension-only body creeps, and past about 450 000 substeps
  it runs away.** The bands one to four cells below the anchor hold a
  steady positive `J` (a hanging body in tension, which is right), but the
  bottom band carries no load at all, so the moment round-off in the SHAPE
  of `F` pushes one principal stretch below 1, a tension-only law offers no
  restoring force and the compression feeds itself: `max |J - 1|` reaches
  0.131 at 900 000 substeps, past the 0.028 the old code reached. Pinning
  the volume moves the error from the volume into the shape, which this one
  material converts back into volume at zero load. The same body in
  `NeoHookeanMaterial`, which resists compression, is flat over the whole
  horizon (+0.000011, max 0.00085). Real cables and membranes are not
  purely tension-only either (bending stiffness, a small compressive
  modulus); adding one is the candidate fix, and it is not built.
- **The pin is CPU only.** On GPU `volume` is rewritten every step by the
  g2p grid-mass gather, so it cannot carry the volume, and `Particle` is
  full at its asserted 128 bytes with no spare slot for a carrier. The GPU
  shaders keep the plain product and its round-off. This belongs with the
  parity work, which already owns the volume/density divergence between the
  two paths.
- **A sand test lost its premise.**
  `pradhana_effect_across_repeated_separate_impact_episodes` asserted that
  its uncorrected baseline gains volume across repeated impact episodes.
  That gain was the round-off: the baseline now reads -2.19e-8, so the sign
  the test needs is gone and it is ignored under that reason. Guarding the
  Pradhana correction needs a scene where the volume gain it corrects is
  physical.

### Not audited yet

Rendering (`systems/render`); the radiation and optics code (its tests were
read, not the code); rod biology (growth, gravitropism, networks,
plasticity); electromagnetics and acoustics; orbital mechanics; the
information measures; the remaining thermodynamics (granular fluidity,
Cosserat field, water saturation); diagnostics; the particle store; the
grip, ratchet, heightmap and kinematic-obstacle boundaries; a law-by-law
re-read of the 17 materials.

### Found during the core audit, outside the current plan

- Positions and velocities are in cells while physical inputs use
  `dx_meters`, and `grid_cell_size` is always 1.0; several docs warn about
  mixing the two. A typed unit split would remove the trap.
- Force fields add no time-step bound of their own. Harmless for smooth
  fields, unsafe if a stiff one (short-range Coulomb, stiff confinement) is
  added.
- The differentiable solver (`spacetime::diff`) is a second, separate
  physics (signed muscles, sticky floor, no gradient through the kernel
  weights' position dependence), so a gait trained there must be re-checked
  in the runtime solver.
- The electric potential field relaxes with a fixed number of Jacobi
  iterations chosen by the caller, with no convergence test.
- Explicit Euler in the LNN controller is stable only while the time step
  stays well below the neuron time constants; nothing checks it.
- `MaterialRegistry::get` maps an unregistered material id to slot 0 with
  only a debug assertion, so outside debug builds a scene that spawns an
  unregistered id silently runs the wrong material. Four
  `implicit_corotated_substep` tests did exactly that until they were fixed.
- `FrictionBoundary` declares no wall law for strict weakly compressible
  fluids (`is_strict_wc_mpm_fluid_compatible` keeps its `false` default), so
  a strict-fluid scene with a friction floor stops on the solver's
  compatibility assertion. Two `physics_correctness` diagnostics
  (`diag_phase_transition_under_load_causes_stress_discontinuity`,
  `diag_repeated_phase_transitions_do_not_cause_cumulative_instability`) are
  ignored for that reason. Choosing the law is a physics decision: Coulomb
  friction fits a granular skeleton, a liquid needs no-slip or Navier slip.
- `nacc_preconsolidates_more_under_deeper_self_weight` (`tests/physics_correctness.rs`)
  is ignored. Its column is spawned without its self-weight stress and
  rebounds after release; with no cohesion (beta = 0) every tensile state
  resets the preconsolidation pressure, so mean alpha ends positive at every
  depth (shallow 0.132, deep 0.259). The ordering the test expects held only
  while the cap return over-hardened compaction. It should be rechecked once
  bodies spawn in equilibrium (the spawn contract in the core plan).
- Two renderer tests (`render_gpu_produces_visible_particle_pixels_not_just_clear_color`
  and its CPU control) find no particle pixel in a 64x64 headless render, on
  real hardware too. The instance buffers hold correct data, so the fault is
  in the draw pass or in quads about 2 pixels wide at that scale; which one
  is not known. Both stay ignored with that reason.
- The Cam-Clay soil model carries four disclosed approximations, all in
  `src/matter/materials/nacc.rs`:
  - Its elastic response uses a constant bulk modulus. Real Cam-Clay
    stiffness is proportional to the pressure (`K = v p / kappa`), so a
    soil near a free surface is modelled far too stiff elastically.
  - On the dry side of the yield ellipse (an overconsolidated soil being
    sheared) the softening is still evaluated at the start of the step,
    unlike the cap and the wet side. Backward Euler is ill-posed there:
    strain softening loses uniqueness, and at a real clay's hardening
    exponent the residual has no root. Measured with the old sinh law, a
    single sheared step could erase the whole preconsolidation; it needs
    re-measuring under the exponential law.
  - The 2D friction slope M comes from sparkl's own dimension-reduced
    relation `M = 4.619 sin(phi) / (3 - sin(phi))`, not from a measurement
    in plane strain, so a soil's triaxial friction angle reaches the model
    through an unverified mapping.
  - p0 never falls below `kappa * 1e-5`, a numerical floor. For a stiff
    material that floor is larger than the soil's own overburden, so it acts
    as a hidden preconsolidation rather than a neutral guard. The frozen
    block of `examples/cpu/permafrost.rs` sits exactly there: its p0 reads
    43.3 against the 1.5 it carries, and the own-weight preconsolidation the
    scene computes (2.35) never applies. That scene's frozen ground not
    yielding is therefore the clamp holding, not measured frozen-soil
    memory, and must not be read as frozen soil validated.
- The engine holds one Cam-Clay parameter set (`NaccMaterial::kaolin`),
  measured on spestone kaolin at Cambridge and cross-checked against a
  second, independent kaolin set, which sits 2.4 times away in hardening
  exponent. Any other soil has to pass its own oedometer numbers through
  `NaccProps`: presets for soils without measurements were removed rather
  than kept unsourced, peat included.
- About a hundred comments point to notes that live outside the repository
  (working notes from past sessions). They should be rewritten to cite the
  code, a test or this file, or dropped.

### Coverage the CI no longer provides

- **GPU path.** The GPU suite (`tests/gpu.rs`), the 43 library tests and the
  11 `tests/solver.rs` tests that need a GPU adapter run only by hand on real
  hardware, because software adapters give different verdicts. A change that
  breaks the GPU path is only caught when someone runs them.
- **Slow long-horizon tests.** Tests that would push a CI shard past 45
  minutes in the debug profile are ignored in the regular suite and run on
  demand through `.github/workflows/slow-tests.yml`, in the quick profile,
  where debug assertions are off.

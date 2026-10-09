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
  Elastoplasticity for Sand Animation,"* SIGGRAPH 2016 ([doi:10.1145/2897824.2925906](https://doi.org/10.1145/2897824.2925906)): "For most of our
  examples, explicit is more efficient... For stiff examples, implicit
  becomes advisable." Their own implicit treatment is "implicit in
  plasticity... not implicit in hardening or friction," a disclosed
  partial scope, not a general answer.
- Fang, Hu, Hu and Jiang, *"A Temporally Adaptive Material Point Method
  with Regional Time Stepping,"* SCA 2018 ([doi:10.1111/cgf.13524](https://doi.org/10.1111/cgf.13524)), Section 8: "it is not always
  the preferred choice especially for cases where stiff materials occupy
  the main portion of a scene... We look forward to exploring mixed
  implicit-explicit integration schemes (IMEX) with regional time
  stepping to handle these cases better." The authors name the exact
  combination that would close their own gap and call it future work.
  We found no paper since, including theirs, that has done it.
- Daviet, *"Mixed Material Point Methods for Stiff Elastoplasticity,"*
  NVIDIA, ACM TOG 45(4), 2026 ([doi:10.1145/3811345](https://doi.org/10.1145/3811345)), the most recent and most general attempt
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
Computer Methods in Applied Mechanics and Engineering 394 (2022)
([doi:10.1016/j.cma.2022.114904](https://doi.org/10.1016/j.cma.2022.114904)), the same
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
| snow | 26.5 m/s (`E = 1.4e5 Pa`, `rho = 200`, Stomakhin 2013, [doi:10.1145/2461912.2461948](https://doi.org/10.1145/2461912.2461948)) | ~880, x4 per rendered frame | 1-2 fps |

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
bug. Fixed at the source (`src/matter/materials/liquid/fluid.rs`), so every
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
  measure cost, not a valid run. That column is the `fluid_pressure_projection`
  example, and it no longer gets that far: it panics on its first frame, a
  particle reaching about 4e7 cells/s within 16 substeps while the CFL step
  collapses to about 3e-9. Replayed at commits back to 13 September 2026, it
  ran two frames at most. The experiments and their toggles live on
  the fork branch `archive/pressure-rhs-audit-2026-09-21`.

  **The rebuild on the standard formulation is experimental. It now passes
  its four gates, on its own; it is not wired into the step yet.** `grid::mac` (feature `experimental`) holds it,
  apart from the step, which does not call it: a staggered grid, a liquid
  level set from the particles, solid face weights, a ghost-fluid free
  surface and a MIC(0) conjugate gradient (Bridson and Muller-Fischer 2007
  course notes, [doi:10.1145/1281500.1281681](https://doi.org/10.1145/1281500.1281681); Batty, Bertails and Bridson
  2007, [doi:10.1145/1276377.1276502](https://doi.org/10.1145/1276377.1276502); Zhu and Bridson; `apic2d`
  as the reference code). Its stop criteria, in `grid/mac/gates.rs`, were
  committed before its code and allowed two failed runs; both runs failed,
  so the attempt stopped there. The four gate scenes and the four probes
  that counted why are ignored tests in that file
  (`cargo test --features experimental --lib grid::mac::gates -- --ignored
  --nocapture`).

  | scene | first run | second run | third attempt |
  | --- | --- | --- | --- |
  | column at rest, 0.38, 1, 2.5 g | pass | pass | pass |
  | droplet in free fall, 0.38, 1, 2.5 g | pass | pass | pass |
  | dam break | fail | fail | pass |
  | drop into a pool | fail | fail | pass |

  The column holds hydrostatic pressure within 0.046 cell of head, and the
  error halves at 0.5 cm cells; the falling droplet keeps zero pressure and
  J within 1e-5 of 1. Between the runs, two things were fixed, both derived
  here from the conditions the solve enforces, not taken from a source. The
  gather is quadratic along each velocity component and linear across it,
  so the divergence a particle reads is the grid's own: deep in the liquid
  it went from 1000 to 3000 times the grid's to equal to it, and J there now
  stays within 2e-4 of 1 over 2 s. And a wall's closed faces hold the mirror
  image of the flow, so the normal velocity read at a flat wall is zero:
  crossings fell from 474 to 107 and the pool keeps every particle.

  What the second run still failed, and the third attempt's three changes,
  each counted by the probes in `grid/mac/gates.rs` before it was made.
  The criteria did not move; two of the changes are to the setup and are
  declared in that file.

  - Tank corners: all 107 crossings started within two cells of a corner,
    where the mirror reflected across the corner's diagonal and reversed
    only the diagonal component. The solid now gives its own image
    (`solid::box_container_image`): across both walls in a corner, the
    method of images for a right angle. Particles out of the tank: 79 to 44.
  - Positions: 71 of the 72 remaining crossings were water stopping against
    the lid within one cell. Forward Euler crosses a wall whose normal
    velocity falls linearly to zero once that gradient times the substep
    exceeds one; the midpoint rule never does, and Bridson and
    Muller-Fischer 3.1 recommend it for trajectories. Out of the tank: 0.
    The particles per interior cell, which had risen from 4.16 to 5.19 over
    the dam break, now hold at 4.01: the packing came from the same Euler
    step.
  - Volume: J drifted only for particles that had come near the free
    surface (mean 0.95 against 0.998 for the others), because the gather
    reads velocity extrapolated into the air, which no solve made divergence
    free. J now advances by the liquid cells' divergence at the particle,
    the one the projection holds. Volume change 0.0000 in both scenes, no J
    at its bounds, in agreement with the particle count.

  Energy never rose in any run. Cost, release build, unoptimised and
  single-threaded: dam break 5.6 substeps per frame and about 2.2 s per
  simulated second for 3200 particles; the weakly compressible liquid needs
  about 67 substeps per frame for a pool of similar size. The pressure
  solve is preconditioned by a multigrid V-cycle (McAdams, Sifakis and
  Teran 2010) instead of MIC(0): iterations nearly flat with grid size (8
  to 20 from 128 to 1024 cells a side, against 45 to 322), and every
  step order independent, so it can run in parallel.

  **Step A1, the compressible projection (Stomakhin et al. 2014, eqs. 14
  to 18), passes its gates** after one amendment the user approved. Run 1
  found a real bug, fixed: the compressible term was also given to the
  cells inside the walls, which let the liquid flow into them (34
  particles through the column's floor in 5 s). Run 2 failed on a pair of
  criteria no initial state could meet (the right pressure at every frame,
  and the surface sinking from an uncompressed start); the pressure is now
  judged from t = 1 s, once the released column has settled. Results: real
  water (K = 2.2 GPa) needs 5.68 substeps per frame on the dam break
  against 5.41 incompressible, so the real bulk modulus brings back no
  acoustic step; sound at `c = 10 m/s` arrives 4.1 % off; the column's J
  matches `1 - g (h - y) / c^2` to 3e-5, its pressure to 0.015 cell of
  head, its surface sinks 0.411 cell for 0.441. Not done yet:
  wiring it into `Simulation::step` (the coupling to the nodal grid, with
  its own sources read first), then removing the old projection
  (`grid/pressure.rs`).
- **Time convergence and energy lost per substep.** With APIC, a free
  elastic block keeps 0.69, 0.51 and 0.37 of its energy after the same
  physical time at 256, 1024 and 4096 steps (the exact answer is 1.0):
  smaller steps mean more artificial damping. The figures once recorded
  here for `asflip_blend` (0.53 at blend 0.5, energy gain at 0.97) were
  measured against a pre-force snapshot that still carried P2G's fused
  stress impulse, the real bug fixed in `f19e331`; see "A sheared column
  runs out less with more substeps" for the corrected, re-measured
  behaviour and its remaining, expected `(1 - blend)` residual. Candidates
  for APIC's own loss above, unaffected by that fix: PolyPIC (Fu et al. 2017,
  [doi:10.1145/3130800.3130878](https://doi.org/10.1145/3130800.3130878)),
  which lowers the loss per transfer without changing the order, and an
  energy-momentum consistent implicit MPM (Love and Sulsky 2006, [doi:10.1002/nme.1512](https://doi.org/10.1002/nme.1512)), which
  conserves energy by construction at the cost of an implicit solve. The
  energy lost per step will be published next to the CFL safety factor.
- **3D.** The code is 2D throughout (about 2,400 `Vec2`, 840 `Mat2` and 250
  `IVec2` uses, 440 WGSL 2D types, no dimension abstraction). A
  per-dimension type alias would be the first seam; nothing else is planned.

### Volume a body loses to nothing

`advance_deformation_gradient` applies each substep as
`F + (exp(dt C) - I) F`, on CPU and GPU alike, with the increment never
formed as one plus a small number. An f32 near 1 is spaced 1.19e-7 above
it and 5.96e-8 below it, so an increment written as `1 + small` was
rounded one way every substep: the plain product `exp(dt C) F` lost
determinant steadily, and the first fix, rescaling `F` each substep onto a
volume ratio carried beside it, fed the same rounding back into the
shape. Measured on the anchored body of
`tests/probes/no_compression_drift_horizon.rs` at one substep of 4.37 ms
(the same substep the adaptive loop picks), mean `J - 1` over the body:

| substeps | tension-only, plain product | tension-only, rescaled | tension-only, now | ordinary elastic, rescaled | ordinary elastic, now |
| --- | --- | --- | --- | --- | --- |
| 150 000 | -0.00066 | -0.000042 | +0.000020 | +0.000025 | +0.000031 |
| 450 000 | -0.00323 | -0.00027 | +0.000015 | +0.000020 | +0.000031 |
| 900 000 | -0.00709 | -0.0115 | +0.000001 | +0.000011 | +0.000031 |

- **The tension-only body no longer runs away within this horizon.**
  `max |J - 1|` after 900 000 substeps is 0.0011, where the rescaled form
  reached 0.131 and the plain product 0.028. What drove the runaway was
  one-way round-off in the shape of `F`, which a law with no compressive
  stiffness turns into volume. The physical gap itself is unchanged: a
  tension-only body still has no restoring force in compression, so a
  real disturbance there is unresisted. Real cables and membranes are not
  purely tension-only either (bending stiffness, a small compressive
  modulus); adding one is the candidate fix, and it is not built
  (issue #37).
- **What is left is unbiased, not zero.** Over 900 000 substeps of a
  prescribed oscillation from a loaded `F` (`tests/probes/f_rounding_horizon.rs`
  with ROUND_FXX=1.0027 ROUND_FYY=0.9899 ROUND_AMP=0.4 ROUND_DT=2.94e-4),
  `ln det F` ends +7.9e-5 from the f64 integral and `ln(F_xx / F_yy)`
  +6.3e-5, about what an unbiased walk of f32 roundings reaches in that
  many steps; the rescaled form held the volume to -1.2e-7 but put -5.6e-3
  into the shape. Storing `F - I` in place of `F` would make each rounding
  relative to the strain rather than to 1; that changes the particle
  layout on both paths and is not built.
- **A sand test lost its premise.**
  `pradhana_effect_across_repeated_separate_impact_episodes` asserted that
  its uncorrected baseline gains volume across repeated impact episodes.
  That gain was round-off: the baseline read -2.19e-8 with the rescaled
  form and reads +1.91e-8 now, so the sign the test needs is round-off
  either way and it stays ignored under that reason. Guarding the
  Pradhana correction needs a scene where the volume gain it corrects is
  physical.

### A resting liquid packs its particles closer than its volume says

A strict weakly compressible liquid takes its pressure from each
particle's own `J = det F`, which the grid's velocity divergence updates.
The particles themselves move with the gathered grid velocity, and the two
drift apart: under gravity a pool at rest packs its particles toward the
floor while every particle's `J` stays where hydrostatics puts it, so no
pressure answers the packing. The volume the solver accounts for is
conserved; the room the liquid visibly fills is not.

Measured on a pool of water 42 x 20 cells at 1 cm cells, 4 particles per
cell, between slip walls under real gravity (K = 2.25e5 Pa, 300 frames of
1/60 s), particles per cell area against the `4 / mean J` the particles'
own volumes give, per 4-cell band:

| band (cells above the floor plane) | t = 0 | t = 1 s | t = 3 s | t = 5 s | 4 / J at 5 s |
| --- | --- | --- | --- | --- | --- |
| 0 to 4 | 4.000 | 4.143 | 4.607 | 4.845 | 4.023 |
| 4 to 8 | 4.000 | 4.095 | 4.256 | 4.464 | 4.018 |
| 8 to 12 | 4.000 | 4.095 | 4.214 | 4.214 | 4.012 |

- Viscosity 100 times water's (0.1 Pa s) leaves it unchanged (bottom band
  4.952 at 5 s), so it is not the stirring of a low-viscosity pool.
- It shrinks with particles per cell, not with the grid. The same physical
  pool, the bottom 4 cm after 5 s, particle number density times mean `J`
  (1.000 = no packing):

  | cell size | particles per cell | bottom 4 cm | next 4 cm |
  | --- | --- | --- | --- |
  | 2 cm | 4 | 1.159 | 1.019 |
  | 1 cm | 4 | 1.205 | 1.111 |
  | 0.5 cm | 4 | 1.256 | 1.103 |
  | 1 cm | 9 | 1.075 | 1.023 |
  | 1 cm | 1 | 2.059 | 1.711 |

  Refining the grid at fixed particles per cell does not converge it, which
  points at the particle quadrature rather than at the grid discretisation.
- It is not the wall: a pool against a `StaticBoxBoundary` face and the
  same pool between the domain's own walls give the same levels and the same
  mean `J` (0.9949 against 0.9952).
- It slows down as the pool settles. A shallower pool (56 x 8 cells, same
  water, cell size and gravity) run for 30 s keeps its bottom band 10.4 to
  12.6 % denser than `4 / J` from 20 to 28 s, while its fastest particle
  stays below 1 cell/s; the bands above stay within 3 %. The last two
  samples read 12.2 and 14.1 % as that speed rose again to 1.9 cells/s,
  so whether it levels off over a longer horizon is not measured.
- What it costs a scene: the free surface sits low. In
  `boiling_cost_probe`'s three columns put in 18-cell beakers under
  gravity, the levels sit 7 to 9 % below the mixture rule after two
  seconds while the densities stay within 0.05 % of it. The boiling and
  cavitation scenes stay weightless for this reason.
- The grid's own mass density does see the packing; the fluid's stress
  reads `J` instead because the grid density lags one substep (see
  `NewtonianFluidMaterial::kirchhoff_stress`). Which correction is right
  is open and not built.

### Solids throw their friction heat away

Only `BinghamFluidMaterial` and `NewtonianFluidMaterial` declare a
specific heat. Every solid and plastic law returns the trait's default,
so the work their yielding and friction dissipate raises nothing's
temperature: sand shearing, metal yielding and rock fracturing are all
adiabatic in the wrong direction, losing the energy instead of keeping
it as heat.

Found in the first pass of the core audit, alongside the spawn contract
and mu(I)'s Euler integration. Both of those are now fixed and this one
is not: it belongs to no phase of the plan, which is why it is written
here rather than left in a note. It is the cross-domain energy question,
not a one-line fix: a law that heats itself needs somewhere for that
heat to go, which is the thermal coupling this engine has for fluids and
not for solids.

### A fluid's cavitation pressure is only half derived

The Tait law these fluids use is a gauge law, zero at rest density, so a
particle above rest volume asks for a negative pressure. That request is
clamped at `MaterialParams::pressure_floor`, and a floor of 0.0 deletes it:
expansion meets no restoring force while compression meets the full one, so
any symmetric noise in the divergence ratchets volume upward forever. This is
what made bodies visibly swell and drift apart the longer a demo ran.

Measured, fixed and closed for the yield-stress family. Before
`BinghamProps::cavitation_pressure_pa` existed, EVERY expanded particle in
every slab was clamped, 105 of 105 at two cells and 657 of 657 at sixteen, and
the slabs climbed past J = 1.002 while their own weight said they should sit
below 0.999. With the cavitation pressure its constants derive, about -280 Pa
from the nucleus term `2*gamma/R`, the same slabs end twenty seconds at 2 ms a
frame within 1.3e-3 of one and all of them BELOW one (0.99986, 0.99981,
0.99988, 0.99957, 0.99875 at one, two, four, eight and sixteen cells). That is
not flat to the fourth decimal, which was the letter of the original criterion:
the four-cell slab moves from 0.99924 to 0.99988 over that window. What is gone
is the upward ratchet. On the same sweep window the drift reads +0.0012 against
+0.0444 percent a second at two cells and -0.0157 against +0.0737 at sixteen,
where the sign has inverted to compaction, and the thickest slab is heading for
the 0.998 that `rho g h / 2K` predicts for 32 mm of it, which is load. Worst
`|J - 1|` on that slab: 0.79 to 0.86 at every window measured with no floor,
0.038 with one.

What is still open:

- Checked, and the other fluid families do NOT have this bug, which is worth
  recording because an earlier draft of this entry claimed they did.
  `BoilingMixtureMaterial`, `CavitatingFluidMaterial` and
  `IsothermalCavitatingFluidMaterial` bound their pressure at `p_sat(T)` by
  construction through `cavitating_eos`, which IS their tension limit and is
  sourced. `GranularFluidMaterial` states `pressure_floor: 0.0` deliberately,
  with its own comment saying so: a cohesionless granular contact carries no
  tension, and zero is the right answer there.
- `NewtonianFluidMaterial::from_physical` does still state -100,000 Pa as a
  bare constant. `2*gamma/R` returns it at a 1.4 micrometre nucleus, so it
  could come out of the same relation the yield-stress family now uses, at
  each fluid's own surface tension instead of water's. It does not.
- The needle-induced-cavitation relation is `P_c = 5E/6 + 2*gamma/R` and only
  the nucleus term is used. The elastic term would deepen the floor and make it
  depend on the fluid's own stiffness. Leaving it out is the conservative
  direction and is stated at the call site, but it has not been measured.

- Useful floors run from about -140 Pa, where clamping stops being the
  dominant effect, to about -2800 Pa. At -10,000 Pa the sixteen-cell slab
  panics on a timestep it cannot represent, so the floor is bounded from below
  by stability and not only by physics. That bound is measured on one scene.
- Within that band, deeper measures better. At twenty seconds, -560 Pa leaves
  3 of 567 expanded particles clamped on the sixteen-cell slab against 60 of
  663 at -280, so 0.5 percent against 9 percent, and worst `|J - 1|` of 0.028
  against 0.038. At -280 the fluid is asking for up to 830 Pa of tension and
  getting 280 back, which is what those 9 percent are. The shipped value is
  the one the cited bound gives, not the one that measures best.
- That bound is itself the weakest link. The 1 mm is where a petrographic
  manual for HARDENED concrete draws the line between entrained and entrapped
  voids, not the largest bubble measured in a fresh paste, and entrapped voids
  above 1 mm exist too and would give a shallower threshold still. The radius
  is a declared modelling choice standing on a published class boundary. Only
  the surface tension in `2*gamma/R` is measured on the fluid family itself.

Ruled out by counting, and worth recording because this entry used to name it:
the free-surface node exclusion in `gather_grid_to_particles`. Instrumented
over the same sweep, it fired 0 times in 418,714,560 node evaluations (issue
 #39), because P2G inserts every in-bounds node of a particle's own stencil.
The invariant it protects is kept as a test,
`a_rigid_translation_reads_no_velocity_gradient`.

### A yield-stress fluid below its yield rings forever

`BinghamFluidMaterial`'s elastoviscoplastic branch (Saramito 2007,
[doi:10.1016/j.jnnfm.2007.04.004](https://doi.org/10.1016/j.jnnfm.2007.04.004), as a
radial return with a Perzyna viscous overstress) is purely elastic below its
yield stress: the viscosity only acts once the material flows. So a block
loaded under its yield and released has nothing to take the energy out, and
it rebounds and rings indefinitely. Seen directly in
`tests/probes/bingham_cursor_yield.rs`: in zero gravity a 1200 Pa block
pushed at half its yield and released keeps oscillating, which is what gives
that row its higher floor, 0.15 mm of apparent change of shape at x0.5
against 0.006 for the 60 Pa block.

It is also a departure from the model this branch cites. Woodbridge, Fonte
and Juel ([arXiv 2609.12229](https://arxiv.org/abs/2609.12229), 2026, a yield-stress spreading study built on
the Saramito family) describe it as: "Below yield, the material behaves as a
linear viscoelastic solid". The dissipation below yield comes from a solvent
viscosity acting at all stresses, which this radial-return version keeps
only inside the plastic flow. A real gel dissipates below yield too. The
same paper notes that one solvent viscosity cannot match both the flow well
above yield and the sub-yield response, so the coefficient would have to be
measured for the sub-yield regime on its own.

Declared (2), not a bug: every column that sits at rest in a scene is
unaffected, and only a column set vibrating below yield shows it. Closing
it means a viscous term acting below yield as well, with that coefficient.

### A settled yield-stress deposit's shear jumps between lines of particles

On the slump demo's 60 Pa deposit, the shear over its own yield differs
between a particle and its neighbours by 0.075 on average, against a spread
of 0.168 across the whole deposit; the demo's stress view shows it as
stripes. It is not the gripping floor: 0.070 on a slip floor against 0.074,
and strongest in the top band, not the bottom one
(`tests/probes/bingham_deposit_state.rs`, `the_stripes_on_each_floor`).
Cause not established; no issue yet.

### GPU snow hardens differently at a body's edge

Two measurements, months apart and from opposite directions, that are
almost certainly the same defect:

- A GPU compaction test found cohesion's differentiation about sixteen
  times weaker than the CPU's: `jp_cohesive` 0.99793 against
  `jp_loose` 0.99792, a 6e-6 gap where the CPU reliably shows 1e-4.
  Correct direction, wrong magnitude
  (`gpu_snow_compacts_and_cohesion_resists_compaction`, ignored with its
  trail).
- The parity matrix leaves snow as its one remaining gap: under uniaxial
  compression the two paths end 9.8e-2 apart in position and 2.5 in
  velocity. `tests/probes/snow_gpu_gap.rs` narrows it: both sides run
  ONE identical substep, after which the hardening and density fields
  part at the body's EDGE while the middle stays identical to the digit.
  Worst particle on CPU carries hardening 2.158 at density 0.255, on GPU
  1.324 at 0.391. The GPU's looser timestep bound (2.34e-3 against
  1.48e-3, so one substep against two) follows from that state, it does
  not cause it.

Both point at the same named suspect: the order in which the GPU clamps
`hardening_scale`/`plastic_volume_ratio` relative to its F update, against
the order `snow.rs` uses. That comparison is line-by-line reading of
`snow_plasticity` in WGSL against the CPU law, not a measurement, and it
is not done. Until it is, these are one entry rather than two.

### The DX12 teardown sometimes kills the process

`tests/gpu_parity.rs` ends about one run in nine with Windows exit code
0xc0000409 (FAST_FAIL), after the test harness has already printed its
result. Measured attribution rather than an assumption: 2 aborts in 18
runs on DX12, 0 in 20 on Vulkan, the matrix printing identical numbers
on both backends, and the existing 52-test GPU suite never showing it.
Nothing of this engine's runs at teardown (no `unsafe`, no `Drop` in
`systems::gpu`), and with `RUST_BACKTRACE=full` and wgpu logging on
there is no panic, no backtrace and no validation warning, so it is not
a Rust panic reaching abort. Consequence: the parity matrix stays a
manual test on real hardware and must not gate CI on DX12.

### A grain stops at a boundary instead of bouncing

A grain that meets a boundary loses its whole normal velocity, whatever
its restitution. In a grid-coupled step (`grains::coupling`),
`GrainPopulation::clean_wall_normal_velocity` removes the inbound normal
velocity of every grain touching a boundary, and
`resolve_wall_contact_forces` applies only the contact's tangential force
and moments. The normal force is left out on purpose: applied against the
grid's position clamp, which keeps resetting the overlap, it refilled
every step and launched the grain. So in its normal direction a boundary
is a perfectly inelastic stop; grain-grain contact is not affected. A
scene that needs a grain to bounce off a wall, such as the
colliding-blocks count in `tests/grains_pi_collisions.rs`, uses a third
grain held in place as the wall. A restitution-aware normal response at
the boundary would close this.

### A wall cannot tell a sealed container from an open one

The equations of state give gauge pressure, zero at the ambient state, so an
empty region next to a body stands for ambient air. A one-sided wall
(`SlipBoundary::new`, `apply_slip_wall_velocity`; the Coulomb walls of
`FrictionBoundary` act only on motion into the wall too) lets matter leave
whenever its velocity points away from the wall. Together they put ambient
air behind every wall: a fluid below ambient pressure pulls itself off the
wall. That is right for an open container and for solids and grains, wrong
for a sealed one, where the absolute pressure stays positive and the fluid
never leaves the wall. Measured on a sealed box of air
(`tests/probes/gas_sound_speed.rs`, `a_rarefaction_...`): a rarefaction
reflects off the one-sided wall with a coefficient of -0.84, like off a free
surface, where a rigid wall gives +1, and a wave running along the top and
bottom walls loses 10 % over 56 cells.

Fixed for sealed containers: `SlipBoundary::sealed` holds matter at its wall
nodes in both directions (reflection +0.93 in the same probe), and a scene
whose container is sealed chooses it. The sound-speed probe, a box filled
wall to wall, uses it.

Still open:
- A wall that decides by itself. The physical rule is that a fluid leaves a
  wall only where ambient air can reach it, at a free surface touching the
  wall, or where it cavitates. Holding every fluid at every wall instead
  would keep water with a free surface stuck under an overhang it should
  fall from. A wall node would need to know whether a free surface reaches
  it, which the grid does not record today.
- The Coulomb friction walls and the GPU walls (`grid_update.wgsl`,
  slip-only and one-sided, see #61) have no sealed variant.
- No gas can sit next to real vacuum. Under gauge pressure every empty
  region is ambient air, so in `examples/cpu/basic_gas.rs` the 0.6x pocket
  is compressed to a mean J of 0.69 by 1.5 s instead of expanding (its
  header says so). Absolute pressure is what vacuum needs, and the gas
  law's own doc records why it was dropped: an unbalanced atmosphere on
  every particle drove J to its maximum.

### A sheared column runs out less with more substeps: APIC transfer dissipation

`sand_column_collapse_runout_against_lube_2005` used to run an 8x16-cell
grid-unit column (4 cells per half-width) and record 0.49 of Lube, Huppert,
Sparks and Freundt 2005's (Phys. Rev. E 72, 041301, series A, Eq. 4)
predicted runout as an unexplained gap. Rebuilt at the sand's real size
(their Tables I-II: 1.95 cm half-width, 29.5° repose) and 16 cells per
half-width, it reproduces their runout within measurement: the farthest
particle (what they measured) reads 0.96 of their prediction, the deposit's
98th percentile 0.79. Most of the old 0.49 was that coarse column, not a
general defect. The pile this same collapse leaves still stands at 26.3°
where dry sand stands at 30-35° (`sand_angle_of_repose_is_physical`, GH
#28), a separate gap from the runout question above, and stays open; see
below.

Four candidates were checked for the old gap and ruled out:
- Matching the cone to 2D Mohr-Coulomb moves the repose pile 0.3°.
- The tension cutoff, which takes 70 % of a settled pile's particles each
  substep (`dp_update_cost_breakdown`), is what lets this sand flow: with
  `use_pradhana` the column does not collapse.
- The volume a cutoff removes, carried in `log_volume_strain`: without it
  (`volume_correction = 0`) the repose pile reads 26.9° and the runout is
  unchanged.
- Forgetting that history once a particle leaves the cutoff, as the
  reference code of Tampubolon et al. 2017 does (ziran2020's
  `DruckerPragerStvkHencky::projectStrain` sets `logJp = 0` outside its
  tension case; this engine, like sparkl, never does): measured in a scratch
  build, repose, runout and fitted slope are unchanged, because the
  particles concerned never leave the cutoff.

The real missing piece was resolution, but not the resolution sweep below:
that one measured the repose pile's *slope* converging near 21°, never this
test's *runout*, so it never actually checked whether runout had converged.
Run at the column's own real size, the bulk deposit (98th percentile)
converges by 16 cells per half-width (4.86 then 4.94 half-widths at 16 vs
40 cells, no friction hardening); the farthest particle keeps creeping out
with every refinement, the same thin-foot effect as the repose test's base.
Release geometry (symmetric vs against a back wall) was checked too and
ruled out, matching within 1% at both 4 and 16 cells per half-width.

New: runout falls with substep count, not stiffness. At 4 cells per
half-width, E = 10 MPa with `material_cfl_coefficient` cut by 10 (36 000
substeps instead of 3 600)
gives the same runout as E = 1 GPa at the normal coefficient (r* 2.70/2.40
against 2.69/2.39, front/98th percentile): the two are the same effect,
and it is the substep count, not E.

First suspected a `FrictionBoundary` wall artefact (the earlier text here
said so); a later control rules that out. A column of this sand resting on
an 8-cell bed of the SAME sand, its own shear at least 8 cells from any
wall, well past the quadratic kernel's 1.5-cell reach, shows the same size
of dependence with no wall anywhere near the shear: r* 4.23/3.66 at x1,
3.11/2.98 at x10, -26%/-19%. A sticking floor, the Coulomb floor, and a
two-field contact floor all show the same 15-37% range; only a frictionless
`SlipBoundary` does not, because that floor lets the column slide as a
near-plug flow with little real shear to begin with, not because walls are
immune to this. So the dependence lives in sheared flow itself, not in how
a wall condition is imposed; a wall-side fix (Nairn's multi-node boundary
condition, Toyota & Umetani's augmented grid points, or reusing the
two-field contact machinery for walls) would not address it.

Isolated with no plasticity and no wall at all: a free `CorotatedMaterial`
block with a standing shear wave as its initial condition (the exact
gradient given to APIC's C matrix), tracked by total energy (kinetic plus
corotated strain energy), which a real elastic solid should conserve. After
13 wave periods, total energy over its start reads 0.72 at 1000 substeps
and 0.13 at 9000 for the same real time; the coherent mode's kinetic peak
reads 0.45 against 0.087. Same mechanism, same direction, with nothing
plastic and nothing granular involved.

The cause is APIC's own transfer dissipation, a documented property of
this transfer family, not an engine-specific defect: Nairn and Hammerquist,
*Material Point Method Simulations using an Approximate Full Mass Matrix
Inverse* (preprint, dated 10 October 2025 on the copy read, no journal
reference found in the text), Sec. 1 calls the lumped mass matrix's energy
loss "detrimental"; Sec. 3.1 states that their PIC-family transfer "does
not converge with reduced time step, dissipation increases as the time
step decreases", because every step replaces particle velocities with
grid-extrapolated ones; Sec. 2.6 states "APIC has significant dissipation
(albeit much less dissipation than non-affine PIC methods)". APIC is this
engine's own transfer. More substeps means more such replacements per real
second, so a material needing more substeps (stiffer, or any scene run at
a finer CFL) loses more of a shear flow's energy, independent of plasticity
or walls.

A side finding from the same probe GAINED energy instead of reducing this
dissipation: confirmed as a real bug, not a property of `asflip_blend`
itself, and fixed in `f19e331`. ASFLIP's FLIP residual and Cundall
damping both compare the grid against its velocity before this substep's
forces; the snapshot was taken after P2G's own momentum normalization,
which already carries this substep's stress impulse (the fused MLS-MPM
transfer adds it there), so the comparison point was never actually
pre-force. At blend 0.97 that let only 3% of the real elastic restoring
force reach the FLIP correction, while the deformation gradient kept
accumulating strain normally: energy gain, not conservation. Cundall
damping shared the identical bug and, with no gravity, wall or contact
force to see, did nothing at all regardless of its coefficient.

Fixed by recomputing the stress term alone (`scatter_particle_stress_impulse`)
and subtracting it back out of the snapshot (`Grid::snapshot_velocities_before_stress`),
matching Fei et al. 2021's own Eq. 12, which transfers the pre-force
velocity without the stress impulse. Re-measured on the same shear-wave
probe: ASFLIP at blend 0.97 now reads 1.31 of its start energy at 0.5 s
(was 24-34x), Cundall damping at 0.75 now reads 0.0026 of an otherwise
undamped 1.04 (was identical to no damping at all). GPU matches CPU to
four digits.

The substep-count dependence on the real sand column shrinks under the
fixed ASFLIP but is not zero: at blend 0.97, x10 substeps run 14% shorter
than x1, against 24% under plain APIC. This is not a remaining bug: the
gather itself is `v_store = new_v + asflip_blend * diff_vel`
(`transfer::g2p`), a direct blend between the dissipative APIC gather and
the FLIP residual that cancels it, so an incomplete correction of order
`(1 - asflip_blend)` is expected by construction, not a leftover defect.
A blend sweep on the same column (0.97, 0.99, 0.995) shows the residual
shrinking toward 0 as blend approaches 1, the signature of that
`(1 - blend)` term, not of an unresolved mechanism. Running every scene
at `asflip_blend` near 1.0 is not free: ASFLIP costs about +20% on this
scene (one extra stress pass every substep) and the FLIP side of the
blend is itself noisier, so this is a real tradeoff, not a strictly
better default.

A property of this model found on the way, not a cause of the repose gap or
the stiffness dependence above: most of a settled pile sits at the tip of
the cone, stress-free, every substep.
More than 8 cells deep, 66 % of the particles take the tension cutoff under
the pile's weight (`settled_pile_creep_by_depth`). Each carries a small
positive volume history (about 1e-3) from the impact, repaid only by
accumulated compression and grown again by any expansion, so it hovers at
the tip; without the history only 1.2 % do. The load is carried by the rest.

Whether the history drives the slow flattening that
`sand_collapse_relaxation_long_horizon_plateau_check` records (29.6° to
10.8° over 100 000 steps of a grid-unit scene) is not settled: an SI column
held its fitted slope at 20.2-20.8° over 25 s with the history and at 20.3°
without it, a far shorter horizon than that test's.

Finer cells do not close the gap either. An SI column (8 x 16 cm,
E 1 MPa, 35°, friction floor) settles to a flank slope of 20.7° at 0.5 cm
cells and 21.4° at 0.25 cm after 3 s, fitted by least squares through the
surface between 20 % and 80 % of the peak (at 1 cm the surface is too
coarse to fit). Height over farthest base particle reads 21.3°, 19.6° and
17.0° at 1, 0.5 and 0.25 cm, but that drop is the measure, not the slope:
the height holds at 5.8-6.0 cm while a thin foot spreads further at finer
cells (98th-percentile base half-width 14.9, 16.0 and 18.3 cm).

The test used to hide this behind `cohesion = 5.0`, tuned to compensate a
"4.7x too far" runout that was the frictionless floor fixed in 70a1b75; with
particles at their real mass (caa97df) that cohesion holds the column up
entirely. `examples/gpu/material_sandbox_gpu.rs` still gives its sand the
same `cohesion = 5.0` for its look; the value has no measured basis and
needs its own re-tuning pass.

### A slow pour builds a tower, not a pile

`sand_pile_built_by_patient_pour_matching_real_creep_timescale` (ignored)
records 30.8°, measured on a frictionless floor; on the current engine the
same pour stands at 84.9° (83.4° before the cone was matched to 2D
Mohr-Coulomb) and fails its own 25-40° band. Like the ~12°
once recorded for `sand_angle_of_repose_is_physical`, the number predates
the floor's friction (70a1b75) and the particles' real mass (caa97df). Not
measured to a cause.

### Not audited yet

Rendering (`systems/render`); rod biology (growth, gravitropism, networks,
plasticity); diagnostics; the particle store; the grip, ratchet, heightmap
and kinematic-obstacle boundaries; a law-by-law re-read of the 17
materials.

Given a lighter pass only (formulas and constants against their sources,
citations, f32 precision traps; not every code path): radiation and optics,
electromagnetics and acoustics, orbital mechanics, the information
measures, and the thermodynamics outside the core audit. It found #51 (the
granular fluidity's pressure factor is inverted) and #52 (heat diffusion
loses sub-kelvin increments in f32), and leaves one question: the Cosserat
field's stability bound does not reduce to seconds in the units it states.

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
  `src/matter/materials/solid/nacc.rs`:
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

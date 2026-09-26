# Known Limitations - emerge

This document tracks places where emerge's design runs into a **genuinely
open problem in the published computational-physics / numerical-methods
literature**. Not our own bugs, not untested code paths, not a TODO we
haven't gotten to yet. Those belong in GitHub issues, where this project
already tracks them.

The bar for an entry here is the same bar as something like the
Navier-Stokes existence-and-smoothness problem: not "we personally
couldn't solve it," but "real, named, published sources show the wider
field hasn't solved it either." Every entry must cite sources that
themselves say, or show through a multi-year line of publications, that
the specific question is still open. A source about the general topic is
not enough on its own.

**What does NOT belong here:** an unimplemented feature, an untested
material combination, a config flag with no code behind it, a constant we
chose by testing rather than deriving, a bug we haven't traced yet. Those
are real and worth tracking, but they are ours to fix, not questions
science hasn't answered yet. They live in GitHub issues instead.

**Rule for every entry:**
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

### 1. No general stability rule exists for APIC under fast, large motion

**The question.** Explicit MPM fluid and solid simulations use a transfer
scheme called APIC to move information between particles and the grid.
Is there a time-step-independent rule that guarantees this scheme stays
stable (no runaway growth) once particles are moving fast and deforming a
lot, not just sitting near rest? As of the sources below, no such rule
exists in the published literature.

**Sources, quoted.**
- Bai and Schroeder, *"Stability analysis of explicit MPM,"* Computer
  Graphics Forum 41(8), 2022. Sections 3.13-3.14 prove, for a simplified
  near-rest single-particle case, that "there is no time step bound for
  which a single-particle simulation with... APIC will be stable for any
  combination of time step sizes." A real, formal impossibility result.
  Their own analysis stays restricted to configurations near rest
  (deformation close to identity). It does not extend to large
  deformation or fast motion, and we found no later paper that does.
- Sun, Shinar and Schroeder, *"Effective time step restrictions for
  explicit MPM simulation,"* SCA 2020, Section 8: "We derived the
  single-particle instability case for PIC transfers; we leave a full
  APIC treatment for future work." The authors of the one existing
  fluid-specific stability formula say plainly, in print, that the
  general APIC case (their own formula only covers plain PIC, no affine
  term) stays unsolved.
- We independently re-derived and numerically checked this ourselves
  (2026-09-17). Both an isolated particle and a dense, periodic lattice,
  tested against this engine's real quadratic B-spline kernel, come out
  provably stable at any coefficient we tried. The real instability we
  hit only shows up once particles move fast during an actual impact,
  exactly the regime the two papers above never cover. This confirms the
  gap is real rather than just repeating the citation.

**What we found in this engine.** Our own water splash demo (GPU) would
disintegrate into scattered droplets on violent impact instead of
splashing and settling. Bisecting the history pinned this to one exact
commit that made an unrelated volume-tracking formula more exact, not
less. That fix was correct on its own; it simply stopped a small,
pre-existing numerical noise from being accidentally smoothed away. The
real growth traces to the particle's own affine velocity state (the
quantity APIC uses to carry local motion) amplifying itself through the
repeated grid round trip during a violent event. It shows up on GPU and
barely on CPU, because GPU's stricter time-step safety check forces far
more, much smaller steps for the same real second of simulation, letting
the same small growth compound many more times before the frame ends.

**What emerge does because this is unresolved.** A shear-relaxation term
in the GPU fluid transfer (`src/systems/gpu/shaders/g2p.wgsl`, around
line 408). Its *shape* is grounded in a real, cited idea (from Lewin et
al., "Position Based MPM," SIGGRAPH 2024): damp only the shear part of
the affine state, the part with no direct physical meaning of its own,
and leave rotation and volume change untouched. The two actual numbers
used (a baseline damping fraction and a hard ceiling) were reached by
testing against the real demo, not derived from a formula, because no
formula in the sources above produces them. GPU only; the CPU solver has
no equivalent yet.

Two later attempts to make this damping smarter (only engage once a real
excursion looks dangerous, or scale by real elapsed time instead of by
substep count) were each tried, measured with a real coherence check --
does the fluid stay one connected body, or do particles end up isolated
from every neighbor -- and reverted: both let real fragmentation back in
that the flat, unconditional version does not show, confirmed live, not
assumed (`examples/gpu/fragmentation_check_gpu.rs` holds the real check).
The flat version's own real cost is real too: a fluid body that lands
correctly but does not visibly keep relaxing afterward. Between a fluid
that freezes in a safe shape and one that quietly loses particles, the
frozen one is the honest choice until a real fix for the underlying
question exists -- not a preference, a measured trade every stronger or
gated variant tried so far has landed on the wrong side of.

**What would close this.** A published stability analysis of APIC that
covers real deformation and real particle speed, the way Bai and
Schroeder's 2022 paper covers the near-rest case.

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
confirmed this fix alone does not solve the splash instability in entry 1
above. The real molecular viscosity, even corrected, is far too small on
its own to explain or calm that particular runaway.

**Closed:** 2026-09-17.

---

## Not tracked here (see GitHub issues instead)

Real, disclosed engineering gaps found while chasing the two open problems
above. Kept in issue tracking, not this document, because they are ours to
fix, not the field's to solve.

- Two fluid-like materials (cavitating fluid, boiling mixture) share entry
  1's safety gate without being individually tested against it.
- Entry 1's fix lives only in the GPU shader, with no CPU equivalent.
- A config flag for letting calm regions take bigger time steps was
  removed on 2026-09-17. It had no real code behind it, left over from an
  earlier rewrite that was undone for unrelated reasons. A direct
  feasibility check confirmed the idea's own precondition, a genuinely
  calm region next to a violent one, does not hold on our current fluid
  scenes anyway. If rebuilt, it should follow the real Fang et al. 2018
  algorithm, on a scene where that precondition actually holds.

---

*Last updated: 2026-09-17.*

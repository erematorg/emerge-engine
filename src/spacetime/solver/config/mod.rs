use glam::Vec2;

/// D⁻¹ = 4.0 for the quadratic B-spline MLS-MPM kernel (always).
/// Not a tunable parameter -- hardcoded from Hu 2018 Table 1.
pub(crate) const KERNEL_D_INVERSE: f32 = 4.0;

/// Parameters that control the physics solver and its runtime behavior.
#[derive(Clone, Copy, Debug)]
pub struct SimConfig {
    pub grid_res: usize,
    pub grid_cell_size: f32,
    pub dt: f32,
    pub adaptive_timestep: bool,
    pub cfl_include_affine_speed: bool,
    /// Safety factor on the ADVECTIVE (velocity) CFL bound: `dt <=
    /// cfl_coefficient * cell_width / max_speed` -- the classic
    /// Courant-Friedrichs-Lewy (1928) condition, applied here to a
    /// particle's own displacement per substep instead of a fixed-grid
    /// advection scheme. Not a value with one "true" derivation -- a CFL
    /// coefficient is inherently a conservative margin under the
    /// theoretical stability limit (Courant number <= 1.0), and 0.9 is a
    /// standard, common choice for the mild advective bound specifically
    /// (see `material_cfl_coefficient` below for why the material/viscous
    /// bounds use a tighter 0.5 instead -- same convention
    /// `rod_cfl_coefficient` documents for the rod solver's own bound).
    pub cfl_coefficient: f32,
    /// Safety factor on the MATERIAL (elastic wave speed) CFL bound,
    /// `elastic_wave_dt`: `dt <= material_cfl_coefficient * cell_width /
    /// c_p` where `c_p = sqrt((lambda+2*mu)/rho)` is the longitudinal
    /// elastic wave speed. Kept tighter (0.5) than the plain advective
    /// `cfl_coefficient` (0.9): a stiffness-driven instability from
    /// under-resolving the acoustic/elastic wave speed tends to blow up
    /// hard and immediately, unlike a mild velocity-CFL overshoot, so a
    /// stricter margin is the standard conservative choice (matches common
    /// explicit-MPM/FEM practice, not a value derived from one specific
    /// formula -- like any CFL number, it's a stability margin, not a
    /// measured material property).
    pub material_cfl_coefficient: f32,
    /// Safety factor on the VISCOUS (diffusive) timestep bound for
    /// viscosity-bearing fluid materials (Newtonian/Bingham) -- a diffusion-
    /// type stability condition (`dt <= C * dx^2 / nu`-shaped, distinct
    /// power of `dx` from the acoustic bound above). Same conservative-
    /// margin reasoning and same 0.5 value as `material_cfl_coefficient`
    /// (viscous diffusion instability is likewise an immediate blowup, not
    /// a mild overshoot).
    pub viscous_timestep_coefficient: f32,
    /// Safety factor for `rod::rod_cfl_dt`'s own bound, folded into
    /// `choose_substep_dt` alongside `material_cfl_coefficient`. Not the same
    /// 0.5 as `material_cfl_coefficient`: `rod_cfl_dt` sums every stiffness/
    /// damping term touching each point (a Gershgorin row-sum bound), which
    /// is real but LOOSE for the rod's geometrically nonlinear dynamics -- 0.5
    /// diverges for a long/stiff-EI cantilever at N=30/40; 0.4 is the
    /// bisected, long-horizon-verified safe value across that regime and a
    /// short/soft blade-of-grass regime (see `project_rod_cfl_gershgorin_and_cookbook_2026-07-21` memory).
    pub rod_cfl_coefficient: f32,
    /// Legacy timestep-granularity hint retained for API compatibility.
    ///
    /// It is deliberately not a lower bound: no solver may raise an
    /// advection, acoustic, or diffusion CFL upper bound to this value.
    pub min_dt: f32,
    pub project_invalid_state: bool,
    pub projection_min_density: f32,
    pub projection_min_volume: f32,
    pub projection_min_deformation_j: f32,
    /// Gravitational acceleration in grid-coordinate units/s².
    /// Use `Vec2::new(x, y)` for angled or planetary gravity. Typical: `Vec2::new(0.0, -9.81)`.
    pub gravity: Vec2,
    /// Direction light is sensed as coming FROM, for `rod::Phototropism`
    /// (see that struct's own doc). A FIXED, externally-set vector, NOT a
    /// real solar/orbital model -- `emerge`/LP work at continuum scale, no
    /// day/night sun-angle system exists (the existing `day_night_thermal_gpu`
    /// demo is a pure scalar ambient-temperature oscillation with no light
    /// direction at all). Default: straight up (`Vec2::new(0.0, 1.0)`,
    /// opposite the default `gravity` direction) -- "light from directly
    /// above," the common illustrative case. Zero cost/no behavior change
    /// for any rod that doesn't opt into `Phototropism`.
    pub light_dir: Vec2,
    pub boundary_thickness: usize,
    pub default_initial_volume: f32,
    /// Request a kernel-density measurement for materials that consume one.
    /// Strict WC-MPM liquids keep their constitutive `rho=rho0/J` state and
    /// are excluded even when this is enabled.
    pub recompute_density_each_step: bool,
    /// Base grid density: particle mass PER CELL AREA, not per particle.
    /// A spawn's actual particle mass is `grid_density * spacing^2`, because a
    /// lattice at `spacing` cells carries `1/spacing^2` particles per cell.
    ///
    /// It is a density, not a mass, for a dimensional reason. `lame_from_si_
    /// physical` converts SI stress to grid units by dividing by
    /// `rho_kg_m3 * dx_meters^2`; the MPM grid update accelerates a node by
    /// `f/m`, i.e. by `sigma_grid / rho_grid`. Those two agree only when
    /// `rho_grid == 1`. Carrying a per-PARTICLE mass here instead made the
    /// gravity/stiffness ratio scale as `1/spacing^2`: a scene sagged 3x too
    /// far at `spacing = 0.5` and 10x at `0.25`, while `spacing = 1.0`
    /// happened to be correct -- which is why the error hid for so long
    /// (measured against the analytic self-weight strain `rho*g*h/2E`; see
    /// `physics_correctness::self_weight_strain_is_spacing_independent`).
    ///
    /// Default 1.0 = "this material IS the reference density," correct for any
    /// single-material scene. For a scene mixing real densities, keep 1.0 and
    /// give each region a `mass_override` of `(rho_i/rho_ref) * spacing^2`
    /// (`SpawnRegion::mass_from` does this), converting every material's
    /// stress with the same `reference_density_kg_m3`.
    pub grid_density: f32,
    /// Shared density that multi-material scenes measure every material
    /// against, so real density CONTRAST survives the grid-unit conversion.
    ///
    /// Only read by `SpawnRegion::mass_from`. A single-material scene never
    /// needs it: converting that material's stress by its own `rho_kg_m3`
    /// already puts it at `grid_density = 1`. Default 1000.0 (water).
    pub reference_density_kg_m3: f32,
    /// Initial substep-resource budget for GPU scheduling and diagnostics.
    ///
    /// It is not a physics cap: a solver step must advance the full requested
    /// `dt` through CFL-safe substeps or explicitly report/defer the work.
    pub max_substeps_per_step: usize,
    /// Real preflight/retry for strict WC-MPM fluid materials, CPU-side (root-caused
    /// 2026-08-08, see project memory's fluid-recovery notes): CFL picks a substep dt from
    /// the PREVIOUS substep's state, which cannot perfectly bound an inherently NONLINEAR
    /// Tait-EOS pressure spike (`eos_power` typically 7) that a fast local compression
    /// event -- confirmed specifically at a hard-wall contact -- can produce WITHIN the
    /// very substep CFL is trying to bound. Verified this is a genuine, convergent
    /// stability limit, not a discrete conservation bug: tightening `material_cfl_
    /// coefficient` GLOBALLY to 1000x default measurably converges toward correct
    /// momentum conservation (not a plateau), but at a real cost no real-time scene can
    /// afford everywhere, all the time, just to cover the rare moments a wall is touched.
    /// When enabled: after computing a substep, the solver checks whether any strict
    /// fluid particle's OWN volume-ratio change this substep exceeded
    /// `fluid_step_retry_threshold`; if so, the substep is rolled back (particle state
    /// only) and retried at half `sub_dt`, up to `FLUID_STEP_RETRY_LIMIT` (a fixed
    /// engineering safety cap on retry attempts, the same category as
    /// `max_substeps_per_step` itself, not a physics tunable) halvings before giving up
    /// and accepting the substep as-is. Default `false` -- zero behavior/cost change for
    /// every existing scene; only checked when a strict fluid material is actually
    /// present. Scoped limitation, disclosed not hidden: the rollback restores particle
    /// state only, not rods/grains/thermal fields that `do_substep` also advances, so a
    /// retry in a scene that mixes a strict fluid with any of those could under/over-count
    /// their own update by one attempt -- fine for a fluid-only scene (this fix's proven
    /// case), not yet verified for a mixed scene.
    pub fluid_step_retry_enabled: bool,
    /// Evaluate `add_phase_rule` predicates once per `step()` instead of once
    /// per substep. `false` (default) keeps the documented per-substep
    /// contract exactly.
    ///
    /// Opt-in because it is only equivalent when a rule's inputs cannot change
    /// WITHIN a frame. That is true for every rule this engine ships today --
    /// they are thermodynamic predicates (freeze/boil/melt), and temperature
    /// advances once per `step()` (the diffusion operators run at their own
    /// stable rate, see `Simulation::step`), so re-testing them 18x per frame
    /// re-reads identical inputs 17 times. It is NOT true for a rule keyed on
    /// something that genuinely varies per substep (velocity, position), which
    /// is exactly why this is a caller's choice rather than a silent change.
    ///
    /// Live-measured on `basic_fluids_gui.rs` (2912 particles, ~18 substeps):
    /// the phase-rule scan was ~3400-4300 us of a ~29000 us step (~12%).
    pub phase_rules_once_per_step: bool,
    /// PROACTIVE companion to `fluid_step_retry_enabled` (2026-08-08): a strict fluid
    /// particle within `boundary_thickness` cells of any wall (the SAME zone the slip
    /// clamp already treats specially -- no new distance invented) gets its own CFL
    /// material-timestep contribution computed with `material_cfl_coefficient` divided by
    /// this factor.
    ///
    /// **Real, second use found 2026-08-08 (see `MEMORY.md`'s fluid-recovery notes, Round
    /// 9): also tightens the gravity-CFL bound (`cfl.rs`) for the SAME near-wall strict-
    /// fluid particles.** This is the one that actually matters once `SimConfig::
    /// fluid_pressure_iterations > 0` (`eos_stiffness=0`): the ORIGINAL acoustic-only
    /// tightening above is structurally INERT there (confirmed live, bit-identical
    /// results at scale=5 vs scale=20 vs disabled -- `NewtonianFluidMaterial::
    /// timestep_bound`'s acoustic term is `c2=0` whenever `eos_stiffness=0`, so dividing
    /// `material_cfl_coefficient` by anything changes nothing). The gravity bound is the
    /// one bound still genuinely ACTIVE and PREDICTIVE for an eos-less fluid at rest, so
    /// tightening THAT one, for near-wall particles specifically, is what actually helps.
    /// Real, measured result on the single hardest known scene (fluid already touching a
    /// wall at spawn, spanning nearly the full domain): combined with
    /// `fluid_pressure_iterations=1`, this is the first configuration tonight that
    /// completes all 120 frames with zero non-finite state (a real, disclosed, bounded
    /// slow drift late in the run, not fully eliminated, but never diverging).
    ///
    /// `1.0` (default) = no scaling on either bound, byte-identical to before this field
    /// existed.
    pub fluid_near_wall_cfl_scale: f32,
    /// Gates the ORIGINAL acoustic-CFL use of `fluid_near_wall_cfl_scale` on ACTUAL
    /// compression, not just wall proximity alone -- required fix, 2026-08-08: a
    /// distance-only trigger fires for ANY fluid resting near a wall, including a calm
    /// puddle sitting on an ordinary floor (the floor IS a wall) -- meaning realistically
    /// every settled water scene would pay the extra cost forever, not just during a
    /// brief compression event. Live-confirmed: exactly this caused a sustained 3-5fps
    /// crawl in `basic_fluids_gui.rs` once the water settled, not a transient slowdown.
    ///
    /// **Does NOT gate the newer gravity-CFL use above** -- that one is deliberately
    /// PREDICTIVE (see its own doc): a compression-based gate is reactive by definition
    /// (needs `J` to have already drifted before it can fire), which is exactly what
    /// fails at the critical first substep, before anything has moved yet. Set this to
    /// `0.0` when using `fluid_near_wall_cfl_scale` for its gravity-bound effect (the
    /// verified `fluid_pressure_projection_gui.rs` demo does exactly this).
    ///
    /// Real water's physical compressibility is negligible (bulk modulus ~2.2 GPa) -- ANY
    /// sustained J deviation past a percent or so is already far outside real water
    /// physics, so `0.01` (1%, the default) is a real, disclosed, measured threshold for
    /// the acoustic-bound use case, not an arbitrary cutoff pretending to be derived from
    /// a closed form.
    ///
    /// **FALLBACK ONLY (2026-08-11) for materials with no acoustic term**
    /// (`MaterialModel::rest_acoustic_c2() == None`, e.g. `eos_stiffness=0.0`
    /// pressure-projection fluids). Any material WITH a real Tait EOS uses
    /// `fluid_near_wall_compression_mach_margin` instead (see that field's
    /// own doc) -- a fixed 1% is only correct for the SPECIFIC EOS stiffness
    /// it happened to be measured against; a deliberately softened EOS (real,
    /// disclosed accuracy/perf trade, `basic_fluids_gpu.rs`'s own doc) has a
    /// much larger NORMAL compression range by design, so this absolute
    /// threshold trips almost permanently for such a material -- root-caused
    /// live: `near_wall=true` continuously, forcing `fluid_near_wall_cfl_scale`'s
    /// 20x tightening on essentially every substep regardless of whether a
    /// genuine impact was happening (1346 substeps/frame measured, ~20x more
    /// than the acoustic term alone would need).
    pub fluid_near_wall_compression_threshold: f32,
    /// Real replacement (2026-08-11) for `fluid_near_wall_compression_threshold`'s
    /// acoustic-bound use, for any material with a genuine Tait EOS
    /// (`MaterialModel::rest_acoustic_c2() == Some(c2_rest)`). Compares
    /// `|J-1|` not to a fixed absolute percentage, but to
    /// `(last_max_particle_speed / sqrt(c2_rest))² *
    /// fluid_near_wall_compression_mach_margin` -- i.e. the compression THIS
    /// material's OWN acoustic stiffness would predict as normal at the
    /// scene's actual current flow speed, times a margin.
    ///
    /// Grounded in the standard WCSPH relation `Ma² ≈ Δρ` (density variation
    /// ≈ squared Mach number; Monaghan 1994, Morris et al. 1997) -- the same
    /// relation this engine's own `weakly_compressible` constructor already
    /// uses to size `eos_stiffness` from a target `c_ref_m_s`. Applying it
    /// HERE too, dynamically, from the scene's actual measured speed rather
    /// than a single a-priori guess made once at material-construction time,
    /// is the real fix for the same class of self-defeating trap
    /// `basic_fluids_gpu.rs`'s own doc history narrates: softening the EOS to
    /// afford more substeps only works if the near-wall gate's OWN threshold
    /// scales with that same softening, not a threshold borrowed from a
    /// stiffer reference EOS. Directly informed by Zhang et al., "A variable
    /// speed of sound formulation for weakly compressible SPH" (UCSPH,
    /// arXiv:2310.04139), whose own time-dependent sound-speed update rule
    /// (`c_s(n+1) = max(10·v_max(n), ...)`, recomputed from the ACTUAL
    /// measured flow state every step rather than a static a-priori bound) is
    /// built on the identical Ma²≈Δρ relation -- adapted here to the
    /// near-wall gate specifically rather than the base EOS stiffness itself,
    /// since making the EOS's own `eos_stiffness` field dynamic would require
    /// per-substep material mutation, a materially larger change deferred as
    /// a real, disclosed follow-up rather than rushed tonight.
    ///
    /// No published number exists for THIS margin specifically (this exact
    /// near-wall-gate application is this engine's own construction, not a
    /// technique the cited paper itself describes) -- `2.0` (default) is a
    /// disclosed engineering choice, not a physics constant: it means the
    /// gate fires once local compression exceeds DOUBLE what the current
    /// flow speed already predicts as normal for this EOS, a round,
    /// minimal-but-clearly-distinguishing margin above 1.0 (which would fire
    /// on literally any compression at all, defeating the gate's purpose the
    /// same way the old absolute 1% did for a soft EOS).
    pub fluid_near_wall_compression_mach_margin: f32,
    /// Per-substep admissible `|ln(J_new/J_old)|` for a strict fluid particle before
    /// `fluid_step_retry_enabled` rejects and retries that substep. NOT the general
    /// `deformation_gradient_cfl_bound` safety margin (`cfl_coefficient`, ~0.5) -- tested
    /// directly against this session's repro and it never once fired: the real failure
    /// mode is many small, individually-under-50%, systematically-same-direction
    /// per-substep changes accumulating over hundreds of substeps, not one large jump.
    /// Calibrate empirically against a real repro before changing, same discipline as
    /// `eos_stiffness`'s own tuning -- this is disclosed as measured, not derived from a
    /// closed-form bound, precisely because no such closed form caught the real failure.
    pub fluid_step_retry_threshold: f32,
    // REMOVED (2026-09-17): `fluid_regional_substepping_gpu_enabled` and
    // `fluid_regional_substepping_fine_tier_margin` used to live here. The
    // real implementation behind them was part of the strict-fluid/DCT-
    // pressure/retry rewrite (`57b83dc`, 2026-08-13) that was wholesale
    // reverted the next day (`gpu_fluid_stable`'s own ignore doc, 2026-08-14)
    // for unrelated reasons -- the config fields survived that revert with
    // no code left to read them, becoming an inert switch that looked like
    // a feature but did nothing. A real feasibility check
    // (`examples/gpu/regional_substep_feasibility_check.rs`, 2026-09-17)
    // also confirmed the technique's own precondition -- a genuinely calm
    // region existing next to a violent one -- does not hold on this
    // engine's real fluid scenes anyway (0% of populated blocks classified
    // "calm" through the whole violent window on both DamBreak and
    // DropletImpact). See `KNOWN_LIMITATIONS.md` entry 2. If this is
    // rebuilt for real, base it on Fang, Hu, Hu & Jiang, "A Temporally
    // Adaptive Material Point Method with Regional Time Stepping," SCA
    // 2018, on a scene where the precondition actually holds.
    /// APIC affine-matrix blend [0, 1].
    /// 1.0 = full APIC (angular-momentum-conserving, taichi default).
    /// 0.0 = pure PIC (maximum numerical dissipation, fastest settling).
    /// Intermediate values blend between the two -- equivalent to taichi's `apic_damping`.
    ///
    /// For strict WC-MPM liquid materials, APIC transports momentum while the
    /// thermodynamic state remains `V=V0 J`, `rho=rho0/J`. Calibrate a spawn
    /// with `m = rho0 * spacing^2` in solver units, so that `V0=m/rho0`
    /// represents its lattice area. Strict WC-MPM fluids require exactly
    /// `1.0`: lowering it changes `div(v)` in their continuity update. Do not
    /// lower this blend as a substitute for a violated acoustic/viscous CFL
    /// condition or an unmodelled free surface/cavitation problem.
    pub apic_blend: f32,
    /// Emergency projection ceiling for solid/plastic AND strict fluid states.
    ///
    /// Originally solid-only ("strict WC-MPM fluid materials never use this:
    /// their `J` is evolved by the continuity equation and an inadmissible
    /// state is reported rather than rescaled" -- the old doc here). Real,
    /// measured gap found 2026-08-09: the "report rather than rescale"
    /// design assumed corruption always shows up as ONE bad substep large
    /// enough for `fluid_step_retry_enabled` to catch and roll back. A
    /// pressure-projection scene (eos_stiffness=0, no elastic backstop
    /// resisting drift at all) instead showed `J` drifting steadily in the
    /// SAME direction for hundreds of individually-under-threshold substeps
    /// -- min_j 0.72 -> 0.0000154 over 120 frames, density_ratio hitting
    /// 64,916x, with `fluid_step_retry_threshold` never once firing because
    /// no single substep's own change was ever large enough. Same class of
    /// gap `fluid_step_retry_threshold`'s own doc already named ("many
    /// small, individually-under-50%... accumulating... not one large
    /// jump") but in the OPPOSITE (compression, not expansion) direction,
    /// and with no per-substep backstop at all until now. Applied every
    /// substep now (see `do_substep`'s pre-P2G pass), not just at retry
    /// exhaustion -- a real, generous safety ceiling (50x volume), not a
    /// physical bound (real water's actual bulk modulus permits ~1%), so it
    /// never touches a legitimately-behaving scene, only a genuinely
    /// runaway one. See `j_min` for the symmetric floor.
    pub j_max: f32,
    /// Symmetric floor to `j_max` (`1.0/j_max` by convention, same 50x
    /// magnitude) -- the compression-direction half of the same real gap
    /// `j_max`'s own doc describes. Applied every substep for strict fluids
    /// alongside `j_max`, not just at retry exhaustion.
    pub j_min: f32,
    /// Speed below which a passive (activation == 0) particle becomes eligible for sleep.
    /// 0.0 = sleep disabled. Typical: 0.01–0.05 grid-cells/s.
    /// Sleeping particles skip P2G and G2P entirely; woken by neighbouring active cells.
    pub sleep_threshold: f32,
    /// Speed below which a whole rod (max over ALL its points) becomes eligible
    /// for sleep -- separate knob from `sleep_threshold` since a rod's natural
    /// residual-sway speed under wind is a different scale than an MPM
    /// particle's. 0.0 = sleep disabled (default -- no existing rod scene's
    /// behavior changes). A rod with an active push (`Rod::push_strength > 0`)
    /// never sleeps regardless of this value. Sleeping rods skip scatter/
    /// gather/internal-force integration AND their own `rod_cfl_dt` term in
    /// `choose_substep_dt` -- the real cost driver for many simultaneous rods.
    pub rod_sleep_threshold: f32,
    /// Coulomb friction coefficient for multi-field contact between a `contact_group != 0`
    /// particle and everything else (Bardenhagen 2001 -- see `Particle::contact_group` doc).
    /// Only has any effect at all when at least one particle actually sets a nonzero
    /// `contact_group`; otherwise `Grid::resolve_contact` never has anything to resolve,
    /// regardless of this value. 0.0 = frictionless (normal no-penetration only, free
    /// tangential slip). Real dry-material Coulomb coefficients are typically 0.3-0.9.
    pub contact_friction: f32,
    /// ASFLIP blend factor [0, 1] (Fei, Guo, Wu, Huang, Gao 2021, "Revisiting Integration in
    /// the Material Point Method: A Scheme for Easier Separation and Less Dissipation", ACM
    /// TOG 40(4)). Reintroduces a FLIP-style velocity/position correction on top of ordinary
    /// APIC, letting granular/debris material separate crisply instead of smearing together.
    /// 0.0 = disabled -- byte-identical to plain APIC, the default for every existing scene/
    /// test. ~0.97 matches the paper's own reference implementation (`nepluno/pyasflip`).
    /// Costs nothing when 0.0: no grid-velocity snapshot is taken, G2P takes the exact
    /// original code path.
    pub asflip_blend: f32,
    /// Cundall local non-viscous damping coefficient [0, 1] (Cundall 1982/1987
    /// "dynamic relaxation"; MPM formulation per Beuth, Benz, Vermeer, Coetzee,
    /// Bonnier & van den Berg 2007, "Formulation and Application of a Quasi-
    /// Static Material Point Method," NUMOG X -- used in production geotechnical
    /// MPM, e.g. Anura3D). Real, material-agnostic fix for the mismatch an
    /// explicit-dynamic MPM solver has with an inherently quasi-static problem
    /// (a granular pile creeping toward equilibrium): damps the component of
    /// each grid cell's velocity change THIS substep (a real proxy for applied
    /// force, since Δv = F·dt/m at fixed dt/mass) that opposes nothing but its
    /// own oscillation -- proportional to the FORCE just applied, not to
    /// velocity itself (that's ordinary viscous damping, a different real
    /// mechanism already available via `ViscoelasticMaterial`). Self-gating by
    /// construction: a cell with zero velocity has nothing to oppose (zero
    /// damping), and steady DIRECTED motion (a creature walking, a fluid
    /// splash) barely engages it -- only genuine wobble/settling does. Lives at
    /// the grid level, not inside any one material's constitutive law, so
    /// every material benefits once enabled, not just granular ones.
    /// 0.0 = disabled (default) -- no velocity snapshot taken, byte-identical
    /// to every existing scene, same zero-cost convention as `asflip_blend`.
    pub cundall_damping: f32,
    /// N-phase mixture coupling drag coefficient (generalizes Tampubolon et al.
    /// 2017, "Multi-species simulation of porous sand and water mixtures" --
    /// Darcy-style momentum exchange between materials wrapped in different
    /// `MixturePhase` slots, see `WithMixturePhase`). Units: mass/time (a per-node drag rate,
    /// NOT the paper's own permeability-derived `c_E` directly -- this is a first,
    /// simplified scalar-coefficient version; mapping to real soil permeability/
    /// porosity is real, disclosed future work, not attempted yet).
    /// 0.0 = disabled (default) -- `Grid::has_mixture_activity()` gates the extra
    /// P2G scatter and the whole resolve pass, zero cost for every scene that
    /// doesn't use `WithMixturePhase`, matching `asflip_blend`'s own convention.
    pub mixture_drag_coefficient: f32,
    /// Jacobi iterations for the mixture incompressibility pressure projection
    /// (`Grid::project_mixture_incompressibility`, see its own doc for the full
    /// derivation and citations). Real fix for a real, root-caused instability:
    /// the drag coupling above conserves momentum but never enforces the
    /// mixture's actual incompressibility constraint, so under sustained/
    /// confined loading (water settled into sand) the violation compounds
    /// silently over hundreds of steps until velocities blow past the CFL
    /// bound. 0 = disabled (default) -- byte-identical to the original
    /// momentum-only coupling, matching every other opt-in field's convention.
    /// Real, disclosed caveat: this is an approximate, real-time-affordable
    /// Jacobi solve, not an exact Poisson solve -- pick this value by measuring
    /// against your actual scene's long-settle behavior (a settled, confined
    /// liquid is the documented worst case for a low iteration count), not by
    /// assuming a small fixed count is free.
    pub mixture_pressure_iterations: u32,
    /// Number of OUTER correction passes per substep for STRICT (single-
    /// phase, non-mixture) fluid incompressibility pressure projection
    /// (`Grid::project_fluid_incompressibility`, see its own doc). Real fix
    /// for the root-caused sustained-wall-contact stability limit (see
    /// `MEMORY.md`'s fluid-recovery notes, Round 9): a stiff Tait EOS
    /// (`eos_stiffness`) needs a tiny acoustic-CFL-bound timestep to stay
    /// stable, and even the tightest near-wall CFL scale tested was too slow
    /// for sustained contact (a settled puddle against a wall/floor). A real
    /// Chorin-style pressure projection (Bridson; the same family already
    /// proven in this codebase for the two-phase mixture case, see
    /// `mixture_pressure_iterations`'s own doc), solved EXACTLY via a
    /// discrete cosine transform (Stam 1999), enforces incompressibility as a
    /// solved constraint instead of an explicit stiff spring -- set the
    /// fluid's own `eos_stiffness` to 0.0 (already a legal, asserted-
    /// permitted value, see `fluid_state::tait_pressure`'s own `>= 0.0`
    /// contract) when using this, so the two mechanisms don't double-count
    /// the same pressure.
    ///
    /// NOT literal Jacobi iterations any more (the DCT solve is exact, not
    /// iterative) -- this is the OUTER repeat count: `project_fluid_
    /// incompressibility` is called this many times in a row per substep,
    /// each pass re-measuring the (hopefully smaller) residual divergence
    /// after the previous pass and correcting again. Real, standard
    /// technique (the same principle PISO/SIMPLE-family incompressible-flow
    /// solvers use: a single projection is only a first-order splitting of a
    /// genuinely violent state). 1 is a reasonable default once enabled; more
    /// passes measurably delay and shrink the known remaining failure mode
    /// below, at a real, linear extra cost per pass.
    ///
    /// **Real, honest, disclosed remaining limitation (Round 9, confirmed by
    /// FOUR independent real attempts, not guessed)**: for the single
    /// hardest known scene (fluid spanning nearly the full domain height,
    /// already touching a wall at spawn), this projection does NOT fully
    /// eliminate a rare failure mode where one particle's own volume ratio
    /// `J` drifts to an extreme value near a true wall, eventually collapsing
    /// the substep budget or crashing outright. Confirmed NOT a solver-
    /// quality artifact: Gauss-Seidel (no spectral basis, can't "ring")
    /// converges to the SAME extreme pressure the DCT solve does, proving
    /// it's the true solution of the constant-density-simplified equation
    /// for that input, not a numerical hallucination. More outer passes
    /// delay it (5 passes: stable ~16 frames; 10 passes: ~34 frames; 20
    /// passes: ~29 frames at much higher cost) but do not eliminate it. The
    /// real, structural cause: the UNIFORM-density simplification (needed to
    /// make the exact DCT solve tractable, see `Grid::
    /// project_fluid_incompressibility`'s own doc) cannot represent a
    /// genuinely non-uniform, hard-packed compression event at a wall
    /// corner -- the correct fix needs a VARIABLE-density formulation done
    /// properly (real per-cell mass, with the free-surface unbounded-alpha
    /// problem solved structurally, e.g. `apic2d`'s own real variational
    /// solid-fraction boundary weighting -- `tmp/apic2d/apic2d/fluidsim.cpp`,
    /// not yet ported), a genuinely separate, larger undertaking. For less
    /// extreme scenes (fluid not already touching a wall at spawn), this
    /// projection is real, verified, and stable -- see the easy-drop-scene
    /// finding in the same memory notes.
    ///
    /// Real, disclosed scope limit: when this is nonzero,
    /// `Simulation::step`'s strict-fluid assertions additionally require
    /// EVERY particle in the scene to be a strict fluid (`owns_deformation_
    /// volume_state() == true`) -- a scene mixing fluid with sand/solid bodies
    /// on the same grid is not yet supported here (would need per-cell
    /// fluid-fraction tracking, the same kind of bookkeeping
    /// `mixture_cells` already does for the porous case, just not built for
    /// this non-porous case -- deliberately out of scope until a real scene
    /// needs it, not a hidden gap).
    pub fluid_pressure_iterations: u32,

    /// Real, opt-in P2G spatial-sort (2026-08-10) -- see `spatial_sort_order`/
    /// `scatter_particles_to_grid_sorted` (`spacetime::transfer::p2g`) for the
    /// full real motivation (Gao et al. 2018 SIGGRAPH Asia: periodic particle
    /// reordering for cache/hashmap locality; this engine's OWN GPU path
    /// already does the equivalent via `particle_sort.wgsl`'s indirection
    /// array). Default `false` -- zero behavior/cost change for every
    /// existing scene. NOT proven universally beneficial yet: real,
    /// disclosed tradeoff is an O(N log N) sort every step against a
    /// real-but-scene-dependent P2G cache-locality win; only enable after
    /// measuring on the actual target scene, same "prove it, don't guess"
    /// discipline as every other perf lever in this engine.
    pub spatial_sort_enabled: bool,

    // ── Physical unit scaling ──────────────────────────────────────────────────
    // Default 1.0 = simulation units (no scaling). Set these to enable SI-calibrated materials.
    // Use `lame_from_si` / `gravity_to_grid` in `materials::utils` to convert SI values.
    /// Physical length of one grid cell in meters. Default 1.0 (grid units).
    ///
    /// Example: if the simulation domain is 64 cells representing 0.64 m, set `dx_meters = 0.01`.
    pub dx_meters: f32,
    /// Physical duration of one simulation time unit in seconds. Default 1.0.
    ///
    /// Typically set to match `config.dt` in physical seconds.
    /// Gravity: `gravity = Vec2::new(0.0, -9.81) * dt_seconds^2 / dx_meters`.
    pub dt_seconds: f32,

    /// Real, opt-in implicit (Newton-CG) grid-velocity update -- see
    /// `spacetime::solver::implicit_corotated`'s own module doc for the
    /// full method (Klar 2016 operator split: implicit elastic solve using
    /// the shared Corotated elastic branch, then the material's own real,
    /// unmodified plastic return-mapping applied once at the end). Default
    /// `false` -- zero behavior change for every existing scene.
    ///
    /// Only engages for a substep where EVERY active particle's material
    /// uses that shared elastic branch (`DruckerPrager`, `Corotated`,
    /// `VonMises`, `Rankine`, `DruckerPragerMuI`) and no rods, grains,
    /// multi-field contact, or mixture coupling are active in the scene --
    /// a real, disclosed v1 scope limit (see the eligibility check itself),
    /// not a silent partial application. `do_substep` falls back to the
    /// normal explicit pipeline, byte-identical to today, whenever this
    /// check fails -- so turning this on for a scene it doesn't yet support
    /// is always safe, just inert.
    ///
    /// Real, measured evidence this is worth having (2026-09-10, standalone
    /// `tests/scratch_implicit_mpm_stage3_drucker_prager_multi_particle.rs`
    /// before this field existed): 10.3x wall-clock speedup at basic_sand's
    /// own real ~2016-particle, ~2247-substep-per-frame scale, real
    /// operator-split plastic-correction error under 1% at far coarser
    /// correction frequency than that.
    pub implicit_corotated_elastic: bool,
}

impl Default for SimConfig {
    /// Safe production defaults: adaptive timestepping on, state projection on.
    /// Use [`SimConfig::standard`] or [`SimConfig::earth`] in practice -- they set the
    /// important physical parameters (grid_res, dt, gravity) from arguments.
    fn default() -> Self {
        Self {
            grid_res: 64,
            grid_cell_size: 1.0,
            dt: 1.0,
            adaptive_timestep: true,
            cfl_include_affine_speed: true,
            cfl_coefficient: 0.9,
            material_cfl_coefficient: 0.5,
            viscous_timestep_coefficient: 0.5,
            rod_cfl_coefficient: 0.4,
            min_dt: 1.0e-3,
            project_invalid_state: true,
            projection_min_density: 1.0e-6,
            projection_min_volume: 1.0e-6,
            projection_min_deformation_j: 1.0e-6,
            gravity: Vec2::new(0.0, -0.05),
            light_dir: Vec2::new(0.0, 1.0),
            boundary_thickness: 2,
            default_initial_volume: 1.0,
            recompute_density_each_step: false,
            grid_density: 1.0,
            reference_density_kg_m3: 1000.0,
            max_substeps_per_step: 64,
            fluid_step_retry_enabled: false,
            phase_rules_once_per_step: false,
            fluid_step_retry_threshold: 0.5,
            fluid_near_wall_cfl_scale: 1.0,
            fluid_near_wall_compression_threshold: 0.01,
            fluid_near_wall_compression_mach_margin: 2.0,
            apic_blend: 1.0,
            j_max: 50.0,
            j_min: 1.0 / 50.0,
            sleep_threshold: 0.0,
            rod_sleep_threshold: 0.0,
            contact_friction: 0.5,
            asflip_blend: 0.0,
            cundall_damping: 0.0,
            mixture_drag_coefficient: 0.0,
            mixture_pressure_iterations: 0,
            fluid_pressure_iterations: 0,
            spatial_sort_enabled: false,
            dx_meters: 1.0,
            dt_seconds: 1.0,
            implicit_corotated_elastic: false,
        }
    }
}

impl SimConfig {
    /// Simulation-ready config: sets the three physical parameters that differ per sim.
    ///
    /// Inherits safe defaults from `Default` (adaptive timestepping, state projection on).
    pub fn standard(grid_res: usize, dt: f32, gravity: Vec2) -> Self {
        Self {
            grid_res,
            dt,
            gravity,
            ..Self::default()
        }
    }

    /// Stripped-down config with adaptive timestepping and state projection disabled.
    ///
    /// Use only for: unit tests that need exact deterministic substeps, benchmarks
    /// where you want to measure a fixed workload, or comparing against an external reference.
    /// Never use for real simulations -- J can go negative and NaN-cascade.
    pub fn unsafe_defaults() -> Self {
        Self {
            adaptive_timestep: false,
            project_invalid_state: false,
            ..Self::default()
        }
    }

    /// Earth-scale simulation preset.
    ///
    /// Derives gravity and unit scaling from real physical constants so that
    /// material parameters passed via `lame_from_si` produce correct behaviour.
    ///
    /// # Arguments
    /// * `grid_res`    -- number of cells per side
    /// * `cell_m`      -- physical size of one grid cell in metres (e.g. `0.01` for 1 cm)
    /// * `dt`          -- frame time step in simulation seconds (e.g. `0.05`)
    ///
    /// # Derived values
    /// `gravity_solver = 9.81 / cell_m` cells/s² (downward, −Y).
    ///
    /// # Example
    /// ```rust,no_run
    /// # extern crate emerge_engine as emerge;
    /// # use emerge::SimConfig;
    /// // 64-cell domain, 1 cm/cell → g = 981 cells/s²
    /// let config = SimConfig::earth(64, 0.01, 0.05);
    /// ```
    pub fn earth(grid_res: usize, cell_m: f32, dt: f32) -> Self {
        // g [cells/s²] = 9.81 [m/s²] / cell_m [m/cell]
        // Derived from v += gravity * sub_dt where sub_dt is in real seconds.
        let g_solver = 9.81 / cell_m;
        Self {
            dx_meters: cell_m,
            dt_seconds: dt,
            ..Self::standard(grid_res, dt, Vec2::new(0.0, -g_solver))
        }
    }

    // ── SI conversion helpers ─────────────────────────────────────────────────

    /// Convert SI Young's modulus (Pa) + Poisson ratio to grid-unit Lamé parameters.
    ///
    /// Equivalent to `lame_from_si(e_pa, nu, rho, self.dx_meters, self.dt_seconds)`.
    /// Requires `earth()` or explicit `dx_meters`/`dt_seconds` to be meaningful.
    pub fn lame_from_si_cfg(&self, e_pa: f32, nu: f32, rho_kg_m3: f32) -> (f32, f32) {
        crate::materials::lame_from_si(e_pa, nu, rho_kg_m3, self.dx_meters, self.dt_seconds)
    }

    /// Dimensionally-correct SI -> grid Lame conversion for this config.
    /// Prefer this over [`Self::lame_from_si_cfg`] for any new scene -- see
    /// [`crate::materials::lame_from_si_physical`] for the measured evidence
    /// (the older path's `dt^2` makes stiffness timestep-dependent, which is
    /// what makes real gravity crush everything and forces each demo to
    /// carry its own `gravity_fraction` fudge).
    ///
    /// Deliberately does NOT read `self.dt_seconds`: a converted stiffness
    /// must not depend on the timestep.
    pub fn lame_from_si_physical_cfg(&self, e_pa: f32, nu: f32, rho_kg_m3: f32) -> (f32, f32) {
        crate::materials::lame_from_si_physical(e_pa, nu, rho_kg_m3, self.dx_meters)
    }

    /// Dimensionally-correct SI stress -> grid conversion: `p_SI / (rho *
    /// dx^2)`, which is exactly the squared wave speed in cells/s. Prefer
    /// over [`Self::stress_from_si`] for any new scene; see
    /// [`crate::materials::lame_from_si_physical`] for why the older one's
    /// `dt^2` is wrong.
    ///
    /// Real, disclosed correction (2026-08-29): pair this ONLY with lambda/mu
    /// that came from [`Self::lame_from_si_physical_cfg`] (the SAME `rho*dx^2`
    /// division) -- a material whose lambda/mu instead came from the raw,
    /// density-agnostic `lame_from_young`/`from_young_modulus` family must
    /// get its cohesion/tensile-strength/viscosity assigned raw SI too, not
    /// run through this. Mixing the two (raw lambda/mu + this conversion on
    /// a term added into the SAME stress tensor) was exactly the bug found
    /// and fixed in `RankineMaterial::ice`'s real call sites this session --
    /// see `q_factor_elastic_viscosity_pa_s`'s own doc for the full writeup.
    pub fn stress_from_si_physical(&self, pa: f32, rho_kg_m3: f32) -> f32 {
        pa / (rho_kg_m3 * self.dx_meters * self.dx_meters)
    }

    /// Dimensionally-correct SI dynamic viscosity -> grid conversion:
    /// `eta_SI / (rho * dx^2)`, i.e. kinematic viscosity in cells^2/s.
    ///
    /// Derived from how viscosity is consumed (`fluid.rs`: `stress +=
    /// eff_viscosity * strain_dev`, strain rate in 1/s, grid stress in
    /// cells^2/s^2). The older [`Self::visc_from_si`] is
    /// `eta * rho * dx^2 / dt^3` -- it multiplies by `rho` and `dx^2` where
    /// it must divide, and carries a spurious `dt^3`.
    ///
    /// Real, disclosed correction (2026-08-29): same pairing rule as
    /// [`Self::stress_from_si_physical`]'s own doc -- only pair this with
    /// lambda/mu from [`Self::lame_from_si_physical_cfg`], never with the
    /// raw `lame_from_young`/`from_young_modulus` family.
    pub fn visc_from_si_physical(&self, eta_pa_s: f32, rho_kg_m3: f32) -> f32 {
        eta_pa_s / (rho_kg_m3 * self.dx_meters * self.dx_meters)
    }

    /// Convert SI stress or pressure (Pa) to grid units.
    ///
    /// Use for: yield stress, tensile strength, eos_stiffness, surface tension.
    /// Scale: `p_grid = p_SI · dt² / (ρ · dx²)`
    pub fn stress_from_si(&self, pa: f32, rho_kg_m3: f32) -> f32 {
        pa * self.dt_seconds * self.dt_seconds / (rho_kg_m3 * self.dx_meters * self.dx_meters)
    }

    /// Convert SI dynamic viscosity (Pa·s) to grid units.
    ///
    /// Viscosity multiplies the velocity gradient (units: 1/step in grid space), so its
    /// non-dimensionalization has one extra factor of dt versus stress:
    /// `η_grid = η_SI · ρ · dx² / dt³`
    pub fn visc_from_si(&self, eta_pa_s: f32, rho_kg_m3: f32) -> f32 {
        eta_pa_s * rho_kg_m3 * self.dx_meters * self.dx_meters
            / (self.dt_seconds * self.dt_seconds * self.dt_seconds)
    }

    /// Validate solver-side numerical and domain constraints.
    pub fn validate(&self) {
        assert!(self.grid_res >= 4, "grid_res must be >= 4");
        assert!(self.grid_cell_size > 0.0, "grid_cell_size must be positive");
        assert!(self.dt > 0.0, "dt must be positive");
        assert!(
            self.cfl_coefficient > 0.0,
            "cfl_coefficient must be positive"
        );
        assert!(
            self.material_cfl_coefficient > 0.0,
            "material_cfl_coefficient must be positive"
        );
        assert!(
            self.rod_cfl_coefficient > 0.0,
            "rod_cfl_coefficient must be positive"
        );
        assert!(
            self.viscous_timestep_coefficient > 0.0,
            "viscous_timestep_coefficient must be positive"
        );
        assert!(self.min_dt > 0.0, "min_dt must be positive");
        assert!(self.min_dt <= self.dt, "min_dt must be <= dt");
        assert!(
            self.projection_min_density > 0.0,
            "projection_min_density must be positive"
        );
        assert!(
            self.projection_min_volume > 0.0,
            "projection_min_volume must be positive"
        );
        assert!(
            self.projection_min_deformation_j > 0.0,
            "projection_min_deformation_j must be positive"
        );
        assert!(self.grid_density > 0.0, "grid_density must be positive");
        assert!(
            self.reference_density_kg_m3 > 0.0,
            "reference_density_kg_m3 must be positive"
        );
        assert!(
            self.contact_friction >= 0.0,
            "contact_friction must be non-negative"
        );
        assert!(
            self.max_substeps_per_step > 0,
            "max_substeps_per_step must be > 0"
        );
        assert!(
            self.default_initial_volume > 0.0,
            "default_initial_volume must be positive"
        );
        assert!(self.j_max > 1.0, "j_max must be > 1.0");
        assert!(
            self.j_min > 0.0 && self.j_min < 1.0,
            "j_min must be in (0.0, 1.0)"
        );
        assert!(
            (0.0..=1.0).contains(&self.apic_blend),
            "apic_blend must be in [0, 1]"
        );
        assert!(
            self.boundary_thickness > 0 && self.boundary_thickness < self.grid_res - 1,
            "boundary_thickness must be in [1, grid_res-2]"
        );
    }
}

// `SpawnRegion` (initial particle layout) + its `SpawnShape` mask and fluent
// builder methods live in spawn.rs -- see that file's own doc comment.
// Re-exported here so every existing `crate::solver::config::SpawnRegion`/
// `SpawnShape` path (and the crate-root `emerge::SpawnRegion`/`SpawnShape`
// re-export in lib.rs) keeps resolving unchanged.
mod spawn;
pub use spawn::{SpawnRegion, SpawnShape};

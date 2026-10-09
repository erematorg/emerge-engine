use glam::Vec2;

/// D⁻¹ = 4.0 for the quadratic B-spline MLS-MPM kernel (always).
/// Not a tunable parameter -- hardcoded from Hu 2018 Table 1.
pub(crate) const KERNEL_D_INVERSE: f32 = 4.0;

/// Default of `SimConfig::material_cfl_coefficient`: the fraction of an
/// explicit scheme's stability limit a step may use. Also what a diffusing
/// operator created on its own (not yet attached to a `Simulation`)
/// sub-cycles at.
pub const DEFAULT_MATERIAL_CFL_COEFFICIENT: f32 = 0.5;

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
    /// bounds use a tighter 0.5 instead, the fraction the rod solver's
    /// own bound also takes).
    pub cfl_coefficient: f32,
    /// The fraction of the acoustic stability limit a substep may use:
    /// `dt <= material_cfl_coefficient * cell_width / c_p`, with
    /// `c_p = sqrt((lambda + 2 mu) / rho)` evaluated per particle from its
    /// own density and each material's own `timestep_bound`, the smallest
    /// winning.
    ///
    /// The limit is the coefficient at which an explicit step stops being
    /// stable, measured near 1.0 (`examples/cpu/cfl_impact_probe.rs`): an
    /// elastic block dropped onto a floor from 10 to 30 cm, flat and on a
    /// corner, holds at 0.90 and breaks at 1.00 when every particle's volume
    /// is the lattice's, and breaks at 1.14, 1.10 and 1.05 for Poisson
    /// ratios 0.3, 0.45 and 0.49 with the volume estimated at spawn. The
    /// default, 0.5, uses half of it: a margin of 2.0 to 2.3 over those
    /// cases, chosen for that margin rather than tuned to a scene. Fluids,
    /// granular materials and stiffer solids have not been measured against
    /// it. A scene should not lower its cost by raising this; a material
    /// validated at a larger fraction is the place to say so, with its
    /// measurement.
    pub material_cfl_coefficient: f32,
    /// Safety factor on the VISCOUS (diffusive) timestep bound for
    /// viscosity-bearing fluid materials (Newtonian/Bingham) -- a diffusion-
    /// type stability condition (`dt <= C * dx^2 / nu`-shaped, distinct
    /// power of `dx` from the acoustic bound above). Same conservative-
    /// margin reasoning and same 0.5 value as `material_cfl_coefficient`
    /// (viscous diffusion instability is likewise an immediate blowup, not
    /// a mild overshoot).
    pub viscous_timestep_coefficient: f32,
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
    /// (see that struct's doc). A FIXED, externally-set vector, NOT a
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
    /// Base grid density: particle mass PER CELL AREA, not per particle.
    /// A spawn's actual particle mass is `grid_density * spacing^2`, because a
    /// lattice at `spacing` cells carries `1/spacing^2` particles per cell.
    ///
    /// It is a density, not a mass, for a dimensional reason. `lame_from_si`
    /// converts SI stress to grid units by dividing by
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
    /// Preflight/retry for strict WC-MPM fluid materials, CPU-side: CFL picks a substep dt from
    /// the PREVIOUS substep's state, which cannot perfectly bound an inherently NONLINEAR
    /// Tait-EOS pressure spike (`eos_power` typically 7) that a fast local compression
    /// event -- confirmed specifically at a hard-wall contact -- can produce WITHIN the
    /// very substep CFL is trying to bound. Verified this is a genuine, convergent
    /// stability limit, not a discrete conservation bug: tightening `material_cfl_
    /// coefficient` GLOBALLY to 1000x default measurably converges toward correct
    /// momentum conservation (not a plateau), but at a cost no real-time scene can
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
    /// something that varies per substep (velocity, position), which
    /// is exactly why this is a caller's choice rather than a silent change.
    ///
    /// Live-measured on `basic_fluids_gui.rs` (2912 particles, ~18 substeps):
    /// the phase-rule scan was ~3400-4300 us of a ~29000 us step (~12%).
    pub phase_rules_once_per_step: bool,
    /// Tightens the CFL of strict fluid particles within `boundary_thickness`
    /// cells of a wall (the zone the slip clamp treats specially): their
    /// material-timestep contribution uses `material_cfl_coefficient`
    /// divided by this factor.
    ///
    /// It also tightens the gravity CFL bound (`cfl.rs`) for the same
    /// particles, the use that matters with `fluid_pressure_iterations > 0`
    /// (`eos_stiffness = 0`): there the acoustic term of
    /// `NewtonianFluidMaterial::timestep_bound` is zero, so the first use
    /// changes nothing (bit-identical at 5, 20 or off), while the gravity
    /// bound stays active and predictive at rest. With
    /// `fluid_pressure_iterations = 1` it lets the wall-contact column (fluid
    /// touching a wall at spawn, nearly full height) complete 120 frames
    /// without non-finite state. That run is not physically valid: J sits at
    /// the [0.5, 2.0] safety clamp from about frame 20 (see the pressure
    /// projection entry in `KNOWN_LIMITATIONS.md`).
    ///
    /// `1.0` (default) = no scaling on either bound.
    pub fluid_near_wall_cfl_scale: f32,
    /// Gates the acoustic-CFL use of `fluid_near_wall_cfl_scale` on actual
    /// compression, not wall proximity alone: a distance-only trigger fires for
    /// any fluid resting on a floor, and a settled `basic_fluids_gui.rs` crawled
    /// at 3-5 fps.
    ///
    /// It does not gate the gravity-CFL use, which must be predictive: a
    /// compression gate needs J to have drifted first, which fails at the first
    /// substep. Set this to `0.0` when using `fluid_near_wall_cfl_scale` for
    /// the gravity bound (as `fluid_pressure_projection_gui.rs` does).
    ///
    /// Water's bulk modulus (~2.2 GPa) makes any sustained J deviation past
    /// about a percent unphysical, so `0.01` (1%, the default) is a measured
    /// threshold for the acoustic use, not a closed-form one.
    ///
    /// Fallback only, for materials without an acoustic term
    /// (`MaterialModel::rest_acoustic_c2() == None`, e.g. `eos_stiffness = 0.0`
    /// projection fluids). A material with a Tait EOS uses
    /// `fluid_near_wall_compression_mach_margin`: a fixed 1% fits one EOS
    /// stiffness, and a softened EOS (`basic_fluids_gpu.rs`) compresses more
    /// in normal flow, so this threshold tripped on nearly every substep (1346
    /// substeps per frame, ~20x what the acoustic term needs).
    pub fluid_near_wall_compression_threshold: f32,
    /// Near-wall compression gate for a material with a Tait EOS
    /// (`MaterialModel::rest_acoustic_c2() == Some(c2_rest)`): compares `|J-1|`
    /// with `(last_max_particle_speed / sqrt(c2_rest))² *
    /// fluid_near_wall_compression_mach_margin`, the compression this EOS
    /// predicts as normal at the scene's current flow speed, times a margin.
    ///
    /// Grounded in the WCSPH relation `Ma² ≈ Δρ` (Monaghan 1994; Morris et
    /// al. 1997), which `weakly_compressible` uses to size `eos_stiffness`.
    /// Applied here from the measured speed, so the gate scales with a
    /// softened EOS instead of borrowing a stiffer EOS's threshold. Informed
    /// by Zhang et al., "A variable speed of sound formulation for weakly
    /// compressible SPH" (arXiv:2310.04139), whose sound speed follows the
    /// measured flow (`c_s(n+1) = max(10·v_max(n), ...)`) on the same
    /// relation; adapted to the gate, since a dynamic `eos_stiffness` would
    /// need per-substep material mutation.
    ///
    /// No published value exists for this margin (the application is this
    /// engine's): `2.0` (default) fires once compression exceeds twice what
    /// the flow speed predicts; 1.0 would fire on any compression.
    pub fluid_near_wall_compression_mach_margin: f32,
    /// Per-substep admissible `|ln(J_new/J_old)|` for a strict fluid particle before
    /// `fluid_step_retry_enabled` rejects and retries that substep. Not the
    /// general `deformation_gradient_cfl_bound` margin (`cfl_coefficient`,
    /// ~0.5), which never fired on the failing scene: the failure is many
    /// small, same-direction changes accumulating over hundreds of substeps,
    /// not one jump. Measured, not derived; calibrate against a reproduction
    /// before changing it.
    pub fluid_step_retry_threshold: f32,
    // No regional substepping switches: the implementation they gated is gone,
    // and `examples/gpu/regional_substep_feasibility_check.rs` found its
    // precondition, a calm region beside a violent one, absent from the fluid
    // scenes (0% of populated blocks calm through the violent window, on both
    // DamBreak and DropletImpact). See `KNOWN_LIMITATIONS.md` entry 2. A
    // rebuild should follow Fang, Hu, Hu & Jiang, "A Temporally Adaptive
    // Material Point Method with Regional Time Stepping," SCA 2018, on a scene
    // where the precondition holds.
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
    /// Emergency projection ceiling for solid/plastic and strict fluid states,
    /// applied every substep (`do_substep`'s pre-P2G pass). A strict fluid's
    /// retry only catches one bad substep; under pressure projection
    /// (`eos_stiffness = 0`, no elastic backstop) J drifted one way over
    /// hundreds of small substeps (min_j 0.72 -> 0.0000154 over 120 frames,
    /// density ratio 64,916x) without `fluid_step_retry_threshold` firing. A
    /// generous safety ceiling (50x volume, water's bulk modulus allows ~1%),
    /// not a physical bound, so it never touches a well-behaved scene. See
    /// `j_min` for the floor.
    pub j_max: f32,
    /// Symmetric floor to `j_max` (`1.0/j_max` by convention, same 50x
    /// magnitude) -- the compression-direction half of the same real gap
    /// `j_max`'s doc describes. Applied every substep for strict fluids
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
    /// `choose_substep_dt` -- the cost driver for many simultaneous rods.
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
    /// MPM, e.g. Anura3D). Material-agnostic fix for the mismatch an
    /// explicit-dynamic MPM solver has with an inherently quasi-static problem
    /// (a granular pile creeping toward equilibrium): damps the component of
    /// each grid cell's velocity change THIS substep (a proxy for applied
    /// force, since Δv = F·dt/m at fixed dt/mass) that opposes nothing but its
    /// own oscillation -- proportional to the FORCE just applied, not to
    /// velocity itself (that's ordinary viscous damping, a different real
    /// mechanism already available via `ViscoelasticMaterial`). Self-gating by
    /// construction: a cell with zero velocity has nothing to oppose (zero
    /// damping), and steady DIRECTED motion (a creature walking, a fluid
    /// splash) barely engages it -- only wobble/settling does. Lives at
    /// the grid level, not inside any one material's constitutive law, so
    /// every material benefits once enabled, not just granular ones.
    /// 0.0 = disabled (default) -- no velocity snapshot taken, byte-identical
    /// to every existing scene, same zero-cost convention as `asflip_blend`.
    pub cundall_damping: f32,
    /// N-phase mixture coupling drag coefficient (generalizes Tampubolon et al.
    /// 2017, "Multi-species simulation of porous sand and water mixtures" --
    /// Darcy-style momentum exchange between materials wrapped in different
    /// `MixturePhase` slots, see `WithMixturePhase`). Units: mass/time (a per-node drag rate,
    /// not the paper's permeability-derived `c_E`; mapping it to soil
    /// permeability and porosity is future work).
    /// 0.0 = disabled (default) -- `Grid::has_mixture_activity()` gates the extra
    /// P2G scatter and the whole resolve pass, zero cost for every scene that
    /// doesn't use `WithMixturePhase`, matching `asflip_blend`'s own convention.
    pub mixture_drag_coefficient: f32,
    /// Jacobi iterations for the mixture incompressibility pressure projection
    /// (`Grid::project_mixture_incompressibility`, see its doc for the
    /// derivation and citations). The drag coupling conserves momentum but does
    /// not enforce the mixture's incompressibility, so under sustained confined
    /// loading (water settled into sand) the violation grows until velocities
    /// pass the CFL bound. 0 = disabled (default). An approximate Jacobi solve,
    /// not an exact Poisson solve: measure the count against the scene's
    /// long-settle behaviour (a settled, confined liquid is the worst case).
    pub mixture_pressure_iterations: u32,
    /// Number of outer correction passes per substep for strict (single-phase,
    /// non-mixture) fluid incompressibility pressure projection
    /// (`Grid::project_fluid_incompressibility`). A stiff Tait EOS needs a tiny
    /// acoustic-CFL step, too slow for sustained wall contact; a Chorin-style
    /// projection (Bridson; the family of `mixture_pressure_iterations`), solved
    /// exactly by a discrete cosine transform (Stam 1999), enforces
    /// incompressibility as a constraint instead. Set the fluid's
    /// `eos_stiffness` to 0.0 with it (allowed, see
    /// `fluid_state::tait_pressure`'s `>= 0.0` contract) so the pressure is not
    /// counted twice.
    ///
    /// The DCT solve is exact; this is the outer repeat count, each pass
    /// re-measuring the residual divergence and correcting again (as in
    /// PISO/SIMPLE solvers: one projection is a first-order splitting of a
    /// violent state). 1 is a reasonable default once enabled.
    ///
    /// Limitation: for fluid spanning nearly the full height and touching a
    /// wall at spawn, one particle's J still drifts to an extreme near the
    /// wall and eventually exhausts the substep budget or crashes. Gauss-Seidel
    /// converges to the same extreme pressure as the DCT, so it is the true
    /// solution of the uniform-density equation for that input. More passes
    /// delay it (5: ~16 frames; 10: ~34; 20: ~29 at much higher cost) without
    /// removing it. A fix needs a variable-density formulation with the free
    /// surface handled structurally (e.g. apic2d's solid-fraction weights,
    /// `tmp/apic2d/apic2d/fluidsim.cpp`).
    ///
    /// Not only the wall case: a falling droplet with no wall anywhere also
    /// drives J to the clamp, at frame 1 at full gravity and at frame 6-7 at
    /// 0.003 g. The mechanisms measured so far (fake divergence at empty
    /// cells, a 1/nodal-mass correction against an average-density solve,
    /// the 0.2 relaxation masking an unstable operator, and node
    /// classification by mass thresholds at walls) are listed in the
    /// pressure projection entry of `KNOWN_LIMITATIONS.md`. Off by default.
    ///
    /// When nonzero, `Simulation::step` requires every particle to be a strict
    /// fluid (`owns_deformation_volume_state() == true`): fluid mixed with sand
    /// or solids on one grid would need per-cell fluid-fraction bookkeeping
    /// like `mixture_cells`, not built for this case.
    pub fluid_pressure_iterations: u32,

    /// Opt-in P2G spatial sort (see `spatial_sort_order`/
    /// `scatter_particles_to_grid_sorted` in `spacetime::transfer::p2g`):
    /// periodic particle reordering for cache and hashmap locality (Gao et al.
    /// 2018, SIGGRAPH Asia), what the GPU path does with `particle_sort.wgsl`'s
    /// indirection array. Default `false`. It costs an O(N log N) sort every
    /// step against a scene-dependent locality gain; enable it after measuring
    /// on the target scene.
    pub spatial_sort_enabled: bool,

    // ── Physical unit scaling ──────────────────────────────────────────────────
    // Default 1.0 = simulation units (no scaling). Set these to enable SI-calibrated materials
    // (`FromSI::from_physical`, or `lame_from_si` / `gravity_to_grid` in `materials::utils`).
    /// Physical length of one grid cell in meters. Default 1.0 (grid units).
    ///
    /// Example: if the simulation domain is 64 cells representing 0.64 m, set `dx_meters = 0.01`.
    pub dx_meters: f32,

    /// Out-of-plane thickness, in metres, of the slab the 2D scene stands
    /// for. `None` (the default) means the scene has not stated one.
    ///
    /// The solver itself never needs it: 2D mechanics works per unit depth.
    /// It matters wherever a real 3D quantity meets a 2D particle: a
    /// particle's radiating face against the mass behind it
    /// (`ThermalConfig::emissivity`), a grid mass turned into kilograms
    /// ([`Self::gravitational_constant_from_si`]), and the path light takes
    /// through the slab (`render::PhysicalRenderContractParams::slice_thickness_m`).
    /// Each of those panics without it rather than assume a thickness, as
    /// each once did (1 mm, 1 m, and none, all different).
    pub slice_thickness_m: Option<f32>,

    /// Opt-in implicit (Newton-CG) grid-velocity update (see
    /// `spacetime::solver::implicit_corotated`: Klar 2016 operator split, an
    /// implicit elastic solve on the shared Corotated branch, then each
    /// material's plastic return mapping once). Default `false`.
    ///
    /// Engages only for a substep where every active particle's material uses
    /// that branch (`DruckerPrager`, `Corotated`, `VonMises`, `Rankine`,
    /// `DruckerPragerMuI`) and no rods, grains, multi-field contact or mixture
    /// coupling are active; otherwise, or when the solve does not converge,
    /// `do_substep` runs the explicit pipeline, so it is always safe to turn on.
    /// At `basic_sand`'s scale it currently always falls back (see that
    /// module's status section).
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
            material_cfl_coefficient: DEFAULT_MATERIAL_CFL_COEFFICIENT,
            viscous_timestep_coefficient: 0.5,
            min_dt: 1.0e-3,
            project_invalid_state: true,
            projection_min_density: 1.0e-6,
            projection_min_volume: 1.0e-6,
            projection_min_deformation_j: 1.0e-6,
            gravity: Vec2::new(0.0, -0.05),
            light_dir: Vec2::new(0.0, 1.0),
            boundary_thickness: 2,
            default_initial_volume: 1.0,
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
            slice_thickness_m: None,
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
    /// Sets gravity and the unit scale from the cell size, so materials built
    /// from SI values (`FromSI::from_physical`, `lame_from_si`) match it.
    ///
    /// # Arguments
    /// * `grid_res`    -- number of cells per side
    /// * `cell_m`      -- physical size of one grid cell in metres (e.g. `0.01` for 1 cm)
    /// * `dt`          -- frame time step in simulation seconds (e.g. `0.05`)
    ///
    /// # Derived values
    /// `gravity_solver = EARTH_GRAVITY_M_S2 / cell_m` cells/s² (downward, −Y).
    ///
    /// # Example
    /// ```rust,no_run
    /// # extern crate emerge_engine as emerge;
    /// # use emerge::SimConfig;
    /// // 64-cell domain, 1 cm/cell → g = 981 cells/s²
    /// let config = SimConfig::earth(64, 0.01, 0.05);
    /// ```
    pub fn earth(grid_res: usize, cell_m: f32, dt: f32) -> Self {
        // g [cells/s²] = g [m/s²] / cell_m [m/cell], since v += gravity * sub_dt
        // with sub_dt in seconds.
        let g_solver = crate::fields::EARTH_GRAVITY_M_S2 / cell_m;
        Self {
            dx_meters: cell_m,
            ..Self::standard(grid_res, dt, Vec2::new(0.0, -g_solver))
        }
    }

    // ── SI conversion helpers ─────────────────────────────────────────────────
    // Grid stress is SI stress over `rho dx^2` (a squared speed in cells/s),
    // never dt-dependent. Pair stress and viscosity conversions only with
    // Lamé parameters from `lame_from_si`: the same `rho dx^2` must divide
    // every term added into one stress tensor.

    /// SI Young's modulus (Pa) and Poisson's ratio to grid Lamé parameters,
    /// through [`crate::materials::lame_from_si`] at this config's `dx_meters`.
    pub fn lame_from_si(&self, e_pa: f32, nu: f32, rho_kg_m3: f32) -> (f32, f32) {
        crate::materials::lame_from_si(e_pa, nu, rho_kg_m3, self.dx_meters)
    }

    /// SI stress or pressure (Pa) to grid units: `p / (rho dx^2)`.
    pub fn stress_from_si(&self, pa: f32, rho_kg_m3: f32) -> f32 {
        pa / (rho_kg_m3 * self.dx_meters * self.dx_meters)
    }

    /// SI dynamic viscosity (Pa s) to grid units: `eta / (rho dx^2)`, the
    /// kinematic viscosity in cells^2/s (grid stress += viscosity * strain
    /// rate in 1/s).
    pub fn visc_from_si(&self, eta_pa_s: f32, rho_kg_m3: f32) -> f32 {
        eta_pa_s / (rho_kg_m3 * self.dx_meters * self.dx_meters)
    }

    /// The stated slice thickness, for a feature that cannot work without
    /// one. Panics naming `feature` when [`Self::slice_thickness_m`] is
    /// unset or not a positive finite length.
    pub fn require_slice_thickness_m(&self, feature: &str) -> f32 {
        match self.slice_thickness_m {
            Some(l) if l.is_finite() && l > 0.0 => l,
            other => panic!(
                "{feature} needs SimConfig::slice_thickness_m, the out-of-plane thickness in \
                 metres the 2D scene stands for; got {other:?}"
            ),
        }
    }

    /// Real gravitational constant (`N m^2 / kg^2`) to the grid units the
    /// gravity fields apply it in: positions in cells, grid masses
    /// (`(rho / reference_density_kg_m3) * cells^2`), accelerations in
    /// cells/s^2.
    ///
    /// A grid mass is `m_SI / (reference_density_kg_m3 * dx_meters^2 * L)`
    /// with `L` the slice thickness, and a grid acceleration is the SI one
    /// over `dx_meters`, so `G_grid = G_SI * reference_density_kg_m3 * L /
    /// dx_meters`. Panics without a stated slice thickness.
    pub fn gravitational_constant_from_si(&self, g_si: f32) -> f32 {
        let l = self.require_slice_thickness_m("gravitational_constant_from_si");
        g_si * self.reference_density_kg_m3 * l / self.dx_meters
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
// builder methods live in spawn.rs -- see that file's doc comment.
// Re-exported here so every existing `crate::solver::config::SpawnRegion`/
// `SpawnShape` path (and the crate-root `emerge::SpawnRegion`/`SpawnShape`
// re-export in lib.rs) keeps resolving unchanged.
mod spawn;
pub use spawn::{SpawnRegion, SpawnShape};

pub mod bingham;
pub mod boiling_mixture;
pub mod cavitating_eos;
pub mod cavitating_fluid;
pub mod corotated;
pub mod elastic;
pub mod fluid;
pub mod gas;
pub mod granular;
pub mod granular_fluid;
pub mod nacc;
pub mod no_compression;
pub mod optical;
pub mod params;
pub mod physical_props;
mod property_dispatch;
pub mod rankine;
pub mod registry;
pub mod rod_material;
pub mod snow;
pub(crate) mod svd;
pub mod utils;
pub mod viscoelastic;
pub mod von_mises;

pub use physical_props::{
    BinghamProps, BrittleProps, Elastic, Elastoplastic, Fluid, FluidGranular, FromSI,
    GranularProps, NaccProps, NoCompression, ParticleMass, PlasticityModel, Pressurized,
    Viscoelastic,
};

pub use bingham::BinghamFluidMaterial;
pub use boiling_mixture::BoilingMixtureMaterial;
pub use cavitating_eos::{CavitatingEosParams, CavitatingEosTable};
pub use cavitating_fluid::{
    CavitatingFluidMaterial, CavitatingFluidMaterialParams, IsothermalCavitatingFluidMaterial,
    IsothermalCavitatingFluidMaterialParams,
};
pub use corotated::CorotatedMaterial;
pub use elastic::NeoHookeanMaterial;
pub use fluid::NewtonianFluidMaterial;
pub use gas::{IdealGasMaterial, IdealGasPhysicalParams};
pub use granular::sand::DruckerPragerMaterial;
pub use granular::sand_mui::MuIRheologyMaterial;
pub use granular_fluid::GranularFluidMaterial;
pub use nacc::{NaccMaterial, NaccMaterialParams};
pub use no_compression::NoCompressionMaterial;
pub use params::MaterialParams;
pub use rankine::RankineMaterial;
pub use registry::{MAX_MATERIAL_SLOTS, MaterialRegistry};
pub use rod_material::RodMaterial;
pub use snow::StomakhinMaterial;
pub use utils::{
    elastic_wave_dt, gravity_to_grid, lame_from_si, lame_from_si_physical, lame_from_young,
    polar_decomposition_2d, rankine_damage_estimate, stokes_drag_rate_from_si,
};
pub use viscoelastic::ViscoelasticMaterial;
pub use von_mises::VonMisesMaterial;

use glam::Mat2;

use crate::particle::{Particle, Particles};

/// Identifies which constitutive model a material implements.
/// `repr(u32)` so this discriminant can be stored directly in GPU uniform buffers.
/// Explicit values are stable across recompiles -- do not change them.
#[non_exhaustive]
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstitutiveModel {
    Fallback = 0,
    Fluid = 1,            // Weakly-compressible Newtonian fluid, Tait EOS
    NeoHookean = 2,       // Neo-Hookean hyperelastic (jelly, soft solids)
    Corotated = 3,        // Corotated linear elastic (stiffer baseline)
    Snow = 4,             // Corotated + SVD plasticity (Stomakhin 2013)
    DruckerPrager = 5,    // Corotated elastic + DP yield surface (sand, soil, rock)
    VonMises = 6,         // J2 perfect plasticity -- ductile flow, no hardening (lava, metal, clay)
    Rankine = 7,          // Tensile cutoff + exponential softening -- brittle rock, bone, ice
    DruckerPragerMuI = 8, // Rate-dependent DP -- µ(I) rheology, granular flow
    Viscoelastic = 9,     // Kelvin-Voigt: NeoHookean elastic + viscous dashpot in parallel
    /// Non-Associated Cam-Clay -- wet soil, clay, bio tissue under
    /// compression. Effectively CPU-only for real dynamics: `NaccMaterial::
    /// params()` deliberately uploads `model: NeoHookean as u32` (2)
    /// instead of this discriminant (see that method's own comment), so a
    /// real `NaccMaterial` never reaches `p2g.wgsl`'s `default` arm at
    /// all -- it silently runs `case 2u`'s NeoHookean stress (`kappa*ln(J)`
    /// volumetric term) instead of NACC's own real law (`kappa/2*(J^2-1)`,
    /// see `nacc.rs::kirchhoff_stress`). Issue #5's `needs_cpu_update`
    /// fallback DOES correctly rerun `NaccMaterial::update_particle`'s
    /// real Cam-Clay return-mapping on CPU every frame, keeping
    /// `deformation_gradient`/plastic state on the real yield surface --
    /// but the stress feeding that same substep's P2G grid transfer is
    /// still NeoHookean's, not NACC's. A first attempt at documenting/
    /// guarding this got the mechanism backwards (checked `params().model`
    /// for `10` and concluded "zero stress" -- that check is unreachable
    /// for a real `NaccMaterial`, since `params()` never emits `10`);
    /// corrected after external review caught it. `GpuSimulation::
    /// with_device` checks the real `constitutive_model()` value (not
    /// `params()`) and panics if NACC is present -- use
    /// `GranularFluidMaterial` (already fully GPU-native) for a
    /// granular-fluid-like GPU scene instead. See issue #5 for the real
    /// WGSL-port option, not pursued here.
    Nacc = 10,
    GranularFluid = 11, // Granular-fluid mixture -- Tait EOS + corotated deviatoric + SVD plasticity
    /// Tension-only (no-compression) reversible elastic -- silk, tendons,
    /// membranes. GPU gap, issue #29: `p2g.wgsl`/`particles_update.wgsl`
    /// have no case-12 branch, so an unrecognised `mat.model` falls through
    /// their `default: { return mat2x2<f32>(); }` arm -- exact zero stress
    /// on GPU, with no compensating CPU fallback the way NACC has (issue
    /// #5). `GpuSimulation::with_device` panics if this model is present
    /// (same real-guard pattern as NACC's own fix) rather than silently
    /// running a cable/membrane/tendon with zero tension resistance.
    NoCompression = 12,
    /// Ideal gas EOS (p=ρRT) -- CPU only. GPU shaders (`p2g.wgsl`,
    /// `particles_update.wgsl`) have no case-13 branch yet; an unrecognised
    /// `mat.model` would fall through their `default: { return mat2x2<f32>(); }`
    /// arm to zero stress, so `GpuSimulation` refuses to start with it (see
    /// `MaterialModel::gpu_unsupported_reason`).
    Gas = 13,
}

// WGSL shaders (p2g.wgsl, particles_update.wgsl) index material branches by the
// ConstitutiveModel discriminant cast to u32. These assertions catch any enum reordering
// that would silently run the wrong GPU stress branch on a material.
const _: () = {
    use ConstitutiveModel as C;
    assert!(C::Fallback as u32 == 0);
    assert!(C::Fluid as u32 == 1);
    assert!(C::NeoHookean as u32 == 2);
    assert!(C::Corotated as u32 == 3);
    assert!(C::Snow as u32 == 4);
    assert!(C::DruckerPrager as u32 == 5);
    assert!(C::VonMises as u32 == 6);
    assert!(C::Rankine as u32 == 7);
    assert!(C::DruckerPragerMuI as u32 == 8);
    assert!(C::Viscoelastic as u32 == 9);
    assert!(C::Nacc as u32 == 10);
    assert!(C::GranularFluid as u32 == 11);
    assert!(C::NoCompression as u32 == 12);
    assert!(C::Gas as u32 == 13);
};

/// Cap on simultaneous mixture phases -- see `MixturePhase`'s own doc.
/// Deliberately small (YAGNI): covers solid + fluid + a real 3rd/4th phase
/// (air, a second fluid) without pre-building for a need that doesn't
/// exist yet. Raising it later is a one-line change (`MixtureCell`'s
/// arrays and `resolve_mixture_coupling`'s solve both size off this
/// constant, nothing else needs touching).
pub const MAX_MIXTURE_PHASES: usize = 4;

/// Which of up to `MAX_MIXTURE_PHASES` roles a material plays in N-phase
/// mixture coupling (generalizes Tampubolon et al. 2017, "Multi-species
/// simulation of porous sand and water mixtures" -- interpenetrating
/// granular-fluid Darcy drag, e.g. water soaking into sand, to N
/// simultaneously-tracked phases exchanging real pairwise momentum). This
/// is a MATERIAL-level classification (via `MaterialModel::mixture_phase`),
/// not a per-particle field -- every particle of a given material shares the
/// same phase, matching how `constitutive_model` already works. `None` (the
/// default for every existing material) opts a scene entirely out of mixture
/// coupling at zero cost -- see `Grid::has_mixture_activity`.
///
/// A plain slot index (0..MAX_MIXTURE_PHASES), not a fixed enum -- unlike
/// the render side's `material_id % 16` convention, this does NOT wrap: an
/// out-of-range index is a real configuration error (asserted where used),
/// since silently colliding two unrelated phases into the same slot would
/// corrupt real physics, not just misdraw a pixel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MixturePhase(pub u8);

impl MixturePhase {
    /// The porous solid (e.g. sand/soil) -- keeps its own full elastic/plastic
    /// deformation, unaffected by mixture coupling beyond the drag force itself.
    pub const SOLID: MixturePhase = MixturePhase(0);
    /// The interpenetrating fluid (e.g. water) -- exchanges momentum with the
    /// solid phase via Darcy-style drag at every node both phases touch.
    pub const FLUID: MixturePhase = MixturePhase(1);
}

/// Blanket downcast hook for `MaterialRegistry`'s enum-dispatch fast path
/// (see `registry::MaterialDispatch`). P2G/G2P/CFL call `kirchhoff_stress`/
/// `stress_volume`/`timestep_bound`/`owns_deformation_volume_state` once per
/// particle per substep; going through `dyn MaterialModel`'s vtable every
/// time is real, measured cost at scale. The registry downcasts each
/// registered material to one of the engine's known concrete types ONCE (at
/// `insert`/`set_default` time) and caches a match-dispatched copy;
/// unrecognised types (a wrapper like `WithMixturePhase`, or an LP-side
/// custom material) simply fail every downcast and keep using the trait
/// object as before -- zero behavior change, only unlocks a fast path for
/// materials the engine already knows about.
///
/// Split into its own blanket-impl'd trait (rather than a default method
/// directly on `MaterialModel`) because `fn as_any(&self) -> &dyn Any { self }`
/// as a *default* method on a `Self: ?Sized`-context trait doesn't typecheck
/// (the unsized coercion needs a concrete, Sized `Self`) -- the standard fix
/// (used by e.g. the `downcast-rs` crate) is a supertrait with a blanket
/// `impl<T: Any> AsAny for T`, which every `Sized` material -- including any
/// external/LP-defined one -- gets automatically, no per-material code needed.
pub trait AsAny: core::any::Any {
    fn as_any(&self) -> &dyn core::any::Any;
}

impl<T: core::any::Any> AsAny for T {
    fn as_any(&self) -> &dyn core::any::Any {
        self
    }
}

pub trait MaterialModel: Send + Sync + core::fmt::Debug + AsAny {
    /// Which constitutive law this material implements.
    /// Used by the GPU shader to select the correct stress branch per particle.
    fn constitutive_model(&self) -> ConstitutiveModel {
        ConstitutiveModel::Fallback
    }
    // Returns the Kirchhoff-like stress used by the transfer kernel.
    // The kernel applies geometry/time factors (dt, kernel_d_inverse, cell_dist, weight).
    fn kirchhoff_stress(&self, _particles: &Particles, _i: usize) -> Mat2 {
        Mat2::ZERO
    }

    // Returns the particle volume used in the stress contribution.
    fn stress_volume(&self, particles: &Particles, i: usize) -> f32 {
        particles.initial_volume[i]
    }

    /// CFL timestep bound for one particle. Takes `density`/`hardening_scale` as plain
    /// scalars rather than `&Particles, i: usize` -- every implementation only ever reads
    /// these two fields, both of which exist directly on `Particle` (AoS) too, so the CPU
    /// (SoA) and GPU (AoS) CFL scans can both call this without either one needing the
    /// other's storage representation.
    fn timestep_bound(
        &self,
        _density: f32,
        _hardening_scale: f32,
        _cell_width: f32,
        _material_cfl: f32,
        _viscous_cfl: f32,
    ) -> f32 {
        f32::INFINITY
    }

    /// Speed-of-sound-squared (c²) this material's own acoustic CFL term
    /// evaluates to AT REST (density == rest_density, the Tait-EOS density
    /// ratio's baseline of 1). `None` (the default) for materials with no
    /// meaningful acoustic term, or a strict fluid whose `eos_stiffness` is
    /// deliberately `0.0` (e.g. pressure-projection incompressible fluids --
    /// see `fluid_pressure_projection_gui.rs`). A real Tait-EOS fluid
    /// overrides this with `eos_stiffness * eos_power / rest_density` -- the
    /// SAME formula its own `timestep_bound` already evaluates at
    /// density_ratio=1, not a new derivation.
    ///
    /// Used by `choose_substep_dt`'s near-wall gate to scale its
    /// compression-anomaly threshold to THIS material's own acoustic
    /// stiffness instead of a fixed absolute percentage tuned for a
    /// different EOS: the standard WCSPH relation Ma² ≈ Δρ (density
    /// variation ≈ squared Mach number; Monaghan 1994, Morris et al. 1997)
    /// means the compression a real flow induces scales with
    /// `(v_max / c_s_rest)²`, not a scene-independent constant -- see
    /// `SimConfig::fluid_near_wall_compression_mach_margin`'s own doc, and
    /// Zhang et al., "A variable speed of sound formulation for weakly
    /// compressible SPH" (arXiv:2310.04139), whose own variable-c_s update
    /// rule is built on the identical Ma²≈Δρ relation.
    fn rest_acoustic_c2(&self) -> Option<f32> {
        None
    }

    /// Real, live-temperature-aware acoustic c^2 -- default just returns
    /// `rest_acoustic_c2()` (every existing material's own construction-
    /// time value, unchanged behavior for every material that doesn't
    /// override this). Exists for the one real class of material whose
    /// own true stiffness genuinely tracks a particle's LIVE temperature,
    /// not a fixed reference value baked in at construction --
    /// `IdealGasMaterial`'s own override is the real, disclosed case this
    /// closes (`c^2=gamma*R*T`, and `T` climbs continuously under active
    /// heating in a real scene, e.g. `phase_states_gui.rs`'s boiling
    /// steam). `cfl.rs`'s own dispatch site adds this as a SEPARATE CFL
    /// term (not folded into `timestep_bound`) -- same established
    /// pattern already used there for the shock-viscosity and single-
    /// particle-instability terms, both added the same way rather than
    /// extending `timestep_bound`'s own trait signature.
    fn acoustic_c2_at_temperature(&self, temperature_k: f32) -> Option<f32> {
        let _ = temperature_k;
        self.rest_acoustic_c2()
    }

    /// Real, live density-AND-temperature-jointly-aware acoustic c^2 --
    /// default falls back to `acoustic_c2_at_temperature(temperature_k)`
    /// (which itself falls back to `rest_acoustic_c2()`), so every
    /// existing material -- including `IdealGasMaterial`'s own real
    /// `acoustic_c2_at_temperature` override -- keeps working unchanged
    /// through this same chain, zero behavior change. Exists for the one
    /// real class of material whose derivative depends on BOTH inputs
    /// JOINTLY, not `T` alone: a genuinely temperature-coupled cavitation
    /// EOS's own mixture band and C^1 patches shift with `T` (unlike an
    /// ideal gas's `c^2=gamma*R*T`, which has no real density dependence
    /// at all), so which real branch/patch a given `(density,T)` pair
    /// lands in cannot be answered from `T` alone (external review's own
    /// explicit point: `acoustic_c2_at_temperature(T)` alone is
    /// insufficient here). See a real T-coupled cavitating material's own
    /// override for the real case this closes.
    fn acoustic_c2_at(&self, density: f32, temperature_k: f32) -> Option<f32> {
        let _ = density;
        self.acoustic_c2_at_temperature(temperature_k)
    }

    /// Real, most general tier of this same chain -- default just
    /// forwards to `acoustic_c2_at(particles.density[i],
    /// particles.temperature[i])`, so every existing material (including
    /// every override above) keeps working unchanged. Exists for the one
    /// real class of material whose derivative depends on a per-particle
    /// SCALAR beyond density/temperature -- a real boiling liquid-vapor
    /// mixture's own local stiffness depends on the particle's own mass
    /// quality (`Particle::friction_hardening`, real, disclosed re-use of
    /// that already-established per-material scratch field -- see
    /// `BoilingMixtureMaterial`'s own doc for the real case this closes),
    /// which neither `density` nor `temperature` alone can express.
    /// `cfl.rs`'s own dispatch calls this tier directly (not `acoustic_c2_at`),
    /// same established pattern as every earlier tier in this chain.
    fn acoustic_c2_at_particle(&self, particles: &Particles, i: usize) -> Option<f32> {
        self.acoustic_c2_at(particles.density[i], particles.temperature[i])
    }

    /// Real, currently-computed internal friction ratio at this particle's
    /// own state (`Some`), or `None` when this material has no such concept
    /// (e.g. an elastic solid, or `GranularFluidMaterial`'s SVD-clamp
    /// plasticity, which has no Drucker-Prager cone at all). Used by MIBF
    /// (`FrictionBoundary`'s `use_material_friction`) to make wall friction
    /// reflect the material's own real, spatially-varying state instead of
    /// one fixed constant -- Blatny & Gaume 2025 (`tmp/ref_matter.md`
    /// sec.19), which reports natural angle-of-repose emergence from
    /// exactly this coupling. Default `None`, same "most materials opt
    /// out" shape as `rest_acoustic_c2`/`corotated_lame_params` above.
    fn current_friction_coefficient(&self, _particles: &Particles, _i: usize) -> Option<f32> {
        None
    }

    /// Advances plastic/deformation state for one particle after G2P's velocity
    /// gather. Takes a `ParticleUpdateCtx` (disjoint per-field borrows), not
    /// `&mut Particles, i` -- every real implementation only ever touches its
    /// own particle's fields, so this shape lets G2P run every particle's
    /// update in parallel (see `ParticleUpdateCtx`'s own doc).
    fn update_particle(&self, _ctx: &mut crate::particle::ParticleUpdateCtx, _dt: f32) {}

    /// Seed per-particle plastic state at spawn time.
    ///
    /// Called once per particle immediately after position/volume assignment.
    /// Default: no-op (elastic materials need no initial plastic state).
    /// Override for materials that have a non-zero neutral accumulator (e.g. sand).
    fn init_particle(&self, _particle: &mut Particle) {}

    /// Seed per-particle state when TRANSITIONING into this material from
    /// another (via `Simulation::phase_transition`/`add_phase_rule`), as
    /// opposed to a fresh spawn. Default: delegates to `init_particle`
    /// unchanged -- exactly today's existing behavior for every material
    /// that doesn't override this, zero behavior change.
    ///
    /// Real, found live 2026-08-18 (`examples/basic_steam.rs`, water
    /// boiling into `IdealGasMaterial` steam): `Simulation::
    /// apply_phase_transition` (`spacetime::solver::particles`) already
    /// rebaselines a transitioning particle to its real, continuous prior
    /// state (F=IDENTITY, `initial_volume`=its actual current volume)
    /// before calling this. For a material whose own fresh-spawn
    /// analytical state (`init_particle`'s own `mass/rest_density`
    /// formula) is close to what it's transitioning FROM, blindly
    /// overwriting that rebaseline is harmless (e.g. water->ice, similar
    /// real densities) -- but for a material transitioning from something
    /// with a dramatically different rest density (water->steam, a real
    /// ~1700x ratio), it makes the particle's claimed VOLUME jump that
    /// same ~1700x in a single instant, injecting a real but wildly
    /// under-resolved force spike (P2G's `stress*volume*kernel_gradient`
    /// scatters that huge volume at the particle's own, unmoved grid
    /// location) -- confirmed live as the direct cause of a real crash.
    ///
    /// Override this (leaving `init_particle` itself untouched for the
    /// fresh-spawn case) when a material's rest state can differ enough
    /// from whatever it might be transitioning from that continuity, not
    /// a fresh analytical reset, is the physically honest choice -- see
    /// `IdealGasMaterial`'s own override for the real, worked pattern (keep the
    /// reference volume TRUE, matching what per-substep dynamics already
    /// assume, and instead set the STARTING deformation gradient to
    /// reflect real compression relative to that true reference, clamped
    /// to the material's own valid range).
    fn init_particle_from_transition(&self, particle: &mut Particle) {
        self.init_particle(particle)
    }

    /// Whether `update_particle` does real work on the CPU.
    ///
    /// Return `false` if plasticity is fully handled on GPU (default).
    /// Return `true` for CPU-only plasticity paths -- the GPU solver uses this to
    /// decide whether to download particles and run the CPU pass each frame.
    fn needs_cpu_update(&self) -> bool {
        false
    }

    /// Why `GpuSimulation` cannot run this material, or `None` when the GPU
    /// shaders implement its law. The GPU solver refuses to start with any
    /// material that gives a reason, instead of running another law in its
    /// place. The reason starts with the material's type name.
    fn gpu_unsupported_reason(&self) -> Option<&'static str> {
        None
    }

    /// Whether this material consumes an optional kernel-density measurement.
    ///
    /// This is for models whose constitutive law explicitly uses that sampled
    /// field. Strict WC-MPM liquids do *not*: their EOS state is
    /// `rho = rho0 / J`, owned together with `V = V0 J`; see
    /// `owns_deformation_volume_state` below. Default: false.
    fn needs_density_recompute(&self) -> bool {
        false
    }

    /// Whether this material owns density and current volume through its
    /// deformation state.  Such materials must not have those values replaced
    /// by a kernel-density gather, whose free-surface bias is a measurement
    /// artifact rather than a constitutive update.
    fn owns_deformation_volume_state(&self) -> bool {
        false
    }

    /// Which two-phase mixture role this material plays, if any -- see
    /// `MixturePhase`'s own doc. `None` (default) means this material never
    /// participates in mixture coupling, the zero-cost-when-unused case that
    /// covers every material/scene that doesn't need this feature.
    fn mixture_phase(&self) -> Option<MixturePhase> {
        None
    }

    /// `Some((lambda, mu))` when this material's `kirchhoff_stress` is
    /// EXACTLY `corotated_elastic_stress(F, lambda, mu)` -- no rate term, no
    /// thermal/activation modifier -- so `spacetime::solver::
    /// implicit_corotated`'s Newton-CG (which evaluates that shared formula
    /// and its JVP at arbitrary trial `F`, not just the particle's current
    /// one) can stand in for this material's elastic response exactly.
    /// `None` (default) opts a material out -- the implicit path falls back
    /// to the normal explicit substep for any scene containing it, so this
    /// is a real safety gate, not a performance hint.
    ///
    /// Verified equivalence for the five real materials that override this
    /// (`tests/scratch_implicit_mpm_stage2_shared_elastic_branch_check.rs`,
    /// 2026-09-10): DruckerPrager, VonMises, Rankine, and MuIRheology all
    /// call `corotated_elastic_stress` directly with `elastic_viscosity ==
    /// 0.0` required (nonzero adds a real Kelvin-Voigt rate term this branch
    /// doesn't model); Corotated itself requires `thermal_expansion == 0.0`
    /// and `active_stress_coeff == 0.0` for the same reason.
    fn corotated_lame_params(&self) -> Option<(f32, f32)> {
        None
    }

    /// Scaling coefficient for activation-driven deviatoric stress.
    ///
    /// When non-zero, the per-particle `activation` field (0.0–1.0) modulates the
    /// deviatoric component of the Kirchhoff stress. This is the engine-level hook for
    /// active matter: muscles, motile cells, contractile tissue.
    ///
    /// Physics: τ_total = τ_elastic + activation × coeff × I  (contractile active pressure)
    /// Default: 0.0 -- activation has no effect on passive materials.
    fn activation_scale(&self) -> f32 {
        0.0
    }

    /// Scaling coefficient for internal pre-stress pressure.
    ///
    /// When non-zero, the per-particle `internal_pressure` field (already SI-
    /// converted to grid stress units) contributes an isotropic `-P·I` term to
    /// the Kirchhoff stress -- the standard "prestressed structure" treatment
    /// (a balloon: envelope tension balanced against internal gas pressure).
    /// Generic engine-level hook, not plant-specific: real motivating case is
    /// turgor pressure (plants aren't held up by cell-wall elasticity alone --
    /// see Niklas 1992's "hydro-skeleton" theory), but applies to any
    /// internally-pressurized body a material wants to model this way.
    ///
    /// Physics: τ_total = τ_elastic + τ_active − internal_pressure × coeff × I
    /// Default: 0.0 -- pre-stress has no effect on materials that don't opt in
    /// (fluids already carry their own EOS pressure and should not double up).
    fn pressure_scale(&self) -> f32 {
        0.0
    }

    /// Returns this material's parameters as a flat, GPU-uploadable struct.
    /// Default returns zeroed params (Fallback model).
    /// Volume ratio `J` at which this material is in equilibrium under a
    /// given pressure, or `None` if it has no equation of state to invert.
    ///
    /// Matter at rest under gravity is not at uniform density: the pressure
    /// rises with depth, and for a compressible material that pressure IS a
    /// density change. Spawning a pool at uniform density therefore creates
    /// a body with no internal pressure at all, which then collapses under
    /// its own weight until the gradient builds -- a real elastic wave, and
    /// exactly what you see when a "resting" pool twitches on its first
    /// frames.
    ///
    /// Answering this lets a caller spawn the body already in equilibrium.
    /// Only the material knows how, because only it knows its own equation
    /// of state.
    fn hydrostatic_volume_ratio(&self, _pressure: f32) -> Option<f32> {
        None
    }

    /// Light this material emits WITHOUT being hot, as a volumetric source
    /// in `W/m^3`. 0 means it does not glow.
    ///
    /// This is luminescence -- a firefly, a glowing fungus, a chemical
    /// light stick. Thermal emission is a separate, already-handled
    /// mechanism (`energy::radiation::blackbody`): matter that glows because
    /// it is hot needs nothing declared here, its temperature is enough.
    ///
    /// It feeds `S` in the photon diffusion equation
    /// `(1/c) dphi/dt = D grad^2 phi - mu_a phi + S`, which is why the unit
    /// is a power per unit volume and why it must be ISOTROPIC: the
    /// diffusion approximation is only valid for a source radiating equally
    /// in all directions. That holds for a luminous organ or a glowing
    /// mineral; it would not hold for a laser or a directed beam, and those
    /// must not be modelled through this.
    fn luminous_emission_w_m3(&self) -> f32 {
        0.0
    }

    /// Measured optical constants of this substance: absorption per colour
    /// band and reduced scattering, both in `m^-1`. `None` means this
    /// material has not declared any.
    ///
    /// Declaring them is what lets the renderer compute this material's
    /// colour from physics instead of reading a painted palette entry. The
    /// values belong to the substance, so they live with the material and
    /// with `matter::materials::optical`'s measured tables, never in the
    /// renderer.
    ///
    /// Like `specific_heat_j_kg_k`, this is per instance rather than per
    /// constitutive model: a Neo-Hookean solid can be jade or muscle, and
    /// they absorb light very differently.
    fn optical_properties(&self) -> Option<crate::energy::radiation::OpticalCoefficientsSi> {
        None
    }

    /// Specific heat capacity `c_p`, J/(kg*K). 0 means this material has not
    /// declared one.
    ///
    /// Per instance rather than per constitutive model, because the model
    /// does not determine it: a Neo-Hookean solid can be rubber or muscle,
    /// and they store heat very differently. A scene that wants dissipated
    /// work to become a real temperature (see
    /// `energy::thermodynamics::frictional_heating`) declares it on the
    /// material it built; a scene that does not gets no temperature change
    /// rather than an invented one.
    fn specific_heat_j_kg_k(&self) -> f32 {
        0.0
    }

    fn params(&self) -> MaterialParams {
        MaterialParams::default()
    }

    /// Energy cost (J/kg, in whatever temperature unit `Particle::temperature` uses)
    /// of transitioning INTO this material via `Simulation::phase_transition` /
    /// `add_phase_rule`, as a function of `from_material_id` (the particle's own
    /// material immediately before this transition). Positive = endothermic (e.g.
    /// melting into a liquid -- absorbs energy, cooling the particle). Negative =
    /// exothermic (e.g. freezing into a solid -- releases energy, warming the
    /// particle). Default: ignores `from_material_id` and returns 0.0 -- no energy
    /// cost, existing behavior for every material unchanged.
    ///
    /// Real, general extension (2026-08-23): a real substance can be the
    /// destination of MULTIPLE physically distinct transitions with DIFFERENT real
    /// energies -- water is the destination for both melting-in (endothermic,
    /// +334,000 J/kg, real fusion) AND condensing-in (exothermic, -2,257,000 J/kg,
    /// real vaporization) -- two genuinely different values a single flat scalar
    /// per material could never represent (confirmed live, `examples/
    /// phase_states_headless.rs`'s own real solid<->liquid<->gas cycle work).
    /// `from_material_id` is what makes that representable: an implementor reads
    /// it to pick the right real value for whichever transition actually
    /// happened, rather than being limited to one value for every incoming path.
    /// Not gas-specific or demo-specific -- any material (solid, liquid, gas,
    /// mixture) transitioning from more than one real physical source can use
    /// this the same way; see `WithLatentHeatTable` for the real, general,
    /// multi-source implementation (`WithLatentHeat` stays the simple single-
    /// value case, unchanged, for materials with only ever one real incoming
    /// transition).
    ///
    /// Applied in `Simulation::phase_transition`/`add_phase_rule` (CPU) against
    /// `ThermalDiffusion::heat_capacity` when a thermal model is configured, and in
    /// `GpuSimulation::phase_transition` against the `heat_capacity` passed to
    /// `attach_thermal_gpu` -- same debit, same formula, on both. GPU has no automatic
    /// `add_phase_rule` counterpart yet (only the manual, one-shot `phase_transition`);
    /// that gap is real and separate from this energy accounting.
    fn latent_heat(&self, from_material_id: u32) -> f32 {
        let _ = from_material_id;
        0.0
    }

    /// Extra cohesion-like yield resistance (Pa-equivalent, same stress-space
    /// units any material's own cohesion concept already uses) contributed by
    /// a coupled `ScalarDiffusionField` value (`Particle::scalar_field` --
    /// e.g. moisture/saturation, but generic: whatever a scene wires that
    /// carrier to mean).
    ///
    /// Generic engine-level hook, deliberately NOT material-specific: any
    /// material with a real yield surface can define how ITS OWN physics
    /// responds to the shared scalar (e.g. `DruckerPragerMaterial` uses real
    /// capillary-cohesion literature for wet sand). A material without a
    /// meaningful notion of cohesion (a fluid's constitutive law, a purely
    /// elastic solid) simply never overrides this -- that is a material
    /// choosing not to participate, not the engine special-casing it. The
    /// hook itself is universal; only the real physics behind an override is
    /// material-specific, the same relationship `pressure_scale`/
    /// `latent_heat` already establish for their own domains.
    ///
    /// Default: 0.0 -- the shared scalar has no effect on any material that
    /// doesn't opt in, byte-identical to every existing scene.
    fn cohesion_bonus_pa(&self, scalar_field: f32) -> f32 {
        let _ = scalar_field;
        0.0
    }
}

/// The `MaterialModel` methods every delegating wrapper below (`WithLatentHeat`,
/// `WithMixturePhase`, `WithPreStress`) forwards to `self.inner` byte-for-byte.
/// Factored into one macro so these three impls can't drift out of sync -- a new
/// `MaterialModel` method that should default-forward gets added here ONCE, not
/// copy-pasted three times.
///
/// The 3 methods each wrapper actually overrides (`init_particle`, `mixture_phase`,
/// `latent_heat`) are NOT in this list -- each wrapper still writes those by hand,
/// forwarding the two it doesn't override itself.
macro_rules! forward_material_model_common {
    () => {
        fn constitutive_model(&self) -> ConstitutiveModel {
            self.inner.constitutive_model()
        }
        fn kirchhoff_stress(&self, particles: &Particles, i: usize) -> Mat2 {
            self.inner.kirchhoff_stress(particles, i)
        }
        fn stress_volume(&self, particles: &Particles, i: usize) -> f32 {
            self.inner.stress_volume(particles, i)
        }
        fn timestep_bound(
            &self,
            density: f32,
            hardening_scale: f32,
            cell_width: f32,
            material_cfl: f32,
            viscous_cfl: f32,
        ) -> f32 {
            self.inner.timestep_bound(
                density,
                hardening_scale,
                cell_width,
                material_cfl,
                viscous_cfl,
            )
        }
        fn update_particle(&self, ctx: &mut crate::particle::ParticleUpdateCtx, dt: f32) {
            self.inner.update_particle(ctx, dt)
        }
        fn needs_cpu_update(&self) -> bool {
            self.inner.needs_cpu_update()
        }
        fn gpu_unsupported_reason(&self) -> Option<&'static str> {
            self.inner.gpu_unsupported_reason()
        }
        fn needs_density_recompute(&self) -> bool {
            self.inner.needs_density_recompute()
        }
        fn owns_deformation_volume_state(&self) -> bool {
            self.inner.owns_deformation_volume_state()
        }
        fn activation_scale(&self) -> f32 {
            self.inner.activation_scale()
        }
        fn pressure_scale(&self) -> f32 {
            self.inner.pressure_scale()
        }
        fn cohesion_bonus_pa(&self, scalar_field: f32) -> f32 {
            self.inner.cohesion_bonus_pa(scalar_field)
        }
        fn params(&self) -> MaterialParams {
            self.inner.params()
        }
    };
}

/// Wraps any `MaterialModel` to give it a non-zero `latent_heat()` without writing a full
/// delegating impl by hand -- none of the 12 built-in materials expose a settable
/// `latent_heat` field directly, since most users never need one.
///
/// ```rust,no_run
/// # extern crate emerge_engine as emerge;
/// # use emerge::{NewtonianFluidMaterial, WithLatentHeat};
/// // Water absorbs 334 (sim-unit) energy per unit mass when transitioning into this material.
/// let water = WithLatentHeat::new(NewtonianFluidMaterial::low_viscosity(1000.0, 1.0e5), 334.0);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct WithLatentHeat<M> {
    pub inner: M,
    pub latent_heat: f32,
}

impl<M> WithLatentHeat<M> {
    pub const fn new(inner: M, latent_heat: f32) -> Self {
        Self { inner, latent_heat }
    }
}

impl<M: MaterialModel> MaterialModel for WithLatentHeat<M> {
    forward_material_model_common!();
    fn init_particle(&self, particle: &mut Particle) {
        self.inner.init_particle(particle)
    }
    // Real, explicit forward (not the trait default): the trait's own
    // default would call THIS wrapper's `init_particle` (i.e. `inner.
    // init_particle`), silently skipping `inner`'s own overridden
    // transition-continuity logic if it has one (e.g. `IdealGasMaterial`) --
    // found live 2026-08-18 while adding this method, same real class of
    // gap the method itself exists to close.
    fn init_particle_from_transition(&self, particle: &mut Particle) {
        self.inner.init_particle_from_transition(particle)
    }
    fn mixture_phase(&self) -> Option<MixturePhase> {
        self.inner.mixture_phase()
    }
    fn latent_heat(&self, from_material_id: u32) -> f32 {
        let _ = from_material_id;
        self.latent_heat
    }
}

/// Real, general multi-source extension of `WithLatentHeat`: instead of ONE
/// flat energy value regardless of where a particle is transitioning FROM,
/// holds a real per-source table -- for a substance that's the destination of
/// multiple physically distinct transitions with different real energies
/// (water: +334,000 J/kg melting-in from ice, -2,257,000 J/kg condensing-in
/// from steam -- confirmed live as a real, structural need, `examples/
/// phase_states_headless.rs`). Not gas/water-specific: any material (solid,
/// liquid, gas, mixture) with more than one real incoming transition can use
/// this the same way -- `WithLatentHeat` stays the simple, unchanged, single-
/// value case for materials that only ever have one real source.
///
/// A `from_material_id` with no matching entry gets 0.0 (no energy cost) --
/// a real, disclosed default (an undeclared transition path is treated as
/// free, not as an error) rather than a hard panic, so adding a new phase
/// rule without updating every destination's table doesn't crash a scene;
/// declare every real transition path you care about explicitly.
///
/// ```rust,no_run
/// # extern crate emerge_engine as emerge;
/// # use emerge::{NewtonianFluidMaterial, WithLatentHeatTable};
/// const ICE_ID: u32 = 0;
/// const STEAM_ID: u32 = 2;
/// let water = WithLatentHeatTable::new(
///     NewtonianFluidMaterial::low_viscosity(1000.0, 1.0e5),
///     vec![(ICE_ID, 334_000.0), (STEAM_ID, -2_257_000.0)],
/// );
/// ```
#[derive(Debug, Clone)]
pub struct WithLatentHeatTable<M> {
    pub inner: M,
    pub latent_heat_by_source: Vec<(u32, f32)>,
}

impl<M> WithLatentHeatTable<M> {
    pub fn new(inner: M, latent_heat_by_source: Vec<(u32, f32)>) -> Self {
        Self {
            inner,
            latent_heat_by_source,
        }
    }
}

impl<M: MaterialModel> MaterialModel for WithLatentHeatTable<M> {
    forward_material_model_common!();
    fn init_particle(&self, particle: &mut Particle) {
        self.inner.init_particle(particle)
    }
    fn init_particle_from_transition(&self, particle: &mut Particle) {
        self.inner.init_particle_from_transition(particle)
    }
    fn mixture_phase(&self) -> Option<MixturePhase> {
        self.inner.mixture_phase()
    }
    fn latent_heat(&self, from_material_id: u32) -> f32 {
        self.latent_heat_by_source
            .iter()
            .find(|(id, _)| *id == from_material_id)
            .map(|(_, e)| *e)
            .unwrap_or(0.0)
    }
}

/// Wraps any `MaterialModel` to opt it into two-phase mixture coupling as either
/// the `Solid` or `Fluid` phase -- see `MixturePhase`'s own doc. Same pattern as
/// `WithLatentHeat`: existing materials (`DruckerPragerMaterial`,
/// `NewtonianFluidMaterial`, etc.) never opt in by themselves, so combining sand
/// and water in a scene that DOESN'T wrap either one behaves exactly as before
/// (two ordinary single-phase materials sharing a grid, no drag coupling) --
/// mixture physics is explicit, per-scene, not a silent side effect of which
/// materials happen to coexist.
///
/// ```rust,no_run
/// # extern crate emerge_engine as emerge;
/// # use emerge::{DruckerPragerMaterial, MixturePhase, WithMixturePhase};
/// let sand = WithMixturePhase::new(DruckerPragerMaterial::cohesionless(1.0e5, 0.2), MixturePhase::SOLID);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct WithMixturePhase<M> {
    pub inner: M,
    pub phase: MixturePhase,
}

impl<M> WithMixturePhase<M> {
    pub const fn new(inner: M, phase: MixturePhase) -> Self {
        Self { inner, phase }
    }
}

impl<M: MaterialModel> MaterialModel for WithMixturePhase<M> {
    forward_material_model_common!();
    fn init_particle(&self, particle: &mut Particle) {
        self.inner.init_particle(particle)
    }
    fn init_particle_from_transition(&self, particle: &mut Particle) {
        self.inner.init_particle_from_transition(particle)
    }
    fn latent_heat(&self, from_material_id: u32) -> f32 {
        self.inner.latent_heat(from_material_id)
    }
    fn mixture_phase(&self) -> Option<MixturePhase> {
        Some(self.phase)
    }
}

/// Wraps any `MaterialModel` to give particles a nonzero `internal_pressure` at spawn
/// time, without writing a full delegating impl by hand -- same pattern as
/// `WithLatentHeat`/`WithMixturePhase`. The wrapped material's own `pressure_scale()`
/// still gates whether the pressure actually contributes stress (see
/// `combined_kirchhoff_stress`); this wrapper only supplies the per-particle value.
///
/// Real motivating case: turgor pressure in plants (see `Particle::internal_pressure`
/// doc) -- but generic, not plant-specific: any internally-pressurized body.
///
/// ```rust,no_run
/// # extern crate emerge_engine as emerge;
/// # use emerge::{NeoHookeanMaterial, WithPreStress};
/// // A turgid plant-tissue stalk: 0.5 MPa turgor pressure already SI-converted to
/// // grid stress units (see `Pressurized::material` for the real conversion).
/// let stalk = WithPreStress::new(NeoHookeanMaterial::new(4000.0, 6000.0), 12.5);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct WithPreStress<M> {
    pub inner: M,
    pub pressure: f32,
}

impl<M> WithPreStress<M> {
    pub const fn new(inner: M, pressure: f32) -> Self {
        Self { inner, pressure }
    }
}

impl<M: MaterialModel> MaterialModel for WithPreStress<M> {
    forward_material_model_common!();
    fn init_particle(&self, particle: &mut Particle) {
        self.inner.init_particle(particle);
        particle.internal_pressure = self.pressure;
    }
    fn init_particle_from_transition(&self, particle: &mut Particle) {
        self.inner.init_particle_from_transition(particle);
        particle.internal_pressure = self.pressure;
    }
    fn mixture_phase(&self) -> Option<MixturePhase> {
        self.inner.mixture_phase()
    }
    fn latent_heat(&self, from_material_id: u32) -> f32 {
        self.inner.latent_heat(from_material_id)
    }
}

/// Internal fallback used when no material is registered for a particle ID.
/// Zero stress, no timestep constraint, no state updates.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct FallbackMaterial;

impl MaterialModel for FallbackMaterial {}

#[cfg(test)]
mod latent_heat_tests {
    use super::*;
    use crate::materials::NeoHookeanMaterial;

    const ICE_ID: u32 = 0;
    const STEAM_ID: u32 = 2;

    #[test]
    fn with_latent_heat_ignores_source_and_stays_flat() {
        // Real regression check: the 2026-08-23 signature change
        // (`latent_heat(&self)` -> `latent_heat(&self, from_material_id: u32)`)
        // must be a no-op for every existing single-value use.
        let water = WithLatentHeat::new(NeoHookeanMaterial::new(10.0, 20.0), 334_000.0);
        assert_eq!(water.latent_heat(ICE_ID), 334_000.0);
        assert_eq!(water.latent_heat(STEAM_ID), 334_000.0);
        assert_eq!(water.latent_heat(999), 334_000.0);
    }

    #[test]
    fn with_latent_heat_table_picks_the_real_value_for_each_real_source() {
        // Water's own real, distinct energies for its two real incoming
        // transitions -- the actual motivating case this type exists for.
        let water = WithLatentHeatTable::new(
            NeoHookeanMaterial::new(10.0, 20.0),
            vec![(ICE_ID, 334_000.0), (STEAM_ID, -2_257_000.0)],
        );
        assert_eq!(water.latent_heat(ICE_ID), 334_000.0);
        assert_eq!(water.latent_heat(STEAM_ID), -2_257_000.0);
    }

    #[test]
    fn with_latent_heat_table_defaults_to_zero_for_an_undeclared_source() {
        // Real, disclosed default (see the type's own doc): an unlisted
        // source is treated as a free transition, not an error, so a new
        // phase rule doesn't crash a scene whose tables weren't updated.
        let water = WithLatentHeatTable::new(
            NeoHookeanMaterial::new(10.0, 20.0),
            vec![(ICE_ID, 334_000.0)],
        );
        assert_eq!(water.latent_heat(STEAM_ID), 0.0);
        assert_eq!(water.latent_heat(999), 0.0);
    }
}

/// Real sanity check for `current_friction_coefficient`'s "most materials
/// opt out" default (MIBF, Blatny & Gaume 2025) -- confirms the trait's
/// own `None` default isn't accidentally overridden anywhere it shouldn't
/// be, across a real spread of material families (elastic, plastic-but-
/// non-granular, and SVD-clamp-plasticity granular with no DP cone).
#[cfg(test)]
mod current_friction_coefficient_default_tests {
    use super::*;
    use crate::materials::{GranularFluidMaterial, NeoHookeanMaterial, VonMisesMaterial};
    use crate::particle::Particle;

    #[test]
    fn non_friction_materials_return_none() {
        let particles = Particles::from(vec![Particle::zeroed()]);
        assert_eq!(
            NeoHookeanMaterial::new(1.0e5, 0.2).current_friction_coefficient(&particles, 0),
            None
        );
        assert_eq!(
            VonMisesMaterial::from_young_modulus(1.0e5, 0.2, 1.0e4)
                .current_friction_coefficient(&particles, 0),
            None
        );
        assert_eq!(
            GranularFluidMaterial::saturated_loam(1.0e5, 0.2)
                .current_friction_coefficient(&particles, 0),
            None,
            "GranularFluidMaterial has SVD-clamp plasticity, no Drucker-Prager cone -- \
             must not silently claim a friction coefficient it has no real concept of"
        );
    }
}

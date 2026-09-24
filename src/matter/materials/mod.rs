pub mod gas;
pub mod liquid;
pub mod mixture;
pub mod params;
pub mod physical_props;
mod property_dispatch;
pub mod registry;
pub mod rod_material;
pub mod solid;
pub(crate) mod svd;
pub mod utils;

pub use physical_props::{
    BrittleProps, Elastic, Elastoplastic, Fluid, FluidGranular, FromSI, NoCompression,
    ParticleMass, PlasticityModel, Pressurized, Viscoelastic,
};

pub use gas::ideal_gas::GasMaterial;
pub use liquid::bingham::BinghamFluidMaterial;
pub use liquid::fluid::NewtonianFluidMaterial;
pub use mixture::granular_fluid::GranularFluidMaterial;
pub use params::MaterialParams;
pub use registry::{MAX_MATERIAL_SLOTS, MaterialRegistry};
pub use rod_material::RodMaterial;
pub use solid::corotated::CorotatedMaterial;
pub use solid::elastic::NeoHookeanMaterial;
pub use solid::granular::sand::DruckerPragerMaterial;
pub use solid::granular::sand_mui::MuIRheologyMaterial;
pub use solid::nacc::NaccMaterial;
pub use solid::no_compression::NoCompressionMaterial;
pub use solid::rankine::RankineMaterial;
pub use solid::snow::StomakhinMaterial;
pub use solid::viscoelastic::ViscoelasticMaterial;
pub use solid::von_mises::VonMisesMaterial;
pub use utils::{
    elastic_wave_dt, gravity_to_grid, lame_from_si, lame_from_young, polar_decomposition_2d,
    rankine_damage_estimate,
};

use glam::Mat2;

use crate::particle::{Particle, Particles};

/// Identifies which constitutive model a material implements.
/// `repr(u32)` so this discriminant can be stored directly in GPU uniform buffers.
/// Explicit values are stable across recompiles — do not change them.
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
    VonMises = 6,         // J2 perfect plasticity — ductile flow, no hardening (lava, metal, clay)
    Rankine = 7,          // Tensile cutoff + exponential softening — brittle rock, bone, ice
    DruckerPragerMuI = 8, // Rate-dependent DP — µ(I) rheology, granular flow
    Viscoelastic = 9,     // Kelvin-Voigt: NeoHookean elastic + viscous dashpot in parallel
    Nacc = 10,            // Non-Associated Cam-Clay — wet soil, clay, bio tissue under compression
    GranularFluid = 11, // Granular-fluid mixture — Tait EOS + corotated deviatoric + SVD plasticity
    NoCompression = 12, // Tension-only (no-compression) reversible elastic — silk, tendons, membranes
    /// Ideal gas EOS (p=ρRT) — CPU only. GPU shaders (`p2g.wgsl`,
    /// `particles_update.wgsl`) have no case-13 branch yet; an unrecognised
    /// `mat.model` falls through their `default: { return mat2x2<f32>(); }`
    /// arm, i.e. zero stress on GPU today. Real, disclosed limitation, not
    /// silent — see `GasMaterial`'s own doc. CPU correctness first, GPU
    /// port second (per this engine's own standing development rule).
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
/// object as before — zero behavior change, only unlocks a fast path for
/// materials the engine already knows about.
///
/// Split into its own blanket-impl'd trait (rather than a default method
/// directly on `MaterialModel`) because `fn as_any(&self) -> &dyn Any { self }`
/// as a *default* method on a `Self: ?Sized`-context trait doesn't typecheck
/// (the unsized coercion needs a concrete, Sized `Self`) — the standard fix
/// (used by e.g. the `downcast-rs` crate) is a supertrait with a blanket
/// `impl<T: Any> AsAny for T`, which every `Sized` material — including any
/// external/LP-defined one — gets automatically, no per-material code needed.
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
    /// scalars rather than `&Particles, i: usize` — every implementation only ever reads
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
    /// boiling into `GasMaterial` steam): `Simulation::
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
    /// `GasMaterial`'s own override for the real, worked pattern (keep the
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
    /// Return `true` for CPU-only plasticity paths — the GPU solver uses this to
    /// decide whether to download particles and run the CPU pass each frame.
    fn needs_cpu_update(&self) -> bool {
        false
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

    /// Scaling coefficient for activation-driven deviatoric stress.
    ///
    /// When non-zero, the per-particle `activation` field (0.0–1.0) modulates the
    /// deviatoric component of the Kirchhoff stress. This is the engine-level hook for
    /// active matter: muscles, motile cells, contractile tissue.
    ///
    /// Physics: τ_total = τ_elastic + activation × coeff × I  (contractile active pressure)
    /// Default: 0.0 — activation has no effect on passive materials.
    fn activation_scale(&self) -> f32 {
        0.0
    }

    /// Scaling coefficient for internal pre-stress pressure.
    ///
    /// When non-zero, the per-particle `internal_pressure` field (already SI-
    /// converted to grid stress units) contributes an isotropic `-P·I` term to
    /// the Kirchhoff stress — the standard "prestressed structure" treatment
    /// (a balloon: envelope tension balanced against internal gas pressure).
    /// Generic engine-level hook, not plant-specific: real motivating case is
    /// turgor pressure (plants aren't held up by cell-wall elasticity alone —
    /// see Niklas 1992's "hydro-skeleton" theory), but applies to any
    /// internally-pressurized body a material wants to model this way.
    ///
    /// Physics: τ_total = τ_elastic + τ_active − internal_pressure × coeff × I
    /// Default: 0.0 — pre-stress has no effect on materials that don't opt in
    /// (fluids already carry their own EOS pressure and should not double up).
    fn pressure_scale(&self) -> f32 {
        0.0
    }

    /// Returns this material's parameters as a flat, GPU-uploadable struct.
    /// Default returns zeroed params (Fallback model).
    fn params(&self) -> MaterialParams {
        MaterialParams::default()
    }

    /// Energy cost (J/kg, in whatever temperature unit `Particle::temperature` uses)
    /// of transitioning INTO this material via `Simulation::phase_transition` /
    /// `add_phase_rule`. Positive = endothermic (e.g. melting into a liquid — absorbs
    /// energy, cooling the particle). Negative = exothermic (e.g. freezing into a
    /// solid — releases energy, warming the particle). Default 0.0 = no energy cost
    /// (existing behavior for every material, unchanged).
    ///
    /// Applied in `Simulation::phase_transition`/`add_phase_rule` (CPU) against
    /// `ThermalDiffusion::heat_capacity` when a thermal model is configured, and in
    /// `GpuSimulation::phase_transition` against the `heat_capacity` passed to
    /// `attach_thermal_gpu` -- same debit, same formula, on both. GPU has no automatic
    /// `add_phase_rule` counterpart yet (only the manual, one-shot `phase_transition`);
    /// that gap is real and separate from this energy accounting.
    fn latent_heat(&self) -> f32 {
        0.0
    }
}

/// The `MaterialModel` methods every delegating wrapper below (`WithLatentHeat`,
/// `WithMixturePhase`, `WithPreStress`) forwards to `self.inner` byte-for-byte.
/// Factored into one macro so these three impls can't drift out of sync — a new
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
        fn params(&self) -> MaterialParams {
            self.inner.params()
        }
    };
}

/// Wraps any `MaterialModel` to give it a non-zero `latent_heat()` without writing a full
/// delegating impl by hand — none of the built-in materials expose a settable
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
    // transition-continuity logic if it has one (e.g. `GasMaterial`) --
    // found live 2026-08-18 while adding this method, same real class of
    // gap the method itself exists to close.
    fn init_particle_from_transition(&self, particle: &mut Particle) {
        self.inner.init_particle_from_transition(particle)
    }
    fn mixture_phase(&self) -> Option<MixturePhase> {
        self.inner.mixture_phase()
    }
    fn latent_heat(&self) -> f32 {
        self.latent_heat
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
    fn latent_heat(&self) -> f32 {
        self.inner.latent_heat()
    }
    fn mixture_phase(&self) -> Option<MixturePhase> {
        Some(self.phase)
    }
}

/// Wraps any `MaterialModel` to give particles a nonzero `internal_pressure` at spawn
/// time, without writing a full delegating impl by hand — same pattern as
/// `WithLatentHeat`/`WithMixturePhase`. The wrapped material's own `pressure_scale()`
/// still gates whether the pressure actually contributes stress (see
/// `combined_kirchhoff_stress`); this wrapper only supplies the per-particle value.
///
/// Real motivating case: turgor pressure in plants (see `Particle::internal_pressure`
/// doc) — but generic, not plant-specific: any internally-pressurized body.
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
    fn latent_heat(&self) -> f32 {
        self.inner.latent_heat()
    }
}

/// Internal fallback used when no material is registered for a particle ID.
/// Zero stress, no timestep constraint, no state updates.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct FallbackMaterial;

impl MaterialModel for FallbackMaterial {}

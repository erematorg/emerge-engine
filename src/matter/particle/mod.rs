use glam::{Mat2, Vec2};

/// A single material point carrying all per-particle simulation state.
///
/// Used as a temporary view / scratch value in material model APIs
/// (`MaterialModel::kirchhoff_stress`, `update_particle`, etc.).
/// Long-term storage lives in [`Particles`] (SoA layout).
///
/// `repr(C)` + `Pod` so the GPU buffers can cast directly without unsafe casts.
/// All fields are `f32` / `u32` / glam types (which are `Pod` with the `bytemuck` feature).
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Particle {
    pub x: Vec2,
    pub v: Vec2,
    /// Local velocity gradient ∂v/∂x (APIC C matrix).
    /// Accumulated during G2P: C = Σ w_i · v_i ⊗ (x_i − x_p) · D⁻¹
    /// Feeds back into P2G to produce a spatially-varying grid velocity field.
    pub velocity_gradient: Mat2,
    pub deformation_gradient: Mat2,
    pub mass: f32,
    pub initial_volume: f32,
    pub volume: f32,
    pub density: f32,
    pub material_id: u32,
    /// Plastic volume ratio Jp = det(Fₚ): cumulative volume change from plastic deformation.
    /// 1.0 = undeformed. < 1.0 = compressed. Updated each step by plasticity models.
    pub plastic_volume_ratio: f32,
    /// Dimensionless hardening scale h = exp(ξ·(1−Jp)). Multiplies µ and λ in corotated stress.
    /// 1.0 = baseline stiffness. > 1.0 = stiffened by compression (e.g. compacted snow).
    pub hardening_scale: f32,
    /// Per-material plastic scalar -- meaning depends on the active material:
    /// - `DruckerPragerMaterial`: Drucker-Prager friction accumulator q (Klar 2016)
    /// - `MuIRheologyMaterial`: current µ(I) value (rate-dependent friction coefficient)
    /// - `VonMisesMaterial`: isotropic hardening κ (equivalent plastic strain)
    /// - `RankineMaterial`: damage accumulator d ∈ [0, 1] (0 = intact, 1 = fully failed)
    pub friction_hardening: f32,
    /// Drucker-Prager cumulative log volumetric plastic strain εᵥ.
    pub log_volume_strain: f32,
    /// Particle temperature in simulation units (K when grid_cell_size is set to SI scale).
    /// Used by LP's rendering emission pass (blackbody glow) and future heat-transfer systems.
    /// Initialize to 0.0; set per-particle for thermal simulations.
    pub temperature: f32,
    /// Caller-defined tag. LP uses this as creature_id for ownership tracking.
    /// Any u32 the consumer wants -- zero means untagged.
    pub user_tag: u32,
    /// Consumer-defined actuation scalar in [0, 1].
    ///
    /// Intended as a generic hook for active-matter materials -- any material
    /// that scales its stress response based on an external drive signal.
    /// 0.0 = fully passive. 1.0 = fully activated.
    /// Particles that are not actively driven keep this at 0.0.
    pub activation: f32,
    /// Muscle fiber direction in the material (reference) frame.
    ///
    /// Unit vector pointing along the contractile axis. Active stress is applied as
    /// τ_active = F · (activation × coeff × n₀⊗n₀) · Fᵀ -- contracting along this
    /// direction and following the body's deformation.
    /// Zero vector = isotropic fallback (same as old behaviour).
    /// LP sets this per muscle region at spawn time.
    pub activation_dir: Vec2,
    /// Index into the controller's muscle group array.
    ///
    /// LP writes one activation scalar per group each AI tick; the solver looks up
    /// p.activation = controller_output[p.muscle_group_id].
    /// 0 = unassigned / passive.
    pub muscle_group_id: u32,
    /// Multi-field frictional contact group (Bardenhagen, Guilkey, Roessig, Brackbill
    /// 2001, "An Improved Contact Algorithm for the Material Point Method"). 0 (default)
    /// = ordinary single-field particle: the solver allocates a second velocity field,
    /// and resolves contact, only at grid nodes touched by at least one particle with
    /// `contact_group != 0`, so a scene that never sets this is unaffected.
    ///
    /// Any nonzero value means "carries its own grip": Coulomb friction (finite,
    /// slip-capable) is resolved between this particle's field and everything with
    /// `contact_group == 0` at shared grid nodes, instead of the default MPM behavior
    /// (all particles share one velocity field, i.e. infinite friction, no slip ever
    /// possible). Distinct nonzero values are not distinguished from each other: this
    /// is 2-field contact (grip vs. rest), not N-body multi-field contact. See
    /// `SimConfig::contact_friction` for the friction coefficient.
    pub contact_group: u32,
    /// GPU sleep flag: 0 = active, 1 = sleeping (skipped by P2G/G2P/plasticity/force
    /// fields on the GPU path). Mirrors `Particles.sleeping` for the CPU `Simulation`'s
    /// own (separate) partition-based sleep bookkeeping -- this field is what travels
    /// with a single particle when converted to/from the AoS form `GpuSimulation` uses
    /// directly. Only meaningful when `SimConfig::sleep_threshold > 0.0`; otherwise
    /// always 0 and has no effect.
    pub sleeping: u32,
    /// Dirichlet/kinematic anchor flag: 0 (default) = ordinary free particle.
    /// Nonzero = fixed-velocity boundary condition. On CPU, every grid node in this
    /// particle's nonzero interpolation support is constrained to zero velocity after
    /// all grid forces and immediately before G2P; G2P also forces this particle's own
    /// `v = 0` and `velocity_gradient = 0`. The grid constraint is essential: resetting
    /// only the particle discards its motion without transmitting the anchor reaction to
    /// nearby continuum material, producing slow creep under constant load. The GPU
    /// path implements only the particle-local reset; full grid-node anchor parity is
    /// not done. Use case: a terrain slab with no pinned particles is a free body that
    /// drifts under the reaction of everything standing on it (measured: terrain
    /// centroid crept y=3.8->7.1 over one locomotion run); a thin pinned "bedrock"
    /// layer under the free top layer anchors the body while the top layer still
    /// deforms underfoot.
    pub pinned: u32,
    /// Generic second scalar carrier for any `ScalarDiffusionField`-shaped quantity
    /// (resource/grass level, pheromone concentration, nutrients, morphogen) that needs
    /// its own field distinct from `temperature`. CPU's closure-based
    /// `ScalarDiffusionField` can point at any field, but the GPU ports bake their
    /// formula: `attach_resource_field_gpu` reads/writes this field and
    /// `attach_thermal_gpu` keeps `temperature`, so the two compose in one scene.
    ///
    /// Appended near the end of the struct rather than next to `temperature`:
    /// inserting a field mid-struct shifts every later field's byte offset, and has
    /// corrupted GPU buffer layout even when Rust and every WGSL mirror agreed
    /// byte-for-byte. New fields go at the end, never the middle.
    /// 0.0 = untouched (every scene that doesn't use a GPU scalar field).
    pub scalar_field: f32,
    /// Generic internal pre-stress pressure, already SI-converted to grid stress
    /// units at construction (same treatment as other converted stress-scale
    /// state -- not raw Pa). Consumed as an isotropic `-P·I` addition to Kirchhoff
    /// stress by any material that opts in via `MaterialModel::pressure_scale()`
    /// (see `combined_kirchhoff_stress`) -- the standard "prestressed structure"
    /// treatment (a balloon: envelope tension balanced against internal gas
    /// pressure). The motivating case is turgor pressure (0.2-2.0 MPa in plant
    /// cells, Niklas 1992), which the self-weight-buckling literature (Niklas's
    /// "hydro-skeleton" theory; pressurized-cylinder self-buckling) treats as a
    /// structural mechanism distinct from bulk elastic stiffness; the field
    /// assumes nothing about what is pressurized (cells, membranes, bladders).
    /// 0.0 = untouched (every scene that doesn't use it).
    ///
    /// The last field: it took the struct's final spare pad slot, so the 128-byte
    /// layout has no room left. Append-only, for the reason in `scalar_field`.
    pub internal_pressure: f32,
}

// The CPU struct and the WGSL `Particle` mirror must agree byte-for-byte, or GPU upload
// silently reads garbage. This is the actual enforcement of the "128 bytes" contract
// documented on every field above -- catches any future field addition/removal that
// forgets to update the WGSL side or the padding.
const _: () = assert!(std::mem::size_of::<Particle>() == 128);

impl Particle {
    /// All-zero particle with identity deformation gradient. Useful in tests and tooling.
    pub const fn zeroed() -> Self {
        Self {
            x: glam::Vec2::ZERO,
            v: glam::Vec2::ZERO,
            velocity_gradient: Mat2::ZERO,
            deformation_gradient: Mat2::IDENTITY,
            mass: 0.0,
            initial_volume: 0.0,
            volume: 0.0,
            density: 0.0,
            material_id: 0,
            plastic_volume_ratio: 1.0,
            hardening_scale: 1.0,
            friction_hardening: 0.0,
            log_volume_strain: 0.0,
            temperature: 0.0,
            user_tag: 0,
            activation: 0.0,
            activation_dir: glam::Vec2::ZERO,
            muscle_group_id: 0,
            contact_group: 0,
            sleeping: 0,
            pinned: 0,
            scalar_field: 0.0,
            internal_pressure: 0.0,
        }
    }

    /// Recompute volume and density from a known elastic Jacobian `J=det(F)`.
    ///
    /// Used by generic solid/plastic material updates. Strict WC-MPM liquids
    /// own `V=V0 J` and `rho=rho0/J` through their continuity update and do
    /// not call this clamping helper.
    #[inline]
    pub fn sync_volume_and_density(&mut self, j: f32) {
        self.volume = (self.initial_volume * j).max(1.0e-6);
        self.density = self.mass / self.volume;
    }

    /// View a particle slice as raw bytes for wgpu buffer upload.
    ///
    /// Byte view of a particle slice -- zero-cost, safe via `bytemuck::Pod`.
    pub fn slice_as_bytes(particles: &[Particle]) -> &[u8] {
        bytemuck::cast_slice(particles)
    }
}

// The SoA `Particles` container (long-term particle storage), its accessor/mutator
// methods, and the iteration/conversion helpers built on top of it live in soa.rs --
// see that file's doc comment. Re-exported here so every existing
// `crate::particle::Particles` / `emerge::particle::Particles` path (and
// `ParticlesIter`) keeps resolving unchanged.
mod grain;
mod rod_points;
mod soa;
pub use grain::Grain;
pub use rod_points::RodPoints;
pub use soa::{ParticleUpdateCtx, Particles, ParticlesIter};

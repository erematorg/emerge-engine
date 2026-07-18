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
    /// Per-material plastic scalar — meaning depends on the active material:
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
    /// Any u32 the consumer wants — zero means untagged.
    pub user_tag: u32,
    /// Consumer-defined actuation scalar in [0, 1].
    ///
    /// Intended as a generic hook for active-matter materials — any material
    /// that scales its stress response based on an external drive signal.
    /// 0.0 = fully passive. 1.0 = fully activated.
    /// Particles that are not actively driven keep this at 0.0.
    pub activation: f32,
    /// Muscle fiber direction in the material (reference) frame.
    ///
    /// Unit vector pointing along the contractile axis. Active stress is applied as
    /// τ_active = F · (activation × coeff × n₀⊗n₀) · Fᵀ — contracting along this
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
    /// = ordinary single-field particle, identical to every material before this field
    /// existed — the solver only allocates a second velocity field, and only resolves
    /// contact, at grid nodes touched by at least one particle with `contact_group != 0`,
    /// so a scene that never sets this is byte-for-byte unaffected.
    ///
    /// Any nonzero value means "carries its own grip" — real Coulomb friction (finite,
    /// slip-capable) is resolved between this particle's field and everything with
    /// `contact_group == 0` at shared grid nodes, instead of the default MPM behavior
    /// (all particles share one velocity field, i.e. infinite friction, no slip ever
    /// possible). Distinct nonzero values are NOT currently distinguished from each
    /// other — this is a 2-field (grip vs. rest) implementation, not full N-body
    /// multi-field contact; a real, disclosed scope limit, not a hidden one. See
    /// `SimConfig::contact_friction` for the friction coefficient.
    pub contact_group: u32,
    /// GPU sleep flag: 0 = active, 1 = sleeping (skipped by P2G/G2P/plasticity/force
    /// fields on the GPU path). Mirrors `Particles.sleeping` for the CPU `Simulation`'s
    /// own (separate) partition-based sleep bookkeeping — this field is what travels
    /// with a single particle when converted to/from the AoS form `GpuSimulation` uses
    /// directly. Only meaningful when `SimConfig::sleep_threshold > 0.0`; otherwise
    /// always 0 and has no effect.
    pub sleeping: u32,
    /// Dirichlet/kinematic anchor flag: 0 (default) = ordinary free particle, identical to
    /// every material before this field existed. Nonzero = fixed-velocity boundary
    /// condition -- G2P forces `v = 0` and `velocity_gradient = 0` for this particle every
    /// substep instead of gathering from the grid, so it never moves and never
    /// accumulates local strain from being dragged, while still scattering its own
    /// mass/stress into P2G so other bodies feel it as a real, immovable anchor (the
    /// standard technique for static/bedrock geometry in deformable-body sims -- a real
    /// Dirichlet BC in continuum-mechanics terms, not a hack). Real motivating case: a
    /// terrain slab with no pinned particles is an ordinary free body that slowly drifts
    /// under the accumulated reaction force of everything standing/walking on it (real,
    /// measured live -- terrain centroid crept y=3.8->7.1 over one `walking_creature`
    /// run); a thin pinned "bedrock" layer under the free top layer anchors the whole
    /// body while the top layer still deforms naturally underfoot.
    pub pinned: u32,
    /// Generic second scalar carrier -- for any `ScalarDiffusionField`-shaped quantity
    /// (resource/grass level, pheromone concentration, nutrients, morphogen) that needs
    /// its OWN field distinct from `temperature`. Real motivating case: GPU's day-night
    /// thermal diffusion and GPU's resource-regrowth field both used to hijack
    /// `temperature` as their carrier (CPU's generic closure-based `ScalarDiffusionField`
    /// never had this problem -- it can point at any field; GPU's baked-formula ports
    /// couldn't stay generic and both defaulted to the one obvious f32 already on the
    /// struct). `attach_resource_field_gpu` reads/writes this field; `attach_thermal_gpu`
    /// keeps `temperature` -- the two now compose freely in the same scene.
    ///
    /// Deliberately placed as the LAST real field, immediately before `_pad`, not
    /// inserted after `temperature` where it semantically "belongs" -- a real, confirmed
    /// bug (2026-07-17): inserting a field in the MIDDLE of the struct (shifting
    /// `user_tag` through `pinned` by 4 bytes each) corrupted particle data on GPU
    /// readback even though both the Rust (`offset_of!`-verified) and all 9 WGSL mirror
    /// declarations agreed byte-for-byte on the resulting layout -- confirmed via a full
    /// bisection (isolated worktree at the last commit passed; reverting just this field
    /// while keeping every other uncommitted change fixed it). Root mechanism not fully
    /// identified; appending at the end (this field replacing one `_pad` slot, nothing
    /// else moving) verified clean instead. 0.0 = untouched (existing behavior for every
    /// scene that doesn't use a GPU scalar field).
    pub scalar_field: f32,
    /// Explicit padding — required after adding `contact_group` (2026-07-11). Real field
    /// data through `scalar_field` totals 124 bytes; `Mat2`'s actual alignment is 16
    /// (verified: `size_of::<Mat2>()==16, align_of::<Mat2>()==16`, NOT 8 as a first guess
    /// assumed), so the struct must round up to 128. `derive(Pod)` refuses implicit
    /// compiler-inserted padding (uninitialized bytes would violate Pod's "every bit
    /// pattern is valid" guarantee for GPU buffer upload), so all remaining bytes must be
    /// real, explicit, always-zeroed fields, not silently left out. A bare `u32` (not
    /// `[u32; 1]`) -- WGSL's `array<u32, 1>` does not reliably byte-match Rust's
    /// `[u32; 1]` in every context checked this session; a bare scalar has unambiguous
    /// 4-byte layout in both languages. Unused; always zero.
    pub _pad: u32,
}

// The CPU struct and the WGSL `Particle` mirror must agree byte-for-byte, or GPU upload
// silently reads garbage. This is the actual enforcement of the "128 bytes" contract
// documented on every field above -- catches any future field addition/removal that
// forgets to update the WGSL side or the padding.
const _: () = assert!(std::mem::size_of::<Particle>() == 128);

impl Particle {
    /// All-zero particle with identity deformation gradient. Useful in tests and tooling.
    pub fn zeroed() -> Self {
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
            _pad: 0,
        }
    }

    /// Recompute volume and density from a known elastic Jacobian J = det(F).
    ///
    /// Called at the end of every `update_particle` implementation after F is updated.
    /// Clamps volume to 1e-6 to prevent divide-by-zero in subsequent stress evaluations.
    #[inline(always)]
    pub fn sync_volume_and_density(&mut self, j: f32) {
        self.volume = (self.initial_volume * j).max(1.0e-6);
        self.density = self.mass / self.volume;
    }

    /// View a particle slice as raw bytes for wgpu buffer upload.
    ///
    /// Byte view of a particle slice — zero-cost, safe via `bytemuck::Pod`.
    pub fn slice_as_bytes(particles: &[Particle]) -> &[u8] {
        bytemuck::cast_slice(particles)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SoA particle storage
// ─────────────────────────────────────────────────────────────────────────────

/// Struct-of-Arrays particle storage.
///
/// Each field is a contiguous `Vec<T>`, giving cache-friendly iteration over
/// individual fields in the hot P2G / G2P loops. Long-term owner of all
/// particle state; [`Particle`] is used only as a temporary view / scratch value.
///
/// # Invariant
/// All vecs have the same length at all times. Methods panic on out-of-bounds.
pub struct Particles {
    // ── Kinematics — hot (read every substep) ────────────────────────────────
    pub x: Vec<Vec2>,
    pub v: Vec<Vec2>,
    pub velocity_gradient: Vec<Mat2>,
    pub deformation_gradient: Vec<Mat2>,

    // ── Volume / mass — hot ───────────────────────────────────────────────────
    pub mass: Vec<f32>,
    pub initial_volume: Vec<f32>,
    pub volume: Vec<f32>,
    pub density: Vec<f32>,
    pub material_id: Vec<u32>,

    // ── Plastic state — warm ──────────────────────────────────────────────────
    pub plastic_volume_ratio: Vec<f32>,
    pub hardening_scale: Vec<f32>,
    pub friction_hardening: Vec<f32>,
    pub log_volume_strain: Vec<f32>,

    // ── Extended — cold ───────────────────────────────────────────────────────
    pub temperature: Vec<f32>,
    pub user_tag: Vec<u32>,
    pub activation: Vec<f32>,
    pub activation_dir: Vec<Vec2>,
    pub muscle_group_id: Vec<u32>,
    /// Multi-field frictional contact group. See `Particle::contact_group` doc.
    pub contact_group: Vec<u32>,
    /// Dirichlet/kinematic anchor flag. See `Particle::pinned` doc.
    pub pinned: Vec<u32>,
    /// Generic second scalar carrier. See `Particle::scalar_field` doc.
    pub scalar_field: Vec<f32>,

    // ── Sleep state — not in the hot path ────────────────────────────────────
    /// True when the particle is in the sleeping partition and skipped by P2G/G2P.
    /// Do not write directly — use `Simulation::wake` / `Simulation::sleep`.
    pub sleeping: Vec<bool>,
}

impl Particles {
    /// Create an empty `Particles` store.
    pub fn new() -> Self {
        Self {
            x: Vec::new(),
            v: Vec::new(),
            velocity_gradient: Vec::new(),
            deformation_gradient: Vec::new(),
            mass: Vec::new(),
            initial_volume: Vec::new(),
            volume: Vec::new(),
            density: Vec::new(),
            material_id: Vec::new(),
            plastic_volume_ratio: Vec::new(),
            hardening_scale: Vec::new(),
            friction_hardening: Vec::new(),
            log_volume_strain: Vec::new(),
            temperature: Vec::new(),
            user_tag: Vec::new(),
            activation: Vec::new(),
            activation_dir: Vec::new(),
            muscle_group_id: Vec::new(),
            contact_group: Vec::new(),
            pinned: Vec::new(),
            scalar_field: Vec::new(),
            sleeping: Vec::new(),
        }
    }

    /// Create an empty `Particles` store pre-allocated for `cap` particles.
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            x: Vec::with_capacity(cap),
            v: Vec::with_capacity(cap),
            velocity_gradient: Vec::with_capacity(cap),
            deformation_gradient: Vec::with_capacity(cap),
            mass: Vec::with_capacity(cap),
            initial_volume: Vec::with_capacity(cap),
            volume: Vec::with_capacity(cap),
            density: Vec::with_capacity(cap),
            material_id: Vec::with_capacity(cap),
            plastic_volume_ratio: Vec::with_capacity(cap),
            hardening_scale: Vec::with_capacity(cap),
            friction_hardening: Vec::with_capacity(cap),
            log_volume_strain: Vec::with_capacity(cap),
            temperature: Vec::with_capacity(cap),
            user_tag: Vec::with_capacity(cap),
            activation: Vec::with_capacity(cap),
            activation_dir: Vec::with_capacity(cap),
            muscle_group_id: Vec::with_capacity(cap),
            contact_group: Vec::with_capacity(cap),
            pinned: Vec::with_capacity(cap),
            scalar_field: Vec::with_capacity(cap),
            sleeping: Vec::with_capacity(cap),
        }
    }

    /// Number of particles.
    #[inline]
    pub fn len(&self) -> usize {
        self.x.len()
    }

    /// True if there are no particles.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.x.is_empty()
    }

    /// Assemble a [`Particle`] view at index `i` (cheap stack copy).
    #[inline]
    pub fn get(&self, i: usize) -> Particle {
        Particle {
            x: self.x[i],
            v: self.v[i],
            velocity_gradient: self.velocity_gradient[i],
            deformation_gradient: self.deformation_gradient[i],
            mass: self.mass[i],
            initial_volume: self.initial_volume[i],
            volume: self.volume[i],
            density: self.density[i],
            material_id: self.material_id[i],
            plastic_volume_ratio: self.plastic_volume_ratio[i],
            hardening_scale: self.hardening_scale[i],
            friction_hardening: self.friction_hardening[i],
            log_volume_strain: self.log_volume_strain[i],
            temperature: self.temperature[i],
            user_tag: self.user_tag[i],
            activation: self.activation[i],
            activation_dir: self.activation_dir[i],
            muscle_group_id: self.muscle_group_id[i],
            contact_group: self.contact_group[i],
            sleeping: self.sleeping[i] as u32,
            pinned: self.pinned[i],
            scalar_field: self.scalar_field[i],
            _pad: 0,
        }
    }

    /// Write a modified [`Particle`] back to index `i`.
    #[inline]
    pub fn set(&mut self, i: usize, p: Particle) {
        self.x[i] = p.x;
        self.v[i] = p.v;
        self.velocity_gradient[i] = p.velocity_gradient;
        self.deformation_gradient[i] = p.deformation_gradient;
        self.mass[i] = p.mass;
        self.initial_volume[i] = p.initial_volume;
        self.volume[i] = p.volume;
        self.density[i] = p.density;
        self.material_id[i] = p.material_id;
        self.plastic_volume_ratio[i] = p.plastic_volume_ratio;
        self.hardening_scale[i] = p.hardening_scale;
        self.friction_hardening[i] = p.friction_hardening;
        self.log_volume_strain[i] = p.log_volume_strain;
        self.temperature[i] = p.temperature;
        self.user_tag[i] = p.user_tag;
        self.activation[i] = p.activation;
        self.activation_dir[i] = p.activation_dir;
        self.muscle_group_id[i] = p.muscle_group_id;
        self.contact_group[i] = p.contact_group;
        self.pinned[i] = p.pinned;
        self.scalar_field[i] = p.scalar_field;
    }

    /// Append a new particle.
    #[inline]
    pub fn push(&mut self, p: Particle) {
        self.x.push(p.x);
        self.v.push(p.v);
        self.velocity_gradient.push(p.velocity_gradient);
        self.deformation_gradient.push(p.deformation_gradient);
        self.mass.push(p.mass);
        self.initial_volume.push(p.initial_volume);
        self.volume.push(p.volume);
        self.density.push(p.density);
        self.material_id.push(p.material_id);
        self.plastic_volume_ratio.push(p.plastic_volume_ratio);
        self.hardening_scale.push(p.hardening_scale);
        self.friction_hardening.push(p.friction_hardening);
        self.log_volume_strain.push(p.log_volume_strain);
        self.temperature.push(p.temperature);
        self.user_tag.push(p.user_tag);
        self.activation.push(p.activation);
        self.activation_dir.push(p.activation_dir);
        self.muscle_group_id.push(p.muscle_group_id);
        self.contact_group.push(p.contact_group);
        self.pinned.push(p.pinned);
        self.scalar_field.push(p.scalar_field);
        // Honor the incoming particle's real sleeping state — needed by GpuSimulation's
        // CPU-plasticity readback path (Particles::from(Vec<Particle>)), which converts
        // live GPU particles (sleeping state included) into this SoA. Freshly-spawned
        // particles always have sleeping=0 already, so this is a no-op for that path.
        self.sleeping.push(p.sleeping != 0);
    }

    /// Swap all SoA fields for indices `a` and `b`. Used by sleep/wake partition logic.
    #[inline]
    pub fn swap(&mut self, a: usize, b: usize) {
        if a == b {
            return;
        }
        self.x.swap(a, b);
        self.v.swap(a, b);
        self.velocity_gradient.swap(a, b);
        self.deformation_gradient.swap(a, b);
        self.mass.swap(a, b);
        self.initial_volume.swap(a, b);
        self.volume.swap(a, b);
        self.density.swap(a, b);
        self.material_id.swap(a, b);
        self.plastic_volume_ratio.swap(a, b);
        self.hardening_scale.swap(a, b);
        self.friction_hardening.swap(a, b);
        self.log_volume_strain.swap(a, b);
        self.temperature.swap(a, b);
        self.user_tag.swap(a, b);
        self.activation.swap(a, b);
        self.activation_dir.swap(a, b);
        self.muscle_group_id.swap(a, b);
        self.contact_group.swap(a, b);
        self.pinned.swap(a, b);
        self.scalar_field.swap(a, b);
        self.sleeping.swap(a, b);
    }

    /// Rotate `[start..end]` so that `[mid..end]` precedes `[start..mid]`.
    /// Used by add_body to insert new particles before the sleeping zone.
    /// Standard 3-reversal algorithm — O(end − start) swaps.
    pub fn rotate_range(&mut self, start: usize, mid: usize, end: usize) {
        if start >= mid || mid >= end {
            return;
        }
        self.reverse_range(start, mid);
        self.reverse_range(mid, end);
        self.reverse_range(start, end);
    }

    fn reverse_range(&mut self, lo: usize, hi: usize) {
        let mut l = lo;
        let mut r = hi;
        while l < r {
            r -= 1;
            self.swap(l, r);
            l += 1;
        }
    }

    /// Collect all particles into a `Vec<Particle>` (for GPU upload / diagnostics).
    pub fn to_vec(&self) -> Vec<Particle> {
        (0..self.len()).map(|i| self.get(i)).collect()
    }

    /// Iterate valid indices.
    #[inline]
    pub fn indices(&self) -> std::ops::Range<usize> {
        0..self.len()
    }

    /// Remove particles where `pred` returns `false`. Stable (preserves order). O(N).
    pub fn retain<F: Fn(&Particle) -> bool>(&mut self, pred: F) {
        let n = self.len();
        let mut write = 0;
        for read in 0..n {
            let p = self.get(read);
            if pred(&p) {
                if write != read {
                    self.set(write, p);
                    // sleeping is not part of the AoS Particle view — copy explicitly.
                    self.sleeping[write] = self.sleeping[read];
                }
                write += 1;
            }
        }
        self.x.truncate(write);
        self.v.truncate(write);
        self.velocity_gradient.truncate(write);
        self.deformation_gradient.truncate(write);
        self.mass.truncate(write);
        self.initial_volume.truncate(write);
        self.volume.truncate(write);
        self.density.truncate(write);
        self.material_id.truncate(write);
        self.plastic_volume_ratio.truncate(write);
        self.hardening_scale.truncate(write);
        self.friction_hardening.truncate(write);
        self.log_volume_strain.truncate(write);
        self.temperature.truncate(write);
        self.user_tag.truncate(write);
        self.activation.truncate(write);
        self.activation_dir.truncate(write);
        self.muscle_group_id.truncate(write);
        self.contact_group.truncate(write);
        self.pinned.truncate(write);
        self.scalar_field.truncate(write);
        self.sleeping.truncate(write);
    }

    /// Apply `f` to every particle, writing all changes back.
    ///
    /// Convenience for examples / LP game code that need to mutate particles in a loop.
    /// For hot inner loops, prefer direct field access (`particles.v[i] += delta`).
    pub fn for_each_mut<F: FnMut(&mut Particle)>(&mut self, mut f: F) {
        for i in 0..self.len() {
            let mut p = self.get(i);
            f(&mut p);
            self.set(i, p);
        }
    }
}

impl Default for Particles {
    fn default() -> Self {
        Self::new()
    }
}

// ── Iteration helpers ────────────────────────────────────────────────────────

/// Lazy iterator over [`Particle`] views from a borrowed [`Particles`] store.
///
/// Constructs each `Particle` on demand from SoA storage — no upfront allocation.
pub struct ParticlesIter<'a> {
    particles: &'a Particles,
    index: usize,
}

impl<'a> Iterator for ParticlesIter<'a> {
    type Item = Particle;
    fn next(&mut self) -> Option<Particle> {
        if self.index >= self.particles.len() {
            return None;
        }
        let p = self.particles.get(self.index);
        self.index += 1;
        Some(p)
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let rem = self.particles.len() - self.index;
        (rem, Some(rem))
    }
}

impl ExactSizeIterator for ParticlesIter<'_> {}

impl Particles {
    pub fn iter(&self) -> ParticlesIter<'_> {
        ParticlesIter {
            particles: self,
            index: 0,
        }
    }

    pub fn iter_enumerated(&self) -> impl Iterator<Item = (usize, Particle)> + '_ {
        self.indices().map(move |i| (i, self.get(i)))
    }
}

impl<'a> IntoIterator for &'a Particles {
    type Item = Particle;
    type IntoIter = ParticlesIter<'a>;
    fn into_iter(self) -> ParticlesIter<'a> {
        self.iter()
    }
}

/// Conversion: collect a `Vec<Particle>` into `Particles` SoA.
impl From<Vec<Particle>> for Particles {
    fn from(v: Vec<Particle>) -> Self {
        let mut soa = Particles::with_capacity(v.len());
        for p in v {
            soa.push(p);
        }
        soa
    }
}

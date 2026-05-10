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
    /// - `SandMaterial`: Drucker-Prager friction accumulator q (Klar 2016)
    /// - `SandMuIMaterial`: current µ(I) value (rate-dependent friction coefficient)
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
    #[doc(hidden)]
    pub _pad: u32, // padding to 112 bytes for GPU alignment — do not use
}

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

/// Owning iterator over collected [`Particle`] views.
pub struct ParticlesIter {
    particles: Vec<Particle>,
    index: usize,
}

impl Iterator for ParticlesIter {
    type Item = Particle;
    fn next(&mut self) -> Option<Particle> {
        if self.index < self.particles.len() {
            let p = self.particles[self.index];
            self.index += 1;
            Some(p)
        } else {
            None
        }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let rem = self.particles.len() - self.index;
        (rem, Some(rem))
    }
}

impl ExactSizeIterator for ParticlesIter {}

impl Particles {
    /// Iterate over all particles as owned [`Particle`] views (cheap copies).
    ///
    /// Each element is an independent stack copy — modifying a particle in the
    /// iterator does not write back to storage. Use index-based access or
    /// [`Particles::set`] to write back.
    pub fn iter(&self) -> impl Iterator<Item = Particle> + '_ {
        self.indices().map(move |i| self.get(i))
    }

    /// Iterate with indices: `(usize, Particle)`.
    pub fn iter_enumerated(&self) -> impl Iterator<Item = (usize, Particle)> + '_ {
        self.indices().map(move |i| (i, self.get(i)))
    }
}

impl<'a> IntoIterator for &'a Particles {
    type Item = Particle;
    type IntoIter = ParticlesIter;
    fn into_iter(self) -> ParticlesIter {
        ParticlesIter { particles: self.to_vec(), index: 0 }
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

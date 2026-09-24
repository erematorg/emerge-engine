//! Struct-of-Arrays particle storage -- split out of `particle.rs` (was its
//! "SoA particle storage" + "Iteration helpers" sections, ~390 of the file's
//! ~600 lines). `Particles` is the long-term owner of all particle state;
//! [`Particle`] (defined in the parent module) is only a temporary AoS view
//! assembled on demand via `get`/`set`.

use glam::{Mat2, Vec2};

use super::Particle;

/// Struct-of-Arrays particle storage.
///
/// Each field is a contiguous `Vec<T>`, giving cache-friendly iteration over
/// individual fields in the hot P2G / G2P loops. Long-term owner of all
/// particle state; [`Particle`] is used only as a temporary view / scratch value.
///
/// # Invariant
/// All vecs have the same length at all times. Methods panic on out-of-bounds.
#[derive(Clone)]
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
    /// Generic internal pre-stress pressure. See `Particle::internal_pressure` doc.
    pub internal_pressure: Vec<f32>,

    // ── Sleep state — not in the hot path ────────────────────────────────────
    /// True when sleeping (skipped by P2G/G2P). `pub(crate)`: write only via
    /// `Simulation::wake`/`sleep`, which keep the tail-partition invariant intact.
    pub(crate) sleeping: Vec<bool>,

    /// Kahan (compensated) summation residual for `x`'s position integration
    /// in `transfer::g2p::gather_grid_to_particles` -- same real technique,
    /// same citation (Kahan 1965), as `RodPoints::position_compensation`
    /// already uses for rods: an ordinary velocity*dt increment can fall
    /// below f32's representable precision at the particle's own grid-
    /// coordinate magnitude (e.g. a barely-moving body far from the domain
    /// origin) even though the underlying velocity is real and sustained --
    /// this tracks the rounding error each addition drops and folds it back
    /// in next time. NOT part of the `Particle` AoS view (no spare byte on
    /// that 128-byte GPU-shared struct -- see its own doc; this is a pure
    /// CPU-integration scratch value, reset to zero on push, same
    /// convention `sleeping` above already established for a field with no
    /// `Particle` counterpart).
    pub(crate) position_compensation: Vec<Vec2>,
}

/// Per-particle mutable view into one particle's warm state, used by
/// `MaterialModel::update_particle` and `BoundaryCondition::post_g2p_particle`.
/// Exists so G2P's per-particle plasticity/boundary pass can run in parallel
/// across particles (rayon) instead of needing `&mut Particles` (the whole
/// SoA struct) one particle at a time -- every field here is disjoint-borrowed
/// straight out of `Particles`' own separate `Vec<T>` fields (real struct-of-
/// arrays, not just in name), so the borrow checker can prove two different
/// particles' contexts never alias, even built concurrently on different
/// threads. Covers exactly the fields every material's `update_particle` (and
/// `GripFrictionBoundary`'s `post_g2p_particle`) actually touches -- verified
/// by grepping every real implementation, not guessed.
pub struct ParticleUpdateCtx<'a> {
    pub x: &'a mut Vec2,
    pub v: &'a mut Vec2,
    pub velocity_gradient: &'a mut Mat2,
    pub deformation_gradient: &'a mut Mat2,
    pub volume: &'a mut f32,
    pub density: &'a mut f32,
    pub hardening_scale: &'a mut f32,
    pub plastic_volume_ratio: &'a mut f32,
    pub log_volume_strain: &'a mut f32,
    pub friction_hardening: &'a mut f32,
    pub mass: f32,
    pub temperature: f32,
    pub initial_volume: f32,
    pub activation: f32,
    pub activation_dir: Vec2,
    /// Gathered granular fluidity `g` from a coupled
    /// `GranularFluidityField` (see `energy::thermodynamics::granular_fluidity`),
    /// for this substep only -- transient, never stored on `Particle` itself
    /// (there is no spare byte for it). 0.0 (the field's own real rest
    /// state) when no such field is wired up for this scene, or when the
    /// reading material doesn't opt in -- provably inert in that case, not
    /// a tuning default.
    pub nonlocal_fluidity: f32,
    /// Gathered micro-curvature (kappa = grad(omega_c)) from a coupled
    /// `CosseratField` (see `energy::thermodynamics::cosserat_field`), for
    /// this substep only -- transient, never stored on `Particle` itself,
    /// same convention `nonlocal_fluidity` already uses. `Vec2::ZERO` (the
    /// field's own real rest state) when no such field is wired up for this
    /// scene, or when the reading material doesn't opt in -- provably inert
    /// in that case, not a tuning default.
    pub cosserat_curvature: Vec2,
}

impl Particles {
    /// Builds a `ParticleUpdateCtx` for one particle by index. For single-
    /// particle/test call sites (needs exclusive `&mut Particles`, so NOT
    /// usable from inside a parallel loop over sliced fields -- the real G2P
    /// hot path builds these directly from its own already-disjoint parallel
    /// slices instead of calling this).
    pub fn update_ctx(&mut self, i: usize) -> ParticleUpdateCtx<'_> {
        ParticleUpdateCtx {
            x: &mut self.x[i],
            v: &mut self.v[i],
            velocity_gradient: &mut self.velocity_gradient[i],
            deformation_gradient: &mut self.deformation_gradient[i],
            volume: &mut self.volume[i],
            density: &mut self.density[i],
            hardening_scale: &mut self.hardening_scale[i],
            plastic_volume_ratio: &mut self.plastic_volume_ratio[i],
            log_volume_strain: &mut self.log_volume_strain[i],
            friction_hardening: &mut self.friction_hardening[i],
            mass: self.mass[i],
            temperature: self.temperature[i],
            initial_volume: self.initial_volume[i],
            activation: self.activation[i],
            activation_dir: self.activation_dir[i],
            nonlocal_fluidity: 0.0,
            cosserat_curvature: Vec2::ZERO,
        }
    }

    /// Create an empty `Particles` store.
    pub const fn new() -> Self {
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
            internal_pressure: Vec::new(),
            sleeping: Vec::new(),
            position_compensation: Vec::new(),
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
            internal_pressure: Vec::with_capacity(cap),
            sleeping: Vec::with_capacity(cap),
            position_compensation: Vec::with_capacity(cap),
        }
    }

    /// Number of particles.
    #[inline]
    pub const fn len(&self) -> usize {
        self.x.len()
    }

    /// True if there are no particles.
    #[inline]
    pub const fn is_empty(&self) -> bool {
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
            internal_pressure: self.internal_pressure[i],
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
        self.internal_pressure[i] = p.internal_pressure;
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
        self.internal_pressure.push(p.internal_pressure);
        // Honor the incoming particle's real sleeping state — needed by GpuSimulation's
        // CPU-plasticity readback path (Particles::from(Vec<Particle>)), which converts
        // live GPU particles (sleeping state included) into this SoA. Freshly-spawned
        // particles always have sleeping=0 already, so this is a no-op for that path.
        self.sleeping.push(p.sleeping != 0);
        // A new particle starts with zero accumulated rounding error, same
        // convention `sleeping` above uses for a field with no `Particle` counterpart.
        self.position_compensation.push(Vec2::ZERO);
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
        self.internal_pressure.swap(a, b);
        self.sleeping.swap(a, b);
        self.position_compensation.swap(a, b);
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
                    // sleeping/position_compensation are not part of the AoS
                    // Particle view — copy explicitly.
                    self.sleeping[write] = self.sleeping[read];
                    self.position_compensation[write] = self.position_compensation[read];
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
        self.internal_pressure.truncate(write);
        self.sleeping.truncate(write);
        self.position_compensation.truncate(write);
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
    pub const fn iter(&self) -> ParticlesIter<'_> {
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

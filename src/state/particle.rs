use glam::{Mat2, Vec2};

/// A single material point carrying all per-particle simulation state.
///
/// `repr(C)` guarantees a stable, deterministic field layout for GPU upload.
/// Use `Particle::as_bytes` to get a raw byte slice for wgpu buffer writes.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Particle {
    pub x: Vec2,
    pub v: Vec2,
    /// APIC affine velocity gradient (C matrix).
    /// Accumulated during G2P: C = Σ w_i · v_i ⊗ (x_i − x_p) · D⁻¹
    /// Encodes the local velocity gradient around the particle.
    /// Feeds back into P2G to produce a spatially-varying grid velocity field.
    pub affine: Mat2,
    pub deformation_gradient: Mat2,
    pub mass: f32,
    pub initial_volume: f32,
    pub volume: f32,
    pub density: f32,
    pub material_id: u32,
    /// Plastic Jacobian Jp: tracks cumulative volume change from plastic deformation.
    /// Always 1.0 for elastic/fluid materials. Updated each step by plasticity models.
    pub plastic_jacobian: f32,
    /// Elastic hardening multiplier h = exp(ξ·(1−Jp)). Scales µ and λ in corotated stress.
    /// 1.0 = no hardening. Rises above 1.0 when snow is compacted (stiffer compressed snow).
    pub elastic_hardening: f32,
    /// Drucker-Prager friction hardening variable q. Accumulates during plastic flow.
    pub plastic_hardening: f32,
    /// Drucker-Prager cumulative log volumetric plastic strain.
    pub log_vol_gain: f32,
}

impl Particle {
    /// View a particle slice as raw bytes for wgpu buffer upload.
    ///
    /// # Safety
    /// `Particle` is `repr(C)` with all-`f32`/`u32`/glam fields — no pointer or
    /// reference fields, no uninit bytes in practice. The cast is safe for GPU upload
    /// purposes. Do not use the resulting slice to construct a `Particle` on the CPU.
    pub fn slice_as_bytes(particles: &[Particle]) -> &[u8] {
        unsafe {
            core::slice::from_raw_parts(
                particles.as_ptr() as *const u8,
                particles.len() * core::mem::size_of::<Particle>(),
            )
        }
    }
}

use glam::{Mat2, Vec2};

#[derive(Clone, Copy, Debug)]
pub struct Particle {
    pub x: Vec2,
    pub v: Vec2,
    pub c: Mat2,
    pub deformation_gradient: Mat2,
    pub mass: f32,
    pub initial_volume: f32,
    pub volume: f32,
    pub density: f32,
    pub material_id: u32,
    // Plastic Jacobian (Jp): tracks cumulative volume change from plastic deformation.
    // Always 1.0 for elastic/fluid materials. Updated by plasticity models (e.g., snow).
    pub plastic_jacobian: f32,
    // Elastic hardening multiplier h = exp(ξ*(1-Jp)). Scales µ and λ in corotated stress.
    // 1.0 = no hardening. Updated by snow plasticity.
    pub elastic_hardening: f32,
    // Drucker-Prager friction hardening variable q. Accumulates during plastic flow.
    // 0.0 = initial state (no accumulated friction hardening yet).
    pub plastic_hardening: f32,
    // Drucker-Prager cumulative log volumetric gain from plastic flow.
    // 0.0 = no volumetric plastic strain accumulated yet.
    pub log_vol_gain: f32,
}

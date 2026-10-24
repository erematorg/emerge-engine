//! The three render paths a demo can cycle through with one key.
//!
//! Its own file, not part of `gui_common/mod.rs`, for the reason that
//! module's doc gives for `cursor_force`: every example that includes
//! `mod.rs` compiles all of it, so an item most examples do not use would
//! fail `-D warnings` on dead code. Include it with
//! `#[path = "../gui_common/render_mode.rs"] mod render_mode;`.

/// Which path draws the frame. `Particles` reads the CPU particles directly;
/// the other two read GPU buffers, which a CPU demo fills with a
/// `emerge::render::CpuRenderBridge`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RenderMode {
    /// One splat per particle (`Renderer::render`).
    Particles,
    /// The physics grid's mass field shaded as an absorbing medium
    /// (`Renderer::render_grid_volume`).
    GridVolume,
    /// A surface reconstructed from the particles and smoothed by curvature
    /// flow (`Renderer::render_surface_reconstruction`).
    Surface,
}

impl RenderMode {
    /// The next mode in the cycle `Particles -> GridVolume -> Surface`.
    pub fn next(self) -> Self {
        match self {
            Self::Particles => Self::GridVolume,
            Self::GridVolume => Self::Surface,
            Self::Surface => Self::Particles,
        }
    }

    /// Short name for a panel or console line.
    pub fn label(self) -> &'static str {
        match self {
            Self::Particles => "particles",
            Self::GridVolume => "grid-volume",
            Self::Surface => "curvature-flow surface",
        }
    }
}

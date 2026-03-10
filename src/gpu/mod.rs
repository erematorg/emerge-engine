/// GPU compute backend for the MLS-MPM solver.
///
/// Architecture: wgpu compute shaders for P2G and G2P transfer.
/// Plasticity (SVD-based: snow, sand) stays on CPU year 1 — GPU SVD is complex.
/// Fluid and NeoHookean stress run fully on GPU.
///
/// Data flow:
///   CPU uploads:  particle buffer, material params uniform buffer
///   GPU computes: P2G scatter → grid momentum update → G2P gather
///   CPU reads:    nothing at runtime — particles stay in VRAM
///   LP renders:   custom Bevy render node reads particle buffer directly via shared wgpu device
///
/// Reference architecture: wgsparkl (Dimforge, tmp/wgsparkl/) — gather pattern,
/// one workgroup per grid block, shared memory for particle data within block.
/// Avoids atomic float scatter: each grid node thread gathers from nearby particles.
///
/// Enabled via `features = ["gpu"]`. Core library compiles without this feature.
#[cfg(feature = "gpu")]
pub mod pipeline;

#[cfg(feature = "gpu")]
pub mod buffers;

// WGSL shader sources — embedded at compile time.
// Each shader is a standalone compute pass; no Bevy dependency.
#[cfg(feature = "gpu")]
pub mod shaders {
    pub const P2G: &str = include_str!("shaders/p2g.wgsl");
    pub const G2P: &str = include_str!("shaders/g2p.wgsl");
    pub const GRID_UPDATE: &str = include_str!("shaders/grid_update.wgsl");
}

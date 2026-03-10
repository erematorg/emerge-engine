/// GPU buffer management for the MPM solver.
///
/// Three persistent buffers live in VRAM for the lifetime of the simulation:
///   - `particles`: array of `Particle` structs (repr(C), 80 bytes each)
///   - `grid`:      array of `Cell` structs  (repr(C), 12 bytes each)
///   - `materials`: array of `MaterialParams` (repr(C), 48 bytes, max 16 materials)
///
/// Upload path (CPU → GPU):
///   - particles:  initial spawn only, then never read back (GPU owns them)
///   - materials:  on spawn + whenever material params change (rare)
///   - grid:       zeroed on GPU each step via compute shader, never uploaded from CPU
///
/// Read path (GPU → CPU):
///   - Never for rendering — LP reads the particle buffer directly in its render node
///   - Optional: diagnostics readback (async, non-blocking) for the egui overlay
use wgpu;

/// All persistent GPU buffers for one MpmSolver instance.
pub struct GpuBuffers {
    /// Particle data. Layout matches `Particle::slice_as_bytes()`.
    /// Usage: STORAGE | COPY_DST (initial upload) | COPY_SRC (optional diag readback)
    pub particles: wgpu::Buffer,
    /// Grid cells. Zeroed each step by grid_clear compute pass.
    /// Usage: STORAGE
    pub grid: wgpu::Buffer,
    /// One MaterialParams per registered material, padded to max_materials.
    /// Usage: UNIFORM | COPY_DST
    pub materials: wgpu::Buffer,
    /// Solver constants (grid_res, dt, d_inverse, gravity, particle_count).
    /// Updated once per substep.
    /// Usage: UNIFORM | COPY_DST
    pub step_params: wgpu::Buffer,

    pub particle_count: u32,
    pub grid_cell_count: u32,
}

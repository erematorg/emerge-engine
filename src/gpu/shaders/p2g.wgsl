// P2G — Particle to Grid scatter
// MLS-MPM, Hu et al. 2018 SIGGRAPH §4.
//
// One workgroup per 8×8 block of grid cells.
// Particles in the block are loaded into shared memory.
// Each thread (= one grid cell) gathers contributions from nearby particles.
// Gather pattern: no atomic floats needed (wgsparkl architecture).
//
// Constants (verified against mls-mpm88-explained.cpp):
//   D_INVERSE = 4.0  (quadratic B-spline, h=1 cell)
//   force_scale = -D_INVERSE  (stress scatter sign)
//
// TODO: implement — port from wgsparkl src/solver/p2g.wgsl
//       removing Bevy #import macros, adapting bind group layout.

// Particle layout (must match Particle repr(C) in Rust — 80 bytes):
struct Particle {
    x:                      vec2<f32>, // position in grid coordinates
    v:                      vec2<f32>, // velocity
    affine:                 mat2x2<f32>, // APIC C matrix
    deformation_gradient:   mat2x2<f32>,
    mass:                   f32,
    initial_volume:         f32,
    volume:                 f32,
    density:                f32,
    material_id:            u32,
    plastic_jacobian:       f32,
    elastic_hardening:      f32,
    plastic_hardening:      f32,
    log_vol_gain:           f32,
    _pad:                   vec3<f32>, // align to 16 bytes
}

// Grid cell layout (must match Cell repr(C) in Rust — 12 bytes, padded to 16):
struct Cell {
    momentum: vec2<f32>,
    mass:     f32,
    _pad:     f32,
}

// Per-substep solver constants:
struct StepParams {
    grid_res:       u32,
    particle_count: u32,
    dt:             f32,
    d_inverse:      f32, // 4.0
    gravity:        f32,
    _pad:           vec3<f32>,
}

@group(0) @binding(0) var<storage, read>       particles:   array<Particle>;
@group(0) @binding(1) var<storage, read_write> grid:        array<Cell>;
@group(0) @binding(2) var<uniform>             step_params: StepParams;

// Quadratic B-spline weight for one axis.
// d: signed distance from particle to cell center, in grid units, |d| <= 1.5
fn bspline_w(d: f32) -> f32 {
    let abs_d = abs(d);
    if abs_d < 0.5 {
        return 0.75 - abs_d * abs_d;
    } else if abs_d < 1.5 {
        let t = 1.5 - abs_d;
        return 0.5 * t * t;
    }
    return 0.0;
}

// TODO: @compute @workgroup_size(8, 8, 1)
// fn p2g_main(@builtin(global_invocation_id) gid: vec3<u32>) { ... }

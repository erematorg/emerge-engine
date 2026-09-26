//! Shared real scene recipe for `cpu/snake_on_terrain.rs` and
//! `gpu/snake_on_terrain_gpu.rs` -- the terrain/snake material values and
//! spawn parameters are the exact same real numbers on both backends;
//! before this module they were hand-duplicated in both files, exactly the
//! kind of drift risk the technique audit already found live in
//! basic_fluids_gpu.rs's own scene (real SI constructors vs the CPU
//! sibling's raw-unit ones, no longer a fair comparison).
//!
//! The two backends' own Simulation-construction APIs genuinely differ
//! (`Simulation::new` + builder methods vs `build_particles` +
//! `MaterialRegistry::with_default` + `GpuSimulation::with_device`), and
//! CPU's `DirectionalContactGrip` steering has no GPU equivalent yet (see
//! `snake_on_terrain_gpu.rs`'s own doc) -- those stay separate per file,
//! only the real, duplicated DATA moves here.

use emerge::{DruckerPragerMaterial, NeoHookeanMaterial, SimConfig, SpawnRegion};
use glam::{IVec2, Vec2};

pub const GRID: usize = 128;
// Physics solver AND CPG must both step by the SAME DT -- DT is whatever a
// physics frame represents in real time, and the CPG has no independent
// awareness of that. Splitting them (solver at 1/60, CPG still stepping by
// an old larger DT) cycles the muscle faster than it was ever tuned for and
// causes real, escalating instability.
pub const DT: f32 = 1.0 / 60.0;
pub const MUSCLE_GROUPS: u32 = 8;
pub const N_RINGS: usize = 2;
pub const N_PER_RING: usize = MUSCLE_GROUPS as usize / N_RINGS;
pub const RING_CROSS_COUPLING: f32 = 0.5;
pub const MUSCLE_AMPLITUDE: f32 = 0.9;
pub const CPG_BURN_IN_STEPS: usize = 600;
pub const SNAKE_CONTACT_GROUP: u32 = 1;
pub const FIBER_DIAG: f32 = 3.0;
pub const BODY_LEN: f32 = 18.0;
pub const BODY_CENTER: Vec2 = Vec2::new(64.0, 20.0);

/// Real, shared base -- each backend layers its own real, disclosed
/// difference on top via struct-update syntax (GPU adds `contact_friction`
/// since it has no `DirectionalContactGrip`; CPU doesn't need that field at
/// all, its contact resolution goes through the grip instead).
pub fn base_config() -> SimConfig {
    SimConfig {
        // `min_dt` is a hard floor on the substep, not a target -- `cfl_bound`
        // clamps the chosen substep to be AT LEAST `min_dt` regardless of what
        // the material's own stability bound requires. A `min_dt` override
        // safe for a soft material can silently become unsafe (forces an
        // oversized step) once stiffness increases. No override here --
        // inherits the safe `1.0e-3` default.
        max_substeps_per_step: 128,
        project_invalid_state: true,
        ..SimConfig::standard(GRID, DT, Vec2::new(0.0, -0.3))
    }
}

/// Matches this engine's own validated real-sand reference
/// (`sand_angle_of_repose_is_physical`, tests/accuracy.rs,
/// `from_young_modulus(1.0e5, 0.2)`), also sparkl/wgsparkl's own canonical
/// demo value -- a much softer value deforms continuously under load and
/// behaves like fluid, not sand.
pub fn terrain_material() -> DruckerPragerMaterial {
    DruckerPragerMaterial::cohesionless(1.0e5, 0.2)
}

pub fn terrain_spawn(config: &SimConfig) -> SpawnRegion {
    SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new(100, 12),
        box_center: Vec2::new(64.0, 10.0),
        material_id: 0,
        precompute_initial_volumes: true,
        ..SpawnRegion::for_sim(config)
    }
}

/// Same real locomotion recipe as `basic_creature.rs`.
pub fn snake_material() -> NeoHookeanMaterial {
    let mut m = NeoHookeanMaterial::new(13.0, 26.0);
    m.active_stress_coeff = 80.0;
    m.viscosity = 150.0;
    m
}

pub fn snake_spawn(config: &SimConfig, material_id: u32) -> SpawnRegion {
    SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new(36, 4),
        box_center: BODY_CENTER,
        material_id,
        precompute_initial_volumes: true,
        ..SpawnRegion::for_sim(config)
    }
}

/// Real per-particle muscle-fiber tagging math, identical on both backends
/// -- given a particle's real X position, returns its `muscle_group_id` and
/// `activation_dir` (fiber direction in material frame). `contact_group`
/// (always `SNAKE_CONTACT_GROUP` for every snake particle) is set directly
/// by each caller since it's a one-line constant, not worth threading
/// through here.
pub fn snake_particle_tag(x: Vec2) -> (u32, Vec2) {
    let body_left = BODY_CENTER.x - BODY_LEN / 2.0;
    let t = ((x.x - body_left) / BODY_LEN).clamp(0.0, 1.0);
    let group = ((t * MUSCLE_GROUPS as f32) as u32).min(MUSCLE_GROUPS - 1);
    let local_y = x.y - BODY_CENTER.y;
    let flip = if group % 2 == 1 { -1.0 } else { 1.0 };
    let dir = if local_y >= 0.0 {
        Vec2::new(-FIBER_DIAG * flip, 1.0).normalize()
    } else {
        Vec2::new(FIBER_DIAG * flip, 1.0).normalize()
    };
    (group, dir)
}

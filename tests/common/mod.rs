//! Shared test helpers -- the official Rust pattern for code shared between
//! integration test binaries (`tests/common/mod.rs`, not `tests/common.rs`,
//! so Cargo doesn't treat this as its own test file; each consumer needs its
//! own `mod common;` since every `tests/*.rs` file compiles as a separate
//! crate). Extracted 2026-08-23: `zero_gravity_config` was hand-duplicated,
//! byte-identical apart from `dt`, across `grains_grid_coupling.rs`,
//! `physics_correctness.rs` and `rod_grid_coupling.rs`.

use emerge::SimConfig;
use glam::Vec2;

/// Real, minimal `SimConfig` for scenes that isolate a mechanism from
/// gravity/settling dynamics -- `dt` stays a real parameter, not hardcoded,
/// since callers use genuinely different real values (0.02 vs 0.05).
pub fn zero_gravity_config(grid_res: usize, dt: f32) -> SimConfig {
    SimConfig {
        grid_res,
        dt,
        gravity: Vec2::ZERO,
        adaptive_timestep: true,
        ..SimConfig::default()
    }
}

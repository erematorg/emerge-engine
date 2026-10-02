//! Shared test helpers, in the standard layout for code shared between
//! integration test binaries (`tests/common/mod.rs`, not `tests/common.rs`,
//! so Cargo does not build it as its own test; each consumer declares
//! `mod common;`, since every `tests/*.rs` file is a separate crate).

use emerge::SimConfig;
use glam::Vec2;

/// Minimal `SimConfig` for scenes that isolate a mechanism from
/// gravity/settling dynamics -- `dt` stays a parameter, not hardcoded,
/// since callers use different real values (0.02 vs 0.05).
pub fn zero_gravity_config(grid_res: usize, dt: f32) -> SimConfig {
    SimConfig {
        grid_res,
        dt,
        gravity: Vec2::ZERO,
        adaptive_timestep: true,
        ..SimConfig::default()
    }
}

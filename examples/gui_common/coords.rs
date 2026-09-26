//! Cursor position to grid coordinates: the exact inverse of the projection
//! the renderer draws with, `Renderer::region_projection`, for code that maps
//! a cursor without holding the renderer. It inverts the engine's own
//! numbers instead of restating the formula, so a click cannot drift from
//! what was drawn. See `gui_common::mod`'s own doc for the real bug this
//! centralization exists to stop from recurring.
//!
//! Split out from `mod.rs` on purpose: many examples only need this pure
//! cursor-to-grid math, not the full wgpu/egui `Gfx` bootstrap. Pointing
//! those examples' `#[path]` straight at this file (instead of `mod.rs`)
//! keeps `Gfx`/`run_egui_frame` out of their binary entirely, so they don't
//! carry dead-code warnings for bootstrap code they never call.

use emerge::Renderer;
use glam::Vec2;

/// What the camera frames, in grid cells, as its lower-left and upper-right
/// corners. A grid resolution means the whole square grid, which is what
/// `Renderer::set_camera` frames; a pair of corners is the rectangle
/// `Renderer::set_camera_region` was given.
pub struct CameraRegion(Vec2, Vec2);

impl From<usize> for CameraRegion {
    fn from(grid_res: usize) -> Self {
        Self(Vec2::ZERO, Vec2::splat(grid_res as f32))
    }
}

impl From<(Vec2, Vec2)> for CameraRegion {
    fn from((min, max): (Vec2, Vec2)) -> Self {
        Self(min, max)
    }
}

pub fn cursor_to_grid(
    cursor_pos: [f32; 2],
    surface_width: u32,
    surface_height: u32,
    view: impl Into<CameraRegion>,
) -> Vec2 {
    let CameraRegion(min, max) = view.into();
    let (sx, tx, sy, ty) = Renderer::region_projection(min, max, surface_width, surface_height);
    let ndc_x = 2.0 * (cursor_pos[0] / surface_width.max(1) as f32) - 1.0;
    let ndc_y = 1.0 - 2.0 * (cursor_pos[1] / surface_height.max(1) as f32);
    Vec2::new((ndc_x - tx) / sx, (ndc_y - ty) / sy)
}

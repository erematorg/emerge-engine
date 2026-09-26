//! Real, aspect-ratio-correct inverse of `Renderer::set_camera`'s own
//! mapping -- mirrors that function's exact math then inverts it, instead
//! of guessing a corrective factor. See `gui_common::mod`'s own doc for the
//! real bug this centralization exists to stop from recurring.
//!
//! Split out from `mod.rs` on purpose: many examples only need this pure
//! cursor-to-grid math, not the full wgpu/egui `Gfx` bootstrap. Pointing
//! those examples' `#[path]` straight at this file (instead of `mod.rs`)
//! keeps `Gfx`/`run_egui_frame` out of their binary entirely, so they don't
//! carry dead-code warnings for bootstrap code they never call.

use glam::Vec2;

pub fn cursor_to_grid(
    cursor_pos: [f32; 2],
    surface_width: u32,
    surface_height: u32,
    grid_res: usize,
) -> Vec2 {
    let w = surface_width.max(1) as f32;
    let h = surface_height.max(1) as f32;
    let aspect = w / h;
    let gr = grid_res as f32;
    let (sx, tx, sy, ty) = if aspect >= 1.0 {
        (2.0 / (gr * aspect), -1.0 / aspect, 2.0 / gr, -1.0)
    } else {
        (2.0 / gr, -1.0, 2.0 * aspect / gr, -aspect)
    };
    let ndc_x = 2.0 * (cursor_pos[0] / w) - 1.0;
    let ndc_y = 1.0 - 2.0 * (cursor_pos[1] / h);
    Vec2::new((ndc_x - tx) / sx, (ndc_y - ty) / sy)
}

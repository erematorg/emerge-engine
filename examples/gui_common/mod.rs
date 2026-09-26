//! Shared winit/wgpu/egui bootstrap for interactive GUI examples.
//!
//! Extracted 2026-08-21 after a direct, concrete finding: `sand_repose_
//! angle_gui.rs` fixed a real cursor-to-grid mapping bug (the naive "screen
//! fraction * grid_res" formula only works when the window is square), but
//! `basic_sand_gui.rs` still carries the old, buggy version -- unexposed
//! only because it happens to launch at a square 480x480 default, not
//! because its own math is actually correct. A bug fixed in one hand-rolled
//! copy of this plumbing does not reach the other copies. Every `*_gui.rs`
//! example in this directory duplicates this exact bootstrap + egui-submit
//! mechanics; this module is where it now lives once.
//!
//! Deliberately NOT in scope: the `App`/`ApplicationHandler` event-dispatch
//! skeleton, `main()`, or panel CONTENT -- those differ enough per example
//! (different keybindings, different modes, different widgets) that forcing
//! them into one shape now would cost real clarity for a smaller win than
//! this part. Extend this module's scope only if a future pass finds the
//! same genuinely-identical-not-just-similar property holds there too.
//!
//! Usage (see `basic_sand_gui.rs` / `sand_repose_angle_gui.rs`):
//! ```ignore
//! #[path = "gui_common/mod.rs"]
//! mod gui_common;
//! use gui_common::Gfx;
//! ```
//!
//! Examples that only need `cursor_to_grid` (not the full `Gfx` bootstrap)
//! should point `#[path]` at `gui_common/coords.rs` instead of this file --
//! see that submodule's own doc for why.
//!
//! `cursor_force` (push/pull interaction) is likewise NOT re-exported here,
//! for the same reason: unlike `cursor_to_grid`, not every `Gfx`-using
//! example needs it, and re-exporting it unconditionally would make every
//! example that includes this file but doesn't use `CursorForce` fail
//! `-D warnings` on dead code. Examples that need it point `#[path]` at
//! `gui_common/cursor_force.rs` directly, alongside their own `gui_common`
//! import -- see `sand_water_saturation.rs` for the pattern.

use egui_wgpu::ScreenDescriptor;
use std::sync::Arc;
use winit::window::Window;

pub mod coords;
pub use coords::cursor_to_grid;

/// The GPU + egui state every interactive example needs, identically
/// constructed. Owns the surface/device/queue (real rendering) and the
/// egui context/state/renderer (real immediate-mode UI) -- NOT the
/// `emerge::render::Renderer` (the particle renderer), which stays
/// per-example since its camera setup (particle scale, aspect handling)
/// is a real, scene-tuned parameter, not shared mechanics.
pub struct Gfx {
    pub surface: wgpu::Surface<'static>,
    pub surface_config: wgpu::SurfaceConfiguration,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub format: wgpu::TextureFormat,
    pub egui_ctx: egui::Context,
    pub egui_state: egui_winit::State,
    pub egui_renderer: egui_wgpu::Renderer,
}

impl Gfx {
    /// Real, shared wgpu+egui bootstrap -- confirmed byte-for-byte identical
    /// (module-doc's own finding) across every GUI example before this
    /// extraction, just independently duplicated.
    pub async fn new(window: &Arc<Window>) -> Self {
        let size = window.inner_size();
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let surface = instance.create_surface(window.clone()).unwrap();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("no GPU adapter");
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await
            .unwrap();
        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(caps.formats[0]);
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::AutoVsync,
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &surface_config);

        let egui_ctx = egui::Context::default();
        let egui_state = egui_winit::State::new(
            egui_ctx.clone(),
            egui_ctx.viewport_id(),
            window.as_ref(),
            None,
            None,
            None,
        );
        let egui_renderer = egui_wgpu::Renderer::new(
            &device,
            format,
            egui_wgpu::RendererOptions {
                msaa_samples: 1,
                ..Default::default()
            },
        );

        Self {
            surface,
            surface_config,
            device,
            queue,
            format,
            egui_ctx,
            egui_state,
            egui_renderer,
        }
    }

    /// Reconfigures the surface for a new window size. Does NOT touch the
    /// particle `Renderer`'s own camera -- callers update that separately
    /// (their own particle_scale/grid_res are scene-specific).
    pub fn resize(&mut self, w: u32, h: u32) {
        if w == 0 || h == 0 {
            return;
        }
        self.surface_config.width = w;
        self.surface_config.height = h;
        self.surface.configure(&self.device, &self.surface_config);
    }
}

/// Real, shared egui frame submit tail -- identical mechanics across every
/// GUI example (only the PANEL CONTENT closure differs per scene). Runs
/// `build_ui`, tessellates, uploads textures, and submits a render pass
/// drawing ON TOP of whatever `view` already holds (the scene's own
/// particle render from earlier in the frame, loaded not cleared).
pub fn run_egui_frame(
    gfx: &mut Gfx,
    window: &Window,
    view: &wgpu::TextureView,
    build_ui: impl FnMut(&egui::Context),
) {
    let raw_input = gfx.egui_state.take_egui_input(window);
    let full_output = gfx.egui_ctx.run(raw_input, build_ui);
    gfx.egui_state
        .handle_platform_output(window, full_output.platform_output);
    let tris = gfx
        .egui_ctx
        .tessellate(full_output.shapes, full_output.pixels_per_point);
    let sd = ScreenDescriptor {
        size_in_pixels: [gfx.surface_config.width, gfx.surface_config.height],
        pixels_per_point: full_output.pixels_per_point,
    };
    for (id, delta) in &full_output.textures_delta.set {
        gfx.egui_renderer
            .update_texture(&gfx.device, &gfx.queue, *id, delta);
    }
    let cmd = {
        let mut enc = gfx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        gfx.egui_renderer
            .update_buffers(&gfx.device, &gfx.queue, &mut enc, &tris, &sd);
        let mut rp = enc
            .begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            })
            .forget_lifetime();
        gfx.egui_renderer.render(&mut rp, &tris, &sd);
        drop(rp);
        enc.finish()
    };
    gfx.queue.submit(std::iter::once(cmd));
    for id in &full_output.textures_delta.free {
        gfx.egui_renderer.free_texture(id);
    }
}

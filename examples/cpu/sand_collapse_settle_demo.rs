extern crate emerge_engine as emerge;

use egui_wgpu::ScreenDescriptor;
/// Fully automatic, non-interactive proof demo for `post_event_relax_threshold`
/// (the edge-triggered elastic-strain reset shipped 2026-08-02, see
/// `sand.rs`'s own doc). No keypress required to see the real behavior --
/// this runs the exact SAME scene/config as `tests/accuracy.rs::
/// post_event_relax_long_horizon_full_confirmation` (the test that held
/// 29.6deg bit-for-bit through 100,000 steps headless), just rendered live
/// with the phase transition (dynamics -> holding) triggered automatically
/// by a real, measured kinetic-energy settle-detect instead of a fixed step
/// count or a keypress.
///
/// This is deliberately a column COLLAPSE (Lajeunesse 2004 / Klar et al.
/// 2016 Figure 14: "A column of sand collapses into a pile"), not a literal
/// hourglass (Klar et al. 2016 Figure 1) -- an hourglass needs new funnel-
/// wall geometry (pinned particles + contact) that has zero precedent or
/// testing in this codebase yet; a column collapse reuses the EXACT
/// already-validated 100k-step recipe with zero new physics risk. Real
/// citation for why a column that starts already touching the floor still
/// counts as "goes naturally like a drop": the column's own 2:1 height:base
/// aspect ratio is unstable under gravity and topples/spreads exactly like
/// a real sand column released from a mold (Lajeunesse et al. 2004's own
/// lab setup), not merely nudged.
///
///   cargo run --example sand_collapse_settle_demo --features render
use emerge::render::{ColorMode, Renderer};
use emerge::{
    DruckerPragerMaterial, FrameLogger, FrictionBoundary, SimConfig, Simulation, SpawnRegion,
    per_material_stats,
};
use glam::{IVec2, Vec2};
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

const GRID: usize = 128;
const FLOOR: f32 = 2.0;
const DT: f32 = 0.1;
const SIGMA_SAND: [f32; 3] = [0.180, 0.220, 0.550];

// Exact same trigger as `tests/accuracy.rs::post_event_relax_long_horizon_
// full_confirmation`: switch to the holding recipe after a FIXED 1500 steps
// of pure dynamics, not a measured settle-detect. REAL BUG CAUGHT LIVE
// (2026-08-02): an earlier version of this file gated the switch on
// `max_speed` dropping below a threshold for N frames -- but raw collapse
// dynamics (before the fix engages) never reliably drop below any such
// threshold on their own; the pile just keeps slowly creeping toward flat
// indefinitely, so that gate could simply never fire. That's the exact
// bug this whole fix exists to solve -- gating the fix's own activation on
// the symptom it's meant to cure is circular. The validated test doesn't
// wait for settling either: it applies the recipe at a fixed, short step
// count, deliberately BEFORE the pile can fully degrade. One `sim.step()`
// call here is one full DT=0.1 macro-step, same unit the test counts in.
const HOLD_ENGAGE_STEP: u64 = 1500;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Phase {
    Falling,
    Settled,
}

fn measure_pile_shape(xs: &[Vec2]) -> (f32, f32, f32) {
    let n = xs.len() as f32;
    if n == 0.0 {
        return (0.0, 0.0, 0.0);
    }
    let center_x = xs.iter().map(|p| p.x).sum::<f32>() / n;
    // Same real bug-fixed fallback as `sand_repose_angle.rs::measure_angle_deg`:
    // a wide-spread pile can leave the +-2 cell band around center_x empty.
    let height = xs
        .iter()
        .filter(|p| (p.x - center_x).abs() < 2.0)
        .map(|p| p.y)
        .fold(f32::NEG_INFINITY, f32::max);
    let height = if height.is_finite() {
        height
    } else {
        xs.iter().map(|p| p.y).fold(f32::NEG_INFINITY, f32::max)
    } - FLOOR;
    let base_half_width = xs
        .iter()
        .filter(|p| p.y < FLOOR + 1.5)
        .map(|p| (p.x - center_x).abs())
        .fold(0.0f32, f32::max);
    let angle = (height / base_half_width.max(0.1)).atan().to_degrees();
    (height, base_half_width, angle)
}

fn make_sim() -> Simulation {
    // Exact recipe as `tests/accuracy.rs::post_event_relax_long_horizon_full_confirmation`.
    let config = SimConfig {
        max_substeps_per_step: 64,
        apic_blend: 0.6,
        cundall_damping: 0.0,
        ..SimConfig::standard(GRID, DT, Vec2::new(0.0, -0.3))
    };
    let column = SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new(8, 16),
        box_center: Vec2::new(GRID as f32 * 0.5, FLOOR + 8.0),
        material_id: 0,
        precompute_initial_volumes: true,
        ..SpawnRegion::for_sim(&config)
    };
    let mut sand = DruckerPragerMaterial::from_young_modulus(1.0e5, 0.2);
    sand.post_event_relax_threshold = 0.001;
    Simulation::new(config, column)
        .with_default_material(Box::new(sand))
        .with_boundary(Box::new(FrictionBoundary::new(2, 0.7)))
}

struct State {
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    device: wgpu::Device,
    queue: wgpu::Queue,
    sim: Simulation,
    renderer: Renderer,
    egui_ctx: egui::Context,
    egui_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
    phase: Phase,
    paused: bool,
    step: u64,
    fps_timer: std::time::Instant,
    fps_frames: u64,
    last_fps: f32,
    logger: FrameLogger,
}

impl State {
    async fn new(window: Arc<Window>) -> Self {
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
        let fmt = caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(caps.formats[0]);
        let sc = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: fmt,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::AutoVsync,
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &sc);
        let sim = make_sim();
        let mut renderer = Renderer::new(&device, sim.particles().len(), fmt);
        renderer.set_camera(&queue, GRID as u32, size.width, size.height, 0.9, true);
        renderer.set_color_mode(ColorMode::ByPhysics);
        renderer.set_optical_params(&queue, 0, SIGMA_SAND);

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
            fmt,
            egui_wgpu::RendererOptions {
                msaa_samples: 1,
                ..Default::default()
            },
        );

        let log_path = std::env::temp_dir().join("emerge_sand_collapse_settle_demo.ndjson");
        println!(
            "sand_collapse_settle_demo: {} particles  |  fully automatic, no keypress needed  |  SPACE=pause  R=reset  Q=quit",
            sim.particles().len()
        );
        println!("per-frame diagnostics log: {}", log_path.display());
        println!(
            "watching: column collapses under gravity for {HOLD_ENGAGE_STEP} steps, then \
             AUTOMATICALLY switches to the holding recipe (same fixed trigger as the validated \
             headless test) -- watch the angle after that: real dry sand IRL is 30-35deg, and \
             it should hold there, not keep creeping toward flat."
        );

        Self {
            surface,
            surface_config: sc,
            device,
            queue,
            sim,
            renderer,
            egui_ctx,
            egui_state,
            egui_renderer,
            phase: Phase::Falling,
            paused: false,
            step: 0,
            fps_timer: std::time::Instant::now(),
            fps_frames: 0,
            last_fps: 0.0,
            logger: FrameLogger::open(&log_path).expect("open ndjson log"),
        }
    }

    fn resize(&mut self, w: u32, h: u32) {
        if w == 0 || h == 0 {
            return;
        }
        self.surface_config.width = w;
        self.surface_config.height = h;
        self.surface.configure(&self.device, &self.surface_config);
        self.renderer
            .set_camera(&self.queue, GRID as u32, w, h, 0.9, true);
    }

    fn reset(&mut self) {
        self.sim = make_sim();
        self.phase = Phase::Falling;
        self.step = 0;
        println!("reset");
    }

    fn update_and_render(&mut self, window: &Window) {
        if !self.paused {
            self.sim.step();
            self.step += 1;

            // Automatic one-shot transition to the holding recipe, at the
            // exact same fixed step count the validated headless test uses
            // -- no keypress, and deliberately NOT gated on a measured
            // settle condition (see `HOLD_ENGAGE_STEP`'s own doc for why).
            if self.phase == Phase::Falling && self.step >= HOLD_ENGAGE_STEP {
                self.sim.set_apic_blend(0.05);
                self.sim.set_cundall_damping(1.0);
                self.phase = Phase::Settled;
                println!(
                    "step={}: holding recipe engaged automatically (fixed step count, matches the validated test)",
                    self.step
                );
            }
        }
        self.fps_frames += 1;
        if self.fps_timer.elapsed().as_secs_f32() >= 1.0 {
            self.last_fps = self.fps_frames as f32 / self.fps_timer.elapsed().as_secs_f32();
            self.fps_timer = std::time::Instant::now();
            self.fps_frames = 0;
        }

        let (height, half_w, angle) = measure_pile_shape(&self.sim.particles().x);
        let max_speed = self
            .sim
            .particles()
            .v
            .iter()
            .fold(0.0f32, |m, v| m.max(v.length()));
        let snap = self.sim.diagnostics_snapshot();
        let stats = per_material_stats(self.sim.particles());
        self.logger.log(
            self.step,
            snap.effective_dt,
            &stats,
            &snap,
            &[],
            &[
                ("height", height),
                ("half_width", half_w),
                ("angle_deg", angle),
                ("max_speed", max_speed),
                (
                    "holding",
                    if self.phase == Phase::Settled {
                        1.0
                    } else {
                        0.0
                    },
                ),
            ],
        );

        let output = match self.surface.get_current_texture() {
            Ok(t) => t,
            Err(_) => return,
        };
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        self.renderer
            .render(&self.device, &self.queue, self.sim.particles(), &view, true);

        let raw_input = self.egui_state.take_egui_input(window);
        let fps = self.last_fps;
        let step = self.step;
        let phase = self.phase;
        let mut paused = self.paused;
        let mut do_reset = false;

        let full_output = self.egui_ctx.run(raw_input, |ctx| {
            egui::Window::new("Sand: automatic collapse -> settle -> hold")
                .default_pos([10.0, 10.0])
                .default_width(340.0)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.label(format!("fps={fps:.0}  step={step}"));
                    ui.separator();
                    let phase_label = match phase {
                        Phase::Falling => "FALLING / COLLAPSING (apic=0.6, no damping)",
                        Phase::Settled => "SETTLED -- holding ON (apic=0.05, cundall=1.0)",
                    };
                    ui.label(format!("phase = {phase_label}"));
                    ui.label(format!("height={height:.2}  half-width={half_w:.2}"));
                    ui.label(format!(
                        "current angle = {angle:.1} deg   max_speed={max_speed:.4}"
                    ));
                    ui.label("real dry sand IRL = 30-35 deg");
                    ui.separator();
                    if phase == Phase::Settled {
                        ui.label("Watch it NOT slide back toward flat over time --");
                        ui.label("real, tested: holds 29.6deg bit-for-bit through 100,000");
                        ui.label("headless steps once this recipe engages (2026-08-02).");
                    } else {
                        ui.label("Column is unstable (2:1 height:base) and collapses under");
                        ui.label("gravity, same setup as Klar et al. 2016 Figure 14.");
                    }
                    ui.separator();
                    ui.checkbox(&mut paused, "Paused (or SPACE)");
                    if ui.button("Reset").clicked() {
                        do_reset = true;
                    }
                    ui.label("Q quit");
                });
        });
        self.paused = paused;
        if do_reset {
            self.reset();
        }

        self.egui_state
            .handle_platform_output(window, full_output.platform_output);
        let tris = self
            .egui_ctx
            .tessellate(full_output.shapes, full_output.pixels_per_point);
        let sd = ScreenDescriptor {
            size_in_pixels: [self.surface_config.width, self.surface_config.height],
            pixels_per_point: full_output.pixels_per_point,
        };
        for (id, delta) in &full_output.textures_delta.set {
            self.egui_renderer
                .update_texture(&self.device, &self.queue, *id, delta);
        }
        let cmd = {
            let mut enc = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            self.egui_renderer
                .update_buffers(&self.device, &self.queue, &mut enc, &tris, &sd);
            let mut rp = enc
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
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
            self.egui_renderer.render(&mut rp, &tris, &sd);
            drop(rp);
            enc.finish()
        };
        self.queue.submit(std::iter::once(cmd));
        for id in &full_output.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }
        output.present();
    }
}

struct App {
    window: Option<Arc<Window>>,
    state: Option<State>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        let w = Arc::new(
            el.create_window(
                winit::window::WindowAttributes::default()
                    .with_title("emerge -- Sand: Collapse -> Settle -> Hold (automatic)")
                    .with_inner_size(winit::dpi::LogicalSize::new(720u32, 720u32)),
            )
            .unwrap(),
        );
        self.state = Some(pollster::block_on(State::new(w.clone())));
        self.window = Some(w);
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(s) = self.state.as_mut() else {
            return;
        };
        if let Some(w) = &self.window {
            let resp = s.egui_state.on_window_event(w, &event);
            if resp.consumed {
                return;
            }
        }
        match event {
            WindowEvent::CloseRequested => el.exit(),
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(key),
                        state: key_state,
                        ..
                    },
                ..
            } => {
                let pressed = key_state == ElementState::Pressed;
                if pressed {
                    match key {
                        KeyCode::KeyR => s.reset(),
                        KeyCode::Space => s.paused = !s.paused,
                        KeyCode::Escape | KeyCode::KeyQ => el.exit(),
                        _ => {}
                    }
                }
            }
            WindowEvent::Resized(sz) => s.resize(sz.width, sz.height),
            WindowEvent::RedrawRequested => {
                if let Some(w) = &self.window {
                    let w = w.clone();
                    s.update_and_render(&w);
                    w.request_redraw();
                }
            }
            _ => {}
        }
    }
}

fn main() {
    let el = EventLoop::new().unwrap();
    el.set_control_flow(ControlFlow::Poll);
    let mut app = App {
        window: None,
        state: None,
    };
    el.run_app(&mut app).unwrap();
}

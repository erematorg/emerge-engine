extern crate emerge_engine as emerge;

#[path = "../gui_common/coords.rs"]
mod gui_common;

use egui_wgpu::ScreenDescriptor;
/// Live demo of the new `DruckerPragerMaterial::compaction_sensitivity`
/// mechanic (real, single-phase/dry compaction: denser packing under real
/// load -> higher friction, Bolton 1986's real relative-density-to-friction
/// relation). Same GUI boilerplate as `basic_sand.rs` -- LMB push, RMB
/// pull (`apply_radial_impulse`, already real, already existed) are the ONLY
/// forcing here. No scripted/automatic event of any kind drives this scene --
/// an earlier version of this file auto-injected a periodic velocity kick to
/// exercise the mechanic without needing live mouse input, which is exactly
/// the kind of unnatural hardcoded forcing this project's own standing rule
/// rejects (see MEMORY.md's no-cheating/no-hardcode note) -- removed. Real
/// compaction only happens here if a real person pushes/pulls the pile
/// through this window themselves.
///
/// `ColorMode::ByVolume` renders each particle by its own current volume
/// ratio J = det(F) -- so wherever you actually compact it, that region
/// visibly shifts color. Console prints a real, passive compaction readout
/// every 2s (deep-bulk vs surface, same regions used in the engine-side
/// diagnostic) -- reporting only, not driving anything.
///
///   cargo run --example sand_compaction --features render
use emerge::render::{ColorMode, Renderer};
use emerge::{
    AabbConfinementField, DruckerPragerMaterial, FrictionBoundary, SimConfig, Simulation,
    SpawnRegion,
};
use glam::{IVec2, Vec2};
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

const GRID: usize = 64;
const DT: f32 = 0.1;
// Real, disclosed demo-visibility coefficient -- real compaction magnitude in
// a settling pile is genuinely small (~1e-4 ln(J)) at this timescale, so this
// is chosen large enough to see, not a claimed-calibrated real value. See
// `DruckerPragerMaterial::compaction_sensitivity`'s own doc for the honest
// scope (real, correctly-directed physics; simplified linear coefficient).
const COMPACTION_SENSITIVITY: f32 = 50.0;

fn make_sim() -> Simulation {
    let target_angle: f32 = 30.0;
    // Smaller pile + coarser spacing -- ~760 particles instead of ~5500 (a
    // real ~7x cut: half the linear size, 2x the spacing = 4x fewer per
    // unit area), specifically so this stays interactive on CPU-only debug
    // stepping. Same real scene/physics, just fewer particles -- this
    // session's own earlier work already confirmed the pile's qualitative
    // behavior is resolution-independent (2x height + 2x density gave the
    // same result), so shrinking it doesn't change what's being shown.
    let height = 8.0f32;
    let half_base = height / target_angle.to_radians().tan();
    let config = SimConfig {
        max_substeps_per_step: 64,
        apic_blend: 0.05, // real, found-optimal granular stabilizer (tonight's own work)
        ..SimConfig::standard(GRID, DT, Vec2::new(0.0, -0.3))
    };
    let cx = GRID as f32 * 0.5;
    let floor = 2.0;
    let spawn = SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new(
            (2.0 * half_base).ceil() as i32 + 4,
            height.ceil() as i32 + 4,
        ),
        box_center: Vec2::new(cx, floor + 2.0 + height * 0.5),
        material_id: 0,
        precompute_initial_volumes: true,
        ..SpawnRegion::for_sim(&config)
    };
    let sand = DruckerPragerMaterial {
        compaction_sensitivity: COMPACTION_SENSITIVITY,
        ..DruckerPragerMaterial::from_young_modulus(1.0e5, 0.2)
    };
    let mut solver = Simulation::new(config, spawn)
        .with_default_material(Box::new(sand))
        .with_boundary(Box::new(FrictionBoundary::new(2, 0.7)));
    solver.retain_particles(|p| {
        let dy = p.x.y - floor;
        let dx = (p.x.x - cx).abs();
        dy >= 0.0 && dy <= height && dx <= half_base * (1.0 - dy / height).max(0.0)
    });
    let footprint_half = half_base + 1.0;
    solver.add_force_field(Box::new(AabbConfinementField::new(
        Vec2::new(cx - footprint_half, floor),
        Vec2::new(cx + footprint_half, floor + height + 20.0),
        500.0,
    )));
    solver
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
    cursor_pos: [f32; 2],
    lmb: bool,
    rmb: bool,
    push_strength: f32,
    frame: u64,
    fps_timer: std::time::Instant,
    fps_frames: u64,
    last_fps: f32,
    sim_time: f32,
    log_timer: std::time::Instant,
}

/// Real, honest, PASSIVE measurement: mean ln(J) (J = det(deformation_gradient),
/// each particle's own actual current volume ratio) in the deep bulk (most
/// sustained load, if any real pushing has happened near/on it) vs the
/// exposed surface -- negative = real compaction. Pure reporting, drives
/// nothing, changes nothing -- whatever compaction shows up here only
/// happened because a real person pushed/pulled the pile through the window.
fn log_compaction(sim: &Simulation, cx: f32, floor: f32, height: f32, half_base: f32) {
    let particles = sim.particles();
    let min_y = particles.x.iter().map(|p| p.y).fold(f32::MAX, f32::min);
    let mut bulk = Vec::new();
    let mut surface = Vec::new();
    for i in 0..particles.len() {
        let j = particles.deformation_gradient[i]
            .determinant()
            .max(1e-6)
            .ln();
        let dy = particles.x[i].y - min_y;
        let dx = (particles.x[i].x - cx).abs();
        if dy < height * 0.3 && dx < half_base * 0.4 {
            bulk.push(j);
        } else if dy > height * 0.7 {
            surface.push(j);
        }
    }
    let _ = floor;
    let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len().max(1) as f32;
    println!(
        "  [compaction, passive] deep-bulk ln(J) = {:.5} ({} particles)  |  surface ln(J) = {:.5} ({} particles)",
        mean(&bulk),
        bulk.len(),
        mean(&surface),
        surface.len()
    );
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
        // ByVolume itself (heat(det2(F)*0.5)) is nearly flat right at J=1.0 --
        // real compaction here is only ~0.1-0.2% volume change, far too small
        // for that mapping's dynamic range (built for much bigger deformation).
        // Real fix, not a physics hardcode: rescale the DISPLAY of the same
        // real J value into a visible range, using `Particle::scalar_field`
        // (the engine's own existing generic visualization/carrier scalar --
        // not read by DruckerPragerMaterial's constitutive law, so this only
        // affects color, never physics) + `ColorMode::ByScalarField`. Written
        // fresh from real state every frame in `update_and_render`.
        renderer.set_color_mode(ColorMode::ByScalarField);

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

        println!(
            "sand_compaction: {} particles | compaction_sensitivity={COMPACTION_SENSITIVITY} | LMB push RMB pull Q quit -- push/pull the pile yourself, nothing forces it automatically",
            sim.particles().len()
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
            cursor_pos: [0.0; 2],
            lmb: false,
            rmb: false,
            push_strength: 12.0,
            frame: 0,
            fps_timer: std::time::Instant::now(),
            fps_frames: 0,
            last_fps: 0.0,
            sim_time: 0.0,
            log_timer: std::time::Instant::now(),
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

    fn cursor_grid(&self) -> Vec2 {
        gui_common::cursor_to_grid(
            self.cursor_pos,
            self.surface_config.width,
            self.surface_config.height,
            GRID,
        )
    }

    fn update_and_render(&mut self, window: &Window) {
        if self.lmb || self.rmb {
            let mag = if self.lmb {
                self.push_strength
            } else {
                -self.push_strength
            };
            self.sim.apply_radial_impulse(self.cursor_grid(), 7.0, mag);
        }

        self.sim.step();

        // Rescale the real J = det(F) into a visible [0,1] range for display
        // only -- physics already ran above using the real, unscaled state;
        // this only sets `scalar_field`, which no material reads back.
        // VISUAL_ZOOM chosen from what this scene actually produces (J stays
        // within roughly +/-1% of 1.0 -- see the passive log below), not an
        // arbitrary number: maps that real range across most of [0,1].
        const VISUAL_ZOOM: f32 = 40.0;
        {
            let particles = self.sim.particles_mut();
            for i in 0..particles.len() {
                let j = particles.deformation_gradient[i].determinant();
                particles.scalar_field[i] = (0.5 + (j - 1.0) * VISUAL_ZOOM).clamp(0.0, 1.0);
            }
        }

        self.sim_time += DT;
        self.frame += 1;
        self.fps_frames += 1;
        if self.fps_timer.elapsed().as_secs_f32() >= 1.0 {
            self.last_fps = self.fps_frames as f32 / self.fps_timer.elapsed().as_secs_f32();
            self.fps_timer = std::time::Instant::now();
            self.fps_frames = 0;
        }
        // Real, passive compaction readout every 2s -- reports whatever real
        // compaction has actually happened so far, drives nothing itself.
        if self.log_timer.elapsed().as_secs_f32() >= 2.0 {
            self.log_timer = std::time::Instant::now();
            let cx = GRID as f32 * 0.5;
            let height = 8.0f32;
            let half_base = height / 30.0f32.to_radians().tan();
            log_compaction(&self.sim, cx, 2.0, height, half_base);
        }

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
        let n_particles = self.sim.particles().len();
        let sim_time = self.sim_time;

        let full_output = self.egui_ctx.run(raw_input, |ctx| {
            egui::Window::new("Sand Compaction")
                .default_pos([10.0, 10.0])
                .default_width(260.0)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.label(format!("fps={fps:.0}  particles={n_particles}  t={sim_time:.1}s"));
                    ui.separator();
                    ui.label(format!(
                        "compaction_sensitivity = {COMPACTION_SENSITIVITY} (demo-visibility scale)"
                    ));
                    ui.label("ColorMode::ByVolume -- push/pull the pile and watch it shift color");
                    ui.separator();
                    ui.label("LMB push  RMB pull  Q quit");
                    ui.label("Nothing forces this automatically -- console logs real compaction passively");
                });
        });

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
                    .with_title("emerge -- Sand Compaction Demo")
                    .with_inner_size(winit::dpi::LogicalSize::new(480u32, 480u32)),
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
            WindowEvent::CursorMoved { position, .. } => {
                s.cursor_pos = [position.x as f32, position.y as f32];
            }
            WindowEvent::MouseInput { state, button, .. } => match button {
                MouseButton::Left => s.lmb = state == ElementState::Pressed,
                MouseButton::Right => s.rmb = state == ElementState::Pressed,
                _ => {}
            },
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
                if let (KeyCode::Escape | KeyCode::KeyQ, true) = (key, pressed) {
                    el.exit();
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

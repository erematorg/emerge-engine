extern crate emerge_engine as emerge;

use egui_wgpu::ScreenDescriptor;
/// `basic_sand.rs` with a real, live egui panel (same wgpu-native egui
/// already used by `rod_blade_of_grass_gui.rs`/`material_sandbox_gpu`):
/// same push/pull cursor interaction as every other sand example (LMB push,
/// RMB pull, `apply_radial_impulse`), POURING (holding P spawns a small
/// trickle of new sand particles at the cursor via `Simulation::add_body`),
/// and a real, live GRAVITY slider (1.0 = genuine IRL 9.81 m/s², via
/// `Simulation::set_gravity` -- both already existed in the engine).
///
/// Real, disclosed limit: the renderer's instance buffer is sized with a
/// fixed extra headroom (`POUR_BUDGET`) at startup (wgpu buffers don't
/// resize live) -- pouring stops once that budget is spent, not a silent
/// overflow.
///
/// DIGGING (D toggles): a directional cursor drag -- nudges nearby particles
/// along the cursor's OWN movement direction, not radially like push/pull.
/// No second body, no mass-ratio tuning: mass-conserving by construction
/// (see MEMORY.md's ecosystem-roadmap note for the two rejected alternatives
/// -- particle deletion, and a kinematic "shovel" body that broke under real
/// gravity).
///
///   cargo run --example basic_sand_gui --features render
use emerge::render::{ColorMode, Renderer};
use emerge::{DruckerPragerMaterial, SimConfig, Simulation, SlipBoundary, SpawnRegion};
use glam::{IVec2, Vec2};
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

const GRID: usize = 64;
const DT: f32 = 0.1;
const MAT_LOOSE: u32 = 0;
const MAT_DENSE: u32 = 1;
const SIGMA_SAND: [f32; 3] = [0.180, 0.220, 0.550];
// Real, disclosed cap on how much new sand pouring can add beyond the
// initial ~2016 particles -- the renderer's wgpu instance buffer is
// allocated once at startup, not resizable live.
const POUR_BUDGET: usize = 2000;
// Real particles added per frame while the pour key is held -- a small
// SpawnRegion, not a single point, so a real (tiny) cone/stream forms
// instead of a perfectly straight line of particles.
const POUR_SPACING: f32 = 0.5;
const POUR_BOX: IVec2 = IVec2::new(2, 1);
// Radius of the directional dig nudge, grid cells.
const DIG_RADIUS: f32 = 4.0;

fn make_sand(lambda: f32, mu: f32, phi_deg: f32) -> DruckerPragerMaterial {
    let mut m = DruckerPragerMaterial::new(lambda, mu);
    m.friction_angle = phi_deg.to_radians();
    m
}

fn make_sim() -> Simulation {
    let config = SimConfig {
        boundary_thickness: 3,
        max_substeps_per_step: 12,
        // No gravity override here -- `earth()`'s own real, correctly-converted
        // IRL gravity (9.81 m/s² / dx_meters) stands, exposed live via the
        // GUI's gravity slider below (see `State::real_gravity`/`gravity_fraction`).
        //
        // Same already-validated CFL margin as basic_sand.rs.
        material_cfl_coefficient: 0.7,
        ..SimConfig::earth(GRID, 0.01, DT)
    };
    let spawn = |c: Vec2, mat, seed| SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new(18, 14),
        box_center: c,
        material_id: mat,
        precompute_initial_volumes: true,
        initial_velocity_scale: 0.0,
        rng_seed: seed,
        position_jitter: 0.5,
        ..SpawnRegion::for_sim(&config)
    };
    let mut solver = Simulation::new(config, spawn(Vec2::new(17.0, 40.0), MAT_LOOSE, 11))
        .with_default_material(Box::new(make_sand(2000.0, 3000.0, 20.0)))
        .with_material(MAT_DENSE, Box::new(make_sand(2000.0, 3000.0, 40.0)))
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));
    let _ = solver.add_body(spawn(Vec2::new(47.0, 40.0), MAT_DENSE, 22));
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
    pouring: bool,
    pour_dense: bool,
    poured_count: usize,
    push_strength: f32,
    digging: bool,
    dig_strength: f32,
    last_cursor_grid: Vec2,
    // Real IRL gravity (9.81 m/s², converted via `earth()`'s own real
    // dx_meters-based formula) captured once at construction -- the slider
    // scales THIS real value, so 1.0 always means genuinely real gravity,
    // not an arbitrary tuned number.
    real_gravity: Vec2,
    gravity_fraction: f32,
    frame: u64,
    fps_timer: std::time::Instant,
    fps_frames: u64,
    last_fps: f32,
    pour_seed: u32,
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
        // Real IRL gravity, captured before anything ever overrides it --
        // `earth()`'s own real conversion, not a tuned constant.
        let real_gravity = sim.config().gravity;
        // Real extra headroom for pouring -- see POUR_BUDGET's own doc.
        let render_capacity = sim.particles().len() + POUR_BUDGET;
        let mut renderer = Renderer::new(&device, render_capacity, fmt);
        // particle_scale=0.9, not the usual 0.6: particles are seeded at
        // spacing=0.5 with position_jitter=0.5 (see make_sim/pour), so a 0.6
        // disc leaves real visible gaps wherever jitter spreads two
        // neighbors apart -- 0.9 keeps discs comfortably overlapping without
        // reaching 1.0 (full-cell, where distinct grains would start
        // visually fusing into unbroken blobs). This is a per-scene render
        // tuning fix, not the deeper "particles vs. a real reconstructed
        // surface" question -- that's the curvature-flow work already
        // planned separately (see render-pipeline-plan memory).
        renderer.set_camera(&queue, GRID as u32, size.width, size.height, 0.9, true);
        renderer.set_color_mode(ColorMode::ByPhysics);
        renderer.set_optical_params(&queue, MAT_LOOSE as usize, SIGMA_SAND);
        renderer.set_optical_params(&queue, MAT_DENSE as usize, SIGMA_SAND);

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
            "basic_sand_gui: {} particles  |  LMB push  RMB pull  D toggle dig  hold P to pour  R reset  Q quit",
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
            pouring: false,
            pour_dense: false,
            poured_count: 0,
            push_strength: 12.0,
            digging: false,
            dig_strength: 18.0,
            last_cursor_grid: Vec2::ZERO,
            real_gravity,
            // 0.001 default: full IRL gravity (1.0) is numerically stable here
            // (no crash/corruption) but reads as violently fast free-fall at
            // this small grid scale -- 0.001 is comfortable, still real
            // Newtonian gravity (F=mg), just a smaller magnitude. Slider still
            // reaches 1.0 for full IRL.
            gravity_fraction: 0.001,
            frame: 0,
            fps_timer: std::time::Instant::now(),
            fps_frames: 0,
            last_fps: 0.0,
            pour_seed: 1000,
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
        Vec2::new(
            self.cursor_pos[0] / self.surface_config.width as f32 * GRID as f32,
            (1.0 - self.cursor_pos[1] / self.surface_config.height as f32) * GRID as f32,
        )
    }

    fn update_and_render(&mut self, window: &Window) {
        // Real, live gravity control -- `gravity_fraction=1.0` is genuine
        // IRL gravity (`real_gravity`, captured from `earth()`'s own real
        // conversion), not an arbitrary tuned constant. `Simulation::
        // set_gravity` already existed in the engine (lifecycle.rs) -- no
        // new engine code needed, just wiring.
        self.sim
            .set_gravity(self.real_gravity * self.gravity_fraction);
        if self.lmb || self.rmb {
            let mag = if self.lmb {
                self.push_strength
            } else {
                -self.push_strength
            };
            self.sim.apply_radial_impulse(self.cursor_grid(), 7.0, mag);
        }
        // Digging: nudges nearby particles along the cursor's OWN movement
        // direction (a furrow), not radially like push/pull -- direct
        // per-particle velocity nudge via SoA access, no second body, no
        // impulse call, so it never collapses into "just push again."
        let cursor = self.cursor_grid();
        if self.digging {
            let delta = cursor - self.last_cursor_grid;
            if delta.length_squared() > 1.0e-8 {
                let dir = delta.normalize();
                let particles = self.sim.particles_mut();
                for i in 0..particles.len() {
                    if (particles.x[i] - cursor).length() < DIG_RADIUS {
                        particles.v[i] += dir * self.dig_strength * DT;
                    }
                }
            }
        }
        self.last_cursor_grid = cursor;
        // Real pour tool: `Simulation::add_body` is the same mid-run
        // body-spawning API the engine already offers elsewhere -- a small
        // SpawnRegion dropped at the cursor each frame while held, capped by
        // POUR_BUDGET so the (fixed-size) render buffer never overflows.
        if self.pouring && self.poured_count < POUR_BUDGET {
            // Real, found-live bug (2026-08-04): pouring with the cursor near
            // the window edge maps to a grid position close enough to the
            // domain boundary that `POUR_BOX` no longer fits inside the
            // spawnable region -- `add_body` then hits `validate_for_sim`'s
            // own real assert and hard-panics the whole demo instead of just
            // declining that frame's pour. Clamp the pour center to the same
            // real bound `fits_in_sim` checks (`boundary_thickness` margin
            // plus half the pour box on each axis) so pouring at the edge
            // just pours as close to the wall as actually fits, not a crash.
            let config = self.sim.config();
            let half = POUR_BOX.as_vec2() * 0.5;
            let domain_min = Vec2::splat(config.boundary_thickness as f32) + half;
            let domain_max =
                Vec2::splat((config.grid_res - config.boundary_thickness) as f32) - half;
            let cursor = self
                .cursor_grid()
                .clamp(domain_min, domain_max.max(domain_min));
            self.pour_seed += 1;
            let mat = if self.pour_dense {
                MAT_DENSE
            } else {
                MAT_LOOSE
            };
            let spawn = SpawnRegion {
                spacing: POUR_SPACING,
                box_size: POUR_BOX,
                box_center: cursor,
                material_id: mat,
                precompute_initial_volumes: true,
                initial_velocity_scale: 0.0,
                rng_seed: self.pour_seed,
                position_jitter: 0.3,
                ..SpawnRegion::for_sim(self.sim.config())
            };
            let before = self.sim.particles().len();
            // TEMP diagnostic (2026-08-05, user-flagged pour-vs-default gap
            // investigation) -- ground-truth the real gap in grid units via
            // stdout instead of trusting a screenshot alone (this project has
            // a documented PrintWindow false-positive on a similar demo).
            // Printed BEFORE add_body so `existing_max_y` excludes this
            // frame's own new particles.
            if self.frame.is_multiple_of(15) {
                let existing_max_y = self
                    .sim
                    .particles()
                    .iter()
                    .map(|p| p.x.y)
                    .fold(f32::MIN, f32::max);
                println!(
                    "POUR_DIAG frame={} cursor_y={:.2} existing_pile_max_y={:.2} gap={:.2}",
                    self.frame,
                    cursor.y,
                    existing_max_y,
                    cursor.y - existing_max_y
                );
            }
            let _ = self.sim.add_body(spawn);
            self.poured_count += self.sim.particles().len() - before;
        }

        self.sim.step();
        self.frame += 1;
        self.fps_frames += 1;
        if self.fps_timer.elapsed().as_secs_f32() >= 1.0 {
            self.last_fps = self.fps_frames as f32 / self.fps_timer.elapsed().as_secs_f32();
            self.fps_timer = std::time::Instant::now();
            self.fps_frames = 0;
            println!(
                "frame={} fps={:.1} particles={}",
                self.frame,
                self.last_fps,
                self.sim.particles().len()
            );
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

        // --- egui panel ---
        let raw_input = self.egui_state.take_egui_input(window);
        let fps = self.last_fps;
        let mut push_strength = self.push_strength;
        let mut pour_dense = self.pour_dense;
        let mut gravity_fraction = self.gravity_fraction;
        let mut digging = self.digging;
        let mut dig_strength = self.dig_strength;
        let n_particles = self.sim.particles().len();
        let poured = self.poured_count;
        let mut reset = false;

        let full_output = self.egui_ctx.run(raw_input, |ctx| {
            egui::Window::new("Sand")
                .default_pos([10.0, 10.0])
                .default_width(240.0)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.label(format!("fps={fps:.0}  particles={n_particles}"));
                    ui.separator();
                    ui.label("Gravity (1.0 = real IRL 9.81 m/s²):");
                    ui.add(egui::Slider::new(&mut gravity_fraction, 0.0..=2.0));
                    ui.separator();
                    ui.label("Push/pull strength:");
                    ui.add(egui::Slider::new(&mut push_strength, 0.0..=40.0));
                    ui.separator();
                    ui.checkbox(&mut digging, "Digging active (or press D)");
                    ui.add(egui::Slider::new(&mut dig_strength, 0.0..=40.0).text("Dig strength"));
                    ui.separator();
                    ui.checkbox(&mut pour_dense, "Pour dense sand (unchecked = loose)");
                    ui.label(format!("Poured: {poured}/{POUR_BUDGET}"));
                    ui.add(
                        egui::ProgressBar::new(poured as f32 / POUR_BUDGET as f32)
                            .desired_width(200.0),
                    );
                    ui.separator();
                    ui.label("LMB push  RMB pull  D toggle dig  hold P to pour  R reset  Q quit");
                    if ui.button("Reset").clicked() {
                        reset = true;
                    }
                });
        });
        self.push_strength = push_strength;
        self.pour_dense = pour_dense;
        self.gravity_fraction = gravity_fraction;
        self.digging = digging;
        self.dig_strength = dig_strength;
        if reset {
            let sim = make_sim();
            self.real_gravity = sim.config().gravity;
            self.sim = sim;
            self.frame = 0;
            self.poured_count = 0;
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
                    .with_title("emerge -- Sand (GUI)")
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
                match key {
                    KeyCode::KeyP => s.pouring = pressed,
                    KeyCode::KeyD if pressed => s.digging = !s.digging,
                    KeyCode::Escape | KeyCode::KeyQ if pressed => el.exit(),
                    KeyCode::KeyR if pressed => {
                        let sim = make_sim();
                        s.real_gravity = sim.config().gravity;
                        s.sim = sim;
                        s.frame = 0;
                        s.poured_count = 0;
                        println!("reset");
                    }
                    _ => {}
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

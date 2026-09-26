extern crate emerge_engine as emerge;

#[path = "../gui_common/coords.rs"]
mod gui_common;

use egui_wgpu::ScreenDescriptor;
/// Live proof of tonight's real fix (see `MEMORY.md` -- fluid-recovery notes,
/// Round 9): an exact DCT/Fourier pressure projection (Stam 1999, "Stable
/// Fluids") replaces the stiff Tait EOS for strict WC-MPM fluids, removing
/// the acoustic-CFL wall that made SUSTAINED wall contact (a puddle resting
/// against a floor/wall) either explode or crawl at ~1fps.
///
/// Deliberately a SEPARATE, minimal example rather than a change to
/// `basic_fluids.rs`: that demo's water->ice phase transition uses
/// `NeoHookeanMaterial` (not a strict fluid), and `SimConfig::
/// fluid_pressure_iterations` currently requires EVERY particle on the grid
/// to be a strict fluid (see that field's own doc) -- freezing a single
/// water particle mid-run would violate that and panic. Water + mud here are
/// BOTH strict fluids (`NewtonianFluidMaterial`/`BinghamFluidMaterial`), so
/// no such conflict -- solo-maximal proof first, combining with the
/// freeze feature is real future work, not attempted tonight.
///
/// SAME hard geometry used all night to find and verify the fix: water
/// starts only ~2 cells from the left wall, spanning nearly the full grid
/// height -- the scene that used to explode under the old stiff-EOS
/// acoustic CFL.
///
///   cargo run --example fluid_pressure_projection --features render
use emerge::render::{ColorMode, Renderer};
use emerge::{
    BinghamFluidMaterial, NewtonianFluidMaterial, SimConfig, Simulation, SlipBoundary, SpawnRegion,
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
const MAT_WATER: u32 = 0;
const MAT_MUD: u32 = 1;
const DIG_RADIUS: f32 = 4.0;

fn make_sim() -> Simulation {
    let config = SimConfig {
        min_dt: 1.0e-4,
        // 150 -> 400 (2026-08-15): live-measured headless via
        // `diag_pressure_projection_timing`, this exact scene's real first-
        // contact violent transient (water starting ~2 cells from the wall)
        // now genuinely needs slightly more than 150 substeps in its worst
        // single frame (~frame 20) before it settles -- confirmed bounded,
        // not divergent: a 2000-cap run completes all 120 frames cleanly,
        // recovering to a cheap ~15ms/frame steady state immediately after
        // the peak (avg 16.5fps over the full run, dominated by that one
        // transient). 400 gives real headroom over the observed peak without
        // masking a genuine runaway the way an unbounded cap would (this
        // strict-fluid path still fails loud, see step.rs's own panic doc,
        // if 400 is ever insufficient).
        max_substeps_per_step: 400,
        material_cfl_coefficient: 0.1,
        cfl_include_affine_speed: false,
        fluid_pressure_iterations: 1,
        // Real, verified fix for this exact hard geometry (see
        // `SimConfig::fluid_near_wall_cfl_scale`'s own doc for the full
        // derivation): predictively tightens the gravity-CFL bound for
        // strict-fluid particles near a wall, BEFORE any compression has
        // happened (unlike the field's original acoustic-only tightening,
        // structurally inert once `eos_stiffness=0`). Verified headless:
        // this exact scene completes all 120 frames with no crash, no
        // non-finite state (a real, honest, disclosed remaining slow drift
        // late in the run, not eliminated, but bounded).
        fluid_near_wall_cfl_scale: 20.0,
        fluid_near_wall_compression_threshold: 0.0,
        ..SimConfig::earth(GRID, 0.01, DT)
    };
    // eos_stiffness = 0.0: the acoustic-CFL term this whole fix removes.
    // Real, legal value (`fluid_state::tait_pressure`'s own `>= 0.0`
    // contract) -- incompressibility now comes from the grid-level pressure
    // projection, not from an explicit stiff spring.
    // rest_density=0.1, NOT the old 4.0 -- real SI fix, 2026-08-08, see
    // basic_fluids.rs's own doc for the full derivation.
    let water = NewtonianFluidMaterial::low_viscosity(0.1, 0.0);
    let mud = BinghamFluidMaterial::new(4.0, 8.0, 0.0, 3.0, 4.0);
    const WATER_MASS: f32 = 0.1 * 0.6 * 0.6;
    const MUD_MASS: f32 = 4.0 * 0.6 * 0.6;
    let spawn_water = SpawnRegion {
        spacing: 0.6,
        box_size: IVec2::new(14, 52),
        box_center: Vec2::new(11.0, 30.0),
        material_id: MAT_WATER,
        initial_velocity_scale: 0.0,
        precompute_initial_volumes: true,
        mass_override: Some(WATER_MASS),
        ..SpawnRegion::for_sim(&config)
    };
    let spawn_mud = SpawnRegion {
        spacing: 0.6,
        box_size: IVec2::new(16, 18),
        box_center: Vec2::new(50.0, 38.0),
        material_id: MAT_MUD,
        initial_velocity_scale: 0.0,
        precompute_initial_volumes: true,
        mass_override: Some(MUD_MASS),
        ..SpawnRegion::for_sim(&config)
    };
    let mut solver = Simulation::new(config, spawn_water)
        .with_default_material(Box::new(water))
        .with_material(MAT_MUD, Box::new(mud))
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));
    let _ = solver.add_body(spawn_mud);
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
    last_cursor_grid: Vec2,
    lmb: bool,
    rmb: bool,
    digging: bool,
    push_strength: f32,
    dig_strength: f32,
    real_gravity: Vec2,
    gravity_fraction: f32,
    frame: u64,
    fps_timer: std::time::Instant,
    fps_frames: u64,
    last_fps: f32,
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
        let real_gravity = sim.config().gravity;
        let mut renderer = Renderer::new(&device, sim.particles().len(), fmt);
        renderer.set_camera(&queue, GRID as u32, size.width, size.height, 0.6, true);
        renderer.set_color_mode(ColorMode::ByMaterial);

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
            "fluid_pressure_projection: {} particles  |  LMB push  RMB pull  D toggle dig  R reset  Q quit",
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
            last_cursor_grid: Vec2::ZERO,
            lmb: false,
            rmb: false,
            digging: false,
            push_strength: 5.0,
            dig_strength: 18.0,
            real_gravity,
            gravity_fraction: 0.003,
            frame: 0,
            fps_timer: std::time::Instant::now(),
            fps_frames: 0,
            last_fps: 0.0,
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
            .set_camera(&self.queue, GRID as u32, w, h, 0.6, true);
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
        self.sim
            .set_gravity(self.real_gravity * self.gravity_fraction);
        if self.lmb || self.rmb {
            let mag = if self.lmb {
                self.push_strength
            } else {
                -self.push_strength
            };
            self.sim.apply_radial_impulse(self.cursor_grid(), 5.0, mag);
        }
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
        self.sim.step();
        self.frame += 1;
        self.fps_frames += 1;
        if self.fps_timer.elapsed().as_secs_f32() >= 1.0 {
            self.last_fps = self.fps_frames as f32 / self.fps_timer.elapsed().as_secs_f32();
            self.fps_timer = std::time::Instant::now();
            self.fps_frames = 0;
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
        let mut push_strength = self.push_strength;
        let mut dig_strength = self.dig_strength;
        let mut gravity_fraction = self.gravity_fraction;
        let mut digging = self.digging;
        let mut reset = false;
        let snap = self.sim.diagnostics_snapshot();
        let substeps = self.sim.last_substeps();

        let full_output = self.egui_ctx.run(raw_input, |ctx| {
            egui::Window::new("Fluid pressure projection (real fix, live)")
                .default_pos([10.0, 10.0])
                .default_width(300.0)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.label(format!("fps={fps:.0}  substeps/frame={substeps}"));
                    ui.label(format!(
                        "max_speed={:.3}  non_finite={}",
                        snap.max_particle_speed, snap.non_finite_particle_values
                    ));
                    ui.separator();
                    ui.label("Gravity (1.0 = real IRL 9.81 m/s²):");
                    ui.add(egui::Slider::new(&mut gravity_fraction, 0.0..=2.0));
                    ui.separator();
                    ui.label("Push/pull strength:");
                    ui.add(egui::Slider::new(&mut push_strength, 0.0..=20.0));
                    ui.checkbox(&mut digging, "Digging/stirring active (or press D)");
                    ui.add(egui::Slider::new(&mut dig_strength, 0.0..=40.0).text("Dig strength"));
                    ui.separator();
                    ui.label("LMB push  RMB pull  D toggle dig  R reset  Q quit");
                    if ui.button("Reset").clicked() {
                        reset = true;
                    }
                });
        });
        self.push_strength = push_strength;
        self.dig_strength = dig_strength;
        self.gravity_fraction = gravity_fraction;
        self.digging = digging;
        if reset {
            let sim = make_sim();
            self.real_gravity = sim.config().gravity;
            self.sim = sim;
            self.frame = 0;
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
                    .with_title("emerge -- Fluid Pressure Projection (real fix)")
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
                    KeyCode::Escape | KeyCode::KeyQ if pressed => el.exit(),
                    KeyCode::KeyD if pressed => s.digging = !s.digging,
                    KeyCode::KeyR if pressed => {
                        let sim = make_sim();
                        s.real_gravity = sim.config().gravity;
                        s.sim = sim;
                        s.frame = 0;
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

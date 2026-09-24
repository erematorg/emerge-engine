extern crate emerge_engine as emerge;

/// GPU Drucker-Prager sand, rendered from the solver's own P2G mass field
/// (`Renderer::render_grid_volume`) instead of one splat per particle —
/// real, already-shipped MPM-native technique (see `render-pipeline-plan`
/// memory): adjacent cells with mass blend into one continuous shape instead
/// of reading as a cloud of discrete dots, at zero extra simulation cost
/// (the solver already builds this field every substep for its own P2G
/// step). G toggles back to the per-particle view to compare directly.
///
///   cargo run --example basic_sand_grid_gpu --features "render"
use std::sync::Arc;

use emerge::diagnostics::log_frame_gpu;
use emerge::render::{ColorMode, GridVolumeSource, Renderer};
use emerge::{
    DruckerPragerMaterial, FixedStepController, GpuSimulation, MaterialRegistry, SimConfig,
    SpawnRegion, build_particles,
};
use glam::{IVec2, Vec2};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

const GRID: usize = 64;
const DT: f32 = 0.1;
const MAT_LOOSE: u32 = 0;
const MAT_DENSE: u32 = 1;
const LABELS: &[(u32, &str)] = &[(MAT_LOOSE, "loose"), (MAT_DENSE, "dense")];
// Real measured sand absorption (Sherman & Waite 1985, iron-oxide quartz sand).
const SIGMA_SAND: [f32; 3] = [0.180, 0.220, 0.550];

struct App {
    window: Option<Arc<Window>>,
    state: Option<State>,
}

struct State {
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    sim: GpuSimulation,
    renderer: Renderer,
    cursor_pos: [f32; 2],
    lmb: bool,
    rmb: bool,
    grid_volume_mode: bool,
    frame: u64,
    fps_timer: std::time::Instant,
    fps_frames: u64,
    /// Real-time-decoupled stepping -- see `basic_fluids_gpu.rs`'s own field
    /// doc for the full real bug/fix writeup.
    stepper: FixedStepController,
    last_instant: std::time::Instant,
    max_steps_seen: usize,
}

fn make_sand(lambda: f32, mu: f32, phi_deg: f32) -> DruckerPragerMaterial {
    let mut m = DruckerPragerMaterial::new(lambda, mu);
    m.friction_angle = phi_deg.to_radians();
    m
}

fn make_sim_data(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> GpuSimulation {
    let config = SimConfig {
        boundary_thickness: 3,
        max_substeps_per_step: 12,
        // Deliberately weak, NOT real IRL gravity (real g_grid ~= 981 via
        // SimConfig::earth) -- tuned down for a calmer, more legible demo at
        // this grid scale. Disclosed, deferred: basic_sand_gui.rs's
        // gravity_fraction slider is the real-IRL-with-live-control
        // pattern, not yet ported to every plain example.
        gravity: Vec2::new(0.0, -0.3),
        // Same already-validated CFL margin as basic_sand.rs's CPU DP-sand.
        material_cfl_coefficient: 0.7,
        ..SimConfig::earth(GRID, 0.01, DT)
    };
    let spawn = |c: Vec2, mat: u32, seed: u32| SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new(18, 14),
        box_center: c,
        material_id: mat,
        precompute_initial_volumes: true,
        rng_seed: seed,
        position_jitter: 0.5,
        ..SpawnRegion::for_sim(&config)
    };
    let mut particles = build_particles(&config, spawn(Vec2::new(17.0, 40.0), MAT_LOOSE, 11));
    particles.extend(build_particles(
        &config,
        spawn(Vec2::new(47.0, 40.0), MAT_DENSE, 22),
    ));
    let mut registry = MaterialRegistry::with_default(Box::new(make_sand(2000.0, 3000.0, 20.0)));
    registry.insert(MAT_DENSE, Box::new(make_sand(2000.0, 3000.0, 40.0)));
    GpuSimulation::with_device(device, queue, config, particles, registry)
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
        let mut sim = make_sim_data(Arc::new(device), Arc::new(queue));
        // Real per-cell material tracking, needed by the grid-volume path to
        // pick each cell's dominant material (loose vs dense) -- see
        // grid_volume.wgsl's own doc.
        sim.attach_grid_material_render_gpu();
        let mut renderer = Renderer::new(sim.device(), sim.particle_count(), fmt);
        renderer.set_camera(sim.queue(), GRID as u32, size.width, size.height, 0.6, true);
        renderer.set_color_mode(ColorMode::ByPhysics);
        renderer.set_optical_params(sim.queue(), MAT_LOOSE as usize, SIGMA_SAND);
        renderer.set_optical_params(sim.queue(), MAT_DENSE as usize, SIGMA_SAND);
        println!(
            "sand grid-volume GPU: {} particles  |  LMB push  RMB pull  G toggle grid/particle view  R reset  Q quit",
            sim.particle_count()
        );
        Self {
            surface,
            surface_config: sc,
            sim,
            renderer,
            cursor_pos: [0.0; 2],
            lmb: false,
            rmb: false,
            // Default ON -- this example exists specifically to show the
            // continuous-surface look; G still lets you A/B against the
            // per-particle view directly.
            grid_volume_mode: true,
            frame: 0,
            fps_timer: std::time::Instant::now(),
            fps_frames: 0,
            stepper: FixedStepController::standard(DT, 60.0),
            last_instant: std::time::Instant::now(),
            max_steps_seen: 0,
        }
    }

    fn resize(&mut self, w: u32, h: u32) {
        if w == 0 || h == 0 {
            return;
        }
        self.surface_config.width = w;
        self.surface_config.height = h;
        self.surface
            .configure(self.sim.device(), &self.surface_config);
        self.renderer
            .set_camera(self.sim.queue(), GRID as u32, w, h, 0.6, true);
    }

    fn cursor_grid(&self) -> Vec2 {
        Vec2::new(
            self.cursor_pos[0] / self.surface_config.width as f32 * GRID as f32,
            (1.0 - self.cursor_pos[1] / self.surface_config.height as f32) * GRID as f32,
        )
    }

    fn reset(&mut self) {
        let (device, queue) = (self.sim.device().clone(), self.sim.queue().clone());
        self.sim = make_sim_data(device, queue);
        self.sim.attach_grid_material_render_gpu();
        self.frame = 0;
        self.stepper.reset();
        self.last_instant = std::time::Instant::now();
        println!("reset");
    }

    fn update_and_render(&mut self) {
        if self.lmb || self.rmb {
            let mag = if self.lmb { 12.0 } else { -12.0 };
            self.sim.apply_radial_impulse(self.cursor_grid(), 7.0, mag);
        }
        let output = match self.surface.get_current_texture() {
            Ok(t) => t,
            Err(_) => return,
        };
        let now = std::time::Instant::now();
        let frame_delta = (now - self.last_instant).as_secs_f32();
        self.last_instant = now;
        let steps = self.stepper.steps_for_frame(frame_delta);
        self.max_steps_seen = self.max_steps_seen.max(steps);
        for _ in 0..steps {
            self.sim.step_frame();
            self.frame += 1;
            if self.frame.is_multiple_of(60) {
                log_frame_gpu(self.frame, DT, self.sim.particles(), LABELS, 1);
                let snap = self.sim.diagnostics_snapshot();
                println!(
                    "  non_finite={}  out_of_bounds={}  max_speed={:.3}  sub={}  cfl={:.4}",
                    snap.non_finite_particle_values,
                    snap.out_of_bounds_particles,
                    snap.max_particle_speed,
                    snap.substeps_last_step,
                    snap.cfl_number,
                );
            }
        }
        self.fps_frames += 1;
        if self.fps_timer.elapsed().as_secs_f32() >= 2.0 {
            let fps = self.fps_frames as f32 / self.fps_timer.elapsed().as_secs_f32();
            println!(
                "frame={} fps={:.0} grid_view={} max_steps_per_render={}",
                self.frame, fps, self.grid_volume_mode, self.max_steps_seen
            );
            self.fps_timer = std::time::Instant::now();
            self.fps_frames = 0;
            self.max_steps_seen = 0;
        }
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        if self.grid_volume_mode {
            self.renderer.render_grid_volume(
                self.sim.device(),
                self.sim.queue(),
                GridVolumeSource {
                    grid: self.sim.grid_buffer(),
                    material_mass: self.sim.material_mass_buffer(),
                    material_mass_enabled: true,
                },
                &view,
                true,
            );
        } else {
            self.renderer.render_gpu(
                self.sim.device(),
                self.sim.queue(),
                self.sim.particle_buffer(),
                self.sim.particle_count(),
                &view,
                true,
            );
        }
        output.present();
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        let w = Arc::new(
            el.create_window(
                winit::window::WindowAttributes::default()
                    .with_title("emerge — Sand, grid-volume render [G: toggle particle view]")
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
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => match key {
                KeyCode::Escape | KeyCode::KeyQ => el.exit(),
                KeyCode::KeyR => s.reset(),
                KeyCode::KeyG => {
                    s.grid_volume_mode = !s.grid_volume_mode;
                    println!(
                        "grid-volume render: {}",
                        if s.grid_volume_mode { "on" } else { "off" }
                    );
                }
                _ => {}
            },
            WindowEvent::Resized(sz) => s.resize(sz.width, sz.height),
            WindowEvent::RedrawRequested => {
                s.update_and_render();
                if let Some(w) = &self.window {
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

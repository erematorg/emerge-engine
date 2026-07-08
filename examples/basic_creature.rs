extern crate emerge_engine as emerge;

use emerge::render::{ColorMode, Renderer};
use emerge::{
    Lnn, NeoHookeanMaterial, RatchetFrictionBoundary, SimConfig, Simulation, SpawnRegion,
};
use glam::{IVec2, Vec2};
/// CPU creature -- NeoHookean soft body with peristaltic muscle activation.
///
/// Traveling wave of vertical muscle contraction, crawling via
/// `RatchetFrictionBoundary` -- directional (setae-style) floor friction that
/// resists backward slip much more than forward slip. This is the mechanism
/// that actually produces net locomotion for this body: plain symmetric floor
/// friction measured near-zero net drift regardless of muscle fiber direction
/// (a symmetric contract/release cycle cancels its own displacement, the same
/// reason you can't swim forward clapping symmetrically underwater); real
/// crawlers break that symmetry structurally (setae/hooks), not by timing
/// friction to muscle phase -- confirmed against SoftZoo (the published MPM
/// soft-robot locomotion benchmark, which uses only symmetric friction + learned
/// actuation) and real-crawler robotics literature. See
/// `tests/physics_correctness.rs::ratchet_friction_produces_real_directed_locomotion`
/// for the headless proof.
///
/// Driven by an `Lnn` (Liquid Time-constant Network) continuous-time CPG, not a
/// hand-coded sine wave -- the same controller LP's creatures use. A bilateral
/// (two-ring, mutually-inhibiting) CPG: left/right steer by biasing one ring
/// harder than the other. NOTE: this body is a straight, non-bending column, so
/// "steering" here shifts which half drives harder but cannot produce a real
/// left/right turn -- that needs a body that can curve, a separate limitation.
/// Up/down adjusts wave speed (LNN clock rate). Space pauses. R resets.
///
///   cargo run --example basic_creature --features "render"
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

const GRID: usize = 64;
const DT: f32 = 0.1;
const MAT_BODY: u32 = 0;
const MUSCLE_GROUPS: u32 = 8;
// Bilateral CPG: 2 mutually-coupled rings (front/back halves of the body),
// 4 segments each. Steering biases one ring harder than the other.
const N_RINGS: usize = 2;
const N_PER_RING: usize = MUSCLE_GROUPS as usize / N_RINGS;
const RING_CROSS_COUPLING: f32 = 1.0;
// Muscle drive is held at the documented activation ceiling; it is never pushed
// above 1.0 (a muscle can't contract >100%), which also keeps active stress
// inside the CFL budget instead of letting a global amplitude knob detonate it.
const MUSCLE_AMPLITUDE: f32 = 0.9;

fn make_cpg() -> Lnn {
    Lnn::coupled_traveling_wave(N_RINGS, N_PER_RING, 1.0, RING_CROSS_COUPLING)
}

// Per-segment colors matching the SoftZoo rainbow palette (ByMaterial slots 0""7).
// ColorMode::ByMaterial assigns color by material_id % 16, so we encode muscle group
// as material_id directly for rendering. Physics still uses MAT_BODY internally.
// For simplicity we render via ByMaterial which gives blue for all (one material).
// Advanced: override muscle group rendering via a custom color callback.

struct App {
    window: Option<Arc<Window>>,
    state: Option<State>,
}

struct State {
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    device: wgpu::Device,
    queue: wgpu::Queue,
    sim: Simulation,
    body_range: std::ops::Range<usize>,
    lnn: Lnn,
    paused: bool,
    wave_speed: f32,
    /// Steering bias in [-1, 1]: drives one CPG ring harder than the other,
    /// breaking the wave's symmetry the way an animal turns. 0 = straight.
    /// ALSO drives the ratchet's crawl direction live (see `update_and_render`):
    /// steer < 0 reverses which way the body actually crawls -- this is real
    /// control, not cosmetic, since it changes `RatchetFrictionBoundary`'s
    /// `easy_direction` on the shared instance the solver is already using.
    steer: f32,
    /// Shared handle to the solver's own ratchet boundary -- steering this
    /// updates the SAME instance driving physics, not a copy.
    ratchet: Arc<RatchetFrictionBoundary>,
    renderer: Renderer,
    frame: u64,
    fps_timer: std::time::Instant,
    fps_frames: u64,
    /// True once an anomaly has been reported, so we WARN on the first frame it
    /// appears rather than spamming every frame after.
    anomaly_latched: bool,
    spawn_centroid: Vec2,
}

fn make_sim() -> (
    Simulation,
    std::ops::Range<usize>,
    Arc<RatchetFrictionBoundary>,
) {
    let mut mat = NeoHookeanMaterial::new(5.0, 10.0);
    mat.active_stress_coeff = 25.0;
    let config = SimConfig {
        min_dt: 0.01,
        // Full CFL headroom + the degenerate-state projection net on: keeps
        // active muscle stress stable under hard driving instead of detonating
        // when a substep can't subdivide enough. See the muscle-stability
        // regression test in tests/physics_correctness.rs.
        max_substeps_per_step: 64,
        project_invalid_state: true,
        ..SimConfig::standard(GRID, DT, Vec2::new(0.0, -0.3))
    };
    let body_center = Vec2::new(32.0, 20.0);
    let spawn = SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new(24, 6),
        box_center: body_center,
        material_id: MAT_BODY,
        precompute_initial_volumes: true,
        ..SpawnRegion::for_sim(&config)
    };
    // Arc'd so this exact instance is shared between the solver (which drives
    // physics through it) and the app (which steers it live from input) --
    // set_easy_direction takes effect immediately, no boundary swap needed.
    let ratchet = Arc::new(RatchetFrictionBoundary::new(4, 0.1, 0.95, Vec2::X));
    let mut solver = Simulation::new(config, spawn)
        .with_default_material(Box::new(mat))
        .with_boundary(Box::new(Arc::clone(&ratchet)));

    let body_range = 0..solver.particles().len();
    let body_left = body_center.x - 12.0;
    {
        let particles = solver.particles_mut();
        for i in body_range.clone() {
            let t = ((particles.x[i].x - body_left) / 24.0).clamp(0.0, 1.0);
            particles.muscle_group_id[i] = (t * MUSCLE_GROUPS as f32) as u32;
            particles.activation_dir[i] = Vec2::Y;
        }
    }
    (solver, body_range, ratchet)
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
                required_limits: adapter.limits(), // use full hardware limits, not wgpu defaults
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
        let (sim, body_range, ratchet) = make_sim();
        let mut renderer = Renderer::new(&device, sim.particles().len(), fmt);
        renderer.set_camera(&queue, GRID as u32, size.width, size.height, 0.6, true);
        renderer.set_color_mode(ColorMode::ByActivation);
        println!(
            "creature: {} particles  |  up/down wave speed  left/right STEER  Space pause  R reset  Q quit",
            sim.particles().len()
        );
        Self {
            surface,
            surface_config: sc,
            device,
            queue,
            sim,
            body_range,
            lnn: make_cpg(),
            paused: false,
            wave_speed: 1.0,
            steer: 0.0,
            ratchet,
            renderer,
            frame: 0,
            fps_timer: std::time::Instant::now(),
            fps_frames: 0,
            anomaly_latched: false,
            spawn_centroid: Vec2::new(32.0, 20.0),
        }
    }

    /// Read the solver's own diagnostics and body geometry, print a full
    /// telemetry line, and WARN immediately if anything is physically wrong.
    /// Returns nothing — this is pure observation, no simulation effect.
    fn log_telemetry(&mut self, fps: f32) {
        let snap = self.sim.diagnostics_snapshot();

        // Body geometry, computed directly from particles.
        let particles = self.sim.particles();
        let n = particles.len().max(1) as f32;
        let mut centroid = Vec2::ZERO;
        let mut min = Vec2::splat(f32::INFINITY);
        let mut max = Vec2::splat(f32::NEG_INFINITY);
        let mut act_sum = 0.0f32;
        let mut act_max = 0.0f32;
        for i in 0..particles.len() {
            let x = particles.x[i];
            centroid += x;
            min = min.min(x);
            max = max.max(x);
            let a = particles.activation[i];
            act_sum += a;
            act_max = act_max.max(a);
        }
        centroid /= n;
        let extent = max - min;
        let drift = centroid - self.spawn_centroid;

        println!(
            "f{:<5} fps={:>3.0} | sub={:>2}/{} eff_dt={:.4} dropped={:.4} cfl={:.2} vmax={:.2} \
             | J=[{:.3},{:.3}] velclamp={} Jproj={} oob={} nan_p={} nan_g={} \
             | centroid=({:.1},{:.1}) drift=({:+.1},{:+.1}) extent=({:.1}x{:.1}) \
             | act mean={:.2} max={:.2} | massErr={:.1e} momErr={:.1e}",
            self.frame,
            fps,
            snap.substeps_last_step,
            self.sim.config().max_substeps_per_step,
            snap.effective_dt,
            snap.sim_time_dropped,
            snap.cfl_number,
            snap.max_particle_speed,
            snap.min_deformation_j,
            snap.max_deformation_j,
            snap.vel_clamp_count,
            snap.j_projection_count,
            snap.out_of_bounds_particles,
            snap.non_finite_particle_values,
            snap.non_finite_grid_values,
            centroid.x,
            centroid.y,
            drift.x,
            drift.y,
            extent.x,
            extent.y,
            act_sum / n,
            act_max,
            snap.relative_mass_error,
            snap.relative_momentum_error,
        );

        // Immediate WARN on the first frame anything goes wrong — pinpoints the
        // exact moment the "huge issues" start, which periodic logging can miss.
        let mut problems: Vec<String> = Vec::new();
        if snap.non_finite_particle_values > 0 || snap.non_finite_grid_values > 0 {
            problems.push(format!(
                "NON-FINITE: {} particle + {} grid values are NaN/Inf",
                snap.non_finite_particle_values, snap.non_finite_grid_values
            ));
        }
        if snap.out_of_bounds_particles > 0 {
            problems.push(format!(
                "{} particles left the grid",
                snap.out_of_bounds_particles
            ));
        }
        if snap.sim_time_dropped > 1e-6 {
            problems.push(format!(
                "solver DROPPED {:.4} of sim time — hit max_substeps and gave up (unstable)",
                snap.sim_time_dropped
            ));
        }
        if snap.min_deformation_j < 0.05 {
            problems.push(format!(
                "near-inverted element: min J = {:.4} (→0 means a particle is collapsing)",
                snap.min_deformation_j
            ));
        }
        if extent.x > 30.0 || extent.y > 30.0 {
            problems.push(format!(
                "body SCATTERING: extent {:.1}x{:.1} (spawned ~12x3)",
                extent.x, extent.y
            ));
        }
        if snap.substeps_last_step >= self.sim.config().max_substeps_per_step {
            problems.push(format!(
                "substeps MAXED ({}) — CFL is fighting hard, near the stability edge",
                snap.substeps_last_step
            ));
        }
        if !problems.is_empty() && !self.anomaly_latched {
            self.anomaly_latched = true;
            eprintln!("  ⚠ FIRST ANOMALY at frame {}:", self.frame);
            for p in &problems {
                eprintln!("      - {p}");
            }
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

    fn update_and_render(&mut self) {
        if !self.paused {
            // wave_speed scales the LNN's internal clock -- faster wave_speed runs the
            // continuous-time ODE forward faster, raising the oscillation frequency, without
            // needing to reconstruct the network (tau/weights stay fixed).
            // Steer by biasing the two rings apart: one drives harder, the wave
            // goes asymmetric, the creature turns. steer=0 → both rings equal → straight.
            self.lnn.set_ring_bias(0, N_PER_RING, self.steer);
            self.lnn.set_ring_bias(1, N_PER_RING, -self.steer);
            // ALSO drive the crawl direction itself: steer<0 reverses which way
            // the ratchet resists slip, so the body actually crawls backward, not
            // just internally-lopsided while still walking the one baked-in way.
            // Same shared instance the solver already uses -- takes effect this substep.
            self.ratchet.set_easy_direction(if self.steer >= 0.0 {
                Vec2::X
            } else {
                Vec2::NEG_X
            });
            self.lnn.step(DT * self.wave_speed);
            let activations: Vec<f32> = self.lnn.activations().collect();
            let body_range = self.body_range.clone();
            let particles = self.sim.particles_mut();
            for i in body_range {
                let group = particles.muscle_group_id[i] as usize;
                // Clamp to the documented [0,1] activation contract — a muscle can't
                // contract past 100%, and staying in-contract keeps active stress
                // inside the CFL budget.
                particles.activation[i] = (MUSCLE_AMPLITUDE * activations[group]).clamp(0.0, 1.0);
            }
            self.sim.step();
            self.frame += 1;
        }
        self.fps_frames += 1;
        // Telemetry ~2x/sec so the log stays readable but catches transients.
        if self.fps_timer.elapsed().as_secs_f32() >= 0.5 {
            let fps = self.fps_frames as f32 / self.fps_timer.elapsed().as_secs_f32();
            self.log_telemetry(fps);
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
        output.present();
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        let w = Arc::new(
            el.create_window(
                winit::window::WindowAttributes::default()
                    .with_title("emerge -- Creature [peristaltic locomotion]")
                    .with_inner_size(winit::dpi::LogicalSize::new(480u32, 480u32)),
            )
            .unwrap(),
        );
        self.state = Some(pollster::block_on(State::new(w.clone())));
        self.window = Some(w);
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(s) = self.state.as_mut() else { return };
        match event {
            WindowEvent::CloseRequested => el.exit(),
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(key),
                        state,
                        ..
                    },
                ..
            } => {
                let pressed = state == ElementState::Pressed;
                match key {
                    KeyCode::Escape | KeyCode::KeyQ if pressed => el.exit(),
                    KeyCode::Space if pressed => {
                        s.paused = !s.paused;
                        println!("{}", if s.paused { "PAUSED" } else { "RUNNING" });
                    }
                    KeyCode::KeyR if pressed => {
                        let (sim, range, ratchet) = make_sim();
                        s.sim = sim;
                        s.body_range = range;
                        s.ratchet = ratchet;
                        s.lnn = make_cpg();
                        s.steer = 0.0;
                        s.frame = 0;
                        s.anomaly_latched = false;
                        println!("reset");
                    }
                    KeyCode::ArrowUp if pressed => s.wave_speed = (s.wave_speed + 0.2).min(6.0),
                    KeyCode::ArrowDown if pressed => s.wave_speed = (s.wave_speed - 0.2).max(0.1),
                    KeyCode::ArrowLeft if pressed => {
                        s.steer = (s.steer - 0.2).max(-1.0);
                        println!("steer {:+.1}", s.steer);
                    }
                    KeyCode::ArrowRight if pressed => {
                        s.steer = (s.steer + 0.2).min(1.0);
                        println!("steer {:+.1}", s.steer);
                    }
                    _ => {}
                }
            }
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

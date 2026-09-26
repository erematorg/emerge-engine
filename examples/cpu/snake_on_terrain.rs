extern crate emerge_engine as emerge;

#[path = "../scenes/snake_on_terrain.rs"]
mod scene;

use emerge::render::{ColorMode, Renderer};
use emerge::{
    DirectionalContactGrip, FixedStepController, FrameLogger, Lnn, Simulation, per_material_stats,
};
use glam::Vec2;
/// Snake crawling on REAL granular sand terrain -- not the abstract floor
/// boundary `basic_creature.rs` uses. Proves the full chain works together:
/// real terrain material (`DruckerPragerMaterial`), real multi-field contact
/// (`Particle::contact_group`, Bardenhagen 2001), and `DirectionalContactGrip`
/// -- the setae-style asymmetric friction mechanism that makes crawling
/// possible at all, generalized from `RatchetFrictionBoundary`'s
/// fixed-floor-only version to an ARBITRARY real contact interface (so it
/// still works if the terrain isn't flat).
///
/// Body proportions, bilayer fiber arch, alternating segments, and CPG
/// traveling wave are the same locomotion recipe as `basic_creature.rs` --
/// this file only changes WHAT the snake grips (real sand particles instead
/// of a world-edge rule).
///
/// One disclosed caveat: min-J can dip close to (but not past) collapse
/// during the snake's initial impact onto the sand -- worth a gentler spawn
/// drop if this becomes more than a proof-of-concept.
///
///   cargo run --example snake_on_terrain --features render
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

fn make_cpg() -> Lnn {
    make_cpg_biased(0.0)
}

fn make_cpg_biased(bias: f32) -> Lnn {
    let mut lnn = Lnn::coupled_traveling_wave(
        scene::N_RINGS,
        scene::N_PER_RING,
        1.0,
        scene::RING_CROSS_COUPLING,
    );
    lnn.set_ring_bias(0, scene::N_PER_RING, bias);
    lnn.set_ring_bias(1, scene::N_PER_RING, -bias);
    for _ in 0..scene::CPG_BURN_IN_STEPS {
        lnn.step(scene::DT);
    }
    lnn
}

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
    snake_range: std::ops::Range<usize>,
    lnn: Lnn,
    paused: bool,
    wave_speed: f32,
    steer: f32,
    last_reburn_steer: f32,
    /// Shared handle to the real multi-field contact grip -- steering this
    /// updates the SAME instance the solver's `resolve_contact` is already
    /// reading, exactly like `basic_creature.rs`'s ratchet boundary handle.
    grip: Arc<DirectionalContactGrip>,
    renderer: Renderer,
    frame: u64,
    fps_timer: std::time::Instant,
    fps_frames: u64,
    anomaly_latched: bool,
    spawn_centroid: Vec2,
    telemetry_log: FrameLogger,
    // Converts real measured elapsed time into the correct number of physics
    // steps per frame -- calling `sim.step()` once per render frame assumes
    // each frame takes exactly `DT` of real time, which it doesn't, and
    // produces jitter.
    stepper: FixedStepController,
    // Real render-interpolation state (2026-09-09, same fix as
    // `basic_fluids.rs` -- see that file's own `prev_x` doc for the full
    // rationale): a snapshot of every particle's position from before the
    // most recent batch of physics steps, blended against the current
    // position at render time so on-screen motion stays smooth even though
    // this scene's own measured real fps (4-6fps) means several sim seconds
    // can land in one rendered frame.
    prev_x: Vec<Vec2>,
    last_instant: std::time::Instant,
}

fn make_sim() -> (
    Simulation,
    std::ops::Range<usize>,
    Arc<DirectionalContactGrip>,
) {
    let config = scene::base_config();
    let mut sim = Simulation::new(config, scene::terrain_spawn(&config))
        .with_default_material(Box::new(scene::terrain_material()));
    let terrain_count = sim.particles().len();

    let snake_mat_id = sim.register_material(Box::new(scene::snake_material()));
    let snake_spawn = scene::snake_spawn(sim.config(), snake_mat_id.0);
    let snake_range_start = terrain_count;
    let _ = sim.add_body(snake_spawn);
    let snake_range = snake_range_start..sim.particles().len();

    {
        let particles = sim.particles_mut();
        for i in snake_range.clone() {
            // Real multi-field contact tag -- the snake gets its own "grip"
            // velocity field against the terrain's "rest" field, instead of
            // unconditional infinite-friction stick.
            particles.contact_group[i] = scene::SNAKE_CONTACT_GROUP;
            let (group, dir) = scene::snake_particle_tag(particles.x[i]);
            particles.muscle_group_id[i] = group;
            particles.activation_dir[i] = dir;
        }
    }

    let grip = Arc::new(DirectionalContactGrip::new(0.5, 0.5, Vec2::X));
    let sim = sim.with_contact_grip(Arc::clone(&grip));
    (sim, snake_range, grip)
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
        let (sim, snake_range, grip) = make_sim();
        let prev_x = sim.particles().x.clone();
        let mut renderer = Renderer::new(&device, sim.particles().len(), fmt);
        renderer.set_camera(
            &queue,
            scene::GRID as u32,
            size.width,
            size.height,
            0.6,
            true,
        );
        renderer.set_color_mode(ColorMode::ByMaterial);
        println!(
            "snake_on_terrain: {} particles ({} snake)  |  up/down wave speed  left/right STEER  Space pause  R reset  Q quit",
            sim.particles().len(),
            snake_range.len()
        );
        let telemetry_log = FrameLogger::open("snake_on_terrain_telemetry.ndjson")
            .expect("failed to open log file");
        let spawn_centroid = {
            let particles = sim.particles();
            let n = snake_range.len() as f32;
            snake_range.clone().map(|i| particles.x[i]).sum::<Vec2>() / n
        };

        Self {
            surface,
            surface_config: sc,
            device,
            queue,
            sim,
            snake_range,
            lnn: make_cpg(),
            paused: false,
            wave_speed: 1.0,
            steer: 0.0,
            last_reburn_steer: 0.0,
            grip,
            renderer,
            frame: 0,
            fps_timer: std::time::Instant::now(),
            fps_frames: 0,
            anomaly_latched: false,
            spawn_centroid,
            telemetry_log,
            stepper: FixedStepController::standard(scene::DT, 1.0 / scene::DT),
            prev_x,
            last_instant: std::time::Instant::now(),
        }
    }

    fn log_telemetry(&mut self, fps: f32) {
        let snap = self.sim.diagnostics_snapshot();
        let particles = self.sim.particles();
        let n = self.snake_range.len().max(1) as f32;
        let mut centroid = Vec2::ZERO;
        let mut min = Vec2::splat(f32::INFINITY);
        let mut max = Vec2::splat(f32::NEG_INFINITY);
        let mut act_sum = 0.0f32;
        let mut act_max = 0.0f32;
        for i in self.snake_range.clone() {
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
            "f{:<5} fps={:>3.0} | sub={:>2}/{} vmax={:.2} | J=[{:.3},{:.3}] oob={} nan_p={} nan_g={} \
             | centroid=({:.1},{:.1}) drift=({:+.3},{:+.3}) extent=({:.1}x{:.1}) | act mean={:.2} max={:.2}",
            self.frame,
            fps,
            snap.substeps_last_step,
            self.sim.config().max_substeps_per_step,
            snap.max_particle_speed,
            snap.min_deformation_j,
            snap.max_deformation_j,
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
        );

        let stats = per_material_stats(self.sim.particles());
        self.telemetry_log.log(
            self.frame,
            self.sim.config().dt,
            &stats,
            &snap,
            &[(0, "terrain")],
            &[("steer", self.steer), ("wave_speed", self.wave_speed)],
        );

        let mut problems: Vec<String> = Vec::new();
        if snap.non_finite_particle_values > 0 || snap.non_finite_grid_values > 0 {
            problems.push(format!(
                "NON-FINITE: {} particle + {} grid values are NaN/Inf",
                snap.non_finite_particle_values, snap.non_finite_grid_values
            ));
        }
        if snap.min_deformation_j < 0.05 {
            problems.push(format!(
                "near-inverted element: min J = {:.4}",
                snap.min_deformation_j
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
            .set_camera(&self.queue, scene::GRID as u32, w, h, 0.6, true);
    }

    fn update_and_render(&mut self) {
        let now = std::time::Instant::now();
        let frame_delta = (now - self.last_instant).as_secs_f32();
        self.last_instant = now;
        if !self.paused {
            let steps = self.stepper.steps_for_frame(frame_delta);
            // Snapshot BEFORE the batch -- see `prev_x`'s own doc. Skipped
            // when `steps==0` (nothing moved, last snapshot stays valid).
            if steps > 0 {
                self.prev_x.clone_from(&self.sim.particles().x);
            }
            for _ in 0..steps {
                if self.steer != 0.0 {
                    let new_dir_sign = if self.steer >= 0.0 { 1.0 } else { -1.0 };
                    if self.steer != self.last_reburn_steer {
                        self.lnn = make_cpg_biased(self.steer);
                        self.last_reburn_steer = self.steer;
                    }
                    self.grip.set_easy_direction(if new_dir_sign >= 0.0 {
                        Vec2::X
                    } else {
                        Vec2::NEG_X
                    });
                    self.grip.set_friction(0.1, 0.95);
                    self.lnn.step(scene::DT * self.wave_speed);
                    let activations: Vec<f32> = self.lnn.activations().collect();
                    let range = self.snake_range.clone();
                    let particles = self.sim.particles_mut();
                    for i in range {
                        let group = particles.muscle_group_id[i] as usize;
                        particles.activation[i] =
                            (scene::MUSCLE_AMPLITUDE * activations[group]).clamp(0.0, 1.0);
                    }
                } else {
                    self.grip.set_friction(0.5, 0.5);
                    let range = self.snake_range.clone();
                    let particles = self.sim.particles_mut();
                    for i in range {
                        particles.activation[i] = 0.0;
                    }
                }
                self.sim.step();
                self.frame += 1;
            }
        }
        self.fps_frames += 1;
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
        // Real render-interpolation (see `prev_x`'s own doc) -- same
        // contained swap-and-restore pattern as `basic_fluids.rs`.
        let alpha = self.stepper.interpolation_alpha();
        if alpha > 0.0 && self.prev_x.len() == self.sim.particles().len() {
            let blended: Vec<Vec2> = self
                .prev_x
                .iter()
                .zip(self.sim.particles().x.iter())
                .map(|(&prev, &now)| prev.lerp(now, alpha))
                .collect();
            let live = std::mem::replace(&mut self.sim.particles_mut().x, blended);
            self.renderer
                .render(&self.device, &self.queue, self.sim.particles(), &view, true);
            self.sim.particles_mut().x = live;
        } else {
            self.renderer
                .render(&self.device, &self.queue, self.sim.particles(), &view, true);
        }
        output.present();
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        let w = Arc::new(
            el.create_window(
                winit::window::WindowAttributes::default()
                    .with_title("emerge -- Snake on real terrain")
                    .with_inner_size(winit::dpi::LogicalSize::new(640u32, 640u32)),
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
                        let (sim, range, grip) = make_sim();
                        s.spawn_centroid = {
                            let particles = sim.particles();
                            let n = range.len() as f32;
                            range.clone().map(|i| particles.x[i]).sum::<Vec2>() / n
                        };
                        s.prev_x = sim.particles().x.clone();
                        s.sim = sim;
                        s.snake_range = range;
                        s.grip = grip;
                        s.lnn = make_cpg();
                        s.steer = 0.0;
                        s.last_reburn_steer = 0.0;
                        s.frame = 0;
                        s.anomaly_latched = false;
                        s.stepper.reset();
                        s.last_instant = std::time::Instant::now();
                        println!("reset");
                    }
                    KeyCode::ArrowUp if pressed => s.wave_speed = (s.wave_speed + 0.2).min(3.0),
                    KeyCode::ArrowDown if pressed => s.wave_speed = (s.wave_speed - 0.2).max(0.1),
                    KeyCode::ArrowLeft if pressed => {
                        s.steer = (s.steer - 0.2).max(-1.0);
                        println!("steer {:+.1}", s.steer);
                    }
                    KeyCode::ArrowRight if pressed => {
                        s.steer = (s.steer + 0.2).min(1.0);
                        println!("steer {:+.1}", s.steer);
                    }
                    KeyCode::ArrowLeft | KeyCode::ArrowRight if !pressed => {
                        s.steer = 0.0;
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

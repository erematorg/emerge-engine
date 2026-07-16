extern crate emerge_engine as emerge;

use emerge::render::{ColorMode, Renderer};
use emerge::{
    DirectionalContactGrip, DruckerPragerMaterial, FixedStepController, FrameLogger, Lnn,
    NeoHookeanMaterial, SimConfig, Simulation, SpawnRegion, per_material_stats,
};
use glam::{IVec2, Vec2};
/// Snake crawling on REAL granular sand terrain -- not the abstract floor
/// boundary `basic_creature.rs` uses. Proves the full chain works together for
/// the first time: real terrain material (`DruckerPragerMaterial`), real
/// multi-field contact (`Particle::contact_group`, Bardenhagen 2001), and the
/// new `DirectionalContactGrip` (2026-07-13) -- the setae-style asymmetric
/// friction mechanism that makes crawling possible at all, generalized from
/// `RatchetFrictionBoundary`'s fixed-floor-only version to an ARBITRARY real
/// contact interface (so it still works if the terrain isn't flat).
///
/// Everything else (body proportions, bilayer fiber arch, alternating
/// segments, CPG traveling wave) is the exact locomotion recipe verified in
/// `basic_creature.rs` this same session -- this file only changes WHAT the
/// snake grips (real sand particles instead of a world-edge rule).
///
/// Headless-verified before this file existed (`diagnose_snake_crawls_on_real_
/// sand_terrain`, deleted after use): real, substantial, growing crawl
/// (drift.x -21.0 by step 3000, still climbing) with NO terrain explosion --
/// terrain stays in a sane 13.8-20.7 range and settles back near baseline.
/// One real, disclosed caveat: min-J dipped to ~0.0000-0.018 during the
/// initial settle (the snake's first impact onto the sand), close to but not
/// past collapse -- worth a gentler spawn drop if this becomes more than a
/// proof-of-concept.
///
///   cargo run --example snake_on_terrain --features render
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

const GRID: usize = 128;
// REAL BUG FOUND AND FIXED 2026-07-15, TWO ROUNDS (found live on the GPU
// counterpart of this exact scene, applies equally here):
//
// Round 1: the original single `DT=0.1` fed the physics solver 0.1 simulated
// seconds per rendered frame -- at a real ~60fps display frame (~0.0167s real
// time), that's the world running at 6x real-time speed (already a known,
// documented issue elsewhere in this codebase -- `gpu_grid_resolution_cost`'s own
// `REAL_TIME_DT` comment, tests/gpu.rs). Once the terrain was recalibrated to real
// sand stiffness, that 6x inflation became a real, measured performance problem.
//
// Round 2 (a REAL BUG INTRODUCED BY ROUND 1's OWN FIX): the first attempt split
// this into `DT=1/60` (correct) and a separate `DT` left at the OLD
// `0.1`, reasoned as "preserving this file's own locomotion tuning." That
// reasoning was backwards. The CPG steps once per physics frame by whatever DT
// it's given, with no awareness of what a frame represents in real time. Before,
// one frame = 0.1 real seconds and the CPG advanced 0.1 per frame -- matched,
// 1:1 real time. After the split, one frame = 1/60 real seconds (6x shorter) but
// the CPG still advanced by the old 0.1 every frame -- meaning the muscle cycled
// 6x FASTER in real wall-clock time than it was ever tuned for. Confirmed live:
// violent "up/down" spasming and a genuine escalating instability (vmax climbing
// from ~2 to >20 over a long run). There is only ONE real DT: whatever a physics
// frame actually represents in real time. Both the physics solver AND the CPG
// must step by that SAME value to preserve the ORIGINAL real-time gait rate --
// splitting them was the actual bug, not fixing anything. Re-verified headlessly
// after this fix: 8000-step real run, vmax stays flat (1.8-2.3), no escalation.
const DT: f32 = 1.0 / 60.0;
const SNAKE_MUSCLE_GROUPS: u32 = 8;
const N_RINGS: usize = 2;
const N_PER_RING: usize = SNAKE_MUSCLE_GROUPS as usize / N_RINGS;
// Same real, verified values as basic_creature.rs -- see that file's own doc
// history for the full sweep evidence behind each one.
const RING_CROSS_COUPLING: f32 = 0.5;
const MUSCLE_AMPLITUDE: f32 = 0.9;
const CPG_BURN_IN_STEPS: usize = 600;
const SNAKE_CONTACT_GROUP: u32 = 1;
const FIBER_DIAG: f32 = 3.0;
const BODY_LEN: f32 = 36.0 * 0.5;
const BODY_CENTER: Vec2 = Vec2::new(64.0, 20.0);

fn make_cpg() -> Lnn {
    make_cpg_biased(0.0)
}

fn make_cpg_biased(bias: f32) -> Lnn {
    let mut lnn = Lnn::coupled_traveling_wave(N_RINGS, N_PER_RING, 1.0, RING_CROSS_COUPLING);
    lnn.set_ring_bias(0, N_PER_RING, bias);
    lnn.set_ring_bias(1, N_PER_RING, -bias);
    for _ in 0..CPG_BURN_IN_STEPS {
        lnn.step(DT);
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
    // REAL BUG FOUND AND FIXED 2026-07-15 (found live on the GPU counterpart of this
    // exact scene -- see that file's own comment for the full explanation): the
    // previous version called `sim.step()` exactly once per rendered frame, silently
    // assuming each render call corresponds to exactly `DT` of real elapsed
    // time. Real frames don't take exactly that long, and the mismatch is exactly
    // what produces jitter/inconsistent motion pacing. `FixedStepController` (an
    // accumulator this engine already ships) converts REAL measured elapsed time
    // into the correct number of physics steps to run each frame.
    stepper: FixedStepController,
    last_instant: std::time::Instant,
}

fn make_sim() -> (
    Simulation,
    std::ops::Range<usize>,
    Arc<DirectionalContactGrip>,
) {
    // Real granular terrain. History, corrected 2026-07-15: this used to be a
    // much softer (133.3, 0.333) -- an early attempt at (5e4, ...) had exploded
    // and this softer value was picked as a workaround at the time, without
    // registering it was ~750x weaker than this engine's own validated real-sand
    // reference (`sand_angle_of_repose_is_physical`, tests/accuracy.rs, uses
    // `from_young_modulus(1.0e5, 0.2)`) -- a real, if long-undetected, physics bug
    // (the terrain visibly behaved like fluid, not sand). Now uses that same
    // validated (1.0e5, 0.2) pair. The earlier explosion at (5e4, ...) was real
    // but had a SEPARATE cause (a stale `min_dt` override too large for that
    // stiffness, see the `DT`/`min_dt` fixes below) -- not a fundamental
    // ceiling on how stiff this terrain can be.
    let terrain_spawn = SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new(100, 12),
        box_center: Vec2::new(64.0, 10.0),
        material_id: 0,
        precompute_initial_volumes: true,
        ..SpawnRegion::for_sim(&SimConfig {
            // REAL BUG FOUND AND FIXED 2026-07-15 -- see the matching config below
            // (and its own longer comment) for the full explanation: no `min_dt`
            // override here now, inherits the safe `1.0e-3` default.
            max_substeps_per_step: 128,
            project_invalid_state: true,
            ..SimConfig::standard(GRID, DT, Vec2::new(0.0, -0.3))
        })
    };
    // REAL BUG FOUND AND FIXED 2026-07-15: the previous `min_dt: 0.01` override was
    // harmless for the OLD, ~750x-softer terrain (E=133.3), whose own real CFL-safe
    // timestep was comfortably above 0.01 so this floor never actually engaged.
    // `cfl_bound` (src/spacetime/solver/step.rs) clamps the chosen substep to be AT
    // LEAST `min_dt` regardless of what the material's own stability bound requires
    // -- once the terrain was correctly recalibrated to E=1e5 (see below), its true
    // safe timestep dropped well below 0.01, and this override then forced an
    // UNSAFE, too-large step every substep, causing a real explosion (confirmed
    // live on the GPU counterpart of this exact scene: the stiffness fix alone,
    // without this one, blew the scene up). Removed -- inherits
    // `SimConfig::default()`'s own safe `1.0e-3`. `max_substeps_per_step` raised to
    // give the adaptive scheme enough real headroom to actually reach a smaller
    // substep within one frame's DT budget.
    let config = SimConfig {
        max_substeps_per_step: 128,
        project_invalid_state: true,
        ..SimConfig::standard(GRID, DT, Vec2::new(0.0, -0.3))
    };
    // REAL FIX 2026-07-15: the original (133.3, 0.333) was ~750x softer than this
    // codebase's OWN validated real-sand reference
    // (`sand_angle_of_repose_is_physical`, tests/accuracy.rs, uses
    // `from_young_modulus(1.0e5, 0.2)` -- E=1e5 is also sparkl/wgsparkl's own
    // canonical demo value). A DP material this soft deforms continuously under any
    // load instead of holding a rigid granular structure before yielding -- the
    // direct, measured cause of a real user observation on the GPU counterpart of
    // this exact scene ("looks like fluid, not sand"). `cohesionless` is a thin
    // wrapper over `from_young_modulus` (same Lamé conversion), so this is a pure
    // recalibration, not a different construction path.
    let mut sim = Simulation::new(config, terrain_spawn)
        .with_default_material(Box::new(DruckerPragerMaterial::cohesionless(1.0e5, 0.2)));
    let terrain_count = sim.particles().len();

    // Snake: exact recipe verified in basic_creature.rs this session.
    let mut snake_mat = NeoHookeanMaterial::new(13.0, 26.0);
    snake_mat.active_stress_coeff = 80.0;
    snake_mat.viscosity = 150.0;
    let snake_mat_id = sim.register_material(Box::new(snake_mat));
    let snake_spawn = SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new(36, 4),
        box_center: BODY_CENTER,
        material_id: snake_mat_id.0,
        precompute_initial_volumes: true,
        ..SpawnRegion::for_sim(sim.config())
    };
    let snake_range_start = terrain_count;
    let _ = sim.add_body(snake_spawn);
    let snake_range = snake_range_start..sim.particles().len();

    let body_left = BODY_CENTER.x - BODY_LEN / 2.0;
    {
        let particles = sim.particles_mut();
        for i in snake_range.clone() {
            // Real multi-field contact tag -- the snake gets its own "grip"
            // velocity field against the terrain's "rest" field, instead of
            // unconditional infinite-friction stick.
            particles.contact_group[i] = SNAKE_CONTACT_GROUP;
            let t = ((particles.x[i].x - body_left) / BODY_LEN).clamp(0.0, 1.0);
            let group = ((t * SNAKE_MUSCLE_GROUPS as f32) as u32).min(SNAKE_MUSCLE_GROUPS - 1);
            particles.muscle_group_id[i] = group;
            let local_y = particles.x[i].y - BODY_CENTER.y;
            let flip = if group % 2 == 1 { -1.0 } else { 1.0 };
            particles.activation_dir[i] = if local_y >= 0.0 {
                Vec2::new(-FIBER_DIAG * flip, 1.0).normalize()
            } else {
                Vec2::new(FIBER_DIAG * flip, 1.0).normalize()
            };
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
        let mut renderer = Renderer::new(&device, sim.particles().len(), fmt);
        renderer.set_camera(&queue, GRID as u32, size.width, size.height, 0.6, true);
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
            stepper: FixedStepController::standard(DT, 1.0 / DT),
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
            .set_camera(&self.queue, GRID as u32, w, h, 0.6, true);
    }

    fn update_and_render(&mut self) {
        let now = std::time::Instant::now();
        let frame_delta = (now - self.last_instant).as_secs_f32();
        self.last_instant = now;
        if !self.paused {
            let steps = self.stepper.steps_for_frame(frame_delta);
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
                    self.lnn.step(DT * self.wave_speed);
                    let activations: Vec<f32> = self.lnn.activations().collect();
                    let range = self.snake_range.clone();
                    let particles = self.sim.particles_mut();
                    for i in range {
                        let group = particles.muscle_group_id[i] as usize;
                        particles.activation[i] =
                            (MUSCLE_AMPLITUDE * activations[group]).clamp(0.0, 1.0);
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

extern crate emerge_engine as emerge;

#[path = "../gui_common/mod.rs"]
mod gui_common;

/// Close-up scene for grain rolling: few grains, the camera zoomed in, nothing else
/// competing for attention, so one grain's rotation can be followed (in
/// `sand_repose_angle.rs`'s Grains mode ~80 grains collapse chaotically). The mechanism
/// is also tested in `tests/grains_grid_coupling.rs` and
/// `tests/particle_neighbor_momentum_transfer.rs`.
///
/// # Inclined-plane physics from the terrain
/// `HeightmapBoundary` derives a local surface normal from the heightmap's slope
/// (`src/forces/boundary/heightmap.rs`), so any scene with terrain gets inclined-plane
/// physics; with a fixed +Y normal a sloped heightmap would act as a staircase of flat
/// blocks that never pushes a body downhill. A flat floor's normal is exactly +Y
/// everywhere (`tests/stress.rs`: `boundary_count_stress`,
/// `flat_floor_normal_is_exactly_up_everywhere`,
/// `sloped_heightmap_normal_is_tilted_and_lets_gravity_drive_motion_downhill`). This scene
/// builds a ramp (rising terrain, then a flat landing) with ordinary straight-down
/// gravity, no per-demo gravity rotation.
///
///   cargo run --example grain_rolling_closeup --features render
use emerge::grains::population::GrainPopulation;
use emerge::materials::granular::grain_contact_law::{ContactLawConfig, critical_timestep};
use emerge::particle::{Grain, Particle};
use emerge::render::{ColorMode, Renderer};
use emerge::{HeightmapBoundary, SimConfig, Simulation};
use glam::{Mat2, Vec2};
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

const GRID: usize = 40;
const FLOOR: f32 = 3.0;
const RAMP_START_X: usize = 4;
const RAMP_END_X: usize = 22;
const N_GRAINS: usize = 4;
const GRAIN_RADIUS: f32 = 1.0;
const GRAIN_MASS: f32 = 1.0;
const GRAINS_MARKER_MAT_ID: u32 = 1;
const GRAINS_ACCENT_MAT_ID: u32 = 2;
const TERRAIN_MARKER_MAT_ID: u32 = 3;
const SIGMA_GRAIN: [f32; 3] = [0.100, 0.300, 0.800];
const SIGMA_ACCENT: [f32; 3] = [0.900, 0.900, 0.900];
const SIGMA_TERRAIN: [f32; 3] = [0.550, 0.400, 0.220];
/// Two terrain markers per grid column -- traces `build_heights` with the
/// SAME linear interpolation `HeightmapBoundary::height_at_f32` uses
/// internally for grain contact (real physics input, not a decorative
/// line): draws exactly the surface the boundary actually enforces, not an
/// approximation of it.
const TERRAIN_SAMPLES_PER_CELL: usize = 2;

/// The stiffness of `sand_repose_angle.rs` (see its `grain_contact_config` for why SI
/// stiffness would need a very fine forced dt).
///
/// `rolling_friction = 0.02`, the low rolling resistance of smooth, hard bodies (steel,
/// glass). `sand_repose_angle.rs`'s `0.2` is calibrated for dry sand's angle of repose,
/// a pile of angular, interlocking grains meant to stay put, not a single smooth ball
/// meant to roll: with it a grain dropped onto this ramp spins briefly on impact, then
/// stops for good (reproduced by `tests/grains_grid_coupling.rs`'s
/// `diag_grain_dropped_onto_22deg_ramp_matches_live_demo_spawn`). With 0.02 the same
/// diagnostic shows continuous, consistent rolling (v.y/v.x tracking -tan(incline)
/// throughout, neither launching nor freezing).
fn grain_contact_config() -> ContactLawConfig {
    let m_eff = GRAIN_MASS * 0.5;
    const DAMPING_RATIO: f32 = 0.6;
    let critical_damping = |k: f32| 2.0 * (k * m_eff).sqrt() * DAMPING_RATIO;
    let normal_stiffness = 1.0e4;
    let tangential_stiffness = 0.8e4;
    let rolling_stiffness = 5.0e2;
    ContactLawConfig {
        normal_stiffness,
        tangential_stiffness,
        rolling_stiffness,
        normal_damping: critical_damping(normal_stiffness),
        tangential_damping: critical_damping(tangential_stiffness),
        rolling_damping: critical_damping(rolling_stiffness),
        friction: (35.0_f32).to_radians().tan(),
        rolling_friction: 0.02,
    }
}

/// Ramp: terrain is elevated for x < RAMP_START_X, descends linearly
/// down to `FLOOR` by x=RAMP_END_X, then a flat landing at `FLOOR` for the
/// rest of the domain -- grains released near the top (small x) roll DOWN
/// toward increasing x, landing and settling on the flat run-out.
/// `incline_deg` controls the ramp's own rise/run ratio (steeper angle =
/// taller top for the same horizontal span), a geometric parameter,
/// not a gravity trick.
fn build_heights(incline_deg: f32) -> Vec<f32> {
    let run = (RAMP_END_X - RAMP_START_X) as f32;
    let rise = run * incline_deg.to_radians().tan();
    (0..GRID)
        .map(|x| {
            if x < RAMP_START_X {
                FLOOR + rise
            } else if x < RAMP_END_X {
                let t = (x - RAMP_START_X) as f32 / run;
                FLOOR + rise * (1.0 - t)
            } else {
                FLOOR
            }
        })
        .collect()
}

/// Linear interpolation between bracketing columns -- mirrors
/// `HeightmapBoundary::height_at_f32` (private to that module) exactly, so
/// the rendered line matches the contact surface a rolling grain
/// actually feels, not a coarser approximation of it.
fn height_at_f32(heights: &[f32], x: f32) -> f32 {
    if heights.is_empty() {
        return 0.0;
    }
    let last = heights.len() - 1;
    let x0 = (x.floor().max(0.0) as usize).min(last);
    let x1 = (x0 + 1).min(last);
    let t = (x - x0 as f32).clamp(0.0, 1.0);
    heights[x0] * (1.0 - t) + heights[x1] * t
}

/// Terrain-surface markers, so the ramp is visible: they trace the `heights` array
/// `make_sim` feeds into `HeightmapBoundary` (the physics input, so the line cannot
/// drift from it), `TERRAIN_SAMPLES_PER_CELL` points per grid column, through
/// `height_at_f32`'s interpolation.
fn terrain_markers(incline_deg: f32) -> Vec<Particle> {
    let heights = build_heights(incline_deg);
    let n = (GRID - 1) * TERRAIN_SAMPLES_PER_CELL;
    (0..=n)
        .map(|i| {
            let x = i as f32 / TERRAIN_SAMPLES_PER_CELL as f32;
            let y = height_at_f32(&heights, x);
            let mut p = Particle::zeroed();
            p.x = Vec2::new(x, y);
            p.mass = 1.0;
            p.initial_volume = 1.0;
            p.volume = 1.0;
            p.density = 1.0;
            p.material_id = TERRAIN_MARKER_MAT_ID;
            let scale = 2.0 * 0.32 / 0.7;
            p.deformation_gradient = Mat2::from_diagonal(Vec2::splat(scale));
            p
        })
        .collect()
}

struct SmallRng(u64);
impl SmallRng {
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((self.0 >> 33) as f32) / (u32::MAX as f32)
    }
}

fn make_sim(incline_deg: f32) -> Simulation {
    let cfg = grain_contact_config();
    let m_eff = GRAIN_MASS * 0.5;
    let dt_crit = critical_timestep(m_eff, &cfg);
    let grain_safe_dt = (dt_crit * 0.2).min(0.02);
    let config = SimConfig {
        grid_res: GRID,
        dt: grain_safe_dt,
        gravity: Vec2::new(0.0, -0.3), // real, ordinary straight-down gravity
        adaptive_timestep: true,
        boundary_thickness: 2,
        ..SimConfig::default()
    };
    // Load-bearing choice, not an oversight: grid-level friction=0.0.
    // The grid's own per-cell boundary correction is noisy for a grain's own
    // kernel-spread momentum (see `GrainPopulation::clean_wall_normal_
    // velocity`'s doc, `src/spacetime/grains/population.rs`) -- the
    // grid now owns NORMAL enforcement only; ALL real tangential/rolling
    // physics for grains comes from the new `resolve_wall_contact`
    // mechanism (real Coulomb friction still applied there, via
    // `grain_contact_config()`'s own `friction` field).
    let heights = build_heights(incline_deg);
    let mut solver = Simulation::empty(config).with_boundary(Box::new(HeightmapBoundary::new(
        heights,
        0.0,
        config.boundary_thickness,
    )));

    // Jitter, as in every grain scene (an unjittered lattice has no asymmetry to roll
    // or topple). Placed near the top of the ramp (just past RAMP_START_X, where the
    // downslope begins) so rolling starts almost at once; each grain's drop height is
    // measured from the ramp's height at its own x (the ramp is not flat).
    let mut rng = SmallRng(0xA11C_E5EE_u64);
    let start_x = RAMP_START_X as f32 + 1.5;
    let spacing = 2.6 * GRAIN_RADIUS;
    let ramp_height_at = |x: f32| -> f32 {
        let col = (x.round() as usize).min(GRID - 1);
        build_heights(incline_deg)[col]
    };
    let grains: Vec<Grain> = (0..N_GRAINS)
        .map(|i| {
            let jx = (rng.next_f32() - 0.5) * 0.3 * spacing;
            let jy = (rng.next_f32() - 0.5) * 0.3 * spacing;
            let x = start_x + i as f32 * spacing + jx;
            let y = ramp_height_at(x) + GRAIN_RADIUS + 1.5 + jy; // real, small drop onto the ramp surface
            let r = GRAIN_RADIUS * (0.9 + 0.2 * rng.next_f32());
            Grain::new(Vec2::new(x, y), r, GRAIN_MASS * (r / GRAIN_RADIUS).powi(2))
        })
        .collect();
    solver.add_grain_population(GrainPopulation::new(grains, cfg));
    solver
}

struct State {
    gfx: gui_common::Gfx,
    sim: Simulation,
    renderer: Renderer,
    incline_deg: f32,
    paused: bool,
    sim_speed: u32,
    step: u64,
    fps_timer: std::time::Instant,
    fps_frames: u64,
    last_fps: f32,
    cursor_pos: [f32; 2],
}

/// Click-to-nudge: LMB applies a small radial impulse to any grain within
/// `NUDGE_RADIUS` of the cursor's grid position, to perturb a settled grain and watch
/// it roll again (the Grains mode of `sand_repose_angle.rs` has no push/pull).
const NUDGE_RADIUS: f32 = 3.0;
const NUDGE_STRENGTH: f32 = 3.0;

impl State {
    async fn new(window: Arc<Window>) -> Self {
        let gfx = gui_common::Gfx::new(&window).await;
        let size = window.inner_size();
        let incline_deg = 22.0;
        let sim = make_sim(incline_deg);
        // Headroom: 2 marker particles per grain (body + rolling accent dot), as
        // `sand_repose_angle.rs`, plus 2 terrain markers per grid column tracing the
        // ramp surface (the `HeightmapBoundary::heights` this scene builds, see
        // `terrain_markers`).
        let render_capacity = 2 * N_GRAINS + TERRAIN_SAMPLES_PER_CELL * GRID;
        let mut renderer = Renderer::new(&gfx.device, render_capacity, gfx.format);
        // Zoomed in tight -- particle_scale=1.4 (well above the usual 0.6-0.9)
        // is the actual point of this scene: a handful of grains filling
        // enough of the screen that individual rotation is unmistakable.
        renderer.set_camera(&gfx.queue, GRID as u32, size.width, size.height, 1.4, true);
        renderer.set_color_mode(ColorMode::ByPhysics);
        renderer.set_optical_params(&gfx.queue, GRAINS_MARKER_MAT_ID as usize, SIGMA_GRAIN);
        renderer.set_optical_params(&gfx.queue, GRAINS_ACCENT_MAT_ID as usize, SIGMA_ACCENT);
        renderer.set_optical_params(&gfx.queue, TERRAIN_MARKER_MAT_ID as usize, SIGMA_TERRAIN);

        println!(
            "grain_rolling_closeup: {N_GRAINS} grains  |  SPACE=pause  R=reset  UP/DOWN=incline angle  LMB=nudge nearest grain  Q=quit"
        );
        Self {
            gfx,
            sim,
            renderer,
            incline_deg,
            paused: false,
            sim_speed: 15,
            step: 0,
            fps_timer: std::time::Instant::now(),
            fps_frames: 0,
            last_fps: 0.0,
            cursor_pos: [0.0; 2],
        }
    }

    fn cursor_grid(&self) -> Vec2 {
        gui_common::cursor_to_grid(
            self.cursor_pos,
            self.gfx.surface_config.width,
            self.gfx.surface_config.height,
            GRID,
        )
    }

    /// Click-to-nudge: applies a fixed impulse, directed away from the click point, to
    /// the nearest grain within `NUDGE_RADIUS`, to perturb a settled grain and watch it
    /// roll again.
    fn nudge_at_cursor(&mut self) {
        let cursor = self.cursor_grid();
        let population = &mut self.sim.grain_populations_mut()[0];
        let Some(grain) = population
            .grains
            .iter_mut()
            .filter(|g| (g.x - cursor).length() < NUDGE_RADIUS)
            .min_by(|a, b| (a.x - cursor).length().total_cmp(&(b.x - cursor).length()))
        else {
            return;
        };
        let dir = (grain.x - cursor).normalize_or_zero();
        let dir = if dir == Vec2::ZERO { Vec2::X } else { dir };
        grain.v += dir * NUDGE_STRENGTH;
    }

    fn resize(&mut self, w: u32, h: u32) {
        self.gfx.resize(w, h);
        if w == 0 || h == 0 {
            return;
        }
        self.renderer
            .set_camera(&self.gfx.queue, GRID as u32, w, h, 1.4, true);
    }

    fn reset(&mut self) {
        self.sim = make_sim(self.incline_deg);
        self.step = 0;
    }

    fn update_and_render(&mut self, window: &Window) {
        if !self.paused {
            for _ in 0..self.sim_speed {
                self.sim.step();
                self.step += 1;
            }
        }
        self.fps_frames += 1;
        if self.fps_timer.elapsed().as_secs_f32() >= 1.0 {
            self.last_fps = self.fps_frames as f32 / self.fps_timer.elapsed().as_secs_f32();
            self.fps_timer = std::time::Instant::now();
            self.fps_frames = 0;
        }

        let output = match self.gfx.surface.get_current_texture() {
            Ok(t) => t,
            Err(_) => return,
        };
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // Same real marker-particle rendering technique already proven in
        // `sand_repose_angle.rs`: grains aren't ordinary `Particle`s, so
        // a zero-physics render proxy carries their physically
        // simulated position/orientation straight through.
        let mut all: Vec<Particle> = terrain_markers(self.incline_deg);
        for grain in &self.sim.grain_populations()[0].grains {
            let mut p = Particle::zeroed();
            p.x = grain.x;
            p.v = grain.v;
            p.mass = 1.0;
            p.initial_volume = 1.0;
            p.volume = 1.0;
            p.density = 1.0;
            p.material_id = GRAINS_MARKER_MAT_ID;
            let scale = 2.0 * grain.radius / 0.7;
            p.deformation_gradient = Mat2::from_diagonal(Vec2::splat(scale));
            all.push(p);

            let mut accent = Particle::zeroed();
            let offset = Vec2::from_angle(grain.orientation) * (grain.radius * 0.6);
            accent.x = grain.x + offset;
            accent.v = grain.v;
            accent.mass = 1.0;
            accent.initial_volume = 1.0;
            accent.volume = 1.0;
            accent.density = 1.0;
            accent.material_id = GRAINS_ACCENT_MAT_ID;
            let accent_scale = 2.0 * (grain.radius * 0.28) / 0.7;
            accent.deformation_gradient = Mat2::from_diagonal(Vec2::splat(accent_scale));
            all.push(accent);
        }
        let marker_particles = emerge::particle::Particles::from(all);
        self.renderer.render(
            &self.gfx.device,
            &self.gfx.queue,
            &marker_particles,
            &view,
            true,
        );

        let fps = self.last_fps;
        let step = self.step;
        let mut paused = self.paused;
        let mut sim_speed = self.sim_speed;
        let mut incline_deg = self.incline_deg;
        let mut do_reset = false;
        let incline_changed_before = incline_deg;

        gui_common::run_egui_frame(&mut self.gfx, window, &view, |ctx| {
            egui::Window::new("Grain rolling close-up")
                .default_pos([10.0, 10.0])
                .default_width(300.0)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.label(format!("fps={fps:.0}  step={step}"));
                    ui.separator();
                    ui.label(
                        "Watch the small accent dot on each grain -- it orbits \
                         the grain's own center as it genuinely rolls, real \
                         integrated `Grain::orientation`, not decoration.",
                    );
                    ui.separator();
                    ui.label("Ramp angle (real HeightmapBoundary slope):");
                    ui.add(egui::Slider::new(&mut incline_deg, 0.0..=45.0).suffix(" deg"));
                    ui.label("Sim speed (physics steps/frame):");
                    ui.add(egui::Slider::new(&mut sim_speed, 1..=60));
                    ui.separator();
                    ui.checkbox(&mut paused, "Paused (or SPACE)");
                    if ui.button("Reset").clicked() {
                        do_reset = true;
                    }
                    ui.separator();
                    ui.label("SPACE pause  R reset  Q quit");
                });
        });
        self.paused = paused;
        self.sim_speed = sim_speed;
        self.incline_deg = incline_deg;
        // Changing the incline rebuilds the ramp's own real heights, so the
        // whole scene resets (a `HeightmapBoundary` isn't live-mutable the
        // way a scalar like gravity is).
        if (incline_deg - incline_changed_before).abs() > 1e-6 || do_reset {
            self.reset();
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
                    .with_title("emerge -- Grain rolling close-up")
                    .with_inner_size(winit::dpi::LogicalSize::new(560u32, 560u32)),
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
            let resp = s.gfx.egui_state.on_window_event(w, &event);
            if resp.consumed {
                return;
            }
        }
        match event {
            WindowEvent::CloseRequested => el.exit(),
            WindowEvent::CursorMoved { position, .. } => {
                s.cursor_pos = [position.x as f32, position.y as f32];
            }
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: winit::event::MouseButton::Left,
                ..
            } => s.nudge_at_cursor(),
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
                    KeyCode::Space if pressed => s.paused = !s.paused,
                    KeyCode::KeyR if pressed => s.reset(),
                    KeyCode::ArrowUp if pressed => {
                        s.incline_deg = (s.incline_deg + 2.0).min(45.0);
                        s.reset();
                    }
                    KeyCode::ArrowDown if pressed => {
                        s.incline_deg = (s.incline_deg - 2.0).max(0.0);
                        s.reset();
                    }
                    KeyCode::Escape | KeyCode::KeyQ if pressed => el.exit(),
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

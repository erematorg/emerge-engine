extern crate emerge_engine as emerge;

#[path = "../gui_common/mod.rs"]
mod gui_common;

/// Newton's cradle at real scale and gravity: five chrome steel balls of
/// 1 cm radius on 14 cm strings, as discs of unit depth through the 2D
/// grain contract (`Grain::from_si`, `GrainPopulation::new_disc`, the line
/// contact of `materials::granular::disc_contact`). Deliberately has NO
/// terrain or boundary: every grain hangs in open space from its own fixed
/// anchor, so the demo stands or falls on the grain-grain contact law.
/// Chrome steel is nearly perfectly elastic, so momentum passes through the
/// row and only the end ball swings out, with very little visible decay:
/// that is the physically right result, not a missing damping.
///
/// # The string
/// The engine has no rigid-joint/constraint solver, so each pendulum string is a rigid
/// distance constraint applied after each physics step: the position is projected
/// back onto the fixed-radius circle around the anchor and the radial velocity is
/// zeroed, the velocity cleaning `GrainPopulation::clean_wall_normal_velocity` applies
/// at walls, along the string instead of a floor normal. This is position-based
/// dynamics (Jakobsen 2001). Gravity, mass and every grain-grain collision are the
/// engine's own physics; only the inextensible string is imposed, like a cradle's
/// wires.
///
///   cargo run --example grain_newtons_cradle --features render
use emerge::grains::population::GrainPopulation;
use emerge::materials::granular::disc_contact::{DiscContactConfig, DiscElastic};
use emerge::particle::{Grain, Particle};
use emerge::render::{ColorMode, Renderer};
use emerge::{Elastic, SimConfig};
use glam::{Mat2, Vec2};
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

/// Gravity for a standalone `GrainPopulation`, which this demo drives directly rather
/// than through a `Simulation`. Grid coupling (`spacetime::grains::coupling`, P2G ->
/// grid_update -> G2P before contact resolution) mixes momentum between close grains in
/// a way that does not shrink with resolution: from grid_cell_size 1.0 to 0.125, with
/// the whole-row gap, the grain3/grain4 ratio stays ~0.67, against ~1.0 standalone (see
/// `tests/grains_grid_coupling.rs::newtons_cradle_two_ball_release_with_real_gap_survives_repeated_strikes`).
/// The quadratic B-spline support is ~3 cells wide, so its physical width scales with
/// the cell size and inter-grain kernel overlap does not shrink relative to a grain's
/// radius. With no terrain and no particles, this scene needs no grid.
///
/// A plain `Vec2`, as uniform gravity is throughout the engine (`SimConfig::gravity`;
/// `Field`'s doc: "uniform body forces go into `SimConfig::gravity`"), not a
/// `Field`/`GrainField`.
const GRAVITY: Vec2 = Vec2::new(0.0, -9.81 / DX_M);

/// One cell is 1 cm: the balls' 1 cm radius is one cell and the geometry in
/// cells is the demo's own. Gravity above is the real 9.81 m/s^2.
const DX_M: f32 = 0.01;

/// Pure rendering-space coordinate convention (camera framing, cursor-to-
/// world mapping) -- no MPM grid exists anymore, this just keeps the same
/// visual scale/framing the demo always used.
const RENDER_SPACE: usize = 40;
const N_GRAINS: usize = 5;
const GRAIN_RADIUS: f32 = 1.0;
const STRING_LENGTH: f32 = 14.0;
const ANCHOR_Y: f32 = 34.0;
const ANCHOR_START_X: f32 = 13.0;
const GRAINS_MARKER_MAT_ID: u32 = 1;
const GRAINS_ACCENT_MAT_ID: u32 = 2;
const STRING_MARKER_MAT_ID: u32 = 3;
const SIGMA_GRAIN: [f32; 3] = [0.100, 0.300, 0.800];
const SIGMA_ACCENT: [f32; 3] = [0.900, 0.900, 0.900];
const SIGMA_STRING: [f32; 3] = [0.700, 0.700, 0.700];
/// Points drawn along each string, anchor to grain, evenly spaced -- real
/// physics positions (linear interpolation of the two true endpoints), not
/// decoration.
const STRING_SAMPLES: usize = 10;

/// Chrome steel, AISI 52100: the alloy Hoover Precision's chrome steel
/// balls are made of ("All sizes and grades are manufactured from AISI Type
/// 52100 steel", hooverprecision.com/chrball.htm, archived April 2000),
/// the balls of the impact table cited at `RESTITUTION`. Young's modulus
/// 29.5e6 psi (203.4 GPa) and density 0.283 lb/in^3 (7833 kg/m^3) from
/// that same page; Carpenter's CarTech 52100 datasheet gives 29.0e3 ksi
/// (199.9 GPa) and 0.2830 lb/in^3. Poisson's ratio 0.29 as stated for
/// hardened 52100 on makeitfrom.com ("Hardened 52100 Chromium Steel",
/// with E 190 GPa and 7.8 g/cm^3), inside the 0.27 to 0.30 AZoM gives for
/// AISI 52100 (azom.com, article 6704). CarTech states no Poisson's ratio;
/// its E over twice its modulus of rigidity, minus one, would give 0.208,
/// but a ratio taken from two separately rounded moduli amplifies their
/// error, so it is not used.
fn steel() -> Elastic {
    Elastic {
        e_pa: 203.4e9,
        nu: 0.29,
        rho_kg_m3: 7833.0,
    }
}

/// The unit conversion the 2D grain contract needs (cell size and
/// reference density); this demo has no MPM grid.
fn units() -> SimConfig {
    SimConfig::earth(RENDER_SPACE, DX_M, 1.0 / 60.0)
}

/// Steel balls in line contact: `RESTITUTION`, sliding friction 0.54 (the
/// same impact table: chrome steel on chrome steel, 0.54 +- 0.05), and no
/// rolling resistance (none measured; head-on strikes barely spin the
/// balls).
fn contact_config() -> DiscContactConfig {
    DiscContactConfig::new(
        DiscElastic::from_si(&steel(), &units()),
        RESTITUTION,
        0.54,
        0.0,
        0.0,
        0.0,
    )
}

/// Chrome steel on chrome steel: 1.00 +- 0.01, measured on 3.18 mm balls
/// (Hoover Precision, 7.83 g/cm^3) in the impact results table of M. Y.
/// Louge's granular flow laboratory at Cornell ("Fall 1999 Impact Parameter
/// Chart", grainflowresearch.mae.cornell.edu/impact/data). Cited as that
/// table, not as Foerster, Louge, Chang and Allia 1994, which predates the
/// chart: the page does not say which row came from which paper. 0.99 sits
/// within its uncertainty and keeps a little loss per strike.
const RESTITUTION: f32 = 0.99;

/// Gap between every adjacent pair in the row, as a fraction of grain radius. Grains
/// touching at an exactly zero gap engage their contact simultaneously with the next
/// collision instead of sequentially, which breaks "N in, N out" momentum transfer
/// (an analytical sequential-collision cross-check shows it; `contact_iterations` and
/// stiffness sweeps up to 100,000x do not fix it, see
/// `tests/grains_grid_coupling.rs::newtons_cradle_two_ball_release_with_real_initial_gap_matches_conservation`).
/// The gap is on every pair, not only the released one, since each re-strike (the
/// launched balls swinging back) meets the rest of the row too. 5% of radius, closer to
/// real touching balls than an exactly zero gap, fixes the first strike and a simulated
/// return strike (grain3/grain4 ratio 0.65 -> 1.002).
const RELEASE_GAP_FRACTION: f32 = 0.05;

/// Anchor spacing is slightly WIDER than exact touching distance (`2 *
/// radius`) -- see `RELEASE_GAP_FRACTION`'s doc for why. Every grain
/// hangs from its own anchor at the same `STRING_LENGTH`, so this spacing
/// alone gives every neighbor pair the same small real gap at rest.
fn anchor(i: usize) -> Vec2 {
    let spacing = 2.0 * GRAIN_RADIUS * (1.0 + RELEASE_GAP_FRACTION);
    Vec2::new(ANCHOR_START_X + i as f32 * spacing, ANCHOR_Y)
}

fn rest_position(i: usize) -> Vec2 {
    anchor(i) + Vec2::new(0.0, -STRING_LENGTH)
}

/// Pulls grain `i` out to the left by `pull_deg`, rotated about its OWN
/// anchor -- generalizes the classic single-ball cradle setup to lifting
/// `pull_count` balls TOGETHER (same angle keeps them at the same real gap
/// from `anchor`'s own spacing, exactly like a hand lifting several
/// balls at once).
fn pulled_position(i: usize, pull_deg: f32) -> Vec2 {
    let theta = pull_deg.to_radians();
    anchor(i) + STRING_LENGTH * Vec2::new(-theta.sin(), -theta.cos())
}

/// Builds the grain population directly (see `RENDER_SPACE`'s doc for
/// why -- this demo drives `GrainPopulation` on its own instead of going
/// through `Simulation`/the MPM grid). Returns the population and its own
/// contact-law-derived stable timestep.
fn make_population(pull_deg: f32, pull_count: usize) -> (GrainPopulation, f32) {
    let units = units();
    let grains: Vec<Grain> = (0..N_GRAINS)
        .map(|i| {
            let pos = if i < pull_count {
                pulled_position(i, pull_deg)
            } else {
                rest_position(i)
            };
            Grain::from_si(pos, GRAIN_RADIUS * DX_M, &steel(), &units)
        })
        .collect();
    let population = GrainPopulation::new_disc(grains, contact_config());
    // The contacts' own stable step, at the fraction the materials use.
    let dt = population.contact_step_limit() * units.material_cfl_coefficient;
    (population, dt)
}

/// Rigid distance constraint (see the file doc), run once per physics step after
/// `GrainPopulation::step()`, on each grain's state.
fn apply_string_constraints(population: &mut GrainPopulation) {
    for (i, grain) in population.grains.iter_mut().enumerate() {
        let a = anchor(i);
        let to_grain = grain.x - a;
        let dist = to_grain.length();
        if dist < 1.0e-6 {
            continue;
        }
        let dir = to_grain / dist;
        grain.x = a + dir * STRING_LENGTH;
        let v_radial = grain.v.dot(dir);
        grain.v -= v_radial * dir;
    }
}

struct State {
    gfx: gui_common::Gfx,
    population: GrainPopulation,
    dt: f32,
    renderer: Renderer,
    pull_deg: f32,
    pull_count: usize,
    /// Per-grain peak `|v|` since the last reset: many swings later the live
    /// `|v|` no longer shows the first strike, which is what the demo is
    /// for.
    peak_speed: [f32; N_GRAINS],
    paused: bool,
    /// Simulated seconds per displayed second.
    time_scale: f32,
    step: u64,
    fps_timer: std::time::Instant,
    fps_frames: u64,
    last_fps: f32,
    cursor_pos: [f32; 2],
}

const NUDGE_RADIUS: f32 = 3.0;
/// Speed a nudge adds, in cells/s (0.5 m/s).
const NUDGE_STRENGTH: f32 = 50.0;
/// Displayed frame time the physics advances by, times `time_scale`.
const FRAME_S: f32 = 1.0 / 60.0;

impl State {
    async fn new(window: Arc<Window>) -> Self {
        let gfx = gui_common::Gfx::new(&window).await;
        let size = window.inner_size();
        let pull_deg = 40.0;
        let pull_count = 2;
        let (population, dt) = make_population(pull_deg, pull_count);
        // 2 marker particles per grain (body + spin accent) + string-line
        // samples for every grain.
        let render_capacity = 2 * N_GRAINS + STRING_SAMPLES * N_GRAINS;
        let mut renderer = Renderer::new(&gfx.device, render_capacity, gfx.format);
        renderer.set_camera(
            &gfx.queue,
            RENDER_SPACE as u32,
            size.width,
            size.height,
            1.1,
            true,
        );
        renderer.set_color_mode(ColorMode::ByPhysics);
        renderer.set_optical_params(&gfx.queue, GRAINS_MARKER_MAT_ID as usize, SIGMA_GRAIN);
        renderer.set_optical_params(&gfx.queue, GRAINS_ACCENT_MAT_ID as usize, SIGMA_ACCENT);
        renderer.set_optical_params(&gfx.queue, STRING_MARKER_MAT_ID as usize, SIGMA_STRING);

        println!(
            "grain_newtons_cradle: {N_GRAINS} grains  |  SPACE=pause  R=reset  LMB=nudge nearest grain  Q=quit"
        );
        Self {
            gfx,
            population,
            dt,
            renderer,
            pull_deg,
            pull_count,
            peak_speed: [0.0; N_GRAINS],
            paused: false,
            time_scale: 0.25,
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
            RENDER_SPACE,
        )
    }

    fn nudge_at_cursor(&mut self) {
        let cursor = self.cursor_grid();
        let Some(grain) = self
            .population
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
            .set_camera(&self.gfx.queue, RENDER_SPACE as u32, w, h, 1.1, true);
    }

    fn reset(&mut self) {
        let (population, dt) = make_population(self.pull_deg, self.pull_count);
        self.population = population;
        self.dt = dt;
        self.step = 0;
        self.peak_speed = [0.0; N_GRAINS];
        println!(
            "RESET pull_deg={:.1} pull_count={} dt={} time_scale={}",
            self.pull_deg, self.pull_count, self.dt, self.time_scale
        );
    }

    fn update_and_render(&mut self, window: &Window) {
        if !self.paused {
            let steps = (self.time_scale * FRAME_S / self.dt).ceil() as u32;
            for _ in 0..steps {
                self.population.step(GRAVITY, self.dt);
                apply_string_constraints(&mut self.population);
                self.step += 1;
                if self.step.is_multiple_of(200_000) {
                    let v: Vec<f32> = self
                        .population
                        .grains
                        .iter()
                        .map(|g| g.v.length())
                        .collect();
                    println!(
                        "LOGSTEP step={} pull_deg={:.1} pull_count={} v={v:?}",
                        self.step, self.pull_deg, self.pull_count
                    );
                }
                for (i, grain) in self.population.grains.iter().enumerate() {
                    self.peak_speed[i] = self.peak_speed[i].max(grain.v.length());
                }
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

        let mut all: Vec<Particle> = Vec::new();
        let grains: Vec<Grain> = self.population.grains.clone();
        for (i, grain) in grains.iter().enumerate() {
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

            let a = anchor(i);
            for s in 0..STRING_SAMPLES {
                let t = s as f32 / (STRING_SAMPLES - 1) as f32;
                let mut sp = Particle::zeroed();
                sp.x = a + (grain.x - a) * t;
                sp.mass = 1.0;
                sp.initial_volume = 1.0;
                sp.volume = 1.0;
                sp.density = 1.0;
                sp.material_id = STRING_MARKER_MAT_ID;
                let ss = 2.0 * 0.12 / 0.7;
                sp.deformation_gradient = Mat2::from_diagonal(Vec2::splat(ss));
                all.push(sp);
            }
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
        let mut time_scale = self.time_scale;
        let mut pull_deg = self.pull_deg;
        let mut pull_count = self.pull_count;
        let mut do_reset = false;
        let pull_before = pull_deg;
        let pull_count_before = pull_count;
        let speeds: Vec<f32> = grains.iter().map(|g| g.v.length()).collect();

        gui_common::run_egui_frame(&mut self.gfx, window, &view, |ctx| {
            egui::Window::new("Newton's cradle")
                .default_pos([10.0, 10.0])
                .default_width(300.0)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.label(format!("fps={fps:.0}  step={step}"));
                    ui.separator();
                    ui.label(
                        "Real grain-grain DEM contact, real rigid string \
                         constraints -- lifting N balls together should \
                         launch exactly N balls out the far end, matched \
                         in speed (real momentum+energy conservation).",
                    );
                    ui.separator();
                    ui.label(
                        "peak = highest |v| reached since Reset (the real, \
                         verified signal -- live |v| alone gets buried by \
                         later swing cycles, see Reset to re-arm):",
                    );
                    for (i, s) in speeds.iter().enumerate() {
                        ui.label(format!(
                            "grain {i}: |v|={:.3} m/s   peak={:.3} m/s",
                            s * DX_M,
                            self.peak_speed[i] * DX_M
                        ));
                    }
                    ui.separator();
                    ui.label("Pull-back angle:");
                    ui.add(egui::Slider::new(&mut pull_deg, 0.0..=60.0).suffix(" deg"));
                    ui.label("Balls lifted together:");
                    ui.add(egui::Slider::new(&mut pull_count, 1..=N_GRAINS - 1));
                    ui.label("Time scale (x real time):");
                    ui.add(egui::Slider::new(&mut time_scale, 0.01..=1.0).logarithmic(true));
                    ui.separator();
                    ui.checkbox(&mut paused, "Paused (or SPACE)");
                    if ui.button("Reset").clicked() {
                        do_reset = true;
                    }
                    ui.separator();
                    ui.label("SPACE pause  R reset  LMB nudge  Q quit");
                });
        });
        self.paused = paused;
        self.time_scale = time_scale;
        self.pull_deg = pull_deg;
        self.pull_count = pull_count;
        if (pull_deg - pull_before).abs() > 1.0e-6 || pull_count != pull_count_before || do_reset {
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
                    .with_title("emerge -- Newton's cradle")
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

extern crate emerge_engine as emerge;

#[path = "../gui_common/cursor_force.rs"]
mod cursor_force;
#[path = "../gui_common/mod.rs"]
mod gui_common;

/// Why hot water tears under a gentler pull.
///
/// A liquid can be stretched. Pull on it and its pressure drops below the
/// air pressure around it, and it holds together on nothing but its own
/// cohesion. It holds until its absolute pressure reaches the saturation
/// pressure of its own temperature, and there it stops resisting and tears
/// into vapour instead. That is cavitation, and the temperature decides
/// how far it can be taken: cold water can be pulled to a near vacuum,
/// water close to boiling gives up almost immediately.
///
/// Three columns of the same water, the same density, the same stiffness.
/// The ONLY difference is the temperature:
///
///   LEFT    20 C   can hold almost a full atmosphere of tension
///   MIDDLE  60 C   gives up at four fifths of that
///   RIGHT   90 C   gives up at under a third
///
/// Those three floors are not written anywhere in this file. There is no
/// pressure constant in this scene at all: the engine reads each one off
/// the IAPWS-IF97 saturation curve at the particle's own temperature, and
/// they land on the published steam table to the pascal:
///
/// ```text
///     water    p_sat published    minus one atmosphere    the engine
///     20 C        2 339 Pa            -98 986 Pa          -98 986 Pa
///     60 C       19 946 Pa            -81 379 Pa          -81 379 Pa
///     90 C       70 182 Pa            -31 143 Pa          -31 143 Pa
/// ```
///
/// Pull with the right mouse button, or hold P to grip both sides of every
/// column and pull them apart at a fixed speed, the way a tensile test
/// grips a specimen. Each column falls until it reaches its own floor and
/// then stops: measured headless (`cavitation_cost_probe`) at a grip of
/// 1.8 m/s, the three reach -98 779, -81 075 and -31 284 Pa against floors
/// of -98 986, -81 379 and -31 143. The hot one is the only one that goes
/// marginally past its floor, by 0.45 %.
///
/// The pressure view (V) paints each particle by how far down its own
/// floor it has come, so a column that is cavitating is a column that has
/// gone bright. Under one pull the hot column lights up long before the
/// cold one moves.
///
/// # Where the numbers come from
///
/// The equation of state is the three-branch cavitating liquid of Lyu,
/// Sun, Colagrossi and Zhang (2023), whose vapour branch is anchored on
/// the IAPWS-IF97 saturation curve. The liquid's Tait exponent 7.0 is
/// Cole's (1948) for water, the vapour's 1.33 is steam's adiabatic index.
///
/// # What is real here, and what is declared
///
///   1. Gravity starts at zero, and not to make the scene behave. Free
///      pools of liquid under gravity flatten and run into each other
///      within two seconds, and no wall in this engine is declared
///      compatible with a strict weakly-compressible liquid (issue #38),
///      so there is nothing honest to keep three of them apart. The
///      gravity slider turns it back on.
///   2. The liquid's sound speed is 60 m/s, not water's real 1480 m/s:
///      the artificial compressibility rule (Monaghan 1994) asks for ten
///      times the fastest speed the scene produces, and this is well past
///      that. It sets how fast tension travels, not where the liquid
///      gives up.
///   3. The vapour branch uses a 6:1 density ratio against the liquid, not
///      steam's real 1673:1, the same declared compression the rest of the
///      engine's two-phase work uses.
///   4. The coupling runs one way. The temperature sets where the liquid
///      tears; tearing does not pay latent heat back and cool it. The
///      two-way closure is a milestone of its own, not faked here.
///
/// # Two ways of pulling that do not work
///
/// Worth keeping out of the next scene's way, both measured. An outward
/// radial impulse has its strongest falloff at the CENTRE, so it blows a
/// column outward rather than stretching it: the tension never got past
/// -5 986 Pa, nowhere near any floor. Prescribing a uniform expansion on
/// every particle does not work either, because the liquid's own pressure
/// cancels the divergence as fast as it is imposed: the velocity gradient
/// settled at -0.057/s against the +0.1/s prescribed, and the volume never
/// moved. Only gripping the edges and leaving the inside free puts a
/// liquid in real tension.
///
/// Cost, measured headless in release (`cavitation_cost_probe`):
///
/// ```text
///   at rest           8 substeps/frame    116 fps
///   under a 1.8 m/s pull    13 substeps     65 fps
/// ```
///
/// Real-time ratio, stated rather than left to be noticed: 0.065 s of
/// water per second of wall clock while being pulled, 15 times slower than
/// life, and 0.116 at rest.
///
///   LMB push  RMB pull  P grip and pull  V pressure view  R reset  Q quit
///   cargo run --release --example basic_cavitation --features render
use emerge::render::{ColorMode, Renderer};
use emerge::{
    CavitatingEosTable, CavitatingFluidMaterial, SimConfig, Simulation, SlipBoundary, SpawnRegion,
};
use glam::{IVec2, Vec2};
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

const GRID: usize = 64;
/// 2 cm cells: a 1.28 m tank holding three columns 20 cm across.
const DX_M: f32 = 0.02;
const COLUMN_CELLS: IVec2 = IVec2::new(10, 10);
const COLUMN_X: [f32; 3] = [14.0, 32.0, 50.0];
const COLUMN_Y: f32 = 32.0;
const COLUMN_LABEL: [&str; 3] = ["left", "mid", "right"];

/// Simulated time advanced per rendered frame, live-adjustable in the
/// panel. A viewing choice, not a physics one: every constant stays real
/// and each substep is identical whatever this is set to.
const DT_S_DEFAULT: f32 = 0.001;

/// The one thing that differs between the columns.
const TEMPERATURE_C: [f32; 3] = [20.0, 60.0, 90.0];
const KELVIN: f32 = 273.15;

const RHO_L_KG_M3: f32 = 1000.0;
/// See the declared approximations in this file's header.
const C_L_M_S: f32 = 60.0;
/// Cole 1948's Tait exponent for water.
const GAMMA_L: f32 = 7.0;
/// The declared 6:1 ratio, not steam's real 1673:1.
const RHO_V_KG_M3: f32 = RHO_L_KG_M3 / 6.0;
/// Steam's adiabatic index.
const GAMMA_V: f32 = 1.33;
/// The mixture band's effective acoustic speed, a model choice.
const C_MIN_M_S: f32 = 1.0;
const WATER_VISCOSITY_PA_S: f32 = 1.0e-3;
/// Full internal vaporization is `J = rho_l/rho_v = 6`; below 0.5 the
/// liquid branch is outside its own range.
const J_MIN: f32 = 0.5;
const J_MAX: f32 = 6.0;
/// What the P key pulls at, in metres per second on each side. The
/// headless probe's own gate speed.
const GRIP_M_S: f32 = 1.8;

fn make_config(gravity_fraction: f32, step_seconds: f32) -> SimConfig {
    let mut config = SimConfig {
        min_dt: 1.0e-7,
        max_substeps_per_step: 256,
        ..SimConfig::earth(GRID, DX_M, step_seconds)
    };
    config.gravity *= gravity_fraction;
    config
}

fn make_table() -> CavitatingEosTable {
    CavitatingEosTable::build(
        RHO_L_KG_M3,
        C_L_M_S,
        GAMMA_L,
        RHO_V_KG_M3,
        GAMMA_V,
        C_MIN_M_S,
        // The coldest liquid state this scene ever holds.
        TEMPERATURE_C[0] + KELVIN,
    )
}

/// Three columns of one water, differing only in temperature. Unlike the
/// boiling scene next door, every column rests at the liquid reference
/// density, so the spacing and the deformation gradient agree at spawn
/// without either being moved (see `cavitating_eos`'s own module doc for
/// what has to be set when they do not).
fn make_sim(
    gravity_fraction: f32,
    step_seconds: f32,
    temperature_shift: f32,
) -> (Simulation, CavitatingEosTable, [usize; 3]) {
    let config = make_config(gravity_fraction, step_seconds);
    let material =
        CavitatingFluidMaterial::new(make_table(), DX_M, WATER_VISCOSITY_PA_S, J_MIN, J_MAX);
    let particle_mass = RHO_L_KG_M3 * (0.5 * DX_M).powi(2);
    let spawn = |slot: usize| SpawnRegion {
        spacing: 0.5,
        box_size: COLUMN_CELLS,
        box_center: Vec2::new(COLUMN_X[slot], COLUMN_Y),
        material_id: 0,
        mass_override: Some(particle_mass),
        initial_velocity_scale: 0.0,
        ..SpawnRegion::for_sim(&config)
    };
    let mut sim = Simulation::new(config, spawn(0))
        .with_default_material(Box::new(material))
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));
    let first = sim.particles().len();
    let _ = sim.add_body(spawn(1));
    let second = sim.particles().len();
    let _ = sim.add_body(spawn(2));
    let bounds = [first, second, sim.particles().len()];
    {
        let particles = sim.particles_mut();
        for i in 0..particles.len() {
            particles.temperature[i] =
                (TEMPERATURE_C[slot_of(&bounds, i)] + temperature_shift + KELVIN).max(KELVIN + 1.0);
        }
    }
    (sim, make_table(), bounds)
}

fn slot_of(bounds: &[usize; 3], i: usize) -> usize {
    if i < bounds[0] {
        0
    } else if i < bounds[1] {
        1
    } else {
        2
    }
}

/// The engine carries density per unit cell area, so one cell of area
/// `dx^2` holds `density * dx^2` of real mass.
fn si_density(grid_density: f32) -> f32 {
    grid_density / (DX_M * DX_M)
}

/// How far down its own saturation floor a particle has come: 0 at rest,
/// 1 when it is cavitating. Negative under compression.
fn toward_floor(table: &CavitatingEosTable, temperature_k: f32, grid_density: f32) -> f32 {
    let eos = table.reconstruct(temperature_k);
    let floor = eos.p_v_gauge_pa;
    if floor >= 0.0 {
        return 0.0;
    }
    (eos.pressure_gauge_pa(si_density(grid_density)) / floor).clamp(-1.0, 1.0)
}

struct State {
    gfx: gui_common::Gfx,
    sim: Simulation,
    table: CavitatingEosTable,
    bounds: [usize; 3],
    renderer: Renderer,
    cursor_pos: [f32; 2],
    lmb: bool,
    rmb: bool,
    /// P: grip both sides of every column and pull, the same tensile test
    /// the headless probe runs.
    gripping: bool,
    cursor_force: cursor_force::CursorForce,
    gravity_fraction: f32,
    /// Moves all three temperatures together, so the ordering can be swept
    /// instead of taken on trust.
    temperature_shift: f32,
    /// The lowest gauge pressure each column has reached since the last
    /// reset, in pascals.
    lowest_pa: [f32; 3],
    step_seconds: f32,
    elapsed_s: f32,
    frame: u64,
    fps_timer: std::time::Instant,
    fps_frames: u64,
    last_fps: f32,
    /// Paints each particle by how close it is to its own floor.
    show_pressure: bool,
}

impl State {
    async fn new(window: Arc<Window>) -> Self {
        let gfx = gui_common::Gfx::new(&window).await;
        let size = window.inner_size();
        let gravity_fraction = 0.0;
        let (sim, table, bounds) = make_sim(gravity_fraction, DT_S_DEFAULT, 0.0);

        let mut renderer = Renderer::new(&gfx.device, sim.particles().len(), gfx.format);
        renderer.set_camera(&gfx.queue, GRID as u32, size.width, size.height, 0.6, true);
        renderer.set_color_mode(ColorMode::ByPhysics);

        println!(
            "basic_cavitation: {} particles, 3 columns of one water, only the temperature differs",
            sim.particles().len()
        );
        for celsius in TEMPERATURE_C {
            let eos = table.reconstruct(celsius + KELVIN);
            println!(
                "  {celsius:.0} C: tears at {:.0} Pa gauge, read off the IAPWS-IF97 saturation curve",
                eos.p_v_gauge_pa
            );
        }
        println!("  LMB push  RMB pull  P grip and pull  V pressure view  R reset  Q quit");

        Self {
            gfx,
            sim,
            table,
            bounds,
            renderer,
            cursor_pos: [0.0; 2],
            lmb: false,
            rmb: false,
            gripping: false,
            cursor_force: cursor_force::CursorForce::new(5.0, 5.0, 5.0),
            gravity_fraction,
            temperature_shift: 0.0,
            lowest_pa: [0.0; 3],
            step_seconds: DT_S_DEFAULT,
            elapsed_s: 0.0,
            frame: 0,
            fps_timer: std::time::Instant::now(),
            fps_frames: 0,
            last_fps: 0.0,
            show_pressure: false,
        }
    }

    fn resize(&mut self, w: u32, h: u32) {
        self.gfx.resize(w, h);
        if w == 0 || h == 0 {
            return;
        }
        self.renderer
            .set_camera(&self.gfx.queue, GRID as u32, w, h, 0.6, true);
    }

    fn cursor_grid(&self) -> Vec2 {
        gui_common::cursor_to_grid(
            self.cursor_pos,
            self.gfx.surface_config.width,
            self.gfx.surface_config.height,
            GRID,
        )
    }

    fn reset(&mut self) {
        let (sim, table, bounds) = make_sim(
            self.gravity_fraction,
            self.step_seconds,
            self.temperature_shift,
        );
        self.sim = sim;
        self.table = table;
        self.bounds = bounds;
        self.lowest_pa = [0.0; 3];
        self.frame = 0;
        self.elapsed_s = 0.0;
    }

    fn update_and_render(&mut self, window: &Window) {
        self.sim
            .set_gravity(make_config(self.gravity_fraction, self.step_seconds).gravity);
        self.sim.set_step_duration(self.step_seconds);

        if self.gripping {
            // One cell at each side of every column, pulled apart at a fixed
            // speed with everything between them left free to answer. An
            // outward impulse or a prescribed uniform expansion both fail to
            // put a liquid in tension; see this file's header.
            let grip_cells = GRIP_M_S / DX_M;
            let half = COLUMN_CELLS.x as f32 * 0.5;
            let bounds = self.bounds;
            let particles = self.sim.particles_mut();
            for i in 0..particles.len() {
                let offset = particles.x[i].x - COLUMN_X[slot_of(&bounds, i)];
                if offset.abs() > half - 1.0 {
                    particles.v[i].x = grip_cells * offset.signum();
                }
            }
        }

        let g = self.sim.config().gravity.length();
        let cursor = self.cursor_grid();
        if self.lmb {
            self.cursor_force.apply(
                self.sim.particles_mut(),
                cursor,
                g.max(9.81),
                self.step_seconds,
                false,
            );
        }
        if self.rmb {
            self.cursor_force.apply(
                self.sim.particles_mut(),
                cursor,
                g.max(9.81),
                self.step_seconds,
                true,
            );
        }

        self.sim.step();

        // The lowest pressure each column has been taken to, which is the
        // number the saturation floor is a claim about.
        for (i, p) in self.sim.particles().iter().enumerate() {
            let eos = self.table.reconstruct(p.temperature);
            let gauge = eos.pressure_gauge_pa(si_density(p.density));
            let slot = slot_of(&self.bounds, i);
            self.lowest_pa[slot] = self.lowest_pa[slot].min(gauge);
        }

        self.frame += 1;
        self.elapsed_s += self.step_seconds;
        self.fps_frames += 1;
        if self.fps_timer.elapsed().as_secs_f32() >= 1.0 {
            self.last_fps = self.fps_frames as f32 / self.fps_timer.elapsed().as_secs_f32();
            self.fps_timer = std::time::Instant::now();
            self.fps_frames = 0;
        }

        if self.show_pressure {
            let table = &self.table;
            let parts = self.sim.particles();
            let toward: Vec<f32> = (0..parts.len())
                .map(|i| toward_floor(table, parts.temperature[i], parts.density[i]).max(0.0))
                .collect();
            self.renderer.set_stress_field(toward);
            // Full scale is a particle sitting exactly on its own floor, so
            // the colour means a fraction of the way to cavitation.
            self.renderer.set_stress_scale(1.0);
        }

        let output = match self.gfx.surface.get_current_texture() {
            Ok(t) => t,
            Err(_) => return,
        };
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        self.renderer.render(
            &self.gfx.device,
            &self.gfx.queue,
            self.sim.particles(),
            &view,
            true,
        );

        let fps = self.last_fps;
        let mut gravity_fraction = self.gravity_fraction;
        let mut temperature_shift = self.temperature_shift;
        let mut step_seconds = self.step_seconds;
        let mut push_strength = self.cursor_force.push_strength;
        let mut pull_strength = self.cursor_force.pull_strength;
        let n_particles = self.sim.particles().len();
        let lowest = self.lowest_pa;
        let floors: Vec<f32> = (0..3)
            .map(|slot| {
                self.table
                    .reconstruct(TEMPERATURE_C[slot] + temperature_shift + KELVIN)
                    .p_v_gauge_pa
            })
            .collect();
        let gripping = self.gripping;
        let mut reset = false;

        gui_common::run_egui_frame(&mut self.gfx, window, &view, |ctx| {
            egui::Window::new("Cavitation -- how far water can be pulled")
                .default_pos([10.0, 10.0])
                .default_width(360.0)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.label(format!("fps={fps:.0}  particles={n_particles}"));
                    ui.label(format!(
                        "{:.2} ms of physics per frame",
                        step_seconds * 1000.0
                    ));
                    ui.add(
                        egui::Slider::new(&mut step_seconds, 0.0005..=0.004)
                            .logarithmic(true)
                            .text("s / frame"),
                    );
                    ui.separator();
                    ui.label("Temperature -- the only difference:");
                    for slot in 0..3 {
                        let t = TEMPERATURE_C[slot] + temperature_shift;
                        let floor = floors[slot];
                        let reached = lowest[slot];
                        let share = if floor < 0.0 {
                            100.0 * (reached / floor).clamp(0.0, 2.0)
                        } else {
                            0.0
                        };
                        ui.label(format!(
                            "{:>5}: {t:5.1} C  tears at {floor:8.0} Pa, taken to {reached:8.0} Pa ({share:3.0} % of the way)",
                            COLUMN_LABEL[slot]
                        ));
                    }
                    ui.add(
                        egui::Slider::new(&mut temperature_shift, -15.0..=8.0)
                            .text("C warmer or colder"),
                    );
                    ui.label("Those floors are the IAPWS-IF97 saturation curve,");
                    ui.label("not a constant written into this scene.");
                    ui.separator();
                    if gripping {
                        ui.label(format!("PULLING at {GRIP_M_S} m/s a side (hold P)"));
                    } else {
                        ui.label("Hold P to grip both sides and pull them apart.");
                    }
                    ui.label("V = pressure view: how far down its own floor.");
                    ui.separator();
                    ui.label(format!(
                        "liquid {RHO_L_KG_M3:.0} kg/m3   c_l {C_L_M_S:.0} m/s   vapour {RHO_V_KG_M3:.1} kg/m3"
                    ));
                    ui.label("Gravity (1.0 = real IRL 9.81 m/s2):");
                    ui.add(egui::Slider::new(&mut gravity_fraction, 0.0..=1.0));
                    ui.label("Push strength:");
                    ui.add(egui::Slider::new(&mut push_strength, 0.0..=15.0));
                    ui.label("Pull strength:");
                    ui.add(egui::Slider::new(&mut pull_strength, 0.0..=15.0));
                    ui.separator();
                    ui.label("LMB push  RMB pull  P grip  V pressure  R reset  Q quit");
                    if ui.button("Reset").clicked() {
                        reset = true;
                    }
                });
        });
        let temperature_changed = (temperature_shift - self.temperature_shift).abs() > f32::EPSILON;
        self.gravity_fraction = gravity_fraction;
        self.temperature_shift = temperature_shift;
        self.step_seconds = step_seconds;
        self.cursor_force.push_strength = push_strength;
        self.cursor_force.pull_strength = pull_strength;
        if temperature_changed {
            // The temperature belongs to the particles, so a new setting has
            // to reach them rather than only the readout.
            let shift = self.temperature_shift;
            let bounds = self.bounds;
            let particles = self.sim.particles_mut();
            for i in 0..particles.len() {
                particles.temperature[i] =
                    (TEMPERATURE_C[slot_of(&bounds, i)] + shift + KELVIN).max(KELVIN + 1.0);
            }
        }
        if reset {
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
                    .with_title("emerge -- cavitation: how far water can be pulled (GUI)")
                    .with_inner_size(winit::dpi::LogicalSize::new(640u32, 640u32)),
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
                    KeyCode::KeyP => s.gripping = pressed,
                    KeyCode::KeyR if pressed => {
                        s.reset();
                        println!("reset");
                    }
                    KeyCode::KeyV if pressed => {
                        s.show_pressure = !s.show_pressure;
                        s.renderer.set_color_mode(if s.show_pressure {
                            ColorMode::ByStress
                        } else {
                            ColorMode::ByPhysics
                        });
                        let on = if s.show_pressure { "ON" } else { "off" };
                        println!("pressure view {on}");
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

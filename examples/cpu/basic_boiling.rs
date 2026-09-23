extern crate emerge_engine as emerge;

#[path = "../gui_common/cursor_force.rs"]
mod cursor_force;
#[path = "../gui_common/mod.rs"]
mod gui_common;

/// What boiling does to room.
///
/// Water that boils does not vanish: part of its mass turns to vapour, and
/// the same mass suddenly needs far more room. Engineers call the boiled
/// share the quality, x, and the standard two-phase rule puts the mixture
/// at the density you get by adding the two specific volumes, each weighted
/// by its share of the mass:
///
/// ```text
///   rho_eq(x) = 1 / ((1 - x) / rho_liquid + x / rho_vapour)
/// ```
///
/// which means the same mass spreads over `J_eq(x) = 1 + (rho_l/rho_v - 1) x`
/// times its liquid volume. That one line is the whole scene.
///
/// Three columns of the same water, the same mass in every particle, each
/// laid down at the size its own quality asks for. The ONLY difference is
/// how much of it has boiled:
///
///   LEFT    x = 0.00   plain liquid, nothing boiled
///   MIDDLE  x = 0.09   about a tenth of the mass turned to vapour
///   RIGHT   x = 0.19   about a fifth
///
/// Each one simply sits there, which is the claim being tested: the
/// mixture rule is an equilibrium, so a column laid down on it should
/// not move at all. Measured headless (`boiling_cost_probe`) after two
/// seconds, with the fastest particle anywhere reading 0.00 m/s:
///
/// ```text
///     quality    rests at        the rule says     width
///      0.00      1000.0 kg/m3      1000.0          9.5 cells
///      0.09       694.4 kg/m3       694.4         12.0 cells
///      0.19       510.2 kg/m3       510.2         13.3 cells
/// ```
///
/// Those widths are the law read off the screen: 12.0/9.5 = 1.26 and
/// 13.3/9.5 = 1.40, against the `sqrt(J_eq)` of 1.2 and 1.4 the rule
/// asks for in each direction.
///
/// Then use the boil slider. It raises all three qualities together and
/// they swell: nothing is added and nothing is moved by hand, the mixture
/// rule simply asks for more room and the pressure law delivers it.
///
/// # Where the numbers come from
///
/// The mixture density is the mass-weighted specific-volume rule (Collier
/// and Thome, "Convective Boiling and Condensation", 3rd ed., section 2.2).
/// The stiffness around it is Wood's mixture sound speed (Wallis,
/// "One-Dimensional Two-Phase Flow", 1969, eq. 4.36). Both fall back
/// exactly onto the pure liquid at x = 0 by construction, not by tuning.
/// The liquid's Tait exponent 7.0 is Cole's (1948) for water; the vapour's
/// 1.33 is steam's adiabatic index.
///
/// # What is real here, and what is declared
///
/// Four things in this scene are deliberate approximations rather than
/// measurements, and the scene is worth nothing without them stated:
///
///   0. Gravity starts at zero, and not to make the scene behave.
///      Three free pools of water under real gravity flatten and run
///      into each other within two seconds (measured: 38 to 47 cells
///      across in a 64-cell tank), and a strict weakly-compressible
///      liquid has no wall in this engine it is declared compatible
///      with, so there is nothing honest to keep them apart. Weightless
///      is what isolates the one law this scene is about. The gravity
///      slider turns it back on, and they do flood.
///   1. Steam at 100 C and one atmosphere is 0.598 kg/m3, a 1673:1 ratio
///      against water. This scene runs 6:1, so a fully boiled particle
///      expands six times rather than sixteen hundred. The rule is exact;
///      the endpoint it is evaluated at is compressed to keep a tank this
///      size on screen.
///   2. The liquid's sound speed is 60 m/s, not water's real 1480 m/s. The
///      artificial compressibility rule (Monaghan 1994) asks for ten times
///      the fastest speed the scene itself produces, about 0.5 m/s here,
///      so this is thirty times that margin and the density error it
///      leaves is 0.01 %.
///   3. The coupling runs one way. The quality drives the mechanical
///      state; a column squeezed by the cursor does not pay that work back
///      into latent heat and boil further. The honest two-way closure
///      needs a real saturation curve and is a milestone of its own, not
///      something faked here.
///
/// # What the panel shows
///
/// Each column's quality, the density the rule predicts for it, the
/// density it actually rests at, its mixture sound speed and its width. The vapour
/// view (V) paints each particle by how much of it has boiled, so the
/// slider's effect is visible as colour before it is visible as size.
///
/// Pressing shows the other half: the more a column has boiled, the softer
/// it is, because a mixture carries the compressibility of both phases at
/// once.
///
/// Cost, measured headless in release (`boiling_cost_probe`):
///
/// ```text
///   2.0 ms/frame    34 substeps    44 fps
///   1.0 ms/frame    17 substeps    87 fps
///   0.5 ms/frame     9 substeps   157 fps
/// ```
///
/// Real-time ratio, stated rather than left to be noticed: 0.087 s of
/// water per second of wall clock, 12 times slower than life, and the
/// slider does not change that. It trades frame rate against smoothness,
/// not against realism: the substeps this water's own sound speed needs
/// per simulated second are the same either way.
///
///   LMB push  RMB pull  V vapour view  R reset  Q quit
///   cargo run --release --example basic_boiling --features render
use emerge::render::{ColorMode, Renderer};
use emerge::{
    BoilingMixtureMaterial, CavitatingEosTable, SimConfig, Simulation, SlipBoundary, SpawnRegion,
};
use glam::{IVec2, Mat2, Vec2};
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

const GRID: usize = 64;
/// 2 cm cells: a 1.28 m tank holding three columns about 20 cm across.
const DX_M: f32 = 0.02;
const FLOOR_CELLS: f32 = 3.0;
const COLUMN_CELLS: IVec2 = IVec2::new(10, 10);
const COLUMN_X: [f32; 3] = [14.0, 32.0, 50.0];
const COLUMN_LABEL: [&str; 3] = ["left", "mid", "right"];

/// Simulated time advanced per rendered frame, live-adjustable in the
/// panel. A viewing choice, not a physics one: every constant stays real
/// and each substep is identical whatever this is set to. See this file's
/// header for the measured table behind it.
const DT_S_DEFAULT: f32 = 0.001;

/// The one thing that differs between the columns. Not round numbers by
/// accident: they put `sqrt(J_eq)` at exactly 1.0, 1.2 and 1.4, so the
/// three lattices differ by a clean factor and every particle carries the
/// same mass.
const QUALITY: [f32; 3] = [0.0, 0.088, 0.192];

const RHO_L_KG_M3: f32 = 1000.0;
/// See the declared approximations in this file's header.
const C_L_M_S: f32 = 60.0;
/// Cole 1948's Tait exponent for water.
const GAMMA_L: f32 = 7.0;
/// The compressed 6:1 ratio, declared in this file's header.
const RHO_V_KG_M3: f32 = RHO_L_KG_M3 / 6.0;
/// Steam's adiabatic index.
const GAMMA_V: f32 = 1.33;
/// The mixture band's effective acoustic speed, a model choice.
const C_MIN_M_S: f32 = 1.0;
const MELTING_POINT_K: f32 = 273.15;
const BOILING_POINT_K: f32 = 373.15;
const WATER_VISCOSITY_PA_S: f32 = 1.0e-3;
/// Full internal vaporization is `J = rho_l/rho_v = 6`; below 0.5 the
/// liquid branch is outside its own range.
const J_MIN: f32 = 0.5;
const J_MAX: f32 = 6.0;

fn make_config(gravity_fraction: f32, step_seconds: f32) -> SimConfig {
    let mut config = SimConfig {
        min_dt: 1.0e-7,
        // 17 substeps at the default frame time; the rest is headroom for a
        // hard press, not a physics cap.
        max_substeps_per_step: 256,
        ..SimConfig::earth(GRID, DX_M, step_seconds)
    };
    config.gravity *= gravity_fraction;
    config
}

fn make_material() -> BoilingMixtureMaterial {
    let table = CavitatingEosTable::build(
        RHO_L_KG_M3,
        C_L_M_S,
        GAMMA_L,
        RHO_V_KG_M3,
        GAMMA_V,
        C_MIN_M_S,
        MELTING_POINT_K,
    );
    BoilingMixtureMaterial::from_table(&table, DX_M, WATER_VISCOSITY_PA_S, J_MIN, J_MAX)
}

/// Three columns, each laid down at the size its own quality asks for.
///
/// Both halves of that matter. The lattice spacing puts the GRID at the
/// mixture's density, and the initial deformation gradient puts each
/// particle's own bookkeeping there too. With only one of the two, a column
/// starts carrying the liquid's density at the mixture's spacing, which is
/// a pressure shock rather than a scene: measured, the first frame then
/// throws particles at 151 m/s instead of 0.01 m/s.
fn make_sim(
    gravity_fraction: f32,
    step_seconds: f32,
) -> (Simulation, BoilingMixtureMaterial, [usize; 3]) {
    let config = make_config(gravity_fraction, step_seconds);
    let material = make_material();
    // Every particle carries the liquid's own mass whatever its column: the
    // quality changes how far apart they sit, not how heavy they are.
    let liquid_spacing = 0.5_f32;
    let particle_mass = RHO_L_KG_M3 * (liquid_spacing * DX_M).powi(2);
    let spawn = |slot: usize| {
        let stretch = material.j_eq(QUALITY[slot]).sqrt();
        SpawnRegion {
            spacing: liquid_spacing * stretch,
            box_size: IVec2::new(
                (COLUMN_CELLS.x as f32 * stretch).round() as i32,
                (COLUMN_CELLS.y as f32 * stretch).round() as i32,
            ),
            box_center: Vec2::new(
                COLUMN_X[slot],
                FLOOR_CELLS + COLUMN_CELLS.y as f32 * stretch * 0.5,
            ),
            material_id: 0,
            mass_override: Some(particle_mass),
            precompute_initial_volumes: false,
            initial_velocity_scale: 0.0,
            initial_deformation_gradient: Mat2::from_diagonal(Vec2::splat(stretch)),
            ..SpawnRegion::for_sim(&config)
        }
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
            particles.temperature[i] = BOILING_POINT_K;
        }
    }
    (sim, material, bounds)
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

/// Measured density and width for one column. Width, not height: the
/// rule predicts a volume, and with the columns weightless it shows up
/// as `sqrt(J_eq)` in each direction.
fn column_state(sim: &Simulation, bounds: &[usize; 3], slot: usize) -> (f32, f32) {
    let (mut rho, mut n, mut left, mut right) = (0.0f32, 0u32, f32::MAX, f32::MIN);
    for (i, p) in sim.particles().iter().enumerate() {
        if slot_of(bounds, i) == slot {
            rho += si_density(p.density);
            n += 1;
            left = left.min(p.x.x);
            right = right.max(p.x.x);
        }
    }
    (rho / n.max(1) as f32, right - left)
}

fn print_state(
    sim: &Simulation,
    material: &BoilingMixtureMaterial,
    bounds: &[usize; 3],
    boil: f32,
    elapsed_s: f32,
) {
    let mut line = format!("BOIL t={elapsed_s:.2}s");
    for slot in 0..3 {
        let x = (QUALITY[slot] + boil).clamp(0.0, 1.0);
        let (rho, width) = column_state(sim, bounds, slot);
        line += &format!(
            "  {}[x={x:.2}, {rho:.0} kg/m3 vs rule {:.0}, {width:.1} cells across]",
            COLUMN_LABEL[slot],
            material.rho_eq_kg_m3(x),
        );
    }
    println!("{line}");
}

struct State {
    gfx: gui_common::Gfx,
    sim: Simulation,
    material: BoilingMixtureMaterial,
    bounds: [usize; 3],
    renderer: Renderer,
    cursor_pos: [f32; 2],
    lmb: bool,
    rmb: bool,
    cursor_force: cursor_force::CursorForce,
    gravity_fraction: f32,
    /// Added to every column's own quality, so the ordering can be swept
    /// instead of taken on trust.
    boil: f32,
    step_seconds: f32,
    elapsed_s: f32,
    frame: u64,
    fps_timer: std::time::Instant,
    fps_frames: u64,
    last_fps: f32,
    /// Paints each particle by how much of it has boiled.
    show_vapour: bool,
}

impl State {
    async fn new(window: Arc<Window>) -> Self {
        let gfx = gui_common::Gfx::new(&window).await;
        let size = window.inner_size();
        let gravity_fraction = 0.0;
        let (sim, material, bounds) = make_sim(gravity_fraction, DT_S_DEFAULT);

        let mut renderer = Renderer::new(&gfx.device, sim.particles().len(), gfx.format);
        renderer.set_camera(&gfx.queue, GRID as u32, size.width, size.height, 0.6, true);
        renderer.set_color_mode(ColorMode::ByPhysics);

        println!(
            "basic_boiling: {} particles, 3 columns of one water, same mass per particle",
            sim.particles().len()
        );
        println!(
            "  rho_l={RHO_L_KG_M3} kg/m3, rho_v={RHO_V_KG_M3:.1} kg/m3 (a declared 6:1, not steam's real 1673:1), c_l={C_L_M_S} m/s"
        );
        println!("  only the boiled fraction differs: {QUALITY:?}");
        println!("  LMB push  RMB pull  V vapour view  R reset  Q quit");

        Self {
            gfx,
            sim,
            material,
            bounds,
            renderer,
            cursor_pos: [0.0; 2],
            lmb: false,
            rmb: false,
            cursor_force: cursor_force::CursorForce::new(5.0, 5.0, 5.0),
            gravity_fraction,
            boil: 0.0,
            step_seconds: DT_S_DEFAULT,
            elapsed_s: 0.0,
            frame: 0,
            fps_timer: std::time::Instant::now(),
            fps_frames: 0,
            last_fps: 0.0,
            show_vapour: false,
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
        let (sim, material, bounds) = make_sim(self.gravity_fraction, self.step_seconds);
        self.sim = sim;
        self.material = material;
        self.bounds = bounds;
        self.boil = 0.0;
        self.frame = 0;
        self.elapsed_s = 0.0;
    }

    fn update_and_render(&mut self, window: &Window) {
        self.sim
            .set_gravity(make_config(self.gravity_fraction, self.step_seconds).gravity);
        self.sim.set_step_duration(self.step_seconds);

        // The quality every particle carries. In a full phase chain the
        // enthalpy method writes this from the heat actually absorbed; here
        // the slider stands in for that heat, so the mixture law can be
        // watched on its own.
        let boil = self.boil;
        let bounds = self.bounds;
        {
            let particles = self.sim.particles_mut();
            for i in 0..particles.len() {
                particles.friction_hardening[i] =
                    (QUALITY[slot_of(&bounds, i)] + boil).clamp(0.0, 1.0);
            }
        }

        let g = self.sim.config().gravity.length();
        let cursor = self.cursor_grid();
        if self.lmb {
            self.cursor_force.apply(
                self.sim.particles_mut(),
                cursor,
                g,
                self.step_seconds,
                false,
            );
        }
        if self.rmb {
            self.cursor_force
                .apply(self.sim.particles_mut(), cursor, g, self.step_seconds, true);
        }

        self.sim.step();

        if self.frame.is_multiple_of(500) {
            print_state(
                &self.sim,
                &self.material,
                &self.bounds,
                self.boil,
                self.elapsed_s,
            );
        }

        self.frame += 1;
        self.elapsed_s += self.step_seconds;
        self.fps_frames += 1;
        if self.fps_timer.elapsed().as_secs_f32() >= 1.0 {
            self.last_fps = self.fps_frames as f32 / self.fps_timer.elapsed().as_secs_f32();
            self.fps_timer = std::time::Instant::now();
            self.fps_frames = 0;
        }

        if self.show_vapour {
            let parts = self.sim.particles();
            let vapour: Vec<f32> = (0..parts.len())
                .map(|i| parts.friction_hardening[i])
                .collect();
            self.renderer.set_stress_field(vapour);
            // Full scale is a fully boiled particle, so the colour means a
            // fraction of the mass, not an arbitrary range.
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
        let mut boil = self.boil;
        let mut step_seconds = self.step_seconds;
        let mut push_strength = self.cursor_force.push_strength;
        let mut pull_strength = self.cursor_force.pull_strength;
        let n_particles = self.sim.particles().len();
        let columns: Vec<(f32, f32)> = (0..3)
            .map(|slot| column_state(&self.sim, &self.bounds, slot))
            .collect();
        let material = self.material;
        let mut reset = false;

        gui_common::run_egui_frame(&mut self.gfx, window, &view, |ctx| {
            egui::Window::new("Boiling -- what vapour does to room")
                .default_pos([10.0, 10.0])
                .default_width(340.0)
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
                    ui.label("Boiled fraction -- the only difference:");
                    for slot in 0..3 {
                        let x = (QUALITY[slot] + boil).clamp(0.0, 1.0);
                        let (rho, width) = columns[slot];
                        ui.label(format!(
                            "{:>5}: x={x:.2}  rests at {rho:6.1} kg/m3, rule says {:6.1}, c_mix {:5.1} m/s, {width:.1} across",
                            COLUMN_LABEL[slot],
                            material.rho_eq_kg_m3(x),
                            material.c_mix2_m2_s2(x).sqrt(),
                        ));
                    }
                    ui.add(egui::Slider::new(&mut boil, 0.0..=0.30).text("boil further"));
                    ui.label("Raise it and they swell: nothing is added, the");
                    ui.label("mixture rule simply asks for more room.");
                    ui.separator();
                    ui.label(format!(
                        "liquid {RHO_L_KG_M3:.0} kg/m3   vapour {RHO_V_KG_M3:.1} kg/m3   c_l {C_L_M_S:.0} m/s"
                    ));
                    ui.label("A declared 6:1 density ratio, not steam's real 1673:1.");
                    ui.separator();
                    ui.label("Gravity (1.0 = real IRL 9.81 m/s2):");
                    ui.add(egui::Slider::new(&mut gravity_fraction, 0.0..=1.0));
                    ui.label("Push strength:");
                    ui.add(egui::Slider::new(&mut push_strength, 0.0..=15.0));
                    ui.label("Pull strength:");
                    ui.add(egui::Slider::new(&mut pull_strength, 0.0..=15.0));
                    ui.separator();
                    ui.label("The more a column has boiled, the softer it is.");
                    ui.label("V = vapour view: how much of each particle boiled.");
                    ui.label("LMB push  RMB pull  V vapour  R reset  Q quit");
                    if ui.button("Reset").clicked() {
                        reset = true;
                    }
                });
        });
        self.gravity_fraction = gravity_fraction;
        self.boil = boil;
        self.step_seconds = step_seconds;
        self.cursor_force.push_strength = push_strength;
        self.cursor_force.pull_strength = pull_strength;
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
                    .with_title("emerge -- boiling: what vapour does to room (GUI)")
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
                    KeyCode::KeyR if pressed => {
                        s.reset();
                        println!("reset");
                    }
                    KeyCode::KeyV if pressed => {
                        s.show_vapour = !s.show_vapour;
                        s.renderer.set_color_mode(if s.show_vapour {
                            ColorMode::ByStress
                        } else {
                            ColorMode::ByPhysics
                        });
                        let on = if s.show_vapour { "ON" } else { "off" };
                        println!("vapour view {on}");
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

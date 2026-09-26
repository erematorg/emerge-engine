extern crate emerge_engine as emerge;

#[path = "../gui_common/cursor_force.rs"]
mod cursor_force;
#[path = "../gui_common/mod.rs"]
mod gui_common;

/// Why coarse gravel and fine sand collapse differently even at the same
/// friction angle.
///
/// Dry granular flow does not have one fixed friction coefficient. It
/// depends on how fast it is shearing, through the inertial number I
/// (grain diameter times shear rate, scaled by the confining pressure).
/// Slow, quasi-static flow feels the STATIC friction angle; fast, inertial
/// flow feels a HIGHER, DYNAMIC one -- the material genuinely resists more
/// per unit stress the harder you push it, the opposite of a viscous
/// fluid thinning out. This is the mu(I) rheology (Jop, Forterre & Pouliquen,
/// Nature 441, 2006; the DPMui model this material implements is Cicoira
/// et al. 2022).
///
/// How fast that crossover happens is set by Q, essentially how coarse the
/// grains are: fine sand (small Q... no, LARGE Q) reaches the fast, dynamic
/// regime at a much lower shear rate than coarse gravel does. Three
/// identical columns, same friction angle (so mu_static and mu_dynamic are
/// literally the same two numbers for all three), same real Young's
/// modulus, same density, same drop. The ONLY difference is the grain-size
/// parameter Q:
///
///   LEFT    Q = 5.58  -- fine sand (d ~ 1 mm). Rate effects kick in early.
///   MIDDLE  Q = 3.00  -- intermediate.
///   RIGHT   Q = 1.12  -- coarse gravel (d ~ 5 mm). Needs a much faster
///                        shear rate before it stiffens.
///
/// Both Q values are Cicoira et al.'s own cited endpoints
/// (`MuIRheologyMaterial::small_grain`/`large_grain`), not picked for this
/// demo. Everything else comes through the engine's real SI dispatch
/// (`Elastoplastic` + `PlasticityModel::GranularRateDependent`), so the
/// friction angle and elastic modulus are real degrees and real pascals;
/// Q itself is not yet part of that dispatch (a real, disclosed engine gap
/// -- see `MuIRheologyMaterial::from_young_modulus`'s own doc), so it is
/// set directly on the constructed material afterward.
///
/// # What to watch
///
/// On the drop alone the three look close to identical: falling under
/// gravity is a slow, quasi-static process everywhere, well inside the
/// regime all three share the same mu_static in. The real difference shows
/// under a hard, fast push: shove the pile and watch how much sooner the
/// left (fine) column stiffens up and resists further shear compared to
/// the right (coarse) one, which keeps flowing more easily at the same
/// push strength. That gap IS the inertial number crossing I0/Q -- not a
/// visual effect, a direct read of each material's own rate sensitivity.
///
///   LMB push  RMB pull  R reset  Q quit
///   cargo run --release --example basic_mui_rheology --features render
use emerge::render::{ColorMode, Renderer};
use emerge::{
    Elastic, FromSI, GranularProps, MuIRheologyMaterial, SimConfig, Simulation, SlipBoundary,
    SpawnRegion,
};
use glam::{IVec2, Vec2};
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

const GRID: usize = 64;
/// Simulated time advanced per rendered frame, live-adjustable in the
/// panel. A viewing choice, not a physics one: the real elastic wave
/// speed at this material's real E=15 MPa, rho=1600 kg/m3 is ~97 m/s,
/// which at dx=0.01m demands a real CFL-safe substep near 0.07ms -- this
/// scene's continuous 3-column collapse stays in that stiff regime the
/// whole time, unlike an interactive pour that mostly idles.
///
/// Measured on this scene, headless, release (`mui_cost_probe`):
///
/// ```text
///   0.3 ms/frame    7.0 substeps    66 fps
///   0.5 ms/frame   11.0 substeps    47 fps
///   1.0 ms/frame   21.0 substeps    25 fps
/// ```
const DT_S_DEFAULT: f32 = 0.0003;
// Same real dry-sand elastic constants and density basic_sand.rs already
// uses and cites -- this scene varies Q, not the elastic response.
const YOUNG_MODULUS_PA: f32 = 15.0e6;
const POISSON_RATIO: f32 = 0.3;
const DENSITY_KG_M3: f32 = 1600.0;
// Same real friction angle for all three columns -- held fixed so Q is
// genuinely the only independent variable.
const FRICTION_ANGLE_DEG: f32 = 30.0;

// Cicoira et al.'s own two cited endpoints, plus their real midpoint.
const INERTIAL_Q: [f32; 3] = [5.58, 3.00, 1.12];
const COLUMN_LABEL: [&str; 3] = ["fine (Q=5.58)", "mid (Q=3.00)", "coarse (Q=1.12)"];
const COLUMN_X: [f32; 3] = [14.0, 32.0, 50.0];
const COLUMN_CELLS: IVec2 = IVec2::new(10, 22);
const MAT_SOFT: u32 = 0;
const MAT_MID: u32 = 1;
const MAT_HARD: u32 = 2;

fn make_sim(gravity_fraction: f32) -> Simulation {
    let mut config = SimConfig {
        min_dt: 1.0e-6,
        max_substeps_per_step: 128,
        material_cfl_coefficient: 0.7,
        ..SimConfig::earth(GRID, 0.01, DT_S_DEFAULT)
    };
    config.gravity *= gravity_fraction;

    let elastic = Elastic {
        e_pa: YOUNG_MODULUS_PA,
        nu: POISSON_RATIO,
        rho_kg_m3: DENSITY_KG_M3,
    };
    let props = GranularProps {
        elastic,
        friction_angle_deg: FRICTION_ANGLE_DEG,
        dilatancy_angle_deg: 0.0,
    };
    let build = |q: f32| {
        let mut m = MuIRheologyMaterial::from_physical(&props, &config);
        // Real, disclosed override: Q is not yet part of the SI dispatch
        // (see this file's own header doc), so it is set directly here --
        // same convention as setting `.optics` after `from_physical` on a
        // fluid material. `mu_static`/`mu_dynamic` came through the real
        // dispatch above and are untouched.
        m.inertial_q = q;
        m
    };

    let spawn = |slot: usize, mat: u32| {
        SpawnRegion {
            spacing: 0.5,
            box_size: COLUMN_CELLS,
            box_center: Vec2::new(COLUMN_X[slot], 4.0 + COLUMN_CELLS.y as f32 * 0.5),
            material_id: mat,
            initial_velocity_scale: 0.0,
            ..SpawnRegion::for_sim(&config)
        }
        .mass_from(&props, &config)
    };

    let mut sim = Simulation::new(config, spawn(0, MAT_SOFT))
        .with_default_material(Box::new(build(INERTIAL_Q[0])))
        .with_material(MAT_MID, Box::new(build(INERTIAL_Q[1])))
        .with_material(MAT_HARD, Box::new(build(INERTIAL_Q[2])))
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));
    let _ = sim.add_body(spawn(1, MAT_MID));
    let _ = sim.add_body(spawn(2, MAT_HARD));
    sim
}

struct State {
    gfx: gui_common::Gfx,
    sim: Simulation,
    renderer: Renderer,
    cursor_pos: [f32; 2],
    lmb: bool,
    rmb: bool,
    cursor_force: cursor_force::CursorForce,
    gravity_fraction: f32,
    step_seconds: f32,
    frame: u64,
    fps_timer: std::time::Instant,
    fps_frames: u64,
    last_fps: f32,
}

impl State {
    async fn new(window: Arc<Window>) -> Self {
        let gfx = gui_common::Gfx::new(&window).await;
        let size = window.inner_size();
        let gravity_fraction = 1.0;
        let sim = make_sim(gravity_fraction);

        let mut renderer = Renderer::new(&gfx.device, sim.particles().len(), gfx.format);
        renderer.set_camera(&gfx.queue, GRID as u32, size.width, size.height, 0.7, true);
        renderer.set_color_mode(ColorMode::ByMaterial);

        println!(
            "basic_mui_rheology: {} particles, 3 columns, same friction angle {FRICTION_ANGLE_DEG} deg, Q={INERTIAL_Q:?}",
            sim.particles().len()
        );
        println!("  LMB push  RMB pull  R reset  Q quit");

        Self {
            gfx,
            sim,
            renderer,
            cursor_pos: [0.0; 2],
            lmb: false,
            rmb: false,
            cursor_force: cursor_force::CursorForce::new(5.0, 8.0, 8.0),
            gravity_fraction,
            step_seconds: DT_S_DEFAULT,
            frame: 0,
            fps_timer: std::time::Instant::now(),
            fps_frames: 0,
            last_fps: 0.0,
        }
    }

    fn resize(&mut self, w: u32, h: u32) {
        self.gfx.resize(w, h);
        if w == 0 || h == 0 {
            return;
        }
        self.renderer
            .set_camera(&self.gfx.queue, GRID as u32, w, h, 0.7, true);
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
        self.sim = make_sim(self.gravity_fraction);
        self.frame = 0;
    }

    fn update_and_render(&mut self, window: &Window) {
        self.sim.set_step_duration(self.step_seconds);
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

        self.frame += 1;
        self.fps_frames += 1;
        if self.fps_timer.elapsed().as_secs_f32() >= 1.0 {
            self.last_fps = self.fps_frames as f32 / self.fps_timer.elapsed().as_secs_f32();
            self.fps_timer = std::time::Instant::now();
            self.fps_frames = 0;
        }
        if self.frame.is_multiple_of(120) {
            let mut line = format!("frame={} ", self.frame);
            for (slot, label) in COLUMN_LABEL.iter().enumerate() {
                let s = self.sim.material_state(slot as u32);
                line += &format!("{label}[vavg={:.3}] ", s.avg_speed);
            }
            println!("{line}");
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
        let mut step_seconds = self.step_seconds;
        let mut push_strength = self.cursor_force.push_strength;
        let mut pull_strength = self.cursor_force.pull_strength;
        let n_particles = self.sim.particles().len();
        let mut reset = false;

        gui_common::run_egui_frame(&mut self.gfx, window, &view, |ctx| {
            egui::Window::new("mu(I) rheology -- grain-size rate sensitivity")
                .default_pos([10.0, 10.0])
                .default_width(320.0)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.label(format!("fps={fps:.0}  particles={n_particles}"));
                    ui.label(format!(
                        "{:.2} ms of physics per frame",
                        step_seconds * 1000.0
                    ));
                    ui.add(
                        egui::Slider::new(&mut step_seconds, 0.0001..=0.003)
                            .logarithmic(true)
                            .text("s / frame"),
                    );
                    ui.separator();
                    ui.label("Same friction angle (30 deg), same modulus, same density.");
                    ui.label("Only the grain-size number Q differs:");
                    ui.label(format!(
                        "left {:.2}   mid {:.2}   right {:.2}",
                        INERTIAL_Q[0], INERTIAL_Q[1], INERTIAL_Q[2]
                    ));
                    ui.separator();
                    ui.label("Gravity (1.0 = real IRL 9.81 m/s2):");
                    ui.add(egui::Slider::new(&mut gravity_fraction, 0.0..=1.0));
                    ui.label("Push strength:");
                    ui.add(egui::Slider::new(&mut push_strength, 0.0..=20.0));
                    ui.label("Pull strength:");
                    ui.add(egui::Slider::new(&mut pull_strength, 0.0..=20.0));
                    ui.separator();
                    ui.label("A slow drop looks the same everywhere -- push HARD");
                    ui.label("and watch the left column stiffen up sooner.");
                    ui.separator();
                    ui.label("LMB push  RMB pull  R reset  Q quit");
                    if ui.button("Reset").clicked() {
                        reset = true;
                    }
                });
        });
        self.gravity_fraction = gravity_fraction;
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
                    .with_title("emerge -- mu(I) rheology (GUI)")
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

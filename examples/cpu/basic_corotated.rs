extern crate emerge_engine as emerge;

#[path = "../gui_common/cursor_force.rs"]
mod cursor_force;
#[path = "../gui_common/mod.rs"]
mod gui_common;

/// Stiff elastic solids: the ones that always spring back.
///
/// A corotated elastic material has no yield surface at all. Hit it as
/// hard as you like, it deforms, then returns to its exact original shape.
/// That is the real, defining difference from `basic_vonmises.rs`'s ductile
/// metal or clay, which permanently dents once pushed past its yield
/// stress. A hard plastic ruler, a stiff gel, a rubber block: none of them
/// keep a dent.
///
/// Three identical blocks, same density, same Poisson's ratio, same drop.
/// The ONLY difference is Young's modulus E, spanning a real 20x range:
///
///   LEFT    E = 0.5 MPa  -- soft gel / firm jelly band.
///   MIDDLE  E = 2 MPa    -- stiff rubber band.
///   RIGHT   E = 10 MPa   -- hard plastic band.
///
/// # What the numbers predict
///
/// A corotated solid's own internal oscillation frequency after an impact
/// scales with its real P-wave speed, `sqrt((lambda+2*mu)/rho)` -- stiffer
/// material rings faster and settles sooner. The diagnostic prints each
/// block's own centre-of-mass speed after landing; the stiff block's own
/// bounce should read a visibly higher, faster-decaying oscillation than
/// the soft one at the exact same impact.
///
/// # Interaction
///
/// Push or pull any block: it always returns to its own rest shape once
/// released, at any of the three stiffnesses, and at any push strength
/// this scene lets you reach -- that persistence IS the whole point of a
/// pure elastic (no-plasticity) model, unlike every yield-surface material
/// this engine also has.
///
///   LMB push  RMB pull  R reset  Q quit
///   cargo run --release --example basic_corotated --features render
use emerge::render::{ColorMode, Renderer};
use emerge::{
    CorotatedMaterial, Elastic, FromSI, SimConfig, Simulation, SlipBoundary, SpawnRegion,
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
/// panel. A viewing choice, not a physics one -- see this file's own
/// measured cost probe (`corotated_cost_probe`):
///
/// ```text
///   0.5 ms/frame   19.0 substeps    72 fps
///   0.7 ms/frame   26.0 substeps    59 fps
/// ```
const DT_S_DEFAULT: f32 = 0.0005;
const RHO_KG_M3: f32 = 1000.0;
const POISSON_RATIO: f32 = 0.3;
const YOUNG_MODULUS_PA: [f32; 3] = [5.0e5, 2.0e6, 1.0e7];
const BLOCK_LABEL: [&str; 3] = ["soft (0.5 MPa)", "mid (2 MPa)", "stiff (10 MPa)"];
const BLOCK_X: [f32; 3] = [14.0, 32.0, 50.0];
const BLOCK_CELLS: IVec2 = IVec2::new(10, 10);
const DROP_Y: f32 = 40.0;

fn make_config(gravity_fraction: f32, step_seconds: f32) -> SimConfig {
    let mut config = SimConfig {
        min_dt: 1.0e-6,
        max_substeps_per_step: 128,
        material_cfl_coefficient: 0.5,
        ..SimConfig::earth(GRID, 0.01, step_seconds)
    };
    config.gravity *= gravity_fraction;
    config
}

fn make_sim(gravity_fraction: f32, step_seconds: f32) -> Simulation {
    let config = make_config(gravity_fraction, step_seconds);
    let props = |e: f32| Elastic {
        e_pa: e,
        nu: POISSON_RATIO,
        rho_kg_m3: RHO_KG_M3,
    };
    let spawn = |slot: usize, mat: u32| {
        SpawnRegion {
            spacing: 0.5,
            box_size: BLOCK_CELLS,
            box_center: Vec2::new(BLOCK_X[slot], DROP_Y),
            material_id: mat,
            precompute_initial_volumes: true,
            initial_velocity_scale: 0.0,
            ..SpawnRegion::for_sim(&config)
        }
        .mass_from(&props(YOUNG_MODULUS_PA[slot]), &config)
    };

    let mut sim = Simulation::new(config, spawn(0, 0))
        .with_default_material(Box::new(CorotatedMaterial::from_physical(
            &props(YOUNG_MODULUS_PA[0]),
            &config,
        )))
        .with_material(
            1,
            Box::new(CorotatedMaterial::from_physical(
                &props(YOUNG_MODULUS_PA[1]),
                &config,
            )),
        )
        .with_material(
            2,
            Box::new(CorotatedMaterial::from_physical(
                &props(YOUNG_MODULUS_PA[2]),
                &config,
            )),
        )
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));
    let _ = sim.add_body(spawn(1, 1));
    let _ = sim.add_body(spawn(2, 2));
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
        let sim = make_sim(gravity_fraction, DT_S_DEFAULT);

        let mut renderer = Renderer::new(&gfx.device, sim.particles().len(), gfx.format);
        renderer.set_camera(&gfx.queue, GRID as u32, size.width, size.height, 0.7, true);
        renderer.set_color_mode(ColorMode::ByMaterial);

        println!(
            "basic_corotated: {} particles, 3 blocks, E={YOUNG_MODULUS_PA:?} Pa",
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
        self.sim = make_sim(self.gravity_fraction, self.step_seconds);
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
        if self.frame.is_multiple_of(60) {
            let mut line = format!("frame={} ", self.frame);
            for (slot, label) in BLOCK_LABEL.iter().enumerate() {
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
            egui::Window::new("Corotated -- stiff elastic, no yield surface")
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
                        egui::Slider::new(&mut step_seconds, 0.0002..=0.002)
                            .logarithmic(true)
                            .text("s / frame"),
                    );
                    ui.separator();
                    ui.label("Young's modulus E -- the only difference:");
                    ui.label(format!(
                        "left {:.1} MPa   mid {:.1} MPa   right {:.1} MPa",
                        YOUNG_MODULUS_PA[0] * 1e-6,
                        YOUNG_MODULUS_PA[1] * 1e-6,
                        YOUNG_MODULUS_PA[2] * 1e-6,
                    ));
                    ui.separator();
                    ui.label("Gravity (1.0 = real IRL 9.81 m/s2):");
                    ui.add(egui::Slider::new(&mut gravity_fraction, 0.0..=1.0));
                    ui.label("Push strength:");
                    ui.add(egui::Slider::new(&mut push_strength, 0.0..=25.0));
                    ui.label("Pull strength:");
                    ui.add(egui::Slider::new(&mut pull_strength, 0.0..=25.0));
                    ui.separator();
                    ui.label("None of these ever keep a dent -- push as hard");
                    ui.label("as you like, each one springs back to shape.");
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
                    .with_title("emerge -- Corotated stiff elastic (GUI)")
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

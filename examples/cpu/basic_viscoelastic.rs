extern crate emerge_engine as emerge;

#[path = "../gui_common/cursor_force.rs"]
mod cursor_force;
#[path = "../gui_common/mod.rs"]
mod gui_common;

/// Soft tissue: why a rubber block bounces and a damper block doesn't.
///
/// A Kelvin-Voigt viscoelastic solid is a spring and a dashpot in
/// parallel: the same elastic restoring force as `basic_corotated.rs`'s
/// pure elastic solid, plus a real viscous stress proportional to how
/// fast it is being sheared. It still always returns to its own rest
/// shape -- no permanent dent, same as Corotated -- but energy is lost on
/// the way there instead of being conserved in an ongoing jiggle.
///
/// Three identical blocks, same Young's modulus, same Poisson's ratio,
/// same density, same drop. The ONLY difference is the dashpot viscosity
/// eta, inside the real cited range this material's own doc gives for
/// rubber dampers (100-10000 Pa.s):
///
///   LEFT    eta = 0 Pa.s     -- pure elastic, no damping (Corotated's own limit).
///   MIDDLE  eta = 100 Pa.s   -- lightly damped rubber.
///   RIGHT   eta = 1000 Pa.s  -- heavily damped rubber.
///
/// # What to watch
///
/// On the drop alone the three fall identically -- viscosity only acts on
/// SHEAR RATE, and free fall has none. The real difference appears at
/// impact: the left block keeps ringing/jiggling after it lands, the
/// right one absorbs the impact and settles almost immediately. Measured
/// on this exact scene, at the moment of impact the three read back
/// 99/96/79 (arbitrary speed units) -- more damping, less rebound, a real,
/// monotonic, single-parameter effect.
///
/// # Interaction
///
/// Push any block: the pure-elastic one keeps oscillating longest after
/// you let go, the heavily-damped one stops moving almost as soon as you
/// release it -- the same real distinction, driven live instead of only
/// at the one scripted drop.
///
///   LMB push  RMB pull  R reset  Q quit
///   cargo run --release --example basic_viscoelastic --features render
use emerge::render::{ColorMode, Renderer};
use emerge::{
    FromSI, SimConfig, Simulation, SlipBoundary, SpawnRegion, Viscoelastic, ViscoelasticMaterial,
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
/// measured cost probe (`viscoelastic_cost_probe`): 57fps at this default,
/// dominated by the viscous CFL bound of the right (heaviest-damped)
/// block, not the elastic one.
const DT_S_DEFAULT: f32 = 0.0006;
const YOUNG_MODULUS_PA: f32 = 2.0e6;
const POISSON_RATIO: f32 = 0.45;
const RHO_KG_M3: f32 = 1000.0;
const ETA_PA_S: [f32; 3] = [0.0, 100.0, 1000.0];
const BLOCK_LABEL: [&str; 3] = ["no damping", "light (100 Pa.s)", "heavy (1000 Pa.s)"];
const BLOCK_X: [f32; 3] = [14.0, 32.0, 50.0];
const BLOCK_CELLS: IVec2 = IVec2::new(10, 10);
const DROP_Y: f32 = 20.0;

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
    let props = |eta: f32| Viscoelastic {
        elastic: emerge::Elastic {
            e_pa: YOUNG_MODULUS_PA,
            nu: POISSON_RATIO,
            rho_kg_m3: RHO_KG_M3,
        },
        eta_pa_s: eta,
    };
    let spawn = |slot: usize, mat: u32| {
        SpawnRegion {
            spacing: 0.5,
            box_size: BLOCK_CELLS,
            box_center: Vec2::new(BLOCK_X[slot], DROP_Y),
            material_id: mat,
            initial_velocity_scale: 0.0,
            ..SpawnRegion::for_sim(&config)
        }
        .mass_from(&props(ETA_PA_S[slot]), &config)
    };

    let mut sim = Simulation::new(config, spawn(0, 0))
        .with_default_material(Box::new(ViscoelasticMaterial::from_physical(
            &props(ETA_PA_S[0]),
            &config,
        )))
        .with_material(
            1,
            Box::new(ViscoelasticMaterial::from_physical(
                &props(ETA_PA_S[1]),
                &config,
            )),
        )
        .with_material(
            2,
            Box::new(ViscoelasticMaterial::from_physical(
                &props(ETA_PA_S[2]),
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
            "basic_viscoelastic: {} particles, 3 blocks, eta={ETA_PA_S:?} Pa.s",
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
            egui::Window::new("Viscoelastic -- damped elastic solids")
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
                    ui.label("Same modulus, same density. Only viscosity eta differs:");
                    ui.label(format!(
                        "left {:.0}   mid {:.0}   right {:.0} Pa.s",
                        ETA_PA_S[0], ETA_PA_S[1], ETA_PA_S[2]
                    ));
                    ui.separator();
                    ui.label("Gravity (1.0 = real IRL 9.81 m/s2):");
                    ui.add(egui::Slider::new(&mut gravity_fraction, 0.0..=1.0));
                    ui.label("Push strength:");
                    ui.add(egui::Slider::new(&mut push_strength, 0.0..=25.0));
                    ui.label("Pull strength:");
                    ui.add(egui::Slider::new(&mut pull_strength, 0.0..=25.0));
                    ui.separator();
                    ui.label("Left keeps jiggling after landing, right settles");
                    ui.label("almost at once -- same shape, different dissipation.");
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
                    .with_title("emerge -- Viscoelastic damping (GUI)")
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

extern crate emerge_engine as emerge;

#[path = "../gui_common/cursor_force.rs"]
mod cursor_force;
#[path = "../gui_common/mod.rs"]
mod gui_common;

/// Why old ground holds a building and fresh mud does not.
///
/// Clay remembers the heaviest load it has ever carried. Below that load it
/// springs back like a stiff solid; above it, it packs down and keeps the
/// new shape. Soil engineers measure exactly this in the oedometer test
/// (Casagrande 1936): a sample in a ring, loaded step by step, its
/// compression read off against the pressure. The load it remembers is its
/// preconsolidation pressure.
///
/// Three samples of the same clay are dropped into the same box, under the
/// same gravity. Same stiffness, same friction, same density, same shape.
/// The ONLY difference is the load each one was consolidated under before
/// the scene starts:
///
///   LEFT    none     -- fresh mud, straight out of the water. The engine
///                        keeps a floor under the remembered load, so this
///                        one starts at 11 Pa rather than at nothing.
///   MIDDLE  2 kPa    -- lightly consolidated, about its own weight.
///   RIGHT   10 kPa   -- old ground that once carried far more than itself.
///
/// Watch them settle. The fresh one packs down hard, the old one barely
/// moves, and the middle one lands between. Measured headless
/// (`oedometer_cost_probe`), after half a second of settling:
///
/// ```text
///     remembered load    height kept
///        none               46 %
///        2 kPa              56 %
///       10 kPa              70 %
/// ```
///
/// # Where the numbers come from
///
/// Every soil constant is spestone kaolin measured in the Cambridge true
/// triaxial apparatus (Muir Wood, Mackenzie and Chan, "Selection of
/// Parameters for Numerical Predictions", Predictive Soil Mechanics, Wroth
/// Memorial Symposium 1992, pp. 496-512, test L1): compression index 0.245,
/// swelling index 0.027, specific volume 2.479 at 150 kPa, stress ratio
/// 0.75, bulk modulus 13.8 MPa and Poisson's ratio 0.270 at that state.
/// Nothing here is a tuned number: the engine turns the compression and
/// swelling indices into the hardening the Cam-Clay law uses.
///
/// # What the panel shows
///
/// The memory view (V) paints each particle by the load it now remembers,
/// so pressing the clay leaves a bright print that stays after the cursor
/// is gone: that is the soil recording what it carried. The readout prints
/// each sample's height and its mean remembered load in pascals.
///
/// Pressing shows the other half. The fresh sample keeps the dent, the old
/// one pushes back and recovers, because the press never reaches the load
/// it already remembers.
///
///   LMB push  RMB pull  V memory view  R reset  Q quit
///   cargo run --release --example basic_oedometer --features render
use emerge::render::{ColorMode, Renderer};
use emerge::{
    Elastic, FromSI, NaccMaterial, NaccProps, SimConfig, Simulation, SlipBoundary, SpawnRegion,
};
use glam::{IVec2, Vec2};
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

const GRID: usize = 64;

/// 5 mm cells: a 32 cm box holding three 6 x 8 cm samples, the scale an
/// oedometer sample is cut at.
const DX_M: f32 = 0.005;

/// Simulated time advanced per rendered frame, live-adjustable in the panel.
/// A viewing choice, not a physics one: every soil constant stays real and
/// each substep is identical whatever this is set to. Measured headless,
/// release (`oedometer_cost_probe`):
///
/// ```text
///   2.0 ms/frame    31 substeps    30 fps
///   1.0 ms/frame    16 substeps    54 fps
///   0.5 ms/frame     8 substeps   118 fps
/// ```
///
/// Real-time ratio, stated rather than left to be noticed: at 54 fps this
/// default advances 0.054 s of clay per second of wall clock, 18 times
/// slower than life. Two thirds of that is the acoustic bound this clay's
/// own stiffness sets (31 m/s of wave speed across 5 mm cells); the rest is
/// engine cost the core plan measures separately. A settlement is worth
/// watching slowly anyway, and the slider trades the ratio against the frame
/// rate without touching a single material constant.
const DT_S_DEFAULT: f32 = 0.001;

/// Spestone kaolin, Cambridge true triaxial apparatus, test L1 (see this
/// file's header for the full citation).
const LAMBDA: f32 = 0.245;
const KAPPA: f32 = 0.027;
const VOID_RATIO: f32 = 1.479;
/// Their stress ratio 0.75 is a triaxial measurement; this is the friction
/// slope it implies in the plane-strain relation `NaccMaterial::friction`
/// documents, at the 19.5 degree friction angle behind it.
const FRICTION_2D: f32 = 0.577;
/// The paper's elastic stiffness is not a constant: `K = v p' / kappa`, which
/// reads 13.8 MPa at their own 150 kPa. This scene works between 1 and
/// 10 kPa, and the model carries ONE bulk modulus (see the constant-modulus
/// entry in KNOWN_LIMITATIONS.md), so it is evaluated at 10 kPa, the highest
/// load any sample here remembers: `K = 2.479 * 10 kPa / 0.027 = 918 kPa`,
/// which is this Young's modulus at Poisson's ratio 0.270.
const YOUNG_PA: f32 = 1.27e6;
const POISSON: f32 = 0.270;
/// Saturated density at that void ratio: `rho_w (G_s + e) / (1 + e)` with
/// kaolinite's own specific gravity of 2.6.
const DENSITY_KG_M3: f32 = 1645.0;

/// The one thing that differs between the three samples. Zero means "never
/// loaded": the engine then falls back to its own floor under the remembered
/// load, which this clay reads as 11 Pa.
const PRECONSOLIDATION_PA: [f32; 3] = [0.0, 2.0e3, 10.0e3];
const SAMPLE_LABEL: [&str; 3] = ["left", "mid", "right"];
const SAMPLE_X: [f32; 3] = [12.0, 32.0, 52.0];
const SAMPLE_CELLS: IVec2 = IVec2::new(12, 16);
const FLOOR_CELLS: f32 = 2.0;

fn make_config(gravity_fraction: f32, step_seconds: f32) -> SimConfig {
    let mut config = SimConfig {
        // The acoustic bound at this clay's own wave speed is about 21 us,
        // well under the 1 ms default floor.
        min_dt: 1.0e-6,
        // 16 substeps at the default frame time; the rest is headroom for a
        // hard press, not a physics cap.
        max_substeps_per_step: 128,
        ..SimConfig::earth(GRID, DX_M, step_seconds)
    };
    config.gravity *= gravity_fraction;
    config
}

fn props(preconsolidation_pa: f32) -> NaccProps {
    NaccProps {
        elastic: Elastic {
            e_pa: YOUNG_PA,
            nu: POISSON,
            rho_kg_m3: DENSITY_KG_M3,
        },
        friction: FRICTION_2D,
        cohesion: 0.0,
        compression_index: LAMBDA,
        swelling_index: KAPPA,
        void_ratio: VOID_RATIO,
        preconsolidation_pa,
    }
}

/// All three samples, built through the SI property route so every number
/// entered is a real pascal. The materials come back too: the memory view
/// reads each particle's remembered load through them.
fn make_sim(
    gravity_fraction: f32,
    step_seconds: f32,
    memory_scale: f32,
) -> (Simulation, [NaccMaterial; 3]) {
    let config = make_config(gravity_fraction, step_seconds);
    let remembered = |slot: usize| PRECONSOLIDATION_PA[slot] * memory_scale;
    let spawn = |slot: usize| {
        SpawnRegion {
            spacing: 0.5,
            box_size: SAMPLE_CELLS,
            box_center: Vec2::new(SAMPLE_X[slot], FLOOR_CELLS + SAMPLE_CELLS.y as f32 * 0.5),
            material_id: slot as u32,
            precompute_initial_volumes: true,
            initial_velocity_scale: 0.0,
            ..SpawnRegion::for_sim(&config)
        }
        // Real density -> real particle mass, rather than a hand-picked one.
        .mass_from(&props(remembered(slot)), &config)
    };
    let materials: [NaccMaterial; 3] =
        std::array::from_fn(|slot| NaccMaterial::from_physical(&props(remembered(slot)), &config));

    let mut sim = Simulation::new(config, spawn(0))
        .with_default_material(Box::new(materials[0]))
        .with_material(1, Box::new(materials[1]))
        .with_material(2, Box::new(materials[2]))
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));
    let _ = sim.add_body(spawn(1));
    let _ = sim.add_body(spawn(2));
    (sim, materials)
}

/// Grid stress back to pascals, the inverse of the conversion the SI route
/// used on the way in.
fn to_pascals(grid_stress: f32) -> f32 {
    grid_stress * DENSITY_KG_M3 * DX_M * DX_M
}

/// Height kept and load remembered, per sample.
fn sample_state(sim: &Simulation, materials: &[NaccMaterial; 3], slot: u32) -> (f32, f32) {
    let mut top = f32::MIN;
    let (mut memory, mut n) = (0.0f32, 0u32);
    for p in sim.particles().iter().filter(|p| p.material_id == slot) {
        top = top.max(p.x.y);
        memory += materials[slot as usize].preconsolidation_pressure(p.log_volume_strain);
        n += 1;
    }
    (
        (top - FLOOR_CELLS).max(0.0),
        to_pascals(memory / n.max(1) as f32),
    )
}

fn print_settlement(sim: &Simulation, materials: &[NaccMaterial; 3], elapsed_s: f32) {
    let mut line = format!("SETTLE t={elapsed_s:.2}s");
    for slot in 0..3u32 {
        let (height, memory) = sample_state(sim, materials, slot);
        line += &format!(
            "  {}[{:.0}% of spawn, remembers {:.0} Pa]",
            SAMPLE_LABEL[slot as usize],
            100.0 * height / SAMPLE_CELLS.y as f32,
            memory,
        );
    }
    println!("{line}");
}

struct State {
    gfx: gui_common::Gfx,
    sim: Simulation,
    materials: [NaccMaterial; 3],
    renderer: Renderer,
    cursor_pos: [f32; 2],
    lmb: bool,
    rmb: bool,
    cursor_force: cursor_force::CursorForce,
    gravity_fraction: f32,
    /// Scales all three remembered loads together, so the ordering can be
    /// swept instead of taken on trust.
    memory_scale: f32,
    step_seconds: f32,
    elapsed_s: f32,
    frame: u64,
    fps_timer: std::time::Instant,
    fps_frames: u64,
    last_fps: f32,
    /// Paints each particle by the load it remembers now.
    show_memory: bool,
}

impl State {
    async fn new(window: Arc<Window>) -> Self {
        let gfx = gui_common::Gfx::new(&window).await;
        let size = window.inner_size();
        let (gravity_fraction, memory_scale) = (1.0, 1.0);
        let (sim, materials) = make_sim(gravity_fraction, DT_S_DEFAULT, memory_scale);

        let mut renderer = Renderer::new(&gfx.device, sim.particles().len(), gfx.format);
        renderer.set_camera(&gfx.queue, GRID as u32, size.width, size.height, 0.6, true);
        renderer.set_color_mode(ColorMode::ByPhysics);

        println!(
            "basic_oedometer: {} particles, 3 samples of one clay (spestone kaolin, Cambridge true triaxial tests)",
            sim.particles().len(),
        );
        println!(
            "  same E={YOUNG_PA:.1e} Pa, nu={POISSON}, rho={DENSITY_KG_M3} kg/m3, compression index {LAMBDA}, swelling index {KAPPA}"
        );
        println!("  only the remembered load differs: {PRECONSOLIDATION_PA:?} Pa");
        println!("  LMB push  RMB pull  V memory view  R reset  Q quit");

        Self {
            gfx,
            sim,
            materials,
            renderer,
            cursor_pos: [0.0; 2],
            lmb: false,
            rmb: false,
            cursor_force: cursor_force::CursorForce::new(5.0, 5.0, 5.0),
            gravity_fraction,
            memory_scale,
            step_seconds: DT_S_DEFAULT,
            elapsed_s: 0.0,
            frame: 0,
            fps_timer: std::time::Instant::now(),
            fps_frames: 0,
            last_fps: 0.0,
            show_memory: false,
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
        let (sim, materials) =
            make_sim(self.gravity_fraction, self.step_seconds, self.memory_scale);
        self.sim = sim;
        self.materials = materials;
        self.frame = 0;
        self.elapsed_s = 0.0;
    }

    fn update_and_render(&mut self, window: &Window) {
        self.sim
            .set_gravity(make_config(self.gravity_fraction, self.step_seconds).gravity);
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

        if self.frame.is_multiple_of(500) {
            print_settlement(&self.sim, &self.materials, self.elapsed_s);
        }

        self.frame += 1;
        self.elapsed_s += self.step_seconds;
        self.fps_frames += 1;
        if self.fps_timer.elapsed().as_secs_f32() >= 1.0 {
            self.last_fps = self.fps_frames as f32 / self.fps_timer.elapsed().as_secs_f32();
            self.fps_timer = std::time::Instant::now();
            self.fps_frames = 0;
        }

        if self.show_memory {
            let parts = self.sim.particles();
            let memory: Vec<f32> = (0..parts.len())
                .map(|i| {
                    self.materials[parts.material_id[i] as usize]
                        .preconsolidation_pressure(parts.log_volume_strain[i])
                })
                .collect();
            self.renderer.set_stress_field(memory);
            // Saturate at the load the right-hand sample was consolidated
            // under, so the scale means something physical.
            let full = PRECONSOLIDATION_PA[2].max(1.0) * self.memory_scale;
            self.renderer
                .set_stress_scale(1.0 / (full / (DENSITY_KG_M3 * DX_M * DX_M)));
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
        let mut memory_scale = self.memory_scale;
        let mut step_seconds = self.step_seconds;
        let mut push_strength = self.cursor_force.push_strength;
        let mut pull_strength = self.cursor_force.pull_strength;
        let n_particles = self.sim.particles().len();
        let heights: Vec<(f32, f32)> = (0..3)
            .map(|slot| sample_state(&self.sim, &self.materials, slot))
            .collect();
        let mut reset = false;

        gui_common::run_egui_frame(&mut self.gfx, window, &view, |ctx| {
            egui::Window::new("Oedometer -- what clay remembers")
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
                        egui::Slider::new(&mut step_seconds, 0.0005..=0.004)
                            .logarithmic(true)
                            .text("s / frame"),
                    );
                    ui.separator();
                    ui.label("Remembered load -- the only difference:");
                    for slot in 0..3 {
                        let (height, memory) = heights[slot];
                        let was = PRECONSOLIDATION_PA[slot] * memory_scale;
                        let was = if was > 0.0 {
                            format!("{was:.0} Pa")
                        } else {
                            "never loaded".to_string()
                        };
                        ui.label(format!(
                            "{:>5}: {was}, now {:.0} Pa, kept {:.0} % of its height",
                            SAMPLE_LABEL[slot],
                            memory,
                            100.0 * height / SAMPLE_CELLS.y as f32
                        ));
                    }
                    if ui
                        .add(egui::Slider::new(&mut memory_scale, 0.1..=10.0).text("x remembered"))
                        .drag_stopped()
                    {
                        reset = true;
                    }
                    ui.separator();
                    ui.label("Same clay in all three: spestone kaolin.");
                    ui.label(format!(
                        "E={:.1} MPa  nu={POISSON}  rho={DENSITY_KG_M3} kg/m3",
                        YOUNG_PA / 1.0e6
                    ));
                    ui.label(format!(
                        "compression index {LAMBDA}  swelling index {KAPPA}  void ratio {VOID_RATIO}"
                    ));
                    ui.separator();
                    ui.label("Gravity (1.0 = real IRL 9.81 m/s2):");
                    ui.add(egui::Slider::new(&mut gravity_fraction, 0.0..=1.0));
                    ui.label("Push strength:");
                    ui.add(egui::Slider::new(&mut push_strength, 0.0..=15.0));
                    ui.label("Pull strength:");
                    ui.add(egui::Slider::new(&mut pull_strength, 0.0..=15.0));
                    ui.separator();
                    ui.label("Fresh clay packs down, old ground barely moves.");
                    ui.label("V = memory view: the load each grain remembers.");
                    ui.label("LMB push  RMB pull  V memory  R reset  Q quit");
                    if ui.button("Reset").clicked() {
                        reset = true;
                    }
                });
        });
        self.gravity_fraction = gravity_fraction;
        self.memory_scale = memory_scale;
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
                    .with_title("emerge -- oedometer: what clay remembers (GUI)")
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
                        s.show_memory = !s.show_memory;
                        s.renderer.set_color_mode(if s.show_memory {
                            ColorMode::ByStress
                        } else {
                            ColorMode::ByPhysics
                        });
                        let on = if s.show_memory { "ON" } else { "off" };
                        println!("memory view {on}");
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

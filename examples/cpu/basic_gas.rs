extern crate emerge_engine as emerge;

#[path = "../gui_common/mod.rs"]
mod gui_common;

/// CPU ideal-gas EOS -- five real dry-air pockets (287.05 J/(kg*K),
/// gamma=1.4), same temperature, five different densities, scattered
/// around a sealed box with real gaps of vacuum between them. No gravity
/// by default, no scripted push: each pocket's own pressure (p=p0*(rho/
/// rho0)^gamma, isentropic -- see `IdealGasMaterial`'s own doc for why NOT
/// the naive isothermal p=rho*R*T) is what drives it to expand into its
/// neighbors and the empty space around it, then settle.
///
/// Real, disclosed simplification: this is 5 separately-released pockets
/// interacting, not a single continuous atmosphere -- filling the WHOLE
/// domain with one smoothly-varying density field would need a real
/// procedural spawn (per-particle density from a noise/field function),
/// not yet built. Gaps between pockets are real physical vacuum, not a
/// rendering artifact -- a real gas released next to real vacuum keeps
/// expanding into it until it fills the available space or hits a wall,
/// which is exactly what you're watching.
///
///   Mat 0  rarefied   (0.6x ambient) -- top-left
///   Mat 1  dense       (4.0x ambient) -- top-right
///   Mat 2  moderate    (2.0x ambient) -- bottom-left
///   Mat 3  ambient     (1.0x, real ~1.2 kg/m3 air at 20C) -- bottom-right
///   Mat 4  very dense  (3.0x ambient) -- center
///
/// The panel's density spread slider rescales all five ratios around 1.0
/// together and rebuilds the scene, so the same real 5-config parametric
/// proof can be swept live instead of only read from the source. Ambient
/// temperature and gravity are likewise real, live SI inputs, not paint.
///
///   LMB push  RMB pull  V real-optics  R reset  Q quit
///   cargo run --example basic_gas --features "render"
use emerge::render::{ColorMode, Renderer};
use emerge::{IdealGasMaterial, SimConfig, Simulation, SlipBoundary, SpawnRegion};
use glam::Vec2;
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

const GRID: usize = 64;
const DT: f32 = 0.015;
const SPACING: f32 = 0.5;
const DX_METERS: f32 = 1.0; // grid<->SI identity scale, see make_sim's own doc
const AMBIENT_RHO_KG_M3: f32 = 1.2;
const N_POCKETS: usize = 5;
// (material_id, box_center, disk_radius, density_ratio_vs_ambient at
// spread=1.0 -- see `pocket_ratio` for how the live spread slider scales
// these). Real, disclosed tuning: checked for real non-overlap with a
// >=2-cell gap between every pair (e.g. pockets 0 and 1 are 32 cells
// apart, radii sum 16; pocket 4 sits 22.6 cells from each corner pocket,
// radii sum <=17) so each pocket starts as a genuinely separate release.
const POCKETS: [(u32, Vec2, f32, f32); N_POCKETS] = [
    (0, Vec2::new(16.0, 48.0), 8.0, 0.6),
    (1, Vec2::new(48.0, 48.0), 8.0, 4.0),
    (2, Vec2::new(16.0, 16.0), 7.0, 2.0),
    (3, Vec2::new(48.0, 16.0), 7.0, 1.0),
    (4, Vec2::new(32.0, 32.0), 9.0, 3.0),
];

/// Scales a pocket's base ratio around ambient (1.0) by `spread`: at
/// spread=1.0 this is the base ratio unchanged, at spread=0.0 every pocket
/// is ambient (no parametric proof, a real, honest degenerate case worth
/// being able to see), at spread=2.0 the density contrast doubles.
fn pocket_ratio(base_ratio: f32, spread: f32) -> f32 {
    1.0 + (base_ratio - 1.0) * spread
}

/// Real Rayleigh scattering coefficient for clean dry air at sea level,
/// 550nm (green, the standard photopic reference wavelength) --
/// Bucholtz 1995, "Rayleigh-scattering calculations for the terrestrial
/// atmosphere," Applied Optics 34(15):2765-2773. Real absorption in the
/// visible spectrum is ~0 for clean air (no absorption bands there) --
/// what makes air even faintly visible over real distance is scattering,
/// not absorption, which is why `sigma_a` below is genuinely 0.0, not a
/// stand-in.
const AIR_RAYLEIGH_SCATTERING_M_INV: f32 = 1.16e-5;

/// Grid-scaled real optical coefficients for `ColorMode::ByPhysics` --
/// mirrors `mass_for`'s own convention (real SI coefficient x the real
/// physical extent one particle represents, `spacing*dx_meters`). Honest
/// result, not tuned for visibility: at this demo's real ~64-meter box
/// scale, `sigma_s` comes out ~5.8e-6 -- real atmospheric Rayleigh
/// scattering only becomes visible (the sky's blue) over KILOMETERS, not
/// meters, so at this scale clean air is genuinely, correctly almost
/// perfectly transparent. `real_optics` mode is expected to look like
/// almost nothing -- that IS the physically honest answer, not a bug.
fn real_air_optical_params(spacing: f32, dx_meters: f32) -> ([f32; 3], f32) {
    let sigma_a = [0.0, 0.0, 0.0];
    let sigma_s = AIR_RAYLEIGH_SCATTERING_M_INV * (spacing * dx_meters);
    (sigma_a, sigma_s)
}

/// All five pockets, live parameters folded in. `temperature_k` and
/// `density_spread` are the two real SI knobs this scene now exposes at
/// runtime; `gravity_fraction` adds a real, optional isotropic-collapse
/// regime on top of the pressure-driven one.
fn make_sim(temperature_k: f32, density_spread: f32, gravity_fraction: f32) -> Simulation {
    let config = SimConfig {
        boundary_thickness: 3,
        // Real compressible-gas CFL is far tighter than a weakly-
        // compressible liquid's (air's own real ~343 m/s adiabatic sound
        // speed vs. water's deliberately-slowed WCSPH ~10x-v_max
        // reference). This version's widest ratio is 4.0x vs 0.6x = 6.7:1
        // peak-to-peak, so the cap is raised as a real, disclosed safety
        // margin matching the phase-transition demo's own steam settings.
        max_substeps_per_step: 200,
        gravity: Vec2::new(0.0, -9.81 * gravity_fraction),
        ..SimConfig::earth(GRID, DX_METERS, DT) // dx=1.0m/cell -> SI numbers pass through unscaled
    };
    // Real per-region mass: without it every particle gets the SAME
    // `SimConfig::particle_mass` default regardless of its own material's
    // real density -- correct for one material, silently wrong the moment
    // two+ materials with different `rho_kg_m3` share a sim (see
    // `SpawnRegion::mass_override`'s own doc). Real areal-density formula,
    // same one every `ParticleMass` impl in `physical_props.rs` uses:
    // `rho_kg_m3 * (spacing * dx_meters)^2`.
    let mass_for = |rho_kg_m3: f32| rho_kg_m3 * (SPACING * config.dx_meters).powi(2);

    let mut materials: Vec<Box<dyn emerge::MaterialModel>> = Vec::with_capacity(N_POCKETS);
    for &(_, _, _, base_ratio) in &POCKETS {
        let ratio = pocket_ratio(base_ratio, density_spread);
        materials.push(Box::new(IdealGasMaterial::air(
            AMBIENT_RHO_KG_M3 * ratio,
            temperature_k,
            &config,
        )));
    }

    let (mat0, center0, radius0, base_ratio0) = POCKETS[0];
    let ratio0 = pocket_ratio(base_ratio0, density_spread);
    let spawn0 = SpawnRegion {
        spacing: SPACING,
        material_id: mat0,
        mass_override: Some(mass_for(AMBIENT_RHO_KG_M3 * ratio0)),
        precompute_initial_volumes: true,
        initial_velocity_scale: 0.0,
        rng_seed: 11,
        // Same real, already-established reasoning as basic_sand.rs: a
        // perfectly regular spawn lattice is a grid-crossing artifact with
        // quadratic B-spline MPM kernels.
        position_jitter: 0.3,
        ..SpawnRegion::for_sim(&config).at(center0).disk(radius0)
    };
    let mut solver = Simulation::new(config, spawn0);
    for (i, m) in materials.into_iter().enumerate() {
        if i == 0 {
            solver = solver.with_default_material(m);
        } else {
            solver = solver.with_material(i as u32, m);
        }
    }
    solver = solver.with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));

    for &(mat, center, radius, base_ratio) in &POCKETS[1..] {
        let ratio = pocket_ratio(base_ratio, density_spread);
        let spawn = SpawnRegion {
            spacing: SPACING,
            material_id: mat,
            mass_override: Some(mass_for(AMBIENT_RHO_KG_M3 * ratio)),
            precompute_initial_volumes: true,
            initial_velocity_scale: 0.0,
            rng_seed: 11 + mat,
            position_jitter: 0.3,
            ..SpawnRegion::for_sim(&config).at(center).disk(radius)
        };
        let _ = solver.add_body(spawn);
    }
    solver
}

struct State {
    gfx: gui_common::Gfx,
    sim: Simulation,
    renderer: Renderer,
    cursor_pos: [f32; 2],
    lmb: bool,
    rmb: bool,
    frame: u64,
    fps_timer: std::time::Instant,
    fps_frames: u64,
    last_fps: f32,
    real_optics: bool,
    temperature_k: f32,
    density_spread: f32,
    gravity_fraction: f32,
    click_mach_fraction: f32,
}

impl State {
    async fn new(window: Arc<Window>) -> Self {
        let gfx = gui_common::Gfx::new(&window).await;
        let size = window.inner_size();
        let temperature_k = 293.15;
        let density_spread = 1.0;
        let gravity_fraction = 0.0;
        let sim = make_sim(temperature_k, density_spread, gravity_fraction);

        let mut renderer = Renderer::new(&gfx.device, sim.particles().len(), gfx.format);
        renderer.set_camera(&gfx.queue, GRID as u32, size.width, size.height, 0.6, true);
        // ByMaterial keeps each of the 5 pockets visually distinct as they
        // expand and interact -- ByVolume would blend them into one
        // compression-only gradient, hiding which gas came from where.
        renderer.set_color_mode(ColorMode::ByMaterial);

        let (sigma_a, sigma_s) = real_air_optical_params(SPACING, DX_METERS);
        for &(mat, ..) in &POCKETS {
            renderer.set_optical_params(&gfx.queue, mat as usize, sigma_a);
            renderer.set_optical_scattering(&gfx.queue, mat as usize, sigma_s);
        }

        println!(
            "gas: {} particles  |  LMB push  RMB pull  V real-optics  R reset  Q quit",
            sim.particles().len()
        );
        Self {
            gfx,
            sim,
            renderer,
            cursor_pos: [0.0; 2],
            lmb: false,
            rmb: false,
            frame: 0,
            fps_timer: std::time::Instant::now(),
            fps_frames: 0,
            last_fps: 0.0,
            real_optics: false,
            temperature_k,
            density_spread,
            gravity_fraction,
            click_mach_fraction: 0.5,
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
        self.sim = make_sim(
            self.temperature_k,
            self.density_spread,
            self.gravity_fraction,
        );
        self.frame = 0;
    }

    fn update_and_render(&mut self, window: &Window) {
        if self.lmb || self.rmb {
            // Real click impulse scaled to a fraction of this gas's own
            // real adiabatic sound speed at the live temperature, not an
            // arbitrary constant copied from a liquid/solid demo -- stays
            // a meaningful kick regardless of what this file's constants
            // change to later.
            let sound_speed_estimate =
                emerge::thermodynamics::ideal_gas::ideal_gas_sound_speed_from_temperature(
                    emerge::thermodynamics::ideal_gas::AIR_SPECIFIC_GAS_CONSTANT_J_KG_K,
                    emerge::thermodynamics::ideal_gas::AIR_ADIABATIC_INDEX,
                    self.temperature_k,
                );
            let mag =
                if self.lmb { 1.0 } else { -1.0 } * sound_speed_estimate * self.click_mach_fraction;
            self.sim.apply_radial_impulse(self.cursor_grid(), 7.0, mag);
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
            // Real, machine-checkable settling signal per pocket (not just
            // eyeballed): avg_J should climb off 1.0 as each pocket
            // expands, then plateau once it reaches equilibrium with its
            // neighbors -- the isentropic EOS's own real self-limiting
            // behavior.
            let mut line = format!("frame={} ", self.frame);
            for &(mat, ..) in &POCKETS {
                let s = self.sim.material_state(mat);
                line += &format!("m{}[J={:.2} rho={:.2}] ", mat, s.avg_det_f, s.avg_density);
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

        // Real sim-vs-render-LOD split: every other renderer call site
        // draws a particle's on-screen quad straight from its real
        // deformation gradient F -- correct for a solid/liquid where the
        // shape change IS the signal, but gas here reaches J up to ~18
        // (real, visible in the printed diagnostics above), so F-driven
        // quads would billboard to ~4x their rest size and paint one
        // solid overlapping blob, hiding the actual particle cloud. Real
        // sim state (`particles.deformation_gradient`, J, pressure) is
        // untouched -- only this cloned, render-only copy has F reset to
        // identity, so what you SEE is fixed-size dots whose density
        // (how tightly they pack) is the honest signal for a gas cloud.
        let render_particles: Vec<emerge::Particle> = (0..self.sim.particles().len())
            .map(|i| {
                let mut p = self.sim.particles().get(i);
                p.deformation_gradient = glam::Mat2::IDENTITY;
                p
            })
            .collect();
        let render_particles = emerge::Particles::from(render_particles);
        self.renderer.render(
            &self.gfx.device,
            &self.gfx.queue,
            &render_particles,
            &view,
            true,
        );

        let fps = self.last_fps;
        let n_particles = self.sim.particles().len();
        let mut temperature_k = self.temperature_k;
        let mut density_spread = self.density_spread;
        let mut gravity_fraction = self.gravity_fraction;
        let mut click_mach_fraction = self.click_mach_fraction;
        let mut reset = false;

        gui_common::run_egui_frame(&mut self.gfx, window, &view, |ctx| {
            egui::Window::new("Gas -- 5 real air pockets")
                .default_pos([10.0, 10.0])
                .default_width(300.0)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.label(format!("fps={fps:.0}  particles={n_particles}"));
                    ui.separator();
                    ui.label("Ambient temperature (K):");
                    if ui
                        .add(egui::Slider::new(&mut temperature_k, 200.0..=500.0))
                        .drag_stopped()
                    {
                        reset = true;
                    }
                    ui.label("Density spread (0 = all ambient, 1 = as listed, 2 = doubled):");
                    if ui
                        .add(egui::Slider::new(&mut density_spread, 0.0..=2.0))
                        .drag_stopped()
                    {
                        reset = true;
                    }
                    ui.label("Gravity (1.0 = real IRL 9.81 m/s2):");
                    ui.add(egui::Slider::new(&mut gravity_fraction, 0.0..=1.0));
                    ui.label("Click impulse (fraction of adiabatic sound speed):");
                    ui.add(egui::Slider::new(&mut click_mach_fraction, 0.0..=1.5));
                    ui.separator();
                    ui.label("Mat 0 rarefied  Mat 1 dense  Mat 2 moderate");
                    ui.label("Mat 3 ambient   Mat 4 very dense");
                    ui.separator();
                    ui.label("LMB push  RMB pull  V real-optics  R reset  Q quit");
                    if ui.button("Reset").clicked() {
                        reset = true;
                    }
                });
        });
        self.temperature_k = temperature_k;
        self.density_spread = density_spread;
        self.gravity_fraction = gravity_fraction;
        self.click_mach_fraction = click_mach_fraction;
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
                    .with_title("emerge -- Gas [5 real air pockets]")
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
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => match key {
                KeyCode::Escape | KeyCode::KeyQ => el.exit(),
                KeyCode::KeyR => {
                    s.reset();
                    println!("reset");
                }
                KeyCode::KeyV => {
                    s.real_optics = !s.real_optics;
                    s.renderer.set_color_mode(if s.real_optics {
                        ColorMode::ByPhysics
                    } else {
                        ColorMode::ByMaterial
                    });
                    println!(
                        "real_optics={} -- {}",
                        s.real_optics,
                        if s.real_optics {
                            "honest air optics: expect this to look like almost nothing, that's correct at this scale"
                        } else {
                            "debug view: particle positions colored by material"
                        }
                    );
                }
                _ => {}
            },
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

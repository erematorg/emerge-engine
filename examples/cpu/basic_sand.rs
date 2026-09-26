extern crate emerge_engine as emerge;

#[path = "../gui_common/mod.rs"]
mod gui_common;

/// `basic_sand.rs` with a real, live egui panel (same wgpu-native egui
/// already used by `rod_blade_and_root.rs`/`material_sandbox_gpu`):
/// same push/pull cursor interaction as every other sand example (LMB push,
/// RMB pull, `apply_radial_impulse`), POURING (holding P spawns a small
/// trickle of new sand particles at the cursor via `Simulation::add_body`),
/// and a real, live GRAVITY slider (1.0 = genuine IRL 9.81 m/s², via
/// `Simulation::set_gravity` -- both already existed in the engine).
///
/// Real, disclosed limit: the renderer's instance buffer is sized with a
/// fixed extra headroom (`POUR_BUDGET`) at startup (wgpu buffers don't
/// resize live) -- pouring stops once that budget is spent, not a silent
/// overflow.
///
/// DIGGING (D toggles): a directional cursor drag -- nudges nearby particles
/// along the cursor's OWN movement direction, not radially like push/pull.
/// No second body, no mass-ratio tuning: mass-conserving by construction
/// (see MEMORY.md's ecosystem-roadmap note for the two rejected alternatives
/// -- particle deletion, and a kinematic "shovel" body that broke under real
/// gravity).
///
///   cargo run --example basic_sand --features render
// `prelude::*` is the documented single-import entry point (covers
// SimConfig/Simulation/SpawnRegion/every material/boundary, plus glam's
// IVec2/Mat2/Vec2 so callers don't need a separate glam dependency) --
// `render` types are the one deliberate exception (feature-gated, not part
// of the prelude's own promise), so that's still its own explicit import.
use emerge::prelude::*;
use emerge::render::{ColorMode, Renderer};
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

const GRID: usize = 64;
const DT: f32 = 0.1;
const MAT_LOOSE: u32 = 0;
const MAT_DENSE: u32 = 1;
const SIGMA_SAND: [f32; 3] = [0.180, 0.220, 0.550];
// Real, disclosed cap on how much new sand pouring can add beyond the
// initial ~2016 particles -- the renderer's wgpu instance buffer is
// allocated once at startup, not resizable live.
const POUR_BUDGET: usize = 2000;
// Real particles added per frame while the pour key is held -- a small
// SpawnRegion, not a single point, so a real (tiny) cone/stream forms
// instead of a perfectly straight line of particles.
const POUR_SPACING: f32 = 0.5;
const POUR_BOX: IVec2 = IVec2::new(2, 1);
// Radius of the directional dig nudge, grid cells.
const DIG_RADIUS: f32 = 4.0;

// Real fix (2026-09-05): was `DruckerPragerMaterial::new(2000.0, 3000.0, ..)`,
// an unsourced grid-unit guess. Real dry sand -- same real citation already
// used and verified tonight for `sand_ngf_collapse.rs` (Haeri & Skonieczny
// 2022 Table 1, Excavation case: E=15 MPa, nu=0.3, rho=1600 kg/m3) -- through
// the dt^2-free `lame_from_si_physical_cfg`. Loose/dense differ only by
// their real friction angle (20 deg loose, 40 deg dense -- both inside the
// real geotechnical range for sand packing states), not by stiffness.
const SAND_YOUNG_MODULUS_PA: f32 = 15.0e6;
const SAND_POISSON_RATIO: f32 = 0.3;
const SAND_DENSITY_KG_M3: f32 = 1600.0;

fn make_sand(lambda: f32, mu: f32, phi_deg: f32) -> DruckerPragerMaterial {
    let mut m = DruckerPragerMaterial::new(lambda, mu);
    m.friction_angle = phi_deg.to_radians();
    m
}

fn make_sim() -> Simulation {
    let config = SimConfig {
        boundary_thickness: 3,
        // Real fix (2026-09-05): the real E=15 MPa stiffness above needs
        // real substep headroom under CFL -- the old 12 silently dropped
        // simulated time instead of crashing (see `step.rs`'s "honest
        // accounting" doc). Measured directly at real full gravity
        // (`tests/scratch_basic_sand_probe.rs`): 2000 still dropped ~11.6%
        // of each step's simulated time; the solver actually uses 2263 once
        // given enough headroom, so 3000 leaves real margin, confirmed
        // zero time dropped.
        max_substeps_per_step: 3000,
        // No gravity override here -- `earth()`'s own real, correctly-converted
        // IRL gravity (9.81 m/s² / dx_meters) stands, exposed live via the
        // GUI's gravity slider below (see `State::real_gravity`/`gravity_fraction`).
        //
        // Same already-validated CFL margin as basic_sand.rs.
        material_cfl_coefficient: 0.7,
        ..SimConfig::earth(GRID, 0.01, DT)
    };
    let (lambda, mu) = config.lame_from_si_physical_cfg(
        SAND_YOUNG_MODULUS_PA,
        SAND_POISSON_RATIO,
        SAND_DENSITY_KG_M3,
    );
    // Real fix (2026-09-05): mass must share the same real density as the
    // stiffness above (see project memory on the grid_density/mass-from
    // gap found migrating basic_membrane.rs/sand_ngf_collapse.rs/
    // basic_jellies.rs the same night) -- was left on the bare
    // `grid_density=1.0` default, computed directly via
    // `ParticleMass::particle_mass`'s own documented formula since the raw
    // `DruckerPragerMaterial::new` constructor bypasses `mass_from`.
    let mass_grid = (SAND_DENSITY_KG_M3 / config.reference_density_kg_m3) * 0.5 * 0.5;
    let spawn = |c: Vec2, mat, seed| SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new(18, 14),
        box_center: c,
        material_id: mat,
        precompute_initial_volumes: true,
        initial_velocity_scale: 0.0,
        rng_seed: seed,
        position_jitter: 0.5,
        mass_override: Some(mass_grid),
        ..SpawnRegion::for_sim(&config)
    };
    let mut solver = Simulation::new(config, spawn(Vec2::new(17.0, 40.0), MAT_LOOSE, 11))
        .with_default_material(Box::new(make_sand(lambda, mu, 20.0)))
        .with_material(MAT_DENSE, Box::new(make_sand(lambda, mu, 40.0)))
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));
    let _ = solver.add_body(spawn(Vec2::new(47.0, 40.0), MAT_DENSE, 22));
    solver
}

struct State {
    gfx: gui_common::Gfx,
    sim: Simulation,
    renderer: Renderer,
    cursor_pos: [f32; 2],
    lmb: bool,
    rmb: bool,
    pouring: bool,
    pour_dense: bool,
    poured_count: usize,
    push_strength: f32,
    digging: bool,
    dig_strength: f32,
    last_cursor_grid: Vec2,
    // Real IRL gravity (9.81 m/s², converted via `earth()`'s own real
    // dx_meters-based formula) captured once at construction -- the slider
    // scales THIS real value, so 1.0 always means genuinely real gravity,
    // not an arbitrary tuned number.
    real_gravity: Vec2,
    gravity_fraction: f32,
    frame: u64,
    fps_timer: std::time::Instant,
    fps_frames: u64,
    last_fps: f32,
    pour_seed: u32,
}

impl State {
    async fn new(window: Arc<Window>) -> Self {
        let gfx = gui_common::Gfx::new(&window).await;
        let size = window.inner_size();
        let sim = make_sim();
        // Real IRL gravity, captured before anything ever overrides it --
        // `earth()`'s own real conversion, not a tuned constant.
        let real_gravity = sim.config().gravity;
        // Real extra headroom for pouring -- see POUR_BUDGET's own doc.
        let render_capacity = sim.particles().len() + POUR_BUDGET;
        let mut renderer = Renderer::new(&gfx.device, render_capacity, gfx.format);
        // particle_scale=0.9, not the usual 0.6: particles are seeded at
        // spacing=0.5 with position_jitter=0.5 (see make_sim/pour), so a 0.6
        // disc leaves real visible gaps wherever jitter spreads two
        // neighbors apart -- 0.9 keeps discs comfortably overlapping without
        // reaching 1.0 (full-cell, where distinct grains would start
        // visually fusing into unbroken blobs). This is a per-scene render
        // tuning fix, not the deeper "particles vs. a real reconstructed
        // surface" question -- that's the curvature-flow work already
        // planned separately (see render-pipeline-plan memory).
        renderer.set_camera(&gfx.queue, GRID as u32, size.width, size.height, 0.9, true);
        renderer.set_color_mode(ColorMode::ByPhysics);
        renderer.set_optical_params(&gfx.queue, MAT_LOOSE as usize, SIGMA_SAND);
        renderer.set_optical_params(&gfx.queue, MAT_DENSE as usize, SIGMA_SAND);

        println!(
            "basic_sand: {} particles  |  LMB push  RMB pull  D toggle dig  hold P to pour  R reset  Q quit",
            sim.particles().len()
        );
        Self {
            gfx,
            sim,
            renderer,
            cursor_pos: [0.0; 2],
            lmb: false,
            rmb: false,
            pouring: false,
            pour_dense: false,
            poured_count: 0,
            push_strength: 12.0,
            digging: false,
            dig_strength: 18.0,
            last_cursor_grid: Vec2::ZERO,
            real_gravity,
            // 0.001 default: full IRL gravity (1.0) is numerically stable here
            // (no crash/corruption) but reads as violently fast free-fall at
            // this small grid scale -- 0.001 is comfortable, still real
            // Newtonian gravity (F=mg), just a smaller magnitude. Slider still
            // reaches 1.0 for full IRL.
            gravity_fraction: 0.001,
            frame: 0,
            fps_timer: std::time::Instant::now(),
            fps_frames: 0,
            last_fps: 0.0,
            pour_seed: 1000,
        }
    }

    fn resize(&mut self, w: u32, h: u32) {
        self.gfx.resize(w, h);
        if w == 0 || h == 0 {
            return;
        }
        self.renderer
            .set_camera(&self.gfx.queue, GRID as u32, w, h, 0.9, true);
    }

    fn cursor_grid(&self) -> Vec2 {
        gui_common::cursor_to_grid(
            self.cursor_pos,
            self.gfx.surface_config.width,
            self.gfx.surface_config.height,
            GRID,
        )
    }

    fn update_and_render(&mut self, window: &Window) {
        // Real, live gravity control -- `gravity_fraction=1.0` is genuine
        // IRL gravity (`real_gravity`, captured from `earth()`'s own real
        // conversion), not an arbitrary tuned constant. `Simulation::
        // set_gravity` already existed in the engine (lifecycle.rs) -- no
        // new engine code needed, just wiring.
        self.sim
            .set_gravity(self.real_gravity * self.gravity_fraction);
        if self.lmb || self.rmb {
            let mag = if self.lmb {
                self.push_strength
            } else {
                -self.push_strength
            };
            self.sim.apply_radial_impulse(self.cursor_grid(), 7.0, mag);
        }
        // Digging: nudges nearby particles along the cursor's OWN movement
        // direction (a furrow), not radially like push/pull -- direct
        // per-particle velocity nudge via SoA access, no second body, no
        // impulse call, so it never collapses into "just push again."
        let cursor = self.cursor_grid();
        if self.digging {
            let delta = cursor - self.last_cursor_grid;
            if delta.length_squared() > 1.0e-8 {
                let dir = delta.normalize();
                let particles = self.sim.particles_mut();
                for i in 0..particles.len() {
                    if (particles.x[i] - cursor).length() < DIG_RADIUS {
                        particles.v[i] += dir * self.dig_strength * DT;
                    }
                }
            }
        }
        self.last_cursor_grid = cursor;
        // Real pour tool: `Simulation::add_body` is the same mid-run
        // body-spawning API the engine already offers elsewhere -- a small
        // SpawnRegion dropped at the cursor each frame while held, capped by
        // POUR_BUDGET so the (fixed-size) render buffer never overflows.
        if self.pouring && self.poured_count < POUR_BUDGET {
            // Real, found-live bug (2026-08-04): pouring with the cursor near
            // the window edge maps to a grid position close enough to the
            // domain boundary that `POUR_BOX` no longer fits inside the
            // spawnable region -- `add_body` then hits `validate_for_sim`'s
            // own real assert and hard-panics the whole demo instead of just
            // declining that frame's pour. Clamp the pour center to the same
            // real bound `fits_in_sim` checks (`boundary_thickness` margin
            // plus half the pour box on each axis) so pouring at the edge
            // just pours as close to the wall as actually fits, not a crash.
            let config = self.sim.config();
            let half = POUR_BOX.as_vec2() * 0.5;
            let domain_min = Vec2::splat(config.boundary_thickness as f32) + half;
            let domain_max =
                Vec2::splat((config.grid_res - config.boundary_thickness) as f32) - half;
            let cursor = self
                .cursor_grid()
                .clamp(domain_min, domain_max.max(domain_min));
            self.pour_seed += 1;
            let mat = if self.pour_dense {
                MAT_DENSE
            } else {
                MAT_LOOSE
            };
            // Real fix (2026-09-05): poured particles must get the same real
            // mass as the initial pile (see `make_sim`'s own note) -- was
            // falling back to the bare `grid_density=1.0` default, silently
            // pouring sand ~1.6x too light relative to the pile it lands on.
            let pour_mass =
                (SAND_DENSITY_KG_M3 / config.reference_density_kg_m3) * POUR_SPACING * POUR_SPACING;
            let spawn = SpawnRegion {
                spacing: POUR_SPACING,
                box_size: POUR_BOX,
                box_center: cursor,
                material_id: mat,
                precompute_initial_volumes: true,
                initial_velocity_scale: 0.0,
                rng_seed: self.pour_seed,
                position_jitter: 0.3,
                mass_override: Some(pour_mass),
                ..SpawnRegion::for_sim(self.sim.config())
            };
            let before = self.sim.particles().len();
            // TEMP diagnostic (2026-08-05, user-flagged pour-vs-default gap
            // investigation) -- ground-truth the real gap in grid units via
            // stdout instead of trusting a screenshot alone (this project has
            // a documented PrintWindow false-positive on a similar demo).
            // Printed BEFORE add_body so `existing_max_y` excludes this
            // frame's own new particles.
            if self.frame.is_multiple_of(15) {
                let existing_max_y = self
                    .sim
                    .particles()
                    .iter()
                    .map(|p| p.x.y)
                    .fold(f32::MIN, f32::max);
                println!(
                    "POUR_DIAG frame={} cursor_y={:.2} existing_pile_max_y={:.2} gap={:.2}",
                    self.frame,
                    cursor.y,
                    existing_max_y,
                    cursor.y - existing_max_y
                );
            }
            let _ = self.sim.add_body(spawn);
            self.poured_count += self.sim.particles().len() - before;
        }

        self.sim.step();
        self.frame += 1;
        self.fps_frames += 1;
        if self.fps_timer.elapsed().as_secs_f32() >= 1.0 {
            self.last_fps = self.fps_frames as f32 / self.fps_timer.elapsed().as_secs_f32();
            self.fps_timer = std::time::Instant::now();
            self.fps_frames = 0;
            println!(
                "frame={} fps={:.1} particles={}",
                self.frame,
                self.last_fps,
                self.sim.particles().len()
            );
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

        // --- egui panel ---
        let fps = self.last_fps;
        let mut push_strength = self.push_strength;
        let mut pour_dense = self.pour_dense;
        let mut gravity_fraction = self.gravity_fraction;
        let mut digging = self.digging;
        let mut dig_strength = self.dig_strength;
        let n_particles = self.sim.particles().len();
        let poured = self.poured_count;
        let mut reset = false;

        gui_common::run_egui_frame(&mut self.gfx, window, &view, |ctx| {
            egui::Window::new("Sand")
                .default_pos([10.0, 10.0])
                .default_width(240.0)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.label(format!("fps={fps:.0}  particles={n_particles}"));
                    ui.separator();
                    ui.label("Gravity (1.0 = real IRL 9.81 m/s²):");
                    ui.add(egui::Slider::new(&mut gravity_fraction, 0.0..=2.0));
                    ui.separator();
                    ui.label("Push/pull strength:");
                    ui.add(egui::Slider::new(&mut push_strength, 0.0..=40.0));
                    ui.separator();
                    ui.checkbox(&mut digging, "Digging active (or press D)");
                    ui.add(egui::Slider::new(&mut dig_strength, 0.0..=40.0).text("Dig strength"));
                    ui.separator();
                    ui.checkbox(&mut pour_dense, "Pour dense sand (unchecked = loose)");
                    ui.label(format!("Poured: {poured}/{POUR_BUDGET}"));
                    ui.add(
                        egui::ProgressBar::new(poured as f32 / POUR_BUDGET as f32)
                            .desired_width(200.0),
                    );
                    ui.separator();
                    ui.label("LMB push  RMB pull  D toggle dig  hold P to pour  R reset  Q quit");
                    if ui.button("Reset").clicked() {
                        reset = true;
                    }
                });
        });
        self.push_strength = push_strength;
        self.pour_dense = pour_dense;
        self.gravity_fraction = gravity_fraction;
        self.digging = digging;
        self.dig_strength = dig_strength;
        if reset {
            let sim = make_sim();
            self.real_gravity = sim.config().gravity;
            self.sim = sim;
            self.frame = 0;
            self.poured_count = 0;
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
                    .with_title("emerge -- Sand (GUI)")
                    .with_inner_size(winit::dpi::LogicalSize::new(480u32, 480u32)),
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
                    KeyCode::KeyP => s.pouring = pressed,
                    KeyCode::KeyD if pressed => s.digging = !s.digging,
                    KeyCode::Escape | KeyCode::KeyQ if pressed => el.exit(),
                    KeyCode::KeyR if pressed => {
                        let sim = make_sim();
                        s.real_gravity = sim.config().gravity;
                        s.sim = sim;
                        s.frame = 0;
                        s.poured_count = 0;
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

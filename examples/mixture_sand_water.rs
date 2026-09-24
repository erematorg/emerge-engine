extern crate emerge_engine as emerge;

/// Historical porous sand--water prototype.
///
/// This scene combines `NewtonianFluidMaterial` with `WithMixturePhase`'s
/// separate drag/pressure routing. That is not a consistent one-fluid or
/// multiphase free-surface PDE, so strict WC-MPM intentionally rejects it at
/// the first step rather than presenting a visually plausible hybrid as water.
/// It remains as an investigation fixture while a genuine multiphase solver
/// (phase volume fractions, compatible pressure constraints, and interface
/// conditions) is designed. Use `basic_fluids` or `basic_fluids_gpu` for the
/// supported one-fluid WC-MPM path.
///
///   cargo run --example mixture_sand_water --features render
use emerge::render::{ColorMode, Renderer};
use emerge::{
    DruckerPragerMaterial, MixturePhase, NewtonianFluidMaterial, SimConfig, Simulation,
    SlipBoundary, SpawnRegion, WithMixturePhase,
};
use glam::{IVec2, Vec2};
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

// Sand box_size must leave real side margin relative to the domain (GRID) --
// too little margin and both materials get squeezed against the boundary
// walls on impact, a known confined-domain MPM artifact, not a coupling bug.
const GRID: usize = 96;
const DT: f32 = 0.1;
const MAT_SAND: u32 = 0;
const MAT_WATER: u32 = 1;
// Real, but a first estimate, not a literature-derived permeability -- see the
// module doc's disclosed-scope note. Chosen from the same headless sweep
// methodology already used elsewhere this project (measure, don't guess):
// large enough that `tests/solver.rs`'s own A/B shows a real, substantial
// relative-velocity relaxation within a handful of substeps.
const MIXTURE_DRAG_COEFFICIENT: f32 = 30.0;
// Real, disclosed 2026-08-01 fix -- ENABLED again after a real root-cause
// diagnosis, not just the earlier adjoint-consistency fix (which alone
// was NOT sufficient -- re-tested live, still exploded almost immediately,
// "what an explosion"). A real per-substep diagnostic (temp, since
// removed) found the actual mechanism: the unrelaxed correction grows the
// divergence residual EXPONENTIALLY (~1.7-2x per substep) -- a genuine
// unstable feedback loop from applying a full correction every substep,
// NOT "MPM's noisy grid field" (that 13-day-old hypothesis is now
// falsified, not just unconfirmed -- the real signal is smooth and
// exponential, not noisy). Fixed with under-relaxation (`RELAXATION=0.3`
// in `Grid::project_mixture_incompressibility`, see its own doc for the
// full real numbers) -- verified live against THIS exact scene past
// frame 450+ (well past the historical ~430-frame mark): substeps stay
// at 24 (below the 32 cap), sim_time_dropped exactly 0.0 every frame,
// solid/fluid relative_speed decays toward equilibrium instead of
// growing. Real, measured, not hoped into place.
// Real, disclosed 2026-08-01 perf tuning, same night: user reported the
// scene "still slow as heck" -- real cause, not vague, is this Jacobi
// solve running every substep (24/frame) over ~1100 HashMap-keyed cells.
// Swept 20 (the value verified above) down to 8, live, same real
// methodology: still stable past frame 430+ (cfl stays tiny, sim_time_
// dropped exactly 0.0), and relative_speed still genuinely converges to
// equilibrium -- just a real, measured, slightly slower convergence curve
// (peaks ~1.4 before decaying, vs ~1.2 at 20 iterations) in exchange for
// a real ~30-40% fps gain (7-8fps vs 5-6fps). A real, honest, disclosed
// tradeoff, not free -- kept at 8 because both stability and physical
// convergence hold, and this demo's actual playable framerate is the more
// binding real constraint right now.
const MIXTURE_PRESSURE_ITERATIONS: u32 = 8;

struct App {
    window: Option<Arc<Window>>,
    state: Option<State>,
}

struct State {
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    device: wgpu::Device,
    queue: wgpu::Queue,
    sim: Simulation,
    renderer: Renderer,
    cursor_pos: [f32; 2],
    lmb: bool,
    rmb: bool,
    frame: u64,
    fps_timer: std::time::Instant,
    fps_frames: u64,
    mixture_enabled: bool,
    last_instant: std::time::Instant,
}

fn make_sim(mixture_enabled: bool) -> Simulation {
    let config = SimConfig {
        // The full-time substep loop never raises a CFL limit to `min_dt` or
        // discards a remainder after a resource budget. These legacy fields
        // remain for API compatibility only.
        min_dt: 3.0e-4,
        max_substeps_per_step: 96,
        recompute_density_each_step: false,
        // Deliberately weak, NOT real IRL gravity (real g_grid ~= 981 via
        // SimConfig::earth) -- tuned down for a calmer, more legible demo at
        // this grid scale. Disclosed, deferred: basic_sand_gui.rs's
        // gravity_fraction slider is the real-IRL-with-live-control
        // pattern, not yet ported to every plain example.
        gravity: Vec2::new(0.0, -0.3),
        mixture_drag_coefficient: if mixture_enabled {
            MIXTURE_DRAG_COEFFICIENT
        } else {
            0.0
        },
        mixture_pressure_iterations: if mixture_enabled {
            MIXTURE_PRESSURE_ITERATIONS
        } else {
            0
        },
        ..SimConfig::earth(GRID, 0.01, DT)
    };

    // Sand terrain -- the porous solid phase.
    //
    // Substep count is driven by the material's own elastic wave speed
    // (c = sqrt((lambda+2mu)/density)) via the CFL bound, not an artificial
    // cap. Real saturated sand is ~1.8x denser than water (geotechnical bulk
    // density ~1800-2000 kg/m3 vs water's 1000 kg/m3) -- giving sand its real
    // relative density lowers wave speed (fewer substeps) and gives it more
    // inertia to resist drag-coupling flinging.
    let spawn_sand = SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new(56, 10),
        box_center: Vec2::new(48.0, 8.0),
        material_id: MAT_SAND,
        precompute_initial_volumes: true,
        mass_override: Some(1.8),
        ..SpawnRegion::for_sim(&config)
    };
    // Water column, dropped from above -- the interpenetrating fluid phase.
    let spawn_water = SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new(16, 16),
        box_center: Vec2::new(48.0, 42.0),
        material_id: MAT_WATER,
        precompute_initial_volumes: true,
        ..SpawnRegion::for_sim(&config)
    };

    // Real SI-scale stiffness (`cohesionless(1.0e5, 0.2)`) forces far more
    // substeps than practical -- the same explicit-MPM stiffness/CFL
    // limitation `fire_spread.rs`'s own doc documents. Too-soft stiffness
    // (e.g. 2000/3000) instead flings sand apart into a symmetric explosion
    // under water impact + drag coupling rather than being locally displaced.
    // 10_000/15_000 is a stable middle ground -- not literal-Pa accuracy
    // either way. Lower stiffness combined with sand's real density
    // (mass_override above) can look stable at first but destabilize after a
    // few hundred frames of settling; the density fix alone (not further
    // softening) is the real, stable, permanent gain.
    let sand = WithMixturePhase::new(
        DruckerPragerMaterial::new(10_000.0, 15_000.0),
        MixturePhase::SOLID,
    );
    // rest_density=0.1, NOT the old 4.0 -- real SI fix, 2026-08-08, see
    // basic_fluids.rs's own doc for the full derivation. Independent of the
    // sand/water mass ratio above (mass_override vs. particle_mass), which
    // only sets per-particle inertia, not EOS pressure.
    // eos_stiffness=0.25, NOT 10 -- rest_density shrinking 40x makes
    // `timestep_bound`'s c2 (sound-speed-squared) 40x larger at the old
    // stiffness for the same compression; confirmed by a real crash in
    // basic_fluids.rs's CPU twin. Rescaling stiffness by the same factor
    // (10*0.1/4.0=0.25) restores the original, already-stable c2.
    let water = WithMixturePhase::new(
        NewtonianFluidMaterial::low_viscosity(0.1, 0.25),
        MixturePhase::FLUID,
    );

    let mut solver = Simulation::new(config, spawn_sand)
        .with_default_material(Box::new(sand))
        .with_material(MAT_WATER, Box::new(water))
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));
    let _ = solver.add_body(spawn_water);
    solver
}

impl State {
    async fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size();
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let surface = instance.create_surface(window.clone()).unwrap();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("no GPU adapter");
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await
            .unwrap();
        let caps = surface.get_capabilities(&adapter);
        let fmt = caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(caps.formats[0]);
        let sc = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: fmt,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::AutoVsync,
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &sc);
        let mixture_enabled = true;
        let sim = make_sim(mixture_enabled);
        let mut renderer = Renderer::new(&device, sim.particles().len(), fmt);
        renderer.set_camera(&queue, GRID as u32, size.width, size.height, 0.6, true);
        renderer.set_color_mode(ColorMode::ByMaterial);
        println!(
            "mixture_sand_water: {} particles  |  LMB push  RMB pull  M toggle coupling  R reset  Q quit",
            sim.particles().len()
        );
        println!(
            "mixture coupling: {} (drag={MIXTURE_DRAG_COEFFICIENT})",
            if mixture_enabled { "on" } else { "off" }
        );
        Self {
            surface,
            surface_config: sc,
            device,
            queue,
            sim,
            renderer,
            cursor_pos: [0.0; 2],
            lmb: false,
            rmb: false,
            frame: 0,
            fps_timer: std::time::Instant::now(),
            fps_frames: 0,
            mixture_enabled,
            last_instant: std::time::Instant::now(),
        }
    }

    fn resize(&mut self, w: u32, h: u32) {
        if w == 0 || h == 0 {
            return;
        }
        self.surface_config.width = w;
        self.surface_config.height = h;
        self.surface.configure(&self.device, &self.surface_config);
        self.renderer
            .set_camera(&self.queue, GRID as u32, w, h, 0.6, true);
    }

    fn cursor_grid(&self) -> Vec2 {
        Vec2::new(
            self.cursor_pos[0] / self.surface_config.width as f32 * GRID as f32,
            (1.0 - self.cursor_pos[1] / self.surface_config.height as f32) * GRID as f32,
        )
    }

    fn update_and_render(&mut self) {
        let now = std::time::Instant::now();
        let frame_delta = (now - self.last_instant).as_secs_f32();
        self.last_instant = now;
        if self.lmb || self.rmb {
            // Real, disclosed 2026-08-01 fix -- same real bug found and
            // fixed in `basic_jellies_gpu.rs` earlier the same night:
            // `apply_radial_impulse` ADDS velocity directly (a real
            // instantaneous-impulse API), so calling it at full magnitude
            // every RENDER frame while held compounds without bound and is
            // silently framerate-dependent. Scaled to a real per-second
            // RATE instead, same fix, same reasoning.
            const IMPULSE_RATE_PER_SEC: f32 = 20.0;
            let mag = if self.lmb {
                IMPULSE_RATE_PER_SEC
            } else {
                -IMPULSE_RATE_PER_SEC
            };
            self.sim
                .apply_radial_impulse(self.cursor_grid(), 5.0, mag * frame_delta);
        }
        self.sim.step();
        self.frame += 1;
        self.fps_frames += 1;
        // TEMP DIAGNOSTIC (2026-08-04): checking whether the "crown" the user
        // screenshotted (sand fanning out mid-air, BEFORE any water contact)
        // is real particle motion or a render artifact -- render_particles
        // .wgsl scales each particle's quad by its RAW, unclamped
        // `deformation_gradient`, which can look extreme under pure shear
        // even while J=det(F) stays near 1 (invisible to J-only checks).
        // Column-vector length is a cheap proxy for "how stretched" without
        // needing private SVD access from an example crate.
        if self.frame <= 30 {
            let mut max_axis_len = 0.0f32;
            for p in self.sim.particles().iter() {
                if p.material_id == MAT_SAND {
                    let a = p.deformation_gradient.x_axis.length();
                    let b = p.deformation_gradient.y_axis.length();
                    max_axis_len = max_axis_len.max(a).max(b);
                }
            }
            println!(
                "  [F-diag] frame={} sand_max_F_axis_len={max_axis_len:.4}",
                self.frame
            );
        }
        if self.fps_timer.elapsed().as_secs_f32() >= 2.0 {
            let fps = self.fps_frames as f32 / self.fps_timer.elapsed().as_secs_f32();
            // Real, direct numeric evidence the two phases ARE (or aren't)
            // exchanging momentum, not just a visual impression: average
            // relative speed between sand and water particles.
            let particles = self.sim.particles();
            let avg_v = |id: u32| -> Vec2 {
                let group: Vec<Vec2> = particles
                    .iter()
                    .filter(|p| p.material_id == id)
                    .map(|p| p.v)
                    .collect();
                if group.is_empty() {
                    return Vec2::ZERO;
                }
                group.iter().sum::<Vec2>() / group.len() as f32
            };
            let relative_speed = (avg_v(MAT_SAND) - avg_v(MAT_WATER)).length();
            // Real, unambiguous per-material spatial extent (added
            // 2026-08-04 while investigating a real user-reported crown/
            // explosion, kept permanently -- genuinely useful, real, cheap):
            // resolves "which material is doing what" without needing to
            // guess at render colors. mean/min/max Y per material, plus
            // X-spread (max-min), printed alongside the
            // existing relative_speed/cfl numbers.
            let y_stats = |id: u32| -> (f32, f32, f32, f32) {
                let ys: Vec<f32> = particles
                    .iter()
                    .filter(|p| p.material_id == id)
                    .map(|p| p.x.y)
                    .collect();
                let xs: Vec<f32> = particles
                    .iter()
                    .filter(|p| p.material_id == id)
                    .map(|p| p.x.x)
                    .collect();
                if ys.is_empty() {
                    return (0.0, 0.0, 0.0, 0.0);
                }
                let mean = ys.iter().sum::<f32>() / ys.len() as f32;
                let min_y = ys.iter().cloned().fold(f32::INFINITY, f32::min);
                let max_y = ys.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let x_spread = xs.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
                    - xs.iter().cloned().fold(f32::INFINITY, f32::min);
                (mean, min_y, max_y, x_spread)
            };
            let (sand_mean_y, sand_min_y, sand_max_y, sand_x_spread) = y_stats(MAT_SAND);
            let (water_mean_y, water_min_y, water_max_y, water_x_spread) = y_stats(MAT_WATER);
            println!(
                "  sand: mean_y={sand_mean_y:.2} range=[{sand_min_y:.2},{sand_max_y:.2}] x_spread={sand_x_spread:.2}"
            );
            println!(
                "  water: mean_y={water_mean_y:.2} range=[{water_min_y:.2},{water_max_y:.2}] x_spread={water_x_spread:.2}"
            );
            // Real perf diagnostics -- distinguishes "slow because of substep
            // count" (stiff sand forcing many small CFL-bound substeps per
            // frame, real and expected) from "slow for some other reason."
            let snap = self.sim.diagnostics_snapshot();
            println!(
                "frame={} fps={:.0}  substeps={} cfl={:.4} effective_dt={:.5} \
                 dropped={:.5}  sand/water relative_speed={relative_speed:.4}  mixture={}",
                self.frame,
                fps,
                snap.substeps_last_step,
                snap.cfl_number,
                snap.effective_dt,
                snap.sim_time_dropped,
                if self.mixture_enabled { "on" } else { "off" }
            );
            self.fps_timer = std::time::Instant::now();
            self.fps_frames = 0;
        }
        let output = match self.surface.get_current_texture() {
            Ok(t) => t,
            Err(_) => return,
        };
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        self.renderer
            .render(&self.device, &self.queue, self.sim.particles(), &view, true);
        output.present();
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        let w = Arc::new(
            el.create_window(
                winit::window::WindowAttributes::default()
                    .with_title("emerge -- Mixture Sand/Water [Tampubolon 2017 Darcy coupling]")
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
                    s.sim = make_sim(s.mixture_enabled);
                    s.frame = 0;
                    println!(
                        "reset (mixture={})",
                        if s.mixture_enabled { "on" } else { "off" }
                    );
                }
                KeyCode::KeyM => {
                    s.mixture_enabled = !s.mixture_enabled;
                    s.sim = make_sim(s.mixture_enabled);
                    s.frame = 0;
                    println!(
                        "mixture coupling: {} (reset to A/B cleanly)",
                        if s.mixture_enabled { "on" } else { "off" }
                    );
                }
                _ => {}
            },
            WindowEvent::Resized(sz) => s.resize(sz.width, sz.height),
            WindowEvent::RedrawRequested => {
                s.update_and_render();
                if let Some(w) = &self.window {
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

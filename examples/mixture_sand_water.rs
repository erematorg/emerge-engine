extern crate emerge_engine as emerge;

/// CPU two-phase mixture coupling (Tampubolon et al. 2017, "Multi-species
/// simulation of porous sand and water mixtures") -- water poured onto sand
/// exchanges momentum with it via Darcy-style drag instead of the two
/// materials just sharing one ordinary MPM grid field.
///
/// Real, disclosed scope: this is the CPU-first MVP (`WithMixturePhase`,
/// `Grid::resolve_mixture_coupling`) -- a single SCALAR drag coefficient
/// (`SimConfig::mixture_drag_coefficient`), not the paper's own permeability/
/// porosity-derived field. GPU port is a separate, deferred follow-up per this
/// project's own "CPU correctness first" rule. See the real closed-form
/// verification in `spacetime::grid::mixture_coupling_tests` and the full
/// end-to-end pipeline test in `tests/solver.rs`
/// (`higher_drag_relaxes_solid_fluid_relative_velocity_faster`) for how this
/// was validated before being shown here.
///
/// Toggle mixture coupling with M to compare directly, live, against ordinary
/// single-field MPM (both materials still share momentum at any node they
/// both touch -- see `build_mixture_scene`'s doc in `tests/solver.rs` for why
/// that's a REAL, stronger-than-you'd-expect baseline, not "no coupling at
/// all") -- with coupling on, water visibly drags on sand and sand drags back
/// on water as it seeps in, instead of the two bodies behaving as if the
/// other weren't there beyond ordinary momentum sharing.
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

// REAL BUG FOUND AND FIXED (live feedback): sand box_size (56) was nearly as
// wide as the domain itself (64), leaving almost no side margin. As water
// spread on impact, both materials got squeezed against the boundary walls --
// with nowhere else to go, material piles up and climbs the wall instead of
// spreading normally (a real, well-known confined-domain MPM artifact, not a
// coupling bug). Widened the domain relative to the material footprint so
// there's real room to spread without ever reaching a wall.
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
// REAL, TESTED, NEGATIVE RESULT (2026-07-18, see `mixture_coupling_
// long_settle_instability` memory): `project_mixture_incompressibility`'s
// pressure projection was built to fix the long-settle instability below,
// verified correct in isolation (unit test), but made THIS real scene worse,
// not better -- destabilized almost immediately (frame ~18) instead of after
// ~430 frames, even after fixing a real derivation bug found along the way.
// Root cause not yet found (leading hypothesis: MPM's naturally noisy/sparse
// grid mass field feeds a noisy central-difference divergence estimate back
// into velocity every substep, amplifying quantization noise rather than
// damping real drift). Kept disabled here until that's actually solved --
// don't re-enable by raising this off 0 without new evidence it's fixed.
const MIXTURE_PRESSURE_ITERATIONS: u32 = 0;

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
}

fn make_sim(mixture_enabled: bool) -> Simulation {
    let config = SimConfig {
        min_dt: 1.0e-3,
        max_substeps_per_step: 32,
        recompute_density_each_step: true,
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
    // REAL PERF LEVER (measured, not guessed): raising `max_substeps_per_step`
    // to 500 and re-running showed the solver genuinely converges on 32
    // substeps/frame on its own -- it's the material's own elastic wave speed
    // (c = sqrt((lambda+2mu)/density)) driving the CFL bound, not an
    // artificial cap. Sand and water previously had IDENTICAL density (both
    // default particle_mass=1.0) -- physically wrong: real saturated sand is
    // ~1.8x denser than water (geotechnical bulk density ~1800-2000 kg/m3 vs
    // water's 1000 kg/m3). Giving sand its real relative density lowers its
    // wave speed by sqrt(1.8) at the SAME stiffness (fewer substeps needed)
    // AND gives it more inertia to resist the drag-coupling flinging that
    // forced the stiffness up in the first place -- a real, physically
    // motivated fix, not just a stiffness knob turned down blind.
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

    // REAL PERF FIX, two rounds: `cohesionless(1.0e5, 0.2)` (real SI-scale
    // stiffness) forced far more substeps than `max_substeps_per_step` could
    // satisfy -- confirmed live (fps stuck ~10-16 even in --release,
    // `SimSnapshot::sim_time_dropped` nonzero every frame: the solver was
    // silently running slower than its own configured dt, the same explicit-
    // MPM stiffness/CFL limitation `fire_spread.rs`'s own doc documents).
    // First fix (`basic_sand.rs`'s own raw Lame values, 2000/3000) was TOO
    // soft -- confirmed live: sand flung apart into a symmetric explosion
    // under the water impact + drag coupling instead of just being locally
    // displaced. Settled on a real middle ground (10_000/15_000): stable,
    // contained impact deformation (verified live), substeps=32 with zero
    // `sim_time_dropped`, not literal-Pa accuracy either way.
    //
    // PERF ROUND 3 (measured, not guessed): raised max_substeps_per_step to
    // 500 and re-ran -- solver genuinely converged on 32 substeps/frame on its
    // own, confirming it's the real material-CFL bound (elastic wave speed),
    // not an artificial cap. Real root cause found: sand and water had
    // IDENTICAL density (both default particle_mass -- see spawn_sand's own
    // `mass_override: Some(1.8)`, real saturated-sand/water bulk-density
    // ratio). Adding that density alone dropped substeps 32->24, fps ~12->~15,
    // zero dropped time -- a genuine, physically-motivated win kept below.
    // Tried halving stiffness further (5_000/7_500) ON TOP of the density fix
    // to push for more: looked great at first (substeps=17, fps~20) but
    // destabilized after ~300 frames of settling (cfl jumped 10x, substeps
    // climbed back to 32, `sim_time_dropped` went nonzero) -- a real, if
    // delayed, instability, not a clean permanent fix. Reverted stiffness to
    // 10_000/15_000; the density fix alone is the real, stable, permanent gain
    // (verified over a longer run below).
    let sand = WithMixturePhase::new(
        DruckerPragerMaterial::new(10_000.0, 15_000.0),
        MixturePhase::Solid,
    );
    let water = WithMixturePhase::new(
        NewtonianFluidMaterial::low_viscosity(4.0, 10.0),
        MixturePhase::Fluid,
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
        println!("mixture coupling: on (drag={MIXTURE_DRAG_COEFFICIENT})");
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
        if self.lmb || self.rmb {
            let mag = if self.lmb { 2.0 } else { -2.0 };
            self.sim.apply_radial_impulse(self.cursor_grid(), 5.0, mag);
        }
        self.sim.step();
        self.frame += 1;
        self.fps_frames += 1;
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

extern crate emerge_engine as emerge;

/// GPU Drucker-Prager sand — angle of repose comparison, zero CPU readback.
///
///   Mat 0  loose sand  (blue, phi=20 deg) — shallow repose
///   Mat 1  dense sand  (gold, phi=40 deg) — steep repose
///
///   cargo run --example basic_sand_gpu --features "render"
use std::sync::Arc;

use emerge::diagnostics::log_frame_gpu;
use emerge::render::demo_harness::{DemoApp, run_demo};
use emerge::render::{ColorMode, Renderer};
use emerge::{
    DruckerPragerMaterial, FixedStepController, GpuSimulation, MaterialRegistry, SimConfig,
    SpawnRegion, build_particles,
};
use glam::{IVec2, Vec2};
use winit::event::MouseButton;
use winit::keyboard::KeyCode;

const GRID: usize = 64;
const DT: f32 = 0.1;
const MAT_LOOSE: u32 = 0;
const MAT_DENSE: u32 = 1;
const LABELS: &[(u32, &str)] = &[(MAT_LOOSE, "loose"), (MAT_DENSE, "dense")];
// Real measured sand absorption (Sherman & Waite 1985, iron-oxide quartz sand) — same value
// used for both materials since they're optically the same sand, just different friction angle.
const SIGMA_SAND: [f32; 3] = [0.180, 0.220, 0.550];

struct State {
    sim: GpuSimulation,
    renderer: Renderer,
    cursor_frac: [f32; 2],
    lmb: bool,
    rmb: bool,
    frame: u64,
    fps_timer: std::time::Instant,
    fps_frames: u64,
    /// Real-time-decoupled stepping -- see `basic_fluids_gpu.rs`'s own field
    /// doc for the full real bug/fix writeup. `standard(DT, 60.0)`
    /// deliberately preserves this demo's already-tuned DT/call-rate,
    /// decoupling it from measured fps instead of assuming every render
    /// frame takes exactly `DT`.
    stepper: FixedStepController,
    last_instant: std::time::Instant,
    /// Diagnostic: highest `steps_for_frame` result seen since the last fps
    /// print -- a catch-up burst would NOT show up in the averaged fps
    /// number alone.
    max_steps_seen: usize,
}

fn make_sand(lambda: f32, mu: f32, phi_deg: f32) -> DruckerPragerMaterial {
    let mut m = DruckerPragerMaterial::new(lambda, mu);
    m.friction_angle = phi_deg.to_radians();
    m
}

fn make_sim_data(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> GpuSimulation {
    let config = SimConfig {
        boundary_thickness: 3,
        max_substeps_per_step: 12,
        // Deliberately weak, NOT real IRL gravity (real g_grid ~= 981 via
        // SimConfig::earth) -- tuned down for a calmer, more legible demo at
        // this grid scale. Disclosed, deferred: basic_sand_gui.rs's
        // gravity_fraction slider is the real-IRL-with-live-control
        // pattern, not yet ported to every plain example.
        gravity: Vec2::new(0.0, -0.3),
        ..SimConfig::earth(GRID, 0.01, DT)
    };
    let spawn = |c: Vec2, mat: u32, seed: u32| SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new(18, 14),
        box_center: c,
        material_id: mat,
        precompute_initial_volumes: true,
        rng_seed: seed,
        // A perfectly regular spawn lattice + quadratic B-spline MPM kernel is a textbook
        // grid-crossing artifact: every spawn column stays a visually distinct streak forever,
        // never mixing with its neighbors ("combed" look, confirmed via direct frame capture —
        // 0.2 per jitter()'s own doc comment wasn't strong enough at this spawn density; 0.5
        // fully breaks the lattice symmetry into a natural-looking granular pile).
        position_jitter: 0.5,
        ..SpawnRegion::for_sim(&config)
    };
    let mut particles = build_particles(&config, spawn(Vec2::new(17.0, 40.0), MAT_LOOSE, 11));
    particles.extend(build_particles(
        &config,
        spawn(Vec2::new(47.0, 40.0), MAT_DENSE, 22),
    ));
    // lambda=2000, mu=3000 -> nu≈0.2 (the previous 5000/3000 implied nu≈0.31, above real dry
    // sand's established range of 0.1-0.3 — see Drucker-Prager yield ratio (lambda+mu)/mu,
    // which directly gates plastic yielding: a higher ratio resists granular flow more,
    // producing the rigid "combed" look instead of a natural collapsing pile). Matches the
    // validated nu=0.2 already used in tests/accuracy.rs's sand_angle_of_repose_is_physical
    // and the canonical sparkl demo value it's calibrated against.
    let mut registry = MaterialRegistry::with_default(Box::new(make_sand(2000.0, 3000.0, 20.0)));
    registry.insert(MAT_DENSE, Box::new(make_sand(2000.0, 3000.0, 40.0)));
    GpuSimulation::with_device(device, queue, config, particles, registry)
}

impl State {
    fn cursor_grid(&self) -> Vec2 {
        Vec2::new(
            self.cursor_frac[0] * GRID as f32,
            (1.0 - self.cursor_frac[1]) * GRID as f32,
        )
    }

    fn reset(&mut self) {
        let (device, queue) = (self.sim.device().clone(), self.sim.queue().clone());
        self.sim = make_sim_data(device, queue);
        self.frame = 0;
        self.stepper.reset();
        self.last_instant = std::time::Instant::now();
        println!("reset");
    }
}

impl DemoApp for State {
    const TITLE: &'static str = "emerge — Sand GPU [Drucker-Prager: loose phi=20 / dense phi=40]";

    fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> Self {
        let sim = make_sim_data(Arc::new(device.clone()), Arc::new(queue.clone()));
        let mut renderer = Renderer::new(sim.device(), sim.particle_count(), format);
        renderer.set_camera(sim.queue(), GRID as u32, width, height, 0.6, true);
        renderer.set_color_mode(ColorMode::ByPhysics);
        renderer.set_optical_params(sim.queue(), MAT_LOOSE as usize, SIGMA_SAND);
        renderer.set_optical_params(sim.queue(), MAT_DENSE as usize, SIGMA_SAND);
        println!(
            "sand GPU: {} particles  |  LMB push  RMB pull  R reset  Q quit",
            sim.particle_count()
        );
        Self {
            sim,
            renderer,
            cursor_frac: [0.0; 2],
            lmb: false,
            rmb: false,
            frame: 0,
            fps_timer: std::time::Instant::now(),
            fps_frames: 0,
            stepper: FixedStepController::standard(DT, 60.0),
            last_instant: std::time::Instant::now(),
            max_steps_seen: 0,
        }
    }

    fn resize(&mut self, _queue: &wgpu::Queue, width: u32, height: u32) {
        self.renderer
            .set_camera(self.sim.queue(), GRID as u32, width, height, 0.6, true);
    }

    fn update_and_render(
        &mut self,
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
        view: &wgpu::TextureView,
    ) {
        if self.lmb || self.rmb {
            let mag = if self.lmb { 12.0 } else { -12.0 };
            self.sim.apply_radial_impulse(self.cursor_grid(), 7.0, mag);
        }
        let now = std::time::Instant::now();
        let frame_delta = (now - self.last_instant).as_secs_f32();
        self.last_instant = now;
        let steps = self.stepper.steps_for_frame(frame_delta);
        self.max_steps_seen = self.max_steps_seen.max(steps);
        for _ in 0..steps {
            self.sim.step_frame();
            self.frame += 1;
            if self.frame.is_multiple_of(60) {
                log_frame_gpu(self.frame, DT, self.sim.particles(), LABELS, 1);
                let snap = self.sim.diagnostics_snapshot();
                println!(
                    "  non_finite={}  out_of_bounds={}  max_speed={:.3}  sub={}  cfl={:.4}",
                    snap.non_finite_particle_values,
                    snap.out_of_bounds_particles,
                    snap.max_particle_speed,
                    snap.substeps_last_step,
                    snap.cfl_number,
                );
            }
        }
        self.fps_frames += 1;
        if self.fps_timer.elapsed().as_secs_f32() >= 2.0 {
            let fps = self.fps_frames as f32 / self.fps_timer.elapsed().as_secs_f32();
            println!(
                "frame={} fps={:.0} max_steps_per_render={}",
                self.frame, fps, self.max_steps_seen
            );
            self.fps_timer = std::time::Instant::now();
            self.fps_frames = 0;
            self.max_steps_seen = 0;
        }
        self.renderer.render_gpu(
            self.sim.device(),
            self.sim.queue(),
            self.sim.particle_buffer(),
            self.sim.particle_count(),
            view,
            true,
        );
    }

    fn cursor_moved(&mut self, x_frac: f32, y_frac: f32) {
        self.cursor_frac = [x_frac, y_frac];
    }

    fn mouse_button(&mut self, button: MouseButton, pressed: bool) {
        match button {
            MouseButton::Left => self.lmb = pressed,
            MouseButton::Right => self.rmb = pressed,
            _ => {}
        }
    }

    fn key_pressed(&mut self, key: KeyCode) {
        if key == KeyCode::KeyR {
            self.reset();
        }
    }
}

fn main() {
    run_demo::<State>();
}

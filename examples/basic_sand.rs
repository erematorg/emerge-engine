extern crate emerge_engine as emerge;

use emerge::render::demo_harness::{DemoApp, run_demo};
use emerge::render::{ColorMode, Renderer};
use emerge::{DruckerPragerMaterial, SimConfig, Simulation, SlipBoundary, SpawnRegion};
use glam::{IVec2, Vec2};
/// CPU Drucker-Prager sand -- angle of repose comparison.
///
///   Mat 0  loose sand  (blue, phi=20 deg) -- shallow repose angle
///   Mat 1  dense sand  (gold, phi=40 deg) -- steep repose angle
///
///   cargo run --example basic_sand --features "render"
use winit::event::MouseButton;
use winit::keyboard::KeyCode;

const GRID: usize = 64;
const DT: f32 = 0.1;
const MAT_LOOSE: u32 = 0;
const MAT_DENSE: u32 = 1;
// Real measured sand absorption (Sherman & Waite 1985, iron-oxide quartz sand) — see
// basic_sand_gpu.rs for the full reasoning.
const SIGMA_SAND: [f32; 3] = [0.180, 0.220, 0.550];

struct State {
    sim: Simulation,
    renderer: Renderer,
    cursor_frac: [f32; 2],
    lmb: bool,
    rmb: bool,
    frame: u64,
    fps_timer: std::time::Instant,
    fps_frames: u64,
}

fn make_sand(lambda: f32, mu: f32, phi_deg: f32) -> DruckerPragerMaterial {
    let mut m = DruckerPragerMaterial::new(lambda, mu);
    m.friction_angle = phi_deg.to_radians();
    m
}

fn make_sim() -> Simulation {
    let config = SimConfig {
        boundary_thickness: 3,
        max_substeps_per_step: 12,
        // Deliberately weak, NOT real IRL gravity (real g_grid ~= 981 via
        // SimConfig::earth) -- tuned down for a calmer, more legible demo at
        // this grid scale. Disclosed, deferred: basic_sand_gui.rs's
        // gravity_fraction slider is the real-IRL-with-live-control
        // pattern, not yet ported to every plain example.
        gravity: Vec2::new(0.0, -0.3),
        // Drucker-Prager sand's own plastic "q-creep" (friction_hardening never
        // fully settles to zero even at apparent rest -- see MEMORY.md's known
        // open issue) means this material pins the substep count at the cap
        // indefinitely rather than dropping once settled -- confirmed live
        // 2026-07-22 (substeps=12 continuously, never lower). 0.7 is the same
        // real, already-validated coefficient (`gpu_relaxed_cfl_coefficient_
        // stays_correct_50k_dpsand`) used for this exact material class at the
        // GPU 50k target -- still within the literature's normal 0.3-1.0 range,
        // not a new gamble.
        material_cfl_coefficient: 0.7,
        ..SimConfig::earth(GRID, 0.01, DT)
    };
    let spawn = |c: Vec2, mat, seed| SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new(18, 14),
        box_center: c,
        material_id: mat,
        precompute_initial_volumes: true,
        initial_velocity_scale: 0.0,
        rng_seed: seed,
        // See basic_sand_gpu.rs's spawn closure for the full reasoning: a perfectly regular
        // spawn lattice is a grid-crossing artifact with quadratic B-spline MPM kernels,
        // confirmed via direct frame capture on the GPU path (same spawn pattern here).
        position_jitter: 0.5,
        ..SpawnRegion::for_sim(&config)
    };
    // lambda=2000, mu=3000 -> nu≈0.2 — see basic_sand_gpu.rs's make_sand call for the full
    // reasoning (the previous 5000/3000 implied nu≈0.31, above real dry sand's established
    // 0.1-0.3 range, and directly resists Drucker-Prager yielding via the (lambda+mu)/mu ratio).
    let mut solver = Simulation::new(config, spawn(Vec2::new(17.0, 40.0), MAT_LOOSE, 11))
        .with_default_material(Box::new(make_sand(2000.0, 3000.0, 20.0)))
        .with_material(MAT_DENSE, Box::new(make_sand(2000.0, 3000.0, 40.0)))
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));
    let _ = solver.add_body(spawn(Vec2::new(47.0, 40.0), MAT_DENSE, 22));
    solver
}

impl State {
    fn cursor_grid(&self) -> Vec2 {
        Vec2::new(
            self.cursor_frac[0] * GRID as f32,
            (1.0 - self.cursor_frac[1]) * GRID as f32,
        )
    }
}

impl DemoApp for State {
    const TITLE: &'static str =
        "emerge -- Sand [Angle of Repose: loose phi=20 deg / dense phi=40 deg]";

    fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> Self {
        let sim = make_sim();
        let mut renderer = Renderer::new(device, sim.particles().len(), format);
        renderer.set_camera(queue, GRID as u32, width, height, 0.6, true);
        renderer.set_color_mode(ColorMode::ByPhysics);
        renderer.set_optical_params(queue, MAT_LOOSE as usize, SIGMA_SAND);
        renderer.set_optical_params(queue, MAT_DENSE as usize, SIGMA_SAND);
        println!(
            "sand: {} particles  |  LMB push  RMB pull  R reset  Q quit",
            sim.particles().len()
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
        }
    }

    fn resize(&mut self, queue: &wgpu::Queue, width: u32, height: u32) {
        self.renderer
            .set_camera(queue, GRID as u32, width, height, 0.6, true);
    }

    fn update_and_render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        view: &wgpu::TextureView,
    ) {
        if self.lmb || self.rmb {
            let mag = if self.lmb { 12.0 } else { -12.0 };
            self.sim.apply_radial_impulse(self.cursor_grid(), 7.0, mag);
        }
        self.sim.step();
        self.frame += 1;
        self.fps_frames += 1;
        if self.fps_timer.elapsed().as_secs_f32() >= 2.0 {
            let fps = self.fps_frames as f32 / self.fps_timer.elapsed().as_secs_f32();
            println!("frame={} fps={:.0}", self.frame, fps);
            self.fps_timer = std::time::Instant::now();
            self.fps_frames = 0;
        }
        self.renderer
            .render(device, queue, self.sim.particles(), view, true);
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
            self.sim = make_sim();
            self.frame = 0;
            println!("reset");
        }
    }
}

fn main() {
    run_demo::<State>();
}

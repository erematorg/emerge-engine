extern crate emerge_engine as emerge;

use emerge::render::demo_harness::{DemoApp, run_demo};
use emerge::render::{ColorMode, Renderer};
use emerge::{
    DruckerPragerMaterial, SimConfig, Simulation, SlipBoundary, SpawnRegion, StomakhinMaterial,
};
use glam::{IVec2, Vec2};
/// CPU snowballs colliding -- Stomakhin 2013 snow plasticity.
///
///   Mat 0  soft powder (blue)  -- low hardening, wide plastic limits
///   Mat 1  packed snow (gold)  -- high hardening, tight limits
///   Mat 2  shatter     (cyan)  -- loose granular after violent impact
///
///   cargo run --example basic_snow --features "render"
use winit::event::MouseButton;
use winit::keyboard::KeyCode;

const GRID: usize = 64;
const DT: f32 = 0.1;
const MAT_SOFT: u32 = 0;
const MAT_PACKED: u32 = 1;
const MAT_SHATTER: u32 = 2;
const BALL_R: f32 = 9.0;
const BALL_A: Vec2 = Vec2::new(16.0, 44.0);
const BALL_B: Vec2 = Vec2::new(48.0, 44.0);
const SPEED: f32 = 15.0;

struct State {
    sim: Simulation,
    renderer: Renderer,
    cursor_frac: [f32; 2],
    lmb: bool,
    frame: u64,
    fps_timer: std::time::Instant,
    fps_frames: u64,
}

fn make_sim() -> Simulation {
    let config = SimConfig {
        max_substeps_per_step: 20,
        // Deliberately weak, NOT real IRL gravity (real g_grid ~= 981 via
        // SimConfig::earth) -- tuned down for a calmer, more legible demo at
        // this grid scale. Disclosed, deferred: basic_snow_gui.rs's
        // gravity_fraction slider is the real-IRL-with-live-control
        // pattern, not yet ported to every plain example.
        gravity: Vec2::new(0.0, -0.08),
        ..SimConfig::earth(GRID, 0.01, DT)
    };
    let spawn = SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new(58, 58),
        rng_seed: 7,
        ..SpawnRegion::for_sim(&config)
    };
    let mut solver = Simulation::new(config, spawn)
        .with_default_material(Box::new(StomakhinMaterial::new(
            1389.0, 2083.0, 7.0, 0.025, 0.0075, 0.6, 20.0,
        )))
        .with_material(
            MAT_PACKED,
            Box::new(
                StomakhinMaterial::new(1389.0, 2083.0, 10.0, 0.012, 0.004, 0.6, 20.0)
                    .with_cohesion(400.0),
            ),
        )
        .with_material(
            MAT_SHATTER,
            Box::new(DruckerPragerMaterial::low_friction(266.7, 0.333)),
        )
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));

    solver.retain_particles(|p| {
        (p.x - BALL_A).length() <= BALL_R || (p.x - BALL_B).length() <= BALL_R
    });
    solver.particles_mut().for_each_mut(|p| {
        if (p.x - BALL_A).length() <= BALL_R {
            p.material_id = MAT_SOFT;
            p.v = Vec2::new(SPEED, 0.0);
        } else {
            p.material_id = MAT_PACKED;
            p.v = Vec2::new(-SPEED, 0.0);
        }
    });
    solver.recompute_initial_volumes();
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
    const TITLE: &'static str = "emerge -- Snow [Soft Powder / Packed Snow collision]";

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
        renderer.set_color_mode(ColorMode::ByMaterial);
        println!(
            "snow: {} particles  |  LMB push  R reset  Q quit",
            sim.particles().len()
        );
        Self {
            sim,
            renderer,
            cursor_frac: [0.0; 2],
            lmb: false,
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
        if self.lmb {
            self.sim.apply_radial_impulse(self.cursor_grid(), 6.0, 10.0);
        }
        self.sim.step();
        // Fracture trigger: real plastic compression (Jp), not raw speed --
        // the old `v.length() > 5.0` fired at launch, before any collision.
        self.sim.phase_transition(
            |p| p.material_id == MAT_PACKED && p.plastic_volume_ratio < 0.9,
            MAT_SHATTER,
        );
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
        if button == MouseButton::Left {
            self.lmb = pressed;
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

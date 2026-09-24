extern crate emerge_engine as emerge;

use emerge::render::demo_harness::{DemoApp, run_demo};
use emerge::render::{ColorMode, Renderer};
/// Real, live demo of `KinematicCircleBoundary`: drag the cursor through the
/// water and it pushes back via a genuine no-penetration + Coulomb friction
/// grid boundary correction (reusing `apply_coulomb_wall`, same primitive
/// `FrictionBoundary` uses) -- NOT `apply_impulse`/`apply_radial_impulse`
/// (no mass, no momentum conservation, a direct velocity write). No shape is
/// drawn for the obstacle itself; the water's own reaction IS the proof,
/// same convention every other push/pull demo in this engine already uses.
///
/// `is_strict_wc_mpm_fluid_compatible` is TEMPORARILY true in
/// `kinematic_obstacle.rs` for this evaluation -- see that file's own
/// comment and project_fluid_solid_coupling_real_root_cause_and_path memory.
///
///   cargo run --example kinematic_obstacle_water_demo --features render
use emerge::{KinematicCircleBoundary, NewtonianFluidMaterial, SimConfig, Simulation, SpawnRegion};
use glam::{IVec2, Vec2};
use std::sync::Arc;
use winit::keyboard::KeyCode;

const GRID: usize = 64;
const DT: f32 = 0.1;
const MAT_WATER: u32 = 0;
const OBSTACLE_RADIUS: f32 = 3.0;

struct State {
    sim: Simulation,
    obstacle: Arc<KinematicCircleBoundary>,
    renderer: Renderer,
    cursor_frac: [f32; 2],
    last_cursor_grid: Vec2,
    frame: u64,
    fps_timer: std::time::Instant,
    fps_frames: u64,
}

fn make_sim() -> (Simulation, Arc<KinematicCircleBoundary>) {
    let config = SimConfig {
        min_dt: 1.0e-4,
        max_substeps_per_step: 400,
        gravity: Vec2::new(0.0, -0.3),
        recompute_density_each_step: false,
        cfl_include_affine_speed: false,
        ..SimConfig::earth(GRID, 0.01, DT)
    };
    // Same real, SI-calibrated water as basic_fluids.rs.
    let water = NewtonianFluidMaterial::low_viscosity(0.1, 2.5);
    const SPACING: f32 = 0.9;
    let spawn_water = SpawnRegion {
        spacing: SPACING,
        mass_override: Some(0.1 * SPACING * SPACING),
        box_size: IVec2::new(24, 20),
        box_center: Vec2::new(32.0, 20.0),
        material_id: MAT_WATER,
        initial_velocity_scale: 0.0,
        ..SpawnRegion::for_sim(&config)
    };
    let obstacle = Arc::new(KinematicCircleBoundary::new(
        Vec2::new(-10.0, -10.0), // off-screen until the cursor first moves
        OBSTACLE_RADIUS,
        0.3,
    ));
    let solver = Simulation::new(config, spawn_water)
        .with_default_material(Box::new(water))
        .with_boundary(Box::new(obstacle.clone()));
    (solver, obstacle)
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
    const TITLE: &'static str = "emerge -- Kinematic Obstacle vs. Real Water";

    fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> Self {
        let (sim, obstacle) = make_sim();
        let mut renderer = Renderer::new(device, sim.particles().len(), format);
        renderer.set_camera(queue, GRID as u32, width, height, 0.6, true);
        renderer.set_color_mode(ColorMode::ByMaterial);
        println!(
            "kinematic obstacle vs. real water: {} particles | move cursor to drag the obstacle | R reset | Q quit",
            sim.particles().len()
        );
        Self {
            sim,
            obstacle,
            renderer,
            cursor_frac: [0.0; 2],
            last_cursor_grid: Vec2::ZERO,
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
        let cursor = self.cursor_grid();
        let vel = (cursor - self.last_cursor_grid) / DT;
        self.obstacle.set_position_velocity(cursor, vel);
        self.last_cursor_grid = cursor;

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

    fn key_pressed(&mut self, key: KeyCode) {
        if key == KeyCode::KeyR {
            let (sim, obstacle) = make_sim();
            self.sim = sim;
            self.obstacle = obstacle;
            self.frame = 0;
            println!("reset");
        }
    }
}

fn main() {
    run_demo::<State>();
}

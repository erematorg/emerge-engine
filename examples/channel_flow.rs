extern crate emerge_engine as emerge;

use emerge::fields::LinearDragField;
use emerge::render::demo_harness::{DemoApp, run_demo};
use emerge::render::{ColorMode, Renderer};
use emerge::{NewtonianFluidMaterial, SimConfig, Simulation, SlipBoundary, SpawnRegion};
use glam::{IVec2, Vec2};
/// Minimal real-forces proof: a real fluid material, no gravity-settling puddle, driven
/// downstream by `LinearDragField` -- the drag/current force field (see its own doc
/// comment for the real physics: Stokes drag / Rayleigh friction, the SAME technique
/// that drives river currents and wind-blown sand in this engine).
///
/// A pool of water spawns on the left; the drag field pushes it rightward the whole run,
/// instead of the fluid just falling and puddling under gravity alone. Same field, same
/// mechanism -- change `target_velocity` to a wind direction and mask to a granular
/// material for wind-blown sand instead of masking to water; not built as a second demo
/// here, this scene exists to prove the ONE new mechanism, not every dressing of it.
///
///   cargo run --example channel_flow --features "render"
use winit::keyboard::KeyCode;

const GRID: usize = 96;
const DT: f32 = 0.1;
const MAT_WATER: u32 = 0;
const CURRENT_SPEED: f32 = 4.0;
const DRAG_COEFFICIENT: f32 = 1.5;

struct State {
    sim: Simulation,
    renderer: Renderer,
    frame: u64,
    fps_timer: std::time::Instant,
    fps_frames: u64,
}

fn make_sim() -> Simulation {
    let config = SimConfig {
        min_dt: 1.0e-3,
        max_substeps_per_step: 8,
        recompute_density_each_step: true,
        cfl_include_affine_speed: false,
        // Deliberately weak, NOT real IRL gravity (real g_grid ~= 981 via
        // SimConfig::earth) -- tuned down for a calmer, more legible demo at
        // this grid scale. Disclosed, deferred: basic_fluids_gui.rs's
        // gravity_fraction slider is the real-IRL-with-live-control
        // pattern, not yet ported to every plain example.
        gravity: Vec2::new(0.0, -0.15),
        ..SimConfig::earth(GRID, 0.01, DT)
    };
    // Real water: Cole 1948 Tait exponent (7.0) + real dynamic viscosity, not a
    // hand-picked 0.1/3.0 pair -- see NewtonianFluidMaterial::low_viscosity.
    // rest_density=0.1, NOT the old 4.0 -- real SI fix, 2026-08-08, see
    // basic_fluids.rs's own doc for the full derivation.
    // eos_stiffness=0.25, NOT 10 -- rest_density shrinking 40x makes
    // `timestep_bound`'s c2 (sound-speed-squared) 40x larger at the old
    // stiffness for the same compression; confirmed by a real crash in
    // basic_fluids.rs's CPU twin. Rescaling stiffness by the same factor
    // (10*0.1/4.0=0.25) restores the original, already-stable c2 -- see
    // basic_fluids.rs's own doc for the full derivation.
    let water = NewtonianFluidMaterial::low_viscosity(0.1, 0.25);
    let spawn_water = SpawnRegion {
        spacing: 0.6,
        box_size: IVec2::new(20, 16),
        box_center: Vec2::new(14.0, 12.0),
        material_id: MAT_WATER,
        initial_velocity_scale: 0.0,
        // Without this, mass falls back to `config.particle_mass` (1.0),
        // completely decoupled from the material's own rest_density=0.1
        // -- a real, separate gap found 2026-08-08 alongside the SI fix
        // (see basic_fluids.rs's doc). m = rho0*spacing^2, same
        // derivation used everywhere else.
        mass_override: Some(0.1 * 0.6 * 0.6),
        ..SpawnRegion::for_sim(&config)
    };
    let current = LinearDragField::new(
        Vec2::new(CURRENT_SPEED, 0.0),
        DRAG_COEFFICIENT,
        1 << MAT_WATER,
    );
    Simulation::new(config, spawn_water)
        .with_default_material(Box::new(water))
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)))
        .with_force_field(Box::new(current))
}

impl DemoApp for State {
    const TITLE: &'static str = "emerge -- Channel Flow [LinearDragField]";
    const SIZE: (u32, u32) = (640, 480);

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
            "channel_flow: {} water particles  |  LinearDragField pushes downstream at target_v=({CURRENT_SPEED},0)  |  R reset  Q quit",
            sim.particles().len()
        );
        Self {
            sim,
            renderer,
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
        self.sim.step();
        self.frame += 1;
        self.fps_frames += 1;
        if self.fps_timer.elapsed().as_secs_f32() >= 2.0 {
            let fps = self.fps_frames as f32 / self.fps_timer.elapsed().as_secs_f32();
            let cx: f32 = self.sim.particles().iter().map(|p| p.x.x).sum::<f32>()
                / self.sim.particles().len() as f32;
            println!(
                "frame={} fps={:.0} water_centroid_x={cx:.2} (started at 14.0 -- should keep climbing)",
                self.frame, fps
            );
            self.fps_timer = std::time::Instant::now();
            self.fps_frames = 0;
        }
        self.renderer
            .render(device, queue, self.sim.particles(), view, true);
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

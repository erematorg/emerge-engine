extern crate emerge_engine as emerge;

use emerge::fields::GravityWellField;
use emerge::render::demo_harness::{DemoApp, run_demo};
use emerge::render::{ColorMode, Renderer};
use emerge::{NeoHookeanMaterial, SimConfig, Simulation, SpawnRegion};
/// Real solar-system-scale orbital mechanics, live: Sun (fixed) + Earth + Mars,
/// real masses/distances (NASA NSSDCA Planetary Fact Sheet), real Newtonian
/// gravity via the engine's existing `GravityWellField` -- proven headless in
/// `tests/orbital_mechanics.rs` (Kepler's third law holds within 0.09%, a
/// real measured/tuned grid-resolution choice -- see that file's own
/// `diag_kepler_error_vs_grid_resolution_sweep`). For a live speed slider,
/// see `basic_orbital_gui.rs`.
///
/// Real, disclosed simplifications for this first pass ("the system itself,
/// not full detail" -- see that test file's own doc for the full real vs.
/// simplified breakdown):
///   - Sun treated as fixed (restricted two-body problem -- Sun is
///     ~333,000x Earth's mass, so this is standard practice, not a hack).
///   - No inter-planet gravity (Earth/Mars don't pull on each other).
///   - Bodies rendered at equal visual size -- real relative sizes AND
///     distances can never be shown to the same scale at once (true of every
///     real astronomy diagram, not an emerge-specific shortcut).
///
///   cargo run --example basic_orbital --features "render"
use glam::{IVec2, Vec2};
use winit::keyboard::KeyCode;

/// 1 grid cell = 250,000 km -- real, measured (not guessed) choice: puts
/// Earth's orbital radius at ~598 grid units, where a real grid-resolution
/// sweep (`tests/orbital_mechanics.rs::diag_kepler_error_vs_grid_resolution_sweep`)
/// measured Kepler's-third-law error at 0.09%, down from 0.36% at the
/// original 4x-coarser scale.
const GRID: usize = 2048;
const DX_METERS: f64 = 2.5e8;
/// 1 real hour per substep -- Earth's real 365-day year plays out in ~8760
/// steps, i.e. ~2.4 real minutes at 60fps. No artificial time compression.
const DT_SECONDS: f64 = 3600.0;

const MU_SUN_SI: f64 = 1.32712e20; // G*M_sun, m^3/s^2 -- NASA/JPL
const AU_M: f64 = 1.496e11;
const MARS_DISTANCE_M: f64 = 228.0e9;
const EARTH_MASS_KG: f32 = 5.97e24;
const MARS_MASS_KG: f32 = 0.642e24;

const MAT_SUN: u32 = 0;
const MAT_EARTH: u32 = 1;
const MAT_MARS: u32 = 2;

fn circular_orbit_speed_grid(r_si: f64) -> f32 {
    ((MU_SUN_SI / r_si).sqrt() / DX_METERS) as f32
}

struct State {
    sim: Simulation,
    renderer: Renderer,
    days_elapsed: f32,
    frame: u64,
}

fn make_sim() -> Simulation {
    let sun_pos = Vec2::splat(GRID as f32 / 2.0);
    let config = SimConfig {
        dx_meters: DX_METERS as f32,
        dt_seconds: DT_SECONDS as f32,
        gravity: Vec2::ZERO,
        ..SimConfig::standard(GRID, DT_SECONDS as f32, Vec2::ZERO)
    };

    let r_earth = (AU_M / DX_METERS) as f32;
    let r_mars = (MARS_DISTANCE_M / DX_METERS) as f32;
    let v_earth = circular_orbit_speed_grid(AU_M);
    let v_mars = circular_orbit_speed_grid(MARS_DISTANCE_M);

    let spawn_sun = SpawnRegion {
        spacing: 1.0,
        box_size: IVec2::new(1, 1),
        box_center: sun_pos,
        position_jitter: 0.0,
        material_id: MAT_SUN,
        mass_override: Some((MU_SUN_SI / 6.674e-11) as f32), // real Sun mass, kg
        ..SpawnRegion::for_sim(&config)
    };
    let spawn_earth = SpawnRegion {
        spacing: 1.0,
        box_size: IVec2::new(1, 1),
        box_center: sun_pos + Vec2::new(r_earth, 0.0),
        position_jitter: 0.0,
        material_id: MAT_EARTH,
        mass_override: Some(EARTH_MASS_KG),
        ..SpawnRegion::for_sim(&config)
    };
    let spawn_mars = SpawnRegion {
        spacing: 1.0,
        box_size: IVec2::new(1, 1),
        box_center: sun_pos + Vec2::new(0.0, r_mars),
        position_jitter: 0.0,
        material_id: MAT_MARS,
        mass_override: Some(MARS_MASS_KG),
        ..SpawnRegion::for_sim(&config)
    };

    let mu_grid = (MU_SUN_SI / (DX_METERS * DX_METERS * DX_METERS)) as f32;
    let sun_well = GravityWellField::point(sun_pos, mu_grid, 1.0, 0.05);

    let mut solver = Simulation::new(config, spawn_sun)
        .with_default_material(Box::new(NeoHookeanMaterial::new(1.0, 1.0)))
        .with_material(MAT_EARTH, Box::new(NeoHookeanMaterial::new(1.0, 1.0)))
        .with_material(MAT_MARS, Box::new(NeoHookeanMaterial::new(1.0, 1.0)))
        .with_force_field(Box::new(sun_well));

    // Sun is fixed (restricted two-body problem, see module doc) -- pin it
    // so it stays put and visible without feeling its own gravity well.
    solver.particles_mut().pinned[0] = 1;

    let _ = solver.add_body(spawn_earth);
    solver.particles_mut().v[1] = Vec2::new(0.0, v_earth);

    let _ = solver.add_body(spawn_mars);
    // Mars starts 90 degrees around from Earth (spawned along +y instead of
    // +x above) -- real tangential velocity for THAT position is along -x.
    solver.particles_mut().v[2] = Vec2::new(-v_mars, 0.0);

    solver
}

impl DemoApp for State {
    const TITLE: &'static str = "emerge -- Orbital [Sun / Earth / Mars]";
    const SIZE: (u32, u32) = (640, 640);

    fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> Self {
        let sim = make_sim();
        let mut renderer = Renderer::new(device, sim.particles().len(), format);
        // particle_scale=6.0 -- purely visual (real relative sizes are
        // un-renderable at this distance scale, see module doc).
        renderer.set_camera(queue, GRID as u32, width, height, 6.0, true);
        renderer.set_color_mode(ColorMode::ByMaterial);
        println!("orbital: Sun + Earth + Mars, real NASA masses/distances  |  R reset  Q quit");
        Self {
            sim,
            renderer,
            days_elapsed: 0.0,
            frame: 0,
        }
    }

    fn resize(&mut self, queue: &wgpu::Queue, width: u32, height: u32) {
        self.renderer
            .set_camera(queue, GRID as u32, width, height, 6.0, true);
    }

    fn update_and_render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        view: &wgpu::TextureView,
    ) {
        self.sim.step();
        self.frame += 1;
        self.days_elapsed += DT_SECONDS as f32 / 86400.0;
        if self.frame.is_multiple_of(720) {
            // ~30 real days per print (720 hourly substeps).
            println!("day {:.0}", self.days_elapsed);
        }
        self.renderer
            .render(device, queue, self.sim.particles(), view, true);
    }

    fn key_pressed(&mut self, key: KeyCode) {
        if key == KeyCode::KeyR {
            self.sim = make_sim();
            self.days_elapsed = 0.0;
            self.frame = 0;
            println!("reset");
        }
    }
}

fn main() {
    run_demo::<State>();
}

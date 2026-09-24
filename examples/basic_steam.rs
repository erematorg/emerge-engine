extern crate emerge_engine as emerge;

use emerge::render::demo_harness::{DemoApp, run_demo};
use emerge::render::{ColorMode, Renderer};
use emerge::thermodynamics::{ThermalConfig, ThermalDiffusion};
use emerge::{
    GasMaterial, NewtonianFluidMaterial, SimConfig, Simulation, SlipBoundary, SpawnRegion,
};
use glam::{IVec2, Vec2};
/// The real, local water cycle: liquid water heats past 373.15K and
/// becomes real steam (`GasMaterial`, water vapor's own real adiabatic
/// index, NOT air's), then steam that cools back below 373.15K (via the
/// engine's already-real Newton-cooling term toward ambient) condenses
/// back into liquid water. ONE real phase rule, run both directions --
/// not two separate scripted effects.
///
/// Real engine-level fix landed here, not a demo-local workaround: water
/// boiling into `GasMaterial` steam is a real ~1700x density-ratio phase
/// transition, and `Simulation::apply_phase_transition` blindly
/// overwriting a transitioning particle's real, continuous prior volume
/// with a fresh `mass/rest_density` analytical value made that volume
/// jump ~1700x in a single instant -- a genuine, wildly under-resolved
/// force spike, confirmed live as a real crash cause. Fixed at the
/// engine level via `MaterialModel::init_particle_from_transition` (a
/// new trait method, defaults to today's `init_particle` behavior for
/// every OTHER material -- zero change to water->ice or any other
/// existing phase-transition demo), which `GasMaterial` now overrides to
/// preserve real volume continuity instead of teleporting. See that
/// method's own doc (`src/matter/materials/mod.rs`) and `GasMaterial::
/// init_particle_from_transition`'s own doc (`gas/ideal_gas.rs`) for the
/// full mechanism.
///
/// Real, disclosed scope limit: steam does not RISE yet (no buoyancy
/// wired into this version -- real buoyant rise needs either a
/// `BuoyancyField` or a real background-air region so a genuine density-
/// driven pressure gradient exists; neither is here tonight), so it heats
/// and condenses roughly where it forms rather than rising to a cold
/// ceiling and dripping back down. The temperature-driven PHASE cycle
/// itself is real; the spatial rise/fall drama is real, deferred future
/// work.
///
/// Real constants: adiabatic index gamma=1.33 (real, standard value for
/// triatomic H2O, unchanged), real boiling point 373.15K (unchanged, still
/// what the phase rule and rendered temperature both use), real steam
/// viscosity ~1.26e-5 Pa*s (standard saturated-steam value at 100C, e.g.
/// NIST steam tables). NOT real: the specific gas constant R and rest
/// density -- see `STEAM_RHO_KG_M3`/`STEAM_SPECIFIC_GAS_CONSTANT_J_KG_K`'s
/// own docs for the full live-measured story of why, and the exact real
/// values (461.5 J/(kg*K), ~0.588 kg/m3) kept there for the record.
///
/// Real, disclosed scope reduction (see `STEAM_RHO_KG_M3`'s own doc for
/// the full live-measured story): steam's REST DENSITY here is NOT the
/// real ideal-gas-law value (~0.588 kg/m3, a real ~1700x ratio vs water)
/// -- even after the real engine-level volume-continuity fix above, the
/// full ratio still needed more substep budget than stayed practical for
/// an interactive demo at this resolution (a genuine numerical stiffness
/// property of that large a ratio, not a remaining bug). Scaled to a
/// ~6:1 ratio instead, matching the same range `basic_gas.rs` already
/// proved stable this session -- still real R/gamma/viscosity, just not
/// the full real density contrast.
///
///   cargo run --example basic_steam --features "render"
use winit::event::MouseButton;
use winit::keyboard::KeyCode;

const GRID: usize = 64;
const DT: f32 = 0.015;
const DX_METERS: f32 = 1.0;
// Real, live-found reason this is far finer than basic_gas.rs's own 0.5
// (2026-08-18): a water particle's mass is sized for water's real density
// (1000 kg/m3); when it phase-transitions to steam, `GasMaterial::
// init_particle` computes that SAME mass's real volume at steam's own
// real, much lower rest density (~0.588 kg/m3, an honest ~1700x ratio --
// real steam expansion, not a bug). At spacing=0.5 that put a SINGLE
// particle's claimed post-transition volume at ~425 grid-area-units
// (~20x20 cells) -- far beyond what one MPM particle's kernel (~2-3 cell
// support) can represent, injecting a real but wildly under-resolved
// force spike that crashed the strict-fluid CFL/retry gate the instant
// steam actually formed (confirmed live: stable for 1000s of idle frames,
// crashed within ~5 frames of the phase transition itself firing).
// Finer spacing -> smaller per-particle mass -> smaller absolute
// post-transition volume for the SAME real density ratio. 0.08 keeps that
// post-transition footprint to a real, disclosed, kernel-tractable
// ~11 grid-area-units. Real, disclosed tradeoff: a small water SAMPLE
// (see WATER_POOL_SIZE_CELLS below), not a full-size pool -- proving the
// real bidirectional phase cycle at a numerically honest scale, not the
// full realistic volume in one shot.
const SPACING: f32 = 0.08;
const MAT_WATER: u32 = 0;
const MAT_STEAM: u32 = 1;
const BOIL_POINT_K: f32 = 373.15;
const ROOM_TEMPERATURE_K: f32 = 293.15;
const WATER_RHO_KG_M3: f32 = 1000.0;
const STEAM_ADIABATIC_INDEX: f32 = 1.33; // real, unchanged -- see STEAM_SPECIFIC_GAS_CONSTANT_J_KG_K's own doc for why R is NOT the real 461.5 here
const STEAM_VISCOSITY_PA_S: f32 = 1.26e-5;
// Real, disclosed scope reduction, live-measured 2026-08-18 -- NOT the
// real ideal-gas-law density (~0.588 kg/m3, a real ~1700x ratio vs
// water). Even after fixing two real bugs (a volume/deformation-gradient
// consistency bug, then a substep-budget shortfall), the full ratio
// stayed unstable at 3000 substeps for even a SINGLE transitioning
// particle. Scaled to a gentler ~6:1 ratio instead, matching the range
// `basic_gas.rs` already proved stable this session.
const STEAM_RHO_KG_M3: f32 = WATER_RHO_KG_M3 / 6.0;
// Real SECOND bug, found live right after the density change above (NOT
// caught until the "gentler" ratio crashed identically): `kirchhoff_stress`
// computes rest pressure as `p0 = rest_density * R * particles.temperature`
// -- raising `STEAM_RHO_KG_M3` 283x (from the real 0.588 to 166.67) to get
// a gentler density RATIO also raised `p0` the SAME 283x (real steam's
// ~460 J/(kg*K) times this demo's real rest density gives a REST pressure
// of ~283 atmospheres, before any compression at all -- the actual
// detonator, not the density ratio itself). Real fix: keep the real
// boiling point (373.15K, still what `particles.temperature` reads for
// the phase rule) and the real adiabatic index (1.33), but scale R DOWN
// to compensate for the inflated rest_density, so `p0` lands back at a
// real, sane ~1 atm regardless of the demo-scaled density -- solved
// directly from `p0=rho0*R*T`, not tuned by trial and error. Genuine
// bonus, not just a wash: since sound speed c=sqrt(gamma*R*T), a smaller
// R also makes the CFL requirement gentler, not worse.
const STEAM_SPECIFIC_GAS_CONSTANT_J_KG_K: f32 = 101_325.0 / (STEAM_RHO_KG_M3 * BOIL_POINT_K);

struct State {
    sim: Simulation,
    renderer: Renderer,
    cursor_frac: [f32; 2],
    heating: bool,
    cooling: bool,
    frame: u64,
    fps_timer: std::time::Instant,
    fps_frames: u64,
}

fn make_sim() -> Simulation {
    let config = SimConfig {
        // Real, live-relearned lesson from basic_gas.rs (same session):
        // `SimConfig::earth()`'s default substep cap is tuned for ordinary
        // materials, nowhere near enough for a real compressible gas's own
        // acoustic CFL. Steam's real adiabatic sound speed at 373.15K
        // (c=sqrt(gamma*R*T)~=479 m/s) is already higher than air's
        // ~343 m/s (itself needing 200 substeps). On top of that, a
        // freshly-transitioned particle starts real-compressed (see
        // `GasMaterial::init_particle_from_transition`'s own doc --
        // clamped to GasMaterial's own 20x ceiling), and the isentropic EOS's own
        // pressure there is ~20^1.33~=45x the rest pressure -- since
        // sound speed scales with sqrt(dp/drho), that's a real ~6.7x
        // spike in required resolution right at the moment of transition,
        // on top of steam's own already-higher baseline. 500 still
        // panicked (same strict-fluid CFL/retry gate) live-measured
        // 2026-08-18. 3000 as a disclosed, deliberately generous budget
        // sized from that real ~6.7x factor, not another blind guess.
        max_substeps_per_step: 3000,
        ..SimConfig::earth(GRID, DX_METERS, DT)
    };
    // Real per-material mass -- same real bug/fix as tonight's basic_gas.rs:
    // without this, both materials share ONE `SimConfig::particle_mass`
    // default regardless of their wildly different real densities (water
    // 1000 kg/m3 vs steam ~0.6 kg/m3, an ~1700x real ratio).
    let mass_for = |rho_kg_m3: f32| rho_kg_m3 * (SPACING * config.dx_meters).powi(2);

    // WCSPH rule (Monaghan 1994; Becker & Teschner 2007): c_ref = 10*v_max
    // limits density variation to ~1%. v_max from real free-fall over the
    // pool's own real height -- shrunk to match WATER_POOL_SIZE_CELLS
    // below (a real small sample, not the original full-size pool).
    const POOL_HEIGHT_CELLS: f32 = 5.0;
    let g_grid = 9.81 / config.dx_meters;
    let v_max_grid = (2.0 * g_grid * POOL_HEIGHT_CELLS).sqrt();
    let c_ref_m_s = 10.0 * v_max_grid * config.dx_meters;
    let water =
        NewtonianFluidMaterial::weakly_compressible(WATER_RHO_KG_M3, 1.0e-3, c_ref_m_s, &config);
    let steam = GasMaterial::from_physical(
        STEAM_RHO_KG_M3,
        STEAM_VISCOSITY_PA_S,
        STEAM_SPECIFIC_GAS_CONSTANT_J_KG_K,
        STEAM_ADIABATIC_INDEX,
        BOIL_POINT_K,
        &config,
    );

    let thermal = ThermalDiffusion::new(
        ThermalConfig {
            conductivity: 0.6,     // real water, W/(m*K)
            heat_capacity: 4182.0, // real water, J/(kg*K)
            density: WATER_RHO_KG_M3,
            ambient: ROOM_TEMPERATURE_K,
            grid_cell_size: config.dx_meters,
            // Newton cooling toward ambient -- the real mechanism that
            // brings steam back down below BOIL_POINT_K over time, closing
            // the cycle. Not tuned against a real measured convective
            // coefficient -- a real, disclosed first cut.
            cooling_rate: 0.05,
            ..Default::default()
        },
        config.grid_res,
    );

    // Real, disclosed small sample -- see SPACING's own doc for why: at
    // this finer spacing, the FULL original 50x20-cell pool would spawn
    // ~156k particles. 6x5 cells keeps count comparable to the first
    // version (~4.7k) while staying numerically tractable through the
    // real ~1700x water->steam volume ratio.
    let spawn_water = SpawnRegion {
        spacing: SPACING,
        box_size: IVec2::new(6, 5),
        box_center: Vec2::new(32.0, 8.0),
        material_id: MAT_WATER,
        mass_override: Some(mass_for(WATER_RHO_KG_M3)),
        precompute_initial_volumes: true,
        initial_velocity_scale: 0.0,
        rng_seed: 7,
        position_jitter: 0.3,
        ..SpawnRegion::for_sim(&config)
    };

    let mut solver = Simulation::new(config, spawn_water)
        .with_default_material(Box::new(water))
        .with_material(MAT_STEAM, Box::new(steam))
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)))
        .with_thermal(thermal)
        .with_phase_rule(|p| {
            if p.material_id == MAT_WATER && p.temperature >= BOIL_POINT_K {
                Some(MAT_STEAM)
            } else if p.material_id == MAT_STEAM && p.temperature < BOIL_POINT_K {
                Some(MAT_WATER)
            } else {
                None
            }
        });

    for t in solver.particles_mut().temperature.iter_mut() {
        *t = ROOM_TEMPERATURE_K;
    }
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
        "emerge -- Steam [real water cycle: heat -> steam -> cool -> water]";

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
        // ByThermal makes the whole point of this demo directly visible:
        // watch the real temperature field climb toward boiling, then
        // watch condensed particles cool back down.
        renderer.set_color_mode(ColorMode::ByThermal);
        // Same real fix as basic_gas.rs, forgotten here until the user
        // caught it live: steam's own J legitimately reaches ~18-19 (real
        // expansion, confirmed via the printed diagnostics), and every
        // OTHER renderer call site draws a particle's on-screen size
        // straight from its real deformation gradient -- correct for
        // water (where the shape change IS the signal) but the same
        // giant-overlapping-blob problem for gas-scale J. Real state is
        // untouched; only what's SENT to the renderer changes.
        renderer.set_rigid_render(true);
        println!(
            "steam: {} particles  |  LMB heat  RMB cool  R reset  Q quit",
            sim.particles().len()
        );
        Self {
            sim,
            renderer,
            cursor_frac: [0.0; 2],
            heating: false,
            cooling: false,
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
        if self.heating || self.cooling {
            // Same real click-to-heat pattern material_sandbox_gpu.rs
            // already establishes -- a direct temperature add near the
            // cursor, not a scripted "spawn steam"/"spawn water" shortcut.
            // Real, symmetric bidirectional control: the SAME phase rule
            // already runs both directions (temperature is the only real
            // driver either way) -- RMB just gives a real, ACTIVE way to
            // drive that besides waiting on the passive ambient
            // Newton-cooling term alone.
            let sign = if self.heating { 1.0 } else { -1.0 };
            let cursor = self.cursor_grid();
            let nearby: Vec<usize> = self.sim.particles_near(cursor, 5.0);
            let particles = self.sim.particles_mut();
            for i in nearby {
                particles.temperature[i] = (particles.temperature[i] + sign * 400.0 * DT).max(0.0);
            }
        }
        self.sim.step();
        self.frame += 1;
        self.fps_frames += 1;
        if self.fps_timer.elapsed().as_secs_f32() >= 2.0 {
            let fps = self.fps_frames as f32 / self.fps_timer.elapsed().as_secs_f32();
            let water = self.sim.material_state(MAT_WATER);
            let steam = self.sim.material_state(MAT_STEAM);
            println!(
                "frame={} fps={:.0}  water[n={} avg_T={:.1}]  steam[n={} avg_T={:.1} avg_J={:.2}]",
                self.frame,
                fps,
                water.count,
                water_avg_temp(&self.sim, MAT_WATER),
                steam.count,
                water_avg_temp(&self.sim, MAT_STEAM),
                steam.avg_det_f,
            );
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
            MouseButton::Left => self.heating = pressed,
            MouseButton::Right => self.cooling = pressed,
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

/// `BodyState::avg_det_f`/`avg_density` cover volume/density; temperature
/// isn't in that struct, so this reads it directly off the matching
/// particles -- a real, simple mean, not a hidden approximation.
fn water_avg_temp(sim: &Simulation, material_id: u32) -> f32 {
    let particles = sim.particles();
    let mut sum = 0.0;
    let mut n = 0u32;
    for i in 0..particles.len() {
        if particles.material_id[i] == material_id {
            sum += particles.temperature[i];
            n += 1;
        }
    }
    if n == 0 { 0.0 } else { sum / n as f32 }
}

fn main() {
    run_demo::<State>();
}

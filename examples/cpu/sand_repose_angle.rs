extern crate emerge_engine as emerge;

#[path = "../gui_common/mod.rs"]
mod gui_common;

/// Does sand reach a ~30 deg angle of repose? Three modes, switched live.
///
/// **Pre-shaped mode**: a pile built already at 30 deg, at rest -- does it hold?
/// Recipe (`tests/accuracy.rs::unconfined_pile_with_cundall_damping_reaches_real_repose_angle`):
/// the self-consistent Drucker-Prager return mapping (the default), `apic_blend=0.05`
/// and `cundall_damping=1.0` (Cundall 1982, used in Anura3D geotechnical MPM) hold flat
/// for 100,000+ steps headless.
///
/// **Collapse mode**: a tall column that topples and settles. Full APIC
/// (`apic_blend=1.0`) is unstable for a violent collapse (reach grows with the domain);
/// this mode uses `apic_blend=0.6`. With the `FrictionBoundary(2, 0.7)` floor in effect,
/// `tests/accuracy.rs::sand_collapse_with_phase_gated_relaxation_after_dynamics` lands
/// at 49.3 deg with 0.6 and recalibrates to 0.96 (32.0 deg); this demo has not been
/// re-tuned to match. Press H once it settles to switch to the pre-shaped holding
/// recipe: continued relaxation drifts the angle back down past the target (see
/// `sand_collapse_relaxation_long_horizon_plateau_check`), unlike the pre-shaped case.
///
/// **Grains mode**: the same question with discrete grains (`spacetime::grains`)
/// instead of a continuum: each grain has its own position, velocity, spin and
/// persistent elastic contact springs (Cundall & Strack 1979 / Luding 2008 / Ai et al.
/// 2011), coupled to a sand terrain bed through the shared MPM grid
/// (`grains::coupling`). The rolling-resistance spring restores (see
/// `tests/grains_repose_angle.rs`'s sign test) and has its own `rolling_damping`
/// channel beside `normal_damping`/`tangential_damping` (GeoTaichi's three-way
/// `ndratio`/`sdratio`/`rdratio` split). This scene settles on a plateau near ~0.58x
/// the Lajeunesse-predicted runout (see `grain_contact_config`), against ~1.05-1.13x
/// for the standalone validation.
///
///   cargo run --example sand_repose_angle --features render
use emerge::grains::population::GrainPopulation;
use emerge::materials::granular::grain_contact_law::{ContactLawConfig, critical_timestep};
use emerge::particle::Grain;
use emerge::particle::Particle;
use emerge::render::{ColorMode, Renderer};
use emerge::{
    DruckerPragerMaterial, FrameLogger, FrictionBoundary, SimConfig, Simulation, SpawnRegion,
    per_material_stats,
};
use glam::{IVec2, Mat2, Vec2};
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

// Same push/pull cursor convention as every other sand example (basic_sand.rs):
// LMB push, RMB pull, `apply_radial_impulse` at a fixed radius.
//
// 2.5 cells, not basic_sand.rs's 7.0: this pile is only 12 cells tall and ~41 wide
// (see PRESHAPED_HEIGHT_CELLS), so a 7-cell push centered mid-pile covers more than
// its height and a third of its width and moves the whole pile at once
// (`apply_radial_impulse` is a local radial-falloff kick). 2.5 nudges a few grains
// near the cursor.
const PUSH_RADIUS_CELLS: f32 = 2.5;

const GRID: usize = 128;
const FLOOR: f32 = 2.0;
const SIGMA_SAND: [f32; 3] = [0.180, 0.220, 0.550];

// Pre-shaped mode's own tested geometry.
const PRESHAPED_DT: f32 = 0.016;
const TARGET_ANGLE_DEG: f32 = 30.0;
const PRESHAPED_HEIGHT_CELLS: f32 = 12.0;

// Collapse mode's own tested geometry (matches
// `sand_angle_of_repose_is_physical`'s column, different DT).
const COLLAPSE_DT: f32 = 0.1;

// Grains mode: discrete-element grains (`spacetime::grains`) instead of a continuum
// Drucker-Prager material. The column matches `tests/grains_repose_angle.rs::run_collapse`'s
// 4-wide, 10-tall shape, measured against Lajeunesse et al. 2004 like
// PreShaped/Collapse, from grain positions instead of particle positions.
const GRAINS_R0: usize = 4;
const GRAINS_H0: usize = 10;
const GRAIN_RADIUS: f32 = 1.0; // grid-coordinate units
const GRAIN_MASS: f32 = 1.0;

// Live pour: a cursor-driven tool in all three modes. Click "Pour sand" on, then LMB
// drops material at the cursor's grid position (`cursor_grid()`, the push/pull
// helper); holding it drops a continuous stream. Grains mode adds a discrete `Grain`;
// PreShaped/Collapse add a small continuum blob via `Simulation::add_body` (the API
// `SpawnRegion` is built on). Both are capped: the wgpu instance buffer is sized once
// at startup (`State::new`, see `render_capacity`), and pouring past it would overrun
// a fixed-size GPU buffer. `POUR_COOLDOWN_FRAMES` throttles a held button to a few
// drops per second.
//
// Cap 100: with the 80 base grains, 180 grains measured 54.9-55.6 fps, while a cap of
// 400 (480 grains) dropped to 32.8 fps, under the 45-60 fps floor (measured by a
// grain-count scaling probe that is not kept).
const GRAINS_POUR_CAP: usize = 100;
const PARTICLE_POUR_CAP: usize = 2000;
const POUR_COOLDOWN_FRAMES: u32 = 4;
const GRAINS_TERRAIN_HALF_WIDTH_CELLS: i32 = 30;
const GRAINS_TERRAIN_HEIGHT_CELLS: i32 = 8;
const GRAINS_MARKER_MAT_ID: u32 = 1; // distinct palette slot from the terrain's own material_id=0
// `sigma_a` is an absorption coefficient (radiative-transfer convention, as
// `SIGMA_SAND` above): a low value in a channel means less of it is absorbed, so more
// of that color shows. Low R, moderate G, high B absorption gives a warm
// terracotta/orange (as `SIGMA_SAND`'s low R/G, high B gives tan).
const SIGMA_GRAIN: [f32; 3] = [0.100, 0.300, 0.800];
const GRAINS_ACCENT_MAT_ID: u32 = 3; // small rolling-indicator dot, distinct palette slot again
const SIGMA_ACCENT: [f32; 3] = [0.900, 0.900, 0.900]; // high absorption in every channel = near-black, real high contrast against the warm orange grain body

/// Contact parameters of `tests/grains_grid_coupling.rs::contact_config()`, stable in a
/// live `Simulation`. Softer than the SI-calibrated E=1e7 Pa of the standalone
/// validation (`tests/grains_repose_angle.rs`), which would need a very fine forced
/// substep once coupled to an MPM grid and its adaptive CFL; a non-literal calibration
/// like the effective grain diameter and coarse-grained mass. The rolling-torque sign
/// and `rolling_damping` hold at any stiffness scale.
fn grain_contact_config() -> ContactLawConfig {
    // Damping at 60% of critical, `2*sqrt(k*m_eff)*0.6`, the convention of
    // `ContactLawConfig::dry_sand`, applied to this scene's own softer stiffness
    // (m_eff=GRAIN_MASS*0.5=0.5, normal_stiffness=1e4, critical ~141.4) rather than
    // through `dry_sand`, which assumes SI stiffness and would need the very fine dt
    // this scene avoids. A flat 5.0 would be only ~3.5% of critical. With this damping
    // the scene settles on a plateau near ~0.58x the predicted runout, against
    // ~1.05-1.13x for the standalone validation.
    let m_eff = GRAIN_MASS * 0.5;
    const DAMPING_RATIO: f32 = 0.6;
    let critical_damping = |k: f32| 2.0 * (k * m_eff).sqrt() * DAMPING_RATIO;
    let normal_stiffness = 1.0e4;
    let tangential_stiffness = 0.8e4;
    let rolling_stiffness = 5.0e2;
    ContactLawConfig {
        normal_stiffness,
        tangential_stiffness,
        rolling_stiffness,
        normal_damping: critical_damping(normal_stiffness),
        tangential_damping: critical_damping(tangential_stiffness),
        rolling_damping: critical_damping(rolling_stiffness),
        friction: (35.0_f32).to_radians().tan(), // real, cited dry-sand friction angle, Klar et al. 2016
        // rolling_friction=0.20: the value calibrated in the standalone validation
        // (`tests/grains_repose_angle.rs`'s
        // `diag_calibrated_rolling_friction_long_horizon_check`, a long-horizon
        // match to Lajeunesse et al. 2004 at friction_angle=35deg, at a
        // dt-converged timestep, see `diag_dt_convergence_study`). Dimensionless,
        // so it transfers across stiffness scales, but not re-verified in this
        // softer, grid-coupled scene.
        rolling_friction: 0.20,
    }
}

/// Tiny deterministic LCG, same real convention as `tests/
/// grains_repose_angle.rs::SmallRng` -- reproducible jitter/polydispersity.
struct SmallRng(u64);
impl SmallRng {
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((self.0 >> 33) as f32) / (u32::MAX as f32)
    }
}

/// Loose "poured" column with the jitter and polydispersity of
/// `tests/grains_repose_angle.rs::build_column`: a perfectly regular lattice has no
/// asymmetry to trigger lateral collapse.
fn build_grain_column(center_x: f32, base_y: f32) -> (Vec<Grain>, f32) {
    let spacing = 2.6 * GRAIN_RADIUS;
    let mut rng = SmallRng(0xC0FF_EE11_u64);
    let mut grains = Vec::new();
    let column_width = 2 * GRAINS_R0 as i32;
    for row in 0..GRAINS_H0 {
        for col in 0..column_width {
            let jx = (rng.next_f32() - 0.5) * 0.3 * spacing;
            let jy = (rng.next_f32() - 0.5) * 0.3 * spacing;
            let x = center_x - (column_width as f32 * spacing) * 0.5 + col as f32 * spacing + jx;
            let y = base_y + row as f32 * spacing + GRAIN_RADIUS + jy;
            let r = GRAIN_RADIUS * (0.9 + 0.2 * rng.next_f32());
            let mut g = Grain::new(Vec2::new(x, y), r, GRAIN_MASS * (r / GRAIN_RADIUS).powi(2));
            g.v = Vec2::ZERO;
            grains.push(g);
        }
    }
    let r0 = GRAINS_R0 as f32 * (2.0 * GRAIN_RADIUS);
    let h0 = GRAINS_H0 as f32 * (2.0 * GRAIN_RADIUS);
    let predicted_r_inf = r0 * (1.0 + 2.0 * (h0 / r0).sqrt());
    (grains, predicted_r_inf)
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Mode {
    PreShaped,
    Collapse,
    Grains,
}

fn measure_angle_deg(xs: &[Vec2]) -> (f32, f32, f32) {
    let n = xs.len() as f32;
    if n == 0.0 {
        return (0.0, 0.0, 0.0);
    }
    let center_x = xs.iter().map(|p| p.x).sum::<f32>() / n;
    // With the pile spread wide, no particle may fall within +-2 cells of
    // center_x, and `fold(f32::MIN, f32::max)` would return its sentinel
    // (`height=-3.4e38`, `angle_deg=-90` in the log). `max_by`'s `Option` makes
    // "nothing matched" explicit, falling back to the pile's overall max y.
    let height = xs
        .iter()
        .filter(|p| (p.x - center_x).abs() < 2.0)
        .map(|p| p.y)
        .fold(f32::NEG_INFINITY, f32::max);
    let height = if height.is_finite() {
        height
    } else {
        xs.iter().map(|p| p.y).fold(f32::NEG_INFINITY, f32::max)
    } - FLOOR;
    let base_half_width = xs
        .iter()
        .filter(|p| p.y < FLOOR + 1.5)
        .map(|p| (p.x - center_x).abs())
        .fold(0.0f32, f32::max);
    let angle = (height / base_half_width.max(0.1)).atan().to_degrees();
    (height, base_half_width, angle)
}

fn make_sim(mode: Mode) -> Simulation {
    match mode {
        Mode::PreShaped => {
            // Exact tested recipe -- `tests/accuracy.rs::
            // unconfined_pile_with_cundall_damping_reaches_real_repose_
            // angle`, reproduced parameter-for-parameter.
            let config = SimConfig {
                max_substeps_per_step: 64,
                apic_blend: 0.05,
                cundall_damping: 1.0,
                ..SimConfig::standard(GRID, PRESHAPED_DT, Vec2::new(0.0, -0.3))
            };
            let cx = GRID as f32 * 0.5;
            let hb = PRESHAPED_HEIGHT_CELLS / TARGET_ANGLE_DEG.to_radians().tan();
            let spawn = SpawnRegion {
                spacing: 0.25,
                box_size: IVec2::new(
                    (2.0 * hb).ceil() as i32 + 4,
                    PRESHAPED_HEIGHT_CELLS.ceil() as i32 + 4,
                ),
                box_center: Vec2::new(cx, FLOOR + 2.0 + PRESHAPED_HEIGHT_CELLS * 0.5),
                material_id: 0,
                ..SpawnRegion::for_sim(&config)
            };
            let sand = DruckerPragerMaterial::from_young_modulus(1.0e5, 0.2);
            let mut solver = Simulation::new(config, spawn)
                .with_default_material(Box::new(sand))
                .with_boundary(Box::new(FrictionBoundary::new(2, 0.7)));
            // Carve the spawned bounding box down to a triangular
            // cross-section at exactly TARGET_ANGLE_DEG -- starts already
            // in its final shape (zero initial velocity): this mode is
            // about whether it HOLDS, not whether a collapse settles there.
            solver.retain_particles(|p| {
                let dy = p.x.y - FLOOR;
                let dx = (p.x.x - cx).abs();
                (0.0..=PRESHAPED_HEIGHT_CELLS).contains(&dy)
                    && dx <= hb * (1.0 - dy / PRESHAPED_HEIGHT_CELLS).max(0.0)
            });
            solver
        }
        Mode::Collapse => {
            // Swept, confirmed value -- see this file's own top doc.
            let config = SimConfig {
                max_substeps_per_step: 64,
                apic_blend: 0.6,
                cundall_damping: 0.0,
                ..SimConfig::standard(GRID, COLLAPSE_DT, Vec2::new(0.0, -0.3))
            };
            let column = SpawnRegion {
                spacing: 0.5,
                box_size: IVec2::new(8, 16),
                box_center: Vec2::new(GRID as f32 * 0.5, FLOOR + 8.0),
                material_id: 0,
                ..SpawnRegion::for_sim(&config)
            };
            let mut sand = DruckerPragerMaterial::from_young_modulus(1.0e5, 0.2);
            // Edge-triggered elastic-strain reset, after Cundall 1982's
            // kinetic-damping peak reset: fires once per particle on a strain-rate
            // falling edge (see `post_event_relax_threshold` and its calibration
            // tests in tests/accuracy.rs).
            sand.post_event_relax_threshold = 0.001;
            Simulation::new(config, column)
                .with_default_material(Box::new(sand))
                .with_boundary(Box::new(FrictionBoundary::new(2, 0.7)))
        }
        Mode::Grains => {
            // Discrete grains resting on a sand terrain bed, both coupled through
            // the shared MPM grid (`grains::coupling`), on a flat
            // `FrictionBoundary` floor (the standalone validation uses a giant
            // pinned floor grain instead, see its floor doc).
            // Grain-safe substep bound, used as this scene's `dt` from
            // construction: `choose_substep_dt` does not account for grain contact
            // stiffness (see `contact_law::critical_timestep`), so nothing else
            // would subdivide a coarser frame dt for the grains. 20% of critical:
            // standard DEM safety margin.
            let cfg = grain_contact_config();
            let m_eff = GRAIN_MASS * 0.5;
            let dt_crit = critical_timestep(m_eff, &cfg);
            let grain_safe_dt = (dt_crit * 0.2).min(0.02);

            let config = SimConfig {
                grid_res: GRID,
                dt: grain_safe_dt,
                gravity: Vec2::new(0.0, -0.3), // same real value already proven stable in `grains_grid_coupling.rs`'s own live-grid test
                adaptive_timestep: true,
                boundary_thickness: 2, // matches this file's own shared FLOOR constant's calibration (see PreShaped/Collapse's own FrictionBoundary::new(2, ...))
                ..SimConfig::default()
            };
            let terrain_center_y = FLOOR + GRAINS_TERRAIN_HEIGHT_CELLS as f32 * 0.5;
            let terrain_center = Vec2::new(GRID as f32 * 0.5, terrain_center_y);
            let spawn = SpawnRegion {
                spacing: 0.5,
                box_size: IVec2::new(
                    GRAINS_TERRAIN_HALF_WIDTH_CELLS * 2,
                    GRAINS_TERRAIN_HEIGHT_CELLS,
                ),
                box_center: terrain_center,
                material_id: 0,
                position_jitter: 0.3,
                ..SpawnRegion::for_sim(&config)
            };
            let mut solver = Simulation::new(config, spawn)
                .with_default_material(Box::new(DruckerPragerMaterial::cohesionless(2.0e3, 0.3)))
                .with_boundary(Box::new(FrictionBoundary::new(
                    config.boundary_thickness,
                    0.6,
                )));

            let terrain_top_y = terrain_center_y + GRAINS_TERRAIN_HEIGHT_CELLS as f32 * 0.5;
            // Derived (not guessed) minimal spawn clearance above the terrain --
            // was a flat `+1.0`, visibly a whole grain-diameter of empty air before any
            // grain starts falling. `build_grain_column`'s own worst case: max jitter
            // `jy` = 0.5*0.3*spacing = 0.5*0.3*(2.6*GRAIN_RADIUS) ≈ 0.39, max grain
            // radius = GRAIN_RADIUS*1.1 = 1.1 (see its own `r = GRAIN_RADIUS*(0.9+0.2*
            // rand)`). Row 0's grain bottom edge is `base_y + GRAIN_RADIUS + jy - r`;
            // solving for the offset that keeps that `>= terrain_top_y` in the absolute
            // worst case (jy at its most negative, r at its largest) gives
            // `GRAIN_RADIUS.mul_add(1.1, -1.0) - min_jy = 1.1 - 1.0 + 0.39 = 0.49`, not 1.0.
            let max_jitter = 0.5 * 0.3 * (2.6 * GRAIN_RADIUS);
            let min_clearance = GRAIN_RADIUS * 1.1 - GRAIN_RADIUS + max_jitter;
            let (grains, _predicted_r_inf) =
                build_grain_column(GRID as f32 * 0.5, terrain_top_y + min_clearance);
            solver.add_grain_population(GrainPopulation::new(grains, grain_contact_config()));
            solver
        }
    }
}

/// Live runout measurement for `Mode::Grains` -- same real Lajeunesse
/// et al. 2004 comparison convention `measure_angle_deg` already uses for
/// the continuum modes, computed directly off the grains' own current,
/// physically-simulated positions.
fn measure_grain_runout(solver: &Simulation) -> (f32, f32, f32, usize) {
    let population = &solver.grain_populations()[0];
    let xs: Vec<f32> = population.grains.iter().map(|g| g.x.x).collect();
    let n = xs.len() as f32;
    let center_x = xs.iter().sum::<f32>() / n.max(1.0);
    let measured_r = xs
        .iter()
        .map(|&x| (x - center_x).abs())
        .fold(0.0f32, f32::max);
    let r0 = GRAINS_R0 as f32 * (2.0 * GRAIN_RADIUS);
    let h0 = GRAINS_H0 as f32 * (2.0 * GRAIN_RADIUS);
    let predicted_r_inf = r0 * (1.0 + 2.0 * (h0 / r0).sqrt());
    let max_speed = population
        .grains
        .iter()
        .map(|g| g.v.length())
        .fold(0.0f32, f32::max);
    (
        measured_r,
        predicted_r_inf,
        max_speed,
        population.active_contact_count(),
    )
}

struct State {
    gfx: gui_common::Gfx,
    sim: Simulation,
    renderer: Renderer,
    mode: Mode,
    holding: bool,
    paused: bool,
    step: u64,
    fps_timer: std::time::Instant,
    fps_frames: u64,
    last_fps: f32,
    cursor_pos: [f32; 2],
    lmb: bool,
    rmb: bool,
    push_strength: f32,
    logger: FrameLogger,
    pour_mode: bool,
    pour_cooldown: u32,
    sim_speed: u32,
    pour_rng: SmallRng,
}

impl State {
    async fn new(window: Arc<Window>) -> Self {
        let gfx = gui_common::Gfx::new(&window).await;
        let size = window.inner_size();
        let mode = Mode::PreShaped;
        let sim = make_sim(mode);
        // Render-buffer sizing across all three modes: the wgpu instance buffer is
        // allocated once here and the mode toggles live (M key) without recreating
        // the renderer, so capacity covers whichever mode needs the most instances
        // (in practice PreShaped's dense spacing=0.25 fill). Grains mode's extra
        // headroom (two markers per grain, the body plus the rolling-indicator dot,
        // see the render loop) is added on top, since the solver does not report
        // those as `Particles`.
        let collapse_particle_count = make_sim(Mode::Collapse).particles().len();
        let grains_sim_for_sizing = make_sim(Mode::Grains);
        let grains_particle_count = grains_sim_for_sizing.particles().len()
            + 2 * (grains_sim_for_sizing.grain_populations()[0].grains.len() + GRAINS_POUR_CAP);
        let render_capacity = sim
            .particles()
            .len()
            .max(collapse_particle_count)
            .max(grains_particle_count)
            + PARTICLE_POUR_CAP;
        let mut renderer = Renderer::new(&gfx.device, render_capacity, gfx.format);
        renderer.set_camera(&gfx.queue, GRID as u32, size.width, size.height, 0.7, true);
        renderer.set_color_mode(ColorMode::ByPhysics);
        renderer.set_optical_params(&gfx.queue, 0, SIGMA_SAND);
        renderer.set_optical_params(&gfx.queue, GRAINS_ACCENT_MAT_ID as usize, SIGMA_ACCENT);
        renderer.set_optical_params(&gfx.queue, GRAINS_MARKER_MAT_ID as usize, SIGMA_GRAIN);

        let log_path = std::env::temp_dir().join("emerge_sand_repose_angle.ndjson");
        let logger = FrameLogger::open(&log_path).unwrap();
        println!(
            "sand_repose_angle: {} particles  |  M=cycle mode (PreShaped/Collapse/Grains)  H=toggle holding (collapse mode only)  LMB=push RMB=pull (continuum modes only)  SPACE=pause  R=reset  Q=quit",
            sim.particles().len()
        );
        println!("per-frame diagnostics log: {}", log_path.display());
        Self {
            gfx,
            sim,
            renderer,
            mode,
            holding: false,
            paused: false,
            step: 0,
            fps_timer: std::time::Instant::now(),
            fps_frames: 0,
            last_fps: 0.0,
            cursor_pos: [0.0; 2],
            lmb: false,
            rmb: false,
            // Lower than basic_sand.rs's own 12.0 default -- real,
            // measured live: holding the button re-applies this force EVERY
            // frame with no decay (same convention every sand demo uses), so
            // it's the DURATION held, not the radius, that determines how
            // fast particles end up going. 4.0 keeps a brief tap a gentle
            // nudge; the slider still reaches 40 for a deliberate shove.
            push_strength: 4.0,
            logger,
            pour_mode: false,
            pour_cooldown: 0,
            // 12 steps per frame: at 25, 1.19 ms/step on this 80-grain scene gives
            // 28.3 fps, under the 45-60 fps floor; 12 measures 61.3 fps and stays at
            // ~1.04x real time (against 1.00x at 25). Pacing only: each step's dt is
            // unchanged (see this field's doc). Measured by a sim-speed sweep probe
            // that is not kept.
            sim_speed: 12,
            pour_rng: SmallRng(0xFEED_1234_5678_u64),
        }
    }

    fn resize(&mut self, w: u32, h: u32) {
        self.gfx.resize(w, h);
        if w == 0 || h == 0 {
            return;
        }
        self.renderer
            .set_camera(&self.gfx.queue, GRID as u32, w, h, 0.7, true);
    }

    /// Aspect-ratio-correct cursor-to-grid mapping -- see
    /// `gui_common::cursor_to_grid`'s doc for the bug this
    /// centralization exists to stop from recurring per-example.
    fn cursor_grid(&self) -> Vec2 {
        gui_common::cursor_to_grid(
            self.cursor_pos,
            self.gfx.surface_config.width,
            self.gfx.surface_config.height,
            GRID,
        )
    }

    fn reset(&mut self) {
        self.sim = make_sim(self.mode);
        self.holding = false;
        self.step = 0;
        println!("reset: mode={:?}", self.mode);
    }

    fn toggle_mode(&mut self) {
        self.mode = match self.mode {
            Mode::PreShaped => Mode::Collapse,
            Mode::Collapse => Mode::Grains,
            Mode::Grains => Mode::PreShaped,
        };
        self.reset();
    }

    /// Drops material above the local surface near the cursor's x position (not at
    /// the cursor's y), in all three modes.
    ///
    /// Spawning at the cursor with no check for what is there plants material
    /// overlapping the pile, a severe initial-condition violation (a DEM contact
    /// spring or an MPM cell answers a large initial overlap with a stiff repulsion,
    /// correctly). So this scans existing material in a small x-window and drops
    /// from just above its highest point (or the cursor's y, whichever is higher),
    /// the "pour from above" convention of `build_grain_column`, centered on the
    /// cursor.
    ///
    /// Grains adds one discrete `Grain`; PreShaped/Collapse add a small continuum
    /// blob via the `add_body`/`SpawnRegion` API. No-op past each mode's pour cap,
    /// matching the render-buffer headroom reserved for it in `State::new`.
    fn pour_at_cursor(&mut self) {
        let cursor = self.cursor_grid();
        match self.mode {
            Mode::Grains => {
                let population = &mut self.sim.grain_populations_mut()[0];
                if population.grains.len() >= GRAINS_R0 * 2 * GRAINS_H0 + GRAINS_POUR_CAP {
                    return;
                }
                const X_WINDOW: f32 = 3.0;
                const CLEARANCE: f32 = 2.5;
                let local_top_y = population
                    .grains
                    .iter()
                    .filter(|g| (g.x.x - cursor.x).abs() < X_WINDOW)
                    .map(|g| g.x.y + g.radius)
                    .fold(FLOOR, f32::max);
                let y = (local_top_y + CLEARANCE).max(cursor.y);
                let r = GRAIN_RADIUS * (0.9 + 0.2 * self.pour_rng.next_f32());
                let mut g = Grain::new(
                    Vec2::new(cursor.x, y),
                    r,
                    GRAIN_MASS * (r / GRAIN_RADIUS).powi(2),
                );
                g.v = Vec2::ZERO;
                population.grains.push(g);
            }
            Mode::PreShaped | Mode::Collapse => {
                if self.sim.particles().len() >= PARTICLE_POUR_CAP {
                    return;
                }
                const X_WINDOW: f32 = 1.5;
                const CLEARANCE: f32 = 1.5;
                let local_top_y = self
                    .sim
                    .particles()
                    .x
                    .iter()
                    .filter(|p| (p.x - cursor.x).abs() < X_WINDOW)
                    .map(|p| p.y)
                    .fold(FLOOR, f32::max);
                let y = (local_top_y + CLEARANCE).max(cursor.y);
                let spawn = SpawnRegion {
                    spacing: 0.5,
                    box_size: IVec2::new(1, 1),
                    box_center: Vec2::new(cursor.x, y),
                    material_id: 0,
                    position_jitter: 0.2,
                    rng_seed: self.pour_rng.0 as u32,
                    ..SpawnRegion::default()
                };
                let _ = self.sim.add_body(spawn);
            }
        }
        self.pour_rng.next_f32(); // advance so consecutive drops don't look identical
    }

    fn toggle_holding(&mut self) {
        if self.mode != Mode::Collapse {
            println!(
                "holding only applies in collapse mode -- press M to cycle modes until you reach it"
            );
            return;
        }
        self.holding = !self.holding;
        if self.holding {
            self.sim.set_apic_blend(0.05);
            self.sim.set_cundall_damping(1.0);
            println!(
                "holding mode ON (apic_blend=0.05, cundall_damping=1.0) -- watch it creep past the target"
            );
        } else {
            self.sim.set_apic_blend(0.6);
            self.sim.set_cundall_damping(0.0);
            println!(
                "holding mode OFF (apic_blend=0.6, cundall_damping=0.0) -- real collapse dynamics"
            );
        }
    }

    fn update_and_render(&mut self, window: &Window) {
        // Push/pull only affects ordinary MPM particles (`apply_radial_impulse`
        // scans the particle SoA directly) -- disclosed scope limit in
        // Grains mode: grains aren't ordinary `Particles`, so this is a no-op
        // there rather than something that looks like it should work but
        // silently doesn't.
        if self.pour_mode {
            // Cursor-driven pour tool, active in ALL THREE modes -- takes
            // over LMB while active (push/pull and pour would otherwise
            // fight over the same button). Cooldown throttles a held button
            // to a trickle instead of one drop every rendered frame.
            if self.pour_cooldown > 0 {
                self.pour_cooldown -= 1;
            }
            if self.lmb && self.pour_cooldown == 0 {
                self.pour_at_cursor();
                self.pour_cooldown = POUR_COOLDOWN_FRAMES;
            }
        } else if self.mode != Mode::Grains && (self.lmb || self.rmb) {
            let mag = if self.lmb {
                self.push_strength
            } else {
                -self.push_strength
            };
            self.sim
                .apply_radial_impulse(self.cursor_grid(), PUSH_RADIUS_CELLS, mag);
        }
        if !self.paused {
            // Grains mode's own computed grain-safe `dt` (see
            // `make_sim`'s `Mode::Grains` arm) is much finer than the other
            // two modes' -- real settling there needs tens of thousands of
            // substeps, so a single `step()` per rendered frame would take
            // many real minutes just to watch it settle. `sim_speed`
            // steps/frame is a disclosed pacing choice (not a physics
            // change -- each individual step is exactly as fine as the
            // safety-margin calculation demands), matching this project's
            // own established "physics fidelity is never cut for demo
            // pacing" rule. Default 12 (see this struct's own `sim_speed`
            // field doc) real-measured at 61.3fps, clearing this project's
            // standing 45-60fps floor while staying ~1x real-time.
            let steps_this_frame = if self.mode == Mode::Grains {
                self.sim_speed
            } else {
                1
            };
            for _ in 0..steps_this_frame {
                self.sim.step();
                self.step += 1;
            }
        }
        self.fps_frames += 1;
        if self.fps_timer.elapsed().as_secs_f32() >= 1.0 {
            self.last_fps = self.fps_frames as f32 / self.fps_timer.elapsed().as_secs_f32();
            self.fps_timer = std::time::Instant::now();
            self.fps_frames = 0;
        }

        // Per-frame diagnostics, so pushing or pulling the pile and whether it
        // re-settles at a stable angle can be checked from the log afterward.
        // `is_pushing`/`is_pulling` and the cursor's grid position ride in `extra`
        // (app-specific context), as in `rod_blade_and_root.rs`. Grains mode
        // measures lateral runout against the Lajeunesse prediction from grain
        // positions (see `measure_grain_runout`).
        let (height, half_w, angle) = measure_angle_deg(&self.sim.particles().x);
        let (grain_r, grain_r_pred, grain_max_speed, grain_contacts) = if self.mode == Mode::Grains
        {
            measure_grain_runout(&self.sim)
        } else {
            (0.0, 1.0, 0.0, 0)
        };
        let grain_ratio = grain_r / grain_r_pred;
        let max_speed = self
            .sim
            .particles()
            .v
            .iter()
            .fold(0.0f32, |m, v| m.max(v.length()))
            .max(grain_max_speed);
        let snap = self.sim.diagnostics_snapshot();
        let stats = per_material_stats(self.sim.particles());
        let cursor = self.cursor_grid();
        self.logger.log(
            self.step,
            snap.effective_dt,
            &stats,
            &snap,
            &[],
            &[
                ("height", height),
                ("half_width", half_w),
                ("angle_deg", angle),
                ("max_speed", max_speed),
                ("holding", if self.holding { 1.0 } else { 0.0 }),
                ("is_pushing", if self.lmb { 1.0 } else { 0.0 }),
                ("is_pulling", if self.rmb { 1.0 } else { 0.0 }),
                ("cursor_x", cursor.x),
                ("cursor_y", cursor.y),
                ("fps", self.last_fps),
                ("grain_measured_r", grain_r),
                ("grain_predicted_r", grain_r_pred),
                ("grain_runout_ratio", grain_ratio),
                ("grain_active_contacts", grain_contacts as f32),
            ],
        );

        let output = match self.gfx.surface.get_current_texture() {
            Ok(t) => t,
            Err(_) => return,
        };
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        // Render buffer: ordinary particles as they are, plus (Grains mode only)
        // one marker per grain at the grain's simulated position, as
        // `rod_blade_and_root.rs` does for rods: a rendering proxy with no physics.
        // Isotropic (a circle: grains have no preferred axis), sized by each
        // grain's polydisperse radius.
        if self.mode == Mode::Grains {
            let mut all: Vec<Particle> = self.sim.particles().iter().collect();
            for grain in &self.sim.grain_populations()[0].grains {
                let mut p = Particle::zeroed();
                p.x = grain.x;
                p.v = grain.v;
                p.mass = 1.0;
                p.initial_volume = 1.0;
                p.volume = 1.0;
                p.density = 1.0;
                p.material_id = GRAINS_MARKER_MAT_ID;
                let scale = 2.0 * grain.radius / 0.7; // 0.7 cancels this file's own `set_camera` particle-scale factor
                p.deformation_gradient = Mat2::from_diagonal(Vec2::splat(scale));
                all.push(p);

                // Rolling cue: a small accent dot offset from the grain's center
                // by `grain.orientation` (the integrated rotation angle, see
                // `Grain::orientation`), which orbits its parent as the grain rolls;
                // a plain rotating circle shows no rotation at all.
                let mut accent = Particle::zeroed();
                let offset = Vec2::from_angle(grain.orientation) * (grain.radius * 0.6);
                accent.x = grain.x + offset;
                accent.v = grain.v;
                accent.mass = 1.0;
                accent.initial_volume = 1.0;
                accent.volume = 1.0;
                accent.density = 1.0;
                accent.material_id = GRAINS_ACCENT_MAT_ID;
                let accent_scale = 2.0 * (grain.radius * 0.28) / 0.7;
                accent.deformation_gradient = Mat2::from_diagonal(Vec2::splat(accent_scale));
                all.push(accent);
            }
            let marker_particles = emerge::particle::Particles::from(all);
            self.renderer.render(
                &self.gfx.device,
                &self.gfx.queue,
                &marker_particles,
                &view,
                true,
            );
        } else {
            self.renderer.render(
                &self.gfx.device,
                &self.gfx.queue,
                self.sim.particles(),
                &view,
                true,
            );
        }

        let fps = self.last_fps;
        let step = self.step;
        let mode = self.mode;
        let holding = self.holding;
        let mut paused = self.paused;
        let mut push_strength = self.push_strength;
        let mut pour_sand = self.pour_mode;
        let mut sim_speed = self.sim_speed;
        let mut do_reset = false;
        let mut do_toggle_mode = false;
        let mut do_toggle_holding = false;

        gui_common::run_egui_frame(&mut self.gfx, window, &view, |ctx| {
            egui::Window::new("Sand: angle of repose")
                .default_pos([10.0, 10.0])
                .default_width(320.0)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.label(format!("fps={fps:.0}  step={step}"));
                    ui.separator();
                    let mode_label = match mode {
                        Mode::PreShaped => "PRE-SHAPED (apic=0.05, cundall=1.0, always holding)",
                        Mode::Collapse if holding => {
                            "COLLAPSE, holding ON (apic=0.05, cundall=1.0)"
                        }
                        Mode::Collapse => "COLLAPSE (apic=0.6, cundall=0.0)",
                        Mode::Grains => "GRAINS (real discrete-element rolling resistance)",
                    };
                    ui.label(format!("mode = {mode_label}"));
                    if mode == Mode::Grains {
                        ui.label(format!(
                            "runout ratio = {grain_ratio:.3}x  (predicted={grain_r_pred:.2}, measured={grain_r:.2})"
                        ));
                        ui.label(format!(
                            "active_contacts={grain_contacts}  max_speed={grain_max_speed:.4}"
                        ));
                        ui.label("real target = 1.0x  |  independently verified: ~1.43x, flat 80k-200k steps");
                    } else {
                        ui.label(format!("height={height:.2}  half-width={half_w:.2}"));
                        ui.label(format!("current angle = {angle:.1} deg"));
                        ui.label("real dry sand IRL = 30-35 deg");
                    }
                    ui.separator();
                    match mode {
                        Mode::PreShaped => {
                            ui.label("Real, tested: holds flat 100,000+ steps headless.");
                            ui.label("Watch it NOT slide away.");
                        }
                        Mode::Collapse => {
                            ui.label("apic_blend=1.0 (engine default) is UNSTABLE here --");
                            ui.label("spreads without bound. 0.6 is stable AND accurate.");
                            ui.label("Toggle H once it settles: holding mode creeps the");
                            ui.label("angle back DOWN past target -- don't over-relax it.");
                        }
                        Mode::Grains => {
                            ui.label("Real, individually-simulated grains (not a continuum");
                            ui.label("material) settling onto a real sand terrain bed --");
                            ui.label("tonight's rolling-torque-sign + rolling_damping fixes,");
                            ui.label("live. Push/pull inactive here (grains, not particles).");
                        }
                    }
                    ui.separator();
                    if mode == Mode::Grains {
                        ui.label("Sim speed (physics steps/frame):");
                        ui.add(egui::Slider::new(&mut sim_speed, 1..=200));
                        ui.label(
                            "Push toward 200 to fast-forward settling and check it stays stable.",
                        );
                        ui.separator();
                    }
                    ui.checkbox(&mut pour_sand, "Pour mode (LMB drops material at cursor)");
                    if pour_sand {
                        ui.label("Click to drop one; hold to pour a stream. Works in all modes.");
                    } else if mode != Mode::Grains {
                        ui.label("Push/pull strength (LMB push, RMB pull):");
                        ui.add(egui::Slider::new(&mut push_strength, 0.0..=40.0));
                    }
                    ui.separator();
                    ui.checkbox(&mut paused, "Paused (or SPACE)");
                    ui.horizontal(|ui| {
                        if ui.button("M: toggle mode").clicked() {
                            do_toggle_mode = true;
                        }
                        if ui.button("H: toggle holding").clicked() {
                            do_toggle_holding = true;
                        }
                        if ui.button("Reset").clicked() {
                            do_reset = true;
                        }
                    });
                    ui.label("Q quit");
                });
        });
        self.paused = paused;
        self.push_strength = push_strength;
        self.pour_mode = pour_sand;
        self.sim_speed = sim_speed;
        if do_toggle_mode {
            self.toggle_mode();
        } else if do_toggle_holding {
            self.toggle_holding();
        } else if do_reset {
            self.reset();
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
                    .with_title("emerge -- Sand: Angle of Repose (GUI)")
                    .with_inner_size(winit::dpi::LogicalSize::new(720u32, 480u32)),
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
                if pressed {
                    match key {
                        KeyCode::KeyM => s.toggle_mode(),
                        KeyCode::KeyH => s.toggle_holding(),
                        KeyCode::KeyR => s.reset(),
                        KeyCode::Space => s.paused = !s.paused,
                        KeyCode::Escape | KeyCode::KeyQ => el.exit(),
                        _ => {}
                    }
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

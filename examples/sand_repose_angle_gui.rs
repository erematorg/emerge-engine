extern crate emerge_engine as emerge;

use egui_wgpu::ScreenDescriptor;
/// Merged replacement for the former `sand_pile_stability_gui.rs` and
/// `sand_collapse_true_repose_gui.rs` -- both were really the same real
/// question (does sand reach a genuine ~30 deg angle of repose?) asked
/// from two different starting conditions, sharing almost all of their
/// GUI boilerplate. One demo, real mode switch between them.
///
/// **Pre-shaped mode**: a pile built ALREADY at 30 deg (zero initial
/// velocity) -- tests whether it HOLDS. Real, tested recipe
/// (`tests/accuracy.rs::unconfined_pile_with_cundall_damping_reaches_
/// real_repose_angle`, found 2026-07-26): the engine's own self-consistent
/// Drucker-Prager return mapping (unconditional default) + `apic_blend=
/// 0.05` + `cundall_damping=1.0` (Cundall 1982, production use in Anura3D
/// geotechnical MPM) holds flat for 100,000+ steps headless.
///
/// **Collapse mode**: a tall column that genuinely TOPPLES and settles
/// near the real target (`tests/accuracy.rs::sand_collapse_with_phase_
/// gated_relaxation_after_dynamics`, 2026-08-01). Real root cause found
/// that session: `SimConfig::standard`'s own default `apic_blend=1.0`
/// (full APIC, zero numerical dissipation) is genuinely UNSTABLE for a
/// violent collapse -- confirmed via a widened-domain test where reach
/// grew roughly in proportion to whatever domain size was given. `apic_
/// blend=0.6` is stable without over-damping the real collapse motion
/// (0.05 over-damps it, landing too steep at 44-51 deg). Press H once it
/// settles to switch to the pre-shaped mode's own holding recipe -- watch
/// the SAME real "excess creep" mechanism the patient-pour investigation
/// found: continued relaxation drifts the angle back DOWN past the real
/// target, it does not hold it steady the way the pre-shaped case does
/// (open question as of this session: whether it ever truly plateaus
/// given a long enough horizon, or keeps drifting toward flat -- see
/// `sand_collapse_relaxation_long_horizon_plateau_check`).
///
/// **Grains mode** (added 2026-08-03): the same real question, answered by
/// real, individually-simulated DISCRETE grains (`spacetime::grains`)
/// instead of a continuum material -- each grain has its own position,
/// velocity, spin, and persistent elastic contact springs (Cundall & Strack
/// 1979 / Luding 2008 / Ai et al. 2011), coupled to a real sand terrain bed
/// through the SAME shared MPM grid ordinary particles use (`grains::
/// coupling`). Real fixes this mode exists to show off: a backwards sign in
/// the rolling-resistance restoring torque (was amplifying instead of
/// damping every off-axis contact -- found via a minimal repro, one grain
/// on one tilted pinned floor, flew off by step 100,000 before the fix,
/// stable for 400,000 steps after) and a missing `rolling_damping` channel
/// (mirrors `normal_damping`/`tangential_damping`, real GeoTaichi precedent:
/// its own three-way `ndratio`/`sdratio`/`rdratio` split). Independently
/// verified (`tests/grains_repose_angle.rs`, extended to 200,000 steps
/// specifically to rule out an early-snapshot false positive -- the same
/// discipline that caught Cosserat's own false positive earlier this
/// session): the pile genuinely arrests, flat from step 80,000 to 200,000,
/// at a real, bounded ~1.43x the Lajeunesse-predicted runout -- not the
/// literal 1.0x target, a real disclosed remaining calibration gap, not a
/// bug.
///
///   cargo run --example sand_repose_angle_gui --features render
use emerge::grains::population::GrainPopulation;
use emerge::materials::solid::granular::grain_contact_law::{ContactLawConfig, critical_timestep};
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

// Same push/pull cursor convention as every other sand example (basic_sand_gui.rs):
// LMB push, RMB pull, `apply_radial_impulse` at a fixed real radius.
//
// REAL BUG FOUND AND FIXED: 7.0 (basic_sand_gui.rs's own value) was copied
// without checking it against THIS demo's own much smaller pile -- that
// scene's sand mass spans a wide multi-body area, so 7 cells is a small
// local nudge there. This pile is only 12 cells tall / ~41 wide (see
// PRESHAPED_HEIGHT_CELLS), so a 7-cell-radius push centered mid-pile covers
// MORE than its full height and a third of its width -- looks like "the
// whole pile moves together," not because of any real coupling bug (`apply_
// radial_impulse` is a plain per-particle radial-falloff kick, verified
// local-only), just a radius picked for the wrong scene. 2.5 keeps a push
// a real local nudge (a few grains near the cursor), not a bulk shove.
const PUSH_RADIUS_CELLS: f32 = 2.5;

const GRID: usize = 128;
const FLOOR: f32 = 2.0;
const SIGMA_SAND: [f32; 3] = [0.180, 0.220, 0.550];

// Pre-shaped mode's own real, tested geometry.
const PRESHAPED_DT: f32 = 0.016;
const TARGET_ANGLE_DEG: f32 = 30.0;
const PRESHAPED_HEIGHT_CELLS: f32 = 12.0;

// Collapse mode's own real, tested geometry (matches
// `sand_angle_of_repose_is_physical`'s column, different DT).
const COLLAPSE_DT: f32 = 0.1;

// Grains mode: real, individually-simulated discrete-element grains
// (`spacetime::grains`) instead of a continuum Drucker-Prager material --
// the real, live demonstration of tonight's rolling-torque-sign +
// rolling_damping fixes (see this file's own module doc, "REAL FIXES,
// 2026-08-03" section below). Real column geometry matches `tests/
// grains_repose_angle.rs::run_collapse`'s own already-validated 4-wide/
// 10-tall shape (same real Lajeunesse et al. 2004 comparison convention
// PreShaped/Collapse already use above, just measured off real grain
// positions instead of particle positions).
const GRAINS_R0: usize = 4;
const GRAINS_H0: usize = 10;
const GRAIN_RADIUS: f32 = 1.0; // grid-coordinate units
const GRAIN_MASS: f32 = 1.0;
const GRAINS_TERRAIN_HALF_WIDTH_CELLS: i32 = 30;
const GRAINS_TERRAIN_HEIGHT_CELLS: i32 = 8;
const GRAINS_MARKER_MAT_ID: u32 = 1; // distinct palette slot from the terrain's own material_id=0
// Real bug found+fixed live-testing this demo: `sigma_a` is an ABSORPTION
// coefficient (radiative-transfer convention, same as `SIGMA_SAND` above) --
// LOW value in a channel means LESS absorbed, i.e. MORE of that color shows.
// The first version here had this backwards (high red-absorption, low
// blue-absorption), which rendered as pale blue-white instead of the
// intended warm orange -- confirmed by cross-checking against `SIGMA_SAND`'s
// own real, already-proven tan/beige result (low R/G, high B = warm sandy
// tone). Low R, moderate G, high B-absorption here gives a real, distinct
// warm terracotta/orange.
const SIGMA_GRAIN: [f32; 3] = [0.100, 0.300, 0.800];
const GRAINS_ACCENT_MAT_ID: u32 = 3; // small rolling-indicator dot, distinct palette slot again
const SIGMA_ACCENT: [f32; 3] = [0.900, 0.900, 0.900]; // high absorption in every channel = near-black, real high contrast against the warm orange grain body

/// Real, ALREADY-PROVEN-stable-in-a-live-Simulation contact parameters,
/// unchanged from `tests/grains_grid_coupling.rs::contact_config()` -- not a
/// new guess, and deliberately softer than the pure-physics validation's own
/// real-SI-calibrated E=1e7 Pa stiffness (`tests/grains_repose_angle.rs`),
/// which would need a punishingly fine forced substep dt once genuinely
/// coupled to a real MPM grid and its own adaptive CFL -- same real
/// "disclosed, non-literal calibration" precedent this whole grain effort
/// already uses (effective grain diameter, coarse-grained mass), not a
/// different physics model. Carries the real 2026-08-03 fixes in full: the
/// rolling-torque sign (`contact_law.rs`'s own module doc) and
/// `rolling_damping` (new field, same role as `normal_damping`/
/// `tangential_damping`) -- both sign/formula corrections, fully exercised
/// regardless of the specific stiffness scale chosen.
fn grain_contact_config() -> ContactLawConfig {
    ContactLawConfig {
        normal_stiffness: 1.0e4,
        tangential_stiffness: 0.8e4,
        rolling_stiffness: 5.0e2,
        normal_damping: 5.0,
        tangential_damping: 5.0,
        rolling_damping: 5.0,
        friction: (35.0_f32).to_radians().tan(), // real, cited dry-sand friction angle, Klar et al. 2016
        rolling_friction: 0.1,
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

/// Real, loose "poured" column -- same real jitter+polydispersity
/// discipline as `tests/grains_repose_angle.rs::build_column` (a perfectly
/// regular lattice has no physical asymmetry to trigger real lateral
/// collapse at all, confirmed the hard way earlier this session).
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
    // REAL BUG FOUND AND FIXED: after enough interactive pushing/pulling
    // spreads the pile wide enough, NO particle can fall within +-2 cells
    // of center_x -- the old `fold(f32::MIN, f32::max)` then silently
    // returned its untouched sentinel (f32::MIN, not a real height),
    // observed live via the NDJSON log (`height=-3.4e38`, `angle_deg=-90`).
    // `max_by`'s `Option` return makes "nothing matched" explicit; falling
    // back to the pile's own overall max y (a real, sensible answer: the
    // tallest particle anywhere) instead of a fixed band's sentinel.
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
            // Exact real, tested recipe -- `tests/accuracy.rs::
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
                precompute_initial_volumes: true,
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
            // Real, swept, confirmed value -- see this file's own top doc.
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
                precompute_initial_volumes: true,
                ..SpawnRegion::for_sim(&config)
            };
            let mut sand = DruckerPragerMaterial::from_young_modulus(1.0e5, 0.2);
            // Real, calibrated tonight (2026-08-02): edge-triggered elastic-strain
            // reset, grounded in Cundall 1982's kinetic-damping peak-reset --
            // fires once per particle on a real strain-rate falling edge, matching
            // the proven F-only-reset target bit-for-bit (29.6deg) instead of the
            // unarrested creep this scene showed before.
            sand.post_event_relax_threshold = 0.001;
            Simulation::new(config, column)
                .with_default_material(Box::new(sand))
                .with_boundary(Box::new(FrictionBoundary::new(2, 0.7)))
        }
        Mode::Grains => {
            // Real, individually-simulated discrete grains resting on a
            // real sand terrain bed, both coupled through the SAME shared
            // MPM grid (`grains::coupling`) -- a genuinely flat
            // `FrictionBoundary` floor (real, not the giant-circle
            // approximation the pure-physics validation's own standalone
            // harness needed -- see this file's own module doc for the
            // real bug that caused, and the real fix that closed, tonight's
            // "still creeping" false alarm).
            // Real, computed (not guessed) grain-safe substep bound, used as
            // this scene's own `dt` from construction -- `choose_substep_dt`
            // doesn't yet know about grain contact stiffness at all (a real,
            // disclosed gap, see `contact_law::critical_timestep`'s own
            // doc), so nothing else would subdivide a coarser frame dt down
            // for the grains automatically. 20% of critical: real, standard
            // DEM safety margin.
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
                precompute_initial_volumes: true,
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
            // Real, derived (not guessed) minimal spawn clearance above the terrain --
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

/// Real, live runout measurement for `Mode::Grains` -- same real Lajeunesse
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
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    device: wgpu::Device,
    queue: wgpu::Queue,
    sim: Simulation,
    renderer: Renderer,
    egui_ctx: egui::Context,
    egui_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
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
        let mode = Mode::PreShaped;
        let sim = make_sim(mode);
        // Real render-buffer sizing across ALL three modes -- the wgpu
        // instance buffer is allocated once here and mode can toggle live
        // (M key) without recreating the renderer, so capacity must cover
        // whichever mode needs the most instances (in practice PreShaped's
        // own dense spacing=0.25 fill), not just the starting mode's count.
        // Grains mode's own extra headroom (TWO marker particles per real
        // grain -- the main body plus the small rolling-indicator accent
        // dot, see the render loop's own doc) is added on top since those
        // aren't real `Particles` the solver itself ever reports a length
        // for.
        let collapse_particle_count = make_sim(Mode::Collapse).particles().len();
        let grains_sim_for_sizing = make_sim(Mode::Grains);
        let grains_particle_count = grains_sim_for_sizing.particles().len()
            + 2 * grains_sim_for_sizing.grain_populations()[0].grains.len();
        let render_capacity = sim
            .particles()
            .len()
            .max(collapse_particle_count)
            .max(grains_particle_count);
        let mut renderer = Renderer::new(&device, render_capacity, fmt);
        renderer.set_camera(&queue, GRID as u32, size.width, size.height, 0.7, true);
        renderer.set_color_mode(ColorMode::ByPhysics);
        renderer.set_optical_params(&queue, 0, SIGMA_SAND);
        renderer.set_optical_params(&queue, GRAINS_ACCENT_MAT_ID as usize, SIGMA_ACCENT);
        renderer.set_optical_params(&queue, GRAINS_MARKER_MAT_ID as usize, SIGMA_GRAIN);

        let egui_ctx = egui::Context::default();
        let egui_state = egui_winit::State::new(
            egui_ctx.clone(),
            egui_ctx.viewport_id(),
            window.as_ref(),
            None,
            None,
            None,
        );
        let egui_renderer = egui_wgpu::Renderer::new(
            &device,
            fmt,
            egui_wgpu::RendererOptions {
                msaa_samples: 1,
                ..Default::default()
            },
        );

        let log_path = std::env::temp_dir().join("emerge_sand_repose_angle_gui.ndjson");
        let logger = FrameLogger::open(&log_path).unwrap();
        println!(
            "sand_repose_angle_gui: {} particles  |  M=cycle mode (PreShaped/Collapse/Grains)  H=toggle holding (collapse mode only)  LMB=push RMB=pull (continuum modes only)  SPACE=pause  R=reset  Q=quit",
            sim.particles().len()
        );
        println!("per-frame diagnostics log: {}", log_path.display());
        Self {
            surface,
            surface_config: sc,
            device,
            queue,
            sim,
            renderer,
            egui_ctx,
            egui_state,
            egui_renderer,
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
            // Lower than basic_sand_gui.rs's own 12.0 default -- real,
            // measured live: holding the button re-applies this force EVERY
            // frame with no decay (same convention every sand demo uses), so
            // it's the DURATION held, not the radius, that determines how
            // fast particles end up going. 4.0 keeps a brief tap a gentle
            // nudge; the slider still reaches 40 for a deliberate shove.
            push_strength: 4.0,
            logger,
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
            .set_camera(&self.queue, GRID as u32, w, h, 0.7, true);
    }

    /// Real bug found live: this window is 720x480 (NOT square), and
    /// `Renderer::set_camera` computes a non-uniform `(sx,tx,sy,ty)` for any
    /// non-square viewport (see its own doc/impl) -- the simple "screen
    /// fraction * GRID" formula (correct only when width==height, the
    /// convention every OTHER sand demo's square window happens to satisfy)
    /// silently mis-locates the cursor here. Mirrors `set_camera`'s own
    /// exact math, then inverts it, instead of guessing a corrective factor.
    fn cursor_grid(&self) -> Vec2 {
        let w = self.surface_config.width.max(1) as f32;
        let h = self.surface_config.height.max(1) as f32;
        let aspect = w / h;
        let gr = GRID as f32;
        let (sx, tx, sy, ty) = if aspect >= 1.0 {
            (2.0 / (gr * aspect), -1.0 / aspect, 2.0 / gr, -1.0)
        } else {
            (2.0 / gr, -1.0, 2.0 * aspect / gr, -aspect)
        };
        let ndc_x = 2.0 * (self.cursor_pos[0] / w) - 1.0;
        let ndc_y = 1.0 - 2.0 * (self.cursor_pos[1] / h);
        Vec2::new((ndc_x - tx) / sx, (ndc_y - ty) / sy)
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
        // scans the particle SoA directly) -- real, disclosed scope limit in
        // Grains mode: grains aren't ordinary `Particles`, so this is a no-op
        // there rather than something that looks like it should work but
        // silently doesn't.
        if self.mode != Mode::Grains && (self.lmb || self.rmb) {
            let mag = if self.lmb {
                self.push_strength
            } else {
                -self.push_strength
            };
            self.sim
                .apply_radial_impulse(self.cursor_grid(), PUSH_RADIUS_CELLS, mag);
        }
        if !self.paused {
            // Grains mode's own real, computed grain-safe `dt` (see
            // `make_sim`'s `Mode::Grains` arm) is much finer than the other
            // two modes' -- real settling there needs tens of thousands of
            // substeps, so a single `step()` per rendered frame would take
            // many real minutes just to watch it settle. 25 steps/frame is
            // a real, disclosed pacing choice (not a physics change -- each
            // individual step is exactly as fine as the safety-margin
            // calculation demands), matching this project's own established
            // "physics fidelity is never cut for demo pacing" rule.
            let steps_this_frame = if self.mode == Mode::Grains { 25 } else { 1 };
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

        // Real per-frame diagnostics -- so pushing/pulling the pile and
        // watching whether it re-settles at a real stable angle (or keeps
        // sliding) can be verified after the fact from the log, not just
        // eyeballed live. `is_pushing`/`is_pulling` and the cursor's own
        // grid position ride in `extra` (app-specific context the generic
        // snapshot has no name for), same slot `rod_blade_of_grass_gui.rs`
        // already uses for its own steer input. Grains mode measures the
        // same real quantity (lateral runout vs the real Lajeunesse
        // prediction) off the grains' own physically-simulated positions
        // instead of particle positions -- `measure_grain_runout`'s own doc.
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

        let output = match self.surface.get_current_texture() {
            Ok(t) => t,
            Err(_) => return,
        };
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        // Real render buffer: ordinary particles as-is, plus (Grains mode
        // only) one synthetic marker per real grain, position copied
        // DIRECTLY from the grain's own real physically-simulated state --
        // same real technique `rod_blade_of_grass_gui.rs` already uses for
        // the rod solver (a different non-Particle solver entity): the
        // marker carries zero physics of its own, it's a rendering proxy
        // for real state. Isotropic scale (a circle, not an oriented
        // ribbon -- grains have no preferred axis), matching each grain's
        // own real, individually-polydisperse radius.
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

                // Real, visible rolling cue: a small accent dot offset from
                // the grain's own center by `grain.orientation` (the real,
                // integrated rotation angle -- see `Grain::orientation`'s
                // own doc, added 2026-08-03 specifically because a plain
                // rotating circle has no visible cue at all). As the grain
                // genuinely rolls, this dot visibly orbits its parent --
                // the actual physical rotation tonight's rolling-resistance
                // fixes are about, made visible for the first time.
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
            self.renderer
                .render(&self.device, &self.queue, &marker_particles, &view, true);
        } else {
            self.renderer
                .render(&self.device, &self.queue, self.sim.particles(), &view, true);
        }

        let raw_input = self.egui_state.take_egui_input(window);
        let fps = self.last_fps;
        let step = self.step;
        let mode = self.mode;
        let holding = self.holding;
        let mut paused = self.paused;
        let mut push_strength = self.push_strength;
        let mut do_reset = false;
        let mut do_toggle_mode = false;
        let mut do_toggle_holding = false;

        let full_output = self.egui_ctx.run(raw_input, |ctx| {
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
                    if mode != Mode::Grains {
                        ui.label("Push/pull strength (LMB push, RMB pull):");
                        ui.add(egui::Slider::new(&mut push_strength, 0.0..=40.0));
                        ui.separator();
                    }
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
        if do_toggle_mode {
            self.toggle_mode();
        } else if do_toggle_holding {
            self.toggle_holding();
        } else if do_reset {
            self.reset();
        }

        self.egui_state
            .handle_platform_output(window, full_output.platform_output);
        let tris = self
            .egui_ctx
            .tessellate(full_output.shapes, full_output.pixels_per_point);
        let sd = ScreenDescriptor {
            size_in_pixels: [self.surface_config.width, self.surface_config.height],
            pixels_per_point: full_output.pixels_per_point,
        };
        for (id, delta) in &full_output.textures_delta.set {
            self.egui_renderer
                .update_texture(&self.device, &self.queue, *id, delta);
        }
        let cmd = {
            let mut enc = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            self.egui_renderer
                .update_buffers(&self.device, &self.queue, &mut enc, &tris, &sd);
            let mut rp = enc
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    ..Default::default()
                })
                .forget_lifetime();
            self.egui_renderer.render(&mut rp, &tris, &sd);
            drop(rp);
            enc.finish()
        };
        self.queue.submit(std::iter::once(cmd));
        for id in &full_output.textures_delta.free {
            self.egui_renderer.free_texture(id);
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
            let resp = s.egui_state.on_window_event(w, &event);
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

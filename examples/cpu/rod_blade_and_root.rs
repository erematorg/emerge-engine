extern crate emerge_engine as emerge;

use egui_wgpu::ScreenDescriptor;
/// The rod-based blade of grass (`examples/rod_blade_of_grass.rs`'s console
/// version), rendered live and pushable with the cursor. The rod is NOT an
/// MPM particle body -- `emerge::render::Renderer` only knows how to draw
/// `Particles`, so each frame we build a small, purely-visual `Particles`
/// buffer whose positions are copied DIRECTLY from the rod's own real,
/// physically-simulated points (`rod.points.x`) -- the markers carry zero
/// physics of their own, they are a rendering proxy for real state, not a
/// stand-in simulation.
///
/// Cursor push is real and hover-only (no click needed): the cursor position
/// sets `rod.push_center` every frame, `push_strength` comes from the egui
/// slider (0 = off), both read fresh by `advance_rod`
/// on EVERY substep -- same persistent-forcing contract wind already had
/// (see `Rod::push_center`'s doc for the real one-shot-impulse bug this
/// replaced).
///
/// egui panel (same wgpu-native egui already used by `material_sandbox_gpu`)
/// exposes the push strength as a live slider.
///
///   cargo run --example rod_blade_and_root --features render
use emerge::particle::{Particle, Particles};
use emerge::render::Renderer;
use emerge::rod::{
    Gravitropism, GravitropismMode, Growth, GrowthResistance, Phototropism, Rod, RodMaterial,
    RodPlasticity, SecondaryGrowth, build_straight_rod,
};
use emerge::{FrameLogger, SimConfig, Simulation, per_material_stats};
use glam::{Mat2, Vec2};
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

const DX_METERS: f32 = 0.01;
const DISPLAY_GRID: usize = 24;
const HEIGHT_M: f32 = 0.10;
// Blade B's own height, below its Greenhill critical height. At HEIGHT_M=0.10 with
// YOUNG_MODULUS_B=5e6 the critical height is ~0.0965 m, so straight is an unstable
// equilibrium and the blade can never return to 0 (see the gravitropism gate below).
// With a clear margin under it, the same blade is a mechanically sound stem: if it
// settles back to 0 after a push, the buckling explanation holds.
const HEIGHT_M_B: f32 = 0.08;
// N=20 is the measured comfortable point (40+ fps, confirmed via the
// NDJSON logger's own fps field) -- N=40 measured at 14fps, too sluggish
// for interactive use. Cost grows faster than quadratically with point
// count (bending's omega scales steeper than axial's as l0 shrinks), so
// pushing past ~20-30 points for a SINGLE rod on CPU needs either an
// implicit integrator or a GPU port, not another CFL-margin trick -- the
// per-point Gershgorin fix already recovered what was recoverable there.
const N_POINTS: usize = 20;
const START: Vec2 = Vec2::new(9.0, 4.0);
// Second blade, same points/damping convention, HALF the Young's modulus --
// a stiffness comparison (matches the soft/dense and soft/packed
// pattern already used for sand and snow). Height is now its OWN constant
// (`HEIGHT_M_B`, see its doc) rather than shared with blade A, so it can sit
// on the mechanically-sound side of its own Greenhill buckling threshold.
const START_B: Vec2 = Vec2::new(17.0, 4.0);
const YOUNG_MODULUS_A: f32 = 1.0e7;
const YOUNG_MODULUS_B: f32 = 5.0e6;
// Cross-section dimensions, shared by construction (`build_blade`, the plasticity
// yield moment below) and rendering (the ribbon ellipse width), so both use one value.
const BLADE_WIDTH_M: f32 = 0.003;
const BLADE_THICKNESS_M: f32 = 0.001;
const ROOT_WIDTH_M: f32 = 0.004;
const ROOT_THICKNESS_M: f32 = 0.002;
// Shared with construction (`build_straight_rod`'s own linear-density arg)
// AND rendering (the reference point secondary growth's own mass gain
// scales the ribbon width against) -- one value, not two independently-
// drifting copies, same reasoning as the width/thickness constants above.
const BLADE_LINEAR_DENSITY_KG_PER_M: f32 = 0.01;
const ROOT_LINEAR_DENSITY_KG_PER_M: f32 = 0.15;
// Disclosed illustrative budget (see the root's own `Growth`
// construction below for the finding this addresses) -- caps total
// achievable growth well within the visible 0.24m world (24 cells * 0.01m)
// regardless of real elapsed time.
const ROOT_GROWTH_BUDGET_M: f32 = 0.08;
// Shared with the ribbon-marker construction below (`deformation_gradient`
// divides this back out to get the absolute grid-unit ellipse size
// regardless of this scale factor) -- one value, not two independently-
// drifting copies.
const PARTICLE_SCALE: f32 = 0.6;
const WIND_DRAG_COEFF: f32 = 1.5;
const WIND_SPEED_REAL_M_S: f32 = 0.05;
const WIND_GUST_PERIOD_SECONDS: f32 = 4.0;
const PUSH_RADIUS: f32 = 3.0;
const DEFAULT_PUSH_STRENGTH: f32 = 400.0;
// Deliberately soft yield stress (see `rod::plasticity` module doc for
// the elastic-perfectly-plastic mechanism, Gere & Goodno) -- blade A's
// own real cross-section (0.003m x 0.001m, same as `build_blade` below)
// gives a yield moment M_yield = sigma_yield*I/c at this stress.
// Empirically calibrated (a headless sweep across push strength/
// duration, not a guess): the DEFAULT push (400) and a quick nudge at any
// strength stay fully elastic at this value; only a firm, HELD push near
// the slider's top end (1000-2000, sustained ~0.5-1s+) visibly overpowers
// it.
const SIGMA_YIELD_A_PA: f32 = 4.0e4;
// Isotropic hardening for the blade's bending plasticity (see `rod::plasticity`'s
// "Cited fix for unbounded creep" section, Melan 1938/Koiter 1956 shakedown theory).
// Without it, a hard enough push leaves blade A so contorted that gravity alone keeps
// exceeding the fixed yield moment, and the permanent bend keeps climbing (for over
// 80 s with the push released). Hardening (like `VonMisesMaterial`'s
// `hardening_modulus`) raises the yield moment as plastic curvature accumulates, so a
// sustained one-direction overload (gravity, a held push) shakes down to a stable
// shape instead of ratcheting. `2.0x` the base yield moment per unit accumulated
// curvature is an illustrative rate (like `Gravitropism`'s constants), not checked
// against a material's hardening curve.
const HARDENING_MULTIPLE_A: f32 = 2.0;

/// Plant-only demo -- terrain/soil interaction is real and valuable but out of
/// scope here (deferred to a separate "mixed" demo); this scene proves the
/// PLANT itself: blade + root + gravitropism + gated growth, nothing else
/// touching the grid. Gravity uses `SimConfig::earth`'s own unmodified
/// value (9.81/dx_meters) -- a -0.3 override elsewhere was specific to a
/// DruckerPragerMaterial fix that doesn't apply with no sand present.
fn make_sim() -> Simulation {
    let config = SimConfig {
        max_substeps_per_step: 5000,
        min_dt: 1.0e-8,
        rod_sleep_threshold: 0.02,
        // Straight up, not angled: gravitropism targets true vertical and
        // phototropism targets `light_dir`, so with an angled source the
        // whole-organ correction settles at a compromise angle (6.39deg for both
        // blades whatever their EI, so it is the tropism balance, not elastic
        // sag). Without a visible growth representation or a steerable light, a
        // crooked stem reads as broken rather than as phototropism. With the
        // light vertical both targets coincide at 0deg, with no special-case
        // code: light straight up grows straight, light at 45deg grows sideways.
        // Angle it once a controllable light source and visible growth exist.
        light_dir: Vec2::new(0.0, 1.0),
        // Shrinks grid_res to fit the scene's real footprint with margin --
        // only affects cell array size / how many cells get cleared+updated
        // per substep, doesn't touch dx_meters, gravity, or any material
        // parameter, so it can't change simulated behavior, only cost.
        ..SimConfig::earth(32, DX_METERS, 0.02)
    };
    let mut solver = Simulation::empty(config);

    // The blades and the root do not keep each other awake: every rod here uses
    // implicit integration, which skips the shared-grid scatter/gather (see
    // `step.rs`'s `!rod.use_implicit_integration` gate), so blade A, blade B and the
    // root are independent. In a 224-second session both blades slept for 200 s
    // while the root was still settling and growing. A single pushed blade reaches
    // sleep at ~15.5 s headless: a hard push takes that long to settle.
    let build_blade =
        |start: Vec2, height_m: f32, young_modulus: f32, damping_fraction: f32| -> Rod {
            let end = Vec2::new(start.x, start.y + height_m / DX_METERS);
            let mut rod_points = build_straight_rod(
                start,
                end,
                N_POINTS,
                BLADE_LINEAR_DENSITY_KG_PER_M,
                DX_METERS,
            );
            rod_points.pinned[0] = 1;
            rod_points.pinned[1] = 1;
            let ea = young_modulus * BLADE_WIDTH_M * BLADE_THICKNESS_M;
            let ei = young_modulus * BLADE_WIDTH_M.powi(3) * BLADE_THICKNESS_M / 12.0;
            // Damping: a fixed fraction of the global modal critical
            // damping (`RodMaterial::modal_critical_damping`, Blevins 1979/Rao's
            // clamped-free mode shape, beta_1*L=1.8751, numerically integrated
            // modal mass) -- not 100% (that gives zero visible oscillation by
            // construction, reads as robotic) and not the naive `ei/l0` local
            // two-point-spring reference (`critical_damping`), which
            // underestimates the true modal value by 100-1700x for a 20-point
            // cantilever. Blade A (0.15) has active tropisms constantly making
            // small corrections even at rest, so it never reads as "dead."
            // Blade B has none (deliberately -- see the buckling-gate comment
            // below), so once it settles into a buckled equilibrium under
            // the SAME fraction, it locks down completely with nothing left to
            // keep it visibly alive. Lower fraction (0.05) for blade B lets a
            // slow residual oscillation linger instead.
            let (axial_critical, bending_critical) =
                RodMaterial::modal_critical_damping(&rod_points, ea, ei);
            let axial_damping = axial_critical * damping_fraction;
            let bending_damping = bending_critical * damping_fraction;
            let material = RodMaterial::from_young_modulus_rectangular(
                young_modulus,
                BLADE_WIDTH_M,
                BLADE_THICKNESS_M,
                axial_damping,
                bending_damping,
            );
            let mut rod = Rod::new(rod_points, material);
            rod.wind_drag_coeff = WIND_DRAG_COEFF;
            rod.push_radius = PUSH_RADIUS;
            // Active recovery, not just passive spring: passive elasticity has
            // no reason to leave a rest shape it
            // already sits in, so a stem that GREW crooked stays crooked
            // forever. A real plant does not rely on elastic stiffness to find
            // "up" -- it ACTIVELY regrows toward it (gravitropism). GSA=PI
            // (Digby & Firn 1995) is exactly "shoot seeking true vertical," the
            // mirror of the root's own GSA=0 below; `WholeOrgan` (Bastien, Bohr,
            // Moulia, Douady 2013) is the mature-organ posture-control regime,
            // where the correction acts along the WHOLE stem rather than only in
            // a growing tip's own bending zone (the root below correctly stays on
            // the tip-only default -- it IS actively elongating).
            //
            // Gated on the rod's Greenhill self-buckling check, not per blade: past
            // its critical height, straight is an unstable equilibrium for the stem's
            // EA/EI/mass, and no curvature-target correction can hold a structure
            // there (on the over-critical blade B it gives a sustained, non-decaying
            // oscillation, across a 500x gain sweep). Plants answer structural
            // buckling with secondary growth (a thicker, stiffer stem), not
            // gravitropism; see `tests/rod_gravitropism_whole_organ.rs`.
            if rod.buckling_warning(9.81).is_none() {
                // A slow correction rate: gravitropic and phototropic
                // reorientation takes hours in a plant, which no interactive
                // demo can wait for, but a gradual drift back reads as growth
                // correction where a fast one reads as a spring fighting the push.
                // Same law and citations with a smaller sensitivity; illustrative,
                // not calibrated to a species.
                rod.gravitropism = Some(
                    Gravitropism::new(0.015, 0.002)
                        .with_gsa(std::f32::consts::PI)
                        .with_mode(GravitropismMode::WholeOrgan),
                );
                // Phototropism (Cholodny & Went) alongside gravitropism, on the
                // mechanically sound blade only, like gravitropism: an
                // over-critical rod's sustained oscillation (see above) applies to
                // any active curvature-target correction. target_angle_rad=0.0 (the
                // default) grows toward `SimConfig::light_dir`.
                rod.phototropism =
                    Some(Phototropism::new(0.01, 0.002).with_mode(GravitropismMode::WholeOrgan));
            }
            // Secondary growth. `bending_rate=1.0` is sped up for interactivity (a
            // tree thickens over years), like this demo's other rates, not a
            // species calibration; the ribbon rendering shows the thickening (it
            // reads `RodPoints::linear_density_kg_per_m` and scales width by
            // sqrt(area) growth). With `HEIGHT_M_B` under blade B's critical height,
            // neither blade is over-critical in this configuration, so this is
            // usually a gated no-op; it fires when a rod needs it (tested in
            // `secondary_growth.rs` and `mod.rs`'s
            // `sustained_bending_stress_raises_greenhill_height_above_actual_height`).
            rod.secondary_growth = Some(SecondaryGrowth::new(0.02, 0.0, 0.02, 0.0));
            // Implicit (backward Euler) integration -- same physics, zero fidelity
            // cut (Baraff & Witkin 1998), real measured win: 1298 substeps/frame -> 1.
            rod.use_implicit_integration = true;
            // 8 implicit substeps (see `Rod::implicit_substeps`): one backward
            // Euler step at the full 0.02 s frame dt adds numerical damping that
            // visibly over-damps a push on top of the physical 15%-of-critical
            // damping (3 tip-direction reversals over 3 s at 1 substep against 9 at
            // 16, same damping ratio and time). 8 restores visible sway within the
            // frame budget.
            rod.implicit_substeps = 16;
            rod
        };
    // Elastic-perfectly-plastic bending (see `rod::plasticity`) on blade A only, the
    // mechanically sound reference; blade B carries the over-critical buckling case,
    // and a second mechanism there would blur which effect is on screen. Blade A's
    // gravitropism/phototropism keep nudging `rest_curvature` toward upright whatever
    // moved it, so a plastic kink appears at once on overload and then heals over the
    // slow gravitropism timescale.
    let mut blade_a = build_blade(START, HEIGHT_M, YOUNG_MODULUS_A, 0.15);
    let yield_moment_a = RodPlasticity::from_young_modulus_rectangular(
        SIGMA_YIELD_A_PA,
        BLADE_WIDTH_M,
        BLADE_THICKNESS_M,
    )
    .yield_moment_n_m;
    blade_a.plasticity = Some(
        RodPlasticity::new(yield_moment_a).with_hardening(HARDENING_MULTIPLE_A * yield_moment_a),
    );
    solver.add_rod(blade_a);
    solver.add_rod(build_blade(START_B, HEIGHT_M_B, YOUNG_MODULUS_B, 0.05));

    // Root material (E=1e5 Pa, 4mm x 2mm section, 0.15 kg/m linear density).
    // Gravitropism (Porat, Riviere, Meroz 2024, J. Exp. Bot. 75(2):620, eq. 2),
    // starting 45 degrees from vertical, a plausible initial growth direction, so
    // the curl back toward straight down is clear.
    let root_len_m = 0.006;
    let root_angle_from_vertical = 45.0_f32.to_radians();
    let root_dir = Vec2::new(
        root_angle_from_vertical.sin(),
        -root_angle_from_vertical.cos(),
    );
    let root_end = START + root_dir * (root_len_m / DX_METERS);
    let mut root_points =
        build_straight_rod(START, root_end, 4, ROOT_LINEAR_DENSITY_KG_PER_M, DX_METERS);
    root_points.pinned[0] = 1;
    let root_l0 = root_len_m / 3.0;
    let root_point_mass = 0.15 * root_l0;
    let root_ea = 1.0e5 * ROOT_WIDTH_M * ROOT_THICKNESS_M;
    let root_ei = 1.0e5 * ROOT_WIDTH_M.powi(3) * ROOT_THICKNESS_M / 12.0;
    // Local critical-damping reference, not `modal_critical_damping()` (unlike the
    // blade): for this root's geometry (4 points, 6mm long, l0=2mm) its bending value
    // is ~73000x the local reference (~1670x for the 20-point/10cm blade), and the
    // implicit solver diverges (max speed 0.4 -> 19.6 -> 0.3 -> 6.0 rad/s within 15
    // frames, diverging by frame 26). `step_rod_implicit`'s K/C Jacobians are central
    // finite differences at a fixed h=1e-4, which lose accuracy when a force term is
    // this stiff at that scale, and an inaccurate C entry behaves like the sign error
    // its doc warns about ("blows up with alternating sign and exponentially growing
    // magnitude"). An engine-level gap for very short, few-point, soft rods.
    const ROOT_DAMPING_FRACTION_OF_CRITICAL: f32 = 0.15;
    let (root_axial_critical, root_bending_critical) =
        RodMaterial::critical_damping(root_l0, root_point_mass, root_ea, root_ei);
    let root_axial_damping = root_axial_critical * ROOT_DAMPING_FRACTION_OF_CRITICAL;
    let root_bending_damping = root_bending_critical * ROOT_DAMPING_FRACTION_OF_CRITICAL;
    let root_material = RodMaterial::from_young_modulus_rectangular(
        1.0e5,
        ROOT_WIDTH_M,
        ROOT_THICKNESS_M,
        root_axial_damping,
        root_bending_damping,
    );
    let mut root = Rod::new(root_points, root_material);
    // Deliberately stays on `GravitropismMode`'s tip-only DEFAULT, unlike the
    // blades above: this root is actively elongating (see its `Growth` below),
    // so Porat 2024's growth-zone-localized model is the genuinely CORRECT
    // one here -- real roots only actively bend within the zone behind the
    // tip, mature tissue further back does not keep re-curving. `WholeOrgan`
    // would be the wrong regime for it, not merely a bigger hammer.
    //
    // Ungated gravitropism can evolve rest_curvature regardless of whether the
    // tip can actually rotate that far, causing unbounded velocity growth --
    // same resistance gate as growth's own, extended to curvature.
    root.gravitropism = Some(
        Gravitropism::new(0.05, 0.005).with_resistance(GrowthResistance {
            turgor_pressure_pa: 0.5e6,
            resistance_per_unit_mass_pa: 2.0e5,
        }),
    );
    // Logistic elongation growth (Verhulst 1838), with the force-balance gate
    // (Bengough & Mullins 1990/1997, Lockhart 1965) and a finite reserve budget
    // (Deleens, Gregory, Bourdu 1984; see `rod::growth`'s "Real finite resource
    // budget" section). This scene has no soil, so the resistance gate never engages;
    // with point insertion the root would grow without bound (23 cm, off the visible
    // 0.24 m world, after ~90 minutes). ROOT_GROWTH_BUDGET_M is illustrative: enough
    // for several cell-division events in a few minutes, small enough to stay in the
    // visible world however long the demo runs.
    root.growth = Some(
        Growth::new(0.05, 0.005)
            .with_resistance(GrowthResistance {
                turgor_pressure_pa: 0.5e6,
                resistance_per_unit_mass_pa: 2.0e5,
            })
            .with_resource_budget(ROOT_GROWTH_BUDGET_M),
    );
    // Same real engine fix as the blade above -- implicit integration,
    // zero fidelity change.
    root.use_implicit_integration = true;
    solver.add_rod(root);

    solver
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
    cursor_pos: [f32; 2],
    wind_enabled: bool,
    wind_time: f32,
    wind_speed_m_s: f32,
    push_strength: f32,
    frame: u64,
    fps_timer: std::time::Instant,
    fps_frames: u64,
    last_fps: f32,
    last_nearest_dist: f32,
    logger: FrameLogger,
    // Temporary A/B toggle: switches both blades live between implicit (backward
    // Euler, `implicit_substeps=8`) and explicit (CFL-substepped, no numerical
    // damping) integration, to tell whether an over-damped look comes from implicit
    // integration.
    use_implicit: bool,
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
        let sim = make_sim();
        let render_capacity = sim.particles().len() + 2 * N_POINTS + 4 + 8;
        let mut renderer = Renderer::new(&device, render_capacity, fmt);
        renderer.set_camera(
            &queue,
            DISPLAY_GRID as u32,
            size.width,
            size.height,
            PARTICLE_SCALE,
            true,
        );

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

        println!(
            "rod_blade_and_root: hover near the blade to push, W toggles wind, R resets, Q quits"
        );
        let log_path = std::env::temp_dir().join("emerge_rod_blade_and_root.ndjson");
        let logger = FrameLogger::open(&log_path).unwrap();
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
            cursor_pos: [0.0; 2],
            // Default OFF: continuous wind gusts keep the blade perpetually
            // disturbed -- W still toggles it on, but the default view should
            // show the settled, anchored state.
            wind_enabled: false,
            wind_time: 0.0,
            wind_speed_m_s: WIND_SPEED_REAL_M_S,
            push_strength: DEFAULT_PUSH_STRENGTH,
            frame: 0,
            fps_timer: std::time::Instant::now(),
            fps_frames: 0,
            last_fps: 0.0,
            last_nearest_dist: 0.0,
            logger,
            use_implicit: true,
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
            .set_camera(&self.queue, DISPLAY_GRID as u32, w, h, PARTICLE_SCALE, true);
    }

    /// Inverse of `Renderer::set_camera`'s ortho projection, which widens or narrows
    /// the visible world range by the window's aspect ratio: for `aspect >= 1.0`, the
    /// visible X range is `grid_res * aspect` world units wide, centered on
    /// `grid_res/2`, not `0..grid_res`. A square-view assumption puts the cursor off,
    /// more so the wider the window. Derived from `set_camera`'s two projection
    /// branches.
    fn cursor_grid(&self) -> Vec2 {
        let gr = DISPLAY_GRID as f32;
        let w = self.surface_config.width.max(1) as f32;
        let h = self.surface_config.height.max(1) as f32;
        let aspect = w / h;
        // Screen fraction in [0,1] -> NDC in [-1,1] (Y flipped: screen-space
        // Y grows downward, NDC/world Y grows upward).
        let ndc_x = self.cursor_pos[0] / w * 2.0 - 1.0;
        let ndc_y = 1.0 - self.cursor_pos[1] / h * 2.0;
        // Exact inverse of `set_camera`'s own (sx, tx, sy, ty), branch for
        // branch -- `world = (ndc - t) / s`.
        let (sx, tx, sy, ty) = if aspect >= 1.0 {
            (2.0 / (gr * aspect), -1.0 / aspect, 2.0 / gr, -1.0)
        } else {
            (2.0 / gr, -1.0, 2.0 * aspect / gr, -aspect)
        };
        Vec2::new((ndc_x - tx) / sx, (ndc_y - ty) / sy)
    }

    fn update_and_render(&mut self, window: &Window) {
        let step_dt = self.sim.config().dt;
        if self.wind_enabled {
            self.wind_time += step_dt;
            let omega = std::f32::consts::TAU / WIND_GUST_PERIOD_SECONDS;
            let gust_speed_real = self.wind_speed_m_s * (self.wind_time * omega).sin();
            let wind = Vec2::new(gust_speed_real / DX_METERS, 0.0);
            self.sim.rods_mut()[0].wind_velocity = wind;
            self.sim.rods_mut()[1].wind_velocity = wind;
        } else {
            self.sim.rods_mut()[0].wind_velocity = Vec2::ZERO;
            self.sim.rods_mut()[1].wind_velocity = Vec2::ZERO;
        }

        // Persistent push state, read every substep inside advance_rod (see
        // Rod::push_center's doc).
        // Hover-only (no click needed): the slider itself is the on/off --
        // at 0 strength, hovering does nothing, sidestepping any risk of
        // egui eating the mouse-down event before it reaches the window.
        // Applied independently to BOTH blades -- whichever one the cursor
        // is actually near responds, same per-blade gate as before.
        let push_center = self.cursor_grid();
        let mut nearest_dist = f32::MAX;
        for blade_idx in 0..2 {
            let dist = {
                let rod = &self.sim.rods()[blade_idx];
                let mut dist = f32::MAX;
                for (x, &pinned) in rod.points.x.iter().zip(&rod.points.pinned) {
                    if pinned != 0 {
                        continue;
                    }
                    dist = dist.min((*x - push_center).length());
                }
                dist
            };
            nearest_dist = nearest_dist.min(dist);
            // Gated by 2D distance, as `push_acceleration` (coupling.rs): a
            // vertical-only gate lets a cursor far away horizontally push at
            // nearly full strength when level with the blade. An unconditional
            // push_strength every frame, even at zero force, keeps the rod awake.
            let rod = &mut self.sim.rods_mut()[blade_idx];
            rod.push_center = Some(push_center);
            rod.push_strength = if dist < PUSH_RADIUS {
                self.push_strength
            } else {
                0.0
            };
        }
        self.last_nearest_dist = nearest_dist;

        // Temporary A/B toggle sync -- see `use_implicit`'s doc.
        for rod in self.sim.rods_mut().iter_mut().take(2) {
            rod.use_implicit_integration = self.use_implicit;
        }

        self.sim.step();
        self.frame += 1;
        self.fps_frames += 1;

        // Per-frame diagnostics through the engine's NDJSON logger:
        // `diagnostics_snapshot()`'s particle-side fields (`per_material_stats`
        // etc.) read mostly zero for this particle-less scene; `snap.rods` carries
        // the rod-solver aggregate (count, sleeping, max speed, tip positions), and
        // per-rod app state (per-blade cursor distance, push strength) rides in
        // `extra`, the logger's slot for app-specific context.
        let blade = &self.sim.rods()[0];
        let tip = blade.points.x[N_POINTS - 1];
        let blade_max_v = blade.points.v.iter().fold(0.0f32, |m, v| m.max(v.length()));
        let blade_max_permanent_bend = blade
            .points
            .rest_curvature
            .iter()
            .fold(0.0f32, |m, &k| m.max(k.abs()));
        let blade_b = &self.sim.rods()[1];
        let tip_b = blade_b.points.x[N_POINTS - 1];
        let blade_b_max_v = blade_b
            .points
            .v
            .iter()
            .fold(0.0f32, |m, v| m.max(v.length()));
        let root = &self.sim.rods()[2];
        let root_tip = *root.points.x.last().unwrap();
        let root_depth_m = (START.y - root_tip.y) * DX_METERS;
        let root_max_v = root.points.v.iter().fold(0.0f32, |m, v| m.max(v.length()));
        let root_tip_dir = (root_tip - root.points.x[root.points.len() - 2]).normalize_or_zero();
        let root_gravity_alignment = root_tip_dir.dot(Vec2::new(0.0, -1.0));
        let snap = self.sim.diagnostics_snapshot();
        let stats = per_material_stats(self.sim.particles());
        self.logger.log(
            self.frame,
            snap.effective_dt,
            &stats,
            &snap,
            &[],
            &[
                ("tip_x", tip.x),
                ("tip_y", tip.y),
                ("nearest_dist", nearest_dist),
                ("push_strength", self.push_strength),
                ("wind_on", if self.wind_enabled { 1.0 } else { 0.0 }),
                ("fps", self.last_fps),
                ("n_points", N_POINTS as f32),
                ("blade_max_v", blade_max_v),
                ("blade_max_permanent_bend", blade_max_permanent_bend),
                ("blade_sleeping", if blade.sleeping { 1.0 } else { 0.0 }),
                ("tip_b_x", tip_b.x),
                ("tip_b_y", tip_b.y),
                ("blade_b_max_v", blade_b_max_v),
                ("blade_b_sleeping", if blade_b.sleeping { 1.0 } else { 0.0 }),
                ("root_depth_m", root_depth_m),
                ("root_max_v", root_max_v),
                ("root_gravity_alignment", root_gravity_alignment),
                ("t_p2g_us", snap.timing.p2g_us as f32),
                ("t_grid_update_us", snap.timing.grid_update_us as f32),
                ("t_g2p_us", snap.timing.g2p_us as f32),
                ("t_cfl_us", snap.timing.cfl_us as f32),
                ("t_phase_sleep_us", snap.timing.phase_sleep_us as f32),
                ("t_total_us", snap.timing.total_us as f32),
            ],
        );

        if self.fps_timer.elapsed().as_secs_f32() >= 1.0 {
            self.last_fps = self.fps_frames as f32 / self.fps_timer.elapsed().as_secs_f32();
            self.fps_timer = std::time::Instant::now();
            self.fps_frames = 0;
        }

        // Visual buffer: the sand particles (their own material_id) plus marker
        // points for both rods, copied from their simulated state. The markers carry
        // no physics, they render state; a distinct material_id per rod gives the
        // palette distinct colors (blade, root, sand).
        //
        // Ribbon look: each marker's `deformation_gradient` orients and stretches it
        // along the rod's local tangent, with the material cross-section width as the
        // perpendicular axis, through the F-based anisotropic splat every MPM
        // material renders with (no new shader). Consecutive markers overlap (reach =
        // 0.75x the longer adjacent edge) so the blade reads as one continuous shape;
        // `/ PARTICLE_SCALE` cancels the renderer's global particle scale so this is
        // the absolute grid-unit size.
        let mut all = Vec::with_capacity(self.sim.particles().len() + 2 * N_POINTS + 8);
        all.extend(self.sim.particles().iter());
        for (rod_index, material_id) in [(0usize, 1u32), (1usize, 3u32), (2usize, 2u32)] {
            let rod = &self.sim.rods()[rod_index];
            let (width_m, initial_density) = if rod_index == 2 {
                (ROOT_WIDTH_M, ROOT_LINEAR_DENSITY_KG_PER_M)
            } else {
                (BLADE_WIDTH_M, BLADE_LINEAR_DENSITY_KG_PER_M)
            };
            let n_points = rod.points.len();
            for i in 0..n_points {
                let pos = rod.points.x[i];
                // Visual thickening: secondary growth
                // (`secondary_growth::apply_secondary_growth`) grows mass and
                // density at a stressed edge as well as stiffness; this renders it.
                // EA=E*A with E constant, so area (hence linear density at fixed
                // length and material density) grows by EA's fraction; with
                // isotropic thickening (width and the implicit out-of-plane
                // thickness together), the linear width scale is the square root of
                // the area scale, not the area scale itself. Local density is the
                // mean of the point's adjacent edges, the lumped-mass convention of
                // `build_straight_rod`/`insert_tip_point`.
                let densities = rod
                    .points
                    .linear_density_kg_per_m
                    .get(i.wrapping_sub(1))
                    .into_iter()
                    .chain(rod.points.linear_density_kg_per_m.get(i))
                    .copied()
                    .collect::<Vec<_>>();
                let local_density = if densities.is_empty() {
                    initial_density
                } else {
                    densities.iter().sum::<f32>() / densities.len() as f32
                };
                let area_scale = (local_density / initial_density).max(1.0);
                let point_width_m = width_m * area_scale.sqrt();
                let half_width_grid = 0.5 * point_width_m / DX_METERS;
                let mut p = Particle::zeroed();
                p.x = pos;
                p.v = rod.points.v[i];
                p.mass = 1.0;
                p.initial_volume = 1.0;
                p.volume = 1.0;
                p.density = 1.0;
                p.material_id = material_id;

                let prev = (i > 0).then(|| rod.points.x[i - 1]);
                let next = (i + 1 < n_points).then(|| rod.points.x[i + 1]);
                let tangent = match (prev, next) {
                    (Some(a), Some(b)) => (b - a).normalize_or_zero(),
                    (Some(a), None) => (pos - a).normalize_or_zero(),
                    (None, Some(b)) => (b - pos).normalize_or_zero(),
                    (None, None) => Vec2::X,
                };
                let normal = Vec2::new(-tangent.y, tangent.x);
                let reach = [prev, next]
                    .into_iter()
                    .flatten()
                    .map(|q| (pos - q).length())
                    .fold(0.0f32, f32::max)
                    * 1.1;
                let half_length_grid = reach.max(half_width_grid);
                // Factor of 2: the unit quad's local coords span [-0.5, +0.5],
                // so `F * (local_pos * particle_scale)` at local_pos.x=0.5 reaches
                // `F_col0 * 0.5`; doubling makes `half_length_grid`/
                // `half_width_grid` the center-to-edge distance.
                p.deformation_gradient = Mat2::from_cols(
                    tangent * (2.0 * half_length_grid / PARTICLE_SCALE),
                    normal * (2.0 * half_width_grid / PARTICLE_SCALE),
                );

                all.push(p);
            }
        }
        let marker_particles = Particles::from(all);

        let output = match self.surface.get_current_texture() {
            Ok(t) => t,
            Err(_) => return,
        };
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        self.renderer
            .render(&self.device, &self.queue, &marker_particles, &view, true);

        // --- egui panel: live push-strength slider ---
        let raw_input = self.egui_state.take_egui_input(window);
        let fps = self.last_fps;
        let mut wind_enabled = self.wind_enabled;
        let mut wind_speed_m_s = self.wind_speed_m_s;
        let mut push_strength = self.push_strength;
        let mut use_implicit = self.use_implicit;
        let mut reset = false;
        let cursor_grid = self.cursor_grid();
        let nearest_dist = self.last_nearest_dist;

        // Live per-rod state -- so what's happening (buckling risk,
        // secondary growth's own progress, phototropism's real lean) is
        // directly visible in the demo itself instead of needing the NDJSON
        // log read back externally. All three numbers come straight off
        // the rod's own real state, nothing recomputed/faked for display.
        let rod_status = |rod: &Rod, label: &str| -> String {
            let weakest_ei = rod.points.ei.iter().cloned().fold(f32::INFINITY, f32::min);
            let buckled = rod.buckling_warning(9.81).is_some();
            let n = rod.points.x.len();
            let tip_edge = rod.points.x[n - 1] - rod.points.x[n - 2];
            let lean_deg = tip_edge
                .normalize_or_zero()
                .angle_to(Vec2::new(0.0, 1.0))
                .to_degrees();
            format!(
                "{label}: {} | weakest EI={weakest_ei:.3e} | lean={lean_deg:+.1} deg",
                if buckled { "OVER-CRITICAL" } else { "stable" }
            )
        };
        let blade_a_status = rod_status(&self.sim.rods()[0], "blade A");
        let blade_b_status = rod_status(&self.sim.rods()[1], "blade B");
        // Root growth and gravitropism state, the numbers logged to the NDJSON
        // (`root_depth_m`/`root_gravity_alignment`/`root_max_v` above), shown
        // live.
        let root_status = format!(
            "root: depth={root_depth_m:.4}m | gravity_alignment={root_gravity_alignment:+.3} | max_v={root_max_v:.3}"
        );

        let full_output = self.egui_ctx.run(raw_input, |ctx| {
            egui::Window::new("Rod Blade of Grass")
                .default_pos([10.0, 10.0])
                .default_width(260.0)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.label(format!("fps={fps:.0}"));
                    ui.separator();
                    ui.label("Push strength (hover near the blade):");
                    ui.add(egui::Slider::new(&mut push_strength, 0.0..=2000.0));
                    ui.label(format!(
                        "cursor=({:.2},{:.2})  nearest rod point={nearest_dist:.2} cells (radius={PUSH_RADIUS})",
                        cursor_grid.x, cursor_grid.y
                    ));
                    ui.separator();
                    ui.checkbox(
                        &mut use_implicit,
                        "Implicit integration (uncheck = explicit, no numerical damping)",
                    );
                    ui.checkbox(&mut wind_enabled, "Wind");
                    ui.label(format!("Wind gust speed ({WIND_GUST_PERIOD_SECONDS:.0}s period):"));
                    ui.add(egui::Slider::new(&mut wind_speed_m_s, 0.0..=0.3).suffix(" m/s"));
                    ui.separator();
                    ui.label(&blade_a_status);
                    ui.label(&blade_b_status);
                    ui.label(&root_status);
                    ui.separator();
                    ui.label("Hover to push  W: toggle wind  R: reset  Q: quit");
                    if ui.button("Reset").clicked() {
                        reset = true;
                    }
                });
        });
        self.push_strength = push_strength;
        self.wind_enabled = wind_enabled;
        self.use_implicit = use_implicit;
        self.wind_speed_m_s = wind_speed_m_s;
        if reset {
            self.sim = make_sim();
            self.frame = 0;
            self.wind_time = 0.0;
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
                    .with_title("emerge -- rod blade of grass")
                    // 960x720: the egui status panel's fixed 260px width
                    // (anchored top-left) fully covers blade A (world x~9 of
                    // 0..DISPLAY_GRID=24) in a 480x480 window.
                    .with_inner_size(winit::dpi::LogicalSize::new(960u32, 720u32)),
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
                    s.sim = make_sim();
                    s.frame = 0;
                    s.wind_time = 0.0;
                    println!("reset");
                }
                KeyCode::KeyW => {
                    s.wind_enabled = !s.wind_enabled;
                    println!("wind: {}", if s.wind_enabled { "on" } else { "off" });
                }
                _ => {}
            },
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

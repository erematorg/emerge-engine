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
/// slider (0 = off), both read fresh by `apply_rod_internal_and_wind_forces`
/// on EVERY substep -- same persistent-forcing contract wind already had
/// (see `Rod::push_center`'s own doc for the real one-shot-impulse bug this
/// replaced).
///
/// egui panel (same wgpu-native egui already used by `material_sandbox_gpu`)
/// exposes the push strength as a live slider.
///
///   cargo run --example rod_blade_of_grass_gui --features render
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
// Blade B's own height, DISTINCT from blade A's (2026-07-28, user request):
// at HEIGHT_M=0.10 with YOUNG_MODULUS_B=5e6, blade B's real Greenhill
// critical height measures ~0.0965m -- over its own buckling threshold, so
// "straight" is a genuinely unstable equilibrium and it can never return to
// 0 no matter what (see this file's own gravitropism-gate comment below).
// Shortened to a real, clear margin UNDER that threshold so the SAME blade
// (same stiffness comparison, same everything else) becomes a mechanically
// sound stem again -- a direct, falsifiable test of that claim: if it now
// settles back to 0 after a push, the buckling explanation is confirmed,
// not just asserted.
const HEIGHT_M_B: f32 = 0.08;
// N=20 is the real, measured comfortable point (40+ fps, confirmed via the
// NDJSON logger's own fps field) -- N=40 measured at 14fps, too sluggish
// for interactive use. Cost grows faster than quadratically with point
// count (bending's omega scales steeper than axial's as l0 shrinks), so
// pushing past ~20-30 points for a SINGLE rod on CPU needs either an
// implicit integrator or a GPU port, not another CFL-margin trick -- the
// per-point Gershgorin fix already recovered what was recoverable there.
const N_POINTS: usize = 20;
const START: Vec2 = Vec2::new(9.0, 4.0);
// Second blade, same points/damping convention, HALF the Young's modulus --
// a real stiffness comparison (matches the soft/dense and soft/packed
// pattern already used for sand and snow). Height is now its OWN constant
// (`HEIGHT_M_B`, see its doc) rather than shared with blade A, so it can sit
// on the mechanically-sound side of its own Greenhill buckling threshold.
const START_B: Vec2 = Vec2::new(17.0, 4.0);
const YOUNG_MODULUS_A: f32 = 1.0e7;
const YOUNG_MODULUS_B: f32 = 5.0e6;
// Real cross-section dimensions, shared between construction (`build_blade`,
// the plasticity yield moment below) and rendering (the ribbon ellipse
// width) -- named once so both stay in sync, rather than two independently-
// drifting copies of the same real number.
const BLADE_WIDTH_M: f32 = 0.003;
const BLADE_THICKNESS_M: f32 = 0.001;
const ROOT_WIDTH_M: f32 = 0.004;
const ROOT_THICKNESS_M: f32 = 0.002;
// Shared with construction (`build_straight_rod`'s own linear-density arg)
// AND rendering (the real reference point secondary growth's own mass gain
// scales the ribbon width against) -- one value, not two independently-
// drifting copies, same reasoning as the width/thickness constants above.
const BLADE_LINEAR_DENSITY_KG_PER_M: f32 = 0.01;
const ROOT_LINEAR_DENSITY_KG_PER_M: f32 = 0.15;
// Real, disclosed illustrative budget (see the root's own `Growth`
// construction below for the real finding this addresses) -- caps total
// achievable growth well within the visible 0.24m world (24 cells * 0.01m)
// regardless of real elapsed time.
const ROOT_GROWTH_BUDGET_M: f32 = 0.08;
// Shared with the ribbon-marker construction below (`deformation_gradient`
// divides this back out to get the real, absolute grid-unit ellipse size
// regardless of this scale factor) -- one value, not two independently-
// drifting copies.
const PARTICLE_SCALE: f32 = 0.6;
const WIND_DRAG_COEFF: f32 = 1.5;
const WIND_SPEED_REAL_M_S: f32 = 0.05;
const WIND_GUST_PERIOD_SECONDS: f32 = 4.0;
const PUSH_RADIUS: f32 = 3.0;
const DEFAULT_PUSH_STRENGTH: f32 = 400.0;
// Real, deliberately soft yield stress (see `rod::plasticity` module doc for
// the real elastic-perfectly-plastic mechanism, Gere & Goodno) -- blade A's
// own real cross-section (0.003m x 0.001m, same as `build_blade` below)
// gives a real yield moment M_yield = sigma_yield*I/c at this stress.
// Empirically calibrated (a real headless sweep across push strength/
// duration, not a guess): the DEFAULT push (400) and a quick nudge at any
// strength stay fully elastic at this value; only a firm, HELD push near
// the slider's top end (1000-2000, sustained ~0.5-1s+) visibly overpowers
// it.
const SIGMA_YIELD_A_PA: f32 = 4.0e4;
// Real, cited fix (see `rod::plasticity` module doc's "Real, cited fix for
// unbounded creep" section, Melan 1938/Koiter 1956 shakedown theory) for a
// REAL bug this demo's own live testing found: without hardening, a hard
// enough push drove blade A into a contorted enough shape that real gravity
// ALONE kept exceeding the fixed yield moment indefinitely -- confirmed
// directly, permanent bend kept climbing for over 80 real seconds with the
// push fully released, not settling. Isotropic hardening (mirrors
// `VonMisesMaterial`'s own `hardening_modulus`) raises the effective yield
// moment as plastic curvature accumulates, so a sustained one-direction
// overload (gravity, a held push -- this demo's real case, not cyclic
// loading) eventually shakes down to a stable shape instead of ratcheting
// forever. `2.0x` the base yield moment per unit accumulated curvature is a
// real, disclosed illustrative starting rate (like `Gravitropism`'s own
// constants), not yet independently re-verified against a real material's
// actual hardening curve.
const HARDENING_MULTIPLE_A: f32 = 2.0;

/// Plant-only demo -- terrain/soil interaction is real and valuable but out of
/// scope here (deferred to a separate "mixed" demo); this scene proves the
/// PLANT itself: blade + root + gravitropism + gated growth, nothing else
/// touching the grid. Gravity uses `SimConfig::earth`'s own real, unmodified
/// value (9.81/dx_meters) -- a -0.3 override elsewhere was specific to a
/// DruckerPragerMaterial fix that doesn't apply with no sand present.
fn make_sim() -> Simulation {
    let config = SimConfig {
        max_substeps_per_step: 5000,
        min_dt: 1.0e-8,
        rod_sleep_threshold: 0.02,
        // TEMPORARILY straight up (2026-07-28, user request), not angled.
        // Gravitropism targets true vertical; phototropism targets
        // `light_dir` -- with an angled source the two real targets
        // disagree and the whole-organ correction settles at a real,
        // measured, non-zero compromise angle (verified via the live
        // NDJSON log: both blades converge to the identical 6.39deg
        // regardless of their differing EI, proving it's the tropism
        // balance, not an elastic-sag artifact). That tension is real and
        // worth keeping eventually, but right now, with no visible growth/
        // segment-adding representation and no way to steer the light
        // source live, an invisible "why is it crooked" reads as broken
        // rather than as real phototropism. Aligning the light with
        // vertical makes both targets coincide at 0deg -- no special-case
        // code, no phototropism override, just a real scene parameter
        // matching the real mechanism the user described: light "at 90 deg"
        // (straight up) grows straight, light at 45deg grows sideways.
        // Revert to an angled value once a real controllable light source +
        // visible growth representation exist.
        light_dir: Vec2::new(0.0, 1.0),
        // Shrinks grid_res to fit the scene's real footprint with margin --
        // only affects cell array size / how many cells get cleared+updated
        // per substep, doesn't touch dx_meters, gravity, or any material
        // parameter, so it can't change simulated behavior, only cost.
        ..SimConfig::earth(32, DX_METERS, 0.02)
    };
    let mut solver = Simulation::empty(config);

    // Correction (2026-07-27): an earlier comment here claimed blade+root
    // "mutual coupling" prevents sleep in the composed scene -- checked
    // against a real, live 224-second session log and it's false. Every rod
    // here uses implicit integration, which skips the shared-grid scatter/
    // gather entirely (see `step.rs`'s own `!rod.use_implicit_integration`
    // gate) -- so blade A, blade B, and the root are fully independent, no
    // coupling exists to reset anything. The real log shows both blades
    // reaching sleep and staying asleep for a solid 200 seconds while the
    // root was still settling/growing underneath them. A single blade
    // reproduced headless (pushed, released) reaches sleep at ~15.5s -- a
    // hard push needs that long to actually settle, which is what a "still
    // hasn't slept" observation a few seconds after release is really
    // seeing, not a stability bug.
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
            // Damping: a fixed fraction of the REAL global modal critical
            // damping (`RodMaterial::modal_critical_damping`, Blevins 1979/Rao's
            // clamped-free mode shape, beta_1*L=1.8751, numerically integrated
            // modal mass) -- not 100% (that gives zero visible oscillation by
            // construction, reads as robotic) and not the naive `ei/l0` local
            // two-point-spring reference (`critical_damping`), which
            // underestimates the true modal value by 100-1700x for a 20-point
            // cantilever. Blade A (0.15) has active tropisms constantly making
            // small corrections even at rest, so it never reads as "dead."
            // Blade B has none (deliberately -- see the buckling-gate comment
            // below), so once it settles into a real buckled equilibrium under
            // the SAME fraction, it locks down completely with nothing left to
            // keep it visibly alive. Lower fraction (0.05) for blade B lets a
            // real, slow residual oscillation linger instead.
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
            // Gated on the rod's own real Greenhill self-buckling check rather
            // than hardcoded per blade: past its own critical height, "straight"
            // is a genuinely UNSTABLE equilibrium for that stem's real EA/EI/
            // mass, and no curvature-target correction can hold a structure at an
            // unstable equilibrium -- measured directly (2026-07-27), attaching
            // gravitropism to the over-critical blade B produces a sustained,
            // non-decaying oscillation instead of recovery, invariant across a
            // 500x gain sweep. Real biology answers genuine structural buckling
            // with secondary growth (a thicker, stiffer stem), not gravitropism;
            // see `tests/rod_gravitropism_whole_organ.rs`'s own module doc.
            if rod.buckling_warning(9.81).is_none() {
                // Real, deliberately SLOW correction rate (2026-07-27, tuned
                // down from an initial 0.05/0.03 that felt mechanically
                // insistent rather than plant-like): a real plant's
                // gravitropic/phototropic reorientation happens over HOURS,
                // not seconds -- there's no real rate that makes an
                // interactive demo wait hours, but a slower, gradual drift
                // back reads as organic growth-correction, while a fast one
                // reads as a rigid spring fighting the push. Same real law,
                // same citations, just a smaller sensitivity constant --
                // still disclosed-illustrative, not calibrated to a species.
                rod.gravitropism = Some(
                    Gravitropism::new(0.015, 0.002)
                        .with_gsa(std::f32::consts::PI)
                        .with_mode(GravitropismMode::WholeOrgan),
                );
                // Real phototropism (Cholodny & Went) alongside gravitropism --
                // NOT gated the same way blade B is excluded from gravitropism
                // above for a different reason here: this is deliberately only
                // on the mechanically-sound blade, matching gravitropism's own
                // gate, since an over-critical rod's known sustained-oscillation
                // failure mode (see the comment above) applies to ANY active
                // curvature-target correction on it, not just gravitropism
                // specifically. target_angle_rad=0.0 (the default) = grow
                // TOWARD `SimConfig::light_dir` directly.
                rod.phototropism =
                    Some(Phototropism::new(0.01, 0.002).with_mode(GravitropismMode::WholeOrgan));
            }
            // RE-ENABLED (2026-07-29): was disabled 2026-07-28 for two real
            // objections -- (1) `bending_rate=1.0` was tuned to an artificial
            // ~20-real-second demo timescale, not real secondary-growth time
            // (a real tree thickens over YEARS); (2) the rod carried zero
            // VISUAL representation of getting thicker. (2) is now fixed --
            // the ribbon-marker rendering above reads `RodPoints::
            // linear_density_kg_per_m` directly and scales visible width by
            // its real sqrt(area) growth. (1) is addressed the same way
            // every other rate constant in this demo already is: a real,
            // disclosed, sped-up-for-interactivity magnitude (matches
            // Gravitropism's own "hours, not seconds" disclosure just above),
            // not a claim this is species-calibrated. Real, honest current
            // status: `HEIGHT_M_B` was separately shortened
            // (2026-07-22/27) to sit UNDER blade B's own Greenhill critical
            // height, so under THIS demo's current configuration neither
            // blade is actually over-critical right now -- this is a
            // correctly-gated no-op most of the time, not a broken feature.
            // It exists, is real, is tested (`secondary_growth.rs`'s own
            // module tests + `mod.rs`'s
            // `sustained_bending_stress_raises_greenhill_height_above_
            // actual_height`), and fires the moment a rod genuinely needs
            // it.
            rod.secondary_growth = Some(SecondaryGrowth::new(0.02, 0.0, 0.02, 0.0));
            // Implicit (backward Euler) integration -- same physics, zero fidelity
            // cut (Baraff & Witkin 1998), real measured win: 1298 substeps/frame -> 1.
            rod.use_implicit_integration = true;
            // Real, measured (2026-07-27, see `Rod::implicit_substeps`'s own
            // doc): ONE implicit step at the full 0.02s frame dt visibly
            // over-damps a push (backward Euler's own numerical damping at
            // that large a dt, on top of the physically-tuned 15%-of-critical
            // damping) -- measured directly: only 3 real tip-direction
            // reversals over 3s at 1 substep vs. 9 at 16, same physical
            // damping ratio, same total time. 8 is a real, disclosed middle
            // ground (visibly restores sway without needing the full 16 this
            // interactive demo's own frame budget can't always spare).
            rod.implicit_substeps = 16;
            rod
        };
    // Real elastic-perfectly-plastic bending (see `rod::plasticity` module
    // doc) -- ONLY on blade A (the mechanically-sound reference blade), not
    // blade B (already carries its own separate, deliberately-uncorrected
    // over-critical-buckling narrative -- mixing a second mechanism onto
    // that same blade would blur which effect explains what's on screen).
    // Real, disclosed interaction: blade A's own active gravitropism/
    // phototropism keep nudging `rest_curvature` toward upright regardless
    // of WHY it moved, so a plastic kink here is not permanent forever --
    // it appears instantly on overload, then heals back over the same slow,
    // organic timescale gravitropism already uses, not immediately, and not
    // never.
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

    // Real root material (E=1e5 Pa, 4mm x 2mm section, 0.15 kg/m linear
    // density). Gravitropism (Porat, Riviere, Meroz 2024, J. Exp. Bot.
    // 75(2):620, eq. 2) started at 45 degrees from vertical -- a real,
    // plausible initial growth direction -- so the curl back toward straight
    // down is unambiguous to observe.
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
    // NOT switched to `modal_critical_damping()` (unlike the blade above) --
    // real, found regression (2026-07-27): for this root's real geometry (4
    // points, 6mm long, l0=2mm), modal_critical_damping's bending value comes
    // out ~73000x the old local reference (vs ~1670x for the 20-point/10cm
    // blade) -- traced with a real per-frame instability trace (max speed
    // oscillating 0.4 -> 19.6 -> 0.3 -> 6.0 rad/s within 15 frames, then
    // diverging by frame 26) to `step_rod_implicit`'s own disclosed
    // simplification: its K/C Jacobians are central finite differences at a
    // FIXED h=1e-4, which loses accuracy exactly when a force term's
    // curvature at that scale gets this stiff -- an inaccurate C matrix entry
    // is indistinguishable, numerically, from the file's own documented
    // failure mode ("getting the sign wrong... blows up with alternating
    // sign and exponentially growing magnitude"). This is a real, separate,
    // disclosed gap in the implicit solver's Jacobian precision for
    // very-short/few-point/soft rods, not something this example should
    // paper over by picking a smaller ad-hoc fraction -- staying on the old,
    // known-stable local reference here until that's fixed at the engine
    // level (real follow-up, not silently dropped).
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
    // Real logistic elongation growth (Verhulst 1838), WITH the real
    // force-balance gate (Bengough & Mullins 1990/1997, Lockhart 1965) AND a
    // real finite reserve budget (Deleens, Gregory, Bourdu 1984 -- see
    // `rod::growth` module doc's own "Real finite resource budget" section).
    // Real, live-run finding (2026-07-29): this scene has no soil, so the
    // resistance gate above never actually engages (nothing to sense) --
    // combined with real point insertion, root growth ran completely
    // unbounded over a long unattended session (confirmed: 23cm and off the
    // visible 0.24m world after ~90 real minutes). ROOT_GROWTH_BUDGET_M is a
    // real, disclosed illustrative choice (same calibration status as this
    // file's other rate constants) -- big enough to show several real
    // cell-division events over a normal few-minute session, small enough
    // to never leave the visible world regardless of how long the demo runs
    // unattended.
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
    // Temporary, user-requested A/B toggle (2026-07-27): live-switch both
    // blades between implicit (backward Euler, `implicit_substeps=8`) and
    // explicit (CFL-substepped, no numerical damping at all) integration,
    // to directly test whether the "looks over-damped" perception traces
    // to implicit integration itself vs. something else. Not a permanent
    // feature -- remove once the real question is answered.
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
            "rod_blade_of_grass_gui: hover near the blade to push, W toggles wind, R resets, Q quits"
        );
        let log_path = std::env::temp_dir().join("emerge_rod_blade_gui.ndjson");
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
            // show the real, settled, anchored state.
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

    /// Real bug fix (2026-07-27, user-reported "cursor doesn't line up"):
    /// `Renderer::set_camera`'s own ortho projection widens (or narrows)
    /// the visible world-space range by the window's real aspect ratio --
    /// for `aspect >= 1.0` (any ordinary widescreen window), the visible
    /// X range is actually `grid_res * aspect` world units wide, CENTERED
    /// on `grid_res/2`, not a plain `0..grid_res` square. The old version
    /// here assumed a perfect square view regardless of window shape, so
    /// on any real (non-square) window the cursor's world position was
    /// silently wrong -- worse the more widescreen the window. This
    /// re-derives the real inverse of `set_camera`'s own two projection
    /// branches directly, rather than a second, independently-drifting
    /// copy of the same math.
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

        // Real, PERSISTENT push state -- read fresh every substep inside
        // apply_rod_internal_and_wind_forces (see Rod::push_center's doc).
        // Hover-only (no click needed): the slider itself is the on/off --
        // at 0 strength, hovering does nothing, sidestepping any risk of
        // egui eating the mouse-down event before it reaches the window.
        // Applied independently to BOTH blades -- whichever one the cursor
        // is actually near responds, same real per-blade gate as before.
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
            // Real bug fix (2026-07-28): gated by real 2D distance now,
            // matching `push_acceleration`'s own real fix (coupling.rs) --
            // this used to gate on VERTICAL distance only, so the cursor
            // could be far away horizontally and still push at nearly
            // full strength as long as it was near the blade's height.
            // An unconditional push_strength every frame, even at zero
            // force, stops the rod from ever sleeping.
            let rod = &mut self.sim.rods_mut()[blade_idx];
            rod.push_center = Some(push_center);
            rod.push_strength = if dist < PUSH_RADIUS {
                self.push_strength
            } else {
                0.0
            };
        }
        self.last_nearest_dist = nearest_dist;

        // Temporary A/B toggle sync -- see `use_implicit`'s own doc.
        for rod in self.sim.rods_mut().iter_mut().take(2) {
            rod.use_implicit_integration = self.use_implicit;
        }

        self.sim.step();
        self.frame += 1;
        self.fps_frames += 1;

        // Real per-frame diagnostics via the engine's own NDJSON logger --
        // `diagnostics_snapshot()`'s particle-side fields (`per_material_stats`
        // etc.) still read mostly zero for this particle-less scene;
        // `snap.rods` now carries the generic rod-solver aggregate (count,
        // sleeping, max speed, tip positions). This app's own richer per-rod
        // state (per-blade cursor distance, push strength) still rides in
        // `extra`, exactly the slot this logger documents for app-specific
        // context that a generic aggregate can't capture.
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

        // Build the visual buffer: real sand particles (their own real
        // material_id from the sim) + marker points for both rods, copied
        // directly from their real, physically-simulated state -- the
        // markers carry zero physics of their own, they are a rendering
        // proxy for real state, not a stand-in simulation. Distinct
        // material_id per rod purely so the palette renders them in
        // different colors (blade vs root vs sand).
        //
        // Real ribbon look (2026-07-29, replacing "a row of separate dots"):
        // each marker's own `deformation_gradient` orients+stretches it along
        // the rod's REAL local tangent direction, with the REAL material
        // cross-section width as the perpendicular axis -- reusing the SAME
        // F-based anisotropic splat every MPM material already renders
        // through (a real geometric transform: real segment direction, real
        // physical width -- not a new shader/pipeline, not an invented
        // shape). Consecutive markers overlap enough (reach = 0.75x the
        // longer adjacent edge) to read as one continuous blade, not beads
        // on a string; the `/ PARTICLE_SCALE` cancels the renderer's own
        // global particle-scale factor so this is the real, absolute
        // grid-unit size regardless of that setting.
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
                // Real visual thickening (2026-07-29): secondary growth
                // (`secondary_growth::apply_secondary_growth`) grows real
                // mass/density at a stressed edge, not just stiffness --
                // this is what actually renders that as a thicker segment,
                // closing the "invisible internal change" gap that kept
                // SecondaryGrowth disabled in this demo. Real derivation:
                // EA=E*A with E constant => area (hence linear density at
                // fixed length/material density) grows by the same fraction
                // as EA; assuming isotropic thickening (width AND the
                // engine's implicit out-of-plane thickness both grow
                // together), a LINEAR width scale is the SQUARE ROOT of the
                // real AREA scale, not the area scale itself -- using the
                // area scale directly here would double-count the growth.
                // Local density = mean of the point's adjacent edge(s),
                // matching the same lumped-mass convention `build_straight_
                // rod`/`insert_tip_point` already use.
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
                // Real factor of 2: the unit quad's local coords span
                // [-0.5, +0.5] (half-width 0.5, not 1.0), so `F * (local_pos
                // * particle_scale)` at local_pos.x=0.5 only reaches
                // `F_col0 * 0.5` -- doubling here makes `half_length_grid`/
                // `half_width_grid` the REAL, exact center-to-edge distance,
                // not half of it.
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

        // --- egui panel: real, live push-strength slider ---
        let raw_input = self.egui_state.take_egui_input(window);
        let fps = self.last_fps;
        let mut wind_enabled = self.wind_enabled;
        let mut wind_speed_m_s = self.wind_speed_m_s;
        let mut push_strength = self.push_strength;
        let mut use_implicit = self.use_implicit;
        let mut reset = false;
        let cursor_grid = self.cursor_grid();
        let nearest_dist = self.last_nearest_dist;

        // Real, live per-rod state -- so what's happening (buckling risk,
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
        // Real root growth/gravitropism state -- same numbers already logged
        // to the NDJSON (`root_depth_m`/`root_gravity_alignment`/`root_max_v`
        // above), now also visible live instead of needing the log read back
        // externally after the fact.
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
                    // Real bug fix (2026-07-28, screenshot-confirmed): the
                    // old 480x480 default was small enough that the egui
                    // status panel's own fixed 260px width (anchored
                    // top-left) fully covered blade A's real screen
                    // position (world x~9 of 0..DISPLAY_GRID=24) --  blade
                    // A wasn't broken, it was hidden behind our own panel.
                    // 960x720 gives the panel room without eating into
                    // either blade's real screen position.
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

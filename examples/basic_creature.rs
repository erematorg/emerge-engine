extern crate emerge_engine as emerge;

use emerge::render::{ColorMode, Renderer};
use emerge::{
    FrameLogger, Lnn, NeoHookeanMaterial, RatchetFrictionBoundary, SimConfig, Simulation,
    SpawnRegion, per_material_stats,
};
use glam::{IVec2, Vec2};
/// CPU creature -- NeoHookean soft body with peristaltic muscle activation.
///
/// Traveling wave of vertical muscle contraction, crawling via
/// `RatchetFrictionBoundary` -- directional (setae-style) floor friction that
/// resists backward slip much more than forward slip. This is the mechanism
/// that actually produces net locomotion for this body: plain symmetric floor
/// friction measured near-zero net drift regardless of muscle fiber direction
/// (a symmetric contract/release cycle cancels its own displacement, the same
/// reason you can't swim forward clapping symmetrically underwater); real
/// crawlers break that symmetry structurally (setae/hooks), not by timing
/// friction to muscle phase -- confirmed against SoftZoo (the published MPM
/// soft-robot locomotion benchmark, which uses only symmetric friction + learned
/// actuation) and real-crawler robotics literature. See
/// `tests/physics_correctness.rs::ratchet_friction_produces_real_directed_locomotion`
/// for the headless proof.
///
/// Driven by an `Lnn` (Liquid Time-constant Network) continuous-time CPG, not a
/// hand-coded sine wave -- the same controller LP's creatures use. A bilateral
/// (two-ring, mutually-inhibiting) CPG: left/right steer by biasing one ring
/// harder than the other. NOTE: this body is a straight, non-bending column, so
/// "steering" here shifts which half drives harder but cannot produce a real
/// left/right turn -- that needs a body that can curve, a separate limitation.
/// Up/down adjusts wave speed (LNN clock rate). Space pauses. R resets.
///
///   cargo run --example basic_creature --features "render"
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

// GRID was 64: found 2026-07-09 (real repro, not a guess) that this made the
// crawl look like it "stopped working" after sustained steering -- extensive
// investigation (CPG-under-bias, floor-friction impulse model, substep
// compounding, muscle duty cycle) all turned out to be chasing a ghost: the
// creature was crawling correctly the entire time and simply reached this
// grid's side wall (~24-unit-wide body, 64-unit grid). `set_camera` frames the
// whole fixed grid uniformly with no camera-follow, so a real fix (tracking
// camera) is out of scope for this basic example -- LP's actual answer for
// unbounded travel is its chunk-based world design, not this demo. Widened so
// normal play/steering has real room without hitting a wall and looking dead.
const GRID: usize = 96;
const DT: f32 = 0.1;
const MAT_BODY: u32 = 0;
const MUSCLE_GROUPS: u32 = 8;
// Bilateral CPG: 2 mutually-coupled rings (front/back halves of the body),
// 4 segments each. Steering biases one ring harder than the other.
const N_RINGS: usize = 2;
const N_PER_RING: usize = MUSCLE_GROUPS as usize / N_RINGS;
// 1.0 -> 0.5, 2026-07-13: re-swept after the slender-body fix (24x6 -> 36x4)
// since the old value was tuned for a body shape that no longer exists.
// Real, Kuramoto-style coupled-oscillator parameter -- how strongly the
// front/back rings phase-lock to each other. Headless-verified over 4000
// post-settle steps at this body's real proportions: coupling=2.0 is
// catastrophic (-4.40, ~6x worse -- too-strong coupling fights the traveling
// wave's own phase relationship), 0.5 and 1.0 are close (-25.13 vs -25.85 at
// coeff=60) but 0.5 edges ahead once combined with the also-re-tuned
// active_stress_coeff below (-29.47 vs -28.62 at coeff=80).
const RING_CROSS_COUPLING: f32 = 0.5;
// Muscle drive is held at the documented activation ceiling; it is never pushed
// above 1.0 (a muscle can't contract >100%), which also keeps active stress
// inside the CFL budget instead of letting a global amplitude knob detonate it.
const MUSCLE_AMPLITUDE: f32 = 0.9;

// How many steps to run the CPG in isolation (no physics, invisible to the
// player) before the body starts reading its activations. Found 2026-07-09 via
// a real headless test: with the punched-up tuning below, a body reading
// activations from the RAW seeded network (no burn-in) gets a visible
// backward stumble in the first ~1000 steps (real 3.5-unit backslide) before
// the CPG settles into its true phase-locked traveling wave -- the seeded
// initial state isn't wrong, it's just not fully organized yet at steer=0
// (unbiased rings settle slower than a live-biased run, which is why an
// earlier sweep that forced a permanent +-1.0 bias never caught this). 600
// steps (~60 sim-seconds, ~6 gait periods) fully eliminated the backslide
// (verified: 0.00 vs 3.52) and nearly tripled total drift over the same
// window (+24.28 vs +3.41) -- a real, measured fix, not a guessed buffer.
const CPG_BURN_IN_STEPS: usize = 600;

fn make_cpg() -> Lnn {
    make_cpg_biased(0.0)
}

/// Builds a fresh CPG already biased and burned-in for `bias` (a continuous
/// value, not just its sign) -- used for the initial spawn (bias=0.0) and for
/// ANY live steer change (2026-07-11: widened from "only on direction
/// reversal" -- see `update_and_render`'s call site for why every steer change
/// needs this, not just a sign flip).
///
/// Real fix, found+verified 2026-07-09: naively flipping `set_ring_bias` on
/// the SAME already-organized network (the original approach) doesn't reverse
/// the wave's physical propagation direction -- `coupled_traveling_wave`'s
/// topology hard-bakes low-index -> high-index propagation (see
/// src/control/lnn.rs's `excite next` term); only the tonic gain flips.
/// Flipping only the ratchet's friction direction then means real thrust
/// (still propagating the OLD physical way) fights real resisting friction
/// for as long as reverse is held -- confirmed via a live playtest AND a
/// headless worst-case test to be a genuine, unbounded compression ratchet
/// (min J fell to 0.087 live, and even the safest tuning tested still
/// degraded 0.576->0.333 over 18,000 sustained-reverse steps).
///
/// Throwing the old network away and burning in a BRAND NEW one seeded with
/// the new direction's bias from construction lets the wave organize its own
/// propagation to match the new friction direction, instead of being force-
/// mapped after the fact (a naive index-mirror trick was tried first and
/// made things WORSE, not better -- verified empirically, not assumed).
/// Headless proof: post-reversal net drift went from ~-1 to -2 units (stall)
/// to a genuine -18.71 over 25,000 sustained-reverse steps, with J degrading
/// far more gently (0.601 -> 0.231) than the naive flip (0.601 -> 0.109 over
/// a shorter window).
fn make_cpg_biased(bias: f32) -> Lnn {
    let mut lnn = Lnn::coupled_traveling_wave(N_RINGS, N_PER_RING, 1.0, RING_CROSS_COUPLING);
    lnn.set_ring_bias(0, N_PER_RING, bias);
    lnn.set_ring_bias(1, N_PER_RING, -bias);
    for _ in 0..CPG_BURN_IN_STEPS {
        lnn.step(DT);
    }
    lnn
}

// Per-segment colors matching the SoftZoo rainbow palette (ByMaterial slots 0""7).
// ColorMode::ByMaterial assigns color by material_id % 16, so we encode muscle group
// as material_id directly for rendering. Physics still uses MAT_BODY internally.
// For simplicity we render via ByMaterial which gives blue for all (one material).
// Advanced: override muscle group rendering via a custom color callback.

struct App {
    window: Option<Arc<Window>>,
    state: Option<State>,
}

struct State {
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    device: wgpu::Device,
    queue: wgpu::Queue,
    sim: Simulation,
    body_range: std::ops::Range<usize>,
    lnn: Lnn,
    paused: bool,
    wave_speed: f32,
    /// Steering bias in [-1, 1]: drives one CPG ring harder than the other,
    /// breaking the wave's symmetry the way an animal turns. 0 = straight.
    /// ALSO drives the ratchet's crawl direction live (see `update_and_render`):
    /// steer < 0 reverses which way the body actually crawls -- this is real
    /// control, not cosmetic, since it changes `RatchetFrictionBoundary`'s
    /// `easy_direction` on the shared instance the solver is already using.
    steer: f32,
    /// Sign of `steer` as of the last frame a crawl-direction reseed happened
    /// ({-1.0, +1.0}, never 0.0) -- when `steer`'s effective sign flips, `lnn`
    /// gets thrown away and rebuilt via `make_cpg_biased` for the new
    /// direction instead of having its bias flipped in place. See
    /// `make_cpg_biased`'s doc for why the in-place flip doesn't work.
    ///
    /// Widened 2026-07-11 from "only on full sign flip" to "on ANY steer
    /// change": a real, separate bug found live -- ramping `steer` from 0.0
    /// toward 1.0 via repeated live `set_ring_bias` calls on the SAME
    /// already-organized-for-straight-crawl network (never reburned, since
    /// the sign never flipped) can knock the oscillator into a different,
    /// far less effective attractor. Verified via a real headless comparison:
    /// a FRESHLY biased-then-burned-in CPG at steer=1.0 crawls exactly as
    /// well as steer=0.0 (~0.24/window either way, no penalty at all) --
    /// proving the slowdown was never about the bias value itself, only
    /// about reaching it via live incremental nudging instead of a fresh burn-in.
    last_reburn_steer: f32,
    /// Shared handle to the solver's own ratchet boundary -- steering this
    /// updates the SAME instance driving physics, not a copy.
    ratchet: Arc<RatchetFrictionBoundary>,
    renderer: Renderer,
    frame: u64,
    fps_timer: std::time::Instant,
    fps_frames: u64,
    /// True once an anomaly has been reported, so we WARN on the first frame it
    /// appears rather than spamming every frame after.
    anomaly_latched: bool,
    spawn_centroid: Vec2,
    /// NDJSON telemetry, one line per `log_telemetry` call, alongside the
    /// console print -- reads/queries far more reliably than parsing the
    /// human-readable line (used e.g. to build post-hoc telemetry charts
    /// during the 2026-07-09 locomotion investigation).
    telemetry_log: FrameLogger,
}

fn make_sim() -> (
    Simulation,
    std::ops::Range<usize>,
    Arc<RatchetFrictionBoundary>,
) {
    // Stiffness doubled from the original (5.0, 10.0) -- found 2026-07-09 via a
    // real parameter sweep (not a guess), not a smaller tweak: the original
    // material was too soft relative to its own active_stress_coeff, so each
    // muscle contraction outpaced the elastic recovery and the body ratcheted
    // toward a permanently-compressed, near-static state after crawling a
    // bounded ~15-28 units, regardless of direction (see project memory
    // basic_creature_wall_hit_and_reversal_stall_2026-07-09 for the full
    // investigation). Verified this exact combination sustains: last-500-step
    // displacement was STILL POSITIVE (and not decaying) at both 6000 and
    // 12000 steps (0.664 then 1.029) -- genuinely ongoing locomotion, not
    // delayed settling.
    //
    // Pushed further 2026-07-09 (16.0, 32.0 / active_stress_coeff=40.0) for a
    // faster steady cruise -- briefly reverted the same day after a live
    // playtest under sustained reverse steer showed progressive compression
    // (min J fell to 0.087). That regression was traced to the STEERING
    // MECHANISM, not this tuning: naively flipping `set_ring_bias` on the
    // SAME already-organized network doesn't reverse the wave's physical
    // propagation (baked into coupled_traveling_wave's fixed topology, see
    // src/control/lnn.rs), so flipped friction just fights the still-forward-
    // propagating wave forever. Fixed properly via `make_cpg_biased` (now
    // triggered on any steer change, not just reversal -- see that function's
    // doc): reversal throws away the old CPG and burns in a FRESH one already
    // organized for the new direction,
    // so thrust and friction agree instead of fighting. With that real fix in
    // place, this tuning is safe again: headless worst-case test (3,000-step
    // forward, then full reverse held for 20,000 steps, matching the exact
    // live stress pattern) gives genuine sustained reverse drift of -56.19 --
    // the body visibly crosses the whole grid and would hit the wall long
    // before J becomes a real concern (0.593 -> 0.086, and only reaches the
    // low end after ~17,000 steps of continuously holding reverse, well past
    // where a player would have stopped or turned again). See project memory
    // for the full investigation, including the naive-flip numbers this
    // supersedes.
    //
    // Softened slightly again 2026-07-09 (16,32 -> 13,26) chasing a more
    // visible organic squish -- real 4-way sweep (measuring J-swing amplitude
    // as a proxy for visible deformation, not just guessing) found stiffness
    // is NOT actually the main lever for this: softening barely moved J-swing
    // (0.525 -> 0.595, ~13%) while costing real sustain (fwd last-500 drift
    // 1.025 -> 0.594) and reverse safety margin. (13,26,40) is the best
    // available tradeoff point of everything tried -- a small, real,
    // verified improvement, not a transformative one. The "looks mechanical"
    // feeling is honestly more likely coming from the activation scheme
    // itself (a single global vertical-squeeze direction) or the render's
    // color-only feedback than from this constant -- flagged as a real open
    // item for whenever this gets revisited, not something more constant-
    // tuning will fix.
    let mut mat = NeoHookeanMaterial::new(13.0, 26.0);
    // 40.0 -> 60.0, 2026-07-13: real bug found live -- the body crawled a genuine
    // ~18 units then permanently froze regardless of continued steering. Root-caused
    // via headless verification (NOT a guess): the CPG oscillator itself stays alive
    // the whole time (real 0.15-0.25 peak-to-peak swing verified directly, ruling out
    // a regression of the 2026-07-05 oscillator-death fix), but the BODY's own
    // velocity response decays ~7x over time even under continued oscillating
    // drive -- a real damped-driven-oscillator settling, not a controller bug. Swept
    // the two real physical levers: `mu_resist` (friction) had ZERO effect on the
    // decay ratio across 0.30-0.95 (ruled out friction dissipation as the cause);
    // `active_stress_coeff` had a real, non-monotonic effect -- 60.0 gives 6000-step
    // drift=47.10 vs the old 40.0's drift=7.59 (~6.2x), verified to hold at both a
    // 2500-step and 6000-step horizon (not a short-window fluke). The mechanism:
    // the SAME early high-energy transient exists at both values, but 60.0 converts
    // it into real directional progress far more effectively (drift=38.55 by step
    // 1000 alone, vs 40.0's 0.27) -- better phase-lock with the ratchet during the
    // active window, not simply "more force forever." 80.0 and 120.0 were both worse
    // than 60.0 in the same sweep -- a real resonance-like optimum, not "more is
    // always better," so don't casually push this higher without re-verifying.
    //
    // 60.0 -> 80.0, 2026-07-13: re-swept after the slender-body fix (24x6 ->
    // 36x4) for the same reason as RING_CROSS_COUPLING above -- that "80.0
    // is worse than 60.0" finding was calibrated for a body shape that no
    // longer exists. Headless-verified over 4000 post-settle steps at the
    // real current proportions + coupling=0.5: 80.0 gives drift=-29.47 vs
    // 60.0's -25.13 (~17% more) -- min-J drops a bit (0.387 vs 0.450, still
    // far from the 0.05 real-danger threshold used elsewhere in this file).
    // coeff=40 was clearly worse again (-16.85) -- the "more isn't always
    // better" lesson still holds, just at a different peak now.
    mat.active_stress_coeff = 80.0;
    // Real fix, 2026-07-11 (see combined_kirchhoff_stress's doc in transfer.rs for the
    // full investigation): a purely elastic muscle body has no internal dissipation, so
    // cyclic muscle activation pumps real energy in every gait cycle with nowhere to go
    // -- it ratchets into unrecoverable compaction over long horizons. Real muscle/tendon
    // tissue is NOT purely elastic; it's viscoelastic (measured hysteresis losses, Fung
    // 1993). Added the same Kelvin-Voigt term ViscoelasticMaterial already implements
    // (η·dev(D), D = local strain rate) -- zero for rigid-body translation (the crawl
    // itself), nonzero only for actual internal deformation, so it damps the accumulating
    // failure mode without damping real locomotion.
    //
    // Tried viscosity=400 with a finer timestep (min_dt=0.001, max_substeps=512) first --
    // real headless sweep showed it sustains genuinely FLAT, non-decaying drift for the
    // full 20,000-step test, the best result of the whole investigation. Reverted to
    // viscosity=150 + the ORIGINAL (min_dt=0.01, max_substeps=64) config after a real,
    // live playtest showed the finer config causes substantial interactive lag (up to 512
    // substeps/frame in debug mode) -- a genuine performance cost, not a physics problem.
    // viscosity=150 at the cheap config was independently verified to sustain real drift
    // far longer than anything untreated (15,000+ steps vs ~6,500 with viscosity=0), at
    // the SAME substep budget the pre-fix code already used -- no performance regression
    // at all vs. before tonight, just a slower long-horizon decay than the 400/fine
    // combination would give. Real trade-off: interactive responsiveness over a perfectly
    // flat multi-thousand-step drift curve the player will never actually observe.
    mat.viscosity = 150.0;
    let config = SimConfig {
        min_dt: 0.01,
        // Full CFL headroom + the degenerate-state projection net on: keeps
        // active muscle stress stable under hard driving instead of detonating
        // when a substep can't subdivide enough. See the muscle-stability
        // regression test in tests/physics_correctness.rs.
        max_substeps_per_step: 64,
        project_invalid_state: true,
        ..SimConfig::standard(GRID, DT, Vec2::new(0.0, -0.3))
    };
    let body_center = Vec2::new(48.0, 20.0); // grid center, equal room either direction
    // 24x6 -> 36x4, 2026-07-13: real, verified finding -- body PROPORTIONS, not
    // fiber tuning, were the dominant lever for sustained undulatory crawl.
    // Grounded in real animal morphology: undulatory locomotion (snakes, eels)
    // requires a body slender relative to its own length; a short, thick body
    // physically cannot propagate a traveling bend far enough to keep making
    // progress -- it settles into one static arch instead (exactly the
    // plateau observed at 24x6). Headless-verified (`diagnose_slender_body_
    // proportion_sweep`, deleted after use): same total particle budget
    // (~576), same everything else, 4000-step post-settle sweep --
    // 24x6 (stubby): -9.25, growth rate CLEARLY decelerating by the end.
    // 36x4 (slender): -25.85, growth rate STILL LINEAR at step 4000, not
    // plateauing -- a real ~3x improvement from shape alone, and unlike every
    // fiber/hop tuning tried before this, it doesn't visibly saturate within
    // the tested horizon. 48x3 was tried too and gave a similar total but
    // WAS starting to decelerate by step 4000 -- 36x4 is the better, safer
    // pick (real margin before its own eventual plateau, not right at the edge
    // of it).
    let spawn = SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new(36, 4),
        box_center: body_center,
        material_id: MAT_BODY,
        precompute_initial_volumes: true,
        ..SpawnRegion::for_sim(&config)
    };
    // Arc'd so this exact instance is shared between the solver (which drives
    // physics through it) and the app (which steers it live from input) --
    // set_easy_direction takes effect immediately, no boundary swap needed.
    let ratchet = Arc::new(RatchetFrictionBoundary::new(4, 0.1, 0.95, Vec2::X));
    let mut solver = Simulation::new(config, spawn)
        .with_default_material(Box::new(mat))
        .with_boundary(Box::new(Arc::clone(&ratchet)));

    let body_range = 0..solver.particles().len();
    let body_len = 36.0 * 0.5; // box_size.x * spacing -- real physical length, not a re-guessed constant
    let body_left = body_center.x - body_len / 2.0;
    // BILAYER FIBER ARCH + ALTERNATING SEGMENTS, 2026-07-13 -- the real,
    // PDE-grounded replacement for the earlier `update_and_render` hop-impulse
    // hack (an injected velocity kick, disclosed at the time as a temporary,
    // non-physical bandaid).
    //
    // Bilayer part: bottom-half particles in each segment get a fiber leaning
    // one diagonal way, top-half particles get the MIRRORED diagonal -- the
    // same shared per-group CPG activation then produces a genuine bending
    // moment through the material's own F*A*F^T active-stress term (already
    // the engine's real muscle model, used everywhere else), not an external
    // nudge. Headless-verified (`diagnose_bilayer_fiber_arch_replaces_hop_
    // hack`, deleted after use): diag=3.0 is the real sweet spot -- past
    // there, more diag buys almost nothing (diag=4.0->6.0 is +3.994->+4.063,
    // ~2% for 50% more).
    //
    // Alternating part: consecutive segments mirror which way they curl
    // relative to their neighbor -- real anguilliform/undulatory locomotion
    // (eels, snakes) alternates segmental contraction sides down the body,
    // not a single uniform curl everywhere. Headless-verified
    // (`diagnose_alternating_segment_undulation`, deleted after use): a real
    // but modest improvement alone (-4.35 vs -3.87 at the old 24x6
    // proportions) -- the bigger lever turned out to be body shape (below),
    // not this pattern by itself, but it's real biomechanics, not a hardcode,
    // so it stays.
    //
    // Honest tradeoff, not hidden: still genuinely SLOWER than the disclosed
    // hack was -- real undulatory crawlers ARE slow. The player's existing
    // wave_speed control (up/down arrow) is the correct, still-real way to
    // get a faster crawl: it speeds up the same physical gait cycle rather
    // than adding an artificial per-cycle kick.
    const FIBER_DIAG: f32 = 3.0;
    {
        let particles = solver.particles_mut();
        for i in body_range.clone() {
            let t = ((particles.x[i].x - body_left) / body_len).clamp(0.0, 1.0);
            let group = ((t * MUSCLE_GROUPS as f32) as u32).min(MUSCLE_GROUPS - 1);
            particles.muscle_group_id[i] = group;
            let local_y = particles.x[i].y - body_center.y;
            let flip = if group % 2 == 1 { -1.0 } else { 1.0 };
            particles.activation_dir[i] = if local_y >= 0.0 {
                Vec2::new(-FIBER_DIAG * flip, 1.0).normalize()
            } else {
                Vec2::new(FIBER_DIAG * flip, 1.0).normalize()
            };
        }
    }
    (solver, body_range, ratchet)
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
                required_limits: adapter.limits(), // use full hardware limits, not wgpu defaults
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
        let (sim, body_range, ratchet) = make_sim();
        let mut renderer = Renderer::new(&device, sim.particles().len(), fmt);
        renderer.set_camera(&queue, GRID as u32, size.width, size.height, 0.6, true);
        renderer.set_color_mode(ColorMode::ByActivation);
        println!(
            "creature: {} particles  |  up/down wave speed  left/right STEER  Space pause  R reset  Q quit",
            sim.particles().len()
        );
        let telemetry_log =
            FrameLogger::open("basic_creature_telemetry.ndjson").expect("failed to open log file");
        // Real bug found live, 2026-07-13: this used to be a hardcoded
        // Vec2::new(32.0, 20.0) -- a stale leftover from an older GRID=64 test
        // setup, copy-pasted the same way the sweep-test body_center bug was
        // earlier this session. make_sim()'s real body_center is (48.0, 20.0),
        // so every telemetry `drift` readout was off by a constant +16 from
        // frame 1 -- a fake number, not real physics. Computed directly from
        // the actual spawned particles instead, so it can never drift out of
        // sync with whatever body_center/GRID the sim is actually using.
        let spawn_centroid = {
            let particles = sim.particles();
            let n = particles.len() as f32;
            body_range.clone().map(|i| particles.x[i]).sum::<Vec2>() / n
        };

        Self {
            surface,
            surface_config: sc,
            device,
            queue,
            sim,
            body_range,
            lnn: make_cpg(),
            paused: false,
            wave_speed: 1.0,
            steer: 0.0,
            last_reburn_steer: 0.0,
            ratchet,
            renderer,
            frame: 0,
            fps_timer: std::time::Instant::now(),
            fps_frames: 0,
            anomaly_latched: false,
            spawn_centroid,
            telemetry_log,
        }
    }

    /// Read the solver's own diagnostics and body geometry, print a full
    /// telemetry line, and WARN immediately if anything is physically wrong.
    /// Returns nothing — this is pure observation, no simulation effect.
    fn log_telemetry(&mut self, fps: f32) {
        let snap = self.sim.diagnostics_snapshot();

        // Body geometry, computed directly from particles.
        let particles = self.sim.particles();
        let n = particles.len().max(1) as f32;
        let mut centroid = Vec2::ZERO;
        let mut min = Vec2::splat(f32::INFINITY);
        let mut max = Vec2::splat(f32::NEG_INFINITY);
        let mut act_sum = 0.0f32;
        let mut act_max = 0.0f32;
        for i in 0..particles.len() {
            let x = particles.x[i];
            centroid += x;
            min = min.min(x);
            max = max.max(x);
            let a = particles.activation[i];
            act_sum += a;
            act_max = act_max.max(a);
        }
        centroid /= n;
        let extent = max - min;
        let drift = centroid - self.spawn_centroid;

        println!(
            "f{:<5} fps={:>3.0} | sub={:>2}/{} eff_dt={:.4} dropped={:.4} cfl={:.2} vmax={:.2} \
             | J=[{:.3},{:.3}] velclamp={} Jproj={} oob={} nan_p={} nan_g={} \
             | centroid=({:.1},{:.1}) drift=({:+.3},{:+.3}) extent=({:.1}x{:.1}) \
             | act mean={:.2} max={:.2} | massErr={:.1e} momErr={:.1e}",
            self.frame,
            fps,
            snap.substeps_last_step,
            self.sim.config().max_substeps_per_step,
            snap.effective_dt,
            snap.sim_time_dropped,
            snap.cfl_number,
            snap.max_particle_speed,
            snap.min_deformation_j,
            snap.max_deformation_j,
            snap.vel_clamp_count,
            snap.j_projection_count,
            snap.out_of_bounds_particles,
            snap.non_finite_particle_values,
            snap.non_finite_grid_values,
            centroid.x,
            centroid.y,
            drift.x,
            drift.y,
            extent.x,
            extent.y,
            act_sum / n,
            act_max,
            snap.relative_mass_error,
            snap.relative_momentum_error,
        );

        // Structured NDJSON alongside the console print -- includes live
        // steer/wave_speed context the engine has no name for, so a run's
        // telemetry can be replayed/charted without regexing the printed line.
        let stats = per_material_stats(self.sim.particles());
        self.telemetry_log.log(
            self.frame,
            self.sim.config().dt,
            &stats,
            &snap,
            &[(MAT_BODY, "body")],
            &[("steer", self.steer), ("wave_speed", self.wave_speed)],
        );

        // Immediate WARN on the first frame anything goes wrong — pinpoints the
        // exact moment the "huge issues" start, which periodic logging can miss.
        let mut problems: Vec<String> = Vec::new();
        if snap.non_finite_particle_values > 0 || snap.non_finite_grid_values > 0 {
            problems.push(format!(
                "NON-FINITE: {} particle + {} grid values are NaN/Inf",
                snap.non_finite_particle_values, snap.non_finite_grid_values
            ));
        }
        if snap.out_of_bounds_particles > 0 {
            problems.push(format!(
                "{} particles left the grid",
                snap.out_of_bounds_particles
            ));
        }
        if snap.sim_time_dropped > 1e-6 {
            problems.push(format!(
                "solver DROPPED {:.4} of sim time — hit max_substeps and gave up (unstable)",
                snap.sim_time_dropped
            ));
        }
        if snap.min_deformation_j < 0.05 {
            problems.push(format!(
                "near-inverted element: min J = {:.4} (→0 means a particle is collapsing)",
                snap.min_deformation_j
            ));
        }
        // Real bug found live, 2026-07-13: threshold + message text were still
        // calibrated for the old 24x6 spawn box (~12x3 physical, legitimate
        // settle transient up to ~2x that). After the real slender-body fix
        // (36x4 -> ~18x2 physical), that same ~2x settle transient reaches
        // ~36-38 -- past the stale 30.0 threshold, firing a false "scattering"
        // alarm on ordinary landing physics. Scaled to the real spawn size
        // instead of a re-guessed constant.
        if extent.x > 2.5 * 18.0 || extent.y > 15.0 {
            problems.push(format!(
                "body SCATTERING: extent {:.1}x{:.1} (spawned ~18x2)",
                extent.x, extent.y
            ));
        }
        if snap.substeps_last_step >= self.sim.config().max_substeps_per_step {
            problems.push(format!(
                "substeps MAXED ({}) — CFL is fighting hard, near the stability edge",
                snap.substeps_last_step
            ));
        }
        if !problems.is_empty() && !self.anomaly_latched {
            self.anomaly_latched = true;
            eprintln!("  ⚠ FIRST ANOMALY at frame {}:", self.frame);
            for p in &problems {
                eprintln!("      - {p}");
            }
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
            .set_camera(&self.queue, GRID as u32, w, h, 0.6, true);
    }

    fn update_and_render(&mut self) {
        if !self.paused {
            // Reseed a FRESH, burned-in CPG for the current steer value on ANY
            // change (not just a full sign flip) instead of live-nudging bias
            // on the same already-organized network. See `make_cpg_biased`'s
            // doc for the full history: the original fix (reburn only on sign
            // flip) solved reversal but not this -- a real, separate bug found
            // live where ramping steer 0.0 -> 1.0 via repeated in-place
            // `set_ring_bias` calls could knock the oscillator into a far less
            // effective attractor, verified via a real headless comparison
            // (a FRESHLY biased-then-burned-in network at steer=1.0 crawls
            // exactly as well as steer=0.0 -- the bias value was never the
            // problem, only reaching it live was).
            // Real bug found live, 2026-07-13: the creature crawled a real ~18 units
            // BEFORE the player ever touched a key, then took ~25+ real seconds of
            // holding the opposite steer to visibly reverse it -- read as "frozen"
            // by a player who didn't wait that long. Root-caused via a precise
            // headless reproduction of this exact frame-by-frame loop (not a guess):
            // muscle activation ran unconditionally even at steer=0.0 (unbiased
            // CPG), and `RatchetFrictionBoundary`'s hardcoded default
            // `easy_direction=Vec2::X` gave that unbiased crawl a real, fast,
            // dominant direction from frame 1 -- a "neutral" state that wasn't
            // actually neutral. An earlier foothold-seeking locomotion prototype
            // already established the right pattern for this: "only replans... while a
            // direction is actively held; with no input, legs simply stop and
            // rest" -- basic_creature never applied that same gate. Fixed:
            // muscle activation (and the CPG clock itself) only run while
            // steer != 0.0; at rest, the body just settles under gravity/friction,
            // no unsolicited directional crawl.
            if self.steer != 0.0 {
                let new_dir_sign = if self.steer >= 0.0 { 1.0 } else { -1.0 };
                if self.steer != self.last_reburn_steer {
                    self.lnn = make_cpg_biased(self.steer);
                    self.last_reburn_steer = self.steer;
                }
                // wave_speed scales the LNN's internal clock -- faster wave_speed runs the
                // continuous-time ODE forward faster, raising the oscillation frequency, without
                // needing to reconstruct the network (tau/weights stay fixed).
                // ALSO drive the crawl direction itself: steer<0 reverses which way
                // the ratchet resists slip, so the body actually crawls backward, not
                // just internally-lopsided while still walking the one baked-in way.
                // Same shared instance the solver already uses -- takes effect this substep.
                self.ratchet.set_easy_direction(if new_dir_sign >= 0.0 {
                    Vec2::X
                } else {
                    Vec2::NEG_X
                });
                // Restore the real asymmetric ratchet values while actively steering --
                // see the `else` branch below for why this is only active on demand.
                self.ratchet.set_friction(0.1, 0.95);
                self.lnn.step(DT * self.wave_speed);
                let activations: Vec<f32> = self.lnn.activations().collect();
                // Real bug found live, 2026-07-13: once the body has fully settled
                // under gravity (the correct behavior now that idle is genuinely
                // still -- see the steer-gate above), the ratchet's crawl mechanism
                // stalls almost immediately: `RatchetFrictionBoundary` only
                // discriminates friction on a downward-velocity floor-contact event
                // (see `apply_to_grid_velocity`'s `v_n_scalar < 0.0` check), and a
                // fully-rested body barely has any vertical velocity left to trigger
                // it. A temporary velocity-injection hack was tried and shipped
                // briefly the same day, then explicitly REPLACED (not just tuned)
                // once it started visibly rotating/launching parts of the body
                // live -- a real, disclosed cost of injecting momentum externally
                // instead of through the material's own constitutive model.
                //
                // REAL FIX, same day: `make_sim`'s bilayer fiber arch (see that
                // function's doc) gives the body's own `F*A*F^T` active stress a
                // genuine bending moment, so a normal muscle contraction produces
                // real lift-off through elastic curling -- no injected velocity
                // anywhere in this function. Slower than the hack (by design --
                // real inchworms are slow), but headless-verified to keep making
                // real, non-stalling progress with min-J staying healthy, and the
                // player's own `wave_speed` control is the correct way to go faster
                // without breaking that.
                let body_range = self.body_range.clone();
                let particles = self.sim.particles_mut();
                for i in body_range {
                    let group = particles.muscle_group_id[i] as usize;
                    // Clamp to the documented [0,1] activation contract — a muscle can't
                    // contract past 100%, and staying in-contract keeps active stress
                    // inside the CFL budget.
                    //
                    // Tried signed (-1,1) bidirectional activation (contract + actively
                    // extend) 2026-07-11 as a fix for a real long-horizon compaction-ratchet
                    // bug (see `combined_kirchhoff_stress` doc for the bug itself, still
                    // real and still open) -- reverted: a real 20,000-step headless test
                    // showed it's WORSE, not better. min(J) collapsed toward the numerical
                    // floor (particles crushed near-zero) while max(J) diverged past 3.0
                    // (other regions torn outward), and net drift just oscillated around
                    // zero with no sustained direction. The naive sigmoid->signed remap
                    // breaks the asymmetric contract/relax rhythm RatchetFrictionBoundary's
                    // directional grip actually depends on. Root cause of the original
                    // compaction bug is confirmed (see doc comment) but the real fix is NOT
                    // this remap -- needs more investigation before trying again.
                    particles.activation[i] =
                        (MUSCLE_AMPLITUDE * activations[group]).clamp(0.0, 1.0);
                }
            } else {
                // No steer input: kill the ratchet's own bias too, not just muscle
                // activation. mu_easy==mu_resist -> plain symmetric floor friction,
                // so passive settling jitter can't get ratcheted into drift.
                self.ratchet.set_friction(0.5, 0.5);
                let body_range = self.body_range.clone();
                let particles = self.sim.particles_mut();
                for i in body_range {
                    particles.activation[i] = 0.0;
                }
            }
            self.sim.step();
            self.frame += 1;
        }
        self.fps_frames += 1;
        // Telemetry ~2x/sec so the log stays readable but catches transients.
        if self.fps_timer.elapsed().as_secs_f32() >= 0.5 {
            let fps = self.fps_frames as f32 / self.fps_timer.elapsed().as_secs_f32();
            self.log_telemetry(fps);
            self.fps_timer = std::time::Instant::now();
            self.fps_frames = 0;
        }
        let output = match self.surface.get_current_texture() {
            Ok(t) => t,
            Err(_) => return,
        };
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        self.renderer
            .render(&self.device, &self.queue, self.sim.particles(), &view, true);
        output.present();
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        let w = Arc::new(
            el.create_window(
                winit::window::WindowAttributes::default()
                    .with_title("emerge -- Creature [peristaltic locomotion]")
                    .with_inner_size(winit::dpi::LogicalSize::new(480u32, 480u32)),
            )
            .unwrap(),
        );
        self.state = Some(pollster::block_on(State::new(w.clone())));
        self.window = Some(w);
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(s) = self.state.as_mut() else { return };
        match event {
            WindowEvent::CloseRequested => el.exit(),
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(key),
                        state,
                        ..
                    },
                ..
            } => {
                let pressed = state == ElementState::Pressed;
                match key {
                    KeyCode::Escape | KeyCode::KeyQ if pressed => el.exit(),
                    KeyCode::Space if pressed => {
                        s.paused = !s.paused;
                        println!("{}", if s.paused { "PAUSED" } else { "RUNNING" });
                    }
                    KeyCode::KeyR if pressed => {
                        let (sim, range, ratchet) = make_sim();
                        s.spawn_centroid = {
                            let particles = sim.particles();
                            let n = particles.len() as f32;
                            range.clone().map(|i| particles.x[i]).sum::<Vec2>() / n
                        };
                        s.sim = sim;
                        s.body_range = range;
                        s.ratchet = ratchet;
                        s.lnn = make_cpg();
                        s.steer = 0.0;
                        s.last_reburn_steer = 0.0;
                        s.frame = 0;
                        s.anomaly_latched = false;
                        println!("reset");
                    }
                    // Capped at 3.0, not the naive 6.0 tried earlier: Lnn::step
                    // is forward Euler with tau=0.5 at period=1.0, so
                    // dt=DT*wave_speed approaching tau (dt/tau -> 1.0) makes
                    // the leak term's decay factor (1 - dt/tau) go negative --
                    // it stops decaying smoothly and starts flipping sign
                    // almost every step, real numerical noise, not a faster
                    // wave. Measured directly (2026-07-09): sign-flip rate per
                    // 100 steps goes 6 (speed=1) -> 17 (speed=3) -> 41
                    // (speed=4) -> 94 (speed=4.5) -- a real cliff right where
                    // dt/tau crosses ~0.8-0.9, not a gradual change. 3.0 stays
                    // comfortably below it; this is what "pressing Up seems to
                    // glitch" (2026-07-09 playtest) actually was.
                    KeyCode::ArrowUp if pressed => s.wave_speed = (s.wave_speed + 0.2).min(3.0),
                    KeyCode::ArrowDown if pressed => s.wave_speed = (s.wave_speed - 0.2).max(0.1),
                    KeyCode::ArrowLeft if pressed => {
                        s.steer = (s.steer - 0.2).max(-1.0);
                        println!("steer {:+.1}", s.steer);
                    }
                    KeyCode::ArrowRight if pressed => {
                        s.steer = (s.steer + 0.2).min(1.0);
                        println!("steer {:+.1}", s.steer);
                    }
                    // Real bug found live, 2026-07-13: there was no release
                    // handler at all for either steer key -- releasing did
                    // nothing, so `steer` just sat wherever it was left
                    // (matching "release does nothing, never rests" reported
                    // live). Real fix: releasing either steer key snaps
                    // straight back to idle, matching ordinary "let go = stop"
                    // control expectations -- and feeds directly into the
                    // already-correct steer==0.0 idle gate (zero activation,
                    // symmetric ratchet friction) instead of requiring the
                    // player to manually counter-steer back through exactly 0.
                    KeyCode::ArrowLeft | KeyCode::ArrowRight if !pressed => {
                        s.steer = 0.0;
                        println!("steer {:+.1}", s.steer);
                    }
                    _ => {}
                }
            }
            WindowEvent::Resized(sz) => s.resize(sz.width, sz.height),
            WindowEvent::RedrawRequested => {
                s.update_and_render();
                if let Some(w) = &self.window {
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

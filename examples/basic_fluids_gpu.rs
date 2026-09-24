extern crate emerge_engine as emerge;

/// GPU viscoplastic fluids — Newtonian water dam-break + Bingham mud blob, zero CPU readback.
///
///   Mat 0  Newtonian water (blue) — Tait EOS + deviatoric viscosity
///   Mat 1  Bingham mud    (gold)  — viscoplastic with yield stress
///
///   cargo run --example basic_fluids_gpu --features "render"
use std::sync::Arc;

use emerge::diagnostics::log_frame_gpu;
use emerge::render::{ColorMode, DualPhaseSurfaceSource, GridVolumeSource, Renderer};
use emerge::{
    BinghamFluidMaterial, FixedStepConfig, FixedStepController, GpuFieldEntry, GpuSimulation,
    MaterialRegistry, NewtonianFluidMaterial, Particle, SimConfig, SpawnRegion, build_particles,
};
use glam::{IVec2, Vec2};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

const GRID: usize = 64;
// Real, disclosed PLAYBACK speed, not a physics constant -- this demo's own
// `gravity: Vec2::new(0.0, -0.3)` (already disclosed as "deliberately weak
// ... tuned for a calmer, more legible demo") and material viscosity were
// tuned by eye while the pre-2026-07-30 fps-coupling bug was secretly
// running the sim ~6x too fast (60fps x the OLD 0.1 dt = 6 simulated
// seconds/real second). Rather than re-tune gravity/viscosity (a real
// physics change) or leave it at a correct-but-unfamiliar-looking true 1x,
// this makes that same ~6x an explicit, disclosed "fast-forward" dial
// (`FixedStepConfig::simulation_speed`), same category as a video player's
// playback-speed control, not a hidden physics fudge.
const PLAYBACK_SPEED: f32 = 6.0;
// Real target render cadence -- NOT `1.0/60.0` used directly as `DT`
// (tried and reverted same session, see `stepper`'s own field doc): at
// PLAYBACK_SPEED=6 that would have needed PLAYBACK_SPEED/DT = 360
// `step_frame()` calls/sec, i.e. 6+ real MPM steps crammed into EVERY
// render frame -- confirmed via live measurement to overload the GPU
// (each `step_frame()` call has fixed dispatch overhead: P2G/grid-update/
// G2P are separate submissions, and multiplying that per frame is real
// cost, not perception). `DT` below is derived FROM this + `PLAYBACK_
// SPEED` specifically so exactly ~1 `step_frame()` call happens per
// render frame at the target fps, regardless of what speed is dialed in.
const RENDER_FPS_TARGET: f32 = 60.0;
// Derived, not independently chosen -- see `RENDER_FPS_TARGET`'s own doc.
// At PLAYBACK_SPEED=6.0 this evaluates to 0.1, the SAME value this demo
// used before the 2026-07-30 pacing fix -- not a coincidence: that's
// exactly the dt size this demo's materials/gravity were tuned against.
const DT: f32 = PLAYBACK_SPEED / RENDER_FPS_TARGET;
const MAT_WATER: u32 = 0;
const MAT_MUD: u32 = 1;
const LABELS: &[(u32, &str)] = &[(MAT_WATER, "water"), (MAT_MUD, "mud")];
// Module scope (not local to `make_sim_data`, which only builds the sim, not
// the renderer): `Renderer::set_grid_reference_cell_mass` needs this same
// real SI value from `State::new`, a separate `impl` method -- see that call
// site's own comment.
const WATER_RHO_GRID: f32 = 0.1;

struct App {
    window: Option<Arc<Window>>,
    state: Option<State>,
}

struct State {
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    sim: GpuSimulation,
    renderer: Renderer,
    cursor_pos: [f32; 2],
    lmb: bool,
    rmb: bool,
    frame: u64,
    fps_timer: std::time::Instant,
    fps_frames: u64,
    /// Cycled with G: Particles -> GridVolume -> Surface -> Particles.
    /// GridVolume's real per-material accumulator is attached lazily on
    /// first switch INTO that mode (NOT eagerly at construction -- that
    /// pattern caused a real, measured perf regression in
    /// material_sandbox_gpu, fixed same session).
    render_mode: RenderMode,
    /// Which spawn geometry is live -- switched with number keys, see
    /// `Pattern`'s own doc. `reset()` re-spawns using this, not always
    /// `DamBreak`, so switching pattern and resetting are the same action.
    pattern: Pattern,
    /// Converts real measured elapsed time into the correct number of physics
    /// steps per frame -- calling `sim.step_frame()` once per render frame
    /// assumes each frame takes exactly `DT` of real time, which it doesn't
    /// (render frame rate varies), and produces the exact "sometimes speeds
    /// up, sometimes slows down" symptom reported live this session. Same
    /// real fix already used in `snake_on_terrain_gpu.rs`, just never
    /// ported here.
    stepper: FixedStepController,
    last_instant: std::time::Instant,
    /// Diagnostic only (2026-07-30): highest `steps_for_frame` result seen
    /// since the last fps print -- a catch-up burst (several real
    /// simulation steps crammed into one render call because the GPU fell
    /// behind real time) would NOT show up in the averaged fps number
    /// below, since a 2-second average smooths occasional slow frames out.
    max_steps_seen: usize,
}

/// The three real rendering paths this demo can show, cycled with G:
/// per-particle instanced splat (`render_gpu`), the solver's own coarse
/// physics-grid density field (`render_grid_volume`), and the finer,
/// resolution-independent curvature-flow surface reconstruction
/// (`render_surface_reconstruction`, shipped 2026-07-29 -- see that
/// method's own doc for the real technique).
#[derive(Clone, Copy, PartialEq, Eq)]
enum RenderMode {
    Particles,
    GridVolume,
    Surface,
}

/// Which real fluid behaviour this scene proves, switched with the number
/// keys (1-4) -- same material/config/render setup throughout, only the
/// spawn geometry (and, for `Vortex`, the initial velocity field) changes.
/// 4 is a reserved slot, not built yet -- see its own doc.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Pattern {
    /// Classic dam-break: a tall column collapses sideways under gravity and
    /// spreads across the floor. This demo's own original, still-default
    /// scene.
    DamBreak,
    /// A small blob falls into a shallow, wide pool -- crown splash and
    /// ripple propagation, a real, distinct behaviour from a collapsing
    /// column (Worthington-style droplet impact), proving the same solver
    /// handles a genuinely different initial geometry, not just a bigger
    /// dam-break.
    DropletImpact,
    /// A resting pool with a real `GravityWellField` "drain" pulling fluid
    /// toward one point, plus a small seed disturbance -- NOT a hand-written
    /// velocity profile. The whirlpool shape (fast core, decaying tail) is
    /// caused by the solver's own APIC angular-momentum conservation (Jiang
    /// et al. 2015) as fluid spirals into the drain, the same way a real
    /// bathtub vortex forms. See its own match-arm doc for the full account.
    Vortex,
    // 4: siphon (real tube geometry, pinned-particle channel + priming) --
    // meaningfully harder than the three above (new static geometry, not
    // just a different spawn), deliberately not rushed in the same pass.
}

impl Pattern {
    fn label(self) -> &'static str {
        match self {
            Pattern::DamBreak => "1: dam break",
            Pattern::DropletImpact => "2: droplet impact",
            Pattern::Vortex => "3: vortex",
        }
    }
}

fn make_sim_data(
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pattern: Pattern,
) -> GpuSimulation {
    let config = SimConfig {
        min_dt: 1.0e-4,
        // Raised from 8 alongside the eos_stiffness fix below (2026-08-07):
        // a correctly-stiff EOS needs real substep headroom (CPU's identical
        // water+mud scene needed 22-63/frame at cfl=0.5) -- 8 would have
        // silently capped it and (with the honest-dropped-time fix already
        // shipped) reported most of each frame's time as unadvanced.
        max_substeps_per_step: 60,
        cfl_include_affine_speed: false,
        // This demo's gravity (below) is ~10x CPU basic_fluids.rs's, so the
        // same eos_stiffness=1000 needs a tighter CFL here to stay
        // admissible -- confirmed live: cfl=0.5 (fine for the weaker-gravity
        // CPU scene) panics on frame 1 here ("GPU strict fluid update became
        // inadmissible"). Not swept per-value for this scene yet; 0.1 is the
        // known-stable pairing from the CPU sweep at higher compression.
        // 0.3, not 0.1 (2026-08-13) -- matches `basic_fluids_gui.rs`'s own
        // change and the standard explicit CFL number range (0.2-0.4;
        // Monaghan 0.25-0.3 for SPH, MLS-MPM commonly 0.3-0.5). See that
        // demo's own comment for the full reasoning: 0.1 was 3x more
        // conservative than any cited solver, and the failure it was
        // protecting against traced to a too-soft EOS, not to C.
        material_cfl_coefficient: 0.3,
        // Real, root-caused fix (2026-08-06, caught live by the user): the
        // old `Vec2::new(0.0, -0.3)` (~3270x weaker than real IRL gravity,
        // g_grid~=981 via SimConfig::earth) left too little real driving
        // force to overcome this material's own EOS-pressure elastic-like
        // response -- the free surface never flattened, showing a
        // persistent, visually "ringing"/wavy standing-pattern instead of
        // settling. Verified via a real headless A/B (temp diagnostic,
        // removed after use): tracking surface-height variance across x,
        // baseline settled to ~1.0-3.0 (never flat) even after 1000 steps;
        // raising drag alone made it WORSE (0.5 drag: variance 24+, real,
        // disclosed negative result, not hidden); a real IRL-proportional
        // gravity (just 0.3% of true g_grid, itself already ~10x the old
        // constant) settled to ~0.01-0.04 -- ~100x flatter, genuinely
        // stabilized. This is the SAME `gravity_fraction`-style real-IRL-
        // scaled convention `basic_sand_gui.rs`/`basic_fluids_gui.rs`
        // already use, ported here directly rather than another hand-picked
        // constant.
        gravity: Vec2::new(0.0, -981.0 * 0.003),
        // Real, CPU-proven mechanism (`fluid_near_wall_cfl_scale`, MEMORY.md's
        // fluid-recovery notes Round 7-9), ported to GPU 2026-08-09 (this
        // exact demo's real, reproduced crash: J up to 34653 under sustained
        // wall-contact compression). Tightens the CFL bound specifically for
        // strict-fluid particles near a wall -- the SAME real mechanism that
        // took `fluid_pressure_projection_gui.rs`'s hardest known scene from
        // exploding to a full, real 120-frame settle, now applied to this
        // demo's stiff-EOS (non-projection) fluid path.
        // Tried lowered to 5.0 (2026-08-09) to cut the substep tax further after
        // the spawn_water geometry fix below -- REVERTED, live-measured worse:
        // max J climbed to 40-50 by frame 120 (max_speed still rising, 57.8 ->
        // 68.6) vs the real, previously-verified ~5-6 stable plateau at 20.0.
        // Not a borderline call -- this is the same wall-contact divergence
        // mechanism this constant exists to stop, just delayed rather than
        // eliminated. Kept at 20.0, the proven-safe value.
        fluid_near_wall_cfl_scale: 20.0,
        // Regional substepping (`purring-swinging-cookie.md` Part A) stays OFF
        // here. Live-measured on this exact scene 2026-08-13, three-way A/B at
        // matched frames (same build, water-only):
        //   flag OFF      J max 36.5 (!), J min 0.22, sub 94 -> 5081 late
        //   flag ON m=1   J max 1.87,     J min 0.66, sub ~1685 throughout
        //   flag ON m=8   J max 11.8,     J min 0.023, sub 188 -> 2398
        // The m=1 run's good J was NOT the tiering working -- it came from the
        // retry ladder repeatedly halving dt (a safety net used as a design
        // mechanism, at ~9x the substep cost). With the correctly-derived
        // margin (see SimConfig::fluid_regional_substepping_fine_tier_margin)
        // the feature is cheap again but does NOT suppress the blowup. The
        // decisive datum: J reaches 36.5 with the feature entirely OFF, so
        // this scene's tensile/expansion blowup is a SEPARATE, pre-existing
        // bug that regional substepping neither causes nor cures. Fix that
        // first; re-evaluate this flag afterward on a scene that is actually
        // stable without it.
        fluid_regional_substepping_gpu_enabled: false,
        ..SimConfig::earth(GRID, 0.01, DT)
    };
    // TRUE root cause, found+proven 2026-08-06 (not the eos_stiffness rabbit hole
    // below, which was real but secondary): `initialize_particles`
    // (spacetime/solver/mod.rs) sets every particle's mass to
    // `config.particle_mass` -- a SimConfig-level CONSTANT, completely
    // independent of this SpawnRegion's own `spacing`. `SimConfig::earth`/
    // `standard` never override it, so it silently stays at `default()`'s 1.0.
    // A uniform material-point lattice represents an initially filled region
    // when `m = rho0 * spacing²`, so ΣV0 approximates the geometric area.
    // This calibrates quadrature mass and reference volume; the WC-MPM EOS
    // itself uses rho=rho0/J, never a kernel-density overwrite.
    // rest_density=0.1 for water, NOT the old 4.0 -- real SI fix, 2026-08-08,
    // see basic_fluids.rs's own doc for the full derivation (`rho_grid =
    // rho_kg_m3*dx_meters^2 = 1000*0.01^2 = 0.1` for real water at this
    // scene's scale). Mud's own `4.0` is intentionally unchanged (no
    // equally solid SI citation established for mud density tonight).
    // 0.5 (2026-08-14) -- matches `basic_fluids_gui.rs` exactly (4 PPC), was
    // 0.9 (~1.2 PPC) since a 45fps fix predating today's GPU solver revert.
    // Real, measured win: denser sampling gives the velocity-divergence
    // estimate less room to spuriously spike (sub=19 steady vs the sparse
    // case's climb into the hundreds).
    const SPACING: f32 = 0.5;
    // Real densities in SI, converted by this engine's own documented rule
    // `rho_grid = rho_kg_m3 * dx_meters^2` (see `NewtonianFluidMaterial::
    // weakly_compressible`), at this scene's dx_meters=0.01:
    //   water 1000 kg/m3 -> 1000 * 0.01^2 = 0.1
    //   mud   1800 kg/m3 -> 1800 * 0.01^2 = 0.18
    // 1800 is inside the real, standard geotechnical range for saturated
    // mud/wet soil (~1600-2000 kg/m3), the same range `FluidGranular::
    // saturated_loam_preset` already cites (rho_kg_m3: 1800).
    //
    // REAL ROOT-CAUSE FIX (2026-08-11) of the GPU fluid fps collapse: mud was
    // `4.0` here, i.e. an implied **40,000 kg/m3** -- nearly 2x denser than
    // osmium (22,590 kg/m3), the densest natural element. That was never a
    // real density: it is a leftover from before the 2026-08-08 SI mass fix,
    // which converted WATER only and explicitly deferred mud ("no equally
    // solid, verified SI citation ... was established tonight", basic_fluids.rs).
    // The result was a **40:1 mass ratio between two materials sharing one MPM
    // grid**. In MPM a shared node's velocity is momentum-weighted, so the
    // light material is effectively slaved to the heavy one: water particles
    // touching mud gathered a velocity gradient dominated by mud's momentum,
    // not their own physics, and their J drifted until it hit the real j_max
    // safety bound -- which then triggered a full (and futile, since the
    // condition is persistent rather than transient) retry ladder every batch.
    // That retry storm IS the measured 55 -> 1 fps collapse.
    // WATER_RHO_GRID itself is module-scope now (renderer setup in `State::new`
    // needs the same real value) -- see its own doc.
    const MUD_RHO_GRID: f32 = 0.18;
    const WATER_MASS: f32 = WATER_RHO_GRID * SPACING * SPACING;
    const MUD_MASS: f32 = MUD_RHO_GRID * SPACING * SPACING;
    let water_region = |box_size: IVec2, box_center: Vec2| SpawnRegion {
        spacing: SPACING,
        box_size,
        box_center,
        material_id: MAT_WATER,
        precompute_initial_volumes: true,
        mass_override: Some(WATER_MASS),
        ..SpawnRegion::for_sim(&config)
    };
    // Non-empty only for Vortex -- registered on `sim` after construction
    // below, see that call site's own comment.
    let mut vortex_fields: Vec<GpuFieldEntry> = Vec::new();
    let particles = match pattern {
        Pattern::DamBreak => {
            // x=20, not the old 11 -- at 11 the column's left edge (x=4) sat
            // only 2 cells past `boundary_thickness`'s near-wall trigger
            // (t=2), so `fluid_near_wall_cfl_scale=20` above was reading
            // almost the WHOLE column as permanently near-wall, not just
            // during real contact events -- confirmed live 2026-08-09:
            // `sub=3965` substeps/frame, cfl=0.0001. At x=20 (left edge
            // x=13) the column starts with real clearance.
            build_particles(
                &config,
                water_region(IVec2::new(14, 52), Vec2::new(20.0, 30.0)),
            )
        }
        Pattern::DropletImpact => {
            // A shallow, wide pool near the floor plus a small blob well
            // above it -- same water material/mass, two spawn regions
            // instead of one. ~28 cell fall distance (blob bottom ~38 to
            // pool top ~15) for a real, visible splash.
            let mut p = build_particles(
                &config,
                water_region(IVec2::new(50, 12), Vec2::new(32.0, 9.0)),
            );
            p.extend(build_particles(
                &config,
                water_region(IVec2::new(7, 7), Vec2::new(32.0, 42.0)),
            ));
            p
        }
        Pattern::Vortex => {
            // A real, contained whirlpool in the middle of a flat, resting
            // pool -- same wide-rectangle-on-the-floor geometry as
            // DropletImpact, not a disk (tried, rejected: read as an
            // isolated blob, not "a sea with a vortex in it").
            //
            // NOT a drain -- no mass leaves. A real sink (mass tapered away
            // near the center) was tried and rejected: it drained the whole
            // pool into a scattered mess within a minute, not the
            // persistent "sits there" feature asked for. A sink isn't
            // needed: Kelvin's circulation theorem (Thomson 1869 --
            // inviscid, barotropic flow under conservative forces conserves
            // circulation around a material loop) is why a real ocean
            // whirlpool (Corryvreckan, Moskstraumen) persists without
            // emptying the sea. Any visible core dip would come from
            // cyclostrophic balance (dp/dr = rho*v^2/r, same relation as
            // tornado/hurricane cores) driven by rotation alone, not mass
            // loss.
            //
            // Mechanism: a `GravityWellField` pull (`GpuFieldEntry::
            // gravity_well` -- the SAME Plummer-softened point-mass field
            // basic_orbital.rs proves via its own Kepler's-third-law test)
            // plus a small constant-angular-momentum ("free vortex") seed,
            // v = L/r floored at r=DRAIN_SOFTENING -- NOT solid-body
            // rotation (v = omega*r), which gives near-zero angular
            // momentum to particles near the center (L = r*(omega*r) ->0 as
            // r->0), so they fall in radially and get flung out hard on
            // close approach (real, live-measured N-body slingshot, not a
            // tuning artifact). Free-vortex gives every particle real
            // angular momentum from the start, avoiding that. Still just an
            // initial condition (real vortices always start from some small
            // pre-existing circulation, Shapiro 1962) -- from there, MPM's
            // own APIC transfer (Jiang et al. 2015) conserves it, the real
            // mechanism this pattern demonstrates.
            //
            // NO RadialConfinement -- tried, rejected: it pushes anything
            // beyond its radius inward unconditionally, so with a wide pool
            // it forcibly shoved ordinary resting water inward every
            // substep (real, measured violence, |v| up to 100+). Every
            // other pattern here already relies on gravity + floor + the
            // domain's slip boundary alone -- this one does too now.
            //
            // GPU CAVEAT: unlike GravityWell on CPU (proven via
            // basic_orbital's Kepler test), this GPU port's
            // grid_update.wgsl application had no test coverage before this
            // session (tests/gpu.rs only exercised linear_drag/
            // spatial_drag/radial_confinement) -- real, complete code, just
            // newly live-exercised.
            //
            // KNOWN, DISCLOSED LIMIT (a real engine gap, not a demo bug):
            // this GPU strict-fluid path uses a weakly-compressible Tait
            // EOS, not true incompressible pressure projection, so
            // cyclostrophic balance is only approximated. The engine's own
            // DCT/Gauss-Seidel solver (`fluid_pressure_projection_gui.rs`)
            // is the accurate path -- real current perf there, 16-30fps
            // depending on `GS_CORRECTION_SWEEPS` (see that constant's own
            // doc, `grid/pressure.rs`), the old "0.1-0.2fps" figure was
            // stale and is now corrected -- but that solver has its own
            // separate, unresolved wall-free-pool instability (see memory
            // project_vortex_siphon_saga_2026-08-15.md), so still not a
            // drop-in replacement here. Scoped as its own project.
            //
            // POOL_BOX/DRAIN_EDGE_R/DRAIN_SOFTENING below are live-verified
            // stable (2026-08-15, 1400+ frames one config, 650+ frames
            // this final one) -- not re-derived from scratch each time.
            // Honest, disclosed constraint: this vortex needs real room
            // (~44-cell working diameter) to stay stable, and GRID=64
            // doesn't leave much margin beyond that for a visibly larger
            // surrounding calm sea -- a bigger GRID would be the honest
            // way to get more visible margin around the same vortex; not
            // changed tonight.
            //
            // PERF STATUS (2026-08-15, real, unresolved -- see memory
            // project_vortex_siphon_saga_2026-08-15.md for the full
            // account, not re-litigated here): this pattern live-measured
            // at 10-15fps steady state vs the other two patterns' 28-38fps.
            // Two real attempts at a fix, BOTH tried and reverted:
            //  1. Weakening DRAIN_EDGE_ACCEL_FRACTION/SEED_EDGE_SPEED
            //     together barely moved fps (10-15 either way) AND broke
            //     the seed/drain's own centripetal balance (particles
            //     started drifting outward instead of orbiting -- the two
            //     constants were tuned as a matched pair, not independent
            //     knobs).
            //  2. Narrowing POOL_BOX (54->44) for more wall clearance --
            //     did NOT fix it either: the pool's own genuine outward
            //     spread (not just its initial footprint) still reached
            //     both walls.
            // Root driver, real not guessed: the GPU CFL scan here has NO
            // gradient/rotation-rate term (unlike the CPU path) -- driven
            // by raw max particle speed. A sustained rotating core keeps
            // SOME particle's speed persistently elevated (unlike a
            // settling puddle, which drops toward zero), so substep count
            // stays high for the pattern's entire lifetime, not just an
            // opening transient. Real fix needs either regional/local
            // substepping (already scoped as its own 5-8 day project) or a
            // fundamentally gentler mechanism -- not a quick constant
            // tweak. REVERTED to the last live-verified-stable values
            // below; perf remains a real, disclosed, open problem.
            const POOL_BOX: IVec2 = IVec2::new(54, 48);
            let pool_center = Vec2::new(32.0, 26.0);
            let drain_center = pool_center;
            const DRAIN_EDGE_R: f32 = 17.0;
            let drain_edge_r = DRAIN_EDGE_R;
            const DRAIN_EDGE_ACCEL_FRACTION: f32 = 0.02;
            let drain_gm =
                DRAIN_EDGE_ACCEL_FRACTION * config.gravity.length() * drain_edge_r * drain_edge_r;
            const DRAIN_SOFTENING: f32 = 2.0; // ~4 particle spacings (SPACING=0.5)
            vortex_fields.push(GpuFieldEntry::gravity_well(
                drain_center,
                drain_gm,
                DRAIN_SOFTENING * DRAIN_SOFTENING,
                0.0, // no cutoff -- softening alone bounds the near-core force
                0.0,
            ));
            // NO RadialConfinement here -- REJECTED, live-tested and
            // reverted 2026-08-15: `radial_confinement` pushes ANYTHING
            // beyond its radius inward unconditionally -- it doesn't know
            // "that's just resting pool water," not overflow from the
            // vortex. With a wide rectangular pool (most of it well beyond
            // any reasonable basin radius, by design -- that's the calm
            // sea), it was forcibly shoving ordinary resting water inward
            // every substep, colliding with the undisturbed pool and
            // producing real, measured violence (|v| up to 100+ within the
            // first ~20 frames). Every other pattern in this file (dam
            // break, droplet impact) already relies on gravity + the
            // floor + the domain's own slip boundary alone, with no
            // confinement field at all -- this pattern does the same now.
            const SWIRL_SEED_RADIUS: f32 = 22.0; // how far out the seed disturbance reaches, not a wall
            let mut p = build_particles(&config, water_region(POOL_BOX, pool_center));
            // Seed edge speed -- an order of magnitude below the drain's
            // own eventual spin-up speeds, a disturbance not a driving
            // force. Free-vortex (constant angular momentum) profile --
            // see the mechanism doc above for why, not solid-body
            // rotation.
            const SEED_EDGE_SPEED: f32 = 1.0;
            let seed_l = SEED_EDGE_SPEED * drain_edge_r;
            for particle in p.iter_mut() {
                let r = particle.x - drain_center;
                let dist = r.length();
                // Only the disk of particles actually within the vortex's
                // own working radius gets the seed -- the rest of the pool
                // is the calm, undisturbed "sea" the vortex sits in,
                // exactly as asked (a flat resting pool, one contained
                // feature in the middle, not the whole body spun up).
                if dist < SWIRL_SEED_RADIUS {
                    let d = dist.max(DRAIN_SOFTENING);
                    particle.v = (seed_l / (d * d)) * Vec2::new(-r.y, r.x);
                }
            }
            p
        }
    };
    // Mud material stays defined/registered below (Pattern::DamBreak's own
    // scene used to spawn it too) but unused by either pattern's own spawn
    // above -- TEMPORARY (2026-08-12): isolating both to water only, so the
    // classic dam-break case (and now droplet impact) can be verified real/
    // correct before mud's own separate, still-open Bingham-yield-stress
    // instability is chased further.
    let _ = MUD_MASS;

    // Real water: Cole 1948 Tait exponent (7.0) + real dynamic viscosity, not a
    // hand-picked 0.1/3.0 pair -- see NewtonianFluidMaterial::low_viscosity.
    //
    // eos_stiffness=2.5, NOT 100 -- rest_density=0.1 (the real SI fix, see
    // basic_fluids.rs's own doc) means `NewtonianFluidMaterial::timestep_bound`'s
    // `c2 = eos_stiffness*eos_power*density_ratio^(power-1)/rest_density` is
    // now 40x larger at the OLD eos_stiffness=100 for any given compression --
    // confirmed 2026-08-08 by basic_fluids.rs's CPU twin actually crashing
    // (`Tait pressure is unrepresentable`) under this exact scenario.
    // eos_stiffness=100 was measured/swept specifically at rest_density=4.0;
    // rescaling by the same factor rest_density shrunk (100*0.1/4.0=2.5)
    // restores the bit-identical c2 -- the already-verified 247fps/2.1% error
    // behavior -- at the new SI-correct density. Exact algebraic correction,
    // not a re-tune.
    // eos_power=3.0, NOT the real Cole 1948 water exponent (7.0) -- real,
    // disclosed compressibility-accuracy trade, found live 2026-08-09 via a
    // temp per-substep CFL-term dump: at this scene's real violent wall
    // impact, J drops to ~0.35-0.4 (genuine ~60% local compression, not a
    // bug), and `c2 = eos_stiffness*eos_power*ratio^(eos_power-1)/rest_density`
    // makes the acoustic term explode as ratio^6 at power=7 (measured
    // max_c2 up to 6605, acoustic_dt down to 0.00006 -- the actual dt-limiting
    // term, confirmed by the same dump: deformation_dt and gravity_dt stayed
    // 1000x+ larger throughout). This is why softening eos_stiffness alone
    // (tried at 0.25, 10x softer) barely moved the substep count: the
    // EXPONENT, not the base stiffness, is what turns a real compression
    // event into a numerical cliff for explicit integration. Lower Tait
    // exponents (n=1..4) are an established real-time-graphics WCSPH
    // trade-off for exactly this reason (Chorin's artificial-compressibility
    // method uses n=1; Monaghan's own WCSPH papers note n=7 is accurate but
    // numerically stiff). eos_stiffness kept near the SI value (1.0, not the
    // fully-correct 2.5) as a modest additional safety margin, not the main
    // lever this time.
    // REAL ROOT-CAUSE FIX (2026-08-11), replacing the soft-EOS approach the
    // comment block above describes. That approach is self-defeating, and the
    // live measurements now prove it: softening the EOS to cut substeps lets J
    // deviate further, and since `c2 = k*gamma*ratio^(gamma-1)/rho0` with
    // `ratio = 1/J`, a large J excursion pushes c2 right back up. Measured on
    // this exact scene at eos_stiffness=1.0/gamma=3.0: **J = [0.145, 24.3]**,
    // max_speed 28, water spread across the entire 64-cell domain, and
    // **sub=784** substeps/frame -- far WORSE than the J~0.35-0.4 excursion
    // that softening was introduced to fix.
    //
    // The decisive number: sound speed at rest was `sqrt(k*gamma/rho0)` =
    // sqrt(1*3/0.1) ~= 5.5 grid-units/s against a measured max flow speed of
    // 28 -- i.e. **Mach ~5**. Weakly-compressible SPH/MPM is only valid at
    // Mach < 0.1 (c_s >= 10*v_max; Monaghan 1994, Morris et al. 1997). At Mach
    // 5 this was not a weakly-compressible liquid at all, it was a gas -- which
    // is exactly why J swung two orders of magnitude and why the CFL needed
    // ~800 substeps (~100 batches x 3 blocking GPU round-trips) to contain it.
    // THAT is the measured 1 fps, and it is a physics failure surfacing as a
    // perf symptom, not a perf problem.
    //
    // A correctly stiff EOS is CHEAPER here, not costlier: it holds J ~= 1, so
    // `ratio^(gamma-1)` stays ~1 and c2 stays at its predictable baseline,
    // instead of being driven up by runaway compression.
    //
    // `c_ref` targets this scene's own real column-height free-fall physics,
    // v_max = sqrt(2*g*h), times the published 10x WCSPH safety factor.
    //
    // DISCLOSED, MEASURED, NOT SILENT: using this scene's real
    // `config.gravity.y` (-981*0.003 = -2.943 grid-cells/s^2 --
    // `config/mod.rs:448`'s own doc confirms `v += gravity*sub_dt` with
    // `sub_dt` in real seconds, so gravity IS an acceleration, the right
    // quantity for Torricelli) gives v_max_grid ~= 17.5, matching the
    // independently measured max_speed=16.03 at frame 60 almost exactly --
    // real confirmation the formula itself is correct.
    //
    // Tested at full strength (2026-08-11) and REJECTED: c2 scales as
    // v_max^2, so the fully-correct gravity made the acoustic term ~9.8x
    // stiffer at rest and the batch never reached frame 60 in 90s (worse
    // than the value below, not better) -- a real, measured regression, not
    // a guess. The peak free-fall speed is also only reached for an instant
    // at the moment of wall impact, a case ALREADY separately guarded by
    // `fluid_near_wall_cfl_scale`'s own dedicated 20x tightening -- sizing
    // the GLOBAL acoustic term to that same instantaneous peak double-pays
    // for one safety margin with another.
    //
    // HONEST STATUS: the constant below is a deliberately reduced,
    // real-time-affordable target, same disclosed category as this file's
    // own `eos_power=3.0` accuracy/perf trade above -- NOT a claim that this
    // is the scene's true v_max. Closing this gap for real (reaching the
    // fully-correct sound speed at 45fps+) needs regional/adaptive
    // substepping so calm parts of the domain stop paying the same CFL cost
    // as the violent wall-impact region -- already scoped, not yet built
    // (see the `regional-substepping` plan).
    const COLUMN_HEIGHT_CELLS: f32 = 52.0;
    const DERATED_GRAVITY_FOR_ACOUSTIC_SIZING: f32 = 0.3;
    let v_max_grid = (2.0 * DERATED_GRAVITY_FOR_ACOUSTIC_SIZING * COLUMN_HEIGHT_CELLS).sqrt();
    let c_ref_m_s = 10.0 * v_max_grid * config.dx_meters;
    // SECOND real bug in the previous version: `NewtonianFluidMaterial::
    // weakly_compressible` hard-codes Cole 1948's gamma=7 internally
    // (`fluid.rs`: `const GAMMA: f32 = 7.0`) -- the EXACT exponent this
    // file's own comment history (above) already measured as catastrophic on
    // this scene (c2 up to 6605 from `ratio^6` amplifying a modest J
    // excursion), which is why eos_power=3.0 was deliberately chosen over 7.0
    // in the first place. Calling `weakly_compressible` silently reintroduced
    // gamma=7. Fixed by inlining that helper's own real formula
    // (`tait_b_pa = rho_kg_m3 * c_ref_m_s^2 / gamma`, `fluid.rs:78`) with
    // this scene's already-justified gamma=3.0 instead.
    const WATER_EOS_POWER: f32 = 3.0;
    let water_tait_b_pa = 1000.0 * c_ref_m_s * c_ref_m_s / WATER_EOS_POWER;
    let mut water =
        NewtonianFluidMaterial::new(WATER_RHO_GRID, 1.0e-3, water_tait_b_pa, WATER_EOS_POWER);
    // Real, sourced bulk (second) viscosity, 2026-08-12 -- `NewtonianFluidMaterial::new`
    // hardcodes `bulk_viscosity: 0.0`, leaving this scene's Navier-Stokes stress tensor
    // (`fluid.rs`'s own `stress += 0.5*bulk_viscosity*div(v)*I`, standard and already
    // correctly implemented, just unused) with NO dissipation for volumetric
    // oscillation -- unlike `artificial_bulk_viscosity` just above it (von Neumann-
    // Richtmyer, correctly gated to compression-only: it's a SHOCK-capturing term,
    // real shocks only form under compression, so that gating is textbook-correct, not
    // a bug). Bulk viscosity is the real, standard, SYMMETRIC (both compression and
    // expansion) dissipative term that damps acoustic ringing after a violent impact --
    // directly matching the literature (Denner et al. 2023, "acoustic damper term in
    // weakly-compressible SPH": dissipates the acoustic component of pressure oscillation
    // from liquid impacts) and root-caused tonight: the tall column's own violent impact
    // shows real (non-diverging, confirmed non_finite=0) but UNDAMPED oscillation in J,
    // consistent with zero volumetric dissipation. Real value: water's bulk viscosity is
    // ~2.8-3.0x its shear (dynamic) viscosity (Litovitz & Davis; confirmed via
    // arxiv.org/pdf/1002.3029's acoustic-spectroscopy remeasurement, ratio ~3 across
    // 7-50C) -- applied here to the SAME `1.0e-3` dynamic_viscosity already used above.
    water.bulk_viscosity = 3.0 * 1.0e-3;
    // `settling_damping` was tried at 0.1 (2026-08-13) alongside the
    // restored J clamp + pressure_floor -- live-measured, that COMBINATION
    // over-damped the scene entirely (reported: "doesn't even move"). Left
    // off (0.0, the constructor default) pending a real, isolated re-test of
    // clamp+pressure_floor alone before adding this back in.
    //
    // 2026-08-14: root-caused instead -- the GPU solver's post-`cac544b`
    // strict-fluid/DCT-pressure/retry rewrite (`dcefbaf`) was the actual
    // source of the sustained instability this demo showed all session
    // (full account: [[project session notes]]), not a missing damping
    // term. The GPU solver internals are reverted to their pre-`dcefbaf`
    // state (proven live: 60fps steady, J bounded, real settling, on a
    // HARDER two-material scene than this one) -- this constant stays off,
    // pending whatever the rebuilt strict-fluid contract needs once that
    // work resumes.
    // Mud gets the SAME real Mach criterion and the SAME gamma as water
    // (2026-08-11): sizing only water correctly would leave mud as the
    // material that drives the CFL minimum (the adaptive dt is a MINIMUM over
    // ALL materials sharing the grid), so the substep count -- and the fps --
    // would barely move.
    //
    // THIRD real bug in the previous version: `mud_eos_stiffness` was derived
    // from `c_target_grid = c_ref_m_s/dx_meters` and `MUD_RHO_GRID` (0.18,
    // grid-converted) -- but `weakly_compressible`'s own doc (`fluid.rs:58-
    // 61`) is explicit that `eos_stiffness` stays in real SI Pascals; only
    // density gets the dx^2 grid conversion. That put mud's acoustic term and
    // water's in two DIFFERENT unit systems on the SAME shared-grid CFL
    // minimum. Fixed by using the identical real formula as water, with mud's
    // REAL density (1800 kg/m3, not the grid-converted MUD_RHO_GRID -- see
    // that constant's own doc above for the citation).
    const MUD_RHO_KG_M3: f32 = 1800.0;
    const MUD_EOS_POWER: f32 = 3.0;
    let mud_tait_b_pa = MUD_RHO_KG_M3 * c_ref_m_s * c_ref_m_s / MUD_EOS_POWER;
    let mut mud = BinghamFluidMaterial::new(MUD_RHO_GRID, 8.0, mud_tait_b_pa, MUD_EOS_POWER, 4.0);
    // Same real bulk-viscosity fix as water above, same ratio (~3x dynamic
    // viscosity) applied to mud's own dynamic_viscosity=8.0 -- mud-specific
    // bulk viscosity isn't a commonly published SI constant the way water's
    // is, so this is a disclosed extrapolation of water's real, cited ratio,
    // not a directly-measured mud citation. Left off would make mud the
    // undamped material instead (same reasoning as the Mach-criterion doc
    // above: this shared-grid CFL minimum is only as good as its weakest
    // link).
    mud.bulk_viscosity = 3.0 * 8.0;
    let mut registry = MaterialRegistry::with_default(Box::new(water));
    registry.insert(MAT_MUD, Box::new(mud));

    let mut sim = GpuSimulation::with_device(device, queue, config, particles, registry);
    // Vortex's real drain + basin -- see that match arm's own doc for the
    // full physical account. Registered once here (persists every substep
    // until cleared, see `add_force_field_gpu`'s own doc) since `sim` only
    // exists from this point on.
    for field in vortex_fields {
        sim.add_force_field_gpu(field);
    }
    // TEMPORARY (2026-08-12): regional-substepping plan's own Step 0 --
    // measure the real per-pass GPU breakdown on this exact hard scene
    // before writing any milestone-1 code (see `purring-swinging-cookie.md`
    // Part A). No new instrumentation -- `enable_profiling`/
    // `last_pass_timings_ns` already exist.
    sim.enable_profiling();
    // Real grid-mediated cohesion (CSF), 2026-08-12 -- built, sign-corrected,
    // real curvature-based physics (see `grid_update.wgsl`'s
    // `grid_cohesion_main_inner` doc for the full story), but DISABLED here.
    // The scene's actual runaway bug was root-caused and fixed elsewhere
    // (`fluid_state::force_stress_volume` -- the P2G force-scatter's own
    // volume input was unbounded, a genuine stress*volume feedback loop;
    // confirmed via a clean cohesion-OFF re-baseline that reproduced the
    // exact same runaway, then confirmed fixed the same way). Re-enabling
    // cohesion on top of that fix was tested directly: it reintroduces a
    // SEPARATE instability of its own (J=25 by frame 3) -- cohesion applies
    // its force directly to grid momentum in its own dedicated pass, which
    // completely bypasses `force_stress_volume`'s cap (that cap only scopes
    // the ordinary particle stress->force P2G scatter). Real surface
    // tension is genuine, wanted physics, but needs its own separate
    // stability pass (likely the same "under-resolved region produces an
    // untrustworthy curvature estimate" class of problem, not yet solved
    // for the grid-space force) before it's safe to ship enabled by
    // default. Left disabled, not deleted -- infra and math are real.
    // sim.set_cohesion_si(0.0728, 1000.0);

    // No scene-wide settling drag is applied.  Momentum changes only through
    // the WC-MPM stress, prescribed gravity, and geometric wall conditions.

    sim
}

/// Shared diagnostic line for a single traced/outlier particle -- DRY
/// extraction (2026-08-15): MIN_J_OUTLIER, OUTLIER, and MUD_OUTLIER below
/// each printed this same field set independently before this. Only mud's
/// own trace needs `friction_hardening` (its Bingham yield state); `None`
/// omits that segment entirely rather than printing a meaningless 0.0 for
/// water.
fn print_particle_trace(
    label: &str,
    idx: usize,
    p: &Particle,
    neighbors_tight: usize,
    neighbors_wide: usize,
    friction_hardening: Option<f32>,
) {
    let j = p.deformation_gradient.determinant();
    let fh = friction_hardening
        .map(|v| format!(" friction_hardening={v:.4}"))
        .unwrap_or_default();
    println!(
        "  {label} idx={idx} x=({:.3},{:.3}) v=({:.3},{:.3}) |v|={:.3} \
         F=[{:.3},{:.3};{:.3},{:.3}] J={j:.4} volume={:.6} initial_volume={:.6} \
         density={:.4} mass={:.6}{fh} neighbors(r=1.5)={neighbors_tight} \
         neighbors(r=3.0)={neighbors_wide}",
        p.x.x,
        p.x.y,
        p.v.x,
        p.v.y,
        p.v.length(),
        p.deformation_gradient.x_axis.x,
        p.deformation_gradient.y_axis.x,
        p.deformation_gradient.x_axis.y,
        p.deformation_gradient.y_axis.y,
        p.volume,
        p.initial_volume,
        p.density,
        p.mass,
    );
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
        let info = adapter.get_info();
        println!(
            "GPU adapter: {} ({:?}, backend={:?})",
            info.name, info.device_type, info.backend
        );
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
        let sim = make_sim_data(Arc::new(device), Arc::new(queue), Pattern::DamBreak);
        let mut renderer = Renderer::new(sim.device(), sim.particle_count(), fmt);
        renderer.set_camera(sim.queue(), GRID as u32, size.width, size.height, 0.6, true);
        renderer.set_color_mode(ColorMode::ByMaterial);
        // Ported from `basic_fluids_gui.rs` (2026-08-14): never called here, so
        // GridVolume/Surface thresholded this scene's real SI water mass
        // (WATER_RHO_GRID=0.1, a FULL cell) against the old absolute
        // mass_floor -- discarding almost everything (live-reported:
        // GridVolume rendered as near-empty fragments).
        renderer.set_grid_reference_cell_mass(WATER_RHO_GRID);
        // Same port: default 6x oversamples the surface grid for this
        // particle count, so each cell sees too few particles and reads
        // density noise as speckle (live-reported: bumpy, mottled Surface
        // mode). 4x gives each cell a real sample, at 0.44x the cost.
        renderer.set_surface_res_multiplier(4);
        // Real Beer-Lambert optics for grid-volume mode's per-material coloring
        // (ByMaterial's palette above is separate/unrelated -- see
        // material_sandbox_gpu.rs's own comment on this exact distinction). Water
        // reuses the same aesthetic SIGMA_WATER other demos use; mud gets a real
        // brownish estimate (not cited -- no real mud reflectance spectrum searched).
        renderer.set_optical_params(sim.queue(), MAT_WATER as usize, [0.85, 0.25, 0.07]);
        renderer.set_optical_params(sim.queue(), MAT_MUD as usize, [0.30, 0.20, 0.12]);
        // Real subsurface scattering + Fresnel specular -- these defaulted to
        // 0.0 (no visible effect at all in grid-volume/curvature-flow modes,
        // even after that shading math shipped) until wired here.
        // Water R0 = ((n_water - n_air) / (n_water + n_air))^2 with
        // n_water=1.33, n_air=1.0 -- a real, precisely derivable Schlick
        // (1994) base reflectance, not an estimate.
        //
        // REVISED (2026-07-30): the first attempt used sigma_s=1.5 (water)
        // / 6.0 (mud), borrowed from an unrelated tissue-scale test value,
        // without checking it against these materials' OWN sigma_a. Real
        // bug this caused, confirmed via screenshot: `albedo = sigma_s /
        // (sigma_s + sigma_a)` (see grid_volume.wgsl/curvature_flow.wgsl's
        // fs_main) -- water's sigma_a is already low (esp. blue at 0.07),
        // so almost ANY sigma_s dominates that channel's albedo, pulling
        // the whole color toward the cream-white scatter_glow instead of
        // water's real blue (read as "foam"/washed-out, user's own word).
        // Fixed by choosing sigma_s relative to each material's own
        // smallest sigma_a channel, targeting a bounded albedo (~0.2-0.3)
        // instead of an unrelated borrowed magnitude: water's min sigma_a
        // is 0.07 (blue) -> sigma_s=0.03 keeps albedo_blue ~= 0.3; mud's
        // min sigma_a is 0.12 -> sigma_s=0.08 keeps albedo_blue ~= 0.4 (mud
        // is real turbid/particulate, tolerates a bit more per limnology,
        // still bounded so it doesn't dominate).
        renderer.set_optical_scattering(sim.queue(), MAT_WATER as usize, 0.03);
        renderer.set_specular_r0(sim.queue(), MAT_WATER as usize, 0.02);
        renderer.set_optical_scattering(sim.queue(), MAT_MUD as usize, 0.08);
        renderer.set_specular_r0(sim.queue(), MAT_MUD as usize, 0.005);
        // NOTE: `Renderer::set_light_dir` (real infrastructure, reuses
        // `SimConfig::light_dir`) exists but is deliberately NOT called here
        // yet -- this config's own `light_dir` is left at its default
        // (straight up, opposite gravity; sensible for phototropism, not
        // necessarily for this demo's lighting look), and wiring it in now
        // would change the visual light angle at the same time as the
        // scattering/specular fix above, confounding which change did what
        // while that's still being visually verified. The renderer's own
        // default light angle (matches the old hardcoded shader value)
        // applies until this is deliberately turned on.
        println!(
            "fluids GPU: {} particles  |  LMB push  RMB pull  G grid-volume  R reset  1/2 pattern (3 vortex disabled: perf, 4 reserved)  Q quit",
            sim.particle_count()
        );
        Self {
            surface,
            surface_config: sc,
            sim,
            renderer,
            cursor_pos: [0.0; 2],
            lmb: false,
            rmb: false,
            frame: 0,
            fps_timer: std::time::Instant::now(),
            fps_frames: 0,
            render_mode: RenderMode::Particles,
            // Vortex stays real, working code (see its own match-arm doc) --
            // just not reachable via the live Digit3 keypress (2026-08-15,
            // real perf gap: 10-15fps vs 28-38fps for patterns 1/2, not
            // ready for casual demo use). Still constructible for real dev
            // testing via this env var, same convention as
            // `EMERGE_DEBUG_PRESSURE` elsewhere this session -- keeps the
            // variant genuinely used (not dead code) without forcing it on
            // anyone running the demo normally.
            pattern: if std::env::var("EMERGE_VORTEX_DEBUG").is_ok() {
                Pattern::Vortex
            } else {
                Pattern::DamBreak
            },
            // NOT `::standard()` (2026-08-07 fix): that hardcodes a 64-step-
            // per-render catch-up cap, sized for cheap physics steps. Once
            // `SimConfig::max_substeps_per_step` needed raising to 150 for a
            // correctly-stiff EOS (see make_sim_data's own doc), the two caps
            // compound: a slow step_frame() call falls behind real time, the
            // accumulator asks for MORE catch-up steps next render, each one
            // ALSO up to 150 substeps plus its own blocking GPU sync -- a
            // real, measured scheduling death-spiral (confirmed live: fps
            // ratchets 4->2->1->0 while GPU usage pins), not physics cost.
            // Capping catch-up at 1 means a slow frame is visually slow
            // motion, never a compounding spiral.
            stepper: FixedStepController::new(FixedStepConfig {
                dt: DT,
                simulation_speed: RENDER_FPS_TARGET * DT,
                max_substeps_per_frame: 1,
                max_frame_delta: 1.0 / 15.0,
            }),
            last_instant: std::time::Instant::now(),
            max_steps_seen: 0,
        }
    }

    fn resize(&mut self, w: u32, h: u32) {
        if w == 0 || h == 0 {
            return;
        }
        self.surface_config.width = w;
        self.surface_config.height = h;
        self.surface
            .configure(self.sim.device(), &self.surface_config);
        self.renderer
            .set_camera(self.sim.queue(), GRID as u32, w, h, 0.6, true);
    }

    fn cursor_grid(&self) -> Vec2 {
        // Exact inverse of set_camera's NDC projection (accounts for
        // aspect-ratio letterboxing) -- the naive width/height scaling this
        // replaced was the same bug already found and fixed in
        // basic_fluids_gui.rs earlier this session, just never ported here:
        // it only matched the grid at a square window, drifting off (LMB/
        // RMB push/pull landing at the wrong point, reading as "the cursor
        // doesn't work") at any other aspect ratio.
        let (gx, gy) = self.renderer.screen_to_grid(
            self.cursor_pos[0],
            self.cursor_pos[1],
            self.surface_config.width,
            self.surface_config.height,
        );
        Vec2::new(gx, gy)
    }

    fn reset(&mut self) {
        let (device, queue) = (self.sim.device().clone(), self.sim.queue().clone());
        self.sim = make_sim_data(device, queue, self.pattern);
        self.frame = 0;
        // Real elapsed time since the LAST reset (possibly seconds ago, if the
        // window was idle) must not be replayed as a burst of catch-up steps.
        self.stepper.reset();
        self.last_instant = std::time::Instant::now();
        println!("reset ({})", self.pattern.label());
    }

    fn update_and_render(&mut self) {
        if self.lmb || self.rmb {
            let mag = if self.lmb { 2.0 } else { -2.0 };
            self.sim.apply_radial_impulse(self.cursor_grid(), 5.0, mag);
        }
        let output = match self.surface.get_current_texture() {
            Ok(t) => t,
            Err(_) => return,
        };
        let now = std::time::Instant::now();
        let frame_delta = (now - self.last_instant).as_secs_f32();
        self.last_instant = now;
        let steps = self.stepper.steps_for_frame(frame_delta);
        self.max_steps_seen = self.max_steps_seen.max(steps);
        for _ in 0..steps {
            self.sim.step_frame();
            self.frame += 1;
            // Gated per real SIMULATION step, not per render call -- `steps`
            // can be 0 for several consecutive render frames right after
            // startup (the accumulator hasn't crossed `DT` of real time
            // yet), which used to make `self.frame` sit at the same value
            // across many renders and re-print the same "frame N" diagnostic
            // repeatedly. Checking it here instead means it fires exactly
            // once every 60 simulated frames, matching what the log is
            // actually meant to sample (simulation state, not render cadence).
            if self.frame.is_multiple_of(1) {
                // TEMPORARY: re-verifying after cleanup, per user's direct challenge
                log_frame_gpu(self.frame, DT, self.sim.particles(), LABELS, 1);
                let snap = self.sim.diagnostics_snapshot();
                println!(
                    "  non_finite={}  out_of_bounds={}  max_speed={:.3}  sub={}  cfl={:.4}",
                    snap.non_finite_particle_values,
                    snap.out_of_bounds_particles,
                    snap.max_particle_speed,
                    snap.substeps_last_step,
                    snap.cfl_number,
                );
                // TEMPORARY (2026-08-15 Step 0, GPU density/volume parity plan):
                // unconditional density/volume range for water -- the existing
                // MIN_J_OUTLIER/OUTLIER traces below never fired in this scene
                // (gated on substeps>2000 or max_speed>20, neither of which this
                // capped-at-60 run reaches), so they never actually captured the
                // one field this investigation needs direct evidence on. J is
                // structurally bounded by the GPU clamp [0.5,2.0] already, but
                // density/volume are a SEPARATE, unguarded field on GPU -- this
                // is the real, direct measurement the plan's Step 0 asked for.
                if self.frame.is_multiple_of(30) {
                    let particles = self.sim.particles();
                    let (mut dmin, mut dmax, mut vmin, mut vmax) =
                        (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
                    for p in particles.iter().filter(|p| p.material_id == MAT_WATER) {
                        dmin = dmin.min(p.density);
                        dmax = dmax.max(p.density);
                        vmin = vmin.min(p.volume);
                        vmax = vmax.max(p.volume);
                    }
                    println!(
                        "  DENSITY_VOLUME water: density=[{dmin:.4},{dmax:.4}] (rest={WATER_RHO_GRID:.4}) volume=[{vmin:.6},{vmax:.6}]"
                    );
                }
                // TEMPORARY: regional-substepping plan's Step 0 measurement
                // -- sparse (every 10 frames), since per-pass GPU profiling
                // readback itself blocks and would distort the very timing
                // being measured if done every frame.
                if self.frame.is_multiple_of(10) {
                    let (cfl_scan_ns, encode_ns, wait_ns, readback_ns, total_ns) =
                        self.sim.last_cpu_timings_ns();
                    println!(
                        "  TIMING cfl_scan={cfl_scan_ns:.0}ns encode={encode_ns:.0}ns wait={wait_ns:.0}ns readback={readback_ns:.0}ns total={total_ns:.0}ns"
                    );
                    if let Some(passes) = self.sim.last_pass_timings_ns() {
                        for (label, ns) in passes {
                            println!("    PASS {label}: {ns:.0}ns");
                        }
                    }
                }
                // TEMPORARY: hunting the "settled fluid still burns 5000+
                // substeps/frame" mystery -- gated on substep count, NOT
                // speed, since the earlier speed-gated OUTLIER trace below
                // never fires during these frames (max_speed is low, 1-2.3,
                // while sub spikes to 5000+). Hypothesis: the acoustic-CFL
                // term (max_c2, Tait EOS stiffness ~ ratio^(power-1)) is
                // pinned high by ONE particle stuck at a low, unchanging J
                // (min J was observed flat at ~0.21-0.22 across many
                // consecutive frames in an earlier run's log, not decaying
                // back toward 1 -- looks like a static wedge, not a
                // transient compression wave settling).
                if snap.substeps_last_step > 2000 {
                    let particles = self.sim.particles();
                    if let Some((idx, p)) = particles
                        .iter()
                        .enumerate()
                        .filter(|(_, p)| p.material_id == MAT_WATER)
                        .min_by(|(_, a), (_, b)| {
                            a.deformation_gradient
                                .determinant()
                                .total_cmp(&b.deformation_gradient.determinant())
                        })
                    {
                        let neighbors_tight = self.sim.count_near(p.x, 1.5, MAT_WATER);
                        let neighbors_wide = self.sim.count_near(p.x, 3.0, MAT_WATER);
                        print_particle_trace(
                            "MIN_J_OUTLIER",
                            idx,
                            p,
                            neighbors_tight,
                            neighbors_wide,
                            None,
                        );
                    }
                }
                // TEMPORARY: trace the exact outlier particle mechanism
                if snap.max_particle_speed > 20.0 {
                    let particles = self.sim.particles();
                    if let Some((idx, p)) = particles
                        .iter()
                        .enumerate()
                        .filter(|(_, p)| p.material_id == MAT_WATER)
                        .max_by(|(_, a), (_, b)| a.v.length().total_cmp(&b.v.length()))
                    {
                        let neighbors_tight = self.sim.count_near(p.x, 1.5, MAT_WATER);
                        let neighbors_wide = self.sim.count_near(p.x, 3.0, MAT_WATER);
                        print_particle_trace(
                            "OUTLIER",
                            idx,
                            p,
                            neighbors_tight,
                            neighbors_wide,
                            None,
                        );
                        // TEMPORARY: dump the outlier's own 9-cell G2P gather
                        // stencil directly (same base/window g2p.wgsl uses:
                        // floor(p.x) +/- 1) to see which cell(s) actually
                        // feed the spike, and whether their values look like
                        // real physics or a stale/misread buffer.
                        let grid_res = self.sim.config().grid_res;
                        let cells = self.sim.grid_cells_blocking();
                        let base_x = p.x.x.floor() as i32;
                        let base_y = p.x.y.floor() as i32;
                        for dj in -1..=1 {
                            for di in -1..=1 {
                                let cx = base_x + di;
                                let cy = base_y + dj;
                                if cx < 0
                                    || cy < 0
                                    || cx >= grid_res as i32
                                    || cy >= grid_res as i32
                                {
                                    println!("    cell({di:+},{dj:+}) OUT_OF_BOUNDS");
                                    continue;
                                }
                                let idx = ((cy as usize) * grid_res + (cx as usize)) * 4;
                                println!(
                                    "    cell({di:+},{dj:+}) [{cx},{cy}] mom_or_vel=({:.4},{:.4}) mass={:.6}",
                                    cells[idx],
                                    cells[idx + 1],
                                    cells[idx + 2],
                                );
                            }
                        }
                    }
                    // TEMPORARY: same trace, for mud specifically -- water's
                    // own trace above showed water genuinely bounded after
                    // the mass-trust-floor fix, but the aggregate mud v/J
                    // range was still spiking (293/210) in the same run,
                    // confirming this is now a mud-specific (Bingham
                    // yield-stress material), not water-specific, remaining
                    // problem.
                    if let Some((idx, p)) = particles
                        .iter()
                        .enumerate()
                        .filter(|(_, p)| p.material_id == MAT_MUD)
                        .max_by(|(_, a), (_, b)| a.v.length().total_cmp(&b.v.length()))
                    {
                        let neighbors_tight = self.sim.count_near(p.x, 1.5, MAT_MUD);
                        let neighbors_wide = self.sim.count_near(p.x, 3.0, MAT_MUD);
                        print_particle_trace(
                            "MUD_OUTLIER",
                            idx,
                            p,
                            neighbors_tight,
                            neighbors_wide,
                            Some(p.friction_hardening),
                        );
                    }
                }
            }
        }
        self.fps_frames += 1;
        if self.fps_timer.elapsed().as_secs_f32() >= 2.0 {
            let fps = self.fps_frames as f32 / self.fps_timer.elapsed().as_secs_f32();
            println!(
                "frame={} fps={:.0} max_steps_per_render={}",
                self.frame, fps, self.max_steps_seen
            );
            self.fps_timer = std::time::Instant::now();
            self.fps_frames = 0;
            self.max_steps_seen = 0;
        }
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        match self.render_mode {
            RenderMode::GridVolume => {
                self.renderer.render_grid_volume(
                    self.sim.device(),
                    self.sim.queue(),
                    GridVolumeSource {
                        grid: self.sim.grid_buffer(),
                        material_mass: self.sim.material_mass_buffer(),
                        material_mass_enabled: true,
                    },
                    &view,
                    true,
                );
            }
            RenderMode::Surface => {
                // Real two-phase reconstruction (shipped 2026-07-30, see
                // `curvature_flow.wgsl`'s own "two-phase extension" doc):
                // water and mud are two genuinely distinct materials in
                // this scene, so each gets its OWN independently-smoothed
                // surface instead of merging into one blob where they
                // touch -- the whole real reason this demo exists (a real
                // phase boundary, not a single-material scene).
                self.renderer.render_surface_reconstruction_dual_phase(
                    self.sim.device(),
                    self.sim.queue(),
                    DualPhaseSurfaceSource {
                        particle_buf: self.sim.particle_buffer(),
                        particle_count: self.sim.particle_count(),
                        grid_res: GRID as u32,
                        material_id_a: MAT_WATER,
                        material_id_b: MAT_MUD,
                        dt: DT,
                    },
                    &view,
                    true,
                );
            }
            RenderMode::Particles => {
                self.renderer.render_gpu(
                    self.sim.device(),
                    self.sim.queue(),
                    self.sim.particle_buffer(),
                    self.sim.particle_count(),
                    &view,
                    true,
                );
            }
        }
        output.present();
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        let w = Arc::new(
            el.create_window(
                winit::window::WindowAttributes::default()
                    .with_title("emerge -- Fluids GPU [Water / Bingham Mud]")
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
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => match key {
                KeyCode::Escape | KeyCode::KeyQ => el.exit(),
                KeyCode::KeyR => s.reset(),
                KeyCode::Digit1 => {
                    s.pattern = Pattern::DamBreak;
                    s.reset();
                }
                KeyCode::Digit2 => {
                    s.pattern = Pattern::DropletImpact;
                    s.reset();
                }
                KeyCode::Digit3 => {
                    println!(
                        "vortex disabled in this build: real, GPU-verified physics but 10-15fps \
                         (vs 28-38fps for patterns 1/2) -- see fluid_solver_perf_reality_check \
                         memory. Not removed: run with EMERGE_VORTEX_DEBUG=1 set to start on it."
                    );
                }
                KeyCode::Digit4 => {
                    println!("pattern slot not built yet");
                }
                KeyCode::KeyG => {
                    s.render_mode = match s.render_mode {
                        RenderMode::Particles => RenderMode::GridVolume,
                        RenderMode::GridVolume => RenderMode::Surface,
                        RenderMode::Surface => RenderMode::Particles,
                    };
                    if s.render_mode == RenderMode::GridVolume {
                        s.sim.attach_grid_material_render_gpu();
                    }
                    println!(
                        "render mode: {}",
                        match s.render_mode {
                            RenderMode::Particles => "particles",
                            RenderMode::GridVolume => "grid-volume",
                            RenderMode::Surface => "curvature-flow surface",
                        }
                    );
                }
                _ => {}
            },
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

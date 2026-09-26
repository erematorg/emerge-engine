extern crate emerge_engine as emerge;

/// GPU Newtonian water dam-break, zero CPU readback.
///
///   Mat 0  Newtonian water (blue) -- Tait EOS + deviatoric viscosity
///
///   cargo run --example basic_fluids_gpu --features "render"
use std::sync::Arc;

use emerge::diagnostics::log_frame_gpu;
use emerge::render::{
    ColorMode, GpuRenderParams, GridVolumeSource, PhysicalRenderContract,
    PhysicalRenderContractParams, Renderer, SurfaceReconstructionSource,
};
use emerge::{
    FixedStepConfig, FixedStepController, GpuFieldEntry, GpuSimulation, MaterialRegistry,
    NewtonianFluidMaterial, Particle, SimConfig, SpawnRegion, build_particles,
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
const LABELS: &[(u32, &str)] = &[(MAT_WATER, "water")];
// Module scope (not local to `make_sim_data`, which only builds the sim, not
// the renderer): `Renderer::set_grid_reference_cell_mass` needs this same
// real SI value from `State::new`, a separate `impl` method -- see that call
// site's own comment.
const WATER_RHO_GRID: f32 = 0.1;

/// Installs the scene's real optical description. There is one path, not a
/// choice of looks: the material declares its own measured constants and the
/// scene declares its own physical scale and lighting. Nothing here selects
/// an appearance.
fn apply_optics(
    renderer: &mut Renderer,
    queue: &wgpu::Queue,
    registry: &emerge::MaterialRegistry,
    dx_meters: f32,
) {
    // Measured constants come from the materials themselves. Adding a
    // material with declared optics needs no change here at all.
    renderer.adopt_material_optics(queue, registry);
    renderer.set_physical_render_contract(
        queue,
        PhysicalRenderContract::new(PhysicalRenderContractParams {
            dx_meters,
            // What this 2-D slice stands for out of plane. The domain is 64
            // cells of 1 cm, so the tank it represents is about this deep.
            // It is a statement about the scene, not a dial for how blue the
            // water should look -- water this shallow is nearly colourless,
            // and that is what a 30 cm tank looks like.
            view_thickness_meters: 0.3,
            // Declared lighting: an overcast sky above, a darker backdrop
            // behind. Equal values would make reflection and scattering
            // cancel out of view.
            incident_radiance_w_m2_sr: [1.0; 3],
            background_radiance_w_m2_sr: [0.25; 3],
            display_white_radiance_w_m2_sr: [1.0; 3],
            camera_direction: glam::Vec3::new(0.0, 0.0, -1.0),
            light_direction: glam::Vec3::new(0.0, 1.0, -1.0),
        })
        .expect("physical render contract"),
    );
    // Fresnel base reflectance from water's real refractive index:
    // R0 = ((1.333 - 1) / (1.333 + 1))^2.
    renderer.set_specular_r0(queue, MAT_WATER as usize, 0.0204);
}

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
        // water scene needed 22-63/frame at cfl=0.5) -- 8 would have
        // silently capped it and (with the honest-dropped-time fix already
        // shipped) reported most of each frame's time as unadvanced.
        //
        // Raised 60 -> 150 (2026-09-16), matching `basic_fluids.rs`'s OWN
        // value for this exact scene (see that file's own doc): GPU's CFL
        // scan was missing several real CPU-only terms (deformation-
        // gradient ODE bound, shock-viscosity compression correction,
        // single-particle instability bound, predictive near-wall
        // tightening), live-confirmed by `sub` pinning at exactly 60.
        //
        // Raised again 150 -> 1000 (2026-09-16, same night): the real
        // `pressure_floor` fix (see `HANDOFF_fluid_gpu_thin_layer_bug.md`'s
        // Tenth/Eleventh passes) lets the EOS generate much larger real
        // pressure gradients in the tension regime -- a larger real
        // pressure response means a larger effective sound speed
        // (`c_sound = sqrt(dp/drho)`), which directly shrinks the
        // CFL-stable timestep. `sub` was pinned at exactly 150 in every
        // run after that fix, live-confirmed evidence the CFL scan was
        // already asking for more and being truncated -- the truncation
        // itself was the source of a genuine, reproducible, wildly
        // unstable transient (`v` spiking past 179 units/s on frame 40).
        // Kept at 1000 as headroom only: with the near-wall tightening gone
        // (see `fluid_near_wall_cfl_scale` below) every pattern takes ~19-27
        // substeps/frame, never near this cap.
        max_substeps_per_step: 1000,
        // TESTED (2026-09-16): raised to `true` (engine default) to check
        // whether the real, cited affine-speed CFL contribution
        // (`affine_cfl_speed_contribution`) would catch the growing
        // velocity-gradient instability traced on particle idx=1994 -- ZERO
        // effect: frames 1-12 (the entire seed-and-growth phase, C growing
        // from exactly 0 to 9.487 during pure free-fall) were byte-identical
        // to the flag-off baseline. This bound only meaningfully tightens dt
        // once grad_norm is already large (dozens+), by which point the
        // runaway has already happened -- it doesn't touch the seed. Left
        // off (matches this file's prior state; no measured benefit to
        // justify the extra CFL-scan cost).
        cfl_include_affine_speed: false,
        // This demo's gravity (below) is ~10x CPU basic_fluids.rs's, so the
        // same eos_stiffness=1000 needs a tighter CFL here to stay
        // admissible -- confirmed live: cfl=0.5 (fine for the weaker-gravity
        // CPU scene) panics on frame 1 here ("GPU strict fluid update became
        // inadmissible"). Not swept per-value for this scene yet; 0.1 is the
        // known-stable pairing from the CPU sweep at higher compression.
        // 0.3, not 0.1 (2026-08-13) -- matches `basic_fluids.rs`'s own
        // change and the standard explicit CFL number range (0.2-0.4;
        // Monaghan 0.25-0.3 for SPH, MLS-MPM commonly 0.3-0.5). See that
        // demo's own comment for the full reasoning: 0.1 was 3x more
        // conservative than any cited solver, and the failure it was
        // protecting against traced to a too-soft EOS, not to C.
        // 0.4, not 0.3 (2026-09-19). Two sources for the number: Becker & Teschner
        // 2007's WCSPH time step (eq. 18, after Monaghan 1992) is `0.4*h/c_s`, and Bai
        // & Schroeder 2022 ("Stability analysis of explicit MPM", CGF 41) put the Von
        // Neumann stability limit of THIS scheme -- APIC with quadratic B-splines -- at
        // `dt <= 1.0*dx/c` (their Figure 4: analytic f = 1.0000, measured 1.0007 in 2D),
        // so 0.4 keeps a 2.5x margin on the proven bound, with boundary/isolated
        // particles covered by the separate Sun/Shinar/Schroeder 2020 bound the CFL scan
        // already applies. It only became safe once the GPU picked its timestep per
        // substep (`adaptive_cfl.wgsl`): with the old frame-frozen dt, 0.4 spiked the
        // vortex's J to 1.28 where the CPU, same scene and coefficient, stayed at 1.025.
        // With the adaptive timestep the GPU measures 1.024 -- the CPU's own value.
        // Costs ~52 substeps/frame instead of ~69. 0.5 was tested too and rejected: the
        // vortex's J goes to 1.43.
        material_cfl_coefficient: 0.4,
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
        // scaled convention `basic_sand.rs`/`basic_fluids.rs`
        // already use, ported here directly rather than another hand-picked
        // constant.
        gravity: Vec2::new(0.0, -981.0 * 0.003),
        // No `fluid_near_wall_cfl_scale` (engine default 1.0). It used to be
        // 20.0 here, a 20x smaller timestep for fluid near a wall, added
        // against wall-contact blow-ups (J up to 34653) that were really two
        // GPU bugs: fixed-point P2G atomics dropping small momentum
        // contributions, and the driver misreading `C[1][1]` in the J update
        // (see `trace2` in `particles_update.wgsl`). With both fixed, all
        // three patterns match their CPU twin without it
        // (`fragmentation_check_{cpu,gpu}.rs`, PATTERN=dam|drop|vortex).
        // Keeping it cost ~20x the substeps once water touched a wall (~400
        // vs ~21 per frame), and the vortex came out damped (peak speed ~9
        // vs the CPU reference's ~22).
        // `fluid_regional_substepping_gpu_enabled` used to be set here.
        // REMOVED (2026-09-17): every "re-tested, no benefit" note this
        // field once carried was, in fact, toggling a config flag with no
        // implementation behind it (the real code was part of a rewrite
        // reverted 2026-08-14 for unrelated reasons; the field survived,
        // the behavior didn't). A real feasibility check confirmed the
        // technique's own precondition -- a calm region existing next to
        // a violent one -- doesn't hold on this scene anyway. Full account:
        // `KNOWN_LIMITATIONS.md` entry 2, `src/spacetime/solver/config/
        // mod.rs`'s own removal note.
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
    // scene's scale).
    // 0.5 (2026-08-14) -- matches `basic_fluids.rs` exactly (4 PPC), was
    // 0.9 (~1.2 PPC) since a 45fps fix predating today's GPU solver revert.
    // Real, measured win: denser sampling gives the velocity-divergence
    // estimate less room to spuriously spike (sub=19 steady vs the sparse
    // case's climb into the hundreds).
    const SPACING: f32 = 0.5;
    // WATER_RHO_GRID itself is module-scope now (renderer setup in `State::new`
    // needs the same real value) -- see its own doc.
    const WATER_MASS: f32 = WATER_RHO_GRID * SPACING * SPACING;
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
            // DCT/Gauss-Seidel solver (`fluid_pressure_projection.rs`)
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
    // Sized from the scene's REAL gravity (v_max ~= 17.5 grid/s, matching the
    // measured peak of ~16). It used to use a derated 0.3 instead of 2.943,
    // chosen while two GPU bugs (fixed-point P2G atomics, the `C[1][1]`
    // driver misread -- see `trace2` in `particles_update.wgsl`) made the
    // full-strength EOS look worse, and while `fluid_near_wall_cfl_scale`
    // was still "guarding" impacts. That left c ~3.1x under the WCSPH rule
    // (Mach ~0.3, not <= 0.1), and it showed: the pool "breathed" -- mean J
    // oscillating 0.988 <-> 1.000 with a ~0.7s period, exactly the acoustic
    // round trip 4*depth/c = 40/55.8 -- so the water bounced on its own
    // compressibility like a jelly. At the real v_max, all three patterns
    // stay coherent (0 isolated particles over 150 frames), J stays within
    // +-6%, mean J holds at 0.999 with no oscillation
    // (`fragmentation_check_gpu.rs`, GRAV_SIZING env). Cost: ~60 substeps/
    // frame instead of ~21.
    const COLUMN_HEIGHT_CELLS: f32 = 52.0;
    let v_max_grid = (2.0 * config.gravity.length() * COLUMN_HEIGHT_CELLS).sqrt();
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
    // REAL FIX (2026-09-17): `dynamic_viscosity` was assigned water's raw SI
    // value (1.0e-3 Pa.s) directly, with NO SI-to-grid conversion -- the
    // SAME bug pattern as `pressure_floor` below, just never caught until
    // now. `fluid.rs`'s own stress law (`stress += eff_viscosity *
    // strain_dev`, strain rate in 1/s grid-time) needs `eff_viscosity` in
    // grid units, and this engine already has the dimensionally-correct
    // conversion for exactly this (`SimConfig::visc_from_si_physical`,
    // `eta_SI/(rho*dx^2)`, doc'd against this exact consumption pattern).
    // Must pair with the SAME density-normalized family `pressure_floor`
    // below already uses (`stress_from_si_physical`) -- mixing raw and
    // density-normalized conventions in the same stress tensor is wrong
    // (see `q_factor_elastic_viscosity_pa_s`'s own doc for a real, prior
    // instance of exactly that mistake, ~917x error, a different material).
    // Real effect here: raw 1.0e-3 was ~10x too weak (correct grid value
    // 0.01) -- real, disclosed, but NOT the fix for the splash-disintegration
    // instability (verified separately: even 10x more molecular viscosity is
    // far too small to explain or damp the observed C-matrix growth rate).
    const WATER_DYNAMIC_VISCOSITY_PA_S: f32 = 1.0e-3;
    const WATER_RHO_SI_KG_M3_FOR_VISC: f32 = 1000.0;
    let water_dynamic_viscosity =
        config.visc_from_si_physical(WATER_DYNAMIC_VISCOSITY_PA_S, WATER_RHO_SI_KG_M3_FOR_VISC);
    let mut water = NewtonianFluidMaterial::new(
        WATER_RHO_GRID,
        water_dynamic_viscosity,
        water_tait_b_pa,
        WATER_EOS_POWER,
    );
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
    // 7-50C) -- applied to the SAME real, now-correctly-SI-converted
    // `water_dynamic_viscosity` above (was `3.0 * 1.0e-3` raw, same unit bug).
    water.bulk_viscosity = 3.0 * water_dynamic_viscosity;
    // This fluid IS water, so it declares water's own measured optical
    // constants. The renderer adopts them; no colour is chosen anywhere.
    water.optics = Some(emerge::materials::optical::pure_water());
    // Liquid water at 25 C, CRC Handbook. Lets frictional dissipation become
    // a real temperature rise rather than only being accounted for.
    water.specific_heat_j_kg_k = 4182.0;
    // REAL FIX (2026-09-16) -- root cause of the long-standing thin-layer
    // collapse (`HANDOFF_fluid_gpu_thin_layer_bug.md`, Tenth pass, full
    // investigation trail there). `pressure_floor` (constructor default
    // -0.1) is a bare GRID-UNIT constant that was never run through this
    // engine's own SI-to-grid conversion pipeline -- unlike `water_tait_b_pa`
    // just above, which IS properly SI-derived in this same file. Real
    // cavitation onset for water in practice (dissolved-gas nucleation, the
    // standard engineering figure) is ~-0.1 MPa = -100,000 Pa gauge.
    // Converted through `stress_from_si_physical` (the SAME conversion
    // `eos_stiffness` itself already uses), that lands orders of magnitude
    // more negative than this demo's own derated `eos_stiffness` -- real
    // water, properly scaled, essentially never cavitates from ordinary
    // splashing (a 2x volumetric expansion is nowhere near its real tensile
    // limit). The un-converted `-0.1` default instead clipped almost the
    // ENTIRE expansion range into a flat, zero-pressure-gradient dead zone
    // with no restoring force once J drifted past it -- confirmed directly:
    // this fix alone took the resting bulk from mean_J=1.5-1.9 pinned at the
    // 2.0 clamp (every depth band, every checkpoint) to mean_J=1.00-1.03
    // (healthy) at every depth band through a full 2000-frame run.
    const REAL_CAVITATION_PRESSURE_PA: f32 = -100_000.0;
    const WATER_RHO_SI_KG_M3: f32 = 1000.0;
    water.pressure_floor =
        config.stress_from_si_physical(REAL_CAVITATION_PRESSURE_PA, WATER_RHO_SI_KG_M3);
    // REAL FIX (2026-09-16) -- root cause of the splash disintegrating into
    // permanently scattered droplets, found bisecting directly against a
    // known-good historical build (`b8b13cc`, 2026-08-13) at the user's own
    // request. Confirmed via a real, adjacent-commit A/B (both patched with
    // the SAME pressure_floor+substep fixes above, isolating this variable
    // alone): `c32d86c` settles cleanly (mean bulk J=[0.92,1.06], 7 self-
    // correcting transient outliers over 900 frames); `f1fea23`, committed
    // 2 minutes later, shows persistent scattering (108 outlier events,
    // ext.y collapsing to 0.0). `f1fea23` itself is correct, well-tested
    // physics (exact rotation-invariant J-integration) -- restoring it is
    // not an option. Mechanism: the OLD, biased `det(I+dt*C)` integrator's
    // spurious `+dt^2*det(C)` expansion term, while wrong for pure rotation,
    // also happened to inflate J (hence soften/dampen the EOS pressure
    // response) specifically in the high-vorticity zones a violent splash
    // front produces -- accidentally providing extra "stickiness" nothing
    // else in the scene supplied. The exact formula correctly removes that
    // bias, exposing water's genuine, physical need for real surface
    // tension to hold a violent splash together, which this material has
    // never had (`surface_tension_coeff` defaults to 0.0, confirmed via
    // `grep -rn "surface_tension_coeff\s*="` finding zero nonzero uses
    // anywhere in this engine's history).
    //
    // Real, cited value, not a tuned constant: water's surface tension at
    // room temperature is gamma=0.0728 N/m (standard, widely-cited figure).
    // `NewtonianFluidMaterial::surface_tension_coeff` adds `gamma_grid*J` to
    // the Kirchhoff stress directly (see fluid.rs's own doc), i.e. it needs
    // PRESSURE units, not force-per-length. Surface tension physically
    // manifests as a pressure jump across a curved interface (Young-Laplace,
    // dp=gamma/R). This discretization cannot resolve any curvature radius
    // R smaller than one grid cell, so R=dx_meters is the natural, real
    // (if conservative/upper-bound) length scale for a bulk, non-curvature-
    // resolving approximation like this one -- the same "the grid IS the
    // resolution limit" reasoning `fluid_near_wall_cfl_scale` and the
    // pressure_floor fix above both already rely on. Converted through the
    // SAME `stress_from_si_physical` pipeline as everything else in this
    // file.
    // TESTED (2026-09-16) at R=dx_meters (the upper-bound length scale
    // derivation): surface_tension_coeff=72.8 caused J to hit the 2.0 clamp
    // by frame 23 and produced a particle at |v|=51.77 that is physically
    // impossible under this demo's own weak gravity (-2.943 cells/s^2 can
    // only produce ~6.8 cells/s after the elapsed sim time) -- a genuine,
    // real instability, the SAME failure signature the deleted grid-based
    // cohesion mechanism already showed. Reverted pending a smaller, still-
    // real derivation, or evidence this bulk-cohesion approach is simply
    // wrong for a macroscopic (cm-to-meter scale) splash, where real
    // surface tension is known to be negligible anyway (its natural length
    // scale is the capillary length, ~2.7mm for water -- millimeter, not
    // centimeter/meter, scale). See fluid_thin_layer_diag_gpu.rs or
    // HANDOFF_fluid_gpu_thin_layer_bug.md for the next real step: verify
    // via RenderMode::Surface (curvature-flow reconstruction) whether the
    // "disintegration" judged from raw point-cloud rendering is even a
    // real physics defect, before adding any more force terms.
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
    let registry = MaterialRegistry::with_default(Box::new(water));

    let mut sim = GpuSimulation::with_device(device, queue, config, particles, registry);
    // A body spawned at uniform density has no internal pressure and
    // collapses under its own weight before anything touches it -- a
    // resting pool that visibly twitches on its first frames. This puts it
    // in hydrostatic equilibrium so "at rest" means at rest. Called on
    // every spawn, which is every reset, so restarting really does restart
    // from a resting state.
    sim.settle_hydrostatic();
    // Real, already-proven mechanism (`fluids_gpu_isolated_droplet_settles_with_damping`,
    // 2026-07-30): an isolated splash particle has no neighbors to form a real
    // deformation/velocity gradient against, so none of this material's stress-based
    // dissipation (shear/bulk/shock viscosity) can ever act on it -- P2G/G2P still
    // smooths a lone particle's own velocity through its own local grid nodes
    // regardless of neighbors, so direct velocity relaxation is the one mechanism that
    // structurally reaches this case (see that memory's own account for why real air
    // drag was tried and rejected -- too weak at this scale/speed to matter). Previously
    // wired only into the Vortex pattern's own field list (needed there for its drain
    // dynamics), never applied to DamBreak/DropletImpact -- re-applied here for every
    // pattern at the same cited rate (0.1, within that investigation's documented
    // 0.05-0.2 water range).
    sim.add_force_field_gpu(GpuFieldEntry::linear_drag(Vec2::ZERO, 0.1, 1 << MAT_WATER));
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
/// extraction (2026-08-15): MIN_J_OUTLIER and OUTLIER below each printed
/// this same field set independently before this.
fn print_particle_trace(
    label: &str,
    idx: usize,
    p: &Particle,
    neighbors_tight: usize,
    neighbors_wide: usize,
) {
    let j = p.deformation_gradient.determinant();
    println!(
        "  {label} idx={idx} x=({:.3},{:.3}) v=({:.3},{:.3}) |v|={:.3} \
         F=[{:.3},{:.3};{:.3},{:.3}] J={j:.4} volume={:.6} initial_volume={:.6} \
         density={:.4} mass={:.6} neighbors(r=1.5)={neighbors_tight} \
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
        // Ported from `basic_fluids.rs` (2026-08-14): never called here, so
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
        let dx = sim.config().dx_meters;
        apply_optics(&mut renderer, sim.queue(), sim.registry(), dx);
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
        // basic_fluids.rs earlier this session, just never ported here:
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
        // Real GPU render-interpolation snapshot ("Fix Your Timestep", Gaffer
        // 2004 -- see `Renderer::snapshot_particle_positions`'s own doc). Same
        // fix already ported to `basic_sand_grid_gpu.rs`; this closes the gap
        // on the OTHER side of the exact demo family this feature was first
        // built for -- `basic_fluids.rs` (CPU) got the real fix back on
        // 2026-09-09, its GPU sibling never did. Taken ONCE per batch, before
        // any step in it runs, so `render_gpu`'s later blend is against the
        // pre-batch state, not a partially-advanced one. Skipped when
        // `steps==0`: the last real snapshot stays valid since nothing moved.
        if steps > 0 {
            self.renderer.snapshot_particle_positions(
                self.sim.device(),
                self.sim.queue(),
                self.sim.particle_buffer(),
                self.sim.particle_count(),
            );
        }
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
                        print_particle_trace("OUTLIER", idx, p, neighbors_tight, neighbors_wide);
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
                // Single-material scene since the mud removal (this file's
                // two-phase reconstruction call is gone with it) -- the
                // plain single-phase path (curvature_flow.wgsl's base
                // technique) is the correct one now, same as basic_fluids.rs's
                // own CPU counterpart.
                self.renderer.render_surface_reconstruction(
                    self.sim.device(),
                    self.sim.queue(),
                    SurfaceReconstructionSource {
                        particle_buf: self.sim.particle_buffer(),
                        particle_count: self.sim.particle_count(),
                        grid_res: GRID as u32,
                        material_slot: MAT_WATER,
                        material_mass_enabled: false,
                        dt: DT,
                    },
                    &view,
                    true,
                );
            }
            RenderMode::Particles => {
                // Real interpolation now active (see the pre-step snapshot
                // above) -- scoped to this render mode only, matching the CPU
                // `basic_fluids.rs` precedent: GridVolume/Surface build their
                // own independent P2G/reconstruction bridge buffers, a real,
                // disclosed, separate follow-up, not done here.
                self.renderer.render_gpu(
                    self.sim.device(),
                    self.sim.queue(),
                    GpuRenderParams {
                        particle_buf: self.sim.particle_buffer(),
                        particle_count: self.sim.particle_count(),
                        output_view: &view,
                        clear: true,
                        interp_alpha: self.stepper.interpolation_alpha(),
                    },
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
                    .with_title("emerge -- Fluids GPU [Water]")
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

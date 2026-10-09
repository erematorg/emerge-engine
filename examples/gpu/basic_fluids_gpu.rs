extern crate emerge_engine as emerge;

#[path = "../gui_common/declared_light.rs"]
mod declared_light;

/// GPU Newtonian water dam-break, zero CPU readback.
///
///   Mat 0  Newtonian water (blue) -- Tait EOS + deviatoric viscosity
///
///   cargo run --example basic_fluids_gpu --features "render"
use std::sync::Arc;

use emerge::diagnostics::log_frame_gpu;
use emerge::render::{
    ColorMode, GpuRenderParams, GridVolumeSource, Renderer, SurfaceReconstructionSource,
};
use emerge::{
    FixedStepConfig, FixedStepController, GpuFieldEntry, GpuSimulation, MaterialRegistry,
    NewtonianFluidMaterial, SimConfig, SpawnRegion, build_particles,
};
use glam::{IVec2, Vec2};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

const GRID: usize = 64;
// Playback speed, not a physics constant: the scene's look (gravity, viscosity) was
// tuned at 6 simulated seconds per real second (60 fps x dt 0.1). Rather than retune
// the physics or show an unfamiliar true 1x, this keeps that pace as an explicit
// fast-forward dial (`FixedStepConfig::simulation_speed`), like a video player's
// playback speed.
const PLAYBACK_SPEED: f32 = 6.0;
// Target render cadence. `DT` below is derived from this and `PLAYBACK_SPEED` so that
// about one `step_frame()` call happens per render frame at the target fps, whatever
// speed is dialed in. Using `1.0/60.0` directly as `DT` at PLAYBACK_SPEED=6 would need
// 360 `step_frame()` calls per second, 6+ MPM steps per render frame, each with fixed
// dispatch overhead (P2G/grid-update/G2P are separate submissions), which overloads
// the GPU.
const RENDER_FPS_TARGET: f32 = 60.0;
// Derived, see `RENDER_FPS_TARGET`. At PLAYBACK_SPEED=6.0 this is 0.1, the dt this
// demo's materials and gravity were tuned against.
const DT: f32 = PLAYBACK_SPEED / RENDER_FPS_TARGET;
const MAT_WATER: u32 = 0;
const LABELS: &[(u32, &str)] = &[(MAT_WATER, "water")];
// Module scope (not local to `make_sim_data`, which only builds the sim, not
// the renderer): `Renderer::set_grid_reference_cell_mass` needs this same
// real SI value from `State::new`, a separate `impl` method -- see that call
// site's comment.
const WATER_RHO_GRID: f32 = 0.1;

/// Installs the scene's real optical description. There is one path, not a
/// choice of looks: the material declares its own measured constants and the
/// scene declares its own physical scale and lighting. Nothing here selects
/// an appearance.
fn apply_optics(
    renderer: &mut Renderer,
    queue: &wgpu::Queue,
    registry: &emerge::MaterialRegistry,
    config: &SimConfig,
) {
    // Measured constants come from the materials themselves. Adding a
    // material with declared optics needs no change here at all.
    renderer.adopt_material_optics(queue, registry);
    renderer.set_physical_render_contract(queue, declared_light::overcast_contract(config));
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
    /// GridVolume's per-material accumulator is attached lazily on first switch
    /// into that mode, not at construction (attaching it eagerly slowed
    /// material_sandbox_gpu measurably).
    render_mode: RenderMode,
    /// Which spawn geometry is live -- switched with number keys, see
    /// `Pattern`'s doc. `reset()` re-spawns using this, not always
    /// `DamBreak`, so switching pattern and resetting are the same action.
    pattern: Pattern,
    /// Converts measured elapsed time into the number of physics steps per frame.
    /// Calling `sim.step_frame()` once per render frame assumes each frame takes
    /// exactly `DT` of real time, which varies with the render rate and makes the
    /// scene speed up and slow down. Same stepper as `snake_on_terrain_gpu.rs`.
    stepper: FixedStepController,
    last_instant: std::time::Instant,
    /// Diagnostic: highest `steps_for_frame` result since the last fps print. A
    /// catch-up burst (several simulation steps in one render call because the GPU
    /// fell behind real time) does not show in the 2-second fps average.
    max_steps_seen: usize,
}

/// The three rendering paths this demo can show, cycled with G: per-particle
/// instanced splat (`render_gpu`), the solver's coarse physics-grid density field
/// (`render_grid_volume`), and the finer, resolution-independent curvature-flow
/// surface reconstruction (`render_surface_reconstruction`, see that method's doc).
#[derive(Clone, Copy, PartialEq, Eq)]
enum RenderMode {
    Particles,
    GridVolume,
    Surface,
}

/// Which real fluid behaviour this scene proves, switched with the number
/// keys (1-4) -- same material/config/render setup throughout, only the
/// spawn geometry (and, for `Vortex`, the initial velocity field) changes.
/// 4 is a reserved slot, not built yet -- see its doc.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Pattern {
    /// Classic dam-break: a tall column collapses sideways under gravity and
    /// spreads across the floor. This demo's own original, still-default
    /// scene.
    DamBreak,
    /// A small blob falls into a shallow, wide pool -- crown splash and
    /// ripple propagation, a distinct behaviour from a collapsing
    /// column (Worthington-style droplet impact), proving the same solver
    /// handles a different initial geometry, not just a bigger
    /// dam-break.
    DropletImpact,
    /// A resting pool with a `GravityWellField` "drain" pulling fluid
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
        // Headroom, not a target: the patterns take tens of substeps per frame
        // (see `material_cfl_coefficient` and the EOS sizing in make_sim_data). A
        // cap the CFL scan runs into truncates the step; at 150, `sub` pinned at
        // exactly 150 and the truncation itself caused an unstable transient (`v`
        // past 179 units/s on frame 40).
        max_substeps_per_step: 1000,
        // Off: the affine-speed CFL contribution (`affine_cfl_speed_contribution`)
        // did not catch the growing velocity-gradient instability traced on
        // particle idx=1994. Frames 1-12 (C growing from exactly 0 to 9.487 during
        // free fall) were byte-identical with it on; the bound only tightens dt
        // once grad_norm is already large (dozens+), after the runaway has
        // started. No measured benefit for its CFL-scan cost.
        cfl_include_affine_speed: false,
        // 0.4, from two sources: Becker & Teschner 2007's WCSPH time step (eq. 18,
        // after Monaghan 1992) is `0.4*h/c_s`, and Bai & Schroeder 2022 ("Stability
        // analysis of explicit MPM", CGF 41) put the Von Neumann stability limit of
        // this scheme -- APIC with quadratic B-splines -- at `dt <= 1.0*dx/c` (their
        // Figure 4: analytic f = 1.0000, measured 1.0007 in 2D), so 0.4 keeps a 2.5x
        // margin on the proven bound, with boundary/isolated particles covered by
        // the separate Sun/Shinar/Schroeder 2020 bound the CFL scan applies. It needs
        // the GPU to pick its timestep per substep (`adaptive_cfl.wgsl`): with a
        // frame-frozen dt, 0.4 spikes the vortex's J to 1.28 where the CPU, same
        // scene and coefficient, stays at 1.025; with the adaptive timestep the GPU
        // measures 1.024. Costs ~52 substeps/frame instead of ~69. At 0.5 the
        // vortex's J goes to 1.43.
        material_cfl_coefficient: 0.4,
        // 0.3% of Earth gravity (g_grid~=981 via SimConfig::earth), the
        // `gravity_fraction` convention of `basic_sand.rs`/`basic_fluids.rs`. Much
        // weaker gravity (-0.3) leaves too little driving force to overcome the
        // EOS's elastic-like response: the free surface never flattens and keeps a
        // wavy standing pattern. Headless A/B on surface-height variance across x:
        // at -0.3 it settles to ~1.0-3.0 after 1000 steps (more drag makes it worse,
        // 24+ at 0.5 drag); at 0.3% of g it settles to ~0.01-0.04, ~100x flatter.
        gravity: Vec2::new(0.0, -981.0 * 0.003),
        // No `fluid_near_wall_cfl_scale` (engine default 1.0). A 20x smaller
        // near-wall timestep is not needed: the wall-contact blow-ups (J up
        // to 34653) it would guard against were two GPU bugs, fixed-point P2G
        // atomics dropping small momentum contributions and the driver
        // misreading `C[1][1]` in the J update (see `trace2` in
        // `particles_update.wgsl`). Without it, the dam, drop and vortex
        // patterns match their CPU twin; with it (20.0), the substep count
        // rises ~20x once water touches a wall (~400 vs ~21 per frame) and
        // the vortex comes out damped (peak speed ~9 vs the CPU reference's
        // ~22).
        // No regional substepping: its precondition, a calm region next to a
        // violent one, does not hold on this scene (see KNOWN_LIMITATIONS.md,
        // entry 2).
        // What this 2-D slice stands for out of plane. The domain is 64
        // cells of 1 cm, so the tank it represents is about this deep. It is
        // a statement about the scene, not a dial for how blue the water
        // should look -- water this shallow is nearly colourless, and that
        // is what a 30 cm tank looks like.
        slice_thickness_m: Some(0.3),
        ..SimConfig::earth(GRID, 0.01, DT)
    };
    // Mass set explicitly: a uniform material-point lattice represents a filled
    // region when `m = rho0 * spacing²`, so ΣV0 approximates the geometric area.
    // This calibrates quadrature mass and reference volume; the WC-MPM EOS itself
    // uses rho=rho0/J, never a kernel-density overwrite. rest_density=0.1 for
    // water: `rho_grid = rho_kg_m3*dx_meters^2 = 1000*0.01^2 = 0.1` at this
    // scene's scale (see basic_fluids.rs). Spacing 0.5 (4 particles per cell, as
    // basic_fluids.rs): denser sampling leaves the velocity-divergence estimate
    // less room to spike (sub=19 steady against a climb into the hundreds at ~1.2
    // particles per cell).
    const SPACING: f32 = 0.5;
    // WATER_RHO_GRID itself is module-scope now (renderer setup in `State::new`
    // needs the same real value) -- see its doc.
    const WATER_MASS: f32 = WATER_RHO_GRID * SPACING * SPACING;
    let water_region = |box_size: IVec2, box_center: Vec2| SpawnRegion {
        spacing: SPACING,
        box_size,
        box_center,
        material_id: MAT_WATER,
        mass_override: Some(WATER_MASS),
        ..SpawnRegion::for_sim(&config)
    };
    // Non-empty only for Vortex -- registered on `sim` after construction
    // below, see that call site's comment.
    let mut vortex_fields: Vec<GpuFieldEntry> = Vec::new();
    let particles = match pattern {
        Pattern::DamBreak => {
            // x=20: the column's left edge (x=13) starts clear of the wall zone
            // (`boundary_thickness`=2), so it meets the wall only through real
            // contact.
            build_particles(
                &config,
                water_region(IVec2::new(14, 52), Vec2::new(20.0, 30.0)),
            )
        }
        Pattern::DropletImpact => {
            // A shallow, wide pool near the floor plus a small blob well
            // above it -- same water material/mass, two spawn regions
            // instead of one. ~28 cell fall distance (blob bottom ~38 to
            // pool top ~15) for a visible splash.
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
            // A contained whirlpool in the middle of a flat, resting pool -- the
            // wide-rectangle-on-the-floor geometry of DropletImpact, not a disk
            // (which reads as an isolated blob, not a sea with a vortex in it).
            //
            // Not a drain: no mass leaves (a sink drains the pool into a scattered
            // mess within a minute). None is needed: by Kelvin's circulation theorem
            // (Thomson 1869: inviscid, barotropic flow under conservative forces
            // conserves circulation around a material loop) an ocean whirlpool
            // (Corryvreckan, Moskstraumen) persists without emptying the sea. Any
            // visible core dip comes from cyclostrophic balance (dp/dr = rho*v^2/r,
            // as in tornado and hurricane cores), from rotation alone.
            //
            // Mechanism: a `GravityWellField` pull (`GpuFieldEntry::gravity_well`,
            // the Plummer-softened point-mass field basic_orbital.rs checks against
            // Kepler's third law) plus a small constant-angular-momentum ("free
            // vortex") seed, v = L/r floored at r=DRAIN_SOFTENING. Solid-body rotation
            // (v = omega*r) gives near-zero angular momentum near the center
            // (L = r*(omega*r) -> 0), so those particles fall in radially and get
            // flung out on close approach (an N-body slingshot); the free vortex gives
            // every particle angular momentum from the start. Only an initial
            // condition (vortices start from some pre-existing circulation, Shapiro
            // 1962); MPM's APIC transfer (Jiang et al. 2015) then conserves it.
            //
            // No RadialConfinement: see below.
            //
            // The GPU gravity-well path in grid_update.wgsl is exercised here
            // (tests/gpu.rs covers linear_drag/spatial_drag/radial_confinement, not
            // gravity_well).
            //
            // Limit: this GPU strict-fluid path is a weakly compressible Tait EOS,
            // not incompressible pressure projection, so cyclostrophic balance is
            // only approximated. The CPU DCT/Gauss-Seidel solver
            // (`fluid_pressure_projection.rs`, 16-30 fps depending on
            // `GS_CORRECTION_SWEEPS`, see `grid/pressure.rs`) is the accurate path but
            // has its own unresolved wall-free-pool instability.
            //
            // POOL_BOX/DRAIN_EDGE_R/DRAIN_SOFTENING are the values that ran stable
            // (1400+ frames in one configuration, 650+ in this one). The vortex needs
            // a ~44-cell working diameter; GRID=64 leaves little margin for a visibly
            // larger calm sea around it, which a bigger GRID would give.
            //
            // Perf: this pattern measured 10-15 fps against 28-38 for the other two
            // (before the GPU picked its timestep per substep). Weakening
            // DRAIN_EDGE_ACCEL_FRACTION/SEED_EDGE_SPEED together barely moved fps and
            // broke the centripetal balance (the two are a matched pair), and
            // narrowing POOL_BOX (54->44) did not help either (the pool's spread still
            // reached both walls). A sustained rotating core keeps some particle fast
            // for the pattern's whole life, unlike a settling puddle, so the substep
            // count stays high throughout.
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
            // No RadialConfinement: `radial_confinement` pushes anything beyond its
            // radius inward, resting pool water included (by design most of this
            // wide pool lies beyond any basin radius), which collides with the
            // undisturbed pool (|v| up to 100+ within the first ~20 frames). Like the
            // other patterns, this relies on gravity, the floor and the domain's
            // slip boundary alone.
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
    // Water: weakly compressible Tait EOS.
    //
    // Exponent 3, not water's 7 (Cole 1948): the acoustic CFL term
    // `c2 = eos_stiffness*eos_power*ratio^(eos_power-1)/rest_density` grows as
    // ratio^6 at 7, so a strong wall compression turns into a timestep cliff
    // (measured at J ~0.35-0.4: max c2 up to 6605, acoustic dt down to 6e-5, while
    // the deformation and gravity bounds stayed 1000x+ larger). Lower Tait
    // exponents (n=1..4) are a known real-time WCSPH trade-off for this reason
    // (Chorin's artificial compressibility uses n=1; Monaghan notes n=7 is
    // accurate but numerically stiff).
    //
    // Stiffness from the weakly compressible rule, `c_ref = 10*v_max` (Mach < 0.1;
    // Monaghan 1994, Morris et al. 1997), `B = rho*c_ref^2/gamma`, with v_max =
    // sqrt(2*g*h) for this scene's column and gravity (~17.5 grid/s, against a
    // measured peak of ~16). A soft EOS is not cheaper: at c ~5.5 grid/s against
    // flows of 28 (Mach ~5) the "liquid" behaves like a gas, J swings over
    // [0.145, 24.3] and the CFL needs ~784 substeps per frame. A stiff EOS holds
    // J ~= 1, so `ratio^(gamma-1)` stays near 1 and c2 at its baseline. At c ~3.1x
    // under the rule (Mach ~0.3) the pool "breathes" (mean J 0.988 <-> 1.000 with
    // a ~0.7 s period, the acoustic round trip 4*depth/c = 40/55.8), bouncing on
    // its own compressibility like a jelly. At the full rule, all three patterns
    // stay coherent (0 isolated particles over 150 frames), J stays within +-6%,
    // and mean J holds at 0.999 with no oscillation. Cost: ~60 substeps/frame
    // instead of ~21.
    const COLUMN_HEIGHT_CELLS: f32 = 52.0;
    let v_max_grid = (2.0 * config.gravity.length() * COLUMN_HEIGHT_CELLS).sqrt();
    let c_ref_m_s = 10.0 * v_max_grid * config.dx_meters;
    // SECOND real bug in the previous version: `NewtonianFluidMaterial::
    // weakly_compressible` hard-codes Cole 1948's gamma=7 internally
    // (`fluid.rs`: `const GAMMA: f32 = 7.0`) -- the EXACT exponent this
    // file's comment history (above) already measured as catastrophic on
    // this scene (c2 up to 6605 from `ratio^6` amplifying a modest J
    // excursion), which is why eos_power=3.0 was deliberately chosen over 7.0
    // in the first place. Calling `weakly_compressible` silently reintroduced
    // gamma=7. Fixed by inlining that helper's own real formula
    // (`tait_b_pa = rho_kg_m3 * c_ref_m_s^2 / gamma`, `fluid.rs:78`) with
    // this scene's already-justified gamma=3.0 instead.
    const WATER_EOS_POWER: f32 = 3.0;
    let water_tait_b_pa = 1000.0 * c_ref_m_s * c_ref_m_s / WATER_EOS_POWER;
    // Viscosity converted to grid units (`SimConfig::visc_from_si`,
    // `eta_SI/(rho*dx^2)`), the family `pressure_floor` below uses
    // (`stress_from_si`): `fluid.rs`'s stress law (`stress += eff_viscosity *
    // strain_dev`) needs grid units, and mixing raw and density-normalized
    // conventions in one stress tensor is wrong (see
    // `q_factor_elastic_viscosity_pa_s` for a ~917x instance of that mistake). The
    // raw 1.0e-3 would be ~10x too weak here (grid value 0.01). Molecular viscosity
    // does not explain or damp the splash's C-matrix growth, even at 10x.
    const WATER_DYNAMIC_VISCOSITY_PA_S: f32 = 1.0e-3;
    const WATER_RHO_SI_KG_M3_FOR_VISC: f32 = 1000.0;
    let water_dynamic_viscosity =
        config.visc_from_si(WATER_DYNAMIC_VISCOSITY_PA_S, WATER_RHO_SI_KG_M3_FOR_VISC);
    let mut water = NewtonianFluidMaterial::new(
        WATER_RHO_GRID,
        water_dynamic_viscosity,
        water_tait_b_pa,
        WATER_EOS_POWER,
    );
    // Bulk (second) viscosity: `NewtonianFluidMaterial::new` sets
    // `bulk_viscosity: 0.0`, leaving the stress tensor (`fluid.rs`'s
    // `stress += 0.5*bulk_viscosity*div(v)*I`) with no dissipation for volumetric
    // oscillation. `artificial_bulk_viscosity` (von Neumann-Richtmyer) acts only
    // under compression, correctly, since it captures shocks; bulk viscosity is the
    // symmetric dissipative term that damps acoustic ringing after an impact
    // (Denner et al. 2023, "acoustic damper term in weakly-compressible SPH"). The
    // tall column's impact shows non-diverging but undamped J oscillation without
    // it. Water's bulk viscosity is ~2.8-3.0x its shear viscosity (Litovitz &
    // Davis; arxiv.org/pdf/1002.3029's acoustic-spectroscopy remeasurement, ratio
    // ~3 across 7-50C), applied to the grid-converted `water_dynamic_viscosity`.
    water.bulk_viscosity = 3.0 * water_dynamic_viscosity;
    // This fluid IS water, so it declares water's own measured optical
    // constants. The renderer adopts them; no colour is chosen anywhere.
    water.optics = Some(emerge::materials::optical::pure_water());
    // Liquid water at 25 C, CRC Handbook. Lets frictional dissipation become
    // a real temperature rise rather than only being accounted for.
    water.specific_heat_j_kg_k = 4182.0;
    // Cavitation floor converted to grid units. The constructor's `pressure_floor`
    // default (-0.1) is a bare grid-unit constant, never SI-converted, unlike
    // `water_tait_b_pa` above. Water's practical cavitation onset (dissolved-gas
    // nucleation, the standard engineering figure) is ~-0.1 MPa = -100,000 Pa
    // gauge; through `stress_from_si` it lands far below the EOS scale, so water
    // essentially never cavitates from ordinary splashing (a 2x expansion is
    // nowhere near its tensile limit). The unconverted -0.1 clips almost the whole
    // expansion range into a zero-pressure-gradient dead zone with no restoring
    // force: the resting bulk sits at mean_J=1.5-1.9, pinned at the 2.0 clamp in
    // every depth band, against mean_J=1.00-1.03 with the converted floor over a
    // 2000-frame run.
    const REAL_CAVITATION_PRESSURE_PA: f32 = -100_000.0;
    const WATER_RHO_SI_KG_M3: f32 = 1000.0;
    water.pressure_floor = config.stress_from_si(REAL_CAVITATION_PRESSURE_PA, WATER_RHO_SI_KG_M3);
    // Surface tension: none (`surface_tension_coeff` stays 0.0). The exact,
    // rotation-invariant J integration removes the biased `det(I+dt*C)` update's
    // spurious `+dt^2*det(C)` expansion, which had softened the EOS in the
    // high-vorticity zones of a splash front and so held it together; without it a
    // violent splash scatters into droplets (108 outlier events and ext.y collapsing
    // to 0.0 over 900 frames, against 7 self-correcting outliers with the biased
    // update).
    //
    // Water's surface tension is gamma=0.0728 N/m at room temperature.
    // `surface_tension_coeff` adds `gamma_grid*J` to the Kirchhoff stress (see
    // fluid.rs), so it needs pressure units: Young-Laplace, dp=gamma/R, with R=dx
    // (the grid cannot resolve a smaller curvature radius), through
    // `stress_from_si`. That upper bound, 72.8, drives J to the 2.0 clamp by frame
    // 23 and a particle to |v|=51.77, impossible under this gravity (-2.943
    // cells/s^2 gives at most ~6.8 cells/s over the elapsed time) -- the failure
    // signature of the grid-based cohesion below. At cm-to-meter scale surface
    // tension is negligible anyway (its length scale, the capillary length, is
    // ~2.7 mm for water). Before adding force terms, check with
    // RenderMode::Surface (curvature-flow reconstruction) whether the
    // "disintegration" judged from raw point-cloud rendering is a physics
    // defect at all.
    let registry = MaterialRegistry::with_default(Box::new(water));

    let mut sim = GpuSimulation::with_device(device, queue, config, particles, registry);
    // A body spawned at uniform density has no internal pressure and
    // collapses under its own weight before anything touches it -- a
    // resting pool that visibly twitches on its first frames. This puts it
    // in hydrostatic equilibrium so "at rest" means at rest. Called on
    // every spawn, which is every reset, so restarting really does restart
    // from a resting state.
    sim.settle_hydrostatic();
    // Linear drag on water (rate 0.1, within the 0.05-0.2 water range of
    // `fluids_gpu_isolated_droplet_settles_with_damping`): an isolated splash
    // particle has no neighbors to form a deformation or velocity gradient against,
    // so no stress-based dissipation (shear/bulk/shock viscosity) acts on it, while
    // P2G/G2P still smooths its velocity through its own grid nodes. Direct velocity
    // relaxation is the mechanism that reaches it (air drag is too weak at this
    // scale and speed to matter).
    sim.add_force_field_gpu(GpuFieldEntry::linear_drag(Vec2::ZERO, 0.1, 1 << MAT_WATER));
    // Vortex's real drain + basin -- see that match arm's doc for the
    // full physical account. Registered once here (persists every substep
    // until cleared, see `add_force_field_gpu`'s doc) since `sim` only
    // exists from this point on.
    for field in vortex_fields {
        sim.add_force_field_gpu(field);
    }
    // Grid-mediated cohesion (CSF, see `grid_update.wgsl`'s
    // `grid_cohesion_main_inner`), disabled. It applies its force to grid momentum
    // in its own pass, bypassing `fluid_state::force_stress_volume`'s cap (which
    // bounds the ordinary stress->force P2G scatter), and on this scene it drives J
    // to 25 by frame 3. Surface tension is wanted physics, but its grid-space force
    // needs its own stability treatment (likely an untrustworthy curvature estimate
    // in under-resolved regions) before it can be on by default.
    // sim.set_cohesion_si(0.0728, 1000.0);

    // No scene-wide settling drag is applied.  Momentum changes only through
    // the WC-MPM stress, prescribed gravity, and geometric wall conditions.

    sim
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
        // As in `basic_fluids.rs`: GridVolume/Surface threshold density against
        // this scene's full-cell water mass (WATER_RHO_GRID=0.1), not an absolute
        // mass floor, which would discard almost everything.
        renderer.set_grid_reference_cell_mass(WATER_RHO_GRID);
        // Same port: default 6x oversamples the surface grid for this
        // particle count, so each cell sees too few particles and reads
        // density noise as speckle (live-reported: bumpy, mottled Surface
        // mode). 4x gives each cell a sample, at 0.44x the cost.
        renderer.set_surface_res_multiplier(4);
        apply_optics(&mut renderer, sim.queue(), sim.registry(), sim.config());
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
            // Vortex is not on a keypress (10-15 fps against 28-38 for patterns 1
            // and 2, see its match arm), but stays constructible through the
            // `EMERGE_VORTEX_DEBUG` environment variable for development.
            pattern: if std::env::var("EMERGE_VORTEX_DEBUG").is_ok() {
                Pattern::Vortex
            } else {
                Pattern::DamBreak
            },
            // Not `::standard()`: its 64-step-per-render catch-up cap is sized for
            // cheap physics steps. With up to `max_substeps_per_step` substeps per
            // step, a slow step_frame() falls behind, the accumulator asks for more
            // catch-up steps, each also slow with its own blocking GPU sync: a
            // scheduling death spiral (fps 4->2->1->0 while GPU usage pins). A cap of
            // 1 makes a slow frame slow motion, never a compounding spiral.
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
        // aspect-ratio letterboxing). Plain width/height scaling matches the grid
        // only in a square window, so push/pull lands at the wrong point at any
        // other aspect ratio.
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
        // Elapsed time since the last reset (possibly seconds, if the window was
        // idle) must not be replayed as a burst of catch-up steps.
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
        // Render-interpolation snapshot ("Fix Your Timestep", Gaffer 2004, see
        // `Renderer::snapshot_particle_positions`), as in `basic_sand_grid_gpu.rs`
        // and `basic_fluids.rs`. Taken once per batch, before any step in it, so
        // `render_gpu`'s blend is against the pre-batch state. Skipped when
        // `steps==0`: the last snapshot stays valid since nothing moved.
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
            // Gated per simulation step, not per render call: `steps` can be 0
            // for several consecutive render frames right after startup (the
            // accumulator has not crossed `DT` of real time yet), which would
            // re-print the same frame. Fires once every 60 simulated frames.
            if self.frame.is_multiple_of(60) {
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
                        grid_res: self.sim.config().grid_res as u32,
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
                        dt: self.sim.mean_substep_dt(),
                    },
                    &view,
                    true,
                );
            }
            RenderMode::Particles => {
                // Interpolation for this render mode only, as in `basic_fluids.rs`:
                // GridVolume/Surface build their own P2G/reconstruction buffers and
                // do not interpolate yet.
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

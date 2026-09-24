extern crate emerge_engine as emerge;

use egui_wgpu::ScreenDescriptor;
use emerge::Particle;
/// `basic_fluids.rs` (Newtonian water dam-break + Bingham mud blob) with a real,
/// live egui panel -- same pattern as `basic_sand_gui.rs`/`basic_snow_gui.rs`:
/// real gravity slider (1.0 = genuine IRL 9.81 m/s²), push/pull, and directional-
/// drag digging (the SAME proven mechanism from `basic_sand_gui.rs`: a per-particle
/// velocity nudge along the cursor's own movement, no second body, no contact_group
/// tuning -- mass-conserving by construction). Materials and the dam-break setup
/// are unchanged from `basic_fluids.rs`.
///
/// Real phase range: water below 273K freezes into ice
/// (`NeoHookeanMaterial` wrapped in `WithLatentHeat(-334_000.0)`, same real
/// exothermic value `latent_heat.rs` already uses, via
/// `Simulation::thermal_config_mut()` -- a real, already-existing engine
/// hook, not new plumbing). Exposed as a discrete Warm/Cold toggle, not a
/// continuous slider -- a continuously-tunable gravity/temperature slider
/// pair turns into a hunt-for-the-right-value loop; a two-state toggle proves
/// the same real phase transition without that.
///
/// This explicit WC-MPM branch has no hidden Jacobian floor: an inadmissible
/// state is reported rather than replaced by a capped deformation. Use the
/// material's sound speed, viscosity, and CFL limit to choose a physically
/// resolved scene rather than treating the gravity slider as a stabilization
/// parameter.
///
///   cargo run --example basic_fluids_gui --features render
use emerge::render::{ColorMode, GridVolumeSource, Renderer, SurfaceReconstructionSource};
use emerge::thermodynamics::{ThermalConfig, ThermalDiffusion};
use emerge::{
    BinghamFluidMaterial, FixedStepConfig, FixedStepController, GravityWellField,
    NeoHookeanMaterial, NewtonianFluidMaterial, SimConfig, Simulation, SlipBoundary, SpawnRegion,
    WithLatentHeat,
};
use glam::{IVec2, Vec2};
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

/// The three real rendering paths this demo can show, cycled with G -- same
/// modes/order as `basic_fluids_gpu.rs`, ported here for the CPU `Simulation`
/// path (2026-08-09). That demo runs on GPU-resident buffers
/// (`GpuSimulation::grid_buffer()`/`particle_buffer()`) already in the right
/// layout for `render_grid_volume`/`render_surface_reconstruction_dual_phase`;
/// this one runs the CPU solver, which has no such persistent GPU buffer, so
/// `grid_bridge_buf`/`material_mass_bridge_buf`/`particle_bridge_buf` below
/// rebuild and upload a snapshot each frame -- same real, disclosed bridging
/// cost/approximation `fire_spread.rs` already established for its own
/// `GridVolume` mode (see `upload_grid_volume_bridge`'s own doc), extended
/// here with a NEW particle-buffer bridge for `Surface` mode (first CPU demo
/// to drive the curvature-flow dual-phase path -- `Particle` is already
/// `repr(C)`/`Pod`/GPU-uploadable by design, so this is a direct
/// `bytemuck::cast_slice` upload of `sim.particles().iter().collect()`, no
/// new layout work).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum RenderMode {
    Particles,
    GridVolume,
    Surface,
}

// Foam/spray (Ihmsen-simplified trapped-air potential + Spray/Foam
// secondary particles) was built and shipped 2026-08-10, then REVERTED same
// day on the user's own direct instruction: real, measured perf cost (see
// [[basic_fluids_gui_foam_spray_shipped_2026-08-10]] for the full postmortem
// -- visual tuning was never confirmed and the user judged it not worth
// carrying while the CORE render/perf/physics work below is still unsettled.
// Deliberately deferred, not abandoned -- pick it back up from that memory
// entry once Surface mode and interaction-fps stability are solid.

const GRID: usize = 64;
const DT: f32 = 0.1;
// Target render fps used ONLY to pick the `FixedStepController`'s
// `simulation_speed` (see `State::new`'s stepper doc) so physics steps land
// ~1-per-render-frame near this rate instead of true real-time (which would
// under-step relative to this demo's ~45-60fps render rate at `DT=0.1` and
// look choppy). Same value + same role as `basic_fluids_gpu.rs`'s own
// `RENDER_FPS_TARGET` -- not an independently re-guessed number.
const RENDER_FPS_TARGET: f32 = 60.0;
// Real, measured 45fps-debug-minimum fix (2026-08-09) -- see `make_sim`'s own
// doc for the full derivation. Promoted to a top-level const (was local to
// `make_sim`) so `State::new`/`resize`'s own `set_camera` calls can size
// `particle_scale` to match -- real, disclosed bug found 2026-08-10: leaving
// `particle_scale` at the OLD spacing's value (0.6) while particles now sit
// 0.9 grid-units apart left visible gaps between them, reading as
// "filtered"/barely-visible fluid, not the actual physics being wrong.
// 0.5, not the old 0.9 (2026-08-13). This is the particle-per-cell (PPC)
// sampling rate, a real MPM discretisation parameter, not a cosmetic one:
// with grid_cell_size = 1.0, spacing s gives PPC = (1/s)^2, so 0.9 gave just
// **1.23 PPC** where standard 2D MPM uses **4** (particles seeded at dx/2 --
// the convention in Hu et al.'s MLS-MPM and every reference implementation in
// tmp/). At 1.23 PPC each grid node is supported by barely one particle, which
// under-resolves the transfer, makes the free surface ragged, and starves both
// render paths (grid-volume's density field and the curvature-flow surface
// reconstruction) of the data they need -- reported live as every render mode
// looking bad, not just the raw-particle one.
//
// 0.5 restores exactly 4 PPC. `box_size` is the region's extent in CELLS, so
// it stays untouched -- the columns keep their exact physical dimensions and
// only the sampling density inside them changes. Particle mass is already
// derived as rho0 * SPACING^2, so it follows automatically. Real cost: ~3.2x
// more particles (water ~928 -> ~2900).
const SPACING: f32 = 0.5;
/// Rendered diameter of one particle, for `RenderMode::Particles`.
///
/// The quad spans `local_pos` in [-0.5, 0.5], so the drawn disc's DIAMETER is
/// exactly the `particle_scale` passed to `set_camera`. Passing `SPACING`
/// (what this demo did until 2026-08-13) makes each disc exactly as wide as
/// the particle pitch -- and circles of diameter = pitch on a square lattice
/// cover only pi/4 = 78.5% of the area, leaving 21.5% of the fluid as visible
/// dark gaps at the diagonals. Live-reported as the particle view looking
/// speckled/scattered rather than like a liquid.
///
/// A material point represents an area of `SPACING^2`, so the disc carrying
/// exactly that area has `pi*r^2 = SPACING^2`, i.e. diameter
/// `2/sqrt(pi) * SPACING ~= 1.128 * SPACING`. That is this constant: each
/// particle draws precisely the fluid area it actually stands for -- no
/// arbitrary fudge factor, and it stays correct automatically if SPACING
/// changes.
const PARTICLE_RENDER_DIAMETER: f32 = SPACING * std::f32::consts::FRAC_2_SQRT_PI;
const MAT_WATER: u32 = 0;
const MAT_MUD: u32 = 1;
const MAT_ICE: u32 = 2;
const FREEZING_POINT: f32 = 273.0;
const ICE_LATENT_HEAT: f32 = -334_000.0; // exothermic: freezing releases energy (real water: 334 kJ/kg)
const WARM_AMBIENT: f32 = 300.0;
const COLD_AMBIENT: f32 = 250.0;
// 1/s, Newton cooling rate for the "Cold ambient" toggle -- models convective
// heat loss to surrounding cold air (a freezer), not bulk Fourier conduction
// through the water's own interior (too slow to visibly freeze this scene's
// real ~0.6m water column within any reasonable play session). Chosen
// empirically for a reasonable interactive wait (ice appears within ~2
// minutes).
const FREEZER_COOLING_RATE: f32 = 0.08;
// Radius of the directional dig nudge, grid cells -- matches basic_sand_gui.rs.
const DIG_RADIUS: f32 = 4.0;

fn make_sim() -> Simulation {
    let config = SimConfig {
        min_dt: 1.0e-4,
        // 12, NOT the real CFL-satisfying ~21 -- a DELIBERATE, DISCLOSED,
        // TEMPORARY dev-time trade (see MEMORY.md [[feedback_incremental_
        // substep_cap_perf_methodology_2026-08-09]]), NOT a repeat of the
        // real bug this exact field once had (see [[fluid_pressure_solve_
        // perf_profiled_and_component_split_ruled_out_2026-08-09]] Round 5:
        // cap=8 two weeks ago silently dropped 61% of requested simulation
        // time every frame while still reporting a flat, comfortable fps --
        // that was NEVER disclosed or tracked, this is). RE-MEASURED
        // 2026-08-10 (previous sweep numbers here were stale, measured under
        // a since-fixed time-dilation bug -- see [[basic_fluids_gui_
        // realtime_stepping_fixed_2026-08-10]]): on a genuinely quiet
        // machine, real-time-corrected stepping, cap=12 -> stable 46-59fps
        // clean 20s (zero spikes); cap=16 -> 38-47fps, real dips below the
        // 45fps floor during warm-up -- REJECTED per the methodology's own
        // rule (raise only kept if it stays >= 45fps). THE PLAN, not
        // optional: every time a real per-substep cost reduction lands
        // (P2G/G2P, render pipeline, etc.), raise this cap by a real
        // increment and re-verify live fps stays >= 45 before keeping the
        // raise -- see that methodology memory entry for the full rule.
        // This demo's gravity_fraction=0.003 is the same
        // ~10x-stronger-than-basic_fluids.rs regime as basic_fluids_gpu.rs,
        // so it needs that file's cfl=0.1, not basic_fluids.rs's unchanged
        // default -- that reasoning still applies to `material_cfl_
        // coefficient` below, unaffected by this cap.
        // TEMPORARY, explicitly disclosed (2026-08-13): was 12, tuned
        // against a scene where mud's EOS pressure was silently dead (a
        // real bug in `BinghamFluidMaterial::update_particle`, just fixed --
        // see that method's own doc). With mud's pressure now genuinely
        // alive, 12 panics ("could not advance... within
        // max_substeps_per_step=12"). Raised to match `basic_fluids_gpu.rs`'s
        // own value as a real, precedented starting point -- the entire
        // 45fps-floor tuning ladder documented below this field needs a
        // fresh re-pass now that mud's physics genuinely changed; not
        // re-done here, flagged as real follow-up work.
        max_substeps_per_step: 150,
        // `spatial_sort_enabled` real-measured 2026-08-10, NOT enabled here:
        // tried at this demo's ~1288 particles (46-59fps -> 22-27fps, a real
        // regression) and re-tried after fixing an initial implementation
        // mistake (was recomputing the sort every substep instead of once
        // per outer step) -- still measured WORSE even at 67,600 particles
        // in a dedicated headless benchmark (52.9ms/step unsorted vs
        // 79.0ms/step sorted, +49%). Real, honest negative result on this
        // engine's actual dev target (debug builds only, see CLAUDE.md) --
        // the O(N log N) sort's real cost in an unoptimized build outweighs
        // the P2G cache-locality win the mechanism itself is real about.
        // Feature kept (opt-in, default `false`, fully tested/correct --
        // see `spatial_sort_order`/`scatter_particles_to_grid_sorted`'s own
        // tests) in case a release build or a very different access pattern
        // ever makes it worthwhile -- just not proven beneficial today.
        // 0.3, not the old 0.1 (2026-08-13). This is the CFL *number* C in the
        // standard explicit acoustic condition `dt <= C * dx / c_sound`, where
        // C < 1 is the stability limit and real solvers run C = 0.2-0.4 for
        // margin (Monaghan 1992/1994 uses 0.25-0.3 for SPH; MLS-MPM commonly
        // 0.3-0.5). 0.1 was 3x more conservative than any of them.
        //
        // Why it was 0.1: the note below records `cfl=0.5 panics on frame 1`.
        // That was measured against the ~100x-too-soft `eos_stiffness = 1.0`
        // (see the water material's own derivation comment) -- with an EOS
        // that soft, the column genuinely collapsed into the J-clamp floor
        // every run, and no CFL number could have saved it. With the stiffness
        // now derived correctly, that failure mode is gone at the source, so
        // the conservative override it forced is no longer justified.
        //
        // Real, measured effect of the stiffness fix alone: substeps/frame
        // 11 -> 54 (correct water is genuinely ~5x more work -- higher sound
        // speed is the whole point of a stiffer EOS), fps ~50 -> ~13. Moving C
        // 0.1 -> 0.3 recovers ~3x of that from the safety margin rather than
        // from the physics.
        // This demo's only phase rule is the water->ice freeze predicate, a
        // thermodynamic test -- and temperature now advances once per step
        // (diffusion runs at its own stable rate), so it cannot change within
        // a frame. Opting in skips ~17 redundant O(N) scans per frame; see
        // SimConfig::phase_rules_once_per_step for why this is a caller's
        // choice rather than a silent default.
        phase_rules_once_per_step: true,
        material_cfl_coefficient: 0.3,
        cfl_include_affine_speed: false,
        // `fluid_near_wall_cfl_scale` (real, proven fix for the wall-contact
        // momentum bug, see project memory) tried here at 1000x and REVERTED,
        // 2026-08-08: this demo's water starts only ~2 cells from a wall, so a
        // large, sustained fraction of the domain reads as "near wall" the
        // whole time, not just during brief contact events -- 1000x turned
        // that into a real, live-confirmed freeze/severe-lag, not a one-off
        // slow frame. The fix is correct but not yet practical for a scene
        // shaped like this one; left at the engine default (1.0, off) here
        // until a cheaper version (narrower spatial trigger, or local-only
        // application) exists.
        ..SimConfig::earth(GRID, 0.01, DT)
    };
    // eos_power=3.0, NOT the real Cole 1948 water exponent (7.0) -- real,
    // disclosed compressibility-accuracy trade, found live 2026-08-09 on this
    // demo's GPU twin (basic_fluids_gpu.rs, see its own doc for the full
    // per-substep CFL-term breakdown that found it): under real violent wall
    // impact, J drops to ~0.3-0.4 (genuine local compression, not a bug), and
    // `c2 = eos_stiffness*eos_power*ratio^(eos_power-1)/rest_density` explodes
    // as ratio^6 at power=7 -- confirmed the dt-limiting term by two orders of
    // magnitude over the deformation-gradient and gravity terms. Lower Tait
    // exponents (n=1..4) are an established real-time-graphics WCSPH trade for
    // exactly this reason (Chorin's artificial-compressibility method uses
    // n=1). eos_stiffness=1.0, not the SI-correct 2.5, as a modest additional
    // margin -- the exponent is the dominant lever, not the base stiffness
    // (tried stiffness alone at 0.25 first, barely moved the substep count).
    // REAL, DERIVED EOS stiffness (2026-08-13) -- replaces a hardcoded
    // `eos_stiffness = 1.0` that was ~100x too soft and was the actual root
    // cause of the "particles get crushed" symptom, calculated:
    //
    //   Hydrostatic load at this column's base:
    //     p = rho * g * h = 1000 kg/m^3 * (9.81 * 0.003) m/s^2 * 0.468 m
    //       = 13.77 Pa                     (h = box_size.y=52 * SPACING=0.9 cells * dx=0.01 m)
    //   Tait EOS solved for the equilibrium compression it implies:
    //     p = B((rho/rho0)^gamma - 1),  gamma = 3
    //     B = 1.0  ->  r^3 = 14.77 -> r = 2.45 -> J = 0.408   <-- CRUSHED
    //     B = 104  ->  r^3 = 1.132 -> r = 1.04 -> J = 0.96    <-- correct
    //
    // J=0.408 sits BELOW this material's own [0.5, 2.0] clamp floor, so every
    // base particle was pinned at exactly J=0.5 permanently -- live-confirmed
    // in this demo's own log line (`water_j=[0.500, ...]`, min pinned at the
    // floor, never moving). The old comment above rationalised this as
    // "genuine local compression, not a bug"; it isn't -- real water at this
    // load compresses by well under 1%, not 60%.
    //
    // `c_ref = 10 * v_max` is the standard weakly-compressible rule limiting
    // density variation to ~1% (Monaghan 1994; Becker & Teschner 2007 WCSPH,
    // both already cited elsewhere in this engine), with v_max from Torricelli
    // for this column. Identical derivation to `basic_fluids_gpu.rs`'s own
    // (that demo already did this correctly; this CPU demo was the holdout) --
    // including its same deliberately-derated gravity for acoustic sizing, so
    // the two demos stay directly comparable.
    const WATER_EOS_POWER: f32 = 3.0;
    const COLUMN_HEIGHT_CELLS: f32 = 52.0 * SPACING;
    const DERATED_GRAVITY_FOR_ACOUSTIC_SIZING: f32 = 0.3;
    let v_max_grid = (2.0 * DERATED_GRAVITY_FOR_ACOUSTIC_SIZING * COLUMN_HEIGHT_CELLS).sqrt();
    let c_ref_m_s = 10.0 * v_max_grid * config.dx_meters;
    let water_tait_b_pa = 1000.0 * c_ref_m_s * c_ref_m_s / WATER_EOS_POWER;
    let water = NewtonianFluidMaterial::new(0.1, 1.0e-3, water_tait_b_pa, WATER_EOS_POWER);
    let mud = BinghamFluidMaterial::new(4.0, 8.0, 100.0, 3.0, 4.0);
    let ice = WithLatentHeat::new(NeoHookeanMaterial::new(4.0, 8.0), ICE_LATENT_HEAT);
    let thermal = ThermalDiffusion::new(
        ThermalConfig {
            conductivity: 0.6,
            heat_capacity: 4182.0,
            density: 1000.0, // kg/m^3, real water -- see ThermalConfig::density's own doc
            ambient: WARM_AMBIENT,
            // Must match the sim's real dx_meters -- ThermalConfig::grid_cell_size
            // requires this, else alpha_grid() is mis-scaled by orders of
            // magnitude.
            grid_cell_size: config.dx_meters,
            ..Default::default()
        },
        config.grid_res,
    );
    // Choose m=rho0*spacing² so the particles' conserved reference volumes
    // fill the intended region.  The strict fluid initializer then sets
    // V0=m/rho0 and rho=rho0; it does not use a kernel-density estimate as
    // thermodynamic state.
    // SPACING (top-level const, see its own doc) = 0.9, NOT the old 0.6 --
    // real, measured 45fps-debug-minimum fix: fewer, larger particles is a
    // real, disclosed RESOLUTION tradeoff, not a physics-accuracy one --
    // material constants below are untouched.
    const WATER_MASS: f32 = 0.1 * SPACING * SPACING;
    const MUD_MASS: f32 = 4.0 * SPACING * SPACING;
    let spawn_water = SpawnRegion {
        spacing: SPACING,
        box_size: IVec2::new(14, 52),
        // x=20, not the old 11 -- matches basic_fluids_gpu.rs's own fix (see
        // that file's doc): at x=11 the column's left edge sat only 2 cells
        // past the near-wall threshold, permanently close to a real wall-
        // contact regime rather than only during genuine interaction.
        box_center: Vec2::new(20.0, 30.0),
        material_id: MAT_WATER,
        initial_velocity_scale: 0.0,
        precompute_initial_volumes: true,
        mass_override: Some(WATER_MASS),
        ..SpawnRegion::for_sim(&config)
    };
    let spawn_mud = SpawnRegion {
        spacing: SPACING,
        box_size: IVec2::new(16, 18),
        box_center: Vec2::new(50.0, 38.0),
        material_id: MAT_MUD,
        initial_velocity_scale: 0.0,
        precompute_initial_volumes: true,
        mass_override: Some(MUD_MASS),
        ..SpawnRegion::for_sim(&config)
    };
    let mut solver = Simulation::new(config, spawn_water)
        .with_default_material(Box::new(water))
        .with_material(MAT_MUD, Box::new(mud))
        .with_material(MAT_ICE, Box::new(ice))
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)))
        .with_thermal(thermal)
        .with_phase_rule(|p| {
            if p.material_id == MAT_WATER && p.temperature < FREEZING_POINT {
                Some(MAT_ICE)
            } else {
                None
            }
        });
    // TEMPORARY (2026-08-13): mud spawn disabled -- isolating to water-only
    // per direct instruction, since the reported "weird" behavior is
    // specifically the water/mud INTERACTION, not either material alone.
    // Material stays registered (`with_material` above) so mud can be
    // re-enabled by uncommenting the line below once water-only is solid.
    // `add_body` appends particles synchronously, so it must run BEFORE this
    // temperature-init loop -- otherwise mud particles are left at
    // `initialize_particles`'s default (effectively 0K), not WARM_AMBIENT.
    let _ = &spawn_mud;
    // let _ = solver.add_body(spawn_mud);
    for t in solver.particles_mut().temperature.iter_mut() {
        *t = WARM_AMBIENT;
    }
    solver
}

// Ported unchanged from basic_fluids_gpu.rs's own PROVEN, live-verified
// `Pattern::Vortex` (see that match arm's own extensive doc, and project_
// vortex_siphon_saga_2026-08-15 memory) -- grid-cell-unit constants, not
// tied to gravity/material specifics, so they transfer directly. Mechanism:
// a `GravityWellField` drain + a free-vortex (constant angular momentum,
// v=L/r, NOT solid-body v=omega*r) seed velocity. NO RadialConfinement, NO
// mass sink -- both real, tried-and-rejected on the GPU side (see that
// file's doc): a sink drains the whole pool into a scattered mess within a
// minute (wrong physical target -- real ocean whirlpools persist via
// Kelvin's circulation theorem, Thomson 1869, no continuous outflow
// needed); RadialConfinement forcibly reshapes ordinary resting pool water
// far outside the vortex's own radius. Headless-verified on THIS demo's own
// config 2026-08-16 (`diag_cpu_vortex_probe.rs`): 400 steps, mass exactly
// conserved, core structure (particles within the seed radius) stays
// orbiting at a stable mean radius/speed, not collapsing or scattering.
const VORTEX_POOL_BOX: IVec2 = IVec2::new(54, 48);
const VORTEX_DRAIN_EDGE_R: f32 = 17.0;
const VORTEX_DRAIN_EDGE_ACCEL_FRACTION: f32 = 0.02;
const VORTEX_DRAIN_SOFTENING: f32 = 2.0;
const VORTEX_SWIRL_SEED_RADIUS: f32 = 22.0;
const VORTEX_SEED_EDGE_SPEED: f32 = 1.0;

/// The drain's real gravitational-parameter (G*M product), scaled by the
/// LIVE gravity magnitude (this demo's gravity is itself a live slider,
/// unlike the GPU demo's fixed derated value) and the drain-strength slider
/// -- same formula shape as the proven recipe
/// (`DRAIN_EDGE_ACCEL_FRACTION * |gravity| * drain_edge_r^2`), evaluated
/// fresh so the drain stays in the same real physical relationship to
/// gravity as the user adjusts either slider, rather than freezing at
/// whatever gravity happened to be when the vortex was turned on.
fn vortex_drain_gm(live_gravity_mag: f32, drain_strength: f32) -> f32 {
    drain_strength
        * VORTEX_DRAIN_EDGE_ACCEL_FRACTION
        * live_gravity_mag
        * VORTEX_DRAIN_EDGE_R
        * VORTEX_DRAIN_EDGE_R
}

fn make_vortex_sim(drain_strength: f32) -> Simulation {
    let mut config = SimConfig {
        min_dt: 1.0e-4,
        max_substeps_per_step: 150,
        phase_rules_once_per_step: true,
        material_cfl_coefficient: 0.3,
        cfl_include_affine_speed: false,
        ..SimConfig::earth(GRID, 0.01, DT)
    };
    // Same real default live gravity this demo already uses everywhere else
    // (`gravity_fraction` default 0.003 against `SimConfig::earth`'s real
    // IRL magnitude) -- not a new number.
    let live_gravity = config.gravity * 0.003;
    config.gravity = live_gravity;

    const WATER_EOS_POWER: f32 = 3.0;
    let pool_height_cells = VORTEX_POOL_BOX.y as f32 * SPACING;
    let v_max_grid = (2.0 * live_gravity.length() * pool_height_cells).sqrt();
    let c_ref_m_s = 10.0 * v_max_grid * config.dx_meters;
    let water_tait_b_pa = 1000.0 * c_ref_m_s * c_ref_m_s / WATER_EOS_POWER;
    let water =
        NewtonianFluidMaterial::new(0.1, 1.0e-3, water_tait_b_pa.max(1.0e-6), WATER_EOS_POWER);

    let pool_center = Vec2::new(32.0, 26.0);
    // Drain near the FLOOR (pool spans ~y=[2,50]), not the pool's vertical
    // midpoint -- real, live feedback 2026-08-16: pulling toward the
    // midpoint reads as a swirl "floating in the middle," not a downward
    // funnel. Same mechanism/formulas, only the attraction point's height
    // changed -- headless-reverified stable (`diag_cpu_vortex_probe.rs`):
    // mass exactly conserved, core structure (mean radius ~13.8) persists
    // over 400 steps, not collapsing/scattering.
    let drain_center = Vec2::new(pool_center.x, 10.0);
    const WATER_MASS: f32 = 0.1 * SPACING * SPACING;
    let spawn_water = SpawnRegion {
        spacing: SPACING,
        box_size: VORTEX_POOL_BOX,
        box_center: pool_center,
        material_id: MAT_WATER,
        initial_velocity_scale: 0.0,
        precompute_initial_volumes: true,
        mass_override: Some(WATER_MASS),
        ..SpawnRegion::for_sim(&config)
    };

    let mut solver = Simulation::new(config, spawn_water)
        .with_default_material(Box::new(water))
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)))
        .with_named_force_field(
            "vortex_drain",
            Box::new(GravityWellField::new(
                vec![(
                    drain_center,
                    vortex_drain_gm(live_gravity.length(), drain_strength),
                )],
                1.0,
                VORTEX_DRAIN_SOFTENING,
            )),
        );

    let seed_l = VORTEX_SEED_EDGE_SPEED * VORTEX_DRAIN_EDGE_R;
    for i in 0..solver.particles().x.len() {
        let x = solver.particles().x[i];
        let r = x - drain_center;
        let dist = r.length();
        if dist < VORTEX_SWIRL_SEED_RADIUS {
            let d = dist.max(VORTEX_DRAIN_SOFTENING);
            solver.particles_mut().v[i] = (seed_l / (d * d)) * Vec2::new(-r.y, r.x);
        }
    }
    for t in solver.particles_mut().temperature.iter_mut() {
        *t = WARM_AMBIENT;
    }
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
    last_cursor_grid: Vec2,
    lmb: bool,
    rmb: bool,
    digging: bool,
    push_strength: f32,
    dig_strength: f32,
    real_gravity: Vec2,
    gravity_fraction: f32,
    cold: bool,
    /// Real, live-verified vortex/drain (see `make_vortex_sim`'s own doc) --
    /// when on, the scene is a flat resting pool with a real
    /// `GravityWellField` drain instead of the dam-break column.
    vortex_mode: bool,
    /// 0 = drain off (still a flat pool, just no force), 1 = the proven
    /// recipe's own real strength, up to 2 for headroom -- multiplies
    /// `vortex_drain_gm`'s formula, does not replace it with an arbitrary
    /// number.
    drain_strength: f32,
    frame: u64,
    fps_timer: std::time::Instant,
    fps_frames: u64,
    last_fps: f32,
    // Real diagnostic added 2026-08-10 -- user's own live report that
    // interaction (push/pull) "slows down the physics" even when the
    // averaged fps counter looks fine. Two headless hypotheses tested and
    // BOTH ruled out with real data (sim_time_dropped unchanged during a
    // scripted push; wall-clock step() cost actually LOWER during a push,
    // not higher) -- neither explains a live-felt slowdown, so the real
    // next diagnostic has to be live, not another isolated guess.
    // `last_fps` is a 1-SECOND AVERAGE, which hides exactly the kind of
    // short spike a user would feel as a stutter during active interaction
    // -- this tracks the single WORST individual `Simulation::step()` call
    // within that same averaging window instead, so a spike becomes
    // directly visible and correlatable with what's actually being done at
    // the time (pushing near a wall, digging, etc.), not silently smoothed
    // away by the average.
    worst_step_ms_this_window: f32,
    last_worst_step_ms: f32,
    fps_log_count: u32,
    // Real bug found live 2026-08-10 (user: "la physique est bizarre, la
    // gravite est pas trop forte?"): this demo called `sim.step()` once per
    // RENDER frame, unconditionally, with `SimConfig::dt_seconds = DT =
    // 0.1s` baked in -- at the live-measured ~48fps that's 100ms of
    // simulated time advanced every ~21ms of wall-clock, a real ~4.8x
    // time-dilation (the whole scene, gravity included, played out ~5x
    // faster than real time). Every OTHER demo in the project already
    // avoids exactly this via `FixedStepController` (see `runtime/README.md`
    // -- "decouples real frame rate from a fixed physics dt... used across
    // every GPU demo"); this CPU demo was the one exception, added after
    // that rollout and never migrated. Fixed by driving `sim.step()` off
    // real elapsed time (`simulation_speed: 1.0` = real-time, no playback-
    // speed knob wanted here) instead of render cadence. NOTE: the substep-
    // cap fps sweep documented on `max_substeps_per_step` above (cap=12 ->
    // 48fps) was measured under the OLD always-step-once-per-frame regime
    // and is now stale -- physics now steps ~10Hz instead of ~48Hz, so real
    // per-frame physics cost dropped ~4.8x; a fresh sweep would be needed to
    // re-tune the cap, not attempted tonight.
    stepper: FixedStepController,
    last_instant: std::time::Instant,
    render_mode: RenderMode,
    grid_bridge_buf: wgpu::Buffer,
    material_mass_bridge_buf: wgpu::Buffer,
    particle_bridge_buf: wgpu::Buffer,
    /// Persistent scratch for the per-frame CPU->GPU render bridges.
    ///
    /// These used to be freshly `vec![]`/`collect()`ed every frame, which at
    /// this scene's size churned ~373 KB (Surface: 2912 particles x 128 B)
    /// or ~320 KB (GridVolume: 64x64x4 + 64x64x16 f32) of allocate-fill-free
    /// per frame for zero benefit -- the contents are fully rewritten each
    /// time either way, so reusing the storage is bit-identical output with
    /// no allocator traffic.
    bridge_particles: Vec<Particle>,
    bridge_dense: Vec<f32>,
    bridge_material_mass: Vec<f32>,
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
        let real_gravity = sim.config().gravity;
        let mut renderer = Renderer::new(&device, sim.particles().len(), fmt);
        renderer.set_camera(
            &queue,
            GRID as u32,
            size.width,
            size.height,
            PARTICLE_RENDER_DIAMETER,
            true,
        );
        renderer.set_color_mode(ColorMode::ByMaterial);
        // Real, disclosed render fix (2026-08-13): the grid-volume and
        // curvature-flow-surface paths threshold on ABSOLUTE cell mass
        // (`mass_floor = 0.15`), a constant written for scenes whose occupied
        // cells weigh "order 0.5-4". This scene is calibrated to REAL water,
        // `rho0 = 1000 kg/m^3 * dx^2 = 0.1` grid units, so a completely full
        // cell weighs 0.1 -- BELOW that floor. Every cell was discarded, so
        // both of those modes rendered the fluid as near-empty while the
        // raw-particle mode showed it correctly (live-confirmed: the three
        // modes visibly disagreed about where the fluid even was). Telling the
        // renderer this scene's real full-cell mass makes the thresholds mean
        // "fraction of a full cell", which is what they were always intended
        // to mean.
        renderer.set_grid_reference_cell_mass(0.1);
        // Surface grid at 4x the physics grid, not the 6x default -- a real
        // sampling-statistics fix for the speckle/"white noise" on the
        // surface, not a quality cut.
        //
        // The reconstruction runs at `grid_res * multiplier`, so at 6x this
        // scene had 384x384 = 147k surface cells for only 2912 particles --
        // ~51 cells per particle. Each cell therefore samples a tiny, noisy
        // subset of the particle distribution, and since the shading normal
        // is a finite difference OF that density field, per-cell density noise
        // becomes per-pixel lighting noise. Coarsening to 4x gives 256x256 =
        // 65k cells (~23 per particle): each cell averages more than twice as
        // many particles, so the field -- and the normals taken from it -- are
        // measurably smoother.
        //
        // Cost scales with the SQUARE of the multiplier, so this is also 0.44x
        // the surface-pass work. Strictly better on both axes at this particle
        // count; raise it again if the particle count rises.
        renderer.set_surface_res_multiplier(4);
        // Splat width left at the plain default (1.0). A derivation from real
        // particle spacing exists (`Renderer::set_particle_spacing_cells`,
        // reasoning in its own doc) and was wired in here 2026-08-14, but
        // live review found a real regression once composed with an
        // additional boundary-truncation factor and this demo's own already-
        // raised `surface_res_multiplier` -- a gap opening at the free
        // surface, worsening further with resolution. Un-wired here pending
        // a proper visual diagnosis of that interaction; the derivation
        // function itself stays available, just not called by default.
        // Optical properties for the volumetric render paths. Without these
        // every slot keeps its 0.0 default, so `grid-volume`/`surface` shade
        // with no absorption, no subsurface scattering and no Fresnel -- which
        // is exactly why this demo's water rendered flat GREY while the same
        // scene in `basic_fluids_gpu.rs` (which does set them) looks like
        // water. Ported from that demo, same values, rather than re-guessed.
        //
        // The absorption triple is physically meaningful, not a palette pick:
        // water's absorption coefficient rises steeply with wavelength, so red
        // is attenuated ~12x more strongly than blue. Encoding that as
        // sigma_a = [0.85, 0.25, 0.07] (R,G,B) makes transmitted light go blue
        // through depth for the real Beer-Lambert reason, instead of being
        // tinted blue by hand.
        renderer.set_optical_params(&queue, MAT_WATER as usize, [0.85, 0.25, 0.07]);
        renderer.set_optical_scattering(&queue, MAT_WATER as usize, 0.03);
        renderer.set_specular_r0(&queue, MAT_WATER as usize, 0.02);
        // Mud + ice keep their own distinct look (both currently unspawned in
        // this water-only isolation, but registered, so their slots must not
        // silently inherit water's).
        renderer.set_optical_params(&queue, MAT_MUD as usize, [0.30, 0.20, 0.12]);
        renderer.set_optical_scattering(&queue, MAT_MUD as usize, 0.08);
        renderer.set_specular_r0(&queue, MAT_MUD as usize, 0.005);
        // Ice: far less absorbing than liquid water (clear ice transmits
        // deeply) and much glossier -- r0 ~0.05 vs water's 0.02.
        renderer.set_optical_params(&queue, MAT_ICE as usize, [0.30, 0.12, 0.05]);
        renderer.set_optical_scattering(&queue, MAT_ICE as usize, 0.06);
        renderer.set_specular_r0(&queue, MAT_ICE as usize, 0.05);

        // CPU->GPU render bridges for RenderMode::GridVolume/Surface -- see
        // RenderMode's own doc for why these exist (no persistent GPU buffer
        // on the CPU `Simulation` path). Sized once at particle-count-fixed
        // scene setup, matching `fire_spread.rs`'s own grid-bridge precedent.
        const RENDER_MATERIAL_SLOTS: u64 = 16;
        let grid_bridge_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("basic_fluids_gui_grid_bridge"),
            size: (GRID * GRID * 4 * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let material_mass_bridge_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("basic_fluids_gui_material_mass_bridge"),
            size: (GRID as u64 * GRID as u64 * RENDER_MATERIAL_SLOTS) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let particle_bridge_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("basic_fluids_gui_particle_bridge"),
            size: (sim.particles().len() * std::mem::size_of::<Particle>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

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
            "basic_fluids_gui: {} particles  |  LMB push  RMB pull  D toggle dig  G render mode  R reset  Q quit",
            sim.particles().len()
        );
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
            last_cursor_grid: Vec2::ZERO,
            lmb: false,
            rmb: false,
            digging: false,
            push_strength: 5.0,
            dig_strength: 18.0,
            real_gravity,
            // 0.01, matching the already-validated sand/snow checkpoint at this
            // same grid scale -- not re-guessed live.
            gravity_fraction: 0.003,
            cold: false,
            vortex_mode: false,
            drain_strength: 1.0,
            frame: 0,
            fps_timer: std::time::Instant::now(),
            fps_frames: 0,
            last_fps: 0.0,
            worst_step_ms_this_window: 0.0,
            last_worst_step_ms: 0.0,
            fps_log_count: 0,
            // simulation_speed = 6.0, NOT 1.0 (true real-time) -- 1.0 was
            // tried first (real-time stepping fix), but analysis caught a
            // real follow-on problem before shipping it as final: with
            // `DT=0.1` (100ms/step), true real-time only steps ~10Hz while
            // this demo renders at ~45-60fps, so positions would freeze for
            // ~4-5 render frames then jump -- classic missing-render-
            // interpolation artifact of decoupled fixed-timestep (matches
            // the user's own separate, live "ca lag" report, though that
            // report's timing versus this exact build was not confirmed).
            // Real fix (not a bandaid): mirror
            // `basic_fluids_gpu.rs`'s OWN already-shipped answer to the
            // identical situation -- that demo uses the SAME `DT=0.1`
            // (`PLAYBACK_SPEED=6.0 / RENDER_FPS_TARGET=60.0`) and picks
            // `simulation_speed = RENDER_FPS_TARGET * DT = 6.0` specifically
            // so 1 physics step lands on ~every render frame at its target
            // fps -- not an arbitrary speedup, the same demo-family tuning
            // this file's `gravity_fraction`/`material_cfl_coefficient`
            // already cross-reference. `max_substeps_per_frame: 1`, matching
            // the GPU demo's OWN value after all -- REAL CORRECTION
            // (2026-08-10, same session): first shipped as `3` on the
            // unverified assumption that "this demo's steps are cheap
            // (`max_substeps_per_step: 12`, not the GPU demo's 150)". Real
            // measurement (a stderr fps/worst_step log, see the `fps_
            // log_count` field below) caught this wrong: `worst_step_ms` on
            // this exact scene is 22-48ms typically, spiking to 127ms -- NOT
            // cheap relative to a 60fps (16.6ms) frame budget. At `3`, a
            // single render frame could cram up to 3 real `Simulation::
            // step()` calls trying to catch up to the 6x-speed target,
            // compounding into a measured ~8-12fps (worse than doing nothing
            // -- a real, self-inflicted regression, not the user
            // misperceiving an actual improvement). `1` restores the GPU
            // demo's own "a slow frame becomes visible slow motion, never a
            // compounding catch-up spiral" property, which turns out to
            // apply here too -- the "steps are cheap here" premise was
            // simply wrong, not measured before being written down.
            stepper: FixedStepController::new(FixedStepConfig {
                dt: DT,
                simulation_speed: RENDER_FPS_TARGET * DT,
                max_substeps_per_frame: 1,
                max_frame_delta: 1.0 / 15.0,
            }),
            last_instant: std::time::Instant::now(),
            render_mode: RenderMode::Particles,
            grid_bridge_buf,
            material_mass_bridge_buf,
            particle_bridge_buf,
            bridge_particles: Vec::new(),
            bridge_dense: Vec::new(),
            bridge_material_mass: Vec::new(),
        }
    }

    /// Rebuilds `grid_bridge_buf`/`material_mass_bridge_buf` from the CPU
    /// solver's current state and uploads them -- same real, disclosed
    /// approximation as `fire_spread.rs`'s own bridge (simplified nearest-
    /// cell scatter, not P2G's full quadratic B-spline kernel; good enough
    /// for dominant-material color selection, not a physics-accuracy claim).
    /// Only called when `render_mode == GridVolume`, so every other mode
    /// (including the default) pays zero extra cost.
    fn upload_grid_volume_bridge(&mut self) {
        const SLOTS: usize = 16;
        let grid = self.sim.grid();
        // Reused across frames -- see `bridge_dense`'s own doc.
        self.bridge_dense.clear();
        self.bridge_dense.resize(GRID * GRID * 4, 0.0);
        let dense = &mut self.bridge_dense;
        for y in 0..GRID {
            for x in 0..GRID {
                let idx = y * GRID + x;
                dense[idx * 4 + 2] = grid.mass_at(IVec2::new(x as i32, y as i32));
            }
        }
        self.queue
            .write_buffer(&self.grid_bridge_buf, 0, bytemuck::cast_slice(dense));

        let particles = self.sim.particles();
        self.bridge_material_mass.clear();
        self.bridge_material_mass.resize(GRID * GRID * SLOTS, 0.0);
        let material_mass = &mut self.bridge_material_mass;
        for i in 0..particles.x.len() {
            let p = particles.x[i];
            let cx = (p.x.round() as i32).clamp(0, GRID as i32 - 1) as usize;
            let cy = (p.y.round() as i32).clamp(0, GRID as i32 - 1) as usize;
            let slot = (particles.material_id[i] as usize) % SLOTS;
            material_mass[(cy * GRID + cx) * SLOTS + slot] += particles.mass[i];
        }
        self.queue.write_buffer(
            &self.material_mass_bridge_buf,
            0,
            bytemuck::cast_slice(material_mass),
        );
    }

    /// Rebuilds `particle_bridge_buf` from the CPU solver's current
    /// particles and uploads it -- first CPU demo to drive the curvature-
    /// flow dual-phase surface path (`RenderMode::Surface`). `Particle` is
    /// already `repr(C)`/`Pod` (GPU-uploadable by design, see CLAUDE.md's
    /// own struct doc), so this is a direct `bytemuck::cast_slice` of the
    /// AoS view `Particles::iter()` already produces elsewhere (e.g.
    /// `Renderer::render`'s own per-particle loop) -- no new layout work,
    /// just a buffer this demo didn't previously need. Only called when
    /// `render_mode == Surface`.
    /// Real, disclosed fix (2026-08-10) for two real bugs the user found live:
    /// `render_surface_reconstruction_dual_phase` only knows 2 material IDs
    /// (`material_id_a`/`material_id_b`, see `DualPhaseSurfaceSource`'s own
    /// doc), so ice (`MAT_ICE`, neither slot) silently vanished from the
    /// surface once water froze -- and the earlier same-day fix (remapping
    /// ice->water in this snapshot) traded that bug for a different one:
    /// ice rendering visually IDENTICAL to water. Real fix: switched the
    /// caller to `render_surface_reconstruction`'s N-material path
    /// (`material_mass_enabled`), which colors every cell from its own real
    /// per-material mass -- water/mud/ice all stay visually distinct, no
    /// remap needed here at all.
    fn upload_particle_bridge(&mut self) {
        // No ice->water remap: render_surface_reconstruction's material_mass_enabled
        // path colors every real material_id (water/mud/ice) from its own per-cell
        // mass, so all 3 stay visually distinct instead of collapsing to one slot.
        self.bridge_particles.clear();
        self.bridge_particles.extend(self.sim.particles().iter());
        self.queue.write_buffer(
            &self.particle_bridge_buf,
            0,
            bytemuck::cast_slice(&self.bridge_particles),
        );
    }

    fn resize(&mut self, w: u32, h: u32) {
        if w == 0 || h == 0 {
            return;
        }
        self.surface_config.width = w;
        self.surface_config.height = h;
        self.surface.configure(&self.device, &self.surface_config);
        self.renderer.set_camera(
            &self.queue,
            GRID as u32,
            w,
            h,
            PARTICLE_RENDER_DIAMETER,
            true,
        );
    }

    fn cursor_grid(&self) -> Vec2 {
        let (gx, gy) = self.renderer.screen_to_grid(
            self.cursor_pos[0],
            self.cursor_pos[1],
            self.surface_config.width,
            self.surface_config.height,
        );
        Vec2::new(gx, gy)
    }

    /// Applies the live gravity slider + freeze/thaw ambient/cooling-rate
    /// toggle to the sim config. Split out of `update_and_render` purely for
    /// readability -- no behavior change.
    fn apply_gravity_and_thermal(&mut self) {
        self.sim
            .set_gravity(self.real_gravity * self.gravity_fraction);
        if let Some(cfg) = self.sim.thermal_config_mut() {
            cfg.ambient = if self.cold {
                COLD_AMBIENT
            } else {
                WARM_AMBIENT
            };
            // Convective (Newton) cooling, not bulk conduction -- see
            // FREEZER_COOLING_RATE's own doc; bulk conduction alone is too slow
            // to ever visibly freeze in a play session.
            cfg.cooling_rate = if self.cold { FREEZER_COOLING_RATE } else { 0.0 };
        }
    }

    /// Applies the LMB/RMB radial push-pull impulse, and returns this
    /// frame's cursor position plus a real digging direction (if actively
    /// digging and the cursor moved) for `step_physics` to apply once per
    /// real physics step below -- see that method's own doc for why the
    /// direction is sampled here (render cadence) but applied there
    /// (physics cadence).
    fn apply_interaction_forces(&mut self) -> (Vec2, Option<Vec2>) {
        if self.lmb || self.rmb {
            let mag = if self.lmb {
                self.push_strength
            } else {
                -self.push_strength
            };
            self.sim.apply_radial_impulse(self.cursor_grid(), 5.0, mag);
        }
        let cursor = self.cursor_grid();
        let dig_dir = if self.digging {
            let delta = cursor - self.last_cursor_grid;
            (delta.length_squared() > 1.0e-8).then(|| delta.normalize())
        } else {
            None
        };
        self.last_cursor_grid = cursor;
        (cursor, dig_dir)
    }

    /// TEMP diagnostic (2026-08-06) -- delete after use. Investigating a real
    /// live-reported "hold shape ~0.2s then sudden brutal collapse" -- this
    /// demo never had per-frame diagnostic printing (unlike basic_fluids_gpu.rs),
    /// so there was no data to check the claim against.
    fn log_early_frame_diagnostics(&self) {
        if self.frame > 20 {
            return;
        }
        let snap = self.sim.diagnostics_snapshot();
        let water_j = self
            .sim
            .particles()
            .deformation_gradient
            .iter()
            .zip(self.sim.particles().material_id.iter())
            .filter(|&(_, &m)| m == MAT_WATER)
            .map(|(f, _)| f.determinant())
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), j| {
                (lo.min(j), hi.max(j))
            });
        let mud_j = self
            .sim
            .particles()
            .deformation_gradient
            .iter()
            .zip(self.sim.particles().material_id.iter())
            .filter(|&(_, &m)| m == MAT_MUD)
            .map(|(f, _)| f.determinant())
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), j| {
                (lo.min(j), hi.max(j))
            });
        eprintln!(
            "frame={}  max_speed={:.3}  non_finite={}  water_j=[{:.3},{:.3}]  mud_j=[{:.3},{:.3}]  gravity_frac={:.3}",
            self.frame,
            snap.max_particle_speed,
            snap.non_finite_particle_values,
            water_j.0,
            water_j.1,
            mud_j.0,
            mud_j.1,
            self.gravity_fraction,
        );
    }

    /// Advances real simulated time by however many physics steps
    /// `FixedStepController` says real elapsed wall-clock time warrants
    /// (see that field's own doc on `State` for the real time-dilation bug
    /// this fixes) -- 0 most render frames at this demo's playback speed,
    /// never more than `max_frame_delta` allows.
    fn step_physics(&mut self, cursor: Vec2, dig_dir: Option<Vec2>) {
        let now = std::time::Instant::now();
        let frame_delta = (now - self.last_instant).as_secs_f32();
        self.last_instant = now;
        let steps = self.stepper.steps_for_frame(frame_delta);
        for _ in 0..steps {
            if let Some(dir) = dig_dir {
                let particles = self.sim.particles_mut();
                for i in 0..particles.len() {
                    if (particles.x[i] - cursor).length() < DIG_RADIUS {
                        particles.v[i] += dir * self.dig_strength * DT;
                    }
                }
            }
            let step_start = std::time::Instant::now();
            self.sim.step();
            let step_ms = step_start.elapsed().as_secs_f32() * 1000.0;
            self.worst_step_ms_this_window = self.worst_step_ms_this_window.max(step_ms);
            // Low-cost permanent tripwire (silent in normal operation) --
            // 2026-08-10, chased a real periodic ~150ms spike that turned
            // out to be system noise, not an engine bug (see
            // [[basic_fluids_gui_perf_regression_and_cleanup_2026-08-10]]
            // items 7-8: substeps_last_step is pinned at the cap regardless
            // of CFL, and a genuinely quiet-machine run showed zero spikes).
            // Left in place with full phase-timing + substep-count context
            // in case a real spike ever recurs for real.
            if step_ms > 50.0 {
                let snap = self.sim.diagnostics_snapshot();
                let t = snap.timing;
                // Every phase timer `StepTiming` actually has -- the previous
                // line printed only 4 of them, leaving ~69% of a 178ms step
                // unattributed and making the real cost impossible to find.
                // `grid_update_us` INCLUDES `pressure_us` (documented subset,
                // not additive); everything else is disjoint, so these should
                // sum to ~`total_us`.
                let accounted = t.p2g_us
                    + t.grid_update_us
                    + t.g2p_us
                    + t.fields_us
                    + t.thermal_us
                    + t.cfl_us
                    + t.spatial_hash_us
                    + t.phase_sleep_us
                    + t.project_us
                    + t.density_us
                    + t.retry_snapshot_us;
                eprintln!(
                    "SPIKE frame={} step={:.1}ms subs={} cfl={:.3} | p2g={} grid_update={} (pressure={}) g2p={} cfl_sel={} project={} spatial_hash={} phase_sleep={} fields={} thermal={} density={} retry_snap={} | accounted={} total={} MISSING={}",
                    self.frame,
                    step_ms,
                    snap.substeps_last_step,
                    snap.cfl_number,
                    t.p2g_us,
                    t.grid_update_us,
                    t.pressure_us,
                    t.g2p_us,
                    t.cfl_us,
                    t.project_us,
                    t.spatial_hash_us,
                    t.phase_sleep_us,
                    t.fields_us,
                    t.thermal_us,
                    t.density_us,
                    t.retry_snapshot_us,
                    accounted,
                    t.total_us,
                    t.total_us.saturating_sub(accounted),
                );
            }
            self.frame += 1;
            self.log_early_frame_diagnostics();
        }
    }

    /// Updates the 1-second-averaged fps/worst-step-ms counters the panel
    /// (and the stderr diagnostic below) display.
    fn update_fps_counters(&mut self) {
        self.fps_frames += 1;
        if self.fps_timer.elapsed().as_secs_f32() >= 1.0 {
            self.last_fps = self.fps_frames as f32 / self.fps_timer.elapsed().as_secs_f32();
            self.fps_timer = std::time::Instant::now();
            self.fps_frames = 0;
            self.last_worst_step_ms = self.worst_step_ms_this_window;
            self.worst_step_ms_this_window = 0.0;
            // TEMP diagnostic (2026-08-10) -- delete after use. Screenshot
            // capture is known-unreliable in this environment (see
            // reference_screenshot_tooling_printwindow.md); this prints the
            // SAME numbers the on-screen panel shows, so real fps/perf can be
            // read from stdout without a screenshot. Capped to the first 20
            // seconds so it doesn't spam a long-running session.
            if self.fps_log_count < 20 {
                self.fps_log_count += 1;
                eprintln!(
                    "sec={}  fps={:.1}  worst_step={:.2}ms  render={:?}",
                    self.fps_log_count, self.last_fps, self.last_worst_step_ms, self.render_mode,
                );
            }
        }
    }

    /// Dispatches to whichever of the 3 real render paths `G` last selected
    /// -- see `RenderMode`'s own doc for what each one is and why the CPU
    /// solver needs a fresh bridge upload for the latter two.
    fn render_scene(&mut self, view: &wgpu::TextureView) {
        match self.render_mode {
            RenderMode::Particles => {
                self.renderer
                    .render(&self.device, &self.queue, self.sim.particles(), view, true);
            }
            RenderMode::GridVolume => {
                self.upload_grid_volume_bridge();
                self.renderer.render_grid_volume(
                    &self.device,
                    &self.queue,
                    GridVolumeSource {
                        grid: &self.grid_bridge_buf,
                        material_mass: &self.material_mass_bridge_buf,
                        material_mass_enabled: true,
                    },
                    view,
                    true,
                );
            }
            RenderMode::Surface => {
                self.upload_particle_bridge();
                // N-material per-cell coloring (`material_mass_enabled`),
                // NOT dual-phase -- real, disclosed switch, 2026-08-10.
                // Dual-phase caps at exactly 2 materials; this demo has 3
                // (water/mud/ice), and the earlier fix (remapping ice's
                // material_id to water's JUST for this render buffer) closed
                // the "ice vanishes" bug but created a real, different
                // problem the user caught live: ice became VISUALLY
                // IDENTICAL to liquid water, losing its own real, distinct
                // optical properties (`OpticalTable` slot 2) even though ice
                // and water are genuinely different materials. This path
                // builds `surface_material_mass` internally from each
                // particle's OWN real `material_id` (same quadratic B-spline
                // kernel as the density splat itself, not the coarser
                // nearest-cell approximation `upload_grid_volume_bridge`
                // uses for `GridVolume` mode) -- every material renders in
                // its own true color, no remap hack needed. Real, disclosed
                // tradeoff kept from switching away from dual-phase: this is
                // ONE shared density/smoothing field, not two independently-
                // smoothed surfaces, so materials can blend slightly AT
                // their exact touching boundary (dual-phase's own real
                // reason to exist) -- correct coloring for 3+ materials was
                // judged the more important property here.
                self.renderer.render_surface_reconstruction(
                    &self.device,
                    &self.queue,
                    SurfaceReconstructionSource {
                        particle_buf: &self.particle_bridge_buf,
                        particle_count: self.sim.particles().len(),
                        grid_res: GRID as u32,
                        material_slot: MAT_WATER,
                        material_mass_enabled: true,
                        dt: DT,
                    },
                    view,
                    true,
                );
            }
        }
    }

    fn update_and_render(&mut self, window: &Window) {
        self.apply_gravity_and_thermal();
        let (cursor, dig_dir) = self.apply_interaction_forces();
        self.step_physics(cursor, dig_dir);
        self.update_fps_counters();

        let output = match self.surface.get_current_texture() {
            Ok(t) => t,
            Err(_) => return,
        };
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        self.render_scene(&view);

        // --- egui panel ---
        let raw_input = self.egui_state.take_egui_input(window);
        let fps = self.last_fps;
        let worst_step_ms = self.last_worst_step_ms;
        let mut push_strength = self.push_strength;
        let mut dig_strength = self.dig_strength;
        let mut gravity_fraction = self.gravity_fraction;
        let mut digging = self.digging;
        let mut cold = self.cold;
        let vortex_mode_before = self.vortex_mode;
        let mut vortex_mode = self.vortex_mode;
        let mut drain_strength = self.drain_strength;
        // Live render-solver dials, read back from the renderer so the widgets
        // always show the value actually in use.
        let mut curvature_iters = self.renderer.curvature_iterations();
        let mut surface_mult = self.renderer.surface_res_multiplier();
        let mut splat_width = self.renderer.splat_width_cells();
        let water_n = self
            .sim
            .particles()
            .iter()
            .filter(|p| p.material_id == MAT_WATER)
            .count();
        let mud_n = self
            .sim
            .particles()
            .iter()
            .filter(|p| p.material_id == MAT_MUD)
            .count();
        let ice_n = self
            .sim
            .particles()
            .iter()
            .filter(|p| p.material_id == MAT_ICE)
            .count();
        let mut reset = false;
        let render_mode_label = match self.render_mode {
            RenderMode::Particles => "particles",
            RenderMode::GridVolume => "grid-volume",
            RenderMode::Surface => "curvature-flow surface",
        };
        let full_output = self.egui_ctx.run(raw_input, |ctx| {
            egui::Window::new("Fluids")
                .default_pos([10.0, 10.0])
                .default_width(260.0)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.label(format!(
                        "fps={fps:.0}  worst_step={worst_step_ms:.1}ms  render={render_mode_label}"
                    ));
                    ui.label(format!("water={water_n}  mud={mud_n}  ice={ice_n}"));
                    ui.separator();
                    ui.label("Gravity (1.0 = real IRL 9.81 m/s², use --release above ~0.1):");
                    ui.add(egui::Slider::new(&mut gravity_fraction, 0.0..=2.0));
                    ui.separator();
                    ui.label("Push/pull strength:");
                    ui.add(egui::Slider::new(&mut push_strength, 0.0..=20.0));
                    ui.checkbox(&mut digging, "Digging/stirring active (or press D)");
                    ui.add(egui::Slider::new(&mut dig_strength, 0.0..=40.0).text("Dig strength"));
                    ui.separator();
                    ui.checkbox(&mut cold, "Cold ambient (water freezes below 273K)");
                    ui.separator();
                    ui.label("Vortex/drain (real GravityWellField + free-vortex seed):");
                    if ui
                        .checkbox(
                            &mut vortex_mode,
                            "Vortex pool (replaces dam-break on toggle)",
                        )
                        .changed()
                    {
                        reset = true;
                    }
                    ui.add_enabled(
                        vortex_mode,
                        egui::Slider::new(&mut drain_strength, 0.0..=2.0).text("Drain strength"),
                    );
                    ui.separator();
                    ui.label("Renderer (surface/grid-volume modes):");
                    ui.add(
                        egui::Slider::new(&mut curvature_iters, 2..=32)
                            .text("Smoothing passes (higher = rounder)"),
                    );
                    ui.add(
                        egui::Slider::new(&mut surface_mult, 1..=10)
                            .text("Surface detail (cost = square!)"),
                    );
                    ui.add(
                        egui::Slider::new(&mut splat_width, 0.2..=2.0)
                            .text("Splat width, cells (lower = sharper)"),
                    );
                    ui.separator();
                    ui.label("LMB push  RMB pull  D toggle dig  G render mode  R reset  Q quit");
                    if ui.button("Reset").clicked() {
                        reset = true;
                    }
                });
        });
        self.push_strength = push_strength;
        self.dig_strength = dig_strength;
        self.gravity_fraction = gravity_fraction;
        self.digging = digging;
        self.cold = cold;
        self.vortex_mode = vortex_mode;
        self.drain_strength = drain_strength;
        // Only call the setters when the value actually moved -- the surface
        // multiplier forces a buffer realloc, so writing it every frame would
        // rebuild the surface buffers continuously.
        if curvature_iters != self.renderer.curvature_iterations() {
            self.renderer.set_curvature_iterations(curvature_iters);
        }
        if surface_mult != self.renderer.surface_res_multiplier() {
            self.renderer.set_surface_res_multiplier(surface_mult);
        }
        if (splat_width - self.renderer.splat_width_cells()).abs() > 1.0e-4 {
            self.renderer.set_splat_width_cells(splat_width);
        }
        if reset {
            let sim = if self.vortex_mode {
                make_vortex_sim(self.drain_strength)
            } else {
                make_sim()
            };
            self.real_gravity = sim.config().gravity;
            self.sim = sim;
            self.frame = 0;
            self.stepper.reset();
            self.last_instant = std::time::Instant::now();
        } else if self.vortex_mode && vortex_mode_before {
            // Live-refresh the drain's real strength every frame it's
            // active -- both `gravity_fraction` and `drain_strength` are
            // live sliders, so the field must stay in the SAME physical
            // relationship to current gravity `make_vortex_sim` established
            // at construction, not freeze at whatever it was when the
            // vortex was first turned on.
            let live_gravity_mag = (self.real_gravity * self.gravity_fraction).length();
            self.sim.remove_force_field("vortex_drain");
            let drain_center = Vec2::new(32.0, 10.0); // must match make_vortex_sim's own
            self.sim.add_named_force_field(
                "vortex_drain",
                Box::new(GravityWellField::new(
                    vec![(
                        drain_center,
                        vortex_drain_gm(live_gravity_mag, self.drain_strength),
                    )],
                    1.0,
                    VORTEX_DRAIN_SOFTENING,
                )),
            );
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
                    .with_title("emerge -- Fluids (GUI)")
                    .with_inner_size(winit::dpi::LogicalSize::new(480u32, 480u32)),
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
                match key {
                    KeyCode::Escape | KeyCode::KeyQ if pressed => el.exit(),
                    KeyCode::KeyD if pressed => s.digging = !s.digging,
                    KeyCode::KeyG if pressed => {
                        s.render_mode = match s.render_mode {
                            RenderMode::Particles => RenderMode::GridVolume,
                            RenderMode::GridVolume => RenderMode::Surface,
                            RenderMode::Surface => RenderMode::Particles,
                        };
                    }
                    KeyCode::KeyR if pressed => {
                        let sim = if s.vortex_mode {
                            make_vortex_sim(s.drain_strength)
                        } else {
                            make_sim()
                        };
                        s.real_gravity = sim.config().gravity;
                        s.sim = sim;
                        s.frame = 0;
                        // Real elapsed time since the LAST render frame (e.g. the
                        // window was idle) must not be replayed as a burst of
                        // catch-up physics steps -- same fix basic_fluids_gpu.rs
                        // already applies on its own reset.
                        s.stepper.reset();
                        s.last_instant = std::time::Instant::now();
                        println!("reset");
                    }
                    _ => {}
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

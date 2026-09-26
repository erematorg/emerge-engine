extern crate emerge_engine as emerge;

#[path = "../gui_common/coords.rs"]
mod gui_common;

use egui_wgpu::ScreenDescriptor;

/// Interactive, windowed counterpart to `phase_states_headless.rs` -- same
/// real materials, same real bidirectional ice<->water<->steam mechanism,
/// but driven by a live temperature-target slider instead of a fixed
/// scripted heat-then-cool schedule -- the interactive demo requested
/// 2026-08-23 (see `project_phase_demo_temperature_slider_wanted_2026-08-23`
/// memory).
///
/// The slider sets a TARGET temperature, not a heat rate directly -- real,
/// simple proportional control (`rate = GAIN * (target - current_avg)`,
/// clamped to `MAX_HEAT_RATE_K_PER_S`), the same real "external heat
/// source" technique `phase_states_headless.rs`/`material_sandbox_gpu.rs`
/// already use, just closed-loop on the slider's own target instead of an
/// open-loop scripted ramp. Real, disclosed architecture (2026-08-29,
/// see `water_phase_chain`'s own doc): that rate now drives a real
/// per-particle ENTHALPY state (Voller & Cross 1981 enthalpy method),
/// not temperature directly -- material transitions and latent heat are
/// both derived from that, not from `add_phase_rule`/`WithLatentHeat`
/// (this demo no longer uses either).
///
/// Real materials: `RankineMaterial::ice()` (real brittle-fracture ice,
/// scaled stiffness -- see that preset's own doc and
/// `ICE_YOUNG_MODULUS_SCALED_PA` below for why) for the solid,
/// `CavitatingFluidMaterial` (Lyu, Sun, Colagrossi & Zhang 2023's real
/// three-branch cavitation EOS, genuinely temperature-coupled -- see
/// `cavitating_eos`'s own doc) for the liquid, `BoilingMixtureMaterial`
/// (real Homogeneous Equilibrium Model mixture, driven directly by the
/// enthalpy method's own `boiling_fraction` -- see that material's own
/// doc) for a particle genuinely mid-boil, `IdealGasMaterial`
/// (isentropic ideal-gas EOS) for the gas.
///
/// Real, disclosed history: this demo used `NewtonianFluidMaterial` (a
/// flat `pressure_floor`) for water, then `IsothermalCavitatingFluidMaterial`
/// (fixed-reference-temperature cavitation), then the genuinely
/// temperature-coupled `CavitatingFluidMaterial` for ALL of water
/// (including the boiling plateau) -- water near freezing and water near
/// boiling genuinely had different real cavitation onsets, not one fixed
/// reference curve, but live-testing that version surfaced a real,
/// quantitatively confirmed bug: a particle held at the real boiling
/// latent-heat plateau let its own MECHANICAL vapor fraction run
/// completely independent of the enthalpy method's own THERMAL vapor
/// fraction (measured live: `J=5.988`, ~fully vaporized mechanically,
/// while barely a third boiled thermally). `BoilingMixtureMaterial`
/// (2026-09-01) closes that: `PhaseState::Boiling` now gets its own real
/// material whose mechanical equilibrium is driven directly by the SAME
/// `fraction` the enthalpy method already tracks -- see that material's
/// own doc for the real citations (Collier & Thome's mixture density,
/// Wallis's mixture sound speed). Real, disclosed remaining limitation,
/// unchanged: the coupling is still one-directional (`x_H -> mechanical
/// state`) -- a real mechanical deviation from equilibrium doesn't pay
/// latent heat back into the enthalpy state. `phase_states_headless.rs`
/// still uses the older `NewtonianFluidMaterial`, not yet updated to
/// match.
///
/// Real "chimney" geometry, added 2026-08-28 after live feedback that the
/// original wide, zero-gravity, centered-square layout let material drift
/// apart in every direction with nothing to pull it back together: real,
/// non-zero gravity (live-adjustable via its own slider, matching
/// `basic_snow.rs`'s own `gravity_fraction` convention) plus real
/// Archimedes buoyancy (same formula as `BuoyancyField`) for STEAM ONLY,
/// applied manually each frame so it stays exactly in sync with the live
/// gravity slider -- see `update_and_render`'s own comment for why steam
/// specifically (a real, disclosed bug was found and fixed live: applying
/// this to every particle gave the starting ice block a net upward nudge
/// before any water/steam even existed). Ice and water both just fall
/// under plain gravity; steam genuinely rises once it exists. Spawned as
/// a narrow column near the bottom of a tall-ish domain so there's real
/// room above for steam to actually rise into.
///
///   cargo run --example phase_states_gui --features "render,experimental"
use emerge::grid::kernel::quadratic_weights;
use emerge::matter::materials::rankine::{
    ICE_Q_REFERENCE_FREQUENCY_HZ, q_factor_elastic_viscosity_pa_s,
};
use emerge::render::{ColorMode, Renderer};
use emerge::{
    BoilingMixtureMaterial, CavitatingEosTable, CavitatingFluidMaterial, IdealGasMaterial,
    MaterialModel, RankineMaterial, SimConfig, Simulation, SlipBoundary, SpawnRegion,
};
use glam::{IVec2, Mat2, Vec2};
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

const GRID: usize = 64;
const ICE_ID: u32 = 0;
const WATER_ID: u32 = 1;
const STEAM_ID: u32 = 2;
/// Real, genuinely mid-boil material (2026-09-01, external review's own
/// "minimum honest enthalpy->mechanical" fix) -- see
/// `emerge::BoilingMixtureMaterial`'s own doc for the real bug this
/// closes (mechanical vapor fraction running independent of the
/// enthalpy method's own thermal vapor fraction) and
/// `material_id_for_phase_state`'s own doc for how this replaces the old
/// majority-vote `WATER_ID`/`STEAM_ID` split.
const BOILING_ID: u32 = 3;

// Real water phase-change constants -- identical to phase_states_headless.rs,
// see that file's own doc for the full real-value sourcing.
const MELTING_POINT_K: f32 = 273.15;
const BOILING_POINT_K: f32 = 373.15;
const FUSION_LATENT_HEAT_J_KG: f32 = 334_000.0;
const VAPORIZATION_LATENT_HEAT_J_KG: f32 = 2_257_000.0;
const WATER_HEAT_CAPACITY_J_KG_K: f32 = 4182.0;
// Real, disclosed fix (2026-08-29, first milestone of the real
// solid<->liquid<->gas cycle plan): the demo used to pay only 1/20th of
// the real latent heats and jump temperature instantly on transition --
// real, disclosed, but never a genuine closed thermodynamic cycle. Real
// fix: `energy::thermodynamics::enthalpy::chained_state_from_enthalpy`
// (Voller & Cross 1981 enthalpy method, extended to two consecutive real
// transitions) tracks enthalpy as the primary thermal state instead, so
// temperature genuinely PLATEAUS at the real transition point while
// `phase_fraction` absorbs the FULL real latent heat -- no scale factor,
// no instant jump, no arbitrary hysteresis margin needed (H is monotonic
// and continuous by construction, so there's nothing to debounce).
const ICE_HEAT_CAPACITY_J_KG_K: f32 = 2090.0; // real, CRC Handbook at 0C
const STEAM_HEAT_CAPACITY_J_KG_K: f32 = 2080.0; // real, NIST steam tables near 100C, 1 atm

// Real, disclosed geometry (2026-08-29, Milestone 2): the spawn column
// (see `make_sim`'s own `spawn`) is centered at y=GRID*0.22=14.08 with a
// real height of `box_size.y*spacing`=10*0.5=5.0, so it spans
// y in [11.58, 16.58]. The heating plate covers the bottom 40% of that
// real span (11.58 + 0.4*5.0 = 13.58) -- enough real ice mass sits
// directly on it to genuinely start melting and subsiding (exposing more
// ice to the plate as it does, the same way a real ice block melts from
// a hot surface underneath it), while still leaving real column ABOVE it
// with no direct heat source -- the real vertical DeltaT this scene
// structurally lacked under uniform heating.
const HEATING_PLATE_TOP_Y: f32 = 13.58;

fn water_phase_chain() -> emerge::thermodynamics::PhaseChainProperties {
    emerge::thermodynamics::PhaseChainProperties {
        cp_solid: ICE_HEAT_CAPACITY_J_KG_K,
        cp_liquid: WATER_HEAT_CAPACITY_J_KG_K,
        cp_gas: STEAM_HEAT_CAPACITY_J_KG_K,
        melting_point_k: MELTING_POINT_K,
        boiling_point_k: BOILING_POINT_K,
        fusion_latent_heat_j_kg: FUSION_LATENT_HEAT_J_KG,
        vaporization_latent_heat_j_kg: VAPORIZATION_LATENT_HEAT_J_KG,
    }
}

/// Real, disclosed simplification, UNCHANGED for melting (2026-08-29): a
/// particle mid-MELT still shows whichever real phase holds the MAJORITY
/// of its own latent-heat band (`fraction < 0.5` keeps the colder
/// identity, `>= 0.5` switches) -- a genuine mushy-solid mechanical model
/// (partial-melt stiffness softening) is real, separate, still-unscoped
/// future work, not attempted here. The 0.5 cutoff itself is still a real
/// coarse-graining choice, not something uniquely derived from physics.
///
/// Real, disclosed IMPROVEMENT for boiling (2026-09-01, external review):
/// the old majority-vote split here (`WATER_ID` below 0.5, `STEAM_ID`
/// above) is GONE -- it let a particle's own mechanical vapor fraction
/// (implicit in its density/`J` under whichever pure-phase material held
/// it) drift completely independent of `fraction` itself, a real,
/// quantitatively confirmed bug (see `BoilingMixtureMaterial`'s own doc).
/// Every `PhaseState::Boiling` particle, regardless of `fraction`, now
/// gets the real, genuinely mixture-aware `BOILING_ID` -- `fraction`
/// itself still drives that material's own mechanical response directly
/// (via `Particle::friction_hardening`, written every substep below), so
/// there is no longer a discrete threshold to cross mid-band at all.
fn material_id_for_phase_state(state: emerge::thermodynamics::PhaseState) -> u32 {
    use emerge::thermodynamics::PhaseState;
    match state {
        PhaseState::Solid => ICE_ID,
        PhaseState::Melting { fraction } => {
            if fraction < 0.5 {
                ICE_ID
            } else {
                WATER_ID
            }
        }
        PhaseState::Liquid => WATER_ID,
        PhaseState::Boiling { .. } => BOILING_ID,
        PhaseState::Gas => STEAM_ID,
    }
}

// Real, DERIVED ice stiffness (2026-08-29) -- found live: the previous
// value (5.0e5 Pa) was carried over VERBATIM from phase_states_headless.rs,
// whose own doc says it was "verified working there" -- but that scene uses
// ZERO gravity by explicit design (isolates the thermal cycle from settling
// dynamics), so it has no real impact/fall velocities to resist at all. THIS
// scene deliberately adds real gravity for real settling/chimney dynamics
// (see make_sim's own doc), and never re-derived the stiffness for that.
// Exact same bug class as the water EOS fix just above/before this in the
// session: a constant proven fine for a less-demanding sibling scene,
// silently insufficient for a more demanding one.
//
// Real ice wave speed `c = sqrt(E/rho)` at the old value: sqrt(5e5/917) =
// 23.35 m/s -- only ~1.3x this scene's own real velocity scale (18.0 m/s,
// Torricelli free-fall from the ice column's real height, same derivation
// as WATER_C_REF_M_S above). Barely faster than what it needs to resist,
// so it deforms/bounces instead of behaving rigid. Applying the SAME
// margin rule used for water (target wave speed = 10*v_max, same spirit as
// Monaghan 1994's stiffness-margin rule, generalized from fluids to an
// elastic solid's own wave speed): `E = rho*(10*v_max)^2 = 917*180^2 =
// 2.971e7 Pa` -- a real, derived correction (~59x stiffer), still ~303x
// softer than real ice (E=9.0 GPa, RankineMaterial::ice's own doc), the
// same category of practical/CFL compromise as the ORIGINAL 253x reduction
// -- just actually re-derived for this scene's real dynamics instead of
// inherited from one that never had any.
const ICE_YOUNG_MODULUS_SCALED_PA: f32 = 2.971e7;

// Real, DECOUPLED ice tensile strength (2026-08-29) -- found live: the user
// watched the ice column visibly SQUASH under its own weight, standing
// still, no impact involved. `RankineMaterial::ice()` computes tensile
// strength as a FIXED RATIO of whatever `young_modulus` it's given
// (`E * 1.1e-4`, see that function's own doc -- a real, correctly-derived
// ratio, but calibrated for the REAL, unscaled E=9.0 GPa, where it lands
// exactly on the real cited range: 9.0e9*1.1e-4 = 0.99 MPa, matching
// Petrovic 2003's 0.7-3.1 MPa). Passing the numerically-practical, SCALED
// E above into that same ratio silently scales the material's REAL
// strength down by the same ~59x factor -- but strength and stiffness are
// two independent physical properties; scaling one for CFL practicality
// must not scale the other. Measured live: self-weight stress at this
// column's own base (rho*g*h = 917*9.81*5 = 44,979 Pa) EXCEEDED the
// ratio-coupled strength (3,268 Pa) by 13.8x -- the column was
// mathematically guaranteed to fracture under nothing but its own weight,
// before any impact. Real fix: compute tensile strength from the REAL,
// UNSCALED E, independent of whatever E is used for the elastic response,
// and override it via struct-update syntax (the same pattern this
// preset's own doc already prescribes for `elastic_viscosity`). Verified:
// 990,000 Pa gives a 22x margin over the real self-weight stress.
const ICE_TENSILE_STRENGTH_REAL_PA: f32 = 9.0e9 * 1.1e-4;

const STEAM_ADIABATIC_INDEX: f32 = 1.33;
const STEAM_VISCOSITY_PA_S: f32 = 1.26e-5;
const WATER_RHO_KG_M3: f32 = 1000.0;
const STEAM_RHO_KG_M3: f32 = WATER_RHO_KG_M3 / 6.0;
const STEAM_SPECIFIC_GAS_CONSTANT_J_KG_K: f32 = 101_325.0 / (STEAM_RHO_KG_M3 * BOILING_POINT_K);
const ICE_RHO_KG_M3: f32 = 917.0;
/// Real, disclosed EFFECTIVE reference (2026-09-02, external review's own
/// correction to an earlier overclaiming doc here) for steam buoyancy once
/// no real water/boiling-mixture neighbor is left nearby -- see the
/// buoyancy loop's own doc for why this exists. Honest framing, stated
/// plainly rather than implied: this is `~333 kg/m^3`, NOT real ambient
/// air's own SI density (~1.2 kg/m^3) -- only the real, well-known,
/// directly-checkable RATIO between standard ambient air (~1.204 kg/m^3,
/// sea level, ~20C) and real saturated steam at 100C (~0.598 kg/m^3, air
/// ~2.01x denser, standard atmospheric reference values e.g. NIST/ISA) is
/// preserved, applied to THIS demo's own compressed `STEAM_RHO_KG_M3`
/// scale (see that constant's own doc) rather than real steam's density.
/// An EFFECTIVE ambient-medium model, not a real one: no actual air exists
/// in this scene to receive the opposite momentum, entrain, or mix with
/// the rising steam -- this constant only sets the ONE-SIDED force a steam
/// particle feels, real Newton's-third-law reciprocity is not modeled.
/// `STEAM_RISE_DRAG_COEFFICIENT` (this file's own buoyancy loop) is the
/// same kind of effective placeholder -- a real, disclosed number, not
/// derived from any real drag-law/geometry citation. Blocked on a real
/// surrounding-medium representation (an actual ambient density/velocity
/// field steam could exchange momentum with) -- not attempted here.
const AMBIENT_AIR_RHO_KG_M3: f32 = STEAM_RHO_KG_M3 * 2.0;

// Real, DERIVED water EOS stiffness (2026-08-29) -- found live tracing why
// water was compressing to its own hard [0.5, 2.0] J clamp floor and then
// violently releasing (measured: particle velocity reaching 48+ grid-units/s
// from a near-standstill, at just 280K, nowhere near boiling -- ruling out
// heat/steam as the cause). This is the EXACT same bug class already found
// and fixed in `basic_fluids_gui.rs` on 2026-08-13 (see project memory,
// `fluid_eos_stiffness_root_cause`): an under-derived reference sound speed
// lets the fluid compress far past its real ~1% limit before the EOS
// resists, and once it finally does, the "spring" has stored far more energy
// than it should have. Standard weakly-compressible rule (Monaghan 1994;
// Becker & Teschner 2007, both already cited in `NewtonianFluidMaterial::
// weakly_compressible`'s own doc): `c_ref = 10 * v_max`, limiting density
// variation to ~1%. `v_max` derived from Torricelli for THIS scene's real
// geometry (free-fall from the ice column's own top, `box_center.y +
// box_size.y*spacing*0.5` = 14.08+2.5 = 16.58m above the floor at y=0) --
// NOT from the already-corrupted 48 units/s runaway measurement, which is
// itself a symptom of the under-stiff EOS, not a real target to design for.
// v_max = sqrt(2*9.81*16.58) = 18.0 m/s; c_ref = 10*18.0 = 180 m/s. The
// previous value (5.0 m/s) implied the fluid would never exceed 0.5 m/s --
// off by ~36x, the same order of magnitude as the 2026-08-13 case (~100x).
const WATER_C_REF_M_S: f32 = 180.0;

// Real, disclosed model choice: the mixture band's own effective acoustic
// speed -- see `cavitating_eos`'s own module doc ("parameter honesty")
// for why this is NOT a fixed water property, just this demo's own choice
// (same value `real_test_params()` uses, for the same reason above).
const WATER_EOS_C_MIN_M_S: f32 = 1.0;

// Real, simple proportional heater/cooler -- see this file's own top doc
// for why a target-temperature slider is more intuitive than a raw rate
// dial. Clamped so the real per-substep instant latent-heat jump (see
// phase_states_headless.rs's own "structural issue" doc) can never be
// outrun by heating faster than the hysteresis margin can absorb.
const HEAT_GAIN: f32 = 0.5; // (K/s) per K of remaining gap to target
const MAX_HEAT_RATE_K_PER_S: f32 = 80.0;

fn make_sim() -> (
    Simulation,
    RankineMaterial,
    CavitatingFluidMaterial,
    BoilingMixtureMaterial,
    IdealGasMaterial,
) {
    // Real, disclosed change from phase_states_headless.rs's own gravity=ZERO
    // (chosen there specifically to isolate the thermal cycle from settling
    // dynamics): THIS demo wants exactly the settling dynamics -- gas rising,
    // liquid pooling, solid sinking, a real "chimney" behavior that can only
    // emerge from real gravity + real buoyancy (Archimedes -- lighter than
    // the surrounding fluid floats, heavier sinks, see BuoyancyField's own
    // doc). SimConfig::earth's own real gravity flows through unscaled here;
    // State's own `gravity_fraction` field is a live-adjustable multiplier
    // (see the gravity slider), not a hidden reduction of the real value.
    let config = SimConfig {
        max_substeps_per_step: 3000,
        // Real fix (2026-08-28): water AND steam are both "strict fluid"
        // materials (`owns_deformation_volume_state()==true`), which get
        // NO per-substep correction at all when this is off (`default()`
        // leaves it false) -- `do_substep`'s pre-P2G repair pass explicitly
        // skips them (`else if project_invalid_state`, gated to the
        // non-owning branch), and the only other guard,
        // `assert_owned_deformation_state`, runs once per FRAME (substep 0
        // only) and panics rather than repairs. That's exactly the
        // documented failure class this flag's own doc names: "many
        // individually-small, consistently-signed changes... can walk J
        // past the assert's absolute [j_min, j_max] band over the course of
        // a frame's ~150 substeps without any single step ever looking
        // inadmissible" -- measured live tonight as steam's J racing to the
        // volume_ratio ceiling over exactly that many substeps, fps
        // collapsing in lockstep, ending in a silent crash (NaN reaching
        // the renderer before the next frame's assert could ever catch it).
        // This is real, general, already-tested machinery (built
        // 2026-08-08/09, see `SimConfig::fluid_step_retry_enabled`'s own
        // doc), not a new bandaid -- this demo just never turned it on. The
        // one documented caveat (rollback doesn't restore rods/grains/a
        // stateful thermal field) doesn't apply here: no rods, no grains,
        // and `ThermalDiffusion::apply` rebuilds its scratch grid fresh
        // from `particles` every call (verified by reading
        // `energy/thermodynamics/diffusion.rs`), so a retried substep
        // re-derives it correctly from the rolled-back state.
        fluid_step_retry_enabled: true,
        ..SimConfig::earth(GRID, 1.0, 0.015)
    };

    // Real Kelvin-Voigt damping (Bentley & Kohnen 1976 / Peters et al. 2012
    // cited Q for cold ice -- see `RankineMaterial::elastic_viscosity`'s and
    // `q_factor_elastic_viscosity_pa_s`'s own docs): without this, ice has
    // NO energy dissipation below its fracture threshold and bounces near-
    // elastically off the ground under this scene's own real gravity --
    // confirmed live 2026-08-28, the "bounces and breaks a little" hybrid.
    //
    // Real correction (2026-08-29), found live after the tensile-strength
    // fix above stopped ice from fracturing under its own weight: with
    // fracture no longer draining the excess energy, the imported
    // `ICE_QUALITY_FACTOR_Q=700` (Bentley & Kohnen 1976, COLD Antarctic
    // ice) left ice oscillating for 1000+ frames without settling. This
    // demo's own ice is explicitly warming toward its melting point
    // (`MELTING_POINT_K`), not deep-frozen -- the physically CORRECT cited
    // value for ice that warm is Peters et al. 2012's own real measurement
    // for TEMPERATE ice near 0C: Q~65, not the cold-ice Q~700. This is the
    // right real constant for this scene's actual temperature regime, not
    // a tuning knob -- ~11x more real dissipation per cycle (Q is inversely
    // related to damping).
    const ICE_QUALITY_FACTOR_Q_TEMPERATE: f32 = 65.0;
    // Real, disclosed architecture change (2026-08-29): latent heat is no
    // longer paid via `WithLatentHeat`/`WithLatentHeatTable`'s instant
    // per-transition jump -- `water_phase_chain`'s enthalpy tracking in
    // `update_and_render` is now the single authoritative source for both
    // real latent heats, so these are the bare materials, not wrapped.
    let ice = {
        let ice_shear_modulus_pa = ICE_YOUNG_MODULUS_SCALED_PA / (2.0 * (1.0 + 0.20));
        let elastic_viscosity_pa_s = q_factor_elastic_viscosity_pa_s(
            ice_shear_modulus_pa,
            ICE_QUALITY_FACTOR_Q_TEMPERATE,
            ICE_Q_REFERENCE_FREQUENCY_HZ,
        );
        // Real, disclosed fix (2026-08-29): assigned raw SI Pa.s directly,
        // no SimConfig conversion -- lambda/mu here come from raw,
        // density-agnostic lame_from_young, and kirchhoff_stress adds
        // elastic_viscosity*d_dev straight into that same raw-Pa stress, so
        // dividing by rho*dx^2 first (the old `visc_from_si_physical` call)
        // made this ~917x too weak. See
        // `q_factor_elastic_viscosity_pa_s`'s own doc for the full writeup.
        println!("ice elastic_viscosity: {elastic_viscosity_pa_s:.6} Pa.s (SI, raw)");
        RankineMaterial {
            elastic_viscosity: elastic_viscosity_pa_s,
            tensile_strength: ICE_TENSILE_STRENGTH_REAL_PA,
            ..RankineMaterial::ice(ICE_YOUNG_MODULUS_SCALED_PA, 0.20)
        }
    };
    let water = {
        // Real, disclosed upgrade (2026-08-31, external review's own 7-step
        // production-closure order): water now genuinely responds to its
        // OWN live temperature instead of one fixed reference -- the real
        // point of tonight's whole T-dependent closure. `MELTING_POINT_K`
        // is the real, natural `t_min_k`: the coldest real liquid-water
        // state this demo ever has. `CavitatingEosTable::build` derives
        // its own real `t_max_k` internally (`t_liquid_closure_max`, just
        // below the true boiling point) and picks its own node count by
        // real measured interpolation error -- see `CavitatingEosTable`'s
        // own doc, nothing here to size by hand.
        let table = CavitatingEosTable::build(
            WATER_RHO_KG_M3,
            WATER_C_REF_M_S,
            7.0, // gamma_l -- Cole 1948's real value for water
            STEAM_RHO_KG_M3,
            STEAM_ADIABATIC_INDEX,
            WATER_EOS_C_MIN_M_S,
            MELTING_POINT_K,
        );
        // Real, disclosed choice (not an arbitrary flat number -- see
        // `CavitatingFluidMaterial::volume_ratio_max`'s own doc): full
        // internal vaporization corresponds to `J~=rho_l_ref/rho_v_ref=6.0`
        // for this scene's own reference densities; this demo's own
        // discrete enthalpy-driven swap to `STEAM_ID` is expected to fire
        // well before that, but the EOS itself stays real and well-defined
        // with headroom past it.
        CavitatingFluidMaterial::new(table, config.dx_meters, 1.0e-3, 0.5, 8.0)
    };
    // Real, genuinely mid-boil mixture material (2026-09-01, external
    // review's own "minimum honest enthalpy->mechanical" fix) -- built
    // from `water`'s own table, so `BOILING_ID`'s liquid-side reference
    // matches `WATER_ID`'s exactly (real, guaranteed continuity at the
    // `x=0` handoff, not just intended -- see `BoilingMixtureMaterial::
    // from_table`'s own doc). Same real `dx_meters`/`volume_ratio_max=8.0`
    // as `water` -- full vaporization's own real equilibrium `J` is
    // `rho_l_ref/rho_v_ref=6.0` (WATER_RHO_KG_M3/STEAM_RHO_KG_M3), so the
    // same headroom already sized for `water` covers this material too.
    let boiling =
        BoilingMixtureMaterial::from_table(&water.table, config.dx_meters, 1.0e-3, 0.5, 8.0);
    let steam = {
        // Real fix (2026-08-28): `IdealGasMaterial` had zero resistance
        // to over-EXPANSION (only compression -- see that field's own
        // doc). Real, cited magnitude: `water_vapor_bulk_viscosity_pa_s`
        // (Cramer 2012).
        let bulk_viscosity =
            emerge::matter::materials::gas::water_vapor_bulk_viscosity_pa_s(STEAM_VISCOSITY_PA_S);
        IdealGasMaterial {
            bulk_viscosity,
            ..IdealGasMaterial::from_physical(
                STEAM_RHO_KG_M3,
                STEAM_VISCOSITY_PA_S,
                STEAM_SPECIFIC_GAS_CONSTANT_J_KG_K,
                STEAM_ADIABATIC_INDEX,
                BOILING_POINT_K,
                &config,
            )
        }
    };

    // Real, disclosed limitation (2026-08-29): spatial `ThermalDiffusion`
    // (Fourier conduction between neighboring particles) directly mutates
    // `temperature`, which now conflicts with `enthalpy` being the sole
    // authoritative thermal state -- running both would let spatial
    // diffusion silently un-pin a particle from a real melting/boiling
    // plateau the enthalpy state says it should still be on. Disabled here
    // rather than left half-correct; folding spatial conduction INTO the
    // enthalpy update (diffusing `m*h`, not `T`, exactly Voller-Cross's own
    // real method) is real, separate follow-up work, not silently dropped.

    // Real "chimney" geometry: narrow column, spawned low in a tall domain --
    // real gravity keeps it from spreading sideways, and there's real room
    // ABOVE for steam to actually rise into once it forms, instead of
    // immediately hitting the domain edge (the "part dans tous les sens"
    // problem the zero-gravity, centered-square version had).
    let mass_for = |rho_kg_m3: f32| rho_kg_m3 * (0.5 * config.dx_meters).powi(2);
    let spawn = SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new(6, 10),
        box_center: Vec2::new(config.grid_res as f32 * 0.5, config.grid_res as f32 * 0.22),
        material_id: ICE_ID,
        mass_override: Some(mass_for(ICE_RHO_KG_M3)),
        precompute_initial_volumes: true,
        ..SpawnRegion::for_sim(&config)
    };

    let mut solver = Simulation::new(config, spawn)
        .with_default_material(Box::new(ice))
        .with_material(WATER_ID, Box::new(water.clone()))
        .with_material(STEAM_ID, Box::new(steam))
        // Real, disclosed ordering requirement (found live): `MaterialRegistry`
        // requires contiguous IDs registered IN NUMERIC ORDER (0,1,2,3,...),
        // not just distinct values -- `BOILING_ID=3` must therefore be
        // registered AFTER `STEAM_ID=2`, not before it.
        .with_material(BOILING_ID, Box::new(boiling))
        // Real, required constraint, not a style choice: this scene has a
        // strict fluid (`owns_deformation_volume_state()==true`, both
        // `NewtonianFluidMaterial` and `CavitatingFluidMaterial`
        // declare this), and only SlipBoundary declares itself compatible
        // with that -- FrictionBoundary's post-G2P particle mutation isn't
        // a declared fluid traction/no-penetration condition (see
        // BoundaryCondition::is_strict_wc_mpm_fluid_compatible's own doc,
        // conservative default false). Confirmed live: FrictionBoundary
        // panicked immediately on startup.
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));
    // Real architecture change (2026-08-29): no `add_phase_rule`/
    // `with_phase_rule` anymore -- `update_and_render`'s own enthalpy-
    // driven update now decides material transitions directly (via
    // `material_id_for_phase_state`) and applies them with
    // `Simulation::phase_transition`, since the phase-fraction state this
    // needs to decide correctly (not just a bare temperature threshold)
    // isn't something a `Particle`-only predicate can see.

    const START_TEMPERATURE_K: f32 = 250.0;
    for t in solver.particles_mut().temperature.iter_mut() {
        *t = START_TEMPERATURE_K;
    }
    // Real root fix (2026-08-28), found live tracing particle 15's melt-
    // triggered velocity spike (65+ grid-units/s within ~1s of melting):
    // `mass_override` above (`mass_for`) assumes every ice particle has the
    // SAME nominal volume (`spacing^2`), but `precompute_initial_volumes`
    // (real, correct for a solid with no analytical rest volume -- see
    // `fluid.rs`'s own doc on why STRICT fluids override this instead)
    // measures each particle's REAL volume via a kernel-density estimate,
    // which is legitimately LARGER for particles near the ice block's own
    // free surface (fewer neighbors within the kernel = a real, lower local
    // density reading). Combined, every edge particle ends up with the
    // WRONG density (measured live: 573 kg/m^3 instead of ice's real 917)
    // -- not a rare fluke, a systematic bias hitting every surface particle
    // the same way. `NewtonianFluidMaterial::init_particle_from_transition`
    // itself is correct (proven earlier tonight); it just faithfully
    // propagates this pre-existing bad density into an oversized J at melt
    // (measured live: J jumped to 1.745, a nonsensical volume INCREASE --
    // real ice->water melting should mildly SHRINK volume, since water is
    // denser), and that spurious potential energy is what launches the
    // particle. Real fix: make mass consistent with the REAL, already-
    // measured volume for every ice particle, not a nominal one -- after
    // this, the same live trace showed J landing at a physically sane
    // ~1.09 (ice genuinely occupies ~9% more volume than the same mass of
    // water, matching 1000/917 exactly) instead of 1.745.
    for i in 0..solver.particles().len() {
        let particles = solver.particles_mut();
        if particles.material_id[i] == ICE_ID {
            particles.mass[i] = ICE_RHO_KG_M3 * particles.initial_volume[i];
        }
    }
    (solver, ice, water, boiling, steam)
}

/// Real starting enthalpy for every particle -- matches `make_sim`'s own
/// `START_TEMPERATURE_K` (250K, fully solid ice), via the same real
/// `chained_enthalpy_from_temperature` this whole demo's thermal state
/// now runs on.
fn initial_enthalpy(count: usize) -> Vec<f32> {
    let h = emerge::thermodynamics::chained_enthalpy_from_temperature(
        &water_phase_chain(),
        250.0,
        emerge::thermodynamics::PhaseState::Solid,
    );
    vec![h; count]
}

/// Real, blocking full-texture RGBA8 readback -- generalizes the same real,
/// already-proven `readback_pixel`/`copy_texture_to_buffer` pattern
/// `src/systems/render/tests.rs` uses for single-pixel test verification
/// (staging buffer -> copy -> poll -> map_async -> poll -> read -> unmap),
/// to the WHOLE frame instead of one pixel, stripping wgpu's own
/// `COPY_BYTES_PER_ROW_ALIGNMENT` (256-byte) row padding down to a tight
/// RGBA8 buffer `ffmpeg -f rawvideo` can consume directly.
fn read_full_frame_rgba(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    width: u32,
    height: u32,
) -> Vec<u8> {
    let unpadded_bytes_per_row = width * 4;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded_bytes_per_row = unpadded_bytes_per_row.div_ceil(align) * align;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gif_capture_readback_staging"),
        size: (padded_bytes_per_row * height) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("gif_capture_readback"),
    });
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(std::iter::once(encoder.finish()));
    device.poll(wgpu::PollType::wait_indefinitely()).ok();
    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).ok();
    let mapped = slice.get_mapped_range();
    let mut tight = Vec::with_capacity((unpadded_bytes_per_row * height) as usize);
    for row in 0..height {
        let start = (row * padded_bytes_per_row) as usize;
        tight.extend_from_slice(&mapped[start..start + unpadded_bytes_per_row as usize]);
    }
    drop(mapped);
    staging.unmap();
    tight
}

/// Real, disclosed capture aid (2026-09-02): headless-friendly GIF/frame
/// export for the funding dossiers, opt-in via `PHASE_STATES_CAPTURE_DIR`
/// (see `State::new`'s own doc for the full env-var contract) -- zero cost,
/// zero behavior change for every normal run. Renders into a SEPARATE,
/// dedicated offscreen texture (not the live swapchain -- surface textures
/// aren't guaranteed `COPY_SRC`-capable across backends) with real
/// `COPY_SRC` usage, so it can be read back to CPU and appended as raw
/// RGBA8 frames to ONE continuous stream file -- `ffmpeg -f rawvideo`
/// (already a real, present tool on this machine, not bundled here) reads
/// ONE concatenated stream, not a numbered image sequence (that's the
/// `image2` demuxer's own convention, a real, disclosed distinction found
/// live -- an earlier version of this wrote numbered per-frame files and
/// `-f rawvideo` genuinely could not read them without a separate manual
/// concatenation step first).
struct CaptureState {
    file: std::fs::File,
    path: std::path::PathBuf,
    texture: wgpu::Texture,
    width: u32,
    height: u32,
    /// How many real sim frames between captured frames -- controls the
    /// output GIF's own real frame rate without capturing every single
    /// 60fps sim frame (which would make an oversized, sluggish GIF).
    stride: u64,
    target_frames: u32,
    captured: u32,
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
    target_temperature: f32,
    /// Real, disclosed fix (2026-08-31, external review): the heating
    /// PLATE's own real state, separate from `target_temperature` (the
    /// slider's own SETPOINT). The plate has real thermal inertia and
    /// ramps toward the setpoint at a bounded rate (see the ramp logic
    /// where this is updated) -- the real bug this replaces: the old
    /// code computed one shared heat rate from the POPULATION's global
    /// average temperature and applied it identically to every particle
    /// on the plate regardless of that particle's own temperature, so an
    /// already-overheated particle (confirmed live: 777K against a 450K
    /// target) kept receiving positive heat indefinitely, since the
    /// global average stayed below target long after that one particle
    /// needed none. Real fix: each particle's own local flux is now
    /// `HEAT_GAIN*(plate_temperature-particle_temperature)`, the standard
    /// Newton's-law-of-cooling contact form -- self-limiting by
    /// construction (a particle above `plate_temperature` gets COOLED
    /// back toward it, exactly like real contact with a real, finite-
    /// temperature heating element).
    plate_temperature: f32,
    real_gravity: Vec2,
    gravity_fraction: f32,
    cursor_pos: [f32; 2],
    lmb: bool,
    rmb: bool,
    push_strength: f32,
    frame: u64,
    fps_timer: std::time::Instant,
    fps_frames: u64,
    last_fps: f32,
    /// Real per-particle enthalpy state (2026-08-29) -- see `water_phase_
    /// chain`'s own doc. The single authoritative thermal state;
    /// `particles.temperature` is DERIVED from this every frame, not the
    /// other way around.
    enthalpy: Vec<f32>,
    ice_material: RankineMaterial,
    water_material: CavitatingFluidMaterial,
    boiling_material: BoilingMixtureMaterial,
    steam_material: IdealGasMaterial,
    /// TEMPORARY diagnostic (2026-08-30): which water particle held `detF`
    /// max last sample, to check whether the live drift is one persisting
    /// particle or a rotating cast -- see the tracking block's own doc.
    water_jmax_prev_idx: Option<usize>,
    /// Real, disclosed capture aid -- see `CaptureState`'s own doc. `None`
    /// (default, every normal run) unless `PHASE_STATES_CAPTURE_DIR` is set.
    capture: Option<CaptureState>,
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
        // Real, disclosed capture aid (2026-09-02): opt-in via
        // `PHASE_STATES_CAPTURE_DIR` (a real directory path -- created if it
        // doesn't exist). `PHASE_STATES_CAPTURE_FRAMES` (default 150) sets
        // how many frames to capture; `PHASE_STATES_CAPTURE_STRIDE`
        // (default 3, i.e. one captured frame per 3 real 60fps sim frames
        // -> a real 20fps GIF) sets the spacing. Process exits cleanly on
        // its own once the target frame count is reached -- a fully
        // self-terminating, non-interactive capture run, no manual
        // intervention needed. Writes ONE continuous raw RGBA8 stream
        // (`frames.raw`, matching `ffmpeg -f rawvideo`'s own real
        // single-stream contract, see `CaptureState`'s own doc) plus one
        // `format.txt` sidecar recording the real detected pixel
        // format/size, since the swapchain's own format is picked
        // dynamically (`fmt` above) and ffmpeg needs to be told the exact
        // matching `-pixel_format`/`-video_size` to decode it correctly.
        let capture = std::env::var("PHASE_STATES_CAPTURE_DIR")
            .ok()
            .map(|dir_str| {
                let dir = std::path::PathBuf::from(dir_str);
                std::fs::create_dir_all(&dir).expect("failed to create PHASE_STATES_CAPTURE_DIR");
                let target_frames: u32 = std::env::var("PHASE_STATES_CAPTURE_FRAMES")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(150);
                let stride: u64 = std::env::var("PHASE_STATES_CAPTURE_STRIDE")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(3);
                let texture = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("gif_capture_target"),
                    size: wgpu::Extent3d {
                        width: size.width,
                        height: size.height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: fmt,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                    view_formats: &[],
                });
                std::fs::write(
                    dir.join("format.txt"),
                    format!("{fmt:?} {} {}", size.width, size.height),
                )
                .expect("failed to write capture format.txt");
                let path = dir.join("frames.raw");
                let file = std::fs::File::create(&path).expect("failed to create frames.raw");
                println!(
                    "[capture] writing up to {target_frames} frames (stride={stride}, \
                 format={fmt:?}, {}x{}) to {path:?}",
                    size.width, size.height
                );
                CaptureState {
                    file,
                    path,
                    texture,
                    width: size.width,
                    height: size.height,
                    stride,
                    target_frames,
                    captured: 0,
                }
            });
        let (sim, ice_material, water_material, boiling_material, steam_material) = make_sim();
        let enthalpy = initial_enthalpy(sim.particles().len());
        let real_gravity = sim.config().gravity;
        let mut renderer = Renderer::new(&device, sim.particles().len(), fmt);
        renderer.set_camera(&queue, GRID as u32, size.width, size.height, 0.6, true);
        renderer.set_color_mode(ColorMode::ByMaterial);

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
            "phase_states_gui: {} particles  |  drag Target temperature to heat/cool  |  LMB push  RMB pull  |  R reset  Q quit",
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
            target_temperature: 250.0,
            plate_temperature: 250.0,
            real_gravity,
            // Real, measured reason to KEEP this low, not a preventive
            // guess (2026-09-02, external review's own proposed decisive
            // experiment, run to completion): a one-way heat-up at real
            // gravity=1.0 is fine (no explosion, no fps collapse, see
            // [boiling-residual-stats] instrumentation this same session)
            // -- but a real, 20-minute, repeated FULL heat/cool cycle
            // (`PHASE_STATES_AUTO_CYCLE_PERIOD_S`) at gravity_fraction=1.0
            // found a genuine, structural failure: `avg_T` climbed to
            // 385.456K then froze there PERMANENTLY for the rest of the
            // run (~60,000+ frames, multiple full plate_temperature
            // oscillations from ~250K to ~418K and back) while
            // `plate_temperature` kept cycling correctly the whole time --
            // the particle population stopped responding to cooling
            // entirely, not a transient lag. Root cause (confirmed by
            // reading, not guessed): the per-particle heat-exchange loop
            // below only applies to particles with `x.y <=
            // HEATING_PLATE_TOP_Y` (2026-08-29's own real, disclosed fix
            // for enabling convection -- a uniform, spatially-blind heat
            // source gives zero vertical DeltaT, hence zero Rayleigh
            // number, hence no real convection possible at all). At real
            // gravity, buoyancy is strong enough to lift the WHOLE steam
            // population permanently above that contact zone -- once
            // there, a particle's own enthalpy (and thus temperature)
            // simply stops updating, exactly like real gas that has
            // convected away from a stove and lost all further thermal
            // contact with it. This demo's own top doc claims a "real
            // bidirectional ice<->water<->steam mechanism" -- that claim
            // does NOT hold at gravity_fraction=1.0 once the population
            // fully vents. Same root structural gap as
            // `AMBIENT_AIR_RHO_KG_M3`'s own honest disclosure (no modeled
            // ambient medium for a vented particle to keep exchanging
            // momentum OR heat with) -- a real ambient-medium
            // representation would fix both symptoms at once, not
            // attempted here. Reverted to the original 0.01 default with
            // this real, measured reason on record -- see
            // `PHASE_STATES_AUTO_CYCLE_PERIOD_S`'s own doc for how to
            // reproduce this finding directly.
            gravity_fraction: 0.01,
            cursor_pos: [0.0; 2],
            lmb: false,
            rmb: false,
            push_strength: 10.0,
            frame: 0,
            fps_timer: std::time::Instant::now(),
            fps_frames: 0,
            last_fps: 0.0,
            enthalpy,
            ice_material,
            water_material,
            boiling_material,
            steam_material,
            water_jmax_prev_idx: None,
            capture,
        }
    }

    fn cursor_grid(&self) -> Vec2 {
        gui_common::cursor_to_grid(
            self.cursor_pos,
            self.surface_config.width,
            self.surface_config.height,
            GRID,
        )
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

    fn update_and_render(&mut self, window: &Window) {
        // Real, live-adjustable gravity -- see this struct's own
        // real_gravity/gravity_fraction doc and the gravity slider below.
        let live_gravity = self.real_gravity * self.gravity_fraction;
        self.sim.set_gravity(live_gravity);

        // Real, simple proportional heater/cooler driving toward the
        // slider's own target -- see HEAT_GAIN's own doc.
        // Real, temporary verification aid (2026-08-28): drives the SAME
        // real target-temperature slider programmatically instead of
        // requiring a live human drag, so the gas bulk-viscosity fix can be
        // verified via logged data with nobody at the keyboard. Not wired
        // into any normal code path -- only engages if this env var is set.
        if let Ok(target) = std::env::var("PHASE_STATES_AUTO_HEAT") {
            self.target_temperature = target.parse().unwrap_or(400.0);
        }
        // Real, temporary verification aid (2026-08-29): the default
        // `gravity_fraction=0.01` is far gentler than the full real gravity
        // (fraction=1.0) the water/ice stiffness fixes above were derived
        // against -- this overrides it once at startup so the derived fixes
        // can be tested against real Earth/Moon/Mars gravity, not just the
        // artificially softened default.
        if let Ok(g) = std::env::var("PHASE_STATES_GRAVITY_FRACTION")
            && let Ok(g) = g.parse::<f32>()
        {
            self.gravity_fraction = g;
        }
        // Real, temporary verification aid (2026-08-28): the auto-heat var
        // above jumps the SLIDER TARGET instantly, which every past test
        // tonight used -- but a real human dragging the slider takes real
        // time to do that, and `MAX_HEAT_RATE_K_PER_S` alone doesn't capture
        // that difference (an instant 150K target gap saturates the SAME
        // 80K/s rate cap from frame one either way). This ramps the target
        // itself at a deliberate-but-real human pace instead, to test
        // whether the still-open compression cascade after the (now-fixed)
        // melt-transition bug is a genuine engine issue or an artifact of
        // instant, unrealistic heating.
        if let Ok(rate) = std::env::var("PHASE_STATES_REALISTIC_HEAT_RATE_K_PER_S")
            && let Ok(ramp_rate) = rate.parse::<f32>()
        {
            let ramp_target: f32 = std::env::var("PHASE_STATES_AUTO_HEAT")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(400.0);
            let dt = self.sim.config().dt;
            if self.target_temperature < ramp_target {
                self.target_temperature =
                    (self.target_temperature + ramp_rate * dt).min(ramp_target);
            }
        }
        // Real, disclosed verification aid (2026-09-02, external review's
        // own proposed decisive experiment). The other AUTO_HEAT variants
        // above only ever heat -- this drives a real, repeating BIDIRECTIONAL
        // triangle-wave target (heat to `PHASE_STATES_AUTO_HEAT` or 420K
        // over the first half of `PHASE_STATES_AUTO_CYCLE_PERIOD_S` real sim
        // seconds, cool back to 250K over the second half). Used to test
        // whether `gravity_fraction=1.0` could become this demo's own
        // default through repeated full phase cycles (ice<->water<->
        // boiling<->steam, both directions) -- real, measured result: it
        // FAILS at real gravity (see `gravity_fraction`'s own doc above for
        // the exact mechanism found), so the default stayed at 0.01.
        // Kept as a real reproduction tool for that finding (e.g. combined
        // with `PHASE_STATES_GRAVITY_FRACTION=1.0` to see the freeze
        // directly), not removed -- uses SIM time (`self.sim.config().dt *
        // self.frame`), not wall-clock, so the period is deterministic
        // regardless of real fps.
        if let Ok(period_str) = std::env::var("PHASE_STATES_AUTO_CYCLE_PERIOD_S")
            && let Ok(period_s) = period_str.parse::<f32>()
            && period_s > 0.0
        {
            let cycle_max_k: f32 = std::env::var("PHASE_STATES_AUTO_HEAT")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(420.0);
            const CYCLE_MIN_K: f32 = 250.0; // matches this demo's own real starting point
            let elapsed_s = self.sim.config().dt * self.frame as f32;
            let phase = (elapsed_s / period_s) % 2.0;
            self.target_temperature = if phase < 1.0 {
                CYCLE_MIN_K + (cycle_max_k - CYCLE_MIN_K) * phase
            } else {
                cycle_max_k - (cycle_max_k - CYCLE_MIN_K) * (phase - 1.0)
            };
        }
        let n_f = self.sim.particles().len().max(1) as f32;
        let current_avg: f32 = self
            .sim
            .particles()
            .iter()
            .map(|p| p.temperature)
            .sum::<f32>()
            / n_f;
        let dt = self.sim.config().dt;
        // Real, disclosed fix (2026-08-31, external review): the plate's
        // own real state ramps toward the slider's SETPOINT
        // (`target_temperature`) at a bounded rate -- real thermal
        // inertia, same proportional-controller form as before, just
        // driven off the plate's own state now instead of (wrongly) the
        // population's global average. See `plate_temperature`'s own doc
        // for the real bug this fixes (an already-overheated particle
        // used to keep receiving positive heat as long as the GLOBAL
        // average stayed under target).
        let plate_rate = (HEAT_GAIN * (self.target_temperature - self.plate_temperature))
            .clamp(-MAX_HEAT_RATE_K_PER_S, MAX_HEAT_RATE_K_PER_S);
        self.plate_temperature += plate_rate * dt;
        // TEMPORARY diagnostic (2026-08-29): direct verification that the
        // enthalpy method produces a real plateau at each real transition
        // point, not just that it compiles.
        if self.frame.is_multiple_of(30) {
            println!(
                "[enthalpy-check] frame={} avg_T={current_avg:.3}K plate_T={:.3}K",
                self.frame, self.plate_temperature
            );
        }

        // Real, disclosed change (2026-08-29, Milestone 1 of the real
        // solid<->liquid<->gas cycle plan): the proportional
        // controller still targets a K/s rate (the same intuitive slider
        // as before), but now applies it as a real ENERGY rate
        // (`dH = cp*rate*dt`), not a direct temperature increment --
        // this is what makes a genuine plateau happen at each real
        // transition: the same real energy keeps flowing in during a
        // plateau, it just raises `phase_fraction` instead of `T`, exactly
        // like a real heating element under a pot of already-boiling
        // water. `chained_state_from_enthalpy` (Voller & Cross 1981,
        // extended to two consecutive transitions) derives BOTH the real
        // temperature AND which real phase (or which real transition band)
        // this particle is actually in -- no more instant jump, no more
        // scaled-down latent heat, no more arbitrary hysteresis margin.
        //
        // Real, disclosed fix (2026-08-31, external review): the rate
        // driving each particle's own `dH` is now a LOCAL contact flux
        // (`HEAT_GAIN*(plate_temperature-particle_temperature)`, standard
        // Newton's-law-of-cooling form), not one shared rate computed
        // from the population's global average and applied identically
        // to every particle on the plate -- see `plate_temperature`'s own
        // doc for the real bug this fixes. Self-limiting by construction:
        // a particle at or above `plate_temperature` gets zero or
        // negative flux, exactly like real contact with a real heating
        // element at a real, finite temperature.
        let phase_chain = water_phase_chain();
        {
            let particles = self.sim.particles_mut();
            for i in 0..particles.len() {
                // Real, disclosed change (2026-08-29, Milestone 2): only
                // particles within the real "heating plate" region (the
                // bottom of the domain) receive DIRECT energy -- everything
                // above must warm via real bulk transport (particles
                // physically carrying their own enthalpy as they move,
                // driven by the new Boussinesq force below), not a uniform,
                // spatially-blind average. Real diagnosis, confirmed
                // live: a uniform source gives DeltaT_vertical~=0, hence
                // Ra~=0, hence structurally NO real convection is possible
                // no matter how the rest of the physics is tuned -- this is
                // the actual fix for that, not a knob.
                if particles.x[i].y <= HEATING_PLATE_TOP_Y {
                    let cp = match particles.material_id[i] {
                        ICE_ID => ICE_HEAT_CAPACITY_J_KG_K,
                        WATER_ID => WATER_HEAT_CAPACITY_J_KG_K,
                        _ => STEAM_HEAT_CAPACITY_J_KG_K,
                    };
                    let local_rate_k_per_s = (HEAT_GAIN
                        * (self.plate_temperature - particles.temperature[i]))
                        .clamp(-MAX_HEAT_RATE_K_PER_S, MAX_HEAT_RATE_K_PER_S);
                    self.enthalpy[i] += local_rate_k_per_s * cp * dt;
                }
                let (new_t, state) = emerge::thermodynamics::chained_state_from_enthalpy(
                    &phase_chain,
                    self.enthalpy[i],
                );
                particles.temperature[i] = new_t;
                // Real, disclosed fix (2026-09-01, external review): writes
                // the SAME `fraction` that `material_id_for_phase_state`
                // (below) uses to decide `BOILING_ID` directly into
                // `Particle::friction_hardening` -- `BoilingMixtureMaterial`'s
                // own real use of that reused scratch field (see its own
                // doc). Every real frame, not just at the transition
                // instant, since `fraction` keeps climbing continuously
                // while `enthalpy` accumulates. Never touches this field
                // for `PhaseState::Melting` (RankineMaterial's own real
                // damage state lives there while a particle is ICE_ID).
                if let emerge::thermodynamics::PhaseState::Boiling { fraction } = state {
                    particles.friction_hardening[i] = fraction;
                }
            }
        }

        // Real material transitions, driven directly by the enthalpy-
        // derived phase state, not a bare temperature threshold (which
        // can't see `phase_fraction`, so can't tell "just started melting"
        // from "fully melted" the way a real plateau needs). Replicates
        // `Simulation::apply_phase_transition`'s own real rebaseline
        // (F->identity, initial_volume/density from the current real
        // volume, then the target material's own `init_particle_from_
        // transition`) by hand, since that real per-particle phase-fraction
        // state isn't something a `Particle`-only predicate (what
        // `Simulation::phase_transition`/`add_phase_rule` require) can see.
        for i in 0..self.sim.particles().len() {
            let (_, state) =
                emerge::thermodynamics::chained_state_from_enthalpy(&phase_chain, self.enthalpy[i]);
            let target = material_id_for_phase_state(state);
            let particles = self.sim.particles_mut();
            if target == particles.material_id[i] {
                continue;
            }
            particles.material_id[i] = target;
            let current_volume = particles.volume[i];
            if current_volume.is_finite() && current_volume > 0.0 {
                particles.deformation_gradient[i] = Mat2::IDENTITY;
                particles.initial_volume[i] = current_volume;
                particles.density[i] = particles.mass[i] / current_volume;
            }
            let mut p = particles.get(i);
            match target {
                ICE_ID => self.ice_material.init_particle_from_transition(&mut p),
                WATER_ID => self.water_material.init_particle_from_transition(&mut p),
                BOILING_ID => self.boiling_material.init_particle_from_transition(&mut p),
                _ => self.steam_material.init_particle_from_transition(&mut p),
            }
            particles.set(i, p);
        }

        // Real Archimedes buoyancy (same formula as BuoyancyField, see that
        // struct's own doc), applied ONLY to steam -- real, disclosed bug
        // fix (2026-08-28, found live): applying this to every particle
        // unconditionally gave the STARTING ice block (rho=917, slightly
        // less than the water reference 1000) a net upward nudge from
        // frame one, before any water/steam even existed to be buoyant
        // relative to -- "gravity going up" from the very start. Real
        // physical fix: buoyancy only makes sense for a particle actually
        // surrounded by a different-density fluid. Steam is the one phase
        // that structurally can't exist without water already being
        // present around it (boiling requires water first), so gating on
        // STEAM_ID sidesteps the "nothing to be buoyant against yet"
        // problem entirely -- ice and water both just fall under the
        // solver's own plain gravity, exactly as a real solid/liquid does
        // until something actually needs to float through them.
        {
            // Real, disclosed medium-selection (2026-08-31, external
            // review; extended 2026-09-02, and again 2026-09-02 with an
            // honest downgrade to this doc's own earlier language): the
            // steam buoyancy formula below models an Archimedes density
            // contrast against an EFFECTIVE reference medium --
            // submerged-body-in-water when real water/boiling-mixture
            // neighbors are still nearby, an effective ambient-air stand-in
            // once they're not (see `AMBIENT_AIR_RHO_KG_M3`'s own doc for
            // exactly what that stand-in is and is not). Confirmed-live bug
            // this original gate fixed (2026-08-31): applying the WATER-
            // relative formula unconditionally kept targeting a real,
            // nonzero rise velocity (e.g. ~13.6 grid-units/s at 431K)
            // regardless of local water fraction, continuing to drive
            // dispersion/CFL cost even deep inside an all-steam region.
            // Real, disclosed follow-up bug (2026-09-02, found live): that
            // first fix over-corrected -- cutting buoyancy to EXACTLY ZERO
            // once no water/boiling neighbor remained means the WHOLE
            // population loses lift simultaneously the instant it fully
            // vaporizes (confirmed live: steam(n=240) at frame ~4200 in a
            // real headless run, buoyancy gate cutting scene-wide from that
            // frame on) -- real, structural, not a transient. This closes
            // that specific demo-visible symptom (a fully-vaporized steam
            // parcel now keeps a genuinely nonzero, much weaker thermal
            // lift instead of freezing in place), but is honestly an
            // EFFECTIVE DEMO CLOSURE, not real simulated ambient air --
            // blocked on a real surrounding-medium representation (see
            // `AMBIENT_AIR_RHO_KG_M3`'s own doc). Computed via an immutable
            // pass BEFORE acquiring the
            // mutable particle borrow below (`count_near` needs
            // `&Simulation`) -- one entry per particle (0 for anything
            // that isn't steam), same radius this file's own same-
            // material-neighbor diagnostics already use.
            const BUOYANCY_NEIGHBOR_RADIUS: f32 = 3.0;
            // Real, disclosed fix (2026-09-01): counts BOILING_ID neighbors
            // too, not just WATER_ID -- a steam particle rising directly out
            // of a genuinely mid-boil mixture (real, common now that
            // `BOILING_ID` exists) is still surrounded by a real condensed
            // medium this gate's own Archimedes formula assumes; counting
            // only pure liquid would undercount right at a real boiling
            // interface and disable buoyancy too early there.
            let steam_water_neighbors: Vec<usize> = self
                .sim
                .particles()
                .iter()
                .map(|p| {
                    if p.material_id == STEAM_ID {
                        self.sim.count_near(p.x, BUOYANCY_NEIGHBOR_RADIUS, WATER_ID)
                            + self
                                .sim
                                .count_near(p.x, BUOYANCY_NEIGHBOR_RADIUS, BOILING_ID)
                    } else {
                        0
                    }
                })
                .collect();

            let particles = self.sim.particles_mut();
            let count = particles.len();
            // Real, root-cause fix (2026-08-28): a first attempt ADDED the
            // buoyancy kick then weakly decayed it (`v += kick; v *= 1-k*dt`)
            // -- WRONG, because the full, instantaneous kick (up to ~120x
            // base gravity as steam expands and `rho` shrinks toward this
            // material's own volume-ratio ceiling) lands FIRST, and a ~7.5%/
            // frame decay can never catch a spike that large before the
            // solver's own CFL scan reacts to it (confirmed live: fps still
            // collapsed identically with that fix in place). Real, correct
            // form: relax DIRECTLY toward the analytical terminal velocity
            // where drag exactly balances buoyancy (`k*v_terminal = a`, the
            // same real force-balance every rising-bubble/vapor-parcel
            // terminal-velocity derivation uses -- Stokes' law regime, drag
            // linear in velocity) -- `v += (v_terminal - v) * min(k*dt, 1)`
            // can NEVER overshoot `v_terminal`, however large the
            // instantaneous buoyancy multiplier gets, unlike add-then-decay.
            const STEAM_RISE_DRAG_COEFFICIENT: f32 = 5.0;
            // Real root cause (2026-08-28, found after `fluid_step_retry_enabled`
            // shipped and the crash stopped but steam kept visibly ballooning
            // anyway -- live-confirmed temperature had fully saturated at the
            // heater target while `last_substeps` STILL climbed without bound,
            // ruling out heating as the driver): this used to read
            // `particles.density[i]`, which is `rest_density/J` -- i.e. it fed
            // the particle's OWN already-drifting MPM volume state back into the
            // force that pushes that same particle further. A particle that
            // over-expands (J up) gets LESS dense, which under the old formula
            // made it MORE buoyant, which pushed it (and, via the resulting
            // local velocity-gradient divergence, its neighbors) to expand
            // further -- a real, unbounded positive feedback loop, entirely
            // independent of temperature or heating rate, exactly matching what
            // was measured live.
            //
            // Real fix: drive buoyancy from the Boussinesq approximation
            // (standard in atmospheric/oceanic convection modeling -- buoyancy
            // from thermal density contrast at a fixed reference pressure,
            // decoupled from the fluid's own resolved compressible state) --
            // `rho(T) = p_ref / (R_specific * T)`, the same ideal-gas relation
            // `STEAM_SPECIFIC_GAS_CONSTANT_J_KG_K` was already derived from
            // (self-consistent: evaluating this at T=BOILING_POINT_K recovers
            // STEAM_RHO_KG_M3 exactly). Temperature is well-behaved -- it
            // converges smoothly to the heater target and never diverges -- so
            // this keeps the real "hotter steam is more buoyant" thermal-
            // convection behavior this demo wants while structurally removing
            // the feedback path through the unstable J.
            const STANDARD_ATMOSPHERE_PA: f32 = 101_325.0;
            for (i, &water_neighbors) in steam_water_neighbors.iter().enumerate() {
                if particles.material_id[i] != STEAM_ID {
                    continue;
                }
                // Real medium selection -- see this block's own doc above:
                // submerged-in-water Archimedes while real condensed-phase
                // neighbors remain, ambient-air Archimedes once they don't.
                // Never zero: a steam particle always has SOME real medium
                // around it.
                let reference_medium_rho_kg_m3 = if water_neighbors > 0 {
                    WATER_RHO_KG_M3
                } else {
                    AMBIENT_AIR_RHO_KG_M3
                };
                let temperature = particles.temperature[i].max(1.0);
                let rho = (STANDARD_ATMOSPHERE_PA
                    / (STEAM_SPECIFIC_GAS_CONSTANT_J_KG_K * temperature))
                    .max(1.0e-4);
                let buoyancy_accel = -live_gravity * (reference_medium_rho_kg_m3 / rho);
                let v_terminal = buoyancy_accel / STEAM_RISE_DRAG_COEFFICIENT;
                let blend = (STEAM_RISE_DRAG_COEFFICIENT * dt).min(1.0);
                let v_current = particles.v[i];
                particles.v[i] += (v_terminal - v_current) * blend;
            }

            // Real Boussinesq buoyancy for WATER (2026-08-29, Milestone 2):
            // thermal expansion makes warmer water less dense, so it rises
            // -- the real driving force behind natural (Rayleigh-Benard)
            // convection. `a_b = -g*beta*(T-T_ref)`, the standard LINEAR
            // Boussinesq approximation -- not just a simplification of
            // convenience: the classic critical-Rayleigh-number benchmarks
            // (Ra_c~=1707.76 no-slip, ~657.5 stress-free) are themselves
            // derived under exactly this same linear approximation, so
            // using it here is what makes that real validation apply at
            // all. `T_ref=MELTING_POINT_K`: water right at its own melting
            // point is the coldest/densest real liquid-water state this
            // scene ever has, so every other real liquid-water particle is
            // relatively buoyant against it, matching a real pot heated
            // from a 0C starting point.
            //
            // Real beta (thermal expansion coefficient): water's own beta
            // is strongly temperature-dependent (CRC Handbook / NIST water
            // data) -- ~2.1e-4/K near 20C, ~4.6e-4/K near 50C, ~7.5e-4/K
            // near 100C. A single constant is a real, disclosed
            // simplification (same convention as this demo's own single-cp-
            // per-phase choice) -- picked at ~50C (323K), roughly the
            // middle of this scene's real liquid-water range.
            const WATER_THERMAL_EXPANSION_COEFF_PER_K: f32 = 4.6e-4;
            // Real bug found live (2026-08-29): a raw `v += a*dt` kick using
            // the coarse per-FRAME dt (not the solver's own much smaller
            // adaptive per-substep dt) pinned water at its hard compression
            // floor (`detF=[0.500,0.500]` exactly, for 11+ real seconds
            // straight) -- the exact same class of bug the steam buoyancy
            // block above already learned from and fixed (see its own
            // "root-cause fix 2026-08-28" comment: an unbounded per-frame
            // kick lands before the solver's CFL scan can ever react to it).
            // Real fix: the SAME bounded relax-toward-terminal-velocity form
            // steam already uses, so this can never overshoot however large
            // `beta*delta_t` gets.
            const WATER_BUOYANCY_RELAX_RATE_PER_S: f32 = 2.0;
            for i in 0..count {
                if particles.material_id[i] != WATER_ID {
                    continue;
                }
                let delta_t = particles.temperature[i] - MELTING_POINT_K;
                let buoyancy_accel =
                    -live_gravity * (WATER_THERMAL_EXPANSION_COEFF_PER_K * delta_t);
                let v_terminal = buoyancy_accel / WATER_BUOYANCY_RELAX_RATE_PER_S;
                let blend = (WATER_BUOYANCY_RELAX_RATE_PER_S * dt).min(1.0);
                let v_current = particles.v[i];
                particles.v[i] += (v_terminal - v_current) * blend;
            }
        }

        // Real cursor push/pull -- same real, already-proven mechanism
        // basic_snow.rs/basic_fluids.rs already use (a normal velocity-space
        // impulse, not restricted by any of the strict-WC-MPM-fluid checks
        // above -- those only gate pinning/contact/mixture/sleep/boundary/
        // apic_blend, never impulses or force fields).
        if self.lmb || self.rmb {
            let mag = if self.lmb {
                self.push_strength
            } else {
                -self.push_strength
            };
            self.sim.apply_radial_impulse(self.cursor_grid(), 6.0, mag);
        }

        self.sim.step();

        // Real, temporary diagnostic (2026-08-28) -- checking a live "sizes
        // don't hold / everything rotates" report against the actual per-
        // particle deformation gradient, not a visual guess. Prints the max
        // |off-diagonal| of F split by material: fluid/gas materials
        // unconditionally re-diagonalize F to a pure isotropic scale every
        // substep (`update_particle`, confirmed by direct code read), so a
        // nonzero water/steam value here would be real, hard evidence of an
        // engine bug -- a zero value would mean the reported "rotation" is
        // real bulk circulation (gravity+buoyancy+cursor), not a per-
        // particle rendering artifact.
        if self.frame.is_multiple_of(60) {
            let particles = self.sim.particles();
            let mut max_offdiag = [0.0_f32; 3]; // [ice, water, steam]
            let mut max_det = [f32::MIN; 3];
            let mut min_det = [f32::MAX; 3];
            // Real diagnostic (2026-08-28) for the "solids still break like
            // elastic" report -- direct, logged evidence of whether the
            // Kelvin-Voigt damping added to RankineMaterial tonight is
            // actually dissipating energy (avg/max ice speed should DECAY
            // toward rest after an impact if it's working) and whether
            // damage is accumulating sanely (RankineMaterial repurposes
            // `friction_hardening` as its damage accumulator, 0=intact,
            // saturating around ~1.5 at this preset's softening_rate=2.0 --
            // see `rankine_damage_saturation_point`).
            let (mut ice_speed_sum, mut ice_n, mut ice_max_speed) = (0.0_f32, 0usize, 0.0_f32);
            let (mut ice_damage_sum, mut ice_max_damage) = (0.0_f32, 0.0_f32);
            // Real diagnostic (2026-08-28): testing whether
            // `IdealGasMaterial::timestep_bound`'s own disclosed blind spot
            // (acoustic bound uses a FIXED reference_temperature_k, not the
            // particle's live temperature) is the real driver of the
            // substep explosion -- live temperature should climb steadily
            // above BOILING_POINT_K=373.15 as sustained heating continues
            // past the transition, if this hypothesis is right.
            let (mut steam_temp_sum, mut steam_n, mut steam_max_temp) = (0.0_f32, 0usize, 0.0_f32);
            // Real diagnostic (2026-08-28): `IdealGasMaterial::update_particle`
            // silently clamps `f_trial`'s determinant to `[volume_ratio_min,
            // volume_ratio_max]` every substep with no record of the PRE-clamp
            // value -- the 20.000 ceiling seen every frame in `detF` above could
            // be a mild, occasional excursion the clamp gently catches, or a
            // violent one masked completely. `trace(velocity_gradient)` (the
            // APIC C matrix) is the exact quantity `f_trial=(I+dt*C)*F` is built
            // from, so its magnitude directly answers that without needing the
            // solver's internal per-substep dt.
            let mut steam_max_abs_c_trace = 0.0_f32;
            // Real diagnostic (2026-08-29): direct user correction -- the
            // deformation-gradient offdiag check above (proven zero all
            // night) only rules out a particle's OWN shape twisting; it says
            // nothing about the velocity FIELD genuinely swirling as a
            // group, which is a real, distinct quantity (vorticity, the
            // antisymmetric half of the velocity gradient -- divergence,
            // tracked above via trace(C), is the symmetric half). A rising,
            // expanding parcel creating real vorticity around itself is
            // correct physics in reality too (a real bubble wake), so this
            // is a genuine open question, not an assumed bug: is what looks
            // like "rotation" at liftoff real fluid vorticity, or the visual
            // signature of several particles being ejected in different
            // directions from the same crowded spot at once (the real
            // compression/ejection event already found tonight)?
            // omega = (dvy/dx - dvx/dy)/2 = (C.x_axis.y - C.y_axis.x)/2.
            let (mut water_max_abs_vorticity, mut steam_max_abs_vorticity) = (0.0_f32, 0.0_f32);
            let mut water_max_abs_c_trace = 0.0_f32;
            let mut water_n = 0usize;
            for p in particles.iter() {
                let slot = match p.material_id {
                    ICE_ID => 0,
                    WATER_ID => 1,
                    STEAM_ID => 2,
                    _ => continue,
                };
                let f = p.deformation_gradient;
                let offdiag = f.x_axis.y.abs().max(f.y_axis.x.abs());
                max_offdiag[slot] = max_offdiag[slot].max(offdiag);
                let det = f.determinant();
                max_det[slot] = max_det[slot].max(det);
                min_det[slot] = min_det[slot].min(det);
                if slot == 0 {
                    let speed = p.v.length();
                    ice_speed_sum += speed;
                    ice_n += 1;
                    ice_max_speed = ice_max_speed.max(speed);
                    ice_damage_sum += p.friction_hardening;
                    ice_max_damage = ice_max_damage.max(p.friction_hardening);
                }
                if slot == 1 {
                    water_n += 1;
                    let c = p.velocity_gradient;
                    let vorticity = (c.x_axis.y - c.y_axis.x).abs() * 0.5;
                    water_max_abs_vorticity = water_max_abs_vorticity.max(vorticity);
                    let c_trace = (c.x_axis.x + c.y_axis.y).abs();
                    water_max_abs_c_trace = water_max_abs_c_trace.max(c_trace);
                }
                if slot == 2 {
                    steam_temp_sum += p.temperature;
                    steam_n += 1;
                    steam_max_temp = steam_max_temp.max(p.temperature);
                    let c = p.velocity_gradient;
                    let c_trace = (c.x_axis.x + c.y_axis.y).abs();
                    steam_max_abs_c_trace = steam_max_abs_c_trace.max(c_trace);
                    let vorticity = (c.x_axis.y - c.y_axis.x).abs() * 0.5;
                    steam_max_abs_vorticity = steam_max_abs_vorticity.max(vorticity);
                }
            }
            let steam_avg_temp = if steam_n > 0 {
                steam_temp_sum / steam_n as f32
            } else {
                f32::NAN
            };
            let ice_avg_speed = if ice_n > 0 {
                ice_speed_sum / ice_n as f32
            } else {
                f32::NAN
            };
            let ice_avg_damage = if ice_n > 0 {
                ice_damage_sum / ice_n as f32
            } else {
                f32::NAN
            };
            // Real fix: `min_det`/`max_det` fold over that slot's own
            // particles starting from `f32::MAX`/`f32::MIN` -- with zero
            // particles in a slot (e.g. no water/steam yet at frame 0,
            // everything still ice), the fold never runs and the raw
            // sentinel prints as-is (the giant ~3.4e38 seen in the log).
            // Same "no data" convention `steam_avg_temp` above already
            // uses: NaN, not a meaningless sentinel or a 0.0 that would
            // misleadingly read as a real collapsed J=0.
            let slot_n = [ice_n, water_n, steam_n];
            for (slot, &n) in slot_n.iter().enumerate() {
                if n == 0 {
                    min_det[slot] = f32::NAN;
                    max_det[slot] = f32::NAN;
                }
            }
            println!(
                "frame={:5}  fps={:5.1} last_substeps={:5}  \
                 max|offdiag| ice={:7.4} water={:7.4} steam={:7.4}  \
                 detF[min,max] ice=[{:.3},{:.3}] water=[{:.3},{:.3}] steam=[{:.3},{:.3}]  \
                 ice(n={:3}) speed[avg,max]=[{:.4},{:.4}] damage[avg,max]=[{:.4},{:.4}]  \
                 steam(n={:3}) temp[avg,max]=[{:.2},{:.2}] (ref=373.15)  \
                 steam max|trace(C)|={:.3}  \
                 vorticity[water,steam]=[{:.3},{:.3}]  divergence[water,steam]=[{:.3},{:.3}]",
                self.frame,
                self.last_fps,
                self.sim.last_substeps(),
                max_offdiag[0],
                max_offdiag[1],
                max_offdiag[2],
                min_det[0],
                max_det[0],
                min_det[1],
                max_det[1],
                min_det[2],
                max_det[2],
                ice_n,
                ice_avg_speed,
                ice_max_speed,
                ice_avg_damage,
                ice_max_damage,
                steam_n,
                steam_avg_temp,
                steam_max_temp,
                steam_max_abs_c_trace,
                water_max_abs_vorticity,
                steam_max_abs_vorticity,
                water_max_abs_c_trace,
                steam_max_abs_c_trace,
            );
            // TEMPORARY diagnostic (2026-08-30, real-material update
            // 2026-08-31): tracks whichever water particle holds the
            // current `detF` max -- one persisting outlier vs a rotating
            // population. Originally built to check whether
            // `NewtonianFluidMaterial`'s `pressure_floor` ratchet was
            // engaged there; that material (and its floor) is gone from
            // this demo now (see this file's own top doc), so the real
            // question it can still answer is whether this same particle
            // is genuinely under real tension (`pressure_gauge<0`) --
            // exactly the state the cavitation EOS's own mixture/vapor
            // branches exist to handle correctly instead of flooring.
            if let Some((max_idx, max_j)) = particles
                .iter()
                .enumerate()
                .filter(|(_, p)| p.material_id == WATER_ID)
                .map(|(i, p)| (i, p.deformation_gradient.determinant()))
                .max_by(|a, b| a.1.total_cmp(&b.1))
            {
                let p = particles.get(max_idx);
                // `rho_grid/J = rest_density_grid/max_j`, and
                // `rest_density_grid = rho_l_ref*dx^2`, so this real SI
                // density is exactly `rho_l_ref/max_j` -- `dx` cancels,
                // same derivation `CavitatingFluidMaterial`'s own private
                // `real_density_si` uses internally. Real, live-temperature
                // reconstruction (2026-08-31): reads THIS particle's own
                // `temperature`, not a fixed reference -- the whole real
                // point of the T-dependent closure.
                let density_si = self.water_material.table.rho_l_ref_kg_m3 / max_j;
                let pressure_gauge = self
                    .water_material
                    .table
                    .reconstruct(p.temperature)
                    .pressure_gauge_pa(density_si);
                let under_tension = pressure_gauge < 0.0;
                let dist_to_wall =
                    p.x.x
                        .min(GRID as f32 - p.x.x)
                        .min(p.x.y)
                        .min(GRID as f32 - p.x.y);
                let same_mat_neighbors = self.sim.count_near(p.x, 3.0, p.material_id);
                let same_as_last = self.water_jmax_prev_idx == Some(max_idx);
                println!(
                    "  [water-jmax/frame={:5}] idx={:4} (same_as_last_sample={}) J={:.4} \
                     T={:.2}K pos=({:.2},{:.2}) dist_to_wall={:.2} same_mat_neighbors(r=3)={} \
                     pressure_gauge={:.2}Pa under_tension={}",
                    self.frame,
                    max_idx,
                    same_as_last,
                    max_j,
                    p.temperature,
                    p.x.x,
                    p.x.y,
                    dist_to_wall,
                    same_mat_neighbors,
                    pressure_gauge,
                    under_tension,
                );
                self.water_jmax_prev_idx = Some(max_idx);
            }
            // TEMPORARY diagnostic (2026-09-01): the real, direct check on
            // `BoilingMixtureMaterial`'s own core claim -- a genuinely
            // mid-boil particle's mechanical `J` should track the real
            // mass-fraction equilibrium `J_eq(x)=1+(rho_l_ref/rho_v_ref-1)*x`
            // this material's own doc derives, not run free the way the
            // old bug let it (the live symptom this whole fix answers:
            // `J=5.988` at `x_H<0.5`, i.e. `J` nearly at the FULL-vapor
            // equilibrium while barely a third boiled). `J/J_eq->1` here is
            // the real, direct confirmation the fix is doing its job.
            if let Some((max_idx, max_j)) = particles
                .iter()
                .enumerate()
                .filter(|(_, p)| p.material_id == BOILING_ID)
                .map(|(i, p)| (i, p.deformation_gradient.determinant()))
                .max_by(|a, b| a.1.total_cmp(&b.1))
            {
                let p = particles.get(max_idx);
                let x = p.friction_hardening.clamp(0.0, 1.0);
                let j_eq = self.boiling_material.j_eq(x);
                let stress = self.boiling_material.kirchhoff_stress(particles, max_idx);
                let pressure_gauge = -stress.x_axis.x;
                // Real, disclosed diagnostic (2026-09-01, external review's
                // own decisive test #1): `J/J_eq != 1` is not automatically
                // a bug -- a column under real gravity needs real internal
                // pressure to hold its own weight, and this material's own
                // `p = c_mix2(x)*(rho-rho_eq(x))` computes exactly that.
                // Converts the observed residual into a real Pa figure and
                // compares it against a direct hydrostatic estimate
                // (`p ~= rho*g*(y_surface-y)`) at this particle's own real
                // depth -- if the orders of magnitude agree, the residual is
                // real physics, not drift. `y_surface` is the real, live
                // top of the condensed-phase column THIS frame (max y over
                // ICE_ID/WATER_ID/BOILING_ID -- steam excluded, it's not
                // part of the hydrostatic medium).
                let y_surface = particles
                    .iter()
                    .filter(|q| matches!(q.material_id, ICE_ID | WATER_ID | BOILING_ID))
                    .map(|q| q.x.y)
                    .fold(f32::NEG_INFINITY, f32::max);
                let depth_m = (y_surface - p.x.y).max(0.0) * self.sim.config().dx_meters;
                let rho_eq_si = self.boiling_material.rho_eq_kg_m3(x);
                let g_si = (self.real_gravity.y * self.gravity_fraction).abs()
                    * self.sim.config().dx_meters;
                let p_hydro_pa = rho_eq_si * g_si * depth_m;
                println!(
                    "  [boiling-jmax/frame={:5}] idx={:4} J={:.4} J_eq={:.4} J/J_eq={:.4} \
                     x={:.4} T={:.2}K pressure_gauge={:.2}Pa depth={:.3}m p_hydro={:.2}Pa",
                    self.frame,
                    max_idx,
                    max_j,
                    j_eq,
                    max_j / j_eq,
                    x,
                    p.temperature,
                    pressure_gauge,
                    depth_m,
                    p_hydro_pa,
                );
            }
            // TEMPORARY diagnostic (2026-08-31, external review): the real
            // A/B this whole steam-runaway investigation needs, per that
            // review's own decisive proposal -- for the worst (max detF)
            // STEAM particle, log the mechanical-equilibrium comparison
            // (`J` vs the real analytic `J_eq(T)` where `p_gauge=0`) and
            // the real neighbor/grid-occupancy picture, not the render's
            // own visual size (proven separately, same review, to be
            // unusable here -- `render_particles.wgsl` draws directly
            // from `F`, blind to `initial_volume`/`volume`, so a
            // constitutive REFERENCE rebase at the water->steam
            // transition reads as a fake size change). Interpretation:
            // `J/J_eq->1` and `p_gauge->0` while positions still disperse
            // means this is forcing-without-a-medium + under-sampling,
            // not an EOS runaway -- stop chasing the solver. `J/J_eq`
            // still GROWING while `p_gauge<0` means the mechanical
            // equilibrium itself is being missed -- the real next step
            // would be a G2P occupied-vs-empty-node contribution check.
            if let Some((max_idx, max_j)) = particles
                .iter()
                .enumerate()
                .filter(|(_, p)| p.material_id == STEAM_ID)
                .map(|(i, p)| (i, p.deformation_gradient.determinant()))
                .max_by(|a, b| a.1.total_cmp(&b.1))
            {
                let p = particles.get(max_idx);
                let temperature = p.temperature.max(1.0);
                // Real, exact mechanical-equilibrium J (p_gauge=0): from
                // `p0=rho0*R*T` (live-T, matches `kirchhoff_stress`'s own
                // formula) and the isentropic relation
                // `p_abs=p0*(rho/rho0)^gamma`, solving `p_abs=reference_
                // pressure_pa` for `J=V/V0=rho0/rho` gives
                // `J_eq=(T/reference_temperature_k)^(1/gamma)` --
                // self-consistent with `reference_temperature_k` itself
                // (T=reference_temperature_k gives J_eq=1 exactly).
                let j_eq = (temperature / self.steam_material.reference_temperature_k)
                    .powf(1.0 / self.steam_material.adiabatic_index);
                let stress = self.steam_material.kirchhoff_stress(particles, max_idx);
                let pressure_gauge = -stress.x_axis.x;
                let all_neighbors = self.sim.particles_near(p.x, 3.0).len();
                let water_neighbors = self.sim.count_near(p.x, 3.0, WATER_ID);
                let weights = quadratic_weights(p.x);
                let grid = self.sim.grid();
                let mut touched_support_nodes = 0u32;
                for gy in 0..3 {
                    for gx in 0..3 {
                        let cell = weights.base_cell + IVec2::new(gx - 1, gy - 1);
                        if grid.mass_at(cell) > 0.0 {
                            touched_support_nodes += 1;
                        }
                    }
                }
                println!(
                    "  [steam-jmax/frame={:5}] idx={:4} J={:.4} J_eq={:.4} J/J_eq={:.4} \
                     pressure_gauge={:.2}Pa all_neighbors={} water_neighbors={} \
                     touched_support_nodes={}/9 |v|={:.3}",
                    self.frame,
                    max_idx,
                    max_j,
                    j_eq,
                    max_j / j_eq,
                    pressure_gauge,
                    all_neighbors,
                    water_neighbors,
                    touched_support_nodes,
                    p.v.length(),
                );
            }
            // TEMPORARY diagnostic (2026-08-28): direct instrumentation
            // (`EMERGE_CFL_DIAGNOSE=2`, see `cfl::diagnose_worst_particle_
            // cfl_term`'s own doc) found particle #15 specifically is the
            // globally-worst-constrained particle in ~66% of 15123 real
            // samples -- not a diffuse steam-population effect, one
            // particular particle in an escalating runaway. Tracking its
            // own real state directly answers what's actually different
            // about it: is it near a domain wall (where reflected/slip
            // forces could compound), was it an early outlier, is its
            // local neighborhood sparse (the qualitative condition the
            // single-particle-instability paper describes, even though
            // that paper's own specific derived bound didn't turn out to
            // be the binding term here).
        }
        // Real, decisive instrumentation (2026-09-02, external review's own
        // recommended go/no-go gate before scoping any two-way mechanical/
        // thermal closure for `BoilingMixtureMaterial`): the single-worst-
        // particle `[boiling-jmax]` trace above answers "is there SOME live
        // residual," not whether it's large, persistent, and depth-coherent
        // enough to matter -- closing the pressure->phase loop before
        // knowing that would risk converting P2G/boundary discretization
        // noise into fake enthalpy/vapor-quality changes. This block covers
        // the WHOLE `BOILING_ID` population instead of one outlier: for
        // each particle, converts its own mechanical gauge pressure to
        // absolute, inverts the real IAPWS-IF97 Region 4 saturation curve
        // to get that pressure's own saturation temperature
        // (`water_saturation_temperature_from_pressure_k`, see that
        // function's own doc), and reports `delta_T_sat = T_sat(p_abs) -
        // BOILING_POINT_K` as median/p10/p90 AND sign -- never just
        // `max(J)` the way the single-particle trace does, per that
        // review's own explicit instruction. Also bins by depth (shallow/
        // mid/deep thirds of the live condensed-phase column) and reports
        // the population's own average |div(v)| (via `trace(velocity_
        // gradient)`, same convention `water_max_abs_c_trace` above uses)
        // and |v|, since a residual significant ONLY during high velocity/
        // divergence is a dynamic/numerical signal, not durable
        // thermodynamic pressure (that review's own decision grid).
        //
        // Converts the pressure residual into the real decision metric
        // that review specified: `epsilon_x_equiv = cp_liquid*|delta_T_sat|
        // /vaporization_latent_heat` -- the vapor-quality change this
        // pressure residual WOULD cause if mechanical and thermal states
        // were coupled, without actually coupling them (this material
        // stays one-directional; see its own module doc). With this
        // engine's real constants (`WATER_HEAT_CAPACITY_J_KG_K=4182`,
        // `VAPORIZATION_LATENT_HEAT_J_KG=2_257_000`), `epsilon_x_equiv
        // ~= 0.00185/K` -- 1K of `delta_T_sat` is ~0.185% quality, 10K is
        // ~1.85%.
        if self.frame.is_multiple_of(300) {
            const STANDARD_ATMOSPHERE_PA: f32 = 101_325.0;
            let particles = self.sim.particles();
            let y_surface = particles
                .iter()
                .filter(|q| matches!(q.material_id, ICE_ID | WATER_ID | BOILING_ID))
                .map(|q| q.x.y)
                .fold(f32::NEG_INFINITY, f32::max);
            let dx_m = self.sim.config().dx_meters;

            struct BoilingSample {
                depth_m: f32,
                delta_t_sat_k: f32,
                abs_div_v: f32,
                speed: f32,
            }
            let mut samples: Vec<BoilingSample> = Vec::new();
            for (i, p) in particles.iter().enumerate() {
                if p.material_id != BOILING_ID {
                    continue;
                }
                let stress = self.boiling_material.kirchhoff_stress(particles, i);
                let pressure_gauge = -stress.x_axis.x;
                let pressure_absolute = pressure_gauge + STANDARD_ATMOSPHERE_PA;
                let t_sat = emerge::thermodynamics::water_saturation::water_saturation_temperature_from_pressure_k(
                    pressure_absolute,
                );
                let c = p.velocity_gradient;
                samples.push(BoilingSample {
                    depth_m: (y_surface - p.x.y).max(0.0) * dx_m,
                    delta_t_sat_k: t_sat - BOILING_POINT_K,
                    abs_div_v: (c.x_axis.x + c.y_axis.y).abs(),
                    speed: p.v.length(),
                });
            }

            if !samples.is_empty() {
                samples.sort_by(|a, b| a.delta_t_sat_k.total_cmp(&b.delta_t_sat_k));
                let n = samples.len();
                let percentile = |q: f32| -> f32 {
                    let idx = (((n - 1) as f32) * q).round() as usize;
                    samples[idx].delta_t_sat_k
                };
                let (p10, median, p90) = (percentile(0.1), percentile(0.5), percentile(0.9));
                let n_positive = samples.iter().filter(|s| s.delta_t_sat_k > 0.0).count();
                let n_negative = samples.iter().filter(|s| s.delta_t_sat_k < 0.0).count();
                let epsilon_x_equiv = |delta_t_k: f32| -> f32 {
                    WATER_HEAT_CAPACITY_J_KG_K * delta_t_k.abs() / VAPORIZATION_LATENT_HEAT_J_KG
                };

                let max_depth_m = samples.iter().map(|s| s.depth_m).fold(0.0_f32, f32::max);
                let band_of = |depth_m: f32| -> usize {
                    if max_depth_m <= 0.0 {
                        0
                    } else {
                        (((depth_m / max_depth_m) * 3.0) as usize).min(2)
                    }
                };
                let mut band_sum = [0.0_f32; 3];
                let mut band_n = [0usize; 3];
                for s in &samples {
                    let b = band_of(s.depth_m);
                    band_sum[b] += s.delta_t_sat_k;
                    band_n[b] += 1;
                }
                let band_avg = |b: usize| -> f32 {
                    if band_n[b] > 0 {
                        band_sum[b] / band_n[b] as f32
                    } else {
                        f32::NAN
                    }
                };
                let avg_abs_div_v = samples.iter().map(|s| s.abs_div_v).sum::<f32>() / n as f32;
                let avg_speed = samples.iter().map(|s| s.speed).sum::<f32>() / n as f32;

                println!(
                    "  [boiling-residual-stats/frame={:5}] n={:4} \
                     delta_T_sat[p10,median,p90]=[{:+.3},{:+.3},{:+.3}]K sign(+/-)={}/{} \
                     eps_x_equiv[median,p90]=[{:.5},{:.5}] \
                     by_depth(shallow,mid,deep)=[{:+.3},{:+.3},{:+.3}]K \
                     avg|div(v)|={:.4} avg|v|={:.4}",
                    self.frame,
                    n,
                    p10,
                    median,
                    p90,
                    n_positive,
                    n_negative,
                    epsilon_x_equiv(median),
                    epsilon_x_equiv(p90),
                    band_avg(0),
                    band_avg(1),
                    band_avg(2),
                    avg_abs_div_v,
                    avg_speed,
                );
            }
        }
        // TEMPORARY diagnostic (2026-08-28): the 60-frame-cadence trace above
        // showed particle 15 accelerating from |v|=5.3 (frame 60, already
        // water) to |v|=65.6 (frame 120) -- NOT an instant-of-transition
        // spike (it was already stable water at frame 60), so the already-
        // fixed `init_particle_from_transition` continuity fix isn't the
        // relevant mechanism here. Every-frame resolution across that exact
        // window to find precisely when and how fast the real acceleration
        // happens, instead of guessing from 60-frame-apart snapshots.
        const TRACKED_PARTICLE_INDEX: usize = 15;
        const TRACKED_PARTICLE_WINDOW_END_FRAME: u64 = 200;
        if self.frame < TRACKED_PARTICLE_WINDOW_END_FRAME {
            let particles = self.sim.particles();
            let p15 = particles.get(TRACKED_PARTICLE_INDEX);
            let dist_to_wall = p15
                .x
                .x
                .min(GRID as f32 - p15.x.x)
                .min(p15.x.y)
                .min(GRID as f32 - p15.x.y);
            let j15 = p15.volume / p15.initial_volume;
            let same_material_neighbors = self.sim.count_near(p15.x, 3.0, p15.material_id);
            println!(
                "  [p15/frame={:4}] material={} pos=({:.3},{:.3}) dist_to_wall={:.2} \
                 v=({:.3},{:.3}) |v|={:.3} J={:.3} temp={:.2} same_mat_neighbors(r=3)={} \
                 mass={:.6} volume={:.6} initial_volume={:.6} density={:.6}",
                self.frame,
                p15.material_id,
                p15.x.x,
                p15.x.y,
                dist_to_wall,
                p15.v.x,
                p15.v.y,
                p15.v.length(),
                j15,
                p15.temperature,
                same_material_neighbors,
                p15.mass,
                p15.volume,
                p15.initial_volume,
                p15.density,
            );
        }

        self.frame += 1;
        self.fps_frames += 1;
        if self.fps_timer.elapsed().as_secs_f32() >= 1.0 {
            self.last_fps = self.fps_frames as f32 / self.fps_timer.elapsed().as_secs_f32();
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

        // Real, disclosed capture aid -- see `CaptureState`'s own doc.
        // Renders a SECOND time into the dedicated offscreen capture
        // texture (not the swapchain view above) so the readback below has
        // a real `COPY_SRC`-capable source. Self-terminates once
        // `target_frames` is reached -- a real, non-interactive, fully
        // automatic capture run.
        if let Some(cap) = &mut self.capture
            && cap.captured < cap.target_frames
            && self.frame.is_multiple_of(cap.stride)
        {
            let capture_view = cap
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default());
            self.renderer.render(
                &self.device,
                &self.queue,
                self.sim.particles(),
                &capture_view,
                true,
            );
            let raw = read_full_frame_rgba(
                &self.device,
                &self.queue,
                &cap.texture,
                cap.width,
                cap.height,
            );
            use std::io::Write;
            cap.file
                .write_all(&raw)
                .unwrap_or_else(|e| panic!("failed to append frame to {:?}: {e}", cap.path));
            cap.captured += 1;
            if cap.captured.is_multiple_of(30) || cap.captured == cap.target_frames {
                println!(
                    "[capture] {}/{} frames written",
                    cap.captured, cap.target_frames
                );
            }
            if cap.captured >= cap.target_frames {
                cap.file.flush().ok();
                println!(
                    "[capture] done -- {} frames in {:?}",
                    cap.captured, cap.path
                );
                std::process::exit(0);
            }
        }

        // --- egui panel ---
        let raw_input = self.egui_state.take_egui_input(window);
        let fps = self.last_fps;
        let mut target_temperature = self.target_temperature;
        let mut gravity_fraction = self.gravity_fraction;
        let mut push_strength = self.push_strength;
        let ice_n = self
            .sim
            .particles()
            .iter()
            .filter(|p| p.material_id == ICE_ID)
            .count();
        let water_n = self
            .sim
            .particles()
            .iter()
            .filter(|p| p.material_id == WATER_ID)
            .count();
        let steam_n = self
            .sim
            .particles()
            .iter()
            .filter(|p| p.material_id == STEAM_ID)
            .count();
        let mut reset = false;

        let full_output = self.egui_ctx.run(raw_input, |ctx| {
            egui::Window::new("Phase states")
                .default_pos([10.0, 10.0])
                .default_width(300.0)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.label(format!("fps={fps:.0}  avg_T={current_avg:.1}K"));
                    ui.label(format!("ice={ice_n}  water={water_n}  steam={steam_n}"));
                    ui.separator();
                    ui.label(format!(
                        "Target temperature (melt={MELTING_POINT_K:.0}K, boil={BOILING_POINT_K:.0}K):"
                    ));
                    // Real, disclosed ceiling (2026-09-02): 600K, not an
                    // arbitrary round number -- comfortably under water's
                    // real IAPWS-IF97 critical point (647.096K,
                    // `water_saturation::WATER_CRITICAL_POINT_K`), past
                    // which the liquid/vapor distinction this whole demo's
                    // chained enthalpy/phase-state model relies on stops
                    // meaning anything. Known, already-disclosed limitation
                    // pushing toward this ceiling: `IdealGasMaterial`'s
                    // acoustic CFL bound uses a FIXED `reference_temperature_k`,
                    // not the particle's own live temperature (see
                    // `project_gas_bulk_viscosity_shipped_steam_lag_
                    // unresolved` memory) -- substep count can climb well
                    // before 600K is reached, real cost, not a crash.
                    ui.add(egui::Slider::new(&mut target_temperature, 150.0..=600.0));
                    ui.separator();
                    ui.label("Gravity (1.0 = real IRL 9.81 m/s²):");
                    ui.add(egui::Slider::new(&mut gravity_fraction, 0.0..=1.0));
                    ui.separator();
                    ui.label("Push/pull strength:");
                    ui.add(egui::Slider::new(&mut push_strength, 0.0..=30.0));
                    ui.separator();
                    ui.label(
                        "Real bidirectional phase transitions: drag Target temperature up to \
                         melt then boil, down to condense then freeze -- same real latent-heat \
                         hysteresis as phase_states_headless.rs, just under your own control. \
                         Real gravity: ice falls and water pools. Real Archimedes \
                         buoyancy: steam rises once it exists.",
                    );
                    ui.label("LMB push  RMB pull  R reset  Q quit");
                    if ui.button("Reset").clicked() {
                        reset = true;
                    }
                });
        });
        self.target_temperature = target_temperature;
        self.gravity_fraction = gravity_fraction;
        self.push_strength = push_strength;
        if reset {
            let (sim, ice_material, water_material, boiling_material, steam_material) = make_sim();
            self.real_gravity = sim.config().gravity;
            self.enthalpy = initial_enthalpy(sim.particles().len());
            self.ice_material = ice_material;
            self.water_material = water_material;
            self.boiling_material = boiling_material;
            self.steam_material = steam_material;
            self.sim = sim;
            self.target_temperature = 250.0;
            self.plate_temperature = 250.0;
            self.frame = 0;
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
                    .with_title("emerge -- Phase states (ice/water/steam)")
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
                    KeyCode::KeyR if pressed => {
                        let (sim, ice_material, water_material, boiling_material, steam_material) =
                            make_sim();
                        s.real_gravity = sim.config().gravity;
                        s.enthalpy = initial_enthalpy(sim.particles().len());
                        s.ice_material = ice_material;
                        s.water_material = water_material;
                        s.boiling_material = boiling_material;
                        s.steam_material = steam_material;
                        s.sim = sim;
                        s.target_temperature = 250.0;
                        s.plate_temperature = 250.0;
                        s.frame = 0;
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

extern crate emerge_engine as emerge;

#[path = "../gui_common/coords.rs"]
mod gui_common;
#[path = "../gui_common/render_mode.rs"]
mod render_mode;
#[path = "../gui_common/scripted.rs"]
mod scripted;

use egui_wgpu::ScreenDescriptor;

/// Interactive, windowed counterpart to `phase_states_headless.rs`: the same
/// ice<->water<->steam cycle, driven by a live target-temperature slider instead of a
/// scripted heat-then-cool schedule.
///
/// The slider sets a target temperature; a heating plate ramps toward it and heats
/// each particle in contact by Newton's law of cooling (see `plate_temperature`). The
/// heat drives a per-particle enthalpy state (Voller & Cross 1981 enthalpy method, see
/// `water_phase_chain`), from which temperature, material transitions and latent heat
/// all follow; this demo does not use `add_phase_rule`/`WithLatentHeat`.
///
/// Materials: `RankineMaterial::ice()` (brittle-fracture ice, scaled stiffness, see
/// `ICE_YOUNG_MODULUS_SCALED_PA`) for the solid; `CavitatingFluidMaterial` (Lyu, Sun,
/// Colagrossi & Zhang 2023's three-branch cavitation EOS, temperature-coupled, see
/// `cavitating_eos`) for the liquid; `BoilingMixtureMaterial` (Homogeneous Equilibrium
/// Model mixture driven by the enthalpy method's `boiling_fraction`) for a particle
/// mid-boil, so its mechanical vapor fraction follows its thermal one; and
/// `IdealGasMaterial` (isentropic ideal-gas EOS) for the gas. The coupling is one-way
/// (`x_H -> mechanical state`): a mechanical deviation from equilibrium does not pay
/// latent heat back into the enthalpy state. `phase_states_headless.rs` still uses
/// `NewtonianFluidMaterial`.
///
/// "Chimney" geometry: a narrow column spawned near the bottom of a tall domain, with
/// gravity (live slider, the `gravity_fraction` convention of `basic_snow.rs`) and
/// Archimedes buoyancy (the `BuoyancyField` formula) applied to steam only, each frame,
/// in step with the gravity slider (see `update_and_render`; applied to every particle
/// it would nudge the starting ice block upward). Ice and water fall under gravity;
/// steam rises into the room above.
///
/// G cycles the view: particles, the grid-volume view, the curvature-flow
/// surface. Ice, water and the boiling mixture declare measured optics
/// (`optical::pure_ice`, Warren & Brandt 2008; `optical::pure_water`, Pope &
/// Fry 1997). Clear ice absorbs about as little as water does, so those two
/// views tell ice from water mostly by the ice being drawn flat, as a solid,
/// and the steam by its density (a sixth of water's mass per cell; it
/// declares no optics). The particle view keeps `ByMaterial`'s placeholder
/// colours.
///
///   cargo run --example phase_states_gui --features "render,experimental"
///
/// `EMERGE_SCRIPT_LOG=<file>` runs a scripted hand instead of the mouse and
/// logs every frame (`gui_common/scripted.rs`): the strongest push into the
/// ice column, then just below it, then a pull. Measured that
/// way, the water and ice stiffness here is sized for 18 m/s, the free fall
/// at real gravity, while the scene runs at 0.01 of it: left alone the ice
/// moves at 0.09 m/s, and the strongest push throws it at 75.7 m/s, Mach
/// 0.42 against the sized 0.1.
use emerge::grid::kernel::quadratic_weights;
use emerge::materials::optical;
use emerge::matter::materials::solid::rankine::{
    ICE_Q_REFERENCE_FREQUENCY_HZ, q_factor_elastic_viscosity_pa_s,
};
use emerge::render::{ColorMode, CpuRenderBridge, Renderer};
use emerge::{
    BoilingMixtureMaterial, CavitatingEosTable, CavitatingFluidMaterial, IdealGasMaterial,
    MaterialModel, RankineMaterial, SimConfig, Simulation, SlipBoundary, SpawnRegion,
};
use glam::{IVec2, Mat2, Vec2};
use render_mode::RenderMode;
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

const GRID: usize = 64;
/// Particle pitch of the ice column, in grid cells.
const SPACING: f32 = 0.5;
/// Surface grid resolution as a multiple of the physics grid.
const SURFACE_RES_MULTIPLIER: u32 = 4;
/// The push slider's top, which a scripted run presses at.
const PUSH_STRENGTH_MAX: f32 = 30.0;
const ICE_ID: u32 = 0;
const WATER_ID: u32 = 1;
const STEAM_ID: u32 = 2;
/// Mid-boil material: see `emerge::BoilingMixtureMaterial` (keeps the mechanical vapor
/// fraction tied to the enthalpy method's thermal one) and
/// `material_id_for_phase_state`.
const BOILING_ID: u32 = 3;

// Water phase-change constants -- identical to phase_states_headless.rs,
// see that file's doc for the sourcing.
const MELTING_POINT_K: f32 = 273.15;
const BOILING_POINT_K: f32 = 373.15;
const FUSION_LATENT_HEAT_J_KG: f32 = 334_000.0;
const VAPORIZATION_LATENT_HEAT_J_KG: f32 = 2_257_000.0;
const WATER_HEAT_CAPACITY_J_KG_K: f32 = 4182.0;
// `energy::thermodynamics::enthalpy::chained_state_from_enthalpy` (Voller & Cross
// 1981 enthalpy method, extended to two consecutive transitions) tracks enthalpy as
// the primary thermal state: temperature plateaus at each transition point while
// `phase_fraction` absorbs the full latent heat, with no scale factor, no instant
// temperature jump and no hysteresis margin (H is monotonic and continuous, so there
// is nothing to debounce).
const ICE_HEAT_CAPACITY_J_KG_K: f32 = 2090.0; // real, CRC Handbook at 0C
const STEAM_HEAT_CAPACITY_J_KG_K: f32 = 2080.0; // real, NIST steam tables near 100C, 1 atm

// Heating plate: the spawn column (see `make_sim`'s `spawn`) is centered at
// y=GRID*0.22=14.08 with height `box_size.y*spacing`=10*0.5=5.0, spanning
// y in [11.58, 16.58]. The plate covers the bottom 40% of that span
// (11.58 + 0.4*5.0 = 13.58): enough ice sits on it to start melting and subsiding
// (exposing more ice to the plate, as a block melts from a hot surface underneath),
// while the column above has no direct heat source, giving the vertical DeltaT that
// uniform heating lacks.
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

/// A particle mid-melt shows whichever phase holds the majority of its latent-heat
/// band (`fraction < 0.5` keeps the colder identity, `>= 0.5` switches); partial-melt
/// stiffness softening is not modeled, and the 0.5 cutoff is a coarse-graining choice,
/// not derived from physics.
///
/// Every `PhaseState::Boiling` particle, whatever its `fraction`, gets `BOILING_ID`,
/// whose mechanical response follows `fraction` directly (through
/// `Particle::friction_hardening`, written every substep below), so there is no
/// threshold to cross mid-band. A water/steam majority split would let the mechanical
/// vapor fraction drift away from `fraction` (see `BoilingMixtureMaterial`'s doc).
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

// Ice stiffness derived for this scene's dynamics. `phase_states_headless.rs` uses
// 5.0e5 Pa at zero gravity, where nothing falls; here, with gravity, ice's wave speed
// at 5.0e5 Pa, `c = sqrt(E/rho)` = sqrt(5e5/917) = 23.35 m/s, is only ~1.3x this
// scene's velocity scale (18.0 m/s, Torricelli free fall from the column's height,
// as for WATER_C_REF_M_S below), so the ice deforms and bounces instead of acting
// rigid. The water margin rule (wave speed = 10*v_max, after Monaghan 1994's
// stiffness margin, applied to an elastic solid's wave speed) gives
// `E = rho*(10*v_max)^2 = 917*180^2 = 2.971e7 Pa`: ~59x stiffer, still ~303x softer
// than real ice (E=9.0 GPa, see RankineMaterial::ice's doc), a CFL compromise.
const ICE_YOUNG_MODULUS_SCALED_PA: f32 = 2.971e7;

// Tensile strength from the unscaled E, independent of the scaled E used for the
// elastic response. `RankineMaterial::ice()` sets strength as a fixed ratio of the
// modulus it is given (`E * 1.1e-4`, see its doc), calibrated for the E=9.0 GPa,
// where it gives 9.0e9*1.1e-4 = 0.99 MPa, inside Petrovic 2003's 0.7-3.1 MPa. Strength
// and stiffness are independent properties; scaling E for CFL must not scale strength.
// With the scaled E the ratio gives 3,268 Pa, 13.8x below the self-weight stress at
// the column base (rho*g*h = 917*9.81*5 = 44,979 Pa), so the column would fracture
// under its own weight. Overridden by struct update (the pattern this preset's doc
// prescribes for `elastic_viscosity`): 990,000 Pa, a 22x margin over self-weight.
const ICE_TENSILE_STRENGTH_REAL_PA: f32 = 9.0e9 * 1.1e-4;

const STEAM_ADIABATIC_INDEX: f32 = 1.33;
const STEAM_VISCOSITY_PA_S: f32 = 1.26e-5;
const WATER_RHO_KG_M3: f32 = 1000.0;
const STEAM_RHO_KG_M3: f32 = WATER_RHO_KG_M3 / 6.0;
const STEAM_SPECIFIC_GAS_CONSTANT_J_KG_K: f32 = 101_325.0 / (STEAM_RHO_KG_M3 * BOILING_POINT_K);
const ICE_RHO_KG_M3: f32 = 917.0;
/// Effective reference density for steam buoyancy once no water or boiling-mixture
/// neighbor is left nearby (see the buoyancy loop). This is `~333 kg/m^3`, not
/// ambient air's ~1.2 kg/m^3: only the ratio between standard ambient air
/// (~1.204 kg/m^3, sea level, ~20C) and saturated steam at 100C (~0.598 kg/m^3, air
/// ~2.01x denser; NIST/ISA reference values) is kept, applied to this demo's
/// compressed `STEAM_RHO_KG_M3` scale (see that constant's doc). An effective
/// ambient-medium model, not a real one: no air exists in the scene to receive the
/// opposite momentum, entrain or mix with the rising steam, so this sets only the
/// one-sided force a steam particle feels. `STEAM_RISE_DRAG_COEFFICIENT` is likewise
/// an effective placeholder, not derived from a drag law. A real fix needs an ambient
/// density/velocity field steam can exchange momentum with.
const AMBIENT_AIR_RHO_KG_M3: f32 = STEAM_RHO_KG_M3 * 2.0;

// Water EOS stiffness derived for this scene. An under-derived reference sound speed
// lets a fluid compress far past its ~1% limit before the EOS resists, then release
// the stored energy violently (with c_ref = 5.0 m/s, water reached its [0.5, 2.0] J
// clamp and particles 48+ grid-units/s from near standstill at 280 K, far from
// boiling). Weakly compressible rule (Monaghan 1994; Becker & Teschner 2007, cited in
// `NewtonianFluidMaterial::weakly_compressible`'s doc): `c_ref = 10 * v_max`, keeping
// density variation near 1%. `v_max` from Torricelli for this geometry (free fall from
// the ice column's top, `box_center.y + box_size.y*spacing*0.5` = 14.08+2.5 = 16.58 m
// above the floor at y=0): v_max = sqrt(2*9.81*16.58) = 18.0 m/s, c_ref = 180 m/s.
const WATER_C_REF_M_S: f32 = 180.0;

// Disclosed model choice: the mixture band's own effective acoustic
// speed -- see `cavitating_eos`'s module doc ("parameter honesty")
// for why this is NOT a fixed water property, just this demo's own choice
// (same value `real_test_params()` uses, for the same reason above).
const WATER_EOS_C_MIN_M_S: f32 = 1.0;

// Simple proportional heater/cooler -- see this file's own top doc
// for why a target-temperature slider is more intuitive than a raw rate
// dial. Clamped so the per-substep instant latent-heat jump (see
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
    // Disclosed change from phase_states_headless.rs's own gravity=ZERO
    // (chosen there specifically to isolate the thermal cycle from settling
    // dynamics): THIS demo wants exactly the settling dynamics -- gas rising,
    // liquid pooling, solid sinking, a real "chimney" behavior that can only
    // emerge from real gravity + real buoyancy (Archimedes -- lighter than
    // the surrounding fluid floats, heavier sinks, see BuoyancyField's own
    // doc). SimConfig::earth's own real gravity flows through unscaled here;
    // State's own `gravity_fraction` field is a live-adjustable multiplier
    // (see the gravity slider), not a hidden reduction of the value.
    let config = SimConfig {
        max_substeps_per_step: 3000,
        // Water and steam are strict-fluid materials
        // (`owns_deformation_volume_state()==true`), which get no per-substep
        // correction when this is off: `do_substep`'s pre-P2G repair pass skips
        // them, and `assert_owned_deformation_state` runs once per frame (substep 0)
        // and panics rather than repairs. Many small, same-signed changes can walk J
        // past the [j_min, j_max] band over a frame's ~150 substeps without any
        // single step looking inadmissible (see `SimConfig::fluid_step_retry_enabled`):
        // without retry, steam's J races to the volume_ratio ceiling, fps collapses and
        // NaN reaches the renderer. The rollback caveat (rods, grains, a stateful
        // thermal field) does not apply: there are no rods or grains, and
        // `ThermalDiffusion::apply` rebuilds its scratch grid from `particles` every
        // call, so a retried substep re-derives it from the rolled-back state.
        fluid_step_retry_enabled: true,
        ..SimConfig::earth(GRID, 1.0, 0.015)
    };

    // Kelvin-Voigt damping (see `RankineMaterial::elastic_viscosity` and
    // `q_factor_elastic_viscosity_pa_s`): without it ice has no energy dissipation
    // below its fracture threshold and bounces near-elastically under gravity.
    // Q~65, Peters et al. 2012's measurement for temperate ice near 0C, since this
    // ice warms toward `MELTING_POINT_K`; the cold-ice `ICE_QUALITY_FACTOR_Q=700`
    // (Bentley & Kohnen 1976, Antarctic ice) leaves it oscillating for 1000+ frames.
    // Q is inversely related to damping: ~11x more dissipation per cycle.
    const ICE_QUALITY_FACTOR_Q_TEMPERATE: f32 = 65.0;
    // Bare materials, not wrapped in `WithLatentHeat`/`WithLatentHeatTable`: the
    // enthalpy tracking of `water_phase_chain` in `update_and_render` pays both
    // latent heats.
    let ice = {
        let ice_shear_modulus_pa = ICE_YOUNG_MODULUS_SCALED_PA / (2.0 * (1.0 + 0.20));
        let elastic_viscosity_pa_s = q_factor_elastic_viscosity_pa_s(
            ice_shear_modulus_pa,
            ICE_QUALITY_FACTOR_Q_TEMPERATE,
            ICE_Q_REFERENCE_FREQUENCY_HZ,
        );
        // Raw SI Pa.s, unconverted: lambda/mu here are raw
        // `lame_from_young` values in the same stress tensor (see
        // `q_factor_elastic_viscosity_pa_s`).
        println!("ice elastic_viscosity: {elastic_viscosity_pa_s:.6} Pa.s (SI, raw)");
        RankineMaterial {
            elastic_viscosity: elastic_viscosity_pa_s,
            tensile_strength: ICE_TENSILE_STRENGTH_REAL_PA,
            optics: Some(optical::pure_ice()),
            ..RankineMaterial::ice(ICE_YOUNG_MODULUS_SCALED_PA, 0.20)
        }
    };
    let water = {
        // Water responds to its own live temperature. `MELTING_POINT_K` is the
        // natural `t_min_k`, the coldest liquid state this demo has.
        // `CavitatingEosTable::build` derives its own `t_max_k`
        // (`t_liquid_closure_max`, just below the boiling point) and picks its
        // node count from measured interpolation error (see `CavitatingEosTable`).
        let table = CavitatingEosTable::build(
            WATER_RHO_KG_M3,
            WATER_C_REF_M_S,
            7.0, // gamma_l -- Cole 1948's real value for water
            STEAM_RHO_KG_M3,
            STEAM_ADIABATIC_INDEX,
            WATER_EOS_C_MIN_M_S,
            MELTING_POINT_K,
        );
        // Disclosed choice (not an arbitrary flat number -- see
        // `CavitatingFluidMaterial::volume_ratio_max`'s doc): full
        // internal vaporization corresponds to `J~=rho_l_ref/rho_v_ref=6.0`
        // for this scene's own reference densities; this demo's own
        // discrete enthalpy-driven swap to `STEAM_ID` is expected to fire
        // well before that, but the EOS itself stays real and well-defined
        // with headroom past it.
        let mut water = CavitatingFluidMaterial::new(table, config.dx_meters, 1.0e-3, 0.5, 8.0);
        water.optics = Some(optical::pure_water());
        water
    };
    // Mid-boil mixture material, built from `water`'s own table, so `BOILING_ID`'s
    // liquid-side reference matches `WATER_ID`'s exactly (continuity at the `x=0`
    // handoff, see `BoilingMixtureMaterial::from_table`). Same `dx_meters` and
    // `volume_ratio_max=8.0` as `water`: full vaporization's equilibrium `J` is
    // `rho_l_ref/rho_v_ref=6.0` (WATER_RHO_KG_M3/STEAM_RHO_KG_M3), so the same
    // headroom covers this material.
    //
    // Same water, so the same measured absorption. The bubbles would also
    // scatter, but that depends on their size, which nothing here measures,
    // so no scattering is declared.
    let mut boiling =
        BoilingMixtureMaterial::from_table(&water.table, config.dx_meters, 1.0e-3, 0.5, 8.0);
    boiling.optics = Some(optical::pure_water());
    let steam = {
        // `IdealGasMaterial` needs bulk viscosity to resist over-expansion (it
        // otherwise resists only compression, see that field's doc). Magnitude:
        // `water_vapor_bulk_viscosity_pa_s` (Cramer 2012).
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

    // No spatial `ThermalDiffusion`: Fourier conduction mutates `temperature`
    // directly, which would pull a particle off a melting or boiling plateau that
    // the enthalpy state (the sole thermal state here) says it is still on.
    // Conduction belongs in the enthalpy update (diffusing `m*h`, not `T`, as
    // Voller-Cross do); not implemented.

    // "Chimney" geometry: a narrow column spawned low in a tall domain. Gravity
    // keeps it from spreading sideways, and there is room above for steam to rise
    // into instead of hitting the domain edge at once.
    let mass_for = |rho_kg_m3: f32| rho_kg_m3 * (SPACING * config.dx_meters).powi(2);
    let spawn = SpawnRegion {
        spacing: SPACING,
        box_size: IVec2::new(6, 10),
        box_center: Vec2::new(config.grid_res as f32 * 0.5, config.grid_res as f32 * 0.22),
        material_id: ICE_ID,
        mass_override: Some(mass_for(ICE_RHO_KG_M3)),
        ..SpawnRegion::for_sim(&config)
    };

    let mut solver = Simulation::new(config, spawn)
        .with_default_material(Box::new(ice))
        .with_material(WATER_ID, Box::new(water.clone()))
        .with_material(STEAM_ID, Box::new(steam))
        // Disclosed ordering requirement (found live): `MaterialRegistry`
        // requires contiguous IDs registered IN NUMERIC ORDER (0,1,2,3,...),
        // not just distinct values -- `BOILING_ID=3` must therefore be
        // registered AFTER `STEAM_ID=2`, not before it.
        .with_material(BOILING_ID, Box::new(boiling))
        // Required constraint, not a style choice: this scene has a
        // strict fluid (`owns_deformation_volume_state()==true`, both
        // `NewtonianFluidMaterial` and `CavitatingFluidMaterial`
        // declare this), and only SlipBoundary declares itself compatible
        // with that -- FrictionBoundary's post-G2P particle mutation isn't
        // a declared fluid traction/no-penetration condition (see
        // BoundaryCondition::is_strict_wc_mpm_fluid_compatible's doc,
        // conservative default false). Confirmed live: FrictionBoundary
        // panicked immediately on startup.
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));
    // No `add_phase_rule`/`with_phase_rule`: `update_and_render`'s enthalpy
    // update decides material transitions (`material_id_for_phase_state`) and
    // applies them with `Simulation::phase_transition`, since the decision needs
    // the phase-fraction state, which a `Particle`-only predicate cannot see.

    const START_TEMPERATURE_K: f32 = 250.0;
    for t in solver.particles_mut().temperature.iter_mut() {
        *t = START_TEMPERATURE_K;
    }
    // Mass consistent with each ice particle's measured volume, not a nominal one.
    // `mass_override` (`mass_for`) assumes every ice particle has volume `spacing^2`,
    // but the spawn measures each particle's volume by kernel-density estimate
    // (right for a solid without an analytical rest volume; strict fluids override
    // it, see `fluid.rs`), which is larger near the block's free surface (fewer
    // neighbors). Together they give every surface particle too low a density
    // (573 kg/m^3 instead of 917), and `init_particle_from_transition` carries that
    // into an oversized J at melt (1.745, a volume increase, where ice->water should
    // shrink), whose spurious potential energy launches the particle (65+
    // grid-units/s within ~1 s of melting). With consistent mass, J lands at ~1.09 at
    // melt (ice occupies ~9% more volume than the same mass of water, 1000/917).
    for i in 0..solver.particles().len() {
        let particles = solver.particles_mut();
        if particles.material_id[i] == ICE_ID {
            particles.mass[i] = ICE_RHO_KG_M3 * particles.initial_volume[i];
        }
    }
    (solver, ice, water, boiling, steam)
}

/// Starting enthalpy for every particle: `make_sim`'s `START_TEMPERATURE_K` (250 K,
/// fully solid ice), through `chained_enthalpy_from_temperature`.
fn initial_enthalpy(count: usize) -> Vec<f32> {
    let h = emerge::thermodynamics::chained_enthalpy_from_temperature(
        &water_phase_chain(),
        250.0,
        emerge::thermodynamics::PhaseState::Solid,
    );
    vec![h; count]
}

/// Blocking full-texture RGBA8 readback -- generalizes the same real,
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

/// Frame capture, opt-in via `PHASE_STATES_CAPTURE_DIR` (see `State::new` for the
/// env-var contract); no effect otherwise. Renders into a dedicated offscreen texture
/// with `COPY_SRC` usage (surface textures are not guaranteed `COPY_SRC`-capable
/// across backends), reads it back to the CPU and appends raw RGBA8 frames to one
/// continuous stream file, which `ffmpeg -f rawvideo` reads directly (numbered
/// per-frame images would need the `image2` demuxer instead).
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
    /// The heating plate's temperature, separate from `target_temperature` (the
    /// slider's setpoint): the plate has thermal inertia and ramps toward the
    /// setpoint at a bounded rate (see where it is updated). Each particle's flux
    /// is `HEAT_GAIN*(plate_temperature-particle_temperature)`, Newton's law of
    /// cooling for contact, self-limiting (a particle above the plate is cooled
    /// toward it, as with a finite-temperature heating element). One heat rate from
    /// the population's average would keep heating a particle already far above
    /// target (777 K against a 450 K target) while the average lagged.
    plate_temperature: f32,
    real_gravity: Vec2,
    gravity_fraction: f32,
    cursor_pos: [f32; 2],
    lmb: bool,
    rmb: bool,
    push_strength: f32,
    frame: u64,
    /// `EMERGE_SCRIPT_LOG`: a scripted run, read from its log (see
    /// `gui_common/scripted.rs`), and where its hand is this frame.
    script: Option<scripted::Script>,
    scripted_at: Option<Vec2>,
    fps_timer: std::time::Instant,
    fps_frames: u64,
    last_fps: f32,
    /// Per-particle enthalpy state (see `water_phase_chain`), the single
    /// authoritative thermal state; `particles.temperature` is derived from it every
    /// frame, not the other way around.
    enthalpy: Vec<f32>,
    ice_material: RankineMaterial,
    water_material: CavitatingFluidMaterial,
    boiling_material: BoilingMixtureMaterial,
    steam_material: IdealGasMaterial,
    /// Temporary diagnostic: which water particle held `detF` max at the last
    /// sample, to tell one persisting particle from a rotating cast (see the
    /// tracking block).
    water_jmax_prev_idx: Option<usize>,
    /// Disclosed capture aid -- see `CaptureState`'s doc. `None`
    /// (default, every normal run) unless `PHASE_STATES_CAPTURE_DIR` is set.
    capture: Option<CaptureState>,
    /// Which render path draws the frame, cycled with G.
    render_mode: RenderMode,
    /// GPU buffers the grid-volume and surface modes read, rebuilt from the
    /// CPU solver on the frames those modes are shown.
    render_bridge: CpuRenderBridge,
}

/// Draws one frame of `sim` into `target` through `mode`'s render path: the
/// swapchain view and the capture texture both go through here, so a capture
/// shows what the window does.
fn draw_frame(
    renderer: &mut Renderer,
    bridge: &mut CpuRenderBridge,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    sim: &Simulation,
    mode: RenderMode,
    target: &wgpu::TextureView,
) {
    match mode {
        RenderMode::Particles => renderer.render(device, queue, sim.particles(), target, true),
        RenderMode::GridVolume => {
            bridge.upload_grid(queue, sim.particles(), sim.grid());
            renderer.render_grid_volume(device, queue, bridge.grid_volume_source(), target, true);
        }
        RenderMode::Surface => {
            bridge.upload_particles(device, queue, sim.particles());
            // Per-material colouring: each phase keeps its own optics slot
            // where the surfaces meet.
            renderer.render_surface_reconstruction(
                device,
                queue,
                bridge.surface_source(ICE_ID, true, sim.mean_substep_dt()),
                target,
                true,
            );
        }
    }
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
        // Frame capture, opt-in via `PHASE_STATES_CAPTURE_DIR` (a directory,
        // created if missing). `PHASE_STATES_CAPTURE_FRAMES` (default 150) sets how
        // many frames to capture, `PHASE_STATES_CAPTURE_STRIDE` (default 3, one
        // captured frame per 3 sim frames at 60 fps -> 20 fps) the spacing. The
        // process exits once the target count is reached. Writes one continuous raw
        // RGBA8 stream (`frames.raw`, what `ffmpeg -f rawvideo` reads, see
        // `CaptureState`) plus a `format.txt` sidecar with the detected pixel format
        // and size, since the swapchain format is picked at runtime (`fmt` above) and
        // ffmpeg needs the matching `-pixel_format`/`-video_size`.
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
        // Marks the materials that hold their shape (a nonzero shear
        // modulus), which the grid-volume and surface modes draw flat; also
        // picks up any measured optics a material declares.
        renderer.adopt_material_optics(&queue, sim.materials());
        // The grid-volume and surface modes threshold on cell mass as a
        // fraction of a full cell of the starting ice, which holds
        // 1/SPACING^2 particles; steam, a sixth as dense, reads thinner.
        renderer.set_grid_reference_cell_mass(sim.particles().mass[0] / (SPACING * SPACING));
        renderer.set_surface_res_multiplier(SURFACE_RES_MULTIPLIER);
        let render_bridge = CpuRenderBridge::new(&device, GRID);

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
            "phase_states_gui: {} particles  |  drag Target temperature to heat/cool  |  LMB push  RMB pull  |  G render mode  R reset  Q quit",
            sim.particles().len()
        );
        // A scripted run (see `gui_common/scripted.rs`): the hand pushes into
        // the ice column, then just below it, then pulls beside it, forty
        // frames each, with the scene left alone in between.
        let script = scripted::Script::from_env(
            vec![
                scripted::Press {
                    at: Vec2::new(32.0, 14.0),
                    from: 60,
                    to: 100,
                    pull: false,
                },
                scripted::Press {
                    at: Vec2::new(32.0, 6.0),
                    from: 160,
                    to: 200,
                    pull: false,
                },
                scripted::Press {
                    at: Vec2::new(38.0, 6.0),
                    from: 260,
                    to: 300,
                    pull: true,
                },
            ],
            360,
            (Vec2::ZERO, Vec2::splat(GRID as f32)),
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
            // Kept low by measurement. A one-way heat-up at gravity_fraction=1.0 is
            // fine, but a 20-minute repeated heat/cool cycle
            // (`PHASE_STATES_AUTO_CYCLE_PERIOD_S`) at 1.0 fails structurally: `avg_T`
            // climbs to 385.456 K and stays there for the rest of the run (~60,000+
            // frames, several full plate cycles between ~250 K and ~418 K) while
            // `plate_temperature` keeps cycling. The heat exchange below applies only
            // to particles with `x.y <= HEATING_PLATE_TOP_Y` (a spatially blind heat
            // source gives no vertical DeltaT, hence no convection), and at full
            // gravity buoyancy lifts the whole steam population above that zone for
            // good, where its enthalpy stops updating, like gas convected away from a
            // stove. So the ice<->water<->steam cycle does not hold at
            // gravity_fraction=1.0 once the population vents: the same missing ambient
            // medium as `AMBIENT_AIR_RHO_KG_M3` (nothing for a vented particle to
            // exchange momentum or heat with). Reproduce with
            // `PHASE_STATES_AUTO_CYCLE_PERIOD_S` (see its doc).
            gravity_fraction: script.as_ref().map_or(0.01, |s| s.gravity(0.01)),
            cursor_pos: [0.0; 2],
            lmb: false,
            rmb: false,
            push_strength: 10.0,
            frame: 0,
            script,
            scripted_at: None,
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
            render_mode: RenderMode::Particles,
            render_bridge,
        }
    }

    fn cursor_grid(&self) -> Vec2 {
        if let Some(at) = self.scripted_at {
            return at;
        }
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
        // Live-adjustable gravity -- see this struct's own
        // real_gravity/gravity_fraction doc and the gravity slider below.
        let live_gravity = self.real_gravity * self.gravity_fraction;
        self.sim.set_gravity(live_gravity);

        // Simple proportional heater/cooler driving toward the
        // slider's own target -- see HEAT_GAIN's doc.
        // Verification aid: drives the target-temperature slider from an env var
        // instead of a live drag, so a run can be checked from its log with nobody
        // at the keyboard. Inactive unless the env var is set.
        if let Ok(target) = std::env::var("PHASE_STATES_AUTO_HEAT") {
            self.target_temperature = target.parse().unwrap_or(400.0);
        }
        // Verification aid: overrides the default `gravity_fraction=0.01` once at
        // startup, to run the scene at full Earth, Moon or Mars gravity, which the
        // water and ice stiffness above is derived for.
        if let Ok(g) = std::env::var("PHASE_STATES_GRAVITY_FRACTION")
            && let Ok(g) = g.parse::<f32>()
        {
            self.gravity_fraction = g;
        }
        // Verification aid: ramps the slider target at a human dragging pace
        // instead of jumping it at once like the auto-heat variable above (an
        // instant 150 K gap saturates the 80 K/s `MAX_HEAT_RATE_K_PER_S` cap from
        // the first frame either way), to tell an engine issue from an artifact
        // of instant heating.
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
        // Verification aid: a repeating bidirectional triangle-wave target (heat
        // to `PHASE_STATES_AUTO_HEAT` or 420 K over the first half of
        // `PHASE_STATES_AUTO_CYCLE_PERIOD_S` sim seconds, cool back to 250 K over
        // the second), where the other AUTO_HEAT variants only heat. It shows the
        // full-gravity failure described at `gravity_fraction` (combine with
        // `PHASE_STATES_GRAVITY_FRACTION=1.0`), which is why the default stays at
        // 0.01. Uses sim time (`self.sim.config().dt * self.frame`), not
        // wall-clock, so the period does not depend on fps.
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
        // The plate ramps toward the slider's setpoint (`target_temperature`)
        // at a bounded rate (thermal inertia), a proportional controller on the
        // plate's own state, not on the population's average (see
        // `plate_temperature`).
        let plate_rate = (HEAT_GAIN * (self.target_temperature - self.plate_temperature))
            .clamp(-MAX_HEAT_RATE_K_PER_S, MAX_HEAT_RATE_K_PER_S);
        self.plate_temperature += plate_rate * dt;
        // Temporary diagnostic: checks that the enthalpy method produces a
        // plateau at each transition point.
        if self.frame.is_multiple_of(30) {
            println!(
                "[enthalpy-check] frame={} avg_T={current_avg:.3}K plate_T={:.3}K",
                self.frame, self.plate_temperature
            );
        }

        // The heating is an energy rate (`dH = cp*rate*dt`), not a direct
        // temperature increment, so a plateau appears at each transition: energy
        // keeps flowing in and raises `phase_fraction` instead of `T`, like a
        // heating element under a pot of boiling water.
        // `chained_state_from_enthalpy` (Voller & Cross 1981, extended to two
        // consecutive transitions) derives both the temperature and the phase (or
        // transition band) the particle is in.
        //
        // Each particle's rate is a local contact flux,
        // `HEAT_GAIN*(plate_temperature-particle_temperature)` (Newton's law of
        // cooling), self-limiting: a particle at or above `plate_temperature` gets
        // zero or negative flux (see `plate_temperature`).
        let phase_chain = water_phase_chain();
        {
            let particles = self.sim.particles_mut();
            for i in 0..particles.len() {
                // Only particles in the heating-plate region (the bottom of the
                // domain) receive energy directly; everything above warms by bulk
                // transport (particles carrying their enthalpy as they move,
                // driven by the Boussinesq force below). A uniform source gives
                // DeltaT_vertical~=0, hence Ra~=0, hence no convection however the
                // rest of the physics is tuned.
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
                // Writes the `fraction` that `material_id_for_phase_state` (below)
                // uses to pick `BOILING_ID` into `Particle::friction_hardening`,
                // which `BoilingMixtureMaterial` reads (see its doc), every frame
                // since `fraction` keeps climbing as `enthalpy` accumulates. Not
                // for `PhaseState::Melting`: RankineMaterial keeps its damage state
                // there while a particle is ICE_ID.
                if let emerge::thermodynamics::PhaseState::Boiling { fraction } = state {
                    particles.friction_hardening[i] = fraction;
                }
            }
        }

        // Material transitions from the enthalpy-derived phase state, not a
        // temperature threshold (which cannot see `phase_fraction`, so cannot tell
        // "just started melting" from "fully melted"). Replicates
        // `Simulation::apply_phase_transition`'s rebaseline by hand (F->identity,
        // initial_volume/density from the current volume, then the target
        // material's `init_particle_from_transition`), since the phase-fraction
        // state is not visible to the `Particle`-only predicate
        // `Simulation::phase_transition`/`add_phase_rule` take.
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

        // Archimedes buoyancy (the BuoyancyField formula, see its doc), for steam
        // only. Buoyancy needs a particle surrounded by a different-density fluid:
        // applied to every particle it gives the starting ice block (rho=917,
        // below the water reference 1000) a net upward nudge from the first frame,
        // before any water exists. Steam cannot exist without water around it
        // first (boiling needs water), so gating on STEAM_ID avoids that; ice and
        // water fall under the solver's plain gravity.
        {
            // Medium selection: the steam buoyancy below is an Archimedes density
            // contrast against an effective reference medium -- water while water
            // or boiling-mixture neighbors are nearby, an effective ambient-air
            // stand-in once they are not (see `AMBIENT_AIR_RHO_KG_M3` for what that
            // stand-in is and is not). The water-relative formula everywhere would
            // keep targeting a nonzero rise velocity (~13.6 grid-units/s at 431 K)
            // deep inside an all-steam region, driving dispersion and CFL cost;
            // zero buoyancy once no condensed neighbor remains would take the lift
            // from the whole population at once when it fully vaporizes (steam
            // n=240 at frame ~4200 in a headless run). An effective demo closure, not
            // simulated ambient air. Computed in an immutable pass before the mutable
            // particle borrow below (`count_near` needs `&Simulation`): one entry per
            // particle (0 for non-steam), at the radius the same-material neighbor
            // diagnostics use.
            const BUOYANCY_NEIGHBOR_RADIUS: f32 = 3.0;
            // Counts BOILING_ID neighbors as well as WATER_ID: a steam particle
            // rising out of a mid-boil mixture is still surrounded by the condensed
            // medium the Archimedes formula assumes, and counting pure liquid only
            // would switch to the air stand-in too early at a boiling interface.
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
            // Relax toward the analytical terminal velocity where drag balances
            // buoyancy (`k*v_terminal = a`, the force balance of rising-bubble and
            // vapor-parcel terminal velocity, drag linear in velocity as in the
            // Stokes regime): `v += (v_terminal - v) * min(k*dt, 1)` never
            // overshoots `v_terminal` however large the buoyancy multiplier gets
            // (up to ~120x base gravity as steam expands and `rho` shrinks toward the
            // volume-ratio ceiling). Adding the kick then decaying it
            // (`v += kick; v *= 1-k*dt`) lands the full spike first, faster than the
            // CFL scan can react, and fps collapses.
            const STEAM_RISE_DRAG_COEFFICIENT: f32 = 5.0;
            // Buoyancy from temperature, not from `particles.density[i]`
            // (`rest_density/J`): reading the particle's own resolved volume feeds its
            // drift back into the force that pushes it, an unbounded positive
            // feedback (over-expansion -> lower density -> more buoyancy -> more
            // expansion, spreading to neighbors through the velocity-gradient
            // divergence) independent of temperature or heating rate: substeps
            // climb without bound while temperature sits saturated at the target.
            //
            // Boussinesq approximation (standard in atmospheric and oceanic
            // convection: buoyancy from thermal density contrast at a fixed reference
            // pressure, decoupled from the resolved compressible state):
            // `rho(T) = p_ref / (R_specific * T)`, the ideal-gas relation
            // `STEAM_SPECIFIC_GAS_CONSTANT_J_KG_K` is derived from (at
            // T=BOILING_POINT_K it gives STEAM_RHO_KG_M3 exactly). Temperature
            // converges smoothly to the heater target, so hotter steam stays more
            // buoyant without the feedback path through J.
            const STANDARD_ATMOSPHERE_PA: f32 = 101_325.0;
            for (i, &water_neighbors) in steam_water_neighbors.iter().enumerate() {
                if particles.material_id[i] != STEAM_ID {
                    continue;
                }
                // Medium selection (see this block's doc): water Archimedes while
                // condensed-phase neighbors remain, ambient-air Archimedes once they
                // do not. Never zero: a steam particle always has some medium
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

            // Boussinesq buoyancy for water: warmer water is less dense and rises,
            // the driving force of natural (Rayleigh-Benard) convection. The linear
            // approximation `a_b = -g*beta*(T-T_ref)` is also the one the critical
            // Rayleigh numbers (Ra_c~=1707.76 no-slip, ~657.5 stress-free) are derived
            // under, so those benchmarks apply. `T_ref=MELTING_POINT_K`: water at its
            // melting point is the coldest, densest liquid state in the scene, so all
            // other water is buoyant against it, like a pot heated from 0C.
            //
            // Water's thermal expansion coefficient depends strongly on temperature
            // (CRC Handbook / NIST water data): ~2.1e-4/K near 20C, ~4.6e-4/K near
            // 50C, ~7.5e-4/K near 100C. One constant, at ~50C (323 K), roughly the
            // middle of the scene's liquid range (like the single cp per phase).
            const WATER_THERMAL_EXPANSION_COEFF_PER_K: f32 = 4.6e-4;
            // The same bounded relax-toward-terminal-velocity form as steam, which
            // cannot overshoot however large `beta*delta_t` gets. A raw `v += a*dt`
            // with the per-frame dt (much coarser than the solver's adaptive
            // substep) lands before the CFL scan can react and pins water at its
            // compression floor (`detF=[0.500,0.500]` for 11+ seconds).
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

        // Cursor push/pull -- the mechanism basic_snow.rs/basic_fluids.rs use (a
        // velocity-space impulse, not restricted by the strict-WC-MPM-fluid checks
        // above, which gate pinning/contact/mixture/sleep/boundary/apic_blend, never
        // impulses or force fields).
        if let Some(script) = &self.script {
            // The hand holds the cursor and the button; the push itself is
            // this demo's own, at its strongest setting.
            let hand = script.hand(self.frame);
            self.lmb = hand.is_some_and(|(_, pull)| !pull);
            self.rmb = hand.is_some_and(|(_, pull)| pull);
            self.scripted_at = hand.map(|(at, _)| at);
            self.push_strength = PUSH_STRENGTH_MAX;
        }
        if self.lmb || self.rmb {
            let mag = if self.lmb {
                self.push_strength
            } else {
                -self.push_strength
            };
            self.sim.apply_radial_impulse(self.cursor_grid(), 6.0, mag);
        }

        self.sim.step();

        if let Some(script) = &mut self.script {
            let done = script.record(
                self.frame,
                self.sim.config().dt,
                &self.sim,
                &[
                    (ICE_ID, "ice"),
                    (WATER_ID, "water"),
                    (STEAM_ID, "steam"),
                    (BOILING_ID, "boiling"),
                ],
                &[
                    ("gravity", self.gravity_fraction),
                    ("push", self.push_strength),
                ],
            );
            if done {
                std::process::exit(0);
            }
        }

        // Temporary diagnostic: max |off-diagonal| of F by material. Fluid and gas
        // materials re-diagonalize F to an isotropic scale every substep
        // (`update_particle`), so a nonzero water/steam value would be an engine
        // bug, and zero means visible "rotation" is bulk circulation
        // (gravity+buoyancy+cursor), not a per-particle artifact.
        if self.frame.is_multiple_of(60) {
            let particles = self.sim.particles();
            let mut max_offdiag = [0.0_f32; 3]; // [ice, water, steam]
            let mut max_det = [f32::MIN; 3];
            let mut min_det = [f32::MAX; 3];
            // Diagnostic: is RankineMaterial's Kelvin-Voigt damping dissipating
            // energy (avg/max ice speed should decay toward rest after an impact),
            // and is damage accumulating sanely (RankineMaterial keeps its damage in
            // `friction_hardening`, 0=intact, saturating around ~1.5 at this preset's
            // softening_rate=2.0, see `rankine_damage_saturation_point`)?
            let (mut ice_speed_sum, mut ice_n, mut ice_max_speed) = (0.0_f32, 0usize, 0.0_f32);
            let (mut ice_damage_sum, mut ice_max_damage) = (0.0_f32, 0.0_f32);
            // Diagnostic: `IdealGasMaterial::timestep_bound`'s acoustic bound uses a
            // fixed reference_temperature_k, not the live temperature. If that drives
            // the substep growth, live temperature should climb steadily above
            // BOILING_POINT_K=373.15 as heating continues past the transition.
            let (mut steam_temp_sum, mut steam_n, mut steam_max_temp) = (0.0_f32, 0usize, 0.0_f32);
            // Diagnostic: `IdealGasMaterial::update_particle` clamps `f_trial`'s
            // determinant to `[volume_ratio_min, volume_ratio_max]` every substep with
            // no record of the pre-clamp value, so the 20.000 ceiling in `detF` above
            // could be a mild excursion or a violent one. `trace(velocity_gradient)`
            // (the APIC C matrix `f_trial=(I+dt*C)*F` is built from) measures it
            // without the solver's per-substep dt.
            let mut steam_max_abs_c_trace = 0.0_f32;
            // Diagnostic: vorticity of the velocity field, the antisymmetric half of
            // the velocity gradient (divergence, via trace(C) above, is the symmetric
            // half). The F off-diagonal check above rules out a particle's own shape
            // twisting, not the field swirling. A rising, expanding parcel has a real
            // wake, so the question is whether "rotation" at liftoff is fluid
            // vorticity or particles ejected in different directions from one crowded
            // spot. omega = (dvy/dx - dvx/dy)/2 = (C.x_axis.y - C.y_axis.x)/2.
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
            // `min_det`/`max_det` fold over the slot's particles from
            // `f32::MAX`/`f32::MIN`, so an empty slot (e.g. no water or steam yet
            // at frame 0) would print the sentinel (~3.4e38). NaN instead, the
            // "no data" convention of `steam_avg_temp` above, rather than a 0.0
            // that would read as a collapsed J=0.
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
            // Temporary diagnostic: tracks whichever water particle holds the
            // current `detF` max (one persisting outlier or a rotating population)
            // and whether it is under tension (`pressure_gauge<0`), the state the
            // cavitation EOS's mixture and vapor branches handle.
            if let Some((max_idx, max_j)) = particles
                .iter()
                .enumerate()
                .filter(|(_, p)| p.material_id == WATER_ID)
                .map(|(i, p)| (i, p.deformation_gradient.determinant()))
                .max_by(|a, b| a.1.total_cmp(&b.1))
            {
                let p = particles.get(max_idx);
                // `rho_grid/J = rest_density_grid/max_j` with
                // `rest_density_grid = rho_l_ref*dx^2`, so the SI density is
                // `rho_l_ref/max_j` (`dx` cancels), as in
                // `CavitatingFluidMaterial`'s private `real_density_si`. The
                // pressure uses this particle's own `temperature`.
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
            // Temporary diagnostic of `BoilingMixtureMaterial`'s core claim: a
            // mid-boil particle's mechanical `J` tracks the mass-fraction
            // equilibrium `J_eq(x)=1+(rho_l_ref/rho_v_ref-1)*x` its doc derives
            // (`J/J_eq->1`), rather than running free (e.g. `J=5.988`, nearly full
            // vapor, at `x_H<0.5`).
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
                // `J/J_eq != 1` is not automatically a bug: a column under gravity
                // needs internal pressure to hold its weight, and this material's
                // `p = c_mix2(x)*(rho-rho_eq(x))` computes it. Converts the residual
                // into Pa and compares it with a hydrostatic estimate
                // (`p ~= rho*g*(y_surface-y)`) at the particle's depth: matching
                // orders of magnitude mean physics, not drift. `y_surface` is this
                // frame's top of the condensed-phase column (max y over
                // ICE_ID/WATER_ID/BOILING_ID; steam is not part of the medium).
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
            // Temporary diagnostic: for the worst (max detF) steam particle, `J`
            // against the analytic `J_eq(T)` where `p_gauge=0`, and its neighbor and
            // grid-occupancy picture. The rendered size cannot answer this:
            // `render_particles.wgsl` draws from `F`, blind to
            // `initial_volume`/`volume`, so a reference rebase at the water->steam
            // transition reads as a size change. `J/J_eq->1` and `p_gauge->0` while
            // positions still disperse means forcing without a medium plus
            // under-sampling, not an EOS runaway; `J/J_eq` still growing while
            // `p_gauge<0` means the mechanical equilibrium is missed (next: a G2P
            // occupied-vs-empty-node contribution check).
            if let Some((max_idx, max_j)) = particles
                .iter()
                .enumerate()
                .filter(|(_, p)| p.material_id == STEAM_ID)
                .map(|(i, p)| (i, p.deformation_gradient.determinant()))
                .max_by(|a, b| a.1.total_cmp(&b.1))
            {
                let p = particles.get(max_idx);
                let temperature = p.temperature.max(1.0);
                // Exact mechanical-equilibrium J (p_gauge=0): from
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
            // Diagnostic: with `EMERGE_CFL_DIAGNOSE=2` in a `research-diagnostics` build (see
            // `cfl::diagnose_worst_particle_cfl_term`), particle #15 is the most
            // constrained particle in ~66% of 15123 samples, one particle in an
            // escalating runaway rather than a diffuse steam effect. Tracks its state:
            // near a wall (where reflected/slip forces could compound), an early
            // outlier, or in a sparse neighborhood (the condition the
            // single-particle-instability paper describes, though that paper's
            // bound is not the binding term here)?
        }
        // Go/no-go gate before any two-way mechanical/thermal closure for
        // `BoilingMixtureMaterial`: closing the pressure->phase loop without knowing
        // whether the residual is large, persistent and depth-coherent could turn
        // P2G/boundary discretization noise into fake enthalpy and vapor-quality
        // changes. Covers the whole `BOILING_ID` population, not one outlier: each
        // particle's gauge pressure is made absolute, the IAPWS-IF97 Region 4
        // saturation curve is inverted for its saturation temperature
        // (`water_saturation_temperature_from_pressure_k`), and
        // `delta_T_sat = T_sat(p_abs) - BOILING_POINT_K` is reported as
        // median/p10/p90 with its sign. Also bins by depth (shallow/mid/deep thirds
        // of the condensed-phase column) and reports the population's average
        // |div(v)| (`trace(velocity_gradient)`) and |v|: a residual significant only
        // at high velocity or divergence is a dynamic, numerical signal, not durable
        // thermodynamic pressure.
        //
        // Decision metric: `epsilon_x_equiv = cp_liquid*|delta_T_sat|
        // /vaporization_latent_heat`, the vapor-quality change the residual would
        // cause if mechanical and thermal states were coupled (they are not; this
        // material is one-directional). With `WATER_HEAT_CAPACITY_J_KG_K=4182` and
        // `VAPORIZATION_LATENT_HEAT_J_KG=2_257_000`, `epsilon_x_equiv ~= 0.00185/K`:
        // 1 K of `delta_T_sat` is ~0.185% quality, 10 K ~1.85%.
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
        // Temporary diagnostic: particle 15 accelerates from |v|=5.3 (frame 60,
        // already water) to |v|=65.6 (frame 120), so it is not a
        // transition-instant spike. Every-frame resolution across that window
        // shows when and how fast the acceleration happens.
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
        draw_frame(
            &mut self.renderer,
            &mut self.render_bridge,
            &self.device,
            &self.queue,
            &self.sim,
            self.render_mode,
            &view,
        );

        // Disclosed capture aid -- see `CaptureState`'s doc.
        // Renders a SECOND time into the dedicated offscreen capture
        // texture (not the swapchain view above) so the readback below has
        // a `COPY_SRC`-capable source. Self-terminates once
        // `target_frames` is reached -- a non-interactive, fully
        // automatic capture run.
        if let Some(cap) = &mut self.capture
            && cap.captured < cap.target_frames
            && self.frame.is_multiple_of(cap.stride)
        {
            let capture_view = cap
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default());
            draw_frame(
                &mut self.renderer,
                &mut self.render_bridge,
                &self.device,
                &self.queue,
                &self.sim,
                self.render_mode,
                &capture_view,
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
                    // 600 K: under water's IAPWS-IF97 critical point (647.096 K,
                    // `water_saturation::WATER_CRITICAL_POINT_K`), past which the
                    // liquid/vapor distinction the chained enthalpy/phase-state
                    // model relies on stops meaning anything. `IdealGasMaterial`'s
                    // acoustic CFL bound uses a fixed `reference_temperature_k`, not
                    // the live temperature, so the substep count can climb well
                    // before 600 K: cost, not a crash.
                    ui.add(egui::Slider::new(&mut target_temperature, 150.0..=600.0));
                    ui.separator();
                    ui.label("Gravity (1.0 = real IRL 9.81 m/s²):");
                    ui.add(egui::Slider::new(&mut gravity_fraction, 0.0..=1.0));
                    ui.separator();
                    ui.label("Push/pull strength:");
                    ui.add(egui::Slider::new(&mut push_strength, 0.0..=PUSH_STRENGTH_MAX));
                    ui.separator();
                    ui.label(
                        "Real bidirectional phase transitions: drag Target temperature up to \
                         melt then boil, down to condense then freeze -- same real latent-heat \
                         hysteresis as phase_states_headless.rs, just under your own control. \
                         Real gravity: ice falls and water pools. Real Archimedes \
                         buoyancy: steam rises once it exists.",
                    );
                    ui.label("LMB push  RMB pull  G render  R reset  Q quit");
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
                    KeyCode::KeyG if pressed => {
                        s.render_mode = s.render_mode.next();
                        println!("render mode: {}", s.render_mode.label());
                    }
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

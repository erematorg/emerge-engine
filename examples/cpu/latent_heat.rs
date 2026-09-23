extern crate emerge_engine as emerge;

/// Headless demo of `MaterialModel::latent_heat()` -- water cooling below freezing
/// transitions to ice via `add_phase_rule`, and the transition itself debits/credits
/// real thermal energy (`temperature -= latent_heat / heat_capacity`), not just an
/// instant, energy-free material_id swap.
///
/// Ice's `latent_heat = -334.0` (negative = exothermic): freezing releases energy,
/// so newly-frozen particles warm up slightly at the moment of transition -- the same
/// real effect that slows down a freezing pond (latent heat release fights further
/// cooling) instead of letting it crash straight to ambient temperature.
///
/// Real, physically-forced scale choice, not an arbitrary demo number: real water's
/// thermal diffusivity (alpha=k/(rho*c_p)~1.434e-7 m^2/s) makes pure-conduction
/// freezing genuinely take MONTHS across a 1-meter slab (tau~L^2/alpha) --
/// shrinking `dx_meters` (tau~dx^2) is the real fix to show freezing in any
/// reasonable step count.
///
/// `alpha_grid()` must use `k/(rho*c_p)`, not `k/c_p` -- omitting density makes
/// every scene diffuse far too fast for a dense material (water, rho=1000). The
/// thermal-CFL bound scales with 1/alpha, so a smaller (correct) alpha makes a
/// small dx_meters safer, not riskier.
///
/// This scene's real substep dt is ~0.0277s regardless of dx_meters (elastic
/// CFL doesn't depend on dx here); `max_substeps_per_step` above the engine's
/// own default (64) makes no difference.
///
///   cargo run --example latent_heat
use emerge::thermodynamics::{ThermalConfig, ThermalDiffusion};
use emerge::{NeoHookeanMaterial, SimConfig, Simulation, SpawnRegion, WithLatentHeat};
use glam::{IVec2, Vec2};

const WATER_ID: u32 = 0;
const ICE_ID: u32 = 1;
const FREEZING_POINT: f32 = 273.0;
const ICE_LATENT_HEAT: f32 = -334_000.0; // exothermic: freezing releases energy (real water: 334 kJ/kg)
const HEAT_CAPACITY: f32 = 4182.0;

fn main() {
    let config = SimConfig {
        gravity: Vec2::ZERO, // isolate the thermal/phase effect from settling dynamics
        ..SimConfig::earth(64, 0.001, 1.0)
    };

    let thermal = ThermalDiffusion::new(
        ThermalConfig {
            conductivity: 0.6,
            heat_capacity: HEAT_CAPACITY,
            density: 1000.0, // kg/m^3, real water -- see ThermalConfig::density's own doc
            ambient: 250.0,  // below freezing -- the whole slab cools toward this
            // Must match the sim's real dx_meters -- ThermalConfig::grid_cell_size
            // requires this, else alpha_grid() is mis-scaled by orders of
            // magnitude.
            grid_cell_size: config.dx_meters,
            ..Default::default()
        },
        config.grid_res,
    );

    let spawn = SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new(20, 20),
        box_center: Vec2::splat(config.grid_res as f32 * 0.5),
        material_id: WATER_ID,
        ..SpawnRegion::for_sim(&config)
    };

    // Grid-native mechanical placeholder, not a real water/ice elastic
    // modulus -- water has no meaningful shear modulus to cite, and this
    // scene's real, physically-forced claims (module doc above) are entirely
    // thermal (alpha, dx_meters, latent heat). NeoHookean exists here only
    // so MPM particles have SOME cohesion; it isn't modeling real mechanics,
    // and the module doc's own note that elastic CFL doesn't gate this
    // scene's substep dt confirms the value is inert either way.
    let water = NeoHookeanMaterial::new(10.0, 20.0);
    let ice = WithLatentHeat::new(NeoHookeanMaterial::new(10.0, 20.0), ICE_LATENT_HEAT);

    let mut solver = Simulation::new(config, spawn)
        .with_default_material(Box::new(water))
        .with_material(ICE_ID, Box::new(ice))
        .with_thermal(thermal)
        .with_phase_rule(|p| {
            if p.material_id == WATER_ID && p.temperature < FREEZING_POINT {
                Some(ICE_ID)
            } else {
                None
            }
        });

    // Start everything warm, well above freezing.
    for t in solver.particles_mut().temperature.iter_mut() {
        *t = 300.0;
    }

    println!(
        "Cooling a warm ~2cm water droplet (dx_meters=0.001) toward ambient=250.0, \
         well below freezing=273.0."
    );
    println!(
        "Ice's latent_heat={ICE_LATENT_HEAT} (exothermic) should produce a visible warm bump \
         right at the freezing transition, before resuming its cool toward ambient.\n"
    );

    let count_of = |sim: &Simulation, id: u32| {
        sim.particles()
            .iter()
            .filter(|p| p.material_id == id)
            .count()
    };
    let avg_temp_of = |sim: &Simulation, id: u32| {
        let (sum, n) = sim
            .particles()
            .iter()
            .filter(|p| p.material_id == id)
            .fold((0.0, 0usize), |(s, n), p| (s + p.temperature, n + 1));
        if n == 0 { f32::NAN } else { sum / n as f32 }
    };

    for step in 1..=1000u64 {
        solver.step_n(1);
        if step % 50 == 0 {
            let water_n = count_of(&solver, WATER_ID);
            let ice_n = count_of(&solver, ICE_ID);
            println!(
                "step={step:4}  water={water_n:4} (avg T={:6.2})  ice={ice_n:4} (avg T={:6.2})",
                avg_temp_of(&solver, WATER_ID),
                avg_temp_of(&solver, ICE_ID),
            );
        }
    }

    println!(
        "\nDone -- watch the ice avg-T column: it should jump up right after ice first \
         appears (latent heat release), then resume cooling toward ambient=250.0."
    );
}

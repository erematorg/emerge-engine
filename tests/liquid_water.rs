//! Liquid water as a material (`LiquidWaterMaterial`): its coefficients are
//! the IAPWS SR6-08 record's at the temperature it is built at, nothing
//! defaulted, and only the total Lagrangian path carries it.

extern crate emerge_engine as emerge;

use emerge::materials::MaterialModel;
use emerge::materials::liquid::liquid_water::{LiquidWater, VALID_K};
use emerge::{LiquidWaterMaterial, ParticleMass, SimConfig, Simulation, SpawnRegion};
use glam::{IVec2, Vec2};

/// Outside the record's range there is no material; inside, its density,
/// heat capacity and modulus are the record's at that temperature.
#[test]
fn the_material_is_the_record_at_its_temperature() {
    let config = SimConfig::earth(24, 0.01, 1.0 / 60.0);
    for t in [VALID_K.0 - 0.01, VALID_K.1 + 0.01] {
        assert!(LiquidWaterMaterial::new(t, &config).is_none(), "{t} K");
    }
    for t in [VALID_K.0, 293.15, VALID_K.1] {
        let water = LiquidWaterMaterial::new(t, &config).expect("in range");
        let record = LiquidWater::at(t).expect("in range");
        assert_eq!(water.bulk_modulus(), record.isentropic_bulk());
        assert_eq!(
            water.rest_density(),
            Some(config.density_from_si(record.density() as f32))
        );
        assert_eq!(water.specific_heat_j_kg_k(), record.cp as f32);
        assert_eq!(
            water.particle_mass(0.5, &config),
            record.density() as f32 * (0.5 * config.dx_meters).powi(2)
        );
        assert!(water.gpu_unsupported_reason().is_some());
    }
}

/// The explicit path has no barotropic law: it refuses the material
/// instead of running another law in its place.
#[test]
#[should_panic(
    expected = "is the barotropic liquid water, which only the total Lagrangian solid path carries"
)]
fn the_explicit_path_refuses_water() {
    let config = SimConfig::earth(24, 0.01, 1.0 / 60.0);
    let water = LiquidWaterMaterial::new(293.15, &config).expect("in range");
    let mut sim = Simulation::empty(config);
    let id = sim.register_material(Box::new(water)).0;
    let _water = sim.add_body(
        SpawnRegion::for_sim(&config)
            .box_of(IVec2::new(4, 4))
            .at(Vec2::new(12.0, 8.0))
            .spacing(0.5)
            .material(id)
            .mass_from(&water, &config),
    );
    sim.step();
}

//! The contact points' start velocity (`build_total_lagrangian_state`).

use super::*;
use crate::materials::physical_props::Elastic as ElasticProps;
use crate::{CorotatedMaterial, FromSI, SimConfig};
use glam::IVec2;

/// A body spawned moving: its contact points start at its velocity,
/// not at rest (they have no mass, so no velocity of their own).
#[test]
fn a_body_spawned_moving_carries_its_contact_points() {
    let mut config = SimConfig::earth(40, 0.01, 1.0 / 60.0);
    config.total_lagrangian = Some(TotalLagrangianConfig::new(1e-2));
    let props = ElasticProps {
        e_pa: 1.125e7,
        nu: 0.3,
        rho_kg_m3: 1700.0,
    };
    let mut sim = Simulation::empty(config);
    let id = sim
        .register_material(Box::new(CorotatedMaterial::from_physical(&props, &config)))
        .0;
    let floor = config.boundary_thickness as f32;
    let _block = sim.add_body(
        SpawnRegion::for_sim(&config)
            .box_of(IVec2::new(8, 4))
            .at(Vec2::new(20.0, floor + 5.0))
            .spacing(0.5)
            .material(id)
            .mass_from(&props, &config),
    );
    let spawned = Vec2::new(30.0, -140.0);
    for v in sim.particles_mut().v.iter_mut() {
        *v = spawned;
    }
    let state = sim.build_total_lagrangian_state((&ConservingSettings::default(), false));
    let expected = GridUnits::of(&config).velocity(spawned);
    assert!(
        state.x.len() > state.particles,
        "the body has contact points"
    );
    for q in state.particles..state.x.len() {
        assert!(
            (state.v[q] - expected).length() <= 1e-10 * expected.length(),
            "contact point {q}: {:?} against {expected:?}",
            state.v[q]
        );
    }
}

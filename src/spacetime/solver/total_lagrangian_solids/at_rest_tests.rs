//! A body at rest on the floor, its contact points at rest: the walls
//! settle its contact (`contact::gap_resolution`) and the step controller
//! keeps the frame's one step.

use super::*;
use crate::materials::physical_props::Elastic as ElasticProps;
use crate::{CorotatedMaterial, FromSI, SimConfig};
use glam::IVec2;

/// The 8 x 4 cm clay block on the floor, placed at its resting
/// compression with its constrained modes (`rest::Rest::settle`), the
/// loaded mode given `kick` of its offset as its rate.
pub(super) fn settled_block(kick: f64) -> Simulation {
    let props = ElasticProps {
        e_pa: 1.125e7,
        nu: 0.3,
        rho_kg_m3: 1700.0,
    };
    let mut config = SimConfig::earth(40, 0.01, 1.0 / 60.0);
    config.total_lagrangian = Some(TotalLagrangianConfig::new(1e-1));
    let mut sim = Simulation::empty(config);
    let id = sim
        .register_material(Box::new(CorotatedMaterial::from_physical(&props, &config)))
        .0;
    let floor = config.boundary_thickness as f32;
    let _block = sim.add_body(
        SpawnRegion::for_sim(&config)
            .box_of(IVec2::new(8, 4))
            .at(Vec2::new(20.0, floor + 2.0))
            .spacing(0.5)
            .material(id)
            .mass_from(&props, &config),
    );
    for _ in 0..2 {
        sim.step();
    }
    let planes = sim.total_lagrangian_walls().0;
    let gravity = sim.total_lagrangian_gravity();
    let mut state = sim.total_lagrangian.take().expect("on the path");
    let cache = state.cache.as_ref().expect("stepped");
    let basis = rest::Basis::build(&state, (&cache.grid, &planes, 0.0)).expect("a basis");
    let Ok(rest) = rest::Rest::start(
        basis,
        &state,
        (&cache.grid, &planes),
        (gravity, state.time, f64::from(sim.config().dt)),
    ) else {
        panic!("a compressive start");
    };
    rest.settle(kick, &mut state);
    sim.total_lagrangian = Some(state);
    sim
}

/// Its contact points at rest, the walls asked for a gap finer than the
/// solve resolves: every try ran to the passes' cap and was rejected, the
/// controller halving the step to 2e-9 s (measured, before the floor).
/// Kicked, its points still start at rest and must settle the same way.
#[test]
fn a_block_at_rest_on_the_floor_steps_once_a_frame() {
    for kick in [0.0, 0.1] {
        let mut sim = settled_block(kick);
        for k in 1..=5 {
            sim.step();
            let frame = sim.total_lagrangian_frame().expect("on the path");
            assert_eq!(
                (frame.steps, frame.rejected, frame.unconverged),
                (1, 0, 0),
                "kick {kick}, frame {k}: one step kept, none rejected or unconverged"
            );
        }
    }
}

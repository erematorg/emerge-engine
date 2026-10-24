//! What the relaxation branches refuse or report: the damped modes' flights
//! that cannot be carried, strains beyond the record's limit, a body of two
//! materials.

use super::*;

/// A free bar of `nu` with `record`, released in its first mode's shape at
/// `speed` m/s, modal flight asked for: `frames` frames, and how many were
/// flown and refused.
fn flight_asked(record: RelaxationSpectrum, nu: f32, speed: f32, frames: usize) -> (u32, u64) {
    let props = Elastic {
        e_pa: E_PA,
        nu,
        rho_kg_m3: RHO,
    };
    let mut config = SimConfig::earth(48, DX_M, FRAME_S);
    config.gravity = Vec2::ZERO;
    config.min_dt = config.min_dt.min(1e-9);
    config.total_lagrangian = Some(TotalLagrangianConfig::new(1e-2).with_modal_flight());
    let mut material = CorotatedMaterial::from_physical(&props, &config);
    material.relaxation = Some(record);
    let mut sim = Simulation::empty(config);
    let id = sim.register_material(Box::new(material)).0;
    let _bar = sim.add_body(
        SpawnRegion::for_sim(&config)
            .box_of(IVec2::new(LENGTH_CELLS, 2))
            .at(Vec2::new(24.0, 24.0))
            .spacing(0.5)
            .material(id)
            .mass_from(&props, &config),
    );
    let positions = sim.particles().x.clone();
    for (v, x) in sim.particles_mut().v.iter_mut().zip(positions) {
        let shape = (std::f32::consts::PI * (x.x - 8.0) / LENGTH_CELLS as f32).cos();
        *v = Vec2::new(shape * speed / DX_M, 0.0);
    }
    let mut flown = 0u32;
    for _ in 0..frames {
        sim.step();
        flown += u32::from(sim.total_lagrangian_frame().expect("on").modal);
    }
    (
        flown,
        sim.total_lagrangian_frame().expect("on").modal_refused,
    )
}

/// What the damped modes do not carry is never flown, and each refusal is
/// counted (the design's gate 5), criteria fixed before measuring:
/// - the synthetic record (P weights twice the shear's): not proportional,
///   its branches do not act through the body's stiffness;
/// - Plasticine rung at 1 cm/s, a strain of 1e-3: its flight's strain bound
///   passes the record's limit of 1e-5.
///
/// Every frame after the first asks to fly and is refused. (A body of two
/// spectra: `a_body_of_two_materials_is_stepped_and_its_damped_half_alone_heats`.)
#[test]
fn damped_flight_refuses_what_its_modes_do_not_carry() {
    const FRAMES: usize = 10;
    let (flown, refused) = flight_asked(synthetic(), 0.25, SPEED_M_S, FRAMES);
    println!("synthetic record: flown {flown}, refused {refused}");
    assert_eq!((flown, refused), (0, FRAMES as u64 - 1));
    let (flown, refused) = flight_asked(
        RelaxationSpectrum::plasticine_crandall_1970(),
        0.0,
        0.01,
        FRAMES,
    );
    println!("beyond the strain limit: flown {flown}, refused {refused}");
    assert_eq!((flown, refused), (0, FRAMES as u64 - 1));
}

/// The validity ledger reports, never hides: the bar rung at a strain of
/// 1e-3, a hundred times the record's linear limit, counts particle-steps
/// beyond it.
#[test]
fn branches_applied_beyond_their_records_strain_limit_are_reported() {
    let props = Elastic {
        e_pa: E_PA,
        nu: 0.0,
        rho_kg_m3: RHO,
    };
    let mut config = SimConfig::earth(48, DX_M, FRAME_S);
    config.gravity = Vec2::ZERO;
    config.total_lagrangian = Some(TotalLagrangianConfig::new(1e-2));
    let mut material = CorotatedMaterial::from_physical(&props, &config);
    material.relaxation = Some(RelaxationSpectrum::plasticine_crandall_1970());
    let mut sim = Simulation::empty(config);
    let id = sim.register_material(Box::new(material)).0;
    let _bar = sim.add_body(
        SpawnRegion::for_sim(&config)
            .box_of(IVec2::new(LENGTH_CELLS, 2))
            .at(Vec2::new(24.0, 24.0))
            .spacing(0.5)
            .material(id)
            .mass_from(&props, &config),
    );
    let positions = sim.particles().x.clone();
    for (v, x) in sim.particles_mut().v.iter_mut().zip(positions) {
        // 1 cm/s in the first mode's shape: a peak strain of 1e-3.
        let shape = (std::f32::consts::PI * (x.x - 8.0) / LENGTH_CELLS as f32).cos();
        *v = Vec2::new(shape * 0.01 / DX_M, 0.0);
    }
    let mut beyond = 0;
    let mut strain = 0.0f64;
    for _ in 0..10 {
        sim.step();
        let frame = sim.total_lagrangian_frame().expect("the path is on");
        beyond += frame.relaxation_beyond;
        strain = strain.max(frame.relaxation_strain);
    }
    println!(
        "rung at 1 cm/s: largest strain {strain:.2e}, particle-steps beyond the limit {beyond}"
    );
    assert!(
        strain > 1e-5 && beyond > 0,
        "strain {strain:e}, beyond {beyond}"
    );
}

/// One body of two materials: the ringing bar's right half carries
/// Plasticine's record, its left half none. With modal flight asked for, a
/// free body: its particles do not all carry one spectrum, so the modes do
/// not decouple its branches and refuse them; every frame is stepped and
/// counted as refused; the ledger closes and only the damped half heats.
#[test]
fn a_body_of_two_materials_is_stepped_and_its_damped_half_alone_heats() {
    const FRAMES: usize = 30;
    let props = Elastic {
        e_pa: E_PA,
        nu: 0.0,
        rho_kg_m3: RHO,
    };
    let mut config = SimConfig::earth(48, DX_M, FRAME_S);
    config.gravity = Vec2::ZERO;
    config.min_dt = config.min_dt.min(1e-7);
    config.total_lagrangian = Some(TotalLagrangianConfig::new(1e-2).with_modal_flight());
    let plain = CorotatedMaterial::from_physical(&props, &config);
    let mut damped = plain;
    damped.relaxation = Some(RelaxationSpectrum::plasticine_crandall_1970());
    let mut sim = Simulation::empty(config);
    let plain_id = sim.register_material(Box::new(plain)).0;
    let damped_id = sim.register_material(Box::new(damped)).0;
    let x0 = 8.0f32;
    let l = LENGTH_CELLS as f32;
    let _bar = sim.add_body(
        SpawnRegion::for_sim(&config)
            .box_of(IVec2::new(LENGTH_CELLS, 2))
            .at(Vec2::new(x0 + 0.5 * l, 24.0))
            .spacing(0.5)
            .material(plain_id)
            .mass_from(&props, &config),
    );
    let positions = sim.particles().x.clone();
    for (i, x) in positions.iter().enumerate() {
        if x.x > x0 + 0.5 * l {
            sim.particles_mut().material_id[i] = damped_id;
        }
        sim.particles_mut().v[i] = Vec2::new(
            (SPEED_M_S / DX_M) * (std::f32::consts::PI * (x.x - x0) / l).cos(),
            0.0,
        );
    }
    let (mut ledger, mut flown) = (Ledger::default(), 0u32);
    for _ in 0..FRAMES {
        sim.step();
        ledger.frame(&sim);
        flown += u32::from(sim.total_lagrangian_frame().expect("on").modal);
    }
    let frame = sim.total_lagrangian_frame().expect("the path is on");
    let heat = ledger.heat;
    println!(
        "two materials: {FRAMES} frames, flown {flown}, refused flights {}, total heat {:.3e} J/m; {}",
        frame.modal_refused,
        frame.heat,
        ledger.summary()
    );
    assert_eq!(flown, 0, "a body of two spectra never flies");
    // The first frame is stepped before a flight can be asked (the free
    // body is told by the contact cache its first step builds): every later
    // frame asks, and is refused and counted.
    assert_eq!(
        frame.modal_refused,
        FRAMES as u64 - 1,
        "each refused flight counted"
    );
    ledger.check("two materials");
    assert!((frame.heat - heat).abs() <= 1e-12 * heat.max(frame.heat));
}

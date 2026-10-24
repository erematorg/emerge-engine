//! Scene probes: the branches' cost and the Plasticine block built from its
//! record.

use super::*;

/// The cost of the branches (report only, criteria none): Newton
/// iterations, GMRES products and wall time per frame, with and without
/// Plasticine's spectrum, on E2's drop (the clay block dropped 10 cells,
/// 45 frames) and on the settled slab's scene (30 frames). The spectrum on
/// these scenes' moduli is a cost measurement, not a material: the record's
/// relaxed moduli are not set (its declared gap). Wall times mean something
/// only on a quiet machine.
#[test]
#[ignore = "cost probe: a quiet machine, run with --ignored --nocapture"]
fn probe_relaxation_cost() {
    let measure = |name: &str, mut sim: Simulation, frames: usize| {
        let (mut newton, mut gmres, mut steps, mut rejected, mut wall) =
            (0u64, 0u64, 0u64, 0u64, 0.0f64);
        let mut slowest = 0.0f64;
        for _ in 0..frames {
            sim.step();
            let f = sim.total_lagrangian_frame().expect("the path is on");
            newton += u64::from(f.newton);
            gmres += u64::from(f.gmres);
            steps += u64::from(f.steps);
            rejected += u64::from(f.rejected);
            wall += f.wall;
            slowest = slowest.max(f.wall);
        }
        let n = frames as f64;
        println!(
            "| {name} | {:.1} | {:.0} | {:.1} | {:.1} | {:.1} | {:.1} |",
            newton as f64 / n,
            gmres as f64 / n,
            steps as f64 / n,
            rejected as f64 / n,
            1e3 * wall / n,
            1e3 * slowest
        );
    };
    println!(
        "| scene | Newton/frame | GMRES/frame | steps/frame | rejected/frame | ms/frame | slowest ms |"
    );
    println!("|---|---|---|---|---|---|---|");
    let clay = Elastic {
        e_pa: 1.125e7,
        nu: 0.3,
        rho_kg_m3: 1700.0,
    };
    for relaxed in [false, true] {
        let mut config = SimConfig::earth(40, 0.01, 1.0 / 60.0);
        config.total_lagrangian = Some(TotalLagrangianConfig::new(1e-1));
        let mut material = CorotatedMaterial::from_physical(&clay, &config);
        if relaxed {
            material.relaxation = Some(RelaxationSpectrum::plasticine_crandall_1970());
        }
        let mut sim = Simulation::empty(config);
        let id = sim.register_material(Box::new(material)).0;
        let floor = config.boundary_thickness as f32;
        let _block = sim.add_body(
            SpawnRegion::for_sim(&config)
                .box_of(IVec2::new(8, 4))
                .at(Vec2::new(20.0, floor + 12.0))
                .spacing(0.5)
                .material(id)
                .mass_from(&clay, &config),
        );
        measure(
            if relaxed {
                "E2 drop, Plasticine spectrum"
            } else {
                "E2 drop"
            },
            sim,
            45,
        );
    }
    let slab = Elastic {
        e_pa: 1.0e5,
        nu: 0.0,
        rho_kg_m3: 1000.0,
    };
    for relaxed in [false, true] {
        let dx = 0.01f32;
        let mut config = SimConfig::earth(48, dx, 1.0 / 60.0);
        config.total_lagrangian = Some(TotalLagrangianConfig::new(1e-1));
        let mut material = CorotatedMaterial::from_physical(&slab, &config);
        if relaxed {
            material.relaxation = Some(RelaxationSpectrum::plasticine_crandall_1970());
        }
        let mut sim = Simulation::empty(config);
        let id = sim.register_material(Box::new(material)).0;
        let floor = config.boundary_thickness as f32;
        let _slab = sim.add_body(
            SpawnRegion::for_sim(&config)
                .box_of(IVec2::new(30, 12))
                .at(Vec2::new(24.0, floor + 6.0))
                .spacing(0.5)
                .material(id)
                .mass_from(&slab, &config),
        );
        measure(
            if relaxed {
                "settled slab, Plasticine spectrum"
            } else {
                "settled slab"
            },
            sim,
            30,
        );
    }
}

/// The Plasticine block: a scene built from one record whole
/// (`ViscoelasticRecord::plasticine_crandall_1970_fresh`: its relaxed
/// moduli, Poisson's ratio, density and spectrum), the E2 drop's geometry
/// (8 by 4 cm, 10 cells above the floor, 1 cm cells, 1/60 s frames).
/// Report only: its impact strains are far beyond the record's linear limit
/// (counted, `relaxation_beyond`), and no plastic law is paired with the
/// record (none read for this material), so the impact is outside what the
/// record covers.
#[test]
#[ignore = "scene probe: run with --ignored --nocapture"]
fn probe_plasticine_block_drop() {
    let record =
        emerge::materials::relaxation::ViscoelasticRecord::plasticine_crandall_1970_fresh();
    println!(
        "Plasticine block ({}): E_r {:.3e} Pa, nu {}, rho {} kg/m^3",
        record.state, record.relaxed.e_pa, record.relaxed.nu, record.relaxed.rho_kg_m3
    );
    // The step tolerance the E2 drop runs at, then tighter: whether the
    // hopping at the frame rate after the first impacts, with almost no heat,
    // is the long steps' (an artefact of the time step) or the scene's.
    // Measured: at 1e-1 the block hops (0.147 m/s at frames 31-45, 14 of
    // frames 21-45 in contact, 0.465 J/m less heat by frame 20); at 1e-2 and
    // 1e-3 it rests and the heat agrees to 1.3 %. A declared gap: the
    // tolerance is relative to the start's `tolerance_energy` (2.1 to 2.4
    // J/m here), which lets a step at 1e-1 carry 0.02 J/m of error while the
    // block's floor mode has `omega dt` near 67 to 117 (an estimate: a
    // quarter wave over the height, not yet measured). Removal criterion: a
    // reference scale derived from the physics, then the gates re-run.
    for tolerance in [1e-1, 1e-2, 1e-3] {
        let mut config = SimConfig::earth(40, 0.01, 1.0 / 60.0);
        config.total_lagrangian = Some(TotalLagrangianConfig::new(tolerance));
        let mut material = CorotatedMaterial::from_physical(&record.relaxed, &config);
        material.relaxation = Some(record.spectrum);
        let mut sim = Simulation::empty(config);
        let id = sim.register_material(Box::new(material)).0;
        let floor = config.boundary_thickness as f32;
        let _block = sim.add_body(
            SpawnRegion::for_sim(&config)
                .box_of(IVec2::new(8, 4))
                .at(Vec2::new(20.0, floor + 12.0))
                .spacing(0.5)
                .material(id)
                .mass_from(&record.relaxed, &config),
        );
        let mut ledger = Ledger::default();
        let (mut steps, mut wall, mut touching, mut heat_21, mut late_speed) =
            (0u32, 0.0f64, 0u32, 0.0f64, 0.0f64);
        for k in 1..=45 {
            sim.step();
            ledger.frame(&sim);
            let f = sim.total_lagrangian_frame().expect("the path is on");
            let p = sim.particles();
            let mass: f64 = p.mass.iter().map(|&m| f64::from(m)).sum();
            let vy =
                p.v.iter()
                    .zip(&p.mass)
                    .map(|(v, &m)| f64::from(v.y) * f64::from(m))
                    .sum::<f64>()
                    / mass
                    * f64::from(config.dx_meters);
            steps += f.steps;
            wall += f.wall;
            if k == 20 {
                heat_21 = -f.heat;
            }
            if k == 45 {
                heat_21 += f.heat;
            }
            if k > 20 && f.contacts > 0 {
                touching += 1;
            }
            if k > 30 {
                late_speed = late_speed.max(vy.abs());
            }
            println!(
                "  tolerance {tolerance:.0e} frame {k}: {} steps, {} rejected, contacts {}, centre speed {vy:+.4} m/s, heat {:.4e} J/m, {:.0} ms",
                f.steps,
                f.rejected,
                f.contacts,
                f.heat,
                1e3 * f.wall
            );
        }
        println!(
            "tolerance {tolerance:.0e}: {steps} steps, {wall:.1} s; frames 21-45: {touching} in contact, heat made {heat_21:.4e} J/m; frames 31-45: largest centre speed {late_speed:.4} m/s; {}",
            ledger.summary()
        );
    }
}

/// The Plasticine block of `probe_plasticine_block_drop` at step tolerance
/// 1e-2, stepped only against modal flight and modal rest asked for: per
/// run, the frames flown, rested and stepped, the hand-backs, the refusals,
/// the steps and the wall per frame, and the frames' rest of the energy
/// ledger. Report only: the impact is beyond the record's linear limit, so
/// most refusals are expected there.
#[test]
#[ignore = "scene probe: a quiet machine, run with --ignored --nocapture"]
fn probe_plasticine_block_modal_report() {
    let record =
        emerge::materials::relaxation::ViscoelasticRecord::plasticine_crandall_1970_fresh();
    for modal in [false, true] {
        let mut config = SimConfig::earth(40, 0.01, 1.0 / 60.0);
        let path = TotalLagrangianConfig::new(1e-2);
        config.total_lagrangian = Some(if modal {
            path.with_modal_flight().with_modal_rest()
        } else {
            path
        });
        let mut material = CorotatedMaterial::from_physical(&record.relaxed, &config);
        material.relaxation = Some(record.spectrum);
        let mut sim = Simulation::empty(config);
        let id = sim.register_material(Box::new(material)).0;
        let floor = config.boundary_thickness as f32;
        let _block = sim.add_body(
            SpawnRegion::for_sim(&config)
                .box_of(IVec2::new(8, 4))
                .at(Vec2::new(20.0, floor + 12.0))
                .spacing(0.5)
                .material(id)
                .mass_from(&record.relaxed, &config),
        );
        let mut ledger = Ledger::default();
        let (mut steps, mut wall, mut stepped, mut flown_wall, mut stepped_wall) =
            (0u32, 0.0f64, 0u32, 0.0f64, 0.0f64);
        let mut statuses = String::new();
        for _ in 1..=45 {
            sim.step();
            ledger.frame(&sim);
            let f = sim.total_lagrangian_frame().expect("the path is on");
            steps += f.steps;
            wall += f.wall;
            if f.modal || f.modal_rest {
                flown_wall += f.wall;
                statuses.push(if f.modal { 'F' } else { 'R' });
            } else {
                stepped += 1;
                stepped_wall += f.wall;
                statuses.push(if f.contacts > 0 { 'c' } else { 's' });
            }
        }
        let f = sim.total_lagrangian_frame().expect("the path is on");
        println!(
            "{}: flown {}, rested {}, stepped {stepped}; hand-backs {} (flight) {} (rest); refused {}, checks failed {}; {steps} steps, {wall:.1} s, {:.3} ms per modal frame, {:.0} ms per stepped frame",
            if modal { "modal asked" } else { "stepped only" },
            f.modal_frames,
            f.modal_rest_frames,
            f.hand_backs,
            f.rest_hand_backs,
            f.modal_refused,
            f.modal_check_failed,
            1e3 * flown_wall / (45 - stepped).max(1) as f64,
            1e3 * stepped_wall / stepped.max(1) as f64
        );
        println!("  frames (F flown, R rested, c stepped in contact, s stepped free): {statuses}");
        println!("  {}", ledger.summary());
    }
}

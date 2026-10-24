//! Replays with their energy ledger, for viewing: the E2 clay block and the
//! Norton plasticine block dropped on the floor on the total Lagrangian
//! path, every frame's positions, the energy's parts, the heat by origin,
//! the walls' work, the balance against what the solves allow, and the wall
//! time against the frame; each scene with the steps alone and with modal
//! flight allowed. The format of `tests/total_lagrangian_path.rs`'s
//! replays, with the ledger added. No check: a probe.
//!
//! Run with `EMERGE_REPLAY_DIR` set (the output directory) and, to label
//! the machine, `EMERGE_REPLAY_LOAD` (free text, written in the header).

extern crate emerge_engine as emerge;

use emerge::materials::MaterialModel;
use emerge::materials::physical_props::Elastic;
use emerge::materials::solid::viscoplastic_records::ViscoplasticRecord;
use emerge::spacetime::solver::total_lagrangian_solids::TotalLagrangianConfig;
use emerge::{CorotatedMaterial, FromSI, SimConfig, Simulation, SpawnRegion};
use glam::{IVec2, Vec2};

/// The 8 x 4 cm block of the material `made` builds, `props`' density,
/// dropped `lift` cells, at the step tolerance `tolerance`, its free frames
/// flown by its modes when `modal`.
fn block(
    (tolerance, modal): (f64, bool),
    lift: f32,
    made: &dyn Fn(&SimConfig) -> Box<dyn MaterialModel>,
    props: &Elastic,
) -> Simulation {
    let mut config = SimConfig::earth(40, 0.01, 1.0 / 60.0);
    let path = TotalLagrangianConfig::new(tolerance);
    config.total_lagrangian = Some(if modal {
        path.with_modal_flight()
    } else {
        path
    });
    let mut sim = Simulation::empty(config);
    let id = sim.register_material(made(&config)).0;
    let floor = config.boundary_thickness as f32;
    let _block = sim.add_body(
        SpawnRegion::for_sim(&config)
            .box_of(IVec2::new(8, 4))
            .at(Vec2::new(20.0, floor + lift + 2.0))
            .spacing(0.5)
            .material(id)
            .mass_from(props, &config),
    );
    sim
}

/// One frame of the replay: the format of `total_lagrangian_path.rs`
/// (time, positions m, speeds m/s, energy, heat, balance J/m, contacts) and
/// the ledger, J/m: the energy's parts (kinetic, stored, gravity's), this
/// frame's heat by origin (plastic work, relaxation branches, dashpots;
/// friction's work), the walls' normal work, the bound the solves allow on
/// the balance accumulated to this frame; the steps and rejected tries,
/// whether the frame was flown by the modes, and its wall time, ms.
fn frame_json(sim: &Simulation, time: f64, allowed: f64) -> String {
    let dx = f64::from(sim.config().dx_meters);
    let p = sim.particles();
    let f = sim.total_lagrangian_frame().expect("the path is on");
    let positions: Vec<String> =
        p.x.iter()
            .map(|q| format!("[{:.6},{:.6}]", f64::from(q.x) * dx, f64::from(q.y) * dx))
            .collect();
    let speeds: Vec<String> =
        p.v.iter()
            .map(|v| format!("{:.5}", f64::from(v.length()) * dx))
            .collect();
    let ledger = format!(
        "{{\"kinetic\":{:.9e},\"stored\":{:.9e},\"gravity\":{:.9e},\"plastic_heat\":{:.6e},\"relaxation_heat\":{:.6e},\"dashpot_heat\":{:.6e},\"friction_work\":{:.6e},\"contact_work\":{:.6e},\"allowed\":{:.3e}}}",
        f.parts[0],
        f.parts[1],
        f.parts[2],
        f.plastic_work,
        f.relaxation_heat,
        f.dissipated,
        f.friction_work,
        f.contact_work,
        allowed
    );
    format!(
        "{{\"time\":{time:.6},\"positions\":[{}],\"speed_m_s\":[{}],\"energy\":{:.9e},\"heat\":{:.9e},\"balance\":{:.3e},\"contacts\":{},\"ledger\":{ledger},\"steps\":{},\"rejected\":{},\"modal\":{},\"wall_ms\":{:.3}}}",
        positions.join(","),
        speeds.join(","),
        f.energy,
        f.heat,
        f.balance,
        f.contacts,
        f.steps,
        f.rejected,
        f.modal,
        1e3 * f.wall
    )
}

/// The particles' layout at the start, measured, not assumed: the distinct
/// columns and rows of their positions, and whether consecutive particles
/// run down a column (`column-major`) or along a row.
fn layout(sim: &Simulation) -> String {
    let x = &sim.particles().x;
    let distinct = |mut values: Vec<f32>| {
        values.sort_by(f32::total_cmp);
        values.dedup_by(|a, b| (*a - *b).abs() < 1e-4);
        values.len()
    };
    let columns = distinct(x.iter().map(|q| q.x).collect());
    let rows = distinct(x.iter().map(|q| q.y).collect());
    let order = if x.len() > 1 && (x[1].x - x[0].x).abs() < 1e-4 {
        "column-major"
    } else {
        "row-major"
    };
    format!("{{\"columns\":{columns},\"rows\":{rows},\"order\":\"{order}\"}}")
}

/// What a scene's header says of it: the tolerance, the material, and what
/// the controller was measured to honour of the tolerance.
struct Labels<'a> {
    tolerance: f64,
    material: &'a str,
    honoured: &'a str,
}

/// Runs `frames` frames and writes `name.json` in `EMERGE_REPLAY_DIR`; the
/// header carries the scene, its labels, the particles' layout, whether
/// frames may fly by the body's modes and how many did, the frame's
/// budget, the wall times and the machine label.
fn record(name: &str, mut sim: Simulation, frames: usize, labels: &Labels<'_>) {
    let frame_s = f64::from(sim.config().dt);
    let grid = layout(&sim);
    let modal = sim
        .config()
        .total_lagrangian
        .is_some_and(|path| path.modal_flight);
    let mut out = Vec::with_capacity(frames);
    let mut allowed = 0.0;
    let mut walls = Vec::with_capacity(frames);
    for k in 1..=frames {
        sim.step();
        let f = sim.total_lagrangian_frame().expect("the path is on");
        allowed += f.residual_work + f.contact_work_bound;
        walls.push(1e3 * f.wall);
        out.push(frame_json(&sim, frame_s * k as f64, allowed));
    }
    let flown = sim.total_lagrangian_frame().map_or(0, |f| f.modal_frames);
    let mut sorted = walls.clone();
    sorted.sort_by(f64::total_cmp);
    let budget = 1e3 * frame_s;
    let over = walls.iter().filter(|&&w| w > budget).count();
    let config = sim.config();
    let header = format!(
        "{{\"scene\":\"{name}\",\"material\":\"{}\",\"dx_m\":{},\"frame_s\":{},\"gravity_m_s2\":{},\"floor_y_m\":{},\"path\":\"total Lagrangian, weighted-gap contact, Acary step doubling\",\"tolerance\":{:e},\"tolerance_honoured\":\"{}\",\"grid\":{grid},\"modal_flight\":{modal},\"modal_frames\":{flown},\"budget_ms\":{budget:.3},\"wall_ms_median\":{:.3},\"wall_ms_largest\":{:.3},\"frames_over_budget\":{over},\"build\":\"test profile (optimized + debuginfo)\",\"machine\":\"{}\",\"precomputed\":true}}",
        labels.material,
        config.dx_meters,
        config.dt,
        config.gravity.length() * config.dx_meters,
        config.boundary_thickness as f32 * config.dx_meters,
        labels.tolerance,
        labels.honoured,
        sorted[sorted.len() / 2],
        sorted[sorted.len() - 1],
        std::env::var("EMERGE_REPLAY_LOAD").unwrap_or_else(|_| "not labelled".into())
    );
    let dir = std::env::var_os("EMERGE_REPLAY_DIR")
        .expect("EMERGE_REPLAY_DIR names the output directory");
    let path = std::path::Path::new(&dir).join(format!("{name}.json"));
    let body = format!(
        "{{\"header\":{header},\"frames\":[\n{}\n]}}\n",
        out.join(",\n")
    );
    std::fs::write(&path, body).expect("the replay directory is writable");
    println!(
        "{name}: {frames} frames written to {}; {flown} flown by the modes; wall median {:.1} ms, largest {:.1} ms, {over} of {frames} frames over the {budget:.1} ms budget",
        path.display(),
        sorted[sorted.len() / 2],
        sorted[sorted.len() - 1]
    );
}

#[test]
#[ignore = "replay export: run with --ignored --nocapture and EMERGE_REPLAY_DIR set"]
fn export_clay_and_plasticine_drops() {
    let clay = Elastic {
        e_pa: 1.125e7,
        nu: 0.3,
        rho_kg_m3: 1700.0,
    };
    let record_np = ViscoplasticRecord::technical_plasticine_sapozhnikov_2014();
    let rho = 119.0 * 0.453_592_37 / (0.3048f64 * 0.3048 * 0.3048);
    let plasticine = Elastic {
        e_pa: record_np.e_pa as f32,
        nu: record_np.nu as f32,
        rho_kg_m3: rho as f32,
    };
    let clay_labels = Labels {
        tolerance: 1e-1,
        material: "medium clay, corotated, E 11.25 MPa, nu 0.3, 1700 kg/m3 (elastic: no heat)",
        honoured: "kept local error about 2.31 times the tolerance (median; 95th 3.26, largest 4.72): probe_estimator_effectivity, 2026-10-20",
    };
    let plasticine_labels = Labels {
        tolerance: 1e-2,
        material: "technical plasticine, Norton flow on J2-Hencky (Sapozhnikov 2014 record), E 30 MPa, nu 0.3, stand-in density 1906 kg/m3",
        honoured: "kept local error about 1.20 times the tolerance (median; largest 1.49): probe_estimator_effectivity, 2026-10-20",
    };
    for (modal, suffix) in [(false, ""), (true, "_modal")] {
        record(
            &format!("clay_drop_e2{suffix}"),
            block(
                (clay_labels.tolerance, modal),
                10.0,
                &|config| Box::new(CorotatedMaterial::from_physical(&clay, config)),
                &clay,
            ),
            40,
            &clay_labels,
        );
        record(
            &format!("plasticine_drop_norton{suffix}"),
            block(
                (plasticine_labels.tolerance, modal),
                10.0,
                &|config| Box::new(record_np.material((record_np.e_pa, rho), config)),
                &plasticine,
            ),
            30,
            &plasticine_labels,
        );
    }
}

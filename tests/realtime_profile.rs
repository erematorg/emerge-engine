//! Where each frame's time goes on the total Lagrangian path, element by
//! element: the real-time step's first measurement, taken before any
//! optimisation. Each scene is one element of the physics, profiled alone,
//! so a later element (or a lever) adds one row, not a new profile:
//! E1 an elastic block at rest, E2 an elastic block bouncing on the floor
//! (contact, the step controller), E3 a block sliding on a Coulomb floor
//! (friction), E4 two blocks colliding (body-body contact), E5 a
//! Kelvin-Voigt bar ringing down (the dashpot), E6 a J2 block splatting
//! (plasticity); E1 and E4 also at 2 and 4 times the particles and on 1, 4
//! and 8 threads; E2 also under the step tolerance's earlier scale.
//!
//! Each prints one row: milliseconds per frame by phase (the frame's own
//! always-on timers, `TotalLagrangianFrame::timing`, every try's time
//! included), the counts per frame (the internal force's evaluations
//! among them, `Timing::internal_evaluations`), the scene's realism gate (the measure
//! its own test asserts, against its exact answer) and the worst energy
//! balance over the derived bound of the solves' tolerances (as
//! `tests/total_lagrangian_path.rs`'s `run` asserts it).
//!
//! Wall time means something only on a quiet machine: no other build or
//! test process, the load recorded before and after the run; one scene at
//! a time (`--test-threads=1`). Measured once: a slide looked 15 times
//! faster on a free machine than beside a test suite. Hence all ignored.

extern crate emerge_engine as emerge;

#[path = "common/cap.rs"]
mod cap;

use emerge::boundary::FrictionBoundary;
use emerge::materials::physical_props::Elastic;
use emerge::spacetime::solver::total_lagrangian_solids::TotalLagrangianConfig;
use emerge::{
    CorotatedMaterial, DuctileProps, FromSI, SimConfig, Simulation, SpawnRegion, VonMisesMaterial,
};
use glam::{IVec2, Vec2};

const CELL_M: f32 = 0.01;
const RES: usize = 40;
/// The 60 Hz frame's budget, ms.
const BUDGET_MS: f64 = 1000.0 / 60.0;

/// Medium clay, as the path's tests use it.
fn clay() -> Elastic {
    Elastic {
        e_pa: 1.125e7,
        nu: 0.3,
        rho_kg_m3: 1700.0,
    }
}

fn config(res: usize, tolerance: f64) -> SimConfig {
    let mut config = SimConfig::earth(res, CELL_M, 1.0 / 60.0);
    config.total_lagrangian = Some(TotalLagrangianConfig::new(tolerance));
    config
}

/// A `width` by `height` cell clay block centred at `x`, its bottom `lift`
/// cells above the floor.
fn block(
    config: &SimConfig,
    material: u32,
    (width, height): (i32, i32),
    x: f32,
    lift: f32,
) -> SpawnRegion {
    let floor = config.boundary_thickness as f32;
    SpawnRegion::for_sim(config)
        .box_of(IVec2::new(width, height))
        .at(Vec2::new(x, floor + lift + height as f32 * 0.5))
        .spacing(0.5)
        .material(material)
        .mass_from(&clay(), config)
}

/// A frame's energies in J/m: the grid's mass unit times a cell's area.
fn energy_unit(sim: &Simulation) -> f64 {
    let config = sim.config();
    f64::from(config.grid_mass_unit_kg_per_m()) * f64::from(config.dx_meters).powi(2)
}

/// The total weight over one cell's height, J/m (the rest scene's scale).
fn weight_scale(sim: &Simulation) -> f64 {
    let mass: f64 = sim.particles().mass.iter().map(|&m| f64::from(m)).sum();
    mass * f64::from(sim.config().gravity.length()) * energy_unit(sim)
}

/// The process's CPU time so far, s, all threads (`GetProcessTimes`):
/// beside the wall time, it reads a row under some background load. NaN
/// off Windows (not measured there).
#[cfg(windows)]
fn process_cpu_seconds() -> f64 {
    #[repr(C)]
    #[derive(Default)]
    struct FileTime {
        low: u32,
        high: u32,
    }
    unsafe extern "system" {
        fn GetCurrentProcess() -> *mut core::ffi::c_void;
        fn GetProcessTimes(
            process: *mut core::ffi::c_void,
            creation: *mut FileTime,
            exit: *mut FileTime,
            kernel: *mut FileTime,
            user: *mut FileTime,
        ) -> i32;
    }
    let (mut creation, mut exit, mut kernel, mut user) = Default::default();
    // SAFETY: the current process's pseudo-handle and four out-structures
    // of the layout the call writes.
    let ok = unsafe {
        GetProcessTimes(
            GetCurrentProcess(),
            &mut creation,
            &mut exit,
            &mut kernel,
            &mut user,
        )
    };
    if ok == 0 {
        return f64::NAN;
    }
    let ticks = |t: &FileTime| (u64::from(t.high) << 32 | u64::from(t.low)) as f64;
    // FILETIME counts 100 ns.
    (ticks(&kernel) + ticks(&user)) * 1e-7
}

#[cfg(not(windows))]
fn process_cpu_seconds() -> f64 {
    f64::NAN
}

/// One frame's record, for the frames' classes.
#[derive(Clone, Copy)]
struct FrameRecord {
    wall: f64,
    steps: u32,
    newton: u32,
    krylov: u32,
    /// The kept steps' contact work bound (`sum lambda a`): zero when no
    /// contact multiplier acted in the frame.
    contact_bound: f64,
    reaching: u32,
}

/// What a run's frames add up to.
#[derive(Default)]
struct Totals {
    frames: usize,
    wall: f64,
    /// The process's CPU time over the run, s (`process_cpu_seconds`).
    cpu: f64,
    records: Vec<FrameRecord>,
    residual: f64,
    jacobian: f64,
    linearisation: f64,
    push: f64,
    responses_jacobian: f64,
    build: f64,
    solve: f64,
    gmres: f64,
    responses: f64,
    contact: f64,
    multipliers: f64,
    grid: f64,
    finishing: f64,
    controller: f64,
    newton: u64,
    krylov: u64,
    evaluations: u64,
    factorizations: u64,
    passes: u64,
    steps: u64,
    rejected: u64,
    /// The worst `|balance|` over the derived bound, any frame.
    balance_over_bound: f64,
}

/// Steps `sim` for `frames` frames, `each` called after every one, and sums
/// the frames' timers and counts.
fn profile(sim: &mut Simulation, frames: usize, mut each: impl FnMut(&Simulation)) -> Totals {
    let mut t = Totals::default();
    let count = sim.particles().len() as f64;
    // The bound `run` asserts: the solves' residual and contact work,
    // accumulated, plus the sums' round-off.
    let mut allowed = 0.0f64;
    let cpu_start = process_cpu_seconds();
    for _ in 0..frames {
        sim.step();
        let f = sim.total_lagrangian_frame().expect("the path is on");
        allowed += f.residual_work + f.contact_work_bound;
        let round_off = count
            * f64::EPSILON
            * (f.parts.iter().map(|e| e.abs()).sum::<f64>() + f.heat.abs() + f.start_energy.abs());
        t.balance_over_bound = t
            .balance_over_bound
            .max(f.balance.abs() / (allowed + round_off));
        let p = f.timing;
        t.frames += 1;
        t.wall += f.wall;
        t.residual += p.residual;
        t.jacobian += p.jacobian;
        t.linearisation += p.linearisation;
        t.push += p.push_derivative;
        t.responses_jacobian += p.responses_jacobian;
        t.build += p.preconditioner_build;
        t.solve += p.preconditioner_solve;
        t.gmres += p.gmres;
        t.responses += p.responses;
        t.contact += p.contact;
        t.multipliers += p.multipliers;
        t.grid += p.grid;
        t.finishing += p.finishing;
        t.controller += p.controller;
        t.newton += u64::from(f.newton);
        t.krylov += u64::from(f.gmres);
        t.evaluations += p.internal_evaluations;
        t.factorizations += p.factorizations;
        t.passes += u64::from(f.passes);
        t.steps += u64::from(f.steps);
        t.rejected += u64::from(f.rejected);
        t.records.push(FrameRecord {
            wall: f.wall,
            steps: f.steps,
            newton: f.newton,
            krylov: f.gmres,
            contact_bound: f.contact_work_bound,
            reaching: f.reaching,
        });
        each(sim);
    }
    t.cpu = process_cpu_seconds() - cpu_start;
    t
}

/// Whether a frame belongs to a class.
type Member = fn(&FrameRecord) -> bool;

/// The frames by class, one line each: free flight (no contact multiplier
/// acted, none reached), impact (a point reached a wall in a kept step),
/// held contact (multipliers acting, none reached): their count, mean and
/// worst wall time and mean steps, Newton and GMRES; then every frame's
/// steps and wall. The real-time question is the worst frame (a missed
/// frame is a visible hitch), not the mean.
fn frame_classes(scene: &str, t: &Totals) {
    let classes: [(&str, Member); 3] = [
        ("free flight", |r| r.contact_bound == 0.0 && r.reaching == 0),
        ("impact", |r| r.reaching > 0),
        ("held contact", |r| r.contact_bound > 0.0 && r.reaching == 0),
    ];
    for (name, member) in classes {
        let picked: Vec<&FrameRecord> = t.records.iter().filter(|r| member(r)).collect();
        if picked.is_empty() {
            println!("| {scene}: {name} | 0 frames |");
            continue;
        }
        let n = picked.len() as f64;
        let mean = |f: fn(&FrameRecord) -> f64| picked.iter().map(|r| f(r)).sum::<f64>() / n;
        let worst = picked.iter().map(|r| r.wall).fold(0.0f64, f64::max);
        println!(
            "| {scene}: {name} | {} frames | wall mean {:.1} ms, worst {:.1} ms (of {BUDGET_MS:.1}) | steps {:.1}, Newton {:.1}, GMRES {:.1} per frame |",
            picked.len(),
            1000.0 * mean(|r| r.wall),
            1000.0 * worst,
            mean(|r| f64::from(r.steps)),
            mean(|r| f64::from(r.newton)),
            mean(|r| f64::from(r.krylov)),
        );
    }
    let list: Vec<String> = t
        .records
        .iter()
        .map(|r| {
            let class = if r.reaching > 0 {
                "I"
            } else if r.contact_bound > 0.0 {
                "C"
            } else {
                "F"
            };
            format!("{class}{}:{:.0}", r.steps, 1000.0 * r.wall)
        })
        .collect();
    println!(
        "| {scene}: frames (class steps:ms; F flight, I impact, C contact) | {} |",
        list.join(" ")
    );
}

/// The table's header, printed with every row so a single run reads alone.
fn header() {
    println!(
        "| scene | particles | threads | ms/frame (of {BUDGET_MS:.1}) | CPU ms/frame (all threads) | residual | Jv | of Jv: push | of Jv: responses' | linearisation | precond build | precond solve | GMRES algebra | responses | contact | multipliers | grid | finishing | controller | Newton | GMRES | force evaluations | factorizations | passes | steps | rejected | gate | balance / bound |"
    );
}

/// One row: per frame, ms by phase and the counts; the gate and the
/// balance's worst ratio to its bound.
fn row(scene: &str, sim: &Simulation, t: &Totals, gate: (&str, f64)) {
    header();
    let n = t.frames.max(1) as f64;
    let ms = |s: f64| 1000.0 * s / n;
    let per = |c: u64| c as f64 / n;
    println!(
        "| {scene} | {} | {} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} | {:.2} | {} {:.4e} | {:.2e} |",
        sim.particles().len(),
        rayon::current_num_threads(),
        ms(t.wall),
        ms(t.cpu),
        ms(t.residual),
        ms(t.jacobian),
        ms(t.push),
        ms(t.responses_jacobian),
        ms(t.linearisation),
        ms(t.build),
        ms(t.solve),
        ms(t.gmres - t.jacobian - t.solve),
        ms(t.responses),
        ms(t.contact),
        ms(t.multipliers),
        ms(t.grid),
        ms(t.finishing),
        ms(t.controller),
        per(t.newton),
        per(t.krylov),
        per(t.evaluations),
        per(t.factorizations),
        per(t.passes),
        per(t.steps),
        per(t.rejected),
        gate.0,
        gate.1,
        t.balance_over_bound
    );
}

/// `run` on a pool of `threads` threads.
fn on_threads(threads: usize, run: impl FnOnce() + Send) {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("a thread pool")
        .install(run);
}

/// E1: an 8 by 4 block (times `scale`) at rest on the floor, 30 frames.
/// Gate: the worst balance over the weight scale (its test: below 1e-6).
fn e1_rest(scale: (i32, i32)) {
    let config = config(RES * scale.0 as usize, 1e-1);
    let mut sim = Simulation::empty(config);
    let id = sim
        .register_material(Box::new(CorotatedMaterial::from_physical(&clay(), &config)))
        .0;
    let middle = config.grid_res as f32 * 0.5;
    let _block = sim.add_body(block(&config, id, (8 * scale.0, 4 * scale.1), middle, 0.0));
    let scale_j = weight_scale(&sim);
    let mut drift = 0.0f64;
    let t = profile(&mut sim, 30, |sim| {
        let f = sim.total_lagrangian_frame().expect("the path is on");
        drift = drift.max(f.balance.abs() / scale_j);
    });
    row(
        &format!("E1 rest x{}", scale.0 * scale.1),
        &sim,
        &t,
        ("drift", drift),
    );
}

/// E2: the 8 by 4 block dropped 10 cells, 45 frames, the step tolerance
/// `tolerance`. Gate: the deepest sink below the floor over the declared
/// penetration `v dt` (its test: at most 1).
fn e2_bounce(tolerance: f64, label: &str) {
    e2_bounce_damped(tolerance, 0.0, label);
}

/// E2 with the generic Newtonian dashpot in the clay (shear viscosity
/// `eta` Pa s, bulk `2 eta / 3`, as `tests/kelvin_voigt_bar.rs`): a scene
/// parameter, the material's own damping, not a change of method. For
/// scale: the block's first mode `omega = pi c / L` (c = sqrt(E / rho), L
/// its 8 cm) is critically damped at `eta = E / omega`, about 3.5e3 Pa s;
/// the cell's modes, ten times faster, at a tenth of it.
fn e2_bounce_damped(tolerance: f64, eta_pa_s: f32, label: &str) {
    let config = config(RES, tolerance);
    let mut sim = Simulation::empty(config);
    let mut material = CorotatedMaterial::from_physical(&clay(), &config);
    material.elastic_viscosity = config.visc_from_si(eta_pa_s);
    material.elastic_bulk_viscosity = config.visc_from_si(2.0 * eta_pa_s / 3.0);
    let id = sim.register_material(Box::new(material)).0;
    let _block = sim.add_body(block(&config, id, (8, 4), RES as f32 * 0.5, 10.0));
    let floor = config.boundary_thickness as f32;
    let bound = (2.0 * config.gravity.length() * 10.0).sqrt() / 60.0;
    let mut deepest = 0.0f32;
    let t = profile(&mut sim, 45, |sim| {
        let lowest = sim
            .particles()
            .x
            .iter()
            .map(|q| q.y)
            .fold(f32::INFINITY, f32::min);
        deepest = deepest.max(floor - lowest);
    });
    row(label, &sim, &t, ("sink / v dt", f64::from(deepest / bound)));
    frame_classes(label, &t);
}

/// E3: the block sliding at 1 m/s on a Coulomb floor, mu 0.3, 30 frames.
/// Gate: the travel over the exact stop distance `v0^2 / (2 mu g)`.
fn e3_slide() {
    let config = config(RES, 1e-1);
    let mut sim = Simulation::empty(config);
    let id = sim
        .register_material(Box::new(CorotatedMaterial::from_physical(&clay(), &config)))
        .0;
    let _block = sim.add_body(block(&config, id, (8, 4), 10.0, 0.0));
    sim.add_boundary_condition(Box::new(FrictionBoundary::new(
        config.boundary_thickness,
        0.3,
    )));
    let speed = 1.0 / CELL_M;
    for v in sim.particles_mut().v.iter_mut() {
        *v = Vec2::new(speed, 0.0);
    }
    let centre = |sim: &Simulation| -> f64 {
        let p = sim.particles();
        let mass: f64 = p.mass.iter().map(|&m| f64::from(m)).sum();
        p.x.iter()
            .zip(&p.mass)
            .map(|(x, &m)| f64::from(x.x) * f64::from(m))
            .sum::<f64>()
            / mass
    };
    let start = centre(&sim);
    let t = profile(&mut sim, 30, |_| {});
    let travel = (centre(&sim) - start) * f64::from(CELL_M);
    let exact = 1.0 / (2.0 * 0.3 * 9.81);
    row("E3 slide", &sim, &t, ("travel / exact", travel / exact));
}

/// E4: two 6 by 4 blocks (times `scale`) closing at 0.25 m/s, 30 frames.
/// Gate: the horizontal momentum's drift (its test: below 1e-5).
fn e4_collision(scale: (i32, i32)) {
    let config = config(RES * scale.0 as usize, 1e-1);
    let mut sim = Simulation::empty(config);
    let id = sim
        .register_material(Box::new(CorotatedMaterial::from_physical(&clay(), &config)))
        .0;
    let size = (6 * scale.0, 4 * scale.1);
    let (left, right) = (8.0 * scale.0 as f32, 18.0 * scale.0 as f32);
    let left_tag = sim.add_body(block(&config, id, size, left, 0.0));
    let _right = sim.add_body(block(&config, id, size, right, 0.0));
    let speed = 0.25 / CELL_M;
    let moving: Vec<bool> = sim
        .particles()
        .user_tag
        .iter()
        .map(|&t| t == left_tag)
        .collect();
    for (v, &m) in sim.particles_mut().v.iter_mut().zip(&moving) {
        if m {
            *v = Vec2::new(speed, 0.0);
        }
    }
    let momentum = |sim: &Simulation| -> f64 {
        sim.particles()
            .v
            .iter()
            .zip(&sim.particles().mass)
            .map(|(v, &m)| f64::from(v.x) * f64::from(m))
            .sum()
    };
    let start = momentum(&sim);
    let t = profile(&mut sim, 30, |_| {});
    let drift = (momentum(&sim) - start).abs() / start.abs();
    row(
        &format!("E4 collision x{}", scale.0 * scale.1),
        &sim,
        &t,
        ("momentum drift", drift),
    );
}

/// E5: the Kelvin-Voigt bar of `tests/kelvin_voigt_bar.rs` (32 by 2 cells,
/// E 100 kPa, nu 0, eta 50 Pa s, no gravity, 2 ms frames), 100 frames.
/// Gate: the damping ratio from the mechanical energy's mean decay (a
/// least-squares line through `ln E`, `E ~ exp(-2 zeta omega t)`) over the
/// exact `eta omega / E`, `omega = pi c / L`.
fn e5_kelvin_voigt() {
    let (e_pa, rho, eta, length) = (1.0e5f32, 1000.0f32, 50.0f32, 32);
    let props = Elastic {
        e_pa,
        nu: 0.0,
        rho_kg_m3: rho,
    };
    let frame_s = 2.0e-3f32;
    let mut config = SimConfig::earth(48, CELL_M, frame_s);
    config.gravity = Vec2::ZERO;
    config.min_dt = config.min_dt.min(1e-7);
    config.total_lagrangian = Some(TotalLagrangianConfig::new(1e-2));
    let mut material = CorotatedMaterial::from_physical(&props, &config);
    material.elastic_viscosity = config.visc_from_si(eta);
    material.elastic_bulk_viscosity = config.visc_from_si(2.0 * eta / 3.0);
    let mut sim = Simulation::empty(config);
    let id = sim.register_material(Box::new(material)).0;
    let (x0, l) = (8.0f32, length as f32);
    let _bar = sim.add_body(
        SpawnRegion::for_sim(&config)
            .box_of(IVec2::new(length, 2))
            .at(Vec2::new(x0 + 0.5 * l, 24.0))
            .spacing(0.5)
            .material(id)
            .mass_from(&props, &config),
    );
    let positions = sim.particles().x.clone();
    for (v, x) in sim.particles_mut().v.iter_mut().zip(positions) {
        *v = Vec2::new(
            (0.01 / CELL_M) * (std::f32::consts::PI * (x.x - x0) / l).cos(),
            0.0,
        );
    }
    let mut samples: Vec<(f64, f64)> = Vec::new();
    let t = profile(&mut sim, 100, |sim| {
        let f = sim.total_lagrangian_frame().expect("the path is on");
        let time = f64::from(frame_s) * (samples.len() + 1) as f64;
        samples.push((time, (f.parts[0] + f.parts[1]).ln()));
    });
    let n = samples.len() as f64;
    let (mt, me) = samples
        .iter()
        .fold((0.0, 0.0), |(a, b), (t, e)| (a + t / n, b + e / n));
    let slope = samples
        .iter()
        .map(|(t, e)| (t - mt) * (e - me))
        .sum::<f64>()
        / samples.iter().map(|(t, _)| (t - mt).powi(2)).sum::<f64>();
    let omega =
        std::f64::consts::PI * f64::from((e_pa / rho).sqrt()) / (f64::from(l) * f64::from(CELL_M));
    let exact = f64::from(eta) * omega / f64::from(e_pa);
    let measured = -slope / (2.0 * omega);
    row(
        "E5 Kelvin-Voigt bar",
        &sim,
        &t,
        ("zeta / exact", measured / exact),
    );
}

/// E6: the J2 block of the splat test (E 1 MPa, nu 0.3, Y 20 kPa, 1700
/// kg/m^3) dropped 10 cells, 40 frames. Gate: the area's change over its
/// derived bound `sqrt(2 A0 E_el / K) + n eps_f32 A0` (its test: at most 1).
fn e6_splat() {
    let config = config(RES, 1e-1);
    let props = DuctileProps {
        elastic: Elastic {
            e_pa: 1.0e6,
            nu: 0.3,
            rho_kg_m3: 1700.0,
        },
        yield_stress_pa: 2.0e4,
    };
    let mut sim = Simulation::empty(config);
    let id = sim
        .register_material(Box::new(VonMisesMaterial::from_physical(&props, &config)))
        .0;
    let floor = config.boundary_thickness as f32;
    let _block = sim.add_body(
        SpawnRegion::for_sim(&config)
            .box_of(IVec2::new(8, 4))
            .at(Vec2::new(RES as f32 * 0.5, floor + 12.0))
            .spacing(0.5)
            .material(id)
            .mass_from(&props.elastic, &config),
    );
    let cell = f64::from(CELL_M);
    let a0 = sim
        .particles()
        .mass
        .iter()
        .map(|&m| f64::from(m))
        .sum::<f64>()
        / f64::from(config.density_from_si(props.elastic.rho_kg_m3))
        * cell
        * cell;
    let t = profile(&mut sim, 40, |_| {});
    let f = sim.total_lagrangian_frame().expect("the path is on");
    let area: f64 = sim
        .particles()
        .volume
        .iter()
        .map(|&v| f64::from(v))
        .sum::<f64>()
        * cell
        * cell;
    let bulk = f64::from(props.elastic.e_pa) / (3.0 * (1.0 - 2.0 * f64::from(props.elastic.nu)));
    let bound = (2.0 * a0 * f.parts[1] / bulk).sqrt()
        + sim.particles().len() as f64 * f64::from(f32::EPSILON) * a0;
    row(
        "E6 J2 splat",
        &sim,
        &t,
        ("area change / bound", (area - a0).abs() / bound),
    );
    frame_classes("E6 J2 splat", &t);
}

#[test]
#[ignore = "real-time profile: a quiet machine only, one scene at a time"]
fn profile_e1_rest() {
    cap::capped("profile_e1_rest", 3600, || {
        e1_rest((1, 1));
    });
}

#[test]
#[ignore = "real-time profile: a quiet machine only, one scene at a time"]
fn profile_e2_bounce() {
    cap::capped("profile_e2_bounce", 3600, || {
        e2_bounce(1e-1, "E2 bounce, tol 1e-1");
    });
}

/// E2 under the step tolerance's earlier scale, the start's total energy
/// with gravity's potential from the grid's corner: the controller decides
/// against `tol sqrt(E)`, so the same decisions come back exactly with the
/// present scale and `tol sqrt(start_energy / tolerance_energy)`; no
/// earlier code is kept. The cost of the scale's correction, as its own row.
#[test]
#[ignore = "real-time profile: a quiet machine only, one scene at a time"]
fn profile_e2_bounce_at_the_earlier_scale() {
    cap::capped("profile_e2_bounce_at_the_earlier_scale", 3600, || {
        let config = config(RES, 1e-1);
        let mut sim = Simulation::empty(config);
        let id = sim
            .register_material(Box::new(CorotatedMaterial::from_physical(&clay(), &config)))
            .0;
        let _block = sim.add_body(block(&config, id, (8, 4), RES as f32 * 0.5, 10.0));
        sim.step();
        let f = sim.total_lagrangian_frame().expect("the path is on");
        let factor = (f.start_energy.abs() / f.tolerance_energy).sqrt();
        e2_bounce(1e-1 * factor, "E2 bounce, earlier scale");
    });
}

/// E2 at a looser and a tighter step tolerance (a lever that changes the
/// answer, so each row's gate stands beside its cost): how much of the
/// impacts' steps the tolerance sets.
#[test]
#[ignore = "real-time profile: a quiet machine only, one scene at a time"]
fn profile_e2_bounce_tolerances() {
    cap::capped("profile_e2_bounce_tolerances", 3600, || {
        e2_bounce(3e-1, "E2 bounce, tol 3e-1");
        e2_bounce(1e-2, "E2 bounce, tol 1e-2");
    });
}

/// E2 with the clay's own damping at three viscosities and two step
/// tolerances: a scene parameter (real clay is not undamped), measured for
/// the user's decision, not a method change.
#[test]
#[ignore = "real-time profile: a quiet machine only, one scene at a time"]
fn profile_e2_bounce_damped() {
    cap::capped("profile_e2_bounce_damped", 3600, || {
        for tolerance in [1e-1, 1e-2] {
            for eta in [10.0f32, 300.0, 1.0e4] {
                e2_bounce_damped(
                    tolerance,
                    eta,
                    &format!("E2 bounce, eta {eta} Pa s, tol {tolerance:e}"),
                );
            }
        }
    });
}

#[test]
#[ignore = "real-time profile: a quiet machine only, one scene at a time"]
fn profile_e3_slide() {
    cap::capped("profile_e3_slide", 3600, || {
        e3_slide();
    });
}

#[test]
#[ignore = "real-time profile: a quiet machine only, one scene at a time"]
fn profile_e4_collision() {
    cap::capped("profile_e4_collision", 3600, || {
        e4_collision((1, 1));
    });
}

#[test]
#[ignore = "real-time profile: a quiet machine only, one scene at a time"]
fn profile_e5_kelvin_voigt() {
    cap::capped("profile_e5_kelvin_voigt", 3600, || {
        e5_kelvin_voigt();
    });
}

#[test]
#[ignore = "real-time profile: a quiet machine only, one scene at a time"]
fn profile_e6_splat() {
    cap::capped("profile_e6_splat", 3600, || {
        e6_splat();
    });
}

/// E1 and E4 at 2 and 4 times the particles (the blocks twice as wide,
/// then twice as tall too; the domain as wide as needed).
#[test]
#[ignore = "real-time profile: a quiet machine only, one scene at a time"]
fn profile_size_scaling() {
    cap::capped("profile_size_scaling", 3600, || {
        for scale in [(2, 1), (2, 2)] {
            e1_rest(scale);
            e4_collision(scale);
        }
    });
}

/// E1 and E4 on 1, 4 and 8 threads.
#[test]
#[ignore = "real-time profile: a quiet machine only, one scene at a time"]
fn profile_thread_scaling() {
    cap::capped("profile_thread_scaling", 3600, || {
        for threads in [1, 4, 8] {
            on_threads(threads, || {
                e1_rest((1, 1));
                e4_collision((1, 1));
            });
        }
    });
}

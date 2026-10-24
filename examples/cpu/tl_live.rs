//! The total Lagrangian solid path (`SimConfig::total_lagrangian`) run live:
//! one clay block dropped on the floor, one `Simulation::step` per wall
//! frame, paced to the frame rate, every frame's wall time recorded.
//!
//! The scene is the drop of `tests/total_lagrangian_path.rs`
//! (`a_dropped_block_bounces_with_its_energy_kept`, the 10-cell lift) and of
//! `tests/realtime_profile.rs` (`e2_bounce`): an 8 by 4 cell block of medium
//! clay (E 11.25 MPa, nu 0.3, 1700 kg/m^3) on a 40-cell grid of 1 cm cells,
//! 1/60 s frames, the step tolerance 1e-1, 45 frames. Every default below is
//! that scene's value; every one is an argument (`--help`).
//!
//! It drives the simulation as `tests/realtime_profile.rs`'s `profile` does
//! (`sim.step()`, then `sim.total_lagrangian_frame()`), and adds the pacing:
//! after a frame's step, the loop sleeps to the next frame's deadline. A
//! frame whose step outlasts the budget is a hitch; the loop does not try
//! to catch up (the next frame starts at once, its deadline one budget
//! later), so the simulated time always advances one frame per wall frame.
//!
//! Wall time means something only on a quiet machine (see
//! `tests/realtime_profile.rs`'s header).

extern crate emerge_engine as emerge;

use std::fmt::Write as _;
use std::time::{Duration, Instant};

use emerge::materials::physical_props::Elastic;
use emerge::spacetime::solver::total_lagrangian_solids::{
    TotalLagrangianConfig, TotalLagrangianFrame,
};
use emerge::{CorotatedMaterial, FromSI, SimConfig, Simulation, SpawnRegion};
use glam::{IVec2, Vec2};

/// The run's settings, each an argument; the defaults are the reference
/// scene's (module doc).
struct Args {
    res: usize,
    cell_m: f32,
    frame_hz: f64,
    frames: usize,
    tolerance: f64,
    e_pa: f32,
    nu: f32,
    rho_kg_m3: f32,
    width: i32,
    height: i32,
    lift: f32,
    spacing: f32,
    /// 0: rayon's own pool (its default, or `RAYON_NUM_THREADS`).
    threads: usize,
    /// Sleep to each frame's deadline; off, the frames run back to back.
    paced: bool,
    /// Carry the block's flights between contacts by its modes
    /// (`TotalLagrangianConfig::with_modal_flight`); off, every frame is
    /// stepped, as in the reference scene.
    modal_flight: bool,
    json: Option<String>,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            res: 40,
            cell_m: 0.01,
            frame_hz: 60.0,
            frames: 45,
            tolerance: 1e-1,
            e_pa: 1.125e7,
            nu: 0.3,
            rho_kg_m3: 1700.0,
            width: 8,
            height: 4,
            lift: 10.0,
            spacing: 0.5,
            threads: 0,
            paced: true,
            modal_flight: false,
            json: None,
        }
    }
}

const USAGE: &str = "tl_live: the total Lagrangian path's drop scene, live at the frame rate.
Defaults: the 10-cell drop of tests/total_lagrangian_path.rs
(a_dropped_block_bounces_with_its_energy_kept) and tests/realtime_profile.rs (e2_bounce).
  --res N          grid cells per side            [40]
  --cell-m M       cell size, m                   [0.01]
  --frame-hz F     frame rate; frame = 1/F s       [60]
  --frames N       frames to run                  [45]
  --tolerance T    step controller tolerance      [0.1]
  --e-pa E         Young's modulus, Pa            [1.125e7]
  --nu NU          Poisson ratio                  [0.3]
  --rho R          density, kg/m^3                [1700]
  --width W        block width, cells             [8]
  --height H       block height, cells            [4]
  --lift L         block bottom above floor, cells [10]
  --spacing S      particle spacing, cells        [0.5]
  --threads N      rayon threads, 0 = rayon's pool [0]
  --unpaced        run frames back to back (no sleep to the deadline)
  --modal-flight   carry flights between contacts by the body's modes [off]
  --json PATH      write the per-frame record to PATH";

fn parse_args() -> Result<Args, String> {
    let mut args = Args::default();
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        if flag == "--help" || flag == "-h" {
            return Err(String::new());
        }
        if flag == "--unpaced" {
            args.paced = false;
            continue;
        }
        if flag == "--modal-flight" {
            args.modal_flight = true;
            continue;
        }
        let value = it.next().ok_or(format!("{flag} needs a value"))?;
        let bad = |e: &dyn std::fmt::Display| format!("{flag} {value}: {e}");
        match flag.as_str() {
            "--res" => args.res = value.parse().map_err(|e| bad(&e))?,
            "--cell-m" => args.cell_m = value.parse().map_err(|e| bad(&e))?,
            "--frame-hz" => args.frame_hz = value.parse().map_err(|e| bad(&e))?,
            "--frames" => args.frames = value.parse().map_err(|e| bad(&e))?,
            "--tolerance" => args.tolerance = value.parse().map_err(|e| bad(&e))?,
            "--e-pa" => args.e_pa = value.parse().map_err(|e| bad(&e))?,
            "--nu" => args.nu = value.parse().map_err(|e| bad(&e))?,
            "--rho" => args.rho_kg_m3 = value.parse().map_err(|e| bad(&e))?,
            "--width" => args.width = value.parse().map_err(|e| bad(&e))?,
            "--height" => args.height = value.parse().map_err(|e| bad(&e))?,
            "--lift" => args.lift = value.parse().map_err(|e| bad(&e))?,
            "--spacing" => args.spacing = value.parse().map_err(|e| bad(&e))?,
            "--threads" => args.threads = value.parse().map_err(|e| bad(&e))?,
            "--json" => args.json = Some(value),
            _ => return Err(format!("unknown argument {flag}")),
        }
    }
    if args.frame_hz <= 0.0 || args.frames == 0 {
        return Err("--frame-hz and --frames must be positive".into());
    }
    Ok(args)
}

/// The scene: the block's bottom `lift` cells above the floor, centred on
/// the domain, as the tests' `scene` and `block` build it.
fn scene(args: &Args) -> Simulation {
    let props = Elastic {
        e_pa: args.e_pa,
        nu: args.nu,
        rho_kg_m3: args.rho_kg_m3,
    };
    let mut config = SimConfig::earth(args.res, args.cell_m, (1.0 / args.frame_hz) as f32);
    let path = TotalLagrangianConfig::new(args.tolerance);
    config.total_lagrangian = Some(if args.modal_flight {
        path.with_modal_flight()
    } else {
        path
    });
    let mut sim = Simulation::empty(config);
    let id = sim
        .register_material(Box::new(CorotatedMaterial::from_physical(&props, &config)))
        .0;
    let floor = config.boundary_thickness as f32;
    let _block = sim.add_body(
        SpawnRegion::for_sim(&config)
            .box_of(IVec2::new(args.width, args.height))
            .at(Vec2::new(
                args.res as f32 * 0.5,
                floor + args.lift + args.height as f32 * 0.5,
            ))
            .spacing(args.spacing)
            .material(id)
            .mass_from(&props, &config),
    );
    sim
}

/// One frame's record: the wall time of its `step` call measured here,
/// the frame report's own wall time, its counts and its energy.
struct Record {
    time: f64,
    wall: f64,
    frame: TotalLagrangianFrame,
    positions: Vec<[f64; 2]>,
}

/// The commit the binary runs from, read at run time (`git rev-parse`);
/// "unknown" outside a checkout or without git.
fn git_commit() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".into())
}

fn header(args: &Args) {
    let profile = if cfg!(debug_assertions) {
        "debug assertions on (dev profile)"
    } else {
        "debug assertions off (optimised profile)"
    };
    let env_threads = std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "unset".into());
    println!(
        "tl_live | build: {profile} | threads: {} (RAYON_NUM_THREADS {env_threads}, --threads {}) | commit: {} | paced: {} | modal flight: {}",
        rayon::current_num_threads(),
        args.threads,
        git_commit(),
        args.paced,
        args.modal_flight
    );
}

/// Runs the frames, paced to the frame rate when asked; returns the
/// records and the loop's elapsed wall time, s.
fn run(sim: &mut Simulation, args: &Args) -> (Vec<Record>, f64) {
    let budget = Duration::from_secs_f64(1.0 / args.frame_hz);
    let dx = f64::from(sim.config().dx_meters);
    let mut records = Vec::with_capacity(args.frames);
    let start = Instant::now();
    let mut deadline = start;
    for i in 0..args.frames {
        let clock = Instant::now();
        sim.step();
        let wall = clock.elapsed().as_secs_f64();
        let frame = sim.total_lagrangian_frame().expect("the path is on");
        let positions = sim
            .particles()
            .x
            .iter()
            .map(|q| [f64::from(q.x) * dx, f64::from(q.y) * dx])
            .collect();
        records.push(Record {
            time: (i + 1) as f64 / args.frame_hz,
            wall,
            frame,
            positions,
        });
        // The next deadline is one budget after this one, or after now when
        // the frame overran: a late frame is not made up.
        deadline = (deadline + budget).max(Instant::now());
        if args.paced {
            let now = Instant::now();
            if deadline > now {
                std::thread::sleep(deadline - now);
            }
        }
    }
    (records, start.elapsed().as_secs_f64())
}

/// The sorted sample's `q` quantile by nearest rank, `ceil(q n)`-th
/// smallest (a choice of definition, exact on the samples it reports).
fn nearest_rank(sorted: &[f64], q: f64) -> f64 {
    let rank = (q * sorted.len() as f64).ceil() as usize;
    sorted[rank.clamp(1, sorted.len()) - 1]
}

fn median(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        0.5 * (sorted[n / 2 - 1] + sorted[n / 2])
    }
}

fn summary(args: &Args, records: &[Record], elapsed: f64) {
    let budget_ms = 1000.0 / args.frame_hz;
    let mut ms: Vec<f64> = records.iter().map(|r| 1000.0 * r.wall).collect();
    let mean = ms.iter().sum::<f64>() / ms.len() as f64;
    ms.sort_by(f64::total_cmp);
    let simulated = records.len() as f64 / args.frame_hz;
    let compute: f64 = records.iter().map(|r| r.wall).sum();
    println!(
        "frames {} | frame ms: mean {mean:.2}, median {:.2}, p95 {:.2}, max {:.2} (budget {budget_ms:.2})",
        records.len(),
        median(&ms),
        nearest_rank(&ms, 0.95),
        ms[ms.len() - 1]
    );
    println!(
        "achieved fps {:.1} (target {}) | real-time factor: loop wall / simulated {:.3}, step wall / simulated {:.3}",
        records.len() as f64 / elapsed,
        args.frame_hz,
        elapsed / simulated,
        compute / simulated
    );
    let hitches: Vec<(usize, &Record)> = records
        .iter()
        .enumerate()
        .filter(|(_, r)| 1000.0 * r.wall > budget_ms)
        .collect();
    println!("hitches (frames over {budget_ms:.2} ms): {}", hitches.len());
    for (i, r) in hitches {
        let f = &r.frame;
        println!(
            "  frame {i} t {:.4} s | {:.2} ms (report {:.2}) | steps {} rejected {} (unconverged {}) reaching {} Newton {} GMRES {} modal {}",
            r.time,
            1000.0 * r.wall,
            1000.0 * f.wall,
            f.steps,
            f.rejected,
            f.rejected_unconverged,
            f.reaching,
            f.newton,
            f.gmres,
            f.modal
        );
    }
    modal_summary(records);
}

/// The frames the modes carried (`TotalLagrangianFrame::modal`) and the
/// path's counters since the start at the last frame (`modal_frames`,
/// `hand_backs`): all zero unless `--modal-flight` is on.
fn modal_summary(records: &[Record]) {
    let modal: Vec<String> = records
        .iter()
        .enumerate()
        .filter(|(_, r)| r.frame.modal)
        .map(|(i, _)| i.to_string())
        .collect();
    let last = records.last().map(|r| r.frame).unwrap_or_default();
    println!(
        "modal frames: {} [{}] | modal_frames {} hand_backs {} (since the start)",
        modal.len(),
        modal.join(" "),
        last.modal_frames,
        last.hand_backs
    );
}

/// The per-frame record as JSON, in the replay layout of
/// `tests/total_lagrangian_path.rs` (`write_replay`: a header, then frames
/// with time, positions in m, energy, heat, balance, contacts), each frame
/// adding its timing and counts.
fn write_json(path: &str, sim: &Simulation, args: &Args, records: &[Record]) {
    let config = sim.config();
    let mut body = String::new();
    let _ = writeln!(
        body,
        "{{\"header\":{{\"scene\":\"tl_live drop\",\"dx_m\":{},\"frame_s\":{},\"gravity_m_s2\":{},\"floor_y_m\":{},\"budget_ms\":{},\"threads\":{},\"commit\":\"{}\",\"debug_assertions\":{},\"paced\":{},\"modal_flight\":{},\"path\":\"total Lagrangian, weighted-gap contact\",\"precomputed\":true}},\"frames\":[",
        config.dx_meters,
        config.dt,
        config.gravity.length() * config.dx_meters,
        config.boundary_thickness as f32 * config.dx_meters,
        1000.0 / args.frame_hz,
        rayon::current_num_threads(),
        git_commit(),
        cfg!(debug_assertions),
        args.paced,
        args.modal_flight
    );
    for (i, r) in records.iter().enumerate() {
        let f = &r.frame;
        let positions: Vec<String> = r
            .positions
            .iter()
            .map(|p| format!("[{:.6},{:.6}]", p[0], p[1]))
            .collect();
        let _ = write!(
            body,
            "{}{{\"index\":{i},\"time\":{:.6},\"wall_ms\":{:.4},\"report_wall_ms\":{:.4},\"steps\":{},\"rejected\":{},\"rejected_unconverged\":{},\"reaching\":{},\"newton\":{},\"gmres\":{},\"contacts\":{},\"energy\":{:.9e},\"heat\":{:.9e},\"balance\":{:.3e},\"modal\":{},\"modal_energy_change\":{:.3e},\"modal_frames\":{},\"hand_backs\":{},\"positions\":[{}]}}",
            if i == 0 { "" } else { ",\n" },
            r.time,
            1000.0 * r.wall,
            1000.0 * f.wall,
            f.steps,
            f.rejected,
            f.rejected_unconverged,
            f.reaching,
            f.newton,
            f.gmres,
            f.contacts,
            f.energy,
            f.heat,
            f.balance,
            f.modal,
            f.modal_energy_change,
            f.modal_frames,
            f.hand_backs,
            positions.join(",")
        );
    }
    body.push_str("\n]}\n");
    match std::fs::write(path, body) {
        Ok(()) => println!("per-frame record written: {path}"),
        Err(e) => eprintln!("could not write {path}: {e}"),
    }
}

fn live(args: &Args) {
    header(args);
    let mut sim = scene(args);
    println!(
        "scene: {} particles, {}x{} block, lift {} cells, tolerance {:e}, {} frames of 1/{} s",
        sim.particles().len(),
        args.width,
        args.height,
        args.lift,
        args.tolerance,
        args.frames,
        args.frame_hz
    );
    let (records, elapsed) = run(&mut sim, args);
    summary(args, &records, elapsed);
    if let Some(path) = &args.json {
        write_json(path, &sim, args, &records);
    }
}

fn main() {
    let args = match parse_args() {
        Ok(args) => args,
        Err(message) => {
            if !message.is_empty() {
                eprintln!("{message}");
            }
            eprintln!("{USAGE}");
            std::process::exit(if message.is_empty() { 0 } else { 2 });
        }
    };
    if args.threads == 0 {
        live(&args);
    } else {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.threads)
            .build()
            .expect("a thread pool")
            .install(|| live(&args));
    }
}

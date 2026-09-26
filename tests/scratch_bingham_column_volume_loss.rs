//! Where does the slump demo's middle column lose six percent of itself?
//!
//! Measured on `bingham_slump_probe` at the scene's own geometry, mean `J`
//! by column over eighteen seconds:
//!
//! ```text
//!   tau_0      first sample   at 18 s    change
//!      2 Pa      0.9998       1.0004     +0.0006
//!     60 Pa      0.9894       0.9380     -0.0514
//!   1200 Pa      0.9925       0.9787     -0.0138
//! ```
//!
//! Two separate things, and an earlier report of this wrongly called it
//! one. There is an OFFSET already present at the first sample, and there
//! is a loss that keeps ACCUMULATING after it. Neither is explained by the
//! columns' own weight: `rho g h / 2` over a 22 mm column is 106 Pa against
//! a bulk modulus of 78,480, so its own load buys `J = 0.9986`. The middle
//! column ends 46 times past that, the right one about 9.
//!
//! The standing suspect is arithmetic, not physics. `BinghamFluidMaterial`
//! takes its elastoviscoplastic branch here (all three columns have a
//! storage modulus), and that branch rebuilds `F` through an SVD every
//! substep: `svd2`, then `hencky_strains`, then a deviatoric rescale, then
//! `exp`, then `reconstruct_f`. In exact arithmetic the rescale preserves
//! volume by construction, because `eps_projected = dev * k + tr/2` leaves
//! `tr` alone and touches only the traceless part. In f32 every one of
//! those steps rounds relative to a singular value near one, which is the
//! same absorption already measured and fixed on the viscous path. An
//! isolated probe (`tests/scratch_evp_volume_rounding.rs`) put it at about
//! 5e-8 a substep, of the right sign. Over the roughly 200,000 substeps of
//! eighteen seconds that is about -1 percent: the right column's size, and
//! five times too small for the middle one.
//!
//! What might make the middle one different is that it sits exactly ON its
//! yield surface, so it takes the plastic branch every substep, where the
//! right column is elastic most of the time.
//!
//! Three cheap tests, one per hypothesis, all on the demo geometry and one
//! column at a time so nothing else can contribute: the DISTRIBUTION of J
//! rather than its mean; HALVING the frame step; and raising the YIELD
//! STRESS at a fixed storage modulus.
//!
//! # What they found, which is none of the above
//!
//! All three came back negative, and the fourth test says why.
//!
//! ```text
//!   1. a 60 Pa column ALONE ends six seconds at mean J = 1.00000, with its
//!      height bands all within 0.001 of one. There is nothing to explain.
//!   2. the frame step from 4 ms down to 0.5 ms: +0.0003, +0.00005, -0.0003,
//!      -0.0004 percent a second. No trend, so no per-substep arithmetic.
//!   3. yield stress 60, 120, 600, 6000 Pa at a fixed storage modulus:
//!      1.00000, 0.99957, 0.99831, 0.99831. Raising it makes the loss
//!      slightly WORSE, so it is not the plastic return path either.
//! ```
//!
//! The variable is not the column. Three IDENTICAL columns, same yield
//! stress, same geometry, same everything, in one simulation:
//!
//! ```text
//!   slot   created by            initial volume    mean J    worst J
//!      0   Simulation::new         0.250000        0.99907    0.986
//!      1   add_body                0.280036        0.94304    0.603
//!      2   add_body                0.280036        0.94441    0.601
//! ```
//!
//! So it is the SPAWN PATH. With no steps run at all, the same column built
//! each way carries a different initial volume: `Simulation::new` gives
//! every particle exactly 0.250000, which is the geometric packing at
//! spacing 0.5 and cannot be a measurement, while `add_body` measures and
//! gets 0.250000 to 0.640000, inflating free-surface particles by up to
//! 2.56 times. Initial volume multiplies stress directly, so those
//! particles push 2.56 times too hard and the body crushes itself.
//!
//! Sweeping what `Simulation::new` does on its own, four configurations,
//! no steps: 0.025000, 0.250000, 0.001000, 0.250000, every one of them
//! UNIFORM (min equals max) and every one of them equal to the particle's
//! mass over its density. It never measures its body's packing, at any dx
//! and from either mass source.
//!
//! `tests/spawn_contract.rs` is not wrong and is not insensitive by design:
//! in ITS scene both bodies read identically, 0.300906 with the same
//! 0.25-to-0.64 spread, so its equality assertion is satisfied by two
//! measured bodies. Why the first body is measured there and not here is
//! the open end of this trail, and so is the question of which of the two
//! values is right. The uniform mass-over-density one is what an
//! undeformed freshly spawned body should carry; the measured one carries
//! the free-surface density underestimate this engine documents elsewhere,
//! which is what makes it inflate edges.
//!
//!   cargo test --profile quick --all-features --test scratch_bingham_column_volume_loss -- --ignored --nocapture
extern crate emerge_engine as emerge;

use emerge::{
    BinghamFluidMaterial, BinghamProps, FromSI, SimConfig, Simulation, SlipBoundary, SpawnRegion,
};
use glam::{IVec2, Vec2};

// The slump demo's own scene, constant for constant.
const GRID: usize = 64;
const DX_M: f32 = 0.002;
const RHO_KG_M3: f32 = 1000.0;
const ETA_PA_S: f32 = 0.5;
const YIELD_STRAIN: f32 = 0.05;
const COLUMN_CELLS: IVec2 = IVec2::new(10, 20);
const FLOOR_CELLS: f32 = 2.0;
const BULK_PA: f32 = 78_480.0;

fn env(name: &str, default: f32) -> f32 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

struct Run {
    sim: Simulation,
    substeps: usize,
}

/// One column of the slump scene, alone in the tank. `shear_modulus_pa` is
/// passed separately from `yield_stress_pa` so the third test can move the
/// yield surface without moving the stiffness, the wave speed or the
/// timestep with it.
fn run_column(yield_stress_pa: f32, shear_modulus_pa: f32, seconds: f32, dt: f32) -> Run {
    let config = SimConfig {
        min_dt: 1.0e-6,
        sleep_threshold: 0.0,
        max_substeps_per_step: 512,
        ..SimConfig::earth(GRID, DX_M, dt)
    };
    let props = BinghamProps {
        rho_kg_m3: RHO_KG_M3,
        eta_pa_s: ETA_PA_S,
        bulk_modulus_pa: BULK_PA,
        yield_stress_pa,
        shear_modulus_pa,
        cavitation_pressure_pa: BinghamProps::air_entrained_cavitation_pressure(),
    };
    let spawn = SpawnRegion {
        spacing: 0.5,
        box_size: COLUMN_CELLS,
        box_center: Vec2::new(GRID as f32 * 0.5, FLOOR_CELLS + COLUMN_CELLS.y as f32 * 0.5),
        material_id: 0,
        initial_velocity_scale: 0.0,
        ..SpawnRegion::for_sim(&config)
    }
    .mass_from(&props, &config);
    let mut sim = Simulation::new(config, spawn)
        .with_default_material(Box::new(BinghamFluidMaterial::from_physical(
            &props, &config,
        )))
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));
    let mut substeps = 0usize;
    for _ in 0..(seconds / dt).round() as usize {
        sim.step();
        substeps += sim.diagnostics_snapshot().substeps_last_step;
    }
    Run { sim, substeps }
}

/// Every particle's volume ratio, sorted, so percentiles are honest.
fn sorted_j(sim: &Simulation) -> Vec<f32> {
    let p = sim.particles();
    let mut v: Vec<f32> = (0..p.len())
        .map(|i| p.deformation_gradient[i].determinant())
        .collect();
    v.sort_by(f32::total_cmp);
    v
}

fn mean(v: &[f32]) -> f64 {
    v.iter().map(|&x| f64::from(x)).sum::<f64>() / v.len().max(1) as f64
}

fn pct(v: &[f32], p: f64) -> f32 {
    v[((v.len() - 1) as f64 * p).round() as usize]
}

// --- 1. the distribution, not the mean --------------------------------------

#[test]
#[ignore = "diagnostic probe kept for reruns, not part of the CI suite"]
fn column_volume_distribution() {
    let seconds = env("COLUMN_SECONDS", 18.0);
    let dt = env("COLUMN_DT", 0.002);
    let tau0 = env("COLUMN_TAU0", 60.0);
    println!("tau_0 {tau0} Pa column alone, {seconds} s at {dt} s a frame");
    println!("  when        mean J      min       p1       p5      p50      p95      max");

    for (label, t) in [("step 1", dt), ("step 2", 2.0 * dt), ("end", seconds)] {
        let run = run_column(tau0, tau0 / YIELD_STRAIN, t, dt);
        let v = sorted_j(&run.sim);
        println!(
            "  {label:<9}  {:>8.5}  {:>7.5}  {:>7.5}  {:>7.5}  {:>7.5}  {:>7.5}  {:>7.5}",
            mean(&v),
            v[0],
            pct(&v, 0.01),
            pct(&v, 0.05),
            pct(&v, 0.50),
            pct(&v, 0.95),
            v[v.len() - 1]
        );
    }

    // And WHERE in the column the loss sits. A mean can be carried by a
    // thin layer pressed against the floor; a profile cannot hide that.
    let run = run_column(tau0, tau0 / YIELD_STRAIN, seconds, dt);
    let p = run.sim.particles();
    let (mut lo, mut hi) = (f32::MAX, f32::MIN);
    for i in 0..p.len() {
        lo = lo.min(p.x[i].y);
        hi = hi.max(p.x[i].y);
    }
    println!("  height band (floor to top)      mean J    particles");
    const BANDS: usize = 5;
    for b in 0..BANDS {
        let (a, z) = (
            lo + (hi - lo) * b as f32 / BANDS as f32,
            lo + (hi - lo) * (b + 1) as f32 / BANDS as f32,
        );
        let band: Vec<f32> = (0..p.len())
            .filter(|&i| p.x[i].y >= a && (p.x[i].y < z || b == BANDS - 1))
            .map(|i| p.deformation_gradient[i].determinant())
            .collect();
        println!(
            "  {b}: {:>5.1} to {:>5.1} mm             {:>7.5}   {:>6}",
            (a - FLOOR_CELLS) * DX_M * 1000.0,
            (z - FLOOR_CELLS) * DX_M * 1000.0,
            mean(&band),
            band.len()
        );
    }
}

// --- 2. halve the frame step ------------------------------------------------

#[test]
#[ignore = "diagnostic probe kept for reruns, not part of the CI suite"]
fn column_volume_loss_against_substep() {
    let seconds = env("COLUMN_SECONDS", 6.0);
    let tau0 = env("COLUMN_TAU0", 60.0);
    println!("tau_0 {tau0} Pa column, {seconds} s, frame step halved three times.");
    println!("A per-substep arithmetic loss is constant PER SUBSTEP, so more substeps");
    println!("for the same physical time means proportionally more of it.");
    println!("  frame dt    substeps    mean J     loss per second   loss per substep");

    for dt in [0.004f32, 0.002, 0.001, 0.0005] {
        let run = run_column(tau0, tau0 / YIELD_STRAIN, seconds, dt);
        let v = sorted_j(&run.sim);
        let m = mean(&v);
        println!(
            "  {dt:>8.4}  {:>10}  {m:>8.5}   {:>15.6} %   {:>16.3e}",
            run.substeps,
            100.0 * (m - 1.0) / f64::from(seconds),
            (m - 1.0) / run.substeps.max(1) as f64
        );
    }
}

// --- 3. move the yield surface, hold the stiffness --------------------------

#[test]
#[ignore = "diagnostic probe kept for reruns, not part of the CI suite"]
fn column_volume_loss_against_yield() {
    let seconds = env("COLUMN_SECONDS", 6.0);
    let dt = env("COLUMN_DT", 0.002);
    // The middle column's own storage modulus, held fixed across the sweep:
    // the wave speed, the timestep and the elastic response stay identical,
    // and only the yield surface moves.
    let shear = 60.0 / YIELD_STRAIN;
    println!(
        "storage modulus fixed at {shear} Pa, only the yield surface moves, {seconds} s at {dt}"
    );
    println!("  tau_0 Pa    substeps    mean J     loss per second   loss per substep");

    for tau0 in [60.0f32, 120.0, 600.0, 6000.0] {
        let run = run_column(tau0, shear, seconds, dt);
        let v = sorted_j(&run.sim);
        let m = mean(&v);
        println!(
            "  {tau0:>8.0}  {:>10}  {m:>8.5}   {:>15.6} %   {:>16.3e}",
            run.substeps,
            100.0 * (m - 1.0) / f64::from(seconds),
            (m - 1.0) / run.substeps.max(1) as f64
        );
    }
}

// --- 4. one column alone against all three together -------------------------

/// The three tests above all run ONE column, and none of them reproduces
/// the demo at all: a 60 Pa column alone ends six seconds at mean
/// `J = 1.00000`, where the same column in `bingham_slump_probe` reads
/// 0.9894 at its first sample and 0.938 at the end. Halving the step does
/// nothing, and raising the yield stress makes the loss slightly WORSE
/// rather than better, so it is not the plastic return path either.
///
/// So the variable is not the column. The demo differs in one structural
/// way: it runs all three columns in ONE simulation, as three materials in
/// one registry, where these tests run one material alone. This puts the
/// two side by side on identical geometry.
#[test]
#[ignore = "diagnostic probe kept for reruns, not part of the CI suite"]
fn column_volume_alone_against_three_together() {
    let seconds = env("COLUMN_SECONDS", 6.0);
    let dt = env("COLUMN_DT", 0.002);
    let taus = [2.0f32, 60.0, 1200.0];
    let xs = [12.0f32, 32.0, 52.0];

    println!("{seconds} s at {dt} s a frame, the demo geometry, mean J per column");
    println!("  tau_0 Pa     alone     together    difference");

    // Alone: one material, one column, three separate simulations.
    let mut alone = [0.0f64; 3];
    for (k, &tau0) in taus.iter().enumerate() {
        let run = run_column(tau0, tau0 / YIELD_STRAIN, seconds, dt);
        alone[k] = mean(&sorted_j(&run.sim));
    }

    // Together: one simulation, three materials, exactly as the demo
    // builds it, including each column standing at its own x.
    let config = SimConfig {
        min_dt: 1.0e-6,
        sleep_threshold: 0.0,
        max_substeps_per_step: 512,
        ..SimConfig::earth(GRID, DX_M, dt)
    };
    let props = |tau0: f32| BinghamProps {
        rho_kg_m3: RHO_KG_M3,
        eta_pa_s: ETA_PA_S,
        bulk_modulus_pa: BULK_PA,
        yield_stress_pa: tau0,
        shear_modulus_pa: tau0 / YIELD_STRAIN,
        cavitation_pressure_pa: BinghamProps::air_entrained_cavitation_pressure(),
    };
    let spawn = |slot: usize| {
        SpawnRegion {
            spacing: 0.5,
            box_size: COLUMN_CELLS,
            box_center: Vec2::new(xs[slot], FLOOR_CELLS + COLUMN_CELLS.y as f32 * 0.5),
            material_id: slot as u32,
            initial_velocity_scale: 0.0,
            ..SpawnRegion::for_sim(&config)
        }
        .mass_from(&props(taus[slot]), &config)
    };
    let mut sim = Simulation::new(config, spawn(0))
        .with_default_material(Box::new(BinghamFluidMaterial::from_physical(
            &props(taus[0]),
            &config,
        )))
        .with_material(
            1,
            Box::new(BinghamFluidMaterial::from_physical(
                &props(taus[1]),
                &config,
            )),
        )
        .with_material(
            2,
            Box::new(BinghamFluidMaterial::from_physical(
                &props(taus[2]),
                &config,
            )),
        )
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));
    let _ = sim.add_body(spawn(1));
    let _ = sim.add_body(spawn(2));
    for _ in 0..(seconds / dt).round() as usize {
        sim.step();
    }
    let p = sim.particles();
    for (k, &tau0) in taus.iter().enumerate() {
        let v: Vec<f32> = (0..p.len())
            .filter(|&i| p.material_id[i] == k as u32)
            .map(|i| p.deformation_gradient[i].determinant())
            .collect();
        let together = mean(&v);
        println!(
            "  {tau0:>8.0}   {:>8.5}    {together:>8.5}    {:>+10.5}   ({} particles)",
            alone[k],
            together - alone[k],
            v.len()
        );
    }
}

// --- 5. which companion does it, and how -----------------------------------

/// Test 4 found that a 60 Pa column alone ends at mean `J = 1.00000` and
/// the SAME column in the same tank as the other two ends at 0.94106. That
/// is a six percent volume loss caused by bodies standing twenty cells
/// away. Three things could carry it and this separates them.
///
///   - CONTACT. The columns are not as far apart as they look once they
///     spread: the soft one deposits to a half-width of 28 mm from x = 12,
///     reaching x = 26, and the middle one reaches back to x = 21. They
///     meet. MPM shares one velocity field by default, so two materials
///     that meet are welded, not merely touching.
///   - THE SHARED TIMESTEP. One simulation takes the smallest step any of
///     its materials demands, so the soft column is integrated at the
///     stiff one's rate. Test 2 already argues against this, having found
///     no per-substep loss at any step from 4 ms down to 0.5 ms, but it
///     tested the step, not the company.
///   - THE REGISTRY. Three materials in one slot table rather than one.
///
/// Each row changes exactly one of those.
#[test]
#[ignore = "diagnostic probe kept for reruns, not part of the CI suite"]
fn which_companion_costs_the_middle_column_its_volume() {
    let seconds = env("COLUMN_SECONDS", 6.0);
    let dt = env("COLUMN_DT", 0.002);

    println!("{seconds} s at {dt} s a frame, always reading the 60 Pa column");
    println!("  companions                          grid   mean J of the 60 Pa column");

    // (label, the yield stresses standing in the tank, x of each, grid)
    let cases: [(&str, &[f32], &[f32], usize); 6] = [
        ("alone", &[60.0], &[32.0], GRID),
        (
            "two more of itself",
            &[60.0, 60.0, 60.0],
            &[12.0, 32.0, 52.0],
            GRID,
        ),
        ("the soft one only", &[60.0, 2.0], &[32.0, 12.0], GRID),
        ("the stiff one only", &[60.0, 1200.0], &[32.0, 52.0], GRID),
        (
            "both, the demo",
            &[60.0, 2.0, 1200.0],
            &[32.0, 12.0, 52.0],
            GRID,
        ),
        (
            "both, too far to meet",
            &[60.0, 2.0, 1200.0],
            &[64.0, 20.0, 108.0],
            128,
        ),
    ];

    for (label, taus, xs, grid) in cases {
        let config = SimConfig {
            min_dt: 1.0e-6,
            sleep_threshold: 0.0,
            max_substeps_per_step: 512,
            ..SimConfig::earth(grid, DX_M, dt)
        };
        let props = |tau0: f32| BinghamProps {
            rho_kg_m3: RHO_KG_M3,
            eta_pa_s: ETA_PA_S,
            bulk_modulus_pa: BULK_PA,
            yield_stress_pa: tau0,
            shear_modulus_pa: tau0 / YIELD_STRAIN,
            cavitation_pressure_pa: BinghamProps::air_entrained_cavitation_pressure(),
        };
        let spawn = |slot: usize| {
            SpawnRegion {
                spacing: 0.5,
                box_size: COLUMN_CELLS,
                box_center: Vec2::new(xs[slot], FLOOR_CELLS + COLUMN_CELLS.y as f32 * 0.5),
                material_id: slot as u32,
                initial_velocity_scale: 0.0,
                ..SpawnRegion::for_sim(&config)
            }
            .mass_from(&props(taus[slot]), &config)
        };
        // Slot 0 is always the 60 Pa column, so the read below is the same
        // body in every row.
        let mut sim = Simulation::new(config, spawn(0))
            .with_default_material(Box::new(BinghamFluidMaterial::from_physical(
                &props(taus[0]),
                &config,
            )))
            .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));
        for (slot, &tau) in taus.iter().enumerate().skip(1) {
            sim.set_material(
                slot as u32,
                Box::new(BinghamFluidMaterial::from_physical(&props(tau), &config)),
            );
            let _ = sim.add_body(spawn(slot));
        }
        for _ in 0..(seconds / dt).round() as usize {
            sim.step();
        }
        let p = sim.particles();
        let v: Vec<f32> = (0..p.len())
            .filter(|&i| p.material_id[i] == 0)
            .map(|i| p.deformation_gradient[i].determinant())
            .collect();
        println!(
            "  {label:<34} {grid:>5}   {:>8.5}   ({} particles)",
            mean(&v),
            v.len()
        );
    }
}

// --- 6. the slot, not the material and not the neighbour --------------------

/// Test 5 read the 60 Pa column at material slot 0 in the demo's own
/// arrangement and found 0.99982. Test 4 read the SAME column, same yield
/// stress, same x, same two neighbours, at slot 1, and found 0.94106.
/// Position is therefore ruled out, the neighbours are ruled out, and the
/// material is ruled out. What is left is the slot.
///
/// This removes the last thing that could confound it: three columns that
/// are identical in every respect INCLUDING their yield stress, read one
/// slot at a time. If slot 0 holds its volume and slots 1 and 2 do not,
/// bodies added through `add_body` are not being integrated the way the
/// body `Simulation::new` starts with is.
#[test]
#[ignore = "diagnostic probe kept for reruns, not part of the CI suite"]
fn three_identical_columns_read_slot_by_slot() {
    let seconds = env("COLUMN_SECONDS", 6.0);
    let dt = env("COLUMN_DT", 0.002);
    let xs = [12.0f32, 32.0, 52.0];

    for tau0 in [60.0f32, 2.0] {
        println!("three identical {tau0} Pa columns, {seconds} s at {dt} s a frame");
        let config = SimConfig {
            min_dt: 1.0e-6,
            sleep_threshold: 0.0,
            max_substeps_per_step: 512,
            ..SimConfig::earth(GRID, DX_M, dt)
        };
        let props = BinghamProps {
            rho_kg_m3: RHO_KG_M3,
            eta_pa_s: ETA_PA_S,
            bulk_modulus_pa: BULK_PA,
            yield_stress_pa: tau0,
            shear_modulus_pa: tau0 / YIELD_STRAIN,
            cavitation_pressure_pa: BinghamProps::air_entrained_cavitation_pressure(),
        };
        let spawn = |slot: usize| {
            SpawnRegion {
                spacing: 0.5,
                box_size: COLUMN_CELLS,
                box_center: Vec2::new(xs[slot], FLOOR_CELLS + COLUMN_CELLS.y as f32 * 0.5),
                material_id: slot as u32,
                initial_velocity_scale: 0.0,
                ..SpawnRegion::for_sim(&config)
            }
            .mass_from(&props, &config)
        };
        let mut sim = Simulation::new(config, spawn(0))
            .with_default_material(Box::new(BinghamFluidMaterial::from_physical(
                &props, &config,
            )))
            .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));
        for slot in 1..3 {
            sim.set_material(
                slot as u32,
                Box::new(BinghamFluidMaterial::from_physical(&props, &config)),
            );
            let _ = sim.add_body(spawn(slot));
        }
        for _ in 0..(seconds / dt).round() as usize {
            sim.step();
        }
        let p = sim.particles();
        println!("  slot   x     mean J      min J    particles   mean initial_volume");
        for slot in 0..3u32 {
            let idx: Vec<usize> = (0..p.len()).filter(|&i| p.material_id[i] == slot).collect();
            let v: Vec<f32> = idx
                .iter()
                .map(|&i| p.deformation_gradient[i].determinant())
                .collect();
            let v0 = idx
                .iter()
                .map(|&i| f64::from(p.initial_volume[i]))
                .sum::<f64>()
                / idx.len().max(1) as f64;
            println!(
                "  {slot:>4}  {:>4.0}   {:>8.5}   {:>8.5}    {:>7}   {v0:>16.6e}",
                xs[slot as usize],
                mean(&v),
                v.iter().copied().fold(f32::INFINITY, f32::min),
                idx.len()
            );
        }
    }
}

// --- 7. the two spawn paths, with no physics at all -------------------------

/// Test 6 showed three identical columns carrying two different initial
/// volumes: 0.250000 for the one `Simulation::new` starts with, 0.280036
/// for the two that `add_body` adds. Initial volume multiplies stress
/// directly, so those are not the same material, and the two added columns
/// crush to a mean `J` of 0.943 with their worst particles at 0.60 while
/// the first holds 0.999.
///
/// This takes the physics out entirely. No steps are run. The same column
/// is built by each path and its initial volume read straight back.
#[test]
#[ignore = "diagnostic probe kept for reruns, not part of the CI suite"]
fn the_two_spawn_paths_disagree_on_initial_volume() {
    let config = SimConfig {
        min_dt: 1.0e-6,
        sleep_threshold: 0.0,
        ..SimConfig::earth(GRID, DX_M, 0.002)
    };
    let props = BinghamProps {
        rho_kg_m3: RHO_KG_M3,
        eta_pa_s: ETA_PA_S,
        bulk_modulus_pa: BULK_PA,
        yield_stress_pa: 60.0,
        shear_modulus_pa: 60.0 / YIELD_STRAIN,
        cavitation_pressure_pa: BinghamProps::air_entrained_cavitation_pressure(),
    };
    let column = |x: f32, slot: u32| {
        SpawnRegion {
            spacing: 0.5,
            box_size: COLUMN_CELLS,
            box_center: Vec2::new(x, FLOOR_CELLS + COLUMN_CELLS.y as f32 * 0.5),
            material_id: slot,
            initial_velocity_scale: 0.0,
            ..SpawnRegion::for_sim(&config)
        }
        .mass_from(&props, &config)
    };
    let material = || Box::new(BinghamFluidMaterial::from_physical(&props, &config));

    let stats = |sim: &Simulation, slot: u32| -> (f64, f32, f32, usize) {
        let p = sim.particles();
        let v: Vec<f32> = (0..p.len())
            .filter(|&i| p.material_id[i] == slot)
            .map(|i| p.initial_volume[i])
            .collect();
        (
            v.iter().map(|&x| f64::from(x)).sum::<f64>() / v.len().max(1) as f64,
            v.iter().copied().fold(f32::INFINITY, f32::min),
            v.iter().copied().fold(f32::NEG_INFINITY, f32::max),
            v.len(),
        )
    };

    // Path A: the body `Simulation::new` starts with.
    let a = Simulation::new(config, column(32.0, 0)).with_default_material(material());
    let (mean_a, min_a, max_a, n_a) = stats(&a, 0);

    // Path B: the identical body, at the identical place, added afterwards.
    // The first body is parked far away so it cannot touch this one.
    let mut b = Simulation::new(config, column(8.0, 0)).with_default_material(material());
    b.set_material(1, material());
    let _ = b.add_body(column(32.0, 1));
    let (mean_b, min_b, max_b, n_b) = stats(&b, 1);

    println!("the same column, no steps run, initial volume in cells squared");
    println!("  path                  mean         min         max      particles");
    println!("  Simulation::new   {mean_a:>10.6}  {min_a:>10.6}  {max_a:>10.6}  {n_a:>10}");
    println!("  add_body          {mean_b:>10.6}  {min_b:>10.6}  {max_b:>10.6}  {n_b:>10}");
    println!(
        "  add_body is {:.4} times the other. Geometric packing at spacing 0.5 is {:.6}.",
        mean_b / mean_a,
        0.5f32 * 0.5
    );
}

// --- 8. why the phase 4 gate does not see it --------------------------------

/// `tests/spawn_contract.rs` compares the two paths' mean initial volume
/// and requires them within a thousandth, and it passes. Test 7 finds them
/// 12 percent apart. One of the two measurements is framed wrong, and this
/// runs the gate's own scene with the spread printed rather than the mean.
#[test]
#[ignore = "diagnostic probe kept for reruns, not part of the CI suite"]
fn the_phase_four_gate_scene_with_its_spread_shown() {
    use emerge::NeoHookeanMaterial;
    let config = SimConfig {
        min_dt: 1.0e-7,
        max_substeps_per_step: 128,
        ..SimConfig::earth(48, 0.01, 0.0005)
    };
    let mass = 1000.0 * (0.5 * 0.01f32).powi(2);
    let spawn = |x: f32| SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new(6, 12),
        box_center: Vec2::new(x, 3.0 + 12.0 * 0.5),
        material_id: 0,
        mass_override: Some(mass),
        initial_velocity_scale: 0.0,
        ..SpawnRegion::for_sim(&config)
    };
    let mut sim = Simulation::new(config, spawn(14.0))
        .with_default_material(Box::new(NeoHookeanMaterial::from_young_modulus(2.0e5, 0.3)))
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));
    let first = sim.particles().len();
    let _ = sim.add_body(spawn(34.0));
    let total = sim.particles().len();

    println!("the phase 4 gate scene, initial volume, no steps run");
    println!("  body                   mean         min         max   particles");
    for (label, range) in [("Simulation::new", 0..first), ("add_body", first..total)] {
        let p = sim.particles();
        let v: Vec<f32> = range.clone().map(|i| p.initial_volume[i]).collect();
        println!(
            "  {label:<16}  {:>10.6}  {:>10.6}  {:>10.6}  {:>9}",
            v.iter().map(|&x| f64::from(x)).sum::<f64>() / v.len() as f64,
            v.iter().copied().fold(f32::INFINITY, f32::min),
            v.iter().copied().fold(f32::NEG_INFINITY, f32::max),
            v.len()
        );
    }
}

// --- 9. what makes `Simulation::new` skip its own measurement ---------------

/// Test 7: in the demo's configuration `Simulation::new` gives every
/// particle exactly 0.250000, which is the geometric packing and cannot be
/// a measurement, while `add_body` measures and gets 0.25 to 0.64. Test 8:
/// in the phase 4 gate's configuration BOTH measure and agree exactly. So
/// `Simulation::new`'s own estimate silently does nothing in some
/// configurations. This sweeps the two things that differ between those
/// scenes, one at a time, with no steps run.
#[test]
#[ignore = "diagnostic probe kept for reruns, not part of the CI suite"]
fn what_makes_the_first_body_skip_its_measurement() {
    println!("initial volume of the FIRST body, `Simulation::new` only, no steps");
    println!("  dx_m     mass source      mean         min         max");
    for dx in [0.01f32, 0.002] {
        for override_mass in [true, false] {
            let config = SimConfig {
                min_dt: 1.0e-6,
                ..SimConfig::earth(GRID, dx, 0.002)
            };
            let props = BinghamProps {
                rho_kg_m3: RHO_KG_M3,
                eta_pa_s: ETA_PA_S,
                bulk_modulus_pa: BULK_PA,
                yield_stress_pa: 60.0,
                shear_modulus_pa: 60.0 / YIELD_STRAIN,
                cavitation_pressure_pa: BinghamProps::air_entrained_cavitation_pressure(),
            };
            let base = SpawnRegion {
                spacing: 0.5,
                box_size: COLUMN_CELLS,
                box_center: Vec2::new(32.0, FLOOR_CELLS + COLUMN_CELLS.y as f32 * 0.5),
                material_id: 0,
                initial_velocity_scale: 0.0,
                ..SpawnRegion::for_sim(&config)
            };
            let spawn = if override_mass {
                SpawnRegion {
                    mass_override: Some(RHO_KG_M3 * (0.5 * dx).powi(2)),
                    ..base
                }
            } else {
                base.mass_from(&props, &config)
            };
            let sim = Simulation::new(config, spawn).with_default_material(Box::new(
                BinghamFluidMaterial::from_physical(&props, &config),
            ));
            let p = sim.particles();
            let v: Vec<f32> = (0..p.len()).map(|i| p.initial_volume[i]).collect();
            println!(
                "  {dx:<7} {:<15} {:>10.6}  {:>10.6}  {:>10.6}",
                if override_mass {
                    "mass_override"
                } else {
                    "mass_from"
                },
                v.iter().map(|&x| f64::from(x)).sum::<f64>() / v.len() as f64,
                v.iter().copied().fold(f32::INFINITY, f32::min),
                v.iter().copied().fold(f32::NEG_INFINITY, f32::max)
            );
        }
    }
}

// --- 10. does the lattice volume close the hydrostatic gap? ----------------

/// `tests/spawn_contract.rs` measures a settled elastic column carrying 13.7
/// percent less than its own weight at 1 cm cells, and its doc shows that
/// halving the cells halves the error, first order, which it reads as
/// discretisation. It may be. But an EDGE artefact converges first order
/// too, because the fraction of particles on the free surface halves each
/// time the cells do, so that table cannot tell the two apart.
///
/// And there is a candidate edge artefact: the same volume estimate that
/// inflated the Bingham columns inflates THIS column's free-surface
/// particles up to 2.56 times, and here nothing overwrites it afterwards,
/// because an elastic material does not set its own volume. Initial volume
/// enters the grid force a particle exerts, so inflated edges push harder.
///
/// This is the gate's own scene, constant for constant, run twice: once as
/// it spawns, and once with every particle's initial volume set to its
/// lattice cell, `spacing^2`, which is exact for a lattice by construction.
///
/// # What it found
///
/// The lattice volume does NOT close the gap. It overshoots it the other
/// way. At 1 cm cells, lower half of the column, time-averaged:
///
/// ```text
///   initial volume        measured     rho g h     error
///   as spawned (0.3009)    -755.6 Pa   -881.4 Pa   +14.3 %  carries less
///   lattice    (0.2500)   -1116.2 Pa   -881.3 Pa   -26.7 %  carries more
/// ```
///
/// Changing ONLY the free-surface particles' volume swings the answer by
/// forty points, so the gap is dominated by the edges, but neither value is
/// right. Refined with `HYDRO_REFINE`, and read only where `rho g h` still
/// comes out near 881 Pa, which is the check that the column is actually at
/// rest:
///
/// ```text
///   initial volume     1 cm      0.5 cm     0.25 cm
///   as spawned        +14.3 %    +5.9 %     not at rest
///   lattice           -26.7 %    not at rest  -6.7 %
/// ```
///
/// Both converge at first order, from opposite sides: -26.7 over a fourfold
/// refinement is -6.7 exactly. That is what two discretisation errors of
/// opposite sign look like, not a defect. The two rows marked "not at rest"
/// read `rho g h` of 666 and 615 Pa for a column whose weight is 881: the
/// column lost its static equilibrium. Which variant loses it flips between
/// resolutions, so it is a stability property of this scene and not an
/// effect of the volume. It is also NOT the substep cap: scaling that cap
/// with the refinement left every number identical to the digit.
///
/// So at a practical resolution the lattice contract roughly DOUBLES the
/// error for an elastic body. Only two valid points per variant, though.
#[test]
#[ignore = "diagnostic probe kept for reruns, not part of the CI suite"]
fn lattice_volume_against_the_hydrostatic_gap() {
    use emerge::{MaterialModel, NeoHookeanMaterial};
    let spacing: f32 = 0.5;
    let young: f32 = 2.0e5;
    let poisson: f32 = 0.3;
    // Refinement keeps the PHYSICAL column (6 x 12 cm) and halves the cell,
    // exactly as the gate's own convergence table does.
    let refine = env("HYDRO_REFINE", 1.0) as i32;
    let grid: usize = 48 * refine as usize;
    let dx: f32 = 0.01 / refine as f32;
    let column: IVec2 = IVec2::new(6 * refine, 12 * refine);

    println!(
        "the phase 4 hydrostatic gate scene at {} cm cells, lower half of the column, time-averaged",
        dx * 100.0
    );
    println!("  initial volume         measured      rho g h      error");
    for lattice in [false, true] {
        let config = SimConfig {
            min_dt: 1.0e-7,
            // The wave speed in CELLS per second doubles each time the cell
            // halves, and so does the substep count a frame needs. A fixed
            // cap of 128 made the refined runs hit it, drop simulated time,
            // and never settle: their own rho*g*h read 666 and 615 Pa for a
            // column whose weight is 881, which is what gave it away.
            max_substeps_per_step: 128 * (refine * refine) as usize * 4,
            ..SimConfig::earth(grid, dx, 0.0005)
        };
        let material = NeoHookeanMaterial::from_young_modulus(young, poisson);
        let mass = RHO_KG_M3 * (spacing * dx).powi(2);
        let spawn = SpawnRegion {
            spacing,
            box_size: column,
            box_center: Vec2::new(
                14.0 * refine as f32,
                3.0 * refine as f32 + column.y as f32 * 0.5,
            ),
            material_id: 0,
            mass_override: Some(mass),
            initial_velocity_scale: 0.0,
            ..SpawnRegion::for_sim(&config)
        };
        let mut sim = Simulation::new(config, spawn)
            .with_default_material(Box::new(material))
            .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));
        let n = sim.particles().len();
        if lattice {
            let cell = spacing * spacing;
            let p = sim.particles_mut();
            for i in 0..n {
                let j = p.deformation_gradient[i].determinant();
                p.initial_volume[i] = cell;
                p.volume[i] = cell * j;
                p.density[i] = p.mass[i] / (cell * j);
            }
        }
        let v0 = {
            let p = sim.particles();
            (0..n).map(|i| f64::from(p.initial_volume[i])).sum::<f64>() / n as f64
        };
        let g = sim.config().gravity.length();
        for _ in 0..2000 {
            sim.step();
        }
        let (mut measured, mut expected, mut samples) = (0.0f64, 0.0f64, 0u32);
        for _ in 0..2000 {
            sim.step();
            let p = sim.particles();
            let top = (0..n).map(|i| p.x[i].y).fold(f32::MIN, f32::max);
            let (mut m, mut e, mut k) = (0.0f64, 0.0f64, 0u32);
            for i in 0..n {
                let depth = top - p.x[i].y;
                if depth < column.y as f32 * 0.5 {
                    continue;
                }
                let j = p.deformation_gradient[i].determinant().max(1.0e-6);
                m += f64::from(material.kirchhoff_stress(p, i).y_axis.y / j);
                e += f64::from(-RHO_KG_M3 * g * dx * depth * dx);
                k += 1;
            }
            if k > 0 {
                measured += m / f64::from(k);
                expected += e / f64::from(k);
                samples += 1;
            }
        }
        let (measured, expected) = (measured / f64::from(samples), expected / f64::from(samples));
        println!(
            "  {:<18} mean {v0:.4}   {measured:>8.1} Pa  {expected:>8.1} Pa   {:>+6.1} %",
            if lattice {
                "lattice spacing^2"
            } else {
                "as spawned"
            },
            100.0 * (measured - expected) / expected.abs()
        );
    }
}

/// What the hydrostatic column looks like when it "loses its equilibrium"
/// at a finer grid, read from text pictures instead of guessed: the column
/// at 0.5 cm cells for both initial volumes, its top height and mean volume
/// ratio every 400 frames, and a picture of where the material is.
///
/// Found: the lattice-volume column topples. Its width grows from 12 to 19
/// cells by 0.8 s, it leans by 1 s and lies as a slab 24 cells wide and 7
/// tall by 2 s. The estimated-volume column rocks (width 11.6 to 14) but
/// stands. Neither is damped: the fastest particle still moves at about
/// 0.5 m/s after 2 s. What seeds the asymmetric mode is not established.
#[test]
#[ignore = "diagnostic probe kept for reruns, not part of the CI suite"]
fn the_refined_column_seen() {
    use emerge::NeoHookeanMaterial;
    use emerge::diagnostics::{OCCUPANCY_BANDS, scene_map};
    let spacing: f32 = 0.5;
    let refine = env("HYDRO_REFINE", 2.0) as i32;
    let grid: usize = 48 * refine as usize;
    let dx: f32 = 0.01 / refine as f32;
    let column: IVec2 = IVec2::new(6 * refine, 12 * refine);
    for lattice in [false, true] {
        let config = SimConfig {
            min_dt: 1.0e-7,
            max_substeps_per_step: 128 * (refine * refine) as usize * 4,
            ..SimConfig::earth(grid, dx, 0.0005)
        };
        let spawn = SpawnRegion {
            spacing,
            box_size: column,
            box_center: Vec2::new(
                14.0 * refine as f32,
                3.0 * refine as f32 + column.y as f32 * 0.5,
            ),
            material_id: 0,
            mass_override: Some(RHO_KG_M3 * (spacing * dx).powi(2)),
            initial_velocity_scale: 0.0,
            ..SpawnRegion::for_sim(&config)
        };
        let mut sim = Simulation::new(config, spawn)
            .with_default_material(Box::new(NeoHookeanMaterial::from_young_modulus(2.0e5, 0.3)))
            .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));
        let n = sim.particles().len();
        if lattice {
            let cell = spacing * spacing;
            let p = sim.particles_mut();
            for i in 0..n {
                p.initial_volume[i] = cell;
                p.volume[i] = cell;
                p.density[i] = p.mass[i] / cell;
            }
        }
        println!(
            "--- {} initial volume, refine {refine}",
            if lattice { "lattice" } else { "estimated" }
        );
        for frame in 0..=4000 {
            if frame % 400 == 0 {
                let p = sim.particles();
                let top = p.x.iter().map(|x| x.y).fold(f32::MIN, f32::max);
                let lo = p.x.iter().map(|x| x.x).fold(f32::MAX, f32::min);
                let hi = p.x.iter().map(|x| x.x).fold(f32::MIN, f32::max);
                let j = (0..n)
                    .map(|i| f64::from(p.deformation_gradient[i].determinant()))
                    .sum::<f64>()
                    / n as f64;
                let vmax = p.v.iter().map(|v| v.length()).fold(0.0f32, f32::max);
                println!(
                    "t={:.2}s top {:.1} cells above spawn floor, width {:.1}, mean J {j:.4}, fastest {:.3} m/s",
                    frame as f32 * 0.0005,
                    top - 3.0 * refine as f32 + column.y as f32 * 0.5 - column.y as f32 * 0.5,
                    hi - lo,
                    vmax * dx
                );
                if frame % 2000 == 0 {
                    let region = (
                        Vec2::new(4.0 * refine as f32, 0.0),
                        Vec2::new(24.0 * refine as f32, 20.0 * refine as f32),
                    );
                    for row in scene_map(p, region, 40, 20, |_| 1.0, &OCCUPANCY_BANDS) {
                        if row.trim().is_empty() {
                            continue;
                        }
                        println!("      |{row}|");
                    }
                }
            }
            sim.step();
        }
    }
}

/// The column loses its equilibrium by toppling: undamped, it rings from
/// its stress-free spawn, and on a frictionless floor the ringing can rock
/// it over (`the_refined_column_seen`). A wide slab cannot topple, so it
/// gives the convergence table issue #41 asks for. A slab 40 cells wide and
/// 12 tall at 1 cm (scaled with the refinement), on the same slip floor,
/// the same material and mass as the column; the vertical stress over its
/// central third, lower half, time-averaged over the second second,
/// against `rho g d` with `d` the depth under the slab's own top there. The
/// top's drift over that second is printed, the check that it is at rest.
///
/// Found (positive: the lower half carries more than the weight above it):
///
/// ```text
///   cells    as spawned   lattice    top drift over the second second
///   1 cm       +4.2 %     +33.1 %    -0.005 / -0.041 cells
///   0.5 cm     +6.8 %     +27.0 %    +0.022 / -0.589 cells
///   0.25 cm    +8.8 %     +13.6 %    -1.625 / -0.428 cells
/// ```
///
/// At 1 cm the estimated volume is eight times closer. Refined, the lattice
/// volume closes in and the estimated one drifts slowly away; neither
/// converges cleanly, and at 0.25 cm the estimated-volume slab is still
/// sinking by 1.6 cells over the measured second, so that row is not at
/// rest. Why it sinks is not established.
#[test]
#[ignore = "diagnostic probe kept for reruns, not part of the CI suite"]
fn the_hydrostatic_slab_for_both_initial_volumes() {
    use emerge::{MaterialModel, NeoHookeanMaterial};
    let spacing: f32 = 0.5;
    println!("  cells    initial volume       measured      rho g d      error     top drift");
    for refine in [1i32, 2, 4] {
        let grid: usize = 48 * refine as usize;
        let dx: f32 = 0.01 / refine as f32;
        let slab = IVec2::new(40 * refine, 12 * refine);
        for lattice in [false, true] {
            let config = SimConfig {
                min_dt: 1.0e-7,
                max_substeps_per_step: 128 * (refine * refine) as usize * 4,
                ..SimConfig::earth(grid, dx, 0.0005)
            };
            let material = NeoHookeanMaterial::from_young_modulus(2.0e5, 0.3);
            let spawn = SpawnRegion {
                spacing,
                box_size: slab,
                box_center: Vec2::new(grid as f32 * 0.5, 3.0 * refine as f32 + slab.y as f32 * 0.5),
                material_id: 0,
                mass_override: Some(RHO_KG_M3 * (spacing * dx).powi(2)),
                initial_velocity_scale: 0.0,
                ..SpawnRegion::for_sim(&config)
            };
            let mut sim = Simulation::new(config, spawn)
                .with_default_material(Box::new(material))
                .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));
            let n = sim.particles().len();
            if lattice {
                let cell = spacing * spacing;
                let p = sim.particles_mut();
                for i in 0..n {
                    p.initial_volume[i] = cell;
                    p.volume[i] = cell;
                    p.density[i] = p.mass[i] / cell;
                }
            }
            let g = sim.config().gravity.length();
            let centre = grid as f32 * 0.5;
            let third = slab.x as f32 / 6.0;
            let middle_top = |sim: &Simulation| {
                let p = sim.particles();
                p.x.iter()
                    .filter(|x| (x.x - centre).abs() < third)
                    .map(|x| x.y)
                    .fold(f32::MIN, f32::max)
            };
            for _ in 0..2000 {
                sim.step();
            }
            let top_start = middle_top(&sim);
            let (mut measured, mut expected, mut samples) = (0.0f64, 0.0f64, 0u32);
            for _ in 0..2000 {
                sim.step();
                let p = sim.particles();
                let top = middle_top(&sim);
                let (mut m, mut e, mut k) = (0.0f64, 0.0f64, 0u32);
                for i in 0..n {
                    if (p.x[i].x - centre).abs() >= third {
                        continue;
                    }
                    let depth = top - p.x[i].y;
                    if depth < slab.y as f32 * 0.5 {
                        continue;
                    }
                    let j = p.deformation_gradient[i].determinant().max(1.0e-6);
                    m += f64::from(material.kirchhoff_stress(p, i).y_axis.y / j);
                    e += f64::from(-RHO_KG_M3 * g * dx * depth * dx);
                    k += 1;
                }
                if k > 0 {
                    measured += m / f64::from(k);
                    expected += e / f64::from(k);
                    samples += 1;
                }
            }
            let drift = middle_top(&sim) - top_start;
            let (measured, expected) =
                (measured / f64::from(samples), expected / f64::from(samples));
            println!(
                "  {:>4} cm  {:<18}  {measured:>9.1} Pa  {expected:>9.1} Pa  {:>+7.1} %   {:+.3} cells",
                dx * 100.0,
                if lattice {
                    "lattice spacing^2"
                } else {
                    "as spawned"
                },
                100.0 * (expected - measured) / expected.abs(),
                drift
            );
        }
    }
}

/// Issue #41's reconciliation: the core audit ranked the two initial
/// volumes the other way (sag of a free column, `self_weight_strain_is_
/// spacing_independent`: about 0.82 of the analytic for the estimated
/// volume, 1.03 for the lattice's), while the slab above ranks them by
/// stress (+4 percent against +33). Two different quantities on two
/// different scenes. This measures BOTH on BOTH, for both volumes, in
/// grid units throughout: the sag as that test reads it (height from the
/// particle centres, time-averaged, over its analytic mean strain), and the
/// lower half's vertical stress over the weight above it, `rho g d` with
/// `rho` the lattice density `mass / spacing^2`, which does not depend on
/// either volume.
///
/// Found, 1 cm cells, spacing 0.5:
///
/// ```text
///   scene               volume      sag / analytic   stress / weight
///   audit column 6x10   estimated       0.913            0.905
///   audit column 6x10   lattice         1.075            1.118
///   wide slab 40x12     estimated       1.053            1.044
///   wide slab 40x12     lattice         1.133            1.332
/// ```
///
/// Within each scene the two quantities rank the volumes the same way; no
/// quantity inverts the order. What changes sign is the estimated volume's
/// error between scenes, 9 percent under on the narrow free column and 4 to
/// 5 over on the slab, while the lattice volume is over on both, 8 to 33
/// percent. The audit's own 0.82 against 1.03 is not reproduced today
/// (0.913 against 1.075 here); it predates the spawn contract fix that
/// changed the estimated volume, which is the likely reason, not checked.
/// The lattice volume is not the more accurate one on either scene.
#[test]
#[ignore = "diagnostic probe kept for reruns, not part of the CI suite"]
fn the_audit_sag_and_the_slab_stress_on_both_scenes() {
    use emerge::{MaterialModel, NeoHookeanMaterial};
    // (label, box, centre, E, nu, boundary cells, frame dt, uniaxial analytic)
    let scenes = [
        (
            "audit column 6x10",
            IVec2::new(6, 10),
            Vec2::new(32.0, 8.0),
            1.0e5f32,
            0.2f32,
            3usize,
            0.005f32,
            true,
        ),
        (
            "wide slab 40x12",
            IVec2::new(40, 12),
            Vec2::new(32.0, 9.0),
            2.0e5,
            0.3,
            2,
            0.0005,
            false,
        ),
    ];
    let spacing = 0.5f32;
    let rho = 1000.0f32;
    println!(
        "  scene               initial volume    sag / analytic    lower-half stress / weight"
    );
    for (label, size, centre, young, poisson, boundary, dt, uniaxial) in scenes {
        for lattice in [false, true] {
            let config = SimConfig {
                boundary_thickness: boundary,
                min_dt: 1.0e-7,
                max_substeps_per_step: 2000,
                ..SimConfig::earth(64, 0.01, dt)
            };
            let spawn = SpawnRegion {
                spacing,
                box_size: size,
                box_center: centre,
                material_id: 0,
                initial_velocity_scale: 0.0,
                ..SpawnRegion::for_sim(&config)
            };
            let (lambda, mu) = config.lame_from_si_physical_cfg(young, poisson, rho);
            let material = NeoHookeanMaterial::new(lambda, mu);
            let mut sim = Simulation::new(config, spawn)
                .with_default_material(Box::new(material))
                .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));
            let n = sim.particles().len();
            if lattice {
                let cell = spacing * spacing;
                let p = sim.particles_mut();
                for i in 0..n {
                    p.initial_volume[i] = cell;
                    p.volume[i] = cell;
                    p.density[i] = p.mass[i] / cell;
                }
            }
            let g = sim.config().gravity.length();
            let rho_grid = sim.particles().mass[0] / (spacing * spacing);
            // The central third across, so the slab's ends do not enter.
            let third = size.x as f32 / 6.0;
            let middle = |x: Vec2| (x.x - centre.x).abs() < third;
            let height = |s: &Simulation| {
                let p = s.particles();
                let top =
                    p.x.iter()
                        .filter(|x| middle(**x))
                        .map(|x| x.y)
                        .fold(f32::MIN, f32::max);
                let bottom =
                    p.x.iter()
                        .filter(|x| middle(**x))
                        .map(|x| x.y)
                        .fold(f32::MAX, f32::min);
                (top, top - bottom)
            };
            let (_, h0) = height(&sim);
            // The audit's analytic: mean strain of a free column, rho g h / 2E.
            // A wide slab is held laterally by itself, so its modulus is the
            // constrained one, lambda + 2 mu.
            let modulus = if uniaxial {
                mu * (3.0 * lambda + 2.0 * mu) / (lambda + mu)
            } else {
                lambda + 2.0 * mu
            };
            let analytic = rho_grid * g * h0 / (2.0 * modulus);
            let settle = (2.0 / dt).round() as usize;
            for _ in 0..settle {
                sim.step();
            }
            let (mut sag, mut stress, mut samples) = (0.0f64, 0.0f64, 0u32);
            for _ in 0..settle {
                sim.step();
                let (top, h) = height(&sim);
                sag += f64::from((h0 - h) / h0);
                let p = sim.particles();
                let (mut m, mut e) = (0.0f64, 0.0f64);
                for i in 0..n {
                    let depth = top - p.x[i].y;
                    if !middle(p.x[i]) || depth < size.y as f32 * 0.5 {
                        continue;
                    }
                    let j = p.deformation_gradient[i].determinant().max(1.0e-6);
                    m += f64::from(-material.kirchhoff_stress(p, i).y_axis.y / j);
                    e += f64::from(rho_grid * g * depth);
                }
                if e > 0.0 {
                    stress += m / e;
                }
                samples += 1;
            }
            println!(
                "  {label:<18}  {:<16}    {:>6.3}            {:>6.3}",
                if lattice { "lattice" } else { "estimated" },
                sag / f64::from(samples) / f64::from(analytic),
                stress / f64::from(samples)
            );
        }
    }
}

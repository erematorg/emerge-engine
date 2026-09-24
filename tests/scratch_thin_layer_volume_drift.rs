//! Why does a spread-out body keep gaining volume?
//!
//! Found by watching a demo: particles grow and drift apart the longer a
//! scene runs, and since the renderer draws each particle deformed by its
//! own F, what is visible IS the volume ratio. Measured on the slump
//! scene, the two columns that hold a shape are flat to the fourth digit
//! over twenty seconds while the one that spreads into a thin wide layer
//! gains about 0.08 % of its volume a second and accelerates. Its own
//! weight says it should sit at J = 0.998, so the sign is wrong and the
//! size is ten times too large: numerical, not a state.
//!
//! This sweeps the one thing that differs. Same material, same width,
//! same mass per particle, same gravity: only the layer's THICKNESS
//! changes. The code names a free-surface mechanism in `transfer/g2p.rs`
//! where a layer thinner than the kernel's own support reads as expanded
//! in every depth band, so if that is what this is, the drift has to grow
//! as the layer thins. If it does not, it is something else.
//!
//! Knobs: DRIFT_SECONDS, DRIFT_GRAVITY (fraction of g, to separate a
//! load-driven effect from one that happens at rest), DRIFT_MATERIAL
//! (`neohookean` for an ordinary solid instead of the yield-stress fluid,
//! to see whether it is one law or the transfer).
//!
//! # What it found
//!
//! The thickness sweep does not behave like a thin-layer effect: the drift
//! does not grow as the layer thins. Splitting each slab into the particles
//! within one cell of its own outline and the rest does separate it, over
//! ten seconds at 2 ms a frame:
//!
//! ```text
//!   thickness    skin J     interior J    ratio
//!     1 cell     0.99986       --         flat, nothing to split
//!     2 cells    1.00389     1.00021      18
//!     4 cells    1.00236     1.00078       3
//!     8 cells    1.00292     1.00051       6
//!    16 cells    1.01527     1.00228       7
//! ```
//!
//! So the volume is gained at the body's outline, three to eighteen times
//! faster than inside it. The obvious suspect is the gather dropping nodes
//! there: `Grid::is_extrapolated` excludes a node that received no scatter,
//! and dropping nodes breaks the kernel's zero-first-moment identity, which
//! is exactly what makes the affine gather blind to a rigid translation.
//!
//! COUNTED, and it is not that. Instrumenting the branch over this sweep:
//! 0 extrapolated nodes of 418,714,560 gathered. The path never fires,
//! because P2G inserts every in-bounds node of a particle's own stencil, so
//! a particle always gathers from a complete one. The exclusion machinery
//! (`included_gx`/`included_gy` and the column discard beneath it) is
//! therefore inert in the current tree, which is worth its own issue but is
//! not this. The invariant it was protecting is kept as a real test,
//! `a_rigid_translation_reads_no_velocity_gradient`, which reads 4.4e-7 on a
//! drifting block.
//!
//! What is left is a one-way pressure ratchet, and that IS measured. A
//! fluid clamps its pressure from below at `pressure_floor`, which
//! `BinghamFluidMaterial::new` left at 0.0. A particle with J > 1 is
//! below rest density, so its Tait pressure is negative and the clamp
//! deleted it: every expanded particle in every slab was clamped (105 of
//! 105, 657 of 657), at a mean asked-for tension of 510 to 830 Pa.
//! Expansion met no restoring force, compression met the full one, so
//! noise ratcheted volume upward.
//!
//! `BinghamProps::cavitation_pressure_pa` is the fix, and
//! `volume_drift_against_cavitation_pressure` below is what sized it.
//! Sweeping the floor at five seconds, with the derived value marked:
//!
//! ```text
//!   floor Pa    2 cells                  16 cells
//!               drift    clamped         drift     clamped   worst |J-1|
//!         0    +0.0444   105/105        +0.0737   657/657      0.857
//!      -140    +0.0041     5/147        -0.0127   124/564      0.047
//!      -280    +0.0012     2/148        -0.0157    27/559      0.039   <- derived
//!      -560    -0.0006     0/149        -0.0165     3/567      0.029
//!     -1120    -0.0006     0/146        -0.0167     0/576      0.014
//!     -2800    -0.0006     0/146        -0.0167     0/573      0.014
//!    -10000    -0.0006     0/146        PANICKED, the timestep could not advance
//! ```
//!
//! The derived floor is where the ratchet stops being the dominant
//! effect, and a factor of ten short of where stability goes. It comes out
//! of `2*gamma/R` at a measured surface tension and a class boundary taken
//! from a petrographic manual, which is a declared modelling choice and not
//! a measured bubble.
//!
//! It is NOT inside the 510 to 830 Pa the fluid was asking for. Only -560
//! is. At -280 the fluid asks for up to 830 Pa of tension and gets 280 back,
//! and that is exactly why 60 of 663 expanded particles are still clamped at
//! sixteen cells, 9 percent, against 0.5 percent at -560. -560 measures
//! better on every column of the table above. It is not what the cited bound
//! gives, so it is not what ships, and this is written down so the choice
//! stays visible.
//!
//! At twenty seconds and 2 ms a frame the slabs then read 0.99986, 0.99981,
//! 0.99988, 0.99957 and 0.99875 at one, two, four, eight and sixteen cells.
//! Not flat to the fourth decimal, which is the letter of the original
//! criterion and is not met: the four-cell slab moves from 0.99924 to
//! 0.99988 over that window. What IS true is that every slab ends within
//! 1.3e-3 of one and all of them BELOW one, and that the same sweep window
//! puts the drift at +0.0012 against +0.0444 percent a second at two cells
//! and -0.0157 against +0.0737 at sixteen: the upward ratchet is gone and
//! the sign at sixteen cells has inverted to compaction. Worst `|J - 1|` on
//! the thickest: 0.79 to 0.86 at every window measured with no floor, 0.038
//! with one. The skin/interior split inverts too, 0.99955 against 0.99867 at
//! sixteen cells, the interior now the more compacted, which is what bearing
//! more load looks like.
//!
//!   cargo test --profile quick --all-features --test scratch_thin_layer_volume_drift -- --ignored --nocapture
extern crate emerge_engine as emerge;

use emerge::{
    BinghamFluidMaterial, BinghamProps, DruckerPragerMaterial, FromSI, MaterialModel,
    NeoHookeanMaterial, NewtonianFluidMaterial, SimConfig, Simulation, SlipBoundary, SpawnRegion,
};
use glam::{IVec2, Vec2};

const GRID: usize = 64;
/// 2 mm cells, the slump scene's own scale, so the numbers are comparable.
const DX_M: f32 = 0.002;
const WIDTH_CELLS: i32 = 40;
const FLOOR_CELLS: f32 = 2.0;
const RHO_KG_M3: f32 = 1000.0;
/// The soft column's own yield stress, the one that drifts.
const YIELD_PA: f32 = 2.0;
/// The slump scene's own bulk modulus, so the floor can be read as a
/// fraction of the stiffness it is clamping.
const BULK_PA: f32 = 78_480.0;
const ETA_PA_S: f32 = 0.5;
const YIELD_STRAIN: f32 = 0.05;
const YOUNG_PA: f32 = 2.0e5;
const POISSON: f32 = 0.3;

fn env(name: &str, default: f32) -> f32 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Mean and worst volume ratio over the whole slab.
fn volume_state(sim: &Simulation) -> (f64, f32) {
    let p = sim.particles();
    let mut sum = 0.0f64;
    let mut worst = 0.0f32;
    for i in 0..p.len() {
        let j = p.deformation_gradient[i].determinant();
        sum += f64::from(j);
        worst = worst.max((j - 1.0).abs());
    }
    (sum / p.len() as f64, worst)
}

/// Mean volume ratio split by how close a particle sits to the body's own
/// outline, within one cell of the left, right or top extent against the
/// rest. The quadratic kernel reaches one and a half cells, so a particle
/// nearer the outline than that gathers from nodes that carry no material
/// on one side. If the residual drift is that, it lives on the skin and
/// the interior stays flat; if it is spread evenly, it is not.
fn skin_and_interior(sim: &Simulation) -> (f64, usize, f64, usize) {
    let p = sim.particles();
    let (mut lo_x, mut hi_x, mut hi_y) = (f32::MAX, f32::MIN, f32::MIN);
    for i in 0..p.len() {
        lo_x = lo_x.min(p.x[i].x);
        hi_x = hi_x.max(p.x[i].x);
        hi_y = hi_y.max(p.x[i].y);
    }
    let (mut skin, mut n_skin, mut core, mut n_core) = (0.0f64, 0usize, 0.0f64, 0usize);
    for i in 0..p.len() {
        let x = p.x[i];
        let j = f64::from(p.deformation_gradient[i].determinant());
        if x.x - lo_x < 1.0 || hi_x - x.x < 1.0 || hi_y - x.y < 1.0 {
            skin += j;
            n_skin += 1;
        } else {
            core += j;
            n_core += 1;
        }
    }
    (
        skin / n_skin.max(1) as f64,
        n_skin,
        core / n_core.max(1) as f64,
        n_core,
    )
}

#[test]
#[ignore = "diagnostic probe kept for reruns, not part of the CI suite"]
fn volume_drift_against_layer_thickness() {
    let seconds = env("DRIFT_SECONDS", 10.0);
    let gravity_fraction = env("DRIFT_GRAVITY", 1.0);
    let which = std::env::var("DRIFT_MATERIAL").unwrap_or_default();
    let neohookean = which == "neohookean";
    let sand = which == "sand";
    // shear_modulus = 0 selects Bingham's purely VISCOUS branch, which
    // carries J by the continuity equation directly with no SVD round
    // trip; anything else selects the elastoviscoplastic one that goes
    // through a decomposition and a reconstruction every substep.
    let viscous = which == "viscous";
    // The control the elastic slab cannot be: a body that FLOWS and
    // collapses like the yield-stress one, under the same stiffness and
    // the same viscosity, but whose law never decomposes and rebuilds F.
    // Comparing a standing elastic slab against a collapsing plastic one
    // changes the law and the motion together; this changes only the law.
    let newtonian = which == "newtonian";
    // Small enough that the adaptive loop takes ONE substep a frame, so
    // the velocity gradient read between frames IS the one the material
    // integrated. Volume can only move through its trace, so the running
    // sum of `dt * tr(C)` must equal `ln J` exactly. Any gap is volume
    // that came from somewhere other than the flow.
    let dt = env("DRIFT_DT", 0.002);

    println!(
        "{} slab, gravity x{gravity_fraction}, {seconds} s, 2 mm cells",
        if neohookean {
            "elastic (NeoHookean, no SVD round trip)"
        } else if sand {
            "sand (Drucker-Prager, SVD round trip)"
        } else if newtonian {
            "newtonian fluid (flows, no SVD)"
        } else if viscous {
            "yield-stress fluid (2 Pa, viscous branch, no SVD)"
        } else {
            "yield-stress fluid (2 Pa, SVD round trip)"
        }
    );
    println!(
        "  thickness   particles    mean J at 0 s   at end    drift per second   per substep   worst |J-1|"
    );

    for thickness in [1i32, 2, 4, 8, 16] {
        let mut config = SimConfig {
            min_dt: 1.0e-6,
            // Off, so a particle keeps its index: the solver rotates the
            // array when one falls asleep, and the per-particle identity
            // check below would then be comparing two different particles.
            sleep_threshold: 0.0,
            max_substeps_per_step: 256,
            ..SimConfig::earth(GRID, DX_M, dt)
        };
        config.gravity *= gravity_fraction;

        // The floor is an SI entry now, so the sweep moves the physical
        // number and lets `from_physical` convert it, instead of poking a
        // grid-unit field behind the constructor's back.
        let floor_pa = env(
            "DRIFT_FLOOR_PA",
            BinghamProps::air_entrained_cavitation_pressure(),
        );
        let props = BinghamProps {
            rho_kg_m3: RHO_KG_M3,
            eta_pa_s: ETA_PA_S,
            bulk_modulus_pa: BULK_PA,
            yield_stress_pa: YIELD_PA,
            shear_modulus_pa: if viscous {
                0.0
            } else {
                YIELD_PA / YIELD_STRAIN
            },
            cavitation_pressure_pa: floor_pa,
        };
        let material: Box<dyn MaterialModel> = if neohookean {
            Box::new(NeoHookeanMaterial::from_young_modulus(YOUNG_PA, POISSON))
        } else if sand {
            Box::new(DruckerPragerMaterial::cohesionless(2000.0, 4000.0))
        } else if newtonian {
            // Built from the yield-stress material's OWN grid-unit fields,
            // so "same stiffness, same viscosity" holds by construction
            // rather than by redoing its unit conversion by hand.
            let bingham = BinghamFluidMaterial::from_physical(&props, &config);
            Box::new(NewtonianFluidMaterial::new(
                bingham.rest_density,
                bingham.dynamic_viscosity,
                bingham.eos_stiffness,
                bingham.eos_power,
            ))
        } else {
            Box::new(BinghamFluidMaterial::from_physical(&props, &config))
        };
        // The same EOS the law uses, read back from the law's own fields so
        // this reports what the solver computed, not a second formula.
        let eos = BinghamFluidMaterial::from_physical(&props, &config);
        let spawn = SpawnRegion {
            spacing: 0.5,
            box_size: IVec2::new(WIDTH_CELLS, thickness),
            box_center: Vec2::new(GRID as f32 * 0.5, FLOOR_CELLS + thickness as f32 * 0.5),
            material_id: 0,
            mass_override: Some(RHO_KG_M3 * (0.5 * DX_M).powi(2)),
            initial_velocity_scale: 0.0,
            ..SpawnRegion::for_sim(&config)
        };
        let mut sim = Simulation::new(config, spawn)
            .with_default_material(material)
            .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));
        let particles = sim.particles().len();

        // One step first: the spawn transient is common to every thickness
        // and measuring across it would put the same constant in every row.
        sim.step();
        let (start, _) = volume_state(&sim);
        let frames = (seconds / dt).round() as usize;
        let mut substeps = 0usize;
        let mut traced = 0.0f64;
        let mut ln_j_start = 0.0f64;
        for frame in 0..frames {
            sim.step();
            substeps += sim.diagnostics_snapshot().substeps_last_step;
            let parts = sim.particles();
            let mid = parts.len() / 2;
            if frame == 0 {
                ln_j_start = f64::from(parts.deformation_gradient[mid].determinant()).ln();
            } else {
                // Only comparable to `ln J` when the adaptive loop takes ONE
                // substep a frame: this samples the gradient once a frame,
                // while the material integrates it once a SUBSTEP. Raise
                // DRIFT_DT until `substeps_last_step` reads 1 before reading
                // the gap as anything.
                let c = parts.velocity_gradient[mid];
                traced += f64::from(dt) * f64::from(c.x_axis.x + c.y_axis.y);
            }
        }
        let ln_j_end = f64::from(
            sim.particles().deformation_gradient[sim.particles().len() / 2].determinant(),
        )
        .ln();
        let (end, worst) = volume_state(&sim);
        println!(
            "  {thickness:>6} cells  {particles:>8}    {start:>11.5}   {end:>7.5}   {:>14.5} %   {:>9.2e}   {worst:>9.5}",
            100.0 * (end - start) / f64::from(seconds),
            (end - start) / substeps.max(1) as f64
        );
        println!(
            "            one particle: ln J moved {:+.3e}, its own trace(C) accounts for {traced:+.3e}, gap {:+.3e}",
            ln_j_end - ln_j_start,
            (ln_j_end - ln_j_start) - traced
        );
        let (skin, n_skin, core, n_core) = skin_and_interior(&sim);
        println!(
            "            skin {skin:.5} over {n_skin} particles, interior {core:.5} over {n_core}"
        );
        let parts = sim.particles();
        let (mut expanded, mut clamped, mut raw_sum) = (0usize, 0usize, 0.0f64);
        for i in 0..parts.len() {
            if parts.deformation_gradient[i].determinant() <= 1.0 {
                continue;
            }
            expanded += 1;
            let density = parts.density[i]
                .max(eos.min_density)
                .min(eos.rest_density * 2.0);
            let raw = eos.eos_stiffness * ((density / eos.rest_density).powf(eos.eos_power) - 1.0);
            raw_sum += f64::from(raw);
            if raw < eos.pressure_floor {
                clamped += 1;
            }
        }
        println!(
            "            floor {floor_pa:.0} Pa ({:.3e} grid): {clamped} of {expanded} expanded particles clamped, mean raw {:.3e} grid",
            eos.pressure_floor,
            raw_sum / expanded.max(1) as f64
        );
    }
}

/// One slab run, reduced to what the cavitation sweep reads off it.
struct SlabRun {
    drift_percent_per_second: f64,
    worst_j_minus_one: f32,
    /// Particles above rest volume whose asked-for tension the floor cut.
    clamped: usize,
    expanded: usize,
    /// Mean tension those expanded particles asked for, back in Pa, so it
    /// can be compared against the floor that is clamping them.
    mean_asked_pa: f64,
}

/// The yield-stress slab of the sweep above, at one thickness and one
/// cavitation pressure. Panics out if the solver cannot advance, which is
/// itself one of the results the sweep is after.
fn run_slab(thickness: i32, floor_pa: f32, seconds: f32, dt: f32) -> SlabRun {
    let config = SimConfig {
        min_dt: 1.0e-6,
        sleep_threshold: 0.0,
        max_substeps_per_step: 256,
        ..SimConfig::earth(GRID, DX_M, dt)
    };
    let props = BinghamProps {
        rho_kg_m3: RHO_KG_M3,
        eta_pa_s: ETA_PA_S,
        bulk_modulus_pa: BULK_PA,
        yield_stress_pa: YIELD_PA,
        shear_modulus_pa: YIELD_PA / YIELD_STRAIN,
        cavitation_pressure_pa: floor_pa,
    };
    let material = BinghamFluidMaterial::from_physical(&props, &config);
    // The engine's own SI-to-grid factor, read back out of the engine
    // rather than recomputed here, so this reports what the solver used.
    let pa_per_grid = {
        let mut gauge = props;
        gauge.cavitation_pressure_pa = -1000.0;
        -1000.0 / BinghamFluidMaterial::from_physical(&gauge, &config).pressure_floor
    };
    let eos = BinghamFluidMaterial::from_physical(&props, &config);
    let spawn = SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new(WIDTH_CELLS, thickness),
        box_center: Vec2::new(GRID as f32 * 0.5, FLOOR_CELLS + thickness as f32 * 0.5),
        material_id: 0,
        mass_override: Some(RHO_KG_M3 * (0.5 * DX_M).powi(2)),
        initial_velocity_scale: 0.0,
        ..SpawnRegion::for_sim(&config)
    };
    let mut sim = Simulation::new(config, spawn)
        .with_default_material(Box::new(material))
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));

    sim.step();
    let (start, _) = volume_state(&sim);
    for _ in 0..(seconds / dt).round() as usize {
        sim.step();
    }
    let (end, worst_j_minus_one) = volume_state(&sim);

    let parts = sim.particles();
    let (mut expanded, mut clamped, mut asked) = (0usize, 0usize, 0.0f64);
    for i in 0..parts.len() {
        if parts.deformation_gradient[i].determinant() <= 1.0 {
            continue;
        }
        expanded += 1;
        let density = parts.density[i]
            .max(eos.min_density)
            .min(eos.rest_density * 2.0);
        let raw = eos.eos_stiffness * ((density / eos.rest_density).powf(eos.eos_power) - 1.0);
        asked += f64::from(raw) * f64::from(pa_per_grid);
        if raw < eos.pressure_floor {
            clamped += 1;
        }
    }
    SlabRun {
        drift_percent_per_second: 100.0 * (end - start) / f64::from(seconds),
        worst_j_minus_one,
        clamped,
        expanded,
        mean_asked_pa: asked / expanded.max(1) as f64,
    }
}

/// At which cavitation pressure does the ratchet stop, and at which does
/// the thickest slab stop panicking?
///
/// `BinghamProps::cavitation_pressure_pa` is a measured coefficient, not a
/// tuning knob, so this does not pick a value. It reports what each value
/// does, so the one the constants derive can be read against its
/// neighbours, against the tension the fluid is actually asking for, and
/// against the bulk modulus it is a fraction of.
///
/// Two slabs: two cells, the thinnest that drifts, and sixteen, the one
/// that panicked outright when the clamp was lifted entirely.
///
///   cargo test --profile quick --all-features --test scratch_thin_layer_volume_drift -- --ignored floor --nocapture
#[test]
#[ignore = "diagnostic probe kept for reruns, not part of the CI suite"]
fn volume_drift_against_cavitation_pressure() {
    let seconds = env("DRIFT_SECONDS", 5.0);
    let dt = env("DRIFT_DT", 0.002);
    let derived = BinghamProps::air_entrained_cavitation_pressure();
    println!(
        "bulk modulus {BULK_PA:.0} Pa, derived cavitation pressure {derived:.0} Pa, {:.4} of it",
        derived.abs() / BULK_PA
    );
    println!(
        "  floor Pa    thickness   drift per second   worst |J-1|   clamped/expanded   mean asked Pa"
    );

    for floor_pa in [
        0.0f32, -56.0, -140.0, -280.0, -560.0, -1120.0, -2800.0, -10_000.0,
    ] {
        for thickness in [2i32, 16] {
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                run_slab(thickness, floor_pa, seconds, dt)
            }));
            match outcome {
                Ok(r) => println!(
                    "  {floor_pa:>8.0}   {thickness:>6} cells   {:>14.5} %   {:>9.5}   {:>10}   {:>13.0}",
                    r.drift_percent_per_second,
                    r.worst_j_minus_one,
                    format!("{}/{}", r.clamped, r.expanded),
                    r.mean_asked_pa
                ),
                Err(_) => println!(
                    "  {floor_pa:>8.0}   {thickness:>6} cells   PANICKED, the timestep could not advance"
                ),
            }
        }
    }
}

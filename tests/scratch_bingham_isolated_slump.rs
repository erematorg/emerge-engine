//! How far does each slump column spread when nothing else is in the way,
//! and what does a floor that grips change?
//!
//! The slump demo stands three columns close enough that their deposits
//! meet (`tests/scratch_bingham_column_symmetry.rs`), so its deposits are
//! not three independent slumps. Spacing them properly needs to know how
//! wide each one gets alone, and that depends on the floor. The demo's floor
//! is frictionless (`SlipBoundary`), and a yield-stress fluid on a
//! frictionless floor is not the slump test: the measurements the demo's
//! inversion comes from are made on a substrate the material does not slip
//! on. Woodbridge, Fonte and Juel (arXiv 2609.12229, 2026), modelling
//! yield-stress drop spreading, impose "no-slip and no-penetration
//! conditions" at the substrate, and measure the same fluids' rheology
//! between "cross-hatched parallel plates to minimise wall slip", because
//! such fluids slip on a smooth wall.
//!
//! `FrictionBoundary` is Coulomb friction, which acts as no slip only where
//! it can hold the shear the base has to carry: `mu * rho * g * h` against
//! roughly `tau_0`. At `mu = 1` that holds with a wide margin for the two
//! columns that spread (216 Pa against 60 under a 22 mm deposit, far more
//! against 2), and not for the 1200 Pa one, which does not spread and so
//! loads its base with almost nothing. A declared approximation (2).
//!
//! Each column alone, spawned on a mirror line of the grid, in a tank twice
//! the demo's width so the softest one does not reach the walls; if it does,
//! the row says so, because a wall-stopped spread is not a free one.
//!
//! # What it found
//!
//! The friction floor runs with this material: the six runs below are the
//! execution that confirms a non-strict fluid accepts it.
//!
//! ```text
//!   tau_0   floor       h mm   L mm   h/L    read back   base nodes sliding
//!     2     slip         2.5  123.9   0.02     0.2 Pa        97.6 %  (reached a wall)
//!     2     friction 1   5.7   93.1   0.06     1.7 Pa         8.7 %
//!    60     slip        10.3   46.6   0.22    11.2 Pa       100.0 %
//!    60     friction 1  20.8   26.3   0.79    80.5 Pa         6.6 %
//!  1200     slip        38.9    9.5   4.08   ~778 Pa         97.6 %
//!  1200     friction 1  38.9    9.5   4.08   ~778 Pa          1.0 %
//! ```
//!
//! On the frictionless floor the thin-layer inversion reads five to ten
//! times low: without basal shear the material spreads too far for this to
//! be a slump test.
//!
//! On the gripping floor there is ONE valid point, and it is still
//! evolving. Only the 2 Pa deposit is in the regime the inversion needs
//! (h/L = 0.06); it reads 1.7 Pa at three seconds and 1.6 at five and eight,
//! still creeping at 0.5 mm/s, so its reading keeps falling as it spreads.
//! The 60 Pa deposit is out of that regime (h/L = 0.79), so its 80.5 Pa is
//! not a reading of `tau_0` by this formula. The 1200 Pa column does not
//! slump at all, and its ~778 Pa is not a reading of anything.
//!
//! `mu = 1` is not perfect adhesion. Coulomb friction holds a node only
//! where the shear it must carry stays under `mu` times the normal load,
//! and while the deposits spread, 8.7 and 6.6 percent of the loaded floor
//! nodes slide, at 1 to 5 percent of the deposit's fastest speed; the rest
//! stick. On the frictionless floor 98 to 100 percent slide.
//!
//! Settled half-widths on the gripping floor: 95.6 mm for 2 Pa at eight
//! seconds, still creeping, 26.3 for 60, 9.5 for 1200.
//!
//!   cargo test --profile quick --all-features --test scratch_bingham_isolated_slump -- --ignored --nocapture
extern crate emerge_engine as emerge;

use emerge::{
    BinghamFluidMaterial, BinghamProps, BoundaryCondition, FrictionBoundary, FromSI, SimConfig,
    Simulation, SlipBoundary, SpawnRegion,
};
use glam::{IVec2, Vec2};

// The slump demo's own constants, in a tank twice as wide.
const GRID: usize = 128;
const DX_M: f32 = 0.002;
const RHO_KG_M3: f32 = 1000.0;
const G: f32 = 9.81;
const ETA_PA_S: f32 = 0.5;
const YIELD_STRAIN: f32 = 0.05;
const COLUMN_CELLS: IVec2 = IVec2::new(10, 20);
const FLOOR_CELLS: f32 = 2.0;
const YIELDS_PA: [f32; 3] = [2.0, 60.0, 1200.0];

fn env(name: &str, default: f32) -> f32 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// The demo's bulk modulus, from its own derivation: sound speed ten times
/// the free-fall speed over the column's height, `K = rho c^2`.
fn bulk_modulus_pa() -> f32 {
    let v_max = (2.0 * G * COLUMN_CELLS.y as f32 * DX_M).sqrt();
    RHO_KG_M3 * (10.0 * v_max).powi(2)
}

struct Deposit {
    h_mm: f32,
    half_width_mm: f32,
    /// Largest particle speed at the end, m/s: near zero means settled.
    vmax_m_s: f32,
    touched_wall: bool,
    /// Share of the floor's loaded nodes that SLID while the deposit was
    /// still spreading, time-averaged. Coulomb friction sets a node's
    /// tangential velocity to exactly zero when it can hold it, so a node
    /// moving along the floor at more than a hundredth of the deposit's
    /// fastest speed is one where the friction was not enough. Measured on
    /// the nodes, where the wall condition acts: the lowest particles sit
    /// half a cell up and move even under perfect no-slip.
    sliding: f32,
    /// Mean tangential speed of those nodes over the deposit's fastest.
    slip_speed_ratio: f32,
}

fn slump(tau0: f32, floor_mu: Option<f32>, seconds: f32, dt: f32) -> Deposit {
    let config = SimConfig {
        min_dt: 1.0e-5,
        max_substeps_per_step: 256,
        ..SimConfig::earth(GRID, DX_M, dt)
    };
    let props = BinghamProps {
        rho_kg_m3: RHO_KG_M3,
        eta_pa_s: ETA_PA_S,
        bulk_modulus_pa: bulk_modulus_pa(),
        yield_stress_pa: tau0,
        shear_modulus_pa: tau0 / YIELD_STRAIN,
        cavitation_pressure_pa: BinghamProps::air_entrained_cavitation_pressure(),
    };
    // A quarter cell right of the tank's axis, so the lattice, which starts
    // at the box's corner (issue #42), is centred ON the axis.
    let axis = GRID as f32 * 0.5;
    let spawn = SpawnRegion {
        spacing: 0.5,
        box_size: COLUMN_CELLS,
        box_center: Vec2::new(axis + 0.25, FLOOR_CELLS + COLUMN_CELLS.y as f32 * 0.5),
        material_id: 0,
        initial_velocity_scale: 0.0,
        ..SpawnRegion::for_sim(&config)
    }
    .mass_from(&props, &config);
    let floor: Box<dyn BoundaryCondition> = match floor_mu {
        None => Box::new(SlipBoundary::new(config.boundary_thickness)),
        Some(mu) => Box::new(FrictionBoundary::new(config.boundary_thickness, mu)),
    };
    let mut sim = Simulation::new(config, spawn)
        .with_default_material(Box::new(BinghamFluidMaterial::from_physical(
            &props, &config,
        )))
        .with_boundary(floor);
    let bt = config.boundary_thickness as i32;
    let (mut slid, mut loaded, mut ratio_sum, mut ratio_frames) = (0usize, 0usize, 0.0f32, 0usize);
    for _ in 0..(seconds / dt).round() as usize {
        sim.step();
        let v_ref = sim
            .particles()
            .v
            .iter()
            .fold(0.0f32, |m, v| m.max(v.length()));
        // Spreading: fastest particle above a millimetre a second.
        if v_ref * DX_M < 1.0e-3 {
            continue;
        }
        let grid = sim.grid();
        let (mut speed_sum, mut nodes) = (0.0f32, 0usize);
        for x in 0..GRID as i32 {
            for y in 0..bt {
                let cell = IVec2::new(x, y);
                if grid.mass_at(cell) <= 0.0 {
                    continue;
                }
                let v_t = grid.velocity_at(cell).x.abs();
                loaded += 1;
                nodes += 1;
                speed_sum += v_t;
                if v_t > 0.01 * v_ref {
                    slid += 1;
                }
            }
        }
        if nodes > 0 {
            ratio_sum += speed_sum / nodes as f32 / v_ref;
            ratio_frames += 1;
        }
    }
    let p = sim.particles();
    let (mut top, mut lo, mut hi, mut vmax) = (f32::MIN, f32::MAX, f32::MIN, 0.0f32);
    for i in 0..p.len() {
        top = top.max(p.x[i].y);
        lo = lo.min(p.x[i].x);
        hi = hi.max(p.x[i].x);
        vmax = vmax.max(p.v[i].length());
    }
    // Same definitions as `bingham_slump_probe`: height above the floor,
    // half the horizontal extent.
    let wall = config.boundary_thickness as f32 + 1.0;
    Deposit {
        h_mm: (top - FLOOR_CELLS).max(0.0) * DX_M * 1000.0,
        half_width_mm: (hi - lo) * 0.5 * DX_M * 1000.0,
        vmax_m_s: vmax * DX_M,
        touched_wall: lo < wall || hi > GRID as f32 - wall,
        sliding: slid as f32 / loaded.max(1) as f32,
        slip_speed_ratio: ratio_sum / ratio_frames.max(1) as f32,
    }
}

#[test]
#[ignore = "diagnostic probe kept for reruns, not part of the CI suite"]
fn each_column_alone_on_a_slip_and_a_gripping_floor() {
    let seconds = env("SLUMP_SECONDS", 3.0);
    let dt = env("SLUMP_DT", 0.001);
    println!(
        "each column alone, {GRID}-cell tank ({} mm), {seconds} s at {dt} s a frame",
        GRID as f32 * DX_M * 1000.0
    );
    println!(
        "  tau_0 Pa  floor          h mm   L mm   h/L    tau_0 read back   vmax m/s   base nodes sliding  slip speed   note"
    );
    // `SLUMP_TAU` runs one column only, to follow a slow one for longer.
    let only = std::env::var("SLUMP_TAU")
        .ok()
        .and_then(|v| v.parse::<f32>().ok());
    for tau0 in YIELDS_PA
        .into_iter()
        .filter(|t| only.is_none_or(|o| o == *t))
    {
        for (label, mu) in [("slip", None), ("friction 1", Some(1.0f32))] {
            let d = slump(tau0, mu, seconds, dt);
            let (h, l) = (d.h_mm * 1.0e-3, d.half_width_mm * 1.0e-3);
            let read_back = RHO_KG_M3 * G * h * h / (2.0 * l.max(1.0e-9));
            println!(
                "  {tau0:>7}   {label:<12} {:>6.1} {:>6.1}  {:>5.2}   {:>9.1} Pa     {:>7.4}   {:>13.1} %   {:>9.3}   {}",
                d.h_mm,
                d.half_width_mm,
                d.h_mm / d.half_width_mm.max(1.0e-6),
                read_back,
                d.vmax_m_s,
                100.0 * d.sliding,
                d.slip_speed_ratio,
                if d.touched_wall { "reached a wall" } else { "" }
            );
        }
    }
}

/// Does the bottom row of MATERIAL stay put, not only the floor's nodes?
///
/// `each_column_alone_on_a_slip_and_a_gripping_floor` counts floor nodes
/// that slide, and finds 91 to 93 percent of them held while the deposits
/// spread. That is measured on the wall layer, the nodes at y = 0 and 1,
/// and the particles start above it at y = 2: the bottom row also reads
/// unconstrained nodes above itself, so it can travel while the wall nodes
/// under it stick. On screen the slumped 60 Pa deposit's lattice rows run
/// outward all the way down to the floor, which is what a sliding base
/// looks like. This follows the bottom row of particles themselves.
#[test]
#[ignore = "diagnostic probe kept for reruns, not part of the CI suite"]
fn the_bottom_row_of_material_on_each_floor() {
    let seconds = env("SLUMP_SECONDS", 3.0);
    let dt = env("SLUMP_DT", 0.001);
    let tau0 = env("SLUMP_TAU", 60.0);
    println!("{tau0} Pa column alone, {seconds} s: its bottom row of particles, start against end");
    println!("  floor          start span mm     end span mm      mean |dx| mm   largest |dx| mm");
    for (label, mu) in [("slip", None), ("friction 1", Some(1.0f32))] {
        let config = SimConfig {
            min_dt: 1.0e-5,
            max_substeps_per_step: 256,
            ..SimConfig::earth(GRID, DX_M, dt)
        };
        let props = BinghamProps {
            rho_kg_m3: RHO_KG_M3,
            eta_pa_s: ETA_PA_S,
            bulk_modulus_pa: bulk_modulus_pa(),
            yield_stress_pa: tau0,
            shear_modulus_pa: tau0 / YIELD_STRAIN,
            cavitation_pressure_pa: BinghamProps::air_entrained_cavitation_pressure(),
        };
        let axis = GRID as f32 * 0.5;
        let spawn = SpawnRegion {
            spacing: 0.5,
            box_size: COLUMN_CELLS,
            box_center: Vec2::new(axis + 0.25, FLOOR_CELLS + COLUMN_CELLS.y as f32 * 0.5),
            material_id: 0,
            initial_velocity_scale: 0.0,
            ..SpawnRegion::for_sim(&config)
        }
        .mass_from(&props, &config);
        let floor: Box<dyn BoundaryCondition> = match mu {
            None => Box::new(SlipBoundary::new(config.boundary_thickness)),
            Some(m) => Box::new(FrictionBoundary::new(config.boundary_thickness, m)),
        };
        let mut sim = Simulation::new(config, spawn)
            .with_default_material(Box::new(BinghamFluidMaterial::from_physical(
                &props, &config,
            )))
            .with_boundary(floor);
        // The bottom row: every particle spawned on the lowest lattice line.
        let bottom: Vec<(usize, f32)> = {
            let p = sim.particles();
            let lowest = (0..p.len()).map(|i| p.x[i].y).fold(f32::MAX, f32::min);
            (0..p.len())
                .filter(|&i| p.x[i].y < lowest + 0.25)
                .map(|i| (i, p.x[i].x))
                .collect()
        };
        for _ in 0..(seconds / dt).round() as usize {
            sim.step();
        }
        let p = sim.particles();
        let span = |xs: &mut dyn Iterator<Item = f32>| {
            xs.fold((f32::MAX, f32::MIN), |(lo, hi), x| (lo.min(x), hi.max(x)))
        };
        let (s0, s1) = span(&mut bottom.iter().map(|&(_, x)| x));
        let (e0, e1) = span(&mut bottom.iter().map(|&(i, _)| p.x[i].x));
        let moves: Vec<f32> = bottom
            .iter()
            .map(|&(i, x0)| (p.x[i].x - x0).abs())
            .collect();
        let mean = moves.iter().sum::<f32>() / moves.len() as f32;
        let most = moves.iter().copied().fold(0.0f32, f32::max);
        let mm = DX_M * 1000.0;
        println!(
            "  {label:<12} {:>6.1} to {:>6.1}   {:>6.1} to {:>6.1}   {:>10.2}     {:>10.2}",
            (s0 - axis) * mm,
            (s1 - axis) * mm,
            (e0 - axis) * mm,
            (e1 - axis) * mm,
            mean * mm,
            most * mm
        );
    }
}

/// Does a slumped deposit RING before it stops?
///
/// Watching the demo, the 60 Pa deposit sways for a long while after it
/// slumps. The standing suspect is the model: below its yield stress the
/// elastoviscoplastic branch is elastic with no dissipation at all (issue
/// #43), and a deposit comes to rest exactly on its yield surface, so it
/// should oscillate as an elastic body and lose energy only when a swing
/// carries part of it back over yield. That predicts a slow, weakly damped
/// oscillation, at roughly the shear-wave period of the deposit. This
/// follows the column's vertical centre-of-mass velocity: oscillation shows
/// as sign changes, damping as how fast their amplitude falls.
#[test]
#[ignore = "diagnostic probe kept for reruns, not part of the CI suite"]
fn a_slumped_deposit_rings_before_it_stops() {
    let seconds = env("SLUMP_SECONDS", 4.0);
    let dt = env("SLUMP_DT", 0.001);
    let tau0 = env("SLUMP_TAU", 60.0);
    let config = SimConfig {
        min_dt: 1.0e-5,
        max_substeps_per_step: 256,
        ..SimConfig::earth(GRID, DX_M, dt)
    };
    let props = BinghamProps {
        rho_kg_m3: RHO_KG_M3,
        eta_pa_s: ETA_PA_S,
        bulk_modulus_pa: bulk_modulus_pa(),
        yield_stress_pa: tau0,
        shear_modulus_pa: tau0 / YIELD_STRAIN,
        cavitation_pressure_pa: BinghamProps::air_entrained_cavitation_pressure(),
    };
    let spawn = SpawnRegion {
        spacing: 0.5,
        box_size: COLUMN_CELLS,
        box_center: Vec2::new(
            GRID as f32 * 0.5 + 0.25,
            FLOOR_CELLS + COLUMN_CELLS.y as f32 * 0.5,
        ),
        material_id: 0,
        initial_velocity_scale: 0.0,
        ..SpawnRegion::for_sim(&config)
    }
    .mass_from(&props, &config);
    let mut sim = Simulation::new(config, spawn)
        .with_default_material(Box::new(BinghamFluidMaterial::from_physical(
            &props, &config,
        )))
        .with_boundary(Box::new(FrictionBoundary::new(
            config.boundary_thickness,
            1.0,
        )));
    let shear_wave = (props.shear_modulus_pa / RHO_KG_M3).sqrt();
    println!("{tau0} Pa column alone on the gripping floor; shear-wave speed {shear_wave:.2} m/s");
    println!(
        "  window        centre-of-mass vy swings   largest |vy| mm/s   largest particle speed mm/s"
    );
    let frames = (seconds / dt).round() as usize;
    let window = (0.5 / dt).round() as usize;
    let (mut crossings, mut peak_vy, mut peak_v, mut last_sign) = (0usize, 0.0f32, 0.0f32, 0.0f32);
    for frame in 0..frames {
        sim.step();
        let p = sim.particles();
        let n = p.len() as f32;
        let vy = p.v.iter().map(|v| v.y).sum::<f32>() / n * DX_M * 1000.0;
        let vmax = p.v.iter().fold(0.0f32, |m, v| m.max(v.length())) * DX_M * 1000.0;
        // Ignore the first half second, the collapse itself.
        if frame >= window {
            let sign = vy.signum();
            if last_sign != 0.0 && sign != last_sign && vy.abs() > 1.0e-4 {
                crossings += 1;
            }
            if vy.abs() > 1.0e-4 {
                last_sign = sign;
            }
            peak_vy = peak_vy.max(vy.abs());
            peak_v = peak_v.max(vmax);
        }
        if frame >= window && (frame + 1) % window == 0 {
            let t1 = (frame + 1) as f32 * dt;
            println!(
                "  {:.1} to {:.1} s        {crossings:>6}             {peak_vy:>10.3}             {peak_v:>10.3}",
                t1 - 0.5,
                t1
            );
            crossings = 0;
            peak_vy = 0.0;
            peak_v = 0.0;
        }
    }
}

/// Is the dilated bottom layer the gripping floor, or the wall itself?
///
/// On the demo's scene the slumped deposits' mean volume ratio sits above
/// one, and all of that excess is in the band of particles next to the
/// floor: J = 1.0108 there under the 60 Pa deposit, the grid-gathered
/// density 0.990 of rest, the law's raw pressure -556 Pa, a fifth of the
/// band on the cavitation floor (`tests/scratch_bingham_deposit_state.rs`).
/// Two things could hold it. The gripping floor, holding the base while
/// the material above flows outward, stretches it; or the wall itself,
/// where the kernel reaches into the empty boundary layer and gathers too
/// little density, the way it does at a free surface. Changing only the
/// floor separates them: the wall is there on both floors, the grip only
/// on one.
///
/// Found, 60 Pa, five seconds at 1 ms: the grip. On the gripping floor the
/// band ends at J 1.0157, 29.5 percent of it at the cavitation pressure; on
/// the slip floor at 0.9995 and 2.2 percent. The dilation builds during the
/// impact, 0.998 at 20 ms, 1.005 at 50 ms, 1.016 by 0.2 s, and then stays
/// frozen to the fourth decimal while the deposit rings down. It is set by
/// the flow and held, not a ratchet at rest. What in the gripped update
/// stretches the layer is not established: issue #44.
#[test]
#[ignore = "diagnostic probe kept for reruns, not part of the CI suite"]
fn the_bottom_layer_volume_on_each_floor() {
    let seconds = env("SLUMP_SECONDS", 3.0);
    let dt = env("SLUMP_DT", 0.001);
    let tau0 = env("SLUMP_TAU", 60.0);
    println!(
        "{tau0} Pa column alone, {seconds} s: the band of particles within 1.5 cells of the floor"
    );
    println!(
        "  floor        time   particles   mean J    gathered rho/rho0   on the cavitation floor   rest of the deposit mean J   fastest particle"
    );
    for (label, mu) in [("slip", None), ("friction 1", Some(1.0f32))] {
        let config = SimConfig {
            min_dt: 1.0e-5,
            max_substeps_per_step: 256,
            ..SimConfig::earth(GRID, DX_M, dt)
        };
        let props = BinghamProps {
            rho_kg_m3: RHO_KG_M3,
            eta_pa_s: ETA_PA_S,
            bulk_modulus_pa: bulk_modulus_pa(),
            yield_stress_pa: tau0,
            shear_modulus_pa: tau0 / YIELD_STRAIN,
            cavitation_pressure_pa: BinghamProps::air_entrained_cavitation_pressure(),
        };
        let material = BinghamFluidMaterial::from_physical(&props, &config);
        let (stiff, power, rest, min_d, floor_p) = (
            material.eos_stiffness,
            material.eos_power,
            material.rest_density,
            material.min_density,
            material.pressure_floor,
        );
        let spawn = SpawnRegion {
            spacing: 0.5,
            box_size: COLUMN_CELLS,
            box_center: Vec2::new(
                GRID as f32 * 0.5 + 0.25,
                FLOOR_CELLS + COLUMN_CELLS.y as f32 * 0.5,
            ),
            material_id: 0,
            initial_velocity_scale: 0.0,
            ..SpawnRegion::for_sim(&config)
        }
        .mass_from(&props, &config);
        let wall: Box<dyn BoundaryCondition> = match mu {
            None => Box::new(SlipBoundary::new(config.boundary_thickness)),
            Some(m) => Box::new(FrictionBoundary::new(config.boundary_thickness, m)),
        };
        let mut sim = Simulation::new(config, spawn)
            .with_default_material(Box::new(material))
            .with_boundary(wall);
        let frames = (seconds / dt).round() as usize;
        for frame in 1..=frames {
            sim.step();
            // Five looks during the run: when the dilation appears says
            // whether it is set by the flow or keeps growing at rest.
            if frame % (frames / 5).max(1) != 0 {
                continue;
            }
            let p = sim.particles();
            let (mut nb, mut jb, mut rb, mut cb, mut nr, mut jr) =
                (0usize, 0.0f64, 0.0f64, 0usize, 0usize, 0.0f64);
            for i in 0..p.len() {
                let j = f64::from(p.deformation_gradient[i].determinant());
                if p.x[i].y < FLOOR_CELLS + 1.5 {
                    let density = p.density[i].max(min_d).min(rest * 2.0);
                    nb += 1;
                    jb += j;
                    rb += f64::from(density / rest);
                    if stiff * ((density / rest).powf(power) - 1.0) < floor_p {
                        cb += 1;
                    }
                } else {
                    nr += 1;
                    jr += j;
                }
            }
            let vmax = (0..p.len()).map(|i| p.v[i].length()).fold(0.0f32, f32::max);
            println!(
                "  {label:<12} t={:.1}s {nb:>5}   {:>8.5}      {:>8.5}              {:>5.1} %                  {:>8.5}      {:.2} mm/s",
                frame as f32 * dt,
                jb / nb.max(1) as f64,
                rb / nb.max(1) as f64,
                100.0 * cb as f64 / nb.max(1) as f64,
                jr / nr.max(1) as f64,
                vmax * DX_M * 1000.0
            );
        }
    }
}

/// The demo's peak collapse speed read 0.787 m/s on the earlier scene
/// (64-cell tank, slip floor, columns touching) and 0.662 m/s on today's
/// (160-cell tank, gripping floor, columns apart). Is it the floor? The
/// 2 Pa column alone, on each floor: the fastest particle over the whole
/// run, in m/s.
#[test]
#[ignore = "diagnostic probe kept for reruns, not part of the CI suite"]
fn the_peak_speed_on_each_floor() {
    let seconds = env("SLUMP_SECONDS", 2.0);
    let dt = env("SLUMP_DT", 0.001);
    let tau0 = env("SLUMP_TAU", 2.0);
    for (label, mu) in [("slip", None), ("friction 1", Some(1.0f32))] {
        let config = SimConfig {
            min_dt: 1.0e-5,
            max_substeps_per_step: 256,
            ..SimConfig::earth(GRID, DX_M, dt)
        };
        let props = BinghamProps {
            rho_kg_m3: RHO_KG_M3,
            eta_pa_s: ETA_PA_S,
            bulk_modulus_pa: bulk_modulus_pa(),
            yield_stress_pa: tau0,
            shear_modulus_pa: tau0 / YIELD_STRAIN,
            cavitation_pressure_pa: BinghamProps::air_entrained_cavitation_pressure(),
        };
        let spawn = SpawnRegion {
            spacing: 0.5,
            box_size: COLUMN_CELLS,
            box_center: Vec2::new(
                GRID as f32 * 0.5 + 0.25,
                FLOOR_CELLS + COLUMN_CELLS.y as f32 * 0.5,
            ),
            material_id: 0,
            initial_velocity_scale: 0.0,
            ..SpawnRegion::for_sim(&config)
        }
        .mass_from(&props, &config);
        let wall: Box<dyn BoundaryCondition> = match mu {
            None => Box::new(SlipBoundary::new(config.boundary_thickness)),
            Some(m) => Box::new(FrictionBoundary::new(config.boundary_thickness, m)),
        };
        let mut sim = Simulation::new(config, spawn)
            .with_default_material(Box::new(BinghamFluidMaterial::from_physical(
                &props, &config,
            )))
            .with_boundary(wall);
        let (mut peak, mut at) = (0.0f32, 0usize);
        for frame in 1..=(seconds / dt).round() as usize {
            sim.step();
            let fastest = sim
                .particles()
                .v
                .iter()
                .map(|v| v.length())
                .fold(0.0f32, f32::max);
            if fastest > peak {
                (peak, at) = (fastest, frame);
            }
        }
        println!(
            "{tau0} Pa column alone, {label:<10}: peak {:.3} m/s at {:.3} s",
            peak * DX_M,
            at as f32 * dt
        );
    }
}

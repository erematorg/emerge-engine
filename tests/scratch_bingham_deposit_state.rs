//! What state are the slump demo's settled deposits actually in?
//!
//! Three questions the demo raised on screen, measured on its own scene
//! (built from `examples/cpu/bingham_slump_scene.rs`, which the demo
//! includes too):
//!
//! 1. The slumped deposits end with a mean volume ratio above one, 1.0043
//!    for 2 Pa and 1.0033 for 60 Pa. Read through the Tait law that is a
//!    tension of some 330 Pa, which a layer whose weight is tens of pascals
//!    and whose shear is capped at a few cannot hold in vertical
//!    equilibrium. So either it is not at rest, or something holds the
//!    dilation. This splits each deposit into height bands and prints, per
//!    band, the volume ratio, the density the grid gathers (which is what
//!    the CPU law turns into a pressure, not the volume ratio), the pressure
//!    the law then computes before its floor, how much of the band sits on
//!    that floor, and the vertical stress the band actually carries beside
//!    the one its depth under the local surface asks for, `-rho g d`. The
//!    vertical stress, not the pressure, is what equilibrium fixes: a
//!    material that holds a shear carries part of its weight in the
//!    deviator.
//! 2. The thin left deposit looks torn, with holes. It is about three cells
//!    thick, at the edge of what the method resolves. This counts isolated
//!    particles with the criterion `examples/cpu/fragmentation_check_cpu.rs`
//!    already uses (nearest neighbour more than four spacings away) and the
//!    largest nearest-neighbour distance.
//! 3. The demo's stress view (`V`) coloured by `von_mises_stress_field`,
//!    which takes the plane-stress von Mises of the FULL stress, pressure
//!    included, and scaled it by the middle column's yield. A pure pressure
//!    reads as its own magnitude there, not zero. This prints that value
//!    beside the deviatoric shear the yield criterion actually tests, which
//!    is what the view now shows.
//!
//! Found, eight seconds at 1 ms:
//!
//! 1. The excess is the bottom quarter of each slumped deposit and nothing
//!    else: J 1.0097 (2 Pa) and 1.0108 (60 Pa) there, gathered density 0.99
//!    of rest, 24 and 22 percent at the cavitation pressure; every band
//!    above at 1 within 1e-3. The gripping floor holds it, issue #44
//!    (`scratch_bingham_isolated_slump`, bottom layer on each floor). The
//!    pressure alone is the wrong comparison with the depth: the vertical
//!    stress, which equilibrium fixes, matches `-rho g d` to 14 percent in
//!    every band of the 60 Pa deposit and 10 in the 1200 Pa one, the
//!    deviator carrying the rest. The 2 Pa deposit, under three cells thick,
//!    does not split into bands that finely.
//! 2. No holes by the fragmentation criterion: no isolated particle in any
//!    deposit, the widest nearest-neighbour gap 1.66 spacings in the thin one.
//! 3. The old colour value was the weight, not the shear: the 1200 Pa column
//!    read 3.36 times the middle yield on average while its shear stood at
//!    0.08 of its own.
//! 4. What the view shows now matches the screen: 96 percent of the 2 Pa
//!    layer at 0.75 of its yield or more, the 1200 Pa column all under
//!    0.25, the 60 Pa deposit 89 percent between 0.5 and 1, where the colour
//!    map changes fastest. Part of that deposit's spread is from one line
//!    of particles to the next (neighbour gap 0.075 against a spread of
//!    0.168), which `the_stripes_on_each_floor` shows is not the floor.
//!
//!   cargo test --profile quick --all-features --test scratch_bingham_deposit_state -- --ignored --nocapture
extern crate emerge_engine as emerge;

#[path = "../examples/cpu/bingham_slump_scene.rs"]
mod bingham_slump_scene;
use bingham_slump_scene::*;

use emerge::{
    BinghamProps, BoundaryCondition, FrictionBoundary, MaterialModel, Simulation, SlipBoundary,
    SpawnRegion,
};
use glam::{Mat2, Vec2};

fn env(name: &str, default: f32) -> f32 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// The formula `MaterialRegistry::von_mises_stress_field` applies, restated
/// here so the comparison does not depend on the code under test.
fn plane_stress_von_mises(s: Mat2) -> f32 {
    let (sxx, syy) = (s.x_axis.x, s.y_axis.y);
    let sxy = 0.5 * (s.x_axis.y + s.y_axis.x);
    (sxx * sxx - sxx * syy + syy * syy + 3.0 * sxy * sxy)
        .max(0.0)
        .sqrt()
}

/// How much a value varies across a set of particles (its standard
/// deviation) against how much it jumps between neighbours (the mean gap
/// between a particle and the mean of those within one cell). A field the
/// grid resolves varies smoothly, so its jumps are small against its
/// spread; stripes from one line of particles to the next are not.
fn spread_and_neighbour_gap(xs: &[Vec2], value: &[f32]) -> (f32, f32) {
    let n = value.len().max(1) as f32;
    let mean = value.iter().sum::<f32>() / n;
    let spread = (value.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n).sqrt();
    let mut gap = 0.0f32;
    for (a, xa) in xs.iter().enumerate() {
        let near: Vec<f32> = xs
            .iter()
            .enumerate()
            .filter(|&(b, xb)| b != a && (*xb - *xa).length() < 1.0)
            .map(|(b, _)| value[b])
            .collect();
        if !near.is_empty() {
            gap += (value[a] - near.iter().sum::<f32>() / near.len() as f32).abs();
        }
    }
    (spread, gap / n)
}

#[test]
#[ignore = "diagnostic probe kept for reruns, not part of the CI suite"]
fn the_settled_deposits_state() {
    let seconds = env("DEPOSIT_SECONDS", 8.0);
    let dt = env("DEPOSIT_DT", 0.001);
    let (mut sim, materials) = make_sim(1.0, 1.0, dt);
    let mut watches = [SlumpWatch::default(); 3];
    for _ in 0..(seconds / dt).round() as usize {
        sim.step();
        for (slot, watch) in watches.iter_mut().enumerate() {
            if let Some((h_m, half_width_m, speed)) = deposit(&sim, slot as u32) {
                watch.read(h_m, half_width_m, speed, 9.81);
            }
        }
    }
    let p = sim.particles();
    println!(
        "the demo's scene, {seconds} s, bulk modulus {:.0} Pa",
        bulk_modulus_pa()
    );

    for slot in 0..3u32 {
        let m = &materials[slot as usize];
        // The engine's own SI-to-grid stress factor for this material, read
        // off the floor it converted: grid pressure per pascal.
        let grid_per_pa = m.pressure_floor / BinghamProps::air_entrained_cavitation_pressure();
        let idx: Vec<usize> = (0..p.len()).filter(|&i| p.material_id[i] == slot).collect();
        let Some((h_m, half_width_m, speed)) = deposit(&sim, slot) else {
            continue;
        };
        println!(
            "{} ({:.0} Pa in): h {:.1} mm, L {:.1} mm, {}",
            COLUMN_LABEL[slot as usize],
            YIELD_STRESS_PA[slot as usize],
            h_m * 1000.0,
            half_width_m * 1000.0,
            describe(&watches[slot as usize].read(h_m, half_width_m, speed, 9.81))
        );

        // 1. Volume and pressure by height band.
        let top = idx.iter().map(|&i| p.x[i].y).fold(f32::MIN, f32::max);
        const BANDS: usize = 4;
        println!(
            "   band (height, mm)   particles   mean J    gathered rho/rho0   raw pressure Pa   on the floor   vertical stress Pa   -rho g d Pa"
        );
        for b in 0..BANDS {
            let lo = FLOOR_CELLS + (top - FLOOR_CELLS) * b as f32 / BANDS as f32;
            let hi = FLOOR_CELLS + (top - FLOOR_CELLS) * (b + 1) as f32 / BANDS as f32;
            let band: Vec<usize> = idx
                .iter()
                .copied()
                .filter(|&i| p.x[i].y >= lo && (p.x[i].y < hi || b == BANDS - 1))
                .collect();
            if band.is_empty() {
                continue;
            }
            let n = band.len() as f64;
            let (mut j, mut rho, mut raw, mut clamped) = (0.0f64, 0.0f64, 0.0f64, 0usize);
            let (mut syy, mut weight) = (0.0f64, 0.0f64);
            for &i in &band {
                j += f64::from(p.deformation_gradient[i].determinant());
                // The CPU law's own pressure, from the density the grid
                // gathered, with its own density clamp, before the floor.
                let density = p.density[i].max(m.min_density).min(m.rest_density * 2.0);
                let pr = m.eos_stiffness * ((density / m.rest_density).powf(m.eos_power) - 1.0);
                rho += f64::from(density / m.rest_density);
                raw += f64::from(pr / grid_per_pa);
                if pr < m.pressure_floor {
                    clamped += 1;
                }
                // Tension positive, so a column's own weight reads negative.
                syy += f64::from(m.kirchhoff_stress(p, i).y_axis.y / grid_per_pa);
                // Depth under the surface right above this particle, not
                // under the deposit's peak: a slumped deposit is not flat.
                let surface = idx
                    .iter()
                    .filter(|&&k| (p.x[k].x - p.x[i].x).abs() < 1.0)
                    .map(|&k| p.x[k].y)
                    .fold(p.x[i].y, f32::max);
                weight += f64::from(-RHO_KG_M3 * 9.81 * (surface - p.x[i].y) * DX_M);
            }
            println!(
                "   {:>5.1} to {:>5.1}      {:>6}     {:>8.5}      {:>8.5}          {:>9.1}        {:>5.1} %         {:>9.1}        {:>9.1}",
                (lo - FLOOR_CELLS) * DX_M * 1000.0,
                (hi - FLOOR_CELLS) * DX_M * 1000.0,
                band.len(),
                j / n,
                rho / n,
                raw / n,
                100.0 * clamped as f64 / n,
                syy / n,
                weight / n
            );
        }

        // 2. Holes: the fragmentation check's own isolation criterion.
        let xs: Vec<Vec2> = idx.iter().map(|&i| p.x[i]).collect();
        let (mut isolated, mut worst_nn) = (0usize, 0.0f32);
        for (a, xa) in xs.iter().enumerate() {
            let nn = xs
                .iter()
                .enumerate()
                .filter(|&(b, _)| b != a)
                .map(|(_, xb)| (*xb - *xa).length())
                .fold(f32::MAX, f32::min);
            worst_nn = worst_nn.max(nn);
            if nn > SPACING * 4.0 {
                isolated += 1;
            }
        }
        println!(
            "   holes: {isolated} of {} particles isolated (nearest neighbour over {} cells), largest nearest-neighbour distance {:.2} mm ({:.2} spacings)",
            xs.len(),
            SPACING * 4.0,
            worst_nn * DX_M * 1000.0,
            worst_nn / SPACING
        );

        // 3. What the stress view shows, against the shear it names.
        let middle_yield = materials[1].yield_stress.max(1.0e-12);
        let (mut vm_max, mut shear_max, mut vm_sum, mut shear_sum) =
            (0.0f32, 0.0f32, 0.0f64, 0.0f64);
        for &i in &idx {
            let tau = m.kirchhoff_stress(p, i);
            let vm = plane_stress_von_mises(tau) / middle_yield;
            let sh = deviatoric_shear(tau) / m.yield_stress.max(1.0e-12);
            vm_max = vm_max.max(vm);
            shear_max = shear_max.max(sh);
            vm_sum += f64::from(vm);
            shear_sum += f64::from(sh);
        }
        let n = idx.len() as f64;
        println!(
            "   stress view: colour value (plane-stress von Mises, pressure included, / middle yield) mean {:.2}, max {:.2};",
            vm_sum / n,
            vm_max
        );
        println!(
            "                deviatoric shear / its OWN yield mean {:.2}, max {:.2}",
            shear_sum / n,
            shear_max
        );

        // 4. What the view paints now: the shear over the particle's own
        // yield, by depth under the local surface, how much of the deposit
        // falls in each part of the colour map, and how much of the spread
        // is between neighbours rather than across the deposit.
        let value: Vec<f32> = idx
            .iter()
            .map(|&i| deviatoric_shear(m.kirchhoff_stress(p, i)) / m.yield_stress.max(1.0e-12))
            .collect();
        let depth: Vec<f32> = idx
            .iter()
            .map(|&i| {
                idx.iter()
                    .filter(|&&k| (p.x[k].x - p.x[i].x).abs() < 1.0)
                    .map(|&k| p.x[k].y)
                    .fold(p.x[i].y, f32::max)
                    - p.x[i].y
            })
            .collect();
        print!("   V by depth under the surface (cells):");
        for (lo, hi) in [(0.0, 1.0), (1.0, 3.0), (3.0, 6.0), (6.0, f32::MAX)] {
            let v: Vec<f32> = (0..idx.len())
                .filter(|&a| depth[a] >= lo && depth[a] < hi)
                .map(|a| value[a])
                .collect();
            if v.is_empty() {
                continue;
            }
            print!(
                "  [{lo}, {}) n {} mean {:.2} min {:.2} max {:.2}",
                if hi == f32::MAX {
                    "..".to_string()
                } else {
                    hi.to_string()
                },
                v.len(),
                v.iter().sum::<f32>() / v.len() as f32,
                v.iter().copied().fold(f32::MAX, f32::min),
                v.iter().copied().fold(f32::MIN, f32::max)
            );
        }
        println!();
        // The colour map's own breakpoints (`heat` in the renderer): blue
        // fades out by 0.5, green peaks at 0.5, red rises from 0.5 to 0.75.
        let share = |lo: f32, hi: f32| {
            100.0 * value.iter().filter(|&&v| v >= lo && v < hi).count() as f32 / value.len() as f32
        };
        println!(
            "   colour classes: blue < 0.25 {:.0} %, teal 0.25-0.5 {:.0} %, green-yellow 0.5-0.75 {:.0} %, orange-red >= 0.75 {:.0} %",
            share(f32::MIN, 0.25),
            share(0.25, 0.5),
            share(0.5, 0.75),
            share(0.75, f32::MAX)
        );
        let xs: Vec<Vec2> = idx.iter().map(|&i| p.x[i]).collect();
        let (spread, gap) = spread_and_neighbour_gap(&xs, &value);
        println!(
            "   spread across the deposit (standard deviation) {spread:.3}; mean gap between a particle and its neighbours within one cell {gap:.3}"
        );
        // The same, band by band up the deposit: a floor effect would sit
        // in the bottom band and fade above it.
        print!("   neighbour gap by height band, bottom first:");
        for b in 0..4 {
            let lo = FLOOR_CELLS + (top - FLOOR_CELLS) * b as f32 / 4.0;
            let hi = FLOOR_CELLS + (top - FLOOR_CELLS) * (b + 1) as f32 / 4.0;
            let in_band: Vec<usize> = (0..idx.len())
                .filter(|&a| xs[a].y >= lo && (xs[a].y < hi || b == 3))
                .collect();
            let bx: Vec<Vec2> = in_band.iter().map(|&a| xs[a]).collect();
            let bv: Vec<f32> = in_band.iter().map(|&a| value[a]).collect();
            let (s, g) = spread_and_neighbour_gap(&bx, &bv);
            print!("  gap {g:.3} (spread {s:.3}, n {})", bv.len());
        }
        println!();
    }
}

/// Are the stripes the V view shows on the middle deposit the same floor
/// effect as its dilated bottom layer? The bottom layer was the gripping
/// floor: the same column on a slip floor had none. If the stripes are that
/// too, they vanish on a slip floor and sit in the bottom band on the
/// gripping one; if they are something else, neither holds. The 60 Pa column
/// alone, on each floor, in the demo's tank.
///
/// Found, five seconds at 1 ms: not the floor. The neighbour gap is 0.070 on
/// the slip floor and 0.074 on the gripping one, against the same spread,
/// 0.17. On the gripping floor it is largest in the TOP band (0.157), not the
/// bottom one (0.065). The stripes are a subject of their own, cause not
/// established. The gripping run also matches the demo's middle deposit band
/// for band (0.065, 0.087, 0.041, 0.157 against 0.065, 0.088, 0.042, 0.157):
/// in the demo the column is as good as alone.
#[test]
#[ignore = "diagnostic probe kept for reruns, not part of the CI suite"]
fn the_stripes_on_each_floor() {
    let seconds = env("DEPOSIT_SECONDS", 5.0);
    let dt = env("DEPOSIT_DT", 0.001);
    println!("60 Pa column alone, {seconds} s: shear / its yield, spread and neighbour gap");
    for (label, grip) in [("slip", false), ("friction 1", true)] {
        let config = make_config(1.0, dt);
        let (material, props) = column_material(60.0, 1.0, &config);
        let yield_stress = material.yield_stress.max(1.0e-12);
        let spawn = SpawnRegion {
            spacing: SPACING,
            box_size: COLUMN_CELLS,
            box_center: Vec2::new(GRID as f32 * 0.5, FLOOR_CELLS + COLUMN_CELLS.y as f32 * 0.5),
            material_id: 0,
            initial_velocity_scale: 0.0,
            ..SpawnRegion::for_sim(&config)
        }
        .mass_from(&props, &config);
        let wall: Box<dyn BoundaryCondition> = if grip {
            Box::new(FrictionBoundary::new(config.boundary_thickness, 1.0))
        } else {
            Box::new(SlipBoundary::new(config.boundary_thickness))
        };
        let mut sim = Simulation::new(config, spawn)
            .with_default_material(Box::new(material))
            .with_boundary(wall);
        for _ in 0..(seconds / dt).round() as usize {
            sim.step();
        }
        let p = sim.particles();
        let xs: Vec<Vec2> = (0..p.len()).map(|i| p.x[i]).collect();
        let value: Vec<f32> = (0..p.len())
            .map(|i| deviatoric_shear(material.kirchhoff_stress(p, i)) / yield_stress)
            .collect();
        let (spread, gap) = spread_and_neighbour_gap(&xs, &value);
        let top = xs.iter().map(|x| x.y).fold(f32::MIN, f32::max);
        print!(
            "  {label:<11} whole: mean {:.2}, spread {spread:.3}, gap {gap:.3};  by height band, bottom first:",
            value.iter().sum::<f32>() / value.len() as f32
        );
        for b in 0..4 {
            let lo = FLOOR_CELLS + (top - FLOOR_CELLS) * b as f32 / 4.0;
            let hi = FLOOR_CELLS + (top - FLOOR_CELLS) * (b + 1) as f32 / 4.0;
            let in_band: Vec<usize> = (0..xs.len())
                .filter(|&a| xs[a].y >= lo && (xs[a].y < hi || b == 3))
                .collect();
            let bx: Vec<Vec2> = in_band.iter().map(|&a| xs[a]).collect();
            let bv: Vec<f32> = in_band.iter().map(|&a| value[a]).collect();
            let (s, g) = spread_and_neighbour_gap(&bx, &bv);
            print!("  gap {g:.3} (spread {s:.3})");
        }
        println!();
    }
}

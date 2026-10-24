//! Galperin's colliding-blocks count through the 2D grain contract.
//!
//! G. Galperin, "Playing pool with pi (the number pi from a billiard point
//! of view)", Regular and Chaotic Dynamics 8(4), 2003, p. 375
//! (doi:10.1070/RD2003v008n04ABEH000252): two balls on a line, the light
//! one between a wall and the heavy one, the heavy one pushed toward it,
//! every collision absolutely elastic, the wall "a non-moving billiard ball
//! of infinite mass". Counting the collisions between the balls plus the
//! light ball's reflections from the wall, a mass ratio of 100 gives 31 and
//! a ratio of 100^2 gives 314 (his section 3), the digits of pi.
//!
//! Here each ball is a steel disc of the grain contract
//! (`GrainPopulation::new_disc`, the line contact of `disc_contact`) at
//! restitution 1, and the wall is a third such disc held in place, 1e12
//! times the light one's mass: the engine's grain boundaries clamp a
//! grain's normal velocity instead of returning it (see KNOWN_LIMITATIONS),
//! so a bounce needs a contact. Everything sits on one axis with no
//! gravity, friction or rolling, and every count comes from the real
//! grain-grain contact: a collision is one stretch of overlap between a
//! pair, counted when it ends.
//!
//! A ratio of 100^3 is not reproduced. At these speeds and gaps the heavy
//! disc drives the light one into the wall disc while it still overlaps the
//! heavy one, so the collisions stop being separate two-body events
//! (measured: 411 hits instead of 3141, with both contacts active at once
//! for thousands of steps). That is a different regime from this test's.

extern crate emerge_engine as emerge;

use emerge::grains::population::GrainPopulation;
use emerge::materials::granular::disc_contact::{DiscContactConfig, DiscElastic};
use emerge::particle::Grain;
use emerge::{Elastic, SimConfig};
use glam::Vec2;

const DX_M: f32 = 0.01;
const RADIUS_CELLS: f32 = 1.0;
const WALL_X: f32 = 10.0;
const LINE_Y: f32 = 32.0;
/// Light disc's start, cells from the wall disc's centre.
const LIGHT_GAP: f32 = 40.0;
/// Heavy disc's start, cells beyond the light one.
const HEAVY_GAP: f32 = 10.0;
/// The heavy disc's initial speed toward the wall, cells per second.
const SPEED: f32 = 50.0;

/// Chrome steel, as `examples/cpu/grain_newtons_cradle.rs` sources it.
fn steel() -> Elastic {
    Elastic {
        e_pa: 203.4e9,
        nu: 0.29,
        rho_kg_m3: 7833.0,
    }
}

struct Outcome {
    wall_hits: u64,
    ball_hits: u64,
    kinetic_energy_ratio: f64,
}

/// Runs the scene at `step_fraction` of the population's own stable contact
/// step (`contact_step_limit` times `material_cfl_coefficient`, what the
/// grain scenes use) until neither disc can meet another again.
fn run(mass_ratio: f64, step_fraction: f32) -> Outcome {
    let units = SimConfig::earth(64, DX_M, 1.0 / 60.0);
    let disc = |x: f32| Grain::from_si(Vec2::new(x, LINE_Y), RADIUS_CELLS * DX_M, &steel(), &units);
    let mut wall = disc(WALL_X);
    let light = disc(WALL_X + LIGHT_GAP);
    let mut heavy = disc(WALL_X + LIGHT_GAP + HEAVY_GAP);
    let m = light.mass;
    heavy.mass = (f64::from(m) * mass_ratio) as f32;
    heavy.v = Vec2::new(-SPEED, 0.0);
    wall.mass = m * 1.0e12;
    let contact = DiscContactConfig::new(
        DiscElastic::from_si(&steel(), &units),
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
    );
    let mut population = GrainPopulation::new_disc(vec![wall, light, heavy], contact);
    let dt = population.contact_step_limit() * units.material_cfl_coefficient * step_fraction;
    let kinetic = |p: &GrainPopulation| -> f64 {
        p.grains[1..]
            .iter()
            .map(|g| 0.5 * f64::from(g.mass) * f64::from(g.v.length_squared()))
            .sum()
    };
    let initial = kinetic(&population);

    let (mut wall_contact, mut ball_contact) = (false, false);
    let (mut wall_hits, mut ball_hits) = (0u64, 0u64);
    for _ in 0..200_000_000u64 {
        population.step(Vec2::ZERO, dt);
        population.grains[0].x = Vec2::new(WALL_X, LINE_Y);
        population.grains[0].v = Vec2::ZERO;
        let [w, l, h] = [
            population.grains[0],
            population.grains[1],
            population.grains[2],
        ];
        let wall_overlap = w.radius + l.radius - (l.x - w.x).length();
        let ball_overlap = l.radius + h.radius - (h.x - l.x).length();
        if wall_overlap > 0.0 {
            wall_contact = true;
        } else if wall_contact {
            wall_contact = false;
            wall_hits += 1;
        }
        if ball_overlap > 0.0 {
            ball_contact = true;
        } else if ball_contact {
            ball_contact = false;
            ball_hits += 1;
        }
        // Done once both move away from the wall, the light one no faster
        // than the heavy one, with no contact under way.
        if !wall_contact && !ball_contact && l.v.x >= 0.0 && h.v.x >= l.v.x {
            return Outcome {
                wall_hits,
                ball_hits,
                kinetic_energy_ratio: kinetic(&population) / initial,
            };
        }
    }
    panic!("mass ratio {mass_ratio}: still colliding after 2e8 steps");
}

/// Galperin's first case at the grain scenes' own contact step: the margin
/// is wide (pi / atan(sqrt(1/100)) = 31.5), so the contact's small
/// per-collision energy error cannot move the count.
#[test]
fn mass_ratio_100_gives_31_collisions() {
    let outcome = run(100.0, 1.0);
    assert_eq!(
        (outcome.wall_hits, outcome.ball_hits),
        (15, 16),
        "31 hits expected; kinetic energy ratio {}",
        outcome.kinetic_energy_ratio
    );
}

/// Galperin's second case. Its margin is thin (pi / atan(sqrt(1/10^4)) =
/// 314.17), thinner than the contact's own per-collision energy error at
/// the grain scenes' step: measured, 313 hits there, then 314 at 0.3, 0.1
/// and 0.03 of it (`probe_counts_against_step`). Converging at all as the
/// step shrinks needs the compensated position update: with a plain
/// `x += v dt` the same scene gave 313 at 0.3 and at 0.1, each step moving
/// the heavy disc less than the f32 spacing at its coordinate.
#[test]
fn mass_ratio_10000_gives_314_collisions_at_a_finer_step() {
    let outcome = run(1.0e4, 0.3);
    assert_eq!(
        (outcome.wall_hits, outcome.ball_hits),
        (157, 157),
        "314 hits expected; kinetic energy ratio {}",
        outcome.kinetic_energy_ratio
    );
}

/// Prints the counts across step fractions. Run manually:
/// `cargo test --test grains_pi_collisions -- --ignored --nocapture`.
#[test]
#[ignore = "probe: prints counts against the step"]
fn probe_counts_against_step() {
    for ratio in [100.0, 1.0e4] {
        for fraction in [1.0, 0.3, 0.1, 0.03] {
            let o = run(ratio, fraction);
            println!(
                "ratio {ratio:e} step x{fraction}: {} hits ({} wall, {} ball), kinetic energy ratio {:.6}",
                o.wall_hits + o.ball_hits,
                o.wall_hits,
                o.ball_hits,
                o.kinetic_energy_ratio
            );
        }
    }
}

//! Newton's cradle through the 2D grain contract, the geometry and material
//! of `examples/cpu/grain_newtons_cradle.rs` (five chrome steel balls of
//! 1 cm radius on 14 cm strings at real gravity, 5 percent of a radius
//! between neighbours), without its window.
//!
//! Written before measuring: one ball released from 30 degrees strikes the
//! row; within the first strike the far ball leaves with at least 95
//! percent of the struck-in ball's impact speed, and every other ball keeps
//! at most 5 percent of it. With restitution 0.99 and the gaps making the
//! four collisions follow one another, each passes `(1 + e) / 2` on, so
//! about 98 percent should reach the far ball.

extern crate emerge_engine as emerge;

use emerge::grains::population::GrainPopulation;
use emerge::materials::granular::disc_contact::{DiscContactConfig, DiscElastic};
use emerge::particle::Grain;
use emerge::{Elastic, SimConfig};
use glam::Vec2;

const DX_M: f32 = 0.01;
const N: usize = 5;
const RADIUS: f32 = 1.0;
const STRING: f32 = 14.0;
const GAP_FRACTION: f32 = 0.05;
const RESTITUTION: f32 = 0.99;

/// The example's chrome steel (AISI 52100); sources in its `steel()`.
fn steel() -> Elastic {
    Elastic {
        e_pa: 203.4e9,
        nu: 0.29,
        rho_kg_m3: 7833.0,
    }
}

fn anchor(i: usize) -> Vec2 {
    Vec2::new(13.0 + i as f32 * 2.0 * RADIUS * (1.0 + GAP_FRACTION), 34.0)
}

/// The example's rigid strings: each ball back on its circle, radial speed
/// removed.
fn strings(population: &mut GrainPopulation) {
    for (i, grain) in population.grains.iter_mut().enumerate() {
        let to_grain = grain.x - anchor(i);
        let dir = to_grain / to_grain.length();
        grain.x = anchor(i) + dir * STRING;
        grain.v -= grain.v.dot(dir) * dir;
    }
}

#[test]
fn one_released_ball_sends_the_far_ball_out() {
    let units = SimConfig::earth(40, DX_M, 1.0 / 60.0);
    let theta = 30.0f32.to_radians();
    let grains: Vec<Grain> = (0..N)
        .map(|i| {
            let hang = if i == 0 {
                Vec2::new(-theta.sin(), -theta.cos())
            } else {
                Vec2::NEG_Y
            };
            Grain::from_si(anchor(i) + STRING * hang, RADIUS * DX_M, &steel(), &units)
        })
        .collect();
    let contact = DiscContactConfig::new(
        DiscElastic::from_si(&steel(), &units),
        RESTITUTION,
        0.54,
        0.0,
        0.0,
        0.0,
    );
    let mut population = GrainPopulation::new_disc(grains, contact);
    let dt = population.contact_step_limit() * units.material_cfl_coefficient;
    let gravity = Vec2::new(0.0, -9.81 / DX_M);
    // A quarter period of a 14 cm pendulum is 0.19 s; 0.3 s covers the
    // swing down and the strike.
    let mut peak = [0.0f32; N];
    let mut struck = false;
    for _ in 0..(0.3 / dt) as usize {
        population.step(gravity, dt);
        strings(&mut population);
        struck |= population.grains[N - 1].v.length() > 1.0;
        for (i, grain) in population.grains.iter().enumerate() {
            // The released ball's peak before the strike, everyone's after.
            if i == 0 && struck {
                continue;
            }
            peak[i] = peak[i].max(grain.v.length());
        }
    }
    let impact = peak[0];
    let after: Vec<f32> = population
        .grains
        .iter()
        .map(|g| g.v.length() / impact)
        .collect();
    let far = peak[N - 1] / impact;
    println!(
        "impact {:.4} m/s; far ball peak {far:.4} of it; speeds at 0.3 s over the impact speed \
         {after:.4?}; middle peaks {:.4?}",
        impact * DX_M,
        peak[1..N - 1]
            .iter()
            .map(|p| p / impact)
            .collect::<Vec<_>>()
    );
    assert!(struck, "the row was never struck");
    assert!(
        far >= 0.95,
        "the far ball left with {far:.4} of the impact speed"
    );
    // What the others keep once the strike is over (in passing, each middle
    // ball crosses its 5 percent gap at nearly the full speed).
    for (i, kept) in after.iter().enumerate().take(N - 1) {
        assert!(*kept <= 0.05, "ball {i} kept {kept:.4} of the impact speed");
    }
}

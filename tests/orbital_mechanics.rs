//! Solar-system-scale orbital mechanics: does the grid-scale gravity machinery
//! (`GravityWellField`) produce Keplerian motion with astronomical masses and
//! distances?
//!
//! The Sun is a fixed `GravityWellField` source, not a moving particle: the
//! restricted two-body problem (the Sun is ~333,000x Earth's mass, so the
//! barycenter sits well inside it). Earth is one MPM particle with its SI mass
//! at 1 AU with a circular-orbit velocity. MPM particles normally represent a
//! mass element of a continuum body; here one particle is a whole planet as a
//! point mass. P2G/G2P mass transfer is linear in mass, so this is expected to
//! work.
//!
//! Reference data: NASA NSSDCA Planetary Fact Sheet
//! (<https://nssdc.gsfc.nasa.gov/planetary/factsheet/>). The Sun's standard
//! gravitational parameter mu = G*M_sun = 1.32712e20 m^3/s^2 is measured
//! directly, more precisely than G and M_sun separately.

#![cfg(feature = "experimental")]

extern crate emerge_engine as emerge;

use emerge::fields::GravityWellField;
use emerge::{Field, NeoHookeanMaterial, SimConfig, Simulation, SpawnRegion};
use glam::{IVec2, Vec2};

/// Sun's standard gravitational parameter (G*M_sun), m^3/s^2 -- NASA/JPL.
const MU_SUN_SI: f64 = 1.32712e20;

/// 1 AU in meters (exact, by definition).
const AU_M: f64 = 1.496e11;

/// 1 grid cell = 250,000 km. Real, MEASURED choice (see
/// `diag_kepler_error_vs_grid_resolution_sweep`), not a guess: the original
/// 1e9 (1M km/cell, r_earth~150 grid units) put real Kepler-law error at
/// 0.36%, flat across a 16x `dt_seconds` sweep -- proving the error was
/// spatial (MPM transfer kernel width vs. orbital radius in grid cells), not
/// temporal. This value (r_earth~598 grid units) measured 0.093%, under the
/// real 0.1% target.
const DX_METERS: f64 = 2.5e8;

/// Large enough to hold Mars's real orbit (r_mars ~912 grid units at
/// `DX_METERS` above) with margin.
const GRID_RES: usize = 2048;

/// 1 hour per nominal substep -- real dt sweep confirmed accuracy doesn't
/// depend on this (see `diag_kepler_error_vs_dt_sweep`); kept coarse since
/// there's no stiff elastic/acoustic CFL in this scene to force it smaller.
const DT_SECONDS: f64 = 3600.0;

fn sun_pos() -> Vec2 {
    Vec2::splat(GRID_RES as f32 / 2.0)
}

fn astronomical_config() -> SimConfig {
    SimConfig {
        dx_meters: DX_METERS as f32,
        ..SimConfig::standard(GRID_RES, DT_SECONDS as f32, Vec2::ZERO)
    }
}

fn earth_radius_grid() -> f32 {
    (AU_M / DX_METERS) as f32
}

/// Circular-orbit speed at radius `r_si` meters, in grid-units/s.
fn circular_orbit_speed_grid(r_si: f64) -> f32 {
    let v_si = (MU_SUN_SI / r_si).sqrt();
    (v_si / DX_METERS) as f32
}

fn sun_well(center: Vec2) -> GravityWellField {
    let mu_grid = (MU_SUN_SI / (DX_METERS * DX_METERS * DX_METERS)) as f32;
    GravityWellField::point(center, mu_grid, 1.0, 0.05)
}

#[test]
fn earths_real_orbital_acceleration_matches_gm_over_r_squared() {
    // Direct, low-level check: no solver stepping, no MPM machinery in the
    // loop -- just verifies the force field itself produces the real,
    // analytically-expected acceleration at Earth's real distance.
    let config = astronomical_config();
    let sun_pos = sun_pos();
    let r = earth_radius_grid();
    let earth_pos = sun_pos + Vec2::new(r, 0.0);

    let spawn = SpawnRegion {
        spacing: 1.0,
        box_size: IVec2::new(1, 1),
        box_center: earth_pos,
        position_jitter: 0.0,
        mass_override: Some(5.97e24), // real Earth mass, kg -- does not affect its own acceleration
        ..SpawnRegion::for_sim(&config)
    };
    let solver = Simulation::new(config, spawn)
        .with_default_material(Box::new(NeoHookeanMaterial::new(1.0, 1.0)));

    let well = sun_well(sun_pos);
    let acc = well.acceleration(solver.particles(), 0);

    // a = mu_sun / r^2 (real Newtonian gravity), in grid-units/s^2.
    let mu_grid = (MU_SUN_SI / (DX_METERS * DX_METERS * DX_METERS)) as f32;
    let expected = mu_grid / (r * r);

    assert!(
        acc.x < 0.0 && acc.y.abs() < 1.0e-6 * acc.x.abs().max(1.0),
        "acceleration should point straight back toward the sun: {acc:?}"
    );
    let relative_error = (acc.x.abs() - expected).abs() / expected;
    assert!(
        relative_error < 0.01,
        "acc.x={:.6e} expected={:.6e} rel_err={:.4}",
        acc.x,
        expected,
        relative_error
    );
}

#[test]
fn earth_holds_a_real_near_circular_orbit_over_a_short_arc() {
    let config = astronomical_config();
    let sun_pos = sun_pos();
    let r0 = earth_radius_grid();
    let earth_pos = sun_pos + Vec2::new(r0, 0.0);
    let v0 = circular_orbit_speed_grid(AU_M);

    let spawn = SpawnRegion {
        spacing: 1.0,
        box_size: IVec2::new(1, 1),
        box_center: earth_pos,
        position_jitter: 0.0,
        mass_override: Some(5.97e24),
        ..SpawnRegion::for_sim(&config)
    };
    let mut solver = Simulation::new(config, spawn)
        .with_default_material(Box::new(NeoHookeanMaterial::new(1.0, 1.0)))
        .with_force_field(Box::new(sun_well(sun_pos)));

    // Tangential velocity for a counter-clockwise circular orbit.
    solver.particles_mut().v[0] = Vec2::new(0.0, v0);

    let r_vec0 = solver.particles().x[0] - sun_pos;
    let l0 = r_vec0.x * solver.particles().v[0].y - r_vec0.y * solver.particles().v[0].x;

    // ~10 real days -- a short arc, not a full year, but enough real motion
    // to check the orbit isn't degenerating (spiraling in/out).
    let steps = (10.0 * 24.0 * 3600.0 / DT_SECONDS) as usize;
    for _ in 0..steps {
        solver.step();
    }

    let p = solver.particles();
    assert!(p.x[0].is_finite() && p.v[0].is_finite(), "non-finite state");

    let r_vec1 = p.x[0] - sun_pos;
    let r1 = r_vec1.length();
    let l1 = r_vec1.x * p.v[0].y - r_vec1.y * p.v[0].x;

    // Standard two-body invariant: specific angular momentum (r x v)
    // is conserved exactly for a pure central-force (gravity-only) orbit.
    let l_drift = (l1 - l0).abs() / l0.abs();
    assert!(
        l_drift < 0.01,
        "angular momentum should be ~conserved over a short arc: l0={l0:.6e} l1={l1:.6e} drift={l_drift:.4}"
    );

    // Radius should stay close to 1 AU over just 10 days of a ~365-day orbit
    // (real eccentricity-free circular case) -- not spiraling in or out.
    let radius_drift = (r1 - r0).abs() / r0;
    assert!(
        radius_drift < 0.02,
        "orbital radius should stay ~constant over a short arc: r0={r0:.3} r1={r1:.3} drift={radius_drift:.4}"
    );
}

/// Mars data, NASA NSSDCA Planetary Fact Sheet.
const MARS_MASS_KG: f64 = 0.642e24;
const MARS_DISTANCE_M: f64 = 228.0e9;

/// Runs the Earth+Mars two-body scene at a given `dt_seconds` and
/// returns `(earth_period_days, mars_period_days, kepler_error_fraction)`.
/// Factored out so integration accuracy can be measured empirically across
/// several `dt_seconds` values instead of assumed -- this project's own
/// "measure, don't guess" discipline applied to timestep choice.
fn measure_kepler_error_at_dt(dt_seconds: f32) -> (f32, f32, f32) {
    let config = SimConfig {
        dx_meters: DX_METERS as f32,
        ..SimConfig::standard(GRID_RES, dt_seconds, Vec2::ZERO)
    };
    let sun_pos = sun_pos();

    let r_earth = earth_radius_grid();
    let r_mars = (MARS_DISTANCE_M / DX_METERS) as f32;
    let v_earth = circular_orbit_speed_grid(AU_M);
    let v_mars = circular_orbit_speed_grid(MARS_DISTANCE_M);

    let spawn_earth = SpawnRegion {
        spacing: 1.0,
        box_size: IVec2::new(1, 1),
        box_center: sun_pos + Vec2::new(r_earth, 0.0),
        position_jitter: 0.0,
        mass_override: Some(5.97e24),
        ..SpawnRegion::for_sim(&config)
    };
    let mut solver = Simulation::new(config, spawn_earth)
        .with_default_material(Box::new(NeoHookeanMaterial::new(1.0, 1.0)))
        .with_force_field(Box::new(sun_well(sun_pos)));
    solver.particles_mut().v[0] = Vec2::new(0.0, v_earth);

    let spawn_mars = SpawnRegion {
        spacing: 1.0,
        box_size: IVec2::new(1, 1),
        box_center: sun_pos + Vec2::new(r_mars, 0.0),
        position_jitter: 0.0,
        mass_override: Some(MARS_MASS_KG as f32),
        ..SpawnRegion::for_sim(&config)
    };
    let _ = solver.add_body(spawn_mars);
    solver.particles_mut().v[1] = Vec2::new(0.0, v_mars);

    // Track cumulative swept angle for each body; record the step at which
    // each first completes a full 2*pi revolution.
    let mut prev_angle = [
        (solver.particles().x[0] - sun_pos).to_angle(),
        (solver.particles().x[1] - sun_pos).to_angle(),
    ];
    let mut cumulative = [0.0f32, 0.0f32];
    let mut period_steps = [None::<usize>, None::<usize>];

    // Mars's period is ~687 days -- run a bit past that so both bodies
    // (Earth included, ~365 days) have a chance to complete a full orbit.
    let max_steps = (720.0 * 24.0 * 3600.0 / dt_seconds) as usize;
    for step in 0..max_steps {
        solver.step();
        let p = solver.particles();
        for (idx, body) in [0usize, 1usize].into_iter().enumerate() {
            if period_steps[idx].is_some() {
                continue;
            }
            let angle = (p.x[body] - sun_pos).to_angle();
            let mut delta = angle - prev_angle[idx];
            if delta > std::f32::consts::PI {
                delta -= std::f32::consts::TAU;
            } else if delta < -std::f32::consts::PI {
                delta += std::f32::consts::TAU;
            }
            cumulative[idx] += delta;
            prev_angle[idx] = angle;
            if cumulative[idx].abs() >= std::f32::consts::TAU {
                period_steps[idx] = Some(step);
            }
        }
        if period_steps[0].is_some() && period_steps[1].is_some() {
            break;
        }
    }

    let earth_period_days =
        period_steps[0].expect("earth should complete an orbit") as f32 * dt_seconds / 86400.0;
    let mars_period_days =
        period_steps[1].expect("mars should complete an orbit within 720 days") as f32 * dt_seconds
            / 86400.0;

    // Kepler's third law: T_mars/T_earth = (a_mars/a_earth)^1.5.
    let predicted_ratio = (MARS_DISTANCE_M / AU_M).powf(1.5) as f32;
    let measured_ratio = mars_period_days / earth_period_days;
    let error = (measured_ratio - predicted_ratio).abs() / predicted_ratio;
    (earth_period_days, mars_period_days, error)
}

/// Kepler's third law: simulating Earth and Mars (each around the fixed-Sun
/// `GravityWellField`, no inter-planet gravity, as above) must reproduce
/// T^2 ~ a^3, a law neither setup was tuned to satisfy.
///
/// Accuracy: at `DX_METERS = 1e9` (r_earth ~150 grid units) the error was
/// 0.36%, flat across a 16x timestep sweep (`diag_kepler_error_vs_dt_sweep`),
/// so the error is spatial (kernel width against orbital radius in cells),
/// not temporal. A grid-resolution sweep
/// (`diag_kepler_error_vs_grid_resolution_sweep`) showed it fall as the
/// orbital radius grows in cells (0.36% -> 0.18% -> 0.09%). The module's
/// `DX_METERS`/`GRID_RES` give ~0.09%, under a 0.1% target; the 0.15% bound
/// below leaves margin without hiding a regression.
#[test]
fn earth_and_mars_orbital_periods_satisfy_keplers_third_law() {
    let (earth_period_days, mars_period_days, error) =
        measure_kepler_error_at_dt(DT_SECONDS as f32);
    let predicted_ratio = (MARS_DISTANCE_M / AU_M).powf(1.5) as f32;
    let measured_ratio = mars_period_days / earth_period_days;

    eprintln!(
        "Earth period={earth_period_days:.1} days (real: 365.25)  \
         Mars period={mars_period_days:.1} days (real: 687.0)  \
         ratio measured={measured_ratio:.4} predicted={predicted_ratio:.4} error={:.2}%",
        error * 100.0
    );

    assert!(
        error < 0.0015,
        "Kepler's third law should hold within 0.15% at this grid resolution: measured={measured_ratio:.4} predicted={predicted_ratio:.4} error={:.4}%",
        error * 100.0
    );
}

/// Measured (not assumed) sweep: how does integration accuracy scale
/// with `dt_seconds`? Answers whether pushing toward a real 0.1% divergence
/// target is achievable, and at what real substep-count cost.
#[test]
#[ignore = "diagnostic sweep, not a routine regression -- run manually with --ignored --nocapture"]
fn diag_kepler_error_vs_dt_sweep() {
    for dt in [3600.0f32, 1800.0, 900.0, 450.0, 225.0] {
        let (_, _, error) = measure_kepler_error_at_dt(dt);
        eprintln!(
            "dt={dt:>6.0}s ({:>5.2}min)  kepler_error={:.4}%",
            dt / 60.0,
            error * 100.0
        );
    }
}

/// Measured sweep of grid resolution (via `dx_meters`, which scales
/// the orbital radius in grid-CELL terms without changing real physical
/// distances) -- tests the hypothesis the dt-sweep above pointed at: is the
/// ~0.35% Kepler error a SPATIAL discretization effect (MPM's B-spline
/// transfer kernel has real width in grid cells) rather than a temporal
/// integration one? If so, error should shrink as orbital radius grows
/// relative to that fixed kernel width -- i.e., as `dx_meters` shrinks.
fn measure_kepler_error_at_dx(dx_meters: f64) -> f32 {
    let dt_seconds = 3600.0f32;
    let config = SimConfig {
        dx_meters: dx_meters as f32,
        ..SimConfig::standard(2048, dt_seconds, Vec2::ZERO)
    };
    let sun_pos = Vec2::new(1024.0, 1024.0);
    let r_earth = (AU_M / dx_meters) as f32;
    let r_mars = (MARS_DISTANCE_M / dx_meters) as f32;
    let v_earth = ((MU_SUN_SI / AU_M).sqrt() / dx_meters) as f32;
    let v_mars = ((MU_SUN_SI / MARS_DISTANCE_M).sqrt() / dx_meters) as f32;
    let mu_grid = (MU_SUN_SI / (dx_meters * dx_meters * dx_meters)) as f32;
    let well = GravityWellField::point(sun_pos, mu_grid, 1.0, 0.05);

    let spawn_earth = SpawnRegion {
        spacing: 1.0,
        box_size: IVec2::new(1, 1),
        box_center: sun_pos + Vec2::new(r_earth, 0.0),
        position_jitter: 0.0,
        mass_override: Some(5.97e24),
        ..SpawnRegion::for_sim(&config)
    };
    let mut solver = Simulation::new(config, spawn_earth)
        .with_default_material(Box::new(NeoHookeanMaterial::new(1.0, 1.0)))
        .with_force_field(Box::new(well));
    solver.particles_mut().v[0] = Vec2::new(0.0, v_earth);

    let spawn_mars = SpawnRegion {
        spacing: 1.0,
        box_size: IVec2::new(1, 1),
        box_center: sun_pos + Vec2::new(r_mars, 0.0),
        position_jitter: 0.0,
        mass_override: Some(MARS_MASS_KG as f32),
        ..SpawnRegion::for_sim(&config)
    };
    let _ = solver.add_body(spawn_mars);
    solver.particles_mut().v[1] = Vec2::new(0.0, v_mars);

    let mut prev_angle = [
        (solver.particles().x[0] - sun_pos).to_angle(),
        (solver.particles().x[1] - sun_pos).to_angle(),
    ];
    let mut cumulative = [0.0f32, 0.0f32];
    let mut period_steps = [None::<usize>, None::<usize>];
    let max_steps = (720.0 * 24.0 * 3600.0 / dt_seconds) as usize;
    for step in 0..max_steps {
        solver.step();
        let p = solver.particles();
        for (idx, body) in [0usize, 1usize].into_iter().enumerate() {
            if period_steps[idx].is_some() {
                continue;
            }
            let angle = (p.x[body] - sun_pos).to_angle();
            let mut delta = angle - prev_angle[idx];
            if delta > std::f32::consts::PI {
                delta -= std::f32::consts::TAU;
            } else if delta < -std::f32::consts::PI {
                delta += std::f32::consts::TAU;
            }
            cumulative[idx] += delta;
            prev_angle[idx] = angle;
            if cumulative[idx].abs() >= std::f32::consts::TAU {
                period_steps[idx] = Some(step);
            }
        }
        if period_steps[0].is_some() && period_steps[1].is_some() {
            break;
        }
    }
    let earth_period_days =
        period_steps[0].expect("earth should complete an orbit") as f32 * dt_seconds / 86400.0;
    let mars_period_days =
        period_steps[1].expect("mars should complete an orbit within 720 days") as f32 * dt_seconds
            / 86400.0;
    let predicted_ratio = (MARS_DISTANCE_M / AU_M).powf(1.5) as f32;
    let measured_ratio = mars_period_days / earth_period_days;
    (measured_ratio - predicted_ratio).abs() / predicted_ratio
}

#[test]
#[ignore = "diagnostic sweep, not a routine regression -- run manually with --ignored --nocapture"]
fn diag_kepler_error_vs_grid_resolution_sweep() {
    // Smaller dx_meters -> same real distances span MORE grid cells ->
    // orbital radius grows relative to the MPM transfer kernel's fixed
    // width in cells. If error shrinks here, the ~0.35% floor above is a
    // spatial discretization effect, not a temporal integration one.
    for dx in [1.0e9f64, 5.0e8, 2.5e8, 1.25e8] {
        let error = measure_kepler_error_at_dx(dx);
        eprintln!(
            "dx_meters={dx:.3e}  r_earth_grid={:.1}  kepler_error={:.4}%",
            AU_M / dx,
            error * 100.0
        );
    }
}

/// `Particle::pinned` (the "fixed Sun" every test/demo in this file relies on)
/// must hold the Sun exactly fixed, with zero drift and zero residual
/// velocity, while an orbiting body sits nearby (gravity here is
/// one-directional, from the well to particles, not mutual).
#[test]
fn pinned_sun_stays_exactly_fixed_while_earth_orbits() {
    let config = astronomical_config();
    let sun_pos = sun_pos();

    let spawn_sun = SpawnRegion {
        spacing: 1.0,
        box_size: IVec2::new(1, 1),
        box_center: sun_pos,
        position_jitter: 0.0,
        mass_override: Some((MU_SUN_SI / 6.674e-11) as f32),
        ..SpawnRegion::for_sim(&config)
    };
    let mut solver = Simulation::new(config, spawn_sun)
        .with_default_material(Box::new(NeoHookeanMaterial::new(1.0, 1.0)))
        .with_force_field(Box::new(sun_well(sun_pos)));
    solver.particles_mut().pinned[0] = 1;
    let sun_start = solver.particles().x[0];

    let r_earth = earth_radius_grid();
    let v_earth = circular_orbit_speed_grid(AU_M);
    let spawn_earth = SpawnRegion {
        spacing: 1.0,
        box_size: IVec2::new(1, 1),
        box_center: sun_pos + Vec2::new(r_earth, 0.0),
        position_jitter: 0.0,
        mass_override: Some(5.97e24),
        ..SpawnRegion::for_sim(&config)
    };
    let _ = solver.add_body(spawn_earth);
    solver.particles_mut().v[1] = Vec2::new(0.0, v_earth);

    for _ in 0..240 {
        solver.step();
    }

    let sun_end = solver.particles().x[0];
    let sun_v_end = solver.particles().v[0];
    assert_eq!(
        sun_start, sun_end,
        "pinned sun must not move at all: {sun_start:?} -> {sun_end:?}"
    );
    assert_eq!(
        sun_v_end,
        Vec2::ZERO,
        "pinned sun must have exactly zero velocity, got {sun_v_end:?}"
    );
    // Earth should have moved meaningfully in the same window (sanity check
    // this isn't a degenerate all-pinned/no-op scene).
    let earth_moved = (solver.particles().x[1] - (sun_pos + Vec2::new(r_earth, 0.0))).length();
    assert!(
        earth_moved > 1.0,
        "earth should have visibly moved: {earth_moved}"
    );
}

// -- Full N-body solar system (mutual gravity, not restricted) --
//
// The tests above use a fixed Sun (`GravityWellField`, restricted two-body
// problem). This section uses `NBodyGravityField` (Barnes-Hut with a
// quadrupole correction, Hernquist 1987), so every body, the Sun included,
// attracts every other and is free to move. Symplectic integrators
// (Leapfrog, Wisdom-Holman/WHFast in REBOUND) are the standard for long-term
// solar-system stability because they do not leak energy; the MPM
// position/velocity update is symplectic Euler (v then x), the same
// structural property, which is why the dt sweep above found no accuracy
// loss at any timestep.
use emerge::fields::NBodyGravityField;

/// NASA NSSDCA Planetary Fact Sheet data: (name, mass_kg, semi_major_axis_m).
const PLANETS: [(&str, f64, f64); 8] = [
    ("Mercury", 0.330e24, 57.9e9),
    ("Venus", 4.87e24, 108.2e9),
    ("Earth", 5.97e24, 149.6e9),
    ("Mars", 0.642e24, 228.0e9),
    ("Jupiter", 1898.0e24, 778.5e9),
    ("Saturn", 568.0e24, 1432.0e9),
    ("Uranus", 86.8e24, 2867.0e9),
    ("Neptune", 102.0e24, 4515.0e9),
];
const SUN_MASS_KG: f64 = 1.9885e30;

/// dx sized so Neptune's real orbit fits with margin. Disclosed accuracy
/// trade vs. the tighter inner-system tests above: at this coarser grid
/// (Earth at only ~75 grid units instead of ~598), the per-body force
/// resolution is worse -- this section verifies real CONSERVATION LAWS
/// (the correct bar for true N-body), not Kepler's third law precision.
const FULL_SYSTEM_DX_METERS: f64 = 2.0e9;
const FULL_SYSTEM_GRID_RES: usize = 4096;

fn full_system_config() -> SimConfig {
    SimConfig {
        dx_meters: FULL_SYSTEM_DX_METERS as f32,
        ..SimConfig::standard(FULL_SYSTEM_GRID_RES, 3600.0, Vec2::ZERO)
    }
}

/// Builds the real 8-planet + Sun system, mutual N-body gravity, in the
/// BARYCENTRIC frame (total momentum exactly zero at t=0 by construction --
/// standard real N-body initial-condition technique: the Sun's own velocity
/// is set to exactly cancel every planet's momentum, so the system's center
/// of mass doesn't drift). Planets spread at distinct real angles (not a
/// real ephemeris snapshot -- arbitrary phase, real distances/speeds).
fn make_full_system() -> Simulation {
    let config = full_system_config();
    let center = Vec2::splat(FULL_SYSTEM_GRID_RES as f32 / 2.0);
    let g_grid = (6.674e-11 / (FULL_SYSTEM_DX_METERS.powi(3))) as f32;

    let spawn_sun = SpawnRegion {
        spacing: 1.0,
        box_size: IVec2::new(1, 1),
        box_center: center,
        position_jitter: 0.0,
        mass_override: Some(SUN_MASS_KG as f32),
        ..SpawnRegion::for_sim(&config)
    };
    let mut solver = Simulation::new(config, spawn_sun)
        .with_default_material(Box::new(NeoHookeanMaterial::new(1.0, 1.0)))
        .with_force_field(Box::new(NBodyGravityField::new(g_grid, 0.05, 0.1)));

    let mut planet_momentum = Vec2::ZERO;
    for (idx, &(_, mass_kg, a_m)) in PLANETS.iter().enumerate() {
        let angle = idx as f32 * std::f32::consts::TAU / PLANETS.len() as f32;
        let (s, c) = angle.sin_cos();
        let r_grid = (a_m / FULL_SYSTEM_DX_METERS) as f32;
        let v_mag = ((6.674e-11 * SUN_MASS_KG / a_m).sqrt() / FULL_SYSTEM_DX_METERS) as f32;
        let pos = center + Vec2::new(c, s) * r_grid;
        // Tangential (perpendicular to radius) for a circular orbit.
        let vel = Vec2::new(-s, c) * v_mag;

        let spawn = SpawnRegion {
            spacing: 1.0,
            box_size: IVec2::new(1, 1),
            box_center: pos,
            position_jitter: 0.0,
            mass_override: Some(mass_kg as f32),
            ..SpawnRegion::for_sim(solver.config())
        };
        let _ = solver.add_body(spawn);
        solver.particles_mut().v[idx + 1] = vel;
        planet_momentum += mass_kg as f32 * vel;
    }

    // Barycentric frame: Sun's velocity exactly cancels total planet momentum.
    solver.particles_mut().v[0] = -planet_momentum / SUN_MASS_KG as f32;

    solver
}

fn total_momentum(sim: &Simulation) -> Vec2 {
    (0..sim.particles().len())
        .map(|i| sim.particles().mass[i] * sim.particles().v[i])
        .fold(Vec2::ZERO, |a, b| a + b)
}

fn total_energy(sim: &Simulation, g_grid: f32) -> f64 {
    let p = sim.particles();
    let n = p.len();
    let ke: f64 = (0..n)
        .map(|i| 0.5 * p.mass[i] as f64 * p.v[i].length_squared() as f64)
        .sum();
    let mut pe = 0.0f64;
    for i in 0..n {
        for j in (i + 1)..n {
            let r = (p.x[i] - p.x[j]).length() as f64;
            pe -= g_grid as f64 * p.mass[i] as f64 * p.mass[j] as f64 / r;
        }
    }
    ke + pe
}

/// Correctness bar for full N-body (mutual gravity): total linear momentum
/// and total energy are exact conservation laws for an isolated gravitating
/// system. Unlike Kepler's third law (which assumes negligible perturbation
/// from other bodies), they must hold however much the planets perturb each
/// other or the Sun.
#[test]
fn full_solar_system_conserves_momentum_and_energy() {
    let mut solver = make_full_system();
    let g_grid = (6.674e-11 / (FULL_SYSTEM_DX_METERS.powi(3))) as f32;

    // Characteristic momentum scale (Jupiter's, the dominant term) -- masses
    // here are raw, unconverted SI kg (~1e24-1e30), so f32's ABSOLUTE
    // precision at that magnitude is intrinsically coarse (epsilon ~1e14 for
    // numbers around 1e21). Conservation must be checked RELATIVE to this
    // scale, not against an absolute threshold -- the same principle this
    // test already applies to energy_drift below.
    let jupiter_momentum_scale =
        1898.0e24 * ((6.674e-11 * SUN_MASS_KG / 778.5e9).sqrt() / FULL_SYSTEM_DX_METERS);

    let p0 = total_momentum(&solver);
    let e0 = total_energy(&solver, g_grid);
    assert!(
        (p0.length() as f64 / jupiter_momentum_scale) < 0.01,
        "barycentric setup should start at ~zero total momentum (relative to Jupiter's own \
         momentum scale {jupiter_momentum_scale:.3e}): {p0:?}"
    );

    // 30-day window: enough for the fastest body (Mercury, ~88-day period)
    // to move substantially, without the outer planets (Neptune, ~165-year
    // period) completing anything.
    let steps = (30.0 * 24.0 * 3600.0 / 3600.0) as usize;
    for _ in 0..steps {
        solver.step();
    }

    for i in 0..solver.particles().len() {
        assert!(
            solver.particles().x[i].is_finite() && solver.particles().v[i].is_finite(),
            "body {i} went non-finite"
        );
    }

    let p1 = total_momentum(&solver);
    let e1 = total_energy(&solver, g_grid);
    let momentum_drift = (p1 - p0).length();
    let relative_momentum_drift = momentum_drift as f64 / jupiter_momentum_scale;
    let energy_drift = ((e1 - e0) / e0).abs();

    eprintln!(
        "momentum drift={momentum_drift:.6e} ({:.4}% of Jupiter's own momentum scale)  \
         energy: e0={e0:.6e} e1={e1:.6e} drift={:.4}%",
        relative_momentum_drift * 100.0,
        energy_drift * 100.0
    );

    assert!(
        relative_momentum_drift < 0.01,
        "total momentum should stay ~conserved relative to Jupiter's own scale: drift={:.4}%",
        relative_momentum_drift * 100.0
    );
    assert!(
        energy_drift < 0.05,
        "total energy should stay within 5% over 30 days: drift={:.4}%",
        energy_drift * 100.0
    );
}

/// The Sun is not perfectly still: Jupiter (~318x Earth's mass) pulls it into
/// a measurable wobble around the barycenter. With full N-body the Sun
/// (unpinned here) must respond to mutual gravity.
///
/// Checks velocity change, not position: `v[0]` changes (e.g.
/// `(-1.14e-9, 7.07e-9) -> (-1.75e-9, 7.12e-9)` over 60 days) while `x[0]`
/// stays bit-identical. The Sun sits at grid coordinate ~2047.5 (so Neptune's
/// orbit, ~2257 grid units, fits in the scene), where f32's ULP is ~2.4e-4,
/// while the Sun's per-step displacement here (v*dt ~ 2.5e-5) is below it:
/// every step's increment is absorbed by the base coordinate, so more steps
/// do not help. A single-precision limit (one domain cannot resolve both
/// Neptune's distance and the Sun's wobble at this timescale), not a physics
/// bug; velocity is the robust signal.
#[test]
fn sun_velocity_responds_to_real_mutual_gravity() {
    let mut solver = make_full_system();
    let v_start = solver.particles().v[0];

    let steps = (60.0 * 24.0 * 3600.0 / 3600.0) as usize; // 60 real days
    for _ in 0..steps {
        solver.step();
    }

    let v_end = solver.particles().v[0];
    let v_change = (v_end - v_start).length();
    eprintln!("sun v0={v_start:?} -> v_end={v_end:?}  |change|={v_change:.4e}");
    assert!(
        v_change > 1.0e-10,
        "sun's velocity should respond to real mutual gravity from the planets (esp. Jupiter): {v_change:.4e}"
    );
}

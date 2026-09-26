//! What does cavitating water cost, and does it stop where its own
//! saturation pressure says it must?
//!
//! Three columns of the same water at three temperatures, pulled outward
//! by the same impulse. A liquid under tension can only be stretched until
//! its absolute pressure reaches the saturation pressure of its own
//! temperature; past that it tears into vapour instead of holding. Hot
//! water has a much higher saturation pressure, so it gives up under a far
//! gentler pull. This prints, for each column, the floor its temperature
//! sets and the lowest pressure it actually reached, next to the substeps
//! the CFL condition asks for and the frame cost.
//!
//!   cargo run --release --example cavitation_cost_probe
//!   CAV_PROBE_PULL=6 cargo run --release --example cavitation_cost_probe
extern crate emerge_engine as emerge;

use emerge::{
    CavitatingEosTable, CavitatingFluidMaterial, SimConfig, Simulation, SlipBoundary, SpawnRegion,
};
use glam::{IVec2, Vec2};

const GRID: usize = 64;
/// 2 cm cells: a 1.28 m tank holding three columns 20 cm across.
const DX_M: f32 = 0.02;
const COLUMN_CELLS: IVec2 = IVec2::new(10, 10);
const COLUMN_X: [f32; 3] = [14.0, 32.0, 50.0];
const COLUMN_Y: f32 = 32.0;

/// The one thing that differs between the columns.
const TEMPERATURE_C: [f32; 3] = [20.0, 60.0, 90.0];
const KELVIN: f32 = 273.15;

const RHO_L_KG_M3: f32 = 1000.0;
/// Weakly compressible, not water's real 1480 m/s: see the boiling scene's
/// own header for the same rule applied to the same tank.
const C_L_M_S: f32 = 60.0;
/// Cole 1948's Tait exponent for water.
const GAMMA_L: f32 = 7.0;
/// The same declared 6:1 density ratio the engine's two-phase work uses,
/// not steam's real 1673:1.
const RHO_V_KG_M3: f32 = RHO_L_KG_M3 / 6.0;
const GAMMA_V: f32 = 1.33;
/// The mixture band's effective acoustic speed, a model choice.
const C_MIN_M_S: f32 = 1.0;
const WATER_VISCOSITY_PA_S: f32 = 1.0e-3;
const J_MIN: f32 = 0.5;
const J_MAX: f32 = 6.0;

fn main() {
    let env = |name: &str, default: f32| -> f32 {
        std::env::var(name)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    };
    let dt = env("CAV_PROBE_DT", 0.001);
    let seconds = env("CAV_PROBE_SECONDS", 0.3);
    // The speed each side of a column is pulled at, in metres per second,
    // the way a tensile test grips a specimen. Two earlier attempts are
    // worth keeping out of anyone else's way: `apply_radial_impulse` has
    // its strongest falloff at the CENTRE, so it blows a column outward
    // rather than stretching it (measured: -5986 Pa of tension, nowhere
    // near any floor), and prescribing a uniform expansion on EVERY
    // particle does not work either, because the liquid's own pressure
    // cancels the divergence as fast as it is imposed (measured:
    // trace(C) settles at -0.057/s against the +0.1/s prescribed, J stays
    // at 1.0001). Only the edges are gripped here; the inside is left to
    // answer, which is what puts it in tension.
    let grip_m_s = env("CAV_PROBE_GRIP", 2.0);
    // Weightless, for the same measured reason the boiling scene states:
    // free pools of liquid under gravity flatten and run into each other,
    // and no wall in this engine is declared compatible with a strict
    // weakly-compressible liquid (issue #38).
    let gravity_fraction = env("CAV_PROBE_G", 0.0);

    let mut config = SimConfig {
        min_dt: 1.0e-7,
        max_substeps_per_step: 512,
        ..SimConfig::earth(GRID, DX_M, dt)
    };
    config.gravity *= gravity_fraction;

    let table = || {
        CavitatingEosTable::build(
            RHO_L_KG_M3,
            C_L_M_S,
            GAMMA_L,
            RHO_V_KG_M3,
            GAMMA_V,
            C_MIN_M_S,
            // The coldest liquid state this scene ever holds.
            TEMPERATURE_C[0] + KELVIN,
        )
    };
    let material = CavitatingFluidMaterial::new(table(), DX_M, WATER_VISCOSITY_PA_S, J_MIN, J_MAX);
    // A second copy to read the EOS with, so the probe evaluates the same
    // numbers the solver does rather than a reimplementation.
    let reference = table();

    let particle_mass = RHO_L_KG_M3 * (0.5 * DX_M).powi(2);
    let spawn = |slot: usize| SpawnRegion {
        spacing: 0.5,
        box_size: COLUMN_CELLS,
        box_center: Vec2::new(COLUMN_X[slot], COLUMN_Y),
        material_id: 0,
        mass_override: Some(particle_mass),
        initial_velocity_scale: 0.0,
        ..SpawnRegion::for_sim(&config)
    };
    let mut sim = Simulation::new(config, spawn(0))
        .with_default_material(Box::new(material))
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));
    let first = sim.particles().len();
    let _ = sim.add_body(spawn(1));
    let second = sim.particles().len();
    let _ = sim.add_body(spawn(2));
    let total = sim.particles().len();
    let slot_of = move |i: usize| {
        if i < first {
            0
        } else if i < second {
            1
        } else {
            2
        }
    };
    {
        let particles = sim.particles_mut();
        for i in 0..particles.len() {
            particles.temperature[i] = TEMPERATURE_C[slot_of(i)] + KELVIN;
        }
    }

    let si_density = |grid_density: f32| grid_density / (DX_M * DX_M);
    let mut lowest = [f32::MAX; 3];
    let frames = (seconds / dt).round() as usize;
    let (mut substeps, mut worst) = (0usize, 0usize);
    let wall = std::time::Instant::now();
    for frame in 0..frames {
        // Grip one cell at each side of every column and pull, leaving
        // everything between them free to answer.
        {
            let grip_cells = grip_m_s / DX_M;
            let half = COLUMN_CELLS.x as f32 * 0.5;
            let particles = sim.particles_mut();
            for i in 0..particles.len() {
                let offset = particles.x[i].x - COLUMN_X[slot_of(i)];
                if offset.abs() > half - 1.0 {
                    particles.v[i].x = grip_cells * offset.signum();
                }
            }
        }
        sim.step();
        let s = sim.diagnostics_snapshot();
        substeps += s.substeps_last_step;
        worst = worst.max(s.substeps_last_step);
        for (i, p) in sim.particles().iter().enumerate() {
            let slot = slot_of(i);
            let eos = reference.reconstruct(p.temperature);
            let gauge = eos.pressure_gauge_pa(si_density(p.density));
            lowest[slot] = lowest[slot].min(gauge);
        }
        if frame % (frames / 4).max(1) == 0 {
            let (mut tr, mut j, mut rho, mut n) = (0.0f32, 0.0f32, 0.0f32, 0u32);
            for (i, p) in sim.particles().iter().enumerate() {
                if slot_of(i) == 0 {
                    tr += p.velocity_gradient.x_axis.x + p.velocity_gradient.y_axis.y;
                    j += p.deformation_gradient.determinant();
                    rho += si_density(p.density);
                    n += 1;
                }
            }
            let n = n as f32;
            println!(
                "    t={:.2}s  lowest gauge {:.0} / {:.0} / {:.0} Pa   cold column: trace(C)={:.4} /s, J={:.4}, rho={:.1} kg/m3",
                frame as f32 * dt,
                lowest[0],
                lowest[1],
                lowest[2],
                tr / n,
                j / n,
                rho / n
            );
        }
    }
    let ms = wall.elapsed().as_secs_f64() * 1000.0 / frames as f64;
    println!(
        "grip={grip_m_s} m/s dt={dt}: {:.1} substeps/frame (worst {worst}), {ms:.2} ms/frame, {:.0} fps, {total} particles",
        substeps as f64 / frames as f64,
        1000.0 / ms
    );
    println!(
        "  real time: {:.3} s of water per second of wall clock, {:.0} times slower than life",
        dt as f64 / (ms / 1000.0),
        (ms / 1000.0) / dt as f64
    );
    // The gate, in words rather than in numbers to be squinted at: each
    // column must come to rest on its OWN saturation floor and stop
    // there. A column that stops short was never pulled hard enough; a
    // column that sails past it is not obeying the curve at all.
    const TOLERANCE: f32 = 0.02;
    let mut verdict = true;
    for slot in 0..3 {
        let t_k = TEMPERATURE_C[slot] + KELVIN;
        let floor = reference.reconstruct(t_k).p_v_gauge_pa;
        let (mut spread, mut n) = (0.0f32, 0u32);
        let (mut left, mut right) = (f32::MAX, f32::MIN);
        for (i, p) in sim.particles().iter().enumerate() {
            if slot_of(i) == slot {
                left = left.min(p.x.x);
                right = right.max(p.x.x);
                spread += p.deformation_gradient.determinant();
                n += 1;
            }
        }
        println!(
            "  {:.0} C: saturation floor {floor:9.0} Pa gauge, reached {:9.0} Pa ({:.0} % of the way down), mean J {:.3}, {:.1} cells across",
            TEMPERATURE_C[slot],
            lowest[slot],
            100.0 * lowest[slot] / floor,
            spread / n as f32,
            right - left
        );
        let off = lowest[slot] / floor - 1.0;
        if off.abs() > TOLERANCE {
            verdict = false;
            println!(
                "        WRONG: it should have stopped at its own floor, and it is {:+.1} % off",
                100.0 * off
            );
        } else {
            println!(
                "        right: stopped on its own floor, {:+.2} % off the steam table",
                100.0 * off
            );
        }
    }
    let hot_gives_first = reference
        .reconstruct(TEMPERATURE_C[2] + KELVIN)
        .p_v_gauge_pa
        > reference
            .reconstruct(TEMPERATURE_C[0] + KELVIN)
            .p_v_gauge_pa;
    if !hot_gives_first {
        verdict = false;
        println!("  WRONG: the hot column should give up before the cold one");
    }
    println!(
        "{}",
        if verdict {
            "VERDICT: every column stopped where its own temperature says it must, and the hottest gave up first."
        } else {
            "VERDICT: FAILED, see the lines above."
        }
    );
}

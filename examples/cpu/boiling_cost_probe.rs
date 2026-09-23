//! What does a boiling mixture cost, and does it rest where the mixture
//! rule says it should?
//!
//! Three columns of the same water, carrying the same mass, differing only
//! in the fraction of that mass which has boiled. The homogeneous
//! equilibrium model puts a mixture of quality `x` at density
//! `1/((1-x)/rho_l + x/rho_v)`, so the same mass must spread over
//! `J_eq(x) = 1 + (rho_l/rho_v - 1) x` times the liquid's volume. Each
//! column is therefore laid down at that spacing and should simply stay
//! there. What this prints: the substeps the CFL condition asks for, the
//! frame cost, the real-time ratio, and each column's own measured density
//! against the rule.
//!
//!   cargo run --release --example boiling_cost_probe
//!   BOIL_PROBE_DT=0.0005 BOIL_PROBE_C=60 cargo run --release --example boiling_cost_probe
extern crate emerge_engine as emerge;

use emerge::{
    BoilingMixtureMaterial, CavitatingEosTable, SimConfig, Simulation, SlipBoundary, SpawnRegion,
};
use glam::{IVec2, Mat2, Vec2};

const GRID: usize = 64;
/// 2 cm cells: a 1.28 m tank holding three columns of about 20 x 24 cm.
const DX_M: f32 = 0.02;
const FLOOR: f32 = 3.0;
const COLUMN: IVec2 = IVec2::new(10, 10);
const COLUMN_X: [f32; 3] = [14.0, 32.0, 50.0];
/// The one thing that differs between the columns: how much of their mass
/// has boiled. These are not round numbers by accident: they put
/// `sqrt(J_eq)` at exactly 1.0, 1.2 and 1.4, so the three columns hold
/// exactly the same particle count at exactly the same mass, and only
/// their size differs.
const QUALITY: [f32; 3] = [0.0, 0.088, 0.192];

const RHO_L_KG_M3: f32 = 1000.0;
/// Weakly compressible, not water's real 1480 m/s: the artificial
/// compressibility rule asks for ten times the fastest speed the scene
/// itself produces (Monaghan 1994), which is about 0.5 m/s here, so this
/// is thirty times the margin that rule needs. Measured against a stiffer
/// liquid, the equilibrium is the same and only the cost moves:
///
/// ```text
///   c_l          substeps/frame   fps   density off the rule
///   180 m/s           50           29         0.00 %
///    60 m/s           17           82         0.01 %
///    30 m/s            9          146         0.06 %
/// ```
const C_L_M_S: f32 = 60.0;
/// Cole 1948's Tait exponent for water.
const GAMMA_L: f32 = 7.0;
/// Steam at 100 C and one atmosphere is 0.598 kg/m^3, a 1673:1 ratio. This
/// scene runs the compressed 6:1 ratio the engine's two-phase work uses, so
/// a fully boiled particle expands six times rather than sixteen hundred:
/// a declared approximation, not a measurement.
const RHO_V_KG_M3: f32 = RHO_L_KG_M3 / 6.0;
const GAMMA_V: f32 = 1.33;
/// The mixture band's effective acoustic speed, a model choice.
const C_MIN_M_S: f32 = 1.0;
const MELTING_POINT_K: f32 = 273.15;
const BOILING_POINT_K: f32 = 373.15;
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
    let dt = env("BOIL_PROBE_DT", 0.001);
    let c_l = env("BOIL_PROBE_C", C_L_M_S);
    let seconds = env("BOIL_PROBE_SECONDS", 2.0);
    // Zero by default, and not to make the scene behave: three free
    // pools of water under gravity flatten and run into each other
    // within two seconds (measured: 38 to 47 cells across in a 64-cell
    // tank), and the engine has no wall a strict WC-MPM liquid is
    // declared compatible with to keep them apart. Without gravity each
    // column holds its own shape, which is what isolates the mixture
    // rule being tested here.
    let gravity_fraction = env("BOIL_PROBE_G", 0.0);
    let mut config = SimConfig {
        min_dt: 1.0e-7,
        max_substeps_per_step: 512,
        ..SimConfig::earth(GRID, DX_M, dt)
    };
    config.gravity *= gravity_fraction;
    let table = CavitatingEosTable::build(
        RHO_L_KG_M3,
        c_l,
        GAMMA_L,
        RHO_V_KG_M3,
        GAMMA_V,
        C_MIN_M_S,
        MELTING_POINT_K,
    );
    let material =
        BoilingMixtureMaterial::from_table(&table, DX_M, WATER_VISCOSITY_PA_S, J_MIN, J_MAX);

    // Every particle carries the liquid's own mass whatever its column: the
    // quality changes how far apart they sit, not how heavy they are.
    let liquid_spacing = 0.5_f32;
    let particle_mass = RHO_L_KG_M3 * (liquid_spacing * DX_M).powi(2);
    let spawn = |slot: usize| {
        let stretch = material.j_eq(QUALITY[slot]).sqrt();
        SpawnRegion {
            spacing: liquid_spacing * stretch,
            box_size: IVec2::new(
                (COLUMN.x as f32 * stretch).round() as i32,
                (COLUMN.y as f32 * stretch).round() as i32,
            ),
            box_center: Vec2::new(COLUMN_X[slot], FLOOR + COLUMN.y as f32 * stretch * 0.5),
            // The other half of the same pre-initialization: the spacing
            // above puts the GRID at the mixture's density, this puts the
            // particle's own bookkeeping there too. Without it every
            // particle starts carrying the liquid's density at the
            // mixture's spacing, which is a pressure shock, not a scene.
            initial_deformation_gradient: Mat2::from_diagonal(Vec2::splat(stretch)),
            material_id: 0,
            mass_override: Some(particle_mass),
            precompute_initial_volumes: false,
            initial_velocity_scale: 0.0,
            ..SpawnRegion::for_sim(&config)
        }
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
            particles.friction_hardening[i] = QUALITY[slot_of(i)];
            particles.temperature[i] = BOILING_POINT_K;
        }
    }

    // Density in SI: the engine carries it per unit cell area, so one cell
    // of area dx^2 holds `density * dx^2` of real mass.
    let si_density = |grid_density: f32| grid_density / (DX_M * DX_M);
    let column = |sim: &Simulation, slot: usize| -> (usize, f32, f32, f32, f32) {
        let (mut n, mut mass, mut rho) = (0usize, 0.0, 0.0);
        let (mut left, mut right, mut top) = (f32::MAX, f32::MIN, f32::MIN);
        for (i, p) in sim.particles().iter().enumerate() {
            if slot_of(i) == slot {
                n += 1;
                mass += p.mass;
                rho += si_density(p.density);
                left = left.min(p.x.x);
                right = right.max(p.x.x);
                top = top.max(p.x.y);
            }
        }
        (n, mass, rho / n as f32, right - left, top)
    };

    let frames = (seconds / dt).round() as usize;
    let (mut substeps, mut worst) = (0usize, 0usize);
    let wall = std::time::Instant::now();
    for frame in 0..frames {
        sim.step();
        let s = sim.diagnostics_snapshot();
        substeps += s.substeps_last_step;
        worst = worst.max(s.substeps_last_step);
        if frame % (frames / 4).max(1) == 0 {
            let fastest = sim
                .particles()
                .iter()
                .map(|p| p.v.length())
                .fold(0.0_f32, f32::max);
            let (_, _, rho, spread, _) = column(&sim, 2);
            println!(
                "    t={:.2}s  x=0.20 column: {rho:.1} kg/m3, {spread:.1} cells across, fastest {:.2} m/s",
                frame as f32 * dt,
                fastest * DX_M
            );
        }
    }
    let ms = wall.elapsed().as_secs_f64() * 1000.0 / frames as f64;
    println!(
        "c_l={c_l} dt={dt}: {:.1} substeps/frame (worst {worst}), {ms:.2} ms/frame, {:.0} fps, {total} particles",
        substeps as f64 / frames as f64,
        1000.0 / ms
    );
    println!(
        "  real time: {:.3} s of water per second of wall clock, {:.0} times slower than life",
        dt as f64 / (ms / 1000.0),
        (ms / 1000.0) / dt as f64
    );
    for (slot, x) in QUALITY.into_iter().enumerate() {
        let (n, mass, rho, spread, _) = column(&sim, slot);
        let expected = material.rho_eq_kg_m3(x);
        println!(
            "  x={x:.2}: {n} particles, {:.3} kg, measured {rho:7.1} kg/m3 against the rule's {expected:7.1} ({:+.2} %), c_mix {:5.1} m/s, {spread:.1} cells across",
            mass / (DX_M * DX_M) * DX_M * DX_M,
            100.0 * (rho / expected - 1.0),
            material.c_mix2_m2_s2(x).sqrt()
        );
    }
}

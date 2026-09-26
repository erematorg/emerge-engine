//! The slump scene, shared by `basic_bingham` and `bingham_slump_probe`.
//!
//! One file for both so the table the demo's header publishes is measured
//! on the demo's own scene, and the yield-stress reading the demo shows is
//! the one the probe prints. They used to be two copies, and an earlier
//! version of that table was never produced by the scene at all: it
//! described a geometry the demo had stopped using.
//!
//! Every item here is used by both includers; one that is not would be
//! dead code in that example.

use crate::emerge::{
    BinghamFluidMaterial, BinghamProps, FrictionBoundary, FromSI, SimConfig, Simulation,
    SpawnRegion,
};
use glam::{IVec2, Mat2, Vec2};

/// 160 cells, 32 cm: wide enough that the three deposits never meet.
/// At 64 cells they did, and each column's shape was set by its neighbours
/// rather than its yield stress. The width is derived from how far each
/// column spreads ALONE on this scene's gripping floor
/// (`tests/scratch_bingham_isolated_slump.rs`): see `COLUMN_X`. A wider
/// tank costs nothing measurable, since the grid is sparse
/// (`BINGHAM_PROBE_GRID` in `bingham_cost_probe`), and the camera frames
/// the material rather than the tank.
pub const GRID: usize = 160;

/// 2 mm cells -- a tabletop tank, the scale a slump of a few
/// centilitres actually happens at. Deposit height scales as tau_0/(rho g),
/// millimetres for these materials, so a metre-scale domain would collapse
/// the whole effect into a single cell.
pub const DX_M: f32 = 0.002;

/// Real yield stresses in pascals, spanning the three bands
/// `BinghamFluidMaterial`'s own doc lists (biological 1-50, mud 50-500,
/// lava 100-2000). Everything else about the three columns is identical.
pub const YIELD_STRESS_PA: [f32; 3] = [2.0, 60.0, 1200.0];
/// Position, not a baked-in yield stress: the panel's slider rescales all
/// three, so a label naming a pascal value would go stale the moment it
/// moves. The `in=` field in each readout carries the live value.
pub const COLUMN_LABEL: [&str; 3] = ["left", "mid", "right"];
/// Column centres, in cells, placed from the measured spreads so the
/// deposits cannot meet. Alone on the gripping floor they settle to half
/// widths of about 96 mm for 2 Pa (48 cells, still creeping at 0.5 mm/s
/// after eight seconds, so 50 are allowed), 26.3 mm for 60 Pa (13.2 cells)
/// and 9.5 mm for 1200 Pa (4.8 cells). Laid left to right with 5 cells,
/// 10 mm, between each deposit and the next and from each wall: 2 Pa spans
/// 7 to 107, 60 Pa 112 to 138, 1200 Pa 143 to 153, inside walls at 2 and
/// 158. The margins are a declared choice, not a measurement.
pub const COLUMN_X: [f32; 3] = [57.0, 125.0, 148.0];

/// Water-based suspensions, so the density is water's.
pub const RHO_KG_M3: f32 = 1000.0;

/// Plastic (post-yield) viscosity, identical for all three columns -- this
/// scene varies exactly one parameter, and this is not it. Inside the
/// 0.1-5 Pa.s band the same doc gives for wet clay.
pub const ETA_PA_S: f32 = 0.5;

/// Yield strain, tau_0/G: how far a yield-stress fluid can be sheared
/// before it starts to flow. Real ones measure in the 1-10% band pretty
/// much regardless of what they are, so 5% is a stated property of the
/// material class rather than three unrelated numbers picked per column.
/// It is what ties each column's storage modulus to its own tau_0, so
/// tau_0 stays the single independent variable of the scene.
pub const YIELD_STRAIN: f32 = 0.05;

/// 20 mm across for 40 tall. The scene used to stand these columns at
/// 8 mm across, an aspect ratio of 5 to 1, and the stiffest one did not
/// demonstrate a yield stress at all: it stood while the soft ones spread,
/// then TOPPLED, and the fall generated the stress that made it flow.
/// Measured (`bingham_slump_probe`): its standing shear sat at 0.59 of its
/// own yield, crossed 1.05 at the instant it fell, and it ended flatter
/// than the column with a twentieth of its yield stress. At 2 to 1 it
/// stays where it is put, which is the behaviour this scene is about.
pub const COLUMN_CELLS: IVec2 = IVec2::new(10, 20);
/// Particle lattice spacing, in cells. Named because the cursor needs it
/// too: it fixes each particle's mass per unit depth, and with it the force
/// a push in pascals has to apply.
pub const SPACING: f32 = 0.5;

pub const FLOOR_CELLS: f32 = 2.0;

/// Weakly-compressible sound-speed derating (Monaghan 1994): resolving
/// water's real 1483 m/s would cost roughly 15000 substeps per frame, so
/// the reference sound speed is 10x the fastest flow speed this scene can
/// produce, which holds density fluctuation under 1%. `v_max` comes from
/// free fall over the column's own height, not from a tuned number, and the
/// bulk modulus is then the definition `K = rho c^2`.
pub fn bulk_modulus_pa() -> f32 {
    let column_height_m = COLUMN_CELLS.y as f32 * DX_M;
    let v_max = (2.0 * 9.81 * column_height_m).sqrt();
    let c_ref = 10.0 * v_max;
    RHO_KG_M3 * c_ref * c_ref
}

pub fn make_config(gravity_fraction: f32, dt: f32) -> SimConfig {
    let mut config = SimConfig {
        // The acoustic CFL bound here is ~2e-4 s, below the 1e-3 s default
        // floor; leaving the default would clamp the substep above its own
        // stability limit.
        min_dt: 1.0e-5,
        // Real arithmetic, not a knob turned until it stopped complaining.
        // The acoustic CFL bound at this sound speed is ~74 us, so a 5 ms
        // frame genuinely needs ~68 substeps, and the 64 default is a
        // budget rather than a physics cap (see its own doc). 256 leaves
        // room for the compression transient at first contact, where the
        // Tait EOS raises the local sound speed above its rest value.
        max_substeps_per_step: 256,
        ..SimConfig::earth(GRID, DX_M, dt)
    };
    config.gravity *= gravity_fraction;
    config
}

/// One column's material, built through the SI property route so every
/// number entered is a real pascal; the three columns differ only in
/// `tau0_pa`.
pub fn column_material(
    tau0_pa: f32,
    yield_scale: f32,
    config: &SimConfig,
) -> (BinghamFluidMaterial, BinghamProps) {
    let props = BinghamProps {
        rho_kg_m3: RHO_KG_M3,
        eta_pa_s: ETA_PA_S,
        bulk_modulus_pa: bulk_modulus_pa(),
        yield_stress_pa: tau0_pa * yield_scale,
        // The storage modulus below the yield point. Without it this
        // material computes its deviatoric stress purely from the CURRENT
        // rate of strain, so at rest it has none and all three columns
        // collapse into identical puddles -- the model's own limitation,
        // not a bug, confirmed against the same model in `tmp/GeoTaichi`.
        // Holding a shape needs stored elastic shear energy; this is it.
        shear_modulus_pa: tau0_pa * yield_scale / YIELD_STRAIN,
        cavitation_pressure_pa: BinghamProps::air_entrained_cavitation_pressure(),
    };
    let mut m = BinghamFluidMaterial::from_physical(&props, config);
    // Measured coefficients, not a substance claim: the continuous phase of
    // all three is water, so all three get water's measured absorption
    // (Pope & Fry 1997). The suspended solids have their own spectrum this
    // engine holds no measurement for; that stays a named gap rather than an
    // invented tint. Identical across the three, so nothing distinguishes
    // them visually except how they move.
    m.optics = Some(crate::emerge::materials::optical::pure_water());
    m.specific_heat_j_kg_k = 4182.0; // water, CRC Handbook
    (m, props)
}

/// All three columns. The materials come back too: their grid-unit
/// `yield_stress` is what the shear colour scale normalizes by.
pub fn make_sim(
    gravity_fraction: f32,
    yield_scale: f32,
    dt: f32,
) -> (Simulation, [BinghamFluidMaterial; 3]) {
    let config = make_config(gravity_fraction, dt);
    let build = |tau0_pa: f32| column_material(tau0_pa, yield_scale, &config);

    let spawn = |slot: usize, props: &BinghamProps| {
        SpawnRegion {
            spacing: SPACING,
            box_size: COLUMN_CELLS,
            box_center: Vec2::new(COLUMN_X[slot], FLOOR_CELLS + COLUMN_CELLS.y as f32 * 0.5),
            material_id: slot as u32,
            initial_velocity_scale: 0.0,
            ..SpawnRegion::for_sim(&config)
        }
        // Real density -> real particle mass, rather than a hand-picked one.
        .mass_from(props, &config)
    };

    let (m0, p0) = build(YIELD_STRESS_PA[0]);
    let (m1, p1) = build(YIELD_STRESS_PA[1]);
    let (m2, p2) = build(YIELD_STRESS_PA[2]);

    let mut sim = Simulation::new(config, spawn(0, &p0))
        .with_default_material(Box::new(m0))
        .with_material(1, Box::new(m1))
        .with_material(2, Box::new(m2))
        // A floor the material grips, not a frictionless one. The slump
        // test is made on a substrate the material does not slip on:
        // Woodbridge, Fonte and Juel (arXiv 2609.12229, 2026) impose
        // "no-slip and no-penetration conditions" at it, and measure the
        // same fluids between "cross-hatched parallel plates to minimise
        // wall slip". Measured here, a frictionless floor makes the
        // thin-layer inversion read five to ten times low. Coulomb friction
        // at mu = 1 grips where mu * rho * g * h exceeds the base shear;
        // while the deposits spread, 7 to 9 percent of the loaded floor
        // nodes still slide (`tests/scratch_bingham_isolated_slump.rs`). A
        // declared approximation of no-slip, not no-slip.
        .with_boundary(Box::new(FrictionBoundary::new(
            config.boundary_thickness,
            1.0,
        )));
    let _ = sim.add_body(spawn(1, &p1));
    let _ = sim.add_body(spawn(2, &p2));
    (sim, [m0, m1, m2])
}

/// A settled deposit's shape: height above the floor and half its width,
/// in metres, and its fastest particle in m/s.
pub fn deposit(sim: &Simulation, slot: u32) -> Option<(f32, f32, f32)> {
    let (mut top, mut lo, mut hi, mut speed, mut n) = (f32::MIN, f32::MAX, f32::MIN, 0.0f32, 0u32);
    for p in sim.particles().iter().filter(|p| p.material_id == slot) {
        top = top.max(p.x.y);
        lo = lo.min(p.x.x);
        hi = hi.max(p.x.x);
        speed = speed.max(p.v.length());
        n += 1;
    }
    (n > 0).then(|| {
        (
            (top - FLOOR_CELLS).max(0.0) * DX_M,
            ((hi - lo) * 0.5).max(1.0e-6) * DX_M,
            speed * DX_M,
        )
    })
}

/// What a deposit says about its own yield stress, and only where the
/// relation used to say it is valid. The relation is chosen from what the
/// deposit LOOKS like, never from the yield stress that went in: choosing
/// by the answer would make the reading circular.
pub enum YieldReading {
    /// No gravity, so nothing loads the column.
    Unloaded,
    /// Still slumping: the shape is far from final, so it says nothing yet.
    Slumping,
    /// Did not slump, so its yield stress exceeds what this height can
    /// reveal: a lower bound.
    AtLeast(f32),
    /// Thin deposit, read by the long-wave force balance.
    ThinLayer(f32),
    /// Slumped but not thin, read by the fitted planar slump law.
    Planar(f32),
    /// No relation here holds for this shape.
    NotApplicable,
}

/// Initial half-width and height of every column, metres.
pub const COLUMN_HALF_WIDTH_M: f32 = COLUMN_CELLS.x as f32 * 0.5 * DX_M;
pub const COLUMN_HEIGHT_M: f32 = COLUMN_CELLS.y as f32 * DX_M;
/// Staron, Lagree, Ray and Popinet, "Scaling laws for the slumping of a
/// Bingham plastic fluid", J. Rheol. 57, 1265 (2013): 2D planar columns on a
/// no-slip base, simulated. Their eq. 13 fits the final height as
/// `H / R0 = 3.01 * sigma^0.66` (exponent +/- 0.03, correlation 0.95), with
/// `sigma = tau_0 / (rho g R0)` and `R0` the initial half-width, over
/// `0.06 <= sigma <= 1.6`; their eq. 14 says a column slumps only when
/// `H0 / R0` exceeds `3.01 * sigma^0.66`.
pub const STARON_PREFACTOR: f32 = 3.01;
pub const STARON_EXPONENT: f32 = 0.66;
pub const STARON_SIGMA_RANGE: std::ops::RangeInclusive<f32> = 0.06..=1.6;

/// A reading and how final it is. A deposit measures its yield stress only
/// at rest; one that still moves gets its reading shown as provisional,
/// with the speed it moves at, rather than hidden until it stops, because
/// a soft deposit can creep for longer than anyone watches.
pub struct Reading {
    pub what: YieldReading,
    /// `Some(speed)` while the fastest particle still moves, metres per
    /// second; `None` once the deposit is at rest.
    pub moving_m_s: Option<f32>,
}

/// Declared: slower than a tenth of a millimetre a second, a deposit is at
/// rest. The deposits that do stop ring down through a few hundredths of a
/// millimetre a second within two and a half seconds
/// (`tests/scratch_bingham_isolated_slump.rs`), well under it.
pub const AT_REST_M_S: f32 = 1.0e-4;
/// Declared: faster than a centimetre a second, five cells a second at this
/// grid, a deposit is still slumping and its shape is no reading at all.
pub const SLUMPING_M_S: f32 = 1.0e-2;

/// Follows one deposit through its slump, so a reading knows whether the
/// collapse is over. Speed alone cannot tell: a column just released moves
/// slowly too, 9.8 mm/s after its first millisecond of fall, but it is
/// speeding up, while a deposit that creeps has slowed down from its
/// fastest. Reset it with the scene.
#[derive(Clone, Copy, Default)]
pub struct SlumpWatch {
    peak_m_s: f32,
}

impl SlumpWatch {
    /// Takes this frame's deposit, from `deposit`, and reads it.
    pub fn read(&mut self, h_m: f32, half_width_m: f32, speed_m_s: f32, g: f32) -> Reading {
        self.peak_m_s = self.peak_m_s.max(speed_m_s);
        let moving = speed_m_s > AT_REST_M_S;
        let what = if speed_m_s > SLUMPING_M_S || (moving && speed_m_s >= self.peak_m_s) {
            YieldReading::Slumping
        } else {
            read_shape(h_m, half_width_m, g)
        };
        Reading {
            what,
            moving_m_s: moving.then_some(speed_m_s),
        }
    }
}

fn read_shape(h_m: f32, half_width_m: f32, g: f32) -> YieldReading {
    if g <= 0.0 {
        return YieldReading::Unloaded;
    }
    let scale = RHO_KG_M3 * g * COLUMN_HALF_WIDTH_M;
    // Declared: within five percent of its starting height, a column has
    // not slumped.
    if h_m >= 0.95 * COLUMN_HEIGHT_M {
        // No slump means H0/R0 <= 3.01 sigma^0.66, so sigma is at least
        // (H0/R0 / 3.01)^(1/0.66). Only trusted where that sigma is inside
        // the range the law was fitted over.
        let sigma =
            (COLUMN_HEIGHT_M / COLUMN_HALF_WIDTH_M / STARON_PREFACTOR).powf(1.0 / STARON_EXPONENT);
        return if STARON_SIGMA_RANGE.contains(&sigma) {
            YieldReading::AtLeast(sigma * scale)
        } else {
            YieldReading::NotApplicable
        };
    }
    // Declared: h/L at most 0.1 is thin enough for the long-wave balance.
    if h_m / half_width_m <= 0.1 {
        // A thin deposit at rest has the yield stress at its base holding
        // up the hydrostatic pressure gradient, tau_0 = rho g h |dh/dx|.
        // Integrated from the front, where h = 0, to the centre, where it
        // is h over a half-width L: h^2 = 2 tau_0 L / (rho g). It holds as
        // h/L goes to zero and nowhere else.
        return YieldReading::ThinLayer(RHO_KG_M3 * g * h_m * h_m / (2.0 * half_width_m));
    }
    let sigma = (h_m / (STARON_PREFACTOR * COLUMN_HALF_WIDTH_M)).powf(1.0 / STARON_EXPONENT);
    if STARON_SIGMA_RANGE.contains(&sigma) {
        YieldReading::Planar(sigma * scale)
    } else {
        YieldReading::NotApplicable
    }
}

pub fn describe(reading: &Reading) -> String {
    let what = match reading.what {
        YieldReading::Unloaded => "no gravity, no reading".to_string(),
        YieldReading::Slumping => return "slumping".to_string(),
        YieldReading::AtLeast(pa) => format!("held its shape: tau_0 > {pa:.0} Pa"),
        YieldReading::ThinLayer(pa) => format!("thin layer: tau_0 = {pa:.1} Pa"),
        YieldReading::Planar(pa) => format!("slumped: tau_0 = {pa:.0} Pa (fitted law)"),
        YieldReading::NotApplicable => "no valid reading for this shape".to_string(),
    };
    match reading.moving_m_s {
        Some(speed) => format!(
            "{what}, provisional: still moving at {:.2} mm/s",
            speed * 1000.0
        ),
        None => what,
    }
}

/// The shear a yield criterion tests: the second invariant of the deviatoric
/// stress, `sqrt(tau_dev : tau_dev / 2)`, the pressure taken out. At rest a
/// yield-stress fluid holds it up to `tau_0` and no further. It is the
/// measure `BinghamFluidMaterial`'s own criterion is stated in (see its
/// `yield_in_frobenius_measure`), so divided by a material's `yield_stress`
/// it reads 1 exactly where that material starts to flow.
pub fn deviatoric_shear(tau: Mat2) -> f32 {
    let mean = 0.5 * (tau.x_axis.x + tau.y_axis.y);
    let dev = Mat2::from_cols(
        Vec2::new(tau.x_axis.x - mean, tau.x_axis.y),
        Vec2::new(tau.y_axis.x, tau.y_axis.y - mean),
    );
    (0.5 * (dev.x_axis.length_squared() + dev.y_axis.length_squared())).sqrt()
}

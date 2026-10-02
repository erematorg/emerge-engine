//! The clay scene, shared by `basic_vonmises` and its headless probe
//! (`tests/probes/vonmises_clay.rs`), so what the demo's header says is
//! measured on the demo's own scene. Every item here is used by both
//! includers.

use crate::emerge::{
    DuctileProps, Elastic, FromSI, SimConfig, Simulation, SlipBoundary, SpawnRegion,
    VonMisesMaterial,
};
use glam::{IVec2, Vec2};

pub const GRID: usize = 64;
pub const DT: f32 = 1.0 / 60.0;
pub const DX_M: f32 = 0.01;
/// Blob side in cells, and the height of its centre above the floor's
/// cells at spawn.
pub const BLOB_CELLS: i32 = 14;
pub const BLOB_CENTRE_Y: f32 = 20.0;

pub const MAT_VERY_SOFT: u32 = 0;
pub const MAT_SOFT: u32 = 1;
pub const MAT_MEDIUM: u32 = 2;

/// Unconfined compressive strength `q_u` [Pa] of the three clays, the
/// middle of each consistency band. FHWA NHI-06-088 (Soils and Foundations
/// Reference Manual, vol. I, 2006), Table 4-2 after Peck et al. 1974: very
/// soft under 25 kPa, soft 25 to 50, medium stiff 50 to 100. NAVFAC DM 7.01
/// (Soil Mechanics, 1986), consistency table: under 0.25, 0.25 to 0.50, 0.50
/// to 1.00 tsf. `q_u` is the uniaxial yield stress of the undrained clay,
/// twice its undrained shear strength `c`.
pub const CLAYS: [(&str, f32); 3] = [("very soft", 20.0e3), ("soft", 37.5e3), ("medium", 75.0e3)];
/// Undrained modulus over undrained shear strength, NAVFAC DM 7.01 chapter
/// 5, Table 2 (after Duncan and Buchignani): 300 for an overconsolidation
/// ratio under 3 and a plasticity index between 30 and 50. It agrees with
/// FHWA NHI-06-088 Table 5-16 (after AASHTO 2004): soft sensitive clay 25 to
/// 150 tsf (2.4 to 14 MPa), medium stiff to stiff 150 to 500 tsf.
pub const MODULUS_OVER_STRENGTH: f32 = 300.0;
/// Undrained Poisson's ratio, inside FHWA NHI-06-088 Table 5-16's 0.4 to
/// 0.5. Exactly 0.5 (incompressible) has no explicit stable step.
pub const CLAY_POISSON: f32 = 0.45;
/// Saturated clay, FHWA NHI-06-088 Table 5-4 after Kulhawy and Mayne 1990:
/// 1.51 to 2.13 times water. One source only.
pub const CLAY_DENSITY_KG_M3: f32 = 1700.0;

pub fn clay(unconfined_strength_pa: f32) -> DuctileProps {
    DuctileProps {
        elastic: Elastic {
            e_pa: MODULUS_OVER_STRENGTH * unconfined_strength_pa / 2.0,
            nu: CLAY_POISSON,
            rho_kg_m3: CLAY_DENSITY_KG_M3,
        },
        yield_stress_pa: unconfined_strength_pa,
    }
}

/// The scene, and its three materials in slot order: the stress view reads
/// each particle against its own material's yield surface.
///
/// Before building it, checks that every clay holds its own weight: a block
/// of height `h` resting on a floor carries `rho g h` at its base, a
/// deviatoric stress of `rho g h / sqrt(2)` in the Frobenius measure the
/// material tests, against `sqrt(2/3) q_u` (`VonMisesMaterial::from_physical`).
pub fn make_sim(gravity_fraction: f32) -> (Simulation, [VonMisesMaterial; 3]) {
    let mut config = SimConfig::earth(GRID, DX_M, DT);
    config.gravity *= gravity_fraction;
    let g = config.gravity.length() * DX_M;
    let blob_height_m = BLOB_CELLS as f32 * DX_M;
    for (name, strength) in CLAYS {
        let weight = CLAY_DENSITY_KG_M3 * g * blob_height_m / std::f32::consts::SQRT_2;
        let threshold = (2.0f32 / 3.0).sqrt() * strength;
        assert!(
            threshold > weight,
            "{name} clay cannot hold its own weight: {weight:.0} Pa against {threshold:.0}"
        );
    }

    // The substeps a frame needs, from the stiffest clay's pressure wave
    // (`material_cfl_coefficient`) or the landing speed (`cfl_coefficient`),
    // whichever bounds the step tighter: no cap below what physics asks for.
    // The engine reads the wave speed with each particle's measured density
    // (`VonMisesMaterial::timestep_bound`), which at a convex corner of a
    // body, where the kernel sees a quarter of material, falls to about a
    // quarter of rest: twice the wave speed. With the rest density alone,
    // 18 percent of each second was dropped at the cap.
    let (_, stiffest) = CLAYS[2];
    let props = clay(stiffest).elastic;
    let p_modulus = props.e_pa * (1.0 - props.nu) / ((1.0 + props.nu) * (1.0 - 2.0 * props.nu));
    let corner_density = props.rho_kg_m3 / 4.0;
    let wave_m_s = (p_modulus / corner_density).sqrt();
    let drop_m =
        (BLOB_CENTRE_Y - BLOB_CELLS as f32 * 0.5 - config.boundary_thickness as f32) * DX_M;
    let landing_m_s = (2.0 * g * drop_m.max(0.0)).sqrt();
    let step = (config.material_cfl_coefficient * DX_M / wave_m_s)
        .min(config.cfl_coefficient * DX_M / landing_m_s.max(f32::MIN_POSITIVE));
    config.max_substeps_per_step = (DT / step).ceil() as usize;

    let spawn = |c: Vec2, mat: u32| {
        SpawnRegion {
            spacing: 0.5,
            box_size: IVec2::new(BLOB_CELLS, BLOB_CELLS),
            box_center: c,
            material_id: mat,
            initial_velocity_scale: 0.0,
            ..SpawnRegion::for_sim(&config)
        }
        .mass_from(&clay(CLAYS[mat as usize].1).elastic, &config)
    };
    let [very_soft, soft, medium] =
        CLAYS.map(|(_, strength)| VonMisesMaterial::from_physical(&clay(strength), &config));

    let mut solver = Simulation::new(config, spawn(Vec2::new(14.0, BLOB_CENTRE_Y), MAT_VERY_SOFT))
        .with_default_material(Box::new(very_soft))
        .with_material(MAT_SOFT, Box::new(soft))
        .with_material(MAT_MEDIUM, Box::new(medium))
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));
    let _ = solver.add_body(spawn(Vec2::new(32.0, BLOB_CENTRE_Y), MAT_SOFT));
    let _ = solver.add_body(spawn(Vec2::new(50.0, BLOB_CENTRE_Y), MAT_MEDIUM));
    (solver, [very_soft, soft, medium])
}

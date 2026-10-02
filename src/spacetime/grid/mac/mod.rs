//! Pressure projection on a staggered (MAC) grid, the standard formulation,
//! built apart from `Simulation::step` so it can be judged on its own first.
//!
//! Experimental (feature `experimental`), and nothing in the engine calls
//! it. The fluid solver in use stays the weakly compressible one;
//! `Grid::project_fluid_incompressibility` is the older projection this
//! replaces, and is removed, once it passes its gates (`gates.rs`, criteria
//! written before the code). Two gate runs so far: the column at rest and
//! the falling droplet pass, the dam break and the drop into a pool do not;
//! `KNOWN_LIMITATIONS.md` has what remains.
//!
//! Sources, read on the documents themselves:
//!
//! - Bridson and Muller-Fischer, *Fluid Simulation*, SIGGRAPH 2007 course
//!   notes: the MAC grid and its discrete gradient and divergence (4.1,
//!   4.2), the pressure equations (4.3), MIC(0) preconditioned conjugate
//!   gradient (4.3.2 to 4.3.4, figs. 4.1 to 4.3), the ghost-fluid free
//!   surface (4.5.1, eqs. 4.35 to 4.37), the variational solid boundary
//!   (4.5.2) and velocity extrapolation (6.3). Their 4.16 is why the grid is
//!   staggered: a collocated central difference reads a checkerboard
//!   velocity field as divergence free.
//! - Batty, Bertails and Bridson, *A Fast Variational Framework for
//!   Accurate Solid-Fluid Coupling*, SIGGRAPH 2007: face weights from the
//!   solid's real position.
//! - Zhu and Bridson, *Animating Sand as a Fluid*: the liquid surface from
//!   particles (section 5, eqs. 6 to 10), particle seeding (4.2.1) and at
//!   most one cell of travel per substep (4.2.5).
//! - `apic2d` (Bridson's variational solve with APIC transfers on a MAC
//!   grid, `tmp/apic2d`) as the reference implementation.
//!
//! Coordinates are the engine's grid coordinates. Cell `(i, j)` spans
//! `[i dx, (i + 1) dx] x [j dx, (j + 1) dx]`; pressure and the liquid level
//! set sit at cell centres, `u` at `(i dx, (j + 1/2) dx)`, `v` at
//! `((i + 1/2) dx, j dx)`. Pressure is kinematic, `p / rho`, as in
//! `apic2d`: one fluid of uniform density needs no density in the solve.

pub mod extrapolate;
pub mod field;
pub mod level_set;
pub mod pcg;
pub mod pressure;
pub mod solid;
pub mod transfer;

#[cfg(test)]
mod gates;

pub use field::{FaceFlags, Field2, MacLayout, MacVelocity};
pub use level_set::SurfaceSettings;
pub use pcg::{Solution, SolverSettings};
pub use solid::FaceWeights;

/// Every tolerance the projection uses, named, with its reason.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProjectionSettings {
    /// Smallest fraction of the line between two cell centres at which the
    /// free surface may cut it (ghost fluid, eq. 4.35). Below it `1 /
    /// theta` would swamp its row; 0.01 is `apic2d`'s floor, the notes
    /// suggest 1e-6. A declared approximation: a surface closer than this
    /// to a liquid cell's centre is read as this close.
    pub theta_floor: f32,
    pub solver: SolverSettings,
    /// Rings of faces the projected velocity is extended over before the
    /// gather. Two, as `apic2d`: the quadratic kernel reaches one and a
    /// half cells from a particle.
    pub extrapolation_layers: u32,
    /// Largest distance, in cells, a particle may travel in one substep
    /// (Zhu and Bridson 4.2.5).
    pub max_cells_per_substep: f32,
}

impl Default for ProjectionSettings {
    fn default() -> Self {
        Self {
            theta_floor: 0.01,
            solver: SolverSettings::default(),
            extrapolation_layers: 2,
            max_cells_per_substep: 1.0,
        }
    }
}

/// Longest substep in which a particle at `max_speed` that gravity keeps
/// accelerating travels at most `max_cells` cells: the positive root of
/// `g dt^2 + v dt = max_cells dx`.
pub fn travel_limited_dt(max_speed: f32, gravity: f32, dx: f32, max_cells: f32) -> f32 {
    let reach = max_cells * dx;
    let (v, g) = (max_speed.abs(), gravity.abs());
    if g == 0.0 {
        return if v > 0.0 { reach / v } else { f32::INFINITY };
    }
    (-v + (v * v + 4.0 * g * reach).sqrt()) / (2.0 * g)
}

/// One projection of the face velocity: equations, solve, update, then
/// the velocity extended past the liquid and held along the walls. The
/// level set must already have the solid folded in (`pressure` module
/// doc).
pub fn project(
    layout: &MacLayout,
    dt: f32,
    vel: &mut MacVelocity,
    weights: &FaceWeights,
    solid_corners: &Field2,
    liquid_phi: &Field2,
    settings: &ProjectionSettings,
) -> Solution {
    let system = pressure::assemble(layout, dt, vel, weights, liquid_phi, settings.theta_floor);
    let solution = pcg::solve(&system, &settings.solver);
    let mut valid = pressure::apply_pressure(
        layout,
        dt,
        &solution.pressure,
        weights,
        liquid_phi,
        settings.theta_floor,
        vel,
    );
    extrapolate::extrapolate_velocity(vel, &mut valid, settings.extrapolation_layers);
    extrapolate::constrain_to_solids(layout, vel, weights, solid_corners);
    solution
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_substep_keeps_travel_within_its_limit() {
        for &(v, g) in &[(0.0f32, 981.0f32), (300.0, 981.0), (50.0, 0.0), (0.0, 0.0)] {
            let dt = travel_limited_dt(v, g, 1.0, 1.0);
            if dt.is_finite() {
                assert!((v + g * dt) * dt <= 1.0 + 1e-4, "v={v} g={g} dt={dt}");
            }
        }
        // At rest under gravity the limit is the fall of one cell.
        let dt = travel_limited_dt(0.0, 981.0, 1.0, 1.0);
        assert!((981.0 * dt * dt - 1.0).abs() < 1e-4);
    }
}

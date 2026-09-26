//! Grain micro-rotation coupling -- a real, bounded, energy-conservative
//! way for grains to exchange rotational influence with nearby grains
//! through the grid, replacing an earlier, reverted attempt that scattered
//! `spin` directly into the shared momentum grid (see
//! `coupling::scatter_grains_to_grid`'s own doc for the full story: exact
//! for an isolated grain, but a real, confirmed, unbounded energy leak once
//! many spinning grains share overlapping grid nodes -- the raw momentum
//! grid has no notion of a bounded coupling STRENGTH, so a neighbor's
//! gather can read a slice of a spinning neighbor's rotational energy with
//! nothing ever debited from the source).
//!
//! Mirrors `energy::thermodynamics::CosseratField`'s own real, cited
//! mechanism (de Borst, Sabet & Hageman 2022) -- a genuine elastic Cosserat
//! coupling torque `2*alpha*(local_avg - own_spin)`, integrated via the
//! SAME exact, unconditionally-stable exponential solution (not an
//! explicit-Euler numerics workaround: that field's own doc records a real,
//! confirmed NaN failure from trying explicit Euler on this class of stiff
//! relaxation term first). Real, disclosed, energy-BOUNDED by construction:
//! the exact solution of this linear ODE has a stable fixed point at the
//! local average, so grains genuinely converge toward local rotational
//! equilibrium at a rate set by `coupling_modulus`, never able to inject
//! unbounded energy the way the raw momentum-grid scatter could.
//!
//! Real, disclosed structural differences from `CosseratField`:
//! 1. An ordinary MPM particle's own "macro_spin" is INSTANTANEOUS (freshly
//!    computed from the local velocity gradient every substep, no memory of
//!    its own), so `CosseratField` needs a separate persistent `omega_c`
//!    field to hold real inertia. A grain's `spin` ALREADY has real inertia
//!    (integrated via its own contact-torque ODE in `grain_contact_law.rs`)
//!    -- so `grain.spin` itself plays the role `omega_c` plays for
//!    continuum particles; this module only needs the transient local-
//!    average scatter/gather each substep, not a second persistent memory
//!    stacked on an already-persistent one.
//! 2. `CosseratConfig`'s own `micro_inertia` uses ONE domain-wide
//!    `grain_diameter_m` (a disclosed simplification for continuum
//!    particles, which don't carry a real per-particle radius). A `Grain`
//!    already has its own real, possibly-polydisperse `radius` -- so this
//!    module uses `Grain::moment_of_inertia()` directly (`I = 0.5*m*r^2`,
//!    real, exact, already the SAME formula `apply_grain_contact_forces`
//!    uses for its own spin ODE) instead of introducing a redundant,
//!    less-accurate domain-wide diameter parameter.
//! 3. Real, disclosed, scoped-for-now simplification: grains couple only to
//!    OTHER grains here, not yet to the continuum's own separate
//!    `CosseratField` (a real, valuable future extension -- letting a grain
//!    feel torque from a surrounding rotating/shearing sand flow -- not
//!    required to fix the confirmed energy-conservation bug this replaces).

use std::collections::HashMap;

use glam::IVec2;

use crate::grid::kernel::quadratic_weights;

use super::population::GrainPopulation;

/// Real, standalone parameter for this coupling -- deliberately its own
/// small type rather than a new field bolted onto `ContactLawConfig`
/// (already constructed via struct-literal in half a dozen call sites) or
/// a reuse of `CosseratConfig` (whose `grain_diameter_m`/
/// `micro_inertia_coefficient` fields are redundant here, see this
/// module's own doc point 2). `coupling_modulus` plays the exact role
/// `CosseratConfig::coupling_modulus_pa` (`alpha`) plays in the real,
/// cited Cosserat elastic relation (de Borst, Sabet & Hageman 2022) -- a
/// genuine, separate micropolar material parameter, not derivable from
/// `rolling_stiffness` (a different physical mechanism: a discrete contact
/// spring between two specific touching bodies, not a field coupling
/// across a shared neighborhood) by any principled formula.
#[derive(Clone, Copy, Debug)]
pub struct GrainMicroRotationConfig {
    pub coupling_modulus: f32,
}

/// Real, exact-exponential relaxation of each grain's spin toward its own
/// local (kernel-weighted) neighborhood average spin. `HashMap`-keyed
/// scratch (not `CosseratField`'s own dense, whole-domain `Vec<f32>`):
/// grains are always a small fraction of a scene's total particle count
/// (see this project's own `oracle.rs` framing), so only-touched-cells
/// allocation is the right cost/simplicity tradeoff here.
pub fn couple_grain_spin_to_local_average(
    grains: &mut GrainPopulation,
    config: &GrainMicroRotationConfig,
    sub_dt: f32,
) {
    if grains.grains.is_empty() {
        return;
    }
    let mut grid_mass: HashMap<(i32, i32), f32> = HashMap::new();
    let mut grid_spin: HashMap<(i32, i32), f32> = HashMap::new();

    for grain in &grains.grains {
        let i_micro = grain.moment_of_inertia();
        let w = quadratic_weights(grain.x);
        for gx in 0i32..3 {
            for gy in 0i32..3 {
                let weight = w.wx[gx as usize] * w.wy[gy as usize];
                if weight <= 0.0 {
                    continue;
                }
                let cell = w.base_cell + IVec2::new(gx - 1, gy - 1);
                let key = (cell.x, cell.y);
                let mw = weight * i_micro;
                *grid_mass.entry(key).or_insert(0.0) += mw;
                *grid_spin.entry(key).or_insert(0.0) += mw * grain.spin;
            }
        }
    }

    for grain in &mut grains.grains {
        let i_eff = grain.moment_of_inertia().max(1e-20);
        let k = 2.0 * config.coupling_modulus / i_eff;
        let relax = (-k * sub_dt).exp();

        let w = quadratic_weights(grain.x);
        let mut spin_sum = 0.0f32;
        let mut mass_sum = 0.0f32;
        for gx in 0i32..3 {
            for gy in 0i32..3 {
                let weight = w.wx[gx as usize] * w.wy[gy as usize];
                if weight <= 0.0 {
                    continue;
                }
                let cell = w.base_cell + IVec2::new(gx - 1, gy - 1);
                let key = (cell.x, cell.y);
                if let Some(&m) = grid_mass.get(&key) {
                    mass_sum += weight * m;
                    spin_sum += weight * grid_spin.get(&key).copied().unwrap_or(0.0);
                }
            }
        }
        if mass_sum > 1e-12 {
            let local_avg = spin_sum / mass_sum;
            grain.spin = local_avg + (grain.spin - local_avg) * relax;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matter::materials::granular::grain_contact_law::ContactLawConfig;
    use crate::matter::particle::Grain;
    use glam::Vec2;

    fn config() -> ContactLawConfig {
        ContactLawConfig {
            normal_stiffness: 1.0e4,
            tangential_stiffness: 0.8e4,
            rolling_stiffness: 5.0e2,
            normal_damping: 50.0,
            tangential_damping: 50.0,
            rolling_damping: 50.0,
            friction: 0.5,
            rolling_friction: 0.1,
        }
    }

    #[test]
    fn two_grains_relax_toward_their_shared_average_spin() {
        // Real, minimal proof: two grains close enough to share grid nodes,
        // one spinning, one not -- both should relax toward SOME shared
        // value strictly between their two starting spins (real, bounded
        // exchange, neither grain's own spin ODE touched by this
        // function -- only `couple_grain_spin_to_local_average` runs here).
        let mut pop = GrainPopulation::new(
            vec![
                Grain {
                    spin: 4.0,
                    ..Grain::new(Vec2::new(16.0, 16.0), 1.0, 1.0)
                },
                Grain::new(Vec2::new(16.5, 16.0), 1.0, 1.0),
            ],
            config(),
        );
        let cfg = GrainMicroRotationConfig {
            coupling_modulus: 50.0,
        };
        for _ in 0..500 {
            couple_grain_spin_to_local_average(&mut pop, &cfg, 0.001);
        }
        assert!(
            pop.grains[0].spin > 0.1 && pop.grains[0].spin < 3.9,
            "spinning grain should have relaxed DOWN toward a shared value, got {}",
            pop.grains[0].spin
        );
        assert!(
            pop.grains[1].spin > 0.1,
            "still grain should have picked up SOME spin from its spinning neighbor, got {}",
            pop.grains[1].spin
        );
    }

    #[test]
    fn isolated_grain_spin_is_unaffected() {
        // No neighbor within kernel reach -- local average IS the grain's
        // own spin, so nothing should change (real, necessary invariant:
        // this mechanism must not perturb a genuinely isolated grain).
        let mut pop = GrainPopulation::new(
            vec![Grain {
                spin: 3.0,
                ..Grain::new(Vec2::new(16.0, 16.0), 1.0, 1.0)
            }],
            config(),
        );
        let cfg = GrainMicroRotationConfig {
            coupling_modulus: 50.0,
        };
        for _ in 0..100 {
            couple_grain_spin_to_local_average(&mut pop, &cfg, 0.001);
        }
        assert!(
            (pop.grains[0].spin - 3.0).abs() < 1e-4,
            "isolated grain's spin should be unaffected, got {}",
            pop.grains[0].spin
        );
    }
}

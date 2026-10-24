//! Grain micro-rotation coupling: a bounded, energy-conservative way for
//! grains to exchange rotation with nearby grains through the grid. Spin is
//! not scattered into the shared momentum grid (see
//! `coupling::scatter_grains_to_grid`): that is exact for an isolated grain
//! but leaks energy without bound once many spinning grains share grid
//! nodes, since the momentum grid has no bounded coupling strength and a
//! neighbor's gather can read part of a spinning grain's rotational energy
//! with nothing debited from the source.
//!
//! Mirrors `energy::thermodynamics::CosseratField` (de Borst, Sabet &
//! Hageman 2022): an elastic Cosserat coupling torque `2*alpha*(local_avg -
//! own_spin)`, integrated with the same exact, unconditionally stable
//! exponential solution (explicit Euler produced NaN on this stiff
//! relaxation term, see that field's doc). Energy-bounded by construction:
//! the linear ODE's exact solution has a stable fixed point at the local
//! average, so grains converge toward local rotational equilibrium at a rate
//! set by `coupling_modulus` and cannot inject energy.
//!
//! Differences from `CosseratField`:
//! 1. An MPM particle's "macro_spin" is instantaneous (recomputed from the
//!    local velocity gradient every substep), so `CosseratField` keeps a
//!    persistent `omega_c` field for inertia. A grain's `spin` already has
//!    inertia (its contact-torque ODE in `grain_contact_law.rs`) and plays
//!    the role of `omega_c`; this module only needs the per-substep
//!    local-average scatter/gather.
//! 2. `CosseratConfig::micro_inertia` uses one domain-wide
//!    `grain_diameter_m`, since continuum particles carry no radius. A
//!    `Grain` has its own, possibly polydisperse `radius`, so this module
//!    uses `Grain::moment_of_inertia()` (`I = 0.5*m*r^2`, the formula
//!    `apply_grain_contact_forces` uses).
//! 3. Grains couple only to other grains, not yet to the continuum's
//!    `CosseratField` (which would let a grain feel torque from a
//!    surrounding shearing sand flow).

use std::collections::HashMap;

use glam::IVec2;

use crate::grid::kernel::quadratic_weights;

use super::population::GrainPopulation;

/// Parameters for this coupling, as a separate type rather than a field on
/// `ContactLawConfig` (built by struct literal at several call sites) or a
/// reuse of `CosseratConfig` (whose `grain_diameter_m`/
/// `micro_inertia_coefficient` are redundant here, see point 2 of the
/// module doc). `coupling_modulus` is the `alpha` of the Cosserat elastic
/// relation (`CosseratConfig::coupling_modulus_pa`, de Borst, Sabet &
/// Hageman 2022): a micropolar material parameter with no principled
/// derivation from `rolling_stiffness`, which is a contact spring between
/// two touching bodies, not a field coupling across a neighborhood.
#[derive(Clone, Copy, Debug)]
pub struct GrainMicroRotationConfig {
    pub coupling_modulus: f32,
}

/// Exact-exponential relaxation of each grain's spin toward its local
/// (kernel-weighted) neighborhood average. Scratch is a `HashMap` over the
/// touched cells rather than `CosseratField`'s dense whole-domain
/// `Vec<f32>`: grains are a small fraction of a scene's particles (see
/// `oracle.rs`).
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
        // Two grains close enough to share grid nodes, one spinning, one
        // not: both relax toward a shared value strictly between their
        // starting spins. Only `couple_grain_spin_to_local_average` runs
        // here; neither grain's own spin ODE is touched.
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
        // own spin, so nothing should change (necessary invariant:
        // this mechanism must not perturb a isolated grain).
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

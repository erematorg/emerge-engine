use std::collections::HashMap;

use glam::{IVec2, Vec2};

use super::{FxU32BuildHasher, Grid, flat_index};
use crate::materials::{MAX_MIXTURE_PHASES, MixturePhase};

/// N-phase mixture coupling cell (generalizes Tampubolon et al. 2017's 2-phase
/// Darcy drag -- see `MixturePhase`'s own doc). Only allocated at nodes touched
/// by at least one mixture-phase particle (via `WithMixturePhase`) — a scene
/// that never wraps a material this way never allocates a single one of these.
///
/// `mass`/`momentum` accumulate during P2G exactly like `Cell`'s own fields,
/// but per phase slot, indexed by `MixturePhase::0` (both ADDITIVE alongside
/// the ordinary `Cell` scatter, not a replacement — mirrors `ContactCell`'s own
/// convention). `resolved_v` is filled in by `Grid::resolve_mixture_coupling`
/// (after `update_velocities` + gravity, same pipeline position as
/// `resolve_contact`) and is what G2P reads for a phase's particles at nodes
/// where this cell exists. A phase slot with no real mass at a node just keeps
/// whatever `resolved_v` it last had (never read -- no particle of an absent
/// phase is ever in kernel range of a node it contributed zero mass to).
#[derive(Clone, Copy, Debug)]
pub(super) struct MixtureCell {
    mass: [f32; MAX_MIXTURE_PHASES],
    momentum: [Vec2; MAX_MIXTURE_PHASES],
    resolved_v: [Vec2; MAX_MIXTURE_PHASES],
}

impl Default for MixtureCell {
    fn default() -> Self {
        Self {
            mass: [0.0; MAX_MIXTURE_PHASES],
            momentum: [Vec2::ZERO; MAX_MIXTURE_PHASES],
            resolved_v: [Vec2::ZERO; MAX_MIXTURE_PHASES],
        }
    }
}

pub(super) type MixtureCellMap = HashMap<u32, MixtureCell, FxU32BuildHasher>;

impl Grid {
    /// Accumulate mass and momentum for one mixture phase during P2G, additively
    /// alongside the normal `add_mass_momentum` call for the SAME particle — see
    /// `MixtureCell` doc. OOB cell position silently ignored; an out-of-range
    /// `phase` (`>= MAX_MIXTURE_PHASES`) is a real configuration error and
    /// panics via the array index, same as any other programmer mistake.
    pub fn add_mixture_mass_momentum(
        &mut self,
        cell_pos: IVec2,
        phase: MixturePhase,
        mass: f32,
        momentum: Vec2,
    ) {
        let Some(idx) = flat_index(cell_pos, self.resolution) else {
            return;
        };
        let p = phase.0 as usize;
        match self.mixture_cells.entry(idx) {
            std::collections::hash_map::Entry::Occupied(mut e) => {
                let cell = e.get_mut();
                cell.mass[p] += mass;
                cell.momentum[p] += momentum;
            }
            std::collections::hash_map::Entry::Vacant(e) => {
                self.mixture_dirty.push(idx);
                let mut cell = MixtureCell::default();
                cell.mass[p] = mass;
                cell.momentum[p] = momentum;
                e.insert(cell);
            }
        }
    }

    /// Resolved velocity for one mixture phase at `cell_pos` — valid after
    /// `resolve_mixture_coupling()`. Falls back to the ordinary total velocity
    /// when no mixture coupling was ever registered at this node, same
    /// convention as `grip_velocity_at`.
    pub fn resolved_velocity_at(&self, cell_pos: IVec2, phase: MixturePhase) -> Vec2 {
        let Some(idx) = flat_index(cell_pos, self.resolution) else {
            return Vec2::ZERO;
        };
        self.mixture_cells.get(&idx).map_or_else(
            || self.velocity_at(cell_pos),
            |c| c.resolved_v[phase.0 as usize],
        )
    }

    /// Resolves N-phase mixture coupling (generalizes Tampubolon et al. 2017's
    /// 2-phase Darcy-style momentum exchange to up to `MAX_MIXTURE_PHASES`
    /// simultaneously-present phases at a node) at every mixture-active node —
    /// call after `update_velocities()` (needs the gravity-applied total
    /// field), same pipeline position as `resolve_contact`.
    ///
    /// Exact linear solve, not an iterative approximation: implicit
    /// backward-Euler pairwise drag between every present phase pair, one
    /// shared scalar `drag_coefficient` (k) for every pair -- a real, disclosed
    /// simplification vs. the paper's own permeability/porosity-derived field,
    /// same simplification the original 2-phase code already made. Per real
    /// mixture theory (Truesdell), phase `i`'s momentum balance is:
    ///   m_i (v_i' - v_i)/dt = sum_{j != i} k (v_j' - v_i')
    /// which rearranges into one linear system per velocity COMPONENT (x, y
    /// decouple -- drag is isotropic) over exactly the phases with real mass
    /// present at this node, `M v' = rhs`:
    ///   M_ii = m_i + dt*k*(count of other present phases)
    ///   M_ij = -dt*k                                  (i != j, both present)
    ///   rhs_i = m_i * v_i
    /// `M` is symmetric and strictly diagonally dominant for any real positive
    /// masses/k (each `m_i > 0` on top of the row's other positive terms) --
    /// guaranteed SPD, guaranteed invertible, solved directly via Gaussian
    /// elimination with no pivoting needed (see `solve_present_phases`). With
    /// exactly 2 phases present this reduces to the original closed-form 2x2
    /// solve exactly (`det = 1+a+b` after dividing by `m_s`/`m_f` respectively)
    /// -- checked directly by a regression test, not just asserted. Momentum
    /// is exactly conserved by construction, verified by a real test.
    ///
    /// Fewer than 2 phases present at a node: no correction, every resolved
    /// velocity just reads the ordinary total field, matching
    /// `resolve_contact`'s own "no real second field" fallback.
    ///
    /// `cell_width`/`pressure_iterations` feed `project_mixture_incompressibility`
    /// (see `mixture::pressure`'s own doc, which stays 2-phase-only -- NOT part
    /// of this generalization): the drag solve above conserves momentum but
    /// never enforces the mixture's incompressibility constraint, so under
    /// sustained/confined loading (e.g. water settled into sand) the violation
    /// compounds silently over hundreds of steps until velocities blow past
    /// the CFL bound. `pressure_iterations == 0` skips the projection entirely.
    pub fn resolve_mixture_coupling(
        &mut self,
        dt: f32,
        gravity: Vec2,
        drag_coefficient: f32,
        cell_width: f32,
        pressure_iterations: u32,
    ) {
        const MIN_MASS_FRACTION: f32 = 1.0e-6;
        if drag_coefficient <= 0.0 {
            // Disabled: every phase just reads the ordinary total velocity —
            // matches every other opt-in system's "true default is a no-op".
            for &idx in &self.mixture_dirty {
                let Some(&total) = self.cells.get(&idx) else {
                    continue;
                };
                if let Some(cell) = self.mixture_cells.get_mut(&idx) {
                    cell.resolved_v = [total.momentum; MAX_MIXTURE_PHASES];
                }
            }
            return;
        }
        for &idx in &self.mixture_dirty {
            let Some(&total) = self.cells.get(&idx) else {
                continue;
            };
            let Some(cell) = self.mixture_cells.get(&idx) else {
                continue;
            };

            let mut present = [0usize; MAX_MIXTURE_PHASES];
            let mut n_present = 0usize;
            for p in 0..MAX_MIXTURE_PHASES {
                if cell.mass[p] > MIN_MASS_FRACTION {
                    present[n_present] = p;
                    n_present += 1;
                }
            }

            if n_present < 2 {
                let cell = self.mixture_cells.get_mut(&idx).unwrap();
                cell.resolved_v = [total.momentum; MAX_MIXTURE_PHASES];
                continue;
            }

            let solved =
                solve_present_phases(cell, &present, n_present, dt, gravity, drag_coefficient);
            let cell = self.mixture_cells.get_mut(&idx).unwrap();
            for local in 0..n_present {
                cell.resolved_v[present[local]] = solved[local];
            }
        }
        if pressure_iterations > 0 {
            self.project_mixture_incompressibility(cell_width, pressure_iterations);
        }
    }
}

/// Builds and solves the `n_present`x`n_present` drag-coupling system for one
/// node's present phases (`present[0..n_present]`, global phase indices) --
/// see `resolve_mixture_coupling`'s own doc for the derivation. Returns the
/// resolved velocity for each LOCAL index in `present`.
fn solve_present_phases(
    cell: &MixtureCell,
    present: &[usize; MAX_MIXTURE_PHASES],
    n_present: usize,
    dt: f32,
    gravity: Vec2,
    k: f32,
) -> [Vec2; MAX_MIXTURE_PHASES] {
    let mut v = [Vec2::ZERO; MAX_MIXTURE_PHASES];
    let mut m = [[0.0f32; MAX_MIXTURE_PHASES]; MAX_MIXTURE_PHASES];
    let mut rhs_x = [0.0f32; MAX_MIXTURE_PHASES];
    let mut rhs_y = [0.0f32; MAX_MIXTURE_PHASES];
    for li in 0..n_present {
        let pi = present[li];
        let mi = cell.mass[pi];
        v[li] = cell.momentum[pi] / mi + gravity * dt;
        // Every OTHER present phase pairs with `li` at the same shared
        // coefficient `-dt*k` -- fill the whole row with that first, then
        // overwrite the diagonal (`m_i + dt*k * other-present-count`).
        for entry in m[li].iter_mut().take(n_present) {
            *entry = -dt * k;
        }
        m[li][li] = mi + dt * k * (n_present - 1) as f32;
        rhs_x[li] = mi * v[li].x;
        rhs_y[li] = mi * v[li].y;
    }

    let mut m_y = m;
    gaussian_eliminate(&mut m, &mut rhs_x, n_present);
    gaussian_eliminate(&mut m_y, &mut rhs_y, n_present);

    let mut resolved = [Vec2::ZERO; MAX_MIXTURE_PHASES];
    for local in 0..n_present {
        resolved[local] = Vec2::new(rhs_x[local], rhs_y[local]);
    }
    resolved
}

/// Solves `m * x = rhs` in place (`rhs` becomes `x`) for the top-left `n x n`
/// submatrix, via plain Gaussian elimination -- no pivoting needed since `m`
/// is proven strictly diagonally dominant by construction (see
/// `resolve_mixture_coupling`'s doc), so every pivot encountered is already
/// the largest-magnitude entry available in its column.
fn gaussian_eliminate(
    m: &mut [[f32; MAX_MIXTURE_PHASES]; MAX_MIXTURE_PHASES],
    rhs: &mut [f32; MAX_MIXTURE_PHASES],
    n: usize,
) {
    for i in 0..n {
        let pivot = m[i][i];
        let row_i = m[i];
        for j in (i + 1)..n {
            let factor = m[j][i] / pivot;
            for (dst, src) in m[j][i..n].iter_mut().zip(row_i[i..n].iter()) {
                *dst -= factor * src;
            }
            rhs[j] -= factor * rhs[i];
        }
    }
    for i in (0..n).rev() {
        let mut sum = rhs[i];
        for j in (i + 1)..n {
            sum -= m[i][j] * rhs[j];
        }
        rhs[i] = sum / m[i][i];
    }
}

// The variable-mobility Jacobi pressure projection that enforces the mixture's
// actual incompressibility constraint (`project_mixture_incompressibility`,
// `pub(super)` so `resolve_mixture_coupling` above can call it) -- a distinct
// algorithm (Zhao & Choo 2020 / Bridson Chorin-style projection) from the
// momentum-exchange drag coupling in this file -- lives in pressure.rs, along
// with its own private helpers and its own test. See that file's own doc.
mod pressure;

#[cfg(test)]
mod mixture_coupling_tests {
    use super::*;
    use crate::materials::MixturePhase;

    /// White-box: constructs a single mixture-active node with known solid/fluid
    /// mass+momentum directly (bypassing P2G), so the resolved velocities can be
    /// checked against the exact closed-form solve `resolve_mixture_coupling`'s
    /// own doc derives (backward-Euler drag exchange reduces to a 2x2 linear
    /// system, solved directly) -- not just "runs without crashing."
    fn setup(m_s: f32, v_s0: Vec2, m_f: f32, v_f0: Vec2) -> Grid {
        let mut grid = Grid::new(8);
        let cell_pos = IVec2::new(2, 2);
        // Ordinary total-field scatter too (resolve_mixture_coupling reads
        // `self.cells` for its "no real second field" fallback check via the
        // shared total, though the dual-mass branch below doesn't use it).
        grid.add_mass_momentum(cell_pos, m_s + m_f, m_s * v_s0 + m_f * v_f0);
        grid.add_mixture_mass_momentum(cell_pos, MixturePhase::SOLID, m_s, m_s * v_s0);
        grid.add_mixture_mass_momentum(cell_pos, MixturePhase::FLUID, m_f, m_f * v_f0);
        // Normalize the total field's raw momentum into true velocity, matching the
        // real pipeline's ordering (resolve_mixture_coupling always runs after
        // update_velocities) -- resolve_mixture_coupling's own "no drag"/"no real
        // second field" fallbacks read `Cell.momentum` assuming it's already velocity.
        grid.update_velocities(0.0, Vec2::ZERO);
        grid
    }

    /// Same as `setup`, but with a real third phase (a plain raw-index
    /// `MixturePhase(2)` -- no material wires this slot to anything yet, this
    /// just proves the grid-level N-phase math itself, per Phase 1's own scope).
    fn setup_three_phase(m: [f32; 3], v0: [Vec2; 3]) -> Grid {
        let mut grid = Grid::new(8);
        let cell_pos = IVec2::new(2, 2);
        let total_mass: f32 = m.iter().sum();
        let total_momentum: Vec2 = (0..3).map(|i| m[i] * v0[i]).sum();
        grid.add_mass_momentum(cell_pos, total_mass, total_momentum);
        let phases = [MixturePhase::SOLID, MixturePhase::FLUID, MixturePhase(2)];
        for i in 0..3 {
            grid.add_mixture_mass_momentum(cell_pos, phases[i], m[i], m[i] * v0[i]);
        }
        grid.update_velocities(0.0, Vec2::ZERO);
        grid
    }

    #[test]
    fn resolved_velocities_match_closed_form_2x2_solve() {
        let (m_s, m_f) = (4.0_f32, 1.0_f32);
        let (v_s0, v_f0) = (Vec2::new(0.0, 0.0), Vec2::new(0.0, -2.0));
        let dt = 0.1_f32;
        let k = 3.0_f32;
        let gravity = Vec2::ZERO; // isolate the drag exchange, no extra gravity term

        let mut grid = setup(m_s, v_s0, m_f, v_f0);
        grid.resolve_mixture_coupling(dt, gravity, k, 1.0, 0);

        let a = dt * k / m_s;
        let b = dt * k / m_f;
        let det = 1.0 + a + b;
        let expected_v_s = ((1.0 + b) * v_s0 + a * v_f0) / det;
        let expected_v_f = (b * v_s0 + (1.0 + a) * v_f0) / det;

        let cell_pos = IVec2::new(2, 2);
        let got_v_s = grid.resolved_velocity_at(cell_pos, MixturePhase::SOLID);
        let got_v_f = grid.resolved_velocity_at(cell_pos, MixturePhase::FLUID);
        assert!(
            (got_v_s - expected_v_s).length() < 1.0e-5,
            "solid velocity mismatch: got={got_v_s:?} expected={expected_v_s:?}"
        );
        assert!(
            (got_v_f - expected_v_f).length() < 1.0e-5,
            "fluid velocity mismatch: got={got_v_f:?} expected={expected_v_f:?}"
        );
    }

    #[test]
    fn momentum_is_exactly_conserved_across_the_coupling() {
        let (m_s, m_f) = (7.0_f32, 2.5_f32);
        let (v_s0, v_f0) = (Vec2::new(1.0, 0.5), Vec2::new(-3.0, 2.0));
        let dt = 0.05_f32;
        let k = 10.0_f32;
        let gravity = Vec2::ZERO;

        let mut grid = setup(m_s, v_s0, m_f, v_f0);
        grid.resolve_mixture_coupling(dt, gravity, k, 1.0, 0);

        let cell_pos = IVec2::new(2, 2);
        let v_s = grid.resolved_velocity_at(cell_pos, MixturePhase::SOLID);
        let v_f = grid.resolved_velocity_at(cell_pos, MixturePhase::FLUID);

        let p_before = m_s * v_s0 + m_f * v_f0;
        let p_after = m_s * v_s + m_f * v_f;
        assert!(
            (p_before - p_after).length() < 1.0e-4,
            "mixture coupling must conserve momentum exactly: before={p_before:?} after={p_after:?}"
        );
    }

    #[test]
    fn drag_pulls_phases_toward_a_shared_velocity_not_apart() {
        // Real, qualitative physical sanity check: whatever the exact numbers,
        // drag must reduce the RELATIVE speed between phases, never increase it
        // (that would mean the coupling is doing something backwards).
        let (m_s, m_f) = (3.0_f32, 3.0_f32);
        let (v_s0, v_f0) = (Vec2::new(0.0, 0.0), Vec2::new(5.0, 0.0));
        let dt = 0.1_f32;
        let k = 1.0_f32;

        let mut grid = setup(m_s, v_s0, m_f, v_f0);
        grid.resolve_mixture_coupling(dt, Vec2::ZERO, k, 1.0, 0);

        let cell_pos = IVec2::new(2, 2);
        let v_s = grid.resolved_velocity_at(cell_pos, MixturePhase::SOLID);
        let v_f = grid.resolved_velocity_at(cell_pos, MixturePhase::FLUID);
        let relative_before = (v_s0 - v_f0).length();
        let relative_after = (v_s - v_f).length();
        assert!(
            relative_after < relative_before,
            "drag should reduce relative velocity: before={relative_before} after={relative_after}"
        );
    }

    #[test]
    fn disabled_when_drag_coefficient_is_zero() {
        // 0.0 is the documented "disabled" sentinel -- both phases must read the
        // ordinary total field, completely unaffected by their own individual
        // momenta (matching every other opt-in system's true-no-op convention).
        let (m_s, m_f) = (4.0_f32, 1.0_f32);
        let (v_s0, v_f0) = (Vec2::new(2.0, 0.0), Vec2::new(-6.0, 0.0));
        let mut grid = setup(m_s, v_s0, m_f, v_f0);
        grid.resolve_mixture_coupling(0.1, Vec2::ZERO, 0.0, 1.0, 0);

        let cell_pos = IVec2::new(2, 2);
        let total_v = (m_s * v_s0 + m_f * v_f0) / (m_s + m_f);
        let v_s = grid.resolved_velocity_at(cell_pos, MixturePhase::SOLID);
        let v_f = grid.resolved_velocity_at(cell_pos, MixturePhase::FLUID);
        assert!((v_s - total_v).length() < 1.0e-5);
        assert!((v_f - total_v).length() < 1.0e-5);
    }

    #[test]
    fn three_phase_reduces_to_the_old_closed_form_when_only_two_are_present() {
        // Regression proof: the new N-phase solve, given exactly the same 2
        // phases the old hardcoded 2x2 closed form handled, must reproduce
        // that exact old result -- not just "a plausible-looking new number".
        let (m_s, m_f) = (4.0_f32, 1.0_f32);
        let (v_s0, v_f0) = (Vec2::new(0.0, 0.0), Vec2::new(0.0, -2.0));
        let dt = 0.1_f32;
        let k = 3.0_f32;

        let mut grid = setup(m_s, v_s0, m_f, v_f0);
        grid.resolve_mixture_coupling(dt, Vec2::ZERO, k, 1.0, 0);

        let a = dt * k / m_s;
        let b = dt * k / m_f;
        let det = 1.0 + a + b;
        let expected_v_s = ((1.0 + b) * v_s0 + a * v_f0) / det;
        let expected_v_f = (b * v_s0 + (1.0 + a) * v_f0) / det;

        let cell_pos = IVec2::new(2, 2);
        let got_v_s = grid.resolved_velocity_at(cell_pos, MixturePhase::SOLID);
        let got_v_f = grid.resolved_velocity_at(cell_pos, MixturePhase::FLUID);
        assert!((got_v_s - expected_v_s).length() < 1.0e-5);
        assert!((got_v_f - expected_v_f).length() < 1.0e-5);
    }

    #[test]
    fn three_phase_momentum_is_exactly_conserved() {
        // Real 3-phase node, deliberately non-round masses/velocities/k (not
        // nice numbers) -- a singular or ill-conditioned 3x3 system would
        // produce NaN/Inf or a non-conserving result here, so a finite,
        // conserved result is real, empirical proof the solve (and the
        // diagonally-dominant/SPD claim behind skipping pivoting) holds for
        // more than 2 phases, not just asserted from the derivation.
        let m = [5.3_f32, 1.7_f32, 2.9_f32];
        let v0 = [
            Vec2::new(1.1, -0.4),
            Vec2::new(-2.3, 0.9),
            Vec2::new(0.6, 1.8),
        ];
        let dt = 0.07_f32;
        let k = 4.1_f32;

        let mut grid = setup_three_phase(m, v0);
        grid.resolve_mixture_coupling(dt, Vec2::ZERO, k, 1.0, 0);

        let cell_pos = IVec2::new(2, 2);
        let phases = [MixturePhase::SOLID, MixturePhase::FLUID, MixturePhase(2)];
        let p_before: Vec2 = (0..3).map(|i| m[i] * v0[i]).sum();
        let p_after: Vec2 = (0..3)
            .map(|i| m[i] * grid.resolved_velocity_at(cell_pos, phases[i]))
            .sum();
        assert!(
            (p_before - p_after).length() < 1.0e-3,
            "3-phase coupling must conserve momentum exactly: before={p_before:?} after={p_after:?}"
        );

        // And a real qualitative check the solve did something sane: pairwise
        // relative speed between every phase pair should shrink, matching the
        // 2-phase `drag_pulls_phases_toward_a_shared_velocity_not_apart` test.
        for i in 0..3 {
            for j in (i + 1)..3 {
                let before = (v0[i] - v0[j]).length();
                let after = (grid.resolved_velocity_at(cell_pos, phases[i])
                    - grid.resolved_velocity_at(cell_pos, phases[j]))
                .length();
                assert!(
                    after < before,
                    "drag should reduce relative velocity between phases {i} and {j}: before={before} after={after}"
                );
            }
        }
    }
}

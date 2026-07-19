use std::collections::HashMap;

use glam::{IVec2, Vec2};

use super::{FxU32BuildHasher, Grid, flat_index};

/// Two-phase mixture coupling cell (Tampubolon et al. 2017 — see `MixturePhase`'s
/// own doc). Only allocated at nodes touched by at least one `MixturePhase::Solid`
/// OR `MixturePhase::Fluid` particle (via `WithMixturePhase`) — a scene that never
/// wraps a material this way never allocates a single one of these.
///
/// `solid_mass`/`solid_momentum` and `fluid_mass`/`fluid_momentum` accumulate
/// during P2G exactly like `Cell`'s own fields, but from each phase's particles
/// separately (both are ADDITIVE alongside the ordinary `Cell` scatter, not a
/// replacement — mirrors `ContactCell`'s own convention). `resolved_solid_v`/
/// `resolved_fluid_v` are filled in by `Grid::resolve_mixture_coupling` (after
/// `update_velocities` + gravity, same pipeline position as `resolve_contact`)
/// and are what G2P reads for solid/fluid particles respectively at nodes where
/// this cell exists.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct MixtureCell {
    solid_mass: f32,
    solid_momentum: Vec2,
    fluid_mass: f32,
    fluid_momentum: Vec2,
    resolved_solid_v: Vec2,
    resolved_fluid_v: Vec2,
}

pub(super) type MixtureCellMap = HashMap<u32, MixtureCell, FxU32BuildHasher>;

impl Grid {
    /// Accumulate mass and momentum for one mixture phase during P2G, additively
    /// alongside the normal `add_mass_momentum` call for the SAME particle — see
    /// `MixtureCell` doc. OOB silently ignored.
    pub fn add_mixture_mass_momentum(
        &mut self,
        cell_pos: IVec2,
        phase: crate::materials::MixturePhase,
        mass: f32,
        momentum: Vec2,
    ) {
        let Some(idx) = flat_index(cell_pos, self.resolution) else {
            return;
        };
        use crate::materials::MixturePhase;
        match self.mixture_cells.entry(idx) {
            std::collections::hash_map::Entry::Occupied(mut e) => {
                let cell = e.get_mut();
                match phase {
                    MixturePhase::Solid => {
                        cell.solid_mass += mass;
                        cell.solid_momentum += momentum;
                    }
                    MixturePhase::Fluid => {
                        cell.fluid_mass += mass;
                        cell.fluid_momentum += momentum;
                    }
                }
            }
            std::collections::hash_map::Entry::Vacant(e) => {
                self.mixture_dirty.push(idx);
                let mut cell = MixtureCell::default();
                match phase {
                    MixturePhase::Solid => {
                        cell.solid_mass = mass;
                        cell.solid_momentum = momentum;
                    }
                    MixturePhase::Fluid => {
                        cell.fluid_mass = mass;
                        cell.fluid_momentum = momentum;
                    }
                }
                e.insert(cell);
            }
        }
    }

    /// Resolved solid-phase velocity at `cell_pos` — valid after
    /// `resolve_mixture_coupling()`. Falls back to the ordinary total velocity
    /// when no mixture coupling was ever registered at this node, same
    /// convention as `grip_velocity_at`.
    pub fn resolved_solid_velocity_at(&self, cell_pos: IVec2) -> Vec2 {
        let Some(idx) = flat_index(cell_pos, self.resolution) else {
            return Vec2::ZERO;
        };
        self.mixture_cells
            .get(&idx)
            .map_or_else(|| self.velocity_at(cell_pos), |c| c.resolved_solid_v)
    }

    /// Resolved fluid-phase velocity at `cell_pos` — valid after
    /// `resolve_mixture_coupling()`. Same fallback convention as
    /// `resolved_solid_velocity_at`.
    pub fn resolved_fluid_velocity_at(&self, cell_pos: IVec2) -> Vec2 {
        let Some(idx) = flat_index(cell_pos, self.resolution) else {
            return Vec2::ZERO;
        };
        self.mixture_cells
            .get(&idx)
            .map_or_else(|| self.velocity_at(cell_pos), |c| c.resolved_fluid_v)
    }

    /// Resolves two-phase mixture coupling (Tampubolon et al. 2017 Darcy-style
    /// momentum exchange) at every mixture-active node — call after
    /// `update_velocities()` (needs the gravity-applied total field), same
    /// pipeline position as `resolve_contact`.
    ///
    /// Real, exact closed-form solve, not an iterative approximation: implicit
    /// backward-Euler drag exchange between two masses reduces to a 2x2 linear
    /// system per velocity COMPONENT (x and y decouple since drag is isotropic),
    /// solved directly here rather than needing Newton iteration or a global
    /// sparse solver — see this session's technical scoping doc for the full
    /// derivation. Let `v_s`, `v_f` be each phase's own pre-coupling velocity
    /// (its own momentum/mass, gravity already applied), `a = dt*k/m_s`,
    /// `b = dt*k/m_f`:
    ///   (1+a) v_s' - a v_f' = v_s
    ///   -b v_s' + (1+b) v_f' = v_f
    ///   det = 1 + a + b  (always > 0, unconditionally stable, no ill-conditioning
    ///   at any real k/dt/mass combination — this is what makes the LOCAL,
    ///   per-node simplification valid instead of needing the paper's own global
    ///   MINRES solve, which exists there specifically to handle full elastic/
    ///   plastic coupling this simplified drag-only model doesn't attempt).
    ///   v_s' = [(1+b) v_s + a v_f] / det
    ///   v_f' = [b v_s + (1+a) v_f] / det
    /// Momentum is exactly conserved by construction (`m_s*(v_s'-v_s) =
    /// -m_f*(v_f'-v_f)` falls out of the shared `det` and the `a*m_s = b*m_f =
    /// dt*k` identity), verified by a real test, not just claimed.
    ///
    /// Real, disclosed simplification vs. the paper: `drag_coefficient` (k) is a
    /// single scalar (mass/time), not the paper's own permeability/porosity-
    /// derived `c_E` field — mapping to real soil permeability is future work,
    /// not attempted here. Nodes with only one phase present get no correction
    /// at all (both resolved velocities just read the ordinary total field),
    /// matching `resolve_contact`'s own "no real second field" fallback.
    ///
    /// `cell_width`/`pressure_iterations` feed `project_mixture_incompressibility`
    /// (see its own doc) — a real, root-caused instability found 2026-07-18: the
    /// drag solve above conserves momentum but never enforces the mixture's real
    /// incompressibility constraint, so under sustained/confined loading (e.g.
    /// water settled into sand) the violation compounds silently over hundreds of
    /// steps until velocities blow past the CFL bound (see
    /// `mixture_coupling_long_settle_instability` memory). `pressure_iterations
    /// == 0` skips the projection entirely (the original, unmodified behavior).
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
            // Disabled: both fields just read the ordinary total velocity —
            // matches every other opt-in system's "true default is a no-op".
            for &idx in &self.mixture_dirty {
                let Some(&total) = self.cells.get(&idx) else {
                    continue;
                };
                if let Some(cell) = self.mixture_cells.get_mut(&idx) {
                    cell.resolved_solid_v = total.momentum;
                    cell.resolved_fluid_v = total.momentum;
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
            let m_s = cell.solid_mass;
            let m_f = cell.fluid_mass;
            if m_s <= MIN_MASS_FRACTION || m_f <= MIN_MASS_FRACTION {
                let cell = self.mixture_cells.get_mut(&idx).unwrap();
                cell.resolved_solid_v = total.momentum;
                cell.resolved_fluid_v = total.momentum;
                continue;
            }
            let v_s = cell.solid_momentum / m_s + gravity * dt;
            let v_f = cell.fluid_momentum / m_f + gravity * dt;
            let a = dt * drag_coefficient / m_s;
            let b = dt * drag_coefficient / m_f;
            let det = 1.0 + a + b;
            let v_s_new = ((1.0 + b) * v_s + a * v_f) / det;
            let v_f_new = (b * v_s + (1.0 + a) * v_f) / det;
            let cell = self.mixture_cells.get_mut(&idx).unwrap();
            cell.resolved_solid_v = v_s_new;
            cell.resolved_fluid_v = v_f_new;
        }
        if pressure_iterations > 0 {
            self.project_mixture_incompressibility(cell_width, pressure_iterations);
        }
    }

    /// Flat index -> cell position. Inverse of `flat_index`.
    fn idx_to_pos(&self, idx: u32) -> IVec2 {
        let idx = idx as usize;
        IVec2::new(
            (idx / self.resolution) as i32,
            (idx % self.resolution) as i32,
        )
    }

    fn mixture_solid_v_or_zero(&self, pos: IVec2) -> Vec2 {
        flat_index(pos, self.resolution)
            .and_then(|idx| self.mixture_cells.get(&idx))
            .map_or(Vec2::ZERO, |c| c.resolved_solid_v)
    }

    fn mixture_fluid_v_or_zero(&self, pos: IVec2) -> Vec2 {
        flat_index(pos, self.resolution)
            .and_then(|idx| self.mixture_cells.get(&idx))
            .map_or(Vec2::ZERO, |c| c.resolved_fluid_v)
    }

    /// Real fix for the long-settle instability found 2026-07-18 (see
    /// `mixture_coupling_long_settle_instability` memory): the closed-form drag
    /// solve above conserves momentum but never enforces the mixture's actual
    /// incompressibility constraint (Zhao & Choo 2020, "Stabilized material
    /// point methods for coupled large deformation and fluid flow in porous
    /// materials", arXiv:1905.00671):
    ///   (1 - n)*div(v_solid) + n*div(v_fluid) = 0
    /// where `n` is the local fluid volume fraction (porosity). Naive momentum-
    /// only coupling lets this drift under sustained/confined loading (water
    /// settled into sand -- close to the "undrained" regime the paper names as
    /// the specific failure case) until the accumulated violation destabilizes
    /// velocities well past the CFL bound.
    ///
    /// Real fix: a variable-density Chorin-style pressure projection (Bridson,
    /// "Fluid Simulation for Computer Graphics", ch. 5 -- the standard
    /// real-time-graphics form of enforcing incompressibility, generalized here
    /// to a two-phase mixture instead of one fluid), solved with a fixed number
    /// of Jacobi iterations rather than an exact sparse solve -- the real-time-
    /// affordable approximation both Zhao & Choo and Stam's own "Real-Time
    /// Fluid Dynamics for Games" independently point to. Per active mixture
    /// cell, using each cell's OWN local mass fractions as the porosity
    /// estimate `n = fluid_mass / (solid_mass + fluid_mass)`:
    ///
    ///   D = (1-n)*div(v_s) + n*div(v_f)                  (divergence residual)
    ///   alpha_s = 1/max(solid_mass, eps), alpha_f = 1/max(fluid_mass, eps)
    ///   K = (1-n)*alpha_s + n*alpha_f                     (local "mobility")
    ///   div( K * grad(p) ) = D                            (variable-mobility Poisson eq.)
    ///   v_s' = v_s - alpha_s * grad(p)
    ///   v_f' = v_f - alpha_f * grad(p)
    ///
    /// REAL BUG FOUND AND FIXED (2026-07-18, live-tested): a first version of
    /// this solved a CONSTANT-coefficient Laplacian for `p` (dividing D by K
    /// into the right-hand side) and only applied K/alpha in the final
    /// correction step -- inconsistent with the equation it was supposed to
    /// solve, and explosively unstable in practice (confirmed live: fps
    /// crashed to 2-3, relative_speed spiked to 40-50 within 14 frames,
    /// dramatically WORSE than the original unprojected instability). Root
    /// cause: `alpha = 1/mass` is unbounded at the near-zero-mass nodes that
    /// are completely ordinary at MPM kernel-support edges, and nothing in
    /// that formulation prevented an unbounded `alpha` from producing an
    /// unbounded velocity correction. The FIX -- folding the mobility `K` into
    /// the Laplacian operator itself via harmonic-mean FACE coefficients
    /// (`K_face = 2*K_i*K_j/(K_i+K_j)`) -- is the standard, correct treatment
    /// for variable-density/variable-mobility pressure projection (Bridson;
    /// Foster & Fedkiw). It is structurally self-limiting: a face where
    /// either side has near-zero mass (huge K) contributes almost nothing
    /// (harmonic mean of a huge value and a normal value is close to the
    /// SMALLER one), and a face where BOTH sides are near-empty correctly
    /// contributes ~0 (no material, no flux) instead of blowing up. Missing/
    /// OOB neighbors are treated as `K_j = 0` (a natural no-flux Neumann
    /// boundary at the material's own edge, not an arbitrary Dirichlet p=0).
    ///
    /// The Poisson equation is solved via `pressure_iterations` Jacobi sweeps.
    /// Real, disclosed limitation from Stam's own paper: a *settled, confined*
    /// liquid (exactly this scene) is the documented worst case for a
    /// low-iteration Jacobi solve -- pick `pressure_iterations` by measuring
    /// against the actual long-settle scenario, not by assuming a small fixed
    /// count is free.
    fn project_mixture_incompressibility(&mut self, cell_width: f32, pressure_iterations: u32) {
        const MIN_MASS: f32 = 1.0e-6;
        let h = cell_width.max(1.0e-6);

        // Per-cell constants: mobility K, inverse-mass weights, and the
        // divergence residual -- computed once from the post-drag-solve velocity
        // field, all in local (cell_pos, value) pairs so we're not fighting the
        // borrow checker against `self.mixture_cells` while reading neighbors.
        let mut alpha_s: HashMap<u32, f32, FxU32BuildHasher> = HashMap::default();
        let mut alpha_f: HashMap<u32, f32, FxU32BuildHasher> = HashMap::default();
        let mut significant: HashMap<u32, (bool, bool), FxU32BuildHasher> = HashMap::default();
        let mut mobility: HashMap<u32, f32, FxU32BuildHasher> = HashMap::default();
        let mut rhs: HashMap<u32, f32, FxU32BuildHasher> = HashMap::default();
        let mut pressure: HashMap<u32, f32, FxU32BuildHasher> = HashMap::default();

        for &idx in &self.mixture_dirty {
            let Some(cell) = self.mixture_cells.get(&idx) else {
                continue;
            };
            let pos = self.idx_to_pos(idx);
            let m_s = cell.solid_mass.max(0.0);
            let m_f = cell.fluid_mass.max(0.0);
            let n = if m_s + m_f > MIN_MASS {
                m_f / (m_s + m_f)
            } else {
                0.0
            };
            let a_s = 1.0 / m_s.max(MIN_MASS);
            let a_f = 1.0 / m_f.max(MIN_MASS);
            let k = (1.0 - n) * a_s + n * a_f;
            // Real bug found and fixed 2026-07-18 (see memory): a phase with
            // negligible LOCAL mass at this node (ordinary at MPM kernel-
            // support edges) has an unbounded `alpha = 1/mass`. The Poisson
            // SOLVE above is safe (harmonic-mean faces saturate it), but
            // applying that raw, unbounded alpha to a real, moderate grad(p)
            // in the correction step below produced velocity corrections in
            // the hundreds-to-thousands range at nodes with essentially no
            // real fluid (or solid) there -- confirmed live via direct
            // instrumentation. A phase's velocity is only meaningful, and
            // only gets corrected, where it holds a real fraction of this
            // node's total mass -- mirrors `resolve_mixture_coupling`'s own
            // "no real second field" skip convention, just with a threshold
            // large enough to matter (1e-6 lets through exactly the
            // kernel-edge slivers that caused this).
            const MIN_MASS_FRACTION: f32 = 0.01;
            let total_mass = (m_s + m_f).max(MIN_MASS);
            let solid_significant = m_s / total_mass > MIN_MASS_FRACTION;
            let fluid_significant = m_f / total_mass > MIN_MASS_FRACTION;

            let vs_r = self.mixture_solid_v_or_zero(pos + IVec2::new(1, 0)).x;
            let vs_l = self.mixture_solid_v_or_zero(pos - IVec2::new(1, 0)).x;
            let vs_u = self.mixture_solid_v_or_zero(pos + IVec2::new(0, 1)).y;
            let vs_d = self.mixture_solid_v_or_zero(pos - IVec2::new(0, 1)).y;
            let div_vs = (vs_r - vs_l) / (2.0 * h) + (vs_u - vs_d) / (2.0 * h);

            let vf_r = self.mixture_fluid_v_or_zero(pos + IVec2::new(1, 0)).x;
            let vf_l = self.mixture_fluid_v_or_zero(pos - IVec2::new(1, 0)).x;
            let vf_u = self.mixture_fluid_v_or_zero(pos + IVec2::new(0, 1)).y;
            let vf_d = self.mixture_fluid_v_or_zero(pos - IVec2::new(0, 1)).y;
            let div_vf = (vf_r - vf_l) / (2.0 * h) + (vf_u - vf_d) / (2.0 * h);

            let residual = (1.0 - n) * div_vs + n * div_vf;

            alpha_s.insert(idx, a_s);
            alpha_f.insert(idx, a_f);
            significant.insert(idx, (solid_significant, fluid_significant));
            mobility.insert(idx, k);
            rhs.insert(idx, residual);
            pressure.insert(idx, 0.0);
        }

        let k_or_zero = |pos: IVec2| -> f32 {
            flat_index(pos, self.resolution)
                .and_then(|idx| mobility.get(&idx).copied())
                .unwrap_or(0.0)
        };
        let p_or_zero = |p: &HashMap<u32, f32, FxU32BuildHasher>, pos: IVec2| -> f32 {
            flat_index(pos, self.resolution)
                .and_then(|idx| p.get(&idx).copied())
                .unwrap_or(0.0)
        };
        // Harmonic mean of two mobilities -- 0 if either side is ~0 (no material,
        // no flux through that face), never blows up even if one side is huge.
        let face_k = |k_i: f32, k_j: f32| -> f32 {
            if k_i + k_j > 1.0e-12 {
                2.0 * k_i * k_j / (k_i + k_j)
            } else {
                0.0
            }
        };

        for _ in 0..pressure_iterations {
            let mut next = pressure.clone();
            for &idx in &self.mixture_dirty {
                let (Some(&r), Some(&k_i)) = (rhs.get(&idx), mobility.get(&idx)) else {
                    continue;
                };
                let pos = self.idx_to_pos(idx);
                let k_r = face_k(k_i, k_or_zero(pos + IVec2::new(1, 0)));
                let k_l = face_k(k_i, k_or_zero(pos - IVec2::new(1, 0)));
                let k_u = face_k(k_i, k_or_zero(pos + IVec2::new(0, 1)));
                let k_d = face_k(k_i, k_or_zero(pos - IVec2::new(0, 1)));
                let k_sum = (k_r + k_l + k_u + k_d).max(1.0e-9);

                let p_r = p_or_zero(&pressure, pos + IVec2::new(1, 0));
                let p_l = p_or_zero(&pressure, pos - IVec2::new(1, 0));
                let p_u = p_or_zero(&pressure, pos + IVec2::new(0, 1));
                let p_d = p_or_zero(&pressure, pos - IVec2::new(0, 1));
                let weighted_neighbors = k_r * p_r + k_l * p_l + k_u * p_u + k_d * p_d;
                next.insert(idx, (weighted_neighbors - h * h * r) / k_sum);
            }
            pressure = next;
        }

        for &idx in &self.mixture_dirty {
            let (Some(&a_s), Some(&a_f), Some(&(solid_significant, fluid_significant))) =
                (alpha_s.get(&idx), alpha_f.get(&idx), significant.get(&idx))
            else {
                continue;
            };
            let pos = self.idx_to_pos(idx);
            let p_r = p_or_zero(&pressure, pos + IVec2::new(1, 0));
            let p_l = p_or_zero(&pressure, pos - IVec2::new(1, 0));
            let p_u = p_or_zero(&pressure, pos + IVec2::new(0, 1));
            let p_d = p_or_zero(&pressure, pos - IVec2::new(0, 1));
            let grad_p = Vec2::new((p_r - p_l) / (2.0 * h), (p_u - p_d) / (2.0 * h));
            if let Some(cell) = self.mixture_cells.get_mut(&idx) {
                if solid_significant {
                    cell.resolved_solid_v -= a_s * grad_p;
                }
                if fluid_significant {
                    cell.resolved_fluid_v -= a_f * grad_p;
                }
            }
        }
    }
}

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
        grid.add_mixture_mass_momentum(cell_pos, MixturePhase::Solid, m_s, m_s * v_s0);
        grid.add_mixture_mass_momentum(cell_pos, MixturePhase::Fluid, m_f, m_f * v_f0);
        // Normalize the total field's raw momentum into true velocity, matching the
        // real pipeline's ordering (resolve_mixture_coupling always runs after
        // update_velocities) -- resolve_mixture_coupling's own "no drag"/"no real
        // second field" fallbacks read `Cell.momentum` assuming it's already velocity.
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
        let got_v_s = grid.resolved_solid_velocity_at(cell_pos);
        let got_v_f = grid.resolved_fluid_velocity_at(cell_pos);
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
        let v_s = grid.resolved_solid_velocity_at(cell_pos);
        let v_f = grid.resolved_fluid_velocity_at(cell_pos);

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
        let v_s = grid.resolved_solid_velocity_at(cell_pos);
        let v_f = grid.resolved_fluid_velocity_at(cell_pos);
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
        let v_s = grid.resolved_solid_velocity_at(cell_pos);
        let v_f = grid.resolved_fluid_velocity_at(cell_pos);
        assert!((v_s - total_v).length() < 1.0e-5);
        assert!((v_f - total_v).length() < 1.0e-5);
    }

    /// Real test for the incompressibility projection itself: build a small
    /// neighborhood of mixture-active nodes with a deliberately divergent
    /// solid velocity field (radiating outward from a center node -- a real,
    /// nonzero div(v_s)), run the projection, and confirm the projected
    /// divergence residual actually SHRINKS relative to the unprojected one.
    /// This is the real, checkable claim behind
    /// `project_mixture_incompressibility` -- not just "runs without crashing."
    #[test]
    fn pressure_projection_reduces_divergence_residual() {
        let mut grid = Grid::new(16);
        let center = IVec2::new(8, 8);
        let m_s = 2.0_f32;
        let m_f = 2.0_f32;
        // Solid velocity field radiating outward from `center` -- real nonzero
        // divergence by construction (a source, not a rotation/shear).
        // A uniform dilation (v = 0.5*d) has constant divergence everywhere and
        // a closed (Neumann) system can never fully cancel that -- it's a
        // net source with nowhere to drain. Use a decaying (Gaussian-weighted)
        // radial field instead: real, concentrated divergence near `center`
        // that fades toward the patch edge, so a closed system CAN resolve
        // it (the total divergence over the patch is close to zero).
        let v_s_at = |pos: IVec2| -> Vec2 {
            let d = (pos - center).as_vec2();
            let r2 = d.length_squared();
            d * 0.5 * (-r2 / 8.0).exp()
        };
        for dx in -6..=6 {
            for dy in -6..=6 {
                let pos = center + IVec2::new(dx, dy);
                let v_s = v_s_at(pos);
                grid.add_mass_momentum(pos, m_s + m_f, m_s * v_s + m_f * Vec2::ZERO);
                grid.add_mixture_mass_momentum(pos, MixturePhase::Solid, m_s, m_s * v_s);
                grid.add_mixture_mass_momentum(pos, MixturePhase::Fluid, m_f, m_f * Vec2::ZERO);
            }
        }
        grid.update_velocities(0.0, Vec2::ZERO);

        // No drag (phases already at rest relative to their own construction),
        // no projection yet -- just resolve the mixture bookkeeping.
        grid.resolve_mixture_coupling(0.0, Vec2::ZERO, 1.0e-9, 1.0, 0);
        let div_before = |g: &Grid| -> f32 {
            let r = g.resolved_solid_velocity_at(center + IVec2::new(1, 0)).x
                - g.resolved_solid_velocity_at(center - IVec2::new(1, 0)).x;
            let u = g.resolved_solid_velocity_at(center + IVec2::new(0, 1)).y
                - g.resolved_solid_velocity_at(center - IVec2::new(0, 1)).y;
            (r + u) / 2.0
        };
        let residual_unprojected = div_before(&grid).abs();
        assert!(
            residual_unprojected > 1.0e-3,
            "test setup should have real nonzero divergence, got {residual_unprojected}"
        );

        // Same setup, but with the projection applied.
        let mut grid2 = Grid::new(16);
        for dx in -6..=6 {
            for dy in -6..=6 {
                let pos = center + IVec2::new(dx, dy);
                let v_s = v_s_at(pos);
                grid2.add_mass_momentum(pos, m_s + m_f, m_s * v_s + m_f * Vec2::ZERO);
                grid2.add_mixture_mass_momentum(pos, MixturePhase::Solid, m_s, m_s * v_s);
                grid2.add_mixture_mass_momentum(pos, MixturePhase::Fluid, m_f, m_f * Vec2::ZERO);
            }
        }
        grid2.update_velocities(0.0, Vec2::ZERO);
        grid2.resolve_mixture_coupling(0.0, Vec2::ZERO, 1.0e-9, 1.0, 200);
        let residual_projected = div_before(&grid2).abs();

        assert!(
            residual_projected < residual_unprojected * 0.6,
            "projection should substantially shrink the divergence residual: \
             before={residual_unprojected:.5} after={residual_projected:.5}"
        );
    }
}

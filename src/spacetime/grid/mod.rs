//! Sparse Eulerian background grid for MLS-MPM P2G/G2P.
//!
//! Split 2026-07-19 (was 1592 lines in one file) by subsystem — the struct
//! already partitioned into 3 nearly-independent field groups (`cells`,
//! `contact_cells`, `mixture_cells`, each with its own dirty list), so this
//! finishes the same sibling-split job `contact_normal.rs`/
//! `directional_grip.rs` already started. `contact.rs` and `mixture.rs` each
//! add their own `impl Grid { ... }` block (ordinary Rust: multiple impl
//! blocks for one type across files) rather than living inside this one —
//! every method these files define was already private to the `grid` module
//! before the split (visible to `grid` and its descendants by Rust's normal
//! privacy rule), so no visibility widening was needed for `Grid`'s own
//! fields; only the two new per-subsystem cell types (`ContactCell`/
//! `ContactCellMap`, `MixtureCell`/`MixtureCellMap`) needed `pub(super)` so
//! this file's own `Grid` struct definition can still name their map types.

pub mod kernel;

mod contact;
mod contact_normal;
mod directional_grip;
mod mixture;

use std::collections::HashMap;
use std::hash::{BuildHasher, Hasher};

use glam::{IVec2, Vec2};

use contact::ContactCellMap;
pub use directional_grip::DirectionalContactGrip;
use mixture::MixtureCellMap;

/// FxHash-style hasher for the grid's `u32` flat-index keys.
///
/// `std::collections::HashMap` defaults to SipHash — a cryptographic, DoS-resistant
/// hash that is deliberately slow. The grid is hashed 9× per particle every substep
/// (the hottest loop in the solver) on an internal `u32` key with no adversarial input,
/// so a single-multiply non-cryptographic hash is both correct and far faster.
/// Constant is the standard FxHash multiplier. Zero dependencies — keeps the
/// glam + bytemuck-only invariant.
const FXHASH_K: u64 = 0x51_7c_c1_b7_27_22_0a_95;

#[derive(Default)]
pub struct FxU32Hasher {
    hash: u64,
}

impl Hasher for FxU32Hasher {
    #[inline]
    fn write_u32(&mut self, i: u32) {
        // Grid keys are always exactly one u32 — this is the only path taken in practice.
        self.hash = (i as u64).wrapping_mul(FXHASH_K);
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        // General fallback so the impl is correct for any key, not just u32.
        for &b in bytes {
            self.hash = (self.hash.rotate_left(5) ^ b as u64).wrapping_mul(FXHASH_K);
        }
    }

    #[inline]
    fn finish(&self) -> u64 {
        self.hash
    }
}

#[derive(Default, Clone, Copy)]
pub struct FxU32BuildHasher;

impl BuildHasher for FxU32BuildHasher {
    type Hasher = FxU32Hasher;
    #[inline]
    fn build_hasher(&self) -> FxU32Hasher {
        FxU32Hasher::default()
    }
}

/// Sparse cell storage keyed by flat index, using the fast non-crypto hasher.
/// `pub(crate)` so `transfer.rs` can build thread-local accumulators for parallel P2G.
pub(crate) type CellMap = HashMap<u32, Cell, FxU32BuildHasher>;
/// Flat-index → velocity snapshot, see `Grid::snapshot_velocities`.
pub type VelocitySnapshot = HashMap<u32, Vec2, FxU32BuildHasher>;

/// Converts a cell position to the flat HashMap key, or `None` if out of domain bounds.
/// Shared by `Grid::add_mass_momentum` and the parallel P2G scatter in `transfer.rs` — both
/// must agree on bounds-checking and indexing, so this is the single source of truth.
pub(crate) fn flat_index(cell_pos: IVec2, resolution: usize) -> Option<u32> {
    if cell_pos.x < 0 || cell_pos.y < 0 {
        return None;
    }
    let x = cell_pos.x as usize;
    let y = cell_pos.y as usize;
    if x >= resolution || y >= resolution {
        return None;
    }
    Some((x * resolution + y) as u32)
}

/// One grid cell — `repr(C)` for stable GPU buffer layout.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Cell {
    /// Dual-phase field.
    /// During P2G scatter: accumulated momentum (mass × velocity).
    /// After `update_velocities`: normalized to true grid velocity (momentum / mass).
    pub momentum: Vec2,
    pub mass: f32,
}

/// Sparse grid — HashMap-backed, only touched cells allocated.
///
/// `resolution` defines the simulation domain (soft boundary enforcement).
/// Memory cost is O(active particles × stencil) not O(resolution²).
/// A 4096-cell domain with 50k particles uses ~4 MB instead of 192 MB.
///
/// All P2G/G2P callers go through the public API. The HashMap key is the flat
/// index `x * resolution + y`, matching the boundary condition convention.
#[derive(Debug)]
pub struct Grid {
    resolution: usize,
    /// Sparse cell storage. Only contains cells touched this frame.
    cells: CellMap,
    /// Flat indices of cells touched this frame, in insertion order.
    /// Separate from `cells` to enable O(touched) clear without iterating HashMap buckets.
    dirty: Vec<u32>,
    /// Second velocity field for multi-field contact — see `contact::ContactCell` doc.
    /// Empty for every scene that never sets `Particle::contact_group`, which is the
    /// critical zero-cost property: `has_contact_activity()` gates the extra work in
    /// P2G/G2P/step so a scene that doesn't use this feature runs unaffected.
    contact_cells: ContactCellMap,
    contact_dirty: Vec<u32>,
    /// Two-phase mixture coupling field — see `mixture::MixtureCell` doc. Empty for
    /// every scene that never uses `WithMixturePhase`, the same zero-cost property
    /// `contact_cells` already has.
    mixture_cells: MixtureCellMap,
    mixture_dirty: Vec<u32>,
}

impl Grid {
    pub fn new(resolution: usize) -> Self {
        assert!(resolution >= 4, "grid resolution must be >= 4");
        Self {
            resolution,
            cells: CellMap::default(),
            dirty: Vec::new(),
            contact_cells: ContactCellMap::default(),
            contact_dirty: Vec::new(),
            mixture_cells: MixtureCellMap::default(),
            mixture_dirty: Vec::new(),
        }
    }

    pub fn resolution(&self) -> usize {
        self.resolution
    }

    /// True if any grip particle touched the grid this substep. Gates the extra
    /// contact-aware work in P2G/G2P/step — when false (every scene that never sets
    /// `Particle::contact_group`), those paths run their original, unmodified logic.
    pub fn has_contact_activity(&self) -> bool {
        !self.contact_dirty.is_empty()
    }

    /// Remove only touched cells. O(touched), not O(resolution²).
    pub fn clear(&mut self) {
        for &idx in &self.dirty {
            self.cells.remove(&idx);
        }
        self.dirty.clear();
        for &idx in &self.contact_dirty {
            self.contact_cells.remove(&idx);
        }
        self.contact_dirty.clear();
        for &idx in &self.mixture_dirty {
            self.mixture_cells.remove(&idx);
        }
        self.mixture_dirty.clear();
    }

    /// True if any mixture-phase particle touched the grid this substep. Gates
    /// the extra mixture-aware work in P2G/G2P/step — same convention as
    /// `has_contact_activity`.
    pub fn has_mixture_activity(&self) -> bool {
        !self.mixture_dirty.is_empty()
    }

    /// Accumulate mass and momentum during P2G. OOB silently ignored.
    pub fn add_mass_momentum(&mut self, cell_pos: IVec2, mass: f32, momentum: Vec2) {
        let Some(idx) = flat_index(cell_pos, self.resolution) else {
            return;
        };
        self.accumulate(idx, mass, momentum);
    }

    /// Accumulate by pre-computed flat index (already bounds-checked by the caller).
    /// Single hash lookup via entry() — was contains_key + insert + get_mut (3 lookups)
    /// in the hottest scatter loop. dirty only grows on first touch of a cell.
    fn accumulate(&mut self, idx: u32, mass: f32, momentum: Vec2) {
        match self.cells.entry(idx) {
            std::collections::hash_map::Entry::Occupied(mut e) => {
                let cell = e.get_mut();
                cell.mass += mass;
                cell.momentum += momentum;
            }
            std::collections::hash_map::Entry::Vacant(e) => {
                self.dirty.push(idx);
                e.insert(Cell { momentum, mass });
            }
        }
    }

    /// Grid velocity at `cell_pos` — valid after `update_velocities()`. Zero for OOB/untouched.
    pub fn velocity_at(&self, cell_pos: IVec2) -> Vec2 {
        if cell_pos.x < 0 || cell_pos.y < 0 {
            return Vec2::ZERO;
        }
        let x = cell_pos.x as usize;
        let y = cell_pos.y as usize;
        if x >= self.resolution || y >= self.resolution {
            return Vec2::ZERO;
        }
        self.cells
            .get(&((x * self.resolution + y) as u32))
            .map_or(Vec2::ZERO, |c| c.momentum)
    }

    /// True if `cell_pos` was touched by P2G this frame.
    #[inline]
    pub fn cell_is_active(&self, cell_pos: IVec2) -> bool {
        if cell_pos.x < 0 || cell_pos.y < 0 {
            return false;
        }
        let x = cell_pos.x as usize;
        let y = cell_pos.y as usize;
        if x >= self.resolution || y >= self.resolution {
            return false;
        }
        self.cells.contains_key(&((x * self.resolution + y) as u32))
    }

    pub fn mass_at(&self, cell_pos: IVec2) -> f32 {
        if cell_pos.x < 0 || cell_pos.y < 0 {
            return 0.0;
        }
        let x = cell_pos.x as usize;
        let y = cell_pos.y as usize;
        if x >= self.resolution || y >= self.resolution {
            return 0.0;
        }
        self.cells
            .get(&((x * self.resolution + y) as u32))
            .map_or(0.0, |c| c.mass)
    }

    /// Normalize momentum → velocity and apply gravity. Operates only on active cells.
    ///
    /// A thin wrapper over `normalize_velocities` + `apply_gravity` — split so ASFLIP
    /// (`SimConfig::asflip_blend`) can snapshot the grid's pre-force velocity in between
    /// the two (see `snapshot_velocities`). Behavior here is unchanged for every existing
    /// caller (`solver::step`, `spacetime::diff`'s `update_velocities_vjp` differentiates
    /// this exact combined formula, so the split must never change its net effect).
    pub fn update_velocities(&mut self, dt: f32, gravity: Vec2) {
        self.normalize_velocities();
        self.apply_gravity(dt, gravity);
    }

    /// Momentum → velocity normalization only (no gravity). Operates only on active cells.
    pub fn normalize_velocities(&mut self) {
        let (dirty, cells) = (&self.dirty, &mut self.cells);
        for &idx in dirty {
            if let Some(cell) = cells.get_mut(&idx)
                && cell.mass > 0.0
            {
                cell.momentum /= cell.mass;
            }
        }
    }

    /// Adds gravity to every active cell's (already-normalized) velocity.
    pub fn apply_gravity(&mut self, dt: f32, gravity: Vec2) {
        let (dirty, cells) = (&self.dirty, &mut self.cells);
        for &idx in dirty {
            if let Some(cell) = cells.get_mut(&idx)
                && cell.mass > 0.0
            {
                cell.momentum += gravity * dt;
            }
        }
    }

    /// Snapshot of every active cell's CURRENT velocity, keyed the same way as `cells`
    /// (flat index → velocity). Used only by ASFLIP (`SimConfig::asflip_blend > 0.0`) to
    /// capture the grid's velocity right after `normalize_velocities` — i.e. before this
    /// substep's gravity, boundary conditions, or contact resolution modify it — so G2P
    /// can later compute the classic FLIP residual `v_p_old - old_v` (Fei et al. 2021).
    /// O(touched cells), not O(grid²): iterates `dirty`, not the full domain. Never called
    /// when ASFLIP is disabled (the default), so this has zero cost for every other scene.
    pub fn snapshot_velocities(&self) -> VelocitySnapshot {
        let mut snapshot = HashMap::with_capacity_and_hasher(self.dirty.len(), FxU32BuildHasher);
        for &idx in &self.dirty {
            if let Some(cell) = self.cells.get(&idx) {
                snapshot.insert(idx, cell.momentum);
            }
        }
        snapshot
    }

    /// Reads a pre-force velocity snapshot (see `snapshot_velocities`) at `cell_pos`,
    /// mirroring `velocity_at`'s own OOB/untouched-is-zero convention exactly.
    pub fn pre_force_velocity_at(&self, snapshot: &VelocitySnapshot, cell_pos: IVec2) -> Vec2 {
        let Some(idx) = flat_index(cell_pos, self.resolution) else {
            return Vec2::ZERO;
        };
        snapshot.get(&idx).copied().unwrap_or(Vec2::ZERO)
    }

    /// Analytic adjoint of one cell's `update_velocities` step w.r.t. its
    /// momentum and mass BEFORE the update (which overwrites `momentum` in
    /// place to become the actual velocity) -- third piece of differentiable
    /// stepping, chains downstream of `p2g_stress_vjp`'s per-cell momentum
    /// gradient output.
    ///
    /// v = p/m + gravity*dt (see `update_velocities` above). `gravity*dt` is
    /// an additive constant -- contributes zero gradient. Given the gradient
    /// flowing back from the resulting velocity, `d_loss_d_v` (a Vec2),
    /// standard scalar/vector calculus (quotient rule on p/m, treating m as
    /// scalar) gives:
    ///
    ///   d_loss_d_momentum = d_loss_d_v / mass
    ///   d_loss_d_mass     = -(d_loss_d_v . momentum) / mass^2
    ///
    /// SCOPED: does not cover boundary-condition application or velocity
    /// clamping, both applied AFTER this in the real substep -- those are
    /// piecewise/conditional (zero out or cap components), differentiable
    /// almost everywhere but with real kinks at the boundary, deliberately
    /// deferred as their own future piece, not silently folded in here.
    /// Verified against central-difference numerical gradients in this
    /// module's own tests, same discipline as every other adjoint so far.
    pub fn update_velocities_vjp(momentum: Vec2, mass: f32, d_loss_d_v: Vec2) -> (Vec2, f32) {
        let d_loss_d_momentum = d_loss_d_v / mass;
        let d_loss_d_mass = -(d_loss_d_v.dot(momentum)) / (mass * mass);
        (d_loss_d_momentum, d_loss_d_mass)
    }

    /// Iterate active cells (read-only). For diagnostics.
    pub fn active_cells(&self) -> impl Iterator<Item = &Cell> {
        let cells = &self.cells;
        self.dirty.iter().filter_map(move |idx| cells.get(idx))
    }

    /// Iterate active cells (mutable). For CFL clamping.
    pub fn active_cells_mut(&mut self) -> impl Iterator<Item = &mut Cell> {
        let (dirty, cells) = (&self.dirty, &mut self.cells);
        // SAFETY: dirty contains unique indices (enforced at insertion), each yielding
        // a distinct &mut Cell. HashMap does not reallocate during iteration here since
        // no inserts occur between clear() and the next add_mass_momentum() call.
        let ptr = cells as *mut CellMap;
        dirty.iter().filter_map(move |idx| {
            // SAFETY: each idx is unique in dirty, so no two iterations alias.
            unsafe { (*ptr).get_mut(idx) }
        })
    }

    /// Iterate active cells with flat index: `(flat_idx, &mut Cell)`.
    /// `flat_idx = x * resolution + y` — same convention used by boundary conditions.
    pub fn active_cells_with_index_mut(&mut self) -> impl Iterator<Item = (usize, &mut Cell)> {
        let (dirty, cells) = (&self.dirty, &mut self.cells);
        let ptr = cells as *mut CellMap;
        dirty.iter().filter_map(move |&idx| {
            // SAFETY: same as active_cells_mut — unique indices, no concurrent inserts.
            unsafe { (*ptr).get_mut(&idx).map(|cell| (idx as usize, cell)) }
        })
    }

    /// Number of cells that received mass this frame.
    pub fn active_cell_count(&self) -> usize {
        self.dirty.len()
    }
}

#[cfg(test)]
mod update_velocities_vjp_tests {
    use super::*;

    /// Forward formula exactly matching `update_velocities`'s own math (minus
    /// the constant `gravity*dt` term, which contributes zero gradient and is
    /// omitted here so the finite-difference check isolates the momentum/mass
    /// dependence being verified).
    fn velocity(momentum: Vec2, mass: f32) -> Vec2 {
        momentum / mass
    }

    /// Scalar loss L(momentum, mass) = g . v(momentum, mass) -- checked one
    /// input component at a time via central differences.
    fn loss(momentum: Vec2, mass: f32, g: Vec2) -> f32 {
        g.dot(velocity(momentum, mass))
    }

    fn check_matches_finite_difference(momentum: Vec2, mass: f32, g: Vec2) {
        let (analytic_d_momentum, analytic_d_mass) = Grid::update_velocities_vjp(momentum, mass, g);
        let h = 1.0e-3_f32;

        let numeric_d_momentum_x = (loss(momentum + Vec2::new(h, 0.0), mass, g)
            - loss(momentum - Vec2::new(h, 0.0), mass, g))
            / (2.0 * h);
        let numeric_d_momentum_y = (loss(momentum + Vec2::new(0.0, h), mass, g)
            - loss(momentum - Vec2::new(0.0, h), mass, g))
            / (2.0 * h);
        let numeric_d_mass =
            (loss(momentum, mass + h, g) - loss(momentum, mass - h, g)) / (2.0 * h);

        let check = |label: &str, analytic: f32, numeric: f32| {
            let diff = (numeric - analytic).abs();
            let scale = numeric.abs().max(analytic.abs()).max(1.0);
            assert!(
                diff / scale < 1.0e-2,
                "update_velocities_vjp mismatch at {label}: analytic={analytic:.6} \
                 numeric(central-diff)={numeric:.6} relative_diff={:.2e} \
                 (momentum={momentum:?}, mass={mass}, g={g:?})",
                diff / scale
            );
        };

        check("d_momentum.x", analytic_d_momentum.x, numeric_d_momentum_x);
        check("d_momentum.y", analytic_d_momentum.y, numeric_d_momentum_y);
        check("d_mass", analytic_d_mass, numeric_d_mass);
    }

    #[test]
    fn matches_finite_difference_unit_mass() {
        check_matches_finite_difference(Vec2::new(2.0, -1.5), 1.0, Vec2::new(0.7, 0.3));
    }

    #[test]
    fn matches_finite_difference_heavier_cell() {
        check_matches_finite_difference(Vec2::new(-3.2, 4.1), 5.5, Vec2::new(-0.9, 1.4));
    }

    #[test]
    fn matches_finite_difference_light_cell_with_asymmetric_gradient() {
        check_matches_finite_difference(Vec2::new(0.8, 0.05), 0.2, Vec2::new(2.0, -3.5));
    }
}

pub mod kernel;

use std::collections::HashMap;
use std::hash::{BuildHasher, Hasher};

use glam::{IVec2, Vec2};

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
}

impl Grid {
    pub fn new(resolution: usize) -> Self {
        assert!(resolution >= 4, "grid resolution must be >= 4");
        Self {
            resolution,
            cells: CellMap::default(),
            dirty: Vec::new(),
        }
    }

    pub fn resolution(&self) -> usize {
        self.resolution
    }

    /// Remove only touched cells. O(touched), not O(resolution²).
    pub fn clear(&mut self) {
        for &idx in &self.dirty {
            self.cells.remove(&idx);
        }
        self.dirty.clear();
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
    pub fn update_velocities(&mut self, dt: f32, gravity: Vec2) {
        let (dirty, cells) = (&self.dirty, &mut self.cells);
        for &idx in dirty {
            if let Some(cell) = cells.get_mut(&idx)
                && cell.mass > 0.0
            {
                cell.momentum /= cell.mass;
                cell.momentum += gravity * dt;
            }
        }
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

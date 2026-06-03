pub mod kernel;

use std::collections::HashMap;

use glam::{IVec2, Vec2};

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
    cells: HashMap<u32, Cell>,
    /// Flat indices of cells touched this frame, in insertion order.
    /// Separate from `cells` to enable O(touched) clear without iterating HashMap buckets.
    dirty: Vec<u32>,
}

impl Grid {
    pub fn new(resolution: usize) -> Self {
        assert!(resolution >= 4, "grid resolution must be >= 4");
        Self {
            resolution,
            cells: HashMap::new(),
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
        if cell_pos.x < 0 || cell_pos.y < 0 {
            return;
        }
        let x = cell_pos.x as usize;
        let y = cell_pos.y as usize;
        if x >= self.resolution || y >= self.resolution {
            return;
        }
        let idx = (x * self.resolution + y) as u32;
        if !self.cells.contains_key(&idx) {
            self.cells.insert(idx, Cell::default());
            self.dirty.push(idx);
        }
        let cell = self.cells.get_mut(&idx).unwrap();
        cell.mass += mass;
        cell.momentum += momentum;
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
            if let Some(cell) = cells.get_mut(&idx) {
                if cell.mass > 0.0 {
                    cell.momentum /= cell.mass;
                    cell.momentum += gravity * dt;
                }
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
        let ptr = cells as *mut HashMap<u32, Cell>;
        dirty.iter().filter_map(move |idx| {
            // SAFETY: each idx is unique in dirty, so no two iterations alias.
            unsafe { (*ptr).get_mut(idx) }
        })
    }

    /// Iterate active cells with flat index: `(flat_idx, &mut Cell)`.
    /// `flat_idx = x * resolution + y` — same convention used by boundary conditions.
    pub fn active_cells_with_index_mut(&mut self) -> impl Iterator<Item = (usize, &mut Cell)> {
        let (dirty, cells) = (&self.dirty, &mut self.cells);
        let ptr = cells as *mut HashMap<u32, Cell>;
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

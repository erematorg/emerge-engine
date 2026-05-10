pub mod kernel;

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

/// Sparse-ready grid — dense `Vec<Cell>` backing with dirty-index tracking.
///
/// `clear()` / `update_velocities()` / iteration operate only on cells touched this
/// frame — O(active particles × stencil) instead of O(grid_res²). For a 1000×1000
/// domain with 10k particles this is ~90k operations vs 1M, enabling large open-world
/// domains without the full HashMap step.
///
/// Upgrade path: swap `Vec<Cell>` for `HashMap<u32, Cell>` when grid_res > ~512.
/// All callers go through the public API — none touch the backing store directly.
#[derive(Debug)]
pub struct Grid {
    resolution: usize,
    cells: Vec<Cell>,
    /// Whether each cell was touched this frame. Parallel to `cells`.
    /// Used to dedup dirty_indices — guarantees each cell appears exactly once.
    touched: Vec<bool>,
    /// Flat indices of touched cells, each appearing exactly once.
    dirty: Vec<u32>,
}

impl Grid {
    pub fn new(resolution: usize) -> Self {
        assert!(resolution >= 4, "grid resolution must be >= 4");
        let n = resolution * resolution;
        Self {
            resolution,
            cells: vec![Cell::default(); n],
            touched: vec![false; n],
            dirty: Vec::with_capacity(n.min(1 << 16)),
        }
    }

    pub fn resolution(&self) -> usize {
        self.resolution
    }

    /// Zero only cells touched since last `clear()`. O(active cells), not O(grid_res²).
    pub fn clear(&mut self) {
        for &idx in &self.dirty {
            let i = idx as usize;
            self.cells[i] = Cell::default();
            self.touched[i] = false;
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
        let idx = x * self.resolution + y;
        let cell = &mut self.cells[idx];
        cell.mass += mass;
        cell.momentum += momentum;
        if !self.touched[idx] {
            self.touched[idx] = true;
            self.dirty.push(idx as u32);
        }
    }

    /// Grid velocity at `cell_pos` — valid after `update_velocities()`. Zero for OOB.
    pub fn velocity_at(&self, cell_pos: IVec2) -> Vec2 {
        if cell_pos.x < 0 || cell_pos.y < 0 {
            return Vec2::ZERO;
        }
        let x = cell_pos.x as usize;
        let y = cell_pos.y as usize;
        if x >= self.resolution || y >= self.resolution {
            return Vec2::ZERO;
        }
        self.cells[x * self.resolution + y].momentum
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
        self.cells[x * self.resolution + y].mass
    }

    /// Normalize momentum → velocity and apply gravity. Operates only on active cells.
    pub fn update_velocities(&mut self, dt: f32, gravity: Vec2) {
        for &idx in &self.dirty {
            let cell = &mut self.cells[idx as usize];
            if cell.mass > 0.0 {
                cell.momentum /= cell.mass;
                cell.momentum += gravity * dt;
            }
        }
    }

    /// Iterate active cells (read-only). For diagnostics.
    pub fn active_cells(&self) -> impl Iterator<Item = &Cell> {
        self.dirty.iter().map(move |&idx| &self.cells[idx as usize])
    }

    /// Iterate active cells (mutable). For CFL clamping.
    pub fn active_cells_mut(&mut self) -> impl Iterator<Item = &mut Cell> {
        let ptr = self.cells.as_mut_ptr();
        self.dirty.iter().map(move |&idx| {
            // SAFETY: dirty contains unique in-bounds indices (enforced at insertion).
            unsafe { &mut *ptr.add(idx as usize) }
        })
    }

    /// Iterate active cells with flat index: `(flat_idx, &mut Cell)`.
    /// `flat_idx = x * resolution + y` — same formula used by boundary conditions.
    pub fn active_cells_with_index_mut(&mut self) -> impl Iterator<Item = (usize, &mut Cell)> {
        let ptr = self.cells.as_mut_ptr();
        self.dirty.iter().map(move |&idx| {
            let i = idx as usize;
            // SAFETY: dirty contains unique in-bounds indices (enforced at insertion).
            (i, unsafe { &mut *ptr.add(i) })
        })
    }

    /// Number of cells that received mass this frame.
    pub fn active_cell_count(&self) -> usize {
        self.dirty.len()
    }
}

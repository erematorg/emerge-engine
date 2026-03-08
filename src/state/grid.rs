use glam::{IVec2, Vec2};

/// One grid cell — `repr(C)` for stable GPU buffer layout.
/// Use `Grid::cells_as_bytes` for wgpu buffer writes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Cell {
    /// Dual-phase field.
    /// During P2G scatter: accumulated momentum (mass × velocity).
    /// After `update_velocities`: normalized to true grid velocity (momentum / mass).
    pub momentum: Vec2,
    pub mass: f32,
}

#[derive(Debug)]
pub struct Grid {
    resolution: usize,
    cells: Vec<Cell>,
}

impl Grid {
    pub fn new(resolution: usize) -> Self {
        assert!(resolution >= 4, "grid resolution must be >= 4");
        Self {
            resolution,
            cells: vec![Cell::default(); resolution * resolution],
        }
    }

    pub fn resolution(&self) -> usize {
        self.resolution
    }

    pub fn cells(&self) -> &[Cell] {
        &self.cells
    }

    pub fn cells_mut(&mut self) -> &mut [Cell] {
        &mut self.cells
    }

    pub fn clear(&mut self) {
        self.cells.fill(Cell::default());
    }

    pub fn index(&self, cell_pos: IVec2) -> usize {
        debug_assert!(cell_pos.x >= 0);
        debug_assert!(cell_pos.y >= 0);
        let x = cell_pos.x as usize;
        let y = cell_pos.y as usize;
        debug_assert!(x < self.resolution);
        debug_assert!(y < self.resolution);
        x * self.resolution + y
    }

    /// Accumulate mass and momentum onto a cell during P2G scatter.
    /// `cell.momentum` holds accumulated momentum until `update_velocities` normalizes it.
    pub fn add_mass_momentum(&mut self, cell_pos: IVec2, mass: f32, momentum: Vec2) {
        let idx = self.index(cell_pos);
        let cell = &mut self.cells[idx];
        cell.mass += mass;
        cell.momentum += momentum;
    }

    pub fn velocity_at(&self, cell_pos: IVec2) -> Vec2 {
        self.cells[self.index(cell_pos)].momentum
    }

    pub fn mass_at(&self, cell_pos: IVec2) -> f32 {
        self.cells[self.index(cell_pos)].mass
    }

    /// Normalize accumulated momentum to velocity (divide by mass), then apply gravity.
    /// After this call, `cell.momentum` holds true grid velocity and can be read by G2P gather.
    pub fn update_velocities(&mut self, dt: f32, gravity: f32) {
        for cell in self.cells.iter_mut() {
            if cell.mass > 0.0 {
                cell.momentum /= cell.mass;
                cell.momentum += Vec2::new(0.0, gravity) * dt;
            }
        }
    }

    /// View the cell buffer as raw bytes for wgpu buffer upload.
    ///
    /// # Safety
    /// `Cell` is `repr(C)` with only glam/f32 fields — no pointer fields, no uninit bytes
    /// in practice. Safe for GPU upload. Do not use to reconstruct `Cell` on CPU.
    pub fn cells_as_bytes(&self) -> &[u8] {
        unsafe {
            core::slice::from_raw_parts(
                self.cells.as_ptr() as *const u8,
                self.cells.len() * core::mem::size_of::<Cell>(),
            )
        }
    }
}

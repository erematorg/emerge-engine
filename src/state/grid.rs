use glam::{IVec2, Vec2};

#[derive(Clone, Copy, Debug, Default)]
pub struct Cell {
    /// During P2G scatter: accumulated momentum (mass × velocity).
    /// After `update_velocities`: normalized to true velocity (momentum / mass).
    /// Never read directly — use the grid's accessor methods which know which phase you're in.
    pub v: Vec2,
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
    /// `v` stores momentum (not velocity) until `update_velocities` normalizes it.
    pub fn add_mass_momentum(&mut self, cell_pos: IVec2, mass: f32, momentum: Vec2) {
        let idx = self.index(cell_pos);
        let cell = &mut self.cells[idx];
        cell.mass += mass;
        cell.v += momentum;
    }

    pub fn velocity_at(&self, cell_pos: IVec2) -> Vec2 {
        self.cells[self.index(cell_pos)].v
    }

    pub fn mass_at(&self, cell_pos: IVec2) -> f32 {
        self.cells[self.index(cell_pos)].mass
    }

    /// Normalize accumulated momentum to velocity (divide by mass), then apply gravity.
    /// After this call, `cell.v` is a true velocity and can be read by G2P gather.
    pub fn update_velocities(&mut self, dt: f32, gravity: f32) {
        for cell in self.cells.iter_mut() {
            if cell.mass > 0.0 {
                cell.v /= cell.mass;
                cell.v += Vec2::new(0.0, gravity) * dt;
            }
        }
    }
}

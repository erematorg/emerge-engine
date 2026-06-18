use glam::Vec2;
use std::collections::HashMap;

/// Flat spatial hash over active particles.
///
/// Particles live in grid-coordinate space (same units as `SimConfig::grid_cell_size`).
/// `cell_size` should match the MPM grid cell width so the natural particle spacing
/// (roughly one particle per cell) puts ≈1 particle per bucket.
///
/// Rebuild once per substep after G2P (positions are final for that substep).
/// Query with `query(center, radius)` — iterates candidate indices; caller does
/// exact distance filtering.
pub(crate) struct SpatialHash {
    inv_cell: f32,
    /// (cx, cy) → particle indices.
    table: HashMap<(i32, i32), Vec<usize>>,
}

impl SpatialHash {
    pub fn new(cell_size: f32) -> Self {
        debug_assert!(cell_size > 0.0, "cell_size must be positive");
        Self {
            inv_cell: 1.0 / cell_size,
            table: HashMap::new(),
        }
    }

    /// Rebuild from the active partition `positions[0..active_count]`.
    ///
    /// Clears and repopulates in one O(active_count) pass.
    /// Reuses existing bucket allocations to avoid repeated heap churn.
    pub fn rebuild(&mut self, positions: &[Vec2], active_count: usize) {
        // Clear buckets without deallocating their inner Vecs.
        for v in self.table.values_mut() {
            v.clear();
        }
        for (i, &pos) in positions.iter().enumerate().take(active_count) {
            let c = self.cell_of(pos);
            self.table.entry(c).or_default().push(i);
        }
    }

    /// Iterate candidate particle indices within `radius` of `center`.
    ///
    /// Covers all cells whose bounding box overlaps the query circle.
    /// Caller is responsible for exact distance filtering.
    pub fn query(&self, center: Vec2, radius: f32) -> impl Iterator<Item = usize> + '_ {
        let r_cells = (radius * self.inv_cell).ceil() as i32;
        let (cx, cy) = self.cell_of(center);
        SpatialHashIter {
            hash: self,
            cx,
            cy,
            r_cells,
            gx: -r_cells,
            gy: -r_cells,
            bucket_pos: 0,
        }
    }

    #[inline(always)]
    fn cell_of(&self, p: Vec2) -> (i32, i32) {
        (
            (p.x * self.inv_cell).floor() as i32,
            (p.y * self.inv_cell).floor() as i32,
        )
    }
}

struct SpatialHashIter<'a> {
    hash: &'a SpatialHash,
    cx: i32,
    cy: i32,
    r_cells: i32,
    gx: i32,
    gy: i32,
    bucket_pos: usize,
}

impl<'a> Iterator for SpatialHashIter<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        loop {
            if self.gx > self.r_cells {
                return None;
            }
            let cell = (self.cx + self.gx, self.cy + self.gy);
            if let Some(bucket) = self.hash.table.get(&cell)
                && self.bucket_pos < bucket.len()
            {
                let idx = bucket[self.bucket_pos];
                self.bucket_pos += 1;
                return Some(idx);
            }
            // Advance to next cell.
            self.bucket_pos = 0;
            self.gy += 1;
            if self.gy > self.r_cells {
                self.gy = -self.r_cells;
                self.gx += 1;
            }
        }
    }
}

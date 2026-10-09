//! A multigrid V-cycle used as the preconditioner of the conjugate
//! gradient (McAdams, Sifakis and Teran, *A parallel multigrid Poisson
//! solver for fluids simulation on large grids*, SCA 2010), in 2D.
//!
//! Why it replaces MIC(0) at scale: their table 1 counts the iterations to
//! a 1e-4 residual reduction at 11 to 13 from 64^3 to 512^3, where
//! incomplete Cholesky needs 72 to 395, and every step of the cycle is a
//! sweep over cells in any order, where MIC(0)'s triangular solves run one
//! cell after another.
//!
//! What is taken from the paper, in 2D:
//!
//! - The cycle (their algorithm 1) with a zero initial guess on every
//!   level, as a preconditioner must start.
//! - Coarsening by 2 x 2 cells: a coarse cell is air (Dirichlet) if any of
//!   its children is, else liquid if any child is, else solid (Neumann).
//!   Outside the grid is solid.
//! - On the coarse levels the voxelized operator (their eq. 2): per open
//!   neighbour, one coupling of the level's `1 / h^2` and the same on the
//!   diagonal, nothing for a solid neighbour.
//! - Restriction `B (x) B` with `B = (1, 3, 3, 1) / 8` over cell-centred
//!   values, prolongation its transpose times 4, which is bilinear
//!   interpolation; both only into liquid cells, reading zero elsewhere.
//! - Damped Jacobi with `omega = 2/3` as the smoother, the same before and
//!   after each correction, and many sweeps as the coarsest solve. With
//!   transposed transfers and a symmetric smoother the cycle is a
//!   symmetric positive definite operator (their 3.3), which conjugate
//!   gradient needs.
//!
//! What differs: the finest level smooths with the system actually being
//! solved (its open-face weights and ghost-fluid diagonal, Batty, Bertails
//! and Bridson 2007), not a voxelized copy; the coarse levels only have to
//! approximate it, which is all a preconditioner asks. The paper's extra
//! Gauss-Seidel sweeps near the boundary are left out: they speed
//! convergence, they are not needed for it (their section 4), and Jacobi
//! alone keeps every sweep order independent.

use rayon::prelude::*;

use super::pressure::PressureSystem;

/// Rows a parallel task takes at least: below it the threads cost more
/// than the cells do.
const ROWS_PER_TASK: usize = 16;

/// Grids smaller than this run their sweeps on the calling thread: a cycle
/// is hundreds of short passes, and on the 72-cell gate scenes handing each
/// to the thread pool made the whole run 75 % slower. The arithmetic is
/// the same either way.
pub(super) const PARALLEL_MIN_CELLS: usize = 128 * 128;

/// Runs `f(row index, row)` over the rows of `out`, in parallel when the
/// grid is large enough.
pub(super) fn for_rows(out: &mut [f32], width: usize, f: impl Fn(usize, &mut [f32]) + Sync + Send) {
    if out.len() >= PARALLEL_MIN_CELLS {
        out.par_chunks_mut(width)
            .with_min_len(ROWS_PER_TASK)
            .enumerate()
            .for_each(|(j, row)| f(j, row));
    } else {
        out.chunks_mut(width)
            .enumerate()
            .for_each(|(j, row)| f(j, row));
    }
}

const OMEGA: f32 = 2.0 / 3.0;

/// Gauss-Seidel sweeps over the boundary band at the finest level,
/// doubled at each coarser one (McAdams et al. section 4: two at the
/// finest level "struck the best balance").
const BAND_SWEEPS: u32 = 2;

/// Cells of a level within an L1 distance of 2 of a cell that is not
/// liquid, or of the grid's edge (their figure 3 draws this band for the
/// stand-alone solver; their preconditioner's own band, from the
/// prolongation stencil, is one to three cells wide). Listed in row order.
fn boundary_band(nx: usize, ny: usize, liquid: impl Fn(usize) -> bool) -> Vec<usize> {
    let mut band = Vec::new();
    for j in 0..ny {
        for i in 0..nx {
            let c = i + nx * j;
            if !liquid(c) {
                continue;
            }
            let near = (-2isize..=2).any(|dj| {
                (-2isize..=2).any(|di| {
                    if di.abs() + dj.abs() > 2 {
                        return false;
                    }
                    let (a, b) = (i as isize + di, j as isize + dj);
                    a < 0
                        || b < 0
                        || a as usize >= nx
                        || b as usize >= ny
                        || !liquid(a as usize + nx * b as usize)
                })
            });
            if near {
                band.push(c);
            }
        }
    }
    band
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Solid,
    Liquid,
    Air,
}

/// One coarse level: its cells and the coefficient of one open face.
struct Level {
    nx: usize,
    ny: usize,
    kind: Vec<Kind>,
    /// `dt / h^2` at this level's spacing.
    coupling: f32,
    /// Diagonal of the voxelized operator, zero outside the liquid.
    diag: Vec<f32>,
    /// Liquid cells near a non-liquid one (`boundary_band`).
    band: Vec<usize>,
}

impl Level {
    fn liquid(&self, i: isize, j: isize) -> bool {
        i >= 0
            && j >= 0
            && (i as usize) < self.nx
            && (j as usize) < self.ny
            && self.kind[i as usize + self.nx * j as usize] == Kind::Liquid
    }

    fn open(&self, i: isize, j: isize) -> bool {
        i >= 0
            && j >= 0
            && (i as usize) < self.nx
            && (j as usize) < self.ny
            && self.kind[i as usize + self.nx * j as usize] != Kind::Solid
    }

    fn new(nx: usize, ny: usize, kind: Vec<Kind>, coupling: f32, extra_diag: f32) -> Self {
        let mut level = Self {
            nx,
            ny,
            kind,
            coupling,
            diag: vec![0.0; nx * ny],
            band: Vec::new(),
        };
        for j in 0..ny {
            for i in 0..nx {
                let c = i + nx * j;
                if level.kind[c] != Kind::Liquid {
                    continue;
                }
                let (i, j) = (i as isize, j as isize);
                let open = [(1, 0), (-1, 0), (0, 1), (0, -1)]
                    .iter()
                    .filter(|(di, dj)| level.open(i + di, j + dj))
                    .count();
                level.diag[c] = open as f32 * coupling + extra_diag;
            }
        }
        level.band = boundary_band(nx, ny, |c| level.kind[c] == Kind::Liquid);
        level
    }

    /// Gauss-Seidel over the band, forward or backward.
    fn gauss_seidel(&self, b: &[f32], u: &mut [f32], forward: bool, sweeps: u32) {
        for _ in 0..sweeps {
            let mut visit = |c: usize| {
                if self.diag[c] <= 0.0 {
                    return;
                }
                let (i, j) = ((c % self.nx) as isize, (c / self.nx) as isize);
                let mut au = self.diag[c] * u[c];
                for (di, dj) in [(1isize, 0isize), (-1, 0), (0, 1), (0, -1)] {
                    if self.liquid(i + di, j + dj) {
                        au -= self.coupling * u[(i + di) as usize + self.nx * (j + dj) as usize];
                    }
                }
                u[c] += (b[c] - au) / self.diag[c];
            };
            if forward {
                self.band.iter().for_each(|&c| visit(c));
            } else {
                self.band.iter().rev().for_each(|&c| visit(c));
            }
        }
    }

    /// `out = b - A u` on the liquid cells, zero elsewhere. Rows in
    /// parallel: every cell's value is its own.
    fn residual(&self, u: &[f32], b: &[f32], out: &mut [f32]) {
        for_rows(out, self.nx, |j, row| {
            for (i, slot) in row.iter_mut().enumerate() {
                let c = i + self.nx * j;
                if self.kind[c] != Kind::Liquid {
                    *slot = 0.0;
                    continue;
                }
                let mut au = self.diag[c] * u[c];
                let (ii, jj) = (i as isize, j as isize);
                for (di, dj) in [(1isize, 0isize), (-1, 0), (0, 1), (0, -1)] {
                    if self.liquid(ii + di, jj + dj) {
                        let n = (ii + di) as usize + self.nx * (jj + dj) as usize;
                        au -= self.coupling * u[n];
                    }
                }
                *slot = b[c] - au;
            }
        });
    }
}

/// The hierarchy built once per solve from the system's cells, with the
/// work vectors every cycle reuses.
pub struct Multigrid {
    levels: Vec<Level>,
    /// Per coarse level: right-hand side, solution, residual, scratch.
    work: Vec<[Vec<f32>; 4]>,
    /// Fine level: residual and scratch.
    fine_work: [Vec<f32>; 2],
    /// Active cells of the fine level near a non-active one.
    fine_band: Vec<usize>,
    /// Damped Jacobi sweeps before and after each correction.
    pub sweeps: u32,
    /// Sweeps that stand in for the solve on the coarsest level.
    pub coarse_sweeps: u32,
}

impl Multigrid {
    /// Coarsens until either side is at most `coarsest` cells.
    pub fn new(sys: &PressureSystem, coarsest: usize, sweeps: u32, coarse_sweeps: u32) -> Self {
        let mut levels = Vec::new();
        let (mut nx, mut ny) = (sys.nx, sys.ny);
        let mut kind: Vec<Kind> = (0..nx * ny)
            .map(|c| {
                if sys.active[c] {
                    Kind::Liquid
                } else if sys.air[c] {
                    Kind::Air
                } else {
                    Kind::Solid
                }
            })
            .collect();
        let mut coupling = sys.scale;
        while nx > coarsest && ny > coarsest {
            let (cx, cy) = (nx.div_ceil(2), ny.div_ceil(2));
            let mut coarse = vec![Kind::Solid; cx * cy];
            for cj in 0..cy {
                for ci in 0..cx {
                    let mut any_air = false;
                    let mut any_liquid = false;
                    for (di, dj) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                        let (i, j) = (2 * ci + di, 2 * cj + dj);
                        if i < nx && j < ny {
                            match kind[i + nx * j] {
                                Kind::Air => any_air = true,
                                Kind::Liquid => any_liquid = true,
                                Kind::Solid => {}
                            }
                        }
                    }
                    coarse[ci + cx * cj] = if any_air {
                        Kind::Air
                    } else if any_liquid {
                        Kind::Liquid
                    } else {
                        Kind::Solid
                    };
                }
            }
            coupling *= 0.25;
            levels.push(Level::new(
                cx,
                cy,
                coarse.clone(),
                coupling,
                sys.compressibility,
            ));
            (nx, ny, kind) = (cx, cy, coarse);
        }
        let work = levels
            .iter()
            .map(|level| {
                let n = level.nx * level.ny;
                [vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]]
            })
            .collect();
        let n0 = sys.nx * sys.ny;
        Self {
            levels,
            work,
            fine_work: [vec![0.0; n0], vec![0.0; n0]],
            fine_band: boundary_band(sys.nx, sys.ny, |c| sys.active[c]),
            sweeps,
            coarse_sweeps,
        }
    }

    /// `z = M r`: one V-cycle from zero for `A z = r`.
    pub fn apply(&mut self, sys: &PressureSystem, r: &[f32], z: &mut [f32]) {
        z.iter_mut().for_each(|v| *v = 0.0);
        let [residual0, scratch0] = &mut self.fine_work;
        jacobi_fine(sys, r, z, scratch0, self.sweeps);
        gauss_seidel_fine(sys, r, z, &self.fine_band, true, BAND_SWEEPS);
        if self.levels.is_empty() {
            gauss_seidel_fine(sys, r, z, &self.fine_band, false, BAND_SWEEPS);
            jacobi_fine(sys, r, z, scratch0, self.sweeps);
            return;
        }
        fine_residual(sys, z, r, residual0);
        let last = self.levels.len() - 1;
        // Downstroke: each level's right-hand side is the restricted
        // residual of the level above.
        for l in 0..self.levels.len() {
            let (above, rest) = self.work.split_at_mut(l);
            let [b, u, res, scratch] = &mut rest[0];
            let level = &self.levels[l];
            b.iter_mut().for_each(|v| *v = 0.0);
            u.iter_mut().for_each(|v| *v = 0.0);
            if l == 0 {
                restrict(residual0, sys.nx, sys.ny, level, b);
            } else {
                let fine = &self.levels[l - 1];
                restrict(&above[l - 1][2], fine.nx, fine.ny, level, b);
            }
            let count = if l == last {
                self.coarse_sweeps
            } else {
                self.sweeps
            };
            jacobi_coarse(level, b, u, scratch, count);
            if l != last {
                level.gauss_seidel(b, u, true, BAND_SWEEPS << (l + 1));
                level.residual(u, b, res);
            }
        }
        // Upstroke.
        for l in (0..self.levels.len()).rev() {
            if l == 0 {
                prolongate_fine(&self.work[0][1], &self.levels[0], sys, z);
                gauss_seidel_fine(sys, r, z, &self.fine_band, false, BAND_SWEEPS);
                jacobi_fine(sys, r, z, scratch0, self.sweeps);
            } else {
                let (above, rest) = self.work.split_at_mut(l);
                let correction = &rest[0][1];
                let [b, u, _, scratch] = &mut above[l - 1];
                let fine = &self.levels[l - 1];
                prolongate(correction, &self.levels[l], fine, u);
                fine.gauss_seidel(b, u, false, BAND_SWEEPS << l);
                jacobi_coarse(fine, b, u, scratch, self.sweeps);
            }
        }
    }
}

/// Gauss-Seidel over the fine band on the system being solved.
fn gauss_seidel_fine(
    sys: &PressureSystem,
    b: &[f32],
    u: &mut [f32],
    band: &[usize],
    forward: bool,
    sweeps: u32,
) {
    let nx = sys.nx;
    for _ in 0..sweeps {
        let mut visit = |c: usize| {
            if sys.diag[c] <= 0.0 {
                return;
            }
            let (i, j) = (c % nx, c / nx);
            let mut au = sys.diag[c] * u[c];
            if i + 1 < nx {
                au += sys.plus_i[c] * u[c + 1];
            }
            if i > 0 {
                au += sys.plus_i[c - 1] * u[c - 1];
            }
            if j + 1 < sys.ny {
                au += sys.plus_j[c] * u[c + nx];
            }
            if j > 0 {
                au += sys.plus_j[c - nx] * u[c - nx];
            }
            u[c] += (b[c] - au) / sys.diag[c];
        };
        if forward {
            band.iter().for_each(|&c| visit(c));
        } else {
            band.iter().rev().for_each(|&c| visit(c));
        }
    }
}

/// Damped Jacobi on the system being solved.
fn jacobi_fine(sys: &PressureSystem, b: &[f32], u: &mut [f32], scratch: &mut [f32], sweeps: u32) {
    for _ in 0..sweeps {
        fine_residual(sys, u, b, scratch);
        let nx = sys.nx;
        let scratch = &*scratch;
        for_rows(u, nx, |j, row| {
            for (i, value) in row.iter_mut().enumerate() {
                let c = i + nx * j;
                if sys.active[c] && sys.diag[c] > 0.0 {
                    *value += OMEGA * scratch[c] / sys.diag[c];
                }
            }
        });
    }
}

fn jacobi_coarse(level: &Level, b: &[f32], u: &mut [f32], scratch: &mut [f32], sweeps: u32) {
    for _ in 0..sweeps {
        level.residual(u, b, scratch);
        let nx = level.nx;
        let scratch = &*scratch;
        for_rows(u, nx, |j, row| {
            for (i, value) in row.iter_mut().enumerate() {
                let c = i + nx * j;
                if level.kind[c] == Kind::Liquid && level.diag[c] > 0.0 {
                    *value += OMEGA * scratch[c] / level.diag[c];
                }
            }
        });
    }
}

/// `out = b - A u` for the fine system, zero off the active cells.
fn fine_residual(sys: &PressureSystem, u: &[f32], b: &[f32], out: &mut [f32]) {
    let nx = sys.nx;
    for_rows(out, nx, |j, row| fine_residual_row(sys, u, b, j, row));
}

fn fine_residual_row(sys: &PressureSystem, u: &[f32], b: &[f32], j: usize, out: &mut [f32]) {
    let nx = sys.nx;
    {
        for (i, slot) in out.iter_mut().enumerate() {
            let c = i + nx * j;
            if !sys.active[c] {
                *slot = 0.0;
                continue;
            }
            let mut au = sys.diag[c] * u[c];
            if i + 1 < nx {
                au += sys.plus_i[c] * u[c + 1];
            }
            if i > 0 {
                au += sys.plus_i[c - 1] * u[c - 1];
            }
            if j + 1 < sys.ny {
                au += sys.plus_j[c] * u[c + nx];
            }
            if j > 0 {
                au += sys.plus_j[c - nx] * u[c - nx];
            }
            *slot = b[c] - au;
        }
    }
}

/// 1D weights of `B` and the fine offsets they apply to, for the coarse
/// cell `I`: fine cells `2I - 1 ..= 2I + 2`.
const B: [(isize, f32); 4] = [(-1, 0.125), (0, 0.375), (1, 0.375), (2, 0.125)];

fn restrict(fine: &[f32], fnx: usize, fny: usize, coarse: &Level, out: &mut [f32]) {
    for_rows(out, coarse.nx, |cj, row| {
        restrict_row(fine, fnx, fny, coarse, cj, row)
    });
}

fn restrict_row(fine: &[f32], fnx: usize, fny: usize, coarse: &Level, cj: usize, out: &mut [f32]) {
    {
        for (ci, slot) in out.iter_mut().enumerate() {
            let c = ci + coarse.nx * cj;
            if coarse.kind[c] != Kind::Liquid {
                continue;
            }
            let mut sum = 0.0;
            for &(dj, wj) in &B {
                let j = 2 * cj as isize + dj;
                if j < 0 || j as usize >= fny {
                    continue;
                }
                for &(di, wi) in &B {
                    let i = 2 * ci as isize + di;
                    if i < 0 || i as usize >= fnx {
                        continue;
                    }
                    sum += wi * wj * fine[i as usize + fnx * j as usize];
                }
            }
            *slot = sum;
        }
    }
}

/// Adds `4 B^T` of the coarse values to the liquid fine cells: bilinear
/// interpolation, the transpose of `restrict` up to that factor. Written
/// as each fine cell gathering from the coarse cells whose stencil covers
/// it (fine `f` is in coarse `I`'s stencil when `f - 2I` is one of `B`'s
/// offsets), so rows can run in parallel.
fn prolongate_into(
    coarse: &[f32],
    cl: &Level,
    fnx: usize,
    fine_liquid: impl Fn(usize) -> bool + Sync,
    out: &mut [f32],
) {
    let parents = |f: usize, len: usize| {
        let mut found = [(0usize, 0.0f32); 2];
        let mut count = 0;
        for &(offset, w) in &B {
            let twice = f as isize - offset;
            if twice >= 0 && twice % 2 == 0 && ((twice / 2) as usize) < len {
                found[count] = ((twice / 2) as usize, w);
                count += 1;
            }
        }
        (found, count)
    };
    for_rows(out, fnx, |j, row| {
        let (pj, nj) = parents(j, cl.ny);
        for (i, slot) in row.iter_mut().enumerate() {
            if !fine_liquid(i + fnx * j) {
                continue;
            }
            let (pi, ni) = parents(i, cl.nx);
            let mut sum = 0.0;
            for &(cj, wj) in &pj[..nj] {
                for &(ci, wi) in &pi[..ni] {
                    sum += 4.0 * wi * wj * coarse[ci + cl.nx * cj];
                }
            }
            *slot += sum;
        }
    });
}

fn prolongate(coarse: &[f32], cl: &Level, fine: &Level, out: &mut [f32]) {
    prolongate_into(coarse, cl, fine.nx, |f| fine.kind[f] == Kind::Liquid, out);
}

fn prolongate_fine(coarse: &[f32], cl: &Level, sys: &PressureSystem, out: &mut [f32]) {
    prolongate_into(coarse, cl, sys.nx, |f| sys.active[f], out);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The 5-point Laplacian on an `n x n` liquid block closed by solid
    /// walls, with one air cell at a corner so the system is definite.
    fn walled_block(n: usize) -> PressureSystem {
        let size = n * n;
        let mut sys = PressureSystem {
            nx: n,
            ny: n,
            active: vec![true; size],
            diag: vec![0.0; size],
            plus_i: vec![0.0; size],
            plus_j: vec![0.0; size],
            rhs: vec![0.0; size],
            air: vec![false; size],
            scale: 1.0,
            compressibility: 0.0,
        };
        sys.active[0] = false;
        sys.air[0] = true;
        for j in 0..n {
            for i in 0..n {
                let c = i + n * j;
                if !sys.active[c] {
                    continue;
                }
                for (di, dj) in [(1isize, 0isize), (-1, 0), (0, 1), (0, -1)] {
                    let (a, b) = (i as isize + di, j as isize + dj);
                    if a < 0 || b < 0 || a as usize >= n || b as usize >= n {
                        continue;
                    }
                    sys.diag[c] += 1.0;
                    let neighbour = a as usize + n * b as usize;
                    if sys.active[neighbour] {
                        if di == 1 {
                            sys.plus_i[c] = -1.0;
                        }
                        if dj == 1 {
                            sys.plus_j[c] = -1.0;
                        }
                    }
                }
            }
        }
        sys
    }

    fn dot(a: &[f32], b: &[f32]) -> f64 {
        a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum()
    }

    /// Conjugate gradient needs the preconditioner symmetric and positive
    /// definite: `x . M y = y . M x` and `x . M x > 0`.
    #[test]
    fn the_cycle_is_symmetric_and_positive() {
        let sys = walled_block(20);
        let mut mg = Multigrid::new(&sys, 3, 2, 30);
        let n = 400;
        let x: Vec<f32> = (0..n)
            .map(|k| {
                if sys.active[k] {
                    ((k * 37) % 11) as f32 - 5.0
                } else {
                    0.0
                }
            })
            .collect();
        let y: Vec<f32> = (0..n)
            .map(|k| {
                if sys.active[k] {
                    ((k * 13) % 7) as f32 - 3.0
                } else {
                    0.0
                }
            })
            .collect();
        let (mut mx, mut my) = (vec![0.0f32; n], vec![0.0f32; n]);
        mg.apply(&sys, &x, &mut mx);
        mg.apply(&sys, &y, &mut my);
        let (xy, yx) = (dot(&y, &mx), dot(&x, &my));
        assert!((xy - yx).abs() <= 1e-4 * xy.abs().max(1.0), "{xy} vs {yx}");
        assert!(dot(&x, &mx) > 0.0);
        assert!(dot(&y, &my) > 0.0);
    }
}

/// Solve time against grid size, MIC(0) against the V-cycle: a basin
/// three quarters full with air above and solid walls, random right-hand
/// side. A probe.
#[cfg(test)]
mod scaling_probe {
    use super::super::pcg::{Preconditioner, SolverSettings, solve};
    use super::super::pressure::PressureSystem;

    fn basin(n: usize) -> PressureSystem {
        let size = n * n;
        let mut sys = PressureSystem {
            nx: n,
            ny: n,
            active: vec![false; size],
            diag: vec![0.0; size],
            plus_i: vec![0.0; size],
            plus_j: vec![0.0; size],
            rhs: vec![0.0; size],
            air: vec![false; size],
            scale: 1.0,
            compressibility: 0.0,
        };
        let surface = 3 * n / 4;
        for j in 0..n {
            for i in 0..n {
                let c = i + n * j;
                sys.active[c] = j < surface;
                sys.air[c] = j >= surface;
            }
        }
        for j in 0..surface {
            for i in 0..n {
                let c = i + n * j;
                for (di, dj) in [(1isize, 0isize), (-1, 0), (0, 1), (0, -1)] {
                    let (a, b) = (i as isize + di, j as isize + dj);
                    if a < 0 || b < 0 || a as usize >= n || b as usize >= n {
                        continue;
                    }
                    sys.diag[c] += 1.0;
                    let k = a as usize + n * b as usize;
                    if sys.active[k] {
                        if di == 1 {
                            sys.plus_i[c] = -1.0;
                        }
                        if dj == 1 {
                            sys.plus_j[c] = -1.0;
                        }
                    }
                }
                sys.rhs[c] = ((c * 7919) % 1000) as f32 / 1000.0 - 0.5;
            }
        }
        sys
    }

    #[test]
    #[ignore = "solver scaling probe: run with --ignored --nocapture"]
    fn probe_solver_scaling() {
        for n in [24usize, 32, 48, 64, 96, 128, 256, 512, 1024] {
            let sys = basin(n);
            // Small solves repeated, so their time is not one call's noise.
            let reps = if n <= 128 { 20 } else { 1 };
            for p in [Preconditioner::Mic0, Preconditioner::Multigrid] {
                let settings = SolverSettings {
                    preconditioner: p,
                    max_iterations: 2000,
                    ..SolverSettings::default()
                };
                let t = std::time::Instant::now();
                let mut s = solve(&sys, &settings);
                for _ in 1..reps {
                    s = solve(&sys, &settings);
                }
                println!(
                    "{n}^2 ({} unknowns) {p:?}: {} iterations, {:.3} ms, converged {}",
                    sys.unknowns(),
                    s.iterations,
                    t.elapsed().as_secs_f64() * 1e3 / f64::from(reps),
                    s.converged
                );
            }
        }
    }
}

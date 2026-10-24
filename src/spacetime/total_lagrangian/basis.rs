//! The grid the step moves: where each velocity value lives (the lattices),
//! how a particle reads and writes it (quadratic B-splines on the reference
//! configuration) and which grid values are kept or folded into others
//! (the extended B-splines).

use super::*;

/// Quadratic B-spline weights of one particle on one lattice: the first
/// face, three weights and their derivatives per axis.
#[derive(Clone, Copy)]
pub(crate) struct Stencil {
    pub(crate) base: (i64, i64),
    pub(crate) wx: [f64; 3],
    pub(crate) wy: [f64; 3],
    pub(crate) dx: [f64; 3],
    pub(crate) dy: [f64; 3],
}

pub(crate) fn axis(coord: f64, offset: f64, h: f64) -> (i64, [f64; 3], [f64; 3]) {
    let s = coord / h - offset;
    let base = (s - 0.5).floor();
    let f = s - base;
    (
        base as i64,
        [
            0.5 * (1.5 - f).powi(2),
            0.75 - (f - 1.0).powi(2),
            0.5 * (f - 0.5).powi(2),
        ],
        [-(1.5 - f) / h, -2.0 * (f - 1.0) / h, (f - 0.5) / h],
    )
}

/// A lattice: its size, offset in cells and where its faces start in the
/// stacked vector of unknowns.
#[derive(Clone, Copy)]
pub(crate) struct Lattice {
    pub(crate) ni: usize,
    pub(crate) nj: usize,
    pub(crate) offset: DVec2,
    pub(crate) start: usize,
}

/// The two velocity components' lattices on an `nx` by `ny` cell domain:
/// both on the cells' corners, each node carrying both components (a
/// collocated grid). The midpoint step keeps the angular momentum only
/// there: it needs each particle's midpoint position in the span of the
/// basis that carries the velocity component it multiplies, which a
/// staggered grid breaks (measured, `gates::probe_disc_impact`: about
/// `6e-3 x strain` of `M v0 R` per impact on the MAC grid, round-off
/// collocated; the proofs of Love and Sulsky and of Jiang, Schroeder and
/// Teran, JCP 338 (2017), sec. 5, are all collocated). A probe may restore
/// the MAC grid's lattices (`hooks::staggered`) to compare.
pub(crate) fn lattices(nx: usize, ny: usize) -> [Lattice; 2] {
    #[cfg(all(test, feature = "experimental"))]
    if hooks::staggered() {
        return [
            Lattice {
                ni: nx + 1,
                nj: ny,
                offset: DVec2::new(0.0, 0.5),
                start: 0,
            },
            Lattice {
                ni: nx,
                nj: ny + 1,
                offset: DVec2::new(0.5, 0.0),
                start: (nx + 1) * ny,
            },
        ];
    }
    let u = Lattice {
        ni: nx + 1,
        nj: ny + 1,
        offset: DVec2::ZERO,
        start: 0,
    };
    [
        u,
        Lattice {
            start: u.ni * u.nj,
            ..u
        },
    ]
}

/// Lattice `a`'s offset in cells (`lattices`): its node `(i, j)` sits at
/// `(i, j) + offset`, for the walls' `Held` closures.
#[cfg(all(test, feature = "experimental"))]
pub(crate) fn lattice_offset(a: usize) -> DVec2 {
    lattices(1, 1)[a].offset
}

/// The unknowns are the faces with some mass that no wall holds, numbered
/// compactly: every vector below has one entry per unknown, the faces
/// outside them being zero.
#[derive(Clone)]
pub(crate) struct Grid {
    /// The cell's width, in the step's length unit: the length scale
    /// of its finite differences.
    pub(crate) cell: f64,
    pub(crate) lattices: [Lattice; 2],
    /// How many unknowns.
    pub(crate) len: usize,
    /// Per particle, per lattice.
    pub(crate) stencils: Vec<[Stencil; 2]>,
    /// The unknowns the solvers move; a wall face holding its material is
    /// not (its value stays zero).
    pub(crate) active: Vec<bool>,
    /// A wall face's inward normal sign along its own component, per
    /// unknown.
    pub(crate) wall: Vec<Option<f64>>,
    /// Each unknown's face in the stacked lattices, and whether it is
    /// stable (`Grid::new`): a wall face that is not, once its wall lets it
    /// go, is extrapolated like any degenerate face rather than left a
    /// free unknown with a sliver of support.
    pub(crate) stacked_of: Vec<usize>,
    pub(crate) stable_unknown: Vec<bool>,
    /// The unknown of each face of the stacked lattices, `usize::MAX` for
    /// none.
    pub(crate) slot: Vec<usize>,
    /// The lattice (velocity component) of each unknown.
    pub(crate) axis_of: Vec<usize>,
    /// For a degenerate face of the extended B-splines, the unknowns its
    /// coefficient is extrapolated from and their weights (`Grid::new`).
    pub(crate) extension: Vec<Option<Extension>>,
    pub(crate) lumped: Vec<f64>,
    /// The consistent mass `M_c`, assembled at its first product and kept
    /// with the masses it was built from (`Grid::consistent_mass_times`):
    /// the grid stays on the reference configuration, so the matrix does
    /// not change while the masses do not. Shared by the grid's clones (the
    /// controller's tries), which are the same grid.
    pub(crate) mass_matrix: std::sync::Arc<std::sync::Mutex<Option<MassMatrix>>>,
    /// Per particle, per lattice, the weight function of the web-splines
    /// and its gradient at the particle's reference position, when the
    /// basis is weighted (`LagrangianCache::weighted`).
    #[cfg(all(test, feature = "experimental"))]
    pub(crate) weights: Option<Vec<[(f64, DVec2); 2]>>,
    /// Per unknown, whether it carries no inertia (`hooks::set_massless`).
    #[cfg(all(test, feature = "experimental"))]
    pub(crate) massless: Option<Vec<bool>>,
}

/// The unknowns a degenerate face's coefficient is extrapolated from, with
/// their weights.
pub(crate) type Extension = Box<[(usize, f64)]>;

/// The consistent mass in compressed rows, `M_c[k][m] = sum_p m_p N_k(X_p)
/// N_m(X_p)` over each lattice (the two components do not couple), and the
/// particle masses it was assembled from.
#[derive(Debug, Default)]
pub(crate) struct MassMatrix {
    mass: Vec<f64>,
    start: Vec<usize>,
    column: Vec<usize>,
    value: Vec<f64>,
}

impl Grid {
    /// The grid of one step. With `occupation` the basis is the extended
    /// B-splines of Kakouris, Chatzis and Triantafyllou, *A high-fidelity
    /// material point method for frictional contact problems*, arXiv
    /// 2403.13534 (2024), section 3.3.1, who take them from Yamaguchi,
    /// Moriguchi and Terada 2021 (not read):
    /// - each knot span's volume fraction is the particles' volume in it
    ///   over its own (their eq. 19), interior above `C_c`, boundary
    ///   between 0 and `C_c` (eq. 20);
    /// - a face whose support holds an interior span is stable, one whose
    ///   support holds only boundary spans degenerate;
    /// - a degenerate face's coefficient is the Lagrange extrapolation of
    ///   the nearest 3 x 3 block of stable ones (eqs. 22, 23), so its
    ///   B-spline is folded into theirs (eq. 25).
    ///
    /// The extrapolation reproduces quadratics, so the basis keeps the
    /// partition of unity and reproduces linear fields (both momenta), and
    /// the step's mass stays the consistent one in this basis, `E^T M E`,
    /// with no term added, so it costs no energy (a ghost penalty added to
    /// the mass took 26 to 46 percent of a drop's energy through its
    /// impacts, measured, and was removed).
    /// The block nearest the face (by its centre) is this engine's reading
    /// of their "adjacent" stable nodes; a held face counts as known (zero);
    /// a degenerate face with no such block keeps its own B-spline.
    pub(crate) fn new(
        layout: &GridLayout,
        x: &[DVec2],
        (mass, volume): (&[f64], &[f64]),
        held: Held<'_>,
        occupation: Option<f64>,
    ) -> Self {
        let (nx, ny) = (layout.nx, layout.ny);
        let [u, v] = lattices(nx, ny);
        let len = v.start + v.ni * v.nj;
        let h = f64::from(layout.dx);
        let stencils: Vec<[Stencil; 2]> = x
            .iter()
            .map(|&q| {
                [u, v].map(|l| {
                    let (bx, wx, dx) = axis(q.x, l.offset.x, h);
                    let (by, wy, dy) = axis(q.y, l.offset.y, h);
                    Stencil {
                        base: (bx, by),
                        wx,
                        wy,
                        dx,
                        dy,
                    }
                })
            })
            .collect();
        let mut grid = Self {
            cell: h,
            lattices: [u, v],
            len: 0,
            stencils,
            active: Vec::new(),
            wall: Vec::new(),
            stacked_of: Vec::new(),
            stable_unknown: Vec::new(),
            slot: vec![usize::MAX; len],
            axis_of: Vec::new(),
            extension: vec![None; len],
            lumped: Vec::new(),
            mass_matrix: Default::default(),
            #[cfg(all(test, feature = "experimental"))]
            weights: None,
            #[cfg(all(test, feature = "experimental"))]
            massless: None,
        };
        let mut reached = vec![0.0; len];
        // The faces any point's support meets, the massless outline points'
        // included: Kakouris et al. (sec. 3.4) carry the boundary as
        // material points, counted in the cells' classification like the
        // bulk's, so a boundary point's B-splines are never exterior.
        // Adaptation, declared: the outline points here stay massless (the
        // paper gives them 0.1 percent of the mass and volume); a face only
        // they reach is degenerate, extrapolated, never an unknown of its
        // own. Dropped instead, an outline point's weights summed to 0.875
        // and a contact between two blocks made momentum (measured: +12
        // percent in the impact's step).
        let mut touched = vec![false; len];
        for (p, &m) in mass.iter().enumerate() {
            for a in 0..2 {
                grid.visit_stacked(p, a, |k, w, _| {
                    reached[k] += w * m;
                    touched[k] |= w > 0.0;
                });
            }
        }
        // Stable faces: every reached one without the extension.
        let mut stable = vec![false; len];
        for (a, l) in grid.lattices.iter().enumerate() {
            let (ni, nj) = (l.ni as i64, l.nj as i64);
            let mut fraction = vec![0.0; l.ni * l.nj];
            if occupation.is_some() {
                for (q, &vol) in x.iter().zip(volume) {
                    let s = *q / h - l.offset;
                    let (m, n) = ((s.x + 0.5).floor() as i64, (s.y + 0.5).floor() as i64);
                    if m >= 0 && n >= 0 && m < ni && n < nj {
                        fraction[(m + ni * n) as usize] += vol / (h * h);
                    }
                }
            }
            for j in 0..l.nj {
                for i in 0..l.ni {
                    let k = l.start + i + l.ni * j;
                    stable[k] = match occupation {
                        None => reached[k] > 0.0,
                        Some(cc) => (-1..=1).any(|dj| {
                            (-1..=1).any(|di| {
                                let (m, n) = (i as i64 + di, j as i64 + dj);
                                m >= 0
                                    && n >= 0
                                    && m < ni
                                    && n < nj
                                    && fraction[(m + ni * n) as usize] > cc
                            })
                        }),
                    };
                    let wall = held(a, i, j);
                    if stable[k] || (wall.is_some() && reached[k] > 0.0) {
                        grid.slot[k] = grid.len;
                        grid.len += 1;
                        grid.axis_of.push(a);
                        grid.wall.push(wall);
                        grid.stacked_of.push(k);
                        grid.stable_unknown.push(stable[k]);
                    }
                }
            }
        }
        // Degenerate faces: extrapolated from the nearest stable block.
        let lagrange = |k: i64, t: i64| -> f64 {
            (0..3)
                .filter(|&b| b != k)
                .map(|b| (t - b) as f64 / (k - b) as f64)
                .product()
        };
        for (a, l) in grid.lattices.iter().enumerate() {
            let (ni, nj) = (l.ni as i64, l.nj as i64);
            let known = |i: i64, j: i64| {
                i >= 0 && j >= 0 && i < ni && j < nj && stable[l.start + (i + ni * j) as usize]
            };
            for j in 0..nj {
                for i in 0..ni {
                    let k = l.start + (i + ni * j) as usize;
                    let wall_only = grid.slot[k] != usize::MAX && !stable[k];
                    if (grid.slot[k] != usize::MAX && !wall_only) || !touched[k] {
                        continue;
                    }
                    // The nearest stable blocks, every one at the least
                    // distance: their extrapolations averaged, each
                    // reproducing quadratics, so the mean does too. One
                    // block kept by a first-found rule preferred the lower
                    // indices, the left and the bottom: a body's two ends
                    // took differently built bases, mirror images of each
                    // other or not.
                    let mut nearest: Vec<(i64, i64)> = Vec::new();
                    let mut least = i64::MAX;
                    for aj in j - 4..=j + 2 {
                        for ai in i - 4..=i + 2 {
                            let block = (0..3).all(|c| (0..3).all(|b| known(ai + b, aj + c)));
                            if !block {
                                continue;
                            }
                            let d = (i - ai - 1).pow(2) + (j - aj - 1).pow(2);
                            if d < least {
                                least = d;
                                nearest.clear();
                            }
                            if d == least {
                                nearest.push((ai, aj));
                            }
                        }
                    }
                    let share = 1.0 / nearest.len().max(1) as f64;
                    if !nearest.is_empty() {
                        let mut list: Vec<(usize, f64)> = Vec::with_capacity(9);
                        for &(ai, aj) in &nearest {
                            for c in 0..3 {
                                for b in 0..3 {
                                    let unknown =
                                        grid.slot[l.start + (ai + b + ni * (aj + c)) as usize];
                                    if unknown == usize::MAX {
                                        continue;
                                    }
                                    let e = share * lagrange(b, i - ai) * lagrange(c, j - aj);
                                    match list.iter_mut().find(|(u, _)| *u == unknown) {
                                        Some(entry) => entry.1 += e,
                                        None => list.push((unknown, e)),
                                    }
                                }
                            }
                        }
                        grid.extension[k] = Some(list.into_boxed_slice());
                    } else if !wall_only && reached[k] > 0.0 {
                        grid.slot[k] = grid.len;
                        grid.len += 1;
                        grid.axis_of.push(a);
                        grid.wall.push(None);
                        grid.stacked_of.push(k);
                        grid.stable_unknown.push(false);
                    }
                }
            }
        }
        grid.active = grid.wall.iter().map(|w| w.is_none()).collect();
        grid.relump(mass);
        #[cfg(all(test, feature = "experimental"))]
        if let Some(faces) = hooks::massless() {
            let mut inert = vec![false; grid.len];
            for &(a, i, j) in faces.iter() {
                let l = grid.lattices[a];
                if i < l.ni && j < l.nj {
                    let unknown = grid.slot[l.start + i + l.ni * j];
                    if unknown != usize::MAX {
                        inert[unknown] = true;
                    }
                }
            }
            grid.massless = Some(inert);
        }
        grid
    }

    /// `M_c y`, the consistent mass's product: the same sums as a pass over
    /// the particles (`sum_p m_p N_k(X_p) sum_m N_m(X_p) y_m`), from the
    /// matrix assembled once for these masses; a different `mass` assembles
    /// it again. Only the summation order differs from the pass: its
    /// round-off, not its value.
    pub(crate) fn consistent_mass_times(&self, y: &[f64], mass: &[f64]) -> Vec<f64> {
        let mut kept = self.mass_matrix.lock().expect("the mass matrix's lock");
        if kept.as_ref().is_none_or(|m| m.mass != mass) {
            let mut entries: Vec<(usize, usize, f64)> = Vec::new();
            for (p, &mp) in mass.iter().enumerate() {
                for a in 0..2 {
                    let mut reach: Vec<(usize, f64)> = Vec::with_capacity(9);
                    self.visit(p, a, |k, w, _| reach.push((k, w)));
                    for &(k, wk) in &reach {
                        for &(m, wm) in &reach {
                            entries.push((k, m, mp * wk * wm));
                        }
                    }
                }
            }
            entries.sort_unstable_by_key(|&(k, m, _)| (k, m));
            // Compressed rows: each row's entries in column order, a
            // repeated column summed.
            let mut matrix = MassMatrix {
                mass: mass.to_vec(),
                start: Vec::with_capacity(self.len + 1),
                ..MassMatrix::default()
            };
            let mut next = entries.iter().peekable();
            for k in 0..self.len {
                let row = matrix.value.len();
                matrix.start.push(row);
                while let Some(&(_, m, value)) = next.next_if(|e| e.0 == k) {
                    if matrix.value.len() > row && matrix.column.last() == Some(&m) {
                        *matrix.value.last_mut().expect("a row entry") += value;
                    } else {
                        matrix.column.push(m);
                        matrix.value.push(value);
                    }
                }
            }
            matrix.start.push(matrix.value.len());
            *kept = Some(matrix);
        }
        let matrix = kept.as_ref().expect("assembled above");
        (0..self.len)
            .map(|k| {
                (matrix.start[k]..matrix.start[k + 1])
                    .map(|e| matrix.value[e] * y[matrix.column[e]])
                    .sum()
            })
            .collect()
    }

    /// Each unknown's lumped mass, through the current basis.
    pub(crate) fn relump(&mut self, mass: &[f64]) {
        let mut lumped = vec![0.0; self.len];
        for (p, &m) in mass.iter().enumerate() {
            for a in 0..2 {
                self.visit(p, a, |k, w, _| lumped[k] += w * m);
            }
        }
        self.lumped = lumped;
    }

    /// Calls `f(unknown, weight, weight gradient)` for every unknown of
    /// lattice `a` particle `p` reaches.
    pub(crate) fn visit(&self, p: usize, a: usize, mut f: impl FnMut(usize, f64, DVec2)) {
        self.visit_stacked(p, a, |k, w, g| {
            let unknown = self.slot[k];
            if unknown != usize::MAX {
                f(unknown, w, g);
            } else if let Some(list) = &self.extension[k] {
                for &(unknown, e) in list.iter() {
                    f(unknown, w * e, g * e);
                }
            }
        });
    }

    /// Calls `f(face, weight, weight gradient)` for every face of lattice
    /// `a` particle `p` reaches, by its stacked index.
    pub(crate) fn visit_stacked(&self, p: usize, a: usize, mut f: impl FnMut(usize, f64, DVec2)) {
        let s = self.stencils[p][a];
        let l = self.lattices[a];
        for (ia, (&wx, &dx)) in s.wx.iter().zip(&s.dx).enumerate() {
            let i = s.base.0 + ia as i64;
            if i < 0 || i as usize >= l.ni {
                continue;
            }
            for (jb, (&wy, &dy)) in s.wy.iter().zip(&s.dy).enumerate() {
                let j = s.base.1 + jb as i64;
                if j < 0 || j as usize >= l.nj {
                    continue;
                }
                let k = l.start + i as usize + l.ni * j as usize;
                let (n, gn) = (wx * wy, DVec2::new(dx * wy, wx * dy));
                // The web-spline's weight: `w N` and its gradient.
                #[cfg(all(test, feature = "experimental"))]
                let (n, gn) = match &self.weights {
                    Some(weights) => {
                        let (w, gw) = weights[p][a];
                        (w * n, w * gn + n * gw)
                    }
                    None => (n, gn),
                };
                f(k, n, gn);
            }
        }
    }

    /// Value and gradient of lattice `a`'s field `y` at particle `p`.
    pub(crate) fn read(&self, y: &[f64], p: usize, a: usize) -> (f64, DVec2) {
        let (mut value, mut grad) = (0.0, DVec2::ZERO);
        self.visit(p, a, |k, w, g| {
            value += w * y[k];
            grad += g * y[k];
        });
        (value, grad)
    }

    /// `M_eps y` on the active faces.
    pub(crate) fn mass_times(&self, y: &[f64], mass: &[f64], eps: f64) -> Vec<f64> {
        let mut out = self.mass_times_unmasked(y, mass, eps);
        for (value, &on) in out.iter_mut().zip(&self.active) {
            if !on {
                *value = 0.0;
            }
        }
        out
    }

    /// `M_eps y` on every unknown.
    pub(crate) fn mass_times_unmasked(&self, y: &[f64], mass: &[f64], eps: f64) -> Vec<f64> {
        // A massless unknown's row and column of the mass are zero.
        let masked: Vec<f64>;
        let y = match self.massless() {
            Some(inert) => {
                masked = y
                    .iter()
                    .zip(inert)
                    .map(|(&e, &none)| if none { 0.0 } else { e })
                    .collect();
                &masked[..]
            }
            None => y,
        };
        let mut out = if eps < 1.0 {
            let mut product = self.consistent_mass_times(y, mass);
            product.iter_mut().for_each(|e| *e *= 1.0 - eps);
            product
        } else {
            vec![0.0; self.len]
        };
        for k in 0..self.len {
            out[k] += eps * self.lumped[k] * y[k];
        }
        if let Some(inert) = self.massless() {
            for (value, &none) in out.iter_mut().zip(inert) {
                if none {
                    *value = 0.0;
                }
            }
        }
        out
    }

    /// `inertia M_eps + K` plus the held particles' penalties (`springs`:
    /// particle, lattice, stiffness; `inertia` is the midpoint's `2 /
    /// dt^2`), `K` the stiffness of the solid's linear elasticity (`sigma =
    /// 2 mu eps + (K - mu) tr(eps) I`, the linearisation of its energy at `F
    /// = I`), in band form, factored.
    /// The unknowns are renumbered body by body, then row by row of the face
    /// positions, so the band is a few rows of one body's faces wide.
    pub(crate) fn approximate_jacobian(
        &self,
        (mass, v0, solids): (&[f64], &[f64], &[Elastic]),
        eps: f64,
        inertia: f64,
        springs: &[Spring],
    ) -> Band {
        let mut band = self.empty_band(mass.len());
        let inert = |k: usize| self.massless().is_some_and(|none| none[k]);
        // Each particle's unknowns with their weights and gradients.
        let reach = |p: usize| -> Vec<(usize, usize, f64, DVec2)> {
            let mut list = Vec::with_capacity(18);
            for a in 0..2 {
                self.visit(p, a, |k, w, g| list.push((k, a, w, g)));
            }
            list
        };
        for (p, (&mp, &vol)) in mass.iter().zip(v0).enumerate() {
            let list = reach(p);
            for &(k, a, wk, gk) in &list {
                for &(m, b, wm, gm) in &list {
                    let mut value = 0.0;
                    if a == b {
                        if !inert(k) && !inert(m) {
                            value += inertia * (1.0 - eps) * mp * wk * wm;
                        }
                        value += vol * solids[p].mu * gk.dot(gm);
                    }
                    let (mu, lame) = (solids[p].mu, solids[p].bulk - solids[p].mu);
                    value += vol * (lame * gk[a] * gm[b] + mu * gk[b] * gm[a]);
                    band.add(k, m, value);
                }
            }
        }
        for k in 0..self.len {
            band.add(k, k, inertia * eps * self.lumped[k]);
        }
        // A held particle's penalty along its plane's normal (`step`).
        for &(p, a, stiffness) in springs {
            let mut list = Vec::with_capacity(9);
            self.visit(p, a, |k, w, _| list.push((k, w)));
            for &(k, wk) in &list {
                for &(m, wm) in &list {
                    band.add(k, m, stiffness * wk * wm);
                }
            }
        }
        band.factor();
        band
    }

    /// `M_eps` on the active unknowns and the identity on the others, in
    /// band form, factored: the projection's exact preconditioner, the
    /// inverse of the masked system itself (`solve`).
    pub(crate) fn mass_band(&self, mass: &[f64], eps: f64) -> Band {
        let mut band = self.empty_band(mass.len());
        let massive = self.massive();
        for (p, &mp) in mass.iter().enumerate() {
            for a in 0..2 {
                let mut list = Vec::with_capacity(9);
                self.visit(p, a, |k, w, _| list.push((k, w)));
                for &(k, wk) in &list {
                    for &(m, wm) in &list {
                        if massive[k] && massive[m] {
                            band.add(k, m, (1.0 - eps) * mp * wk * wm);
                        }
                    }
                }
            }
        }
        for (k, &on) in massive.iter().enumerate() {
            let diagonal = if on { eps * self.lumped[k] } else { 1.0 };
            band.add(k, k, diagonal);
        }
        band.factor();
        band
    }

    /// The active unknowns with some mass: all of them, unless some are
    /// massless (`Grid::massless`), which the grid velocity's projection
    /// leaves at zero, having no inertia and so no velocity of their own.
    pub(crate) fn massive(&self) -> Vec<bool> {
        match self.massless() {
            Some(inert) => self
                .active
                .iter()
                .zip(inert)
                .map(|(&on, &none)| on && !none)
                .collect(),
            None => self.active.clone(),
        }
    }

    /// The unknowns that carry no inertia, a probe's massless contact
    /// boundary (`hooks::set_massless`); `None`: every unknown has its mass.
    #[cfg(all(test, feature = "experimental"))]
    pub(crate) fn massless(&self) -> Option<&[bool]> {
        self.massless.as_deref()
    }

    #[cfg(not(all(test, feature = "experimental")))]
    pub(crate) fn massless(&self) -> Option<&[bool]> {
        None
    }

    /// An empty band over the unknowns, renumbered body by body and within
    /// each row by row of the face positions, as wide as the particles'
    /// reach (`approximate_jacobian`).
    pub(crate) fn empty_band(&self, particles: usize) -> Band {
        let n = self.len;
        // Position of each unknown, in half cells, for the ordering.
        let mut key = vec![(0i64, 0i64); n];
        for (a, l) in self.lattices.iter().enumerate() {
            for j in 0..l.nj {
                for i in 0..l.ni {
                    let unknown = self.slot[l.start + i + l.ni * j];
                    if unknown != usize::MAX {
                        let q = (DVec2::new(i as f64, j as f64) + l.offset) * 2.0;
                        key[unknown] = (q.y.round() as i64, q.x.round() as i64 * 2 + a as i64);
                    }
                }
            }
        }
        // Each particle's unknowns with their weights and gradients.
        let reach = |p: usize| -> Vec<(usize, usize, f64, DVec2)> {
            let mut list = Vec::with_capacity(18);
            for a in 0..2 {
                self.visit(p, a, |k, w, g| list.push((k, a, w, g)));
            }
            list
        };
        // Bodies that share no unknown share no entry of the system: each
        // is numbered as a block of its own, so the band spans one body's
        // rows, not every body's on the same rows. A body here is the
        // unknowns the particles' supports join, found by union-find.
        let mut parent: Vec<usize> = (0..n).collect();
        let root = |parent: &mut [usize], mut k: usize| {
            while parent[k] != k {
                parent[k] = parent[parent[k]];
                k = parent[k];
            }
            k
        };
        for p in 0..particles {
            let list = reach(p);
            if let Some(&(first, ..)) = list.first() {
                for &(k, ..) in &list[1..] {
                    let (a, b) = (root(&mut parent, first), root(&mut parent, k));
                    parent[a.max(b)] = a.min(b);
                }
            }
        }
        let roots: Vec<usize> = (0..n).map(|k| root(&mut parent, k)).collect();
        let mut body_key = vec![(i64::MAX, i64::MAX); n];
        for (k, &r) in roots.iter().enumerate() {
            body_key[r] = body_key[r].min(key[k]);
        }
        let mut by_position: Vec<usize> = (0..n).collect();
        by_position.sort_by_key(|&k| (body_key[roots[k]], key[k]));
        let mut order = vec![0; n];
        for (position, &k) in by_position.iter().enumerate() {
            order[k] = position;
        }
        let mut width = 0;
        for p in 0..particles {
            let list = reach(p);
            for &(k, ..) in &list {
                for &(m, ..) in &list {
                    width = width.max(order[k].abs_diff(order[m]));
                }
            }
        }
        Band {
            n,
            width,
            data: vec![0.0; n * (width + 1)],
            order,
        }
    }

    /// The diagonal of `M_eps`, for the tests' Jacobi preconditioner.
    #[cfg(all(test, feature = "experimental"))]
    pub(crate) fn mass_diagonal(&self, mass: &[f64], eps: f64) -> Vec<f64> {
        let mut diag = vec![0.0; self.len];
        for (p, &m) in mass.iter().enumerate() {
            for a in 0..2 {
                self.visit(p, a, |k, w, _| diag[k] += (1.0 - eps) * m * w * w);
            }
        }
        for (d, &l) in diag.iter_mut().zip(&self.lumped) {
            *d += eps * l;
        }
        diag
    }
}

/// `sum_p body(p)` over the particles in parallel, each adding into a
/// vector of `len` entries. The particles are cut into chunks that depend
/// on their number alone and the chunks' sums are added in order, so the
/// result is the same bits whatever the threads (measured: rayon's own
/// fold and reduce, which split by the threads' schedule, made two runs of
/// one drop part by 2 mm once a wall's release went the other way).
pub(crate) fn parallel_scatter(
    len: usize,
    n: usize,
    body: impl Fn(usize, &mut [f64]) + Sync,
) -> Vec<f64> {
    let chunk = n.div_ceil(64).max(64);
    let parts: Vec<Vec<f64>> = (0..n.div_ceil(chunk))
        .into_par_iter()
        .map(|c| {
            let mut acc = vec![0.0; len];
            for p in c * chunk..((c + 1) * chunk).min(n) {
                body(p, &mut acc);
            }
            acc
        })
        .collect();
    let mut total = vec![0.0; len];
    for part in &parts {
        total.iter_mut().zip(part).for_each(|(x, y)| *x += y);
    }
    total
}

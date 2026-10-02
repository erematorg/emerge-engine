//! Conjugate gradient with a MIC(0) preconditioner, as Bridson and
//! Muller-Fischer give it for this grid (4.3.2 to 4.3.4, figs. 4.1 to 4.3,
//! written out in 2D). It takes a system and returns a solution; nothing
//! else in the projection depends on how the solution was found.

use super::pressure::PressureSystem;

/// When the solve stops, and the preconditioner's two parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SolverSettings {
    /// Stop once the largest residual is this fraction of the largest
    /// right-hand side. The residual is the divergence the velocity will
    /// keep, so the notes stop on its infinity norm (4.3.2); `apic2d` uses
    /// a relative 1e-4. 1e-5 keeps the divergence left per substep a
    /// hundred thousand times below the one being removed.
    pub relative_tolerance: f32,
    /// Or once it is below this, in 1/s. The notes suggest 1e-6 1/s in
    /// double precision; in f32 the divergence of velocities of a few
    /// hundred cells per second carries round-off near 1e-5 1/s, so the
    /// floor sits one order above it, where a solve can still reach.
    pub absolute_tolerance: f32,
    /// The notes find 100 reasonable to start with, as `apic2d` uses. A
    /// solve that reaches it returns its best iterate and says so.
    pub max_iterations: u32,
    /// Blend between incomplete Cholesky (0) and its modified form (1);
    /// 0.97 in the notes and `apic2d`.
    pub mic_tau: f32,
    /// A pivot below this fraction of its diagonal falls back to the
    /// diagonal, `apic2d`'s guard (0.25). The preconditioner only changes
    /// how fast the solve converges, not what it converges to.
    pub mic_min_diagonal_ratio: f32,
}

impl Default for SolverSettings {
    fn default() -> Self {
        Self {
            relative_tolerance: 1e-5,
            absolute_tolerance: 1e-4,
            max_iterations: 100,
            mic_tau: 0.97,
            mic_min_diagonal_ratio: 0.25,
        }
    }
}

/// Pressure per cell (zero where there is no unknown), with how the solve
/// went.
#[derive(Clone, Debug, PartialEq)]
pub struct Solution {
    pub pressure: Vec<f32>,
    pub iterations: u32,
    /// Largest remaining residual, in 1/s.
    pub residual: f32,
    pub converged: bool,
}

pub fn solve(sys: &PressureSystem, settings: &SolverSettings) -> Solution {
    let n = sys.nx * sys.ny;
    let mut p = vec![0.0f32; n];
    let mut r = sys.rhs.clone();
    let initial = max_abs(&r, &sys.active);
    if initial == 0.0 {
        return Solution {
            pressure: p,
            iterations: 0,
            residual: 0.0,
            converged: true,
        };
    }
    let tolerance = (settings.relative_tolerance * initial).max(settings.absolute_tolerance);
    if initial <= tolerance {
        return Solution {
            pressure: p,
            iterations: 0,
            residual: initial,
            converged: true,
        };
    }
    let precon = mic0(sys, settings);
    let mut z = vec![0.0f32; n];
    let mut q = vec![0.0f32; n];
    apply_preconditioner(sys, &precon, &r, &mut q, &mut z);
    let mut s = z.clone();
    let mut sigma = dot(&z, &r, &sys.active);
    let mut residual = initial;
    for iteration in 0..settings.max_iterations {
        apply_a(sys, &s, &mut z);
        let alpha = sigma / dot(&z, &s, &sys.active);
        for c in 0..n {
            if sys.active[c] {
                p[c] += (alpha * s[c] as f64) as f32;
                r[c] -= (alpha * z[c] as f64) as f32;
            }
        }
        residual = max_abs(&r, &sys.active);
        if residual <= tolerance {
            return Solution {
                pressure: p,
                iterations: iteration + 1,
                residual,
                converged: true,
            };
        }
        apply_preconditioner(sys, &precon, &r, &mut q, &mut z);
        let sigma_new = dot(&z, &r, &sys.active);
        let beta = sigma_new / sigma;
        for c in 0..n {
            if sys.active[c] {
                s[c] = z[c] + (beta * s[c] as f64) as f32;
            }
        }
        sigma = sigma_new;
    }
    Solution {
        pressure: p,
        iterations: settings.max_iterations,
        residual,
        converged: false,
    }
}

fn max_abs(v: &[f32], active: &[bool]) -> f32 {
    v.iter()
        .zip(active)
        .filter(|(_, a)| **a)
        .fold(0.0f32, |m, (x, _)| m.max(x.abs()))
}

/// Dot product accumulated in f64, so the step lengths do not carry the
/// round-off of a long f32 sum.
fn dot(a: &[f32], b: &[f32], active: &[bool]) -> f64 {
    a.iter()
        .zip(b)
        .zip(active)
        .filter(|(_, act)| **act)
        .map(|((x, y), _)| *x as f64 * *y as f64)
        .sum()
}

/// `out = A s`.
fn apply_a(sys: &PressureSystem, s: &[f32], out: &mut [f32]) {
    let nx = sys.nx;
    for j in 0..sys.ny {
        for i in 0..nx {
            let c = i + nx * j;
            if !sys.active[c] {
                out[c] = 0.0;
                continue;
            }
            let mut v = sys.diag[c] * s[c];
            if i + 1 < nx {
                v += sys.plus_i[c] * s[c + 1];
            }
            if i > 0 {
                v += sys.plus_i[c - 1] * s[c - 1];
            }
            if j + 1 < sys.ny {
                v += sys.plus_j[c] * s[c + nx];
            }
            if j > 0 {
                v += sys.plus_j[c - nx] * s[c - nx];
            }
            out[c] = v;
        }
    }
}

/// Reciprocals of the MIC(0) factor's diagonal (the notes' fig. 4.2, in
/// 2D, with `apic2d`'s small-pivot guard).
fn mic0(sys: &PressureSystem, settings: &SolverSettings) -> Vec<f32> {
    let nx = sys.nx;
    let tau = settings.mic_tau;
    let mut precon = vec![0.0f32; nx * sys.ny];
    for j in 0..sys.ny {
        for i in 0..nx {
            let c = i + nx * j;
            if !sys.active[c] {
                continue;
            }
            let mut e = sys.diag[c];
            if i > 0 {
                let (a, pre) = (sys.plus_i[c - 1], precon[c - 1]);
                e -= (a * pre).powi(2) + tau * a * sys.plus_j[c - 1] * pre * pre;
            }
            if j > 0 {
                let (a, pre) = (sys.plus_j[c - nx], precon[c - nx]);
                e -= (a * pre).powi(2) + tau * a * sys.plus_i[c - nx] * pre * pre;
            }
            if e < settings.mic_min_diagonal_ratio * sys.diag[c] {
                e = sys.diag[c];
            }
            precon[c] = 1.0 / (e + 1e-30).sqrt();
        }
    }
    precon
}

/// `z = M r` by the two triangular solves of the notes' fig. 4.3, in 2D.
fn apply_preconditioner(
    sys: &PressureSystem,
    precon: &[f32],
    r: &[f32],
    q: &mut [f32],
    z: &mut [f32],
) {
    let (nx, ny) = (sys.nx, sys.ny);
    for j in 0..ny {
        for i in 0..nx {
            let c = i + nx * j;
            if !sys.active[c] {
                q[c] = 0.0;
                continue;
            }
            let mut t = r[c];
            if i > 0 {
                t -= sys.plus_i[c - 1] * precon[c - 1] * q[c - 1];
            }
            if j > 0 {
                t -= sys.plus_j[c - nx] * precon[c - nx] * q[c - nx];
            }
            q[c] = t * precon[c];
        }
    }
    for j in (0..ny).rev() {
        for i in (0..nx).rev() {
            let c = i + nx * j;
            if !sys.active[c] {
                z[c] = 0.0;
                continue;
            }
            let mut t = q[c];
            if i + 1 < nx {
                t -= sys.plus_i[c] * precon[c] * z[c + 1];
            }
            if j + 1 < ny {
                t -= sys.plus_j[c] * precon[c] * z[c + nx];
            }
            z[c] = t * precon[c];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The 5-point Laplacian on an `n x n` block with `p = 0` all round
    /// (every neighbour outside the block is air at a full cell), scaled
    /// to unit spacing.
    fn dirichlet_block(n: usize) -> PressureSystem {
        let size = n * n;
        let mut sys = PressureSystem {
            nx: n,
            ny: n,
            active: vec![true; size],
            diag: vec![4.0; size],
            plus_i: vec![0.0; size],
            plus_j: vec![0.0; size],
            rhs: vec![0.0; size],
        };
        for j in 0..n {
            for i in 0..n {
                let c = i + n * j;
                if i + 1 < n {
                    sys.plus_i[c] = -1.0;
                }
                if j + 1 < n {
                    sys.plus_j[c] = -1.0;
                }
            }
        }
        sys
    }

    #[test]
    fn it_recovers_a_known_solution() {
        let n = 24;
        let mut sys = dirichlet_block(n);
        let exact: Vec<f32> = (0..n * n)
            .map(|c| ((c % n) as f32 * 0.3).sin() + ((c / n) as f32 * 0.2).cos())
            .collect();
        let mut rhs = vec![0.0; n * n];
        apply_a(&sys, &exact, &mut rhs);
        sys.rhs = rhs;
        let settings = SolverSettings {
            relative_tolerance: 1e-6,
            absolute_tolerance: 0.0,
            max_iterations: 400,
            ..SolverSettings::default()
        };
        let solution = solve(&sys, &settings);
        assert!(solution.converged);
        for (c, (got, want)) in solution.pressure.iter().zip(&exact).enumerate() {
            assert!((got - want).abs() < 1e-3, "cell {c}");
        }
    }

    /// The modified factorisation is there to need fewer iterations than
    /// the plain incomplete one (the notes, 4.3.4).
    #[test]
    fn the_modified_factorisation_needs_fewer_iterations() {
        let n = 64;
        let mut sys = dirichlet_block(n);
        sys.rhs = (0..n * n)
            .map(|c| if c % 7 == 0 { 1.0 } else { -0.2 })
            .collect();
        let strict = SolverSettings {
            relative_tolerance: 1e-5,
            absolute_tolerance: 0.0,
            max_iterations: 1000,
            ..SolverSettings::default()
        };
        let modified = solve(&sys, &strict);
        let plain = solve(
            &sys,
            &SolverSettings {
                mic_tau: 0.0,
                ..strict
            },
        );
        assert!(modified.converged && plain.converged);
        assert!(
            modified.iterations < plain.iterations,
            "MIC(0) {} iterations, IC(0) {}",
            modified.iterations,
            plain.iterations
        );
    }

    #[test]
    fn a_zero_right_hand_side_returns_at_once() {
        let sys = dirichlet_block(8);
        let solution = solve(&sys, &SolverSettings::default());
        assert!(solution.converged);
        assert_eq!(solution.iterations, 0);
        assert!(solution.pressure.iter().all(|&p| p == 0.0));
    }
}

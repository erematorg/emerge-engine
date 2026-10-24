//! A banded matrix and its Cholesky factor: the preconditioner the step's
//! linear solves lean on.

/// A symmetric positive definite matrix in band form, factored `L L^T`
/// in place (Cholesky), its unknowns renumbered by `order` to keep the
/// band narrow. A non-positive pivot is lifted to a small positive one, so
/// the factor stays usable as a preconditioner.
#[derive(Clone)]
pub(crate) struct Band {
    pub(crate) n: usize,
    pub(crate) width: usize,
    /// `L(i, j)` for `i - width <= j <= i`, row by row.
    pub(crate) data: Vec<f64>,
    /// The band's position of each unknown.
    pub(crate) order: Vec<usize>,
}

impl Band {
    /// Adds `value` at the unknowns `(k, m)`, kept on the lower triangle.
    pub(crate) fn add(&mut self, k: usize, m: usize, value: f64) {
        let (i, j) = (self.order[k], self.order[m]);
        if i >= j {
            let slot = self.at(i, j);
            self.data[slot] += value;
        }
    }

    pub(crate) fn at(&self, i: usize, j: usize) -> usize {
        i * (self.width + 1) + (j + self.width - i)
    }

    pub(crate) fn factor(&mut self) {
        let w = self.width;
        for i in 0..self.n {
            let lo = i.saturating_sub(w);
            for j in lo..=i {
                let mut sum = self.data[self.at(i, j)];
                for k in lo.max(j.saturating_sub(w))..j {
                    sum -= self.data[self.at(i, k)] * self.data[self.at(j, k)];
                }
                let slot = self.at(i, j);
                if i == j {
                    let floor = 1e-12 * self.data[slot].abs().max(f64::MIN_POSITIVE);
                    self.data[slot] = sum.max(floor).sqrt();
                } else {
                    self.data[slot] = sum / self.data[self.at(j, j)];
                }
            }
        }
    }

    pub(crate) fn solve(&self, rhs: &[f64]) -> Vec<f64> {
        let (n, w) = (self.n, self.width);
        let mut y = vec![0.0; n];
        for (k, &value) in rhs.iter().enumerate() {
            y[self.order[k]] = value;
        }
        for i in 0..n {
            let lo = i.saturating_sub(w);
            let sum: f64 = (lo..i).map(|k| self.data[self.at(i, k)] * y[k]).sum();
            y[i] = (y[i] - sum) / self.data[self.at(i, i)];
        }
        for i in (0..n).rev() {
            let hi = (i + w + 1).min(n);
            let sum: f64 = (i + 1..hi).map(|k| self.data[self.at(k, i)] * y[k]).sum();
            y[i] = (y[i] - sum) / self.data[self.at(i, i)];
        }
        (0..n).map(|k| y[self.order[k]]).collect()
    }
}

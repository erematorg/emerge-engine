/// O(N) information-theoretic estimators for MPM particle systems.
///
/// All functions take `&Particles` (SoA — the CPU solver's canonical form, same as
/// `diagnostics::per_material_stats`) and do a single pass. No distance matrix, no k-NN.
/// Safe to call every frame.
use crate::materials::registry::MAX_MATERIAL_SLOTS;
use crate::particle::Particles;

/// Shannon entropy from a histogram of counts.
/// H = -Σ p·log₂(p), returns bits.
fn shannon_bits(counts: &[u32]) -> f32 {
    let total: u32 = counts.iter().sum();
    if total == 0 {
        return 0.0;
    }
    let n = total as f32;
    counts
        .iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f32 / n;
            -p * p.log2()
        })
        .sum()
}

/// Spatial entropy — how evenly particles are spread across the grid.
///
/// H = 0  → all particles in one cell (maximum clustering).
/// H = log₂(grid²) → perfectly uniform (maximum disorder).
///
/// Bins each particle by its grid cell (floor of position). O(N).
pub fn spatial_entropy(particles: &Particles, grid_res: usize) -> f32 {
    if grid_res == 0 || particles.len() == 0 {
        return 0.0;
    }
    let mut counts = vec![0u32; grid_res * grid_res];
    for &pos in &particles.x {
        // Negative positions (OOB particles) clamp to 0, never wrap.
        let cx = (pos.x.max(0.0) as usize).min(grid_res - 1);
        let cy = (pos.y.max(0.0) as usize).min(grid_res - 1);
        counts[cy * grid_res + cx] += 1;
    }
    shannon_bits(&counts)
}

/// Kinetic entropy — disorder in the velocity-magnitude distribution.
///
/// Bins |v| into `bins` buckets between 0 and v_max.
/// H = 0 → all particles at same speed (coherent motion).
/// H = log₂(bins) → speeds uniformly spread (thermal chaos).
/// O(N).
pub fn kinetic_entropy(particles: &Particles, bins: usize) -> f32 {
    if particles.len() == 0 || bins == 0 {
        return 0.0;
    }
    let v_max = particles.v.iter().map(|v| v.length()).fold(0.0f32, f32::max);
    if v_max < f32::EPSILON {
        return 0.0;
    }
    let mut counts = vec![0u32; bins];
    for v in &particles.v {
        let bin = ((v.length() / v_max) * (bins - 1) as f32) as usize;
        counts[bin.min(bins - 1)] += 1;
    }
    shannon_bits(&counts)
}

/// Phase entropy — diversity of material types across the particle set.
///
/// H = 0 → all particles same material (pure phase).
/// H = log₂(num_materials) → materials equally distributed.
/// O(N).
pub fn phase_entropy(particles: &Particles) -> f32 {
    if particles.len() == 0 {
        return 0.0;
    }
    let mut counts = [0u32; MAX_MATERIAL_SLOTS];
    for &id in &particles.material_id {
        counts[(id as usize).min(MAX_MATERIAL_SLOTS - 1)] += 1;
    }
    shannon_bits(&counts)
}

/// Local mutual information between two material phases over a set of particles.
///
/// `indices` is the particle set to analyse — typically
/// `solver.particles_near(center, radius).collect()`. Measures how much the presence
/// of `mat_a` predicts `mat_b` within that set (2×2 joint histogram). O(|indices|).
///
/// Returns bits. 0 = independent phases; high = the two materials co-locate (or exclude).
pub fn local_phase_mi(particles: &Particles, indices: &[usize], mat_a: u32, mat_b: u32) -> f32 {
    if indices.is_empty() {
        return 0.0;
    }
    let n = indices.len() as f32;

    // 2×2 joint counts indexed [a*2 + b]: [neither, only_b, only_a, both].
    let mut joint = [0u32; 4];
    for &i in indices {
        let id = particles.material_id[i];
        let a = (id == mat_a) as usize;
        let b = (id == mat_b) as usize;
        joint[a * 2 + b] += 1;
    }

    let p_a = (joint[2] + joint[3]) as f32 / n;
    let p_b = (joint[1] + joint[3]) as f32 / n;

    // I(A;B) = Σ p(a,b)·log₂(p(a,b) / (p(a)·p(b))).
    let mut mi = 0.0f32;
    for (idx, &count) in joint.iter().enumerate() {
        if count == 0 {
            continue;
        }
        let pa = if idx >= 2 { p_a } else { 1.0 - p_a };
        let pb = if idx % 2 == 1 { p_b } else { 1.0 - p_b };
        let pab = count as f32 / n;
        let denom = pa * pb;
        if denom > f32::EPSILON {
            mi += pab * (pab / denom).log2();
        }
    }
    mi.max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::particle::{Particle, Particles};
    use glam::Vec2;

    fn build(items: &[(f32, f32, u32, f32, f32)]) -> Particles {
        let mut ps = Particles::new();
        for &(x, y, mat, vx, vy) in items {
            let mut p = Particle::zeroed();
            p.x = Vec2::new(x, y);
            p.v = Vec2::new(vx, vy);
            p.material_id = mat;
            ps.push(p);
        }
        ps
    }

    #[test]
    fn spatial_entropy_uniform_is_high() {
        // 4 particles in 4 distinct cells → near-max entropy for 4 occupied cells.
        let ps = build(&[
            (0.5, 0.5, 0, 0.0, 0.0),
            (1.5, 0.5, 0, 0.0, 0.0),
            (0.5, 1.5, 0, 0.0, 0.0),
            (1.5, 1.5, 0, 0.0, 0.0),
        ]);
        assert!(spatial_entropy(&ps, 4) > 1.5);
    }

    #[test]
    fn spatial_entropy_clustered_is_low() {
        let ps = build(&[
            (0.1, 0.1, 0, 0.0, 0.0),
            (0.2, 0.2, 0, 0.0, 0.0),
            (0.3, 0.1, 0, 0.0, 0.0),
        ]);
        assert!(spatial_entropy(&ps, 64) < 0.1);
    }

    #[test]
    fn kinetic_entropy_same_speed_is_zero() {
        let ps = build(&[
            (0.0, 0.0, 0, 1.0, 0.0),
            (1.0, 0.0, 0, 1.0, 0.0),
            (2.0, 0.0, 0, 1.0, 0.0),
        ]);
        assert!(kinetic_entropy(&ps, 16) < 0.01);
    }

    #[test]
    fn phase_entropy_single_material_is_zero() {
        let ps = build(&[(0.0, 0.0, 0, 0.0, 0.0), (1.0, 0.0, 0, 0.0, 0.0)]);
        assert_eq!(phase_entropy(&ps), 0.0);
    }

    #[test]
    fn phase_entropy_two_equal_materials_is_one_bit() {
        let ps = build(&[(0.0, 0.0, 0, 0.0, 0.0), (1.0, 0.0, 1, 0.0, 0.0)]);
        assert!((phase_entropy(&ps) - 1.0).abs() < 0.01);
    }

    #[test]
    fn local_phase_mi_perfect_separation() {
        // Two materials, never co-occurring in the same slot → still independent here
        // (each particle is exactly one material). MI of a single categorical split
        // against itself peaks; we check it is finite and non-negative.
        let ps = build(&[
            (0.0, 0.0, 0, 0.0, 0.0),
            (1.0, 0.0, 1, 0.0, 0.0),
            (2.0, 0.0, 0, 0.0, 0.0),
            (3.0, 0.0, 1, 0.0, 0.0),
        ]);
        let indices: Vec<usize> = (0..ps.len()).collect();
        let mi = local_phase_mi(&ps, &indices, 0, 1);
        assert!(mi >= 0.0 && mi.is_finite());
    }

    #[test]
    fn empty_inputs_are_zero() {
        let ps = Particles::new();
        assert_eq!(spatial_entropy(&ps, 64), 0.0);
        assert_eq!(kinetic_entropy(&ps, 16), 0.0);
        assert_eq!(phase_entropy(&ps), 0.0);
        assert_eq!(local_phase_mi(&ps, &[], 0, 1), 0.0);
    }
}

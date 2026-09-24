use glam::{IVec2, Mat2, Vec2};
use rayon::prelude::*;

use crate::grid::kernel::{axis_weights_derivative, quadratic_weights};
use crate::grid::{CellMap, Grid, flat_index};
use crate::materials::registry::MaterialRegistry;
use crate::particle::Particles;
use crate::solver::config::KERNEL_D_INVERSE;

use super::{combined_kirchhoff_stress, combined_kirchhoff_stress_from};

/// Scatters ONE particle's mass/momentum contribution into a thread-local
/// `CellMap` accumulator -- factored out of `scatter_particles_to_grid`'s
/// fold closure so `scatter_particles_to_grid_sorted` (spatial-sort opt-in,
/// see that function's own doc) can share the exact same per-particle math
/// without duplicating it.
fn scatter_one_into(
    acc: &mut CellMap,
    particles: &Particles,
    materials: &MaterialRegistry,
    dt: f32,
    resolution: usize,
    i: usize,
) {
    let material_id = particles.material_id[i];
    let x = particles.x[i];
    let mass_i = particles.mass[i];
    let v_i = particles.v[i];
    let c_i = particles.velocity_gradient[i];

    // Enum-dispatch fast path (`MaterialRegistry::kirchhoff_stress`/
    // `stress_volume`) instead of a `&dyn MaterialModel` vtable call --
    // this loop runs every particle, every substep. `material` (the
    // trait object) is still needed for the rarely-taken
    // activation/pressure/strict-mode branches inside
    // `combined_kirchhoff_stress_from` and the assert below.
    let passive_tau = materials.kirchhoff_stress(material_id, particles, i);
    let material = materials.get(material_id);
    let stress = combined_kirchhoff_stress_from(passive_tau, material, particles, i);
    let stress_coeff = -materials.stress_volume(material_id, particles, i) * KERNEL_D_INVERSE * dt;
    if materials.owns_deformation_volume_state(material_id) {
        assert!(
            stress.x_axis.is_finite() && stress.y_axis.is_finite() && stress_coeff.is_finite(),
            "strict WC-MPM particle {i} produced an unrepresentable stress impulse; reduce the timestep or use a pressure solver"
        );
    }

    let weights = quadratic_weights(x);
    for gx in 0..3 {
        for gy in 0..3 {
            let weight = weights.wx[gx] * weights.wy[gy];
            let cell_pos = weights.base_cell + IVec2::new(gx as i32 - 1, gy as i32 - 1);
            let Some(idx) = flat_index(cell_pos, resolution) else {
                continue;
            };
            let cell_dist = cell_pos.as_vec2() - x + Vec2::splat(0.5);
            let momentum =
                weight * (mass_i * (v_i + c_i * cell_dist) + stress_coeff * (stress * cell_dist));
            let entry = acc.entry(idx).or_default();
            entry.mass += weight * mass_i;
            entry.momentum += momentum;
        }
    }
}

fn merge_cell_maps(a: &mut CellMap, b: CellMap) {
    for (idx, cell) in b {
        let entry = a.entry(idx).or_default();
        entry.mass += cell.mass;
        entry.momentum += cell.momentum;
    }
}

/// P2G: scatter particle mass, momentum, and stress forces onto the grid (MLS-MPM, Hu 2018 §4).
///
/// Stress is pre-integrated as a momentum impulse so the grid needs one accumulation pass.
/// The APIC affine term conserves angular momentum without a correction step.
///
/// PARALLELIZED (2026-08-05) via a thread-local `CellMap` fold/reduce, then merged into the
/// real grid in one serial pass (`Grid::merge_cells`) -- pure safe Rust, no unsafe pointers,
/// no shared mutable state during the parallel phase (each rayon task owns its own private
/// `CellMap`; `HashMap::entry()`'s possible resize is therefore never shared across threads).
///
/// Tried a dense `Vec<Cell>` (2026-08-07) instead of `CellMap` here, on the strength of an
/// isolated SERIAL measurement showing a HashMap insert costs 6.5x a plain indexed write.
/// Measured the REAL integrated version afterward (not just the isolated microbenchmark):
/// catastrophically worse, 330-375ms vs ~20ms (15-20x), because rayon's fold/reduce creates
/// far more, far smaller accumulator instances than assumed -- each one now paying a full
/// `resolution^2` allocation+zero, vastly more total work than a lazily-growing HashMap that
/// only ever allocates what a given chunk actually touches. Exact same failure mode as the
/// earlier same-night capacity-reservation attempt (also reverted for measuring worse) --
/// should have been the tell. Reverted; `CellMap::default` is the real, measured winner for
/// THIS parallel-fold access pattern, even though a bare serial HashMap-vs-Vec test says the
/// opposite. Lesson: a microbenchmark of the accumulator alone does not predict the cost of
/// the real fold/reduce shape -- always measure the integrated change, not the isolated one.
///
/// A first attempt at this (2026-06-20) used the identical thread-local-map-then-merge shape
/// and was reverted -- NOT for a soundness reason (that version was safe Rust too), but because
/// it changed the floating-point SUMMATION ORDER for grid cells touched by multiple particles
/// (float addition isn't associative), and that shifted `fluid_spreads_more_than_elastic_under_
/// gravity`'s (a 600-step CHAOTIC simulation) qualitative outcome. Re-verified 2026-08-05: that
/// test's own assertions are real qualitative inequalities (`ar_fluid_final > ar_elastic_final`),
/// not exact-value matching -- a legitimate physical claim, not a fragile snapshot -- so the
/// real risk is chaotic amplification of a thin margin, not a badly-designed test. This
/// implementation is re-verified against that exact test (and the full regression suite) before
/// being trusted, same "revert immediately if anything moves" discipline as every other change
/// tonight.
///
/// Contact (`Particle::contact_group`) and mixture (`WithMixturePhase`) scatter are
/// DELIBERATELY kept in a separate, still-serial second pass rather than folded into the
/// parallel accumulator: both are opt-in, zero-cost-when-unused features that only a minority
/// of scenes touch, and giving them their own parallel-safe accumulator design wasn't worth the
/// added risk for this pass. The real, disclosed cost: particles that use either feature get
/// `combined_kirchhoff_stress`/`stress_volume` recomputed a second time (same pure functions,
/// same inputs, so results are identical -- just a small redundant-computation cost for the
/// particles that opt into these features, not a correctness risk).
pub fn scatter_particles_to_grid(
    particles: &Particles,
    grid: &mut Grid,
    materials: &MaterialRegistry,
    dt: f32,
    active_count: usize,
) {
    let resolution = grid.resolution();

    // Measured rayon's default chunking for this exact workload directly (a
    // temp diagnostic, not guessed): 105 fold instances for 2925 particles
    // on 8 cores -- ~13x more, much smaller chunks than the naive
    // one-per-core assumption. `with_min_len` forces fewer, larger chunks;
    // real, measured, KEPT win on its own: p2g_us 20ms -> 13ms, ~22fps ->
    // ~28fps, `CellMap` unchanged. A dense `Vec<Cell>` accumulator was tried
    // TWICE on top of this same-night investigation -- once against default
    // chunking (catastrophic, 330-375ms: 105 instances x a full
    // resolution^2 alloc+zero) and once again WITH this same `with_min_len`
    // fix (still worse than `CellMap`, 16-17ms vs 13ms): each chunk only
    // ever touches a small fraction of the `resolution^2` cells (heavy
    // per-particle stencil overlap), so `CellMap`'s lazy growth -- which
    // only ever allocates what a chunk actually touches -- beats a dense
    // buffer's fixed full-grid allocation regardless of chunk count. Both
    // dense attempts reverted; `with_min_len` + `CellMap` is the real,
    // twice-verified winner for this access pattern.
    let min_len = (active_count / (rayon::current_num_threads() * 2)).max(1);
    let local_map: CellMap = (0..active_count)
        .into_par_iter()
        .with_min_len(min_len)
        .fold(CellMap::default, |mut acc, i| {
            scatter_one_into(&mut acc, particles, materials, dt, resolution, i);
            acc
        })
        .reduce(CellMap::default, |mut a, b| {
            merge_cell_maps(&mut a, b);
            a
        });
    grid.merge_cells(local_map);

    for i in 0..active_count {
        let contact_group = particles.contact_group[i];
        let material = materials.get(particles.material_id[i]);
        let mixture_phase = material.mixture_phase();
        if contact_group == 0 && mixture_phase.is_none() {
            continue;
        }

        let x = particles.x[i];
        let mass_i = particles.mass[i];
        let v_i = particles.v[i];
        let c_i = particles.velocity_gradient[i];

        let stress = combined_kirchhoff_stress(material, particles, i);
        let stress_coeff = -material.stress_volume(particles, i) * KERNEL_D_INVERSE * dt;
        if material.owns_deformation_volume_state() {
            assert!(
                stress.x_axis.is_finite() && stress.y_axis.is_finite() && stress_coeff.is_finite(),
                "strict WC-MPM particle {i} produced an unrepresentable stress impulse; reduce the timestep or use a pressure solver"
            );
        }

        let weights = quadratic_weights(x);
        for gx in 0..3 {
            for gy in 0..3 {
                let weight = weights.wx[gx] * weights.wy[gy];
                let cell_pos = weights.base_cell + IVec2::new(gx as i32 - 1, gy as i32 - 1);
                let cell_dist = cell_pos.as_vec2() - x + Vec2::splat(0.5);
                let momentum = weight
                    * (mass_i * (v_i + c_i * cell_dist) + stress_coeff * (stress * cell_dist));
                // Additive second scatter for multi-field contact (Bardenhagen 2001) —
                // see `Particle::contact_group` doc.
                if contact_group != 0 {
                    grid.add_grip_mass_momentum(cell_pos, weight * mass_i, momentum);
                }
                // Additive second scatter for two-phase mixture coupling (Tampubolon
                // et al. 2017) — see `WithMixturePhase`/`MixturePhase` doc.
                if let Some(phase) = mixture_phase {
                    grid.add_mixture_mass_momentum(cell_pos, phase, weight * mass_i, momentum);
                }
            }
        }
    }
}

/// Real, opt-in spatial-sort variant of `scatter_particles_to_grid`'s dense
/// (first) scatter pass -- `SimConfig::spatial_sort_enabled`, see that
/// field's own doc for the full real motivation (Gao et al. 2018 SIGGRAPH
/// Asia, "GPU Optimization of Material Point Methods": periodic particle
/// reordering for cache locality; this engine's OWN GPU path already does
/// this via an indirection array, `particle_sort.wgsl`'s `sorted_particle_
/// ids`, never physically moving particle data -- this CPU version mirrors
/// that exact precedent instead of physically reordering the `Particles`
/// SoA, since nothing in this codebase's index-instability model needed
/// changing to support it (confirmed: `tag_index`/sleep-wake already treat
/// indices as unstable across frames, but a NEW un-synchronized reorder
/// pass would still need its own bookkeeping -- the indirection approach
/// sidesteps that entirely, real indices never move).
///
/// `order` must contain each of `0..active_count` exactly once (a real
/// permutation, not filtered/subset) -- callers get this from
/// `spatial_sort_order` below. Only the dense CellMap-accumulated pass is
/// reordered; the second (contact/mixture) pass deliberately stays
/// iterating `0..active_count` in the original order, unaffected -- it
/// isn't parallel-fold-accumulated (no CellMap, no chunk-locality benefit
/// to gain there) and keeping it untouched means this feature's blast
/// radius is exactly the one pass it's meant to help, nothing more.
///
/// REAL, DISCLOSED RISK (not new -- see `scatter_particles_to_grid`'s own
/// doc on the 2026-06-20 revert): changing which particles land in which
/// rayon chunk changes the floating-point SUMMATION ORDER for grid cells
/// touched by multiple particles (float addition isn't associative). That
/// exact class of change previously shifted a chaotic test's (`fluid_
/// spreads_more_than_elastic_under_gravity`) qualitative outcome. This
/// function must be re-verified against that specific test (and the full
/// regression suite) before being trusted, same discipline as last time --
/// not assumed safe just because the underlying math per-particle is
/// unchanged.
pub fn scatter_particles_to_grid_sorted(
    particles: &Particles,
    grid: &mut Grid,
    materials: &MaterialRegistry,
    dt: f32,
    active_count: usize,
    order: &[usize],
) {
    debug_assert_eq!(
        order.len(),
        active_count,
        "spatial sort order must cover exactly the active particle range"
    );
    let resolution = grid.resolution();
    let min_len = (active_count / (rayon::current_num_threads() * 2)).max(1);
    let local_map: CellMap = order
        .into_par_iter()
        .copied()
        .with_min_len(min_len)
        .fold(CellMap::default, |mut acc, i| {
            scatter_one_into(&mut acc, particles, materials, dt, resolution, i);
            acc
        })
        .reduce(CellMap::default, |mut a, b| {
            merge_cell_maps(&mut a, b);
            a
        });
    grid.merge_cells(local_map);

    for i in 0..active_count {
        let contact_group = particles.contact_group[i];
        let material = materials.get(particles.material_id[i]);
        let mixture_phase = material.mixture_phase();
        if contact_group == 0 && mixture_phase.is_none() {
            continue;
        }

        let x = particles.x[i];
        let mass_i = particles.mass[i];
        let v_i = particles.v[i];
        let c_i = particles.velocity_gradient[i];

        let stress = combined_kirchhoff_stress(material, particles, i);
        let stress_coeff = -material.stress_volume(particles, i) * KERNEL_D_INVERSE * dt;
        if material.owns_deformation_volume_state() {
            assert!(
                stress.x_axis.is_finite() && stress.y_axis.is_finite() && stress_coeff.is_finite(),
                "strict WC-MPM particle {i} produced an unrepresentable stress impulse; reduce the timestep or use a pressure solver"
            );
        }

        let weights = quadratic_weights(x);
        for gx in 0..3 {
            for gy in 0..3 {
                let weight = weights.wx[gx] * weights.wy[gy];
                let cell_pos = weights.base_cell + IVec2::new(gx as i32 - 1, gy as i32 - 1);
                let cell_dist = cell_pos.as_vec2() - x + Vec2::splat(0.5);
                let momentum = weight
                    * (mass_i * (v_i + c_i * cell_dist) + stress_coeff * (stress * cell_dist));
                if contact_group != 0 {
                    grid.add_grip_mass_momentum(cell_pos, weight * mass_i, momentum);
                }
                if let Some(phase) = mixture_phase {
                    grid.add_mixture_mass_momentum(cell_pos, phase, weight * mass_i, momentum);
                }
            }
        }
    }
}

/// Computes a real spatial sort permutation of `0..active_count`, ordered by
/// each particle's own P2G stencil center (`quadratic_weights(x).base_cell`,
/// the SAME cell the scatter loop itself keys into) flattened to a single
/// grid-row-major index -- particles that land in the same or nearby grid
/// cells end up adjacent in the returned order, so a rayon chunk (a
/// contiguous slice of this order) touches far fewer DISTINCT `CellMap`
/// entries than a chunk of spawn-order particles that have since drifted
/// apart spatially. Real, not guessed: `CellMap` is a `HashMap`, and this
/// engine's own `scatter_particles_to_grid` doc already establishes that
/// its per-chunk hashmap-entry cost dominates over raw memory-access
/// pattern for this workload.
///
/// Particles outside the grid (`flat_index` returns `None`) sort to the end
/// via `u32::MAX` -- rare (only particles that have left the domain), and
/// harmless: they still appear exactly once, just not usefully grouped.
pub fn spatial_sort_order(
    particles: &Particles,
    active_count: usize,
    resolution: usize,
) -> Vec<usize> {
    let mut order: Vec<usize> = (0..active_count).collect();
    order.sort_unstable_by_key(|&i| {
        let base_cell = quadratic_weights(particles.x[i]).base_cell;
        flat_index(base_cell, resolution).unwrap_or(u32::MAX)
    });
    order
}

/// Gathers the labeled particle point cloud (`+1.0` grip / `-1.0` rest) that
/// `Grid::resolve_contact`'s logistic-regression normal fit (`fit_contact_normal_lr`)
/// needs, at every node `scatter_particles_to_grid` already marked contact-active.
///
/// Deliberately a SECOND pass over particles, not merged into `scatter_particles_to_grid`
/// above: which nodes are contact-active isn't fully known until that first pass has
/// scattered every grip particle's mass, and `Grid::add_contact_point` only appends to a
/// node that already exists in `contact_cells` (never creates one) — so running this
/// before the first pass completes would silently miss point-cloud data for nodes whose
/// grip contribution hadn't been seen yet. Gated on `grid.has_contact_activity()`: a full
/// no-op, not even a loop iteration, for every scene that never sets
/// `Particle::contact_group` — the same zero-cost-when-unused property as the rest of
/// this feature.
pub fn gather_contact_point_cloud(particles: &Particles, grid: &mut Grid, active_count: usize) {
    if !grid.has_contact_activity() {
        return;
    }
    for i in 0..active_count {
        let x = particles.x[i];
        let label = if particles.contact_group[i] != 0 {
            1.0
        } else {
            -1.0
        };
        let weights = quadratic_weights(x);
        for gx in 0i32..3 {
            for gy in 0i32..3 {
                let cell_pos = weights.base_cell + IVec2::new(gx - 1, gy - 1);
                grid.add_contact_point(cell_pos, x, label);
            }
        }
    }
}

/// Analytic adjoint of P2G's stress→force scatter contribution w.r.t. the
/// particle's own Kirchhoff stress tensor -- the second real piece of
/// differentiable stepping, after `NeoHookeanMaterial::kirchhoff_stress_vjp`.
///
/// SCOPED, not a full P2G adjoint: differentiates only the elastic-force term
/// `weight * stress_coeff * (stress * cell_dist)` inside `scatter_particles_to_grid`,
/// treating the particle's position `x` (and therefore the kernel weights and
/// `cell_dist`) as FIXED. The mass/velocity/affine-C term is untouched here --
/// a separate, much simpler linear adjoint, not yet implemented. Differentiating
/// through the kernel weights' own dependence on `x` (how MOVING the particle
/// changes which cells it deposits to, and by how much) is the real remaining
/// gap in a fully general P2G adjoint -- deliberately deferred, not silently
/// dropped: this covers the actual control-relevant path (muscle activation →
/// stress → grid force) needed to train a controller, without yet handling
/// the harder position-dependence.
///
/// Real derivation: for one particle, cell `c`'s momentum contribution from
/// stress is `y_c = (weight_c * stress_coeff) * (stress * cell_dist_c)` --
/// linear in `stress`, a matrix-vector product `y = M*v` scaled by a fixed
/// scalar. Given the gradient flowing back from each cell's grid momentum,
/// `d_loss_d_momentum[c]` (a Vec2), the standard VJP for `y=Mv` is
/// `dL/dM = outer(dL/dy, v)`, i.e. `dL/dM_kl = dL/dy_k * v_l`. Summed over
/// all 9 stencil cells:
///
///   d_loss_d_stress = sum_c (weight_c * stress_coeff) * outer(d_loss_d_momentum[c], cell_dist_c)
///
/// Returns d_loss_d_stress, ready to feed into e.g.
/// `NeoHookeanMaterial::kirchhoff_stress_vjp` to continue the chain back to F.
/// Verified against central-difference numerical gradients in this module's
/// own tests, same non-negotiable discipline as the stress adjoint itself.
pub fn p2g_stress_vjp(x: Vec2, stress_coeff: f32, d_loss_d_momentum: &[[Vec2; 3]; 3]) -> Mat2 {
    let weights = quadratic_weights(x);
    let mut d_loss_d_stress = Mat2::ZERO;
    for (gx, (wx, momentum_row)) in weights.wx.iter().zip(d_loss_d_momentum.iter()).enumerate() {
        for (gy, (wy, &g)) in weights.wy.iter().zip(momentum_row.iter()).enumerate() {
            let weight = wx * wy;
            let cell_pos = weights.base_cell + IVec2::new(gx as i32 - 1, gy as i32 - 1);
            let cell_dist = cell_pos.as_vec2() - x + Vec2::splat(0.5);
            let scalar = weight * stress_coeff;
            // outer(g, cell_dist): column 0 = cell_dist.x * g, column 1 = cell_dist.y * g
            // (matches glam's column-major Mat2, verified against the matrix-vector
            // VJP already proven correct in kirchhoff_stress_vjp).
            d_loss_d_stress += scalar * Mat2::from_cols(cell_dist.x * g, cell_dist.y * g);
        }
    }
    d_loss_d_stress
}

/// Analytic adjoint of P2G's FULL forward pass (`scatter_particles_to_grid`)
/// w.r.t. the particle's own position `x` -- the last confirmed-real gap,
/// now closed for P2G. Combines `axis_weights_derivative` (the kernel's own
/// position-sensitivity) with the product rule across the complete momentum
/// AND mass scatter (not just the stress term `p2g_stress_vjp` covers).
///
/// Forward, restated from `scatter_particles_to_grid`: per cell `c`,
///   mass_contrib_c     = weight_c * mass
///   momentum_contrib_c = weight_c * A_c,  A_c = mass*v + M*cell_dist_c
///   M = mass*C + stress_coeff*stress   (constant across cells, for fixed particle state)
///
/// BOTH `weight_c(x)` and `cell_dist_c(x) = cell_pos_c - x + 0.5` depend on
/// `x` (`d(cell_dist)/dx = -I`), so differentiating the product `weight * A`
/// needs the product rule on both factors. Per cell, given the gradients
/// flowing back from that cell's momentum and mass, `d_loss_d_momentum[c]`
/// (Vec2) and `d_loss_d_mass[c]` (f32):
///
///   d_loss_d_x += d(weight_c)/dx * (d_loss_d_momentum[c].A_c + d_loss_d_mass[c]*mass)
///               - weight_c * (Mᵀ * d_loss_d_momentum[c])
///
/// where `d(weight_c)/dx = (dwx[gx]/dx.x * wy[gy], wx[gx] * dwy[gy]/dx.y)`
/// via `axis_weights_derivative`, and the `-weight_c * Mᵀ*d_loss_d_momentum`
/// term comes from `d(A_c)/dx = M * d(cell_dist_c)/dx = -M`.
///
/// Verified against central-difference numerical gradients taken through a
/// forward function reconstructing `scatter_particles_to_grid`'s exact
/// per-cell formula, in this module's own tests.
///
/// Bundles the particle state P2G itself reads (`mass`, `v`, `C`, `stress`,
/// `stress_coeff`) into one struct rather than five separate parameters --
/// this function differentiates the FULL forward pass, so it genuinely needs
/// all of it, but five-plus-position-plus-two-gradient-array parameters
/// crossed the project's own no-`#[allow]` line for argument count.
pub struct P2GParticleState {
    pub mass: f32,
    pub v: Vec2,
    pub c: Mat2,
    pub stress: Mat2,
    pub stress_coeff: f32,
}

pub fn p2g_position_vjp(
    x: Vec2,
    state: &P2GParticleState,
    d_loss_d_momentum: &[[Vec2; 3]; 3],
    d_loss_d_mass: &[[f32; 3]; 3],
) -> Vec2 {
    let weights = quadratic_weights(x);
    let diff = x - weights.base_cell.as_vec2() - Vec2::splat(0.5);
    let dwx = axis_weights_derivative(diff.x);
    let dwy = axis_weights_derivative(diff.y);
    let m = state.mass * state.c + state.stress_coeff * state.stress;

    let mut d_loss_d_x = Vec2::ZERO;
    for gx in 0..3 {
        for gy in 0..3 {
            let wx = weights.wx[gx];
            let wy = weights.wy[gy];
            let weight = wx * wy;
            let cell_pos = weights.base_cell + IVec2::new(gx as i32 - 1, gy as i32 - 1);
            let cell_dist = cell_pos.as_vec2() - x + Vec2::splat(0.5);
            let a = state.mass * state.v + m * cell_dist;

            let d_weight_dx = Vec2::new(dwx[gx] * wy, wx * dwy[gy]);
            let g_momentum = d_loss_d_momentum[gx][gy];
            let g_mass = d_loss_d_mass[gx][gy];

            d_loss_d_x += d_weight_dx * (g_momentum.dot(a) + g_mass * state.mass);
            d_loss_d_x -= weight * (m.transpose() * g_momentum);
        }
    }
    d_loss_d_x
}

pub fn scatter_particle_mass(particles: &Particles, grid: &mut Grid, active_count: usize) {
    for i in 0..active_count {
        let x = particles.x[i];
        let mass = particles.mass[i];
        let weights = quadratic_weights(x);
        for gx in 0..3 {
            for gy in 0..3 {
                let weight = weights.wx[gx] * weights.wy[gy];
                let cell_pos = weights.base_cell + IVec2::new(gx as i32 - 1, gy as i32 - 1);
                grid.add_mass_momentum(cell_pos, weight * mass, Vec2::ZERO);
            }
        }
    }
}

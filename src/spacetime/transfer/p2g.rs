use glam::{IVec2, Mat2, Vec2};
use rayon::prelude::*;

use crate::grid::kernel::{axis_weights_derivative, quadratic_weights};
use crate::grid::{
    CellMap, FrictionCellMap, Grid, accumulate_friction, flat_index, merge_friction_maps,
};
use crate::materials::registry::MaterialRegistry;
use crate::particle::Particles;
use crate::solver::config::KERNEL_D_INVERSE;

use super::{combined_kirchhoff_stress, combined_kirchhoff_stress_from};

/// Decomposition of one grid node's P2G contribution, for the
/// structural-boundary impulse ledger (research diagnostic).
#[cfg(any(test, feature = "research-diagnostics"))]
#[derive(Clone, Copy, Debug, Default)]
pub struct GridNodeP2GComponents {
    pub mass: f32,
    pub translation_momentum: Vec2,
    pub affine_momentum: Vec2,
    pub stress_momentum: Vec2,
    /// Volume-weighted lower-wall normal Kirchhoff stress numerator and
    /// denominator. Their ratio estimates `n·tau·n` at the node; unlike the
    /// stress *force* above, its sign is the contact-traction sign.
    pub weighted_tau_yy: f32,
    pub stress_volume_weight: f32,
}

/// Reconstruct the exact particle part of P2G, separated into translation,
/// APIC-affine and stress impulses for every in-domain node.
///
/// This intentionally mirrors `scatter_one_into` term for term and is called
/// from inside the substep with that attempt's own `dt`. Rod/grain scatters
/// are outside its scope; the controlled boundary experiment contains neither.
#[cfg(any(test, feature = "research-diagnostics"))]
pub fn diagnose_grid_p2g_components(
    particles: &Particles,
    materials: &MaterialRegistry,
    dt: f32,
    resolution: usize,
    active_count: usize,
) -> Vec<GridNodeP2GComponents> {
    let mut out = vec![GridNodeP2GComponents::default(); resolution * resolution];
    for i in 0..active_count {
        let material_id = particles.material_id[i];
        let x = particles.x[i];
        let mass_i = particles.mass[i];
        let v_i = particles.v[i];
        let c_i = particles.velocity_gradient[i];
        let passive_tau = materials.kirchhoff_stress(material_id, particles, i);
        let material = materials.get(material_id);
        let stress = combined_kirchhoff_stress_from(passive_tau, material, particles, i);
        let stress_coeff =
            -materials.stress_volume(material_id, particles, i) * KERNEL_D_INVERSE * dt;
        let weights = quadratic_weights(x);
        for gx in 0..3 {
            for gy in 0..3 {
                let cell_pos = weights.base_cell + IVec2::new(gx as i32 - 1, gy as i32 - 1);
                let Some(idx) = flat_index(cell_pos, resolution) else {
                    continue;
                };
                let weight = weights.wx[gx] * weights.wy[gy];
                let cell_dist = cell_pos.as_vec2() - x + Vec2::splat(0.5);
                let node = &mut out[idx as usize];
                node.mass += weight * mass_i;
                node.translation_momentum += weight * mass_i * v_i;
                node.affine_momentum += weight * mass_i * (c_i * cell_dist);
                node.stress_momentum += weight * stress_coeff * (stress * cell_dist);
                let stress_volume = materials.stress_volume(material_id, particles, i);
                node.weighted_tau_yy += weight * stress_volume * stress.y_axis.y;
                node.stress_volume_weight += weight * stress_volume;
            }
        }
    }
    out
}

/// Scatters ONE particle's mass/momentum contribution into a thread-local
/// `CellMap` accumulator -- factored out of `scatter_particles_to_grid`'s
/// fold closure so `scatter_particles_to_grid_sorted` (spatial-sort opt-in,
/// see that function's doc) can share the exact same per-particle math
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

    // Node momentum `m (v + C d) + s (S d)` is `A + B d`, with `A = m v` and
    // `B = m C + s S` built once per particle, and `d` steps by one cell
    // between nodes, so each node adds `B`'s columns to `A + B d0` instead of
    // computing two matrix-vector products. The same sum, rounded in a
    // different order (`p2g_cost_breakdown`: the node math 106-128 ns -> 55 ns
    // per particle).
    let weights = quadratic_weights(x);
    let affine_and_stress = c_i * mass_i + stress * stress_coeff;
    let cell_dist_base = weights.base_cell.as_vec2() - x + Vec2::splat(0.5);
    let momentum_at_base = mass_i * v_i + affine_and_stress * cell_dist_base;
    for gx in 0..3 {
        for gy in 0..3 {
            let weight = weights.wx[gx] * weights.wy[gy];
            let cell_pos = weights.base_cell + IVec2::new(gx as i32 - 1, gy as i32 - 1);
            let Some(idx) = flat_index(cell_pos, resolution) else {
                continue;
            };
            let momentum = weight
                * (momentum_at_base
                    + (gx as f32 - 1.0) * affine_and_stress.x_axis
                    + (gy as f32 - 1.0) * affine_and_stress.y_axis);
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
/// Parallel through a thread-local `CellMap` fold/reduce, merged into the grid
/// in one serial pass (`Grid::merge_cells`): safe Rust, no shared mutable
/// state during the parallel phase (each rayon task owns its `CellMap`).
///
/// Not a dense `Vec<Cell>`: although a serial microbenchmark puts a HashMap
/// insert at 6.5x an indexed write, rayon's fold/reduce makes many small
/// accumulators, each paying a full `resolution^2` allocation and zeroing
/// (330-375 ms against ~20 ms). The lazily growing map only allocates what a
/// chunk touches.
///
/// The parallel fold changes the floating-point summation order at cells
/// shared by several particles, which can move a chaotic run: checked
/// against `fluid_spreads_more_than_elastic_under_gravity` (600 steps,
/// asserting `ar_fluid_final > ar_elastic_final`) and the full suite.
///
/// On 8 threads (4 cores) this fold runs no faster than a serial map
/// scatter (57 600 particles: 4.8-5.6 ms either way), yet the structure is
/// not what costs. Measured serially, a particle's work here is 220-320 ns,
/// of which its Kirchhoff stress is about 35 ns and the nine weighted adds
/// about 25 ns; the rest is the per-particle math of `scatter_one_into`.
/// Two rewrites aimed at the storage and the reduce were measured against
/// this one, alternating release binaries, and reverted:
/// - Block-sparse cell storage (8x8 blocks, a block table, touched masks):
///   grid passes +30 to 40 %, a serial scatter slower than the map, P2G
///   unchanged; the bookkeeping costs as much as the hash it replaces.
/// - Per-block 10x10 tiles filled in parallel, then committed to the map:
///   P2G -9 to -12 %, but grid passes +50 to 100 % on a large body (not
///   measured to a cause) and +30 % P2G on a small body in
///   a 1024-cell domain (the counting sort walks every block of the domain
///   each substep).
///
/// Pinned nodes and the contact, mixture and friction scatters follow in a
/// serial second pass, `scatter_second_pass`.
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
    // measured, KEPT win on its own: p2g_us 20ms -> 13ms, ~22fps ->
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
    scatter_second_pass(particles, grid, materials, dt, active_count);
}

/// The stress term of `scatter_one_into`'s node momentum on its own,
/// `sum_p w_ip s_p (S_p d_ip)`, for every node a particle touches. Each
/// entry's `momentum` holds that impulse; `mass` stays zero.
///
/// ASFLIP's FLIP residual and Cundall damping both need the grid velocity
/// from before this substep's forces. The fused MLS-MPM transfer has
/// already added the stress impulse to the node momentum, so
/// `Grid::snapshot_velocities_before_stress` takes this map back out. Fei,
/// Guo, Wu, Huang and Gao 2021 (ACM TOG 40(4) 109, Eq. 12) transfer
/// `m_p (v_p + C_p (x_i - x_p))` alone for that velocity, and their
/// reference code (pyasflip) scatters it as a separate `grid_v0`. Left in,
/// the residual cancels the stress force: at blend 0.97 particles received
/// 3 % of it while `F` kept straining, and an elastic shear wave gained
/// 12-34x its energy.
///
/// Recomputes each particle's stress, so it is called only when one of
/// the two features is on.
pub(crate) fn scatter_particle_stress_impulse(
    particles: &Particles,
    materials: &MaterialRegistry,
    dt: f32,
    resolution: usize,
    active_count: usize,
) -> CellMap {
    let min_len = (active_count / (rayon::current_num_threads() * 2)).max(1);
    (0..active_count)
        .into_par_iter()
        .with_min_len(min_len)
        .fold(CellMap::default, |mut acc, i| {
            let material_id = particles.material_id[i];
            let x = particles.x[i];
            let passive_tau = materials.kirchhoff_stress(material_id, particles, i);
            let material = materials.get(material_id);
            let stress = combined_kirchhoff_stress_from(passive_tau, material, particles, i);
            let stress_coeff =
                -materials.stress_volume(material_id, particles, i) * KERNEL_D_INVERSE * dt;
            let weights = quadratic_weights(x);
            let impulse = stress * stress_coeff;
            for gx in 0..3 {
                for gy in 0..3 {
                    let cell_pos = weights.base_cell + IVec2::new(gx as i32 - 1, gy as i32 - 1);
                    let Some(idx) = flat_index(cell_pos, resolution) else {
                        continue;
                    };
                    let weight = weights.wx[gx] * weights.wy[gy];
                    let cell_dist = cell_pos.as_vec2() - x + Vec2::splat(0.5);
                    acc.entry(idx).or_default().momentum += weight * (impulse * cell_dist);
                }
            }
            acc
        })
        .reduce(CellMap::default, |mut a, b| {
            merge_cell_maps(&mut a, b);
            a
        })
}

/// Opt-in spatial-sort variant of `scatter_particles_to_grid`'s dense first
/// pass (`SimConfig::spatial_sort_enabled`; Gao et al. 2018 SIGGRAPH Asia,
/// "GPU Optimization of Material Point Methods": periodic reordering for cache
/// locality). Like the GPU path (`particle_sort.wgsl`'s `sorted_particle_ids`)
/// it goes through an indirection array instead of moving the `Particles`
/// SoA, so particle indices never move.
///
/// `order` must contain each of `0..active_count` exactly once (a
/// permutation), as `spatial_sort_order` returns. Only the `CellMap` pass is
/// reordered; the contact/mixture pass keeps iterating `0..active_count` (it
/// is not fold-accumulated, so it gains nothing).
///
/// Reordering changes which particles share a rayon chunk and so the
/// floating-point summation order at shared cells, which once moved a
/// chaotic test's outcome (`fluid_spreads_more_than_elastic_under_gravity`):
/// check it and the full suite before relying on this.
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
    scatter_second_pass(particles, grid, materials, dt, active_count);
}

/// The pass after the parallel `CellMap` scatter, shared by
/// `scatter_particles_to_grid` and its sorted variant.
///
/// Marks the grid nodes a pinned particle supports: essential boundary
/// conditions are constraints on grid DOFs, not a post-G2P particle reset,
/// and the solver enforces these nodes after all grid forces and immediately
/// before G2P.
///
/// Then adds the extra per-field scatters a particle opts into: multi-field
/// contact (Bardenhagen 2001, `Particle::contact_group`), a mixture phase
/// (Tampubolon et al. 2017, `WithMixturePhase`), and material-induced
/// boundary friction (Blatny & Gaume 2025,
/// `MaterialModel::current_friction_coefficient`). Contact and mixture take
/// the particle's momentum, so they recompute its stress (the same pure
/// functions as the parallel pass, the same result). Friction takes only
/// each node's mass share, so a particle carrying friction alone skips that
/// recompute: every Drucker-Prager or mu(I) sand particle, each substep.
///
/// Those friction-only particles are the bulk of a sand scene, and their
/// scatter is an additive reduction (`sum(w*m*mu)`, `sum(w*m)` per node),
/// so it runs as a parallel fold/reduce like the main scatter, with the
/// same consequence: the summation order at shared nodes changes, so the
/// node coefficient can differ in its last bits from a serial sum. The rest
/// (pinned nodes, contact, mixture) stays serial.
fn scatter_second_pass(
    particles: &Particles,
    grid: &mut Grid,
    materials: &MaterialRegistry,
    dt: f32,
    active_count: usize,
) {
    let resolution = grid.resolution();
    let min_len = (active_count / (rayon::current_num_threads() * 2)).max(1);
    let friction: FrictionCellMap = (0..active_count)
        .into_par_iter()
        .with_min_len(min_len)
        .fold(FrictionCellMap::default, |mut acc, i| {
            if particles.contact_group[i] == 0
                && materials
                    .get(particles.material_id[i])
                    .mixture_phase()
                    .is_none()
                && let Some(mu) =
                    materials.current_friction_coefficient(particles.material_id[i], particles, i)
            {
                let mass_i = particles.mass[i];
                let weights = quadratic_weights(particles.x[i]);
                for gx in 0..3 {
                    for gy in 0..3 {
                        let weight = weights.wx[gx] * weights.wy[gy];
                        let cell_pos = weights.base_cell + IVec2::new(gx as i32 - 1, gy as i32 - 1);
                        if let Some(idx) = flat_index(cell_pos, resolution) {
                            accumulate_friction(&mut acc, idx, weight * mass_i, mu);
                        }
                    }
                }
            }
            acc
        })
        .reduce(FrictionCellMap::default, |mut a, b| {
            merge_friction_maps(&mut a, b);
            a
        });
    grid.merge_friction_cells(friction);

    for i in 0..active_count {
        if particles.pinned[i] != 0 {
            let weights = quadratic_weights(particles.x[i]);
            for gx in 0..3 {
                for gy in 0..3 {
                    if weights.wx[gx] * weights.wy[gy] > 0.0 {
                        let cell_pos = weights.base_cell + IVec2::new(gx as i32 - 1, gy as i32 - 1);
                        grid.mark_pinned_node(cell_pos);
                    }
                }
            }
        }

        let contact_group = particles.contact_group[i];
        let material = materials.get(particles.material_id[i]);
        let mixture_phase = material.mixture_phase();
        if contact_group == 0 && mixture_phase.is_none() {
            // Friction-only or plain: scattered above, or nothing to add.
            continue;
        }
        let friction_coefficient =
            materials.current_friction_coefficient(particles.material_id[i], particles, i);

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
                if let Some(mu) = friction_coefficient {
                    grid.add_friction_mass(cell_pos, weight * mass_i, mu);
                }
            }
        }
    }
}

/// Computes a spatial sort permutation of `0..active_count`, ordered by
/// each particle's own P2G stencil center (`quadratic_weights(x).base_cell`,
/// the SAME cell the scatter loop itself keys into) flattened to a single
/// grid-row-major index -- particles that land in the same or nearby grid
/// cells end up adjacent in the returned order, so a rayon chunk (a
/// contiguous slice of this order) touches far fewer DISTINCT `CellMap`
/// entries than a chunk of spawn-order particles that have since drifted
/// apart spatially. Not guessed: `CellMap` is a `HashMap`, and this
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
/// node that already exists in `contact_cells` (never creates one) -- so running this
/// before the first pass completes would silently miss point-cloud data for nodes whose
/// grip contribution hadn't been seen yet. Gated on `grid.has_contact_activity()`: a full
/// no-op, not even a loop iteration, for every scene that never sets
/// `Particle::contact_group` -- the same zero-cost-when-unused property as the rest of
/// this feature.
///
/// Most particles in a contact scene touch no contact-active node (a whole
/// sand bed around one body), yet each one used to pay for its inverse
/// deformation and nine hash lookups serially: 25% of a substep on
/// `tests/probes/contact_cost.rs`'s scene. The lookups now run in parallel
/// over contiguous particle chunks, read-only, and only particles with a hit
/// compute their extent. The hits are appended serially in chunk order, so
/// every node receives its points in exactly the particle order the serial
/// loop gave it, and the normal fit sees the same input.
pub fn gather_contact_point_cloud(
    particles: &Particles,
    grid: &mut Grid,
    materials: &MaterialRegistry,
    active_count: usize,
) {
    if !grid.has_contact_activity() {
        return;
    }
    let chunk = (active_count / (rayon::current_num_threads() * 2)).max(1);
    let chunk_count = active_count.div_ceil(chunk);
    let grid_view: &Grid = grid;
    let hits: Vec<Vec<ContactPointHit>> = (0..chunk_count)
        .into_par_iter()
        .map(|c| {
            let mut out = Vec::new();
            for i in c * chunk..((c + 1) * chunk).min(active_count) {
                let x = particles.x[i];
                let weights = quadratic_weights(x);
                let mut extent = None;
                for gx in 0i32..3 {
                    for gy in 0i32..3 {
                        let cell_pos = weights.base_cell + IVec2::new(gx - 1, gy - 1);
                        if !grid_view.is_contact_node(cell_pos) {
                            continue;
                        }
                        let (inverse_deformation, half_size) =
                            *extent.get_or_insert_with(|| particle_extent(particles, materials, i));
                        out.push(ContactPointHit {
                            cell_pos,
                            position: x,
                            label: if particles.contact_group[i] != 0 {
                                1.0
                            } else {
                                -1.0
                            },
                            inverse_deformation,
                            half_size,
                        });
                    }
                }
            }
            out
        })
        .collect();
    for hit in hits.into_iter().flatten() {
        grid.add_contact_point(
            hit.cell_pos,
            hit.position,
            hit.label,
            hit.inverse_deformation,
            hit.half_size,
        );
    }
}

/// One point `gather_contact_point_cloud` appends to a contact node.
struct ContactPointHit {
    cell_pos: IVec2,
    position: Vec2,
    label: f32,
    inverse_deformation: Mat2,
    half_size: f32,
}

/// Where a particle's deformed edge sits (Nairn, Hammerquist and Smith 2020,
/// eq. 25): the inverse of its deformation gradient and its undeformed half
/// size. The undeformed area is `mass / rest_density` when the material knows
/// its density (see `MaterialModel::rest_density` for why not
/// `initial_volume`).
fn particle_extent(particles: &Particles, materials: &MaterialRegistry, i: usize) -> (Mat2, f32) {
    let inverse_deformation = particles.deformation_gradient[i].inverse();
    let undeformed_area = materials
        .get(particles.material_id[i])
        .rest_density()
        .filter(|&rho| rho > 0.0)
        .map_or(particles.initial_volume[i], |rho| particles.mass[i] / rho);
    (inverse_deformation, 0.5 * undeformed_area.max(0.0).sqrt())
}

/// Analytic adjoint of P2G's stress→force scatter contribution w.r.t. the
/// particle's Kirchhoff stress, the step after
/// `NeoHookeanMaterial::kirchhoff_stress_vjp` in differentiable stepping.
///
/// Only the elastic-force term `weight * stress_coeff * (stress * cell_dist)`
/// of `scatter_particles_to_grid`, with the position `x` (so the weights and
/// `cell_dist`) fixed. Not covered: the mass/velocity/affine term (a simpler
/// linear adjoint) and the weights' dependence on `x`. This covers the
/// control path (muscle activation → stress → grid force) a controller is
/// trained through.
///
/// Derivation: for one particle, cell `c`'s momentum from stress is `y_c =
/// (weight_c * stress_coeff) * (stress * cell_dist_c)`, linear in `stress`, a
/// scaled matrix-vector product `y = M*v`. Given each cell's momentum gradient
/// `d_loss_d_momentum[c]` (a Vec2), the VJP of `y = Mv` is
/// `dL/dM = outer(dL/dy, v)`. Summed over the 9 stencil cells:
///
///   d_loss_d_stress = sum_c (weight_c * stress_coeff) * outer(d_loss_d_momentum[c], cell_dist_c)
///
/// Returns d_loss_d_stress, to feed e.g.
/// `NeoHookeanMaterial::kirchhoff_stress_vjp` to continue the chain back to F.
/// Checked against central-difference gradients in the tests.
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
/// this function differentiates the FULL forward pass, so it needs
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

/// One material's real contribution to one grid node's P2G mass/momentum --
/// see `diagnose_particle_node_material_sources`'s doc.
#[derive(Debug, Clone, Copy)]
pub struct NodeMaterialSource {
    pub material_id: u32,
    pub mass: f32,
    pub advective_momentum: Vec2,
    pub stress_momentum: Vec2,
}

/// One of a tracked particle's 9 P2G/G2P support nodes, broken down by
/// which material contributed what -- see `diagnose_particle_node_material_
/// sources`'s doc.
#[derive(Debug, Clone)]
pub struct TrackedNodeBreakdown {
    pub cell_pos: IVec2,
    /// This tracked particle's own G2P kernel weight for this node -- how
    /// much this node's velocity counts toward the tracked particle's next
    /// G2P gather.
    pub tracked_particle_weight: f32,
    /// Normalized velocity at this node, from an independent full
    /// `scatter_particles_to_grid` on a fresh grid (see this function's doc).
    pub real_velocity: Vec2,
    pub real_mass: f32,
    /// Every material's contribution to this node, reproducing `scatter_
    /// one_into`'s exact formula split by `material_id`.
    pub sources: Vec<NodeMaterialSource>,
}

/// Diagnostic: for one tracked particle's 9 support nodes, breaks down every
/// particle's mass and momentum contribution by `material_id`. Tests whether
/// a particle's runaway acceleration (a water particle before its boiling
/// threshold, `phase_states_gui.rs`'s Moon-gravity run) is inherited from a
/// buoyant steam neighbour through the shared grid: self-pressure impulses
/// sum to zero under partition of unity and APIC conserves linear momentum,
/// so neither accelerates an isolated particle as a whole.
///
/// Step 1 runs an independent `scatter_particles_to_grid` on a fresh `Grid`
/// for the merged mass and velocity at each node. Step 2 re-scans every
/// particle with `scatter_one_into`'s per-particle formula (advective and
/// stress-impulse terms apart), restricted to the 9 nodes and bucketed by
/// `material_id`; each node's buckets must sum to step 1's values exactly.
///
/// O(active_count * 9), for on-demand use, not the per-substep path.
pub fn diagnose_particle_node_material_sources(
    particles: &Particles,
    materials: &MaterialRegistry,
    dt: f32,
    resolution: usize,
    active_count: usize,
    tracked_index: usize,
) -> Vec<TrackedNodeBreakdown> {
    let mut ground_truth_grid = Grid::new(resolution);
    scatter_particles_to_grid(
        particles,
        &mut ground_truth_grid,
        materials,
        dt,
        active_count,
    );
    // `velocity_at` is only a velocity AFTER normalization -- see its
    // doc ("valid after update_velocities()"). Pure momentum/mass here
    // (no gravity/boundary), matching what P2G alone produces before the
    // grid-update phase -- the right snapshot for this diagnostic.
    ground_truth_grid.normalize_velocities();

    let tracked_x = particles.x[tracked_index];
    let tracked_weights = quadratic_weights(tracked_x);
    let mut breakdowns: Vec<TrackedNodeBreakdown> = Vec::with_capacity(9);
    for gx in 0..3 {
        for gy in 0..3 {
            let tracked_particle_weight = tracked_weights.wx[gx] * tracked_weights.wy[gy];
            let cell_pos = tracked_weights.base_cell + IVec2::new(gx as i32 - 1, gy as i32 - 1);
            breakdowns.push(TrackedNodeBreakdown {
                cell_pos,
                tracked_particle_weight,
                real_velocity: ground_truth_grid.velocity_at(cell_pos),
                real_mass: ground_truth_grid.mass_at(cell_pos),
                sources: Vec::new(),
            });
        }
    }

    for i in 0..active_count {
        let material_id = particles.material_id[i];
        let x = particles.x[i];
        let mass_i = particles.mass[i];
        let v_i = particles.v[i];
        let c_i = particles.velocity_gradient[i];

        let passive_tau = materials.kirchhoff_stress(material_id, particles, i);
        let material = materials.get(material_id);
        let stress = combined_kirchhoff_stress_from(passive_tau, material, particles, i);
        let stress_coeff =
            -materials.stress_volume(material_id, particles, i) * KERNEL_D_INVERSE * dt;

        let weights = quadratic_weights(x);
        for gx in 0..3 {
            for gy in 0..3 {
                let cell_pos = weights.base_cell + IVec2::new(gx as i32 - 1, gy as i32 - 1);
                let Some(target) = breakdowns.iter_mut().find(|b| b.cell_pos == cell_pos) else {
                    continue;
                };
                let weight = weights.wx[gx] * weights.wy[gy];
                let cell_dist = cell_pos.as_vec2() - x + Vec2::splat(0.5);
                let mass = weight * mass_i;
                let advective_momentum = weight * mass_i * (v_i + c_i * cell_dist);
                let stress_momentum = weight * stress_coeff * (stress * cell_dist);

                if let Some(entry) = target
                    .sources
                    .iter_mut()
                    .find(|s| s.material_id == material_id)
                {
                    entry.mass += mass;
                    entry.advective_momentum += advective_momentum;
                    entry.stress_momentum += stress_momentum;
                } else {
                    target.sources.push(NodeMaterialSource {
                        material_id,
                        mass,
                        advective_momentum,
                        stress_momentum,
                    });
                }
            }
        }
    }

    breakdowns
}

/// Three-way decomposition of a tracked particle's pre-grid-update `tr(C)`:
/// translation (`w*m*v`), affine (`w*m*C*d`), stress (`w*stress_coeff*(tau*d)`),
/// separating the particle's own EOS pressure pushing its neighbours
/// (`trace_stress`) from inherited motion (`trace_translation` +
/// `trace_affine`).
///
/// Call it inside the executing substep, right after the P2G scatter and
/// with its `dt` (see `Simulation::do_substep`), so a retry's `actual_dt`
/// cannot differ: it recomputes `scatter_one_into`'s per-particle formula for
/// the particle's 9-node stencil, bucketed by contribution.
///
/// The three buckets share the node's total mass, so `trace_translation +
/// trace_affine + trace_stress` equals the combined `trace(C)` to float
/// precision. Returns `(trace_translation, trace_affine, trace_stress)`.
///
/// O(active_count), for on-demand use, not the hot path.
pub fn diagnose_particle_divergence_decomposition(
    particles: &Particles,
    materials: &MaterialRegistry,
    dt: f32,
    active_count: usize,
    tracked_index: usize,
    apic_blend: f32,
) -> (f32, f32, f32) {
    let tracked_x = particles.x[tracked_index];
    let tracked_weights = quadratic_weights(tracked_x);
    let mut node_cell_pos = [IVec2::ZERO; 9];
    for gx in 0..3 {
        for gy in 0..3 {
            node_cell_pos[gx * 3 + gy] =
                tracked_weights.base_cell + IVec2::new(gx as i32 - 1, gy as i32 - 1);
        }
    }
    let mut node_mass = [0.0f32; 9];
    let mut node_translation = [Vec2::ZERO; 9];
    let mut node_affine = [Vec2::ZERO; 9];
    let mut node_stress = [Vec2::ZERO; 9];

    for i in 0..active_count {
        let material_id = particles.material_id[i];
        let x = particles.x[i];
        let mass_i = particles.mass[i];
        let v_i = particles.v[i];
        let c_i = particles.velocity_gradient[i];

        let passive_tau = materials.kirchhoff_stress(material_id, particles, i);
        let material = materials.get(material_id);
        let stress = combined_kirchhoff_stress_from(passive_tau, material, particles, i);
        let stress_coeff = -materials.stress_volume(material_id, particles, i)
            * crate::solver::config::KERNEL_D_INVERSE
            * dt;

        let weights = quadratic_weights(x);
        for gx in 0..3 {
            for gy in 0..3 {
                let cell_pos = weights.base_cell + IVec2::new(gx as i32 - 1, gy as i32 - 1);
                let Some(node_idx) = node_cell_pos.iter().position(|&c| c == cell_pos) else {
                    continue;
                };
                let weight = weights.wx[gx] * weights.wy[gy];
                let cell_dist = cell_pos.as_vec2() - x + Vec2::splat(0.5);
                node_mass[node_idx] += weight * mass_i;
                node_translation[node_idx] += weight * mass_i * v_i;
                node_affine[node_idx] += weight * mass_i * (c_i * cell_dist);
                node_stress[node_idx] += weight * stress_coeff * (stress * cell_dist);
            }
        }
    }

    let mut b_translation = Mat2::ZERO;
    let mut b_affine = Mat2::ZERO;
    let mut b_stress = Mat2::ZERO;
    for gx in 0..3 {
        for gy in 0..3 {
            let idx = gx * 3 + gy;
            let weight = tracked_weights.wx[gx] * tracked_weights.wy[gy];
            let cell_pos = node_cell_pos[idx];
            let dist = cell_pos.as_vec2() - tracked_x + Vec2::splat(0.5);
            let mass = node_mass[idx].max(1.0e-12);
            let v_translation = weight * (node_translation[idx] / mass);
            let v_affine = weight * (node_affine[idx] / mass);
            let v_stress = weight * (node_stress[idx] / mass);
            b_translation += Mat2::from_cols(v_translation * dist.x, v_translation * dist.y);
            b_affine += Mat2::from_cols(v_affine * dist.x, v_affine * dist.y);
            b_stress += Mat2::from_cols(v_stress * dist.x, v_stress * dist.y);
        }
    }
    let scale = crate::solver::config::KERNEL_D_INVERSE * apic_blend;
    let c_translation = b_translation * scale;
    let c_affine = b_affine * scale;
    let c_stress = b_stress * scale;
    (
        c_translation.x_axis.x + c_translation.y_axis.y,
        c_affine.x_axis.x + c_affine.y_axis.y,
        c_stress.x_axis.x + c_stress.y_axis.y,
    )
}

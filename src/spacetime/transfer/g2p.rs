use glam::{IVec2, Mat2, Vec2};
use rayon::prelude::*;

use crate::boundary::BoundaryCondition;
use crate::grid::Grid;
use crate::grid::kernel::quadratic_weights;
use crate::materials::registry::MaterialRegistry;
use crate::particle::{ParticleUpdateCtx, Particles};
use crate::solver::config::KERNEL_D_INVERSE;

/// Raw pointers to every mutable field `gather_grid_to_particles`' merged
/// parallel pass needs, taken once before the loop -- see that function's own
/// SAFETY comment for why indexing these concurrently is sound.
struct MutFieldPtrs {
    x: *mut Vec2,
    v: *mut Vec2,
    velocity_gradient: *mut Mat2,
    deformation_gradient: *mut Mat2,
    volume: *mut f32,
    density: *mut f32,
    hardening_scale: *mut f32,
    plastic_volume_ratio: *mut f32,
    log_volume_strain: *mut f32,
    friction_hardening: *mut f32,
    /// Kahan compensation residual for `x`'s position integration -- see
    /// `Particles::position_compensation`'s own doc. Deliberately NOT part
    /// of `ParticleUpdateCtx` (that struct is public API for external
    /// `MaterialModel`/`BoundaryCondition` implementors; this is a pure
    /// internal scratch value with no meaning outside this function), so
    /// it gets its own accessor method below instead.
    position_compensation: *mut Vec2,
}
// SAFETY: raw pointers aren't Send/Sync by default, but this type is only ever
// used to derive disjoint per-index references (see the SAFETY comment where
// it's constructed) -- sharing the pointers themselves across threads is safe,
// only concurrent access to the SAME index would not be, and that never happens.
unsafe impl Send for MutFieldPtrs {}
unsafe impl Sync for MutFieldPtrs {}

impl MutFieldPtrs {
    /// Builds the real `ParticleUpdateCtx` directly, for one index -- a method
    /// (not direct field access) on purpose: Rust 2021's disjoint closure
    /// captures would otherwise capture individual `*mut T` fields directly
    /// (never `Sync`, even though the `MutFieldPtrs` wrapper is), silently
    /// bypassing the `unsafe impl Sync` above. Routing through a method call
    /// forces the closure to capture `MutFieldPtrs` as a whole instead.
    ///
    /// SAFETY: caller must ensure `i` is unique across every concurrent call
    /// (see the SAFETY comment where `MutFieldPtrs` is constructed).
    #[allow(clippy::too_many_arguments)]
    unsafe fn ctx_at(
        &self,
        i: usize,
        mass: f32,
        temperature: f32,
        initial_volume: f32,
        activation: f32,
        activation_dir: Vec2,
        nonlocal_fluidity: f32,
        cosserat_curvature: Vec2,
    ) -> ParticleUpdateCtx<'_> {
        unsafe {
            ParticleUpdateCtx {
                x: &mut *self.x.add(i),
                v: &mut *self.v.add(i),
                velocity_gradient: &mut *self.velocity_gradient.add(i),
                deformation_gradient: &mut *self.deformation_gradient.add(i),
                volume: &mut *self.volume.add(i),
                density: &mut *self.density.add(i),
                hardening_scale: &mut *self.hardening_scale.add(i),
                plastic_volume_ratio: &mut *self.plastic_volume_ratio.add(i),
                log_volume_strain: &mut *self.log_volume_strain.add(i),
                friction_hardening: &mut *self.friction_hardening.add(i),
                mass,
                temperature,
                initial_volume,
                activation,
                activation_dir,
                nonlocal_fluidity,
                cosserat_curvature,
            }
        }
    }

    /// SAFETY: same contract as `ctx_at` -- caller must ensure `i` is unique
    /// across every concurrent call. A real, separate method (not direct
    /// field access) for the same reason `ctx_at` is: routing through a
    /// method call forces the closure to capture `MutFieldPtrs` as a whole,
    /// not the individual (non-`Sync`) raw pointer field.
    ///
    /// Clippy's `mut_from_ref` heuristic can't see the disjoint-index proof
    /// above (same as `ctx_at`'s identical pattern); the `&self` receiver is
    /// required so callers keep capturing `MutFieldPtrs` as a whole.
    #[allow(clippy::mut_from_ref)]
    unsafe fn position_compensation_at(&self, i: usize) -> &mut Vec2 {
        unsafe { &mut *self.position_compensation.add(i) }
    }
}

pub struct G2PParams<'a> {
    pub apic_blend: f32,
    pub active_count: usize,
    /// ASFLIP blend factor (`SimConfig::asflip_blend`, Fei et al. 2021). 0.0 = disabled,
    /// the exact original G2P formula below (see `pre_force_snapshot`'s doc for the gate).
    pub asflip_blend: f32,
    /// The grid's pre-force velocity snapshot (see `Grid::snapshot_velocities`), or `None`
    /// when ASFLIP is disabled. This, not `asflip_blend` alone, is the real gate: the ASFLIP
    /// correction below only runs when `Some`, so a caller that never opts in (passes `None`)
    /// gets the byte-identical original code path regardless of what `asflip_blend` holds.
    pub pre_force_snapshot: Option<&'a crate::grid::VelocitySnapshot>,
    /// Gathered granular fluidity `g`, one entry per active particle, from a
    /// coupled `GranularFluidityField` computed at the END of the PREVIOUS
    /// substep (see `Simulation::step`'s own ordering comment for why this
    /// one-substep lag matches the already-established thermal/scalar-field
    /// convention, not a shortcut specific to this feature). Empty (`&[]`)
    /// when no such field is configured for this scene -- every read below
    /// falls back to `0.0` in that case, matching `ParticleUpdateCtx::
    /// nonlocal_fluidity`'s own real-rest-state default.
    pub nonlocal_fluidity: &'a [f32],
    /// Gathered Cosserat micro-curvature, one entry per active particle,
    /// from a coupled `CosseratField` computed at the END of the PREVIOUS
    /// substep -- same one-substep-lag convention as `nonlocal_fluidity`
    /// above. Empty (`&[]`) when no such field is configured for this scene
    /// -- every read below falls back to `Vec2::ZERO`, matching
    /// `ParticleUpdateCtx::cosserat_curvature`'s own real-rest-state default.
    pub cosserat_curvature: &'a [Vec2],
}

/// Analytic adjoint of G2P's velocity gather (`new_v = sum_c weight_c *
/// grid.velocity_at(cell_c)`, see `gather_grid_to_particles`'s Phase 1) w.r.t.
/// the 9 grid velocities in the particle's stencil -- fifth real piece of
/// differentiable stepping, and the mathematical transpose of
/// `p2g_stress_vjp`: same quadratic kernel weights, same 3x3 stencil, but
/// scattering a gradient back out to the grid instead of gathering a value in
/// from it (the well-known P2G/G2P transpose relationship in MPM literature,
/// e.g. Jiang et al. 2016 "The Material Point Method for Simulating
/// Continuum Materials", carries over directly to differentiation).
///
/// SCOPED, matching the P2G adjoint's own scoping: treats particle position
/// `x` (and therefore the kernel weights) as FIXED. Also covers only the new
/// velocity `new_v`, not the APIC affine matrix `b`/`velocity_gradient` G2P
/// computes alongside it (`b = sum_c weight_c * outer(v_grid_c, dist_c)`) --
/// a related, still-open piece: same per-cell structure, needs its own
/// derivation and verification, not silently folded in here. Also doesn't
/// cover the position boundary condition applied after this in the real
/// G2P (piecewise/conditional, same deferred-with-a-name status as the
/// grid-update boundary gap).
///
/// Given the gradient flowing back from the particle's new velocity,
/// `d_loss_d_new_v` (a Vec2), the adjoint of a weighted sum distributes it
/// back to each grid cell by the SAME weight it was gathered with:
///
///   d_loss_d_v_grid[c] = weight_c * d_loss_d_new_v
///
/// Returns the per-cell gradient in the same `[[Vec2; 3]; 3]` shape
/// `p2g_stress_vjp` consumes, so a real trainer can pass this straight
/// through to the P2G side once both meet at the same grid cells. Verified
/// against central-difference numerical gradients in this module's own
/// tests.
pub fn g2p_velocity_vjp(x: Vec2, d_loss_d_new_v: Vec2) -> [[Vec2; 3]; 3] {
    let weights = quadratic_weights(x);
    let mut out = [[Vec2::ZERO; 3]; 3];
    for (row, wx) in out.iter_mut().zip(weights.wx.iter()) {
        for (cell, wy) in row.iter_mut().zip(weights.wy.iter()) {
            *cell = (wx * wy) * d_loss_d_new_v;
        }
    }
    out
}

/// Analytic adjoint of G2P's APIC affine matrix (`velocity_gradient`)
/// computation w.r.t. the 9 grid velocities -- the piece `g2p_velocity_vjp`
/// deliberately left open, now closed. Real, externally cross-checked: this
/// exact term appears in ChainQueen's own hand-written CUDA backward pass
/// (`backward.cu`, `P2G_backward`'s "(C)" comment) as
/// `invD * N * grad_C_next[alpha][beta] * dpos[beta]` -- confirms both that
/// this term is genuinely needed (not paranoia) and, since it algebraically
/// matches the independently-derived formula below once ChainQueen's `invD`
/// is read as this codebase's `KERNEL_D_INVERSE`, that the derivation is
/// right. `apic_blend` is an emerge-specific extra factor ChainQueen's own
/// formula doesn't have (see `gather_grid_to_particles`'s `vg = b *
/// KERNEL_D_INVERSE * apic_blend`), included here since it's part of
/// emerge's own forward formula.
///
/// Forward (see `gather_grid_to_particles`'s Phase 1): `new_c = scale *
/// sum_c weight_c * outer(v_grid_c, dist_c)`, where `scale =
/// KERNEL_D_INVERSE * apic_blend` and `outer(v,d)` has column 0 = `d.x*v`,
/// column 1 = `d.y*v` (same convention as `p2g_stress_vjp`'s own outer
/// product). Linear in each `v_grid_c`; given the gradient flowing back from
/// the affine matrix, `d_loss_d_new_c` (a Mat2), the VJP of `outer(v,d)`
/// w.r.t. `v` is `M*d` (matrix-vector product, standard result for an outer
/// product's adjoint):
///
///   d_loss_d_v_grid[c] = weight_c * scale * (d_loss_d_new_c * dist_c)
///
/// Callers combine this additively with `g2p_velocity_vjp`'s output (both
/// scatter to the SAME 9 grid cells, since `new_v` and `new_c` are computed
/// from the same stencil in the same G2P pass) to get the true total
/// per-cell gradient. Verified against central-difference numerical
/// gradients in this module's own tests, independently and composed with
/// `g2p_velocity_vjp`.
pub fn g2p_affine_vjp(
    x: Vec2,
    kernel_d_inverse: f32,
    apic_blend: f32,
    d_loss_d_new_c: Mat2,
) -> [[Vec2; 3]; 3] {
    let weights = quadratic_weights(x);
    let scale = kernel_d_inverse * apic_blend;
    let mut out = [[Vec2::ZERO; 3]; 3];
    for (gx, (row, wx)) in out.iter_mut().zip(weights.wx.iter()).enumerate() {
        for (gy, (cell, wy)) in row.iter_mut().zip(weights.wy.iter()).enumerate() {
            let cell_pos = weights.base_cell + IVec2::new(gx as i32 - 1, gy as i32 - 1);
            let dist = cell_pos.as_vec2() - x + Vec2::splat(0.5);
            *cell = (wx * wy * scale) * (d_loss_d_new_c * dist);
        }
    }
    out
}

/// Analytic adjoint of the deformation-gradient update `F_new = (I + dt*C) *
/// F_old` w.r.t. both `C` (the APIC affine matrix / velocity_gradient G2P
/// produces) and `F_old` -- sixth real piece of differentiable stepping, and
/// the one that actually CLOSES the loop: `C` comes from G2P, `F_old` is the
/// previous substep's deformation gradient, and this update's own output
/// (`F_new`) is exactly what `kirchhoff_stress_vjp` needs as input for the
/// NEXT substep. Chaining this repeatedly is what backprop-through-multiple-
/// substeps actually means.
///
/// This exact formula is universal MPM kinematics, not any one material's own
/// logic -- confirmed by grep: every material in `matter::materials`
/// (NeoHookean, Corotated, Viscoelastic, and every plastic model's F_trial
/// before its own return-mapping) computes `F_new`/`F_trial` this identical
/// way. Lives here in `spacetime::transfer`, not any material file, for that
/// reason.
///
/// Derivation: let `A = I + dt*C`, so `F_new = A * F_old` -- a plain matrix
/// product. Standard VJP for `Y = A*B`: `dL/dA = Ḡ*Bᵀ`, `dL/dB = Aᵀ*Ḡ`. Since
/// `A` is linear in `C` (`dA/dC = dt` component-wise), `dL/dC = dt * dL/dA`:
///
///   d_loss_d_C     = dt * (d_loss_d_F_new * F_oldᵀ)
///   d_loss_d_F_old = (I + dt*C)ᵀ * d_loss_d_F_new
///
/// Verified against central-difference numerical gradients in this module's
/// own tests, on both outputs independently.
pub fn f_update_vjp(c: Mat2, f_old: Mat2, dt: f32, d_loss_d_f_new: Mat2) -> (Mat2, Mat2) {
    let a = Mat2::IDENTITY + dt * c;
    let d_loss_d_c = dt * (d_loss_d_f_new * f_old.transpose());
    let d_loss_d_f_old = a.transpose() * d_loss_d_f_new;
    (d_loss_d_c, d_loss_d_f_old)
}

/// G2P: read grid velocities back into particles, advance state, apply boundaries.
///
/// The return value is retained for source compatibility with the old
/// `last_vel_clamp_count` diagnostic. The solver no longer clips velocities,
/// so it is always zero.
#[allow(clippy::too_many_arguments)]
pub fn gather_grid_to_particles(
    particles: &mut Particles,
    grid: &Grid,
    dt: f32,
    gravity: Vec2,
    boundary_thickness: usize,
    boundaries: &[Box<dyn BoundaryCondition>],
    materials: &MaterialRegistry,
    params: G2PParams,
) -> usize {
    let G2PParams {
        apic_blend,
        active_count,
        asflip_blend,
        pre_force_snapshot,
        nonlocal_fluidity,
        cosserat_curvature,
    } = params;
    let grid_res = grid.resolution();

    // Single parallel pass: grid gather -> v/velocity_gradient/position advance
    // -> plasticity update -> boundary post-hooks, one particle at a time, in
    // parallel across particles. Used to be two phases (a parallel gather, then
    // a forced-sequential plasticity/boundary pass, because `MaterialModel::
    // update_particle`/`BoundaryCondition::post_g2p_particle` used to need
    // `&mut Particles` -- the WHOLE struct -- per call). Now both take a
    // `ParticleUpdateCtx` (disjoint per-field borrows of just this particle's
    // own state), so the whole thing runs as one parallel loop -- real, measured
    // win: the plasticity/SVD update is the single most expensive per-particle
    // math in the solver, and it was serial before this.
    //
    // Read-only fields: ordinary indexed slices, safe (shared refs, no unsafe).
    let contact_groups = &particles.contact_group[..active_count];
    let pinned_flags = &particles.pinned[..active_count];
    let material_ids = &particles.material_id[..active_count];
    let masses = &particles.mass[..active_count];
    let temperatures = &particles.temperature[..active_count];
    let initial_volumes = &particles.initial_volume[..active_count];
    let activations = &particles.activation[..active_count];
    let activation_dirs = &particles.activation_dir[..active_count];
    // Mutable fields: raw pointers taken once, before the parallel loop --
    // avoids an 18-way nested `.zip()` (unreadable, error-prone to extend).
    // SAFETY: `(0..active_count).into_par_iter()` is an IndexedParallelIterator
    // -- every index in range is visited by exactly one task, so every pointer
    // offset below is to a genuinely distinct particle's own memory. Same
    // "unique indices never alias" argument `Grid::active_cells_mut` already
    // uses for the identical problem (many disjoint mutable borrows driven by
    // a known-unique index set).
    let ptrs = MutFieldPtrs {
        x: particles.x.as_mut_ptr(),
        v: particles.v.as_mut_ptr(),
        velocity_gradient: particles.velocity_gradient.as_mut_ptr(),
        deformation_gradient: particles.deformation_gradient.as_mut_ptr(),
        volume: particles.volume.as_mut_ptr(),
        density: particles.density.as_mut_ptr(),
        hardening_scale: particles.hardening_scale.as_mut_ptr(),
        plastic_volume_ratio: particles.plastic_volume_ratio.as_mut_ptr(),
        log_volume_strain: particles.log_volume_strain.as_mut_ptr(),
        friction_hardening: particles.friction_hardening.as_mut_ptr(),
        position_compensation: particles.position_compensation.as_mut_ptr(),
    };
    // Gate once, not per particle: when no grip particle ever touched the grid this
    // substep (every scene that doesn't use `Particle::contact_group`), this is false
    // and the loop below takes the exact same path it always has — a plain
    // `grid.velocity_at` lookup, no extra branching cost worth measuring.
    let contact_active = grid.has_contact_activity();
    // Same gate for two-phase mixture coupling (Tampubolon et al. 2017) — see
    // `WithMixturePhase` doc. False (the default) for every scene that never
    // wraps a material this way, same zero-cost property as contact above.
    let mixture_active = grid.has_mixture_activity();

    // Same real, measured lesson as P2G tonight (see
    // `transfer::p2g::scatter_particles_to_grid`'s own doc): rayon's default
    // chunking splits far more, far smaller tasks than a naive
    // one-per-core assumption. This loop has no per-chunk accumulator to
    // allocate (direct unique-pointer writes, not a fold/reduce), so the
    // failure mode that broke P2G's dense-buffer attempt doesn't apply --
    // but fewer/larger chunks still cuts rayon's own task-scheduling
    // overhead. Measure before trusting, same discipline as every change
    // tonight.
    let min_len = (active_count / (rayon::current_num_threads() * 2)).max(1);
    let clamp_count: usize = (0..active_count)
        .into_par_iter()
        .with_min_len(min_len)
        .map(|i| {
            let contact_group = contact_groups[i];
            let pinned = pinned_flags[i];
            let material_id = material_ids[i];
            let material = materials.get(material_id);
            // SAFETY: see the SAFETY comment on `ptrs` above -- index `i` is
            // unique to this task for the whole closure body below.
            let mut ctx = unsafe {
                ptrs.ctx_at(
                    i,
                    masses[i],
                    temperatures[i],
                    initial_volumes[i],
                    activations[i],
                    activation_dirs[i],
                    nonlocal_fluidity.get(i).copied().unwrap_or(0.0),
                    cosserat_curvature.get(i).copied().unwrap_or(Vec2::ZERO),
                )
            };
            // SAFETY: same contract as `ctx` above -- index `i` is unique to
            // this task.
            let pos_compensation = unsafe { ptrs.position_compensation_at(i) };
            let mixture_phase = if mixture_active {
                material.mixture_phase()
            } else {
                None
            };

            if pinned != 0 {
                // Dirichlet/kinematic anchor (`Particle::pinned`): force v=0 and
                // velocity_gradient=0 instead of gathering from the grid, so a
                // pinned particle never moves and never accumulates local strain
                // from being dragged — while its own mass/stress still scattered
                // into P2G normally, so it acts as a real, immovable anchor other
                // bodies push against (the standard technique for static/bedrock
                // geometry in deformable-body sims). Position is deliberately left
                // completely untouched, not just re-clamped to itself, avoiding any
                // float drift from a v=0*dt add-then-reclamp round trip. Plastic/
                // stress state (below) still updates normally either way.
                *ctx.v = Vec2::ZERO;
                *ctx.velocity_gradient = Mat2::ZERO;
            } else {
                let v_old = *ctx.v;
                let weights = quadratic_weights(*ctx.x);
                let mut new_v = Vec2::ZERO;
                let mut b = Mat2::ZERO;

                for gx in 0..3 {
                    for gy in 0..3 {
                        let weight = weights.wx[gx] * weights.wy[gy];
                        let cell_pos = weights.base_cell + IVec2::new(gx as i32 - 1, gy as i32 - 1);
                        let dist = cell_pos.as_vec2() - *ctx.x + Vec2::splat(0.5);
                        // Multi-field contact routing (Bardenhagen 2001): a grip particle
                        // reads the resolved grip field, a non-grip particle reads the
                        // resolved rest field, at nodes where contact was ever registered
                        // this substep. Both helpers fall back to the ordinary total
                        // velocity where no contact exists at that node, so this is exact
                        // everywhere, not just near contact.
                        let node_v = if contact_active {
                            if contact_group != 0 {
                                grid.grip_velocity_at(cell_pos)
                            } else {
                                grid.rest_velocity_at(cell_pos)
                            }
                        } else if let Some(phase) = mixture_phase {
                            // N-phase mixture coupling routing (generalizes Tampubolon et
                            // al. 2017): a phase-`p` particle reads that phase's own
                            // resolved velocity field, falling back to the ordinary total
                            // velocity where no coupling was registered at that node, same
                            // convention as contact.
                            grid.resolved_velocity_at(cell_pos, phase)
                        } else {
                            // Free-surface velocity extrapolation: empty nodes
                            // take THIS particle's own velocity, not ~zero --
                            // see `velocity_at_or_extrapolated`'s own doc for
                            // the measurement and the citation.
                            grid.velocity_at_or_extrapolated(
                                cell_pos,
                                *ctx.v,
                                gravity,
                                dt,
                                boundary_thickness,
                            )
                        };
                        let weighted_velocity = node_v * weight;
                        let term =
                            Mat2::from_cols(weighted_velocity * dist.x, weighted_velocity * dist.y);
                        b += term;
                        new_v += weighted_velocity;
                    }
                }

                // ASFLIP (Fei, Guo, Wu, Huang, Gao 2021, "Revisiting Integration in the
                // Material Point Method" -- see `SimConfig::asflip_blend` doc). Reintroduces
                // the classic FLIP residual (`v_p_old - old_v`) on top of the PIC/APIC gather
                // above -- `old_v` is a PIC-style gather against the grid's PRE-FORCE velocity
                // (`pre_force_snapshot`, taken right after P2G's own momentum normalization,
                // before this substep's gravity/boundary/contact modified it), using the SAME
                // stencil weights as `new_v` above. `pre_force_snapshot` being `None` (the
                // default, `asflip_blend=0.0`) is the real gate: `v_store`/`v_position` both
                // stay exactly `new_v`, reproducing the original formula below bit-for-bit.
                //
                // `gamma` (position-correction strength) is 0 while the local velocity
                // gradient indicates compression (`trace(b) < 0` -- two bodies pressing
                // together, e.g. a creature pushing into terrain via multi-field contact, or
                // material pressing against a boundary, since boundary conditions are already
                // baked into `new_v`/`b` by the time G2P reads the grid) and 1 while
                // separating -- exactly the paper's own "easier separation" adaptivity,
                // avoiding injecting extra positional noise while two bodies are in contact.
                let (mut v_store, mut v_position) = (new_v, new_v);
                if let Some(snapshot) = pre_force_snapshot {
                    let mut old_v = Vec2::ZERO;
                    for gx in 0..3 {
                        for gy in 0..3 {
                            let weight = weights.wx[gx] * weights.wy[gy];
                            let cell_pos =
                                weights.base_cell + IVec2::new(gx as i32 - 1, gy as i32 - 1);
                            old_v += grid.pre_force_velocity_at(snapshot, cell_pos) * weight;
                        }
                    }
                    let diff_vel = v_old - old_v;
                    let trace_b = b.x_axis.x + b.y_axis.y;
                    let gamma = if trace_b < 0.0 { 0.0 } else { 1.0 };
                    v_store = new_v + asflip_blend * diff_vel;
                    v_position = new_v + gamma * asflip_blend * diff_vel;
                }

                // Kahan (compensated) summation (Kahan 1965) -- same real
                // technique, same citation, `rod::coupling::gather_grid_to_rod`
                // already uses for rods (see `Particles::position_compensation`'s
                // own doc for the full derivation): a real, sustained velocity's
                // own `v*dt` increment can fall below f32's representable
                // precision at the particle's own grid-coordinate magnitude,
                // silently rounding away to nothing every substep even though
                // the underlying motion is real. Tracks the rounding error each
                // addition drops and folds it back in next time. The boundary
                // clamp below is a separate, unrelated, already-existing
                // mechanism -- Kahan compensation only concerns the raw
                // addition itself, not what a boundary does to the result
                // afterward.
                let y = v_position * dt - *pos_compensation;
                let t = *ctx.x + y;
                *pos_compensation = (t - *ctx.x) - y;
                let mut new_pos = t;
                for boundary in boundaries.iter() {
                    new_pos = boundary.clamp_particle_position(new_pos, grid_res);
                }

                *ctx.v = v_store;
                *ctx.velocity_gradient = b * KERNEL_D_INVERSE * apic_blend;
                *ctx.x = new_pos;
            }

            // Plasticity update + boundary post-hooks, now inline in the same
            // parallel task (used to be a forced-sequential second pass -- see
            // this function's own top doc). Runs unconditionally, even for a
            // pinned particle above: its kinematic x/v/velocity_gradient are
            // frozen, but its stress/plastic state must keep evolving normally
            // (matches the pinned branch's own "acts as a real anchor" comment).
            materials.update_particle(material_id, &mut ctx, dt);
            for boundary in boundaries.iter() {
                boundary.post_g2p_particle(&mut ctx, grid_res, dt);
            }

            0
        })
        .sum();

    clamp_count
}

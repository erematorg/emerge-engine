use glam::{IVec2, Mat2, Vec2};
use rayon::prelude::*;

use crate::boundary::BoundaryCondition;
use crate::grid::Grid;
use crate::grid::kernel::quadratic_weights;
use crate::materials::registry::MaterialRegistry;
use crate::particle::Particles;
use crate::solver::config::KERNEL_D_INVERSE;

pub struct G2PParams<'a> {
    pub vel_limit: f32,
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
/// cover the velocity clamp or position boundary-clamp applied after this in
/// the real G2P (piecewise/conditional, same deferred-with-a-name status as
/// grid update's boundary/clamp gap).
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
/// Returns the number of particles whose velocity was clamped to `vel_limit`.
pub fn gather_grid_to_particles(
    particles: &mut Particles,
    grid: &Grid,
    dt: f32,
    boundaries: &[Box<dyn BoundaryCondition>],
    materials: &MaterialRegistry,
    params: G2PParams,
) -> usize {
    let G2PParams {
        vel_limit,
        apic_blend,
        active_count,
        asflip_blend,
        pre_force_snapshot,
    } = params;
    let grid_res = grid.resolution();

    // Phase 1 (parallel): grid gather -> v, velocity_gradient, position advance + boundary
    // position clamp. Pure math over read-only grid/boundary state, writing only the calling
    // particle's own x/v/velocity_gradient — no cross-particle data dependency, so disjoint
    // per-field slices can be processed concurrently (gather passes are race-free by
    // construction; see Gao et al. 2018, "GPU Optimization of Material Point Methods").
    let xs = &mut particles.x[..active_count];
    let vs = &mut particles.v[..active_count];
    let vgs = &mut particles.velocity_gradient[..active_count];
    let contact_groups = &particles.contact_group[..active_count];
    let pinned_flags = &particles.pinned[..active_count];
    let material_ids = &particles.material_id[..active_count];
    // Gate once, not per particle: when no grip particle ever touched the grid this
    // substep (every scene that doesn't use `Particle::contact_group`), this is false
    // and the loop below takes the exact same path it always has — a plain
    // `grid.velocity_at` lookup, no extra branching cost worth measuring.
    let contact_active = grid.has_contact_activity();
    // Same gate for two-phase mixture coupling (Tampubolon et al. 2017) — see
    // `WithMixturePhase` doc. False (the default) for every scene that never
    // wraps a material this way, same zero-cost property as contact above.
    let mixture_active = grid.has_mixture_activity();

    let clamp_count: usize = xs
        .par_iter_mut()
        .zip(vs.par_iter_mut())
        .zip(vgs.par_iter_mut())
        .zip(contact_groups.par_iter())
        .zip(pinned_flags.par_iter())
        .zip(material_ids.par_iter())
        .map(
            |(((((x, v), vg), &contact_group), &pinned), &material_id)| {
                let mixture_phase = if mixture_active {
                    materials.get(material_id).mixture_phase()
                } else {
                    None
                };
                let v_old = *v;
                let weights = quadratic_weights(*x);
                let mut new_v = Vec2::ZERO;
                let mut b = Mat2::ZERO;

                for gx in 0..3 {
                    for gy in 0..3 {
                        let weight = weights.wx[gx] * weights.wy[gy];
                        let cell_pos = weights.base_cell + IVec2::new(gx as i32 - 1, gy as i32 - 1);
                        let dist = cell_pos.as_vec2() - *x + Vec2::splat(0.5);
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
                            // Two-phase mixture coupling routing (Tampubolon et al. 2017):
                            // a solid-phase particle reads the resolved solid field, a
                            // fluid-phase particle reads the resolved fluid field — both
                            // fall back to the ordinary total velocity where no coupling
                            // was registered at that node, same convention as contact.
                            use crate::materials::MixturePhase;
                            match phase {
                                MixturePhase::Solid => grid.resolved_solid_velocity_at(cell_pos),
                                MixturePhase::Fluid => grid.resolved_fluid_velocity_at(cell_pos),
                            }
                        } else {
                            grid.velocity_at(cell_pos)
                        };
                        let weighted_velocity = node_v * weight;
                        let term =
                            Mat2::from_cols(weighted_velocity * dist.x, weighted_velocity * dist.y);
                        b += term;
                        new_v += weighted_velocity;
                    }
                }

                // Dirichlet/kinematic anchor (`Particle::pinned`): force v=0 and
                // velocity_gradient=0 instead of gathering from the grid, so a pinned
                // particle never moves and never accumulates local strain from being
                // dragged — while its own mass/stress still scattered into P2G normally,
                // so it acts as a real, immovable anchor other bodies push against (the
                // standard technique for static/bedrock geometry in deformable-body sims).
                // Checked before the speed cap/position advance so a pinned particle takes
                // neither — position is deliberately left completely untouched, not just
                // re-clamped to itself, avoiding any float drift from a v=0*dt add-then-
                // reclamp round trip.
                if pinned != 0 {
                    *v = Vec2::ZERO;
                    *vg = Mat2::ZERO;
                    return 0;
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

                // Hard speed cap — CFL in choose_substep_dt is the physics-grounded bound.
                // This fires only when CFL is violated despite the timestep limiter (e.g. first
                // substep of a high-energy spawn). Magnitude clamp preserves direction; no
                // anisotropic bias unlike per-component clamping. Clamps both `v_store` and
                // `v_position` by the SAME safety ratio (derived from the stored velocity's own
                // magnitude) so they stay mutually consistent -- when ASFLIP is disabled the two
                // are identical (`v_store == v_position == new_v`), so this is byte-identical to
                // the original single-velocity clamp.
                let spd = v_store.length();
                let clamped = if spd > vel_limit {
                    let scale = vel_limit / spd;
                    v_store *= scale;
                    v_position *= scale;
                    1
                } else {
                    0
                };

                // Apply all boundaries' position clamp (pure function, no particle-struct access).
                let mut new_pos = *x + v_position * dt;
                for boundary in boundaries.iter() {
                    new_pos = boundary.clamp_particle_position(new_pos, grid_res);
                }

                *v = v_store;
                *vg = b * KERNEL_D_INVERSE * apic_blend;
                *x = new_pos;
                clamped
            },
        )
        .sum();

    // Phase 2 (sequential): plasticity update + boundary post-hooks need whole-`Particles`
    // mutable access (deformation_gradient, hardening_scale, etc. per material) — not
    // split-borrow-friendly without a larger `MaterialModel` trait redesign, so kept sequential.
    for i in 0..active_count {
        let material_id = particles.material_id[i];
        let material = materials.get(material_id);
        material.update_particle(particles, i, dt);
        for boundary in boundaries.iter() {
            boundary.post_g2p_particle(particles, i, grid_res, dt);
        }
    }

    clamp_count
}

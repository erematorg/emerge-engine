use glam::{IVec2, Mat2, Vec2};
use std::collections::BTreeMap;

use crate::grid::Grid;
use crate::grid::kernel::quadratic_weights;
use crate::transfer::{f_update_vjp, g2p_affine_vjp, g2p_velocity_vjp, p2g_stress_vjp};

use super::body_plan::BodyPlan;
use super::config::{DiffConfig, DiffState, FeedbackController, SinusoidController, StepRecord};
use super::forward::{rollout, rollout_feedback};
use super::stress::{StressEval, signed_active_stress_vjp};

// ── Backward ──────────────────────────────────────────────────────────────────

/// Gradients flowing INTO one substep from everything after it.
struct IncomingGrad<'a> {
    x: &'a [Vec2],
    v: &'a [Vec2],
    c: &'a [Mat2],
    f: &'a [Mat2],
}

/// Gradients flowing OUT of one substep to the step before it, plus this
/// substep's own per-group activation gradient.
struct OutgoingGrad {
    x: Vec<Vec2>,
    v: Vec<Vec2>,
    c: Vec<Mat2>,
    f: Vec<Mat2>,
    act: Vec<f32>,
}

/// Everything the backward pass needs about ONE recorded forward substep.
struct SubstepCtx<'a> {
    state: &'a DiffState,
    next_state: &'a DiffState,
    record: &'a StepRecord,
    acts: &'a [f32],
}

/// Backward through one substep, built from the individually FD-verified
/// adjoints in `spacetime::transfer`/`grid` plus this module's own
/// FD-verified glue:
/// - position update `x' = x + v'*dt`: `g_v' += g_x'*dt` and (identity term)
///   `g_x += g_x'` -- the chain the constant-activation prototype could skip
///   (its velocity transients died against elastic restoring forces) but a
///   time-varying gait cannot;
/// - gravity: additive constant, gradient passes through unchanged;
/// - sticky floor: cells recorded stuck forward pass NO gradient back through
///   their velocity (their output was the constant zero).
fn backward_substep(
    ctx: &SubstepCtx,
    plan: &BodyPlan,
    eval: &mut StressEval,
    cfg: &DiffConfig,
    incoming: &IncomingGrad,
) -> OutgoingGrad {
    let SubstepCtx {
        state,
        next_state,
        record,
        acts,
    } = *ctx;
    let n = state.x.len();

    // Position/F/C bookkeeping at the output side.
    let mut g_v_total = vec![Vec2::ZERO; n];
    let mut g_c_total = vec![Mat2::ZERO; n];
    let mut g_f_old_running = vec![Mat2::ZERO; n];
    for i in 0..n {
        g_v_total[i] = incoming.v[i] + incoming.x[i] * cfg.dt;
        let (g_c_from_f, g_f_old_a) =
            f_update_vjp(next_state.c[i], state.f[i], cfg.dt, incoming.f[i]);
        g_c_total[i] = incoming.c[i] + g_c_from_f;
        g_f_old_running[i] = g_f_old_a;
    }

    // G2P transpose: per-cell velocity gradient (post-floor).
    let mut g_vel_post: BTreeMap<(i32, i32), Vec2> = BTreeMap::new();
    for (i, &x) in state.x.iter().enumerate() {
        let g_from_c = g2p_affine_vjp(x, cfg.kernel_d_inverse, cfg.apic_blend, g_c_total[i]);
        let g_from_v = g2p_velocity_vjp(x, g_v_total[i]);
        let w = quadratic_weights(x);
        for gx in 0..3 {
            for gy in 0..3 {
                let cell_pos = w.base_cell + IVec2::new(gx as i32 - 1, gy as i32 - 1);
                *g_vel_post
                    .entry((cell_pos.x, cell_pos.y))
                    .or_insert(Vec2::ZERO) += g_from_c[gx][gy] + g_from_v[gx][gy];
            }
        }
    }

    // Grid update backward: sticky floor kills gradient; gravity is a
    // constant shift (pass-through); then velocity = momentum/mass.
    let mut mass_map: BTreeMap<(i32, i32), f32> = BTreeMap::new();
    for &x in state.x.iter() {
        let w = quadratic_weights(x);
        for gx in 0..3 {
            for gy in 0..3 {
                let weight = w.wx[gx] * w.wy[gy];
                let cell_pos = w.base_cell + IVec2::new(gx as i32 - 1, gy as i32 - 1);
                *mass_map.entry((cell_pos.x, cell_pos.y)).or_insert(0.0) += weight * cfg.mass;
            }
        }
    }
    let mut g_momentum_map: BTreeMap<(i32, i32), Vec2> = BTreeMap::new();
    for (&cell, &g_v_cell) in g_vel_post.iter() {
        if record.stuck.contains(&cell) {
            continue; // forward output was the constant zero
        }
        let mass_c = mass_map[&cell];
        if mass_c <= 0.0 {
            continue; // never produced a real velocity forward
        }
        let (g_m, _g_mass) = Grid::update_velocities_vjp(Vec2::ZERO, mass_c, g_v_cell);
        g_momentum_map.insert(cell, g_m);
    }

    // P2G backward per particle.
    let mut out = OutgoingGrad {
        x: incoming.x.to_vec(), // identity term of x' = x + v'*dt
        v: vec![Vec2::ZERO; n],
        c: vec![Mat2::ZERO; n],
        f: vec![Mat2::ZERO; n],
        act: vec![0.0; plan.n_groups],
    };
    for (i, &x) in state.x.iter().enumerate() {
        let w = quadratic_weights(x);
        let mut g_momentum_local = [[Vec2::ZERO; 3]; 3];
        for (gx, row) in g_momentum_local.iter_mut().enumerate() {
            for (gy, cell) in row.iter_mut().enumerate() {
                let cell_pos = w.base_cell + IVec2::new(gx as i32 - 1, gy as i32 - 1);
                *cell = *g_momentum_map
                    .get(&(cell_pos.x, cell_pos.y))
                    .unwrap_or(&Vec2::ZERO);
            }
        }

        let g_stress = p2g_stress_vjp(x, cfg.stress_coeff, &g_momentum_local);
        let g_c_from_p2g = p2g_stress_vjp(x, cfg.mass, &g_momentum_local);

        let mut g_v_accum = Vec2::ZERO;
        for (gx, wx) in w.wx.iter().enumerate() {
            for (gy, wy) in w.wy.iter().enumerate() {
                g_v_accum += (wx * wy) * cfg.mass * g_momentum_local[gx][gy];
            }
        }

        let g_f_passive = eval.passive_vjp(state.f[i], g_stress);
        let act = plan.group[i].map_or(0.0, |g| acts[g]);
        let fiber = plan.group[i].map_or(Vec2::Y, |g| plan.fiber_dir[g]);
        let (g_f_active, g_act) =
            signed_active_stress_vjp(state.f[i], act, cfg.act_strength, fiber, g_stress);

        out.v[i] = g_v_accum;
        out.c[i] = g_c_from_p2g;
        out.f[i] = g_f_old_running[i] + g_f_passive + g_f_active;
        if let Some(g) = plan.group[i] {
            out.act[g] += g_act;
        }
    }

    out
}

/// Gradient of the locomotion loss `L = -(mean_x(final) - mean_x(rest))`
/// w.r.t. the controller's weights and bias, via full backprop through time.
pub fn controller_gradient(
    plan: &BodyPlan,
    controller: &SinusoidController,
    eval: &mut StressEval,
    cfg: &DiffConfig,
    steps: usize,
) -> (Vec<f32>, Vec<f32>) {
    let n = plan.positions.len();
    // Average the drift over the last `loss_window` states instead of
    // reading only the final one: a final-state-only loss rewards ending
    // far right by ANY means, and gradient descent found the exploit --
    // ballistic end-of-rollout hops ("goes flying, not walking", observed
    // live 2026-07-11). Averaging over a window rewards being consistently
    // far right through sustained ground contact instead. dL/dx for each
    // windowed state is -1/(n*K); the position-identity chain in
    // `backward_substep` accumulates them correctly across steps.
    let window = cfg.loss_window.clamp(1, steps);
    let per_state = -1.0 / (n as f32 * window as f32);
    // Bounce penalty: mean squared vertical velocity over the WHOLE rollout
    // (not just the drift window) -- bouncing anywhere should be
    // discouraged, not only near the end. dL/dv.y = 2*lambda*v.y/(n*steps).
    // See `DiffConfig::bounce_penalty` for why this exists (a window alone
    // doesn't stop gradient descent from choosing a hop over a walk).
    let bounce_coeff = 2.0 * cfg.bounce_penalty / (n as f32 * steps as f32);
    backprop_through_time(
        plan,
        controller,
        eval,
        cfg,
        steps,
        &mut |t, next_state, seed| {
            if t >= steps - window {
                for g in seed.x.iter_mut() {
                    g.x += per_state;
                }
            }
            if bounce_coeff != 0.0 {
                for (g, s) in seed.v.iter_mut().zip(next_state.v.iter()) {
                    g.y += bounce_coeff * s.y;
                }
            }
        },
    )
}

/// Same backprop through time, but for an arbitrary loss seed: `seed_g_x[i]`
/// = dL/d(final position of particle i). `controller_gradient` is the
/// centroid-drift special case. Exposed separately because losses that
/// aren't pure-centroid are the only way to finite-difference-verify the
/// chain in a contact-free regime -- momentum conservation makes the
/// centroid EXACTLY invariant there (the analytic gradient correctly
/// reports zero), leaving nothing to measure.
pub fn controller_gradient_seeded(
    plan: &BodyPlan,
    controller: &SinusoidController,
    eval: &mut StressEval,
    cfg: &DiffConfig,
    steps: usize,
    seed_g_x: &[Vec2],
) -> (Vec<f32>, Vec<f32>) {
    backprop_through_time(
        plan,
        controller,
        eval,
        cfg,
        steps,
        &mut |t, _next_state, seed| {
            if t == steps - 1 {
                for (g, s) in seed.x.iter_mut().zip(seed_g_x.iter()) {
                    *g += *s;
                }
            }
        },
    )
}

/// The two running gradients a loss-seeding closure can add into.
pub(super) struct GradSeed<'a> {
    pub(super) x: &'a mut [Vec2],
    pub(super) v: &'a mut [Vec2],
}

/// Core backprop-through-time loop. `inject_seed(t, next_state, seed)`
/// is called at each substep (in reverse order) BEFORE that substep's
/// backward pass, and adds dL/d(state-after-substep-t's positions/
/// velocities) into the running gradients -- this is how a loss that reads
/// MULTIPLE states along the rollout (an averaged-drift loss, a bounce
/// penalty) seeds its gradient, not just a final-state loss. `next_state` is
/// that substep's own forward result, for losses that need to read it (e.g.
/// the bounce penalty reads `next_state.v`).
pub(super) fn backprop_through_time(
    plan: &BodyPlan,
    controller: &SinusoidController,
    eval: &mut StressEval,
    cfg: &DiffConfig,
    steps: usize,
    inject_seed: &mut dyn FnMut(usize, &DiffState, &mut GradSeed),
) -> (Vec<f32>, Vec<f32>) {
    let (history, acts_cache) = rollout(plan, controller, eval, cfg, steps);
    let n = plan.positions.len();
    let rest = DiffState::rest(plan);

    let mut g_x = vec![Vec2::ZERO; n];
    let mut g_v = vec![Vec2::ZERO; n];
    let mut g_c = vec![Mat2::ZERO; n];
    let mut g_f = vec![Mat2::ZERO; n];

    let mut g_weights = vec![0.0f32; controller.weights.len()];
    let mut g_bias = vec![0.0f32; controller.bias.len()];

    for t in (0..steps).rev() {
        inject_seed(
            t,
            &history[t].0,
            &mut GradSeed {
                x: &mut g_x,
                v: &mut g_v,
            },
        );
        let state = if t == 0 { &rest } else { &history[t - 1].0 };
        let (next_state, record) = &history[t];
        let incoming = IncomingGrad {
            x: &g_x,
            v: &g_v,
            c: &g_c,
            f: &g_f,
        };
        let ctx = SubstepCtx {
            state,
            next_state,
            record,
            acts: &acts_cache[t],
        };
        let out = backward_substep(&ctx, plan, eval, cfg, &incoming);

        // Chain each group's activation gradient into the controller
        // parameters: act = tanh(pre), d(act)/d(pre) = 1 - tanh(pre)^2;
        // pre is linear in weights (sin basis values) and bias. The
        // control-effort penalty (mean act^2 over all groups/substeps) adds
        // DIRECTLY to the gradient w.r.t. `act` itself (same variable the
        // physics gradient `out.act` already targets), before the shared
        // tanh-derivative chain -- not a separate path.
        let time = t as f32 * cfg.dt;
        let effort_coeff =
            2.0 * cfg.control_effort_penalty / (controller.n_groups as f32 * steps as f32);
        for (g, &g_act_physics) in out.act.iter().enumerate() {
            let act = acts_cache[t][g];
            let g_act = g_act_physics + effort_coeff * act;
            let d_pre = g_act * (1.0 - act * act);
            // Mirrored groups reuse another group's weights/bias (see
            // `SinusoidController::mirror_of`'s doc), so their gradient
            // must land on the SAME underlying parameters, at the SAME
            // phase offset the forward pass actually used -- a shared
            // parameter's total gradient is the sum of every use's
            // contribution (standard multivariable chain rule), which
            // summing into `g_weights[src]`/`g_bias[src]` across every
            // group `g` whose `mirror_of` resolves to `src` achieves
            // automatically.
            let src = controller.mirror_of[g].unwrap_or(g);
            let extra_phase = controller.phase_offset[g];
            for j in 0..controller.n_waves {
                let phase = 2.0 * std::f32::consts::PI * j as f32 / controller.n_waves as f32;
                g_weights[src * controller.n_waves + j] +=
                    d_pre * (cfg.omega * time + phase + extra_phase).sin();
            }
            g_bias[src] += d_pre;
        }

        g_x = out.x;
        g_v = out.v;
        g_c = out.c;
        g_f = out.f;
    }

    (g_weights, g_bias)
}

/// Gradient of the same locomotion loss as `controller_gradient`, but for a
/// `FeedbackController`. Same windowed-drift + bounce-penalty objective;
/// the real difference is the activation gradient chains through
/// `FeedbackController::backward` (feature-extraction + linear + tanh)
/// instead of the sinusoid's tanh+weights, and that backward ALSO returns
/// position/velocity gradient contributions (the controller READ this
/// substep's state to decide its own activation) that must be ADDED onto
/// `out.x`/`out.v` -- a real, new path the open-loop controller never had
/// (its activation depended only on `t`, never on the body's own state).
pub fn feedback_controller_gradient(
    plan: &BodyPlan,
    controller: &FeedbackController,
    eval: &mut StressEval,
    cfg: &DiffConfig,
    steps: usize,
) -> (Vec<f32>, Vec<f32>) {
    let n = plan.positions.len();
    let window = cfg.loss_window.clamp(1, steps);
    let per_state = -1.0 / (n as f32 * window as f32);
    let bounce_coeff = 2.0 * cfg.bounce_penalty / (n as f32 * steps as f32);
    feedback_backprop_through_time(
        plan,
        controller,
        eval,
        cfg,
        steps,
        &mut |t, next_state, seed| {
            if t >= steps - window {
                for g in seed.x.iter_mut() {
                    g.x += per_state;
                }
            }
            if bounce_coeff != 0.0 {
                for (g, s) in seed.v.iter_mut().zip(next_state.v.iter()) {
                    g.y += bounce_coeff * s.y;
                }
            }
        },
    )
}

/// `FeedbackController` analogue of `controller_gradient_seeded` -- same
/// role (arbitrary final-state loss seed, for FD verification in a
/// contact-free regime where the centroid loss's true gradient is exactly
/// zero by momentum conservation).
pub fn feedback_controller_gradient_seeded(
    plan: &BodyPlan,
    controller: &FeedbackController,
    eval: &mut StressEval,
    cfg: &DiffConfig,
    steps: usize,
    seed_g_x: &[Vec2],
) -> (Vec<f32>, Vec<f32>) {
    feedback_backprop_through_time(
        plan,
        controller,
        eval,
        cfg,
        steps,
        &mut |t, _next_state, seed| {
            if t == steps - 1 {
                for (g, s) in seed.x.iter_mut().zip(seed_g_x.iter()) {
                    *g += *s;
                }
            }
        },
    )
}

/// `FeedbackController` analogue of `backprop_through_time`. Structurally
/// identical to the sinusoid version (rollout, then reverse-order
/// `backward_substep` calls threading g_x/g_v/g_c/g_f between steps) --
/// the one real difference is chaining `out.act` through
/// `FeedbackController::backward` (recomputing that step's `(feat, count)`
/// from `state`, the same cheap-recompute-in-backward pattern used
/// throughout this module for kernel weights) instead of a fixed tanh+sin
/// formula, and adding that backward's own `(g_x, g_v)` outputs onto
/// `out.x`/`out.v` before they become the next iteration's incoming
/// gradients.
fn feedback_backprop_through_time(
    plan: &BodyPlan,
    controller: &FeedbackController,
    eval: &mut StressEval,
    cfg: &DiffConfig,
    steps: usize,
    inject_seed: &mut dyn FnMut(usize, &DiffState, &mut GradSeed),
) -> (Vec<f32>, Vec<f32>) {
    let (history, acts_cache) = rollout_feedback(plan, controller, eval, cfg, steps);
    let n = plan.positions.len();
    let rest = DiffState::rest(plan);

    let mut g_x = vec![Vec2::ZERO; n];
    let mut g_v = vec![Vec2::ZERO; n];
    let mut g_c = vec![Mat2::ZERO; n];
    let mut g_f = vec![Mat2::ZERO; n];
    let mut g_weights = vec![0.0f32; controller.weights.len()];
    let mut g_bias = vec![0.0f32; controller.bias.len()];

    for t in (0..steps).rev() {
        inject_seed(
            t,
            &history[t].0,
            &mut GradSeed {
                x: &mut g_x,
                v: &mut g_v,
            },
        );
        let state = if t == 0 { &rest } else { &history[t - 1].0 };
        let (next_state, record) = &history[t];
        let incoming = IncomingGrad {
            x: &g_x,
            v: &g_v,
            c: &g_c,
            f: &g_f,
        };
        let ctx = SubstepCtx {
            state,
            next_state,
            record,
            acts: &acts_cache[t],
        };
        let out = backward_substep(&ctx, plan, eval, cfg, &incoming);

        let effort_coeff =
            2.0 * cfg.control_effort_penalty / (controller.n_groups as f32 * steps as f32);
        let g_act: Vec<f32> = out
            .act
            .iter()
            .enumerate()
            .map(|(g, &g_act_physics)| g_act_physics + effort_coeff * acts_cache[t][g])
            .collect();

        let (feat, count) = FeedbackController::features(plan, state);
        let (g_w_step, g_b_step, g_x_ctrl, g_v_ctrl) =
            controller.backward(plan, &feat, &count, &g_act);
        for (gw, s) in g_weights.iter_mut().zip(g_w_step.iter()) {
            *gw += s;
        }
        for (gb, s) in g_bias.iter_mut().zip(g_b_step.iter()) {
            *gb += s;
        }

        g_x = out.x;
        g_v = out.v;
        for (gx, extra) in g_x.iter_mut().zip(g_x_ctrl.iter()) {
            *gx += *extra;
        }
        for (gv, extra) in g_v.iter_mut().zip(g_v_ctrl.iter()) {
            *gv += *extra;
        }
        g_c = out.c;
        g_f = out.f;
    }

    (g_weights, g_bias)
}

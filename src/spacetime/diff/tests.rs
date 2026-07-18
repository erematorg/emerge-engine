//! Test suite for `diff.rs` (differentiable MLS-MPM forward+reverse stepping
//! for offline gait-controller training) -- split out into its own file (was
//! ~900 of the parent file's ~2530 lines), same pattern as
//! `gpu/solver/device_lost_tests.rs`. Pure mechanical line-range extraction
//! (dedented, not retyped) to eliminate transcription risk in this project's
//! single most gradient-sensitive code -- every VJP here is verified against
//! central-difference numerical gradients, exactly where a silent copy error
//! would be hardest to notice.

use super::*;

fn setup() -> (BodyPlan, DiffConfig, StressEval) {
    let plan = BodyPlan::walker(Vec2::new(20.0, 1.3), 0.5);
    let cfg = DiffConfig::default();
    let eval = StressEval::new(NeoHookeanMaterial::new(900.0, 700.0));
    (plan, cfg, eval)
}

/// Same walker, but placed near the origin so absolute positions (and
/// with them the f32 quantization floor on an absolute-position loss)
/// stay small -- FD verification only.
fn setup_origin() -> (BodyPlan, DiffConfig, StressEval) {
    let plan = BodyPlan::walker(Vec2::new(1.5, 1.3), 0.5);
    let cfg = DiffConfig::default();
    let eval = StressEval::new(NeoHookeanMaterial::new(900.0, 700.0));
    (plan, cfg, eval)
}

#[test]
fn signed_active_stress_vjp_matches_finite_difference() {
    let f = Mat2::from_cols(Vec2::new(1.1, 0.08), Vec2::new(-0.06, 0.92));
    let strength = 12.0;
    let fiber = Vec2::Y;
    let g = Mat2::from_cols(Vec2::new(0.4, -0.7), Vec2::new(0.6, 0.3));
    let h = 1.0e-3_f32;

    let loss = |f: Mat2, act: f32| -> f32 {
        let tau = signed_active_stress(f, act, strength, fiber);
        g.x_axis.x * tau.x_axis.x
            + g.x_axis.y * tau.x_axis.y
            + g.y_axis.x * tau.y_axis.x
            + g.y_axis.y * tau.y_axis.y
    };

    // The signed case's whole point: check at a NEGATIVE activation,
    // where the engine variant's guard would kill the gradient.
    for act in [-0.6f32, -0.05, 0.3] {
        let (analytic_f, analytic_act) = signed_active_stress_vjp(f, act, strength, fiber, g);

        let numeric_act = (loss(f, act + h) - loss(f, act - h)) / (2.0 * h);
        let diff = (numeric_act - analytic_act).abs();
        let scale = numeric_act.abs().max(analytic_act.abs()).max(1.0);
        assert!(
            diff / scale < 1.0e-2,
            "signed act-gradient mismatch at act={act}: analytic={analytic_act:.6} \
             numeric={numeric_act:.6}"
        );

        let mut f_plus = f;
        f_plus.x_axis.y += h;
        let mut f_minus = f;
        f_minus.x_axis.y -= h;
        let numeric_f = (loss(f_plus, act) - loss(f_minus, act)) / (2.0 * h);
        let diff = (numeric_f - analytic_f.x_axis.y).abs();
        let scale = numeric_f.abs().max(analytic_f.x_axis.y.abs()).max(1.0);
        assert!(
            diff / scale < 1.0e-2,
            "signed F-gradient mismatch at act={act}: analytic={:.6} numeric={numeric_f:.6}",
            analytic_f.x_axis.y
        );
    }
}

/// FD check of the full smooth pipeline -- sinusoid controller, tanh
/// chain, signed active stress, position-identity chain, multi-particle
/// shared grid, backprop through time -- with the CONTACT KINK removed
/// (no gravity, floor placed out of reach). The sticky floor is a
/// genuine, hard non-differentiability: a finite-difference perturbation
/// flips stick/unstick branch decisions mid-rollout, so numeric and
/// analytic (detached-branch subgradient) values legitimately diverge
/// there -- the standard, documented contact-gradient limitation every
/// differentiable contact simulator shares, DiffTaichi's canonical
/// walker included (its floor produces the same kink; its training works
/// anyway). The contact regime is therefore verified separately by
/// `contact_gradient_is_a_descent_direction` below with the property
/// training actually needs, while THIS test pins every smooth piece of
/// the chain against exact finite differences.
///
/// Uses a per-particle loss (two particles, mixed x/y weights), NOT the
/// centroid loss: with no external contact, momentum conservation makes
/// the centroid exactly invariant, so its true gradient is zero and an
/// FD check of it measures nothing (a fact this module's own
/// `internal_stress_conserves_centroid` pins down separately).
#[test]
fn controller_gradient_matches_finite_difference_smooth_regime() {
    let (plan, mut cfg, mut eval) = setup_origin();
    cfg.gravity = 0.0;
    cfg.floor_y = -100.0; // out of reach: no contact anywhere in the rollout
    // Gentler actuation than the training default, and a body placed
    // near the origin (same trick as DiffTaichi's unit-box domain): the
    // loss reads absolute particle positions, so f32 quantization sets a
    // noise floor of one ULP of |x| per step -- at x~20 that floor
    // swamps small-amplitude gradients, at x~2 it is 16x finer. The
    // remaining systematic disagreement is the chain's one deliberate
    // scope exclusion (kernel-weight position dependence, see module
    // docs): measured at ~5-7% at full training strength, shrinking
    // with actuation amplitude exactly as an excluded motion-
    // proportional term should.
    cfg.act_strength = 4.0;
    let steps = 40;
    let mut controller = SinusoidController::seeded(plan.n_groups, cfg.n_waves);
    // Non-trivial starting point so gradients aren't at a symmetric zero.
    controller.bias[1] = 0.15;
    controller.weights[2] = 0.2;

    let n = plan.positions.len();
    // Loss L = x_final[3].x - 0.5 * x_final[40].y : deliberately
    // asymmetric, touches both components, sensitive to internal motion.
    let mut seed = vec![Vec2::ZERO; n];
    seed[3] = Vec2::new(1.0, 0.0);
    seed[40] = Vec2::new(0.0, -0.5);

    let (g_w, g_b) = controller_gradient_seeded(&plan, &controller, &mut eval, &cfg, steps, &seed);

    let loss_of = |c: &SinusoidController, eval: &mut StressEval| -> f32 {
        let (history, _) = rollout(&plan, c, eval, &cfg, steps);
        let fin = &history[steps - 1].0;
        fin.x[3].x - 0.5 * fin.x[40].y
    };

    // h must be large enough that the loss change (~gradient * h) clears
    // f32 quantization: the loss lives at |x| ~ 20 grid units where one
    // ULP is ~2e-6, and gradients here are ~1e-3, so h = 2e-3 would ask
    // the FD numerator to resolve ~3 ULPs -- it reads exactly zero. At
    // h = 5e-2 the signal is ~75 ULPs while tanh/sin curvature error
    // (O(h^2)) stays well inside the tolerance.
    let h = 5.0e-2_f32;
    // A few representative weights across groups/waves, plus one bias.
    for &wi in &[0usize, 2, 5, 9, 14] {
        let mut c_plus = controller.clone();
        c_plus.weights[wi] += h;
        let mut c_minus = controller.clone();
        c_minus.weights[wi] -= h;
        let numeric = (loss_of(&c_plus, &mut eval) - loss_of(&c_minus, &mut eval)) / (2.0 * h);
        let diff = (numeric - g_w[wi]).abs();
        let scale = numeric.abs().max(g_w[wi].abs()).max(1.0e-3);
        assert!(
            diff / scale < 5.0e-2,
            "controller weight[{wi}] gradient mismatch (smooth regime): analytic={:.6} \
             numeric(central-diff)={numeric:.6} relative_diff={:.2e}",
            g_w[wi],
            diff / scale
        );
    }
    {
        let mut c_plus = controller.clone();
        c_plus.bias[1] += h;
        let mut c_minus = controller.clone();
        c_minus.bias[1] -= h;
        let numeric = (loss_of(&c_plus, &mut eval) - loss_of(&c_minus, &mut eval)) / (2.0 * h);
        let diff = (numeric - g_b[1]).abs();
        let scale = numeric.abs().max(g_b[1].abs()).max(1.0e-3);
        assert!(
            diff / scale < 5.0e-2,
            "controller bias[1] gradient mismatch (smooth regime): analytic={:.6} \
             numeric={numeric:.6} relative_diff={:.2e}",
            g_b[1],
            diff / scale
        );
    }
}

/// The real end-to-end gate for the whole `FeedbackController` chain:
/// `feedback_controller_backward_matches_finite_difference` above only
/// checks the controller in isolation (no physics); this backprops
/// through real P2G/grid-update/G2P/F-update substeps AND the
/// controller's own state-read backward, together, in the smooth
/// (contact-free) regime -- same rationale as the sinusoid controller's
/// own smooth-regime test (non-centroid loss, origin-placed body, gentle
/// actuation for the O(h^2) FD error margin).
#[test]
fn feedback_controller_gradient_matches_finite_difference_smooth_regime() {
    let plan = BodyPlan::biped(Vec2::new(1.5, 1.3), 0.5);
    let cfg = DiffConfig {
        gravity: 0.0,
        floor_y: -100.0,
        act_strength: 4.0,
        ..DiffConfig::default()
    };
    let mut eval = StressEval::new(NeoHookeanMaterial::new(900.0, 700.0));
    let steps = 40;
    let mut controller = FeedbackController::seeded_with(plan.n_groups, 7);
    controller.bias[1] = 0.15;
    controller.weights[3] = 0.2;

    let n = plan.positions.len();
    let mut seed = vec![Vec2::ZERO; n];
    seed[2] = Vec2::new(1.0, 0.0);
    seed[n - 3] = Vec2::new(0.0, -0.5);

    let (g_w, g_b) =
        feedback_controller_gradient_seeded(&plan, &controller, &mut eval, &cfg, steps, &seed);

    let loss_of = |c: &FeedbackController, eval: &mut StressEval| -> f32 {
        let (history, _) = rollout_feedback(&plan, c, eval, &cfg, steps);
        let fin = &history[steps - 1].0;
        fin.x[2].x - 0.5 * fin.x[n - 3].y
    };

    let h = 5.0e-2_f32;
    for &wi in &[0usize, 3, 9, 20] {
        let mut c_plus = controller.clone();
        c_plus.weights[wi] += h;
        let mut c_minus = controller.clone();
        c_minus.weights[wi] -= h;
        let numeric = (loss_of(&c_plus, &mut eval) - loss_of(&c_minus, &mut eval)) / (2.0 * h);
        let diff = (numeric - g_w[wi]).abs();
        let scale = numeric.abs().max(g_w[wi].abs()).max(1.0e-3);
        assert!(
            diff / scale < 5.0e-2,
            "feedback controller weight[{wi}] gradient mismatch (end-to-end): analytic={:.6} \
             numeric(central-diff)={numeric:.6} relative_diff={:.2e}",
            g_w[wi],
            diff / scale
        );
    }
    {
        let mut c_plus = controller.clone();
        c_plus.bias[1] += h;
        let mut c_minus = controller.clone();
        c_minus.bias[1] -= h;
        let numeric = (loss_of(&c_plus, &mut eval) - loss_of(&c_minus, &mut eval)) / (2.0 * h);
        let diff = (numeric - g_b[1]).abs();
        let scale = numeric.abs().max(g_b[1].abs()).max(1.0e-3);
        assert!(
            diff / scale < 5.0e-2,
            "feedback controller bias[1] gradient mismatch (end-to-end): analytic={:.6} \
             numeric={numeric:.6} relative_diff={:.2e}",
            g_b[1],
            diff / scale
        );
    }
}

/// Every FD test above uses `BodyPlan::walker`, whose 4 groups all share
/// the SAME fiber direction (`Vec2::Y`) -- moving `fiber_dir` from a
/// single `DiffConfig` value onto per-group `BodyPlan` storage is a real
/// behavior change (mixed directions ACROSS a body, the whole point of
/// `BodyPlan::biped`'s thigh/foot split) that none of them exercise.
/// Same smooth-regime + origin-placement rationale, `BodyPlan::biped`
/// instead of `walker`.
#[test]
fn mixed_fiber_directions_gradient_matches_finite_difference() {
    let plan = BodyPlan::biped(Vec2::new(1.5, 1.3), 0.5);
    let cfg = DiffConfig {
        gravity: 0.0,
        floor_y: -100.0,
        act_strength: 4.0,
        ..DiffConfig::default()
    };
    let mut eval = StressEval::new(NeoHookeanMaterial::new(900.0, 700.0));
    let steps = 40;
    let mut controller = SinusoidController::seeded(plan.n_groups, cfg.n_waves);
    controller.bias[1] = 0.15; // left foot (horizontal fiber)
    controller.weights[2 * cfg.n_waves] = 0.2; // right thigh (vertical fiber)

    let n = plan.positions.len();
    let mut seed = vec![Vec2::ZERO; n];
    seed[2] = Vec2::new(1.0, 0.0);
    seed[n - 3] = Vec2::new(0.0, -0.5);

    let (g_w, g_b) = controller_gradient_seeded(&plan, &controller, &mut eval, &cfg, steps, &seed);

    let loss_of = |c: &SinusoidController, eval: &mut StressEval| -> f32 {
        let (history, _) = rollout(&plan, c, eval, &cfg, steps);
        let fin = &history[steps - 1].0;
        fin.x[2].x - 0.5 * fin.x[n - 3].y
    };

    let h = 5.0e-2_f32;
    for &wi in &[
        0usize,
        cfg.n_waves + 1,
        2 * cfg.n_waves,
        3 * cfg.n_waves + 2,
    ] {
        let mut c_plus = controller.clone();
        c_plus.weights[wi] += h;
        let mut c_minus = controller.clone();
        c_minus.weights[wi] -= h;
        let numeric = (loss_of(&c_plus, &mut eval) - loss_of(&c_minus, &mut eval)) / (2.0 * h);
        let diff = (numeric - g_w[wi]).abs();
        let scale = numeric.abs().max(g_w[wi].abs()).max(1.0e-3);
        assert!(
            diff / scale < 5.0e-2,
            "mixed-fiber weight[{wi}] gradient mismatch: analytic={:.6} \
             numeric(central-diff)={numeric:.6} relative_diff={:.2e}",
            g_w[wi],
            diff / scale
        );
    }
    {
        let mut c_plus = controller.clone();
        c_plus.bias[1] += h;
        let mut c_minus = controller.clone();
        c_minus.bias[1] -= h;
        let numeric = (loss_of(&c_plus, &mut eval) - loss_of(&c_minus, &mut eval)) / (2.0 * h);
        let diff = (numeric - g_b[1]).abs();
        let scale = numeric.abs().max(g_b[1].abs()).max(1.0e-3);
        assert!(
            diff / scale < 5.0e-2,
            "mixed-fiber bias[1] gradient mismatch: analytic={:.6} numeric={numeric:.6} \
             relative_diff={:.2e}",
            g_b[1],
            diff / scale
        );
    }
}

/// Physical invariant, and the reason the FD test above needs a
/// non-centroid loss: with no gravity and no floor, muscle stress is
/// purely internal, so the body's mass centroid must not move no matter
/// how hard the controller fires -- Newton's third law flowing through
/// P2G/G2P intact. Also the physics fact that makes contact necessary
/// for locomotion at all.
#[test]
fn internal_stress_conserves_centroid() {
    let (plan, mut cfg, mut eval) = setup();
    cfg.gravity = 0.0;
    cfg.floor_y = -100.0;
    let mut controller = SinusoidController::seeded(plan.n_groups, cfg.n_waves);
    // Fire the muscles hard: conservation must hold regardless.
    controller.bias = vec![0.8, -0.6, 0.7, -0.5];
    for w in controller.weights.iter_mut() {
        *w += 0.3;
    }

    let d = drift(&plan, &controller, &mut eval, &cfg, 60);
    assert!(
        d.abs() < 1.0e-3,
        "internal muscle stress must not move the centroid (no contact, no gravity): \
         drift={d:.6}"
    );
}

/// Contact-regime verification: exact FD equality is unattainable across
/// the sticky floor's branch kinks (see the smooth-regime test's doc),
/// so verify the property gradient descent actually relies on instead --
/// the analytic gradient must be a real descent direction of the true
/// (kinked) loss: stepping against it must reduce the loss.
#[test]
fn contact_gradient_is_a_descent_direction() {
    let (plan, cfg, mut eval) = setup();
    let steps = 60;
    let mut controller = SinusoidController::seeded(plan.n_groups, cfg.n_waves);
    controller.bias[1] = 0.15;
    controller.weights[2] = 0.2;

    let loss_of = |c: &SinusoidController, eval: &mut StressEval| -> f32 {
        -drift(&plan, c, eval, &cfg, steps)
    };
    let loss_base = loss_of(&controller, &mut eval);

    let (g_w, g_b) = controller_gradient(&plan, &controller, &mut eval, &cfg, steps);
    let grad_norm_sq: f32 = g_w.iter().chain(g_b.iter()).map(|g| g * g).sum();
    assert!(
        grad_norm_sq > 0.0,
        "contact-regime gradient must be nonzero to test descent"
    );

    // A real descent step (same order as training's lr) must reduce loss.
    let alpha = 0.5;
    let mut stepped = controller.clone();
    for (w, g) in stepped.weights.iter_mut().zip(g_w.iter()) {
        *w -= alpha * g;
    }
    for (b, g) in stepped.bias.iter_mut().zip(g_b.iter()) {
        *b -= alpha * g;
    }
    let loss_stepped = loss_of(&stepped, &mut eval);

    assert!(
        loss_stepped < loss_base,
        "stepping against the analytic gradient must reduce the true loss: \
         base={loss_base:.6} stepped={loss_stepped:.6}"
    );
}

/// FD check of the WINDOWED (multi-state) loss seeding -- the mechanism
/// `controller_gradient` uses when `loss_window > 1`, where dL/dx is
/// injected at several substeps and must compose correctly with the
/// position-identity chain. Same smooth regime + single-particle loss +
/// origin placement rationale as the single-seed FD test above; loss =
/// average of particle 3's x over the last K states.
#[test]
fn windowed_loss_gradient_matches_finite_difference() {
    let (plan, mut cfg, mut eval) = setup_origin();
    cfg.gravity = 0.0;
    cfg.floor_y = -100.0;
    cfg.act_strength = 4.0;
    let steps = 40;
    let window = 8usize;
    let mut controller = SinusoidController::seeded(plan.n_groups, cfg.n_waves);
    controller.bias[1] = 0.15;
    controller.weights[2] = 0.2;

    let per_state = 1.0 / window as f32;
    let (g_w, g_b) = backprop_through_time(
        &plan,
        &controller,
        &mut eval,
        &cfg,
        steps,
        &mut |t, _next_state, seed| {
            if t >= steps - window {
                seed.x[3].x += per_state;
            }
        },
    );

    let loss_of = |c: &SinusoidController, eval: &mut StressEval| -> f32 {
        let (history, _) = rollout(&plan, c, eval, &cfg, steps);
        history[steps - window..]
            .iter()
            .map(|(s, _)| s.x[3].x)
            .sum::<f32>()
            / window as f32
    };

    let h = 5.0e-2_f32;
    for &wi in &[0usize, 2, 9] {
        let mut c_plus = controller.clone();
        c_plus.weights[wi] += h;
        let mut c_minus = controller.clone();
        c_minus.weights[wi] -= h;
        let numeric = (loss_of(&c_plus, &mut eval) - loss_of(&c_minus, &mut eval)) / (2.0 * h);
        let diff = (numeric - g_w[wi]).abs();
        let scale = numeric.abs().max(g_w[wi].abs()).max(1.0e-3);
        assert!(
            diff / scale < 5.0e-2,
            "windowed loss weight[{wi}] gradient mismatch: analytic={:.6} \
             numeric(central-diff)={numeric:.6} relative_diff={:.2e}",
            g_w[wi],
            diff / scale
        );
    }
    {
        let mut c_plus = controller.clone();
        c_plus.bias[1] += h;
        let mut c_minus = controller.clone();
        c_minus.bias[1] -= h;
        let numeric = (loss_of(&c_plus, &mut eval) - loss_of(&c_minus, &mut eval)) / (2.0 * h);
        let diff = (numeric - g_b[1]).abs();
        let scale = numeric.abs().max(g_b[1].abs()).max(1.0e-3);
        assert!(
            diff / scale < 5.0e-2,
            "windowed loss bias[1] gradient mismatch: analytic={:.6} numeric={numeric:.6} \
             relative_diff={:.2e}",
            g_b[1],
            diff / scale
        );
    }
}

/// FD check of `bounce_penalty`'s own gradient contribution -- the real
/// fix for the "still flies" gap `loss_window` alone left open (see
/// `DiffConfig::bounce_penalty`'s doc). Uses `controller_gradient`
/// directly (the public API a real trainer calls) with both the drift
/// window AND the penalty active together, so it verifies they compose
/// correctly, not just that the penalty term is correct in isolation.
#[test]
fn bounce_penalty_gradient_matches_finite_difference() {
    let (plan, mut cfg, mut eval) = setup_origin();
    cfg.gravity = 0.0;
    cfg.floor_y = -100.0;
    cfg.act_strength = 4.0;
    cfg.loss_window = 10;
    cfg.bounce_penalty = 2.0;
    let steps = 40;
    let mut controller = SinusoidController::seeded(plan.n_groups, cfg.n_waves);
    controller.bias[1] = 0.15;
    controller.weights[2] = 0.2;

    let (g_w, g_b) = controller_gradient(&plan, &controller, &mut eval, &cfg, steps);

    let loss_of = |c: &SinusoidController, eval: &mut StressEval| -> f32 {
        let (history, _) = rollout(&plan, c, eval, &cfg, steps);
        let n = plan.positions.len();
        let window = cfg.loss_window;
        let drift = history[steps - window..]
            .iter()
            .map(|(s, _)| s.mean_x())
            .sum::<f32>()
            / window as f32
            - DiffState::rest(&plan).mean_x();
        let bounce = history
            .iter()
            .map(|(s, _)| s.v.iter().map(|v| v.y * v.y).sum::<f32>())
            .sum::<f32>()
            / (n as f32 * steps as f32);
        -drift + cfg.bounce_penalty * bounce
    };

    let h = 5.0e-2_f32;
    for &wi in &[0usize, 2, 9] {
        let mut c_plus = controller.clone();
        c_plus.weights[wi] += h;
        let mut c_minus = controller.clone();
        c_minus.weights[wi] -= h;
        let numeric = (loss_of(&c_plus, &mut eval) - loss_of(&c_minus, &mut eval)) / (2.0 * h);
        let diff = (numeric - g_w[wi]).abs();
        let scale = numeric.abs().max(g_w[wi].abs()).max(1.0e-3);
        assert!(
            diff / scale < 5.0e-2,
            "bounce-penalty weight[{wi}] gradient mismatch: analytic={:.6} \
             numeric(central-diff)={numeric:.6} relative_diff={:.2e}",
            g_w[wi],
            diff / scale
        );
    }
    {
        let mut c_plus = controller.clone();
        c_plus.bias[1] += h;
        let mut c_minus = controller.clone();
        c_minus.bias[1] -= h;
        let numeric = (loss_of(&c_plus, &mut eval) - loss_of(&c_minus, &mut eval)) / (2.0 * h);
        let diff = (numeric - g_b[1]).abs();
        let scale = numeric.abs().max(g_b[1].abs()).max(1.0e-3);
        assert!(
            diff / scale < 5.0e-2,
            "bounce-penalty bias[1] gradient mismatch: analytic={:.6} numeric={numeric:.6} \
             relative_diff={:.2e}",
            g_b[1],
            diff / scale
        );
    }
}

/// FD check of `control_effort_penalty`'s gradient contribution -- the
/// "torque cost" half of reward shaping, alongside `bounce_penalty`'s
/// "don't bounce" half. Both active together, verifying composition
/// exactly like the bounce-penalty test above.
#[test]
fn control_effort_penalty_gradient_matches_finite_difference() {
    let (plan, mut cfg, mut eval) = setup_origin();
    cfg.gravity = 0.0;
    cfg.floor_y = -100.0;
    cfg.act_strength = 4.0;
    cfg.loss_window = 10;
    cfg.bounce_penalty = 1.0;
    cfg.control_effort_penalty = 3.0;
    let steps = 40;
    let mut controller = SinusoidController::seeded(plan.n_groups, cfg.n_waves);
    controller.bias[1] = 0.15;
    controller.weights[2] = 0.2;

    let (g_w, g_b) = controller_gradient(&plan, &controller, &mut eval, &cfg, steps);

    let loss_of = |c: &SinusoidController, eval: &mut StressEval| -> f32 {
        let (history, acts_cache) = rollout(&plan, c, eval, &cfg, steps);
        let n = plan.positions.len();
        let window = cfg.loss_window;
        let drift = history[steps - window..]
            .iter()
            .map(|(s, _)| s.mean_x())
            .sum::<f32>()
            / window as f32
            - DiffState::rest(&plan).mean_x();
        let bounce = history
            .iter()
            .map(|(s, _)| s.v.iter().map(|v| v.y * v.y).sum::<f32>())
            .sum::<f32>()
            / (n as f32 * steps as f32);
        let effort = acts_cache
            .iter()
            .map(|acts| acts.iter().map(|a| a * a).sum::<f32>())
            .sum::<f32>()
            / (c.n_groups as f32 * steps as f32);
        -drift + cfg.bounce_penalty * bounce + cfg.control_effort_penalty * effort
    };

    let h = 5.0e-2_f32;
    for &wi in &[0usize, 2, 9] {
        let mut c_plus = controller.clone();
        c_plus.weights[wi] += h;
        let mut c_minus = controller.clone();
        c_minus.weights[wi] -= h;
        let numeric = (loss_of(&c_plus, &mut eval) - loss_of(&c_minus, &mut eval)) / (2.0 * h);
        let diff = (numeric - g_w[wi]).abs();
        let scale = numeric.abs().max(g_w[wi].abs()).max(1.0e-3);
        assert!(
            diff / scale < 5.0e-2,
            "control-effort weight[{wi}] gradient mismatch: analytic={:.6} \
             numeric(central-diff)={numeric:.6} relative_diff={:.2e}",
            g_w[wi],
            diff / scale
        );
    }
    {
        let mut c_plus = controller.clone();
        c_plus.bias[1] += h;
        let mut c_minus = controller.clone();
        c_minus.bias[1] -= h;
        let numeric = (loss_of(&c_plus, &mut eval) - loss_of(&c_minus, &mut eval)) / (2.0 * h);
        let diff = (numeric - g_b[1]).abs();
        let scale = numeric.abs().max(g_b[1].abs()).max(1.0e-3);
        assert!(
            diff / scale < 5.0e-2,
            "control-effort bias[1] gradient mismatch: analytic={:.6} numeric={numeric:.6} \
             relative_diff={:.2e}",
            g_b[1],
            diff / scale
        );
    }
}

/// FD check of the bilateral-symmetry gradient redirection --
/// mirrored groups (right leg) must accumulate their gradient onto the
/// SAME underlying weights/bias as their source group (left leg), at
/// the source's own phase offset (0) reflected through the mirror's
/// extra phase, not their own dead weight slots. Uses `BodyPlan::biped`
/// with `with_biped_symmetry()` -- the actual configuration meant to
/// fix the one-legged-hop degenerate gait.
#[test]
fn bilateral_symmetry_gradient_matches_finite_difference() {
    let plan = BodyPlan::biped(Vec2::new(1.5, 1.3), 0.5);
    let cfg = DiffConfig {
        gravity: 0.0,
        floor_y: -100.0,
        act_strength: 4.0,
        ..DiffConfig::default()
    };
    let mut eval = StressEval::new(NeoHookeanMaterial::new(900.0, 700.0));
    let steps = 40;
    let mut controller =
        SinusoidController::seeded(plan.n_groups, cfg.n_waves).with_biped_symmetry();
    controller.bias[0] = 0.15; // left thigh (source of group 2's mirror)
    controller.weights[cfg.n_waves] = 0.2; // left foot (source of group 3's mirror)

    let n = plan.positions.len();
    let mut seed = vec![Vec2::ZERO; n];
    seed[2] = Vec2::new(1.0, 0.0);
    seed[n - 3] = Vec2::new(0.0, -0.5);

    let (g_w, g_b) = controller_gradient_seeded(&plan, &controller, &mut eval, &cfg, steps, &seed);

    // Groups 2 and 3 (right leg) are mirrors -- their own weight/bias
    // slots must never receive gradient; only groups 0 and 1 should.
    for j in 0..cfg.n_waves {
        assert_eq!(
            g_w[2 * cfg.n_waves + j],
            0.0,
            "mirrored group's own weight slot must stay untouched (gradient lands on source)"
        );
        assert_eq!(
            g_w[3 * cfg.n_waves + j],
            0.0,
            "mirrored group's own weight slot must stay untouched (gradient lands on source)"
        );
    }

    let loss_of = |c: &SinusoidController, eval: &mut StressEval| -> f32 {
        let (history, _) = rollout(&plan, c, eval, &cfg, steps);
        let fin = &history[steps - 1].0;
        fin.x[2].x - 0.5 * fin.x[n - 3].y
    };

    let h = 5.0e-2_f32;
    for &wi in &[0usize, cfg.n_waves + 1] {
        let mut c_plus = controller.clone();
        c_plus.weights[wi] += h;
        let mut c_minus = controller.clone();
        c_minus.weights[wi] -= h;
        let numeric = (loss_of(&c_plus, &mut eval) - loss_of(&c_minus, &mut eval)) / (2.0 * h);
        let diff = (numeric - g_w[wi]).abs();
        let scale = numeric.abs().max(g_w[wi].abs()).max(1.0e-3);
        assert!(
            diff / scale < 5.0e-2,
            "bilateral-symmetry weight[{wi}] gradient mismatch: analytic={:.6} \
             numeric(central-diff)={numeric:.6} relative_diff={:.2e}",
            g_w[wi],
            diff / scale
        );
    }
    {
        let mut c_plus = controller.clone();
        c_plus.bias[0] += h;
        let mut c_minus = controller.clone();
        c_minus.bias[0] -= h;
        let numeric = (loss_of(&c_plus, &mut eval) - loss_of(&c_minus, &mut eval)) / (2.0 * h);
        let diff = (numeric - g_b[0]).abs();
        let scale = numeric.abs().max(g_b[0].abs()).max(1.0e-3);
        assert!(
            diff / scale < 5.0e-2,
            "bilateral-symmetry bias[0] gradient mismatch: analytic={:.6} numeric={numeric:.6} \
             relative_diff={:.2e}",
            g_b[0],
            diff / scale
        );
    }
}

/// FD check of `FeedbackController::backward` in COMPLETE isolation
/// from the physics chain (no rollout, no P2G/G2P) -- pins down the
/// feature-extraction + centroid + linear-layer adjoint on its own
/// before it's ever trusted wired into a real substep, since this is
/// the highest-risk new derivation of the whole controller (mean/
/// centroid backward is exactly the kind of "looks obvious, has a
/// transpose/sign trap" derivation this project's discipline exists to
/// catch before it's trusted).
#[test]
fn feedback_controller_backward_matches_finite_difference() {
    let plan = BodyPlan::biped(Vec2::new(1.5, 1.3), 0.5);
    let controller = FeedbackController::seeded_with(plan.n_groups, 3);

    // A real, non-rest state: nonzero velocities and perturbed
    // positions, so both `rel` and `vel` features are nonzero.
    let mut state = DiffState::rest(&plan);
    for (i, x) in state.x.iter_mut().enumerate() {
        *x += Vec2::new(0.02 * (i as f32 % 5.0 - 2.0), 0.01 * (i as f32 % 3.0));
    }
    for (i, v) in state.v.iter_mut().enumerate() {
        *v = Vec2::new(0.03 * (i as f32 % 4.0 - 1.5), -0.02 * (i as f32 % 2.0));
    }

    // Loss reads two arbitrary particles' positions and velocities --
    // exercises both the `rel` (position) and `vel` gradient paths,
    // and (via the centroid) every particle's OWN position gradient,
    // not just the group directly queried.
    let g = Vec2::new(0.6, -0.4);
    let loss_of = |state: &DiffState| -> f32 {
        let acts = controller.activations(&plan, state);
        acts.iter()
            .enumerate()
            .map(|(g_id, a)| a * (1.0 + g_id as f32))
            .sum::<f32>()
            + g.dot(state.x[5])
            + g.dot(state.v[20])
    };

    let (feat, count) = FeedbackController::features(&plan, &state);
    let acts = controller.activations_from_features(&feat);
    // d(loss)/d(act[g]) from the sum-of-scaled-activations term.
    let g_act: Vec<f32> = (0..plan.n_groups).map(|g_id| 1.0 + g_id as f32).collect();
    let (g_w, g_b, mut g_x, mut g_v) = controller.backward(&plan, &feat, &count, &g_act);
    // Direct loss contributions bypassing the controller entirely.
    g_x[5] += g;
    g_v[20] += g;
    let _ = acts;

    let h = 1.0e-3_f32;

    // Weight and bias gradients.
    for &wi in &[0usize, 5, 12] {
        let mut c_plus = FeedbackController {
            weights: controller.weights.clone(),
            bias: controller.bias.clone(),
            n_groups: controller.n_groups,
        };
        c_plus.weights[wi] += h;
        let mut c_minus = FeedbackController {
            weights: controller.weights.clone(),
            bias: controller.bias.clone(),
            n_groups: controller.n_groups,
        };
        c_minus.weights[wi] -= h;
        let numeric = (c_plus
            .activations(&plan, &state)
            .iter()
            .enumerate()
            .map(|(g_id, a)| a * (1.0 + g_id as f32))
            .sum::<f32>()
            - c_minus
                .activations(&plan, &state)
                .iter()
                .enumerate()
                .map(|(g_id, a)| a * (1.0 + g_id as f32))
                .sum::<f32>())
            / (2.0 * h);
        let diff = (numeric - g_w[wi]).abs();
        let scale = numeric.abs().max(g_w[wi].abs()).max(1.0e-3);
        assert!(
            diff / scale < 5.0e-2,
            "feedback controller weight[{wi}] gradient mismatch: analytic={:.6} \
             numeric(central-diff)={numeric:.6} relative_diff={:.2e}",
            g_w[wi],
            diff / scale
        );
    }
    for (bi, &g_b_bi) in g_b.iter().enumerate() {
        let mut c_plus = FeedbackController {
            weights: controller.weights.clone(),
            bias: controller.bias.clone(),
            n_groups: controller.n_groups,
        };
        c_plus.bias[bi] += h;
        let mut c_minus = FeedbackController {
            weights: controller.weights.clone(),
            bias: controller.bias.clone(),
            n_groups: controller.n_groups,
        };
        c_minus.bias[bi] -= h;
        let numeric = (c_plus
            .activations(&plan, &state)
            .iter()
            .enumerate()
            .map(|(g_id, a)| a * (1.0 + g_id as f32))
            .sum::<f32>()
            - c_minus
                .activations(&plan, &state)
                .iter()
                .enumerate()
                .map(|(g_id, a)| a * (1.0 + g_id as f32))
                .sum::<f32>())
            / (2.0 * h);
        let diff = (numeric - g_b_bi).abs();
        let scale = numeric.abs().max(g_b_bi.abs()).max(1.0e-3);
        assert!(
            diff / scale < 5.0e-2,
            "feedback controller bias[{bi}] gradient mismatch: analytic={:.6} \
             numeric(central-diff)={numeric:.6} relative_diff={:.2e}",
            g_b_bi,
            diff / scale
        );
    }

    // Position and velocity gradients (the new, real-risk paths):
    // perturb particle 5's x and particle 20's v directly.
    for &(pi, axis) in &[(5usize, 0usize), (5, 1), (20, 0), (20, 1)] {
        let mut s_plus = state.clone();
        let mut s_minus = state.clone();
        if axis == 0 {
            s_plus.x[pi].x += h;
            s_minus.x[pi].x -= h;
        } else {
            s_plus.x[pi].y += h;
            s_minus.x[pi].y -= h;
        }
        let numeric = (loss_of(&s_plus) - loss_of(&s_minus)) / (2.0 * h);
        let analytic = if axis == 0 { g_x[pi].x } else { g_x[pi].y };
        let diff = (numeric - analytic).abs();
        let scale = numeric.abs().max(analytic.abs()).max(1.0e-3);
        assert!(
            diff / scale < 5.0e-2,
            "feedback controller position[{pi}].{axis} gradient mismatch: analytic={analytic:.6} \
             numeric(central-diff)={numeric:.6} relative_diff={:.2e}",
            diff / scale
        );
    }
    for &(pi, axis) in &[(20usize, 0usize), (20, 1)] {
        let mut s_plus = state.clone();
        let mut s_minus = state.clone();
        if axis == 0 {
            s_plus.v[pi].x += h;
            s_minus.v[pi].x -= h;
        } else {
            s_plus.v[pi].y += h;
            s_minus.v[pi].y -= h;
        }
        let numeric = (loss_of(&s_plus) - loss_of(&s_minus)) / (2.0 * h);
        let analytic = if axis == 0 { g_v[pi].x } else { g_v[pi].y };
        let diff = (numeric - analytic).abs();
        let scale = numeric.abs().max(analytic.abs()).max(1.0e-3);
        assert!(
            diff / scale < 5.0e-2,
            "feedback controller velocity[{pi}].{axis} gradient mismatch: analytic={analytic:.6} \
             numeric(central-diff)={numeric:.6} relative_diff={:.2e}",
            diff / scale
        );
    }

    // A particle with NO group (passive torso) must still receive a
    // nonzero position gradient purely through the centroid term.
    let torso_particle = plan
        .group
        .iter()
        .position(|g| g.is_none())
        .expect("body plan must have passive torso particles");
    assert!(
        g_x[torso_particle].length() > 0.0,
        "passive (ungrouped) particles must still get a centroid-mediated position gradient"
    );
}

/// The end-to-end claim: gradient descent on the sinusoid controller
/// finds a gait that beats the (near-zero-drift) untrained start.
#[test]
fn training_finds_a_gait() {
    let (plan, cfg, mut eval) = setup();
    let steps = 80;
    let mut controller = SinusoidController::seeded(plan.n_groups, cfg.n_waves);

    let drift_before = drift(&plan, &controller, &mut eval, &cfg, steps);
    let drifts = train(&plan, &mut controller, &mut eval, &cfg, steps, 40, 0.5);
    let drift_after = *drifts.last().unwrap();

    println!(
        "training_finds_a_gait: {} particles, {steps} substeps, 40 iterations\n  \
         untrained drift: {drift_before:.4}\n  trained drift:   {drift_after:.4}",
        plan.positions.len()
    );

    assert!(
        drift_after > drift_before && drift_after > 0.0,
        "trained gait should produce real positive drift beyond the untrained start: \
         before={drift_before:.4} after={drift_after:.4}"
    );
}

use super::backward::{controller_gradient, feedback_controller_gradient};
use super::body_plan::BodyPlan;
use super::config::{DiffConfig, DiffState, FeedbackController, SinusoidController};
use super::forward::{rollout, rollout_feedback};
use super::stress::StressEval;

/// Plain gradient descent on the controller parameters. Returns per-iteration
/// drift so callers can report/plot training progress.
///
/// Keeps the BEST-drift parameters seen, not the last: measured on the real
/// walker, late training oscillates (a 600-substep horizon regressed from a
/// 0.74 best back to 0.06 by the final iteration -- the classic
/// backprop-through-time instability, sharpened here by the contact kinks),
/// so `controller` is restored to its best-scoring snapshot before returning.
/// Standard model selection, not a workaround specific to this trainer.
///
/// Uses Adam (Kingma & Ba 2014), not plain gradient descent: a real, measured
/// symptom found live 2026-07-11 -- with a fixed step size, the same body
/// went from "flies" (contact 0.31) to "frozen" (drift 0.05, near-zero
/// movement) between bounce-penalty values 0.05 and 0.1, a razor-thin usable
/// range. That's the textbook fixed-step-size failure: SGD takes the same
/// size step regardless of how consistent or noisy a parameter's gradient
/// history has been. Adam tracks per-parameter first/second moment
/// estimates and scales each parameter's step by them, damping steps for
/// noisy/spiky gradients and taking confident steps where gradients are
/// small but consistent -- standard fix for exactly this symptom, not a
/// tuning trick specific to this trainer. `lr` here is Adam's own learning
/// rate (typically much smaller than an SGD-tuned one, ~1e-2 to 1e-1 for
/// this problem's scale, not ~1.0).
pub fn train(
    plan: &BodyPlan,
    controller: &mut SinusoidController,
    eval: &mut StressEval,
    cfg: &DiffConfig,
    steps: usize,
    iterations: usize,
    lr: f32,
) -> Vec<f32> {
    const BETA1: f32 = 0.9;
    const BETA2: f32 = 0.999;
    const EPS: f32 = 1.0e-8;

    let mut drifts = Vec::with_capacity(iterations);
    let mut best_score = f32::NEG_INFINITY;
    let mut best = controller.clone();
    let window = cfg.loss_window.clamp(1, steps);
    let rest_x = DiffState::rest(plan).mean_x();

    let mut m_w = vec![0.0f32; controller.weights.len()];
    let mut v_w = vec![0.0f32; controller.weights.len()];
    let mut m_b = vec![0.0f32; controller.bias.len()];
    let mut v_b = vec![0.0f32; controller.bias.len()];

    for iter in 1..=iterations {
        let (g_w, g_b) = controller_gradient(plan, controller, eval, cfg, steps);
        let bias_correction1 = 1.0 - BETA1.powi(iter as i32);
        let bias_correction2 = 1.0 - BETA2.powi(iter as i32);

        for (((w, g), m), v) in controller
            .weights
            .iter_mut()
            .zip(g_w.iter())
            .zip(m_w.iter_mut())
            .zip(v_w.iter_mut())
        {
            *m = BETA1 * *m + (1.0 - BETA1) * g;
            *v = BETA2 * *v + (1.0 - BETA2) * g * g;
            let m_hat = *m / bias_correction1;
            let v_hat = *v / bias_correction2;
            *w -= lr * m_hat / (v_hat.sqrt() + EPS);
        }
        for (((b, g), m), v) in controller
            .bias
            .iter_mut()
            .zip(g_b.iter())
            .zip(m_b.iter_mut())
            .zip(v_b.iter_mut())
        {
            *m = BETA1 * *m + (1.0 - BETA1) * g;
            *v = BETA2 * *v + (1.0 - BETA2) * g * g;
            let m_hat = *m / bias_correction1;
            let v_hat = *v / bias_correction2;
            *b -= lr * m_hat / (v_hat.sqrt() + EPS);
        }

        // Score by the SAME full objective training optimizes (windowed
        // drift, bounce penalty, control-effort penalty), so model
        // selection can't quietly reintroduce an exploit the objective was
        // extended to remove.
        let (history, acts_cache) = rollout(plan, controller, eval, cfg, steps);
        let n = plan.positions.len();
        let windowed_drift = history[steps - window..]
            .iter()
            .map(|(s, _)| s.mean_x() - rest_x)
            .sum::<f32>()
            / window as f32;
        let bounce = history
            .iter()
            .map(|(s, _)| s.v.iter().map(|v| v.y * v.y).sum::<f32>())
            .sum::<f32>()
            / (n as f32 * steps as f32);
        let effort = acts_cache
            .iter()
            .map(|acts| acts.iter().map(|a| a * a).sum::<f32>())
            .sum::<f32>()
            / (controller.n_groups as f32 * steps as f32);
        let score =
            windowed_drift - cfg.bounce_penalty * bounce - cfg.control_effort_penalty * effort;
        if score > best_score {
            best_score = score;
            best = controller.clone();
        }
        drifts.push(history[steps - 1].0.mean_x() - rest_x);
    }
    *controller = best;
    drifts
}

/// `FeedbackController` analogue of `train` -- identical Adam loop and
/// keep-best model selection, only the gradient source and rollout differ.
pub fn train_feedback(
    plan: &BodyPlan,
    controller: &mut FeedbackController,
    eval: &mut StressEval,
    cfg: &DiffConfig,
    steps: usize,
    iterations: usize,
    lr: f32,
) -> Vec<f32> {
    const BETA1: f32 = 0.9;
    const BETA2: f32 = 0.999;
    const EPS: f32 = 1.0e-8;

    let mut drifts = Vec::with_capacity(iterations);
    let mut best_score = f32::NEG_INFINITY;
    let mut best = controller.clone();
    let window = cfg.loss_window.clamp(1, steps);
    let rest_x = DiffState::rest(plan).mean_x();

    let mut m_w = vec![0.0f32; controller.weights.len()];
    let mut v_w = vec![0.0f32; controller.weights.len()];
    let mut m_b = vec![0.0f32; controller.bias.len()];
    let mut v_b = vec![0.0f32; controller.bias.len()];

    for iter in 1..=iterations {
        let (g_w, g_b) = feedback_controller_gradient(plan, controller, eval, cfg, steps);
        let bias_correction1 = 1.0 - BETA1.powi(iter as i32);
        let bias_correction2 = 1.0 - BETA2.powi(iter as i32);

        for (((w, g), m), v) in controller
            .weights
            .iter_mut()
            .zip(g_w.iter())
            .zip(m_w.iter_mut())
            .zip(v_w.iter_mut())
        {
            *m = BETA1 * *m + (1.0 - BETA1) * g;
            *v = BETA2 * *v + (1.0 - BETA2) * g * g;
            let m_hat = *m / bias_correction1;
            let v_hat = *v / bias_correction2;
            *w -= lr * m_hat / (v_hat.sqrt() + EPS);
        }
        for (((b, g), m), v) in controller
            .bias
            .iter_mut()
            .zip(g_b.iter())
            .zip(m_b.iter_mut())
            .zip(v_b.iter_mut())
        {
            *m = BETA1 * *m + (1.0 - BETA1) * g;
            *v = BETA2 * *v + (1.0 - BETA2) * g * g;
            let m_hat = *m / bias_correction1;
            let v_hat = *v / bias_correction2;
            *b -= lr * m_hat / (v_hat.sqrt() + EPS);
        }

        let (history, acts_cache) = rollout_feedback(plan, controller, eval, cfg, steps);
        let n = plan.positions.len();
        let windowed_drift = history[steps - window..]
            .iter()
            .map(|(s, _)| s.mean_x() - rest_x)
            .sum::<f32>()
            / window as f32;
        let bounce = history
            .iter()
            .map(|(s, _)| s.v.iter().map(|v| v.y * v.y).sum::<f32>())
            .sum::<f32>()
            / (n as f32 * steps as f32);
        let effort = acts_cache
            .iter()
            .map(|acts| acts.iter().map(|a| a * a).sum::<f32>())
            .sum::<f32>()
            / (controller.n_groups as f32 * steps as f32);
        let score =
            windowed_drift - cfg.bounce_penalty * bounce - cfg.control_effort_penalty * effort;
        if score > best_score {
            best_score = score;
            best = controller.clone();
        }
        drifts.push(history[steps - 1].0.mean_x() - rest_x);
    }
    *controller = best;
    drifts
}

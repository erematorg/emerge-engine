use super::backward::{controller_gradient, feedback_controller_gradient};
use super::body_plan::BodyPlan;
use super::config::{DiffConfig, DiffState, FeedbackController, SinusoidController, StepRecord};
use super::forward::{rollout, rollout_feedback};
use super::stress::StressEval;

/// Trains the controller parameters by gradient descent with Adam. Returns
/// per-iteration drift so callers can report/plot training progress.
///
/// Keeps the best-drift parameters seen, not the last: late training
/// oscillates (on the walker, a 600-substep horizon fell from a 0.74 best
/// back to 0.06 by the final iteration, the usual backprop-through-time
/// instability sharpened by contact kinks), so `controller` is restored to
/// its best-scoring snapshot before returning (standard model selection).
///
/// Uses Adam (see `Adam`), not fixed-step gradient descent: with a
/// fixed step the same body went from "flies" (contact 0.31) to "frozen"
/// (drift 0.05) between bounce-penalty values 0.05 and 0.1, a very narrow
/// usable range. Adam tracks per-parameter first and second moment
/// estimates and scales each step by them, damping noisy gradients and
/// taking confident steps where gradients are small but consistent. `lr`
/// is Adam's learning rate (~1e-2 to 1e-1 at this problem's scale, much
/// smaller than an SGD-tuned ~1.0).
pub fn train(
    plan: &BodyPlan,
    controller: &mut SinusoidController,
    eval: &mut StressEval,
    cfg: &DiffConfig,
    steps: usize,
    iterations: usize,
    lr: f32,
) -> Vec<f32> {
    let mut drifts = Vec::with_capacity(iterations);
    let mut best_score = f32::NEG_INFINITY;
    let mut best = controller.clone();
    let rest_x = DiffState::rest(plan).mean_x();
    let mut adam_w = Adam::new(controller.weights.len());
    let mut adam_b = Adam::new(controller.bias.len());

    for iter in 1..=iterations {
        let (g_w, g_b) = controller_gradient(plan, controller, eval, cfg, steps);
        adam_w.step(&mut controller.weights, &g_w, lr, iter);
        adam_b.step(&mut controller.bias, &g_b, lr, iter);

        let (history, acts_cache) = rollout(plan, controller, eval, cfg, steps);
        let score = rollout_score(
            plan,
            &history,
            &acts_cache,
            controller.n_groups,
            rest_x,
            cfg,
        );
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
    let mut drifts = Vec::with_capacity(iterations);
    let mut best_score = f32::NEG_INFINITY;
    let mut best = controller.clone();
    let rest_x = DiffState::rest(plan).mean_x();
    let mut adam_w = Adam::new(controller.weights.len());
    let mut adam_b = Adam::new(controller.bias.len());

    for iter in 1..=iterations {
        let (g_w, g_b) = feedback_controller_gradient(plan, controller, eval, cfg, steps);
        adam_w.step(&mut controller.weights, &g_w, lr, iter);
        adam_b.step(&mut controller.bias, &g_b, lr, iter);

        let (history, acts_cache) = rollout_feedback(plan, controller, eval, cfg, steps);
        let score = rollout_score(
            plan,
            &history,
            &acts_cache,
            controller.n_groups,
            rest_x,
            cfg,
        );
        if score > best_score {
            best_score = score;
            best = controller.clone();
        }
        drifts.push(history[steps - 1].0.mean_x() - rest_x);
    }
    *controller = best;
    drifts
}

/// Adam (Kingma & Ba 2015, "Adam: A Method for Stochastic Optimization",
/// ICLR, Algorithm 1) over one parameter vector, with the paper's default
/// decay rates and epsilon.
struct Adam {
    /// First moment estimate, one per parameter.
    m: Vec<f32>,
    /// Second raw moment estimate, one per parameter.
    v: Vec<f32>,
}

impl Adam {
    const BETA1: f32 = 0.9;
    const BETA2: f32 = 0.999;
    const EPS: f32 = 1.0e-8;

    fn new(len: usize) -> Self {
        Self {
            m: vec![0.0; len],
            v: vec![0.0; len],
        }
    }

    /// One bias-corrected update of `params` along `grads`, `iter` counting
    /// from 1.
    fn step(&mut self, params: &mut [f32], grads: &[f32], lr: f32, iter: usize) {
        let bias_correction1 = 1.0 - Self::BETA1.powi(iter as i32);
        let bias_correction2 = 1.0 - Self::BETA2.powi(iter as i32);
        for (((p, g), m), v) in params
            .iter_mut()
            .zip(grads.iter())
            .zip(self.m.iter_mut())
            .zip(self.v.iter_mut())
        {
            *m = Self::BETA1 * *m + (1.0 - Self::BETA1) * g;
            *v = Self::BETA2 * *v + (1.0 - Self::BETA2) * g * g;
            let m_hat = *m / bias_correction1;
            let v_hat = *v / bias_correction2;
            *p -= lr * m_hat / (v_hat.sqrt() + Self::EPS);
        }
    }
}

/// Model-selection score of one rollout: the same full objective training
/// optimizes (windowed drift, bounce penalty, control-effort penalty), so
/// model selection can't quietly reintroduce an exploit the objective was
/// extended to remove.
fn rollout_score(
    plan: &BodyPlan,
    history: &[(DiffState, StepRecord)],
    acts_cache: &[Vec<f32>],
    n_groups: usize,
    rest_x: f32,
    cfg: &DiffConfig,
) -> f32 {
    let steps = history.len();
    let window = cfg.loss_window.clamp(1, steps);
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
        / (n_groups as f32 * steps as f32);
    windowed_drift - cfg.bounce_penalty * bounce - cfg.control_effort_penalty * effort
}

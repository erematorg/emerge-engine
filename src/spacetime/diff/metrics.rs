use super::body_plan::BodyPlan;
use super::config::{DiffConfig, DiffState, FeedbackController, SinusoidController, StepRecord};
use super::forward::{rollout, rollout_feedback};
use super::stress::StressEval;

/// Forward drift (grid units) of the body's mean x over a rollout -- the
/// quantity training maximizes.
pub fn drift(
    plan: &BodyPlan,
    controller: &SinusoidController,
    eval: &mut StressEval,
    cfg: &DiffConfig,
    steps: usize,
) -> f32 {
    let (history, _) = rollout(plan, controller, eval, cfg, steps);
    history[steps - 1].0.mean_x() - DiffState::rest(plan).mean_x()
}

/// `FeedbackController` analogue of `drift`.
pub fn drift_feedback(
    plan: &BodyPlan,
    controller: &FeedbackController,
    eval: &mut StressEval,
    cfg: &DiffConfig,
    steps: usize,
) -> f32 {
    let (history, _) = rollout_feedback(plan, controller, eval, cfg, steps);
    history[steps - 1].0.mean_x() - DiffState::rest(plan).mean_x()
}

/// Gait-quality metrics for judging whether a trained controller WALKS
/// rather than hurls itself -- the difference a drift number alone can't
/// see (observed live: a final-state-drift-trained gait "goes flying").
pub struct GaitMetrics {
    /// Final-state mean-x drift (what `drift` reports).
    pub final_drift: f32,
    /// Drift averaged over the last `cfg.loss_window` states (the windowed
    /// training objective).
    pub windowed_drift: f32,
    /// Fraction of substeps where the body's lowest particle is within one
    /// particle spacing of the floor -- ~1.0 for a grounded walk, small for
    /// ballistic hopping.
    pub contact_fraction: f32,
    /// Highest the body's LOWEST particle ever gets above the floor -- a
    /// direct "how airborne did it go" measure (grid units).
    pub max_clearance: f32,
}

pub fn gait_metrics(
    plan: &BodyPlan,
    controller: &SinusoidController,
    eval: &mut StressEval,
    cfg: &DiffConfig,
    steps: usize,
    contact_band: f32,
) -> GaitMetrics {
    let (history, _) = rollout(plan, controller, eval, cfg, steps);
    gait_metrics_from_history(plan, cfg, steps, contact_band, &history)
}

/// `FeedbackController` analogue of `gait_metrics`.
pub fn gait_metrics_feedback(
    plan: &BodyPlan,
    controller: &FeedbackController,
    eval: &mut StressEval,
    cfg: &DiffConfig,
    steps: usize,
    contact_band: f32,
) -> GaitMetrics {
    let (history, _) = rollout_feedback(plan, controller, eval, cfg, steps);
    gait_metrics_from_history(plan, cfg, steps, contact_band, &history)
}

fn gait_metrics_from_history(
    plan: &BodyPlan,
    cfg: &DiffConfig,
    steps: usize,
    contact_band: f32,
    history: &[(DiffState, StepRecord)],
) -> GaitMetrics {
    let rest_x = DiffState::rest(plan).mean_x();

    let final_drift = history[steps - 1].0.mean_x() - rest_x;
    let window = cfg.loss_window.clamp(1, steps);
    let windowed_drift = history[steps - window..]
        .iter()
        .map(|(s, _)| s.mean_x() - rest_x)
        .sum::<f32>()
        / window as f32;

    let mut contact_steps = 0usize;
    let mut max_clearance = 0.0f32;
    for (state, _) in history.iter() {
        let lowest = state.x.iter().map(|p| p.y).fold(f32::INFINITY, f32::min);
        let clearance = (lowest - cfg.floor_y).max(0.0);
        if clearance <= contact_band {
            contact_steps += 1;
        }
        max_clearance = max_clearance.max(clearance);
    }

    GaitMetrics {
        final_drift,
        windowed_drift,
        contact_fraction: contact_steps as f32 / steps as f32,
        max_clearance,
    }
}

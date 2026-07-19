use glam::{IVec2, Mat2, Vec2};
use std::collections::{BTreeMap, BTreeSet};

use crate::grid::kernel::quadratic_weights;

use super::body_plan::BodyPlan;
use super::config::{DiffConfig, DiffState, FeedbackController, SinusoidController, StepRecord};
use super::stress::{StressEval, signed_active_stress};

// ── Forward ───────────────────────────────────────────────────────────────────

fn total_stress(
    eval: &mut StressEval,
    f: Mat2,
    act: f32,
    fiber_dir: Vec2,
    cfg: &DiffConfig,
) -> Mat2 {
    eval.passive(f) + signed_active_stress(f, act, cfg.act_strength, fiber_dir)
}

/// One full differentiable substep: P2G scatter -> grid update (gravity +
/// sticky floor) -> G2P gather -> position/F update. `acts` is one signed
/// activation per muscle group at this substep.
pub fn forward_substep(
    state: &DiffState,
    plan: &BodyPlan,
    acts: &[f32],
    eval: &mut StressEval,
    cfg: &DiffConfig,
) -> (DiffState, StepRecord) {
    let n = state.x.len();
    let mut momentum_map: BTreeMap<(i32, i32), Vec2> = BTreeMap::new();
    let mut mass_map: BTreeMap<(i32, i32), f32> = BTreeMap::new();

    for i in 0..n {
        let act = plan.group[i].map_or(0.0, |g| acts[g]);
        let fiber = plan.group[i].map_or(Vec2::Y, |g| plan.fiber_dir[g]);
        let stress = total_stress(eval, state.f[i], act, fiber, cfg);
        let w = quadratic_weights(state.x[i]);
        for gx in 0..3 {
            for gy in 0..3 {
                let weight = w.wx[gx] * w.wy[gy];
                let cell_pos = w.base_cell + IVec2::new(gx as i32 - 1, gy as i32 - 1);
                let cell_dist = cell_pos.as_vec2() - state.x[i] + Vec2::splat(0.5);
                let momentum = weight
                    * (cfg.mass * (state.v[i] + state.c[i] * cell_dist)
                        + cfg.stress_coeff * (stress * cell_dist));
                let key = (cell_pos.x, cell_pos.y);
                *momentum_map.entry(key).or_insert(Vec2::ZERO) += momentum;
                *mass_map.entry(key).or_insert(0.0) += weight * cfg.mass;
            }
        }
    }

    let mut vel_map: BTreeMap<(i32, i32), Vec2> = BTreeMap::new();
    let mut stuck = BTreeSet::new();
    for (&cell, &mass_c) in mass_map.iter() {
        // Zero-weight guard, same as the real `Grid::update_velocities`.
        if mass_c <= 0.0 {
            continue;
        }
        let mut v = momentum_map[&cell] / mass_c;
        v.y -= cfg.gravity * cfg.dt;
        // Sticky floor (verified-real DiffTaichi walker behavior): a floor
        // cell moving downward loses its velocity entirely.
        if (cell.1 as f32) <= cfg.floor_y && v.y < 0.0 {
            v = Vec2::ZERO;
            stuck.insert(cell);
        }
        vel_map.insert(cell, v);
    }

    let mut next = DiffState {
        x: Vec::with_capacity(n),
        v: Vec::with_capacity(n),
        c: Vec::with_capacity(n),
        f: Vec::with_capacity(n),
    };
    for (&x, &f) in state.x.iter().zip(state.f.iter()) {
        let w = quadratic_weights(x);
        let mut new_v = Vec2::ZERO;
        let mut b = Mat2::ZERO;
        for gx in 0..3 {
            for gy in 0..3 {
                let weight = w.wx[gx] * w.wy[gy];
                let cell_pos = w.base_cell + IVec2::new(gx as i32 - 1, gy as i32 - 1);
                let cell_dist = cell_pos.as_vec2() - x + Vec2::splat(0.5);
                let v_cell = *vel_map
                    .get(&(cell_pos.x, cell_pos.y))
                    .unwrap_or(&Vec2::ZERO);
                let weighted = v_cell * weight;
                new_v += weighted;
                b += Mat2::from_cols(weighted * cell_dist.x, weighted * cell_dist.y);
            }
        }
        let new_c = b * (cfg.kernel_d_inverse * cfg.apic_blend);
        next.x.push(x + new_v * cfg.dt);
        next.v.push(new_v);
        next.c.push(new_c);
        next.f.push((Mat2::IDENTITY + cfg.dt * new_c) * f);
    }

    (next, StepRecord { stuck })
}

/// Full rollout: `steps` substeps from the rest state, returning per-step
/// results and the per-step group activations used (cached for backward).
pub fn rollout(
    plan: &BodyPlan,
    controller: &SinusoidController,
    eval: &mut StressEval,
    cfg: &DiffConfig,
    steps: usize,
) -> (Vec<(DiffState, StepRecord)>, Vec<Vec<f32>>) {
    let mut history = Vec::with_capacity(steps);
    let mut acts_cache = Vec::with_capacity(steps);
    let mut current = DiffState::rest(plan);
    for t in 0..steps {
        let acts = controller.activations(cfg, t);
        let (next, record) = forward_substep(&current, plan, &acts, eval, cfg);
        history.push((next.clone(), record));
        acts_cache.push(acts);
        current = next;
    }
    (history, acts_cache)
}

/// Same rollout, but activation comes from the CURRENT state
/// (`FeedbackController::activations`) instead of a fixed time-based
/// rhythm -- reuses `forward_substep` unchanged (it already takes
/// activations as a plain slice, indifferent to their source).
pub fn rollout_feedback(
    plan: &BodyPlan,
    controller: &FeedbackController,
    eval: &mut StressEval,
    cfg: &DiffConfig,
    steps: usize,
) -> (Vec<(DiffState, StepRecord)>, Vec<Vec<f32>>) {
    let mut history = Vec::with_capacity(steps);
    let mut acts_cache = Vec::with_capacity(steps);
    let mut current = DiffState::rest(plan);
    for _ in 0..steps {
        let acts = controller.activations(plan, &current);
        let (next, record) = forward_substep(&current, plan, &acts, eval, cfg);
        history.push((next.clone(), record));
        acts_cache.push(acts);
        current = next;
    }
    (history, acts_cache)
}

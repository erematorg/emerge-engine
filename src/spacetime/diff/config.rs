use glam::{Mat2, Vec2};
use std::collections::BTreeSet;

use crate::solver::config::KERNEL_D_INVERSE;

use super::body_plan::BodyPlan;

// ── Config / controller / state ───────────────────────────────────────────────

pub struct DiffConfig {
    pub mass: f32,
    /// P2G stress premultiplier: `-V0 * KERNEL_D_INVERSE * dt` in the real
    /// solver; a free constant here.
    pub stress_coeff: f32,
    pub dt: f32,
    pub kernel_d_inverse: f32,
    pub apic_blend: f32,
    /// Downward gravitational acceleration (grid units / s^2).
    pub gravity: f32,
    /// Sticky-floor height: grid cells at `y <= floor_y` moving downward get
    /// their velocity zeroed (the verified real behavior of DiffTaichi's
    /// canonical walker floor).
    pub floor_y: f32,
    /// Active-stress scale (DiffTaichi's `act_strength`).
    pub act_strength: f32,
    /// Sinusoid basis size per group.
    pub n_waves: usize,
    /// Gait angular frequency (rad/s of *simulated* time).
    pub omega: f32,
    /// Training loss averages drift over the LAST this-many states of the
    /// rollout (1 = final state only). A final-state-only loss rewards
    /// ending far right by any means -- including ballistic hops; a window
    /// rewards sustained progress. See `controller_gradient`.
    pub loss_window: usize,
    /// Coefficient penalizing mean squared vertical velocity across the
    /// WHOLE rollout, added to the training loss. Real root cause found
    /// live 2026-07-11: `loss_window` alone rewards "consistently far right
    /// through sustained contact" but does not PENALIZE vertical motion
    /// itself -- if a ballistic hop still covers more ground per unit loss
    /// than a grounded gait, gradient descent takes the hop regardless of
    /// how the actuators are arranged (confirmed: adding horizontal
    /// push-off muscles alone, see `BodyPlan::biped`, improved vy/vx from
    /// 1.76 to 1.46-1.58 but left contact fraction at ~0.22-0.23 once
    /// actuation was strong enough to move any real distance). Direct
    /// penalty on vertical velocity is the standard fix in published
    /// legged-locomotion reward functions (torso-height/vertical-velocity
    /// penalties are near-universal there) -- 0.0 disables it (backward
    /// compatible default).
    pub bounce_penalty: f32,
    /// Coefficient penalizing mean squared activation across the whole
    /// rollout and all groups -- the "torque cost" half of standard
    /// locomotion reward shaping (`bounce_penalty` is the other half).
    /// Discourages the controller from firing muscles harder than needed,
    /// which tends to produce smoother, less erratic gaits as a side
    /// effect. 0.0 disables it (backward compatible default).
    pub control_effort_penalty: f32,
}

impl Default for DiffConfig {
    fn default() -> Self {
        Self {
            mass: 1.0,
            stress_coeff: -0.05,
            dt: 0.01,
            kernel_d_inverse: KERNEL_D_INVERSE,
            apic_blend: 1.0,
            gravity: 2.0,
            floor_y: 1.0,
            act_strength: 12.0,
            n_waves: 4,
            omega: 16.0,
            loss_window: 1,
            bounce_penalty: 0.0,
            control_effort_penalty: 0.0,
        }
    }
}

/// Open-loop sinusoid-basis controller (DiffTaichi `compute_actuation`):
/// `act[t, g] = tanh( sum_j w[g,j] * sin(omega * t * dt + 2*pi*j/n_waves) + b[g] )`.
#[derive(Clone)]
pub struct SinusoidController {
    /// Row-major `[group][wave]`.
    pub weights: Vec<f32>,
    pub bias: Vec<f32>,
    pub n_groups: usize,
    pub n_waves: usize,
    /// Bilateral-symmetry constraint: if `mirror_of[g] = Some(s)`, group `g`
    /// reuses group `s`'s weights/bias (a mirrored, not independent, muscle)
    /// instead of having its own free parameters. `None` = free/trainable.
    ///
    /// Real technique, found live 2026-07-11 after a diagnosed failure: a
    /// trained biped with fully independent left/right controllers found a
    /// ONE-LEGGED HOP (one leg permanently retracted, the other doing all
    /// the work) -- nothing in a pure drift/bounce/effort loss requires the
    /// two legs to alternate, and that degenerate solution is simpler for
    /// gradient descent to find than genuine alternation. Cross-checked
    /// against EvoSoro's real, published soft-robot evolution source
    /// (`evosoro/networks.py`, `enforce_symmetry()`): it mirrors left/right
    /// genome parameters structurally so an asymmetric solution can't even
    /// be represented, rather than hoping a loss term discourages it.
    /// Combined here with `phase_offset` (standard CPG anti-phase coupling
    /// for ALTERNATING, not synchronized, gaits): mirroring alone would make
    /// both legs move identically in phase (a two-legged synchronized hop,
    /// not a walk); the phase offset is what turns that into alternation.
    pub mirror_of: Vec<Option<usize>>,
    /// Extra phase (radians) added to group `g`'s sinusoid argument.
    pub phase_offset: Vec<f32>,
}

impl SinusoidController {
    /// Small deterministic pseudo-random init (DiffTaichi uses N(0, 0.01);
    /// this uses a hash-based equivalent so runs reproduce exactly). No
    /// bilateral symmetry by default (`mirror_of` all `None`, `phase_offset`
    /// all 0) -- fully independent per-group parameters, as before.
    pub fn seeded(n_groups: usize, n_waves: usize) -> Self {
        Self::seeded_with(n_groups, n_waves, 0)
    }

    /// Same as `seeded`, but with an explicit `seed` -- every call to
    /// `seeded` (no seed argument) used the SAME index-only hash for the
    /// entire session, meaning every hyperparameter sweep started from the
    /// literal same initial weights every time. Non-convex training
    /// standardly needs multiple random restarts, not just hyperparameter
    /// search over a single fixed starting point -- real gap, found late
    /// 2026-07-11 after several sweeps converged to different DEGENERATE
    /// solutions (frozen, one-legged hop, monotonic tilt) without ever
    /// trying a different basin of attraction.
    pub fn seeded_with(n_groups: usize, n_waves: usize, seed: u32) -> Self {
        let rand = |i: usize| -> f32 {
            let x = (i as u32)
                .wrapping_mul(2_654_435_761)
                .wrapping_add(seed.wrapping_mul(40_503));
            let x = x ^ (x >> 15);
            let x = x.wrapping_mul(0x27d4_eb2d);
            let x = x ^ (x >> 15);
            ((x % 2000) as f32 / 1000.0 - 1.0) * 0.01
        };
        Self {
            weights: (0..n_groups * n_waves).map(rand).collect(),
            bias: vec![0.0; n_groups],
            n_groups,
            n_waves,
            mirror_of: vec![None; n_groups],
            phase_offset: vec![0.0; n_groups],
        }
    }

    /// Configures bilateral symmetry matching `BodyPlan::biped`'s group
    /// layout (0=left-thigh, 1=left-foot, 2=right-thigh, 3=right-foot):
    /// the right leg's groups mirror the left leg's, anti-phase (half a
    /// gait cycle apart) -- see `mirror_of`'s doc for why both pieces
    /// (mirroring AND the phase offset) are needed together.
    pub fn with_biped_symmetry(mut self) -> Self {
        assert_eq!(self.n_groups, 4, "biped symmetry needs exactly 4 groups");
        self.mirror_of = vec![None, None, Some(0), Some(1)];
        self.phase_offset = vec![0.0, 0.0, std::f32::consts::PI, std::f32::consts::PI];
        self
    }

    fn pre_activation(&self, cfg: &DiffConfig, t: usize, g: usize) -> f32 {
        let src = self.mirror_of[g].unwrap_or(g);
        let time = t as f32 * cfg.dt;
        let extra_phase = self.phase_offset[g];
        let mut pre = self.bias[src];
        for j in 0..self.n_waves {
            let phase = 2.0 * std::f32::consts::PI * j as f32 / self.n_waves as f32;
            pre += self.weights[src * self.n_waves + j]
                * (cfg.omega * time + phase + extra_phase).sin();
        }
        pre
    }

    /// Signed activation in (-1, 1) for group `g` at substep `t`.
    pub fn activation(&self, cfg: &DiffConfig, t: usize, g: usize) -> f32 {
        self.pre_activation(cfg, t, g).tanh()
    }

    /// All groups' activations at substep `t`.
    pub fn activations(&self, cfg: &DiffConfig, t: usize) -> Vec<f32> {
        (0..self.n_groups)
            .map(|g| self.activation(cfg, t, g))
            .collect()
    }
}

/// Closed-loop state-feedback controller -- ChainQueen's real `walker_2d.py`
/// design (verified against the real source, `demos/walker_2d.py`): each
/// muscle group's mean position (relative to the body's own centroid, for
/// translation invariance) and mean velocity feed ONE shared linear layer +
/// tanh, producing all groups' activations together (so one group's muscle
/// can depend on ANY group's sensed state, not just a private clock phase).
///
/// Built after `SinusoidController` (open-loop, time-driven) repeatedly
/// collapsed into degenerate gaits -- frozen, one-legged hop, monotonic
/// tip-over -- across a 12-seed search at its best-found hyperparameters,
/// even with bilateral symmetry and anti-phase coupling. That's real
/// evidence the missing piece isn't more tuning: an open-loop clock can't
/// sense and correct for what the body is actually doing; a feedback
/// controller can.
///
/// Per-group feature layout: `[rel_x, rel_y, vel_x, vel_y]`, concatenated
/// group-major -- `feature_len() = n_groups * 4`.
#[derive(Clone)]
pub struct FeedbackController {
    /// Row-major `[output_group][input_feature]`.
    pub weights: Vec<f32>,
    pub bias: Vec<f32>,
    pub n_groups: usize,
}

impl FeedbackController {
    pub fn feature_len(n_groups: usize) -> usize {
        n_groups * 4
    }

    /// Deterministic pseudo-random init, same hash family as
    /// `SinusoidController::seeded_with` (small magnitude, reproducible).
    pub fn seeded_with(n_groups: usize, seed: u32) -> Self {
        let flen = Self::feature_len(n_groups);
        let rand = |i: usize| -> f32 {
            let x = (i as u32)
                .wrapping_mul(2_654_435_761)
                .wrapping_add(seed.wrapping_mul(40_503).wrapping_add(1));
            let x = x ^ (x >> 15);
            let x = x.wrapping_mul(0x27d4_eb2d);
            let x = x ^ (x >> 15);
            ((x % 2000) as f32 / 1000.0 - 1.0) * 0.01
        };
        Self {
            weights: (0..n_groups * flen).map(rand).collect(),
            bias: vec![0.0; n_groups],
            n_groups,
        }
    }

    /// Per-group mean position (relative to the body centroid) and mean
    /// velocity, flattened group-major. Also returns each group's particle
    /// count (needed by the backward pass to distribute gradient back to
    /// individual particles) and the total particle count (for the
    /// centroid's own gradient).
    pub(super) fn features(plan: &BodyPlan, state: &DiffState) -> (Vec<f32>, Vec<usize>) {
        let n_groups = plan.n_groups;
        let mut sum_x = vec![Vec2::ZERO; n_groups];
        let mut sum_v = vec![Vec2::ZERO; n_groups];
        let mut count = vec![0usize; n_groups];
        let mut centroid = Vec2::ZERO;
        for (i, group) in plan.group.iter().enumerate() {
            centroid += state.x[i];
            if let Some(g) = *group {
                sum_x[g] += state.x[i];
                sum_v[g] += state.v[i];
                count[g] += 1;
            }
        }
        centroid /= plan.positions.len() as f32;

        let mut feat = vec![0.0f32; n_groups * 4];
        for g in 0..n_groups {
            let n = count[g].max(1) as f32;
            let rel = sum_x[g] / n - centroid;
            let vel = sum_v[g] / n;
            feat[g * 4] = rel.x;
            feat[g * 4 + 1] = rel.y;
            feat[g * 4 + 2] = vel.x;
            feat[g * 4 + 3] = vel.y;
        }
        (feat, count)
    }

    pub(super) fn activations_from_features(&self, feat: &[f32]) -> Vec<f32> {
        let flen = feat.len();
        (0..self.n_groups)
            .map(|g| {
                let mut pre = self.bias[g];
                for (k, &fk) in feat.iter().enumerate() {
                    pre += self.weights[g * flen + k] * fk;
                }
                pre.tanh()
            })
            .collect()
    }

    /// All groups' activations given the body's CURRENT state (read at the
    /// start of the substep whose stress they'll drive).
    pub fn activations(&self, plan: &BodyPlan, state: &DiffState) -> Vec<f32> {
        let (feat, _) = Self::features(plan, state);
        self.activations_from_features(&feat)
    }

    /// Adjoint of the whole feature-extraction + linear + tanh pipeline.
    /// Given `g_act` (gradient flowing back from each group's activation,
    /// already summed with whatever downstream physics/penalty terms
    /// contribute to it -- same role as `SinusoidController`'s `g_act` in
    /// its own tanh chain), returns the controller parameter gradients
    /// AND, critically, each PARTICLE's gradient contribution for having
    /// been read as an input to this controller -- these must be ADDED to
    /// the position/velocity gradients already flowing from the physics
    /// chain for this same substep's state, not treated as a separate path.
    ///
    /// Derivation, in order:
    /// 1. `d_pre[g] = g_act[g] * (1 - act[g]^2)` (tanh derivative, same as
    ///    `SinusoidController`).
    /// 2. Linear layer adjoint (standard `Y = W*x + b`):
    ///    `d_weights[g,k] = d_pre[g] * feat[k]`, `d_bias[g] = d_pre[g]`,
    ///    `d_feat[k] = sum_g W[g,k] * d_pre[g]` (`W^T * d_pre`).
    /// 3. Unpack `d_feat` per group into `(d_rel, d_vel)`:
    ///    - `d_vel` distributes evenly to every particle in that group:
    ///      `d(v[i]) += d_vel / count[g]`.
    ///    - `rel = mean_x[g] - centroid` is a difference, so its adjoint
    ///      splits two ways: `d(mean_x[g]) += d_rel` (direct term,
    ///      distributed evenly to the group's own particles) AND
    ///      `d(centroid) -= d_rel`, accumulated across EVERY group (since
    ///      centroid feeds every group's `rel` term) then distributed
    ///      EVENLY TO EVERY PARTICLE IN THE BODY (not just one group's --
    ///      centroid is a mean over all particles, passive torso included).
    pub(super) fn backward(
        &self,
        plan: &BodyPlan,
        feat: &[f32],
        count: &[usize],
        g_act: &[f32],
    ) -> (Vec<f32>, Vec<f32>, Vec<Vec2>, Vec<Vec2>) {
        let flen = feat.len();
        let n_particles = plan.positions.len();
        let mut g_weights = vec![0.0f32; self.weights.len()];
        let mut g_bias = vec![0.0f32; self.n_groups];
        let mut g_feat = vec![0.0f32; flen];

        for g in 0..self.n_groups {
            let act = self.activations_from_features(feat)[g];
            let d_pre = g_act[g] * (1.0 - act * act);
            for (k, &fk) in feat.iter().enumerate() {
                g_weights[g * flen + k] += d_pre * fk;
                g_feat[k] += self.weights[g * flen + k] * d_pre;
            }
            g_bias[g] += d_pre;
        }

        let mut g_x = vec![Vec2::ZERO; n_particles];
        let mut g_v = vec![Vec2::ZERO; n_particles];
        let mut g_centroid = Vec2::ZERO;

        for g in 0..self.n_groups {
            let n = count[g].max(1) as f32;
            let g_rel = Vec2::new(g_feat[g * 4], g_feat[g * 4 + 1]);
            let g_vel = Vec2::new(g_feat[g * 4 + 2], g_feat[g * 4 + 3]);
            g_centroid -= g_rel;
            for (i, group) in plan.group.iter().enumerate() {
                if *group == Some(g) {
                    g_x[i] += g_rel / n;
                    g_v[i] += g_vel / n;
                }
            }
        }

        let per_particle_centroid = g_centroid / n_particles as f32;
        for gx in g_x.iter_mut() {
            *gx += per_particle_centroid;
        }

        (g_weights, g_bias, g_x, g_v)
    }
}

/// Per-particle dynamic state of the mini-sim.
#[derive(Clone)]
pub struct DiffState {
    pub x: Vec<Vec2>,
    pub v: Vec<Vec2>,
    pub c: Vec<Mat2>,
    pub f: Vec<Mat2>,
}

impl DiffState {
    pub fn rest(plan: &BodyPlan) -> Self {
        let n = plan.positions.len();
        Self {
            x: plan.positions.clone(),
            v: vec![Vec2::ZERO; n],
            c: vec![Mat2::ZERO; n],
            f: vec![Mat2::IDENTITY; n],
        }
    }

    pub fn mean_x(&self) -> f32 {
        self.x.iter().map(|p| p.x).sum::<f32>() / self.x.len() as f32
    }
}

/// What one forward substep records for its backward pass: the sticky-floor
/// branch decisions (everything else is recomputed from the stored states).
pub struct StepRecord {
    pub(super) stuck: BTreeSet<(i32, i32)>,
}

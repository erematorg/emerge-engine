use glam::Vec2;

// ── Body plan ─────────────────────────────────────────────────────────────────

/// Rest layout of a trainable body: particle positions, which muscle group
/// (if any) each particle belongs to, and each group's own fiber direction.
///
/// `fiber_dir` moved here from a single global config value (`DiffConfig`
/// used to carry one shared `Vec2::Y` for the whole body) after a real
/// diagnostic: the walker's trained gait measurably bounced rather than
/// walked (vertical velocity^2 1.76x horizontal, mean-height swing 0.94
/// grid units over a ~1.5-unit-tall body) -- because EVERY group could only
/// push straight up/down, net horizontal drift could only emerge indirectly
/// through the sticky floor's timing, which is inherently a pogo motion,
/// not a step. Cross-checked against EvoGym's real, published, walking
/// voxel robots (Bhatia et al. 2021, source in `evogym/utils.py`): their
/// voxels come in two actuator types, `H_ACT` (horizontal) and `V_ACT`
/// (vertical) -- real walkers mix both, using vertical actuators for
/// stance/lift and horizontal actuators for push-off (the actual Newton's-
/// third-law mechanism real legged locomotion uses: push the ground
/// backward, the ground pushes the body forward). `signed_active_stress`
/// already took `fiber_dir` as a parameter, so this is a real fix, not new
/// derivation -- no new adjoint math, just plumbing a per-group value
/// through where a global constant was hardcoded before.
pub struct BodyPlan {
    pub positions: Vec<Vec2>,
    /// Muscle group per particle; `None` = passive tissue (torso).
    pub group: Vec<Option<usize>>,
    pub n_groups: usize,
    /// Fiber direction per muscle group (indexed by group id).
    pub fiber_dir: Vec<Vec2>,
}

impl BodyPlan {
    /// DiffTaichi-`robot()`-style walker: a passive torso slab with four
    /// actuated legs hanging under its ends, one muscle group per leg, ALL
    /// vertical fiber -- kept for comparison against `biped` (the reference
    /// walker's own vertical-only convention, real but measurably bouncy).
    /// `origin` is the lower-left corner of the *legs*, in grid coords;
    /// `spacing` the particle spacing.
    pub fn walker(origin: Vec2, spacing: f32) -> Self {
        let mut positions = Vec::new();
        let mut group = Vec::new();

        // Four legs, 2 columns x 3 rows each, at columns {0-1, 2-3, 8-9, 10-11}.
        let leg_cols: [(usize, usize); 4] = [(0, 0), (2, 1), (8, 2), (10, 3)];
        for (col0, g) in leg_cols {
            for c in 0..2 {
                for r in 0..3 {
                    positions.push(origin + Vec2::new((col0 + c) as f32, r as f32) * spacing);
                    group.push(Some(g));
                }
            }
        }

        // Torso: 12 columns x 3 rows sitting on top of the legs, passive.
        for c in 0..12 {
            for r in 0..3 {
                positions.push(origin + Vec2::new(c as f32, (3 + r) as f32) * spacing);
                group.push(None);
            }
        }

        Self {
            positions,
            group,
            n_groups: 4,
            fiber_dir: vec![Vec2::Y; 4],
        }
    }

    /// Two-legged biped, each leg split into a THIGH (upper, vertical fiber
    /// -- lift/stance) and a FOOT (lower, horizontal fiber -- push-off),
    /// mirroring EvoGym's real V_ACT/H_ACT mix. 4 muscle groups total:
    /// left-thigh, left-foot, right-thigh, right-foot. `origin` is the
    /// lower-left corner of the feet.
    pub fn biped(origin: Vec2, spacing: f32) -> Self {
        let mut positions = Vec::new();
        let mut group = Vec::new();

        const LEFT_THIGH: usize = 0;
        const LEFT_FOOT: usize = 1;
        const RIGHT_THIGH: usize = 2;
        const RIGHT_FOOT: usize = 3;

        // Each leg: 2 columns wide. Foot = bottom 2 rows (horizontal fiber),
        // thigh = next 3 rows up (vertical fiber). Legs at columns {0-1} and
        // {6-7}, leaving a gap for a natural stride stance.
        let legs: [(usize, usize, usize); 2] =
            [(0, LEFT_FOOT, LEFT_THIGH), (6, RIGHT_FOOT, RIGHT_THIGH)];
        for (col0, foot_g, thigh_g) in legs {
            for c in 0..2 {
                for r in 0..2 {
                    positions.push(origin + Vec2::new((col0 + c) as f32, r as f32) * spacing);
                    group.push(Some(foot_g));
                }
                for r in 0..3 {
                    positions.push(origin + Vec2::new((col0 + c) as f32, (2 + r) as f32) * spacing);
                    group.push(Some(thigh_g));
                }
            }
        }

        // Torso: spans both legs, sitting on top, passive.
        for c in 0..8 {
            for r in 0..3 {
                positions.push(origin + Vec2::new(c as f32, (5 + r) as f32) * spacing);
                group.push(None);
            }
        }

        Self {
            positions,
            group,
            n_groups: 4,
            fiber_dir: vec![
                Vec2::Y, // left thigh: vertical (stance/lift)
                Vec2::X, // left foot: horizontal (push-off)
                Vec2::Y, // right thigh: vertical
                Vec2::X, // right foot: horizontal
            ],
        }
    }
}

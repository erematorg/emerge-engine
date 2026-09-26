#[derive(Debug, Clone, Copy)]
pub struct FixedStepConfig {
    pub dt: f32,
    pub simulation_speed: f32,
    pub max_substeps_per_frame: usize,
    pub max_frame_delta: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct FixedStepController {
    config: FixedStepConfig,
    accumulator: f32,
}

impl FixedStepController {
    /// Standard interactive stepper -- `hz` solver steps per real second, capped at 64/frame.
    ///
    /// Equivalent to `FixedStepController::new(FixedStepConfig { dt, simulation_speed: hz * dt,
    /// max_substeps_per_frame: 64, max_frame_delta: 1.0 / 15.0 })`.
    pub fn standard(dt: f32, hz: f32) -> Self {
        Self::new(FixedStepConfig {
            dt,
            simulation_speed: hz * dt,
            max_substeps_per_frame: 64,
            max_frame_delta: 1.0 / 15.0,
        })
    }

    pub fn new(config: FixedStepConfig) -> Self {
        assert!(config.dt > 0.0, "dt must be positive");
        assert!(
            config.simulation_speed >= 0.0,
            "simulation_speed must be non-negative"
        );
        assert!(
            config.max_substeps_per_frame > 0,
            "max_substeps_per_frame must be > 0"
        );
        assert!(
            config.max_frame_delta > 0.0,
            "max_frame_delta must be positive"
        );

        Self {
            config,
            accumulator: 0.0,
        }
    }

    pub fn set_simulation_speed(&mut self, speed: f32) {
        assert!(speed >= 0.0, "simulation_speed must be non-negative");
        self.config.simulation_speed = speed;
    }

    pub const fn dt(&self) -> f32 {
        self.config.dt
    }
    /// Real, additive accessor for the leftover fractional step -- how far
    /// (as a `[0,1)` fraction of one `dt`) real elapsed time has advanced
    /// PAST the last completed physics step. A renderer can use this to
    /// interpolate between the previous and current physics state
    /// (`x_render = lerp(x_prev, x_now, alpha)`, the standard "Fix Your
    /// Timestep" render-interpolation pattern -- Gaffer 2004) so on-screen
    /// motion stays visually smooth even when the real achievable physics-
    /// step cadence itself varies frame to frame, without changing any
    /// physics value. Zero behavior change for every existing caller that
    /// doesn't read this (`steps_for_frame`'s own math is untouched).
    pub fn interpolation_alpha(&self) -> f32 {
        (self.accumulator / self.config.dt).clamp(0.0, 1.0)
    }
    pub const fn simulation_speed(&self) -> f32 {
        self.config.simulation_speed
    }
    /// Reset the time accumulator -- call on save-load or pause-resume to prevent stutter.
    pub const fn reset(&mut self) {
        self.accumulator = 0.0;
    }

    pub fn steps_for_frame(&mut self, frame_delta_seconds: f32) -> usize {
        let clamped_delta = frame_delta_seconds.min(self.config.max_frame_delta);
        self.accumulator += clamped_delta * self.config.simulation_speed;

        let raw_steps = (self.accumulator / self.config.dt).floor() as usize;
        let steps = raw_steps.min(self.config.max_substeps_per_frame);
        self.accumulator -= steps as f32 * self.config.dt;
        steps
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interpolation_alpha_tracks_leftover_fraction() {
        let mut stepper = FixedStepController::new(FixedStepConfig {
            dt: 0.1,
            simulation_speed: 1.0, // 1 real second = 1 simulated second
            max_substeps_per_frame: 64,
            max_frame_delta: 1.0,
        });
        // Half a dt's worth of real time -- no step taken yet (raw_steps=0),
        // so the whole 0.05s should sit in the accumulator as alpha=0.5.
        assert_eq!(stepper.steps_for_frame(0.05), 0);
        assert!((stepper.interpolation_alpha() - 0.5).abs() < 1e-6);

        // Another 0.05s completes exactly one dt -- one step taken, leftover
        // fraction drops back to (near) zero.
        assert_eq!(stepper.steps_for_frame(0.05), 1);
        assert!(stepper.interpolation_alpha() < 1e-5);
    }

    #[test]
    fn interpolation_alpha_stays_in_zero_one_range_even_capped() {
        let mut stepper = FixedStepController::new(FixedStepConfig {
            dt: 0.01,
            simulation_speed: 1.0,
            max_substeps_per_frame: 2, // deliberately tiny cap
            max_frame_delta: 1.0,
        });
        // Real elapsed time worth far more than the cap allows -- the
        // accumulator keeps the UNCONSUMED backlog (by design, see
        // `steps_for_frame`'s own doc), so alpha must stay clamped to
        // [0,1) rather than reporting a nonsensical multi-step overrun.
        let steps = stepper.steps_for_frame(1.0);
        assert_eq!(steps, 2);
        let alpha = stepper.interpolation_alpha();
        assert!((0.0..=1.0).contains(&alpha), "alpha={alpha} out of range");
    }
}

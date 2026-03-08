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

    pub fn steps_for_frame(&mut self, frame_delta_seconds: f32) -> usize {
        let clamped_delta = frame_delta_seconds.min(self.config.max_frame_delta);
        self.accumulator += clamped_delta * self.config.simulation_speed;

        let raw_steps = (self.accumulator / self.config.dt).floor() as usize;
        let steps = raw_steps.min(self.config.max_substeps_per_frame);
        self.accumulator -= steps as f32 * self.config.dt;
        steps
    }
}

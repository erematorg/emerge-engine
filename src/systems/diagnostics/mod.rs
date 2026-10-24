pub mod logger;
pub mod per_material;
pub mod plugin;
pub mod position_resolution;
pub mod rules;
pub mod scene_map;
pub mod snapshot;

pub use logger::FrameLogger;
pub use per_material::{
    MaterialStats, log_frame_full, log_frame_gpu, per_material_stats, per_material_stats_of,
};
pub use plugin::{
    ActivationStatsPlugin, DiagnosticsFrame, DiagnosticsPlugin, DiagnosticsRegistry,
    MaterialCountPlugin, RollingPlugin, ThermalStatsPlugin,
};
pub use position_resolution::{PositionResolution, PositionResolutionPlugin, position_resolution};
pub use rules::{StabilityStatus, StabilityThresholds, evaluate_stability};
pub use scene_map::{HEAT_BANDS, OCCUPANCY_BANDS, scene_map};
pub use snapshot::{
    RodSnapshot, SiSnapshot, SimSnapshot, StepTiming, collect_rod_snapshot, collect_snapshot,
    collect_snapshot_particles_only,
};

/// Reads the research-diagnostic switch `name` (an `EMERGE_*` environment
/// variable) from the environment. Always `None` in a build without the
/// `research-diagnostics` feature (unit tests aside), so a default build
/// never reads the environment and every trace it guards is dead code.
#[inline]
pub(crate) fn research_switch(name: &str) -> Option<String> {
    if cfg!(any(test, feature = "research-diagnostics")) {
        std::env::var(name).ok()
    } else {
        None
    }
}

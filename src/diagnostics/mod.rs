pub mod logger;
pub mod per_material;
pub mod plugin;
pub mod rules;
pub mod snapshot;

pub use logger::FrameLogger;
pub use per_material::{
    MaterialStats, log_frame, log_frame_full, log_frame_gpu, per_material_stats,
    per_material_stats_of,
};
pub use plugin::{
    ActivationStatsPlugin, DiagnosticsFrame, DiagnosticsPlugin, DiagnosticsRegistry,
    MaterialCountPlugin, RollingPlugin, ThermalStatsPlugin,
};
pub use rules::{MpmHealthStatus, MpmHealthThresholds, evaluate_mpm_health};
pub use snapshot::{MpmSnapshot, collect_mpm_snapshot};

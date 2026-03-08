use crate::diagnostics::rules::{MpmHealthThresholds, evaluate_mpm_health};
use crate::diagnostics::snapshot::MpmSnapshot;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MpmStateEvent {
    Recovered,
    Degraded,
}

#[derive(Debug, Clone)]
pub struct MpmReportingPolicy {
    pub thresholds: MpmHealthThresholds,
    pub report_interval_secs: f32,
    pub healthy_heartbeat_secs: f32,
    pub issue_cooldown_secs: f32,
    pub log_healthy: bool,
}

impl Default for MpmReportingPolicy {
    fn default() -> Self {
        Self {
            thresholds: MpmHealthThresholds::default(),
            report_interval_secs: 1.0,
            healthy_heartbeat_secs: 5.0,
            issue_cooldown_secs: 3.0,
            log_healthy: false,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct MpmReportingState {
    pub elapsed: f32,
    pub healthy_elapsed: f32,
    pub unhealthy_elapsed: f32,
    pub last_healthy: Option<bool>,
    pub last_issue_mask: Option<u16>,
    pub healthy_report_streak: u32,
}

#[derive(Debug, Clone)]
pub struct MpmReportBundle {
    pub event_line: Option<String>,
    pub report_line: String,
}

pub fn format_mpm_report_line(
    snapshot: &MpmSnapshot,
    status: crate::diagnostics::rules::MpmHealthStatus,
    healthy_report_streak: u32,
) -> String {
    let healthy = status.healthy();
    let issues = if healthy {
        String::new()
    } else {
        format!(
            " issues=[{}] invalid_physical={} non_finite={} out_of_bounds={} recommended_dt<={:.4}",
            status.issue_labels().join(","),
            snapshot.invalid_physical_particle_values,
            snapshot.non_finite_particle_values + snapshot.non_finite_grid_values,
            snapshot.out_of_bounds_particles,
            snapshot.recommended_max_dt_from_velocity_cfl
        )
    };
    format!(
        "[diag] frame={} dt={:.4}/{:.4} particles={}/{} active_cells={} particles_per_cell={:.1} mix_cells={:.3} mix_particles={:.3} cfl={:.3} mass_err={:.1e} momentum_err={:.2e} vmax_p={:.3} vmax_g={:.3} j=[{:.3},{:.3}] jp=[{:.3},{:.3}] h={:.3} healthy={} streak={}{}",
        snapshot.frame_index,
        snapshot.effective_dt,
        snapshot.configured_dt,
        snapshot.valid_particle_count,
        snapshot.particle_count,
        snapshot.active_grid_cells,
        snapshot.particles_per_active_cell,
        snapshot.mixed_material_cell_ratio,
        snapshot.mixed_material_particle_ratio,
        snapshot.cfl_number,
        snapshot.relative_mass_error,
        snapshot.relative_momentum_error,
        snapshot.max_particle_speed,
        snapshot.max_grid_speed,
        snapshot.min_deformation_j,
        snapshot.max_deformation_j,
        snapshot.min_plastic_jacobian,
        snapshot.avg_plastic_jacobian,
        snapshot.avg_elastic_hardening,
        healthy,
        healthy_report_streak,
        issues
    )
}

pub fn update_mpm_reporting(
    state: &mut MpmReportingState,
    delta_secs: f32,
    snapshot: &MpmSnapshot,
    policy: &MpmReportingPolicy,
) -> Option<MpmReportBundle> {
    state.elapsed += delta_secs;
    state.healthy_elapsed += delta_secs;
    state.unhealthy_elapsed += delta_secs;
    if state.elapsed < policy.report_interval_secs {
        return None;
    }
    state.elapsed = 0.0;

    let status = evaluate_mpm_health(snapshot, &policy.thresholds);
    let healthy = status.healthy();
    let issue_mask = status.issue_mask();
    let changed = state.last_healthy.is_none_or(|prev| prev != healthy);
    let issue_changed = state.last_issue_mask != Some(issue_mask);
    let healthy_heartbeat = healthy && state.healthy_elapsed >= policy.healthy_heartbeat_secs;
    let unhealthy_cooldown = !healthy && state.unhealthy_elapsed >= policy.issue_cooldown_secs;

    if healthy {
        state.healthy_report_streak = state.healthy_report_streak.saturating_add(1);
    } else {
        state.healthy_report_streak = 0;
    }

    let should_log = if healthy {
        changed || policy.log_healthy || healthy_heartbeat
    } else {
        changed || issue_changed || unhealthy_cooldown
    };
    if !should_log {
        state.last_healthy = Some(healthy);
        state.last_issue_mask = Some(issue_mask);
        return None;
    }

    let event_line = if changed {
        let event = if healthy {
            MpmStateEvent::Recovered
        } else {
            MpmStateEvent::Degraded
        };
        let event_name = match event {
            MpmStateEvent::Recovered => "recovered",
            MpmStateEvent::Degraded => "degraded",
        };
        let issues = if healthy {
            String::new()
        } else {
            format!(" issues=[{}]", status.issue_labels().join(","))
        };
        Some(format!(
            "[diag-event] state={} frame={}{}",
            event_name, snapshot.frame_index, issues
        ))
    } else {
        None
    };

    if healthy {
        state.healthy_elapsed = 0.0;
    } else {
        state.unhealthy_elapsed = 0.0;
    }

    state.last_healthy = Some(healthy);
    state.last_issue_mask = Some(issue_mask);

    Some(MpmReportBundle {
        event_line,
        report_line: format_mpm_report_line(snapshot, status, state.healthy_report_streak),
    })
}

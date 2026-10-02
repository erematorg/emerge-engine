//! Diagnostic probes: measurements and investigations kept for reruns,
//! built as one test binary instead of one per file. Most are
//! `#[ignore]`d and print what they measure; run one with
//! `cargo test --test probes <module>:: -- --ignored --nocapture`.

extern crate emerge_engine as emerge;

mod basic_jellies_gpu_probe;
mod basic_jellies_probe;
mod basic_sand_probe;
mod basic_showcase_probe;
mod basic_snow_gpu_probe;
mod basic_snow_probe;
mod bingham_column_symmetry;
mod bingham_column_volume_loss;
mod bingham_cursor_yield;
mod bingham_deposit_state;
mod bingham_isolated_slump;
mod bingham_slope_yield;
mod bingham_substep_gap;
mod column_convergence;
mod contact_cost;
mod dt_spike_overshoot_diagnostic;
mod evp_volume_rounding;
mod f_rounding_horizon;
mod falling_droplet_pressure_projection_check;
mod fluid_j_rounding;
mod gas_sound_speed;
mod gpu_boundary_recheck;
mod grain_coarse_graining_check;
mod grain_contact_derived_phi_gate;
mod grain_pure_dem_pour_repose_angle;
mod grain_real_substep_count_measurement;
mod grid_coupled_grain_pour_repose_angle;
mod implicit_corotated_real_fps_measurement;
mod implicit_corotated_wiring_diagnostic;
mod implicit_mpm_stage2_corotated_jvp;
mod implicit_mpm_stage2_shared_elastic_branch_check;
mod implicit_mpm_stage3_drucker_prager_multi_particle;
mod implicit_stiffness_vs_scale_isolation;
mod kinematic_reactive_obstacle_water_verify;
mod membrane_gravity_probe;
mod no_compression_drift_horizon;
mod position_resolution_scenes;
mod rod_load_ab;
mod sand_mibf_clean_test;
mod sand_mibf_plus_switch_step;
mod sand_patient_pour_with_post_event_relax;
mod sand_pour_lateral_motion_diagnostic;
mod sand_pour_with_post_event_relax;
mod sand_switch_step_extrapolation;
mod self_weight_shortening_convergence;
mod snow_gpu_gap;
mod stress_view_before_after;
mod thin_layer_volume_drift;
mod trophic_predation_probe;
mod vonmises_clay;
mod vonmises_clay_gpu;
mod wall_contact_regression_check_after_divergence_fix;

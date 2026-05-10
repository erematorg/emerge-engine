use crate::diagnostics::snapshot::MpmSnapshot;

#[derive(Debug, Clone, Copy)]
pub struct MpmHealthThresholds {
    pub max_cfl: f32,
    pub max_relative_mass_error: f32,
    pub max_relative_momentum_error: f32,
    pub min_particle_count: usize,
    pub min_active_grid_cells: usize,
    pub max_particles_per_active_cell: f32,
    pub max_mixed_material_cell_ratio: f32,
    pub max_mixed_material_particle_ratio: f32,
    pub max_out_of_bounds_particles: usize,
    pub max_invalid_physical_particle_values: usize,
    pub max_non_finite_values: usize,
}

impl MpmHealthThresholds {
    pub fn for_spacing(spacing: f32) -> Self {
        Self::for_spacing_with_options(spacing, f32::INFINITY, f32::INFINITY)
    }

    pub fn for_spacing_with_options(
        spacing: f32,
        max_mixed_material_cell_ratio: f32,
        max_mixed_material_particle_ratio: f32,
    ) -> Self {
        let mut thresholds = Self::default();
        let expected_particles_per_cell = (1.0 / spacing.max(1.0e-6)).powi(2);
        // This catches collapse where too many particles numerically concentrate in few cells.
        thresholds.max_particles_per_active_cell = expected_particles_per_cell * 32.0;
        thresholds.max_mixed_material_cell_ratio = max_mixed_material_cell_ratio;
        thresholds.max_mixed_material_particle_ratio = max_mixed_material_particle_ratio;
        thresholds
    }
}

impl Default for MpmHealthThresholds {
    fn default() -> Self {
        Self {
            max_cfl: 1.0,
            max_relative_mass_error: 1.0e-3,
            // Momentum error is noisier near boundaries, so keep this threshold looser.
            max_relative_momentum_error: 2.0,
            min_particle_count: 1,
            min_active_grid_cells: 1,
            max_particles_per_active_cell: f32::INFINITY,
            max_mixed_material_cell_ratio: f32::INFINITY,
            max_mixed_material_particle_ratio: f32::INFINITY,
            max_out_of_bounds_particles: 0,
            max_invalid_physical_particle_values: 0,
            max_non_finite_values: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct MpmHealthStatus {
    pub particle_count_violation: bool,
    pub inactive_grid_violation: bool,
    pub cell_concentration_violation: bool,
    pub mixed_material_violation: bool,
    pub cfl_violation: bool,
    pub mass_drift_violation: bool,
    pub momentum_drift_violation: bool,
    pub out_of_bounds_violation: bool,
    pub invalid_physical_state_violation: bool,
    pub non_finite_violation: bool,
}

impl MpmHealthStatus {
    pub fn healthy(self) -> bool {
        !self.particle_count_violation
            && !self.inactive_grid_violation
            && !self.cell_concentration_violation
            && !self.mixed_material_violation
            && !self.cfl_violation
            && !self.mass_drift_violation
            && !self.momentum_drift_violation
            && !self.out_of_bounds_violation
            && !self.invalid_physical_state_violation
            && !self.non_finite_violation
    }

    pub fn issue_labels(self) -> Vec<&'static str> {
        let mut labels = Vec::new();
        if self.particle_count_violation {
            labels.push("particle_count");
        }
        if self.inactive_grid_violation {
            labels.push("inactive_grid");
        }
        if self.cell_concentration_violation {
            labels.push("cell_concentration");
        }
        if self.mixed_material_violation {
            labels.push("material_mixing");
        }
        if self.cfl_violation {
            labels.push("cfl");
        }
        if self.mass_drift_violation {
            labels.push("mass");
        }
        if self.momentum_drift_violation {
            labels.push("momentum");
        }
        if self.out_of_bounds_violation {
            labels.push("out_of_bounds");
        }
        if self.invalid_physical_state_violation {
            labels.push("physical_state");
        }
        if self.non_finite_violation {
            labels.push("non_finite");
        }
        labels
    }

    pub fn issue_mask(self) -> u16 {
        let mut mask = 0u16;
        if self.particle_count_violation {
            mask |= 1 << 0;
        }
        if self.inactive_grid_violation {
            mask |= 1 << 1;
        }
        if self.cell_concentration_violation {
            mask |= 1 << 2;
        }
        if self.mixed_material_violation {
            mask |= 1 << 3;
        }
        if self.cfl_violation {
            mask |= 1 << 4;
        }
        if self.mass_drift_violation {
            mask |= 1 << 5;
        }
        if self.momentum_drift_violation {
            mask |= 1 << 6;
        }
        if self.out_of_bounds_violation {
            mask |= 1 << 7;
        }
        if self.invalid_physical_state_violation {
            mask |= 1 << 8;
        }
        if self.non_finite_violation {
            mask |= 1 << 9;
        }
        mask
    }
}

pub fn evaluate_mpm_health(
    snapshot: &MpmSnapshot,
    thresholds: &MpmHealthThresholds,
) -> MpmHealthStatus {
    let non_finite_total = snapshot.non_finite_particle_values + snapshot.non_finite_grid_values;

    MpmHealthStatus {
        particle_count_violation: snapshot.particle_count < thresholds.min_particle_count,
        inactive_grid_violation: snapshot.particle_count > 0
            && snapshot.active_grid_cells < thresholds.min_active_grid_cells,
        cell_concentration_violation: snapshot.active_grid_cells > 0
            && snapshot.particles_per_active_cell > thresholds.max_particles_per_active_cell,
        mixed_material_violation: snapshot.mixed_material_cell_ratio
            > thresholds.max_mixed_material_cell_ratio
            || snapshot.mixed_material_particle_ratio
                > thresholds.max_mixed_material_particle_ratio,
        cfl_violation: snapshot.cfl_number > thresholds.max_cfl,
        mass_drift_violation: snapshot.relative_mass_error > thresholds.max_relative_mass_error,
        momentum_drift_violation: snapshot.relative_momentum_error
            > thresholds.max_relative_momentum_error,
        out_of_bounds_violation: snapshot.out_of_bounds_particles
            > thresholds.max_out_of_bounds_particles,
        invalid_physical_state_violation: snapshot.invalid_physical_particle_values
            > thresholds.max_invalid_physical_particle_values,
        non_finite_violation: non_finite_total > thresholds.max_non_finite_values,
    }
}

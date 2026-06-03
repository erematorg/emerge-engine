use glam::Vec2;
use std::collections::HashMap;

use crate::solver::config::SolverConfig;
use crate::{grid::Grid, particle::Particles};

#[derive(Debug, Clone, Copy, Default)]
pub struct MpmSnapshot {
    pub frame_index: u64,
    /// Frame duration as configured: the total simulation time advanced by one `step()` call.
    pub configured_dt: f32,
    /// Duration of the *last substep* within the most recent `step()` call.
    /// Equal to `configured_dt` when adaptive timestepping is off or only one substep ran.
    /// Useful for per-substep CFL diagnostics; not the per-frame dt.
    pub effective_dt: f32,
    pub substeps_last_step: usize,
    pub particle_count: usize,
    pub valid_particle_count: usize,
    pub active_grid_cells: usize,
    pub particles_per_active_cell: f32,
    pub mixed_material_cell_ratio: f32,
    pub mixed_material_particle_ratio: f32,
    pub total_particle_mass: f32,
    pub total_grid_mass: f32,
    pub relative_mass_error: f32,
    pub total_particle_momentum: Vec2,
    pub total_grid_momentum: Vec2,
    pub relative_momentum_error: f32,
    pub max_particle_speed: f32,
    pub max_grid_speed: f32,
    pub cfl_number: f32,
    pub min_deformation_j: f32,
    pub max_deformation_j: f32,
    pub out_of_bounds_particles: usize,
    pub invalid_physical_particle_values: usize,
    pub non_finite_particle_values: usize,
    pub non_finite_grid_values: usize,
    pub recommended_max_dt_from_velocity_cfl: f32,
    /// Average plastic Jacobian (Jp) across all particles. 1.0 = no plastic deformation.
    /// Drops below 1.0 when material compresses plastically (e.g. snow after impact).
    pub avg_plastic_jacobian: f32,
    /// Minimum Jp across all particles. Shows where compression is most severe.
    pub min_plastic_jacobian: f32,
    /// Average elastic hardening multiplier h = exp(ξ*(1−Jp)). 1.0 = no hardening.
    /// Rises above 1.0 when snow is compacted (compressed snow is stiffer).
    pub avg_elastic_hardening: f32,
    /// Active particles (not sleeping). Sleeping particles are excluded from P2G/G2P.
    pub active_count: usize,
    /// Sleeping particles (excluded from this step's physics).
    pub sleeping_count: usize,
    /// Particles whose velocity was clamped to the CFL limit during G2P this step.
    /// Nonzero = CFL was violated; substep budget or material stiffness needs attention.
    pub vel_clamp_count: usize,
    /// Particles whose deformation state was projected back to admissible this step.
    /// Nonzero = explicit integration diverged; check dt, material params, or stiffness.
    pub j_projection_count: usize,
    /// Simulation time (seconds) dropped due to `max_substeps_per_step` cap this step.
    /// Nonzero = simulation running slower than real-time; reduce stiffness or raise cap.
    pub sim_time_dropped: f32,
}

pub fn collect_mpm_snapshot(
    frame_index: u64,
    particles: &Particles,
    grid: &Grid,
    config: &SolverConfig,
    step_dt: f32,
    substeps_last_step: usize,
) -> MpmSnapshot {
    #[derive(Clone, Copy, Debug)]
    struct MaterialCellState {
        first_material_id: u32,
        has_multiple_materials: bool,
        particle_count: usize,
    }

    let mut snapshot = MpmSnapshot {
        frame_index,
        configured_dt: config.dt,
        effective_dt: step_dt,
        substeps_last_step,
        particle_count: particles.len(),
        recommended_max_dt_from_velocity_cfl: f32::INFINITY,
        min_deformation_j: f32::INFINITY,
        max_deformation_j: f32::NEG_INFINITY,
        min_plastic_jacobian: f32::INFINITY,
        avg_plastic_jacobian: 1.0,
        avg_elastic_hardening: 1.0,
        ..Default::default()
    };
    let mut jp_sum = 0.0f32;
    let mut h_sum = 0.0f32;
    let mut material_cells = HashMap::<usize, MaterialCellState>::new();

    let min_bound = config.boundary_thickness.saturating_sub(1) as f32;
    let max_bound = config.grid_res.saturating_sub(config.boundary_thickness) as f32;

    for i in particles.indices() {
        let mass = particles.mass[i];
        let v = particles.v[i];
        let x = particles.x[i];
        let deformation_j = particles.deformation_gradient[i].determinant();
        let jp = particles.plastic_volume_ratio[i];
        let h = particles.hardening_scale[i];
        let mat_id = particles.material_id[i];

        snapshot.total_particle_mass += mass;
        snapshot.total_particle_momentum += mass * v;
        snapshot.max_particle_speed = snapshot.max_particle_speed.max(v.length());

        if deformation_j.is_finite() {
            snapshot.min_deformation_j = snapshot.min_deformation_j.min(deformation_j);
            snapshot.max_deformation_j = snapshot.max_deformation_j.max(deformation_j);
        }

        if x.x < min_bound || x.x > max_bound || x.y < min_bound || x.y > max_bound {
            snapshot.out_of_bounds_particles += 1;
        }

        if let Some(cell_index) = particle_cell_index(x, config.grid_res) {
            let entry = material_cells
                .entry(cell_index)
                .or_insert_with(|| MaterialCellState {
                    first_material_id: mat_id,
                    has_multiple_materials: false,
                    particle_count: 0,
                });
            entry.particle_count += 1;
            if !entry.has_multiple_materials && mat_id != entry.first_material_id {
                entry.has_multiple_materials = true;
            }
        }

        if jp.is_finite() {
            jp_sum += jp;
            snapshot.min_plastic_jacobian = snapshot.min_plastic_jacobian.min(jp);
        }
        if h.is_finite() {
            h_sum += h;
        }

        let non_finite_values = count_non_finite_particle_values(particles, i);
        snapshot.non_finite_particle_values += non_finite_values;
        if non_finite_values == 0 {
            snapshot.valid_particle_count += 1;
        }
        snapshot.invalid_physical_particle_values +=
            count_invalid_particle_values(particles, i, deformation_j);
    }

    if !particles.is_empty() {
        let n = particles.len() as f32;
        snapshot.avg_plastic_jacobian = jp_sum / n;
        snapshot.avg_elastic_hardening = h_sum / n;
        if snapshot.min_plastic_jacobian.is_infinite() {
            snapshot.min_plastic_jacobian = 1.0;
        }
    }

    for cell in grid.active_cells() {
        if cell.mass > 0.0 {
            snapshot.active_grid_cells += 1;
        }
        snapshot.total_grid_mass += cell.mass;
        snapshot.total_grid_momentum += cell.mass * cell.momentum;
        snapshot.max_grid_speed = snapshot.max_grid_speed.max(cell.momentum.length());
        snapshot.non_finite_grid_values +=
            count_non_finite_grid_values(cell.mass, cell.momentum.x, cell.momentum.y);
    }

    if snapshot.active_grid_cells > 0 {
        snapshot.particles_per_active_cell =
            snapshot.particle_count as f32 / snapshot.active_grid_cells as f32;
    }

    if !material_cells.is_empty() {
        let mut mixed_cell_count = 0usize;
        let mut particles_in_mixed_cells = 0usize;
        for cell in material_cells.values() {
            if cell.has_multiple_materials {
                mixed_cell_count += 1;
                particles_in_mixed_cells += cell.particle_count;
            }
        }
        snapshot.mixed_material_cell_ratio = mixed_cell_count as f32 / material_cells.len() as f32;
        if snapshot.particle_count > 0 {
            snapshot.mixed_material_particle_ratio =
                particles_in_mixed_cells as f32 / snapshot.particle_count as f32;
        }
    }

    if snapshot.total_particle_mass > f32::EPSILON {
        snapshot.relative_mass_error = (snapshot.total_grid_mass - snapshot.total_particle_mass)
            .abs()
            / snapshot.total_particle_mass;
    }

    let momentum_scale = snapshot
        .total_particle_momentum
        .length()
        .max(snapshot.total_grid_momentum.length())
        .max(snapshot.total_particle_mass * 1.0e-3);
    snapshot.relative_momentum_error =
        (snapshot.total_grid_momentum - snapshot.total_particle_momentum).length() / momentum_scale;

    let cell_size = config.grid_cell_size.max(f32::EPSILON);
    snapshot.cfl_number = snapshot.max_particle_speed * step_dt / cell_size;
    if snapshot.max_particle_speed > f32::EPSILON {
        snapshot.recommended_max_dt_from_velocity_cfl = cell_size / snapshot.max_particle_speed;
    }

    snapshot
}

fn particle_cell_index(position: Vec2, grid_res: usize) -> Option<usize> {
    if !position.is_finite() {
        return None;
    }
    let ix = position.x.floor() as i32;
    let iy = position.y.floor() as i32;
    if ix < 0 || iy < 0 {
        return None;
    }
    let ux = ix as usize;
    let uy = iy as usize;
    if ux >= grid_res || uy >= grid_res {
        return None;
    }
    Some(ux * grid_res + uy)
}

fn count_non_finite_particle_values(particles: &Particles, i: usize) -> usize {
    let c = particles.velocity_gradient[i];
    let f = particles.deformation_gradient[i];
    let values = [
        particles.x[i].x,
        particles.x[i].y,
        particles.v[i].x,
        particles.v[i].y,
        c.x_axis.x,
        c.x_axis.y,
        c.y_axis.x,
        c.y_axis.y,
        f.x_axis.x,
        f.x_axis.y,
        f.y_axis.x,
        f.y_axis.y,
        particles.mass[i],
        particles.initial_volume[i],
        particles.volume[i],
        particles.density[i],
        particles.plastic_volume_ratio[i],
    ];
    values.iter().filter(|v| !v.is_finite()).count()
}

fn count_non_finite_grid_values(mass: f32, vx: f32, vy: f32) -> usize {
    [mass, vx, vy]
        .iter()
        .filter(|value| !value.is_finite())
        .count()
}

fn count_invalid_particle_values(particles: &Particles, i: usize, deformation_j: f32) -> usize {
    let mut invalid = 0usize;
    if particles.mass[i] <= 0.0 {
        invalid += 1;
    }
    if particles.volume[i] <= 0.0 {
        invalid += 1;
    }
    if particles.initial_volume[i] <= 0.0 {
        invalid += 1;
    }
    if particles.density[i] <= 0.0 {
        invalid += 1;
    }
    if deformation_j <= 0.0 {
        invalid += 1;
    }
    if particles.plastic_volume_ratio[i] <= 0.0 {
        invalid += 1;
    }
    invalid
}

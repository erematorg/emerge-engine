use glam::Vec2;
use std::collections::HashMap;

use crate::solver::config::SolverConfig;
use crate::state::{grid::Grid, particle::Particle};

#[derive(Debug, Clone, Copy, Default)]
pub struct MpmSnapshot {
    pub frame_index: u64,
    pub configured_dt: f32,
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
}

pub fn collect_mpm_snapshot(
    frame_index: u64,
    particles: &[Particle],
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

    for particle in particles {
        snapshot.total_particle_mass += particle.mass;
        snapshot.total_particle_momentum += particle.mass * particle.v;
        snapshot.max_particle_speed = snapshot.max_particle_speed.max(particle.v.length());

        let deformation_j = particle.deformation_gradient.determinant();
        if deformation_j.is_finite() {
            snapshot.min_deformation_j = snapshot.min_deformation_j.min(deformation_j);
            snapshot.max_deformation_j = snapshot.max_deformation_j.max(deformation_j);
        }

        if particle.x.x < min_bound
            || particle.x.x > max_bound
            || particle.x.y < min_bound
            || particle.x.y > max_bound
        {
            snapshot.out_of_bounds_particles += 1;
        }

        if let Some(cell_index) = particle_cell_index(particle.x, config.grid_res) {
            let entry = material_cells
                .entry(cell_index)
                .or_insert_with(|| MaterialCellState {
                    first_material_id: particle.material_id,
                    has_multiple_materials: false,
                    particle_count: 0,
                });
            entry.particle_count += 1;
            if !entry.has_multiple_materials && particle.material_id != entry.first_material_id {
                entry.has_multiple_materials = true;
            }
        }

        if particle.plastic_jacobian.is_finite() {
            jp_sum += particle.plastic_jacobian;
            snapshot.min_plastic_jacobian = snapshot.min_plastic_jacobian.min(particle.plastic_jacobian);
        }
        if particle.elastic_hardening.is_finite() {
            h_sum += particle.elastic_hardening;
        }

        let non_finite_values = count_non_finite_particle_values(particle);
        snapshot.non_finite_particle_values += non_finite_values;
        if non_finite_values == 0 {
            snapshot.valid_particle_count += 1;
        }
        snapshot.invalid_physical_particle_values +=
            count_invalid_particle_values(particle, deformation_j);
    }

    if !particles.is_empty() {
        let n = particles.len() as f32;
        snapshot.avg_plastic_jacobian = jp_sum / n;
        snapshot.avg_elastic_hardening = h_sum / n;
        if snapshot.min_plastic_jacobian.is_infinite() {
            snapshot.min_plastic_jacobian = 1.0;
        }
    }

    for cell in grid.cells() {
        if cell.mass > 0.0 {
            snapshot.active_grid_cells += 1;
        }
        snapshot.total_grid_mass += cell.mass;
        snapshot.total_grid_momentum += cell.mass * cell.momentum;
        snapshot.max_grid_speed = snapshot.max_grid_speed.max(cell.momentum.length());
        snapshot.non_finite_grid_values += count_non_finite_grid_values(cell.mass, cell.momentum.x, cell.momentum.y);
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
        snapshot.relative_mass_error =
            (snapshot.total_grid_mass - snapshot.total_particle_mass).abs() / snapshot.total_particle_mass;
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

fn count_non_finite_particle_values(particle: &Particle) -> usize {
    let values = [
        particle.x.x,
        particle.x.y,
        particle.v.x,
        particle.v.y,
        particle.c.x_axis.x,
        particle.c.x_axis.y,
        particle.c.y_axis.x,
        particle.c.y_axis.y,
        particle.deformation_gradient.x_axis.x,
        particle.deformation_gradient.x_axis.y,
        particle.deformation_gradient.y_axis.x,
        particle.deformation_gradient.y_axis.y,
        particle.mass,
        particle.initial_volume,
        particle.volume,
        particle.density,
        particle.plastic_jacobian,
    ];
    values.iter().filter(|value| !value.is_finite()).count()
}

fn count_non_finite_grid_values(mass: f32, vx: f32, vy: f32) -> usize {
    [mass, vx, vy]
        .iter()
        .filter(|value| !value.is_finite())
        .count()
}

fn count_invalid_particle_values(particle: &Particle, deformation_j: f32) -> usize {
    let mut invalid = 0usize;
    if particle.mass <= 0.0 {
        invalid += 1;
    }
    if particle.volume <= 0.0 {
        invalid += 1;
    }
    if particle.initial_volume <= 0.0 {
        invalid += 1;
    }
    if particle.density <= 0.0 {
        invalid += 1;
    }
    if deformation_j <= 0.0 {
        invalid += 1;
    }
    if particle.plastic_jacobian <= 0.0 {
        invalid += 1;
    }
    invalid
}

//! TEMPORARY accepted-substep impulse ledger for the structural wall-bounce
//! investigation. Opt-in only; remove once the boundary mechanism is settled.

use glam::{DVec2, Vec2};

use super::Simulation;
use crate::grid::{Grid, flat_index};
use crate::solver::config::KERNEL_D_INVERSE;
use crate::transfer::GridNodeP2GComponents;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoundaryImpulseExperiment {
    Baseline,
    TractionAwareRelease,
    DeepQuadraticBand,
    TractionAwareDeepBand,
}

impl BoundaryImpulseExperiment {
    pub(crate) fn from_env() -> Option<Self> {
        let value = std::env::var("EMERGE_DIAG_BOUNDARY_IMPULSE").ok()?;
        match value.to_ascii_lowercase().as_str() {
            "baseline" | "a" => Some(Self::Baseline),
            "traction" | "b" => Some(Self::TractionAwareRelease),
            "deep" | "c" => Some(Self::DeepQuadraticBand),
            "both" | "d" => Some(Self::TractionAwareDeepBand),
            _ => None,
        }
    }

    const fn traction_aware(self) -> bool {
        matches!(
            self,
            Self::TractionAwareRelease | Self::TractionAwareDeepBand
        )
    }

    const fn deep_band(self) -> bool {
        matches!(self, Self::DeepQuadraticBand | Self::TractionAwareDeepBand)
    }
}

#[derive(Clone, Debug, Default)]
pub struct BoundaryNodeImpulseLedger {
    pub cell_index: usize,
    pub x: usize,
    pub y: usize,
    pub mass: f32,
    pub translation_momentum: Vec2,
    pub affine_momentum: Vec2,
    pub stress_momentum: Vec2,
    pub velocity_before_wall: Vec2,
    pub velocity_after_wall: Vec2,
    pub gravity_impulse: Vec2,
    pub wall_impulse: Vec2,
    /// Lower-wall normal force per unit out-of-plane thickness and per nodal
    /// face width. Negative means compression into the wall.
    pub estimated_normal_traction: f32,
    pub velocity_condition_active: bool,
    pub compressive_traction_active: bool,
    pub released_while_compressive: bool,
}

#[derive(Clone, Debug, Default)]
pub struct AcceptedBoundaryImpulseLedger {
    pub dt: f32,
    pub particle_momentum_start: Vec2,
    pub grid_momentum_after_p2g: Vec2,
    pub gravity_impulse: Vec2,
    pub grid_momentum_before_wall: Vec2,
    pub grid_momentum_after_wall: Vec2,
    pub wall_impulse: Vec2,
    pub contact_impulse: Vec2,
    pub other_grid_impulse: Vec2,
    pub particle_momentum_end: Vec2,
    pub residual: Vec2,
    pub particle_momentum_start_f64: DVec2,
    pub grid_momentum_after_p2g_f64: DVec2,
    pub gravity_impulse_f64: DVec2,
    pub grid_momentum_before_wall_f64: DVec2,
    pub grid_momentum_after_wall_f64: DVec2,
    pub wall_impulse_f64: DVec2,
    pub contact_impulse_f64: DVec2,
    pub other_grid_impulse_f64: DVec2,
    pub grid_momentum_before_g2p_f64: DVec2,
    pub particle_momentum_end_f64: DVec2,
    pub residual_f64: DVec2,
    pub g2p_transfer_residual_f64: DVec2,
    pub g2p_unexplained_residual_f64: DVec2,
    pub g2p_mass_closure: G2pMassClosureLedger,
    pub bottom_nodes: Vec<BoundaryNodeImpulseLedger>,
    /// Algebraic consistency of the exact regular-grid quadratic stencil used
    /// by this accepted retry attempt, restricted to particles whose support
    /// touches the lower wall's constrained node band. This is deliberately
    /// read-only: it does not alter P2G, the wall projection, or G2P.
    pub mls_consistency: BoundaryMlsConsistencyLedger,
}

#[derive(Clone, Debug, Default)]
pub struct G2pNodeMassClosure {
    pub cell_index: usize,
    pub x: usize,
    pub y: usize,
    pub stored_mass: f32,
    pub resummed_mass: f64,
    pub mass_gap: f64,
    pub velocity: Vec2,
    pub delta_p: DVec2,
}

#[derive(Clone, Debug, Default)]
pub struct G2pMassClosureLedger {
    pub active_nodes: u64,
    pub mass_gap_l1: f64,
    pub max_abs_mass_gap: f64,
    pub delta_p_sum: DVec2,
    pub delta_p_l1: f64,
    pub max_delta_p_norm: f64,
    pub near_wall_delta_p_l1: f64,
    /// Largest nodes for this accepted substep, ordered by `|delta_p|`.
    pub worst_nodes: Vec<G2pNodeMassClosure>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct BoundaryMlsConsistencyLedger {
    pub particle_count: u64,
    /// Infinity-norm condition number of the local 3x3 linear-MLS moment
    /// matrix, averaged over near-wall particles.
    pub condition_inf_mean: f32,
    pub condition_inf_max: f32,
    /// RMS errors for reproduction by the current, unmodified quadratic
    /// stencil. Linear-value error is measured in particle-local coordinates;
    /// linear-gradient error uses the same `KERNEL_D_INVERSE` reconstruction
    /// as real G2P.
    pub constant_reproduction_rms: f32,
    pub linear_value_reproduction_rms: f32,
    pub linear_gradient_reproduction_rms: f32,
    /// Reproduction after the current lower-wall velocity projection of the
    /// synthetic affine normal field `v_y=y-y_wall`, which satisfies the
    /// physical Dirichlet datum `v_y=0` at the particle clamp plane. APIC P2G
    /// reproduces this affine field exactly before the wall; these two errors
    /// therefore isolate the wall-projection/G2P composition.
    pub projected_linear_value_reproduction_rms: f32,
    pub projected_linear_gradient_reproduction_rms: f32,
    /// Distance to the particle-position clamp plane (`thickness - 1`) for
    /// the closest sampled particle. Kept separate from the velocity-node
    /// cutoff (`node row < thickness`) because the current boundary has no
    /// single explicit geometric wall surface.
    pub closest_clamp_plane_distance: f32,
}

#[derive(Clone, Debug)]
pub struct BoundaryImpulseReport {
    pub mode: BoundaryImpulseExperiment,
    pub accepted_substeps: u64,
    pub residual_sum: Vec2,
    pub residual_rms: Vec2,
    pub max_abs_residual: Vec2,
    pub residual_f64_rms: DVec2,
    pub max_abs_residual_f64: DVec2,
    pub g2p_transfer_residual_f64_rms: DVec2,
    pub max_abs_g2p_transfer_residual_f64: DVec2,
    pub g2p_unexplained_residual_f64_rms: DVec2,
    pub max_abs_g2p_unexplained_residual_f64: DVec2,
    pub g2p_delta_p_sum_rms: DVec2,
    pub max_abs_g2p_delta_p_sum: DVec2,
    pub max_abs_g2p_node_mass_gap: f64,
    pub max_g2p_node_delta_p_norm: f64,
    pub g2p_near_wall_delta_p_l1_fraction: f64,
    pub worst_g2p_node: Option<G2pNodeMassClosure>,
    pub worst_g2p_node_accepted_substep: u64,
    pub gravity_impulse_sum: Vec2,
    pub wall_impulse_sum: Vec2,
    pub contact_impulse_sum: Vec2,
    pub other_grid_impulse_sum: Vec2,
    pub compressive_release_events: u64,
    pub row_wall_impulse_sum: [Vec2; 3],
    pub row_stress_impulse_sum: [Vec2; 3],
    pub first_accepted: Option<AcceptedBoundaryImpulseLedger>,
    pub last_accepted: Option<AcceptedBoundaryImpulseLedger>,
    pub mls_particle_samples: u64,
    pub mls_condition_inf_mean: f32,
    pub mls_condition_inf_max: f32,
    pub mls_closest_clamp_plane_distance: f32,
    pub mls_constant_reproduction_rms: f32,
    pub mls_linear_value_reproduction_rms: f32,
    pub mls_linear_gradient_reproduction_rms: f32,
    pub mls_projected_linear_value_reproduction_rms: f32,
    pub mls_projected_linear_gradient_reproduction_rms: f32,
    /// Pearson correlations across accepted substeps against `abs(R_y)`.
    /// NaN means the MLS metric had no numerically meaningful variance, so a
    /// correlation is mathematically unidentified rather than zero.
    pub mls_condition_residual_correlation: f32,
    pub mls_constant_residual_correlation: f32,
    pub mls_linear_value_residual_correlation: f32,
    pub mls_linear_gradient_residual_correlation: f32,
    pub mls_projected_linear_value_residual_correlation: f32,
    pub mls_projected_linear_gradient_residual_correlation: f32,
}

impl BoundaryImpulseReport {
    fn new(mode: BoundaryImpulseExperiment) -> Self {
        Self {
            mode,
            accepted_substeps: 0,
            residual_sum: Vec2::ZERO,
            residual_rms: Vec2::ZERO,
            max_abs_residual: Vec2::ZERO,
            residual_f64_rms: DVec2::ZERO,
            max_abs_residual_f64: DVec2::ZERO,
            g2p_transfer_residual_f64_rms: DVec2::ZERO,
            max_abs_g2p_transfer_residual_f64: DVec2::ZERO,
            g2p_unexplained_residual_f64_rms: DVec2::ZERO,
            max_abs_g2p_unexplained_residual_f64: DVec2::ZERO,
            g2p_delta_p_sum_rms: DVec2::ZERO,
            max_abs_g2p_delta_p_sum: DVec2::ZERO,
            max_abs_g2p_node_mass_gap: 0.0,
            max_g2p_node_delta_p_norm: 0.0,
            g2p_near_wall_delta_p_l1_fraction: 0.0,
            worst_g2p_node: None,
            worst_g2p_node_accepted_substep: 0,
            gravity_impulse_sum: Vec2::ZERO,
            wall_impulse_sum: Vec2::ZERO,
            contact_impulse_sum: Vec2::ZERO,
            other_grid_impulse_sum: Vec2::ZERO,
            compressive_release_events: 0,
            row_wall_impulse_sum: [Vec2::ZERO; 3],
            row_stress_impulse_sum: [Vec2::ZERO; 3],
            first_accepted: None,
            last_accepted: None,
            mls_particle_samples: 0,
            mls_condition_inf_mean: 0.0,
            mls_condition_inf_max: 0.0,
            mls_closest_clamp_plane_distance: f32::INFINITY,
            mls_constant_reproduction_rms: 0.0,
            mls_linear_value_reproduction_rms: 0.0,
            mls_linear_gradient_reproduction_rms: 0.0,
            mls_projected_linear_value_reproduction_rms: 0.0,
            mls_projected_linear_gradient_reproduction_rms: 0.0,
            mls_condition_residual_correlation: f32::NAN,
            mls_constant_residual_correlation: f32::NAN,
            mls_linear_value_residual_correlation: f32::NAN,
            mls_linear_gradient_residual_correlation: f32::NAN,
            mls_projected_linear_value_residual_correlation: f32::NAN,
            mls_projected_linear_gradient_residual_correlation: f32::NAN,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct OnlineCorrelation {
    n: u64,
    sum_x: f64,
    sum_y: f64,
    sum_xx: f64,
    sum_yy: f64,
    sum_xy: f64,
}

impl OnlineCorrelation {
    fn push(&mut self, x: f32, y: f32) {
        let (x, y) = (f64::from(x), f64::from(y));
        self.n += 1;
        self.sum_x += x;
        self.sum_y += y;
        self.sum_xx += x * x;
        self.sum_yy += y * y;
        self.sum_xy += x * y;
    }

    fn pearson(self) -> f32 {
        if self.n < 2 {
            return f32::NAN;
        }
        let n = self.n as f64;
        let var_x = (self.sum_xx - self.sum_x * self.sum_x / n).max(0.0);
        let var_y = (self.sum_yy - self.sum_y * self.sum_y / n).max(0.0);
        // All consistency errors below are expected at f32 roundoff for an
        // intact regular stencil. Refuse to turn that arithmetic noise into a
        // physically suggestive correlation coefficient.
        let x_scale = (self.sum_xx / n).sqrt().max(1.0);
        if var_x.sqrt() / n.sqrt() <= 1.0e-6 * x_scale || var_y <= 1.0e-24 {
            return f32::NAN;
        }
        ((self.sum_xy - self.sum_x * self.sum_y / n) / (var_x * var_y).sqrt()) as f32
    }
}

#[derive(Debug)]
pub(crate) struct BoundaryImpulseDiagnostic {
    pub(crate) mode: BoundaryImpulseExperiment,
    pub(crate) pending: Option<AcceptedBoundaryImpulseLedger>,
    report: BoundaryImpulseReport,
    residual_square_sum: Vec2,
    residual_f64_square_sum: DVec2,
    g2p_transfer_residual_f64_square_sum: DVec2,
    g2p_unexplained_residual_f64_square_sum: DVec2,
    g2p_delta_p_sum_square_sum: DVec2,
    g2p_delta_p_l1_sum: f64,
    g2p_near_wall_delta_p_l1_sum: f64,
    mls_condition_sum: f64,
    mls_constant_square_sum: f64,
    mls_linear_value_square_sum: f64,
    mls_linear_gradient_square_sum: f64,
    mls_projected_linear_value_square_sum: f64,
    mls_projected_linear_gradient_square_sum: f64,
    condition_residual_correlation: OnlineCorrelation,
    constant_residual_correlation: OnlineCorrelation,
    linear_value_residual_correlation: OnlineCorrelation,
    linear_gradient_residual_correlation: OnlineCorrelation,
    projected_linear_value_residual_correlation: OnlineCorrelation,
    projected_linear_gradient_residual_correlation: OnlineCorrelation,
}

impl BoundaryImpulseDiagnostic {
    pub(crate) fn new(mode: BoundaryImpulseExperiment) -> Self {
        Self {
            mode,
            pending: None,
            report: BoundaryImpulseReport::new(mode),
            residual_square_sum: Vec2::ZERO,
            residual_f64_square_sum: DVec2::ZERO,
            g2p_transfer_residual_f64_square_sum: DVec2::ZERO,
            g2p_unexplained_residual_f64_square_sum: DVec2::ZERO,
            g2p_delta_p_sum_square_sum: DVec2::ZERO,
            g2p_delta_p_l1_sum: 0.0,
            g2p_near_wall_delta_p_l1_sum: 0.0,
            mls_condition_sum: 0.0,
            mls_constant_square_sum: 0.0,
            mls_linear_value_square_sum: 0.0,
            mls_linear_gradient_square_sum: 0.0,
            mls_projected_linear_value_square_sum: 0.0,
            mls_projected_linear_gradient_square_sum: 0.0,
            condition_residual_correlation: OnlineCorrelation::default(),
            constant_residual_correlation: OnlineCorrelation::default(),
            linear_value_residual_correlation: OnlineCorrelation::default(),
            linear_gradient_residual_correlation: OnlineCorrelation::default(),
            projected_linear_value_residual_correlation: OnlineCorrelation::default(),
            projected_linear_gradient_residual_correlation: OnlineCorrelation::default(),
        }
    }

    pub(crate) fn accept_pending(&mut self) {
        let Some(ledger) = self.pending.take() else {
            return;
        };
        self.report.accepted_substeps += 1;
        self.report.residual_sum += ledger.residual;
        self.residual_square_sum += ledger.residual * ledger.residual;
        let n = self.report.accepted_substeps as f32;
        let mean_square = self.residual_square_sum / n;
        self.report.residual_rms = Vec2::new(mean_square.x.sqrt(), mean_square.y.sqrt());
        self.report.max_abs_residual = self.report.max_abs_residual.max(ledger.residual.abs());
        self.residual_f64_square_sum += ledger.residual_f64 * ledger.residual_f64;
        self.g2p_transfer_residual_f64_square_sum +=
            ledger.g2p_transfer_residual_f64 * ledger.g2p_transfer_residual_f64;
        self.g2p_unexplained_residual_f64_square_sum +=
            ledger.g2p_unexplained_residual_f64 * ledger.g2p_unexplained_residual_f64;
        self.g2p_delta_p_sum_square_sum +=
            ledger.g2p_mass_closure.delta_p_sum * ledger.g2p_mass_closure.delta_p_sum;
        let n_f64 = self.report.accepted_substeps as f64;
        let component_rms = |sum: DVec2| {
            let mean = sum / n_f64;
            DVec2::new(mean.x.sqrt(), mean.y.sqrt())
        };
        self.report.residual_f64_rms = component_rms(self.residual_f64_square_sum);
        self.report.g2p_transfer_residual_f64_rms =
            component_rms(self.g2p_transfer_residual_f64_square_sum);
        self.report.g2p_unexplained_residual_f64_rms =
            component_rms(self.g2p_unexplained_residual_f64_square_sum);
        self.report.g2p_delta_p_sum_rms = component_rms(self.g2p_delta_p_sum_square_sum);
        self.report.max_abs_residual_f64 = self
            .report
            .max_abs_residual_f64
            .max(ledger.residual_f64.abs());
        self.report.max_abs_g2p_transfer_residual_f64 = self
            .report
            .max_abs_g2p_transfer_residual_f64
            .max(ledger.g2p_transfer_residual_f64.abs());
        self.report.max_abs_g2p_unexplained_residual_f64 = self
            .report
            .max_abs_g2p_unexplained_residual_f64
            .max(ledger.g2p_unexplained_residual_f64.abs());
        self.report.max_abs_g2p_delta_p_sum = self
            .report
            .max_abs_g2p_delta_p_sum
            .max(ledger.g2p_mass_closure.delta_p_sum.abs());
        self.report.max_abs_g2p_node_mass_gap = self
            .report
            .max_abs_g2p_node_mass_gap
            .max(ledger.g2p_mass_closure.max_abs_mass_gap);
        self.g2p_delta_p_l1_sum += ledger.g2p_mass_closure.delta_p_l1;
        self.g2p_near_wall_delta_p_l1_sum += ledger.g2p_mass_closure.near_wall_delta_p_l1;
        self.report.g2p_near_wall_delta_p_l1_fraction = if self.g2p_delta_p_l1_sum > 0.0 {
            self.g2p_near_wall_delta_p_l1_sum / self.g2p_delta_p_l1_sum
        } else {
            0.0
        };
        if ledger.g2p_mass_closure.max_delta_p_norm > self.report.max_g2p_node_delta_p_norm {
            self.report.max_g2p_node_delta_p_norm = ledger.g2p_mass_closure.max_delta_p_norm;
            self.report.worst_g2p_node = ledger.g2p_mass_closure.worst_nodes.first().cloned();
            self.report.worst_g2p_node_accepted_substep = self.report.accepted_substeps;
        }
        self.report.gravity_impulse_sum += ledger.gravity_impulse;
        self.report.wall_impulse_sum += ledger.wall_impulse;
        self.report.contact_impulse_sum += ledger.contact_impulse;
        self.report.other_grid_impulse_sum += ledger.other_grid_impulse;
        let mls = ledger.mls_consistency;
        if mls.particle_count > 0 {
            let sample_count = mls.particle_count as f64;
            self.report.mls_particle_samples += mls.particle_count;
            self.mls_condition_sum += f64::from(mls.condition_inf_mean) * sample_count;
            self.mls_constant_square_sum +=
                f64::from(mls.constant_reproduction_rms).powi(2) * sample_count;
            self.mls_linear_value_square_sum +=
                f64::from(mls.linear_value_reproduction_rms).powi(2) * sample_count;
            self.mls_linear_gradient_square_sum +=
                f64::from(mls.linear_gradient_reproduction_rms).powi(2) * sample_count;
            self.mls_projected_linear_value_square_sum +=
                f64::from(mls.projected_linear_value_reproduction_rms).powi(2) * sample_count;
            self.mls_projected_linear_gradient_square_sum +=
                f64::from(mls.projected_linear_gradient_reproduction_rms).powi(2) * sample_count;
            self.report.mls_condition_inf_max =
                self.report.mls_condition_inf_max.max(mls.condition_inf_max);
            self.report.mls_closest_clamp_plane_distance = self
                .report
                .mls_closest_clamp_plane_distance
                .min(mls.closest_clamp_plane_distance);
            let total = self.report.mls_particle_samples as f64;
            self.report.mls_condition_inf_mean = (self.mls_condition_sum / total) as f32;
            self.report.mls_constant_reproduction_rms =
                (self.mls_constant_square_sum / total).sqrt() as f32;
            self.report.mls_linear_value_reproduction_rms =
                (self.mls_linear_value_square_sum / total).sqrt() as f32;
            self.report.mls_linear_gradient_reproduction_rms =
                (self.mls_linear_gradient_square_sum / total).sqrt() as f32;
            self.report.mls_projected_linear_value_reproduction_rms =
                (self.mls_projected_linear_value_square_sum / total).sqrt() as f32;
            self.report.mls_projected_linear_gradient_reproduction_rms =
                (self.mls_projected_linear_gradient_square_sum / total).sqrt() as f32;

            let abs_residual_y = ledger.residual_f64.y.abs() as f32;
            self.condition_residual_correlation
                .push(mls.condition_inf_mean, abs_residual_y);
            self.constant_residual_correlation
                .push(mls.constant_reproduction_rms, abs_residual_y);
            self.linear_value_residual_correlation
                .push(mls.linear_value_reproduction_rms, abs_residual_y);
            self.linear_gradient_residual_correlation
                .push(mls.linear_gradient_reproduction_rms, abs_residual_y);
            self.projected_linear_value_residual_correlation
                .push(mls.projected_linear_value_reproduction_rms, abs_residual_y);
            self.projected_linear_gradient_residual_correlation.push(
                mls.projected_linear_gradient_reproduction_rms,
                abs_residual_y,
            );
            self.report.mls_condition_residual_correlation =
                self.condition_residual_correlation.pearson();
            self.report.mls_constant_residual_correlation =
                self.constant_residual_correlation.pearson();
            self.report.mls_linear_value_residual_correlation =
                self.linear_value_residual_correlation.pearson();
            self.report.mls_linear_gradient_residual_correlation =
                self.linear_gradient_residual_correlation.pearson();
            self.report.mls_projected_linear_value_residual_correlation =
                self.projected_linear_value_residual_correlation.pearson();
            self.report
                .mls_projected_linear_gradient_residual_correlation = self
                .projected_linear_gradient_residual_correlation
                .pearson();
        }
        for node in &ledger.bottom_nodes {
            self.report.compressive_release_events += node.released_while_compressive as u64;
            if node.y < 3 {
                self.report.row_wall_impulse_sum[node.y] += node.wall_impulse;
                self.report.row_stress_impulse_sum[node.y] += node.stress_momentum;
            }
        }
        if self.report.first_accepted.is_none() {
            self.report.first_accepted = Some(ledger.clone());
        }
        self.report.last_accepted = Some(ledger);
    }

    fn report(&self) -> &BoundaryImpulseReport {
        &self.report
    }
}

pub(crate) fn particle_momentum(sim: &Simulation) -> Vec2 {
    (0..sim.active_count)
        .map(|i| sim.particles.mass[i] * sim.particles.v[i])
        .sum()
}

pub(crate) fn particle_momentum_f64(sim: &Simulation) -> DVec2 {
    (0..sim.active_count).fold(DVec2::ZERO, |sum, i| {
        sum + f64::from(sim.particles.mass[i]) * sim.particles.v[i].as_dvec2()
    })
}

pub(crate) fn begin_ledger(
    sim: &Simulation,
    dt: f32,
    components: &[GridNodeP2GComponents],
) -> AcceptedBoundaryImpulseLedger {
    let particle_momentum_start = particle_momentum(sim);
    let total_mass: f32 = sim.particles.mass[..sim.active_count].iter().sum();
    let total_mass_f64: f64 = sim.particles.mass[..sim.active_count]
        .iter()
        .map(|&mass| f64::from(mass))
        .sum();
    let mut bottom_nodes = Vec::new();
    for (cell_index, component) in components.iter().enumerate() {
        let y = cell_index % sim.config.grid_res;
        if y < 3 && component.mass > 0.0 {
            bottom_nodes.push(BoundaryNodeImpulseLedger {
                cell_index,
                x: cell_index / sim.config.grid_res,
                y,
                mass: component.mass,
                translation_momentum: component.translation_momentum,
                affine_momentum: component.affine_momentum,
                stress_momentum: component.stress_momentum,
                gravity_impulse: component.mass * sim.config.gravity * dt,
                estimated_normal_traction: if component.stress_volume_weight > 0.0 {
                    component.weighted_tau_yy / component.stress_volume_weight
                } else {
                    0.0
                },
                compressive_traction_active: component.weighted_tau_yy < 0.0,
                ..BoundaryNodeImpulseLedger::default()
            });
        }
    }
    AcceptedBoundaryImpulseLedger {
        dt,
        particle_momentum_start,
        grid_momentum_after_p2g: sim.grid.raw_momentum_sum(),
        gravity_impulse: total_mass * sim.config.gravity * dt,
        particle_momentum_start_f64: particle_momentum_f64(sim),
        grid_momentum_after_p2g_f64: sim.grid.raw_momentum_sum_f64(),
        gravity_impulse_f64: total_mass_f64 * sim.config.gravity.as_dvec2() * f64::from(dt),
        bottom_nodes,
        mls_consistency: measure_lower_wall_mls_consistency(sim),
        ..AcceptedBoundaryImpulseLedger::default()
    }
}

/// Measure the algebraic consistency of the *actual* regular quadratic
/// stencil without changing any state. The local polynomial coordinates are
/// `r_i = x_i - x_p`, exactly `cell_dist` in P2G/G2P. For basis
/// `P=[1,r_x,r_y]`, `M=sum(w P P^T)`. Constant and linear reproduction follow
/// directly from its first row and lower 2x2 block.
fn measure_lower_wall_mls_consistency(sim: &Simulation) -> BoundaryMlsConsistencyLedger {
    let thickness = sim.config.boundary_thickness as i32;
    let clamp_plane = sim.config.boundary_thickness.saturating_sub(1) as f32;
    let mut count = 0_u64;
    let mut condition_sum = 0.0_f64;
    let mut condition_max = 0.0_f64;
    let mut constant_square_sum = 0.0_f64;
    let mut linear_value_square_sum = 0.0_f64;
    let mut linear_gradient_square_sum = 0.0_f64;
    let mut projected_linear_value_square_sum = 0.0_f64;
    let mut projected_linear_gradient_square_sum = 0.0_f64;
    let mut closest_distance = f32::INFINITY;

    for &x in &sim.particles.x[..sim.active_count] {
        let weights = crate::grid::kernel::quadratic_weights(x);
        // The smallest touched row is `base_y - 1`; only sample particles
        // whose real 3x3 support intersects the lower constrained-node band.
        if weights.base_cell.y > thickness {
            continue;
        }
        let mut m = [[0.0_f64; 3]; 3];
        let mut projected_value = 0.0_f64;
        let mut projected_gradient = [0.0_f64; 2];
        for gx in 0..3 {
            for gy in 0..3 {
                let cell_pos = weights.base_cell + glam::IVec2::new(gx as i32 - 1, gy as i32 - 1);
                if flat_index(cell_pos, sim.config.grid_res).is_none() {
                    continue;
                }
                let w = f64::from(weights.wx[gx] * weights.wy[gy]);
                let r = cell_pos.as_vec2() - x + Vec2::splat(0.5);
                let p = [1.0, f64::from(r.x), f64::from(r.y)];
                for row in 0..3 {
                    for col in 0..3 {
                        m[row][col] += w * p[row] * p[col];
                    }
                }
                // Exact affine APIC P2G would put `node_y-y_wall` on this
                // node. The current lower SlipBoundary then projects its
                // inward (negative) part to zero. Gathering this synthetic
                // field with the real weights isolates the composed boundary
                // operator without mutating the simulation.
                let projected_node_value =
                    f64::from((cell_pos.y as f32 + 0.5 - clamp_plane).max(0.0));
                projected_value += w * projected_node_value;
                projected_gradient[0] += w * projected_node_value * f64::from(r.x);
                projected_gradient[1] += w * projected_node_value * f64::from(r.y);
            }
        }

        let condition = matrix_condition_inf(m);
        let constant_error = (m[0][0] - 1.0).abs();
        let linear_value_error = m[0][1].hypot(m[0][2]);
        let k = f64::from(KERNEL_D_INVERSE);
        let gradient_error_x = ((k * m[1][1] - 1.0).powi(2) + (k * m[1][2]).powi(2)).sqrt();
        let gradient_error_y = ((k * m[2][1]).powi(2) + (k * m[2][2] - 1.0).powi(2)).sqrt();
        let linear_gradient_error = gradient_error_x.max(gradient_error_y);
        let expected_value = f64::from((x.y - clamp_plane).max(0.0));
        let projected_linear_value_error = (projected_value - expected_value).abs();
        projected_gradient[0] *= k;
        projected_gradient[1] *= k;
        let projected_linear_gradient_error =
            projected_gradient[0].hypot(projected_gradient[1] - 1.0);

        count += 1;
        condition_sum += condition;
        condition_max = condition_max.max(condition);
        constant_square_sum += constant_error * constant_error;
        linear_value_square_sum += linear_value_error * linear_value_error;
        linear_gradient_square_sum += linear_gradient_error * linear_gradient_error;
        projected_linear_value_square_sum +=
            projected_linear_value_error * projected_linear_value_error;
        projected_linear_gradient_square_sum +=
            projected_linear_gradient_error * projected_linear_gradient_error;
        closest_distance = closest_distance.min(x.y - clamp_plane);
    }

    if count == 0 {
        return BoundaryMlsConsistencyLedger::default();
    }
    let n = count as f64;
    BoundaryMlsConsistencyLedger {
        particle_count: count,
        condition_inf_mean: (condition_sum / n) as f32,
        condition_inf_max: condition_max as f32,
        constant_reproduction_rms: (constant_square_sum / n).sqrt() as f32,
        linear_value_reproduction_rms: (linear_value_square_sum / n).sqrt() as f32,
        linear_gradient_reproduction_rms: (linear_gradient_square_sum / n).sqrt() as f32,
        projected_linear_value_reproduction_rms: (projected_linear_value_square_sum / n).sqrt()
            as f32,
        projected_linear_gradient_reproduction_rms: (projected_linear_gradient_square_sum / n)
            .sqrt() as f32,
        closest_clamp_plane_distance: closest_distance,
    }
}

fn matrix_condition_inf(m: [[f64; 3]; 3]) -> f64 {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if !det.is_finite() || det.abs() <= f64::EPSILON {
        return f64::INFINITY;
    }
    let inv = [
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) / det,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) / det,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) / det,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) / det,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) / det,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) / det,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) / det,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) / det,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) / det,
        ],
    ];
    let norm_inf = |a: [[f64; 3]; 3]| {
        a.into_iter()
            .map(|row| row.into_iter().map(f64::abs).sum::<f64>())
            .fold(0.0, f64::max)
    };
    norm_inf(m) * norm_inf(inv)
}

pub(crate) fn capture_before_wall(grid: &Grid, ledger: &mut AcceptedBoundaryImpulseLedger) {
    ledger.grid_momentum_before_wall = grid.velocity_field_momentum_sum();
    ledger.grid_momentum_before_wall_f64 = grid.velocity_field_momentum_sum_f64();
    for node in &mut ledger.bottom_nodes {
        node.velocity_before_wall = grid.velocity_at_index(node.cell_index);
    }
}

/// Re-sum the particle masses onto nodes in canonical particle/stencil order,
/// then compare that f64 result to the real f32 mass stored by P2G. The grid
/// velocity is the final value immediately before real G2P. No solver state is
/// modified.
pub(crate) fn measure_g2p_mass_closure(sim: &Simulation) -> G2pMassClosureLedger {
    let resolution = sim.config.grid_res;
    let mut resummed_mass = vec![0.0_f64; resolution * resolution];
    for i in 0..sim.active_count {
        let weights = crate::grid::kernel::quadratic_weights(sim.particles.x[i]);
        let mass = f64::from(sim.particles.mass[i]);
        for gx in 0..3 {
            for gy in 0..3 {
                let cell_pos = weights.base_cell + glam::IVec2::new(gx as i32 - 1, gy as i32 - 1);
                let Some(index) = flat_index(cell_pos, resolution) else {
                    continue;
                };
                let weight = f64::from(weights.wx[gx]) * f64::from(weights.wy[gy]);
                resummed_mass[index as usize] += mass * weight;
            }
        }
    }

    let mut report = G2pMassClosureLedger::default();
    let mut nodes = Vec::new();
    for (index, &resummed) in resummed_mass.iter().enumerate() {
        let cell_pos = glam::IVec2::new((index / resolution) as i32, (index % resolution) as i32);
        let stored = sim.grid.mass_at(cell_pos);
        if stored == 0.0 && resummed == 0.0 {
            continue;
        }
        let velocity = sim.grid.velocity_at_index(index);
        let mass_gap = resummed - f64::from(stored);
        let delta_p = velocity.as_dvec2() * mass_gap;
        let delta_p_norm = delta_p.length();
        report.active_nodes += 1;
        report.mass_gap_l1 += mass_gap.abs();
        report.max_abs_mass_gap = report.max_abs_mass_gap.max(mass_gap.abs());
        report.delta_p_sum += delta_p;
        report.delta_p_l1 += delta_p_norm;
        report.max_delta_p_norm = report.max_delta_p_norm.max(delta_p_norm);
        if index % resolution < sim.config.boundary_thickness + 1 {
            report.near_wall_delta_p_l1 += delta_p_norm;
        }
        nodes.push(G2pNodeMassClosure {
            cell_index: index,
            x: index / resolution,
            y: index % resolution,
            stored_mass: stored,
            resummed_mass: resummed,
            mass_gap,
            velocity,
            delta_p,
        });
    }
    nodes.sort_by(|a, b| {
        b.delta_p
            .length_squared()
            .total_cmp(&a.delta_p.length_squared())
            .then_with(|| a.cell_index.cmp(&b.cell_index))
    });
    nodes.truncate(8);
    report.worst_nodes = nodes;
    report
}

/// Apply only the controlled diagnostic variations after the ordinary boundary
/// callback. The physical particle wall remains unchanged in every mode.
pub(crate) fn apply_experimental_lower_wall(
    grid: &mut Grid,
    grid_res: usize,
    boundary_thickness: usize,
    mode: BoundaryImpulseExperiment,
    ledger: &mut AcceptedBoundaryImpulseLedger,
) {
    let traction_aware = mode.traction_aware();
    let deepest_row = if mode.deep_band() {
        boundary_thickness
    } else {
        boundary_thickness.saturating_sub(1)
    };
    for node in &mut ledger.bottom_nodes {
        let in_band = node.y <= deepest_row;
        let velocity_active = in_band && node.velocity_before_wall.y < 0.0;
        node.velocity_condition_active = velocity_active;
        node.released_while_compressive =
            in_band && node.velocity_before_wall.y > 0.0 && node.compressive_traction_active;
        if let Some(cell) = grid.cell_at_index_mut(node.cell_index) {
            if mode.deep_band() && node.y == boundary_thickness {
                cell.momentum.y = cell.momentum.y.max(0.0);
            }
            if traction_aware && in_band && node.compressive_traction_active {
                cell.momentum.y = 0.0;
            }
            node.velocity_after_wall = cell.momentum;
            node.wall_impulse = node.mass * (node.velocity_after_wall - node.velocity_before_wall);
        }
    }
    let _ = grid_res;
}

impl Simulation {
    /// TEMPORARY explicit opt-in used by the controlled structural-bounce test.
    /// Normal users remain on the exact production path.
    pub fn enable_boundary_impulse_diagnostic(&mut self, mode: BoundaryImpulseExperiment) {
        self.boundary_impulse_diagnostic = Some(BoundaryImpulseDiagnostic::new(mode));
    }

    pub fn boundary_impulse_report(&self) -> Option<&BoundaryImpulseReport> {
        self.boundary_impulse_diagnostic
            .as_ref()
            .map(BoundaryImpulseDiagnostic::report)
    }
}

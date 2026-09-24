use glam::Vec2;

/// Rod-solver diagnostics -- `spacetime::rod` is a separate solver from the
/// MPM particle grid (see `rod` module doc), so its state doesn't fall out
/// of `collect_snapshot`'s particle/grid scan and needs its own small
/// aggregation. Not `Copy` (holds a `Vec`), unlike `SimSnapshot` itself.
#[derive(Debug, Clone, Default)]
pub struct RodSnapshot {
    pub count: usize,
    /// Rods with `Rod::sleeping == true` -- skipped entirely by
    /// scatter/gather/internal-force integration this step.
    pub sleeping_count: usize,
    /// Max point speed across all rods (sleeping rods contribute 0.0, same
    /// convention as `SimSnapshot::max_particle_speed` excluding sleeping
    /// particles from the scan would -- sleeping rods are physically at rest).
    pub max_speed: f32,
    /// Tip position (last point) of each rod, in the same order as
    /// `Simulation::rods()`.
    pub tip_positions: Vec<Vec2>,
}

#[derive(Debug, Clone, Default)]
pub struct SimSnapshot {
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
    /// Legacy compatibility counter for former G2P velocity clipping.
    /// The solver no longer clips velocity, so this remains zero; CFL is met
    /// by substepping or an inadmissible state is reported.
    pub vel_clamp_count: usize,
    /// Particles whose deformation state was projected back to admissible this step.
    /// Nonzero = explicit integration diverged; check dt, material params, or stiffness.
    pub j_projection_count: usize,
    /// Legacy compatibility field for former max-substep time loss.
    /// A solver step now advances the full requested time, so this is zero.
    pub sim_time_dropped: f32,
    /// Wall-clock time breakdown for the last `step()` call. All values in microseconds.
    /// Accumulated across all substeps — divide by `substeps_last_step` for per-substep cost.
    pub timing: StepTiming,
    /// Total kinetic energy (sum of `0.5 * mass * |v|^2`) across all particles.
    /// Real, generic sanity signal for ANY scene: should decay toward a steady value
    /// under damping (viscous/plastic materials) or oscillate boundedly for a purely
    /// elastic one — unbounded growth with no external force driving it is a real bug,
    /// not just "high energy."
    pub total_kinetic_energy: f32,
    /// Max speed among particles with `Particle::pinned != 0`. Should read exactly 0.0
    /// for any scene using pinned/Dirichlet anchors — G2P forces `v=0` on pinned
    /// particles every substep (see `transfer.rs`). Nonzero here means the pinning
    /// mechanism itself is broken (a real engine bug), not a scene-tuning issue —
    /// added specifically so this class of bug is directly observable instead of
    /// inferred indirectly from a body slowly drifting.
    pub max_pinned_particle_speed: f32,
    /// Rod-solver diagnostics -- empty/default for any scene with no rods
    /// (`Simulation::rods()` empty), zero cost in that case.
    pub rods: RodSnapshot,
}

/// Real-world (SI) unit conversion of a `SimSnapshot`'s key physical
/// quantities -- lets any demo/test print actual m/s, kg*m/s, and Joules
/// instead of raw grid units, so "is this realistic" is a direct read, not
/// a mental `* dx_meters` every time. Mass is already real kilograms
/// throughout this engine (`Particle::mass`'s own convention) -- only
/// length/velocity-derived quantities need `dx_meters` scaling.
#[derive(Debug, Clone, Copy, Default)]
pub struct SiSnapshot {
    pub sim_time_elapsed_s: f32,
    pub max_particle_speed_m_s: f32,
    pub max_grid_speed_m_s: f32,
    pub total_particle_mass_kg: f32,
    pub total_particle_momentum_kg_m_s: Vec2,
    pub total_kinetic_energy_joules: f32,
    pub recommended_max_dt_s: f32,
}

impl SimSnapshot {
    /// Convert to real SI units given the scene's `dx_meters` (meters per
    /// grid cell). Velocity: v_real = v_grid * dx_meters. Momentum
    /// (mass*v): scales by the same one factor of dx_meters. Kinetic
    /// energy (mass*v^2): scales by dx_meters^2. `configured_dt` is
    /// already real seconds throughout this engine (the same convention
    /// `SimConfig::dt` itself uses), so `sim_time_elapsed_s` and
    /// `recommended_max_dt_s` need no further conversion.
    pub fn to_si(&self, dx_meters: f32) -> SiSnapshot {
        SiSnapshot {
            sim_time_elapsed_s: self.frame_index as f32 * self.configured_dt,
            max_particle_speed_m_s: self.max_particle_speed * dx_meters,
            max_grid_speed_m_s: self.max_grid_speed * dx_meters,
            total_particle_mass_kg: self.total_particle_mass,
            total_particle_momentum_kg_m_s: self.total_particle_momentum * dx_meters,
            total_kinetic_energy_joules: self.total_kinetic_energy * dx_meters * dx_meters,
            recommended_max_dt_s: self.recommended_max_dt_from_velocity_cfl,
        }
    }
}

/// Wall-clock timing breakdown for one `step()` call (sum of all substeps).
/// Measured with `std::time::Instant` — zero external dependencies.
/// Read via `solver.diagnostics_snapshot().timing`.
#[derive(Debug, Clone, Copy, Default)]
pub struct StepTiming {
    /// P2G scatter: particle → grid momentum/stress accumulation.
    pub p2g_us: u64,
    /// Grid update: momentum normalization + gravity + boundary application.
    /// Includes `pressure_us` below (a real cost, not double-counted against
    /// `total_us`) -- `pressure_us` exists purely to break out how much of
    /// this bucket is the fluid pressure-projection loop specifically, since
    /// that loop was folded in here silently before 2026-08-09.
    pub grid_update_us: u64,
    /// The `SimConfig::fluid_pressure_iterations` corrector loop
    /// (`Grid::project_fluid_incompressibility` + its own re-applied
    /// boundary pass) -- a SUBSET of `grid_update_us`, not additive to it.
    /// Zero for any scene that doesn't enable pressure projection. Split out
    /// 2026-08-09 because the prior "63% of frame time is grid_update_us"
    /// profiling number couldn't distinguish the pressure solve from
    /// gravity/boundary/contact/mixture/Cundall -- see MEMORY.md.
    pub pressure_us: u64,
    /// G2P gather: grid → particle velocity/position + plasticity update.
    pub g2p_us: u64,
    /// Force fields (NBody, gravity wells, Coulomb). Zero if no fields registered.
    pub fields_us: u64,
    /// Thermal + scalar diffusion. Zero if neither is active.
    pub thermal_us: u64,
    /// CFL timestep selection (choose_substep_dt) — iterates all particles once per substep.
    pub cfl_us: u64,
    /// Spatial hash rebuild (O(N) per substep) — powers particles_near / count_near queries.
    pub spatial_hash_us: u64,
    /// Phase rule evaluation + sleep scoring (O(N) per substep).
    pub phase_sleep_us: u64,
    /// `project_invalid_state` admissibility scan (O(N) pre-P2G, only when standard config).
    pub project_us: u64,
    /// Density recompute via P2G volume estimation (only when fluid materials present).
    pub density_us: u64,
    /// `do_substep_with_retry`'s own `self.particles.clone()` snapshot -- taken
    /// unconditionally once per attempt (up to `FLUID_STEP_RETRY_LIMIT+1`, worst
    /// case 17x) whenever `SimConfig::fluid_step_retry_enabled` is on, regardless
    /// of whether that attempt actually needed a retry. A full SoA deep-copy at
    /// full particle count, added 2026-08-09 alongside the retry/backstop fixes --
    /// not visible in any other bucket before this field existed (fell into the
    /// unaccounted `total_us` - sum(other fields) residual). Zero when retry is
    /// disabled (the default) or no material owns deformation/volume state.
    pub retry_snapshot_us: u64,
    /// Total wall time for the step (includes overhead not captured in individual phases).
    pub total_us: u64,
}

// Snapshot aggregation from Particles/Grid state (collect_snapshot,
// collect_snapshot_particles_only, and their private per-particle/per-cell
// helpers) split into collect.rs -- was ~330 of this file's ~524 lines of
// pure computation, as opposed to the data definitions above. See that
// file's own doc comment.
mod collect;
pub use collect::{collect_snapshot, collect_snapshot_particles_only};

/// Aggregate `RodSnapshot` from a `Simulation`'s live rods -- called from
/// `Simulation::diagnostics_snapshot`, kept here (not in `collect.rs`) since
/// it scans `rod::Rod` not `Particles`/`Grid`.
pub fn collect_rod_snapshot(rods: &[crate::rod::Rod]) -> RodSnapshot {
    let mut snap = RodSnapshot {
        count: rods.len(),
        ..Default::default()
    };
    for rod in rods {
        if rod.sleeping {
            snap.sleeping_count += 1;
        } else {
            for v in &rod.points.v {
                snap.max_speed = snap.max_speed.max(v.length());
            }
        }
        if let Some(&tip) = rod.points.x.last() {
            snap.tip_positions.push(tip);
        }
    }
    snap
}

#[cfg(test)]
mod rod_snapshot_tests {
    use super::*;
    use crate::rod::{Rod, RodMaterial, build_straight_rod};

    fn straight_rod(start: Vec2, end: Vec2) -> Rod {
        let points = build_straight_rod(start, end, 4, 0.01, 0.01);
        let material = RodMaterial::new(1.0e-3, 1.0e-6, 1.0e-4, 1.0e-6);
        Rod::new(points, material)
    }

    #[test]
    fn empty_rod_list_gives_empty_snapshot() {
        let snap = collect_rod_snapshot(&[]);
        assert_eq!(snap.count, 0);
        assert_eq!(snap.sleeping_count, 0);
        assert_eq!(snap.max_speed, 0.0);
        assert!(snap.tip_positions.is_empty());
    }

    #[test]
    fn counts_and_tip_positions_match_real_rod_state() {
        let mut awake = straight_rod(Vec2::new(0.0, 0.0), Vec2::new(0.0, 3.0));
        awake.points.v[2] = Vec2::new(5.0, 0.0); // a real, nonzero interior speed
        let mut asleep = straight_rod(Vec2::new(10.0, 0.0), Vec2::new(10.0, 3.0));
        asleep.sleeping = true;
        asleep.points.v[1] = Vec2::new(99.0, 0.0); // must NOT count -- sleeping rods don't move

        let rods = [awake, asleep];
        let snap = collect_rod_snapshot(&rods);

        assert_eq!(snap.count, 2);
        assert_eq!(snap.sleeping_count, 1);
        assert!(
            (snap.max_speed - 5.0).abs() < 1.0e-6,
            "max_speed must come from the awake rod's real 5.0 m/s point, \
             not the sleeping rod's stale 99.0 leftover velocity"
        );
        assert_eq!(snap.tip_positions.len(), 2);
        assert_eq!(snap.tip_positions[0], rods[0].points.x[3]);
        assert_eq!(snap.tip_positions[1], rods[1].points.x[3]);
    }
}

#[cfg(test)]
mod si_conversion_tests {
    use super::*;

    /// Real, hand-checkable conversion: a particle moving at exactly 1
    /// grid-cell/second in a scene where 1 cell = 0.01m must read as
    /// exactly 0.01 m/s once converted -- not an approximation.
    #[test]
    fn to_si_converts_known_values_exactly() {
        let snap = SimSnapshot {
            frame_index: 100,
            configured_dt: 0.1,
            max_particle_speed: 1.0,
            max_grid_speed: 2.0,
            total_particle_mass: 5.0,
            total_particle_momentum: Vec2::new(3.0, 0.0),
            total_kinetic_energy: 10.0,
            recommended_max_dt_from_velocity_cfl: 0.005,
            ..Default::default()
        };
        let dx_meters = 0.01;
        let si = snap.to_si(dx_meters);

        assert!(
            (si.sim_time_elapsed_s - 10.0).abs() < 1.0e-6,
            "100 steps * 0.1s = 10s real time"
        );
        assert!((si.max_particle_speed_m_s - 0.01).abs() < 1.0e-6);
        assert!((si.max_grid_speed_m_s - 0.02).abs() < 1.0e-6);
        assert!(
            (si.total_particle_mass_kg - 5.0).abs() < 1.0e-6,
            "mass is already real kg, unscaled"
        );
        assert!((si.total_particle_momentum_kg_m_s.x - 0.03).abs() < 1.0e-6);
        assert!(
            (si.total_kinetic_energy_joules - 10.0 * 0.01 * 0.01).abs() < 1.0e-9,
            "KE scales by dx_meters^2 (velocity is squared)"
        );
        assert!(
            (si.recommended_max_dt_s - 0.005).abs() < 1.0e-6,
            "dt is already real seconds"
        );
    }
}

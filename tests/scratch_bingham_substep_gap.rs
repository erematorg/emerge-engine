//! Why does the GPU run a third of the CPU's substeps on Bingham?
//!
//! `tests/gpu_parity.rs` measures the gap (220 substeps against 57 in free
//! fall, 660 against 177 standing) but not its cause. This prints what each
//! side actually chose, next to the material's own bound evaluated by hand,
//! so the missing term can be named rather than guessed.
extern crate emerge_engine as emerge;

#[cfg(feature = "gpu")]
mod probe {
    use emerge::gpu::GpuSimulation;
    use emerge::{
        BinghamFluidMaterial, MaterialModel, MaterialRegistry, SimConfig, SlipBoundary,
        SpawnRegion, build_particles,
    };
    use glam::{IVec2, Vec2};
    use pollster::block_on;

    #[test]
    #[ignore = "diagnostic probe kept for reruns, not part of the CI suite"]
    fn bingham_substep_gap() {
        let config = SimConfig {
            max_substeps_per_step: 64,
            ..SimConfig::standard(32, 0.002, Vec2::new(0.0, -9.81))
        };
        let spawn = SpawnRegion {
            spacing: 0.5,
            box_size: IVec2::new(6, 6),
            box_center: Vec2::splat(16.0),
            initial_velocity_scale: 0.0,
            ..SpawnRegion::for_sim(&config)
        };
        let material = BinghamFluidMaterial::low_yield(4.0, 10.0);
        println!(
            "config: dt={} grid_cell_size={} min_dt={} max_substeps={} adaptive={} material_cfl={} viscous_cfl={} cfl={}",
            config.dt,
            config.grid_cell_size,
            config.min_dt,
            config.max_substeps_per_step,
            config.adaptive_timestep,
            config.material_cfl_coefficient,
            config.viscous_timestep_coefficient,
            config.cfl_coefficient
        );
        println!(
            "material: rest_density={} eos_stiffness={} eos_power={} eta={} yield={} critical_shear_rate={} shear_modulus={} bulk_viscosity={}",
            material.rest_density,
            material.eos_stiffness,
            material.eos_power,
            material.dynamic_viscosity,
            material.yield_stress,
            material.critical_shear_rate,
            material.shear_modulus,
            material.bulk_viscosity
        );

        let mut cpu = emerge::Simulation::new(config, spawn)
            .with_default_material(Box::new(material))
            .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));
        let particles = build_particles(&config, spawn);
        let rest_density_grid = particles[0].density;
        println!(
            "spawn density (grid units) = {rest_density_grid}, mass = {}",
            particles[0].mass
        );

        // The material's own bound, at the spawn state, with the config's
        // own coefficients: what both paths are supposed to be reading.
        let bound = material.timestep_bound(
            rest_density_grid,
            1.0,
            config.grid_cell_size,
            config.material_cfl_coefficient,
            config.viscous_timestep_coefficient,
        );
        println!(
            "timestep_bound at spawn = {bound:.6e} s  -> {:.1} substeps for a {} s frame",
            config.dt / bound,
            config.dt
        );

        let gpu_registry = MaterialRegistry::with_default(Box::new(material));
        let mut gpu = block_on(GpuSimulation::new(config, particles, gpu_registry));

        for frame in 0..5 {
            cpu.step();
            gpu.step_frame();
            gpu.sync_particles_blocking();
            println!(
                "frame {frame}: cpu {} substeps (sub_dt {:.3e}), gpu {} substeps (effective_dt {:.3e}), gpu dropped {:.3e} s",
                cpu.diagnostics_snapshot().substeps_last_step,
                config.dt / cpu.diagnostics_snapshot().substeps_last_step.max(1) as f32,
                gpu.last_substeps(),
                gpu.effective_dt(),
                gpu.last_sim_time_dropped()
            );
            let c = cpu.particles().get(0);
            let g = gpu.particles()[0];
            println!(
                "          cpu p0: density={:.4} volume={:.4} y={:.5} | gpu p0: density={:.4} volume={:.4} y={:.5}",
                c.density, c.volume, c.x.y, g.density, g.volume, g.x.y
            );
        }
    }
}

//! Why does the GPU answer a snow compression differently?
//!
//! `tests/gpu_parity.rs` leaves one known gap: under uniaxial compression
//! the GPU's snow ends 9.8e-2 away in position and 2.5 in velocity, and
//! runs 19 substeps against the CPU's 39. This prints, frame by frame,
//! what each side carries in the two state variables snow's own law is
//! made of, its hardening and its plastic volume ratio, so the gap can be
//! attributed to the substep count, to the plasticity, or to the stress.
extern crate emerge_engine as emerge;

#[cfg(feature = "gpu")]
mod probe {
    use emerge::gpu::GpuSimulation;
    use emerge::{
        MaterialModel, MaterialRegistry, SimConfig, SlipBoundary, SpawnRegion, StomakhinMaterial,
        build_particles,
    };
    use glam::{IVec2, Mat2, Vec2};
    use pollster::block_on;

    #[test]
    #[ignore = "diagnostic probe kept for reruns, not part of the CI suite"]
    fn snow_compression_gap() {
        let config = SimConfig {
            max_substeps_per_step: 64,
            ..SimConfig::standard(32, 0.002, Vec2::ZERO)
        };
        let spawn = SpawnRegion {
            spacing: 0.5,
            box_size: IVec2::new(6, 6),
            box_center: Vec2::splat(16.0),
            initial_velocity_scale: 0.0,
            initial_deformation_gradient: Mat2::from_diagonal(Vec2::new(0.9, 1.0)),
            ..SpawnRegion::for_sim(&config)
        };
        let material = StomakhinMaterial::from_young_modulus(10_000.0, 0.3);
        println!(
            "snow: lambda={:.1} mu={:.1} hardening_exponent={} compression_limit={} stretch_limit={}",
            material.lambda,
            material.mu,
            material.hardening_exponent,
            material.compression_limit,
            material.stretch_limit
        );

        let mut cpu = emerge::Simulation::new(config, spawn)
            .with_default_material(Box::new(material))
            .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));
        let particles = build_particles(&config, spawn);
        let gpu_registry = MaterialRegistry::with_default(Box::new(material));
        let mut gpu = block_on(GpuSimulation::new(config, particles, gpu_registry));

        let report = |tag: &str, h: f32, jp: f32, j: f32, v: f32, x: f32| {
            println!("    {tag}: hardening={h:.5} Jp={jp:.5} J={j:.5} |v|={v:.5} x={x:.5}");
        };
        for frame in 0..8 {
            cpu.step();
            gpu.step_frame();
            gpu.sync_particles_blocking();
            let c = cpu.particles();
            let g = gpu.particles();
            // Particle 0 is a corner; the middle of the block is where the
            // law works hardest, so report both.
            let mid = c.len() / 2;
            println!(
                "frame {frame}: cpu {} substeps, gpu {} substeps (dropped {:.2e} s)",
                cpu.diagnostics_snapshot().substeps_last_step,
                gpu.last_substeps(),
                gpu.last_sim_time_dropped()
            );
            report(
                "cpu mid",
                c.hardening_scale[mid],
                c.plastic_volume_ratio[mid],
                c.deformation_gradient[mid].determinant(),
                c.v[mid].length(),
                c.x[mid].x,
            );
            // The same bound evaluated on each side's own state: this is
            // what each scan is supposed to be reading.
            let mut cpu_min = f32::INFINITY;
            let mut gpu_min = f32::INFINITY;
            let (mut cpu_worst, mut gpu_worst) = (0usize, 0usize);
            for i in 0..c.len() {
                let a = material.timestep_bound(
                    c.density[i],
                    c.hardening_scale[i],
                    config.grid_cell_size,
                    config.material_cfl_coefficient,
                    config.viscous_timestep_coefficient,
                );
                if a < cpu_min {
                    cpu_min = a;
                    cpu_worst = i;
                }
                let b = material.timestep_bound(
                    g[i].density,
                    g[i].hardening_scale,
                    config.grid_cell_size,
                    config.material_cfl_coefficient,
                    config.viscous_timestep_coefficient,
                );
                if b < gpu_min {
                    gpu_min = b;
                    gpu_worst = i;
                }
            }
            println!(
                "    bound over the whole body: cpu {cpu_min:.6} (particle {cpu_worst}, h={:.4} rho={:.4}), gpu {gpu_min:.6} (particle {gpu_worst}, h={:.4} rho={:.4})",
                c.hardening_scale[cpu_worst],
                c.density[cpu_worst],
                g[gpu_worst].hardening_scale,
                g[gpu_worst].density
            );
            report(
                "gpu mid",
                g[mid].hardening_scale,
                g[mid].plastic_volume_ratio,
                g[mid].deformation_gradient.determinant(),
                g[mid].v.length(),
                g[mid].x.x,
            );
        }
    }
}

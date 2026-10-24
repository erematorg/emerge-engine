//! The clay scene of `basic_vonmises` (`examples/cpu/vonmises_clay_scene.rs`,
//! included) on the GPU, beside the CPU: substeps, wall time per frame,
//! time dropped, and each blob's mean and smallest J, to see the cost and
//! whether `VonMisesMaterial`'s known GPU compression drift shows on this
//! nearly incompressible clay. Needs a GPU adapter. A probe.

// The scene module is included once, by `vonmises_clay`, and shared from there.

#[cfg(feature = "gpu")]
mod gpu_clay {
    use crate::emerge::MaterialRegistry;
    use crate::emerge::gpu::GpuSimulation;
    use crate::emerge::particle::Particle;
    use crate::vonmises_clay::vonmises_clay_scene::*;
    use pollster::block_on;

    fn blobs(particles: &[Particle]) -> String {
        let mut out = String::new();
        for (slot, (name, _)) in CLAYS.iter().enumerate() {
            let (mut j, mut jmin, mut n, mut bottom) = (0.0f32, f32::MAX, 0.0f32, f32::MAX);
            for p in particles.iter().filter(|p| p.material_id == slot as u32) {
                let det = p.deformation_gradient.determinant();
                j += det;
                jmin = jmin.min(det);
                bottom = bottom.min(p.x.y);
                n += 1.0;
            }
            out += &format!("  {name}: J {:.4}/{jmin:.4} bottom {bottom:.2}", j / n);
        }
        out
    }

    #[test]
    #[ignore = "probe: needs a real GPU; run with --features gpu --ignored --nocapture"]
    fn the_clay_scene_on_the_gpu() {
        let (cpu, materials) = make_sim(1.0);
        let config = *cpu.config();
        let particles: Vec<Particle> = cpu.particles().iter().collect();
        let mut registry = MaterialRegistry::with_default(Box::new(materials[0]));
        registry.insert(MAT_SOFT, Box::new(materials[1]));
        registry.insert(MAT_MEDIUM, Box::new(materials[2]));
        let mut gpu = block_on(GpuSimulation::new(config, particles, registry));
        let (mut substeps, mut dropped) = (0usize, 0.0f32);
        let started = std::time::Instant::now();
        let frames = 600;
        for frame in 1..=frames {
            gpu.step_frame();
            substeps += gpu.last_substeps();
            dropped += gpu.last_sim_time_dropped();
            if [10, 20, 60, 120, 300, 600].contains(&frame) {
                gpu.sync_particles_blocking();
                println!("GPU frame {frame}:{}", blobs(gpu.particles()));
            }
        }
        gpu.sync_particles_blocking();
        let wall = started.elapsed().as_secs_f32();
        println!(
            "GPU {frames} frames ({:.1} s simulated): {:.1} substeps a frame, {dropped:.3} s dropped, {:.1} ms a frame (with the six readbacks), simulated time at {:.3}x real time",
            frames as f32 * DT,
            substeps as f32 / frames as f32,
            wall * 1000.0 / frames as f32,
            frames as f32 * DT / wall
        );
    }
}

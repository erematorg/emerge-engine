//! Test suite for `Renderer` -- split out of `mod.rs` (was ~150 of its ~930
//! lines), same pattern as `gpu/solver/device_lost_tests.rs`.

use super::*;
use crate::particle::Particle;
use glam::Mat2;

fn headless_device() -> (wgpu::Device, wgpu::Queue) {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::None,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .expect("no GPU adapter available for render test");
    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
        .expect("failed to create device")
}

/// Real subsurface scattering must actually change ByPhysics's output, not be
/// dead-stored data -- two materials with identical absorption but different
/// `sigma_s` must render differently (see prep_instances.wgsl's ByPhysics
/// branch for the real single-scattering-albedo derivation this mirrors).
#[test]
fn scattering_changes_by_physics_color() {
    let (device, _queue) = headless_device();
    let mut r = Renderer::new(&device, 16, wgpu::TextureFormat::Rgba8UnormSrgb);
    r.set_color_mode(ColorMode::ByPhysics);
    r.set_optical_params(0, [0.3, 0.3, 0.3]);
    r.set_optical_params(1, [0.3, 0.3, 0.3]);
    r.set_optical_scattering(1, 5.0); // real tissue-scale reduced scattering coeff

    let mut p0 = Particle::zeroed();
    p0.material_id = 0;
    p0.deformation_gradient = Mat2::IDENTITY;
    let mut p1 = p0;
    p1.material_id = 1;

    let c0 = r.particle_color(&p0);
    let c1 = r.particle_color(&p1);
    assert_ne!(
        c0, c1,
        "identical absorption but different sigma_s must render differently"
    );
}

/// Real specular Fresnel reflectance must actually change ByPhysics's output --
/// same check as scattering, for the R0 term.
#[test]
fn specular_r0_changes_by_physics_color() {
    let (device, _queue) = headless_device();
    let mut r = Renderer::new(&device, 16, wgpu::TextureFormat::Rgba8UnormSrgb);
    r.set_color_mode(ColorMode::ByPhysics);
    r.set_optical_params(0, [0.3, 0.3, 0.3]);
    r.set_optical_params(1, [0.3, 0.3, 0.3]);
    r.set_specular_r0(1, 0.02); // real water-scale Fresnel base reflectance

    let mut p0 = Particle::zeroed();
    p0.material_id = 0;
    p0.deformation_gradient = Mat2::IDENTITY;
    let mut p1 = p0;
    p1.material_id = 1;

    let c0 = r.particle_color(&p0);
    let c1 = r.particle_color(&p1);
    assert_ne!(
        c0, c1,
        "identical absorption but different specular R0 must render differently"
    );
}

/// `Renderer::new` must succeed and `upload_optical_params` must not panic with
/// the extended (scattering + specular) `OpticalTable` layout -- a real,
/// end-to-end check that the WGSL struct and Rust struct stayed in sync (a
/// mismatch here would show up as a wgpu validation panic, not a silent bug).
#[test]
fn renderer_construction_and_optical_upload_survive_extended_table() {
    let (device, queue) = headless_device();
    let mut r = Renderer::new(&device, 16, wgpu::TextureFormat::Rgba8UnormSrgb);
    r.set_optical_params(0, [0.18, 0.22, 0.55]);
    r.set_optical_scattering(0, 8.0);
    r.set_specular_r0(0, 0.02);
    r.upload_optical_params(&queue);
}

/// End-to-end GPU path (the one LP actually uses, `render_gpu`): real
/// particles on a real `GpuSimulation`, real compute dispatch through
/// `prep_instances.wgsl` with the extended `OpticalTable`, real render pass to
/// an offscreen texture. Proves the whole pipeline survives, not just that
/// `Renderer::new` compiles the shader in isolation.
#[test]
fn render_gpu_survives_scattering_and_specular_end_to_end() {
    use crate::gpu::GpuSimulation;
    use crate::{MaterialRegistry, NeoHookeanMaterial, SimConfig, SpawnRegion, build_particles};
    use std::sync::Arc;

    let (device, queue) = headless_device();
    let device = Arc::new(device);
    let queue = Arc::new(queue);

    let config = SimConfig::standard(32, 0.1, glam::Vec2::new(0.0, -0.3));
    let particles = build_particles(
        &config,
        SpawnRegion::for_sim(&config)
            .at(glam::Vec2::splat(16.0))
            .disk(4.0)
            .spacing(0.5)
            .material(0)
            .precompute_volumes(),
    );
    let registry = MaterialRegistry::with_default(Box::new(NeoHookeanMaterial::new(100.0, 50.0)));
    let sim =
        GpuSimulation::with_device(device.clone(), queue.clone(), config, particles, registry);

    let fmt = wgpu::TextureFormat::Rgba8UnormSrgb;
    let mut r = Renderer::new(&device, sim.particle_count(), fmt);
    r.set_color_mode(ColorMode::ByPhysics);
    r.set_optical_params(0, [0.18, 0.22, 0.55]);
    r.set_optical_scattering(0, 8.0);
    r.set_specular_r0(0, 0.02);
    r.set_camera(&queue, 32, 64, 64, 0.6, true);

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("render_gpu_test_target"),
        size: wgpu::Extent3d {
            width: 64,
            height: 64,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: fmt,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    r.render_gpu(
        &device,
        &queue,
        sim.particle_buffer(),
        sim.particle_count(),
        &view,
        true,
    );
    device.poll(wgpu::PollType::wait_indefinitely()).ok();
}

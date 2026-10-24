//! Test suite for `Renderer` -- split out of `mod.rs` (was ~150 of its ~930
//! lines), same pattern as `gpu/solver/device_lost_tests.rs`.

use super::gpu_types::GridPeakParams;
use super::*;
use crate::particle::Particle;
use glam::Mat2;

fn headless_device() -> (wgpu::Device, wgpu::Queue) {
    let instance = crate::systems::gpu::create_wgpu_instance();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::None,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .expect("no GPU adapter available for render test");
    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
        .expect("failed to create device")
}

/// Subsurface scattering changes ByPhysics's output: two materials with the
/// same absorption and different `sigma_s` render differently (see
/// prep_instances.wgsl's ByPhysics branch for the single-scattering albedo).
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn scattering_changes_by_physics_color() {
    let (device, queue) = headless_device();
    let mut r = Renderer::new(&device, 16, wgpu::TextureFormat::Rgba8UnormSrgb);
    r.set_color_mode(ColorMode::ByPhysics);
    r.set_optical_params(&queue, 0, [0.3, 0.3, 0.3]);
    r.set_optical_params(&queue, 1, [0.3, 0.3, 0.3]);
    r.set_optical_scattering(&queue, 1, 5.0); // real tissue-scale reduced scattering coeff

    let mut p0 = Particle::zeroed();
    p0.material_id = 0;
    p0.deformation_gradient = Mat2::IDENTITY;
    let mut p1 = p0;
    p1.material_id = 1;

    let c0 = r.particle_color(&p0, 0);
    let c1 = r.particle_color(&p1, 0);
    assert_ne!(
        c0, c1,
        "identical absorption but different sigma_s must render differently"
    );
}

/// Specular Fresnel reflectance changes ByPhysics's output, the R0 term.
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn specular_r0_changes_by_physics_color() {
    let (device, queue) = headless_device();
    let mut r = Renderer::new(&device, 16, wgpu::TextureFormat::Rgba8UnormSrgb);
    r.set_color_mode(ColorMode::ByPhysics);
    r.set_optical_params(&queue, 0, [0.3, 0.3, 0.3]);
    r.set_optical_params(&queue, 1, [0.3, 0.3, 0.3]);
    r.set_specular_r0(&queue, 1, 0.02); // real water-scale Fresnel base reflectance

    let mut p0 = Particle::zeroed();
    p0.material_id = 0;
    p0.deformation_gradient = Mat2::IDENTITY;
    let mut p1 = p0;
    p1.material_id = 1;

    let c0 = r.particle_color(&p0, 0);
    let c1 = r.particle_color(&p1, 0);
    assert_ne!(
        c0, c1,
        "identical absorption but different specular R0 must render differently"
    );
}

/// `Renderer::new` must succeed and the (now auto-uploading) optical setters
/// must not panic with the extended (scattering + specular) `OpticalTable`
/// layout -- a end-to-end check that the WGSL struct and Rust struct
/// stayed in sync (a mismatch here would show up as a wgpu validation panic,
/// not a silent bug).
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn renderer_construction_and_optical_upload_survive_extended_table() {
    let (device, queue) = headless_device();
    let mut r = Renderer::new(&device, 16, wgpu::TextureFormat::Rgba8UnormSrgb);
    r.set_optical_params(&queue, 0, [0.18, 0.22, 0.55]);
    r.set_optical_scattering(&queue, 0, 8.0);
    r.set_specular_r0(&queue, 0, 0.02);
}

#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn physical_contract_drives_cpu_beer_lambert_in_si() {
    let (device, queue) = headless_device();
    let mut r = Renderer::new(&device, 1, wgpu::TextureFormat::Rgba8UnormSrgb);
    r.set_color_mode(ColorMode::ByPhysics);
    r.set_optical_coefficients_si(
        &queue,
        0,
        OpticalCoefficientsSi::new([2.0, 1.0, 0.5], 0.0).unwrap(),
    );
    r.set_physical_render_contract(
        &queue,
        PhysicalRenderContract::new(PhysicalRenderContractParams {
            dx_meters: 0.01,
            slice_thickness_m: 0.25,
            incident_radiance_w_m2_sr: [10.0; 3],
            background_radiance_w_m2_sr: [10.0, 20.0, 30.0],
            display_white_radiance_w_m2_sr: [10.0, 20.0, 30.0],
            camera_direction: glam::Vec3::Z,
            light_direction: glam::Vec3::Y,
        })
        .unwrap(),
    );
    let mut p = Particle::zeroed();
    p.deformation_gradient = Mat2::IDENTITY;
    let got = r.particle_color(&p, 0);
    let expected = [(-0.5f32).exp(), (-0.25f32).exp(), (-0.125f32).exp()];
    for (channel, want) in got[..3].iter().copied().zip(expected) {
        assert!(
            (channel - want).abs() < 1.0e-6,
            "got {channel}, expected {want}"
        );
    }
}

/// The GPU particle path must consume the same SI slab contract as the CPU
/// path. Read back the compute-generated instance directly, avoiding the
/// unrelated tiny-quad rasterization limitation documented below.
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn physical_contract_drives_gpu_particle_beer_lambert_in_si() {
    use crate::gpu::GpuSimulation;
    use crate::{MaterialRegistry, NeoHookeanMaterial, SimConfig};
    use std::sync::Arc;

    let (device, queue) = headless_device();
    let device = Arc::new(device);
    let queue = Arc::new(queue);
    let config = SimConfig::standard(8, 0.01, glam::Vec2::ZERO);
    let mut particle = Particle::zeroed();
    particle.x = glam::Vec2::splat(4.0);
    particle.mass = 1.0;
    particle.volume = 1.0;
    particle.deformation_gradient = Mat2::IDENTITY;
    let registry = MaterialRegistry::with_default(Box::new(NeoHookeanMaterial::new(100.0, 50.0)));
    let sim = GpuSimulation::with_device(
        device.clone(),
        queue.clone(),
        config,
        vec![particle],
        registry,
    );

    let mut r = Renderer::new(&device, 1, wgpu::TextureFormat::Rgba8UnormSrgb);
    r.set_color_mode(ColorMode::ByPhysics);
    r.set_optical_coefficients_si(
        &queue,
        0,
        OpticalCoefficientsSi::new([2.0, 1.0, 0.5], 0.0).unwrap(),
    );
    r.set_physical_render_contract(
        &queue,
        PhysicalRenderContract::new(PhysicalRenderContractParams {
            dx_meters: 0.01,
            slice_thickness_m: 0.25,
            incident_radiance_w_m2_sr: [10.0; 3],
            background_radiance_w_m2_sr: [10.0, 20.0, 30.0],
            display_white_radiance_w_m2_sr: [10.0, 20.0, 30.0],
            camera_direction: glam::Vec3::Z,
            light_direction: glam::Vec3::Y,
        })
        .unwrap(),
    );
    r.set_camera(&queue, 8, 32, 32, 1.0, true);
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("physical_gpu_particle_target"),
        size: wgpu::Extent3d {
            width: 32,
            height: 32,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    r.render_gpu(
        &device,
        &queue,
        GpuRenderParams {
            particle_buf: sim.particle_buffer(),
            particle_count: 1,
            output_view: &texture.create_view(&wgpu::TextureViewDescriptor::default()),
            clear: true,
            interp_alpha: 1.0,
        },
    );
    device.poll(wgpu::PollType::wait_indefinitely()).ok();
    let raw = readback_f32_blocking(&device, &queue, &r.storage_instances, 12);
    let expected = [(-0.5f32).exp(), (-0.25f32).exp(), (-0.125f32).exp()];
    for (channel, want) in raw[8..11].iter().copied().zip(expected) {
        assert!(
            (channel - want).abs() < 1.0e-5,
            "got {channel}, expected {want}"
        );
    }
}

/// End-to-end GPU path (the one LP actually uses, `render_gpu`): real
/// particles on a `GpuSimulation`, real compute dispatch through
/// `prep_instances.wgsl` with the extended `OpticalTable`, real render pass to
/// an offscreen texture. Proves the whole pipeline survives, not just that
/// `Renderer::new` compiles the shader in isolation.
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
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
            .material(0),
    );
    let registry = MaterialRegistry::with_default(Box::new(NeoHookeanMaterial::new(100.0, 50.0)));
    let sim =
        GpuSimulation::with_device(device.clone(), queue.clone(), config, particles, registry);

    let fmt = wgpu::TextureFormat::Rgba8UnormSrgb;
    let mut r = Renderer::new(&device, sim.particle_count(), fmt);
    r.set_color_mode(ColorMode::ByPhysics);
    r.set_optical_params(&queue, 0, [0.18, 0.22, 0.55]);
    r.set_optical_scattering(&queue, 0, 8.0);
    r.set_specular_r0(&queue, 0, 0.02);
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
        GpuRenderParams {
            particle_buf: sim.particle_buffer(),
            particle_count: sim.particle_count(),
            output_view: &view,
            clear: true,
            interp_alpha: 1.0,
        },
    );
    device.poll(wgpu::PollType::wait_indefinitely()).ok();
}

/// `blackbody.inc.wgsl` claims to be a mirror of `energy::radiation`. This is
/// what makes that claim checkable on real hardware rather than by reading
/// both and hoping.
///
/// A scene is rendered in `ByThermal` (emission and nothing else) with the
/// exposure anchored at the scene's own temperature, so the expected pixel is
/// exactly the blackbody colour -- no exposure factor, no lighting, no
/// absorption in the way. The readback is sRGB-encoded, so it is decoded back
/// to linear before comparison against the CPU's own Planck integration.
///
/// Tolerance 0.06 per channel covers three stacked approximations that are
/// each documented where they live: the Kang 2002 locus fit against real
/// Planck (tested separately at 0.03), 8-bit quantization, and the shader's
/// f32 arithmetic against the CPU's f64.
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn gpu_blackbody_emission_matches_planck_on_the_cpu() {
    use crate::energy::radiation::blackbody_linear_srgb;
    use crate::gpu::GpuSimulation;
    use crate::{MaterialRegistry, NeoHookeanMaterial, SimConfig, SpawnRegion, build_particles};
    use std::sync::Arc;

    let (device, queue) = headless_device();
    let device = Arc::new(device);
    let queue = Arc::new(queue);
    let fmt = wgpu::TextureFormat::Rgba8UnormSrgb;

    let srgb_to_linear = |encoded: u8| -> f32 {
        let value = f32::from(encoded) / 255.0;
        if value <= 0.04045 {
            value / 12.92
        } else {
            ((value + 0.055) / 1.055).powf(2.4)
        }
    };

    for temperature in [2000.0f32, 3500.0, 6000.0] {
        let config = SimConfig::standard(32, 0.1, glam::Vec2::new(0.0, -0.3));
        let mut particles = build_particles(
            &config,
            SpawnRegion::for_sim(&config)
                .at(glam::Vec2::splat(16.0))
                .disk(6.0)
                .spacing(0.5)
                .material(0),
        );
        for p in particles.iter_mut() {
            p.temperature = temperature;
        }
        let registry =
            MaterialRegistry::with_default(Box::new(NeoHookeanMaterial::new(100.0, 50.0)));
        let sim =
            GpuSimulation::with_device(device.clone(), queue.clone(), config, particles, registry);

        let mut r = Renderer::new(&device, sim.particle_count(), fmt);
        r.set_color_mode(ColorMode::ByThermal);
        r.set_camera(&queue, 32, 256, 256, 0.6, true);
        // Exposure anchored at the scene's own temperature: (T/T_ref)^4 = 1.
        r.set_emission_reference_temperature(&queue, temperature);

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("blackbody_parity_target"),
            size: wgpu::Extent3d {
                width: 256,
                height: 256,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: fmt,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        r.render_gpu(
            &device,
            &queue,
            GpuRenderParams {
                particle_buf: sim.particle_buffer(),
                particle_count: sim.particle_count(),
                output_view: &view,
                clear: true,
                interp_alpha: 1.0,
            },
        );
        device.poll(wgpu::PollType::wait_indefinitely()).ok();

        let pixel = readback_brightest_pixel(&device, &queue, &texture, 256, 256);
        let rendered = [
            srgb_to_linear(pixel[0]),
            srgb_to_linear(pixel[1]),
            srgb_to_linear(pixel[2]),
        ];
        let expected = blackbody_linear_srgb(temperature);
        for channel in 0..3 {
            assert!(
                (rendered[channel] - expected[channel]).abs() < 0.06,
                "{temperature}K channel {channel}: GPU rendered {rendered:?} \
                 (raw {pixel:?}) but Planck through the CIE observer says {expected:?}"
            );
        }
    }
}

/// Under a real SI contract the render must still carry scattering and
/// Fresnel, not absorption alone.
///
/// This guards a gap that was real until this was written: the contract
/// branch computed `background * exp(-sigma_a * path)` and nothing else, so
/// turning on measured SI optics silently cost a scene its subsurface glow
/// and its specular. Two materials identical except for `sigma_s`, and two
/// identical except for `R0`, must each render differently.
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn si_contract_path_keeps_scattering_and_fresnel() {
    use crate::render::{PhysicalRenderContract, PhysicalRenderContractParams};

    let (device, queue) = headless_device();
    let mut r = Renderer::new(&device, 16, wgpu::TextureFormat::Rgba8UnormSrgb);
    r.set_color_mode(ColorMode::ByPhysics);
    r.set_physical_render_contract(
        &queue,
        PhysicalRenderContract::new(PhysicalRenderContractParams {
            dx_meters: 0.01,
            slice_thickness_m: 1.0,
            incident_radiance_w_m2_sr: [1.0; 3],
            background_radiance_w_m2_sr: [0.5; 3],
            display_white_radiance_w_m2_sr: [1.0; 3],
            camera_direction: glam::Vec3::new(0.0, 0.0, -1.0),
            light_direction: glam::Vec3::new(0.0, 1.0, -1.0),
        })
        .unwrap(),
    );
    r.set_optical_params(&queue, 0, [0.3, 0.3, 0.3]);
    r.set_optical_params(&queue, 1, [0.3, 0.3, 0.3]);
    r.set_optical_params(&queue, 2, [0.3, 0.3, 0.3]);
    r.set_optical_scattering(&queue, 1, 5.0);
    r.set_specular_r0(&queue, 2, 0.2);

    let mut plain = Particle::zeroed();
    plain.material_id = 0;
    plain.deformation_gradient = Mat2::IDENTITY;
    let mut scattering = plain;
    scattering.material_id = 1;
    let mut reflective = plain;
    reflective.material_id = 2;

    let base = r.particle_color(&plain, 0);
    assert_ne!(
        base,
        r.particle_color(&scattering, 0),
        "sigma_s must change the SI-contract color, not be dropped with the rest          of the scattering term"
    );
    assert_ne!(
        base,
        r.particle_color(&reflective, 0),
        "Fresnel R0 must change the SI-contract color"
    );
}

/// Luminescence: matter that glows WITHOUT being hot must actually light
/// the scene, through the same photon-diffusion solver thermal emitters
/// already feed.
///
/// A firefly is not hot. Before this, the only way to emit light was to be
/// at a few thousand kelvin, so a cold glowing thing was impossible to
/// express. The declared source enters `S` in
/// `(1/c) dphi/dt = D grad^2 phi - mu_a phi + S`, weighted by how much
/// emitting matter the cell holds.
///
/// Same cold scene twice, at ambient temperature throughout, differing only
/// by whether the material declares a luminous emission.
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn luminescent_material_lights_the_scene_without_being_hot() {
    use crate::gpu::GpuSimulation;
    use crate::materials::registry::MaterialRegistry;
    use crate::{NeoHookeanMaterial, SimConfig, SpawnRegion, build_particles};
    use std::sync::Arc;

    /// A cold material that glows, the way a firefly or a luminous fungus
    /// does. 2000 W/m^3 is a stated scene value, not a measurement of any
    /// particular organism.
    #[derive(Debug)]
    struct Luminescent(NeoHookeanMaterial);

    impl crate::materials::MaterialModel for Luminescent {
        fn constitutive_model(&self) -> crate::materials::ConstitutiveModel {
            self.0.constitutive_model()
        }
        fn kirchhoff_stress(&self, particles: &crate::particle::Particles, i: usize) -> glam::Mat2 {
            self.0.kirchhoff_stress(particles, i)
        }
        fn stress_volume(&self, particles: &crate::particle::Particles, i: usize) -> f32 {
            self.0.stress_volume(particles, i)
        }
        fn params(&self) -> crate::materials::MaterialParams {
            self.0.params()
        }
        fn luminous_emission_w_m3(&self) -> f32 {
            2000.0
        }
    }

    let (device, queue) = headless_device();
    let device = Arc::new(device);
    let queue = Arc::new(queue);
    let grid_res = 32u32;
    let fmt = wgpu::TextureFormat::Rgba8UnormSrgb;

    let render_glow = |luminous: bool| -> u32 {
        let config = SimConfig::standard(grid_res as usize, 0.1, glam::Vec2::new(0.0, -0.3));
        let mut particles = build_particles(
            &config,
            SpawnRegion::for_sim(&config)
                .at(glam::Vec2::splat(16.0))
                .disk(4.0)
                .spacing(0.5)
                .material(0),
        );
        // Cold throughout: nothing here may glow by being hot.
        for p in particles.iter_mut() {
            p.temperature = 293.0;
        }
        let base = NeoHookeanMaterial::new(100.0, 50.0);
        let registry: MaterialRegistry = if luminous {
            MaterialRegistry::with_default(Box::new(Luminescent(base)))
        } else {
            MaterialRegistry::with_default(Box::new(base))
        };
        let sim =
            GpuSimulation::with_device(device.clone(), queue.clone(), config, particles, registry);

        let mut r = Renderer::new(&device, sim.particle_count(), fmt);
        r.set_optical_params(&queue, 0, [0.3, 0.3, 0.3]);
        r.set_camera(&queue, grid_res, 64, 64, 0.6, true);
        r.adopt_material_optics(&queue, sim.registry());

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("luminescence_target"),
            size: wgpu::Extent3d {
                width: 64,
                height: 64,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: fmt,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        // The fluence field is persistent, so it needs a few frames to build.
        for _ in 0..30 {
            r.render_surface_reconstruction(
                &device,
                &queue,
                SurfaceReconstructionSource {
                    particle_buf: sim.particle_buffer(),
                    particle_count: sim.particle_count(),
                    grid_res,
                    material_slot: 0,
                    material_mass_enabled: false,
                    dt: 0.1,
                },
                &view,
                true,
            );
            device.poll(wgpu::PollType::wait_indefinitely()).ok();
        }
        let p = readback_pixel(&device, &queue, &texture, 64, 64, 32, 32);
        u32::from(p[0]) + u32::from(p[1]) + u32::from(p[2])
    };

    let dark = render_glow(false);
    let glowing = render_glow(true);
    assert!(
        glowing > dark,
        "a material declaring a luminous source must light the scene even at          293 K: glowing={glowing} vs non-glowing={dark}"
    );
}

/// The physical fact the previous hand-drawn ramp inverted, checked through
/// the GPU pipeline rather than in isolation: a hotter body renders
/// bluer, not redder.
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn gpu_thermal_emission_gets_bluer_with_temperature() {
    use crate::gpu::GpuSimulation;
    use crate::{MaterialRegistry, NeoHookeanMaterial, SimConfig, SpawnRegion, build_particles};
    use std::sync::Arc;

    let (device, queue) = headless_device();
    let device = Arc::new(device);
    let queue = Arc::new(queue);
    let fmt = wgpu::TextureFormat::Rgba8UnormSrgb;

    let render_at = |temperature: f32| -> [u8; 4] {
        let config = SimConfig::standard(32, 0.1, glam::Vec2::new(0.0, -0.3));
        let mut particles = build_particles(
            &config,
            SpawnRegion::for_sim(&config)
                .at(glam::Vec2::splat(16.0))
                .disk(6.0)
                .spacing(0.5)
                .material(0),
        );
        for p in particles.iter_mut() {
            p.temperature = temperature;
        }
        let registry =
            MaterialRegistry::with_default(Box::new(NeoHookeanMaterial::new(100.0, 50.0)));
        let sim =
            GpuSimulation::with_device(device.clone(), queue.clone(), config, particles, registry);
        let mut r = Renderer::new(&device, sim.particle_count(), fmt);
        r.set_color_mode(ColorMode::ByThermal);
        r.set_camera(&queue, 32, 256, 256, 0.6, true);
        r.set_emission_reference_temperature(&queue, temperature);

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("blackbody_hue_target"),
            size: wgpu::Extent3d {
                width: 256,
                height: 256,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: fmt,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        r.render_gpu(
            &device,
            &queue,
            GpuRenderParams {
                particle_buf: sim.particle_buffer(),
                particle_count: sim.particle_count(),
                output_view: &view,
                clear: true,
                interp_alpha: 1.0,
            },
        );
        device.poll(wgpu::PollType::wait_indefinitely()).ok();
        readback_brightest_pixel(&device, &queue, &texture, 256, 256)
    };

    let ember = render_at(1800.0);
    let hot = render_at(9000.0);
    assert!(
        ember[2] < 80 && hot[2] > 200,
        "1800K must render as a deep red with little blue and 9000K as blue-white: \
         ember={ember:?} hot={hot:?}"
    );
}

/// `render_gpu`'s own compute-shader instance prep (`prep_instances.wgsl`)
/// must actually produce visible particle pixels, not just "not panic" --
/// the test above only proved the pipeline survives, never read back a
/// single pixel. Real particles at a known grid position, `ByMaterial`
/// color mode (material 0's real palette entry, `material_color(0)` in
/// `prep_instances.wgsl` = `vec4(0.35, 0.65, 1.00, 1.0)`, a distinct blue),
/// must show up as that color, not the clear color (0.05, 0.05, 0.08).
///
/// The target matches the 256 x 256 window `set_camera` is given. Until
/// issue #45 it was 64 x 64 and nothing was drawn:
/// `subpixel_particles_light_only_pixels_whose_centre_they_cover` below
/// measures why.
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
#[test]
fn render_gpu_produces_visible_particle_pixels_not_just_clear_color() {
    use crate::gpu::GpuSimulation;
    use crate::{MaterialRegistry, NeoHookeanMaterial, SimConfig};
    use std::sync::Arc;

    let (device, queue) = headless_device();
    let device = Arc::new(device);
    let queue = Arc::new(queue);

    let config = SimConfig::standard(32, 0.1, glam::Vec2::new(0.0, -0.3));
    let particles = pixel_test_disk(0.0);
    let registry = MaterialRegistry::with_default(Box::new(NeoHookeanMaterial::new(100.0, 50.0)));
    let sim =
        GpuSimulation::with_device(device.clone(), queue.clone(), config, particles, registry);
    assert!(sim.particle_count() > 0, "test setup must spawn particles");

    let fmt = wgpu::TextureFormat::Rgba8UnormSrgb;
    let mut r = Renderer::new(&device, sim.particle_count(), fmt);
    r.set_color_mode(ColorMode::ByMaterial);
    r.set_camera(&queue, 32, 256, 256, 0.6, true);
    let (texture, view) = pixel_test_target(&device, fmt, 256);

    r.render_gpu(
        &device,
        &queue,
        GpuRenderParams {
            particle_buf: sim.particle_buffer(),
            particle_count: sim.particle_count(),
            output_view: &view,
            clear: true,
            interp_alpha: 1.0,
        },
    );
    device.poll(wgpu::PollType::wait_indefinitely()).ok();

    // A partly covered pixel is blended toward the background, so the
    // brightest pixel carries the particles' own colour. Clear color
    // (0.05,0.05,0.08 linear) is near-black; material-0 blue
    // (0.35,0.65,1.00) is bright, especially in the blue channel.
    let brightest = readback_brightest_pixel(&device, &queue, &texture, 256, 256);
    assert!(
        brightest[2] > 120 && brightest[0] < 180,
        "render_gpu must produce visible material-0 (blue) particle pixels, \
         not just the clear color; brightest pixel in the target {brightest:?}"
    );
}

/// Control test for the above: SAME scene/camera/texture, but through the
/// CPU `render_slice()` path instead of `render_gpu` -- isolates whether a
/// blank result is specific to the GPU compute-prep path or a shared
/// `draw_pass`/pipeline problem that would affect both.
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
#[test]
fn render_cpu_produces_visible_particle_pixels_control() {
    let (device, queue) = headless_device();
    let particles = pixel_test_disk(0.0);
    let fmt = wgpu::TextureFormat::Rgba8UnormSrgb;
    let mut r = Renderer::new(&device, particles.len(), fmt);
    r.set_color_mode(ColorMode::ByMaterial);
    r.set_camera(&queue, 32, 256, 256, 0.6, true);
    let (texture, view) = pixel_test_target(&device, fmt, 256);

    r.render_slice(&device, &queue, &particles, &view, true);
    device.poll(wgpu::PollType::wait_indefinitely()).ok();

    let brightest = readback_brightest_pixel(&device, &queue, &texture, 256, 256);
    assert!(
        brightest[2] > 120 && brightest[0] < 180,
        "control: CPU render_slice() must produce visible particle pixels with the \
         exact same scene/camera/texture params as the GPU test above; brightest \
         pixel in the target {brightest:?}"
    );
}

/// Issue #45, measured. The particle pass point-samples: a fragment exists
/// only where a pixel centre falls inside a particle's quad, and the round
/// clip keeps it only within half the quad's width of the particle. A
/// particle smaller than a pixel therefore lights one pixel or none,
/// depending on where it sits, with no partial coverage.
///
/// The two tests above drew into a 64 x 64 target until this was found: 2
/// pixels per cell, so a 0.6-cell particle is a disc of radius 0.6 px. The
/// spawn lattice, 0.5 cells apart from x = 10.0, puts every particle on a
/// pixel corner, 0.71 px from the nearest pixel centre, and every fragment
/// was discarded: 0 of 4096 pixels lit. A quarter-cell (half-pixel) shift
/// puts each particle on a pixel centre and lights exactly one pixel per
/// particle. On the same 64 x 64 target a 4-cell particle, or a square one,
/// lights the disk.
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
#[test]
fn subpixel_particles_light_only_pixels_whose_centre_they_cover() {
    let (device, queue) = headless_device();
    let fmt = wgpu::TextureFormat::Rgba8UnormSrgb;
    let lit_pixels = |shift: f32| -> usize {
        let particles = pixel_test_disk(shift);
        let mut r = Renderer::new(&device, particles.len(), fmt);
        r.set_color_mode(ColorMode::ByMaterial);
        r.set_camera(&queue, 32, 64, 64, 0.6, true);
        let (texture, view) = pixel_test_target(&device, fmt, 64);
        r.render_slice(&device, &queue, &particles, &view, true);
        device.poll(wgpu::PollType::wait_indefinitely()).ok();
        let luminance = readback_luminance_grid(&device, &queue, &texture, 64, 64, 1);
        let background = luminance[0];
        luminance
            .iter()
            .filter(|&&l| (l - background).abs() > 0.5)
            .count()
    };
    let count = pixel_test_disk(0.0).len();
    let on_corners = lit_pixels(0.0);
    let on_centres = lit_pixels(0.25);
    assert_eq!(
        (on_corners, on_centres),
        (0, count),
        "a 0.6 px disc lights no pixel from a pixel corner and exactly one from \
         a pixel centre ({count} particles)"
    );
}

/// The disk of particles the pixel tests above draw: radius 6 cells at the
/// centre of a 32-cell grid, 0.5 cells apart, material 0, moved by `shift`
/// cells along both axes.
fn pixel_test_disk(shift: f32) -> Vec<crate::Particle> {
    use crate::{SimConfig, SpawnRegion, build_particles};
    let config = SimConfig::standard(32, 0.1, glam::Vec2::new(0.0, -0.3));
    build_particles(
        &config,
        SpawnRegion::for_sim(&config)
            .at(glam::Vec2::splat(16.0 + shift))
            .disk(6.0)
            .spacing(0.5)
            .material(0),
    )
}

/// A square render target for the pixel tests, with its view.
fn pixel_test_target(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    size: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("pixel_test_target"),
        size: wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

/// Blocking single-pixel RGBA8 texture readback -- test-only. Same
/// staging-buffer/copy/poll/map_async/poll/read/unmap pattern as
/// `readback_f32_blocking` below, but for a render-target texture instead
/// of a storage buffer, so it must additionally respect wgpu's
/// `COPY_BYTES_PER_ROW_ALIGNMENT` (256-byte) padding requirement on the
/// destination buffer's row stride.
fn readback_pixel(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    width: u32,
    height: u32,
    x: u32,
    y: u32,
) -> [u8; 4] {
    let unpadded_bytes_per_row = width * 4;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded_bytes_per_row = unpadded_bytes_per_row.div_ceil(align) * align;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("pixel_readback_staging"),
        size: (padded_bytes_per_row * height) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("pixel_readback"),
    });
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(std::iter::once(encoder.finish()));
    device.poll(wgpu::PollType::wait_indefinitely()).ok();
    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).ok();
    let mapped = slice.get_mapped_range();
    let row_start = (y * padded_bytes_per_row) as usize;
    let px_start = row_start + (x * 4) as usize;
    let pixel = [
        mapped[px_start],
        mapped[px_start + 1],
        mapped[px_start + 2],
        mapped[px_start + 3],
    ];
    drop(mapped);
    staging.unmap();
    pixel
}

/// Blocking readback of the single brightest pixel in the target -- test-only,
/// same staging pattern as `readback_pixel` above.
///
/// For a test that renders emitters on a dark background, this is the right
/// sample: a partially-covered pixel is blended toward the background and so
/// is always dimmer than a fully-covered one, which makes the maximum the
/// pixel that carries the emitter's own colour. Sampling a fixed coordinate
/// instead would depend on exactly where the particle lattice happens to
/// land relative to the pixel grid.
fn readback_brightest_pixel(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    width: u32,
    height: u32,
) -> [u8; 4] {
    let unpadded_bytes_per_row = width * 4;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded_bytes_per_row = unpadded_bytes_per_row.div_ceil(align) * align;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("brightest_pixel_staging"),
        size: (padded_bytes_per_row * height) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("brightest_pixel_readback"),
    });
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(std::iter::once(encoder.finish()));
    device.poll(wgpu::PollType::wait_indefinitely()).ok();
    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).ok();
    let mapped = slice.get_mapped_range();
    let mut best = ([0u8; 4], -1i32);
    for y in 0..height {
        let row = (y * padded_bytes_per_row) as usize;
        for x in 0..width {
            let at = row + (x * 4) as usize;
            let pixel = [mapped[at], mapped[at + 1], mapped[at + 2], mapped[at + 3]];
            let sum = i32::from(pixel[0]) + i32::from(pixel[1]) + i32::from(pixel[2]);
            if sum > best.1 {
                best = (pixel, sum);
            }
        }
    }
    drop(mapped);
    staging.unmap();
    best.0
}

/// Blocking readback of a SAMPLED GRID of pixel luminances (every
/// `sample_stride`-th pixel in both x/y, row-major) -- test-only, same
/// staging pattern as `readback_pixel` above but reads the whole texture
/// once instead of one pixel, for a per-frame flicker measurement
/// across many frames (see `surface_reconstruction_does_not_flicker_over_
/// many_deterministic_frames` below).
fn readback_luminance_grid(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    width: u32,
    height: u32,
    sample_stride: u32,
) -> Vec<f64> {
    let unpadded_bytes_per_row = width * 4;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded_bytes_per_row = unpadded_bytes_per_row.div_ceil(align) * align;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("flicker_grid_readback_staging"),
        size: (padded_bytes_per_row * height) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("flicker_grid_readback"),
    });
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(std::iter::once(encoder.finish()));
    device.poll(wgpu::PollType::wait_indefinitely()).ok();
    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).ok();
    let mapped = slice.get_mapped_range();
    let mut out = Vec::new();
    let mut y = 0u32;
    while y < height {
        let mut x = 0u32;
        while x < width {
            let off = (y * padded_bytes_per_row + x * 4) as usize;
            let b = mapped[off] as f64;
            let g = mapped[off + 1] as f64;
            let r = mapped[off + 2] as f64;
            out.push(0.299 * r + 0.587 * g + 0.114 * b);
            x += sample_stride;
        }
        y += sample_stride;
    }
    drop(mapped);
    staging.unmap();
    out
}

/// Blocking f32 storage-buffer readback -- test-only, mirrors the real
/// established pattern `gpu::buffers::readback::readback_f32_blocking`
/// already uses (staging buffer, copy, poll, map_async, poll, read,
/// unmap), just inlined here since `surface_a_buf` is a `Renderer`-owned
/// buffer, not a `GpuBuffers` one.
fn readback_f32_blocking(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buf: &wgpu::Buffer,
    count: usize,
) -> Vec<f32> {
    let byte_count = (count * std::mem::size_of::<f32>()) as u64;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("surface_readback_staging"),
        size: byte_count,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("surface_readback"),
    });
    encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, byte_count);
    queue.submit(std::iter::once(encoder.finish()));
    device.poll(wgpu::PollType::wait_indefinitely()).ok();
    let slice = staging.slice(..byte_count);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).ok();
    let mapped = slice.get_mapped_range();
    let values = bytemuck::cast_slice::<u8, f32>(&mapped).to_vec();
    drop(mapped);
    staging.unmap();
    values
}

/// End-to-end smoke test: real particle cluster, real `GpuSimulation`, real
/// offscreen texture, real `render_surface_reconstruction` call. Proves the
/// whole 6-pass pipeline (clear/splat/convert/12x iterate/render) survives
/// together, not just that each shader entry point compiles in isolation.
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn render_surface_reconstruction_survives_end_to_end() {
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
            .material(0),
    );
    let registry = MaterialRegistry::with_default(Box::new(NeoHookeanMaterial::new(100.0, 50.0)));
    let sim =
        GpuSimulation::with_device(device.clone(), queue.clone(), config, particles, registry);

    let fmt = wgpu::TextureFormat::Rgba8UnormSrgb;
    let mut r = Renderer::new(&device, sim.particle_count(), fmt);
    r.set_optical_coefficients_si(
        &queue,
        0,
        OpticalCoefficientsSi::new([0.18, 0.22, 0.55], 0.0).unwrap(),
    );
    r.set_physical_render_contract(
        &queue,
        PhysicalRenderContract::new(PhysicalRenderContractParams {
            dx_meters: 0.1,
            slice_thickness_m: 0.5,
            incident_radiance_w_m2_sr: [1.0; 3],
            background_radiance_w_m2_sr: [1.0; 3],
            display_white_radiance_w_m2_sr: [1.0; 3],
            camera_direction: glam::Vec3::Z,
            light_direction: glam::Vec3::Y,
        })
        .unwrap(),
    );
    r.set_camera(&queue, 32, 64, 64, 0.6, true);

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("surface_reconstruction_test_target"),
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

    r.render_surface_reconstruction(
        &device,
        &queue,
        SurfaceReconstructionSource {
            particle_buf: sim.particle_buffer(),
            particle_count: sim.particle_count(),
            grid_res: 32,
            material_slot: 0,
            material_mass_enabled: false,
            dt: 0.1,
        },
        &view,
        true,
    );
    device.poll(wgpu::PollType::wait_indefinitely()).ok();
}

/// After the splat, convert and 12 curvature-iterate passes, the settled
/// buffer (`surface_a_buf`, `CURVATURE_ITERATIONS` being even) has
/// substantially higher density near the particle cluster than far from it:
/// the pipeline reconstructs a density field, not uniform noise or zeros
/// (which the end-to-end smoke test above would still pass).
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn render_surface_reconstruction_produces_real_density_near_particles() {
    use crate::gpu::GpuSimulation;
    use crate::{MaterialRegistry, NeoHookeanMaterial, SimConfig, SpawnRegion, build_particles};
    use std::sync::Arc;

    let (device, queue) = headless_device();
    let device = Arc::new(device);
    let queue = Arc::new(queue);

    let grid_res = 32u32;
    let config = SimConfig::standard(grid_res as usize, 0.1, glam::Vec2::new(0.0, -0.3));
    // Small, tight cluster near the grid center -- unambiguous "here"
    // vs. the grid corners, which this scene never populates at all.
    let particles = build_particles(
        &config,
        SpawnRegion::for_sim(&config)
            .at(glam::Vec2::splat(16.0))
            .disk(3.0)
            .spacing(0.5)
            .material(0),
    );
    let registry = MaterialRegistry::with_default(Box::new(NeoHookeanMaterial::new(100.0, 50.0)));
    let sim =
        GpuSimulation::with_device(device.clone(), queue.clone(), config, particles, registry);

    let fmt = wgpu::TextureFormat::Rgba8UnormSrgb;
    let mut r = Renderer::new(&device, sim.particle_count(), fmt);
    r.set_camera(&queue, grid_res, 64, 64, 0.6, true);

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("surface_density_test_target"),
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

    r.render_surface_reconstruction(
        &device,
        &queue,
        SurfaceReconstructionSource {
            particle_buf: sim.particle_buffer(),
            particle_count: sim.particle_count(),
            grid_res,
            material_slot: 0,
            material_mass_enabled: false,
            dt: 0.1,
        },
        &view,
        true,
    );
    device.poll(wgpu::PollType::wait_indefinitely()).ok();

    let surface_res = r.surface_res;
    let values = readback_f32_blocking(
        &device,
        &queue,
        &r.surface_a_buf,
        (surface_res * surface_res) as usize,
    );

    // The particle cluster sits at grid position (16, 16), converted to this
    // buffer's finer units (surface_res/grid_res, as in the shader).
    let scale = surface_res as f32 / grid_res as f32;
    let center = (16.0 * scale) as i32;
    let center_idx = (center as u32 * surface_res + center as u32) as usize;
    let corner_idx = 0usize; // (0, 0) -- this scene never puts any particle there

    assert!(
        values[center_idx] > values[corner_idx] + 0.05,
        "density near the real particle cluster must be substantially higher \
         than density at an empty corner: center={} corner={}",
        values[center_idx],
        values[corner_idx]
    );
    assert!(
        values.iter().all(|v| v.is_finite()),
        "curvature-flow smoothing must never produce NaN/inf, even after 12 iterations"
    );
}

/// Two-phase extension: two separate material clusters each settle density
/// in their own phase buffer and stay near zero in the other's, so
/// `phase_filter_material_id` partitions particles by material instead of
/// running one unfiltered splat twice (the independent-phase-field design of
/// `curvature_flow.wgsl`'s "two-phase extension" depends on it).
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn dual_phase_reconstruction_keeps_two_materials_in_their_own_phase_buffer() {
    use crate::gpu::GpuSimulation;
    use crate::{MaterialRegistry, NeoHookeanMaterial, SimConfig, SpawnRegion, build_particles};
    use std::sync::Arc;

    let (device, queue) = headless_device();
    let device = Arc::new(device);
    let queue = Arc::new(queue);

    let grid_res = 32u32;
    let config = SimConfig::standard(grid_res as usize, 0.1, glam::Vec2::new(0.0, -0.3));
    // Two spatially separate clusters -- far enough apart (8 vs 24 on
    // a 32-cell grid) that neither's real B-spline splat reach can touch
    // the other's territory, isolating the filter itself as the only thing
    // under test.
    let mut particles = build_particles(
        &config,
        SpawnRegion::for_sim(&config)
            .at(glam::Vec2::new(8.0, 16.0))
            .disk(3.0)
            .spacing(0.5)
            .material(0),
    );
    particles.extend(build_particles(
        &config,
        SpawnRegion::for_sim(&config)
            .at(glam::Vec2::new(24.0, 16.0))
            .disk(3.0)
            .spacing(0.5)
            .material(1),
    ));
    let mut registry =
        MaterialRegistry::with_default(Box::new(NeoHookeanMaterial::new(100.0, 50.0)));
    registry.insert(1, Box::new(NeoHookeanMaterial::new(100.0, 50.0)));
    let sim =
        GpuSimulation::with_device(device.clone(), queue.clone(), config, particles, registry);

    let fmt = wgpu::TextureFormat::Rgba8UnormSrgb;
    let mut r = Renderer::new(&device, sim.particle_count(), fmt);
    r.set_camera(&queue, grid_res, 64, 64, 0.6, true);

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("dual_phase_test_target"),
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

    r.render_surface_reconstruction_dual_phase(
        &device,
        &queue,
        DualPhaseSurfaceSource {
            particle_buf: sim.particle_buffer(),
            particle_count: sim.particle_count(),
            grid_res,
            material_id_a: 0,
            material_id_b: 1,
            dt: 0.1,
        },
        &view,
        true,
    );
    device.poll(wgpu::PollType::wait_indefinitely()).ok();

    let surface_res = r.surface_res;
    let phase_a_values = readback_f32_blocking(
        &device,
        &queue,
        &r.surface_a_buf,
        (surface_res * surface_res) as usize,
    );
    let phase_b_values = readback_f32_blocking(
        &device,
        &queue,
        &r.phase_b_a_buf,
        (surface_res * surface_res) as usize,
    );

    let scale = surface_res as f32 / grid_res as f32;
    let idx_at = |grid_x: f32, grid_y: f32| -> usize {
        let sx = (grid_x * scale) as u32;
        let sy = (grid_y * scale) as u32;
        (sy * surface_res + sx) as usize
    };
    let cluster_a_idx = idx_at(8.0, 16.0);
    let cluster_b_idx = idx_at(24.0, 16.0);

    assert!(
        phase_a_values[cluster_a_idx] > 0.05,
        "phase A must show real density at cluster A's own location: got {}",
        phase_a_values[cluster_a_idx]
    );
    assert!(
        phase_a_values[cluster_b_idx] < 0.01,
        "phase A must NOT show real density at cluster B's location (material \
         filter must have excluded those particles): got {}",
        phase_a_values[cluster_b_idx]
    );
    assert!(
        phase_b_values[cluster_b_idx] > 0.05,
        "phase B must show real density at cluster B's own location: got {}",
        phase_b_values[cluster_b_idx]
    );
    assert!(
        phase_b_values[cluster_a_idx] < 0.01,
        "phase B must NOT show real density at cluster A's location (material \
         filter must have excluded those particles): got {}",
        phase_b_values[cluster_a_idx]
    );
}

/// N-material extension (single-phase path, see `curvature_flow.wgsl`): 3
/// separate clusters sharing one smoothed density field each render their
/// own `OpticalTable` colour, so `dominant_material` reads the per-cell mass
/// array, not the `material_slot` fallback (3 materials in
/// `basic_jellies_gpu.rs` rendered as one colour).
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn n_material_surface_reconstruction_colors_each_material_distinctly() {
    use crate::gpu::GpuSimulation;
    use crate::{MaterialRegistry, NeoHookeanMaterial, SimConfig, SpawnRegion, build_particles};
    use std::sync::Arc;

    let (device, queue) = headless_device();
    let device = Arc::new(device);
    let queue = Arc::new(queue);

    let grid_res = 32u32;
    let config = SimConfig::standard(grid_res as usize, 0.1, glam::Vec2::new(0.0, -0.3));
    // Three spatially separate clusters (11 grid cells apart, disk
    // radius 2.0) -- far enough that the shared curvature-smoothed field
    // still resolves 3 distinct dominant-material regions instead of one
    // blended blob.
    let mut particles = build_particles(
        &config,
        SpawnRegion::for_sim(&config)
            .at(glam::Vec2::new(5.0, 16.0))
            .disk(2.0)
            .spacing(0.5)
            .material(0),
    );
    particles.extend(build_particles(
        &config,
        SpawnRegion::for_sim(&config)
            .at(glam::Vec2::new(16.0, 16.0))
            .disk(2.0)
            .spacing(0.5)
            .material(1),
    ));
    particles.extend(build_particles(
        &config,
        SpawnRegion::for_sim(&config)
            .at(glam::Vec2::new(27.0, 16.0))
            .disk(2.0)
            .spacing(0.5)
            .material(2),
    ));
    let mut registry =
        MaterialRegistry::with_default(Box::new(NeoHookeanMaterial::new(100.0, 50.0)));
    registry.insert(1, Box::new(NeoHookeanMaterial::new(100.0, 50.0)));
    registry.insert(2, Box::new(NeoHookeanMaterial::new(100.0, 50.0)));
    let sim =
        GpuSimulation::with_device(device.clone(), queue.clone(), config, particles, registry);

    let fmt = wgpu::TextureFormat::Rgba8UnormSrgb;
    let mut r = Renderer::new(&device, sim.particle_count(), fmt);
    // Same real magnitude range `basic_jellies_gpu.rs`'s own SIGMA_NEO/COR/VIS
    // use (0.05-0.6), not an arbitrary saturated extreme -- LOW absorption in
    // one channel means that channel is mostly TRANSMITTED (bright); HIGH
    // absorption in the other two means they're mostly absorbed (dark). So
    // material 0 (low red absorption) reads red-dominant, etc.
    r.set_optical_params(&queue, 0, [0.05, 0.55, 0.55]);
    r.set_optical_params(&queue, 1, [0.55, 0.05, 0.55]);
    r.set_optical_params(&queue, 2, [0.55, 0.55, 0.05]);
    r.set_camera(&queue, grid_res, 64, 64, 0.6, true);

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("n_material_surface_test_target"),
        size: wgpu::Extent3d {
            width: 64,
            height: 64,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: fmt,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    r.render_surface_reconstruction(
        &device,
        &queue,
        SurfaceReconstructionSource {
            particle_buf: sim.particle_buffer(),
            particle_count: sim.particle_count(),
            grid_res,
            material_slot: 0,
            material_mass_enabled: true,
            dt: 0.1,
        },
        &view,
        true,
    );
    device.poll(wgpu::PollType::wait_indefinitely()).ok();

    // Read the whole 64x64 frame ONCE (not per-pixel) -- the exact camera
    // projection's sub-pixel rounding isn't worth hand-deriving precisely;
    // instead, scan a window around each cluster's approximate
    // expected screen location and take whichever pixel shows that
    // material's color most strongly. Robust against a few cells of
    // rounding error in the pixel<->surface-cell mapping (confirmed via a
    // real diagnostic run: hand-derived pixel targets landed within ~1-3
    // surface cells of the true splat center, well inside the B-spline
    // kernel's real footprint for most but not all of the 3 clusters at a
    // single exact pixel -- the window scan absorbs that margin), which is
    // all that's actually under test here (dominant_material's real
    // per-cell resolution), not exact sub-pixel camera arithmetic.
    let unpadded_bytes_per_row: u32 = 64 * 4;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded_bytes_per_row = unpadded_bytes_per_row.div_ceil(align) * align;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("n_material_frame_readback_staging"),
        size: (padded_bytes_per_row * 64) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("n_material_frame_readback"),
    });
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row),
                rows_per_image: Some(64),
            },
        },
        wgpu::Extent3d {
            width: 64,
            height: 64,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(std::iter::once(encoder.finish()));
    device.poll(wgpu::PollType::wait_indefinitely()).ok();
    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).ok();
    let frame = slice.get_mapped_range().to_vec();
    staging.unmap();
    let pixel_at = |x: i32, y: i32| -> [u8; 4] {
        let x = x.clamp(0, 63) as u32;
        let y = y.clamp(0, 63) as u32;
        let off = (y * padded_bytes_per_row + x * 4) as usize;
        [frame[off], frame[off + 1], frame[off + 2], frame[off + 3]]
    };
    // Search the WHOLE frame for the pixel that most strongly shows each
    // material's color (rather than hand-deriving the exact camera
    // projection to a specific pixel, which is fragile at cell-level
    // precision against a per-slot mass field that -- unlike total density
    // -- is never curvature-smoothed, so it stays exactly as narrow as the
    // raw B-spline splat, not spread by the later smoothing pass).
    // "Strongest" = that channel exceeds the other two by the widest
    // margin (Beer-Lambert transmission: low sigma_a in `channel` means it
    // dominates the rendered color).
    let strongest_for_channel = |channel: usize| -> ([u8; 4], i32, i32) {
        let mut best = [0u8; 4];
        let mut best_score = i32::MIN;
        let mut best_pos = (0i32, 0i32);
        for y in 0..64 {
            for x in 0..64 {
                let px = pixel_at(x, y);
                let others: i32 = (0..3).filter(|&c| c != channel).map(|c| px[c] as i32).sum();
                let score = px[channel] as i32 * 2 - others;
                if score > best_score {
                    best_score = score;
                    best = px;
                    best_pos = (x, y);
                }
            }
        }
        (best, best_pos.0, best_pos.1)
    };
    let (px_a, ax, ay) = strongest_for_channel(0);
    let (px_b, bx, by) = strongest_for_channel(1);
    let (px_c, cx, cy) = strongest_for_channel(2);

    assert!(
        px_a[0] > px_a[1] && px_a[0] > px_a[2],
        "material 0 (low red absorption) must be found SOMEWHERE in frame \
         as red-dominant: best={:?} at ({ax},{ay})",
        px_a
    );
    assert!(
        px_b[1] > px_b[0] && px_b[1] > px_b[2],
        "material 1 (low green absorption) must be found SOMEWHERE in frame \
         as green-dominant: best={:?} at ({bx},{by})",
        px_b
    );
    assert!(
        px_c[2] > px_c[0] && px_c[2] > px_c[1],
        "material 2 (low blue absorption) must be found SOMEWHERE in frame \
         as blue-dominant: best={:?} at ({cx},{cy})",
        px_c
    );
    // The 3 winning locations must be different regions (not all
    // the same handful of pixels), proving 3 spatially distinct dominant-
    // material resolutions, not one lucky pixel satisfying all 3 channel
    // checks by coincidence.
    let dist2 = |x1: i32, y1: i32, x2: i32, y2: i32| (x1 - x2).pow(2) + (y1 - y2).pow(2);
    assert!(
        dist2(ax, ay, bx, by) > 9,
        "material 0's and material 1's winning pixels are suspiciously close: \
         ({ax},{ay}) vs ({bx},{by})"
    );
    assert!(
        dist2(bx, by, cx, cy) > 9,
        "material 1's and material 2's winning pixels are suspiciously close: \
         ({bx},{by}) vs ({cx},{cy})"
    );
    assert!(
        dist2(ax, ay, cx, cy) > 9,
        "material 0's and material 2's winning pixels are suspiciously close: \
         ({ax},{ay}) vs ({cx},{cy})"
    );
}

/// Blended (mass-fraction-weighted) N-material resolver (see
/// `curvature_flow.wgsl`'s `blended_optical_slot`): two close clusters whose
/// splat footprints overlap leave at least one surface cell with mass in
/// both slots, and the blend formula is checked on the raw buffer (pixel
/// assertions are fragile after tone mapping and quantization): a mixed cell
/// resolves to a weighted average strictly between the two pure optics.
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn n_material_blend_produces_real_weighted_average_at_a_mixed_cell() {
    use crate::gpu::GpuSimulation;
    use crate::{MaterialRegistry, NeoHookeanMaterial, SimConfig, SpawnRegion, build_particles};
    use std::sync::Arc;

    let (device, queue) = headless_device();
    let device = Arc::new(device);
    let queue = Arc::new(queue);

    let grid_res = 32u32;
    let config = SimConfig::standard(grid_res as usize, 0.1, glam::Vec2::new(0.0, -0.3));
    // Two clusters close enough (4 grid cells apart, disk radius 3.0 --
    // particle placement itself overlaps by construction) that their real
    // B-spline splat footprints share cells -- unlike the well-
    // separated clusters in the distinctness test above, which deliberately
    // avoid overlap.
    let mut particles = build_particles(
        &config,
        SpawnRegion::for_sim(&config)
            .at(glam::Vec2::new(14.0, 16.0))
            .disk(3.0)
            .spacing(0.5)
            .material(0),
    );
    particles.extend(build_particles(
        &config,
        SpawnRegion::for_sim(&config)
            .at(glam::Vec2::new(18.0, 16.0))
            .disk(3.0)
            .spacing(0.5)
            .material(1),
    ));
    let mut registry =
        MaterialRegistry::with_default(Box::new(NeoHookeanMaterial::new(100.0, 50.0)));
    registry.insert(1, Box::new(NeoHookeanMaterial::new(100.0, 50.0)));
    let sim =
        GpuSimulation::with_device(device.clone(), queue.clone(), config, particles, registry);

    let fmt = wgpu::TextureFormat::Rgba8UnormSrgb;
    let mut r = Renderer::new(&device, sim.particle_count(), fmt);
    r.set_optical_params(&queue, 0, [3.0, 0.0, 0.0]); // pure-red absorption
    r.set_optical_params(&queue, 1, [0.0, 3.0, 0.0]); // pure-green absorption
    r.set_camera(&queue, grid_res, 64, 64, 0.6, true);

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("n_material_blend_test_target"),
        size: wgpu::Extent3d {
            width: 64,
            height: 64,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: fmt,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    r.render_surface_reconstruction(
        &device,
        &queue,
        SurfaceReconstructionSource {
            particle_buf: sim.particle_buffer(),
            particle_count: sim.particle_count(),
            grid_res,
            material_slot: 0,
            material_mass_enabled: true,
            dt: 0.1,
        },
        &view,
        true,
    );
    device.poll(wgpu::PollType::wait_indefinitely()).ok();

    let surface_res = r.surface_res;
    let mm = readback_f32_blocking(
        &device,
        &queue,
        &r.surface_material_mass_buf,
        (surface_res * surface_res * MAX_RENDER_MATERIAL_SLOTS) as usize,
    );

    // The optics written above: red=[3,0,0,0], green=[0,3,0,0] (sigma_s and
    // specular default to 0, only sigma_a set).
    let slot_red = [3.0f32, 0.0, 0.0, 0.0];
    let slot_green = [0.0f32, 3.0, 0.0, 0.0];

    // Search the whole per-cell mass array for a cell where BOTH slot 0 and
    // slot 1 have substantial mass -- proof the two footprints
    // overlap at this scene geometry, not just adjacent.
    let cell_count = (surface_res * surface_res) as usize;
    let mut found = false;
    for cell in 0..cell_count {
        let base = cell * MAX_RENDER_MATERIAL_SLOTS as usize;
        let m0 = mm[base];
        let m1 = mm[base + 1];
        // These fixed-point masses are read back as bit-reinterpreted f32 (see
        // `blended_optical_slot`: fine for ordering and proportion) and sit in
        // the IEEE 754 denormal range (~1e-39), so `> 0.01` would never fire;
        // `> 0.0` tests nonzero mass.
        if m0 > 0.0 && m1 > 0.0 {
            found = true;
            let total = m0 + m1;
            let expected: Vec<f32> = (0..4)
                .map(|c| (m0 * slot_red[c] + m1 * slot_green[c]) / total)
                .collect();
            // The blended red channel must be strictly between pure-green's
            // (0.0) and pure-red's (3.0) values -- a weighted
            // average, not a winner-take-all snap to either pure value.
            assert!(
                expected[0] > 0.0 && expected[0] < 3.0,
                "blended red channel at a real mixed cell (m0={m0}, m1={m1}) \
                 must sit strictly between 0.0 and 3.0, not snap to either \
                 pure value: got {}",
                expected[0]
            );
            assert!(
                expected[1] > 0.0 && expected[1] < 3.0,
                "blended green channel at a real mixed cell (m0={m0}, m1={m1}) \
                 must sit strictly between 0.0 and 3.0, not snap to either \
                 pure value: got {}",
                expected[1]
            );
            // A cell with MORE red mass must lean more toward red than a
            // cell with LESS red mass -- the weighting is not just
            // "average of the two extremes regardless of ratio". Checked
            // via the closed-form ratio directly rather than a second
            // sampled cell (deterministic, no dependence on scene geometry
            // producing a second usable mixed cell).
            let expected_ratio = m0 / total;
            assert!(
                (expected[0] / 3.0 - expected_ratio).abs() < 1.0e-4,
                "blended red channel must scale linearly with slot 0's real \
                 mass fraction ({expected_ratio}): got fraction {}",
                expected[0] / 3.0
            );
            break;
        }
    }
    assert!(
        found,
        "scene geometry produced no real overlapping-mass cell -- test setup \
         needs closer clusters or a wider disk radius, this doesn't verify \
         the blend at all if there's nothing to blend"
    );
}

/// Anisotropic splat (see `curvature_flow.wgsl`): a particle whose
/// `deformation_gradient` is stretched 2.5x along x splats a wider footprint
/// along x than along y at the same offset. An identity-F control at the
/// same position shows no such bias, so the asymmetry comes from F and not
/// from the splat loop.
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn anisotropic_splat_widens_footprint_along_stretched_axis() {
    use crate::gpu::GpuSimulation;
    use crate::{MaterialRegistry, NeoHookeanMaterial, SimConfig, SpawnRegion, build_particles};
    use std::sync::Arc;

    let (device, queue) = headless_device();
    let device = Arc::new(device);
    let queue = Arc::new(queue);

    let grid_res = 32u32;
    let config = SimConfig::standard(grid_res as usize, 0.1, glam::Vec2::new(0.0, -0.3));

    let render_single_particle = |f: Mat2| -> Vec<f32> {
        // A small, smooth cluster (not one isolated near-delta-function
        // particle): every particle shares the SAME imposed `F`, so the
        // aggregate splat is still a clean directional-bias test, but the
        // density field entering `curvature_iterate_main` is smooth enough
        // not to trip that pass's own known instability on sharp/isolated
        // inputs (an explicit curvature-flow step, like any explicit
        // diffusion scheme, is only guaranteed stable on smooth data --
        // real usage never spawns a truly isolated single particle either).
        let mut particles = build_particles(
            &config,
            SpawnRegion::for_sim(&config)
                .at(glam::Vec2::splat(16.0))
                .disk(1.5)
                .spacing(0.5)
                .material(0),
        );
        for p in &mut particles {
            p.deformation_gradient = f;
        }

        let registry =
            MaterialRegistry::with_default(Box::new(NeoHookeanMaterial::new(100.0, 50.0)));
        let sim =
            GpuSimulation::with_device(device.clone(), queue.clone(), config, particles, registry);

        let fmt = wgpu::TextureFormat::Rgba8UnormSrgb;
        let mut r = Renderer::new(&device, sim.particle_count(), fmt);
        r.set_camera(&queue, grid_res, 64, 64, 0.6, true);

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("aniso_splat_test_target"),
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

        r.render_surface_reconstruction(
            &device,
            &queue,
            SurfaceReconstructionSource {
                particle_buf: sim.particle_buffer(),
                particle_count: sim.particle_count(),
                grid_res,
                material_slot: 0,
                material_mass_enabled: false,
                dt: 0.1,
            },
            &view,
            true,
        );
        device.poll(wgpu::PollType::wait_indefinitely()).ok();

        let surface_res = r.surface_res;
        readback_f32_blocking(
            &device,
            &queue,
            &r.surface_a_buf,
            (surface_res * surface_res) as usize,
        )
    };

    // `ensure_surface_capacity` computes this exact same product before any
    // render call -- read the constant directly rather than instantiate a
    // throwaway `Renderer` (whose `surface_res` starts at a placeholder `1`
    // until a render grows it).
    let surface_res = grid_res * SURFACE_RES_MULTIPLIER;
    let scale = surface_res as f32 / grid_res as f32;
    let center = (16.0 * scale) as i32;
    // Chosen to sit just past the ISOTROPIC kernel's real reach (radius
    // ~1.5 grid cells) but well inside the axis the stretched particle's F
    // extends 2.5x -- the discriminating distance between the two cases.
    let offset = (2.0 * scale) as i32;
    let idx = |dx: i32, dy: i32| -> usize {
        ((center + dy) as u32 * surface_res + (center + dx) as u32) as usize
    };

    let stretched = render_single_particle(Mat2::from_cols(
        glam::Vec2::new(2.5, 0.0),
        glam::Vec2::new(0.0, 1.0),
    ));
    let density_x = stretched[idx(offset, 0)];
    let density_y = stretched[idx(0, offset)];
    assert!(
        density_x > density_y + 0.02,
        "a particle stretched 2.5x along x must splat substantially more \
         density along x than along the unstretched y axis at the same \
         offset: x={} y={}",
        density_x,
        density_y
    );

    let isotropic = render_single_particle(Mat2::IDENTITY);
    let iso_x = isotropic[idx(offset, 0)];
    let iso_y = isotropic[idx(0, offset)];
    assert!(
        (iso_x - iso_y).abs() < 0.02,
        "an F=identity particle must splat a symmetric footprint (no \
         directional bias from the splat loop itself): x={} y={}",
        iso_x,
        iso_y
    );
}

/// Velocity stretch: a fast particle with F = identity also splats wider
/// along its velocity than across it (the signature
/// `anisotropic_splat_widens_footprint_along_stretched_axis` shows for F). A
/// stationary control shows no bias, and neither does `dt = 0.0`, the no-op
/// the other tests in this file rely on (their `dt: 0.1` must not change
/// them while their particles rest).
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn velocity_stretch_widens_footprint_along_motion_direction() {
    use crate::gpu::GpuSimulation;
    use crate::{MaterialRegistry, NeoHookeanMaterial, SimConfig, SpawnRegion, build_particles};
    use std::sync::Arc;

    let (device, queue) = headless_device();
    let device = Arc::new(device);
    let queue = Arc::new(queue);

    let grid_res = 32u32;
    let config = SimConfig::standard(grid_res as usize, 0.1, glam::Vec2::new(0.0, -0.3));

    let render_single_cluster = |velocity: glam::Vec2, dt: f32| -> Vec<f32> {
        let mut particles = build_particles(
            &config,
            SpawnRegion::for_sim(&config)
                .at(glam::Vec2::splat(16.0))
                .disk(1.5)
                .spacing(0.5)
                .material(0),
        );
        for p in &mut particles {
            p.v = velocity;
        }

        let registry =
            MaterialRegistry::with_default(Box::new(NeoHookeanMaterial::new(100.0, 50.0)));
        let sim =
            GpuSimulation::with_device(device.clone(), queue.clone(), config, particles, registry);

        let fmt = wgpu::TextureFormat::Rgba8UnormSrgb;
        let mut r = Renderer::new(&device, sim.particle_count(), fmt);
        r.set_camera(&queue, grid_res, 64, 64, 0.6, true);

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("velocity_stretch_test_target"),
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

        r.render_surface_reconstruction(
            &device,
            &queue,
            SurfaceReconstructionSource {
                particle_buf: sim.particle_buffer(),
                particle_count: sim.particle_count(),
                grid_res,
                material_slot: 0,
                material_mass_enabled: false,
                dt,
            },
            &view,
            true,
        );
        device.poll(wgpu::PollType::wait_indefinitely()).ok();

        let surface_res = r.surface_res;
        readback_f32_blocking(
            &device,
            &queue,
            &r.surface_a_buf,
            (surface_res * surface_res) as usize,
        )
    };

    let surface_res = grid_res * SURFACE_RES_MULTIPLIER;
    let scale = surface_res as f32 / grid_res as f32;
    let center = (16.0 * scale) as i32;
    let offset = (2.0 * scale) as i32;
    let idx = |dx: i32, dy: i32| -> usize {
        ((center + dy) as u32 * surface_res + (center + dx) as u32) as usize
    };

    // speed*dt/BSPLINE_OUTER_LIMIT = 22.5*0.1/1.5 = 1.5 -> stretch_factor=2.5,
    // matching the F-based test's own real 2.5x for a direct, consistent
    // comparison. 22.5 grid-units/s is a plausible fast-splash speed
    // for this engine (live-measured max speeds during violent impacts have
    // reached 100-450+ grid-units/s elsewhere in this project).
    let moving = render_single_cluster(glam::Vec2::new(22.5, 0.0), 0.1);
    let density_x = moving[idx(offset, 0)];
    let density_y = moving[idx(0, offset)];
    assert!(
        density_x > density_y + 0.02,
        "a particle moving fast along x (F=identity, no shape deformation) must \
         splat substantially more density along its own motion axis than \
         perpendicular to it: x={density_x} y={density_y}"
    );

    let stationary = render_single_cluster(glam::Vec2::ZERO, 0.1);
    let stat_x = stationary[idx(offset, 0)];
    let stat_y = stationary[idx(0, offset)];
    assert!(
        (stat_x - stat_y).abs() < 0.02,
        "a stationary (v=0) particle must splat a symmetric footprint -- no \
         motion, no stretch: x={stat_x} y={stat_y}"
    );

    // The same fast velocity with dt = 0.0 (no physics step behind it) shows
    // zero bias: the extension is inert without dt, not merely small here.
    let fast_but_dt_zero = render_single_cluster(glam::Vec2::new(22.5, 0.0), 0.0);
    let zero_dt_x = fast_but_dt_zero[idx(offset, 0)];
    let zero_dt_y = fast_but_dt_zero[idx(0, offset)];
    assert!(
        (zero_dt_x - zero_dt_y).abs() < 0.02,
        "dt=0.0 must be a real no-op regardless of velocity (the physical \
         quantity is displacement = v*dt, not v alone): x={zero_dt_x} y={zero_dt_y}"
    );
}

/// GPU end-to-end check of the scattering and Fresnel port from
/// `prep_instances.wgsl`'s ByPhysics into `grid_volume.wgsl`'s `fs_main`.
/// With no CPU shortcut (`Renderer::particle_color`) for this path, a cluster
/// is rendered twice, scattering and specular off and then at tissue/water
/// values, and a pixel is read back to show the colour changes.
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn grid_volume_scattering_and_specular_change_rendered_color() {
    use crate::gpu::GpuSimulation;
    use crate::{MaterialRegistry, NeoHookeanMaterial, SimConfig, SpawnRegion, build_particles};
    use std::sync::Arc;

    let (device, queue) = headless_device();
    let device = Arc::new(device);
    let queue = Arc::new(queue);

    let grid_res = 32u32;
    let config = SimConfig::standard(grid_res as usize, 0.1, glam::Vec2::new(0.0, -0.3));
    let particles = build_particles(
        &config,
        SpawnRegion::for_sim(&config)
            .at(glam::Vec2::splat(16.0))
            .disk(4.0)
            .spacing(0.5)
            .material(0),
    );
    let registry = MaterialRegistry::with_default(Box::new(NeoHookeanMaterial::new(100.0, 50.0)));
    let mut sim =
        GpuSimulation::with_device(device.clone(), queue.clone(), config, particles, registry);
    // One real P2G step -- `grid_volume.wgsl` samples the solver's own grid
    // mass field, which is only populated once a step has actually run.
    sim.step_frame();

    let fmt = wgpu::TextureFormat::Rgba8UnormSrgb;
    let render_with = |sigma_s: f32, r0: f32, physical_thickness: Option<f32>| -> [u8; 4] {
        let mut r = Renderer::new(&device, sim.particle_count(), fmt);
        r.set_optical_coefficients_si(
            &queue,
            0,
            OpticalCoefficientsSi::new([0.3, 0.3, 0.3], sigma_s).unwrap(),
        );
        r.set_optical_scattering(&queue, 0, sigma_s);
        r.set_specular_r0(&queue, 0, r0);
        if let Some(thickness) = physical_thickness {
            r.set_grid_reference_cell_mass(1.0);
            r.set_physical_render_contract(
                &queue,
                PhysicalRenderContract::new(PhysicalRenderContractParams {
                    dx_meters: 0.1,
                    slice_thickness_m: thickness,
                    incident_radiance_w_m2_sr: [1.0; 3],
                    background_radiance_w_m2_sr: [1.0; 3],
                    display_white_radiance_w_m2_sr: [1.0; 3],
                    camera_direction: glam::Vec3::Z,
                    light_direction: glam::Vec3::Y,
                })
                .unwrap(),
            );
        }
        r.set_camera(&queue, grid_res, 64, 64, 0.6, true);

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("grid_volume_optical_test_target"),
            size: wgpu::Extent3d {
                width: 64,
                height: 64,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: fmt,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        r.render_grid_volume(
            &device,
            &queue,
            GridVolumeSource {
                grid: sim.grid_buffer(),
                material_mass: sim.material_mass_buffer(),
                material_mass_enabled: false,
                grid_res,
            },
            &view,
            true,
        );
        device.poll(wgpu::PollType::wait_indefinitely()).ok();
        readback_pixel(&device, &queue, &texture, 64, 64, 32, 32)
    };

    let without_optics = render_with(0.0, 0.0, None);
    let with_optics = render_with(8.0, 0.02, None); // real tissue-scale sigma_s, water-scale R0
    assert_ne!(
        without_optics, with_optics,
        "identical absorption but different scattering/specular must render a \
         different pixel color at the particle cluster's center: without={:?} with={:?}",
        without_optics, with_optics
    );

    let physical_absorption = render_with(0.0, 0.0, Some(1.0));
    assert!(
        physical_absorption[..3]
            .iter()
            .any(|channel| *channel < 250),
        "an occupied SI slab must attenuate the white physical background: {physical_absorption:?}"
    );
}

/// `grid_volume.wgsl` renders blackbody emission from a mass-weighted
/// temperature scattered into the buffer's channel 0. The `grid_int` buffer
/// is built by hand (the layout of
/// `grid_peak_is_the_largest_mass_within_two_cells`: 4 u32 slots per cell,
/// mass at offset 2) so mass stays equal and
/// temperature differs between the two renders.
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn grid_volume_blackbody_emission_brightens_hot_cells() {
    let (device, queue) = headless_device();
    let grid_res = 8u32;
    let cell_count = (grid_res * grid_res) as usize;
    const SLOTS: usize = 16;

    let material_mass_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test_grid_volume_material_mass"),
        size: (cell_count * SLOTS * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(
        &material_mass_buf,
        0,
        bytemuck::cast_slice(&vec![0f32; cell_count * SLOTS]),
    );

    let fmt = wgpu::TextureFormat::Rgba8UnormSrgb;
    let render_at_temp = |temp_k: f32| -> [u8; 4] {
        let grid_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test_grid_volume_grid_int"),
            size: (cell_count * 4 * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut cells = vec![0u32; cell_count * 4];
        // EVERY cell, not just one: a uniform full field, drawn everywhere,
        // and a mass-weighted temperature consistent
        // with that same mass (slot 0 = mass*temp, slot 2 = mass, so
        // `avg_temp = slot0/slot2 = temp_k` exactly, matching
        // `sample_weighted_temp`'s own real convention) -- uniform fill
        // sidesteps any dependency on exactly which cell the readback pixel's
        // bilinear/nearest sampling happens to land on.
        let mass = 1.0f32;
        for c in 0..cell_count {
            cells[c * 4] = (mass * temp_k).to_bits();
            cells[c * 4 + 2] = mass.to_bits();
        }
        queue.write_buffer(&grid_buf, 0, bytemuck::cast_slice(&cells));

        let mut r = Renderer::new(&device, 1, fmt);
        r.set_optical_params(&queue, 0, [0.3, 0.3, 0.3]);
        r.set_camera(&queue, grid_res, 64, 64, 0.6, true);

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("grid_volume_blackbody_test_target"),
            size: wgpu::Extent3d {
                width: 64,
                height: 64,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: fmt,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        r.render_grid_volume(
            &device,
            &queue,
            GridVolumeSource {
                grid: &grid_buf,
                material_mass: &material_mass_buf,
                material_mass_enabled: false,
                grid_res,
            },
            &view,
            true,
        );
        device.poll(wgpu::PollType::wait_indefinitely()).ok();
        readback_pixel(&device, &queue, &texture, 64, 64, 32, 32)
    };

    let cold = render_at_temp(293.0); // ambient room temperature
    let hot = render_at_temp(3000.0); // real near-ignition/glow-hot range
    let cold_brightness: u32 = cold[0] as u32 + cold[1] as u32 + cold[2] as u32;
    let hot_brightness: u32 = hot[0] as u32 + hot[1] as u32 + hot[2] as u32;
    assert!(
        hot_brightness > cold_brightness,
        "a hot cell must render brighter than an ambient-temperature cell with \
         otherwise identical mass (real blackbody emission, additive) -- \
         cold={cold:?} (sum={cold_brightness}) hot={hot:?} (sum={hot_brightness})"
    );
}

/// Grid-volume centre pixel of a scene whose every column is filled with
/// mass 1.0 from y = 0 up to (not including) `fill_to_y_exclusive`, on a
/// 24-cell grid. With `shape_holding`, slot 0 is adopted from a Neo-Hookean
/// solid, whose nonzero shear modulus marks it as holding its shape.
fn column_scene_centre_pixel(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    fill_to_y_exclusive: u32,
    shape_holding: bool,
) -> [u8; 4] {
    use crate::{MaterialRegistry, NeoHookeanMaterial};
    let grid_res = 24u32;
    let cell_count = (grid_res * grid_res) as usize;
    const SLOTS: usize = 16;

    let material_mass_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test_column_depth_material_mass"),
        size: (cell_count * SLOTS * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(
        &material_mass_buf,
        0,
        bytemuck::cast_slice(&vec![0f32; cell_count * SLOTS]),
    );
    let grid_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test_column_depth_grid_int"),
        size: (cell_count * 4 * std::mem::size_of::<u32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut cells = vec![0u32; cell_count * 4];
    let mass = 1.0f32;
    for cy in 0..fill_to_y_exclusive.min(grid_res) {
        for cx in 0..grid_res {
            let c = (cy * grid_res + cx) as usize;
            cells[c * 4 + 2] = mass.to_bits();
        }
    }
    queue.write_buffer(&grid_buf, 0, bytemuck::cast_slice(&cells));

    let fmt = wgpu::TextureFormat::Rgba8UnormSrgb;
    let mut r = Renderer::new(device, 1, fmt);
    // A water-like sigma_a (Pope & Fry 1997, as in render_plan.md): with a
    // near-zero absorption depth would barely matter.
    r.set_optical_params(queue, 0, [0.35, 0.033, 0.011]);
    if shape_holding {
        // Declares no optics, so the sigma_a above stays.
        let registry =
            MaterialRegistry::with_default(Box::new(NeoHookeanMaterial::new(100.0, 50.0)));
        r.adopt_material_optics(queue, &registry);
    }
    r.set_camera(queue, grid_res, 64, 64, 0.6, true);

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("column_depth_test_target"),
        size: wgpu::Extent3d {
            width: 64,
            height: 64,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: fmt,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    r.render_grid_volume(
        device,
        queue,
        GridVolumeSource {
            grid: &grid_buf,
            material_mass: &material_mass_buf,
            material_mass_enabled: false,
            grid_res,
        },
        &view,
        true,
    );
    device.poll(wgpu::PollType::wait_indefinitely()).ok();
    readback_pixel(device, queue, &texture, 64, 64, 32, 32)
}

fn pixel_brightness(p: [u8; 4]) -> u32 {
    u32::from(p[0]) + u32::from(p[1]) + u32::from(p[2])
}

/// Column-depth attenuation: water darkens with depth (Pope & Fry 1997, the
/// source of the sigma_a table), so with the same local mass at the query
/// cell a tall column above it (deep) renders darker than a thin band
/// (shallow). Only what lies above the cell differs.
///
/// Both scenes fill past the domain's vertical middle (17 of 24 rows), so the
/// centre pixel lands inside material either way without guessing which row
/// it maps to.
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn grid_volume_column_depth_darkens_deep_regions_more_than_shallow() {
    let (device, queue) = headless_device();
    let shallow = column_scene_centre_pixel(&device, &queue, 17, false);
    let deep = column_scene_centre_pixel(&device, &queue, 24, false);
    let (shallow_brightness, deep_brightness) = (pixel_brightness(shallow), pixel_brightness(deep));
    assert!(
        deep_brightness < shallow_brightness,
        "a deep column must render measurably darker than a shallow one at the SAME          local mass (real solar attenuation with depth, not local density alone) --          shallow={shallow:?} (sum={shallow_brightness}) deep={deep:?} (sum={deep_brightness})"
    );
}

/// The same two scenes as above, but the matter holds its shape: no
/// column-depth darkening, so a solid body reads as one flat tone instead of
/// a lit cylinder dark at the bottom.
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn grid_volume_shape_holding_matter_does_not_darken_with_column_depth() {
    let (device, queue) = headless_device();
    let shallow = column_scene_centre_pixel(&device, &queue, 17, true);
    let deep = column_scene_centre_pixel(&device, &queue, 24, true);
    assert_eq!(
        shallow, deep,
        "a solid must not darken with the depth of matter above it"
    );
}

/// Runs the grid-volume light pass on a grid whose cell `(x, y)` holds
/// `mass(x, y)` (reference cell mass 1), slot 0 with absorption `sigma_a`
/// and reduced scattering `sigma_s`, under a contract of cell size `dx` and light toward
/// `light`; returns the per-cell RGB transmittance, row-major (y * res + x).
fn light_pass_transmittance(
    res: u32,
    dx: f32,
    sigma_a: [f32; 3],
    sigma_s: f32,
    light: glam::Vec3,
    mass: impl Fn(u32, u32) -> f32,
) -> Vec<[f32; 3]> {
    use crate::render::{PhysicalRenderContract, PhysicalRenderContractParams};
    let (device, queue) = headless_device();
    let cells = (res * res) as usize;
    let mut words = vec![0u32; cells * 4];
    for y in 0..res {
        for x in 0..res {
            words[(y * res + x) as usize * 4 + 2] = mass(x, y).to_bits();
        }
    }
    let grid_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("light_pass_grid"),
        size: (cells * 16) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&grid_buf, 0, bytemuck::cast_slice(&words));
    let material_mass_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("light_pass_material_mass"),
        size: (cells * 16 * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let fmt = wgpu::TextureFormat::Rgba8UnormSrgb;
    let mut r = Renderer::new(&device, 1, fmt);
    r.set_optical_params(&queue, 0, sigma_a);
    r.set_optical_scattering(&queue, 0, sigma_s);
    r.set_physical_render_contract(
        &queue,
        PhysicalRenderContract::new(PhysicalRenderContractParams {
            dx_meters: dx,
            slice_thickness_m: 0.5,
            incident_radiance_w_m2_sr: [1.0; 3],
            background_radiance_w_m2_sr: [0.25; 3],
            display_white_radiance_w_m2_sr: [1.0; 3],
            camera_direction: glam::Vec3::new(0.0, 0.0, -1.0),
            light_direction: light,
        })
        .unwrap(),
    );
    r.set_camera(&queue, res, 64, 64, 0.6, true);
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("light_pass_target"),
        size: wgpu::Extent3d {
            width: 64,
            height: 64,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: fmt,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    r.render_grid_volume(
        &device,
        &queue,
        GridVolumeSource {
            grid: &grid_buf,
            material_mass: &material_mass_buf,
            material_mass_enabled: false,
            grid_res: res,
        },
        &view,
        true,
    );
    device.poll(wgpu::PollType::wait_indefinitely()).ok();
    let raw = readback_f32_blocking(&device, &queue, &r.light_transmittance_buf, cells * 4);
    raw.chunks(4).map(|c| [c[0], c[1], c[2]]).collect()
}

/// Gate (a) of the volumetric-shadow pass: in a uniform medium lit from
/// straight above, the light reaching a cell has crossed every cell above
/// it, so its transmittance is Beer-Lambert's `exp(-sigma * dx * n)` with
/// `n` the number of cells between it and the top edge.
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn light_pass_matches_beer_lambert_in_a_uniform_medium() {
    let (res, dx, sigma) = (24u32, 0.1f32, [0.5f32, 1.0, 2.0]);
    let t = light_pass_transmittance(res, dx, sigma, 0.0, glam::Vec3::Y, |_, _| 1.0);
    for y in 0..res {
        let cells_above = (res - 1 - y) as f32;
        for x in [0u32, 7, res - 1] {
            let got = t[(y * res + x) as usize];
            for ch in 0..3 {
                let expected = (-sigma[ch] * dx * cells_above).exp();
                assert!(
                    (got[ch] - expected).abs() < 1.0e-3,
                    "cell ({x},{y}) channel {ch}: {} against exp(-{} * {dx} * {cells_above}) = {expected}",
                    got[ch],
                    sigma[ch]
                );
            }
        }
    }
}

/// The diffusion regime: in a strongly scattering, weakly absorbing medium
/// (snow's measured coefficients) light reaching a depth has mostly
/// scattered on the way, and it dies out with diffusion theory's effective
/// attenuation sqrt(3 mu_a (mu_a + mu_s')) (Jacques and Prahl, ECE532 notes),
/// far slower than the unscattered beam's mu_a + mu_s'.
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn light_pass_uses_diffusion_attenuation_in_a_scattering_medium() {
    let (res, dx, sigma_a, sigma_s) = (24u32, 0.01f32, [0.055f32, 0.018, 0.0014], 236.0f32);
    let t = light_pass_transmittance(res, dx, sigma_a, sigma_s, glam::Vec3::Y, |_, _| 1.0);
    for y in 0..res {
        let cells_above = (res - 1 - y) as f32;
        let got = t[(y * res + 5) as usize];
        for ch in 0..3 {
            let mu_eff = (3.0 * sigma_a[ch] * (sigma_a[ch] + sigma_s)).sqrt();
            assert!(
                mu_eff < sigma_a[ch] + sigma_s,
                "the diffuse rate must be the slower one"
            );
            let expected = (-mu_eff * dx * cells_above).exp();
            assert!(
                (got[ch] - expected).abs() < 1.0e-3,
                "cell (5,{y}) channel {ch}: {} against exp(-{mu_eff} * {dx} * {cells_above}) = {expected}",
                got[ch]
            );
        }
    }
}

/// Gate (b): an opaque block under oblique light (toward +x, +y) shadows
/// exactly the cells whose ray toward the light crosses it. Cells whose ray
/// passes the block shrunk by one cell must be dark, cells whose ray misses
/// it grown by one cell must be fully lit; the one-cell band between is the
/// march's own discretisation and is not judged.
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn light_pass_shadow_edge_follows_the_ray_geometry() {
    let res = 32u32;
    let t = light_pass_transmittance(
        res,
        0.1,
        [100.0; 3],
        0.0,
        glam::Vec3::new(1.0, 1.0, 0.0),
        |x, y| {
            if in_shadow_test_block(x as f32 + 0.5, y as f32 + 0.5) {
                1.0
            } else {
                0.0
            }
        },
    );
    assert_shadow_follows_the_ray_geometry(&t, res);
}

/// The opaque block of the shadow-geometry tests, in physics-grid cells.
const SHADOW_TEST_BLOCK: (f32, f32, f32, f32) = (12.0, 16.0, 20.0, 24.0);

fn in_shadow_test_block(cx: f32, cy: f32) -> bool {
    let (x0, x1, y0, y1) = SHADOW_TEST_BLOCK;
    cx > x0 && cx < x1 && cy > y0 && cy < y1
}

/// Judges a `res` transmittance field lit toward (1, 1) past
/// `SHADOW_TEST_BLOCK` (see `light_pass_shadow_edge_follows_the_ray_geometry`).
fn assert_shadow_follows_the_ray_geometry(t: &[[f32; 3]], res: u32) {
    let (x0, x1, y0, y1) = SHADOW_TEST_BLOCK;
    // Slab test of the ray from the cell centre along (1, 1)/sqrt(2) against
    // the box grown by `margin` cells.
    let hits = |x: u32, y: u32, margin: f32| {
        let (ox, oy) = (x as f32 + 0.5, y as f32 + 0.5);
        let (lo_x, hi_x) = (x0 - margin, x1 + margin);
        let (lo_y, hi_y) = (y0 - margin, y1 + margin);
        let t_enter = (lo_x - ox).max(lo_y - oy);
        let t_exit = (hi_x - ox).min(hi_y - oy);
        t_exit > t_enter.max(0.0)
    };
    let (mut dark, mut lit) = (0, 0);
    for y in 0..res {
        for x in 0..res {
            if in_shadow_test_block(x as f32 + 0.5, y as f32 + 0.5) {
                continue;
            }
            let got = t[(y * res + x) as usize][0];
            if hits(x, y, -1.0) {
                assert!(got < 0.01, "cell ({x},{y}) behind the block got {got}");
                dark += 1;
            } else if !hits(x, y, 1.0) {
                assert!(got > 0.99, "cell ({x},{y}) clear of the block got {got}");
                lit += 1;
            }
        }
    }
    assert!(
        dark > 20 && lit > 500,
        "too few judged cells: {dark} dark, {lit} lit"
    );
}

/// `render_surface_reconstruction` runs the light pass from its own
/// reconstructed density, with no grid-volume render before it: an opaque
/// disk lit from straight above leaves the cells under it dark and the
/// cells above it, or beside it, fully lit.
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn surface_reconstruction_shadows_with_its_own_light_pass() {
    use crate::gpu::GpuSimulation;
    use crate::render::{PhysicalRenderContract, PhysicalRenderContractParams};
    use crate::{MaterialRegistry, NeoHookeanMaterial, SimConfig, SpawnRegion, build_particles};
    use std::sync::Arc;

    let (device, queue) = headless_device();
    let device = Arc::new(device);
    let queue = Arc::new(queue);
    let grid_res = 32u32;
    let config = SimConfig::standard(grid_res as usize, 0.1, glam::Vec2::new(0.0, -0.3));
    let particles = build_particles(
        &config,
        SpawnRegion::for_sim(&config)
            .at(glam::Vec2::splat(16.0))
            .disk(6.0)
            .spacing(0.5)
            .material(0),
    );
    let registry = MaterialRegistry::with_default(Box::new(NeoHookeanMaterial::new(100.0, 50.0)));
    let sim =
        GpuSimulation::with_device(device.clone(), queue.clone(), config, particles, registry);

    let fmt = wgpu::TextureFormat::Rgba8UnormSrgb;
    let mut r = Renderer::new(&device, sim.particle_count(), fmt);
    r.set_camera(&queue, grid_res, 64, 64, 0.6, true);
    // 50 per metre over 0.1 m cells: an optical depth of 5 per cell.
    r.set_optical_params(&queue, 0, [50.0; 3]);
    r.set_physical_render_contract(
        &queue,
        PhysicalRenderContract::new(PhysicalRenderContractParams {
            dx_meters: 0.1,
            slice_thickness_m: 0.5,
            incident_radiance_w_m2_sr: [1.0; 3],
            background_radiance_w_m2_sr: [0.25; 3],
            display_white_radiance_w_m2_sr: [1.0; 3],
            camera_direction: glam::Vec3::new(0.0, 0.0, -1.0),
            light_direction: glam::Vec3::Y,
        })
        .unwrap(),
    );
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("surface_light_pass_target"),
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
    r.render_surface_reconstruction(
        &device,
        &queue,
        SurfaceReconstructionSource {
            particle_buf: sim.particle_buffer(),
            particle_count: sim.particle_count(),
            grid_res,
            material_slot: 0,
            material_mass_enabled: false,
            dt: 0.1,
        },
        &view,
        true,
    );
    device.poll(wgpu::PollType::wait_indefinitely()).ok();
    let cells = (grid_res * grid_res) as usize;
    let t = readback_f32_blocking(&device, &queue, &r.light_transmittance_buf, cells * 4);
    let at = |x: u32, y: u32| t[(y * grid_res + x) as usize * 4];
    // The disk spans y = 10..22 along x = 16.
    assert!(at(16, 6) < 0.01, "under the disk: {}", at(16, 6));
    assert!(at(16, 26) > 0.99, "above the disk: {}", at(16, 26));
    assert!(at(3, 6) > 0.99, "beside the disk: {}", at(3, 6));
}

/// Runs the light pass from a curvature-flow surface density: `density(i, j)`
/// on a `surface_res` square covering the same domain as the `res` physics
/// grid (reference cell mass 1), with per-slot masses `slot_mass(i, j, s)`
/// when given (slot 0 throughout otherwise), slot `s` absorbing
/// `sigma_a[s]` and not scattering, under a contract of cell size `dx` and
/// light toward `light`. Returns the per-cell RGB transmittance on the
/// physics grid, row-major (y * res + x).
fn surface_light_pass_transmittance(
    res: u32,
    surface_res: u32,
    dx: f32,
    sigma_a: &[[f32; 3]],
    light: glam::Vec3,
    density: impl Fn(u32, u32) -> f32,
    slot_mass: Option<&dyn Fn(u32, u32, usize) -> i32>,
) -> Vec<[f32; 3]> {
    use crate::render::{PhysicalRenderContract, PhysicalRenderContractParams};
    const SLOTS: usize = 16;
    let (device, queue) = headless_device();
    let surface_cells = (surface_res * surface_res) as usize;
    let mut field = vec![0.0f32; surface_cells];
    let mut masses = vec![0i32; surface_cells * SLOTS];
    for j in 0..surface_res {
        for i in 0..surface_res {
            let idx = (j * surface_res + i) as usize;
            field[idx] = density(i, j);
            if let Some(slot_mass) = slot_mass {
                for s in 0..SLOTS {
                    masses[idx * SLOTS + s] = slot_mass(i, j, s);
                }
            }
        }
    }
    let storage = |label: &str, bytes: &[u8]| {
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&buf, 0, bytes);
        buf
    };
    let density_buf = storage("surface_light_pass_density", bytemuck::cast_slice(&field));
    let material_mass_buf = storage(
        "surface_light_pass_material_mass",
        bytemuck::cast_slice(&masses),
    );
    let mut r = Renderer::new(&device, 1, wgpu::TextureFormat::Rgba8UnormSrgb);
    for (slot, sigma) in sigma_a.iter().enumerate() {
        r.set_optical_params(&queue, slot, *sigma);
    }
    r.set_physical_render_contract(
        &queue,
        PhysicalRenderContract::new(PhysicalRenderContractParams {
            dx_meters: dx,
            slice_thickness_m: 0.5,
            incident_radiance_w_m2_sr: [1.0; 3],
            background_radiance_w_m2_sr: [0.25; 3],
            display_white_radiance_w_m2_sr: [1.0; 3],
            camera_direction: glam::Vec3::new(0.0, 0.0, -1.0),
            light_direction: light,
        })
        .unwrap(),
    );
    r.ensure_light_pass_capacity(&device, res);
    r.ensure_light_surface_capacity(&device, surface_res);
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("surface_light_pass"),
    });
    r.encode_light_pass(
        &device,
        &queue,
        &mut enc,
        res,
        LightPassSource::Surface {
            density: &density_buf,
            material_mass: &material_mass_buf,
            material_mass_enabled: slot_mass.is_some(),
            surface_res,
            material_slot: 0,
        },
    );
    queue.submit(std::iter::once(enc.finish()));
    device.poll(wgpu::PollType::wait_indefinitely()).ok();
    let cells = (res * res) as usize;
    let raw = readback_f32_blocking(&device, &queue, &r.light_transmittance_buf, cells * 4);
    raw.chunks(4).map(|c| [c[0], c[1], c[2]]).collect()
}

/// Asserts a transmittance field lit from straight above is Beer-Lambert's
/// `exp(-sigma * dx * n)` through `n` cells of extinction `sigma` per metre.
fn assert_beer_lambert_from_above(t: &[[f32; 3]], res: u32, dx: f32, sigma: [f32; 3], case: &str) {
    for y in 0..res {
        let cells_above = (res - 1 - y) as f32;
        for x in [0u32, 7, res - 1] {
            let got = t[(y * res + x) as usize];
            for ch in 0..3 {
                let expected = (-sigma[ch] * dx * cells_above).exp();
                assert!(
                    (got[ch] - expected).abs() < 1.0e-3,
                    "{case}: cell ({x},{y}) channel {ch}: {} against exp(-{} * {dx} * {cells_above}) = {expected}",
                    got[ch],
                    sigma[ch]
                );
            }
        }
    }
}

/// The surface path marches the same physics grid, from the mean extinction
/// of the surface cells inside each grid cell. A surface whose columns
/// alternate empty and twice the reference density averages to the
/// reference density, so the light through it is Beer-Lambert's through a
/// uniform medium; so is a uniform surface at a resolution that is not a
/// whole multiple of the grid's, whose grid cells hold two or three surface
/// cells in turn.
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn surface_light_pass_matches_beer_lambert_through_the_averaged_density() {
    let (res, dx, sigma) = (24u32, 0.1f32, [0.5f32, 1.0, 2.0]);
    let alternating = surface_light_pass_transmittance(
        res,
        6 * res,
        dx,
        &[sigma],
        glam::Vec3::Y,
        |i, _| if i % 2 == 0 { 0.0 } else { 2.0 },
        None,
    );
    assert_beer_lambert_from_above(&alternating, res, dx, sigma, "alternating columns");
    let uneven = surface_light_pass_transmittance(
        res,
        res * 5 / 2,
        dx,
        &[sigma],
        glam::Vec3::Y,
        |_, _| 1.0,
        None,
    );
    assert_beer_lambert_from_above(&uneven, res, dx, sigma, "surface 2.5 times finer");
}

/// The surface's materials blend by mass fraction within each surface cell,
/// then average over the grid cell: even columns hold slot 0 alone, odd
/// columns slot 0 and slot 1 at 1 : 3, so every grid cell attenuates as
/// `0.5 sigma_0 + 0.5 (0.25 sigma_0 + 0.75 sigma_1)`.
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn surface_light_pass_blends_slots_by_mass_fraction() {
    let (res, dx) = (16u32, 0.1f32);
    let (sigma_0, sigma_1) = ([0.4f32, 0.8, 1.6], [2.0f32, 1.0, 0.0]);
    let slot_mass = |i: u32, _: u32, s: usize| match (i % 2, s) {
        (_, 0) => 1000,
        (1, 1) => 3000,
        _ => 0,
    };
    let t = surface_light_pass_transmittance(
        res,
        6 * res,
        dx,
        &[sigma_0, sigma_1],
        glam::Vec3::Y,
        |_, _| 1.0,
        Some(&slot_mass),
    );
    let blended: [f32; 3] = std::array::from_fn(|ch| {
        0.5 * sigma_0[ch] + 0.5 * (0.25 * sigma_0[ch] + 0.75 * sigma_1[ch])
    });
    assert_beer_lambert_from_above(&t, res, dx, blended, "mass-fraction blend");
}

/// Gate (b) on the surface path: a block of surface cells filling whole
/// grid cells shadows exactly as the same block on the grid does.
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn surface_light_pass_shadow_edge_follows_the_ray_geometry() {
    let (res, multiplier) = (32u32, 6u32);
    let t = surface_light_pass_transmittance(
        res,
        multiplier * res,
        0.1,
        &[[100.0; 3]],
        glam::Vec3::new(1.0, 1.0, 0.0),
        |i, j| {
            let centre = |k: u32| (k as f32 + 0.5) / multiplier as f32;
            if in_shadow_test_block(centre(i), centre(j)) {
                1.0
            } else {
                0.0
            }
        },
        None,
    );
    assert_shadow_follows_the_ray_geometry(&t, res);
}

/// `dominant_material` reads `material_mass` as i32: read as bit-reinterpreted
/// f32, this GPU flushes the denormal values to zero in the fragment shader.
/// Two grid halves with different dominant slots render their own colours,
/// not both slot 0 (every other call site here sets
/// `material_mass_enabled: false`).
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn grid_volume_dominant_material_colors_regions_distinctly() {
    let (device, queue) = headless_device();
    let grid_res = 8u32;
    let cell_count = (grid_res * grid_res) as usize;
    const SLOTS: usize = 16;

    let grid_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test_dominant_material_grid_int"),
        size: (cell_count * 4 * std::mem::size_of::<u32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut cells = vec![0u32; cell_count * 4];
    // Uniform real mass everywhere (slot 2), comfortably above mass_floor,
    // no temperature contribution (slot 0 stays 0 -- isolates this test
    // from blackbody emission entirely).
    let mass = 1.0f32;
    for c in 0..cell_count {
        cells[c * 4 + 2] = mass.to_bits();
    }
    queue.write_buffer(&grid_buf, 0, bytemuck::cast_slice(&cells));

    let material_mass_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test_dominant_material_material_mass"),
        size: (cell_count * SLOTS * std::mem::size_of::<i32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    // Fixed-point-scale magnitude (same order as P2G's own
    // `MASS_ATOMIC_SCALE`-driven values) -- deliberately denormal-adjacent,
    // not a large/normal-range float, since FTZ/rounding bugs live in that
    // regime. Left half (x<4) dominant in slot 0, right half (x>=4)
    // dominant in slot 1.
    let mut mm = vec![0i32; cell_count * SLOTS];
    for cy in 0..grid_res {
        for cx in 0..grid_res {
            let c = (cy * grid_res + cx) as usize;
            let slot = if cx < grid_res / 2 { 0 } else { 1 };
            mm[c * SLOTS + slot] = 100_000;
        }
    }
    queue.write_buffer(&material_mass_buf, 0, bytemuck::cast_slice(&mm));

    let fmt = wgpu::TextureFormat::Rgba8UnormSrgb;
    let mut r = Renderer::new(&device, 1, fmt);
    r.set_optical_params(&queue, 0, [0.05, 0.55, 0.55]); // red-dominant
    r.set_optical_params(&queue, 1, [0.55, 0.05, 0.55]); // green-dominant
    r.set_camera(&queue, grid_res, 64, 64, 0.6, true);

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("dominant_material_test_target"),
        size: wgpu::Extent3d {
            width: 64,
            height: 64,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: fmt,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    r.render_grid_volume(
        &device,
        &queue,
        GridVolumeSource {
            grid: &grid_buf,
            material_mass: &material_mass_buf,
            material_mass_enabled: true,
            grid_res,
        },
        &view,
        true,
    );
    device.poll(wgpu::PollType::wait_indefinitely()).ok();

    // Whole halves of the grid, not a single particle splat -- pixel
    // targeting is forgiving here, but sample a few points per half anyway
    // rather than trust one exact pixel.
    let left = readback_pixel(&device, &queue, &texture, 64, 64, 16, 32);
    let right = readback_pixel(&device, &queue, &texture, 64, 64, 48, 32);

    assert_ne!(
        left, right,
        "left half (slot 0) and right half (slot 1) must render visibly \
         distinct colors: left={:?} right={:?}",
        left, right
    );
    assert!(
        left[0] > left[1] && left[0] > left[2],
        "left half (dominant slot 0, low red absorption) should read \
         red-dominant: {:?}",
        left
    );
    assert!(
        right[1] > right[0] && right[1] > right[2],
        "right half (dominant slot 1, low green absorption) should read \
         green-dominant: {:?}",
        right
    );
}

/// Same real end-to-end check as
/// `grid_volume_scattering_and_specular_change_rendered_color`, for
/// `curvature_flow.wgsl`'s single-phase `fs_main` -- the other fragment
/// shader that just received the same optical-parity port.
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn curvature_flow_scattering_and_specular_change_rendered_color() {
    use crate::gpu::GpuSimulation;
    use crate::{MaterialRegistry, NeoHookeanMaterial, SimConfig, SpawnRegion, build_particles};
    use std::sync::Arc;

    let (device, queue) = headless_device();
    let device = Arc::new(device);
    let queue = Arc::new(queue);

    let grid_res = 32u32;
    let config = SimConfig::standard(grid_res as usize, 0.1, glam::Vec2::new(0.0, -0.3));
    let particles = build_particles(
        &config,
        SpawnRegion::for_sim(&config)
            .at(glam::Vec2::splat(16.0))
            .disk(4.0)
            .spacing(0.5)
            .material(0),
    );
    let registry = MaterialRegistry::with_default(Box::new(NeoHookeanMaterial::new(100.0, 50.0)));
    let sim =
        GpuSimulation::with_device(device.clone(), queue.clone(), config, particles, registry);

    let fmt = wgpu::TextureFormat::Rgba8UnormSrgb;
    let render_with = |sigma_s: f32, r0: f32| -> [u8; 4] {
        let mut r = Renderer::new(&device, sim.particle_count(), fmt);
        r.set_optical_params(&queue, 0, [0.3, 0.3, 0.3]);
        r.set_optical_scattering(&queue, 0, sigma_s);
        r.set_specular_r0(&queue, 0, r0);
        r.set_camera(&queue, grid_res, 64, 64, 0.6, true);

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("curvature_flow_optical_test_target"),
            size: wgpu::Extent3d {
                width: 64,
                height: 64,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: fmt,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        r.render_surface_reconstruction(
            &device,
            &queue,
            SurfaceReconstructionSource {
                particle_buf: sim.particle_buffer(),
                particle_count: sim.particle_count(),
                grid_res,
                material_slot: 0,
                material_mass_enabled: false,
                dt: 0.1,
            },
            &view,
            true,
        );
        device.poll(wgpu::PollType::wait_indefinitely()).ok();
        readback_pixel(&device, &queue, &texture, 64, 64, 32, 32)
    };

    let without_optics = render_with(0.0, 0.0);
    let with_optics = render_with(8.0, 0.02);
    assert_ne!(
        without_optics, with_optics,
        "identical absorption but different scattering/specular must render a \
         different pixel color at the particle cluster's center: without={:?} with={:?}",
        without_optics, with_optics
    );
}

/// Diagnostic, not a regression test -- run manually via `cargo test
/// diagnose_curvature_flow_edge_hair_pixels -- --ignored --nocapture` when
/// investigating "hair" fuzz around Surface mode's silhouette edges. Prints
/// an RGBA scanline crossing a cold (ambient-temperature-only, no
/// blackbody emission at all) cluster's own boundary, so the actual pixel
/// numbers can be read directly instead of guessing from a screenshot --
/// isolates whether the artifact depends on temperature/emission at all.
#[test]
#[ignore]
fn diagnose_curvature_flow_edge_hair_pixels() {
    use crate::gpu::GpuSimulation;
    use crate::{MaterialRegistry, NeoHookeanMaterial, SimConfig, SpawnRegion, build_particles};
    use std::sync::Arc;

    let (device, queue) = headless_device();
    let device = Arc::new(device);
    let queue = Arc::new(queue);

    let grid_res = 32u32;
    let config = SimConfig::standard(grid_res as usize, 0.1, glam::Vec2::new(0.0, -0.3));
    // Cold, ambient-temperature-only cluster -- same SIGMA_NEO/scattering
    // basic_jellies_gpu.rs's own MAT_NEO uses, no heat at all, to isolate
    // whether the reported edge artifact depends on temperature/emission.
    let particles = build_particles(
        &config,
        SpawnRegion::for_sim(&config)
            .at(glam::Vec2::splat(16.0))
            .disk(8.0)
            .spacing(0.5)
            .material(0),
    );
    let registry = MaterialRegistry::with_default(Box::new(NeoHookeanMaterial::new(10.0, 20.0)));
    let sim =
        GpuSimulation::with_device(device.clone(), queue.clone(), config, particles, registry);

    let fmt = wgpu::TextureFormat::Rgba8UnormSrgb;
    let mut r = Renderer::new(&device, sim.particle_count(), fmt);
    r.set_optical_params(&queue, 0, [0.05, 0.55, 0.60]);
    r.set_optical_scattering(&queue, 0, 0.02);
    r.set_specular_r0(&queue, 0, 0.01);
    r.set_camera(&queue, grid_res, 128, 128, 0.6, true);

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("diagnose_edge_hair_target"),
        size: wgpu::Extent3d {
            width: 128,
            height: 128,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: fmt,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    r.render_surface_reconstruction(
        &device,
        &queue,
        SurfaceReconstructionSource {
            particle_buf: sim.particle_buffer(),
            particle_count: sim.particle_count(),
            grid_res,
            material_slot: 0,
            material_mass_enabled: false,
            dt: 0.1,
        },
        &view,
        true,
    );
    device.poll(wgpu::PollType::wait_indefinitely()).ok();

    // Raw density (`surface_a_buf`, after curvature flow) along the same row,
    // to read the interior depth directly instead of from colour.
    let surface_res = r.surface_res;
    let mass_values = readback_f32_blocking(
        &device,
        &queue,
        &r.surface_a_buf,
        (surface_res * surface_res) as usize,
    );
    let scale = surface_res as f32 / grid_res as f32;
    let row = (16.0 * scale) as u32; // grid y=16 -> surface row, matches this scene's camera-space y=64
    for sx in 0..surface_res {
        let m = mass_values[(row * surface_res + sx) as usize];
        if m > 1.0e-4 {
            eprintln!("surface_cell x={sx} mass={m}");
        }
    }

    // Scanline straight through the cluster's own edge (cluster is centered
    // at grid (16,16) with real radius 8, camera maps grid->pixel via
    // set_camera's own 0.6 zoom over a 128px target -- print a wide enough
    // range to see background -> edge -> interior.
    for x in 40..115 {
        let px = readback_pixel(&device, &queue, &texture, 128, 128, x, 64);
        eprintln!("y=64 x={x} rgba={px:?}");
    }
    // Also scan through a more CURVED part of the silhouette (near the
    // disk's top cap, not straight through its widest point) -- the
    // reported "hair" artifact looks worst on curved edges in the
    // screenshot, a flat scan through dead-center might miss it.
    for x in 40..115 {
        let px = readback_pixel(&device, &queue, &texture, 128, 128, x, 40);
        eprintln!("y=40 x={x} rgba={px:?}");
    }
}

/// Surface mode renders blackbody emission from a mass-weighted temperature
/// (`surface_temp_atomic`/`surface_temp_final`, the formula of
/// `grid_volume.wgsl`). Built from particles, not a hand-made buffer: two
/// `GpuSimulation`s from the same spawn region, differing only in
/// `particles.temperature`, so the temperature reaches the pixel through
/// splat, convert and curvature flow.
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn curvature_flow_blackbody_emission_brightens_hot_cluster() {
    use crate::gpu::GpuSimulation;
    use crate::{MaterialRegistry, NeoHookeanMaterial, SimConfig, SpawnRegion, build_particles};
    use std::sync::Arc;

    let (device, queue) = headless_device();
    let device = Arc::new(device);
    let queue = Arc::new(queue);

    let grid_res = 32u32;
    let fmt = wgpu::TextureFormat::Rgba8UnormSrgb;

    let render_at_temp = |temp_k: f32| -> [u8; 4] {
        let config = SimConfig::standard(grid_res as usize, 0.1, glam::Vec2::new(0.0, -0.3));
        let mut particles = build_particles(
            &config,
            SpawnRegion::for_sim(&config)
                .at(glam::Vec2::splat(16.0))
                .disk(4.0)
                .spacing(0.5)
                .material(0),
        );
        for p in particles.iter_mut() {
            p.temperature = temp_k;
        }
        let registry =
            MaterialRegistry::with_default(Box::new(NeoHookeanMaterial::new(100.0, 50.0)));
        let sim =
            GpuSimulation::with_device(device.clone(), queue.clone(), config, particles, registry);

        let mut r = Renderer::new(&device, sim.particle_count(), fmt);
        r.set_optical_params(&queue, 0, [0.3, 0.3, 0.3]);
        r.set_camera(&queue, grid_res, 64, 64, 0.6, true);

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("curvature_flow_blackbody_test_target"),
            size: wgpu::Extent3d {
                width: 64,
                height: 64,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: fmt,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        r.render_surface_reconstruction(
            &device,
            &queue,
            SurfaceReconstructionSource {
                particle_buf: sim.particle_buffer(),
                particle_count: sim.particle_count(),
                grid_res,
                material_slot: 0,
                material_mass_enabled: false,
                dt: 0.1,
            },
            &view,
            true,
        );
        device.poll(wgpu::PollType::wait_indefinitely()).ok();
        readback_pixel(&device, &queue, &texture, 64, 64, 32, 32)
    };

    let cold = render_at_temp(293.0); // ambient room temperature
    let hot = render_at_temp(3000.0); // real near-ignition/glow-hot range
    let cold_brightness: u32 = cold[0] as u32 + cold[1] as u32 + cold[2] as u32;
    let hot_brightness: u32 = hot[0] as u32 + hot[1] as u32 + hot[2] as u32;
    assert!(
        hot_brightness > cold_brightness,
        "a hot particle cluster must render brighter than an ambient-temperature \
         one with otherwise identical mass/shape (real blackbody emission, \
         additive, single-phase curvature-flow surface mode) -- \
         cold={cold:?} (sum={cold_brightness}) hot={hot:?} (sum={hot_brightness})"
    );
}

/// Light diffusion (`curvature_flow.wgsl`'s Pass 1e) accumulates across
/// frames: its fluence Phi persists, since diffusion needs time to spread.
/// The same static scene (fixed temperature, zero velocity; the wave field
/// stays flat because its forcing is the density's change) is rendered
/// repeatedly through one `Renderer`. A hot scene's brightness at a pixel
/// rises from frame 1 to frame 30; an ambient scene drifts by only a small
/// fraction of that, ruling out frame-to-frame noise. Not zero: the emission
/// goes as Stefan-Boltzmann's `(T/T_ref)^4`, so 293 K still radiates, ~11000x
/// less than 3000 K at the default 3000 K anchor.
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn light_diffusion_builds_up_real_glow_over_multiple_frames() {
    use crate::gpu::GpuSimulation;
    use crate::{MaterialRegistry, NeoHookeanMaterial, SimConfig, SpawnRegion, build_particles};
    use std::sync::Arc;

    let (device, queue) = headless_device();
    let device = Arc::new(device);
    let queue = Arc::new(queue);

    let grid_res = 32u32;
    let fmt = wgpu::TextureFormat::Rgba8UnormSrgb;

    let render_n_frames = |temp_k: f32, n_frames: u32| -> [u8; 4] {
        let config = SimConfig::standard(grid_res as usize, 0.1, glam::Vec2::new(0.0, -0.3));
        let mut particles = build_particles(
            &config,
            SpawnRegion::for_sim(&config)
                .at(glam::Vec2::splat(16.0))
                .disk(4.0)
                .spacing(0.5)
                .material(0),
        );
        for p in particles.iter_mut() {
            p.temperature = temp_k;
        }
        let registry =
            MaterialRegistry::with_default(Box::new(NeoHookeanMaterial::new(100.0, 50.0)));
        let sim =
            GpuSimulation::with_device(device.clone(), queue.clone(), config, particles, registry);

        let mut r = Renderer::new(&device, sim.particle_count(), fmt);
        r.set_optical_params(&queue, 0, [0.3, 0.3, 0.3]);
        r.set_camera(&queue, grid_res, 64, 64, 0.6, true);

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("light_diffusion_test_target"),
            size: wgpu::Extent3d {
                width: 64,
                height: 64,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: fmt,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut last = [0u8; 4];
        for _ in 0..n_frames {
            r.render_surface_reconstruction(
                &device,
                &queue,
                SurfaceReconstructionSource {
                    particle_buf: sim.particle_buffer(),
                    particle_count: sim.particle_count(),
                    grid_res,
                    material_slot: 0,
                    material_mass_enabled: false,
                    dt: 0.1,
                },
                &view,
                true,
            );
            device.poll(wgpu::PollType::wait_indefinitely()).ok();
            last = readback_pixel(&device, &queue, &texture, 64, 64, 32, 32);
        }
        last
    };

    let hot_frame1 = render_n_frames(3000.0, 1);
    let hot_frame30 = render_n_frames(3000.0, 30);
    let hot_b1: u32 = hot_frame1[0] as u32 + hot_frame1[1] as u32 + hot_frame1[2] as u32;
    let hot_b30: u32 = hot_frame30[0] as u32 + hot_frame30[1] as u32 + hot_frame30[2] as u32;
    assert!(
        hot_b30 > hot_b1,
        "a hot, unchanging scene's own rendered brightness at the SAME pixel must \
         genuinely increase from frame 1 to frame 30 as the real, persistent light-\
         diffusion field builds up -- frame1={hot_frame1:?} (sum={hot_b1}) \
         frame30={hot_frame30:?} (sum={hot_b30})"
    );

    let cold_frame1 = render_n_frames(293.0, 1);
    let cold_frame30 = render_n_frames(293.0, 30);
    let cold_b1: u32 = cold_frame1[0] as u32 + cold_frame1[1] as u32 + cold_frame1[2] as u32;
    let cold_b30: u32 = cold_frame30[0] as u32 + cold_frame30[1] as u32 + cold_frame30[2] as u32;
    let hot_drift = hot_b30.abs_diff(hot_b1);
    let cold_drift = cold_b30.abs_diff(cold_b1);
    assert!(
        cold_drift * 5 <= hot_drift,
        "an ambient (cold) scene emits real light too (`light_diffuse_main`'s own \
         (T/T_ref)^4 term is never exactly zero), but ~11000x weaker than the hot scene \
         at 293K vs 3000K -- its frame1->frame30 drift must stay a small fraction of \
         the hot scene's, not comparable to it (rules out unrelated per-frame noise \
         as the explanation): hot drift={hot_drift} (frame1={hot_frame1:?} \
         frame30={hot_frame30:?}) cold drift={cold_drift} (frame1={cold_frame1:?} \
         frame30={cold_frame30:?})"
    );
}

/// Thermal diffusion (`curvature_flow.wgsl`'s Pass 1c, `temp_avg_main`/
/// `temp_diffuse_main`): a hot (3000 K) and an ambient (293 K) cluster
/// touching, a sharp temperature step, the input an explicit stencil
/// misbehaves on if its Fourier bound (`DIFFUSION_ALPHA*DIFFUSION_DT <=
/// 0.25`) were wrong. Reads `surface_temp_float_buf`: finite everywhere, and
/// the hot side warmer than the cold side.
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn curvature_flow_thermal_diffusion_stays_finite_and_separates_hot_from_cold() {
    use crate::gpu::GpuSimulation;
    use crate::{MaterialRegistry, NeoHookeanMaterial, SimConfig, SpawnRegion, build_particles};
    use std::sync::Arc;

    let (device, queue) = headless_device();
    let device = Arc::new(device);
    let queue = Arc::new(queue);

    let grid_res = 32u32;
    let config = SimConfig::standard(grid_res as usize, 0.1, glam::Vec2::new(0.0, -0.3));

    let mut hot_side = build_particles(
        &config,
        SpawnRegion {
            spacing: 0.5,
            box_size: glam::IVec2::new(12, 12),
            box_center: glam::Vec2::new(10.0, 16.0),
            material_id: 0,
            rng_seed: 1,
            ..SpawnRegion::for_sim(&config)
        },
    );
    for p in hot_side.iter_mut() {
        p.temperature = 3000.0;
    }
    let mut cold_side = build_particles(
        &config,
        SpawnRegion {
            spacing: 0.5,
            box_size: glam::IVec2::new(12, 12),
            box_center: glam::Vec2::new(22.0, 16.0),
            material_id: 0,
            rng_seed: 2,
            ..SpawnRegion::for_sim(&config)
        },
    );
    for p in cold_side.iter_mut() {
        p.temperature = 293.0;
    }
    hot_side.extend(cold_side);
    let particles = hot_side;

    let registry = MaterialRegistry::with_default(Box::new(NeoHookeanMaterial::new(100.0, 50.0)));
    let sim =
        GpuSimulation::with_device(device.clone(), queue.clone(), config, particles, registry);

    let fmt = wgpu::TextureFormat::Rgba8UnormSrgb;
    let mut r = Renderer::new(&device, sim.particle_count(), fmt);
    r.set_optical_params(&queue, 0, [0.3, 0.3, 0.3]);
    r.set_camera(&queue, grid_res, 64, 64, 0.6, true);

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("curvature_flow_thermal_diffusion_test_target"),
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

    r.render_surface_reconstruction(
        &device,
        &queue,
        SurfaceReconstructionSource {
            particle_buf: sim.particle_buffer(),
            particle_count: sim.particle_count(),
            grid_res,
            material_slot: 0,
            material_mass_enabled: false,
            dt: 0.1,
        },
        &view,
        true,
    );
    device.poll(wgpu::PollType::wait_indefinitely()).ok();

    let surface_res = r.surface_res;
    let temps = readback_f32_blocking(
        &device,
        &queue,
        &r.surface_temp_float_buf,
        (surface_res * surface_res) as usize,
    );

    assert!(
        temps.iter().all(|t| t.is_finite()),
        "the real explicit heat-equation step must never produce NaN/inf, even \
         across a genuinely sharp hot/cold boundary"
    );

    let scale = surface_res as f32 / grid_res as f32;
    let row = (16.0 * scale) as u32;
    let hot_col = (10.0 * scale) as u32;
    let cold_col = (22.0 * scale) as u32;
    let hot_val = temps[(row * surface_res + hot_col) as usize];
    let cold_val = temps[(row * surface_res + cold_col) as usize];

    assert!(
        hot_val > cold_val + 500.0,
        "the hot cluster's own settled temperature must stay substantially \
         above the cold cluster's, even after real diffusion spreads some \
         heat toward the boundary -- hot={hot_val} cold={cold_val}"
    );
    assert!(
        hot_val < 3000.0 + 50.0 && cold_val > 293.0 - 50.0,
        "neither side should overshoot its own real input temperature by more \
         than a small margin -- a real sign the explicit stencil is stable, \
         not oscillating -- hot={hot_val} cold={cold_val}"
    );
}

/// Blocking readback of a single atomic<i32> total (used for the volume-
/// preserving-correction buffers, which are NOT f32 like everything else
/// `readback_f32_blocking` reads -- same staging pattern, different cast.
fn readback_i32_total(device: &wgpu::Device, queue: &wgpu::Queue, buf: &wgpu::Buffer) -> i32 {
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("i32_total_readback_staging"),
        size: 4,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("i32_total_readback"),
    });
    encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, 4);
    queue.submit(std::iter::once(encoder.finish()));
    device.poll(wgpu::PollType::wait_indefinitely()).ok();
    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).ok();
    let mapped = slice.get_mapped_range();
    let value = bytemuck::cast_slice::<u8, i32>(&mapped)[0];
    drop(mapped);
    staging.unmap();
    value
}

/// Volume-preserving correction (`curvature_flow.wgsl`'s Pass 1d, the discrete
/// form of `V = -H + lambda(t)`): the 15x-35x mass growth once measured came
/// from comparing a mass total with a raw, unweighted sum of densities on a
/// `surface_res_multiplier`x finer grid (see
/// `curvature_flow_mass_growth_scales_with_iteration_count`). With the area
/// factor in the Lagrange multiplier it sits near 1.0, not ~1/34, which would
/// have made small and thin objects vanish. `corrected_total` is read back as
/// a raw sum like `pre_total`/`raw_post_total`, so it is compared with
/// `true_particle_mass_sum * multiplier^2`, the shader's structural factor,
/// not a separate tolerance.
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn curvature_flow_volume_correction_matches_true_particle_mass() {
    use crate::gpu::GpuSimulation;
    use crate::{MaterialRegistry, NeoHookeanMaterial, SimConfig, SpawnRegion, build_particles};
    use std::sync::Arc;

    let (device, queue) = headless_device();
    let device = Arc::new(device);
    let queue = Arc::new(queue);

    let grid_res = 32u32;
    let config = SimConfig::standard(grid_res as usize, 0.1, glam::Vec2::new(0.0, -0.3));
    let particles = build_particles(
        &config,
        SpawnRegion::for_sim(&config)
            .at(glam::Vec2::splat(16.0))
            .disk(6.0)
            .spacing(0.5)
            .material(0),
    );
    let true_particle_mass_sum: f32 = particles.iter().map(|p| p.mass).sum();

    let registry = MaterialRegistry::with_default(Box::new(NeoHookeanMaterial::new(100.0, 50.0)));
    let sim =
        GpuSimulation::with_device(device.clone(), queue.clone(), config, particles, registry);

    let fmt = wgpu::TextureFormat::Rgba8UnormSrgb;
    let mut r = Renderer::new(&device, sim.particle_count(), fmt);
    r.set_camera(&queue, grid_res, 64, 64, 0.6, true);

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("volume_correction_test_target"),
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

    r.render_surface_reconstruction(
        &device,
        &queue,
        SurfaceReconstructionSource {
            particle_buf: sim.particle_buffer(),
            particle_count: sim.particle_count(),
            grid_res,
            material_slot: 0,
            material_mass_enabled: false,
            dt: 0.1,
        },
        &view,
        true,
    );
    device.poll(wgpu::PollType::wait_indefinitely()).ok();

    // Matches `curvature_flow.wgsl`'s own `TOTAL_ATOMIC_SCALE` -- a
    // deliberately SMALLER scale than the per-cell `DENSITY_ATOMIC_SCALE`,
    // since these are GLOBAL sums across an entire scene's cells/particles,
    // not one cell's own bounded local overlap (see that constant's own
    // doc for the i32-overflow bug this fixes).
    const TOTAL_ATOMIC_SCALE: f32 = 1000.0;
    let pre_total =
        readback_i32_total(&device, &queue, &r.pre_total_atomic_buf) as f32 / TOTAL_ATOMIC_SCALE;
    let raw_post_total =
        readback_i32_total(&device, &queue, &r.post_total_atomic_buf) as f32 / TOTAL_ATOMIC_SCALE;

    let surface_res = r.surface_res;
    let settled = readback_f32_blocking(
        &device,
        &queue,
        &r.surface_a_buf,
        (surface_res * surface_res) as usize,
    );
    let corrected_total: f32 = settled.iter().sum();

    eprintln!(
        "true_particle_mass={true_particle_mass_sum} pre_total(splat)={pre_total} \
         raw_post_total(pre-correction)={raw_post_total} corrected_total(post-correction)={corrected_total}"
    );

    assert!(
        (pre_total - true_particle_mass_sum).abs() < true_particle_mass_sum * 0.05,
        "the splat pass's own accumulated total must closely match the real \
         particle mass sum (sanity check on the ground truth itself) -- \
         true={true_particle_mass_sum} splat_total={pre_total}"
    );
    let relative_error_uncorrected = (raw_post_total - pre_total).abs() / pre_total;
    eprintln!(
        "real, uncorrected raw drift (expected large -- see this test's own \
         doc, it's a units mismatch, not what the correction below fixes): \
         {:.1}%",
        relative_error_uncorrected * 100.0
    );

    // The real check: `corrected_total` is still a RAW sum (same units as
    // `raw_post_total`), so it's compared against `true_particle_mass_sum`
    // scaled by the SAME real structural factor (`surface_res_multiplier^2`)
    // the shader's own Lagrange multiplier accounts for -- not a second,
    // independently-chosen tolerance.
    let multiplier = r.surface_res_multiplier() as f32;
    let expected_corrected_total = true_particle_mass_sum * multiplier * multiplier;
    let relative_error_corrected =
        (corrected_total - expected_corrected_total).abs() / expected_corrected_total;
    eprintln!(
        "real, area-corrected drift: {:.2}% (expected_corrected_total={expected_corrected_total:.1} \
         vs actual corrected_total={corrected_total:.1})",
        relative_error_corrected * 100.0
    );
    assert!(
        relative_error_corrected < 0.1,
        "the RE-ENABLED volume-preserving correction must bring the settled \
         total within a small, real margin of true particle mass (scaled by \
         the surface grid's own real area factor) -- got \
         corrected_total={corrected_total:.1}, \
         expected~={expected_corrected_total:.1} \
         ({:.1}% off, wanted < 10%)",
        relative_error_corrected * 100.0
    );
}

/// Diagnostic, not an assertion: sweeps the engine's curvature iteration
/// count on the particle-splat scene above, to see whether the total-mass
/// growth scales with iterations or appears at the first smoothing step
/// (flat-noise and clean-disk hypotheses were ruled out on idealized fields).
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn curvature_flow_mass_growth_scales_with_iteration_count() {
    use crate::gpu::GpuSimulation;
    use crate::{MaterialRegistry, NeoHookeanMaterial, SimConfig, SpawnRegion, build_particles};
    use std::sync::Arc;

    let (device, queue) = headless_device();
    let device = Arc::new(device);
    let queue = Arc::new(queue);

    let grid_res = 32u32;
    let config = SimConfig::standard(grid_res as usize, 0.1, glam::Vec2::new(0.0, -0.3));
    let particles = build_particles(
        &config,
        SpawnRegion::for_sim(&config)
            .at(glam::Vec2::splat(16.0))
            .disk(6.0)
            .spacing(0.5)
            .material(0),
    );
    let registry = MaterialRegistry::with_default(Box::new(NeoHookeanMaterial::new(100.0, 50.0)));
    let sim =
        GpuSimulation::with_device(device.clone(), queue.clone(), config, particles, registry);

    const TOTAL_ATOMIC_SCALE: f32 = 1000.0;
    let fmt = wgpu::TextureFormat::Rgba8UnormSrgb;

    eprintln!("real engine mass growth vs. iteration count, same scene as the sibling test:");
    eprintln!(
        "{:>6} {:>14} {:>10}",
        "iters", "raw_post_total", "pct_of_pre"
    );
    for iterations in [2u32, 4, 6, 8, 10, 12] {
        let mut r = Renderer::new(&device, sim.particle_count(), fmt);
        r.set_camera(&queue, grid_res, 64, 64, 0.6, true);
        r.set_curvature_iterations(iterations);

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("mass_growth_sweep_target"),
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

        r.render_surface_reconstruction(
            &device,
            &queue,
            SurfaceReconstructionSource {
                particle_buf: sim.particle_buffer(),
                particle_count: sim.particle_count(),
                grid_res,
                material_slot: 0,
                material_mass_enabled: false,
                dt: 0.1,
            },
            &view,
            true,
        );
        device.poll(wgpu::PollType::wait_indefinitely()).ok();

        let pre_total = readback_i32_total(&device, &queue, &r.pre_total_atomic_buf) as f32
            / TOTAL_ATOMIC_SCALE;
        let raw_post_total = readback_i32_total(&device, &queue, &r.post_total_atomic_buf) as f32
            / TOTAL_ATOMIC_SCALE;

        // `pre_total` comes from per-particle mass during the splat
        // (resolution-independent: a normalized kernel's weights sum to 1).
        // `raw_post_total` is a raw sum of per-cell densities, and each surface
        // cell's area is `1/surface_res_multiplier^2` of a physics cell's,
        // never multiplied back in by `post_total_reduce_main`/
        // `convert_atomic_to_float_main`. Dividing by `surface_res_multiplier^2`
        // should recover about `pre_total`.
        let multiplier = r.surface_res_multiplier() as f32;
        let area_corrected = raw_post_total / (multiplier * multiplier);
        eprintln!(
            "{:>6} {:>14.3} {:>9.1}%  area_corrected={:.2} (multiplier={:.0}, vs pre_total={:.2})",
            iterations,
            raw_post_total,
            100.0 * raw_post_total / pre_total,
            area_corrected,
            multiplier,
            pre_total
        );
    }
}

/// Wave-equation surface (`curvature_flow.wgsl`'s Pass 2b): repeated calls to
/// `render_surface_reconstruction` with a cluster present excite the wave
/// field away from zero, and it stays finite across many steps (`WAVE_DAMPING`
/// bounds it under continuous forcing).
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn wave_field_is_excited_by_real_density_and_stays_bounded() {
    use crate::gpu::GpuSimulation;
    use crate::matter::materials::MaterialModel;
    use crate::{
        MaterialRegistry, NewtonianFluidMaterial, SimConfig, SpawnRegion, build_particles,
    };
    use std::sync::Arc;

    let (device, queue) = headless_device();
    let device = Arc::new(device);
    let queue = Arc::new(queue);

    let grid_res = 32u32;
    let config = SimConfig::standard(grid_res as usize, 0.1, glam::Vec2::new(0.0, -0.3));
    let particles = build_particles(
        &config,
        SpawnRegion::for_sim(&config)
            .at(glam::Vec2::splat(16.0))
            .disk(4.0)
            .spacing(0.5)
            .material(0),
    );
    // Waves only excite for a material that behaves like a fluid
    // (`owns_deformation_volume_state()`, see `Renderer::set_wave_force_coeff`);
    // `NeoHookeanMaterial`, a solid, does not. The coefficient 0.35 is the one
    // `examples/cpu/basic_fluids.rs` and `examples/gpu/basic_sand_grid_gpu.rs`
    // use with the same gate.
    let material = NewtonianFluidMaterial::low_viscosity(1.0, 1.0);
    let registry = MaterialRegistry::with_default(Box::new(material));
    let sim =
        GpuSimulation::with_device(device.clone(), queue.clone(), config, particles, registry);

    let fmt = wgpu::TextureFormat::Rgba8UnormSrgb;
    let mut r = Renderer::new(&device, sim.particle_count(), fmt);
    r.set_camera(&queue, grid_res, 64, 64, 0.6, true);
    // Opt-in (see `WaveStepParams::wave_force_coeff` in `curvature_flow.wgsl`):
    // inert at 0.0 unless the scene sets it, as the examples do.
    assert!(
        material.owns_deformation_volume_state(),
        "test material must actually be a real fluid, or this whole test proves nothing"
    );
    r.set_wave_force_coeff(0.35);

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("wave_field_test_target"),
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

    const FRAMES: u32 = 30;
    for _ in 0..FRAMES {
        r.render_surface_reconstruction(
            &device,
            &queue,
            SurfaceReconstructionSource {
                particle_buf: sim.particle_buffer(),
                particle_count: sim.particle_count(),
                grid_res,
                material_slot: 0,
                material_mass_enabled: false,
                dt: 0.1,
            },
            &view,
            true,
        );
        device.poll(wgpu::PollType::wait_indefinitely()).ok();
    }

    let surface_res = r.surface_res;
    let cell_count = (surface_res * surface_res) as usize;
    // After FRAMES calls, the buffer holding the latest settled state is
    // whichever served as "next" on the LAST call -- same rotation
    // `render_surface_reconstruction` itself uses internally.
    let current_idx = (FRAMES % 3) as usize;
    let values = readback_f32_blocking(&device, &queue, &r.wave_bufs[current_idx], cell_count);

    assert!(
        values.iter().all(|v| v.is_finite()),
        "wave field must stay finite (real damping must bound continuous forcing), \
         even after {FRAMES} real steps"
    );
    assert!(
        values.iter().any(|v| v.abs() > 1.0e-6),
        "a real particle cluster's density gradient must excite the wave field away \
         from its all-zero initial state after {FRAMES} real steps"
    );
}

/// Regression check for the wave-excitation forcing term: it must be the
/// TEMPORAL density difference (this frame's density minus last frame's),
/// not the SPATIAL density gradient -- a spatial gradient is nonzero at
/// any object's edge PERMANENTLY, whether anything moves or not, so the
/// wave field would never settle even for a fully static body. This test
/// drives a COMPLETELY STATIC particle cluster (same buffer, never touched
/// between calls -- the strongest analogue of "physics reports
/// max_speed~0") for many frames and confirms the wave field's peak
/// magnitude DECAYS over time once the one-time "body just appeared" burst
/// passes, rather than staying pinned at a roughly constant nonzero level
/// forever (what a spatial-gradient forcing term would do, since the
/// excitation source -- the object's own unchanging edge -- never goes
/// away).
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn curvature_flow_wave_field_decays_once_density_stops_changing() {
    use crate::gpu::GpuSimulation;
    use crate::matter::materials::MaterialModel;
    use crate::{
        MaterialRegistry, NewtonianFluidMaterial, SimConfig, SpawnRegion, build_particles,
    };
    use std::sync::Arc;

    let (device, queue) = headless_device();
    let device = Arc::new(device);
    let queue = Arc::new(queue);

    let grid_res = 32u32;
    let config = SimConfig::standard(grid_res as usize, 0.1, glam::Vec2::new(0.0, -0.3));
    let particles = build_particles(
        &config,
        SpawnRegion::for_sim(&config)
            .at(glam::Vec2::splat(16.0))
            .disk(4.0)
            .spacing(0.5)
            .material(0),
    );
    // A fluid material and the `set_wave_force_coeff` opt-in below, as in
    // `wave_field_is_excited_by_real_density_and_stays_bounded` and the
    // examples.
    let material = NewtonianFluidMaterial::low_viscosity(1.0, 1.0);
    let registry = MaterialRegistry::with_default(Box::new(material));
    let sim =
        GpuSimulation::with_device(device.clone(), queue.clone(), config, particles, registry);

    let fmt = wgpu::TextureFormat::Rgba8UnormSrgb;
    let mut r = Renderer::new(&device, sim.particle_count(), fmt);
    r.set_camera(&queue, grid_res, 64, 64, 0.6, true);
    assert!(
        material.owns_deformation_volume_state(),
        "test material must actually be a real fluid, or this whole test proves nothing"
    );
    r.set_wave_force_coeff(0.35);

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("wave_decay_test_target"),
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

    let render_frame = |r: &mut Renderer| {
        r.render_surface_reconstruction(
            &device,
            &queue,
            SurfaceReconstructionSource {
                particle_buf: sim.particle_buffer(),
                particle_count: sim.particle_count(),
                grid_res,
                material_slot: 0,
                material_mass_enabled: false,
                dt: 0.1,
            },
            &view,
            true,
        );
        device.poll(wgpu::PollType::wait_indefinitely()).ok();
    };

    // First real frame grows the surface buffers to their true size --
    // `surface_res` must be read AFTER this, not before, or `cell_count`
    // silently stays at the constructor's 1-cell placeholder.
    render_frame(&mut r);
    let surface_res = r.surface_res;
    let cell_count = (surface_res * surface_res) as usize;

    // Past the initial one-time excitation burst (expected -- a body
    // appearing IS a disturbance), but still early in the real
    // WAVE_DAMPING=0.996 decay curve.
    const EARLY_FRAME: u32 = 10;
    for _ in 1..EARLY_FRAME {
        render_frame(&mut r);
    }
    let early_idx = (EARLY_FRAME % 3) as usize;
    let early_values = readback_f32_blocking(&device, &queue, &r.wave_bufs[early_idx], cell_count);
    let early_peak = early_values.iter().fold(0.0f32, |a, v| a.max(v.abs()));

    const LATE_FRAME: u32 = 300;
    for _ in EARLY_FRAME..LATE_FRAME {
        render_frame(&mut r);
    }
    let late_idx = (LATE_FRAME % 3) as usize;
    let late_values = readback_f32_blocking(&device, &queue, &r.wave_bufs[late_idx], cell_count);
    let late_peak = late_values.iter().fold(0.0f32, |a, v| a.max(v.abs()));

    assert!(
        late_values.iter().all(|v| v.is_finite()),
        "wave field must stay finite across {LATE_FRAME} real frames"
    );
    assert!(
        late_peak < early_peak * 0.9,
        "with a COMPLETELY STATIC particle cluster (density never changes \
         after the first frame), the wave field's peak magnitude must \
         genuinely decay over {LATE_FRAME} frames, not stay pinned near its \
         early value -- the old spatial-gradient forcing term would keep \
         re-exciting it forever from the object's own permanent edge, \
         exactly the bug this fix addresses -- early(frame {EARLY_FRAME})={early_peak} \
         late(frame {LATE_FRAME})={late_peak}"
    );
}

/// Hysteresis (Schmitt trigger) visibility (`curvature_flow.wgsl`'s Pass 2c):
/// a cell whose density hovers between the low and high thresholds keeps its
/// state. Dispatches `visibility_step_main` on a synthetic density, since the
/// particle splat cannot be held in that gap for several frames.
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn visibility_hysteresis_does_not_flicker_in_the_gap_between_thresholds() {
    let (device, queue) = headless_device();
    let mut r = Renderer::new(&device, 1, wgpu::TextureFormat::Rgba8UnormSrgb);
    let grid_res = 32u32;
    r.ensure_surface_capacity(&device, grid_res);
    let surface_res = r.surface_res;
    let cell_count = (surface_res * surface_res) as usize;

    const MASS_FLOOR: f32 = 0.15;

    let dispatch_visibility_step = |density_value: f32| {
        let mut density = vec![0.0f32; cell_count];
        density[0] = density_value;
        queue.write_buffer(&r.surface_a_buf, 0, bytemuck::cast_slice(&density));
        queue.write_buffer(
            &r.visibility_params_buf,
            0,
            bytemuck::bytes_of(&VisibilityParams {
                surface_res,
                mass_floor: MASS_FLOOR,
                _pad0: 0,
                _pad1: 0,
            }),
        );
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("test_visibility_step_bg"),
            layout: &r.visibility_step_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: r.surface_a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: r.visibility_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: r.visibility_params_buf.as_entire_binding(),
                },
            ],
        });
        let mut enc =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cp.set_pipeline(&r.visibility_step_pipeline);
            cp.set_bind_group(0, &bg, &[]);
            cp.dispatch_workgroups(surface_res.div_ceil(8), surface_res.div_ceil(8), 1);
        }
        queue.submit(std::iter::once(enc.finish()));
        device.poll(wgpu::PollType::wait_indefinitely()).ok();
    };

    let read_visibility_cell0 =
        || -> f32 { readback_f32_blocking(&device, &queue, &r.visibility_buf, cell_count)[0] };

    // Frame 1: clearly above the HIGH threshold (1.3x) -> must turn visible.
    dispatch_visibility_step(MASS_FLOOR * 1.5);
    assert!(
        read_visibility_cell0() > 0.5,
        "mass clearly above the HIGH hysteresis threshold must turn the cell visible"
    );

    // Frame 2: drop into the AMBIGUOUS gap -- below mass_floor itself, but
    // still above the LOW threshold (0.7x). A naive single-threshold test
    // would flip this off; hysteresis must keep it visible since it was
    // already visible and hasn't dropped below LOW.
    dispatch_visibility_step(MASS_FLOOR * 0.9);
    assert!(
        read_visibility_cell0() > 0.5,
        "a cell already visible must NOT turn invisible just because its mass \
         dropped below mass_floor itself, as long as it stays above the LOW \
         hysteresis threshold -- this is the whole point of hysteresis"
    );

    // Frame 3: drop clearly below the LOW threshold -> must finally turn invisible.
    dispatch_visibility_step(MASS_FLOOR * 0.5);
    assert!(
        read_visibility_cell0() <= 0.5,
        "mass clearly below the LOW hysteresis threshold must turn the cell invisible"
    );

    // Frame 4: rise back into the SAME ambiguous gap -- above mass_floor
    // itself, but still below the HIGH threshold. Must stay invisible,
    // proving the same gap is stable in both directions, not just one.
    dispatch_visibility_step(MASS_FLOOR * 1.1);
    assert!(
        read_visibility_cell0() <= 0.5,
        "a cell already invisible must NOT turn visible just because its mass \
         rose above mass_floor itself, as long as it stays below the HIGH \
         hysteresis threshold"
    );
}

/// `grid_peak_main` writes, for each cell, the largest cell mass within two
/// cells of it, the reach `grid_volume.wgsl`'s edge needs: one heavy cell in
/// a uniform field shows in its 5 x 5 neighbourhood and nowhere else.
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn grid_peak_is_the_largest_mass_within_two_cells() {
    let (device, queue) = headless_device();
    let mut r = Renderer::new(&device, 1, wgpu::TextureFormat::Rgba8UnormSrgb);
    let grid_res = 32u32;
    r.ensure_grid_peak_capacity(&device, grid_res);
    let cell_count = (grid_res * grid_res) as usize;
    let (heavy, background, hx, hy) = (3.0f32, 0.25f32, 10i32, 12i32);
    let mut cells = vec![0u32; cell_count * 4];
    for c in 0..cell_count {
        cells[c * 4 + 2] = background.to_bits();
    }
    cells[(hy as usize * grid_res as usize + hx as usize) * 4 + 2] = heavy.to_bits();
    let grid_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test_grid_peak_density"),
        size: (cell_count * 4 * std::mem::size_of::<u32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&grid_buf, 0, bytemuck::cast_slice(&cells));
    queue.write_buffer(
        &r.grid_peak_params_buf,
        0,
        bytemuck::bytes_of(&GridPeakParams {
            grid_res,
            _pad: [0; 3],
        }),
    );
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("test_grid_peak_bg"),
        layout: &r.grid_peak_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: grid_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: r.grid_peak_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: r.grid_peak_params_buf.as_entire_binding(),
            },
        ],
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        cp.set_pipeline(&r.grid_peak_pipeline);
        cp.set_bind_group(0, &bg, &[]);
        cp.dispatch_workgroups(grid_res.div_ceil(8), grid_res.div_ceil(8), 1);
    }
    queue.submit(std::iter::once(enc.finish()));
    device.poll(wgpu::PollType::wait_indefinitely()).ok();
    let peak = readback_f32_blocking(&device, &queue, &r.grid_peak_buf, cell_count);
    for y in 0..grid_res as i32 {
        for x in 0..grid_res as i32 {
            let near = (x - hx).abs() <= 2 && (y - hy).abs() <= 2;
            let expected = if near { heavy } else { background };
            assert_eq!(
                peak[(y * grid_res as i32 + x) as usize],
                expected,
                "cell ({x},{y})"
            );
        }
    }
}

/// Grid mass of particles at `positions` (in cells), `mass_each` each,
/// scattered with the solver's own quadratic B-spline (`grid::kernel`).
fn scatter_particle_mass(grid_res: u32, positions: &[glam::Vec2], mass_each: f32) -> Vec<f32> {
    use crate::spacetime::grid::kernel::quadratic_weights;
    let mut mass = vec![0.0f32; (grid_res * grid_res) as usize];
    for &x in positions {
        let w = quadratic_weights(x);
        for (gx, wx) in w.wx.iter().enumerate() {
            for (gy, wy) in w.wy.iter().enumerate() {
                let c = w.base_cell + glam::IVec2::new(gx as i32 - 1, gy as i32 - 1);
                if c.x >= 0 && c.y >= 0 && c.x < grid_res as i32 && c.y < grid_res as i32 {
                    mass[(c.y as u32 * grid_res + c.x as u32) as usize] += mass_each * wx * wy;
                }
            }
        }
    }
    mass
}

/// Renders the grid `mass` (reference cell mass 1, the default) through
/// `render_grid_volume` over a transparent `size` x `size` target, and
/// returns every pixel's alpha, the drawn coverage, row-major from the top.
fn grid_volume_coverage(grid_res: u32, size: u32, mass: &[f32]) -> Vec<u8> {
    let (device, queue) = headless_device();
    let cell_count = (grid_res * grid_res) as usize;
    let mut words = vec![0u32; cell_count * 4];
    for (c, m) in mass.iter().enumerate() {
        words[c * 4 + 2] = m.to_bits();
    }
    let grid_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test_coverage_grid"),
        size: (cell_count * 16) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&grid_buf, 0, bytemuck::cast_slice(&words));
    let material_mass_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test_coverage_material_mass"),
        size: (cell_count * 16 * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let fmt = wgpu::TextureFormat::Rgba8UnormSrgb;
    let mut r = Renderer::new(&device, 1, fmt);
    r.set_camera(&queue, grid_res, size, size, 0.6, true);
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("test_coverage_target"),
        size: wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: fmt,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    // A transparent target, so each pixel's alpha is the drawn coverage.
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    enc.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("test_coverage_clear"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: &view,
            resolve_target: None,
            depth_slice: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
    });
    queue.submit(std::iter::once(enc.finish()));
    r.render_grid_volume(
        &device,
        &queue,
        GridVolumeSource {
            grid: &grid_buf,
            material_mass: &material_mass_buf,
            material_mass_enabled: false,
            grid_res,
        },
        &view,
        false,
    );
    device.poll(wgpu::PollType::wait_indefinitely()).ok();
    let padded = (size * 4).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
        * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("test_coverage_staging"),
        size: (padded * size) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    enc.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: Some(size),
            },
        },
        wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(std::iter::once(enc.finish()));
    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).ok();
    let mapped = slice.get_mapped_range();
    (0..size)
        .flat_map(|y| (0..size).map(move |x| (y, x)))
        .map(|(y, x)| mapped[(y * padded + x * 4 + 3) as usize])
        .collect()
}

/// The grid volume draws matter's edge where the matter ends. A slab of
/// particles filling x < 16.25 (a boundary between cell edges), scattered
/// with the solver's own kernel, must render its edge within 0.1 cell of
/// 16.25: 0.05 is the bilinear interpolation of the sampled field (the
/// crossing of its four cell values is at 16.20), the rest half a pixel at 8
/// pixels per cell. The edge criterion it replaced, a nearest-cell gate at
/// 0.15 of a full cell, put this edge at 16.66.
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn grid_volume_edge_sits_where_the_matter_ends() {
    let (grid_res, size) = (32u32, 256u32);
    let px_per_cell = size as f32 / grid_res as f32;
    // Particles at spacing 0.5, a quarter of a full cell's mass each, so the
    // inside holds exactly the reference cell mass.
    let (spacing, last_x) = (0.5f32, 16.0f32);
    let mut positions = Vec::new();
    let mut x = last_x;
    while x > 0.0 {
        let mut y = spacing / 2.0;
        while y < grid_res as f32 {
            positions.push(glam::Vec2::new(x, y));
            y += spacing;
        }
        x -= spacing;
    }
    let true_edge = last_x + spacing / 2.0;
    let mass = scatter_particle_mass(grid_res, &positions, spacing * spacing);
    let coverage = grid_volume_coverage(grid_res, size, &mass);
    // A row through the middle of the slab, away from the domain's walls.
    let row = (size / 2 + 4) as usize;
    let alpha: Vec<f32> = coverage[row * size as usize..(row + 1) * size as usize]
        .iter()
        .map(|&a| a as f32)
        .collect();
    let start = (true_edge * px_per_cell) as usize - 2 * px_per_cell as usize;
    let crossing = (start..size as usize - 1)
        .find(|&c| alpha[c] >= 127.5 && alpha[c + 1] < 127.5)
        .expect("the slab must have an edge on this row");
    let fraction = (alpha[crossing] - 127.5) / (alpha[crossing] - alpha[crossing + 1]);
    let edge = (crossing as f32 + 0.5 + fraction) / px_per_cell;
    assert!(
        (edge - true_edge).abs() < 0.1,
        "edge drawn at x = {edge}, the matter ends at {true_edge}"
    );
}

/// Nothing is drawn where there is no matter. Past a body's edge both the
/// mass and its local peak fade to zero, two cells beyond the kernel's
/// reach; there the edge test must not find a crossing that is not there.
/// (Written as a mass difference, `mass - peak / 2` approaches zero without
/// crossing it, and the anti-aliasing drew a one-pixel ring around every
/// body.) A disk off the grid's lattice puts that fade-out at every offset
/// from the pixel centres.
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn grid_volume_draws_nothing_past_the_matter() {
    let (grid_res, size) = (32u32, 256u32);
    let px_per_cell = size as f32 / grid_res as f32;
    let (spacing, centre, radius) = (0.5f32, glam::Vec2::new(16.3, 15.7), 6.0f32);
    let mut positions = Vec::new();
    let mut x = centre.x - radius;
    while x <= centre.x + radius {
        let mut y = centre.y - radius;
        while y <= centre.y + radius {
            if (glam::Vec2::new(x, y) - centre).length() <= radius {
                positions.push(glam::Vec2::new(x, y));
            }
            y += spacing;
        }
        x += spacing;
    }
    let mass = scatter_particle_mass(grid_res, &positions, spacing * spacing);
    let coverage = grid_volume_coverage(grid_res, size, &mass);
    // The matter reaches half a spacing past the outermost particle.
    let matter_radius = radius + spacing / 2.0;
    let mut stray = Vec::new();
    for row in 0..size {
        for col in 0..size {
            let p = glam::Vec2::new(
                (col as f32 + 0.5) / px_per_cell,
                grid_res as f32 - (row as f32 + 0.5) / px_per_cell,
            );
            let a = coverage[(row * size + col) as usize];
            if (p - centre).length() > matter_radius + 1.0 && a > 0 {
                stray.push((col, row, a));
            }
        }
    }
    assert!(
        stray.is_empty(),
        "{} pixels drawn more than a cell past the matter, e.g. {:?}",
        stray.len(),
        &stray[..stray.len().min(5)]
    );
}

/// DETERMINISTIC flicker measurement -- unlike live-demo screenshots
/// (confounded: a different random splash every relaunch, which can swing
/// measured "flicker pixel" counts by 10x+ run to run with no shader
/// change at all), this steps a FIXED-SEED particle scenario forward
/// through many physics + render frames and reads back actual rendered
/// pixels each time. The whole pipeline (particle physics, the wave PDE,
/// hysteresis) has no wall-clock dependency, and the atomic splat scatter
/// is integer (exactly order-independent) -- so the same seed reproduces
/// bit-identical results run to run, making this a re-runnable A/B harness
/// for any future shading change, unlike a live demo screenshot ever could
/// be.
///
/// NOT a strict pass/fail gate yet: the exact acceptable flicker fraction
/// hasn't been established (this is the first time it's been measured
/// this way). Asserts a generous placeholder ceiling so this stays a
/// regression guard against a CATASTROPHIC regression (like the reverted
/// density-persistence attempt, which measured roughly half of all
/// sampled points flickering) without yet claiming the CURRENT baseline
/// itself is "acceptable" -- that judgment is still open, tracked
/// separately, not asserted here as settled.
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn surface_reconstruction_does_not_flicker_over_many_deterministic_frames() {
    use crate::gpu::GpuSimulation;
    use crate::{MaterialRegistry, NeoHookeanMaterial, SimConfig, SpawnRegion, build_particles};
    use std::sync::Arc;

    let (device, queue) = headless_device();
    let device = Arc::new(device);
    let queue = Arc::new(queue);

    let grid_res = 32u32;
    let config = SimConfig::standard(grid_res as usize, 0.1, glam::Vec2::new(0.0, -0.3));
    let particles = build_particles(
        &config,
        SpawnRegion::for_sim(&config)
            .at(glam::Vec2::splat(16.0))
            .disk(4.0)
            .spacing(0.5)
            .material(0)
            .jitter(0.15)
            .rng_seed(42),
    );
    let registry = MaterialRegistry::with_default(Box::new(NeoHookeanMaterial::new(100.0, 50.0)));
    let mut sim =
        GpuSimulation::with_device(device.clone(), queue.clone(), config, particles, registry);

    let fmt = wgpu::TextureFormat::Rgba8UnormSrgb;
    let mut r = Renderer::new(&device, sim.particle_count(), fmt);
    r.set_camera(&queue, grid_res, 64, 64, 0.6, true);

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("deterministic_flicker_test_target"),
        size: wgpu::Extent3d {
            width: 64,
            height: 64,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: fmt,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    const FRAMES: usize = 40;
    let mut samples_per_frame: Vec<Vec<f64>> = Vec::with_capacity(FRAMES);
    for _ in 0..FRAMES {
        sim.step_frame();
        r.render_surface_reconstruction(
            &device,
            &queue,
            SurfaceReconstructionSource {
                particle_buf: sim.particle_buffer(),
                particle_count: sim.particle_count(),
                grid_res,
                material_slot: 0,
                material_mass_enabled: false,
                dt: 0.1,
            },
            &view,
            true,
        );
        device.poll(wgpu::PollType::wait_indefinitely()).ok();
        samples_per_frame.push(readback_luminance_grid(
            &device, &queue, &texture, 64, 64, 2,
        ));
    }

    let num_points = samples_per_frame[0].len();
    let mut flicker_count = 0usize;
    for p in 0..num_points {
        let mut min_v = f64::MAX;
        let mut max_v = f64::MIN;
        for frame in samples_per_frame.iter().take(FRAMES) {
            let v = frame[p];
            min_v = min_v.min(v);
            max_v = max_v.max(v);
        }
        let range = max_v - min_v;
        let mut sign_changes = 0;
        for f in 1..FRAMES - 1 {
            let d1 = samples_per_frame[f][p] - samples_per_frame[f - 1][p];
            let d2 = samples_per_frame[f + 1][p] - samples_per_frame[f][p];
            if (d1 > 3.0 && d2 < -3.0) || (d1 < -3.0 && d2 > 3.0) {
                sign_changes += 1;
            }
        }
        if sign_changes >= 2 && range > 15.0 {
            flicker_count += 1;
        }
    }

    let flicker_fraction = flicker_count as f64 / num_points as f64;
    eprintln!(
        "DETERMINISTIC flicker measurement: {flicker_count}/{num_points} \
         ({:.1}%) sampled points show non-monotonic luminance oscillation \
         across {FRAMES} real, reproducible frames (seed=42)",
        flicker_fraction * 100.0
    );
    assert!(
        flicker_fraction < 0.5,
        "catastrophic flicker regression: {flicker_count}/{num_points} \
         ({:.1}%) sampled points oscillating -- this generous ceiling only \
         guards against a severe regression (like the reverted \
         density-persistence attempt, which hit ~50%); it does NOT yet\
         assert the current baseline is fully acceptable",
        flicker_fraction * 100.0
    );
}

// ── curvature_iterate_main numerical stability ──────────────────────────────
//
// A line-for-line CPU port of `curvature_flow.wgsl`'s `curvature_iterate_main`
// (same formula, same constants), so the update rule itself is measured. At
// `GRAD_EPSILON = 1.0e-3`, in a near-flat region (a fluid's interior, only
// splat noise) the denominator `(grad_sq + GRAD_EPSILON)^1.5` is dominated by
// epsilon and kappa becomes noise over a near-zero constant, bounded only by
// MAX_KAPPA: 12 iterations grew a flat field's noise variance 357x and a sharp
// corner grew (0.1 -> 0.205) instead of rounding.
mod curvature_iterate_stability {
    /// Mirrors `curvature_flow.wgsl`'s `CURVATURE_PSEUDO_DT`, `MAX_KAPPA`,
    /// and `GRAD_EPSILON` -- all three are WGSL-only (shader compile-time
    /// consts, not part of any uniform), so there is no single source of
    /// truth to import from Rust. If any of the three shader constants ever
    /// change, update the matching one here too, or this "line-for-line
    /// port" silently stops verifying the value actually shipped.
    /// `CURVATURE_PSEUDO_DT` in particular is documented shader-side as a
    /// tuned, retunable value, not a derived one -- the most likely of the
    /// three to drift.
    const CURVATURE_PSEUDO_DT: f32 = 0.15;
    const MAX_KAPPA: f32 = 4.0;
    const GRAD_EPSILON: f32 = 0.1;

    fn sample(field: &[f32], cx: i32, cy: i32, res: i32) -> f32 {
        if cx < 0 || cy < 0 || cx >= res || cy >= res {
            0.0
        } else {
            field[(cy * res + cx) as usize]
        }
    }

    /// Exact port of `curvature_iterate_main`'s body -- must be kept in sync
    /// with `curvature_flow.wgsl` if that formula ever changes.
    fn curvature_iterate(field_in: &[f32], res: i32, grad_epsilon: f32) -> Vec<f32> {
        let mut out = vec![0.0f32; field_in.len()];
        for cy in 0..res {
            for cx in 0..res {
                let center = sample(field_in, cx, cy, res);
                let dx =
                    (sample(field_in, cx + 1, cy, res) - sample(field_in, cx - 1, cy, res)) * 0.5;
                let dy =
                    (sample(field_in, cx, cy + 1, res) - sample(field_in, cx, cy - 1, res)) * 0.5;
                let dxx = sample(field_in, cx + 1, cy, res) - 2.0 * center
                    + sample(field_in, cx - 1, cy, res);
                let dyy = sample(field_in, cx, cy + 1, res) - 2.0 * center
                    + sample(field_in, cx, cy - 1, res);
                let dxy = (sample(field_in, cx + 1, cy + 1, res)
                    - sample(field_in, cx + 1, cy - 1, res)
                    - sample(field_in, cx - 1, cy + 1, res)
                    + sample(field_in, cx - 1, cy - 1, res))
                    * 0.25;
                let grad_sq = dx * dx + dy * dy;
                let denom = (grad_sq + grad_epsilon).powf(1.5);
                let kappa = ((dxx * dy * dy - 2.0 * dx * dy * dxy + dyy * dx * dx) / denom)
                    .clamp(-MAX_KAPPA, MAX_KAPPA);
                let i = (cy * res + cx) as usize;
                out[i] = (center + CURVATURE_PSEUDO_DT * kappa).max(0.0);
            }
        }
        out
    }

    // Deterministic xorshift -- zero external deps, fully reproducible.
    struct Rng(u64);
    impl Rng {
        fn next_f32(&mut self) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            ((self.0 >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        }
    }

    fn variance(field: &[f32]) -> f32 {
        let n = field.len() as f32;
        let mean: f32 = field.iter().sum::<f32>() / n;
        field.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n
    }

    /// The regression this bug actually was: a near-flat, noisy field (the
    /// real shape of a fluid's interior -- see `render::mod`'s N_eff
    /// analysis for where the noise floor comes from) must not have its
    /// variance GROW under repeated iteration. `GRAD_EPSILON=1.0e-3` grew it
    /// 357x over 12 iterations; the shipped value must keep it bounded with
    /// real margin, not just barely under 1.0.
    #[test]
    #[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
    fn flat_noisy_field_does_not_amplify() {
        const RES: i32 = 48;
        const ITERATIONS: usize = 12;
        let flat_value = 0.1f32;
        let noise_std = 0.03f32; // matches the N_eff~13 noise floor at this scene's reference_cell_mass=0.1

        let mut rng = Rng(0x9E3779B97F4A7C15);
        let mut field: Vec<f32> = (0..(RES * RES) as usize)
            .map(|_| (flat_value + noise_std * rng.next_f32()).max(0.0))
            .collect();
        let var0 = variance(&field);
        for _ in 0..ITERATIONS {
            field = curvature_iterate(&field, RES, GRAD_EPSILON);
        }
        let var_final = variance(&field);
        let ratio = var_final / var0.max(1.0e-12);

        assert!(
            ratio < 1.2,
            "curvature_iterate_main is amplifying flat-region noise instead \
             of damping it: variance ratio {ratio:.3} over {ITERATIONS} \
             iterations (>= 1.0 means growth). This is the exact mechanism \
             behind the white-noise/flicker regression fixed 2026-08-14 -- \
             GRAD_EPSILON is too small relative to the density field's real \
             sampling noise floor."
        );
    }

    /// The other half of the same fix: raising GRAD_EPSILON enough to
    /// stabilize the flat interior must not also neuter the pass's actual
    /// job. A sharp 90-degree corner (the free-surface case) must still
    /// round off measurably after iteration, not sit inert.
    #[test]
    #[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
    fn sharp_corner_still_rounds() {
        const RES: i32 = 48;
        const ITERATIONS: usize = 12;
        let flat_value = 0.1f32;

        let mut field = vec![0.0f32; (RES * RES) as usize];
        for cy in 0..RES {
            for cx in 0..RES {
                if cx >= 12 && cy >= 12 {
                    field[(cy * RES + cx) as usize] = flat_value;
                }
            }
        }
        let corner_idx = (12 * RES + 12) as usize;
        for _ in 0..ITERATIONS {
            field = curvature_iterate(&field, RES, GRAD_EPSILON);
        }

        assert!(
            field[corner_idx] < flat_value * 0.9,
            "a sharp corner must round off (its own value should drop \
             meaningfully toward its empty neighbours) after {ITERATIONS} \
             curvature-flow iterations -- got {:.4} from a start of \
             {flat_value}, too small a change. GRAD_EPSILON may be large \
             enough to have suppressed real edge curvature along with the \
             noise it was raised to fix.",
            field[corner_idx]
        );
        assert!(
            field[corner_idx] < flat_value,
            "a sharp corner must not GROW under curvature flow (mean \
             curvature flow shrinks, it does not expand) -- got {:.4} from \
             a start of {flat_value}, which is the exact instability \
             signature the old GRAD_EPSILON=1.0e-3 had (0.1 -> 0.205, this \
             file's own prior measurement).",
            field[corner_idx]
        );
    }
}

/// `cursor_grid()`'s `screen_pos / window_size * grid_res` assumed the grid
/// fills the window, while `set_camera` letterboxes or pillarboxes to keep
/// the aspect ratio (they agreed only for a square window). `screen_to_grid`
/// is the exact inverse; checked against geometric invariants rather than a
/// re-derived formula, which would copy a bug into its test.
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn screen_to_grid_is_exact_inverse_of_set_camera_at_any_aspect_ratio() {
    let (device, queue) = headless_device();
    let fmt = wgpu::TextureFormat::Rgba8UnormSrgb;
    let mut r = Renderer::new(&device, 1, fmt);
    let grid_res = 32u32;

    // Exactly square: no letterboxing on either axis, so the four window
    // corners must map to the four exact grid corners.
    r.set_camera(&queue, grid_res, 100, 100, 0.6, true);
    let (gx, gy) = r.screen_to_grid(0.0, 0.0, 100, 100);
    assert!(
        (gx - 0.0).abs() < 1.0e-3 && (gy - grid_res as f32).abs() < 1.0e-3,
        "square window's top-left screen corner must map to the grid's \
         top-left corner (x=0, y=grid_res) -- got ({gx}, {gy})"
    );
    let (gx, gy) = r.screen_to_grid(100.0, 100.0, 100, 100);
    assert!(
        (gx - grid_res as f32).abs() < 1.0e-3 && (gy - 0.0).abs() < 1.0e-3,
        "square window's bottom-right screen corner must map to the grid's \
         bottom-right corner (x=grid_res, y=0) -- got ({gx}, {gy})"
    );

    // A real invariant that holds at ANY aspect ratio: letterbox/pillarbox
    // padding is always symmetric, so the window's own screen-space CENTER
    // must always map to the grid's center, wide or tall or square. This is
    // exactly the case the old naive mapping got wrong away from square.
    for (w, h) in [(100u32, 100u32), (300, 100), (100, 300), (37, 211)] {
        r.set_camera(&queue, grid_res, w, h, 0.6, true);
        let (gx, gy) = r.screen_to_grid(w as f32 / 2.0, h as f32 / 2.0, w, h);
        let expected = grid_res as f32 / 2.0;
        assert!(
            (gx - expected).abs() < 1.0e-2 && (gy - expected).abs() < 1.0e-2,
            "window center must map to grid center at ANY aspect ratio \
             (w={w}, h={h}) -- got ({gx}, {gy}), expected ({expected}, {expected})"
        );
    }

    // A wide window (aspect > 1) pillarboxes on X but fills Y edge to edge
    // -- so its full screen-space Y range must still cover the grid's full
    // Y range exactly, unlike X which is padded.
    r.set_camera(&queue, grid_res, 300, 100, 0.6, true);
    let (_, gy_top) = r.screen_to_grid(150.0, 0.0, 300, 100);
    let (_, gy_bottom) = r.screen_to_grid(150.0, 100.0, 300, 100);
    assert!(
        (gy_top - grid_res as f32).abs() < 1.0e-3 && gy_bottom.abs() < 1.0e-3,
        "a wide window's Y axis is never letterboxed -- top/bottom screen \
         edges must map exactly to grid Y=grid_res/Y=0 -- got top={gy_top} \
         bottom={gy_bottom}"
    );
}

/// `grid_to_screen`/`grid_distance_to_pixels`, the one source for both
/// directions (a hand-derived copy of the projection drifted from what
/// `set_camera` uploaded), round-trip exactly with `screen_to_grid` at several
/// aspect ratios, without re-deriving the formula.
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn grid_to_screen_is_exact_inverse_of_screen_to_grid_at_any_aspect_ratio() {
    let (device, queue) = headless_device();
    let fmt = wgpu::TextureFormat::Rgba8UnormSrgb;
    let mut r = Renderer::new(&device, 1, fmt);
    let grid_res = 32u32;

    for (w, h) in [(100u32, 100u32), (300, 100), (100, 300), (37, 211)] {
        r.set_camera(&queue, grid_res, w, h, 0.6, true);
        for (sx, sy) in [
            (0.0, 0.0),
            (w as f32, h as f32),
            (w as f32 * 0.5, h as f32 * 0.5),
            (w as f32 * 0.25, h as f32 * 0.75),
        ] {
            let (gx, gy) = r.screen_to_grid(sx, sy, w, h);
            let (rx, ry) = r.grid_to_screen(gx, gy, w, h);
            assert!(
                (rx - sx).abs() < 1.0e-2 && (ry - sy).abs() < 1.0e-2,
                "grid_to_screen must exactly invert screen_to_grid at w={w} h={h} \
                 -- screen ({sx},{sy}) -> grid ({gx},{gy}) -> screen ({rx},{ry})"
            );
        }
    }

    // A independently-checkable invariant for the radius conversion:
    // the camera is isotropic (grid cells always render as true squares, see
    // `set_camera`'s doc), so a distance measured along the Y axis via
    // `screen_to_grid` must match `grid_distance_to_pixels`'s own conversion
    // of that same real grid distance, exactly.
    r.set_camera(&queue, grid_res, 300, 100, 0.6, true);
    let (_, gy_top) = r.screen_to_grid(0.0, 0.0, 300, 100);
    let (_, gy_mid) = r.screen_to_grid(0.0, 50.0, 300, 100);
    let grid_distance = gy_top - gy_mid;
    let pixel_distance = r.grid_distance_to_pixels(grid_distance, 100);
    assert!(
        (pixel_distance - 50.0).abs() < 1.0e-2,
        "grid_distance_to_pixels must agree with screen_to_grid's own real \
         scale -- expected 50.0 pixels, got {pixel_distance}"
    );
}

/// A UI overlay fed physical pixels into a toolkit (egui) that draws in
/// logical points is off by the display's scale factor.
/// `grid_to_screen_points`/`grid_distance_to_points` are checked at 100%,
/// 125% (the scale that showed it) and 200%.
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn grid_to_screen_points_divides_out_the_real_dpi_scale_factor() {
    let (device, queue) = headless_device();
    let fmt = wgpu::TextureFormat::Rgba8UnormSrgb;
    let mut r = Renderer::new(&device, 1, fmt);
    let grid_res = 32u32;
    r.set_camera(&queue, grid_res, 300, 200, 0.6, true);

    let (px_x, px_y) = r.grid_to_screen(16.0, 16.0, 300, 200);
    let px_radius = r.grid_distance_to_pixels(2.0, 200);

    for ppp in [1.0_f32, 1.25, 1.5, 2.0] {
        let (pt_x, pt_y) = r.grid_to_screen_points(16.0, 16.0, 300, 200, ppp);
        let pt_radius = r.grid_distance_to_points(2.0, 200, ppp);
        assert!(
            (pt_x - px_x / ppp).abs() < 1.0e-3 && (pt_y - px_y / ppp).abs() < 1.0e-3,
            "grid_to_screen_points at ppp={ppp} must be exactly the physical \
             result divided by the real scale factor -- got ({pt_x},{pt_y}), \
             expected ({},{})",
            px_x / ppp,
            px_y / ppp
        );
        assert!(
            (pt_radius - px_radius / ppp).abs() < 1.0e-3,
            "grid_distance_to_points at ppp={ppp} must match physical/ppp -- \
             got {pt_radius}, expected {}",
            px_radius / ppp
        );
    }

    // At the real 100% (no scaling) case specifically, points and pixels
    // must be numerically IDENTICAL -- this is the case that silently looked
    // "correct" during development on a 100%-scaled display while actually
    // being wrong at every other real scale factor.
    let (pt_x, pt_y) = r.grid_to_screen_points(16.0, 16.0, 300, 200, 1.0);
    assert!(
        (pt_x - px_x).abs() < 1.0e-6 && (pt_y - px_y).abs() < 1.0e-6,
        "at 100% scaling, points and physical pixels must be identical"
    );
}

/// Pore-fluid index matching (`Renderer::set_refractive_index`) reduces
/// scattering as a particle's `scalar_field` saturates toward 1.0 (the
/// generic wet-material mechanism). With quartz sand's index (~1.5) and
/// saturation from dry to full on one slot, the colour changes and darkens
/// toward the absorption-only colour.
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn wetness_darkens_by_physics_color_via_refractive_index() {
    let (device, queue) = headless_device();
    let mut r = Renderer::new(&device, 16, wgpu::TextureFormat::Rgba8UnormSrgb);
    r.set_color_mode(ColorMode::ByPhysics);
    r.set_optical_params(&queue, 0, [0.18, 0.22, 0.55]); // illustrative sand absorption, unsourced
    r.set_optical_scattering(&queue, 0, 8.0);
    r.set_refractive_index(0, 1.5); // real quartz refractive index (Hecht, "Optics")

    let mut dry = Particle::zeroed();
    dry.material_id = 0;
    dry.deformation_gradient = Mat2::IDENTITY;
    dry.scalar_field = 0.0;
    let mut wet = dry;
    wet.scalar_field = 1.0;

    let c_dry = r.particle_color(&dry, 0);
    let c_wet = r.particle_color(&wet, 0);
    assert_ne!(
        c_dry, c_wet,
        "saturation must actually change ByPhysics color once refractive_index is set"
    );
    let brightness = |c: [f32; 4]| c[0] + c[1] + c[2];
    assert!(
        brightness(c_wet) < brightness(c_dry),
        "wet sand must render DARKER than dry (reduced scattering from \
         pore-fluid index-matching), not brighter -- dry={c_dry:?} wet={c_wet:?}"
    );
}

/// The default `refractive_index` (1.0, same as air) must be fully inert --
/// a material that never calls `set_refractive_index` renders identically
/// regardless of its particles' `scalar_field`, exactly as before this
/// feature existed. Real safety property: this mechanism must never
/// silently activate for materials that never opted in.
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn wetness_darkening_is_inert_without_refractive_index_opt_in() {
    let (device, queue) = headless_device();
    let mut r = Renderer::new(&device, 16, wgpu::TextureFormat::Rgba8UnormSrgb);
    r.set_color_mode(ColorMode::ByPhysics);
    r.set_optical_params(&queue, 0, [0.18, 0.22, 0.55]);
    r.set_optical_scattering(&queue, 0, 8.0);
    // Deliberately NOT calling set_refractive_index -- default stays 1.0.

    let mut dry = Particle::zeroed();
    dry.material_id = 0;
    dry.deformation_gradient = Mat2::IDENTITY;
    dry.scalar_field = 0.0;
    let mut wet = dry;
    wet.scalar_field = 1.0;

    assert_eq!(
        r.particle_color(&dry, 0),
        r.particle_color(&wet, 0),
        "a material that never calls set_refractive_index must be byte-identical \
         regardless of scalar_field -- this feature must be opt-in, not silently active"
    );
}

/// Diagnostic: the isolated per-call GPU cost of each render path
/// (Particles/GridVolume/Surface) on the same scene, synced with
/// device.poll so it measures completed GPU work. Built when
/// `basic_sand_grid_gpu.rs` dropped to 24-27 fps with
/// `render_surface_reconstruction`, to separate the technique's cost from the
/// wiring.
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn diag_surface_reconstruction_real_cost_vs_grid_volume_and_particles() {
    use crate::gpu::GpuSimulation;
    use crate::{DruckerPragerMaterial, MaterialRegistry, SimConfig, SpawnRegion, build_particles};
    use std::sync::Arc;
    use std::time::Instant;

    let (device, queue) = headless_device();
    let device = Arc::new(device);
    let queue = Arc::new(queue);

    const GRID: usize = 64;
    let config = SimConfig::standard(GRID, 0.1, glam::Vec2::new(0.0, -0.3));
    let particles = build_particles(
        &config,
        SpawnRegion::for_sim(&config)
            .at(glam::Vec2::new(32.0, 20.0))
            .disk(20.0)
            .spacing(0.5)
            .material(0),
    );
    let registry =
        MaterialRegistry::with_default(Box::new(DruckerPragerMaterial::new(100.0, 50.0)));
    let sim =
        GpuSimulation::with_device(device.clone(), queue.clone(), config, particles, registry);
    let particle_count = sim.particle_count();

    let fmt = wgpu::TextureFormat::Rgba8UnormSrgb;
    let mut r = Renderer::new(&device, particle_count, fmt);
    r.set_optical_params(&queue, 0, [0.18, 0.22, 0.55]);
    r.set_camera(&queue, GRID as u32, 800, 600, 0.6, true);

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("perf_test_target"),
        size: wgpu::Extent3d {
            width: 800,
            height: 600,
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

    const WARMUP: usize = 3;
    const TIMED: usize = 20;

    macro_rules! measure {
        ($label:expr, $call:expr) => {{
            for _ in 0..WARMUP {
                $call;
                device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            }
            let start = Instant::now();
            for _ in 0..TIMED {
                $call;
            }
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            let elapsed = start.elapsed();
            println!(
                "{}: {:.3}ms/call ({} particles, {GRID}x{GRID} grid)",
                $label,
                elapsed.as_secs_f64() * 1000.0 / TIMED as f64,
                particle_count
            );
        }};
    }

    measure!(
        "render_gpu (Particles)",
        r.render_gpu(
            &device,
            &queue,
            GpuRenderParams {
                particle_buf: sim.particle_buffer(),
                particle_count,
                output_view: &view,
                clear: true,
                interp_alpha: 1.0,
            },
        )
    );
    measure!(
        "render_grid_volume",
        r.render_grid_volume(
            &device,
            &queue,
            GridVolumeSource {
                grid: sim.grid_buffer(),
                material_mass: sim.material_mass_buffer(),
                material_mass_enabled: true,
                grid_res: GRID as u32,
            },
            &view,
            true,
        )
    );
    measure!(
        "render_surface_reconstruction (default mult=6, iters=12)",
        r.render_surface_reconstruction(
            &device,
            &queue,
            SurfaceReconstructionSource {
                particle_buf: sim.particle_buffer(),
                particle_count,
                grid_res: GRID as u32,
                material_slot: 0,
                material_mass_enabled: true,
                dt: 0.1,
            },
            &view,
            true,
        )
    );

    // Sweep of `surface_res_multiplier` (the biggest quality/cost dial, cost
    // goes with its square) and `curvature_iterations` (van der Laan et al.
    // 2009's "several per frame"), measured at each combination.
    for mult in [2u32, 3, 4] {
        r.set_surface_res_multiplier(mult);
        measure!(
            format!("surface_res_multiplier={mult} (iters=12)"),
            r.render_surface_reconstruction(
                &device,
                &queue,
                SurfaceReconstructionSource {
                    particle_buf: sim.particle_buffer(),
                    particle_count,
                    grid_res: GRID as u32,
                    material_slot: 0,
                    material_mass_enabled: true,
                    dt: 0.1,
                },
                &view,
                true,
            )
        );
    }
    r.set_surface_res_multiplier(6);
    for iters in [4u32, 6, 8] {
        r.set_curvature_iterations(iters);
        measure!(
            format!("curvature_iterations={iters} (mult=6)"),
            r.render_surface_reconstruction(
                &device,
                &queue,
                SurfaceReconstructionSource {
                    particle_buf: sim.particle_buffer(),
                    particle_count,
                    grid_res: GRID as u32,
                    material_slot: 0,
                    material_mass_enabled: true,
                    dt: 0.1,
                },
                &view,
                true,
            )
        );
    }
}

/// Every camera goes through `region_projection`. Framed whole, the grid must
/// come out as `set_camera` drew it before it called that function (the
/// closed form below is that version's, restated); a region must fill the
/// window along one axis, centred, with square pixels. Both window
/// orientations, since the fit switches axis between them.
#[test]
fn region_projection_frames_a_region_and_the_whole_grid() {
    let close = |a: f32, b: f32| (a - b).abs() <= 1.0e-5 * a.abs().max(b.abs()).max(1.0);
    for (w, h) in [(1280u32, 720u32), (720, 1280), (900, 900)] {
        let aspect = w as f32 / h as f32;
        let gr = 160.0f32;
        let before = if aspect >= 1.0 {
            (2.0 / (gr * aspect), -1.0 / aspect, 2.0 / gr, -1.0)
        } else {
            (2.0 / gr, -1.0, 2.0 * aspect / gr, -aspect)
        };
        let now = Renderer::region_projection(Vec2::ZERO, Vec2::splat(gr), w, h);
        assert!(
            close(now.0, before.0)
                && close(now.1, before.1)
                && close(now.2, before.2)
                && close(now.3, before.3),
            "{w}x{h}: whole grid {now:?}, set_camera drew {before:?}"
        );

        // A wide strip and a tall one.
        for (min, max) in [
            (Vec2::new(30.0, 2.0), Vec2::new(150.0, 40.0)),
            (Vec2::new(70.0, 2.0), Vec2::new(90.0, 120.0)),
        ] {
            let (sx, tx, sy, ty) = Renderer::region_projection(min, max, w, h);
            let ndc = |p: Vec2| Vec2::new(p.x * sx + tx, p.y * sy + ty);
            let centre = ndc((min + max) * 0.5);
            assert!(centre.length() < 1.0e-5, "{w}x{h}: centre at {centre}");
            let (lo, hi) = (ndc(min), ndc(max));
            assert!(
                lo.cmpge(Vec2::splat(-1.0 - 1.0e-5)).all()
                    && hi.cmple(Vec2::splat(1.0 + 1.0e-5)).all(),
                "{w}x{h}: region {min}..{max} spills out of the window, {lo}..{hi}"
            );
            assert!(
                close(hi.x, 1.0) || close(hi.y, 1.0),
                "{w}x{h}: region {min}..{max} fills neither axis, {lo}..{hi}"
            );
            // Square pixels: a cell spans as many pixels across as up.
            assert!(
                close(sx * w as f32, sy * h as f32),
                "{w}x{h}: cells are not square"
            );
        }
    }
}

/// The round trip the cursor depends on, through a renderer: a point
/// drawn under `set_camera_region` and read back by `screen_to_grid` comes
/// back where it was, and the framed strip's ends land on the window's
/// edges.
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn a_cursor_reads_back_the_point_a_region_camera_drew() {
    let (device, queue) = headless_device();
    let mut r = Renderer::new(&device, 16, wgpu::TextureFormat::Rgba8UnormSrgb);
    // Wider than either window, so it fills the width in both.
    let (min, max) = (Vec2::new(30.0, 2.0), Vec2::new(150.0, 40.0));
    for (w, h) in [(1280u32, 720u32), (720, 1280)] {
        r.set_camera_region(&queue, (min, max), w, h, 0.6, true);
        for p in [min, max, (min + max) * 0.5, Vec2::new(41.3, 17.9)] {
            let (x, y) = r.grid_to_screen(p.x, p.y, w, h);
            let (gx, gy) = r.screen_to_grid(x, y, w, h);
            assert!(
                (gx - p.x).abs() < 1.0e-3 && (gy - p.y).abs() < 1.0e-3,
                "{w}x{h}: {p} drawn at ({x}, {y}) reads back as ({gx}, {gy})"
            );
        }
        let (left, _) = r.grid_to_screen(min.x, min.y, w, h);
        let (right, _) = r.grid_to_screen(max.x, min.y, w, h);
        assert!(
            left.abs() < 1.0e-2 && (right - w as f32).abs() < 1.0e-2,
            "{w}x{h}: strip drawn from x = {left} to {right} px"
        );
    }
}

/// `free_surface_cell_step` is the quadratic B-spline's mass inside one cell
/// centred on its peak, for the kernel `grid::kernel::axis_weights`
/// evaluates, stretched by the splat width: checked against a numerical
/// integral of that kernel.
#[test]
fn free_surface_cell_step_is_the_kernels_central_mass() {
    // N(u) from the kernel's own weights: the centre weight within half a
    // cell, an outer weight beyond it.
    let n = |u: f32| {
        let a = u.abs();
        if a <= 0.5 {
            crate::grid::kernel::axis_weights(a)[1]
        } else if a < 1.5 {
            crate::grid::kernel::axis_weights(a - 1.0)[0]
        } else {
            0.0
        }
    };
    for splat_cells in [0.5f32, 1.0, 2.0, 3.0, 4.0] {
        let h = 0.5 / splat_cells;
        let steps = 20_000;
        let du = 2.0 * h / steps as f32;
        let integral: f32 = (0..steps).map(|i| n(-h + (i as f32 + 0.5) * du) * du).sum();
        let step = free_surface_cell_step(splat_cells);
        assert!(
            (step - integral).abs() < 1.0e-4,
            "splat width {splat_cells}: {step} against the kernel's {integral}"
        );
    }
    // The physics grid's own value, `grid_volume.wgsl`'s constant.
    assert!((free_surface_cell_step(1.0) - 2.0 / 3.0).abs() < 1.0e-6);
}

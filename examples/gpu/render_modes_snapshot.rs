extern crate emerge_engine as emerge;

/// Offscreen render harness: runs `basic_fluids_gpu.rs`'s own dam-break scene
/// headlessly and writes one image per render mode, so the three modes can be
/// compared without a window (and without screenshotting anything).
///
/// Writes raw BGRA bytes plus a one-line `.txt` header next to them; convert
/// with the companion PowerShell snippet, or read them with any raw-image
/// viewer at the stated size.
///
///   cargo run --release --example render_modes_snapshot --features "gpu render"
///
/// Environment knobs:
///   FRAMES=60          how many simulation frames before the snapshot
///   OUT_DIR=.          where to write `<mode>.bgra` / `<mode>.txt`
///   SIZE=512           square output resolution
use emerge::gpu::GpuSimulation;
use emerge::render::{
    ColorMode, GpuRenderParams, GridVolumeSource, Renderer, SurfaceReconstructionSource,
};
use emerge::{
    GpuFieldEntry, MaterialRegistry, NewtonianFluidMaterial, SimConfig, SpawnRegion,
    build_particles,
};
use glam::{IVec2, Vec2};
use pollster::block_on;
use wgpu::InstanceDescriptor;

const GRID: usize = 64;
const MAT_WATER: u32 = 0;
const SPACING: f32 = 0.5;
const WATER_RHO_GRID: f32 = 0.1;
const COLUMN_HEIGHT_CELLS: f32 = 52.0;

fn main() {
    let size: u32 = env_or("SIZE", 512.0) as u32;
    let frames: u64 = env_or("FRAMES", 60.0) as u64;
    let out_dir = std::env::var("OUT_DIR").unwrap_or_else(|_| ".".into());

    let instance = wgpu::Instance::new(&InstanceDescriptor::default());
    let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .expect("adapter");
    let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_limits: adapter.limits(),
        ..Default::default()
    }))
    .expect("device");
    let device = std::sync::Arc::new(device);
    let queue = std::sync::Arc::new(queue);

    // Same scene and material as the demo's dam break.
    let dt = 0.1;
    let config = SimConfig {
        min_dt: 1.0e-4,
        max_substeps_per_step: 1000,
        cfl_include_affine_speed: false,
        material_cfl_coefficient: 0.4,
        gravity: Vec2::new(0.0, -981.0 * 0.003),
        ..SimConfig::earth(GRID, 0.01, dt)
    };
    let particles = build_particles(
        &config,
        SpawnRegion {
            spacing: SPACING,
            box_size: IVec2::new(14, 52),
            box_center: Vec2::new(20.0, 30.0),
            material_id: MAT_WATER,
            precompute_initial_volumes: true,
            mass_override: Some(WATER_RHO_GRID * SPACING * SPACING),
            ..SpawnRegion::for_sim(&config)
        },
    );
    // TEMP_K: every particle's temperature. 0 (the default) leaves thermal
    // emission on its early-out path; a hot value forces the full blackbody
    // evaluation, which is what a render-cost A/B needs to compare.
    let temperature_k = env_or("TEMP_K", 0.0);
    let mut particles = particles;
    if temperature_k > 0.0 {
        for p in particles.iter_mut() {
            p.temperature = temperature_k;
        }
    }

    let v_max_grid = (2.0 * config.gravity.length() * COLUMN_HEIGHT_CELLS).sqrt();
    let c_ref_m_s = 10.0 * v_max_grid * config.dx_meters;
    let visc = config.visc_from_si_physical(1.0e-3, 1000.0);
    let mut water = NewtonianFluidMaterial::new(
        WATER_RHO_GRID,
        visc,
        1000.0 * c_ref_m_s * c_ref_m_s / 3.0,
        3.0,
    );
    water.bulk_viscosity = 3.0 * visc;
    water.pressure_floor = config.stress_from_si_physical(-100_000.0, 1000.0);

    let registry = MaterialRegistry::with_default(Box::new(water));
    let mut sim =
        GpuSimulation::with_device(device.clone(), queue.clone(), config, particles, registry);
    sim.add_force_field_gpu(GpuFieldEntry::linear_drag(Vec2::ZERO, 0.1, 1 << MAT_WATER));
    sim.attach_grid_material_render_gpu();

    let fmt = wgpu::TextureFormat::Rgba8UnormSrgb;
    let mut renderer = Renderer::new(&device, sim.particle_count(), fmt);
    renderer.set_camera(&queue, GRID as u32, size, size, 0.6, true);
    renderer.set_color_mode(ColorMode::ByMaterial);
    renderer.set_grid_reference_cell_mass(WATER_RHO_GRID);
    renderer.set_surface_res_multiplier(env_or("SURF_MULT", 4.0) as u32);
    if let Ok(v) = std::env::var("CURV_ITERS")
        && let Ok(n) = v.parse::<u32>()
    {
        renderer.set_curvature_iterations(n);
    }
    if std::env::var("SPLAT_FROM_SPACING").is_ok() {
        renderer.set_particle_spacing_cells(SPACING);
    } else if let Ok(v) = std::env::var("SPLAT_CELLS")
        && let Ok(w) = v.parse::<f32>()
    {
        renderer.set_splat_width_cells(w);
    }
    // PHYS=1 switches from the legacy dimensionless optical numbers to real
    // measured ones: pure water's own absorption spectrum (Pope & Fry 1997)
    // band-averaged to sRGB, read through a declared SI contract. SLAB_M is
    // the out-of-plane thickness this 2-D slice stands for, which is the only
    // honest way to make water look blue -- the coefficients are measured and
    // are not ours to inflate.
    if std::env::var("PHYS").is_ok() {
        let slab_m = env_or("SLAB_M", 0.3);
        // Radiance the scene declares, W/(m^2 sr). Defaults put an equally
        // bright light and backdrop everywhere, which is the simplest case
        // and also the one where reflection and scattering cancel out of
        // view. A real scene lights from one side against a darker
        // backdrop, and that is what makes them visible.
        let incident = env_or("INCIDENT", 1.0);
        let background = env_or("BACKGROUND", 1.0);
        let white = env_or("WHITE", 1.0);
        let mut water = emerge::matter::materials::optical::pure_water();
        // SCATTER_M_INV overrides pure water's own molecular scattering.
        // Real water bodies scatter far more than pure water because of
        // suspended particles -- that is a property of the mixture, not of
        // water, so it is stated by the scene rather than baked into the
        // substance.
        if let Ok(v) = std::env::var("SCATTER_M_INV")
            && let Ok(sigma_s) = v.parse::<f32>()
        {
            water.reduced_scattering_m_inv = sigma_s;
        }
        renderer.set_optical_coefficients_si(&queue, MAT_WATER as usize, water);
        renderer.set_physical_render_contract(
            &queue,
            emerge::render::PhysicalRenderContract::new(
                emerge::render::PhysicalRenderContractParams {
                    dx_meters: config.dx_meters,
                    view_thickness_meters: slab_m,
                    incident_radiance_w_m2_sr: [incident; 3],
                    background_radiance_w_m2_sr: [background; 3],
                    display_white_radiance_w_m2_sr: [white; 3],
                    camera_direction: glam::Vec3::new(0.0, 0.0, -1.0),
                    light_direction: glam::Vec3::new(0.0, 1.0, -1.0),
                },
            )
            .expect("physical render contract"),
        );
        println!("physical optics: water {:?}", pure_water_sigma());
    } else {
        renderer.set_optical_params(&queue, MAT_WATER as usize, [0.85, 0.25, 0.07]);
        renderer.set_optical_scattering(&queue, MAT_WATER as usize, 0.03);
    }
    renderer.set_specular_r0(&queue, MAT_WATER as usize, 0.02);

    for _ in 0..frames {
        sim.step_frame();
    }
    sim.sync_particles_blocking();
    println!(
        "stepped {frames} frames, {} particles",
        sim.particle_count()
    );

    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("snapshot_target"),
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
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());

    let only = std::env::var("MODE").unwrap_or_default();
    let tag = std::env::var("TAG").unwrap_or_default();
    for mode in ["particles", "grid_volume", "surface"] {
        if !only.is_empty() && only != mode {
            continue;
        }
        render_mode(mode, &device, &queue, &mut renderer, &sim, &view, dt);
        // RENDER_REPEAT=N: time N more renders of this mode (relative cost only --
        // any other GPU load on the machine penalises every variant equally).
        let repeat = env_or("RENDER_REPEAT", 0.0) as u32;
        if repeat > 0 {
            device.poll(wgpu::PollType::wait_indefinitely()).ok();
            let t0 = std::time::Instant::now();
            for _ in 0..repeat {
                render_mode(mode, &device, &queue, &mut renderer, &sim, &view, dt);
            }
            device.poll(wgpu::PollType::wait_indefinitely()).ok();
            println!(
                "{mode}: {:.2}ms per render ({repeat} renders)",
                t0.elapsed().as_secs_f64() * 1e3 / f64::from(repeat)
            );
        }
        let pixels = readback(&device, &queue, &target, size);
        let path = format!("{out_dir}/{mode}{tag}.bgra");
        std::fs::write(&path, &pixels).expect("write image bytes");
        std::fs::write(
            format!("{out_dir}/{mode}.txt"),
            format!("{size} {size} RGBA8\n"),
        )
        .expect("write header");
        let lit = pixels
            .as_chunks::<4>()
            .0
            .iter()
            .filter(|p| p[0] as u32 + p[1] as u32 + p[2] as u32 > 120)
            .count();
        println!("{mode}: wrote {path}, {lit} pixels above background");
    }
}

#[allow(clippy::too_many_arguments)]
fn render_mode(
    mode: &str,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    renderer: &mut Renderer,
    sim: &GpuSimulation,
    view: &wgpu::TextureView,
    dt: f32,
) {
    match mode {
        "particles" => renderer.render_gpu(
            device,
            queue,
            GpuRenderParams {
                particle_buf: sim.particle_buffer(),
                particle_count: sim.particle_count(),
                output_view: view,
                clear: true,
                interp_alpha: 1.0,
            },
        ),
        "grid_volume" => renderer.render_grid_volume(
            device,
            queue,
            GridVolumeSource {
                grid: sim.grid_buffer(),
                material_mass: sim.material_mass_buffer(),
                material_mass_enabled: true,
            },
            view,
            true,
        ),
        _ => renderer.render_surface_reconstruction(
            device,
            queue,
            SurfaceReconstructionSource {
                particle_buf: sim.particle_buffer(),
                particle_count: sim.particle_count(),
                grid_res: GRID as u32,
                material_slot: MAT_WATER,
                material_mass_enabled: false,
                dt,
            },
            view,
            true,
        ),
    }
}

fn pure_water_sigma() -> [f32; 3] {
    emerge::matter::materials::optical::pure_water().absorption_m_inv
}

fn env_or(key: &str, default: f32) -> f32 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn readback(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    size: u32,
) -> Vec<u8> {
    let unpadded = size * 4;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded = unpadded.div_ceil(align) * align;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("snapshot_staging"),
        size: (padded * size) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("snapshot_readback"),
    });
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
    queue.submit(Some(enc.finish()));
    let slice = staging.slice(..);
    let flag = std::sync::Arc::new(std::sync::Mutex::new(None));
    let flag2 = flag.clone();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        *flag2.lock().unwrap() = Some(r);
    });
    device.poll(wgpu::PollType::wait_indefinitely()).ok();
    flag.lock().unwrap().take().expect("map").expect("mapped");
    let data = slice.get_mapped_range();
    let mut out = Vec::with_capacity((unpadded * size) as usize);
    for row in 0..size {
        let start = (row * padded) as usize;
        out.extend_from_slice(&data[start..start + unpadded as usize]);
    }
    drop(data);
    staging.unmap();
    out
}

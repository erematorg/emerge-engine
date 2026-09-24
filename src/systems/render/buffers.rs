//! GPU buffer allocation for `Renderer` -- split out of `Renderer::new`
//! (was ~390 of `mod.rs`'s ~1318 lines, one long flat sequence of
//! independent `device.create_buffer` calls with no interdependency between
//! them). Pure resource creation, same shape as the pipeline builders
//! already split into `pipelines.rs` -- extracting it changes nothing about
//! when/how these buffers are built or which fields `Renderer` itself has.
//!
//! Two real, mechanical patterns cover 39 of the 43 buffers here --
//! `placeholder_buffer` (4-byte lazy-growth storage placeholders, STORAGE |
//! COPY_DST [| COPY_SRC]) and `uniform_buffer` (a UNIFORM | COPY_DST buffer
//! sized to exactly one `T`) -- factored out so the per-buffer doc comments
//! (each real, explaining WHY that specific buffer needs COPY_SRC or not)
//! stay attached to their own call site instead of being duplicated 20+
//! times over inside an identical `BufferDescriptor` literal.

use std::mem;

use wgpu::util::DeviceExt;

use super::gpu_types::{
    BandHysteresisParams, CameraParams, GridVisibilityParams, GridVolumeParams, InstanceData,
    LightDiffuseParams, OpticalTable, RenderConfig, SurfaceParams, SurfaceRenderParams,
    VisibilityParams, WaveStepParams,
};

/// 4-byte lazy-growth storage placeholder -- real, standard convention used
/// throughout this file: allocate minimally at construction, `ensure_*_capacity`
/// grows the real buffer the first time its true size (e.g. `grid_res`) is
/// known. `copy_src` is per-buffer real intent (readback/diagnostic tools
/// need to copy FROM some of these, not others) -- kept as a caller-supplied
/// flag, not inferred, so each call site's own doc comment still explains it.
fn placeholder_buffer(device: &wgpu::Device, label: &str, copy_src: bool) -> wgpu::Buffer {
    let mut usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
    if copy_src {
        usage |= wgpu::BufferUsages::COPY_SRC;
    }
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: 4,
        usage,
        mapped_at_creation: false,
    })
}

/// UNIFORM | COPY_DST buffer sized to exactly one `T` -- the other real,
/// repeated pattern in this file (camera/config/optics/surface/wave/
/// visibility/band-hysteresis params all share this exact shape).
fn uniform_buffer<T>(device: &wgpu::Device, label: &str) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: mem::size_of::<T>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// Every GPU buffer `Renderer` owns, allocated once at construction.
/// Destructured back into `Renderer::new`'s own local bindings immediately
/// after construction -- `Renderer`'s own field layout is untouched by this
/// split.
pub(super) struct RenderBuffers {
    pub instance_buffer: wgpu::Buffer,
    pub storage_instances: wgpu::Buffer,
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub camera_buffer: wgpu::Buffer,
    pub render_config_buf: wgpu::Buffer,
    pub optical_table_buf: wgpu::Buffer,
    pub grid_volume_params_buf: wgpu::Buffer,
    pub grid_visibility_buf: wgpu::Buffer,
    pub grid_visibility_params_buf: wgpu::Buffer,
    pub surface_atomic_buf: wgpu::Buffer,
    pub surface_temp_atomic_buf: wgpu::Buffer,
    pub surface_temp_float_buf: wgpu::Buffer,
    pub pre_total_atomic_buf: wgpu::Buffer,
    pub post_total_atomic_buf: wgpu::Buffer,
    pub surface_temp_b_buf: wgpu::Buffer,
    pub surface_a_buf: wgpu::Buffer,
    pub surface_b_buf: wgpu::Buffer,
    pub surface_params_buf: wgpu::Buffer,
    pub surface_render_params_buf: wgpu::Buffer,
    pub surface_material_mass_buf: wgpu::Buffer,
    pub surface_moments_buf: wgpu::Buffer,
    pub phase_b_atomic_buf: wgpu::Buffer,
    pub phase_b_temp_atomic_buf: wgpu::Buffer,
    pub phase_b_temp_float_buf: wgpu::Buffer,
    pub phase_b_pre_total_atomic_buf: wgpu::Buffer,
    pub phase_b_post_total_atomic_buf: wgpu::Buffer,
    pub phase_b_a_buf: wgpu::Buffer,
    pub phase_b_b_buf: wgpu::Buffer,
    pub phase_b_raw_splat_history_buf: wgpu::Buffer,
    pub phase_b_params_buf: wgpu::Buffer,
    pub render_params_b_buf: wgpu::Buffer,
    pub phase_b_wave_bufs: [wgpu::Buffer; 3],
    pub phase_b_wave_density_prev_buf: wgpu::Buffer,
    pub phase_b_visibility_buf: wgpu::Buffer,
    pub phase_b_band_state_buf: wgpu::Buffer,
    pub wave_bufs: [wgpu::Buffer; 3],
    pub wave_params_buf: wgpu::Buffer,
    pub wave_density_prev_buf: wgpu::Buffer,
    pub visibility_buf: wgpu::Buffer,
    pub visibility_params_buf: wgpu::Buffer,
    pub band_state_buf: wgpu::Buffer,
    pub band_hysteresis_params_buf: wgpu::Buffer,
    pub raw_splat_history_buf: wgpu::Buffer,
    /// Real, persistent light-fluence diffusion field (`curvature_flow.wgsl`'s
    /// "Pass 1e") -- two buffers, not three: unlike the wave field's real
    /// second-order-in-time leapfrog (needs current+previous to read, next to
    /// write), diffusion is first-order-in-time -- one "current" to read, one
    /// "next" to write, swapped each frame (see `light_frame_index`'s own doc).
    pub light_phi_bufs: [wgpu::Buffer; 2],
    pub light_diffuse_params_buf: wgpu::Buffer,
}

impl RenderBuffers {
    pub(super) fn new(device: &wgpu::Device, cap: usize) -> Self {
        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("render_instances"),
            size: (cap * mem::size_of::<InstanceData>()) as u64,
            // VERTEX for draw; COPY_DST for both the CPU fill path and the GPU compute copy.
            // COPY_SRC (2026-08-15): kept permanently -- lets tests read back
            // this buffer directly to check what actually landed here, not
            // just what was written to it. Used by the `#[ignore]`d
            // render_gpu/render_cpu pixel tests in tests.rs (see
            // basic_fluids_gpu_blank_render_unconfirmed memory) to prove
            // storage_instances -> instance_buffer copies correctly even
            // when the final drawn pixels don't show it.
            usage: wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // GPU compute write target. Kept distinct from the vertex buffer: wgpu treats a
        // read_write storage buffer as an exclusive usage, so sharing one buffer for both
        // compute-write and vertex-read trips its usage tracker. Copied into instance_buffer.
        let storage_instances = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("render_instances_storage"),
            size: (cap * mem::size_of::<InstanceData>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("render_quad_verts"),
            contents: bytemuck::cast_slice::<[f32; 2], u8>(&[
                [-0.5f32, -0.5],
                [0.5, -0.5],
                [0.5, 0.5],
                [-0.5, 0.5],
            ]),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("render_quad_idx"),
            contents: bytemuck::cast_slice::<u16, u8>(&[0u16, 1, 2, 0, 2, 3]),
            usage: wgpu::BufferUsages::INDEX,
        });

        let camera_buffer = uniform_buffer::<CameraParams>(device, "render_camera");
        let render_config_buf = uniform_buffer::<RenderConfig>(device, "render_config");
        let optical_table_buf = uniform_buffer::<OpticalTable>(device, "render_optics");
        let grid_volume_params_buf =
            uniform_buffer::<GridVolumeParams>(device, "grid_volume_params");

        // Real hysteresis visibility state for the grid-native path --
        // same real, disclosed all-zero starting bias as `visibility_buf`
        // above (every cell starts "not visible" until it genuinely earns
        // visibility on its own first real frame).
        let grid_visibility_buf =
            placeholder_buffer(device, "grid_visibility_state", /* copy_src */ true);
        let grid_visibility_params_buf =
            uniform_buffer::<GridVisibilityParams>(device, "grid_visibility_params");

        // Curvature-flow surface buffers -- allocated at a minimal 1-cell
        // placeholder size; `ensure_surface_capacity` (called from
        // `render_surface_reconstruction`, the only place `grid_res` is
        // actually known) grows all three together the first time it's
        // needed, same lazy-growth pattern `ensure_capacity` already uses
        // for the particle instance buffers.
        let surface_atomic_buf = placeholder_buffer(device, "surface_atomic", false);
        // Real mass-weighted temperature pair -- same minimal-placeholder-
        // then-grow convention as `surface_atomic_buf` above.
        let surface_temp_atomic_buf = placeholder_buffer(device, "surface_temp_atomic", false);
        // COPY_SRC: `fs_main`'s own final temperature source, same real
        // "readback/diagnostic tools need to copy FROM it" reason
        // `surface_a_buf` is COPY_SRC (see that field's own doc).
        let surface_temp_float_buf = placeholder_buffer(device, "surface_temp_float", true);
        // Real volume-preserving-correction totals. Always exactly 4 bytes
        // (one atomic i32), never grown by `ensure_surface_capacity` -- a
        // global scalar, not a per-cell field. COPY_SRC: diagnostic/test
        // readback, same real reason `surface_a_buf` has it.
        let pre_total_atomic_buf = placeholder_buffer(device, "pre_total_atomic", true);
        let post_total_atomic_buf = placeholder_buffer(device, "post_total_atomic", true);
        // Ping-pong partner for the real thermal-diffusion PDE -- see
        // `temp_avg_pipeline`'s own doc.
        let surface_temp_b_buf = placeholder_buffer(device, "surface_temp_b", false);
        // COPY_SRC: `surface_a` is where `CURVATURE_ITERATIONS` (even)
        // always settles the final result -- readback/diagnostic tools
        // need to copy FROM it, not just the render pass reading it.
        let surface_a_buf = placeholder_buffer(device, "surface_a", true);
        let surface_b_buf = placeholder_buffer(device, "surface_b", false);
        let surface_params_buf = uniform_buffer::<SurfaceParams>(device, "surface_params");
        let surface_render_params_buf =
            uniform_buffer::<SurfaceRenderParams>(device, "surface_render_params");
        // N-material extension -- minimal placeholder, same lazy-growth
        // convention as the surface buffers above, but grown independently
        // (only on opt-in) by `ensure_surface_material_mass_capacity`, not
        // by `ensure_surface_capacity`. COPY_SRC: diagnostic/test readback,
        // same real reason `surface_a_buf` has it.
        let surface_material_mass_buf = placeholder_buffer(device, "surface_material_mass", true);
        // Yu & Turk neighbourhood moments -- sized off `grid_res` (not
        // `surface_res`) and grown by `ensure_surface_capacity` alongside the
        // surface buffers. Same minimal-placeholder-then-grow convention.
        let surface_moments_buf = placeholder_buffer(device, "surface_moments", false);

        // Two-phase extension's own phase-B buffers -- same minimal-
        // placeholder-then-grow convention as phase A's own buffers above.
        let phase_b_atomic_buf = placeholder_buffer(device, "phase_b_atomic", false);
        // Phase B's own temperature pair -- see `surface_temp_atomic_buf`'s own doc.
        let phase_b_temp_atomic_buf = placeholder_buffer(device, "phase_b_temp_atomic", false);
        let phase_b_temp_float_buf = placeholder_buffer(device, "phase_b_temp_float", false);
        // Phase B's own volume-preserving-correction totals -- see
        // `pre_total_atomic_buf`'s own doc.
        let phase_b_pre_total_atomic_buf =
            placeholder_buffer(device, "phase_b_pre_total_atomic", true);
        let phase_b_post_total_atomic_buf =
            placeholder_buffer(device, "phase_b_post_total_atomic", true);
        let phase_b_a_buf = placeholder_buffer(device, "phase_b_a", true);
        let phase_b_b_buf = placeholder_buffer(device, "phase_b_b", false);
        let phase_b_raw_splat_history_buf =
            placeholder_buffer(device, "phase_b_raw_splat_history", true);
        let phase_b_params_buf = uniform_buffer::<SurfaceParams>(device, "phase_b_params");
        let render_params_b_buf =
            uniform_buffer::<SurfaceRenderParams>(device, "surface_render_params_b");

        // Phase B's own wave/visibility/band state -- same real, disclosed
        // placeholder-then-grow convention as every other buffer above.
        let phase_b_wave_bufs = std::array::from_fn(|i| {
            placeholder_buffer(
                device,
                match i {
                    0 => "phase_b_wave_0",
                    1 => "phase_b_wave_1",
                    _ => "phase_b_wave_2",
                },
                true,
            )
        });
        let phase_b_wave_density_prev_buf =
            placeholder_buffer(device, "phase_b_wave_density_prev", true);
        let phase_b_visibility_buf = placeholder_buffer(device, "phase_b_visibility_state", true);
        let phase_b_band_state_buf = placeholder_buffer(device, "phase_b_band_state", true);

        // Real, persistent wave field -- placeholder-then-grow, same
        // convention as the surface buffers above. WebGPU/wgpu guarantees
        // newly created buffers start zero-filled (undisturbed water genuinely
        // starts at zero height -- no explicit clear pass needed). THREE
        // buffers, not two -- see this struct's own `wave_bufs` field doc.
        let wave_bufs = std::array::from_fn(|i| {
            placeholder_buffer(
                device,
                match i {
                    0 => "wave_0",
                    1 => "wave_1",
                    _ => "wave_2",
                },
                true,
            )
        });
        let wave_params_buf = uniform_buffer::<WaveStepParams>(device, "wave_params");
        // Real temporal-disturbance history -- see `wave_density_prev_buf`'s
        // own doc. Starts all-zero (WebGPU guarantee), but the render call
        // seeds it from the real settled density before the first `wave_step`
        // ever reads it (`wave_prev_seeded`), so this zero is never actually
        // read as a "previous" density.
        let wave_density_prev_buf = placeholder_buffer(device, "wave_density_prev", true);

        // Real hysteresis visibility state -- single persistent buffer,
        // starts all-zero (WebGPU guarantee), meaning every cell starts
        // "not visible" until it genuinely earns visibility on its own
        // first real frame (a harmless, expected one-time conservative
        // bias from hysteresis itself, not a bug). COPY_SRC: readback/
        // diagnostic tools (incl. this crate's own tests) need to copy FROM it.
        let visibility_buf = placeholder_buffer(device, "visibility_state", true);
        let visibility_params_buf = uniform_buffer::<VisibilityParams>(device, "visibility_params");

        // Real hysteresis color-band state -- single persistent buffer,
        // same real, disclosed all-zero starting bias as `visibility_buf`
        // (every cell starts in band 0 until it genuinely earns a
        // different one on its own first real frame).
        let band_state_buf = placeholder_buffer(device, "band_state", true);
        let band_hysteresis_params_buf =
            uniform_buffer::<BandHysteresisParams>(device, "band_hysteresis_params");

        // Real, persistent raw-splat history -- starts all-zero (WebGPU
        // guarantee), a real, harmless one-time bias (first frame's blend
        // ramps up to the true value over a few frames).
        let raw_splat_history_buf = placeholder_buffer(device, "raw_splat_history", true);

        // Real, persistent light-fluence diffusion field -- placeholder-then-
        // grow, same convention as the wave field above. Starts all-zero
        // (WebGPU guarantee) -- a real, harmless one-time bias (no light has
        // diffused yet on the very first frame, which is simply true).
        let light_phi_bufs = std::array::from_fn(|i| {
            placeholder_buffer(
                device,
                match i {
                    0 => "light_phi_0",
                    _ => "light_phi_1",
                },
                true,
            )
        });
        let light_diffuse_params_buf =
            uniform_buffer::<LightDiffuseParams>(device, "light_diffuse_params");

        Self {
            instance_buffer,
            storage_instances,
            vertex_buffer,
            index_buffer,
            camera_buffer,
            render_config_buf,
            optical_table_buf,
            grid_volume_params_buf,
            grid_visibility_buf,
            grid_visibility_params_buf,
            surface_atomic_buf,
            surface_temp_atomic_buf,
            surface_temp_float_buf,
            pre_total_atomic_buf,
            post_total_atomic_buf,
            surface_temp_b_buf,
            surface_a_buf,
            surface_b_buf,
            surface_params_buf,
            surface_render_params_buf,
            surface_material_mass_buf,
            surface_moments_buf,
            phase_b_atomic_buf,
            phase_b_temp_atomic_buf,
            phase_b_temp_float_buf,
            phase_b_pre_total_atomic_buf,
            phase_b_post_total_atomic_buf,
            phase_b_a_buf,
            phase_b_b_buf,
            phase_b_raw_splat_history_buf,
            phase_b_params_buf,
            render_params_b_buf,
            phase_b_wave_bufs,
            phase_b_wave_density_prev_buf,
            phase_b_visibility_buf,
            phase_b_band_state_buf,
            wave_bufs,
            wave_params_buf,
            wave_density_prev_buf,
            visibility_buf,
            visibility_params_buf,
            band_state_buf,
            band_hysteresis_params_buf,
            raw_splat_history_buf,
            light_phi_bufs,
            light_diffuse_params_buf,
        }
    }
}

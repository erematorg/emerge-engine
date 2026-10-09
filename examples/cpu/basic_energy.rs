extern crate emerge_engine as emerge;

/// A live, windowed run of `energy::electromagnetics`'s Laplace potential solver and
/// dielectric-breakdown leader (Niemeyer, Pietronero & Wiesmann 1984) -- see that
/// module's doc for the physics and citations.
///
/// Not MPM: there are no `Particle`s and no MPM grid here. An energy field (electric
/// potential) exists at every point in space, matter or not, so it is a plain 2D grid
/// (Eulerian), not discrete particles (Lagrangian). The field and leader state are
/// rasterized to a CPU RGBA buffer each frame and blitted through a minimal
/// fullscreen-texture shader (`gui_common/fullscreen_texture.wgsl`); the particle, grid
/// volume and curvature-flow renderers read only MPM state.
///
/// The physics runs at its own speed: the leader grows to completion once, before the
/// window opens, at the cited stepped-leader speed (~1.5e5 m/s, Rakov & Uman) and this
/// scene's 1-meter-per-cell scale, and the physical duration is printed (see
/// `State::new`), with no slow-motion factor in the simulation's clock. The window
/// shows a replay of that completed event at a separately labeled playback rate, as a
/// high-speed camera relates to slow-motion footage: the bolt happened at full speed,
/// the projector runs slow. `PLAYBACK_CELLS_PER_SECOND` is that projector rate.
///
///   cargo run --example basic_energy --features "render experimental"
use emerge::energy::electromagnetics::{DielectricBreakdownLeader, ElectricPotentialField};
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

const WIDTH: usize = 96;
const HEIGHT: usize = 140;
const CLOUD_POTENTIAL: f32 = 0.0;
const GROUND_POTENTIAL: f32 = 1.0;
// The model's eta (Niemeyer-Pietronero-Wiesmann), the value checked against the
// reference implementation (github.com/diluuuu10/triggered-discharge, cloned in tmp/).
const ETA: f32 = 2.5;
const RELAX_ITERATIONS_PER_STEP: usize = 8;
// Stepped-leader speed (Rakov & Uman; the stepped-leader literature measures
// 1.5e5-4e5 m/s). 1 grid cell = 1 meter, a compact illustrative scale: a storm cloud
// base sits on the order of 1-2 km up, and this scene is 140 cells tall.
const LEADER_SPEED_M_S: f32 = 1.5e5;
const CELL_METERS: f32 = 1.0;
const REAL_SECONDS_PER_STEP: f32 = CELL_METERS / LEADER_SPEED_M_S;
// Fixed seed: a reload (press R) reproduces the exact same channel, byte for byte, a
// checkable determinism test (LcgRng is a deterministic PRNG: complex-looking, fully
// determined by the seed). Make it time-derived for a different channel per run.
const RNG_SEED: u32 = 20260827; // any fixed seed; this one reads as a date
// Playback rate ONLY -- how many already-computed real cells the replay
// reveals per real second on screen. This is the "projector speed," never
// the simulation's own clock (see this file's own top doc) -- the leader
// is grown to full completion, at its real physical speed, before this
// number is ever used.
const PLAYBACK_CELLS_PER_SECOND: f32 = 80.0;

struct App {
    window: Option<Arc<Window>>,
    state: Option<State>,
}

struct State {
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    texture: wgpu::Texture,
    // The real, ALREADY-COMPLETED result -- see this file's own top doc.
    // `field`'s final relaxed state is used as the background for the
    // whole replay (a disclosed simplification: the potential field
    // changes shape as the channel grows, but re-deriving and
    // storing every intermediate field snapshot just to animate the
    // background gradient would cost real memory for no change to what
    // this scene exists to show -- the channel's own real shape).
    field: ElectricPotentialField,
    channel_cells: Vec<(usize, usize)>,
    revealed: usize,
    last_frame_instant: std::time::Instant,
    rgba: Vec<u8>,
}

/// The physics, run to completion once, at its own speed (see the file doc for why
/// there is no simulation-speed dial). Shared by `State::new` and `State::reload`, so a
/// reload re-runs the same computation and the R-key determinism check compares like
/// with like.
fn compute_leader() -> (
    ElectricPotentialField,
    Vec<(usize, usize)>,
    f32,
    std::time::Duration,
) {
    let mut field = ElectricPotentialField::new(WIDTH, HEIGHT, CLOUD_POTENTIAL, GROUND_POTENTIAL);
    field.relax_n(HEIGHT * HEIGHT / 4);
    let mut leader =
        DielectricBreakdownLeader::new(&mut field, WIDTH / 2, 0, CLOUD_POTENTIAL, ETA, RNG_SEED);
    let compute_start = std::time::Instant::now();
    loop {
        if !leader.grow_step(&mut field, RELAX_ITERATIONS_PER_STEP) {
            break;
        }
        if let Some(&(_, y)) = leader.growth_order().last()
            && y >= HEIGHT - 2
        {
            break;
        }
    }
    let compute_wall_time = compute_start.elapsed();
    let channel_cells = leader.growth_order().to_vec();
    let real_physical_duration_s = channel_cells.len() as f32 * REAL_SECONDS_PER_STEP;
    println!(
        "basic_energy: leader grown to completion -- {} real cells, real physical duration \
         {:.2} microseconds (computed in {:.2} ms of actual CPU time) -- last cell {:?}",
        channel_cells.len(),
        real_physical_duration_s * 1.0e6,
        compute_wall_time.as_secs_f64() * 1000.0,
        channel_cells.last()
    );
    (
        field,
        channel_cells,
        real_physical_duration_s,
        compute_wall_time,
    )
}

fn rasterize(field: &ElectricPotentialField, channel: &[(usize, usize)], rgba: &mut [u8]) {
    let (w, h) = (field.width(), field.height());
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 4;
            // Background: the solved potential, cloud (0) to ground (1), as a dim
            // blue-grey ramp.
            let phi = field.phi_at(x, y).clamp(0.0, 1.0);
            let bg = 12.0 + phi * 10.0;
            rgba[i] = (bg * 1.1) as u8;
            rgba[i + 1] = (bg * 1.3) as u8;
            rgba[i + 2] = (bg * 1.8) as u8;
            rgba[i + 3] = 255;
        }
    }
    for &(x, y) in channel {
        let i = (y * w + x) * 4;
        rgba[i] = 235;
        rgba[i + 1] = 245;
        rgba[i + 2] = 255;
    }
}

impl State {
    async fn new(window: Arc<Window>) -> Self {
        // Minimal wgpu bootstrap rather than gui_common::Gfx, which also sets up an
        // egui panel this scene (no panel, no mouse) does not use; reusing it would
        // leave those fields and functions unread in this binary (dead_code). This
        // scene draws one fullscreen texture.
        let size = window.inner_size();
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let surface = instance.create_surface(window.clone()).unwrap();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("no GPU adapter");
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await
            .unwrap();
        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(caps.formats[0]);
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::AutoVsync,
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &surface_config);

        // Checkable fingerprint -- same seed (RNG_SEED, fixed, see its
        // doc) means this must be byte-identical across runs/reloads.
        // Press R to reload and compare this line directly.
        let (field, channel_cells, _duration_s, _wall_time) = compute_leader();

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("energy_field_texture"),
            size: wgpu::Extent3d {
                width: WIDTH as u32,
                height: HEIGHT as u32,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fullscreen_texture"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../gui_common/fullscreen_texture.wgsl").into(),
            ),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("energy_bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("energy_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("energy_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("energy_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(format.into())],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let mut rgba = vec![0u8; WIDTH * HEIGHT * 4];
        rasterize(&field, &[], &mut rgba);

        println!(
            "basic_energy: {WIDTH}x{HEIGHT} grid  |  1 cell = {CELL_METERS:.1} m  |  real leader speed {LEADER_SPEED_M_S:.1e} m/s  |  playback {PLAYBACK_CELLS_PER_SECOND:.0} cells/s (viewing rate only, not physics)  |  R reload  Q quit"
        );

        Self {
            surface,
            surface_config,
            device,
            queue,
            pipeline,
            bind_group,
            texture,
            field,
            channel_cells,
            revealed: 0,
            last_frame_instant: std::time::Instant::now(),
            rgba,
        }
    }

    /// Reload: re-runs the same physics (see `compute_leader`) and resets the replay,
    /// without touching any GPU resource (device, surface, texture and pipeline stay).
    /// Rebuilding the whole `State` would create a new `wgpu::Instance`/adapter/device/
    /// surface for the same OS window while the old one is alive, and wgpu rejects the
    /// resulting texture ("Texture::create_view ... Texture ... is invalid"). A physics
    /// reload needs no new GPU context.
    fn reload(&mut self) {
        let (field, channel_cells, _duration_s, _wall_time) = compute_leader();
        self.field = field;
        self.channel_cells = channel_cells;
        self.revealed = 0;
        self.last_frame_instant = std::time::Instant::now();
    }

    fn resize(&mut self, w: u32, h: u32) {
        if w == 0 || h == 0 {
            return;
        }
        self.surface_config.width = w;
        self.surface_config.height = h;
        self.surface.configure(&self.device, &self.surface_config);
    }

    fn update_and_render(&mut self) {
        // Replay ONLY -- the physics already ran to completion in `new`, at
        // its own speed (see this file's own top doc). This just
        // reveals more of that already-computed, already-real recording
        // based on actual elapsed wall-clock time, at the explicit
        // PLAYBACK_CELLS_PER_SECOND viewing rate -- never re-simulates or
        // re-paces the physics itself.
        let now = std::time::Instant::now();
        let dt = (now - self.last_frame_instant).as_secs_f32();
        self.last_frame_instant = now;
        if self.revealed < self.channel_cells.len() {
            let advance = (dt * PLAYBACK_CELLS_PER_SECOND).round() as usize;
            self.revealed = (self.revealed + advance.max(1)).min(self.channel_cells.len());
        }

        rasterize(
            &self.field,
            &self.channel_cells[..self.revealed],
            &mut self.rgba,
        );
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &self.rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some((WIDTH * 4) as u32),
                rows_per_image: Some(HEIGHT as u32),
            },
            wgpu::Extent3d {
                width: WIDTH as u32,
                height: HEIGHT as u32,
                depth_or_array_layers: 1,
            },
        );

        let Ok(frame) = self.surface.get_current_texture() else {
            return;
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut rp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("energy_render_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
            rp.set_pipeline(&self.pipeline);
            rp.set_bind_group(0, &self.bind_group, &[]);
            rp.draw(0..3, 0..1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        frame.present();
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        let w = Arc::new(
            el.create_window(
                winit::window::WindowAttributes::default()
                    .with_title(
                        "emerge -- basic_energy: dielectric breakdown leader (R reload, Q quit)",
                    )
                    .with_inner_size(winit::dpi::LogicalSize::new(384u32, 560u32)),
            )
            .unwrap(),
        );
        self.state = Some(pollster::block_on(State::new(w.clone())));
        self.window = Some(w);
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(s) = self.state.as_mut() else {
            return;
        };
        match event {
            WindowEvent::CloseRequested => el.exit(),
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(key),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => {
                if matches!(key, KeyCode::Escape | KeyCode::KeyQ) {
                    el.exit();
                } else if key == KeyCode::KeyR {
                    // Checkable determinism test -- RNG_SEED is fixed
                    // (see its doc), so this must print the exact same
                    // fingerprint line every time, and the visible channel
                    // must be pixel-identical. See `State::reload`'s own
                    // doc for why this does NOT touch GPU resources.
                    println!(
                        "--- reload (same RNG_SEED -- compare the fingerprint line above) ---"
                    );
                    s.reload();
                }
            }
            WindowEvent::Resized(sz) => s.resize(sz.width, sz.height),
            WindowEvent::RedrawRequested => {
                s.update_and_render();
                if let Some(w) = &self.window {
                    w.request_redraw();
                }
            }
            _ => {}
        }
    }
}

fn main() {
    let el = EventLoop::new().unwrap();
    el.set_control_flow(ControlFlow::Poll);
    let mut app = App {
        window: None,
        state: None,
    };
    el.run_app(&mut app).unwrap();
}

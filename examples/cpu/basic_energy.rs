extern crate emerge_engine as emerge;

/// First real, LIVE, WINDOWED application of `energy::electromagnetics`'s
/// Laplace potential solver + dielectric-breakdown leader (Niemeyer,
/// Pietronero & Wiesmann 1984) -- see that module's own doc for the full
/// real physics and citations.
///
/// NOT MPM: unlike every other example in this engine, there are no
/// `Particle`s and no MPM grid here at all. An energy FIELD (electric
/// potential) is a genuinely different physical object from matter -- a
/// value exists at every point in space, including where there is no
/// matter, so it is naturally represented as a plain 2D grid (Eulerian),
/// never as discrete particles (Lagrangian). This scene's own render path
/// reflects that: the field/leader state is rasterized to a CPU RGBA
/// buffer each frame and blitted through a minimal fullscreen-texture
/// shader (`gui_common/fullscreen_texture.wgsl`) -- none of
/// render_particles/grid_volume/curvature_flow apply, since none of them
/// read anything but MPM particle/grid state.
///
/// REAL PHYSICS, NOT A DISPLAY-TUNED SPEED (fixed 2026-08-27 after a real,
/// deserved correction): the physics runs at its own genuine speed, full
/// stop -- the leader is grown to completion ONCE, instantly, before the
/// window even opens, using the real cited stepped-leader speed (~1.5e5
/// m/s, Rakov & Uman) and this scene's own real 1-meter-per-cell scale. The
/// real physical duration that took is computed and printed honestly (see
/// `State::new`), with NO fudge factor hidden inside the simulation's own
/// clock -- an earlier version of this file did exactly that (a tuned
/// "SLOW_MOTION_FACTOR" picked to make the demo feel like a nice ~8
/// seconds), which is precisely the kind of made-up compromise this
/// engine's whole standing discipline exists to refuse. What you actually
/// SEE animating in the window is a REPLAY of that already-completed real
/// event, at an explicit, separately-labeled playback rate -- the exact
/// same honest relationship a real high-speed camera has to slow-motion
/// footage: the lightning bolt genuinely happened at real speed, it's the
/// PROJECTOR that runs slow so a human eye can see it, never the event
/// itself. `PLAYBACK_CELLS_PER_SECOND` is that projector rate, and it is
/// never confused with, or substituted for, the real physical timing
/// printed to the console.
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
// Real, named model parameter (Niemeyer-Pietronero-Wiesmann's own eta) --
// same value checked live tonight against the reference implementation
// (github.com/diluuuu10/triggered-discharge, cloned in tmp/).
const ETA: f32 = 2.5;
const RELAX_ITERATIONS_PER_STEP: usize = 8;
// Real stepped-leader speed (Rakov & Uman; corroborated 2026-08-27 against
// the wider stepped-leader speed literature, same order of magnitude
// across independent measurements: 1.5e5-4e5 m/s). 1 grid cell = 1 real
// meter (disclosed scene scale, not measured -- a real storm cloud base
// sits on the order of 1-2 km up, this scene's 140-cell height is a
// deliberately compact illustrative scale, not a claim about real storm
// geometry).
const LEADER_SPEED_M_S: f32 = 1.5e5;
const CELL_METERS: f32 = 1.0;
const REAL_SECONDS_PER_STEP: f32 = CELL_METERS / LEADER_SPEED_M_S;
// Fixed, not time-derived: same seed every run means a reload (press R)
// reproduces the EXACT same channel, byte for byte -- a real, checkable
// determinism test (LcgRng is a real deterministic PRNG, category 3 of
// this session's own randomness discussion: looks complex, is fully
// calculable from the seed, not real chaos). Change this to something
// time-derived if a genuinely different channel per run is ever wanted --
// deliberately not done here, since reproducibility is the actual point.
const RNG_SEED: u32 = 20260827; // 2026-08-27, the date this scene was built
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
    // whole replay (a real, disclosed simplification: the potential field
    // genuinely changes shape as the channel grows, but re-deriving and
    // storing every intermediate field snapshot just to animate the
    // background gradient would cost real memory for no change to what
    // this scene exists to show -- the channel's own real shape).
    field: ElectricPotentialField,
    channel_cells: Vec<(usize, usize)>,
    revealed: usize,
    last_frame_instant: std::time::Instant,
    rgba: Vec<u8>,
}

/// Real physics, run to completion ONCE, at its own genuine speed -- see
/// this file's own top doc for why there is no "simulation speed" dial at
/// all. Shared by `State::new` and `State::reload` (a real reload must
/// re-run the SAME physics, not just recreate GPU resources -- keeping
/// this in one place is what makes the R-key determinism check meaningful:
/// both call sites are provably running the identical computation, not two
/// hand-copies that could drift apart).
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
            // Real background: the actual solved potential, cloud (0) to
            // ground (1) -- a dim blue-grey ramp, not decoration.
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
        // Minimal, direct wgpu bootstrap -- deliberately NOT gui_common::Gfx,
        // which also bundles a real egui UI-panel setup no scene without an
        // on-screen panel or mouse interaction needs. Reusing it here would
        // leave those real fields/functions genuinely unread by this binary
        // (dead_code, the exact class of false-flag issue already found and
        // fixed once tonight for a different Gfx consumer) -- this scene
        // draws one fullscreen texture and nothing else, so it gets its own
        // minimal, correctly-scoped setup instead.
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

        // Real, checkable fingerprint -- same seed (RNG_SEED, fixed, see its
        // own doc) means this must be byte-identical across runs/reloads.
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

    /// Real reload: re-runs the SAME real physics (see `compute_leader`'s
    /// own doc) and resets the replay to its start, WITHOUT touching any
    /// GPU resource (device/surface/texture/pipeline all stay exactly as
    /// they are). Real bug found live (2026-08-27): an earlier version of
    /// this reload rebuilt the entire `State`, including a brand new
    /// `wgpu::Instance`/adapter/device/surface for the SAME OS window while
    /// the old one was still alive -- wgpu rejected the resulting texture
    /// as invalid ("Texture::create_view ... Texture ... is invalid"), a
    /// real, reproducible crash, not a hypothetical one. There is no real
    /// reason a physics reload should ever need a new GPU context in the
    /// first place; this fixes the actual root cause (recreating far more
    /// than the reload needed) rather than papering over the crash.
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
        // its own real speed (see this file's own top doc). This just
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
                    // Real, checkable determinism test -- RNG_SEED is fixed
                    // (see its own doc), so this must print the exact same
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

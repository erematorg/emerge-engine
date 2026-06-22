extern crate emerge_engine as emerge;

/// Real-time particle-count ramp test — finds the actual current FPS ceiling, not a
/// guessed one. Falling-sand-game style: ONE fixed pour point near the top of the grid (an
/// hourglass neck, not a sweeping nozzle), sand falls through it continuously and
/// automatically forever, piling up on the floor (domain boundary). No input required —
/// nothing to configure, nothing to click. Rendered live, FPS shown on-screen.
///
/// FINDING (confirmed 2026-06-21): with growth running every frame and no throttle, FPS was
/// already DEGRADED (35-52fps) at just ~250 particles — not from real physics/render cost, but
/// because the regrowth stall (rebuilding the whole GpuSimulation every single frame) costs
/// ~20-28ms on its own, regardless of how little is actually being simulated. This proves
/// "grow via full recreation" cannot be a real runtime particle-creation mechanism at any
/// frequency above occasional — LP needs a genuine incremental add-particles GPU API.
///
/// IMPORTANT CAVEAT (this is itself a finding, not just a benchmark detail): GpuSimulation has
/// no "add particles to a running simulation" API. To grow the particle count, this example
/// reads back the current particles, builds a bigger combined list, and recreates the whole
/// GpuSimulation + Renderer from scratch — the only way possible with today's API. LP's
/// roadmap explicitly wants runtime particle creation (creature spawning, "particle reseeding"
/// for cell division/phase changes) — this example's growth mechanism is a workaround for a
/// missing capability, and the regrowth stall itself is measured and printed below so the cost
/// of that workaround is visible, not hidden inside the FPS average.
///
///   cargo run --release --example stress_realtime_scaling --features render
use std::sync::Arc;

use egui_wgpu::ScreenDescriptor;
use emerge::diagnostics::log_frame_gpu;
use emerge::gpu::GpuSimulation;
use emerge::render::{ColorMode, Renderer};
use emerge::{DruckerPragerMaterial, MaterialRegistry, SimConfig, SpawnRegion, build_particles};
use glam::Vec2;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

// 128 (matches LP's own mpm::World::GRID_RES), small enough that the whole domain -- pour
// point AND floor -- fits in frame at the default camera zoom. GRID=512 was technically safe
// re: the grid_resolution_cost cliff (1024-2048) but made the floor invisible off-camera,
// making it impossible to actually see whether sand was piling up.
const GRID: usize = 128;
const DT: f32 = 0.05;
const MAT_ID: u32 = 0;
const LABELS: &[(u32, &str)] = &[(MAT_ID, "body")];

const BATCH_PARTICLES: usize = 200; // bigger pour per drop -- ~1000 particles/sec at the interval below
const GROWTH_INTERVAL_SECS: f32 = 0.2; // faster cadence, still above the ~25-30ms regrowth-stall floor
const GROWTH_CAP: usize = 600_000; // past LP's stated 500k target, to see where it actually breaks

struct App {
    window: Option<Arc<Window>>,
    state: Option<State>,
}

struct State {
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    sim: GpuSimulation,
    renderer: Renderer,
    fmt: wgpu::TextureFormat,
    config: SimConfig,
    frame: u64,
    fps_timer: std::time::Instant,
    fps_frames: u64,
    last_fps: f32,
    last_growth: std::time::Instant,
    growth_index: u32,
    last_regrowth_stall_ms: f64,
    egui_ctx: egui::Context,
    egui_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
}

fn make_registry() -> MaterialRegistry {
    // Granular sand -- classic falling-sand-game material, settles into a pile under the
    // fixed pour point instead of bouncing/jiggling.
    MaterialRegistry::with_default(Box::new(DruckerPragerMaterial::cohesionless(133.3, 0.333)))
}

impl State {
    async fn new(window: Arc<Window>) -> Self {
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
            .expect("device request failed");
        let device = Arc::new(device);
        let queue = Arc::new(queue);

        let caps = surface.get_capabilities(&adapter);
        let fmt = caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(caps.formats[0]);
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: fmt,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::AutoVsync,
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &surface_config);

        let config = SimConfig {
            gravity: Vec2::new(0.0, -0.3),
            apic_blend: 0.3,
            max_substeps_per_step: 8,
            ..SimConfig::standard(GRID, DT, Vec2::new(0.0, -0.3))
        };

        let particles = build_initial_batch(&config);
        let sim = GpuSimulation::with_device(
            device.clone(),
            queue.clone(),
            config,
            particles,
            make_registry(),
        );

        let mut renderer = Renderer::new(&device, sim.particle_count(), fmt);
        renderer.set_camera(&queue, GRID as u32, size.width, size.height, 0.6, true);
        renderer.set_color_mode(ColorMode::ByVelocity);

        let egui_ctx = egui::Context::default();
        let egui_state = egui_winit::State::new(
            egui_ctx.clone(),
            egui_ctx.viewport_id(),
            window.as_ref(),
            None,
            None,
            None,
        );
        let egui_renderer = egui_wgpu::Renderer::new(
            &device,
            fmt,
            egui_wgpu::RendererOptions {
                msaa_samples: 1,
                ..Default::default()
            },
        );

        println!(
            "stress_realtime_scaling: starting at {} particles, +1 ball ({BATCH_PARTICLES} particles) every {GROWTH_INTERVAL_SECS}s, cap={GROWTH_CAP}",
            sim.particle_count()
        );
        println!("  [Q] quit -- that's the only input, the balls fall on their own");

        Self {
            surface,
            surface_config,
            device,
            queue,
            sim,
            renderer,
            fmt,
            config,
            frame: 0,
            fps_timer: std::time::Instant::now(),
            fps_frames: 0,
            last_fps: 0.0,
            last_growth: std::time::Instant::now(),
            growth_index: 0,
            last_regrowth_stall_ms: 0.0,
            egui_ctx,
            egui_state,
            egui_renderer,
        }
    }

    fn resize(&mut self, w: u32, h: u32) {
        if w == 0 || h == 0 {
            return;
        }
        self.surface_config.width = w;
        self.surface_config.height = h;
        self.surface.configure(&self.device, &self.surface_config);
        self.renderer
            .set_camera(&self.queue, GRID as u32, w, h, 0.6, true);
    }

    /// Reads back current particles, appends a new batch, and rebuilds the whole
    /// GpuSimulation + Renderer. Times the stall explicitly -- this IS the cost of LP's
    /// missing "spawn at runtime" capability, made visible instead of averaged away.
    fn grow(&mut self) {
        let stall_start = std::time::Instant::now();

        self.sim.sync_particles_blocking();
        let mut particles: Vec<emerge::Particle> = self.sim.particles().to_vec();
        self.growth_index += 1;
        particles.extend(build_initial_batch(&self.config));

        let n = particles.len();
        self.sim = GpuSimulation::with_device(
            self.device.clone(),
            self.queue.clone(),
            self.config,
            particles,
            make_registry(),
        );
        self.renderer = Renderer::new(&self.device, n, self.fmt);
        self.renderer.set_camera(
            &self.queue,
            GRID as u32,
            self.surface_config.width,
            self.surface_config.height,
            0.6,
            true,
        );
        self.renderer.set_color_mode(ColorMode::ByVelocity);

        self.last_regrowth_stall_ms = stall_start.elapsed().as_secs_f64() * 1000.0;
        println!(
            "GROWTH #{}: now {n} particles (+{BATCH_PARTICLES}) -- regrowth stall = {:.1}ms",
            self.growth_index, self.last_regrowth_stall_ms
        );
        let _ = std::io::Write::flush(&mut std::io::stdout());
    }

    fn update_and_render(&mut self, window: &Window) {
        let interval_ok = self.sim.particle_count() < GROWTH_CAP
            && self.last_growth.elapsed().as_secs_f32() >= GROWTH_INTERVAL_SECS;
        if interval_ok {
            self.grow();
            self.last_growth = std::time::Instant::now();
        }

        let output = match self.surface.get_current_texture() {
            Ok(t) => t,
            Err(_) => return,
        };

        self.sim.step_frame();
        self.frame += 1;
        self.fps_frames += 1;

        if self.fps_timer.elapsed().as_secs_f32() >= 1.0 {
            self.last_fps = self.fps_frames as f32 / self.fps_timer.elapsed().as_secs_f32();
            println!(
                "n={:>7} fps={:>6.1} ({})",
                self.sim.particle_count(),
                self.last_fps,
                if self.last_fps >= 58.0 {
                    "OK 60fps"
                } else if self.last_fps >= 30.0 {
                    "DEGRADED"
                } else {
                    "BAD"
                }
            );
            let _ = std::io::Write::flush(&mut std::io::stdout());
            self.fps_timer = std::time::Instant::now();
            self.fps_frames = 0;
        }
        if self.frame.is_multiple_of(120) {
            self.sim.sync_particles_blocking();
            log_frame_gpu(self.frame, DT, self.sim.particles(), LABELS, 1);
        }

        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        self.renderer.render_gpu(
            &self.device,
            &self.queue,
            self.sim.particle_buffer(),
            self.sim.particle_count(),
            &view,
            true,
        );

        // --- egui overlay: read-only FPS/particle stats, no controls (one entrance only) ---
        let raw_input = self.egui_state.take_egui_input(window);
        let fps = self.last_fps;
        let n = self.sim.particle_count();
        let stall_ms = self.last_regrowth_stall_ms;
        let status = if fps >= 58.0 {
            "OK 60fps"
        } else if fps >= 30.0 {
            "DEGRADED"
        } else {
            "BAD"
        };

        let full_output = self.egui_ctx.run(raw_input, |ctx| {
            egui::Window::new("stress_realtime_scaling")
                .default_pos([10.0, 10.0])
                .default_width(260.0)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.heading(format!("fps = {fps:.0}  ({status})"));
                    ui.label(format!("particles = {n}"));
                    ui.label(format!("last regrowth stall = {stall_ms:.1}ms"));
                    ui.separator();
                    ui.label("Sand falls automatically through one fixed point, forever.");
                    ui.label("Q to quit");
                });
        });

        self.egui_state
            .handle_platform_output(window, full_output.platform_output);
        let tris = self
            .egui_ctx
            .tessellate(full_output.shapes, full_output.pixels_per_point);
        let sd = ScreenDescriptor {
            size_in_pixels: [self.surface_config.width, self.surface_config.height],
            pixels_per_point: full_output.pixels_per_point,
        };
        for (id, delta) in &full_output.textures_delta.set {
            self.egui_renderer
                .update_texture(&self.device, &self.queue, *id, delta);
        }
        let cmd = {
            let mut enc = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            self.egui_renderer
                .update_buffers(&self.device, &self.queue, &mut enc, &tris, &sd);
            let mut rp = enc
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    ..Default::default()
                })
                .forget_lifetime();
            self.egui_renderer.render(&mut rp, &tris, &sd);
            drop(rp);
            enc.finish()
        };
        self.queue.submit(std::iter::once(cmd));
        for id in &full_output.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }
        output.present();
    }
}

/// Builds one small batch of sand falling through ONE fixed point near the top of the grid --
/// an hourglass neck, not a sweeping nozzle. Always the same (x, y); the pile grows underneath
/// it exactly like a falling-sand game / hourglass.
fn build_initial_batch(config: &SimConfig) -> Vec<emerge::Particle> {
    const POUR_RADIUS: f32 = 4.0; // cells -- ~200 particles at spacing 0.5 (matches BATCH_PARTICLES)
    let center = Vec2::new(GRID as f32 * 0.5, GRID as f32 * 0.92);
    build_particles(
        config,
        SpawnRegion::for_sim(config)
            .at(center)
            .disk(POUR_RADIUS)
            .spacing(0.5)
            .material(MAT_ID)
            .precompute_volumes(),
    )
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(
                    winit::window::WindowAttributes::default()
                        .with_title("emerge -- realtime particle-count ramp [GPU]")
                        .with_inner_size(winit::dpi::LogicalSize::new(640u32, 640u32)),
                )
                .unwrap(),
        );
        self.state = Some(pollster::block_on(State::new(window.clone())));
        self.window = Some(window);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(s) = self.state.as_mut() else { return };
        if let Some(w) = &self.window {
            let resp = s.egui_state.on_window_event(w, &event);
            if resp.consumed {
                return;
            }
        }
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(key),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => {
                if key == KeyCode::Escape || key == KeyCode::KeyQ {
                    event_loop.exit();
                }
            }
            WindowEvent::Resized(sz) => s.resize(sz.width, sz.height),
            WindowEvent::RedrawRequested => {
                if let Some(w) = self.window.clone() {
                    s.update_and_render(&w);
                    w.request_redraw();
                }
            }
            _ => {}
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App {
        window: None,
        state: None,
    };
    event_loop.run_app(&mut app).unwrap();
}

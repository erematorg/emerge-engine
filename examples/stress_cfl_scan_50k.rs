extern crate emerge_engine as emerge;

/// Real-time, vsync-paced verification of the CFL-scan fix at the actual 0.1.0 target scene
/// (~50k DP-sand particles, grid_res=320, dt=1/60) — built specifically to check whether the
/// real, paced interactive case behaves like the synthetic tight-loop headless benchmarks
/// (which showed wild 16-919ms per-frame variance, almost certainly a benchmark-pattern
/// artifact, not a real-use problem). Prints live FPS every 2 seconds.
///
///   cargo run --release --example stress_cfl_scan_50k --features "gpu render"
use std::sync::Arc;

use emerge::render::{ColorMode, Renderer};
use emerge::{
    DruckerPragerMaterial, GpuSimulation, MaterialRegistry, SimConfig, SpawnRegion, build_particles,
};
use glam::{IVec2, Vec2};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

const GRID: usize = 320;
const REAL_TIME_DT: f32 = 1.0 / 60.0;
const TARGET: usize = 50_000;

struct App {
    window: Option<Arc<Window>>,
    state: Option<State>,
}

struct State {
    surface: wgpu::Surface<'static>,
    sim: GpuSimulation,
    renderer: Renderer,
    frame: u64,
    fps_timer: std::time::Instant,
    fps_frames: u64,
    worst_frame_ms: f64,
    worst_render_ms: f64,
    worst_present_ms: f64,
    worst_acquire_ms: f64,
}

fn make_sim_data(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> GpuSimulation {
    let config = SimConfig {
        max_substeps_per_step: 4,
        // Default 0.5 is a real, conservative CFL safety margin (matches Klar/sparkl's typical
        // usage). 0.7 is still inside the literature's normal range (commonly 0.3-1.0
        // depending on scheme) — a modest, principled relaxation, not an extreme gamble. Real
        // per-frame GPU cost scales ~linearly with substep count (3 substeps for this stiff
        // DP-sand material at the default), so cutting substeps is the legitimate lever for
        // closing the gap to 60fps, not a band-aid — verify stability carefully, don't just trust it.
        material_cfl_coefficient: 0.7,
        ..SimConfig::standard(GRID, REAL_TIME_DT, Vec2::new(0.0, -0.3))
    };
    let side = ((TARGET as f32) / 4.0).sqrt().ceil() as i32;
    let particles = build_particles(
        &config,
        SpawnRegion {
            spacing: 0.5,
            box_size: IVec2::splat(side),
            box_center: Vec2::splat(GRID as f32 * 0.5),
            precompute_initial_volumes: true,
            ..SpawnRegion::for_sim(&config)
        },
    );
    let registry =
        MaterialRegistry::with_default(Box::new(DruckerPragerMaterial::new(2000.0, 3000.0)));
    let mut sim = GpuSimulation::with_device(device, queue, config, particles, registry);
    // This example renders directly from the GPU particle buffer (render_gpu takes
    // sim.particle_buffer(), never the CPU mirror) and never calls particles()/
    // sync_particles_blocking(). DP-sand also needs zero CPU-side plasticity (any_cpu=false).
    // The default readback_stride=1 was paying for a full GPU->CPU particle copy every frame
    // for a CPU mirror nothing here ever reads — measured as the single largest chunk of
    // step_frame's CPU-side cost (4-7.5ms of ~6.6-10.6ms total). Effectively disabled: a
    // pure-rendering scene like this one has no use for it.
    sim.readback_stride = 1_000_000;
    sim
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
                required_features: adapter.features() & wgpu::Features::TIMESTAMP_QUERY,
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await
            .unwrap();
        let caps = surface.get_capabilities(&adapter);
        let fmt = caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(caps.formats[0]);
        let sc = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: fmt,
            width: size.width,
            height: size.height,
            present_mode: if caps.present_modes.contains(&wgpu::PresentMode::Mailbox) {
                wgpu::PresentMode::Mailbox
            } else {
                println!("Mailbox not supported on this backend, falling back to AutoNoVsync");
                wgpu::PresentMode::AutoNoVsync
            },
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &sc);
        let mut sim = make_sim_data(Arc::new(device), Arc::new(queue));
        if !sim.enable_profiling() {
            println!("TIMESTAMP_QUERY not supported on this device, live GPU pass timing disabled");
        }
        let mut renderer = Renderer::new(sim.device(), sim.particle_count(), fmt);
        renderer.set_camera(sim.queue(), GRID as u32, size.width, size.height, 0.6, true);
        renderer.set_color_mode(ColorMode::ByMaterial);
        println!(
            "stress_cfl_scan_50k: {} particles, grid_res={GRID}, dt={REAL_TIME_DT:.5} | watch FPS below",
            sim.particle_count()
        );
        Self {
            surface,
            sim,
            renderer,
            frame: 0,
            fps_timer: std::time::Instant::now(),
            fps_frames: 0,
            worst_frame_ms: 0.0,
            worst_render_ms: 0.0,
            worst_present_ms: 0.0,
            worst_acquire_ms: 0.0,
        }
    }

    fn update_and_render(&mut self) {
        let acquire_start = std::time::Instant::now();
        let output = match self.surface.get_current_texture() {
            Ok(t) => t,
            Err(_) => return,
        };
        let acquire_ms = acquire_start.elapsed().as_secs_f64() * 1000.0;
        self.worst_acquire_ms = self.worst_acquire_ms.max(acquire_ms);
        let frame_start = std::time::Instant::now();
        self.sim.step_frame();
        let frame_ms = frame_start.elapsed().as_secs_f64() * 1000.0;
        self.worst_frame_ms = self.worst_frame_ms.max(frame_ms);
        self.frame += 1;
        self.fps_frames += 1;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let render_start = std::time::Instant::now();
        self.renderer.render_gpu(
            self.sim.device(),
            self.sim.queue(),
            self.sim.particle_buffer(),
            self.sim.particle_count(),
            &view,
            true,
        );
        let render_ms = render_start.elapsed().as_secs_f64() * 1000.0;
        self.worst_render_ms = self.worst_render_ms.max(render_ms);
        let present_start = std::time::Instant::now();
        output.present();
        let present_ms = present_start.elapsed().as_secs_f64() * 1000.0;
        self.worst_present_ms = self.worst_present_ms.max(present_ms);
        if self.fps_timer.elapsed().as_secs_f32() >= 2.0 {
            let fps = self.fps_frames as f32 / self.fps_timer.elapsed().as_secs_f32();
            let substeps = self.sim.diagnostics_snapshot().substeps_last_step;
            println!(
                "frame={} fps={:.0} substeps={substeps} worst_acquire_ms={:.2} worst_step_frame_ms={:.2} worst_render_ms={:.2} worst_present_ms={:.2}",
                self.frame,
                fps,
                self.worst_acquire_ms,
                self.worst_frame_ms,
                self.worst_render_ms,
                self.worst_present_ms
            );
            if let Some(timings) = self.sim.last_pass_timings_ns() {
                let total: f32 = timings.iter().map(|(_, ns)| ns).sum();
                println!(
                    "  LIVE GPU pass total (one substep, while actually rendering): {:.3}ms",
                    total / 1.0e6
                );
                for (label, ns) in &timings {
                    println!("    {label:<28} {:.4}ms", ns / 1.0e6);
                }
            }
            self.fps_timer = std::time::Instant::now();
            self.fps_frames = 0;
            self.worst_frame_ms = 0.0;
            self.worst_render_ms = 0.0;
            self.worst_present_ms = 0.0;
            self.worst_acquire_ms = 0.0;
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        // Borderless fullscreen — bypasses DWM window composition overhead, a known real fix
        // for compositor-induced present/acquire stalls on Windows that windowed swapchains
        // can suffer from.
        let w = Arc::new(
            el.create_window(
                winit::window::WindowAttributes::default()
                    .with_title("stress_cfl_scan_50k — real-time, vsync-paced, ~50k particles")
                    .with_fullscreen(Some(winit::window::Fullscreen::Borderless(None))),
            )
            .unwrap(),
        );
        self.state = Some(pollster::block_on(State::new(w.clone())));
        self.window = Some(w);
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(s) = self.state.as_mut() else { return };
        match event {
            WindowEvent::CloseRequested => el.exit(),
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

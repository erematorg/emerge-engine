extern crate emerge_engine as emerge;

/// Physics-derived rendering -- Beer-Lambert absorption + blackbody emission.
///
/// GPU simulation (wgpu compute) + zero-readback GPU renderer.
/// Physics and rendering both run fully on GPU.
///
///   Mat 0  Light fluid (rho=700)  -- sigma_a -> pinkish-red, rises in water
///   Mat 1  Water   (rho=1000)    -- sigma_a -> blue-cyan
///
///   cargo run --example render_physics --features render
use std::sync::Arc;

use emerge::diagnostics::log_frame_gpu;
use emerge::gpu::GpuSimulation;
use emerge::render::{ColorMode, Renderer};
use emerge::{Fluid, MaterialRegistry, SimConfig, SpawnRegion, build_particles};
use glam::{IVec2, Vec2};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

const GRID: usize = 64;
const DT: f32 = 0.05;

const BLOB_ID: u32 = 0;
const WATER_ID: u32 = 1;
const LABELS: &[(u32, &str)] = &[(BLOB_ID, "blob"), (WATER_ID, "water")];

const SIGMA_TISSUE: [f32; 3] = [0.05, 0.55, 0.60];
const SIGMA_WATER: [f32; 3] = [0.85, 0.25, 0.07];

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
    cursor_pos: [f32; 2],
    lmb: bool,
    rmb: bool,
    physics_colors: bool,
    frame: u64,
    fps_timer: std::time::Instant,
    fps_frames: u64,
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
                required_limits: adapter.limits(), // use full hardware limits, not wgpu defaults
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

        // -- Simulation ----------------------------------------------------
        // Game-scale gravity (0.3 cells/s²) — matches all other GPU examples.
        // Earth gravity (981 cells/s²) causes floor impact in <7 frames on a 64-cell grid.
        // apic_blend=0.1: 10% APIC + 90% PIC → natural energy dissipation.
        let config = SimConfig {
            gravity: Vec2::new(0.0, -0.3),
            apic_blend: 0.1,
            max_substeps_per_step: 20,
            ..SimConfig::earth(GRID, 0.01, DT)
        };

        // Two-layer stratification: heavy water below, light blob layer above.
        // Both start at rest — no fall, no impact spike. Gentle pressure equilibration
        // at the interface; apic_blend=0.1 damps sloshing within ~2 s.
        // Water: y = 3..15 (12 cells deep)
        let mut particles = build_particles(
            &config,
            SpawnRegion {
                spacing: 0.5,
                box_size: IVec2::new(GRID as i32 - 8, 12),
                box_center: Vec2::new(GRID as f32 * 0.5, 9.0),
                material_id: WATER_ID,
                precompute_initial_volumes: true,
                ..SpawnRegion::for_sim(&config)
            },
        );
        // Blob layer: y = 13..21 (8 cells, directly on top of water)
        particles.extend(build_particles(
            &config,
            SpawnRegion {
                spacing: 0.5,
                box_size: IVec2::new(GRID as i32 - 8, 8),
                box_center: Vec2::new(GRID as f32 * 0.5, 17.0),
                material_id: BLOB_ID,
                precompute_initial_volumes: true,
                ..SpawnRegion::for_sim(&config)
            },
        ));

        // Blob: lighter than water (rho=700 kg/m³) → floats via MPM pressure stratification.
        // Heavier water particles drive grid momentum downward; lighter blob stays above.
        let mut registry = MaterialRegistry::with_default(
            Fluid {
                rho_kg_m3: 700.0,
                eta_pa_s: 0.05,
                bulk_modulus_pa: 64_000.0,
                yield_stress_pa: None,
            }
            .material(&config),
        );
        // Weakly-compressible water: K ≈ 128 kPa (WCSPH rule, c_ref ≈ 30 m/s).
        registry.insert(
            WATER_ID,
            Fluid {
                rho_kg_m3: 1000.0,
                eta_pa_s: 0.001,
                bulk_modulus_pa: 128_571.0,
                yield_stress_pa: None,
            }
            .material(&config),
        );

        let sim =
            GpuSimulation::with_device(device.clone(), queue.clone(), config, particles, registry);

        // -- Renderer ------------------------------------------------------
        let mut renderer = Renderer::new(&device, sim.particle_count(), fmt);
        renderer.set_camera(&queue, GRID as u32, size.width, size.height, 0.6, true);
        renderer.set_color_mode(ColorMode::ByPhysics);
        renderer.set_optical_params(BLOB_ID as usize, SIGMA_TISSUE);
        renderer.set_optical_params(WATER_ID as usize, SIGMA_WATER);

        println!("render_physics [GPU]: {} particles", sim.particle_count());
        println!(
            "  [G] toggle Beer-Lambert/material  [1-4] color modes  LMB push  RMB pull  [Q] quit"
        );

        Self {
            surface,
            surface_config,
            device,
            queue,
            sim,
            renderer,
            cursor_pos: [0.0; 2],
            lmb: false,
            rmb: false,
            physics_colors: true,
            frame: 0,
            fps_timer: std::time::Instant::now(),
            fps_frames: 0,
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

    fn cursor_grid(&self) -> Vec2 {
        Vec2::new(
            self.cursor_pos[0] / self.surface_config.width as f32 * GRID as f32,
            (1.0 - self.cursor_pos[1] / self.surface_config.height as f32) * GRID as f32,
        )
    }

    fn update_and_render(&mut self) {
        if self.lmb || self.rmb {
            let mag = if self.lmb { 3.0 } else { -3.0 };
            self.sim.apply_radial_impulse(self.cursor_grid(), 6.0, mag);
        }

        // Acquire surface before stepping physics: if surface unavailable, skip the
        // entire frame so physics and visuals stay in sync (no silent jump-ahead).
        let output = match self.surface.get_current_texture() {
            Ok(t) => t,
            Err(_) => return,
        };

        self.sim.step_frame();
        self.frame += 1;
        self.fps_frames += 1;

        if self.fps_timer.elapsed().as_secs_f32() >= 2.0 {
            let fps = self.fps_frames as f32 / self.fps_timer.elapsed().as_secs_f32();
            println!("frame={} fps={:.0}", self.frame, fps);
            self.fps_timer = std::time::Instant::now();
            self.fps_frames = 0;
        }
        if self.frame % 60 == 0 {
            log_frame_gpu(self.frame, DT, self.sim.particles(), LABELS, 1);
            let snap = self.sim.diagnostics_snapshot();
            println!(
                "  non_finite={}  out_of_bounds={}  max_speed={:.3}  sub={}  cfl={:.4}",
                snap.non_finite_particle_values,
                snap.out_of_bounds_particles,
                snap.max_particle_speed,
                snap.substeps_last_step,
                snap.cfl_number,
            );
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
        output.present();
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(
                    winit::window::WindowAttributes::default()
                        .with_title("emerge -- Beer-Lambert rendering [GPU]")
                        .with_inner_size(winit::dpi::LogicalSize::new(480u32, 480u32)),
                )
                .unwrap(),
        );
        self.state = Some(pollster::block_on(State::new(window.clone())));
        self.window = Some(window);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(s) = self.state.as_mut() else { return };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::CursorMoved { position, .. } => {
                s.cursor_pos = [position.x as f32, position.y as f32];
            }
            WindowEvent::MouseInput { state, button, .. } => match button {
                MouseButton::Left => s.lmb = state == ElementState::Pressed,
                MouseButton::Right => s.rmb = state == ElementState::Pressed,
                _ => {}
            },
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(key),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => match key {
                KeyCode::Escape | KeyCode::KeyQ => event_loop.exit(),
                KeyCode::KeyG => {
                    s.physics_colors = !s.physics_colors;
                    s.renderer.set_color_mode(if s.physics_colors {
                        ColorMode::ByPhysics
                    } else {
                        ColorMode::ByMaterial
                    });
                }
                KeyCode::Digit1 => s.renderer.set_color_mode(ColorMode::ByPhysics),
                KeyCode::Digit2 => s.renderer.set_color_mode(ColorMode::ByVelocity),
                KeyCode::Digit3 => s.renderer.set_color_mode(ColorMode::ByVolume),
                KeyCode::Digit4 => s.renderer.set_color_mode(ColorMode::ByMaterial),
                _ => {}
            },
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
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App {
        window: None,
        state: None,
    };
    event_loop.run_app(&mut app).unwrap();
}

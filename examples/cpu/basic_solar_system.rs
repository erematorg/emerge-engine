extern crate emerge_engine as emerge;

use egui_wgpu::ScreenDescriptor;
use emerge::fields::NBodyGravityField;
use emerge::render::{ColorMode, Renderer};
use emerge::{NeoHookeanMaterial, SimConfig, Simulation, SpawnRegion};
/// TRUE full N-body solar system, live: Sun + all 8 real planets, REAL MUTUAL
/// gravity (every body pulls on every other, Sun included and free to move) --
/// the real structural upgrade from `basic_orbital.rs`'s restricted
/// two-body model (fixed Sun, Earth+Mars only). Uses the engine's existing
/// `NBodyGravityField` (Barnes-Hut + a real quadrupole correction, Hernquist
/// 1987) -- the same real technique proven headless in
/// `tests/orbital_mechanics.rs::full_solar_system_conserves_momentum_and_energy`
/// (momentum drift 0.0020%, energy drift 0.0033% over 30 real days).
///
/// Real technique grounding (WebSearch, 2026-08-11): symplectic integrators
/// (Leapfrog, Wisdom-Holman/WHFast -- REBOUND, the real standard N-body
/// astronomy code) are the established real technique for long-term
/// solar-system stability. emerge's own MPM position/velocity update is
/// already semi-implicit/symplectic-Euler by construction -- the same real
/// structural property, not a new addition.
///
/// Real, disclosed limitations:
///   - Real LINEAR distance scale (not logarithmic) -- Mercury sits ~78x
///     closer than Neptune, so inner planets cluster tightly near the Sun.
///     Every real solar-system diagram is "not to scale" for exactly this
///     reason; this one IS to scale, which is why it looks this way.
///   - Real, found-not-hidden precision limit: at this domain's scale
///     (needed to fit Neptune's real orbit), the Sun's own tiny wobble
///     velocity produces a per-step position increment below f32's local
///     precision -- confirmed in `tests/orbital_mechanics.rs`'s own doc
///     (`sun_velocity_responds_to_real_mutual_gravity`): velocity responds
///     correctly to real gravity, position does not visibly accumulate the
///     wobble at this scale/timeframe. A real, structural float-precision
///     constraint, not a physics bug.
///   - Real orbital phase is arbitrary (planets spread at even angles, not
///     a real ephemeris snapshot) -- real distances/masses/speeds throughout.
///
///   cargo run --example basic_solar_system --features render
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

/// dx sized so Neptune's real orbit fits with margin -- same real, measured
/// choice as `tests/orbital_mechanics.rs`'s full-system section.
const DX_METERS: f64 = 2.0e9;
const GRID: usize = 4096;
const DT_SECONDS: f64 = 3600.0;

const G_SI: f64 = 6.674e-11;
const SUN_MASS_KG: f64 = 1.9885e30;

/// Real NASA NSSDCA Planetary Fact Sheet data: (name, mass_kg, semi_major_axis_m).
const PLANETS: [(&str, f64, f64); 8] = [
    ("Mercury", 0.330e24, 57.9e9),
    ("Venus", 4.87e24, 108.2e9),
    ("Earth", 5.97e24, 149.6e9),
    ("Mars", 0.642e24, 228.0e9),
    ("Jupiter", 1898.0e24, 778.5e9),
    ("Saturn", 568.0e24, 1432.0e9),
    ("Uranus", 86.8e24, 2867.0e9),
    ("Neptune", 102.0e24, 4515.0e9),
];

fn make_sim() -> Simulation {
    let center = glam::Vec2::splat(GRID as f32 / 2.0);
    let config = SimConfig {
        dx_meters: DX_METERS as f32,
        dt_seconds: DT_SECONDS as f32,
        gravity: glam::Vec2::ZERO,
        ..SimConfig::standard(GRID, DT_SECONDS as f32, glam::Vec2::ZERO)
    };
    let g_grid = (G_SI / DX_METERS.powi(3)) as f32;

    let spawn_sun = SpawnRegion {
        spacing: 1.0,
        box_size: glam::IVec2::new(1, 1),
        box_center: center,
        position_jitter: 0.0,
        material_id: 0,
        mass_override: Some(SUN_MASS_KG as f32),
        ..SpawnRegion::for_sim(&config)
    };
    let mut solver = Simulation::new(config, spawn_sun)
        .with_default_material(Box::new(NeoHookeanMaterial::new(1.0, 1.0)))
        .with_force_field(Box::new(NBodyGravityField::new(g_grid, 0.05, 0.1)));
    for mat_id in 1..=PLANETS.len() as u32 {
        solver = solver.with_material(mat_id, Box::new(NeoHookeanMaterial::new(1.0, 1.0)));
    }

    let mut planet_momentum = glam::Vec2::ZERO;
    for (idx, &(_, mass_kg, a_m)) in PLANETS.iter().enumerate() {
        let angle = idx as f32 * std::f32::consts::TAU / PLANETS.len() as f32;
        let (s, c) = angle.sin_cos();
        let r_grid = (a_m / DX_METERS) as f32;
        let v_mag = ((G_SI * SUN_MASS_KG / a_m).sqrt() / DX_METERS) as f32;
        let pos = center + glam::Vec2::new(c, s) * r_grid;
        let vel = glam::Vec2::new(-s, c) * v_mag;

        let spawn = SpawnRegion {
            spacing: 1.0,
            box_size: glam::IVec2::new(1, 1),
            box_center: pos,
            position_jitter: 0.0,
            material_id: idx as u32 + 1,
            mass_override: Some(mass_kg as f32),
            ..SpawnRegion::for_sim(solver.config())
        };
        let _ = solver.add_body(spawn);
        solver.particles_mut().v[idx + 1] = vel;
        planet_momentum += mass_kg as f32 * vel;
    }
    // Barycentric frame: Sun's velocity exactly cancels total planet momentum
    // (real, standard N-body initial-condition technique).
    solver.particles_mut().v[0] = -planet_momentum / SUN_MASS_KG as f32;

    solver
}

struct State {
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    device: wgpu::Device,
    queue: wgpu::Queue,
    sim: Simulation,
    renderer: Renderer,
    egui_ctx: egui::Context,
    egui_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
    days_elapsed: f32,
    steps_per_frame: u32,
    paused: bool,
    zoom: f32,
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
            present_mode: wgpu::PresentMode::AutoVsync,
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &sc);
        let sim = make_sim();
        let mut renderer = Renderer::new(&device, sim.particles().len(), fmt);
        let zoom = 1.0;
        renderer.set_camera(
            &queue,
            (GRID as f32 / zoom) as u32,
            size.width,
            size.height,
            6.0,
            true,
        );
        renderer.set_color_mode(ColorMode::ByMaterial);

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
            "solar system: Sun + 8 real planets, TRUE mutual N-body gravity  |  R reset  Q quit"
        );
        Self {
            surface,
            surface_config: sc,
            device,
            queue,
            sim,
            renderer,
            egui_ctx,
            egui_state,
            egui_renderer,
            days_elapsed: 0.0,
            steps_per_frame: 4,
            paused: false,
            zoom,
        }
    }

    fn apply_camera(&mut self, w: u32, h: u32) {
        self.renderer.set_camera(
            &self.queue,
            (GRID as f32 / self.zoom) as u32,
            w,
            h,
            6.0,
            true,
        );
    }

    fn resize(&mut self, w: u32, h: u32) {
        if w == 0 || h == 0 {
            return;
        }
        self.surface_config.width = w;
        self.surface_config.height = h;
        self.surface.configure(&self.device, &self.surface_config);
        self.apply_camera(w, h);
    }

    fn update_and_render(&mut self, window: &Window) {
        if !self.paused {
            for _ in 0..self.steps_per_frame {
                self.sim.step();
                self.days_elapsed += DT_SECONDS as f32 / 86400.0;
            }
        }

        let output = match self.surface.get_current_texture() {
            Ok(t) => t,
            Err(_) => return,
        };
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        self.renderer
            .render(&self.device, &self.queue, self.sim.particles(), &view, true);

        let raw_input = self.egui_state.take_egui_input(window);
        let mut steps_per_frame = self.steps_per_frame as f32;
        let mut paused = self.paused;
        let mut zoom = self.zoom;
        let days = self.days_elapsed;
        let mut reset = false;
        let (w, h) = (self.surface_config.width, self.surface_config.height);
        let full_output = self.egui_ctx.run(raw_input, |ctx| {
            egui::Window::new("Solar System (true N-body)")
                .default_pos([10.0, 10.0])
                .default_width(280.0)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.label(format!(
                        "day {days:.0}  ({:.2} years)  |  9 real bodies, mutual gravity",
                        days / 365.25
                    ));
                    ui.separator();
                    ui.checkbox(&mut paused, "Paused");
                    ui.label("Speed (real steps per rendered frame):");
                    ui.add(egui::Slider::new(&mut steps_per_frame, 1.0..=400.0).logarithmic(true));
                    ui.label(
                        "Zoom (real linear distance scale -- Mercury ~78x closer than Neptune):",
                    );
                    ui.add(egui::Slider::new(&mut zoom, 0.2..=20.0).logarithmic(true));
                    ui.separator();
                    ui.label("R reset  Q quit");
                    if ui.button("Reset").clicked() {
                        reset = true;
                    }
                });
        });
        self.steps_per_frame = steps_per_frame.round().max(1.0) as u32;
        self.paused = paused;
        if (zoom - self.zoom).abs() > 1.0e-4 {
            self.zoom = zoom;
            self.apply_camera(w, h);
        }
        if reset {
            self.sim = make_sim();
            self.days_elapsed = 0.0;
        }

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

struct App {
    window: Option<Arc<Window>>,
    state: Option<State>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        let w = Arc::new(
            el.create_window(
                winit::window::WindowAttributes::default()
                    .with_title("emerge -- True Solar System (N-body)")
                    .with_inner_size(winit::dpi::LogicalSize::new(720u32, 720u32)),
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
        if let Some(w) = &self.window {
            let resp = s.egui_state.on_window_event(w, &event);
            if resp.consumed {
                return;
            }
        }
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
            } => match key {
                KeyCode::Escape | KeyCode::KeyQ => el.exit(),
                KeyCode::KeyR => {
                    s.sim = make_sim();
                    s.days_elapsed = 0.0;
                    println!("reset");
                }
                _ => {}
            },
            WindowEvent::Resized(sz) => s.resize(sz.width, sz.height),
            WindowEvent::RedrawRequested => {
                if let Some(w) = &self.window {
                    let w = w.clone();
                    s.update_and_render(&w);
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

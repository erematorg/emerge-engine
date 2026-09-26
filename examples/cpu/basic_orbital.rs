extern crate emerge_engine as emerge;

use egui_wgpu::ScreenDescriptor;
use emerge::fields::GravityWellField;
use emerge::render::{ColorMode, Renderer};
use emerge::{NeoHookeanMaterial, SimConfig, Simulation, SpawnRegion};
/// `basic_orbital.rs` (Sun + Earth + Mars, real `GravityWellField` gravity)
/// with a real, live egui panel -- same pattern as `basic_fluids.rs`: a
/// speed slider (steps-per-frame, NOT `dt_seconds` -- keeps the validated
/// integration accuracy fixed regardless of playback speed) and a live real
/// day/year readout.
///
/// Real, measured accuracy tuning (2026-08-11, see `tests/orbital_mechanics.rs`
/// for the full sweep data): `DX_METERS`/`GRID` below were chosen from a real
/// grid-resolution sweep, not guessed -- Kepler's third law (T^2 ~ a^3,
/// checked headless between Earth and Mars) holds within 0.09% at this scale,
/// down from 0.36% at the original (coarser) grid. A separate `dt_seconds`
/// sweep proved accuracy does NOT depend on timestep here (spatial
/// discretization, not temporal, was the real limiting factor) -- so the
/// speed slider is free to change playback pace without touching accuracy.
///
///   cargo run --example basic_orbital --features render
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

/// 1 grid cell = 250,000 km -- the real, measured (not guessed) choice from
/// `tests/orbital_mechanics.rs`'s own grid-resolution sweep: puts Earth's
/// orbital radius at ~598 grid units, where Kepler's third law measured
/// 0.09% error (vs. 0.36% at the original, 4x coarser scale).
const DX_METERS: f64 = 2.5e8;
/// Large enough to hold Mars's real orbit (~912 grid units) with margin.
const GRID: usize = 2048;
/// 1 real hour per nominal substep -- proven (via the same test file's own
/// `diag_kepler_error_vs_dt_sweep`) NOT to be the accuracy bottleneck here.
const DT_SECONDS: f64 = 3600.0;

const MU_SUN_SI: f64 = 1.32712e20; // G*M_sun, m^3/s^2 -- NASA/JPL
const AU_M: f64 = 1.496e11;
const MARS_DISTANCE_M: f64 = 228.0e9;
const EARTH_MASS_KG: f32 = 5.97e24;
const MARS_MASS_KG: f32 = 0.642e24;

const MAT_SUN: u32 = 0;
const MAT_EARTH: u32 = 1;
const MAT_MARS: u32 = 2;

fn circular_orbit_speed_grid(r_si: f64) -> f32 {
    ((MU_SUN_SI / r_si).sqrt() / DX_METERS) as f32
}

fn make_sim() -> Simulation {
    let sun_pos = glam::Vec2::splat(GRID as f32 / 2.0);
    let config = SimConfig {
        dx_meters: DX_METERS as f32,
        dt_seconds: DT_SECONDS as f32,
        gravity: glam::Vec2::ZERO,
        ..SimConfig::standard(GRID, DT_SECONDS as f32, glam::Vec2::ZERO)
    };

    let r_earth = (AU_M / DX_METERS) as f32;
    let r_mars = (MARS_DISTANCE_M / DX_METERS) as f32;
    let v_earth = circular_orbit_speed_grid(AU_M);
    let v_mars = circular_orbit_speed_grid(MARS_DISTANCE_M);

    let spawn_sun = SpawnRegion {
        spacing: 1.0,
        box_size: glam::IVec2::new(1, 1),
        box_center: sun_pos,
        position_jitter: 0.0,
        material_id: MAT_SUN,
        mass_override: Some((MU_SUN_SI / 6.674e-11) as f32), // real Sun mass, kg
        ..SpawnRegion::for_sim(&config)
    };
    let spawn_earth = SpawnRegion {
        spacing: 1.0,
        box_size: glam::IVec2::new(1, 1),
        box_center: sun_pos + glam::Vec2::new(r_earth, 0.0),
        position_jitter: 0.0,
        material_id: MAT_EARTH,
        mass_override: Some(EARTH_MASS_KG),
        ..SpawnRegion::for_sim(&config)
    };
    let spawn_mars = SpawnRegion {
        spacing: 1.0,
        box_size: glam::IVec2::new(1, 1),
        box_center: sun_pos + glam::Vec2::new(0.0, r_mars),
        position_jitter: 0.0,
        material_id: MAT_MARS,
        mass_override: Some(MARS_MASS_KG),
        ..SpawnRegion::for_sim(&config)
    };

    let mu_grid = (MU_SUN_SI / (DX_METERS * DX_METERS * DX_METERS)) as f32;
    let sun_well = GravityWellField::point(sun_pos, mu_grid, 1.0, 0.05);

    let mut solver = Simulation::new(config, spawn_sun)
        .with_default_material(Box::new(NeoHookeanMaterial::new(1.0, 1.0)))
        .with_material(MAT_EARTH, Box::new(NeoHookeanMaterial::new(1.0, 1.0)))
        .with_material(MAT_MARS, Box::new(NeoHookeanMaterial::new(1.0, 1.0)))
        .with_force_field(Box::new(sun_well));

    // Sun is fixed (restricted two-body problem, see module doc) -- pin it
    // so it stays put and visible without feeling its own gravity well.
    // Real, permanent regression proof this actually holds:
    // `tests/orbital_mechanics.rs::pinned_sun_stays_exactly_fixed_while_earth_orbits`.
    solver.particles_mut().pinned[0] = 1;

    let _ = solver.add_body(spawn_earth);
    solver.particles_mut().v[1] = glam::Vec2::new(0.0, v_earth);

    let _ = solver.add_body(spawn_mars);
    // Mars starts 90 degrees around from Earth (spawned along +y instead of
    // +x above) -- real tangential velocity for THAT position is along -x.
    solver.particles_mut().v[2] = glam::Vec2::new(-v_mars, 0.0);

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
    /// Real steps-per-frame, NOT a `dt_seconds` change -- keeps the
    /// validated 0.09% Kepler-law accuracy fixed regardless of playback
    /// speed (accuracy was proven dt-independent in this scene, but
    /// changing dt would still be a real, separate physics change; a
    /// steps-per-frame multiplier is purely a playback-speed control).
    steps_per_frame: u32,
    paused: bool,
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
        // particle_scale=6.0 -- purely visual (real relative sizes are
        // un-renderable at this distance scale, see module doc).
        renderer.set_camera(&queue, GRID as u32, size.width, size.height, 6.0, true);
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

        println!("orbital: Sun + Earth + Mars, real NASA masses/distances  |  R reset  Q quit");
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
            steps_per_frame: 1,
            paused: false,
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
            .set_camera(&self.queue, GRID as u32, w, h, 6.0, true);
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

        // --- egui panel ---
        let raw_input = self.egui_state.take_egui_input(window);
        let mut steps_per_frame = self.steps_per_frame as f32;
        let mut paused = self.paused;
        let days = self.days_elapsed;
        let mut reset = false;
        let full_output = self.egui_ctx.run(raw_input, |ctx| {
            egui::Window::new("Orbital")
                .default_pos([10.0, 10.0])
                .default_width(260.0)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.label(format!(
                        "day {days:.0}  ({:.2} years)  |  Earth year: 365.25d  Mars: 687d",
                        days / 365.25
                    ));
                    ui.separator();
                    ui.checkbox(&mut paused, "Paused");
                    ui.label("Speed (real steps per rendered frame):");
                    ui.add(egui::Slider::new(&mut steps_per_frame, 1.0..=200.0).logarithmic(true));
                    ui.separator();
                    ui.label("R reset  Q quit");
                    if ui.button("Reset").clicked() {
                        reset = true;
                    }
                });
        });
        self.steps_per_frame = steps_per_frame.round() as u32;
        self.paused = paused;
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
                    .with_title("emerge -- Orbital [Sun / Earth / Mars] (GUI)")
                    .with_inner_size(winit::dpi::LogicalSize::new(640u32, 640u32)),
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

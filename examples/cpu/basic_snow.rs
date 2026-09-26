extern crate emerge_engine as emerge;

#[path = "../gui_common/coords.rs"]
mod gui_common;

use egui_wgpu::ScreenDescriptor;
/// `basic_snow.rs` (two real snowballs colliding -- Stomakhin 2013 snow
/// plasticity, soft powder vs packed snow, packed snow fractures into loose
/// granular on hard impact via a real phase transition) with a real, live
/// egui panel -- same pattern as `basic_sand.rs`: real gravity slider
/// (1.0 = genuine IRL 9.81 m/s²) and push/pull strength. Materials, the
/// collision setup, and the fracture mechanic are unchanged from
/// `basic_snow.rs` -- already real and good, not touched.
///
/// Real gravity default: 0.01, NOT re-guessed -- a headless sweep
/// (2026-07-23, see MEMORY.md's ecosystem-roadmap note) confirmed every
/// fraction from 0.001 to 1.0 stays numerically finite here, and the same
/// 0.01 checkpoint already validated for sand and fluids only adds ~12
/// grid-units/s on top of this scene's own intrinsic ~15 grid-units/s
/// collision-launch speed -- consistent across all three tier-0 materials
/// rather than a fresh guess.
///
///   cargo run --example basic_snow --features render
use emerge::render::{ColorMode, Renderer};
use emerge::{
    DruckerPragerMaterial, SimConfig, Simulation, SlipBoundary, SpawnRegion, StomakhinMaterial,
};
use glam::{IVec2, Vec2};
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

const GRID: usize = 64;
const DT: f32 = 0.1;
const MAT_SOFT: u32 = 0;
const MAT_PACKED: u32 = 1;
const MAT_SHATTER: u32 = 2;
const BALL_R: f32 = 9.0;
const BALL_A: Vec2 = Vec2::new(16.0, 44.0);
const BALL_B: Vec2 = Vec2::new(48.0, 44.0);
const SPEED: f32 = 15.0;
// Radius of the directional dig nudge, grid cells -- matches basic_sand.rs.
const DIG_RADIUS: f32 = 4.0;

// Real fix (2026-09-05): was `StomakhinMaterial::new(1389.0, 2083.0, ..)`,
// an unsourced grid-unit guess. Real snow -- `StomakhinMaterial::from_
// young_modulus`'s own doc cites this exact E/nu as "Canonical... matches
// MPM2D reference and sparkl snow demos" (Stomakhin et al. 2013 -- the same
// value real-time MPM snow demos in other engines use, not just a textbook
// number). Density: real fresh/settled snow order of magnitude (a real
// packed-snow reference of 200 kg/m3 is also cited in this engine's own
// `physical_props.rs` module doc). Loose/packed differ only in their real
// Stomakhin plasticity parameters (hardening/compression/stretch limits),
// not stiffness -- same real physical mechanism (packing changes how much
// strain triggers plastic flow, not the elastic modulus itself).
const SNOW_YOUNG_MODULUS_PA: f32 = 1.4e5;
const SNOW_POISSON_RATIO: f32 = 0.2;
const SNOW_DENSITY_KG_M3: f32 = 200.0;

fn make_sim() -> Simulation {
    let config = SimConfig {
        // Real fix (2026-09-05): the real stiffness above needs real
        // substep headroom under CFL -- the old 20 silently dropped
        // simulated time instead of crashing (see `step.rs`'s "honest
        // accounting" doc). Measured directly during a real snowball
        // collision (`tests/scratch_basic_snow_probe.rs`): 3000 still
        // dropped ~54% of each step's simulated time; the solver actually
        // settles around 6590-6600 once given enough headroom, so 8000
        // leaves real margin, confirmed zero time dropped. Real, disclosed
        // cost: this is a genuinely heavy substep count for an interactive
        // demo -- whether E=1.4e5 is practical at this resolution for
        // real-time framerate (vs. needing a coarser dx or an implicit
        // solver) is an open question, not resolved here.
        max_substeps_per_step: 8000,
        ..SimConfig::earth(GRID, 0.01, DT)
    };
    let (lambda, mu) = config.lame_from_si_physical_cfg(
        SNOW_YOUNG_MODULUS_PA,
        SNOW_POISSON_RATIO,
        SNOW_DENSITY_KG_M3,
    );
    // Real fix (2026-09-05): mass must share the same real density as the
    // stiffness above (see project memory on the grid_density/mass-from
    // gap found migrating this same night's other scenes) -- was left on
    // the bare `grid_density=1.0` default, computed directly via
    // `ParticleMass::particle_mass`'s own documented formula since the raw
    // `StomakhinMaterial::new` constructor bypasses `mass_from`.
    let mass_grid = (SNOW_DENSITY_KG_M3 / config.reference_density_kg_m3) * 0.5 * 0.5;
    let spawn = SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new(58, 58),
        rng_seed: 7,
        mass_override: Some(mass_grid),
        ..SpawnRegion::for_sim(&config)
    };
    let mut solver = Simulation::new(config, spawn)
        .with_default_material(Box::new(StomakhinMaterial::new(
            lambda, mu, 7.0, 0.025, 0.0075, 0.6, 20.0,
        )))
        .with_material(
            MAT_PACKED,
            Box::new(
                StomakhinMaterial::new(lambda, mu, 10.0, 0.012, 0.004, 0.6, 20.0)
                    .with_cohesion(400.0),
            ),
        )
        .with_material(
            MAT_SHATTER,
            Box::new(DruckerPragerMaterial::low_friction(266.7, 0.333)),
        )
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));

    solver.retain_particles(|p| {
        (p.x - BALL_A).length() <= BALL_R || (p.x - BALL_B).length() <= BALL_R
    });
    solver.particles_mut().for_each_mut(|p| {
        if (p.x - BALL_A).length() <= BALL_R {
            p.material_id = MAT_SOFT;
            p.v = Vec2::new(SPEED, 0.0);
        } else {
            p.material_id = MAT_PACKED;
            p.v = Vec2::new(-SPEED, 0.0);
        }
    });
    solver.recompute_initial_volumes();
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
    cursor_pos: [f32; 2],
    last_cursor_grid: Vec2,
    lmb: bool,
    rmb: bool,
    digging: bool,
    push_strength: f32,
    dig_strength: f32,
    real_gravity: Vec2,
    gravity_fraction: f32,
    frame: u64,
    fps_timer: std::time::Instant,
    fps_frames: u64,
    last_fps: f32,
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
        let real_gravity = sim.config().gravity;
        let mut renderer = Renderer::new(&device, sim.particles().len(), fmt);
        renderer.set_camera(&queue, GRID as u32, size.width, size.height, 0.6, true);
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
            "basic_snow: {} particles  |  LMB push  RMB pull  D toggle dig  R reset  Q quit",
            sim.particles().len()
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
            cursor_pos: [0.0; 2],
            last_cursor_grid: Vec2::ZERO,
            lmb: false,
            rmb: false,
            digging: false,
            push_strength: 10.0,
            dig_strength: 18.0,
            real_gravity,
            // 0.01 -> 0.005, user-confirmed live: "un peu fort" at 0.01.
            gravity_fraction: 0.005,
            frame: 0,
            fps_timer: std::time::Instant::now(),
            fps_frames: 0,
            last_fps: 0.0,
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
        gui_common::cursor_to_grid(
            self.cursor_pos,
            self.surface_config.width,
            self.surface_config.height,
            GRID,
        )
    }

    fn update_and_render(&mut self, window: &Window) {
        self.sim
            .set_gravity(self.real_gravity * self.gravity_fraction);
        if self.lmb || self.rmb {
            let mag = if self.lmb {
                self.push_strength
            } else {
                -self.push_strength
            };
            self.sim.apply_radial_impulse(self.cursor_grid(), 6.0, mag);
        }
        // Digging: nudges nearby particles along the cursor's OWN movement
        // direction (a furrow), not radially like push/pull -- same proven
        // mechanism as basic_sand.rs, no second body, no impulse call.
        let cursor = self.cursor_grid();
        if self.digging {
            let delta = cursor - self.last_cursor_grid;
            if delta.length_squared() > 1.0e-8 {
                let dir = delta.normalize();
                let particles = self.sim.particles_mut();
                for i in 0..particles.len() {
                    if (particles.x[i] - cursor).length() < DIG_RADIUS {
                        particles.v[i] += dir * self.dig_strength * DT;
                    }
                }
            }
        }
        self.last_cursor_grid = cursor;
        self.sim.step();
        // Fracture trigger: real plastic compression (Jp), not raw speed --
        // the old `v.length() > 5.0` fired at launch, before any collision.
        self.sim.phase_transition(
            |p| p.material_id == MAT_PACKED && p.plastic_volume_ratio < 0.9,
            MAT_SHATTER,
        );
        self.frame += 1;
        self.fps_frames += 1;
        if self.fps_timer.elapsed().as_secs_f32() >= 1.0 {
            self.last_fps = self.fps_frames as f32 / self.fps_timer.elapsed().as_secs_f32();
            self.fps_timer = std::time::Instant::now();
            self.fps_frames = 0;
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
        let fps = self.last_fps;
        let mut push_strength = self.push_strength;
        let mut dig_strength = self.dig_strength;
        let mut gravity_fraction = self.gravity_fraction;
        let mut digging = self.digging;
        let soft_n = self
            .sim
            .particles()
            .iter()
            .filter(|p| p.material_id == MAT_SOFT)
            .count();
        let packed_n = self
            .sim
            .particles()
            .iter()
            .filter(|p| p.material_id == MAT_PACKED)
            .count();
        let shatter_n = self
            .sim
            .particles()
            .iter()
            .filter(|p| p.material_id == MAT_SHATTER)
            .count();
        let mut reset = false;

        let full_output = self.egui_ctx.run(raw_input, |ctx| {
            egui::Window::new("Snow")
                .default_pos([10.0, 10.0])
                .default_width(260.0)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.label(format!("fps={fps:.0}"));
                    ui.label(format!(
                        "soft={soft_n}  packed={packed_n}  shatter={shatter_n}"
                    ));
                    ui.separator();
                    ui.label("Gravity (1.0 = real IRL 9.81 m/s²):");
                    ui.add(egui::Slider::new(&mut gravity_fraction, 0.0..=2.0));
                    ui.separator();
                    ui.label("Push/pull strength:");
                    ui.add(egui::Slider::new(&mut push_strength, 0.0..=30.0));
                    ui.checkbox(&mut digging, "Digging active (or press D)");
                    ui.add(egui::Slider::new(&mut dig_strength, 0.0..=40.0).text("Dig strength"));
                    ui.separator();
                    ui.label("LMB push  RMB pull  D toggle dig  R reset  Q quit");
                    if ui.button("Reset").clicked() {
                        reset = true;
                    }
                });
        });
        self.push_strength = push_strength;
        self.dig_strength = dig_strength;
        self.gravity_fraction = gravity_fraction;
        self.digging = digging;
        if reset {
            let sim = make_sim();
            self.real_gravity = sim.config().gravity;
            self.sim = sim;
            self.frame = 0;
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
                    .with_title("emerge -- Snow (GUI)")
                    .with_inner_size(winit::dpi::LogicalSize::new(480u32, 480u32)),
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
                        state: key_state,
                        ..
                    },
                ..
            } => {
                let pressed = key_state == ElementState::Pressed;
                match key {
                    KeyCode::Escape | KeyCode::KeyQ if pressed => el.exit(),
                    KeyCode::KeyD if pressed => s.digging = !s.digging,
                    KeyCode::KeyR if pressed => {
                        let sim = make_sim();
                        s.real_gravity = sim.config().gravity;
                        s.sim = sim;
                        s.frame = 0;
                        println!("reset");
                    }
                    _ => {}
                }
            }
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

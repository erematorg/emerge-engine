extern crate emerge_engine as emerge;

#[path = "../gui_common/coords.rs"]
mod gui_common;
#[path = "../gui_common/render_mode.rs"]
mod render_mode;

use egui_wgpu::ScreenDescriptor;
/// Two snowballs colliding (Stomakhin 2013 snow plasticity; soft powder vs packed snow,
/// packed snow fracturing into loose granular on a hard impact through a phase
/// transition) with a live egui panel, the pattern of `basic_sand.rs`: gravity slider
/// (1.0 = Earth's 9.81 m/s²) and push/pull strength.
///
/// Default gravity fraction 0.01: a headless sweep keeps every fraction from 0.001 to
/// 1.0 finite, and 0.01 (the checkpoint used for sand and fluids) adds only ~12
/// grid-units/s on top of the scene's ~15 grid-units/s collision-launch speed.
///
/// G cycles the view: particles, the grid-volume view, the curvature-flow
/// surface. Those two views colour the snow from measured optics
/// (`optical::snow`: Henley et al. 2024's snow model on Warren & Brandt
/// 2008's ice absorption), which makes it white; the particle view keeps
/// `ByMaterial`'s placeholder colours.
///
///   cargo run --example basic_snow --features render
use emerge::materials::optical;
use emerge::render::{ColorMode, CpuRenderBridge, Renderer};
use emerge::{
    DruckerPragerMaterial, SimConfig, Simulation, SlipBoundary, SpawnRegion, StomakhinMaterial,
};
use glam::{IVec2, Vec2};
use render_mode::RenderMode;
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
/// Particle pitch of both snowballs, in grid cells.
const SPACING: f32 = 0.5;
/// Surface grid resolution as a multiple of the physics grid.
const SURFACE_RES_MULTIPLIER: u32 = 4;

// Snow: `StomakhinMaterial::from_young_modulus`'s doc cites this E/nu as "Canonical...
// matches MPM2D reference and sparkl snow demos" (Stomakhin et al. 2013). Density: fresh
// or settled snow, order of magnitude (`physical_props.rs`'s module doc also cites a
// packed-snow reference of 200 kg/m3). Loose and packed differ only in their Stomakhin
// plasticity parameters (hardening, compression and stretch limits), not stiffness:
// packing changes how much strain triggers plastic flow, not the elastic modulus.
const SNOW_YOUNG_MODULUS_PA: f32 = 1.4e5;
const SNOW_POISSON_RATIO: f32 = 0.2;
const SNOW_DENSITY_KG_M3: f32 = 200.0;
// Optical grain radius for the snow's optics (`optical::snow`): 242.5
// micrometres, the ground-truth radius Henley et al. measured on one real
// natural snow sample (arXiv:2310.20068v2, p. 21; their samples ran from
// fine fresh powder to coarse refrozen snow, p. 21). A real measured grain,
// not this demo's own snow, which no one measured.
const SNOW_GRAIN_RADIUS_M: f32 = 242.5e-6;

fn make_sim() -> Simulation {
    let config = SimConfig {
        // Substep headroom for E=1.4e5: a budget the CFL scan runs into drops
        // simulated time (see `step.rs`). During a snowball collision
        // (`tests/probes/basic_snow_probe.rs`) 3000 still dropped ~54% of each
        // step's time; the solver settles around 6590-6600 given room, so 8000
        // leaves margin with zero time dropped. A heavy count for an interactive
        // demo: whether E=1.4e5 is practical in real time at this resolution (or
        // needs a coarser dx or an implicit solver) is open.
        max_substeps_per_step: 8000,
        ..SimConfig::earth(GRID, 0.01, DT)
    };
    let (lambda, mu) = config.lame_from_si(
        SNOW_YOUNG_MODULUS_PA,
        SNOW_POISSON_RATIO,
        SNOW_DENSITY_KG_M3,
    );
    // Mass from the same density as the stiffness above, not the `grid_density=1.0`
    // default, through `ParticleMass::particle_mass`'s formula, since the raw
    // `StomakhinMaterial::new` constructor bypasses `mass_from`.
    let mass_grid = (SNOW_DENSITY_KG_M3 / config.reference_density_kg_m3) * SPACING * SPACING;
    let spawn = SpawnRegion {
        spacing: SPACING,
        box_size: IVec2::new(58, 58),
        rng_seed: 7,
        mass_override: Some(mass_grid),
        ..SpawnRegion::for_sim(&config)
    };
    let mut solver = Simulation::new(config, spawn)
        .with_default_material(Box::new(StomakhinMaterial {
            optics: Some(optical::snow(
                SNOW_GRAIN_RADIUS_M,
                SNOW_DENSITY_KG_M3 / optical::ICE_DENSITY_KG_M3,
            )),
            ..StomakhinMaterial::new(lambda, mu, 7.0, 0.025, 0.0075, 0.6, 20.0)
        }))
        .with_material(
            MAT_PACKED,
            Box::new(StomakhinMaterial {
                optics: Some(optical::snow(
                    SNOW_GRAIN_RADIUS_M,
                    SNOW_DENSITY_KG_M3 / optical::ICE_DENSITY_KG_M3,
                )),
                ..StomakhinMaterial::new(lambda, mu, 10.0, 0.012, 0.004, 0.6, 20.0)
                    .with_cohesion(400.0)
            }),
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
    /// Which render path draws the frame, cycled with G.
    render_mode: RenderMode,
    /// GPU buffers the grid-volume and surface modes read, rebuilt from the
    /// CPU solver on the frames those modes are shown.
    render_bridge: CpuRenderBridge,
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
        // Marks the materials that hold their shape (a nonzero shear
        // modulus), which the grid-volume and surface modes draw flat; also
        // picks up any measured optics a material declares.
        renderer.adopt_material_optics(&queue, sim.materials());
        // The fragments are the same snow, broken up; their granular model
        // carries no optics of its own, so they get the snow's here.
        let fragment_optics = optical::snow(
            SNOW_GRAIN_RADIUS_M,
            SNOW_DENSITY_KG_M3 / optical::ICE_DENSITY_KG_M3,
        );
        renderer.set_optical_params(
            &queue,
            MAT_SHATTER as usize,
            fragment_optics.absorption_m_inv,
        );
        renderer.set_optical_scattering(
            &queue,
            MAT_SHATTER as usize,
            fragment_optics.reduced_scattering_m_inv,
        );
        // The grid-volume and surface modes threshold on cell mass as a
        // fraction of a full cell of snow, which holds 1/SPACING^2 particles.
        renderer.set_grid_reference_cell_mass(sim.particles().mass[0] / (SPACING * SPACING));
        renderer.set_surface_res_multiplier(SURFACE_RES_MULTIPLIER);
        let render_bridge = CpuRenderBridge::new(&device, GRID);

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
            "basic_snow: {} particles  |  LMB push  RMB pull  D toggle dig  G render mode  R reset  Q quit",
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
            render_mode: RenderMode::Particles,
            render_bridge,
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
        match self.render_mode {
            RenderMode::Particles => {
                self.renderer
                    .render(&self.device, &self.queue, self.sim.particles(), &view, true)
            }
            RenderMode::GridVolume => {
                self.render_bridge
                    .upload_grid(&self.queue, self.sim.particles(), self.sim.grid());
                self.renderer.render_grid_volume(
                    &self.device,
                    &self.queue,
                    self.render_bridge.grid_volume_source(),
                    &view,
                    true,
                );
            }
            RenderMode::Surface => {
                self.render_bridge.upload_particles(
                    &self.device,
                    &self.queue,
                    self.sim.particles(),
                );
                // Per-material colouring: the two snows and the fragments
                // keep their own optics slot where the surfaces meet.
                self.renderer.render_surface_reconstruction(
                    &self.device,
                    &self.queue,
                    self.render_bridge
                        .surface_source(MAT_SOFT, true, self.sim.mean_substep_dt()),
                    &view,
                    true,
                );
            }
        }

        // --- egui panel ---
        let raw_input = self.egui_state.take_egui_input(window);
        let fps = self.last_fps;
        let render_mode = self.render_mode;
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
                    ui.label(format!("render: {} (G to cycle)", render_mode.label()));
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
                    ui.label("LMB push  RMB pull  D dig  G render  R reset  Q quit");
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
                    KeyCode::KeyG if pressed => {
                        s.render_mode = s.render_mode.next();
                        println!("render mode: {}", s.render_mode.label());
                    }
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

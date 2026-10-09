extern crate emerge_engine as emerge;

#[path = "../gui_common/coords.rs"]
mod gui_common;
#[path = "../gui_common/render_mode.rs"]
mod render_mode;

use emerge::materials::optical;
use emerge::render::{ColorMode, CpuRenderBridge, Renderer};
use emerge::{
    DruckerPragerMaterial, NeoHookeanMaterial, NewtonianFluidMaterial, SimConfig, Simulation,
    SlipBoundary, SpawnRegion,
};
use glam::{IVec2, Vec2};
use render_mode::RenderMode;
/// CPU three-material showcase -- sand terrain, fluid pool, elastic blob.
///
///   Mat 0  NeoHookean elastic (blue)  -- creature body, arrow-key drive
///   Mat 1  Sand Drucker-Prager (gold) -- terrain
///   Mat 2  Newtonian fluid  (cyan)    -- water pool
///
/// G cycles the view: particles, the grid-volume view, the curvature-flow
/// surface. In those two views the water is coloured from its measured
/// absorption (`optical::pure_water`, Pope & Fry 1997); the sand and the
/// elastic blob declare no optics and show grey, drawn flat because they
/// hold their shape.
///
///   ^v<>  drive elastic blob  |  LMB push  RMB pull  |  G render  R reset  Q quit
///   cargo run --example basic_showcase --features "render"
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

const GRID: usize = 64;
const DT: f32 = 0.1;
const ELASTIC_ID: u32 = 0;
const SAND_ID: u32 = 1;
const FLUID_ID: u32 = 2;
const SPACING: f32 = 0.7;
/// Surface grid resolution as a multiple of the physics grid.
const SURFACE_RES_MULTIPLIER: u32 = 4;
/// The water's rest density in grid units (`rho * dx^2` for water at
/// `dx = 0.01`, see `make_sim`), which is also its mass per full cell.
const WATER_REST_DENSITY_GRID: f32 = 0.1;

// Dry sand (Haeri & Skonieczny 2022 Table 1, Excavation case: E=15 MPa, nu=0.3,
// rho=1600 kg/m3, as `basic_sand.rs`/`sand_ngf_collapse.rs`) through `lame_from_si`.
const SAND_YOUNG_MODULUS_PA: f32 = 15.0e6;
const SAND_POISSON_RATIO: f32 = 0.3;
const SAND_DENSITY_KG_M3: f32 = 1600.0;

struct App {
    window: Option<Arc<Window>>,
    state: Option<State>,
}

struct State {
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    device: wgpu::Device,
    queue: wgpu::Queue,
    sim: Simulation,
    renderer: Renderer,
    cursor_pos: [f32; 2],
    lmb: bool,
    rmb: bool,
    arrow_up: bool,
    arrow_down: bool,
    arrow_left: bool,
    arrow_right: bool,
    frame: u64,
    fps_timer: std::time::Instant,
    fps_frames: u64,
    /// Which render path draws the frame, cycled with G.
    render_mode: RenderMode,
    /// GPU buffers the grid-volume and surface modes read, rebuilt from the
    /// CPU solver on the frames those modes are shown.
    render_bridge: CpuRenderBridge,
}

fn make_sim() -> Simulation {
    let config = SimConfig {
        min_dt: 0.005,
        // Substep headroom for E=15 MPa sand: the value measured with zero dropped
        // simulated time for the same citation, grid and dx in basic_sand.rs
        // (`tests/probes/basic_sand_probe.rs`).
        max_substeps_per_step: 3000,
        // Deliberately weak, NOT real IRL gravity (real g_grid ~= 981 via
        // SimConfig::earth) -- tuned down for a calmer, more legible demo at
        // this grid scale. Disclosed, deferred: basic_sand.rs's
        // gravity_fraction slider is the real-IRL-with-live-control
        // pattern, not yet ported to every plain example.
        gravity: Vec2::new(0.0, -0.3),
        ..SimConfig::earth(GRID, 0.01, DT)
    };
    // Grid units, unlike `sand`/`fluid` below, as `basic_creature.rs`/`grass_field.rs`:
    // this body is player-driven (arrow keys, see `update_and_render`), and an SI
    // stiffness would change its response to the same drive impulse, which needs
    // checking live (is it still controllable), not only with a headless stability
    // probe.
    let elastic = NeoHookeanMaterial::new(40.0, 80.0);
    let (sand_lambda, sand_mu) = config.lame_from_si(
        SAND_YOUNG_MODULUS_PA,
        SAND_POISSON_RATIO,
        SAND_DENSITY_KG_M3,
    );
    let sand = DruckerPragerMaterial::new(sand_lambda, sand_mu);
    // Water through `NewtonianFluidMaterial::low_viscosity` (Tait exponent 7.0, Cole
    // 1948; water viscosity). rest_density=0.1 (`rho*dx^2` for water at dx=0.01, see
    // basic_fluids.rs). eos_stiffness=0.25: with rest_density 40x smaller than 4.0,
    // `timestep_bound`'s c2 (sound speed squared) is 40x larger at a given stiffness and
    // compression, so the stiffness is scaled by the same factor (10*0.1/4.0=0.25) to
    // keep the same c2.
    let mut fluid = NewtonianFluidMaterial::low_viscosity(WATER_REST_DENSITY_GRID, 0.25);
    fluid.optics = Some(optical::pure_water());
    // Same density-consistency fix as basic_sand.rs: mass must share the
    // same real SAND_DENSITY_KG_M3 the stiffness above uses, not
    // `config.grid_density`'s unrelated bare default.
    let sand_mass = (SAND_DENSITY_KG_M3 / config.reference_density_kg_m3) * SPACING * SPACING;

    let mut solver = Simulation::empty(config)
        .with_default_material(Box::new(elastic))
        .with_material(SAND_ID, Box::new(sand))
        .with_material(FLUID_ID, Box::new(fluid))
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));

    let _ = solver.add_body(SpawnRegion {
        spacing: SPACING,
        box_size: IVec2::new(22, 14),
        box_center: Vec2::new(19.0, 9.0),
        material_id: SAND_ID,
        mass_override: Some(sand_mass),
        ..SpawnRegion::for_sim(&config)
    });
    let _ = solver.add_body(SpawnRegion {
        spacing: SPACING,
        box_size: IVec2::new(22, 14),
        box_center: Vec2::new(45.0, 9.0),
        material_id: FLUID_ID,
        // Mass set explicitly, m = rho0*spacing^2 with the material's
        // rest_density=0.1 (see basic_fluids.rs), rather than the scene's grid
        // density.
        mass_override: Some(WATER_REST_DENSITY_GRID * SPACING * SPACING),
        ..SpawnRegion::for_sim(&config)
    });
    let _ = solver.add_body(SpawnRegion {
        spacing: SPACING,
        box_size: IVec2::new(12, 12),
        box_center: Vec2::new(32.0, 46.0),
        material_id: ELASTIC_ID,
        ..SpawnRegion::for_sim(&config)
    });
    solver
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
        renderer.set_camera(&queue, GRID as u32, size.width, size.height, 0.6, true);
        renderer.set_color_mode(ColorMode::ByMaterial);
        // Marks the materials that hold their shape (a nonzero shear
        // modulus), which the grid-volume and surface modes draw flat, and
        // picks up the water's measured optics.
        renderer.adopt_material_optics(&queue, sim.materials());
        // The grid-volume and surface modes threshold on cell mass as a
        // fraction of a full cell. The water is the lightest of the three
        // bodies per cell, so measuring against it keeps all three above
        // the visibility floor.
        renderer.set_grid_reference_cell_mass(WATER_REST_DENSITY_GRID);
        renderer.set_surface_res_multiplier(SURFACE_RES_MULTIPLIER);
        let render_bridge = CpuRenderBridge::new(&device, GRID);
        println!(
            "showcase: {} particles  |  ^v<> drive blob  LMB push  RMB pull  G render  R reset  Q quit",
            sim.particles().len()
        );
        Self {
            surface,
            surface_config: sc,
            device,
            queue,
            sim,
            renderer,
            cursor_pos: [0.0; 2],
            lmb: false,
            rmb: false,
            arrow_up: false,
            arrow_down: false,
            arrow_left: false,
            arrow_right: false,
            frame: 0,
            fps_timer: std::time::Instant::now(),
            fps_frames: 0,
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

    fn update_and_render(&mut self) {
        // Arrow-key drive: find elastic centroid, apply impulse.
        let mut dir = Vec2::ZERO;
        if self.arrow_up {
            dir.y += 1.0;
        }
        if self.arrow_down {
            dir.y -= 1.0;
        }
        if self.arrow_left {
            dir.x -= 1.0;
        }
        if self.arrow_right {
            dir.x += 1.0;
        }
        if dir != Vec2::ZERO {
            let particles = self.sim.particles();
            let (sum, n) = particles
                .indices()
                .filter(|&i| particles.material_id[i] == ELASTIC_ID)
                .fold((Vec2::ZERO, 0usize), |(s, n), i| {
                    (s + particles.x[i], n + 1)
                });
            if n > 0 {
                let centroid = sum / n as f32;
                let impulse = dir.normalize() * 10.0;
                self.sim.apply_impulse(centroid, 12.0, impulse);
            }
        }

        if self.lmb || self.rmb {
            let mag = if self.lmb { 2.0 } else { -2.0 };
            self.sim.apply_radial_impulse(self.cursor_grid(), 5.0, mag);
        }

        self.sim.step();
        self.frame += 1;
        self.fps_frames += 1;
        if self.fps_timer.elapsed().as_secs_f32() >= 2.0 {
            let fps = self.fps_frames as f32 / self.fps_timer.elapsed().as_secs_f32();
            let substeps = self.sim.diagnostics_snapshot().substeps_last_step;
            // Keep the low end visible during the release FPS audit: rounding
            // to an integer turns a measured sub-1 fps result into
            // an unhelpful bare `0`.
            println!("frame={} fps={:.2} substeps={substeps}", self.frame, fps);
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
                // Per-material colouring: each body keeps its own optics slot
                // where the surfaces meet.
                self.renderer.render_surface_reconstruction(
                    &self.device,
                    &self.queue,
                    self.render_bridge
                        .surface_source(FLUID_ID, true, self.sim.mean_substep_dt()),
                    &view,
                    true,
                );
            }
        }
        output.present();
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        let w = Arc::new(
            el.create_window(
                winit::window::WindowAttributes::default()
                    .with_title("emerge -- Showcase [Sand / Fluid / Elastic]")
                    .with_inner_size(winit::dpi::LogicalSize::new(480u32, 480u32)),
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
                        state,
                        ..
                    },
                ..
            } => {
                let pressed = state == ElementState::Pressed;
                match key {
                    KeyCode::Escape | KeyCode::KeyQ if pressed => el.exit(),
                    KeyCode::KeyR if pressed => {
                        s.sim = make_sim();
                        s.frame = 0;
                        println!("reset");
                    }
                    KeyCode::KeyG if pressed => {
                        s.render_mode = s.render_mode.next();
                        println!("render mode: {}", s.render_mode.label());
                    }
                    KeyCode::ArrowUp => s.arrow_up = pressed,
                    KeyCode::ArrowDown => s.arrow_down = pressed,
                    KeyCode::ArrowLeft => s.arrow_left = pressed,
                    KeyCode::ArrowRight => s.arrow_right = pressed,
                    _ => {}
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

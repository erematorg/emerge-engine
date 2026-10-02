extern crate emerge_engine as emerge;

#[path = "../gui_common/render_mode.rs"]
mod render_mode;

use emerge::fields::LinearDragField;
use emerge::materials::optical::pure_water;
use emerge::render::{ColorMode, CpuRenderBridge, Renderer};
use emerge::{NewtonianFluidMaterial, SimConfig, Simulation, SlipBoundary, SpawnRegion};
use glam::{IVec2, Vec2};
/// Minimal real-forces proof: a real fluid material, no gravity-settling puddle, driven
/// downstream by `LinearDragField` -- the drag/current force field (see its doc
/// comment for the real physics: Stokes drag / Rayleigh friction, the SAME technique
/// that drives river currents and wind-blown sand in this engine).
///
/// A pool of water spawns on the left; the drag field pushes it rightward the whole run,
/// instead of the fluid just falling and puddling under gravity alone. Same field, same
/// mechanism -- change `target_velocity` to a wind direction and mask to a granular
/// material for wind-blown sand instead of masking to water; not built as a second demo
/// here, this scene exists to prove the ONE new mechanism, not every dressing of it.
///
/// G cycles the view: particles, the grid-volume view, the curvature-flow
/// surface. The water carries pure water's measured absorption
/// (`materials::optical::pure_water`, which cites its source), so every
/// view colours it the same way.
///
///   cargo run --example channel_flow --features "render"
use render_mode::RenderMode;
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

const GRID: usize = 96;
const DT: f32 = 0.1;
const MAT_WATER: u32 = 0;
const CURRENT_SPEED: f32 = 4.0;
const DRAG_COEFFICIENT: f32 = 1.5;
/// Particle pitch of the pool, in grid cells.
const SPACING: f32 = 0.6;
/// Surface grid resolution as a multiple of the physics grid. Measured on an
/// integrated Radeon (Vulkan): 6 to 7 ms per render at 4x, 4 ms at 2x,
/// inside this 60 fps demo's frame either way.
const SURFACE_RES_MULTIPLIER: u32 = 4;

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
        min_dt: 1.0e-3,
        max_substeps_per_step: 8,
        cfl_include_affine_speed: false,
        // Deliberately weak, NOT real IRL gravity (real g_grid ~= 981 via
        // SimConfig::earth) -- tuned down for a calmer, more legible demo at
        // this grid scale. Disclosed, deferred: basic_fluids.rs's
        // gravity_fraction slider is the real-IRL-with-live-control
        // pattern, not yet ported to every plain example.
        gravity: Vec2::new(0.0, -0.15),
        ..SimConfig::earth(GRID, 0.01, DT)
    };
    // Water through `NewtonianFluidMaterial::low_viscosity` (Tait exponent 7.0, Cole
    // 1948; water viscosity). rest_density=0.1 (`rho*dx^2` for water at dx=0.01, see
    // basic_fluids.rs). eos_stiffness=0.25: with rest_density 40x smaller than 4.0,
    // `timestep_bound`'s c2 (sound speed squared) is 40x larger at a given stiffness and
    // compression, so the stiffness is scaled by the same factor (10*0.1/4.0=0.25) to
    // keep the same c2.
    let mut water = NewtonianFluidMaterial::low_viscosity(0.1, 0.25);
    water.optics = Some(pure_water());
    let spawn_water = SpawnRegion {
        spacing: SPACING,
        box_size: IVec2::new(20, 16),
        box_center: Vec2::new(14.0, 12.0),
        material_id: MAT_WATER,
        initial_velocity_scale: 0.0,
        // Mass set explicitly, m = rho0*spacing^2 with the material's
        // rest_density=0.1 (see basic_fluids.rs), rather than the scene's grid
        // density.
        mass_override: Some(0.1 * SPACING * SPACING),
        ..SpawnRegion::for_sim(&config)
    };
    let current = LinearDragField::new(
        Vec2::new(CURRENT_SPEED, 0.0),
        DRAG_COEFFICIENT,
        1 << MAT_WATER,
    );
    Simulation::new(config, spawn_water)
        .with_default_material(Box::new(water))
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)))
        .with_force_field(Box::new(current))
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
        renderer.set_camera(&queue, GRID as u32, size.width, size.height, 0.6, true);
        renderer.set_color_mode(ColorMode::ByPhysics);
        renderer.adopt_material_optics(&queue, sim.materials());
        // The grid-volume and surface modes threshold on cell mass as a
        // fraction of a full cell, which holds 1/SPACING^2 particles.
        renderer.set_grid_reference_cell_mass(sim.particles().mass[0] / (SPACING * SPACING));
        renderer.set_surface_res_multiplier(SURFACE_RES_MULTIPLIER);
        let render_bridge = CpuRenderBridge::new(&device, GRID);
        println!(
            "channel_flow: {} water particles  |  LinearDragField pushes downstream at target_v=({CURRENT_SPEED},0)  |  G render mode  R reset  Q quit",
            sim.particles().len()
        );
        Self {
            surface,
            surface_config: sc,
            device,
            queue,
            sim,
            renderer,
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

    fn update_and_render(&mut self) {
        self.sim.step();
        self.frame += 1;
        self.fps_frames += 1;
        if self.fps_timer.elapsed().as_secs_f32() >= 2.0 {
            let fps = self.fps_frames as f32 / self.fps_timer.elapsed().as_secs_f32();
            let cx: f32 = self.sim.particles().iter().map(|p| p.x.x).sum::<f32>()
                / self.sim.particles().len() as f32;
            println!(
                "frame={} fps={:.0} water_centroid_x={cx:.2} (started at 14.0 -- should keep climbing)",
                self.frame, fps
            );
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
                // One material, so one optics slot colours the whole surface.
                self.renderer.render_surface_reconstruction(
                    &self.device,
                    &self.queue,
                    self.render_bridge
                        .surface_source(MAT_WATER, false, self.sim.mean_substep_dt()),
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
                    .with_title("emerge -- Channel Flow [LinearDragField]")
                    .with_inner_size(winit::dpi::LogicalSize::new(640u32, 480u32)),
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
            } => match key {
                KeyCode::Escape | KeyCode::KeyQ => el.exit(),
                KeyCode::KeyG => {
                    s.render_mode = s.render_mode.next();
                    println!("render mode: {}", s.render_mode.label());
                }
                KeyCode::KeyR => {
                    s.sim = make_sim();
                    s.frame = 0;
                    println!("reset");
                }
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
    let el = EventLoop::new().unwrap();
    el.set_control_flow(ControlFlow::Poll);
    let mut app = App {
        window: None,
        state: None,
    };
    el.run_app(&mut app).unwrap();
}

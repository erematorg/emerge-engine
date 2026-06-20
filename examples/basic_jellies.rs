extern crate emerge_engine as emerge;

use egui_wgpu::ScreenDescriptor;
use emerge::render::{ColorMode, Renderer};
use emerge::{
    CorotatedMaterial, NeoHookeanMaterial, SimConfig, Simulation, SlipBoundary, SpawnRegion,
    ViscoelasticMaterial,
};
use glam::{IVec2, Vec2};
/// CPU elastic solids -- NeoHookean / Corotated / Viscoelastic, three-blob comparison.
///
///   G  toggle ByPhysics/ByMaterial  |  LMB push  RMB pull  |  R reset  Q quit
///   cargo run --example basic_jellies --features "render"
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

const GRID: usize = 64;
const DT: f32 = 0.1;
const MAT_NEO: u32 = 0;
const MAT_COR: u32 = 1;
const MAT_VIS: u32 = 2;

const SIGMA_NEO: [f32; 3] = [0.05, 0.55, 0.60];
const SIGMA_COR: [f32; 3] = [0.10, 0.45, 0.50];
const SIGMA_VIS: [f32; 3] = [0.08, 0.35, 0.45];

struct App {
    window: Option<Arc<Window>>,
    state: Option<State>,
}

struct Params {
    neo_lambda: f32,
    neo_mu: f32,
    cor_lambda: f32,
    cor_mu: f32,
    vis_lambda: f32,
    vis_mu: f32,
    vis_viscosity: f32,
    gravity: f32,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            neo_lambda: 10.0,
            neo_mu: 20.0,
            cor_lambda: 30.0,
            cor_mu: 60.0,
            vis_lambda: 10.0,
            vis_mu: 15.0,
            vis_viscosity: 0.15,
            gravity: -0.3,
        }
    }
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
    p: Params,
    cursor_pos: [f32; 2],
    lmb: bool,
    rmb: bool,
    physics_colors: bool,
    frame: u64,
    fps_timer: std::time::Instant,
    fps_frames: u64,
    last_fps: f32,
}

fn make_sim(p: &Params) -> Simulation {
    let config = SimConfig {
        min_dt: 0.01,
        max_substeps_per_step: 8,
        gravity: Vec2::new(0.0, p.gravity),
        ..SimConfig::earth(GRID, 0.01, DT)
    };
    let spawn = |c: Vec2, mat| SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new(14, 14),
        box_center: c,
        material_id: mat,
        precompute_initial_volumes: true,
        initial_velocity_scale: 0.0,
        ..SpawnRegion::for_sim(&config)
    };
    let mut solver = Simulation::new(config, spawn(Vec2::new(14.0, 50.0), MAT_NEO))
        .with_default_material(Box::new(NeoHookeanMaterial::new(p.neo_lambda, p.neo_mu)))
        .with_material(
            MAT_COR,
            Box::new(CorotatedMaterial::new(p.cor_lambda, p.cor_mu)),
        )
        .with_material(
            MAT_VIS,
            Box::new(ViscoelasticMaterial::new(
                p.vis_lambda,
                p.vis_mu,
                p.vis_viscosity,
            )),
        )
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));
    let _ = solver.add_body(spawn(Vec2::new(32.0, 50.0), MAT_COR));
    let _ = solver.add_body(spawn(Vec2::new(50.0, 50.0), MAT_VIS));
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

        let p = Params::default();
        let sim = make_sim(&p);

        let mut renderer = Renderer::new(&device, sim.particles().len(), fmt);
        renderer.set_camera(&queue, GRID as u32, size.width, size.height, 0.6, true);
        renderer.set_color_mode(ColorMode::ByPhysics);
        renderer.set_optical_params(MAT_NEO as usize, SIGMA_NEO);
        renderer.set_optical_params(MAT_COR as usize, SIGMA_COR);
        renderer.set_optical_params(MAT_VIS as usize, SIGMA_VIS);

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
            "jellies: {} particles  G=colors  LMB/RMB=push/pull  R=reset  Q=quit",
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
            p,
            cursor_pos: [0.0; 2],
            lmb: false,
            rmb: false,
            physics_colors: true,
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
        Vec2::new(
            self.cursor_pos[0] / self.surface_config.width as f32 * GRID as f32,
            (1.0 - self.cursor_pos[1] / self.surface_config.height as f32) * GRID as f32,
        )
    }

    fn reset(&mut self) {
        self.sim = make_sim(&self.p);
        self.frame = 0;
    }

    fn update_and_render(&mut self, window: &Window) {
        // Push live params to solver
        self.sim.set_gravity(Vec2::new(0.0, self.p.gravity));
        self.sim
            .set_default_material(Box::new(NeoHookeanMaterial::new(
                self.p.neo_lambda,
                self.p.neo_mu,
            )));
        self.sim.set_material(
            MAT_COR,
            Box::new(CorotatedMaterial::new(self.p.cor_lambda, self.p.cor_mu)),
        );
        self.sim.set_material(
            MAT_VIS,
            Box::new(ViscoelasticMaterial::new(
                self.p.vis_lambda,
                self.p.vis_mu,
                self.p.vis_viscosity,
            )),
        );

        if self.lmb || self.rmb {
            let mag = if self.lmb { 2.0 } else { -2.0 };
            self.sim.apply_radial_impulse(self.cursor_grid(), 6.0, mag);
        }
        self.sim.step();
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

        // --- egui --- extract locals to avoid borrow conflict with closure
        let raw_input = self.egui_state.take_egui_input(window);
        let n = self.sim.particles().len();
        let fps = self.last_fps;
        let mut reset = false;
        let p = &mut self.p;

        let full_output = self.egui_ctx.run(raw_input, |ctx| {
            egui::Window::new("Jellies")
                .default_pos([10.0, 10.0])
                .default_width(260.0)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.label(format!("fps={:.0}  n={}  [G] toggle colors", fps, n));
                    ui.separator();
                    ui.add(egui::Slider::new(&mut p.gravity, -3.0..=0.0).text("gravity"));
                    ui.separator();
                    ui.colored_label(egui::Color32::from_rgb(240, 133, 69), "NeoHookean");
                    ui.add(egui::Slider::new(&mut p.neo_lambda, 1.0..=200.0).text("lambda"));
                    ui.add(egui::Slider::new(&mut p.neo_mu, 1.0..=400.0).text("mu"));
                    ui.separator();
                    ui.colored_label(egui::Color32::from_rgb(64, 199, 166), "Corotated");
                    ui.add(egui::Slider::new(&mut p.cor_lambda, 1.0..=200.0).text("lambda"));
                    ui.add(egui::Slider::new(&mut p.cor_mu, 1.0..=400.0).text("mu"));
                    ui.separator();
                    ui.colored_label(egui::Color32::from_rgb(184, 102, 230), "Viscoelastic");
                    ui.add(egui::Slider::new(&mut p.vis_lambda, 1.0..=200.0).text("lambda"));
                    ui.add(egui::Slider::new(&mut p.vis_mu, 1.0..=400.0).text("mu"));
                    ui.add(egui::Slider::new(&mut p.vis_viscosity, 0.0..=5.0).text("viscosity"));
                    ui.separator();
                    ui.label("LMB push  RMB pull  G colors  R reset");
                    if ui.button("Reset").clicked() {
                        reset = true;
                    }
                });
        });
        if reset {
            self.reset();
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

impl ApplicationHandler for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        let w = Arc::new(
            el.create_window(
                winit::window::WindowAttributes::default()
                    .with_title("emerge -- Jellies [NeoHookean / Corotated / Viscoelastic]")
                    .with_inner_size(winit::dpi::LogicalSize::new(480u32, 480u32)),
            )
            .unwrap(),
        );
        self.state = Some(pollster::block_on(State::new(w.clone())));
        self.window = Some(w);
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(s) = self.state.as_mut() else { return };
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
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => match key {
                KeyCode::Escape | KeyCode::KeyQ => el.exit(),
                KeyCode::KeyG => {
                    s.physics_colors = !s.physics_colors;
                    s.renderer.set_color_mode(if s.physics_colors {
                        ColorMode::ByPhysics
                    } else {
                        ColorMode::ByMaterial
                    });
                }
                KeyCode::KeyR => {
                    s.reset();
                    println!("reset");
                }
                _ => {}
            },
            WindowEvent::Resized(sz) => s.resize(sz.width, sz.height),
            WindowEvent::RedrawRequested => {
                if let Some(w) = &self.window {
                    s.update_and_render(w);
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

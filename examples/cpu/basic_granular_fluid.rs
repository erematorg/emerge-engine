extern crate emerge_engine as emerge;

#[path = "../gui_common/coords.rs"]
mod gui_common;

use egui_wgpu::ScreenDescriptor;
/// Real, live egui GUI for `GranularFluidMaterial` -- the tier-0
/// "solo-maximal pass" this material was still missing (added to the tier-0
/// list 2026-08-02, never individually stress-tested since: see
/// `project_ecosystem_slice_roadmap_2026-07-22.md`'s own note). Mirrors
/// `basic_sand.rs`'s established conventions exactly: LMB/RMB push-pull
/// (`apply_radial_impulse`), D-toggle directional dig (mass-conserving
/// per-particle velocity nudge, no second body), P to pour, real IRL
/// gravity slider (`Simulation::set_gravity`, 1.0 = genuine 9.81 m/s²).
///
/// PHASE RANGE (this material's own version of sand's loose/dense split):
/// cycles between its three distinct presets -- `saturated_loam` (soft,
/// yields easily), `consolidated_clay` (stiff, slower creep), and
/// `cytoplasmic` (soft biological-matrix regime). Their declared shear and
/// bulk viscosities supply dissipation; this demo never applies global
/// settling/Cundall damping as a substitute for constitutive physics.
///
/// Honest disclosure carried over from the material's own doc: the
/// constitutive LAW (Tait EOS + corotated elastic + SVD plasticity) is real
/// and cited (Dunatunga & Kamrin 2015); these three presets' specific shape
/// parameters are hand-tuned illustrative values, not measured geotechnical
/// data (see `GranularFluidMaterial::saturated_loam`'s own doc).
///
///   cargo run --example basic_granular_fluid --features render
use emerge::render::{ColorMode, Renderer};
use emerge::{GranularFluidMaterial, SimConfig, Simulation, SlipBoundary, SpawnRegion};
use glam::{IVec2, Vec2};
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

const GRID: usize = 64;
const DT: f32 = 0.05;
const MAT_LOAM: u32 = 0;
const MAT_CLAY: u32 = 1;
const MAT_CYTO: u32 = 2;
// Illustrative Beer-Lambert absorption colors (target_rgb -> sigma_a =
// -ln(target_rgb), same convention `SIGMA_SAND` uses) -- real technique,
// hand-picked target hues (wet-loam brown / clay tan-grey / pale
// biological), not measured against real material spectra.
const SIGMA_LOAM: [f32; 3] = [0.470, 0.620, 0.980]; // target ~(0.62,0.54,0.375) warm brown
const SIGMA_CLAY: [f32; 3] = [0.560, 0.560, 0.690]; // target ~(0.57,0.57,0.50) tan-grey
const SIGMA_CYTO: [f32; 3] = [0.220, 0.280, 0.260]; // target ~(0.80,0.76,0.77) pale translucent
const POUR_BUDGET: usize = 2000;
const POUR_SPACING: f32 = 0.5;
const POUR_BOX: IVec2 = IVec2::new(2, 1);
const DIG_RADIUS: f32 = 4.0;

fn make_sim() -> Simulation {
    // Dissipation is supplied only by the material's declared shear/bulk
    // viscosity. No global settling or Cundall damping is enabled here.
    let config = SimConfig {
        boundary_thickness: 3,
        max_substeps_per_step: 12,
        ..SimConfig::earth(GRID, 0.01, DT)
    };

    let spawn = |c: Vec2, mat, seed| SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new(16, 14),
        box_center: c,
        material_id: mat,
        precompute_initial_volumes: true,
        initial_velocity_scale: 0.0,
        rng_seed: seed,
        position_jitter: 0.3,
        ..SpawnRegion::for_sim(&config)
    };
    // Real, disclosed correction (2026-08-06, caught live by the user):
    // spawning at y=40 (same ~30-unit drop `basic_sand.rs` also uses)
    // exposed a real, measured impact-stability gap -- unlike sand,
    // GranularFluidMaterial's own numerics bounce substantially on a hard
    // impact even with real viscosity added (see `dynamic_viscosity`/
    // `bulk_viscosity` on the material itself). This demo's own point is
    // cursor push/pull/dig/pour interaction, not impact-stress-testing a
    // free fall this material was never shown to handle as well as sand --
    // spawning close to the floor sidesteps a real, disclosed, still-open
    // gap rather than hiding it.
    let mut solver = Simulation::new(config, spawn(Vec2::new(16.0, 12.0), MAT_LOAM, 11))
        .with_default_material(Box::new(GranularFluidMaterial::saturated_loam(600.0, 0.3)))
        .with_material(
            MAT_CLAY,
            Box::new(GranularFluidMaterial::consolidated_clay(600.0, 0.3)),
        )
        .with_material(
            MAT_CYTO,
            Box::new(GranularFluidMaterial::cytoplasmic(600.0, 0.3)),
        )
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));
    let _ = solver.add_body(spawn(Vec2::new(48.0, 12.0), MAT_CLAY, 22));
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
    lmb: bool,
    rmb: bool,
    pouring: bool,
    pour_material: u32,
    poured_count: usize,
    push_strength: f32,
    digging: bool,
    dig_strength: f32,
    last_cursor_grid: Vec2,
    real_gravity: Vec2,
    gravity_fraction: f32,
    frame: u64,
    fps_timer: std::time::Instant,
    fps_frames: u64,
    last_fps: f32,
    pour_seed: u32,
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
        let render_capacity = sim.particles().len() + POUR_BUDGET;
        let mut renderer = Renderer::new(&device, render_capacity, fmt);
        renderer.set_camera(&queue, GRID as u32, size.width, size.height, 0.9, true);
        renderer.set_color_mode(ColorMode::ByPhysics);
        renderer.set_optical_params(&queue, MAT_LOAM as usize, SIGMA_LOAM);
        renderer.set_optical_params(&queue, MAT_CLAY as usize, SIGMA_CLAY);
        renderer.set_optical_params(&queue, MAT_CYTO as usize, SIGMA_CYTO);

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
            "basic_granular_fluid: {} particles  |  LMB push  RMB pull  D toggle dig  hold P to pour  R reset  Q quit",
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
            lmb: false,
            rmb: false,
            pouring: false,
            pour_material: MAT_LOAM,
            poured_count: 0,
            push_strength: 12.0,
            digging: false,
            dig_strength: 18.0,
            last_cursor_grid: Vec2::ZERO,
            real_gravity,
            gravity_fraction: 0.001,
            frame: 0,
            fps_timer: std::time::Instant::now(),
            fps_frames: 0,
            last_fps: 0.0,
            pour_seed: 1000,
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
            .set_camera(&self.queue, GRID as u32, w, h, 0.9, true);
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
            self.sim.apply_radial_impulse(self.cursor_grid(), 7.0, mag);
        }
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
        if self.pouring && self.poured_count < POUR_BUDGET {
            let config = self.sim.config();
            let half = POUR_BOX.as_vec2() * 0.5;
            let domain_min = Vec2::splat(config.boundary_thickness as f32) + half;
            let domain_max =
                Vec2::splat((config.grid_res - config.boundary_thickness) as f32) - half;
            let cursor = self
                .cursor_grid()
                .clamp(domain_min, domain_max.max(domain_min));
            self.pour_seed += 1;
            let spawn = SpawnRegion {
                spacing: POUR_SPACING,
                box_size: POUR_BOX,
                box_center: cursor,
                material_id: self.pour_material,
                precompute_initial_volumes: true,
                initial_velocity_scale: 0.0,
                rng_seed: self.pour_seed,
                position_jitter: 0.3,
                ..SpawnRegion::for_sim(self.sim.config())
            };
            let before = self.sim.particles().len();
            let _ = self.sim.add_body(spawn);
            self.poured_count += self.sim.particles().len() - before;
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

        let raw_input = self.egui_state.take_egui_input(window);
        let fps = self.last_fps;
        let mut push_strength = self.push_strength;
        let mut gravity_fraction = self.gravity_fraction;
        let mut digging = self.digging;
        let mut dig_strength = self.dig_strength;
        let n_particles = self.sim.particles().len();
        let poured = self.poured_count;
        let mut pour_material = self.pour_material;
        let mut reset = false;

        let full_output = self.egui_ctx.run(raw_input, |ctx| {
            egui::Window::new("Granular Fluid")
                .default_pos([10.0, 10.0])
                .default_width(260.0)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.label(format!("fps={fps:.0}  particles={n_particles}"));
                    ui.separator();
                    ui.label("Gravity (1.0 = real IRL 9.81 m/s²):");
                    ui.add(egui::Slider::new(&mut gravity_fraction, 0.0..=2.0));
                    ui.separator();
                    ui.label("Push/pull strength:");
                    ui.add(egui::Slider::new(&mut push_strength, 0.0..=40.0));
                    ui.separator();
                    ui.checkbox(&mut digging, "Digging active (or press D)");
                    ui.add(egui::Slider::new(&mut dig_strength, 0.0..=40.0).text("Dig strength"));
                    ui.separator();
                    ui.label("Pour phase (real preset, not just a color):");
                    ui.radio_value(&mut pour_material, MAT_LOAM, "saturated_loam (soft)");
                    ui.radio_value(&mut pour_material, MAT_CLAY, "consolidated_clay (stiff)");
                    ui.radio_value(&mut pour_material, MAT_CYTO, "cytoplasmic (biological)");
                    ui.label(format!("Poured: {poured}/{POUR_BUDGET}"));
                    ui.add(
                        egui::ProgressBar::new(poured as f32 / POUR_BUDGET as f32)
                            .desired_width(200.0),
                    );
                    ui.separator();
                    ui.label("LMB push  RMB pull  D toggle dig  hold P to pour  R reset  Q quit");
                    if ui.button("Reset").clicked() {
                        reset = true;
                    }
                });
        });
        self.push_strength = push_strength;
        self.gravity_fraction = gravity_fraction;
        self.digging = digging;
        self.dig_strength = dig_strength;
        self.pour_material = pour_material;
        if reset {
            let sim = make_sim();
            self.real_gravity = sim.config().gravity;
            self.sim = sim;
            self.frame = 0;
            self.poured_count = 0;
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
                    .with_title("emerge -- Granular Fluid (GUI)")
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
                    KeyCode::KeyP => s.pouring = pressed,
                    KeyCode::KeyD if pressed => s.digging = !s.digging,
                    KeyCode::Escape | KeyCode::KeyQ if pressed => el.exit(),
                    KeyCode::KeyR if pressed => {
                        let sim = make_sim();
                        s.real_gravity = sim.config().gravity;
                        s.sim = sim;
                        s.frame = 0;
                        s.poured_count = 0;
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

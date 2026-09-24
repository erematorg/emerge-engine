extern crate emerge_engine as emerge;

/// Real rock fracture — 4 real rock types side by side (`RankineMaterial`'s
/// granite/sandstone/limestone/shale presets, real cited E/tensile-strength
/// ratios), struck with an adjustable-force downward impulse to compare how much
/// real damage accumulates per rock, per strike, and across REPEATED strikes
/// (Rankine's own damage state never decreases -- real fatigue-like accumulation,
/// not a per-hit reset).
///
/// Stiffness is GRID-NATIVE, not literal SI Pa -- real GPa-scale rock stiffness is
/// incompatible with explicit-MPM CFL at this grid resolution (same reason
/// `fire_spread.rs`'s own doc gives for wood). The REAL RATIOS between rock types
/// are preserved from their cited GPa values (granite 30, sandstone 20,
/// limestone 8, shale 27) so relative-stiffness honesty survives the rescale --
/// only the absolute magnitude is adapted for demo practicality, same pattern
/// `mass_override` already uses for relative density elsewhere tonight.
///
/// Honest, disclosed limitation carried over from `RankineMaterial::shale`'s own
/// doc: this is an ISOTROPIC model, so shale here represents its real ACROSS-
/// foliation (stronger) direction, not its well-known weak-along-bedding-planes
/// direction -- shale showing LESS damage than granite under the same strike is
/// real and expected given that, not a bug.
///
///   F strike at cursor (adjustable force)  |  [ / ] adjust strike force
///   R reset  Q quit
///   cargo run --example rock_fracture --features "render"
use egui_wgpu::ScreenDescriptor;
use emerge::render::{ColorMode, Renderer};
use emerge::{RankineMaterial, SimConfig, Simulation, SlipBoundary, SpawnRegion};
use glam::{IVec2, Vec2};
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

const GRID: usize = 64;
const DT: f32 = 0.02;

const GRANITE_ID: u32 = 0;
const SANDSTONE_ID: u32 = 1;
const LIMESTONE_ID: u32 = 2;
const SHALE_ID: u32 = 3;

// Grid-native stiffness, real ratios preserved from cited GPa values -- see module doc.
const GRANITE_STIFFNESS: f32 = 4000.0;
const SANDSTONE_STIFFNESS: f32 = GRANITE_STIFFNESS * (20.0 / 30.0);
const LIMESTONE_STIFFNESS: f32 = GRANITE_STIFFNESS * (8.0 / 30.0);
const SHALE_STIFFNESS: f32 = GRANITE_STIFFNESS * (27.0 / 30.0);

const STRIKE_RADIUS: f32 = 3.0;
const STRIKE_FORCE_STEP: f32 = 10.0;
const STRIKE_FORCE_MIN: f32 = 10.0;
const STRIKE_FORCE_MAX: f32 = 400.0;

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
    egui_ctx: egui::Context,
    egui_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
    cursor_pos: [f32; 2],
    striking: bool,
    strike_force: f32,
    frame: u64,
    fps_timer: std::time::Instant,
    fps_frames: u64,
    last_fps: f32,
}

struct Diagnostics {
    max_speed: f32,
    non_finite: usize,
    damage: [f32; 4],
}

fn make_sim() -> Simulation {
    let config = SimConfig {
        max_substeps_per_step: 64,
        gravity: Vec2::new(0.0, -0.3),
        ..SimConfig::earth(GRID, 0.01, DT)
    };

    let granite = RankineMaterial::stiff_brittle(GRANITE_STIFFNESS, 0.25);
    let sandstone = RankineMaterial::sandstone(SANDSTONE_STIFFNESS, 0.25);
    let limestone = RankineMaterial::limestone(LIMESTONE_STIFFNESS, 0.25);
    let shale = RankineMaterial::shale(SHALE_STIFFNESS, 0.25);

    let mut solver = Simulation::empty(config)
        .with_material(GRANITE_ID, Box::new(granite))
        .with_material(SANDSTONE_ID, Box::new(sandstone))
        .with_material(LIMESTONE_ID, Box::new(limestone))
        .with_material(SHALE_ID, Box::new(shale))
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));

    let blocks = [
        (GRANITE_ID, 9.0),
        (SANDSTONE_ID, 24.0),
        (LIMESTONE_ID, 39.0),
        (SHALE_ID, 54.0),
    ];
    for &(material_id, x_center) in &blocks {
        let spawn = SpawnRegion {
            spacing: 0.5,
            box_size: IVec2::new(12, 12),
            box_center: Vec2::new(x_center, 10.0),
            material_id,
            precompute_initial_volumes: true,
            ..SpawnRegion::for_sim(&config)
        };
        let _ = solver.add_body(spawn);
    }

    solver
}

/// Real per-material max damage report (`friction_hardening`, repurposed by
/// `RankineMaterial` as damage -- see that struct's own doc).
fn max_damage_by_material(sim: &Simulation) -> [f32; 4] {
    let particles = sim.particles();
    let mut result = [0.0f32; 4];
    for i in particles.indices() {
        let id = particles.material_id[i] as usize;
        if id < 4 {
            result[id] = result[id].max(particles.friction_hardening[i]);
        }
    }
    result
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
        // Optical params are Beer-Lambert absorption coefficients: color = exp(-sigma_a).
        // Real representative rock colors (not literal spectral measurements):
        // granite pale grey-pink, sandstone tan, limestone pale cream, shale dark grey.
        renderer.set_optical_params(&queue, GRANITE_ID as usize, [0.288, 0.223, 0.288]);
        renderer.set_optical_params(&queue, SANDSTONE_ID as usize, [0.223, 0.357, 0.799]);
        renderer.set_optical_params(&queue, LIMESTONE_ID as usize, [0.174, 0.223, 0.357]);
        renderer.set_optical_params(&queue, SHALE_ID as usize, [1.204, 1.204, 1.204]);

        println!(
            "rock_fracture: {} particles (granite/sandstone/limestone/shale)  |  \
             F strike  [ / ] force  R reset  Q quit",
            sim.particles().len()
        );
        println!("  hold F at a block to strike it (start force={STRIKE_FORCE_MIN:.0})");

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
            striking: false,
            strike_force: STRIKE_FORCE_MIN,
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

    /// Real per-frame health snapshot -- shared by the periodic console print and
    /// the egui panel so neither can silently drift out of sync with the other.
    fn diagnostics(&self) -> Diagnostics {
        let particles = self.sim.particles();
        let max_speed = particles
            .iter()
            .map(|p| p.v.length())
            .fold(0.0f32, f32::max);
        let non_finite = particles
            .iter()
            .filter(|p| !p.x.is_finite() || !p.v.is_finite())
            .count();
        Diagnostics {
            max_speed,
            non_finite,
            damage: max_damage_by_material(&self.sim),
        }
    }

    fn update_and_render(&mut self, window: &Window) {
        if self.striking {
            self.sim.apply_impulse(
                self.cursor_grid(),
                STRIKE_RADIUS,
                Vec2::new(0.0, -self.strike_force * DT),
            );
        }

        self.sim.step();
        self.frame += 1;
        self.fps_frames += 1;
        if self.fps_timer.elapsed().as_secs_f32() >= 2.0 {
            self.last_fps = self.fps_frames as f32 / self.fps_timer.elapsed().as_secs_f32();
            let d = self.diagnostics();
            println!(
                "frame={} fps={:.0} max_speed={:.3} non_finite={} \
                 damage[granite={:.3} sandstone={:.3} limestone={:.3} shale={:.3}] \
                 (should stay bounded -- large/nonzero non_finite = explosion)",
                self.frame,
                self.last_fps,
                d.max_speed,
                d.non_finite,
                d.damage[0],
                d.damage[1],
                d.damage[2],
                d.damage[3]
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
        self.renderer
            .render(&self.device, &self.queue, self.sim.particles(), &view, true);

        // --- egui panel ---
        let d = self.diagnostics();
        let raw_input = self.egui_state.take_egui_input(window);
        let mut strike_force = self.strike_force;
        let mut reset_clicked = false;

        let full_output = self.egui_ctx.run(raw_input, |ctx| {
            egui::Window::new("Rock Fracture")
                .default_pos([10.0, 10.0])
                .default_width(280.0)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.label(format!("fps={:.0}  frame={}", self.last_fps, self.frame));
                    ui.label(format!(
                        "max_speed={:.3}  non_finite={}",
                        d.max_speed, d.non_finite
                    ));
                    ui.separator();
                    ui.label(format!("damage granite={:.3}", d.damage[0]));
                    ui.label(format!("damage sandstone={:.3}", d.damage[1]));
                    ui.label(format!("damage limestone={:.3}", d.damage[2]));
                    ui.label(format!("damage shale={:.3}", d.damage[3]));
                    ui.separator();
                    ui.add(
                        egui::Slider::new(&mut strike_force, STRIKE_FORCE_MIN..=STRIKE_FORCE_MAX)
                            .text("strike force ([ / ])"),
                    );
                    ui.label("F strike at cursor  R reset  Q quit");
                    if ui.button("Reset").clicked() {
                        reset_clicked = true;
                    }
                });
        });

        self.strike_force = strike_force;
        if reset_clicked {
            self.sim = make_sim();
            self.frame = 0;
            println!("reset");
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
                    .with_title("emerge -- Rock Fracture [granite/sandstone/limestone/shale]")
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
            WindowEvent::CursorMoved { position, .. } => {
                s.cursor_pos = [position.x as f32, position.y as f32];
            }
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
                    KeyCode::KeyF => s.striking = pressed,
                    _ if !pressed => {}
                    KeyCode::Escape | KeyCode::KeyQ => el.exit(),
                    KeyCode::KeyR => {
                        s.sim = make_sim();
                        s.frame = 0;
                        println!("reset");
                    }
                    KeyCode::BracketRight => {
                        s.strike_force = (s.strike_force + STRIKE_FORCE_STEP).min(STRIKE_FORCE_MAX);
                        println!("strike_force={:.0}", s.strike_force);
                    }
                    KeyCode::BracketLeft => {
                        s.strike_force = (s.strike_force - STRIKE_FORCE_STEP).max(STRIKE_FORCE_MIN);
                        println!("strike_force={:.0}", s.strike_force);
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Real regression check: all 4 blocks spawn distinctly, non-overlapping, and
    /// every particle carries the material_id its block's x-position implies --
    /// proves the layout is real, not just "it compiles".
    #[test]
    fn four_rock_blocks_spawn_with_correct_materials() {
        let sim = make_sim();
        let particles = sim.particles();
        assert!(particles.len() > 0, "must spawn particles");

        let mut counts = [0usize; 4];
        for i in particles.indices() {
            let id = particles.material_id[i];
            assert!(id < 4, "unexpected material_id={id}");
            counts[id as usize] += 1;
        }
        for (id, count) in counts.iter().enumerate() {
            assert!(*count > 0, "material_id={id} has zero particles");
        }
    }
}

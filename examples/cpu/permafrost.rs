extern crate emerge_engine as emerge;

#[path = "../gui_common/coords.rs"]
mod gui_common;

/// Permafrost freeze/thaw: a block of ice-bonded soil (`NaccMaterial::kaolin` at 250x its
/// thawed stiffness) that softens once ambient warming takes it past the freezing point
/// (273.15 K). A live version of
/// `tests/solver.rs::permafrost_thaws_at_freezing_point_with_real_latent_heat_debit` and
/// `frozen_ground_resists_a_strike_more_than_thawed_ground`.
///
/// Water/ice latent heat of fusion (334, the value used elsewhere for water) is absorbed
/// on thaw through `WithLatentHeat` + `add_phase_rule`, the machinery of
/// `fire_spread.rs`'s combustion, not scaled by permafrost's ice-content fraction
/// (ice-bonded soil, not pure ice).
///
/// Frozen soil is two to three orders of magnitude stiffer than thawed (~100 MPa
/// unfrozen soil vs ~23-30 GPa frozen fine sand, Andersland & Ladanyi-adjacent
/// research). The ratio here is ~250x, the midpoint of that ~230-300x range; a
/// substep-headroom sweep under continuous strikes uses 8% (thawed) and 31% (frozen) of
/// the budget, so CFL does not force a smaller ratio. See `THAWED_STIFFNESS`/
/// `FROZEN_STIFFNESS` below.
///
///   W hold to warm ambient (simulate seasonal thaw)  |  C hold to cool it back down
///   F strike at cursor (adjustable force)  |  [ / ] adjust strike force
///   R reset  Q quit
///   cargo run --example permafrost --features "render"
use egui_wgpu::ScreenDescriptor;
use emerge::render::{ColorMode, Renderer};
use emerge::thermodynamics::{ThermalConfig, ThermalDiffusion};
use emerge::{NaccMaterial, SimConfig, Simulation, SlipBoundary, SpawnRegion, WithLatentHeat};
use glam::{IVec2, Vec2};
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

const GRID: usize = 64;
const GRAVITY_MAGNITUDE: f32 = 0.3;
const BLOCK_CELLS: IVec2 = IVec2::new(32, 20);
const DT: f32 = 0.02;

const FROZEN_ID: u32 = 0;
const THAWED_ID: u32 = 1;

const FREEZING_POINT_K: f32 = 273.15;
const LATENT_HEAT_FUSION: f32 = 334.0;
const AMBIENT_START_K: f32 = 260.0; // real permafrost winter-range starting temperature
const AMBIENT_RATE: f32 = 15.0; // K/s while holding W or C -- demo pacing, not a measured rate

// Measured: at dt=0.02/max_substeps_per_step=64, under continuous strikes, thawed uses
// 8% and frozen 31% of the substep budget, so the frozen/thawed ratio is the literature
// ~250x (midpoint of the ~230-300x of ~100 MPa unfrozen soil vs ~23-30 GPa frozen fine
// sand, see the module doc).
const THAWED_STIFFNESS: f32 = 18000.0;
const FROZEN_STIFFNESS: f32 = THAWED_STIFFNESS * 250.0;

/// Mean stress a layer already carries from everything above it, in grid
/// units: the weight per cell of each layer above plus half of its own,
/// times gravity, turned into a mean stress with Jaky's earth-pressure
/// coefficient at rest, `K0 = 1 - sin(phi')`, so `p = sigma_v (1 + K0)/2`
/// in plane strain. A soil in place has carried this for a long time, so
/// its clay starts preconsolidated under it instead of as fresh slurry.
const CLAY_FRICTION_ANGLE_SIN: f32 = 0.436; // kaolin, 25.9 degrees

const STRIKE_RADIUS: f32 = 3.0;
const STRIKE_FORCE_STEP: f32 = 10.0;
const STRIKE_FORCE_MIN: f32 = 10.0;
const STRIKE_FORCE_MAX: f32 = 300.0;

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
    warming: bool,
    cooling: bool,
    frame: u64,
    fps_timer: std::time::Instant,
    fps_frames: u64,
    last_fps: f32,
}

fn make_sim() -> Simulation {
    let config = SimConfig {
        max_substeps_per_step: 64,
        gravity: Vec2::new(0.0, -GRAVITY_MAGNITUDE),
        ..SimConfig::earth(GRID, 0.01, DT)
    };

    let thermal = ThermalDiffusion::new(
        ThermalConfig {
            conductivity: 2.2, // real, ice-rich soil, higher than dry soil (ice conducts well)
            heat_capacity: 2000.0,
            density: 1800.0,
            ambient: AMBIENT_START_K,
            grid_cell_size: config.dx_meters,
            ..Default::default()
        },
        config.grid_res,
    );

    // The block is BLOCK_CELLS.y deep at the default grid density, so its
    // clay starts consolidated under the weight above its own mid-depth.
    let sigma_v = GRAVITY_MAGNITUDE * BLOCK_CELLS.y as f32 * 0.5;
    let preconsolidation = sigma_v * (2.0 - CLAY_FRICTION_ANGLE_SIN) * 0.5;
    let mut frozen = NaccMaterial::kaolin(FROZEN_STIFFNESS, 0.3);
    frozen.initial_preconsolidation = preconsolidation;
    let mut thawed_clay = NaccMaterial::kaolin(THAWED_STIFFNESS, 0.3);
    thawed_clay.initial_preconsolidation = preconsolidation;
    let thawed = WithLatentHeat::new(thawed_clay, LATENT_HEAT_FUSION);

    let mut solver = Simulation::empty(config)
        .with_material(FROZEN_ID, Box::new(frozen))
        .with_material(THAWED_ID, Box::new(thawed))
        .with_thermal(thermal)
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)))
        .with_phase_rule(|p| {
            if p.material_id == FROZEN_ID && p.temperature > FREEZING_POINT_K {
                Some(THAWED_ID)
            } else if p.material_id == THAWED_ID && p.temperature < FREEZING_POINT_K {
                Some(FROZEN_ID)
            } else {
                None
            }
        });

    let spawn = SpawnRegion {
        spacing: 0.5,
        box_size: BLOCK_CELLS,
        box_center: Vec2::new(32.0, 12.0),
        material_id: FROZEN_ID,
        ..SpawnRegion::for_sim(&config)
    };
    let _ = solver.add_body(spawn);

    for t in solver.particles_mut().temperature.iter_mut() {
        *t = AMBIENT_START_K;
    }

    solver
}

struct Diagnostics {
    frozen: usize,
    thawed: usize,
    avg_temp: f32,
    ambient: f32,
    max_speed: f32,
    non_finite: usize,
    min_volume_ratio: f32,
    width: f32,
    height: f32,
}

fn material_counts(sim: &Simulation) -> (usize, usize) {
    let particles = sim.particles();
    let frozen = particles
        .indices()
        .filter(|&i| particles.material_id[i] == FROZEN_ID)
        .count();
    let thawed = particles
        .indices()
        .filter(|&i| particles.material_id[i] == THAWED_ID)
        .count();
    (frozen, thawed)
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
        // Representative colors: frozen = pale blue-white (ice-bonded), thawed = dark
        // wet-clay brown; not spectral measurements.
        renderer.set_optical_params(&queue, FROZEN_ID as usize, [0.588, 0.470, 0.357]);
        renderer.set_optical_params(&queue, THAWED_ID as usize, [1.386, 1.139, 0.799]);

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

        let (frozen, thawed) = material_counts(&sim);
        println!(
            "permafrost: frozen={frozen} thawed={thawed}  |  W warm  C cool  F strike  \
             [ / ] force  R reset  Q quit"
        );
        println!("  freezing point={FREEZING_POINT_K}K, starting ambient={AMBIENT_START_K}K");

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
            warming: false,
            cooling: false,
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

    /// Per-frame health snapshot, shared by the periodic console print and the egui
    /// panel so the two stay in sync.
    fn diagnostics(&mut self) -> Diagnostics {
        let (frozen, thawed) = material_counts(&self.sim);
        let particles = self.sim.particles();
        let avg_temp: f32 =
            particles.temperature.iter().sum::<f32>() / particles.len().max(1) as f32;
        let max_speed = particles
            .iter()
            .map(|p| p.v.length())
            .fold(0.0f32, f32::max);
        let non_finite = particles
            .iter()
            .filter(|p| !p.x.is_finite() || !p.v.is_finite())
            .count();
        let min_volume_ratio = particles
            .initial_volume
            .iter()
            .zip(particles.volume.iter())
            .map(|(&v0, &v)| if v0 > 1.0e-9 { v / v0 } else { 1.0 })
            .fold(f32::INFINITY, f32::min);
        // Aggregate shape: per-particle volume (above) can stay exactly 1.0 while the
        // block spreads laterally through shear or plastic flow (particles sliding
        // past each other, not compressing), so the block's bounding-box width and
        // height show whether it flattened.
        let (mut min_x, mut max_x, mut min_y, mut max_y) = (
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::INFINITY,
            f32::NEG_INFINITY,
        );
        for p in particles.iter() {
            min_x = min_x.min(p.x.x);
            max_x = max_x.max(p.x.x);
            min_y = min_y.min(p.x.y);
            max_y = max_y.max(p.x.y);
        }
        let ambient = self
            .sim
            .thermal_config_mut()
            .map(|t| t.ambient)
            .unwrap_or(f32::NAN);
        Diagnostics {
            frozen,
            thawed,
            avg_temp,
            ambient,
            max_speed,
            non_finite,
            min_volume_ratio,
            width: max_x - min_x,
            height: max_y - min_y,
        }
    }

    fn update_and_render(&mut self, window: &Window) {
        if self.warming || self.cooling {
            let sign = if self.warming { 1.0 } else { -1.0 };
            if let Some(thermal) = self.sim.thermal_config_mut() {
                thermal.ambient += sign * AMBIENT_RATE * DT;
            }
            // Fourier diffusion alone is too slow to be playable (diffusion.rs:
            // soil-scale conduction is ~18000 s against MPM's ~0.002 s mechanical
            // CFL), so particle temperature is also nudged directly, a demo-pacing
            // simplification like `fire_spread.rs`'s `ignite_at_cursor` (direct
            // Newton-style relaxation), so warming and cooling are watchable.
            let particles = self.sim.particles_mut();
            for t in particles.temperature.iter_mut() {
                *t += sign * AMBIENT_RATE * DT;
            }
        }
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
                "frame={} fps={:.0} frozen={} thawed={} \
                 avg_temp={:.1}K ambient={:.1}K max_speed={:.3} \
                 non_finite={} min_volume_ratio={:.4} \
                 width={:.2} height={:.2} aspect={:.2} \
                 (large/nonzero non_finite = explosion, min_volume_ratio near 0 = per-particle \
                 collapse, growing aspect = aggregate flattening)",
                self.frame,
                self.last_fps,
                d.frozen,
                d.thawed,
                d.avg_temp,
                d.ambient,
                d.max_speed,
                d.non_finite,
                d.min_volume_ratio,
                d.width,
                d.height,
                d.width / d.height.max(1.0e-6)
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
        let mut warm_clicked = false;
        let mut cool_clicked = false;

        let full_output = self.egui_ctx.run(raw_input, |ctx| {
            egui::Window::new("Permafrost")
                .default_pos([10.0, 10.0])
                .default_width(280.0)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.label(format!("fps={:.0}  frame={}", self.last_fps, self.frame));
                    ui.label(format!("frozen={} thawed={}", d.frozen, d.thawed));
                    ui.label(format!(
                        "avg_temp={:.1}K  ambient={:.1}K",
                        d.avg_temp, d.ambient
                    ));
                    ui.label(format!(
                        "freezing point={FREEZING_POINT_K}K  latent heat={LATENT_HEAT_FUSION}"
                    ));
                    ui.separator();
                    ui.label(format!(
                        "max_speed={:.3}  non_finite={}",
                        d.max_speed, d.non_finite
                    ));
                    ui.label(format!(
                        "min_volume_ratio={:.4}  width={:.2} height={:.2} aspect={:.2}",
                        d.min_volume_ratio,
                        d.width,
                        d.height,
                        d.width / d.height.max(1.0e-6)
                    ));
                    ui.separator();
                    ui.add(
                        egui::Slider::new(&mut strike_force, STRIKE_FORCE_MIN..=STRIKE_FORCE_MAX)
                            .text("strike force ([ / ])"),
                    );
                    ui.horizontal(|ui| {
                        if ui.button("Warm now (+5K)").clicked() {
                            warm_clicked = true;
                        }
                        if ui.button("Cool now (-5K)").clicked() {
                            cool_clicked = true;
                        }
                    });
                    ui.label("W hold warm  C hold cool  F strike  R reset  Q quit");
                    if ui.button("Reset").clicked() {
                        reset_clicked = true;
                    }
                });
        });

        self.strike_force = strike_force;
        if warm_clicked || cool_clicked {
            let sign = if warm_clicked { 1.0 } else { -1.0 };
            if let Some(thermal) = self.sim.thermal_config_mut() {
                thermal.ambient += sign * 5.0;
            }
            for t in self.sim.particles_mut().temperature.iter_mut() {
                *t += sign * 5.0;
            }
        }
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
                    .with_title("emerge -- Permafrost [freeze/thaw]")
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
                    KeyCode::KeyW => s.warming = pressed,
                    KeyCode::KeyC => s.cooling = pressed,
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

    /// The block spawns fully frozen, and warming the ambient past the freezing point
    /// thaws it (material_id changes), through this demo's `make_sim()` and phase-rule
    /// wiring, beyond the unit tests in `tests/solver.rs`.
    #[test]
    fn warming_past_freezing_point_thaws_the_block() {
        let mut sim = make_sim();
        let (frozen0, thawed0) = material_counts(&sim);
        assert!(frozen0 > 0, "must start with frozen particles");
        assert_eq!(thawed0, 0, "must start with zero thawed particles");

        // Sets particle temperature directly: Fourier diffusion alone would take an
        // unplayable sim time to warm the block (diffusion.rs: soil-scale conduction
        // ~18000 s against MPM's ~0.002 s mechanical CFL). This tests the phase-rule
        // and latent-heat wiring, as `tests/solver.rs`'s permafrost tests do; the W/C
        // keys are a UI concern.
        if let Some(thermal) = sim.thermal_config_mut() {
            thermal.ambient = 300.0; // well above freezing
        }
        for t in sim.particles_mut().temperature.iter_mut() {
            *t = 300.0;
        }
        sim.step();

        let (_frozen1, thawed1) = material_counts(&sim);
        assert!(
            thawed1 > 0,
            "warming past the real freezing point must thaw at least some of the block"
        );
    }
}

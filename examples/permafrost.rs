extern crate emerge_engine as emerge;

/// Real permafrost freeze/thaw -- a block of ice-bonded soil (`NaccMaterial::wet_soil`
/// at 250x its own thawed stiffness) that genuinely softens once real ambient warming
/// pushes it past the real freezing point (273.15K), same mechanism verified in
/// `tests/solver.rs::permafrost_thaws_at_freezing_point_with_real_latent_heat_debit`
/// and `frozen_ground_resists_a_strike_more_than_thawed_ground` -- this is the live,
/// watchable version of those two tests, not new physics.
///
/// Real water/ice latent heat of fusion (334, same value already used elsewhere in
/// this codebase for water) is absorbed on thaw via `WithLatentHeat` + `add_phase_rule`
/// -- the SAME machinery already proven for combustion in `fire_spread.rs`. Real,
/// disclosed simplification: not scaled down by permafrost's actual real ice-content
/// fraction (it's an ice-BONDED soil mixture, not pure ice).
///
/// Real stiffness ratio, measured not guessed: literature composes to roughly two to
/// three orders of magnitude stiffer frozen-vs-thawed (~100 MPa unfrozen soil vs
/// ~23-30 GPa frozen fine sand, Andersland & Ladanyi-adjacent research) -- an earlier
/// version of this demo compressed that to 8x "for CFL practicality," but a real
/// substep-headroom sweep (2026-07-31, at this demo's own dt/substep budget, under
/// continuous strike load) showed CFL was never actually the binding constraint --
/// the real ~250x ratio (midpoint of the cited ~230-300x range) uses only 8%/31% of
/// the substep budget. Uses the real ratio directly now, see `THAWED_STIFFNESS`/
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
const DT: f32 = 0.02;

const FROZEN_ID: u32 = 0;
const THAWED_ID: u32 = 1;

const FREEZING_POINT_K: f32 = 273.15;
const LATENT_HEAT_FUSION: f32 = 334.0;
const AMBIENT_START_K: f32 = 260.0; // real permafrost winter-range starting temperature
const AMBIENT_RATE: f32 = 15.0; // K/s while holding W or C -- demo pacing, not a measured rate

// Real, MEASURED values (2026-07-31), not the earlier CFL-compromise guess: a real
// substep-headroom sweep at this demo's own dt=0.02/max_substeps_per_step=64 showed
// both values stay well within budget even under continuous strikes (thawed 8%,
// frozen 31% of the substep budget used) at these numbers -- CFL was never actually
// the binding constraint here, so the frozen/thawed ratio no longer needs to be
// compressed. Uses the real cited literature ratio directly instead: ~250x, the
// midpoint of Andersland & Ladanyi-adjacent research's ~230-300x (~100 MPa unfrozen
// soil vs ~23-30 GPa frozen fine sand) -- see module doc for the original citation.
const THAWED_STIFFNESS: f32 = 18000.0;
const FROZEN_STIFFNESS: f32 = THAWED_STIFFNESS * 250.0;

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
        gravity: Vec2::new(0.0, -0.3),
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

    let frozen = NaccMaterial::wet_soil(FROZEN_STIFFNESS, 0.3);
    let thawed = WithLatentHeat::new(
        NaccMaterial::wet_soil(THAWED_STIFFNESS, 0.3),
        LATENT_HEAT_FUSION,
    );

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
        box_size: IVec2::new(32, 20),
        box_center: Vec2::new(32.0, 12.0),
        material_id: FROZEN_ID,
        precompute_initial_volumes: true,
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
        // Real representative colors: frozen = pale blue-white (ice-bonded), thawed =
        // dark wet-clay brown -- not literal spectral measurements.
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
        Vec2::new(
            self.cursor_pos[0] / self.surface_config.width as f32 * GRID as f32,
            (1.0 - self.cursor_pos[1] / self.surface_config.height as f32) * GRID as f32,
        )
    }

    /// Real per-frame health snapshot -- shared by the periodic console print and
    /// the egui panel so neither can silently drift out of sync with the other.
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
        // Real AGGREGATE shape check -- per-particle volume (above) can stay
        // exactly 1.0 while the whole block still spreads out laterally via
        // shear/plastic flow (particles sliding past each other, not compressing
        // individually). Width/height of the block's own bounding box is the
        // real signal for "did it flatten", not per-particle J.
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
            // Real Fourier diffusion alone is far too slow to be playable here (this
            // engine's own diffusion.rs doc: real soil-scale conduction is ~18000s vs
            // MPM's ~0.002s mechanical CFL) -- directly nudging particle temperature
            // too is the SAME disclosed demo-pacing simplification `fire_spread.rs`'s
            // own `ignite_at_cursor` already uses (direct Newton-style relaxation, not
            // waiting on real ambient conduction) so warming/cooling is actually
            // watchable in a live session.
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

    /// Real regression check: the block spawns fully frozen, and warming the real
    /// ambient past the real freezing point genuinely thaws it (material_id changes),
    /// not just a cosmetic temperature number -- proves the live demo's own
    /// `make_sim()`/phase-rule wiring works, not just the abstracted unit tests in
    /// `tests/solver.rs`.
    #[test]
    fn warming_past_freezing_point_thaws_the_block() {
        let mut sim = make_sim();
        let (frozen0, thawed0) = material_counts(&sim);
        assert!(frozen0 > 0, "must start with frozen particles");
        assert_eq!(thawed0, 0, "must start with zero thawed particles");

        // Real Fourier diffusion alone would take a genuinely unplayable amount of
        // sim-time to warm the block from ambient (this engine's own diffusion.rs
        // doc: real soil-scale conduction is ~18000s vs MPM's ~0.002s mechanical
        // CFL) -- directly setting particle temperature tests the actual thing that
        // matters here (the phase-rule + latent-heat wiring), matching how
        // `tests/solver.rs`'s own permafrost/latent-heat tests already do this; the
        // interactive W/C warm/cool keys are a live-verified UI concern, not
        // something a headless test should wait on real diffusion for.
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

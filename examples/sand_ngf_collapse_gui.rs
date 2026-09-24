extern crate emerge_engine as emerge;

use egui_wgpu::ScreenDescriptor;
/// Side-by-side, real-SI proof of what Nonlocal Granular Fluidity (NGF) does
/// and does not fix for `DruckerPragerMaterial` sand -- same real physical
/// scene as `sand::ngf_verification_tests::run_column_collapse` (a column of
/// real dry sand, E=15MPa/nu=0.3/rho=1600kg/m3, aspect ratio 4:1, real
/// seconds/meters via `SimConfig::earth`), rendered live instead of measured
/// headless.
///
/// Two real, cited reference lines are drawn on screen at the Lajeunesse et
/// al. 2004 predicted final spread (`R_inf = r0*(1+2*sqrt(h0/r0))`, the same
/// formula the headless test uses) -- real dry sand of this aspect ratio
/// stops there. Watch the pile blow straight past both lines regardless of
/// which mode is active: that is the real, still-open sand accuracy gap
/// this whole effort exists to chip at. NGF (press N) measurably narrows the
/// final spread vs baseline (press B) but nowhere near enough to stop at the
/// lines -- do not expect a dramatic visual difference between the two
/// modes, the real measured effect is ~3% tighter, not a fix.
///
///   cargo run --example sand_ngf_collapse_gui --features render
use emerge::render::{ColorMode, Renderer};
use emerge::thermodynamics::{GranularFluidityConfig, GranularFluidityField};
use emerge::{
    DruckerPragerMaterial, FrictionBoundary, Particle, SimConfig, Simulation, SpawnRegion,
};
use glam::{IVec2, Vec2};
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

const GRID: usize = 96;
const CELL_M: f32 = 0.01;
const DT_S: f32 = 0.01;
const SIGMA_SAND: [f32; 3] = [0.180, 0.220, 0.550];

// Real dry sand, same values used throughout tonight's NGF verification
// (Haeri & Skonieczny 2022 Table 1, Excavation case).
const YOUNG_MODULUS_PA: f32 = 15.0e6;
const POISSON_RATIO: f32 = 0.3;
const BULK_DENSITY_KG_M3: f32 = 1600.0;
const FRICTION_DEG: f32 = 35.0;

// Column geometry -- identical aspect ratio (4:1) to the headless Lajeunesse
// test, so the same predicted-R_inf formula applies unmodified.
const R0_CELLS: f32 = 4.0;
const H0_CELLS: f32 = 16.0;
const FLOOR_CELLS: f32 = 5.0; // 0.05m / CELL_M

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Mode {
    Baseline,
    Ngf,
}

// Bare fn pointer (no captures) -- `GranularFluidityField::pressure_and_ratio`
// requires this, same constraint already documented in
// `sand::ngf_verification_tests`. Real SI (not grid-scaled) Lame parameters
// -- the material's own internal `lambda`/`mu` ARE grid-scaled (needed for
// MPM stress), but this reaction term's `sqrt(P/rho_s)*d` needs real
// Pascals, so it's recomputed fresh here (`lame_from_young` is public).
//
// Recovers the 2 singular values of `deformation_gradient` from two
// rotation-invariant scalars instead of calling the engine's own SVD
// (`materials::svd` is `pub(crate)`, not reachable from example code):
// Frobenius norm² = sigma1²+sigma2² (SVD-invariant: trace(FᵀF)), and
// det(F) = sigma1*sigma2 whenever F isn't inverted (real settling sand
// never is). Two equations, two unknowns -- exact, not an approximation,
// same real Hencky-strain pressure/stress-ratio formula
// `MuIRheologyMaterial::update_particle` uses either way.
fn ngf_pressure_and_ratio(p: &Particle) -> (f32, f32) {
    use emerge::materials::utils::lame_from_young;
    let (lambda, mu) = lame_from_young(YOUNG_MODULUS_PA, POISSON_RATIO);
    let f = p.deformation_gradient;
    let sum_sq = f.x_axis.length_squared() + f.y_axis.length_squared(); // sigma1^2+sigma2^2
    let det = f.determinant().max(1.0e-6); // sigma1*sigma2 (guard against inversion)
    let sum = (sum_sq + 2.0 * det).max(0.0).sqrt(); // sigma1+sigma2
    let diff = (sum_sq - 2.0 * det).max(0.0).sqrt(); // |sigma1-sigma2|
    let sigma1 = ((sum + diff) * 0.5).max(1.0e-6);
    let sigma2 = ((sum - diff) * 0.5).max(1.0e-6);
    let eps = Vec2::new(
        sigma1.ln() + p.log_volume_strain * 0.5,
        sigma2.ln() + p.log_volume_strain * 0.5,
    );
    let trace = eps.x + eps.y;
    let dev = eps - Vec2::splat(trace * 0.5);
    let dev_norm = dev.length();
    let p_trial = -(lambda + mu) * trace;
    let mu_ratio = if p_trial > 1.0e-6 {
        std::f32::consts::SQRT_2 * mu * dev_norm / p_trial
    } else {
        0.0
    };
    (p_trial.max(0.0), mu_ratio)
}

fn ngf_config() -> GranularFluidityConfig {
    const EFFECTIVE_GRAIN_DIAMETER_M: f32 = 0.008;
    const GRAIN_DENSITY_KG_M3: f32 = 2583.0;
    let pressure_floor_pa = GRAIN_DENSITY_KG_M3 * 9.81 * EFFECTIVE_GRAIN_DIAMETER_M;
    GranularFluidityConfig {
        mu_s: 0.70,
        grain_diameter_m: EFFECTIVE_GRAIN_DIAMETER_M,
        grain_density_kg_m3: GRAIN_DENSITY_KG_M3,
        nonlocal_amplitude: 0.48,
        b: 0.278,
        t0_s: 1.0e-4,
        pressure_floor_pa,
    }
}

fn predicted_r_inf_cells() -> f32 {
    let aspect = H0_CELLS / R0_CELLS;
    R0_CELLS * (1.0 + 2.0 * aspect.sqrt())
}

fn make_sim(mode: Mode) -> Simulation {
    let config = SimConfig {
        max_substeps_per_step: 4000,
        ..SimConfig::earth(GRID, CELL_M, DT_S)
    };
    let column = SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new(8, 16),
        box_center: Vec2::new(GRID as f32 * 0.5, FLOOR_CELLS + 8.0),
        material_id: 0,
        precompute_initial_volumes: true,
        ..SpawnRegion::for_sim(&config)
    };
    // Same real SI -> grid-Lame conversion `DruckerPragerMaterial::from_physical`
    // uses internally (`SimConfig::lame_from_si_cfg` is the public SI helper
    // documented for exactly this -- the `GranularProps` bridging struct that
    // wraps it is deliberately `pub(super)`, not part of the public property-
    // system API, see `physical_props.rs`'s own doc).
    let (lambda, mu) = config.lame_from_si_cfg(YOUNG_MODULUS_PA, POISSON_RATIO, BULK_DENSITY_KG_M3);
    let sand = DruckerPragerMaterial {
        friction_angle: FRICTION_DEG.to_radians(),
        dilatancy_angle: 0.0,
        ngf_enabled: mode == Mode::Ngf,
        ..DruckerPragerMaterial::new(lambda, mu)
    };
    let mut solver = Simulation::new(config, column)
        .with_default_material(Box::new(sand))
        .with_boundary(Box::new(FrictionBoundary::new(2, 0.7)));
    if mode == Mode::Ngf {
        let field = GranularFluidityField::new(ngf_config(), ngf_pressure_and_ratio, GRID);
        solver = solver.with_granular_fluidity(field);
    }
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
    mode: Mode,
    paused: bool,
    step: u64,
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
        let mode = Mode::Baseline;
        let sim = make_sim(mode);
        let mut renderer = Renderer::new(&device, sim.particles().len(), fmt);
        renderer.set_camera(&queue, GRID as u32, size.width, size.height, 0.9, true);
        renderer.set_color_mode(ColorMode::ByPhysics);
        renderer.set_optical_params(&queue, 0, SIGMA_SAND);

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
            "sand_ngf_collapse_gui: {} particles  |  B=baseline reset  N=ngf reset  SPACE=pause  Q=quit",
            sim.particles().len()
        );
        println!(
            "Lajeunesse predicted R_inf = {:.2} cells (real dry sand, aspect ratio {:.1}:1)",
            predicted_r_inf_cells(),
            H0_CELLS / R0_CELLS
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
            mode,
            paused: false,
            step: 0,
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
            .set_camera(&self.queue, GRID as u32, w, h, 0.9, true);
    }

    fn reset(&mut self, mode: Mode) {
        self.sim = make_sim(mode);
        self.mode = mode;
        self.step = 0;
        println!("reset: mode={:?}", mode);
    }

    /// Real, live measurement -- same `max |x - center_x|` metric the
    /// headless Lajeunesse test uses, in cells.
    fn measured_spread_cells(&self) -> f32 {
        let xs = &self.sim.particles().x;
        let n = xs.len() as f32;
        if n == 0.0 {
            return 0.0;
        }
        let center_x = xs.iter().map(|p| p.x).sum::<f32>() / n;
        xs.iter()
            .map(|p| (p.x - center_x).abs())
            .fold(0.0f32, f32::max)
    }

    fn update_and_render(&mut self, window: &Window) {
        if !self.paused {
            self.sim.step();
            self.step += 1;
        }
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

        // Real, cited reference lines: where dry sand of this aspect ratio
        // actually stops IRL (Lajeunesse et al. 2004). Grid-x -> screen-x
        // matches the camera mapping `Renderer::set_camera` itself uses
        // (linear, GRID cells across the viewport) -- computed against
        // egui's own LOGICAL screen rect (points, not physical pixels)
        // inside the closure below: egui's painter takes point-space
        // coordinates and applies `pixels_per_point` itself when
        // rasterizing, so feeding it physical-pixel values directly (as an
        // earlier version of this file did) draws in the wrong place on
        // any display with a scale factor other than 1.0.
        let center_x_cells = GRID as f32 * 0.5;
        let r_inf = predicted_r_inf_cells();

        let raw_input = self.egui_state.take_egui_input(window);
        let fps = self.last_fps;
        let mode = self.mode;
        let step = self.step;
        let elapsed_s = step as f32 * DT_S;
        let measured = self.measured_spread_cells();
        let ratio = measured / r_inf;
        let mut paused = self.paused;
        let mut do_baseline = false;
        let mut do_ngf = false;

        let full_output = self.egui_ctx.run(raw_input, |ctx| {
            // Logical (point-space) width/height -- egui's own painter
            // coordinate system, distinct from the physical-pixel
            // `surface_config` dimensions the GPU camera uses.
            let screen = ctx.content_rect();
            let w = screen.width();
            let h = screen.height();
            let left_line_x = (center_x_cells - r_inf) / GRID as f32 * w;
            let right_line_x = (center_x_cells + r_inf) / GRID as f32 * w;

            let painter = ctx.debug_painter();
            let stroke = egui::Stroke::new(2.0_f32, egui::Color32::from_rgb(255, 60, 60));
            painter.line_segment(
                [egui::pos2(left_line_x, 0.0), egui::pos2(left_line_x, h)],
                stroke,
            );
            painter.line_segment(
                [egui::pos2(right_line_x, 0.0), egui::pos2(right_line_x, h)],
                stroke,
            );
            painter.text(
                egui::pos2(right_line_x + 4.0, 4.0),
                egui::Align2::LEFT_TOP,
                "real dry sand\nstops here",
                egui::FontId::proportional(12.0),
                egui::Color32::from_rgb(255, 120, 120),
            );

            egui::Window::new("Sand: NGF vs baseline")
                .default_pos([10.0, 10.0])
                .default_width(280.0)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.label(format!("fps={fps:.0}  step={step}  t={elapsed_s:.2}s"));
                    ui.separator();
                    ui.label(format!("mode = {mode:?}"));
                    ui.label(format!(
                        "predicted R_inf (Lajeunesse 2004) = {r_inf:.2} cells"
                    ));
                    ui.label(format!(
                        "measured spread              = {measured:.2} cells"
                    ));
                    ui.label(format!("ratio (measured/predicted)    = {ratio:.2}x"));
                    ui.separator();
                    ui.label("Red lines = where real dry sand stops.");
                    ui.label("Watch it blow past them either way -- that");
                    ui.label("gap is the real, still-open accuracy issue.");
                    ui.label("NGF narrows the ratio a few %, doesn't close it.");
                    ui.separator();
                    ui.checkbox(&mut paused, "Paused (or SPACE)");
                    ui.horizontal(|ui| {
                        if ui.button("B: reset baseline").clicked() {
                            do_baseline = true;
                        }
                        if ui.button("N: reset NGF").clicked() {
                            do_ngf = true;
                        }
                    });
                    ui.label("Q quit");
                });
        });
        self.paused = paused;
        if do_baseline {
            self.reset(Mode::Baseline);
        } else if do_ngf {
            self.reset(Mode::Ngf);
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
                    .with_title("emerge -- Sand NGF vs Baseline (GUI)")
                    .with_inner_size(winit::dpi::LogicalSize::new(720u32, 720u32)),
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
                        state: key_state,
                        ..
                    },
                ..
            } => {
                let pressed = key_state == ElementState::Pressed;
                if pressed {
                    match key {
                        KeyCode::KeyB => s.reset(Mode::Baseline),
                        KeyCode::KeyN => s.reset(Mode::Ngf),
                        KeyCode::Space => s.paused = !s.paused,
                        KeyCode::Escape | KeyCode::KeyQ => el.exit(),
                        _ => {}
                    }
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

extern crate emerge_engine as emerge;

/// Real soil-horizon layering -- the O/A/B/C genetic horizon sequence (Jenny 1941,
/// "Factors of Soil Formation"; USDA-NRCS Soil Survey Manual horizon nomenclature),
/// each horizon a REAL material already in this engine, not new physics:
///
///   O (organic litter/humus)      -> DruckerPragerMaterial::low_friction
///                                     (closest existing loose/low-cohesion preset)
///   A (mineral+organic topsoil)   -> GranularFluidMaterial::saturated_loam
///                                     ("loam" IS the real A-horizon texture class)
///   B (clay-illuviated subsoil)   -> NaccMaterial::wet_soil (Cam-Clay -- real clay
///                                     accumulation zone)
///   C (weathered parent material) -> DruckerPragerMaterial::dilatant (denser,
///                                     closer to intact rock than A/O)
///
/// A-horizon note: `GranularFluidMaterial::saturated_loam` was originally found --
/// via this very demo -- to be genuinely UNSTABLE under real gravity (max_speed
/// reaching 80-140 grid-units/s, never settling), a real pre-existing engine bug,
/// not a soil-horizons bug (root-caused and fixed 2026-07-31: `eos_power=7` was the
/// near-incompressible WATER value applied to a 40%-compressible preset, combined
/// with a hardening-scale floor that let dilation soften the material below its own
/// baseline stiffness -- a genuine unbounded positive feedback; see
/// `granular_fluid_saturated_loam_instability_found_2026-07-31` and its follow-up
/// fix memory for the full writeup). Verified settling cleanly at this file's own
/// `young_modulus=1200` before switching back -- the temporary `DruckerPragerMaterial
/// ::cohesionless` substitution is no longer needed.
///
/// Real cited bulk-density RATIOS relative to water=1.0 (same convention
/// `mixture_sand_water.rs` already uses via `mass_override`), web-verified against
/// real soil-science figures (not a single USDA horizon table, which wasn't found
/// in this exact form -- these compose from several independently-confirmed real
/// numbers): organic/peaty soils <0.5 g/cm^3, loam ~1.2-1.5 g/cm^3, clay ~1.0-1.4
/// g/cm^3 (but B-horizon clay is real-world DENSER than surface clay of the same
/// texture, from illuviation/compaction -- the actual reason B differs from A, not
/// texture alone), compact/glacial-till C-horizons specifically measured at
/// 1.76-1.95 g/cm^3. Representative picks within/near each verified range, not
/// universal constants -- real soil depth/density varies hugely by climate/parent
/// material (Jenny's own thesis).
/// Layer THICKNESS ratios (O thin, C thickest) are the real, uncontroversial
/// qualitative ordering pedology gives -- exact depths vary by soil type/location,
/// so these are representative proportions, not a literal profile.
///
/// Horizon colors are representative real pedology description (dark organic O,
/// brown A, reddish-orange B from iron-oxide illuviation, pale weathered C) --
/// NOT literal Munsell soil-color-chart values, which weren't looked up.
///
/// Real interaction: LMB pushes soil aside, revealing the real cross-section of
/// layers as you excavate -- proves the layering isn't a static texture.
///
///   cargo run --example soil_horizons --features "render"
use egui_wgpu::ScreenDescriptor;
use emerge::render::{ColorMode, Renderer};
use emerge::{
    DruckerPragerMaterial, GranularFluidMaterial, NaccMaterial, SimConfig, Simulation,
    SlipBoundary, SpawnRegion,
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
const SPACING: f32 = 0.5;

const O_ID: u32 = 0;
const A_ID: u32 = 1;
const B_ID: u32 = 2;
const C_ID: u32 = 3;

// Real bulk-density ratios vs water=1.0 (USDA-NRCS Soil Survey Manual typical
// ranges, representative midpoints -- see module doc).
const O_DENSITY_RATIO: f32 = 0.2;
const A_DENSITY_RATIO: f32 = 1.2;
const B_DENSITY_RATIO: f32 = 1.5;
const C_DENSITY_RATIO: f32 = 1.8; // within the verified 1.76-1.95 glacial-till C-horizon range

// Column geometry: bottom of C horizon sits just above the floor boundary; total
// soil column depth is split across horizons using the real, uncontroversial
// qualitative ordering (O thin, C thickest) -- see module doc for the caveat on
// exact proportions.
const COLUMN_HALF_WIDTH: i32 = 24;
const SOIL_BOTTOM: f32 = 2.0;
const O_THICKNESS: f32 = 2.0;
const A_THICKNESS: f32 = 8.0;
const B_THICKNESS: f32 = 14.0;
const C_THICKNESS: f32 = 16.0;

const DIG_RADIUS: f32 = 4.0;
const DIG_STRENGTH: f32 = 10.0;

// Real footstep-force probe: hold F at the cursor to press straight down, like a
// creature's foot loading the ground. PRESS_RADIUS approximates a real footprint
// footprint's contact patch relative to this column's own scale.
const PRESS_RADIUS: f32 = 3.0;
const PRESS_FORCE_STEP: f32 = 5.0;
const PRESS_FORCE_MIN: f32 = 5.0;
const PRESS_FORCE_MAX: f32 = 200.0;

struct App {
    window: Option<Arc<Window>>,
    state: Option<State>,
}

struct Diagnostics {
    max_speed: f32,
    non_finite: usize,
    o_count: usize,
    a_count: usize,
    b_count: usize,
    c_count: usize,
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
    pressing: bool,
    press_force: f32,
    // Real sag/absorption measurement state: surface height at the press column
    // captured the instant pressing starts, and the lowest height reached while
    // held -- lets us report BOTH how far it sagged under load and, after release,
    // how much of that sag was permanent (absorbed/plastic) vs recovered
    // (elastic rebound), rather than just "it moved".
    press_baseline_height: Option<f32>,
    press_min_height: f32,
    was_pressing: bool,
    frame: u64,
    fps_timer: std::time::Instant,
    fps_frames: u64,
    last_fps: f32,
}

fn make_sim() -> Simulation {
    let config = SimConfig {
        max_substeps_per_step: 16,
        // Deliberately weak, NOT real IRL gravity (real g_grid ~= 981 via
        // SimConfig::earth) -- tuned down for a calmer, more legible demo at this
        // grid scale, same disclosed convention `basic_showcase.rs`/`fire_spread.rs`
        // already use.
        gravity: Vec2::new(0.0, -0.3),
        ..SimConfig::earth(GRID, 0.01, DT)
    };

    // Stiffness doubled across all four horizons vs the original values (2026-07-31)
    // -- measured, not guessed: a real CFL/substep-headroom sweep at this demo's own
    // dt=0.1/max_substeps_per_step=16 showed EVERY horizon staying at 50-63% of its
    // substep budget at this doubled E (real margin left for interactive dig/press
    // spikes on top of quiescent settling, which is all the sweep itself measured).
    // Real ceiling is higher still (O measured safe to 16x, C to 4x, B to ~4x) --
    // this is a conservative real step, not the maximum, so there's known headroom
    // left if it still feels too soft. Real relative ordering/ratios between
    // horizons (O softest .. C stiffest) preserved exactly, just uniformly doubled.
    // (A-horizon's own headroom wasn't re-measured against this doubled value --
    // it uses a different material now, see below -- but its own isolated settling
    // was directly reverified at this exact E right before switching back.)
    // O: loose organic litter -- closest existing preset, see module doc.
    let o_horizon = DruckerPragerMaterial::low_friction(600.0, 0.3);
    // A: loamy topsoil -- real name-match, now fixed and settling cleanly (see
    // module doc's A-horizon note).
    let a_horizon = GranularFluidMaterial::saturated_loam(1200.0, 0.3);
    // B: clay-illuviated subsoil -- Non-Associated Cam-Clay, real wet-clay regime.
    let b_horizon = NaccMaterial::wet_soil(1800.0, 0.3);
    // C: weathered parent material -- denser, closer to intact rock.
    let c_horizon = DruckerPragerMaterial::dilatant(2400.0, 0.3);

    let mut solver = Simulation::empty(config)
        .with_material(O_ID, Box::new(o_horizon))
        .with_material(A_ID, Box::new(a_horizon))
        .with_material(B_ID, Box::new(b_horizon))
        .with_material(C_ID, Box::new(c_horizon))
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));

    let center_x = GRID as f32 * 0.5;
    let horizons = [
        (C_ID, C_THICKNESS, C_DENSITY_RATIO, SOIL_BOTTOM),
        (
            B_ID,
            B_THICKNESS,
            B_DENSITY_RATIO,
            SOIL_BOTTOM + C_THICKNESS,
        ),
        (
            A_ID,
            A_THICKNESS,
            A_DENSITY_RATIO,
            SOIL_BOTTOM + C_THICKNESS + B_THICKNESS,
        ),
        (
            O_ID,
            O_THICKNESS,
            O_DENSITY_RATIO,
            SOIL_BOTTOM + C_THICKNESS + B_THICKNESS + A_THICKNESS,
        ),
    ];

    for &(material_id, thickness, density_ratio, y_bottom) in &horizons {
        // box_size is in world/grid units directly (same units as box_center), NOT
        // a particle/cell count to be divided by spacing -- spacing only controls
        // how densely particles pack WITHIN that world-space extent.
        let spawn = SpawnRegion {
            spacing: SPACING,
            box_size: IVec2::new(COLUMN_HALF_WIDTH * 2, thickness.round().max(1.0) as i32),
            box_center: Vec2::new(center_x, y_bottom + thickness * 0.5),
            material_id,
            precompute_initial_volumes: true,
            mass_override: Some(density_ratio),
            ..SpawnRegion::for_sim(&config)
        };
        let _ = solver.add_body(spawn);
    }

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
        // Optical params are Beer-Lambert absorption coefficients, not direct RGB:
        // color = exp(-sigma_a), so sigma_a = -ln(target) for a target color.
        // Targets are representative real pedology description, see module doc.
        renderer.set_optical_params(&queue, O_ID as usize, [1.386, 1.715, 2.120]); // dark organic
        renderer.set_optical_params(&queue, A_ID as usize, [0.799, 1.139, 1.609]); // brown loam
        renderer.set_optical_params(&queue, B_ID as usize, [0.511, 1.139, 1.897]); // reddish clay
        renderer.set_optical_params(&queue, C_ID as usize, [0.431, 0.511, 0.693]); // pale weathered rock

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

        let particles = sim.particles();
        let count_of = |id: u32| {
            particles
                .indices()
                .filter(|&i| particles.material_id[i] == id)
                .count()
        };
        println!(
            "soil_horizons: O={} A={} B={} C={} particles  |  LMB dig  F press  R reset  Q quit",
            count_of(O_ID),
            count_of(A_ID),
            count_of(B_ID),
            count_of(C_ID),
        );

        println!(
            "  hold F to press down at cursor (footstep force probe)  \
             [ / ] adjust press force (start {PRESS_FORCE_MIN:.0})"
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
            pressing: false,
            press_force: PRESS_FORCE_MIN,
            press_baseline_height: None,
            press_min_height: f32::INFINITY,
            was_pressing: false,
            frame: 0,
            fps_timer: std::time::Instant::now(),
            fps_frames: 0,
            last_fps: 0.0,
        }
    }

    /// Real surface-height probe: highest y among particles within `PRESS_RADIUS`
    /// of `x_center` -- the actual local ground-surface height at that column,
    /// not a fixed/assumed value, so sag is measured against reality each time.
    fn surface_height_near(&self, x_center: f32) -> f32 {
        self.sim
            .particles()
            .iter()
            .filter(|p| (p.x.x - x_center).abs() <= PRESS_RADIUS)
            .map(|p| p.x.y)
            .fold(f32::NEG_INFINITY, f32::max)
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
        let count_of = |id: u32| {
            particles
                .indices()
                .filter(|&i| particles.material_id[i] == id)
                .count()
        };
        Diagnostics {
            max_speed,
            non_finite,
            o_count: count_of(O_ID),
            a_count: count_of(A_ID),
            b_count: count_of(B_ID),
            c_count: count_of(C_ID),
        }
    }

    fn update_and_render(&mut self, window: &Window) {
        if self.lmb || self.rmb {
            let mag = if self.lmb {
                DIG_STRENGTH
            } else {
                -DIG_STRENGTH
            };
            self.sim
                .apply_radial_impulse(self.cursor_grid(), DIG_RADIUS, mag);
        }

        let press_x = self.cursor_grid().x;
        if self.pressing {
            if self.press_baseline_height.is_none() {
                let h = self.surface_height_near(press_x);
                self.press_baseline_height = Some(h);
                self.press_min_height = h;
                println!(
                    "press start: force={:.0} baseline_height={h:.2}",
                    self.press_force
                );
            }
            self.sim.apply_impulse(
                self.cursor_grid(),
                PRESS_RADIUS,
                Vec2::new(0.0, -self.press_force * DT),
            );
            let h = self.surface_height_near(press_x);
            self.press_min_height = self.press_min_height.min(h);
        }
        // Real sag/absorption report, printed once right as the foot lifts --
        // compares the settled height AFTER release against both the original
        // baseline and the deepest point reached under load, so "how much
        // recovered" and "how much stayed sunk" are both real, measured numbers.
        if self.was_pressing && !self.pressing {
            if let Some(baseline) = self.press_baseline_height {
                let recovered = self.surface_height_near(press_x);
                let max_sag = baseline - self.press_min_height;
                let permanent_sag = baseline - recovered;
                let absorbed_fraction = if max_sag > 1.0e-6 {
                    (permanent_sag / max_sag).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                println!(
                    "press end: force={:.0} max_sag={max_sag:.2} permanent_sag={permanent_sag:.2} \
                     absorbed_fraction={absorbed_fraction:.2} (0=fully elastic rebound, 1=fully absorbed/plastic)",
                    self.press_force
                );
            }
            self.press_baseline_height = None;
            self.press_min_height = f32::INFINITY;
        }
        self.was_pressing = self.pressing;

        self.sim.step();
        self.frame += 1;
        self.fps_frames += 1;
        if self.fps_timer.elapsed().as_secs_f32() >= 2.0 {
            self.last_fps = self.fps_frames as f32 / self.fps_timer.elapsed().as_secs_f32();
            let d = self.diagnostics();
            println!(
                "frame={} fps={:.0} max_speed={:.3} non_finite={} \
                 (should stay small/bounded for a settling soil column -- large/nonzero = explosion)",
                self.frame, self.last_fps, d.max_speed, d.non_finite
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
        let mut press_force = self.press_force;
        let mut reset_clicked = false;

        let full_output = self.egui_ctx.run(raw_input, |ctx| {
            egui::Window::new("Soil Horizons")
                .default_pos([10.0, 10.0])
                .default_width(280.0)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.label(format!("fps={:.0}  frame={}", self.last_fps, self.frame));
                    ui.label(format!(
                        "O={} A={} B={} C={} particles",
                        d.o_count, d.a_count, d.b_count, d.c_count
                    ));
                    ui.separator();
                    ui.label(format!(
                        "max_speed={:.3}  non_finite={}",
                        d.max_speed, d.non_finite
                    ));
                    ui.separator();
                    ui.add(
                        egui::Slider::new(&mut press_force, PRESS_FORCE_MIN..=PRESS_FORCE_MAX)
                            .text("press force ([ / ])"),
                    );
                    ui.label("LMB dig  RMB fill  F press (footstep probe)  R reset  Q quit");
                    if ui.button("Reset").clicked() {
                        reset_clicked = true;
                    }
                });
        });

        self.press_force = press_force;
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
                    .with_title("emerge -- Soil Horizons [O/A/B/C layering]")
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
                    KeyCode::KeyF => s.pressing = pressed,
                    _ if !pressed => {}
                    KeyCode::Escape | KeyCode::KeyQ => el.exit(),
                    KeyCode::KeyR => {
                        s.sim = make_sim();
                        s.frame = 0;
                        println!("reset");
                    }
                    KeyCode::BracketRight => {
                        s.press_force = (s.press_force + PRESS_FORCE_STEP).min(PRESS_FORCE_MAX);
                        println!("press_force={:.0}", s.press_force);
                    }
                    KeyCode::BracketLeft => {
                        s.press_force = (s.press_force - PRESS_FORCE_STEP).max(PRESS_FORCE_MIN);
                        println!("press_force={:.0}", s.press_force);
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

    /// Real regression check on the layering logic itself: every particle's
    /// material must match the REAL horizon its y-position falls in (O topmost,
    /// then A, B, C bottommost), and every horizon must be non-empty -- proves
    /// the depth-based assignment is actually correct, not just "it compiles".
    #[test]
    fn particles_are_assigned_to_the_correct_horizon_by_depth() {
        let sim = make_sim();
        let particles = sim.particles();
        assert!(particles.len() > 0, "soil column must not be empty");

        let c_top = SOIL_BOTTOM + C_THICKNESS;
        let b_top = c_top + B_THICKNESS;
        let a_top = b_top + A_THICKNESS;
        let material_at = |y: f32| -> u32 {
            if y > a_top {
                O_ID
            } else if y > b_top {
                A_ID
            } else if y > c_top {
                B_ID
            } else {
                C_ID
            }
        };

        let mut counts = [0usize; 4];
        for i in particles.indices() {
            let y = particles.x[i].y;
            let id = particles.material_id[i];
            counts[id as usize] += 1;
            // Adjacent horizons are spawned as separate abutting boxes, so a
            // particle can legitimately land exactly on a shared boundary line --
            // accept either horizon on the two sides of that line, not just one.
            let eps = 1.0e-3;
            let candidates = [material_at(y - eps), material_at(y + eps)];
            assert!(
                candidates.contains(&id),
                "particle at y={y:.2} has material_id={id} but its depth allows \
                 only {candidates:?} (c_top={c_top:.1} b_top={b_top:.1} a_top={a_top:.1})"
            );
        }

        for (id, count) in counts.iter().enumerate() {
            assert!(*count > 0, "horizon material_id={id} has zero particles");
        }
    }
}

extern crate emerge_engine as emerge;

use egui_wgpu::ScreenDescriptor;
/// `fire_spread.rs` (real ignition + Fourier heat-driven combustion propagation) with a
/// live egui material picker -- compares how fire spread depends on a material's
/// thermal diffusivity (alpha = k/(rho*c_p)) and ignition point. Same mechanism
/// throughout (`WithLatentHeat` combustion exotherm + `ThermalDiffusion` conduction +
/// `phase_rule` ignition), only the material's own real constants change.
///
/// Real cited constants for all three:
///   - Wood: unchanged from `fire_spread.rs` (conductivity 0.147 W/(m*K), heat_capacity
///     1700 J/(kg*K), density 500 kg/m3 real pine, ignition 603.15K/330C midpoint of the
///     real 300-365C piloted range, combustion -18.5MJ/kg oven-dry wood).
///   - Paper: conductivity 0.05 W/(m*K) (cross-grain, real cited range ~0.05-0.07),
///     heat_capacity 1340 J/(kg*K) (cellulose/paper specific heat, real range
///     ~1300-1500), density 100 kg/m3 -- real, but for LOOSELY CRUMPLED paper (what
///     actually burns), not a pressed stack/cardboard (700-1200): air gaps between
///     sheets give real bulk density ~50-150 kg/m3, same reason snow's bulk density is
///     far below solid ice's. Paper's "catches fast" behavior is mostly a
///     low-thermal-mass effect: alpha=k/(rho*c_p) is inversely proportional to density,
///     so a lighter material heats and diffuses faster for the identical heat input --
///     the same fire-science "thin fuels ignite faster than thick fuels" principle.
///     Ignition 503.15K/230C (real cited piloted-ignition range 218-246C, midpoint).
///     Combustion -16MJ/kg (real cellulose/paper heat of combustion, range ~15-17MJ/kg).
///   - Stone (granite): conductivity 2.5 W/(m*K) (real granite range 2.0-3.5),
///     heat_capacity 790 J/(kg*K) (real granite range ~790-800), density 2700 kg/m3
///     (real granite). Ignition point = f32::INFINITY -- honest, not an arbitrarily
///     high finite number: rock is genuinely non-combustible, there is no real
///     "ignition temperature" to cite, so the phase-rule condition structurally
///     never fires rather than merely being unlikely to.
///
///   cargo run --example fire_spread_gui --features "render"
use emerge::render::{ColorMode, GridVolumeSource, Renderer};
use emerge::thermodynamics::{ThermalConfig, ThermalDiffusion};
use emerge::{
    DruckerPragerMaterial, SimConfig, Simulation, SlipBoundary, SpawnRegion, ViscoelasticMaterial,
    WithLatentHeat,
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

const FUEL_ID: u32 = 0;
const ASH_ID: u32 = 1;

const AMBIENT_K: f32 = 293.15;
// 1/s, Newton cooling (natural convective heat loss to still air). 0.001
// (tau=1000s) is within the real natural-convection range for a wood-sized solid
// in still air, chosen empirically so a several-minute play session shows
// meaningful, differentiated spread across materials.
const COOLING_RATE: f32 = 0.001;
// Real per-material emissivity radiates heat away faster than conduction/combustion
// can build it up at this demo's real dt/dx scale -- measured directly on
// fire_spread.rs's identical wood/thermal setup, including several scaled-down values
// (0.02, 0.01, with/without COOLING_RATE): EVERY nonzero emissivity stalls the fire at
// 6-7% burned within a few hundred seconds, vs. 18%-and-still-climbing with it off.
// Stefan-Boltzmann's T^4 term grows faster than this conduction-only spread mechanic
// can compensate for right in the temperature range spread depends on -- a real
// physical effect, but incompatible with keeping these demos' fire actually spreading.
// Kept at 0.0 so the mechanic stays intact; the real, tested mechanism itself lives in
// `ThermalConfig::emissivity` for scenes where it's a good fit (e.g. lava cooling).
const EMISSIVITY_DEMO_SCALE: f32 = 0.0;

const PLANK_HALF_LEN: i32 = 22;
const PLANK_HALF_HEIGHT: i32 = 4;
const IGNITE_RADIUS: f32 = 2.5;
const MATCH_TEMP: f32 = 973.15;
const IGNITION_HEAT_TRANSFER: f32 = 0.06;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum FuelKind {
    Paper,
    Wood,
    Stone,
}

struct FuelProps {
    conductivity: f32,
    heat_capacity: f32,
    density: f32,
    ignition_k: f32,
    combustion_enthalpy: f32,
    /// Real cited emissivity (Incropera). Scaled by `EMISSIVITY_DEMO_SCALE` for
    /// actual use in `ThermalConfig` -- see that constant's own doc.
    emissivity: f32,
    /// Beer-Lambert absorption coefficient sigma_a, NOT a direct RGB target --
    /// rendered color is `exp(-sigma_a)` per channel (higher sigma_a = more
    /// absorbed = darker). Computed as `-ln(target_srgb)` from the intended
    /// appearance.
    absorption: [f32; 3],
    name: &'static str,
}

impl FuelKind {
    fn props(self) -> FuelProps {
        match self {
            FuelKind::Paper => FuelProps {
                conductivity: 0.05,
                heat_capacity: 1340.0,
                // Real, but for LOOSELY CRUMPLED paper (what actually burns), not a
                // pressed stack/cardboard -- air gaps between sheets give real bulk
                // density ~50-150 kg/m^3, same reason snow's bulk density is far
                // below solid ice's.
                density: 100.0,
                ignition_k: 503.15,
                combustion_enthalpy: -16_000_000.0,
                emissivity: 0.92, // real, white/cream paper (Incropera)
                // Target appearance: pale cream (0.92, 0.90, 0.80).
                // sigma_a = -ln(target), verified via particle_color() to
                // reproduce it (see fire_spread_gui_real_colors test).
                absorption: [0.083, 0.105, 0.223],
                name: "Paper (very flammable)",
            },
            FuelKind::Wood => FuelProps {
                conductivity: 0.147,
                heat_capacity: 1700.0,
                density: 500.0,
                ignition_k: 603.15,
                combustion_enthalpy: -18_500_000.0,
                emissivity: 0.85, // real, wood (Incropera) -- the measured reference value
                // Target appearance: medium brown (0.55, 0.35, 0.20).
                // sigma_a = -ln(target).
                absorption: [0.598, 1.050, 1.609],
                name: "Wood (moderate)",
            },
            FuelKind::Stone => FuelProps {
                conductivity: 2.5,
                heat_capacity: 790.0,
                density: 2700.0,
                // Real, honest: rock is non-combustible -- no finite ignition
                // temperature exists to cite, so the phase rule structurally
                // never fires rather than merely being set improbably high.
                ignition_k: f32::INFINITY,
                combustion_enthalpy: 0.0,
                emissivity: 0.90, // real, rough stone/concrete (Incropera)
                // Target appearance: medium grey (0.50, 0.50, 0.50).
                absorption: [0.693, 0.693, 0.693],
                name: "Stone (fireproof)",
            },
        }
    }
}

fn make_sim(fuel_kind: FuelKind) -> Simulation {
    let fuel = fuel_kind.props();
    let config = SimConfig {
        max_substeps_per_step: 16,
        // Deliberately weak, NOT real IRL gravity (real g_grid ~= 981 via
        // SimConfig::earth) -- tuned down for a calmer, more legible demo at
        // this grid scale. Disclosed, deferred: basic_sand_gui.rs's
        // gravity_fraction slider is the real-IRL-with-live-control
        // pattern, not yet ported to every plain example.
        gravity: Vec2::new(0.0, -0.08),
        ..SimConfig::earth(GRID, 0.01, DT)
    };

    let thermal = ThermalDiffusion::new(
        ThermalConfig {
            conductivity: fuel.conductivity,
            heat_capacity: fuel.heat_capacity,
            density: fuel.density,
            ambient: AMBIENT_K,
            grid_cell_size: config.dx_meters,
            cooling_rate: COOLING_RATE,
            emissivity: fuel.emissivity * EMISSIVITY_DEMO_SCALE,
        },
        config.grid_res,
    );

    let spawn = SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new(2 * PLANK_HALF_LEN, 2 * PLANK_HALF_HEIGHT),
        box_center: Vec2::new(GRID as f32 * 0.5, GRID as f32 * 0.4),
        material_id: FUEL_ID,
        precompute_initial_volumes: true,
        ..SpawnRegion::for_sim(&config)
    };

    // ViscoelasticMaterial (Kelvin-Voigt), not pure NeoHookean -- undamped elastic
    // solids bounce indefinitely off a frictionless boundary. Viscosity=100 settles
    // on first landing; same grid-native stiffness tier as fire_spread.rs (real
    // GPa-scale stiffness is incompatible with this grid's CFL).
    let solid = ViscoelasticMaterial::new(100.0, 50.0, 100.0);
    let ash = WithLatentHeat::new(
        DruckerPragerMaterial::low_friction(266.7, 0.333),
        fuel.combustion_enthalpy,
    );

    let ignition_k = fuel.ignition_k;
    let mut solver = Simulation::new(config, spawn)
        .with_default_material(Box::new(solid))
        .with_material(ASH_ID, Box::new(ash))
        .with_thermal(thermal)
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)))
        .with_phase_rule(move |p| {
            if p.material_id == FUEL_ID && p.temperature > ignition_k {
                Some(ASH_ID)
            } else {
                None
            }
        });

    for t in solver.particles_mut().temperature.iter_mut() {
        *t = AMBIENT_K;
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
    cursor_pos: [f32; 2],
    lmb: bool,
    fuel_kind: FuelKind,
    frame: u64,
    fps_timer: std::time::Instant,
    fps_frames: u64,
    last_fps: f32,
    log_timer: std::time::Instant,
    burned_count: usize,
    grid_bridge_buf: wgpu::Buffer,
    material_mass_bridge_buf: wgpu::Buffer,
    grid_volume_mode: bool,
}

/// `render_grid_volume`'s GPU shader reads a separate GPU-resident buffer from the
/// regular splat path -- `Renderer::set_optical_params` takes `&Queue` and uploads
/// immediately so both paths stay in sync.
fn set_optical_params(renderer: &mut Renderer, queue: &wgpu::Queue, fuel_kind: FuelKind) {
    renderer.set_optical_params(queue, FUEL_ID as usize, fuel_kind.props().absorption);
    renderer.set_optical_params(queue, ASH_ID as usize, [0.4, 0.4, 0.4]);
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
        let fuel_kind = FuelKind::Wood;
        let sim = make_sim(fuel_kind);
        let mut renderer = Renderer::new(&device, sim.particles().len(), fmt);
        renderer.set_camera(&queue, GRID as u32, size.width, size.height, 0.6, true);
        renderer.set_color_mode(ColorMode::ByPhysics);
        set_optical_params(&mut renderer, &queue, fuel_kind);

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
            "fire_spread_gui: {} particles  |  click to ignite (real match, {MATCH_TEMP}K)  |  \
             G grid-volume  R reset  Q quit",
            sim.particles().len()
        );

        const RENDER_MATERIAL_SLOTS: u64 = 16;
        let grid_bridge_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fire_spread_gui_grid_bridge"),
            size: (GRID * GRID * 4 * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let material_mass_bridge_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fire_spread_gui_material_mass_bridge"),
            size: (GRID as u64 * GRID as u64 * RENDER_MATERIAL_SLOTS) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

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
            fuel_kind,
            frame: 0,
            fps_timer: std::time::Instant::now(),
            fps_frames: 0,
            last_fps: 0.0,
            log_timer: std::time::Instant::now(),
            burned_count: 0,
            grid_bridge_buf,
            material_mass_bridge_buf,
            grid_volume_mode: false,
        }
    }

    fn upload_grid_volume_bridge(&self) {
        const SLOTS: usize = 16;
        let grid = self.sim.grid();
        let mut dense = vec![0f32; GRID * GRID * 4];
        for y in 0..GRID {
            for x in 0..GRID {
                let idx = y * GRID + x;
                dense[idx * 4 + 2] = grid.mass_at(IVec2::new(x as i32, y as i32));
            }
        }
        // Real mass-weighted temperature scatter into the previously-unused
        // channel 0 -- same P2G scatter convention `ThermalDiffusion` already
        // uses, real fix for `grid_volume.wgsl`'s own disclosed "blackbody not
        // ported, no per-pixel temperature" gap (confirmed live via a user
        // side-by-side screenshot). Grid-cell mass already exists above;
        // temperature isn't tracked per-cell by the CPU solver, so scatter it
        // here the same way the solver's own P2G would (nearest-cell,
        // mass-weighted) directly from particle state.
        let particles = self.sim.particles();
        for i in 0..particles.x.len() {
            let p = particles.x[i];
            let cx = (p.x.round() as i32).clamp(0, GRID as i32 - 1) as usize;
            let cy = (p.y.round() as i32).clamp(0, GRID as i32 - 1) as usize;
            let idx = cy * GRID + cx;
            dense[idx * 4] += particles.mass[i] * particles.temperature[i];
        }
        self.queue
            .write_buffer(&self.grid_bridge_buf, 0, bytemuck::cast_slice(&dense));

        let mut material_mass = vec![0f32; GRID * GRID * SLOTS];
        for i in 0..particles.x.len() {
            let p = particles.x[i];
            let cx = (p.x.round() as i32).clamp(0, GRID as i32 - 1) as usize;
            let cy = (p.y.round() as i32).clamp(0, GRID as i32 - 1) as usize;
            let slot = (particles.material_id[i] as usize) % SLOTS;
            material_mass[(cy * GRID + cx) * SLOTS + slot] += particles.mass[i];
        }
        self.queue.write_buffer(
            &self.material_mass_bridge_buf,
            0,
            bytemuck::cast_slice(&material_mass),
        );
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

    fn ignite_at_cursor(&mut self) {
        let center = self.cursor_grid();
        self.sim.particles_mut().for_each_mut(|p| {
            if p.material_id == FUEL_ID && (p.x - center).length() <= IGNITE_RADIUS {
                p.temperature += IGNITION_HEAT_TRANSFER * (MATCH_TEMP - p.temperature);
            }
        });
    }

    fn fuel_temp_stats(&self) -> (f32, f32, usize) {
        let mut max_t = f32::NEG_INFINITY;
        let mut sum_t = 0.0f32;
        let mut n = 0usize;
        for p in self
            .sim
            .particles()
            .iter()
            .filter(|p| p.material_id == FUEL_ID)
        {
            max_t = max_t.max(p.temperature);
            sum_t += p.temperature;
            n += 1;
        }
        let avg_t = if n > 0 { sum_t / n as f32 } else { f32::NAN };
        (max_t, avg_t, n)
    }

    fn reset(&mut self) {
        self.sim = make_sim(self.fuel_kind);
        set_optical_params(&mut self.renderer, &self.queue, self.fuel_kind);
        self.frame = 0;
        self.burned_count = 0;
        println!("reset -- material: {}", self.fuel_kind.props().name);
    }

    fn update_and_render(&mut self, window: &Window) {
        if self.lmb {
            self.ignite_at_cursor();
        }
        self.sim.step();
        self.burned_count = self
            .sim
            .particles()
            .iter()
            .filter(|p| p.material_id == ASH_ID)
            .count();
        self.frame += 1;
        self.fps_frames += 1;
        if self.fps_timer.elapsed().as_secs_f32() >= 1.0 {
            self.last_fps = self.fps_frames as f32 / self.fps_timer.elapsed().as_secs_f32();
            self.fps_timer = std::time::Instant::now();
            self.fps_frames = 0;
        }
        // Periodic console log alongside the egui panel -- the panel is ephemeral
        // (nothing to scroll back through), this gives a persistent terminal record.
        if self.log_timer.elapsed().as_secs_f32() >= 2.0 {
            let (max_t, avg_t, fuel_n) = self.fuel_temp_stats();
            let ignition_k = self.fuel_kind.props().ignition_k;
            let gap = if ignition_k.is_finite() {
                format!("{:.1}K", ignition_k - max_t)
            } else {
                "n/a (fireproof)".to_string()
            };
            println!(
                "[{}] frame={} fps={:.0}  burned={}/{}  fuel remaining n={fuel_n} \
                 max_T={max_t:.1}K avg_T={avg_t:.1}K  gap_to_ignition={gap}",
                self.fuel_kind.props().name,
                self.frame,
                self.last_fps,
                self.burned_count,
                self.sim.particles().len(),
            );
            self.log_timer = std::time::Instant::now();
        }

        let output = match self.surface.get_current_texture() {
            Ok(t) => t,
            Err(_) => return,
        };
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        if self.grid_volume_mode {
            self.upload_grid_volume_bridge();
            self.renderer.render_grid_volume(
                &self.device,
                &self.queue,
                GridVolumeSource {
                    grid: &self.grid_bridge_buf,
                    material_mass: &self.material_mass_bridge_buf,
                    material_mass_enabled: true,
                },
                &view,
                true,
            );
        } else {
            self.renderer
                .render(&self.device, &self.queue, self.sim.particles(), &view, true);
        }

        // --- egui panel ---
        let raw_input = self.egui_state.take_egui_input(window);
        let fps = self.last_fps;
        let (max_t, avg_t, fuel_n) = self.fuel_temp_stats();
        let ignition_k = self.fuel_kind.props().ignition_k;
        let total = self.sim.particles().len();
        let burned = self.burned_count;
        let mut selected = self.fuel_kind;
        let mut reset_clicked = false;
        let mut grid_volume_mode = self.grid_volume_mode;

        let full_output = self.egui_ctx.run(raw_input, |ctx| {
            egui::Window::new("Fire Spread")
                .default_pos([10.0, 10.0])
                .default_width(280.0)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.label(format!("fps={fps:.0}"));
                    ui.label(format!(
                        "burned={burned}/{total}  fuel remaining n={fuel_n}"
                    ));
                    let gap = if ignition_k.is_finite() {
                        format!("{:.1}K", ignition_k - max_t)
                    } else {
                        "never (fireproof)".to_string()
                    };
                    ui.label(format!(
                        "max_T={max_t:.1}K avg_T={avg_t:.1}K  gap to ignition={gap}"
                    ));
                    ui.separator();
                    ui.label("Material:");
                    for kind in [FuelKind::Paper, FuelKind::Wood, FuelKind::Stone] {
                        if ui
                            .radio_value(&mut selected, kind, kind.props().name)
                            .changed()
                        {
                            // handled after closure via `selected` diffing below
                        }
                    }
                    ui.separator();
                    ui.checkbox(&mut grid_volume_mode, "Grid-volume render (or press G)");
                    ui.separator();
                    ui.label("Click/hold to ignite  R reset  Q quit");
                    if ui.button("Reset").clicked() {
                        reset_clicked = true;
                    }
                });
        });

        self.grid_volume_mode = grid_volume_mode;
        if selected != self.fuel_kind {
            self.fuel_kind = selected;
            self.reset();
        } else if reset_clicked {
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

struct App {
    window: Option<Arc<Window>>,
    state: Option<State>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        let w = Arc::new(
            el.create_window(
                winit::window::WindowAttributes::default()
                    .with_title("emerge -- Fire Spread (GUI, material picker)")
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
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => {
                s.lmb = state == ElementState::Pressed;
            }
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
                    KeyCode::KeyR if pressed => {
                        s.reset();
                    }
                    KeyCode::KeyG if pressed => {
                        s.grid_volume_mode = !s.grid_volume_mode;
                        println!(
                            "grid-volume render: {}",
                            if s.grid_volume_mode { "on" } else { "off" }
                        );
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

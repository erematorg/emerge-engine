extern crate emerge_engine as emerge;

/// Real fire spread through wood -- composition of already-shipped mechanisms, zero
/// new engine infrastructure: `add_phase_rule`/`phase_transition` (wood -> ash past
/// ignition), `WithLatentHeat` (exothermic combustion releases real heat), and
/// `ThermalDiffusion` (Fourier diffusion spreads heat to neighbors, which can then
/// cross ignition themselves -- emergent chain reaction, not scripted).
///
/// Honest scope: heat-DRIVEN ignition PROPAGATION through a solid, not full
/// combustion chemistry (no O2 consumption, no smoke/soot, no gas-phase flame
/// front). Next tier would be Gillespie/SSA reaction kinetics on top of
/// `ScalarDiffusionField` -- not attempted here.
///
/// Real cited constants:
///   - Piloted ignition 300-365 C -> midpoint 330 C = 603.15 K.
///   - Oven-dry wood heat of combustion ~18.5 MJ/kg.
///   - Wood thermal conductivity ~0.147 W/(m*K) across the grain (yellow pine).
///   - Wood specific heat ~1700 J/(kg*K) -- standard engineering estimate, disclosed
///     as an estimate rather than a precise citation.
///   - Wood/ash mechanical stiffness is grid-native (`NeoHookeanMaterial`-tier Lame
///     values), not literal real-Pa: real GPa-scale wood stiffness is incompatible
///     with this scene's fine grid (`dx_meters=0.01`) under explicit-MPM CFL --
///     needs a coarser grid or an implicit integrator. Relative stiffness
///     (wood >> ash) is still respected. `ThermalConfig`'s real SI conductivity/
///     heat-capacity values are separate and unaffected.
///   - `WithLatentHeat` must wrap the material being transitioned INTO (ash), not
///     the one left behind (wood) -- `phase_transition`/`add_phase_rule` apply the
///     NEW material's `latent_heat()`.
///
/// Spread is visually gradual, not an instant flash, for real physical reasons:
/// wood's thermal diffusivity (alpha = k / (rho*c_p)) is genuinely low -- wood is a
/// real insulator, same reason a log takes real time to catch fully alight.
///
///   cargo run --example fire_spread --features "render"
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

const WOOD_ID: u32 = 0;
const ASH_ID: u32 = 1;

const AMBIENT_K: f32 = 293.15; // 20 C room temperature
const IGNITION_K: f32 = 603.15; // 330 C -- midpoint of the real 300-365 C piloted-ignition range
const COMBUSTION_ENTHALPY: f32 = -18_500_000.0; // J/kg, oven-dry wood ~18.5 MJ/kg, exothermic
// W/(m*K), real -- yellow pine across grain measures 0.147 W/(m*K), same species
// this demo's E=9.5 GPa/rho=500 kg/m3 pine values already assume.
const WOOD_CONDUCTIVITY: f32 = 0.147;
const WOOD_HEAT_CAPACITY: f32 = 1700.0; // J/(kg*K), standard engineering estimate for dry wood
// kg/m^3, real pine density -- feeds ThermalConfig::density, which directly scales
// diffusion rate (alpha = k / (rho*c_p)).
const WOOD_DENSITY: f32 = 500.0;
// 1/s, Newton cooling (natural convective heat loss to still air). 0.001
// (tau=1000s) is within the real natural-convection range for a wood-sized solid
// in still air, and chosen empirically so a several-minute play session shows
// meaningful spread -- real physics alone reads as too slow for that timescale.
const COOLING_RATE: f32 = 0.001;
// Real wood emissivity is ~0.85 (Incropera). Measured directly (headless comparisons,
// same constants as this file, 2000s sim-time each): at the real value the plank NEVER
// ignites (radiative loss equilibrates around 340K). Tried scaling it down (0.02, 0.01,
// combined with/without COOLING_RATE) -- EVERY nonzero value stalls the fire at 6-7%
// burned within a few hundred seconds, vs. 18%-and-still-climbing with emissivity=0.0.
// Root cause: Stefan-Boltzmann's T^4 term grows much faster than this plank's
// conduction can compensate for right in the 500-600K range spread depends on -- a
// real physical effect (radiative-loss-driven flame extinction), but this demo's
// conduction-only spread mechanic is too delicately balanced to carry ANY of it.
// Kept off (0.0) so the actual "fire spreads" mechanic stays intact; the real,
// tested Stefan-Boltzmann mechanism itself lives in `ThermalConfig::emissivity` for
// scenes where it's a good fit (e.g. lava cooling), just not this one.
const WOOD_EMISSIVITY: f32 = 0.0;

const PLANK_HALF_LEN: i32 = 22;
const PLANK_HALF_HEIGHT: i32 = 4;
const IGNITE_RADIUS: f32 = 2.5;
// 700 C (973.15K) -- real match-flame temperature is 600-800 C, using the midpoint.
const MATCH_TEMP: f32 = 973.15;
// Dimensionless per-render-frame contact-heating fraction (Newton relaxation toward
// MATCH_TEMP: T += rate*(target-T)). Applied per RENDER frame, not per physics
// substep -- `ignite_at_cursor` runs outside `sim.step()`'s dt. No literature source
// for a match-to-wood contact heat-transfer coefficient; tuned so a brief click
// barely warms the wood while a sustained hold ramps it toward ignition.
const IGNITION_HEAT_TRANSFER: f32 = 0.06;

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
    frame: u64,
    fps_timer: std::time::Instant,
    fps_frames: u64,
    burned_count: usize,
    /// CPU-simulation grid-volume render bridge (G to toggle): this scene runs on the
    /// CPU `Simulation`, which has no GPU-resident grid buffer the way `GpuSimulation`
    /// does, so `render_grid_volume` has nothing to read directly. Rebuilt from the
    /// CPU solver's `Grid`/`Particles` each frame and uploaded -- extra per-frame
    /// cost, but reuses the same GPU render path/shader.
    grid_bridge_buf: wgpu::Buffer,
    /// Per-cell per-material mass, built via a simplified nearest-cell scatter (not
    /// P2G's full quadratic B-spline kernel) -- good enough for dominant-material
    /// color selection, not a physics-accuracy claim. Read-only by rendering.
    material_mass_bridge_buf: wgpu::Buffer,
    grid_volume_mode: bool,
}

fn make_sim() -> Simulation {
    let config = SimConfig {
        // Matches comparable "solid elastic" demos (basic_jellies_gpu/basic_showcase
        // use 12-16) -- substep count alone can't compensate for wrong stiffness scale.
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
            conductivity: WOOD_CONDUCTIVITY,
            heat_capacity: WOOD_HEAT_CAPACITY,
            density: WOOD_DENSITY,
            ambient: AMBIENT_K,
            // Must match the sim's real dx_meters -- ThermalConfig::grid_cell_size
            // requires this, else diffusion rate is silently wrong.
            grid_cell_size: config.dx_meters,
            cooling_rate: COOLING_RATE,
            emissivity: WOOD_EMISSIVITY,
        },
        config.grid_res,
    );

    // The SI property system (`Elastic{e_pa,nu,rho_kg_m3}`) is incompatible with this
    // scene's fine grid scale -- both materials below use plain grid-native
    // constructors instead, matching `basic_snow.rs`'s convention.
    let spawn = SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new(2 * PLANK_HALF_LEN, 2 * PLANK_HALF_HEIGHT),
        box_center: Vec2::new(GRID as f32 * 0.5, GRID as f32 * 0.4),
        material_id: WOOD_ID,
        precompute_initial_volumes: true,
        ..SpawnRegion::for_sim(&config)
    };

    // ViscoelasticMaterial (Kelvin-Voigt), not pure NeoHookean -- undamped elastic
    // wood bounces indefinitely off a frictionless boundary, which real wood doesn't.
    // Viscosity=100 settles on first landing.
    let wood = ViscoelasticMaterial::new(100.0, 50.0, 100.0);
    // Ash: crumbly granular -- burned wood collapses into loose material, not just
    // changes color. `low_friction` is `basic_snow.rs`'s preset at this grid/dt/dx.
    //
    // `WithLatentHeat` must wrap the transition TARGET (ash), not the source (wood) --
    // `phase_transition`/`add_phase_rule` apply the NEW material's `latent_heat()`.
    let ash = WithLatentHeat::new(
        DruckerPragerMaterial::low_friction(266.7, 0.333),
        COMBUSTION_ENTHALPY,
    );

    let mut solver = Simulation::new(config, spawn)
        .with_default_material(Box::new(wood))
        .with_material(ASH_ID, Box::new(ash))
        .with_thermal(thermal)
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)))
        .with_phase_rule(|p| {
            if p.material_id == WOOD_ID && p.temperature > IGNITION_K {
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
        // ByThermal is a pure blackbody-glow mode -- near-black at rest (normalizes
        // against a 1500K ceiling). ByPhysics gives a real base material color
        // (Beer-Lambert absorption) plus the same thermal glow layered on top once hot.
        renderer.set_color_mode(ColorMode::ByPhysics);
        // Optical params are Beer-Lambert absorption coefficients, not direct RGB:
        // color = exp(-sigma_a), so sigma_a = -ln(target) for a target color.
        renderer.set_optical_params(&queue, WOOD_ID as usize, [0.598, 1.050, 1.609]);
        renderer.set_optical_params(&queue, ASH_ID as usize, [0.4, 0.4, 0.4]);
        println!(
            "fire_spread: {} wood particles  |  click to ignite (real match, {MATCH_TEMP}K)  |  \
             G grid-volume  R reset  Q quit",
            sim.particles().len()
        );
        println!(
            "ignition point={IGNITION_K}K (330C, real piloted-ignition range 300-365C)  \
             combustion enthalpy={COMBUSTION_ENTHALPY}J/kg (real, oven-dry wood ~18.5MJ/kg)"
        );
        const RENDER_MATERIAL_SLOTS: u64 = 16;
        let grid_bridge_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fire_spread_grid_bridge"),
            size: (GRID * GRID * 4 * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let material_mass_bridge_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fire_spread_material_mass_bridge"),
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
            cursor_pos: [0.0; 2],
            lmb: false,
            frame: 0,
            fps_timer: std::time::Instant::now(),
            fps_frames: 0,
            burned_count: 0,
            grid_bridge_buf,
            material_mass_bridge_buf,
            grid_volume_mode: false,
        }
    }

    /// Rebuilds `grid_bridge_buf`/`material_mass_bridge_buf` from the CPU solver's
    /// current state and uploads them -- see those fields' own doc for the real,
    /// disclosed cost/approximation. Only called when `grid_volume_mode` is on, so
    /// the default splat-mode path pays zero extra cost.
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
        // channel 0 -- same fix as `fire_spread_gui.rs`'s own bridge, see
        // `grid_volume.wgsl`'s own doc for the real formula this feeds.
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

        // Simplified nearest-cell scatter (real, disclosed approximation -- see
        // material_mass_bridge_buf's own doc): good enough for a 2-material dominant-
        // color decision, not claiming P2G-kernel accuracy.
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
            if p.material_id == WOOD_ID && (p.x - center).length() <= IGNITE_RADIUS {
                p.temperature += IGNITION_HEAT_TRANSFER * (MATCH_TEMP - p.temperature);
            }
        });
    }

    /// (max_temp, avg_temp, count) over still-unburned wood -- lets you tell "still
    /// heating up, will ignite eventually" from "genuinely stalled, not receiving heat"
    /// without guessing from the burn count alone.
    fn wood_temp_stats(&self) -> (f32, f32, usize) {
        let mut max_t = f32::NEG_INFINITY;
        let mut sum_t = 0.0f32;
        let mut n = 0usize;
        for p in self
            .sim
            .particles()
            .iter()
            .filter(|p| p.material_id == WOOD_ID)
        {
            max_t = max_t.max(p.temperature);
            sum_t += p.temperature;
            n += 1;
        }
        let avg_t = if n > 0 { sum_t / n as f32 } else { f32::NAN };
        (max_t, avg_t, n)
    }

    fn update_and_render(&mut self) {
        // Held-button pattern (matches basic_snow.rs) re-applies every frame using the
        // current cursor_pos, not frozen at click-moment -- lets you drag to ignite.
        if self.lmb {
            self.ignite_at_cursor();
        }
        self.sim.step();
        let before = self.burned_count;
        self.burned_count = self
            .sim
            .particles()
            .iter()
            .filter(|p| p.material_id == ASH_ID)
            .count();
        if self.burned_count != before && self.burned_count.is_multiple_of(50) {
            let (max_t, avg_t, wood_n) = self.wood_temp_stats();
            println!(
                "frame={} burned={}/{}  remaining wood: max_T={max_t:.1}K avg_T={avg_t:.1}K \
                 (ignition={IGNITION_K}K)",
                self.frame,
                self.burned_count,
                self.sim.particles().len(),
            );
            let _ = wood_n;
        }
        self.frame += 1;
        self.fps_frames += 1;
        if self.fps_timer.elapsed().as_secs_f32() >= 2.0 {
            let fps = self.fps_frames as f32 / self.fps_timer.elapsed().as_secs_f32();
            let (max_t, avg_t, wood_n) = self.wood_temp_stats();
            let max_speed = self
                .sim
                .particles()
                .iter()
                .map(|p| p.v.length())
                .fold(0.0f32, f32::max);
            println!(
                "frame={} fps={:.0} burned={}/{}  remaining wood: n={wood_n} max_T={max_t:.1}K \
                 avg_T={avg_t:.1}K (ignition={IGNITION_K}K, gap={:.1}K)  max_speed={max_speed:.3} \
                 (should stay small/bounded for a resting plank -- large/NaN = explosion)",
                self.frame,
                fps,
                self.burned_count,
                self.sim.particles().len(),
                IGNITION_K - max_t,
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
        output.present();
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        let w = Arc::new(
            el.create_window(
                winit::window::WindowAttributes::default()
                    .with_title("emerge -- Fire Spread [real ignition + Fourier diffusion]")
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
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => match key {
                KeyCode::Escape | KeyCode::KeyQ => el.exit(),
                KeyCode::KeyR => {
                    s.sim = make_sim();
                    s.frame = 0;
                    s.burned_count = 0;
                    println!("reset");
                }
                KeyCode::KeyG => {
                    s.grid_volume_mode = !s.grid_volume_mode;
                    println!(
                        "grid-volume render: {}",
                        if s.grid_volume_mode { "on" } else { "off" }
                    );
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

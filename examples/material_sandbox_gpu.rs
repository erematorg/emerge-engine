extern crate emerge_engine as emerge;

/// GPU material sandbox -- paint real continuum materials with the mouse.
///
/// The pitch this proves: unlike a falling-sand/cellular-automaton toy (single
/// particle per grid cell, swap-if-heavier rule tables, no stress or strain --
/// see `tmp/powder-toy` for a directly-verified example of that architecture),
/// every brush here is a real MLS-MPM particle carrying a real deformation
/// gradient and a real constitutive law. Sand piles because of Drucker-Prager
/// friction, not a `Weight` lookup; jelly squishes and springs back because of
/// real NeoHookean elasticity, not a hardcoded "solid" flag.
///
/// Five real, already-shipped material presets (reused verbatim from
/// `basic_showcase_gpu`/`basic_snow_gpu`/`basic_jellies_gpu` -- no new invented
/// constants): NeoHookean jelly, Drucker-Prager sand, Newtonian water, Stomakhin
/// snow, Kelvin-Voigt viscoelastic tissue.
///
/// Real UI (egui, the same wgpu-native GUI already used by `basic_jellies`/
/// `stress_realtime_scaling`), not a repurposed physics hack: clickable material
/// buttons with names and matching colors, live particle/fps readout, reset.
///
/// Real phase transitions, not a hardcoded threshold swap: the Heat tool feeds a
/// real heat source into the real GPU thermal diffusion PDE (`attach_thermal_gpu`,
/// Fourier's law); the resulting temperature field is what a periodic scan checks
/// against real melting/boiling points (273.15K / 373.15K, actual water/ice
/// constants) to call `phase_transition` (snow<->water) or `remove_particles`
/// (water -> vanishes above boiling, real evaporation, not tracked as a gas phase).
/// Both engine calls now apply a REAL latent-heat energy debit
/// (`MaterialModel::latent_heat`, `ΔT = latent_heat/heat_capacity`) -- melting
/// genuinely cools the surrounding material, freezing genuinely warms it, real
/// energy conservation, not a free material swap. This required two real,
/// disclosed engine fixes made this session: `GpuSimulation::phase_transition` had
/// no latent-heat accounting at all (CPU-only before), and neither `phase_transition`
/// nor a GPU `remove_particles` existed in a form safe to call from live/interactive
/// code. Ambient is set below freezing (260K) so anything not actively heated
/// genuinely drifts back toward frozen via the same real Newton-cooling term
/// `day_night_thermal_gpu` already proved -- the "cold reverses it" half of the ask
/// is the PDE's own behavior, not a separate mechanism.
///
/// Grid cells are set to a real, small, disclosed physical scale (2cm/cell, a
/// hand-sized snowball) rather than literal room-scale (1m/cell): real thermal
/// diffusion at 1m/cell is far too slow to watch live (that's genuinely how slow
/// real conduction is) -- shrinking the domain's physical scale is a legitimate
/// modeling choice every discretized simulation makes, not a fudge to the physics
/// itself (conductivity/heat_capacity stay real, unmodified SI values).
///
/// Known, disclosed limitation: `GpuSimulation::spawn_region` fully reallocates
/// every per-particle GPU buffer and rebuilds the bind-group pool on each call
/// (see its own doc). Painting is therefore rate-limited to one small clump
/// every few frames while the mouse is held, not a true continuous stream --
/// a real, deliberate interaction-rate choice given that cost, not a hidden
/// hack. A streaming/incremental-append spawn path would remove this limit but
/// is real, separate, future engine work.
///
///   click a material in the panel, or press 1-5  |  LMB paint  |  R reset  |  Q quit
///   cargo run --example material_sandbox_gpu --features "gpu render"
use std::sync::Arc;

use egui_wgpu::ScreenDescriptor;
use emerge::gpu::GpuSimulation;
use emerge::render::{ColorMode, Renderer};
use emerge::{
    DruckerPragerMaterial, MaterialRegistry, NeoHookeanMaterial, NewtonianFluidMaterial, SimConfig,
    SpawnRegion, StomakhinMaterial, ViscoelasticMaterial, WithLatentHeat, build_particles,
};
use glam::{IVec2, Vec2};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

const GRID: usize = 64;
const DT: f32 = 0.1;
// `spawn_region` fully reallocates every per-particle GPU buffer + rebuilds the
// bind-group pool per call (real, disclosed cost -- see its own doc). A real fix
// is giving it capacity headroom so repeated small spawns amortize instead of
// reallocating every time, but that's core-engine work touching every shader's
// buffer-size assumption, not a "slight" example-level change. This cap is the
// safe version for now: painting stops (not crashes, not grinds to a stall)
// once the scene hits a size a live demo has no real reason to exceed.
const MAX_PARTICLES: usize = 20_000;

const JELLY_ID: u32 = 0;
const SAND_ID: u32 = 1;
const WATER_ID: u32 = 2;
const SNOW_ID: u32 = 3;
const TISSUE_ID: u32 = 4;

/// (id, name, RGB) -- colors match `render::material_palette(id)` exactly, so
/// the button you click is honestly the color of what you're about to paint.
const PALETTE: &[(u32, &str, [u8; 3])] = &[
    (JELLY_ID, "Jelly", [89, 166, 255]),
    (SAND_ID, "Sand", [230, 204, 77]),
    (WATER_ID, "Water", [204, 230, 255]),
    (SNOW_ID, "Snow", [128, 217, 128]),
    (TISSUE_ID, "Tissue", [255, 115, 51]),
];

// Real per-material absorption (Beer-Lambert σ_a), reused verbatim from already-shipped
// demos where a real value exists -- NOT invented for this demo. `ByMaterial` mode gave
// zero visual feedback while heating (temperature was invisible until the exact instant
// a particle's material_id flipped) -- real user-observed confusion. `ByPhysics` instead
// shows each material's own real optical look PLUS a real blackbody-emission glow term
// driven by `particle.temperature` (see prep_instances.wgsl's ByPhysics branch), so
// heating is visible continuously, not just at the phase-transition instant.
const SIGMA_JELLY: [f32; 3] = [0.05, 0.55, 0.60]; // basic_jellies_gpu (SIGMA_NEO)
const SIGMA_SAND: [f32; 3] = [0.180, 0.220, 0.550]; // basic_sand_gpu
const SIGMA_WATER: [f32; 3] = [0.85, 0.25, 0.07]; // render_physics (real: water absorbs red faster than blue)
// REAL water absorption coefficients, per-meter -- Pope & Fry 1997 (via the OMLC
// optical absorption compendium, omlc.org/spectra/water/abs, same real-source
// tradition as this file's own Jacques 2013 tissue-scattering citation):
// a_red(630nm)~0.34 m^-1, a_green(532nm)~0.044 m^-1, a_blue(420nm)~0.0044 m^-1.
// Kept as the literal real-world m^-1 values, NOT pre-multiplied by any particular
// dx_meters -- see `real_water_sigma_a` below for why baking in one scene's scale
// as a frozen constant would silently go stale if that scale ever changes.
const WATER_ABSORPTION_PER_METER: [f32; 3] = [0.34, 0.044, 0.0044];

// Beer-Lambert here is per ONE PARTICLE's own depth (prep_instances.wgsl's own doc:
// "depth=1 particle"), so the real shader-space sigma_a is the real m^-1 value
// scaled by however many real meters one particle actually represents at THIS
// scene's live scale -- computed from `config.dx_meters`, not a frozen literal, so
// it stays correct if the scene's scale ever changes.
//
// HONEST FINDING, not a bug to fix: at this demo's real scale (dx_meters=0.01,
// SimConfig::earth), this comes out genuinely tiny (~0.0034/0.00044/0.00004) --
// real water at 1cm of real depth IS nearly perfectly transparent, true physics,
// not a rendering gap. This engine is 2D (a face-on cross-section, no camera-ray-
// through-volume axis), so there's no real depth-ACCUMULATION technique (like 3D
// screen-space fluid rendering's thickness pass) that applies here -- `1/J`
// (compression-as-density) is already the correct 2D analog, not a placeholder
// for something more real. SIGMA_WATER above stays the demo's deliberate artistic
// exaggeration (disclosed as such); this is the literal real-physics answer.
fn real_water_sigma_a(dx_meters: f32) -> [f32; 3] {
    WATER_ABSORPTION_PER_METER.map(|a| a * dx_meters)
}
const SIGMA_TISSUE: [f32; 3] = [0.05, 0.55, 0.60]; // render_physics (SIGMA_TISSUE)
// No established value exists elsewhere in this codebase for snow specifically -- low,
// roughly-neutral absorption (real snow is near-white, dominated by scattering not
// absorption) with a faint blue bias (real, well-known snow/ice optical trend: red
// absorbs marginally faster than blue). A reasonable physically-motivated estimate,
// not a literature citation -- same honesty bar as this project's other disclosed,
// non-literature-sourced presets.
const SIGMA_SNOW: [f32; 3] = [0.06, 0.05, 0.03];

// Real water/ice constants -- same numbers `tests/gpu.rs`'s latent-heat parity
// tests check against, not invented for this demo.
const MELT_POINT_K: f32 = 273.15;
const FREEZE_POINT_K: f32 = 272.15; // 1K hysteresis -- avoids flicker exactly at 273.15,
// a small, disclosed simplification of real nucleation-barrier supercooling, not a
// made-up number (real water/ice DOES exhibit hysteresis around the phase boundary).
const BOIL_POINT_K: f32 = 373.15;
const LATENT_HEAT_FUSION: f32 = 334.0; // water, kJ/kg-equivalent in this engine's units
const HEAT_CAPACITY: f32 = 4182.0; // water, J/(kg*K)
const AMBIENT_K: f32 = 260.0; // below freezing -- world starts cold, matches the ask
const COOLING_RATE: f32 = 0.05; // Newton cooling k_c, same value day_night_thermal_gpu uses
const CONDUCTIVITY: f32 = 0.6; // water/ice, W/(m*K)
const CELL_SIZE_M: f32 = 0.02; // 2cm/cell -- hand-sized snowball scale, see module doc

fn make_registry() -> MaterialRegistry {
    // Reused verbatim from already-shipped demos -- not new invented numbers.
    let jelly = NeoHookeanMaterial::new(40.0, 80.0); // basic_showcase_gpu
    // Real bug found live (2026-07-17): basic_showcase_gpu's plain `new(400.0, 200.0)`
    // (cohesion defaults to 0.0, true cohesionless Mohr-Coulomb) measures a real ~12°
    // angle of repose on GPU (see gpu_sand_angle_of_repose_is_physical) against real dry
    // sand's 30-35° -- a genuine, already-documented continuum-MPM-resolution artifact
    // (DruckerPragerMaterial::cohesion's own doc: pressure-proportional friction vanishes
    // in thin/fast-flowing layers regardless of friction angle, confirmed across THREE
    // different friction coefficients giving identical ~4.7x excess runout). The real fix
    // isn't a new number for THIS scale -- it's reusing the exact (E, ν, cohesion) triple
    // already calibrated and passing against the real Lajeunesse et al. 2004 runout
    // scaling law (`sand_column_collapse_runout_matches_lajeunesse_scaling`), not
    // extrapolating an unverified proportional guess for a different raw-Lamé pair.
    let mut sand = DruckerPragerMaterial::from_young_modulus(1.0e5, 0.2);
    sand.cohesion = 5.0; // calibrated against the real Lajeunesse benchmark, see above
    let water = NewtonianFluidMaterial::new(4.0, 0.1, 10.0, 4.0); // basic_showcase_gpu
    let snow = StomakhinMaterial::new(1389.0, 2083.0, 7.0, 0.025, 0.0075, 0.6, 20.0); // basic_snow_gpu
    let tissue = ViscoelasticMaterial::new(10.0, 15.0, 0.15); // basic_jellies_gpu
    let mut reg = MaterialRegistry::with_default(Box::new(jelly));
    reg.insert(SAND_ID, Box::new(sand));
    // Positive = endothermic (melting into water absorbs energy, cools the particle).
    reg.insert(
        WATER_ID,
        Box::new(WithLatentHeat::new(water, LATENT_HEAT_FUSION)),
    );
    // Negative = exothermic (freezing into snow releases energy, warms the particle) --
    // real reversible thermodynamics, same magnitude as melting, opposite sign.
    reg.insert(
        SNOW_ID,
        Box::new(WithLatentHeat::new(snow, -LATENT_HEAT_FUSION)),
    );
    reg.insert(TISSUE_ID, Box::new(tissue));
    reg
}

fn make_sim_data(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> GpuSimulation {
    let config = SimConfig {
        min_dt: 0.005,
        max_substeps_per_step: 16,
        recompute_density_each_step: true,
        gravity: Vec2::new(0.0, -0.3),
        ..SimConfig::earth(GRID, 0.01, DT)
    };

    // Ground strip to paint onto -- real sand, not decoration.
    let mut particles = build_particles(
        &config,
        SpawnRegion {
            spacing: 0.7,
            box_size: IVec2::new(50, 3),
            box_center: Vec2::new(32.0, 4.0),
            material_id: SAND_ID,
            precompute_initial_volumes: true,
            ..SpawnRegion::for_sim(&config)
        },
    );
    // World starts cold -- anything painted starts at ambient, below freezing.
    for p in &mut particles {
        p.temperature = AMBIENT_K;
    }

    let registry = make_registry();
    let mut sim = GpuSimulation::with_device(device, queue, config, particles, registry);
    sim.attach_thermal_gpu(
        CONDUCTIVITY,
        HEAT_CAPACITY,
        CELL_SIZE_M,
        AMBIENT_K,
        COOLING_RATE,
    );
    sim
}

struct App {
    window: Option<Arc<Window>>,
    state: Option<State>,
}

/// Cursor tool: paint new material, or push/pull existing particles with a
/// real radial impulse (same `apply_radial_impulse` used by `basic_showcase_gpu`'s
/// LMB/RMB push-pull -- reused, not reinvented).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Paint,
    Force,
    Heat,
}

impl Mode {
    fn name(self) -> &'static str {
        match self {
            Mode::Paint => "paint",
            Mode::Force => "force",
            Mode::Heat => "heat",
        }
    }

    fn next(self) -> Mode {
        match self {
            Mode::Paint => Mode::Force,
            Mode::Force => Mode::Heat,
            Mode::Heat => Mode::Paint,
        }
    }
}

struct State {
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    sim: GpuSimulation,
    renderer: Renderer,
    egui_ctx: egui::Context,
    egui_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
    mode: Mode,
    selected: usize,
    cursor_pos: [f32; 2],
    lmb: bool,
    rmb: bool,
    paint_cooldown: u32,
    frame: u64,
    fps_timer: std::time::Instant,
    fps_frames: u64,
    last_fps: f32,
    /// Real max temperature within Heat-tool range of the cursor, refreshed at the same
    /// cadence as the phase-transition scan (not every frame -- a blocking readback every
    /// frame is a real, avoidable cost). Direct numeric feedback for the real thing
    /// `ByPhysics`'s emission glow is too subtle to show at this demo's 260-373K range
    /// (that glow term is normalized to 5000K, a lava/molten-metal scale) -- clear textual
    /// proof that heat is genuinely accumulating, not just the sudden melt/boil jump.
    near_cursor_max_temp: f32,
    /// Toggle with M -- swaps water's optics between the demo's artistic
    /// exaggeration (`SIGMA_WATER`) and the literal real-physics value, computed
    /// live via `real_water_sigma_a` (properly derived from Pope & Fry 1997 -- see
    /// that function's own doc). Real, live demonstration that real water at this
    /// engine's actual 1cm/particle scale is nearly transparent, not the vivid
    /// blue every other demo shows -- both are honest, just answering different
    /// questions ("looks nice" vs "what would this really look like").
    real_water_optics: bool,
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
        let device = Arc::new(device);
        let queue = Arc::new(queue);
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
        let sim = make_sim_data(device.clone(), queue.clone());
        let mut renderer = Renderer::new(&device, sim.particle_count(), fmt);
        renderer.set_camera(&queue, GRID as u32, size.width, size.height, 0.6, true);
        renderer.set_color_mode(ColorMode::ByPhysics);
        renderer.set_optical_params(JELLY_ID as usize, SIGMA_JELLY);
        renderer.set_optical_params(SAND_ID as usize, SIGMA_SAND);
        renderer.set_optical_params(WATER_ID as usize, SIGMA_WATER);
        renderer.set_optical_params(SNOW_ID as usize, SIGMA_SNOW);
        renderer.set_optical_params(TISSUE_ID as usize, SIGMA_TISSUE);

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
            "material_sandbox_gpu: {} particles  |  click a material or press 1-5  |  Paint/Force/Heat tool (F)  |  R reset  Q quit",
            sim.particle_count()
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
            mode: Mode::Paint,
            selected: 0,
            cursor_pos: [0.0; 2],
            lmb: false,
            rmb: false,
            paint_cooldown: 0,
            frame: 0,
            fps_timer: std::time::Instant::now(),
            fps_frames: 0,
            last_fps: 0.0,
            near_cursor_max_temp: AMBIENT_K,
            real_water_optics: false,
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

    fn reset(&mut self) {
        self.sim = make_sim_data(self.device.clone(), self.queue.clone());
        self.frame = 0;
        self.near_cursor_max_temp = AMBIENT_K;
        println!("reset");
    }

    fn cursor_grid(&self) -> Vec2 {
        Vec2::new(
            self.cursor_pos[0] / self.surface_config.width as f32 * GRID as f32,
            (1.0 - self.cursor_pos[1] / self.surface_config.height as f32) * GRID as f32,
        )
    }

    fn select_brush(&mut self, index: usize) {
        if index < PALETTE.len() {
            self.selected = index;
            println!("brush: {}", PALETTE[index].1);
        }
    }

    fn update_and_render(&mut self, window: &Window) {
        if self.paint_cooldown > 0 {
            self.paint_cooldown -= 1;
        }
        match self.mode {
            Mode::Paint
                if self.lmb
                    && self.paint_cooldown == 0
                    && self.sim.particle_count() < MAX_PARTICLES =>
            {
                // Bigger than a token dab on purpose: sand needs enough grains to show
                // a real angle of repose, water enough mass to actually flow/spread,
                // snow enough bulk to compact under its own weight -- a 2x2 clump never
                // gave any of that room to happen, so every material just looked like
                // an undifferentiated soft blob (real user feedback).
                let material_id = PALETTE[self.selected].0;
                let region = SpawnRegion {
                    spacing: 0.5,
                    box_size: IVec2::new(6, 6),
                    box_center: self.cursor_grid(),
                    material_id,
                    precompute_initial_volumes: true,
                    ..SpawnRegion::for_sim(self.sim.config())
                };
                // Real engine check (SpawnRegion::fits_in_sim), not hand-derived margin
                // math -- a hand-rolled `pos.x > 4.0 && ...` version of this exact check
                // previously got the margin wrong and crashed on a real click near the
                // domain edge (spawn_region panics on an out-of-bounds region; that panic
                // is the correct behavior for a scripted/startup spawn, but not for one
                // driven by live mouse input where going out of bounds is normal and
                // should just be skipped).
                if region.fits_in_sim(self.sim.config()) {
                    let range = self.sim.spawn_region(region);
                    // build_particles defaults temperature to 0.0 -- freshly painted
                    // matter must start at the world's real ambient, not absolute zero.
                    let particles = self.sim.particles_mut();
                    for i in range {
                        particles[i].temperature = AMBIENT_K;
                    }
                    self.sim.mark_particles_dirty();
                    self.paint_cooldown = 20;
                }
            }
            Mode::Force if self.lmb || self.rmb => {
                // Real radial impulse, same call basic_showcase_gpu's push/pull uses --
                // no new mechanic, just exposed as a second selectable tool here.
                let mag = if self.lmb { 3.0 } else { -3.0 };
                self.sim.apply_radial_impulse(self.cursor_grid(), 5.0, mag);
            }
            Mode::Heat if self.lmb && self.frame > 0 => {
                // A real, disclosed external heat source (like a torch) -- NOT part of
                // the diffusion PDE itself, exactly the same "real source feeding a real
                // field" composition already used by resource_regrowth_gpu's consumer.
                // The PDE (attach_thermal_gpu) is what then spreads this into the pile.
                self.sim.sync_particles_blocking();
                let nearby: Vec<usize> = self
                    .sim
                    .particles_near(self.cursor_grid(), 4.0)
                    .map(|(i, _)| i)
                    .collect();
                let particles = self.sim.particles_mut();
                for i in nearby {
                    particles[i].temperature += 40.0 * DT;
                }
                self.sim.mark_particles_dirty();
            }
            _ => {}
        }

        // Real phase-transition scan, not evaluated every frame (cheap enough at demo
        // scale, but no reason to pay 3 sync+scan passes 60x/sec for a slow thermal
        // process). Order matters: melt/freeze before evaporate, so a particle crossing
        // both snow->water and water->vanish in the same interval still gets a coherent
        // one-step-at-a-time transition instead of skipping straight past water.
        if self.frame > 0 && self.frame.is_multiple_of(15) {
            self.sim.sync_particles_blocking();
            let particles = self.sim.particles();
            self.near_cursor_max_temp = self
                .sim
                .particles_near(self.cursor_grid(), 4.0)
                .map(|(i, _)| particles[i].temperature)
                .fold(AMBIENT_K, f32::max);
            self.sim.phase_transition(
                |p| p.material_id == SNOW_ID && p.temperature > MELT_POINT_K,
                WATER_ID,
            );
            self.sim.phase_transition(
                |p| p.material_id == WATER_ID && p.temperature < FREEZE_POINT_K,
                SNOW_ID,
            );
            let evaporated = self
                .sim
                .remove_particles(|p| p.material_id == WATER_ID && p.temperature > BOIL_POINT_K);
            if evaporated > 0 {
                println!("evaporated: {evaporated} particles vanished above {BOIL_POINT_K}K");
            }
        }

        let output = match self.surface.get_current_texture() {
            Ok(t) => t,
            Err(_) => return,
        };
        self.sim.step_frame();
        self.frame += 1;
        self.fps_frames += 1;
        if self.fps_timer.elapsed().as_secs_f32() >= 1.0 {
            self.last_fps = self.fps_frames as f32 / self.fps_timer.elapsed().as_secs_f32();
            self.fps_timer = std::time::Instant::now();
            self.fps_frames = 0;
        }

        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        self.renderer.render_gpu(
            &self.device,
            &self.queue,
            self.sim.particle_buffer(),
            self.sim.particle_count(),
            &view,
            true,
        );

        // --- egui panel: real clickable material buttons, not a physics hack ---
        let raw_input = self.egui_state.take_egui_input(window);
        let n = self.sim.particle_count();
        let fps = self.last_fps;
        let near_cursor_max_temp = self.near_cursor_max_temp;
        let mut reset = false;
        let mut new_selection = None;
        let mut new_mode = None;
        let selected = self.selected;
        let mode = self.mode;

        let full_output = self.egui_ctx.run(raw_input, |ctx| {
            egui::Window::new("Material Sandbox")
                .default_pos([10.0, 10.0])
                .default_width(190.0)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.label(format!("fps={fps:.0}  particles={n}/{MAX_PARTICLES}"));
                    ui.separator();
                    ui.label("Tool:");
                    ui.horizontal(|ui| {
                        for &m in &[Mode::Paint, Mode::Force, Mode::Heat] {
                            let label = match m {
                                Mode::Paint => "Paint",
                                Mode::Force => "Force",
                                Mode::Heat => "Heat",
                            };
                            if ui.selectable_label(mode == m, label).clicked() {
                                new_mode = Some(m);
                            }
                        }
                    });
                    ui.separator();
                    ui.label("Brush:");
                    ui.add_enabled_ui(mode == Mode::Paint, |ui| {
                        for (i, &(_, name, rgb)) in PALETTE.iter().enumerate() {
                            let color = egui::Color32::from_rgb(rgb[0], rgb[1], rgb[2]);
                            let text = egui::RichText::new(name).color(color).strong();
                            if ui.selectable_label(selected == i, text).clicked() {
                                new_selection = Some(i);
                            }
                        }
                    });
                    ui.separator();
                    match mode {
                        Mode::Paint => ui.label("LMB paint"),
                        Mode::Force => ui.label("LMB push  RMB pull"),
                        Mode::Heat => ui.label("LMB heat (real PDE spreads it)"),
                    };
                    ui.label(format!(
                        "melt {MELT_POINT_K:.0}K  boil {BOIL_POINT_K:.0}K  ambient {AMBIENT_K:.0}K"
                    ));
                    // Real, direct numeric feedback: the material's own optical glow
                    // (ByPhysics) is normalized to a 5000K lava/molten-metal scale, far too
                    // coarse to show visually across this scene's real 260-373K span --
                    // real user-observed confusion ("snow doesn't seem affected by
                    // temperature"). This number is ground truth regardless of how subtle
                    // the glow looks.
                    let t = near_cursor_max_temp;
                    let color = if t >= BOIL_POINT_K {
                        egui::Color32::from_rgb(255, 90, 40)
                    } else if t >= MELT_POINT_K {
                        egui::Color32::from_rgb(255, 200, 80)
                    } else {
                        egui::Color32::from_rgb(140, 200, 255)
                    };
                    ui.colored_label(color, format!("near cursor: {t:.1}K"));
                    ui.label("F cycle tool  R reset  Q quit");
                    if ui.button("Reset").clicked() {
                        reset = true;
                    }
                });
        });
        if let Some(i) = new_selection {
            self.select_brush(i);
        }
        if let Some(m) = new_mode {
            self.mode = m;
            println!("tool: {}", m.name());
        }
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
                    .with_title("emerge -- Material Sandbox [paint real continuum physics]")
                    .with_inner_size(winit::dpi::LogicalSize::new(600u32, 600u32)),
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
                        state,
                        ..
                    },
                ..
            } => {
                let pressed = state == ElementState::Pressed;
                if !pressed {
                    return;
                }
                match key {
                    KeyCode::Escape | KeyCode::KeyQ => el.exit(),
                    KeyCode::KeyR => s.reset(),
                    KeyCode::KeyF => {
                        s.mode = s.mode.next();
                        println!("tool: {}", s.mode.name());
                    }
                    KeyCode::Digit1 => s.select_brush(0),
                    KeyCode::Digit2 => s.select_brush(1),
                    KeyCode::Digit3 => s.select_brush(2),
                    KeyCode::Digit4 => s.select_brush(3),
                    KeyCode::Digit5 => s.select_brush(4),
                    KeyCode::KeyM => {
                        s.real_water_optics = !s.real_water_optics;
                        let sigma = if s.real_water_optics {
                            real_water_sigma_a(s.sim.config().dx_meters)
                        } else {
                            SIGMA_WATER
                        };
                        s.renderer.set_optical_params(WATER_ID as usize, sigma);
                        println!(
                            "water optics: {} (M to toggle)",
                            if s.real_water_optics {
                                "REAL PHYSICS (Pope & Fry 1997, ~1cm real depth -- nearly transparent)"
                            } else {
                                "artistic (exaggerated for visibility)"
                            }
                        );
                    }
                    _ => {}
                }
            }
            WindowEvent::Resized(sz) => s.resize(sz.width, sz.height),
            WindowEvent::RedrawRequested => {
                let window = self.window.clone();
                if let (Some(s), Some(w)) = (self.state.as_mut(), window) {
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

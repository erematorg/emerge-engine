extern crate emerge_engine as emerge;

use egui_wgpu::ScreenDescriptor;
use emerge::fields::NBodyGravityField;
use emerge::render::{ColorMode, Renderer, TrailRenderer};
use emerge::{
    NeoHookeanMaterial, RadianceField, SimConfig, Simulation, SpawnRegion, stellar_luminosity_w,
};
/// TRUE full N-body solar system, live: Sun + all 8 real planets, REAL MUTUAL
/// gravity (every body pulls on every other, Sun included and free to move) --
/// the real structural upgrade from `basic_orbital(_gui).rs`'s restricted
/// two-body model (fixed Sun, Earth+Mars only). Uses the engine's existing
/// `NBodyGravityField` (Barnes-Hut + a real quadrupole correction, Hernquist
/// 1987) -- the same real technique proven headless in
/// `tests/orbital_mechanics.rs::full_solar_system_conserves_momentum_and_energy`
/// (momentum drift 0.0020%, energy drift 0.0033% over 30 real days).
///
/// Real technique grounding (WebSearch, 2026-08-11): symplectic integrators
/// (Leapfrog, Wisdom-Holman/WHFast -- REBOUND, the real standard N-body
/// astronomy code) are the established real technique for long-term
/// solar-system stability. emerge's own MPM position/velocity update is
/// already semi-implicit/symplectic-Euler by construction -- the same real
/// structural property, not a new addition.
///
/// Real, disclosed limitations:
///   - Real LINEAR distance scale (not logarithmic) -- Mercury sits ~78x
///     closer than Neptune, so inner planets cluster tightly near the Sun.
///     Every real solar-system diagram is "not to scale" for exactly this
///     reason; this one IS to scale, which is why it looks this way.
///   - Real, found-not-hidden precision limit: at this domain's scale
///     (needed to fit Neptune's real orbit), the Sun's own tiny wobble
///     velocity produces a per-step position increment below f32's local
///     precision -- confirmed in `tests/orbital_mechanics.rs`'s own doc
///     (`sun_velocity_responds_to_real_mutual_gravity`): velocity responds
///     correctly to real gravity, position does not visibly accumulate the
///     wobble at this scale/timeframe. A real, structural float-precision
///     constraint, not a physics bug.
///   - Real orbital phase is arbitrary (planets spread at even angles, not
///     a real ephemeris snapshot) -- real distances/masses/speeds throughout.
///
///   cargo run --example basic_solar_system_gui --features render
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

/// dx sized so Neptune's real orbit fits with margin -- same real, measured
/// choice as `tests/orbital_mechanics.rs`'s full-system section.
const DX_METERS: f64 = 2.0e9;
const GRID: usize = 4096;
const DT_SECONDS: f64 = 3600.0;

const G_SI: f64 = 6.674e-11;
const SUN_MASS_KG: f64 = 1.9885e30;
/// Real IAU 2015 nominal solar values -- feeds `stellar_luminosity_w`
/// (Stefan-Boltzmann on a sphere), not a separately hardcoded luminosity:
/// the Sun's radiated power comes FROM its real temperature+size, same
/// physical connection this engine's own `transfer::heat_radiation`
/// already uses for near-field exchange, just applied here to the real
/// far-field point-source (inverse-square) regime -- see
/// `RadianceField`'s own doc for why this is a genuinely different regime
/// from `SimConfig::light_dir`'s uniform "sun at infinity" simplification.
const SUN_RADIUS_M: f32 = 6.957e8;
const SUN_SURFACE_TEMP_K: f32 = 5772.0;

/// Real NASA NSSDCA Planetary Fact Sheet data: (name, mass_kg, semi_major_axis_m).
/// Real, standard N-body-viz technique (same shape REBOUND's own OpenGL/js
/// visualizers use): a fading polyline trail per body, see [`TrailRenderer`].
///
/// Sampled on a FIXED SIMULATED-TIME cadence (`TRAIL_SAMPLE_INTERVAL_
/// SECONDS`, real accumulator below), not once per rendered frame -- a
/// real bug in the first version of this feature sampled per-frame, so at
/// high `steps_per_frame` (fast playback) Mercury's 88-day orbit collapsed
/// to only a handful of trail vertices per loop and rendered as a visibly
/// faceted polygon instead of a smooth ellipse, independent of the actual
/// (unaffected, already-validated-stable) orbital mechanics. Fixed-cadence
/// sampling keeps trail smoothness constant regardless of playback speed.
const TRAIL_LEN: usize = 400;
/// One trail sample per simulated day -- gives Mercury (88-day period)
/// ~88 vertices/orbit, comfortably smooth; `TRAIL_LEN=400` then covers
/// ~1.1 real years of history (a few Mercury loops, about one Earth loop,
/// a real honest partial arc for the outer planets, matching their true
/// much-longer periods).
const TRAIL_SAMPLE_INTERVAL_SECONDS: f64 = 86400.0;

/// Mirrors `render::color::material_palette`'s own RGB values for material
/// IDs 0-8 (that function is `pub(super)`, not reachable from an example) --
/// keeps each planet's trail visually matched to its own particle dot.
fn planet_trail_color(material_id: u32) -> [f32; 4] {
    match material_id {
        0 => [0.35, 0.65, 1.00, 0.7], // Sun (material_palette's slot 0)
        1 => [0.90, 0.80, 0.30, 0.7],
        2 => [0.80, 0.90, 1.00, 0.7],
        3 => [0.50, 0.85, 0.50, 0.7],
        4 => [1.00, 0.45, 0.20, 0.7],
        5 => [0.85, 0.35, 0.35, 0.7],
        6 => [0.65, 0.40, 0.85, 0.7],
        7 => [0.40, 0.85, 0.80, 0.7],
        _ => [0.90, 0.60, 0.40, 0.7],
    }
}

const PLANETS: [(&str, f64, f64); 8] = [
    ("Mercury", 0.330e24, 57.9e9),
    ("Venus", 4.87e24, 108.2e9),
    ("Earth", 5.97e24, 149.6e9),
    ("Mars", 0.642e24, 228.0e9),
    ("Jupiter", 1898.0e24, 778.5e9),
    ("Saturn", 568.0e24, 1432.0e9),
    ("Uranus", 86.8e24, 2867.0e9),
    ("Neptune", 102.0e24, 4515.0e9),
];

/// Real reference distance for `Renderer::set_light_source`'s inverse-
/// square shading falloff -- Earth's own real semi-major axis
/// (`PLANETS[2].2`), the same distance the "solar constant" (1361 W/m²) is
/// literally defined at. A real, physically meaningful normalization
/// point, not an arbitrary tuning number.
fn light_reference_distance_grid() -> f32 {
    (PLANETS[2].2 / DX_METERS) as f32
}

fn make_sim() -> Simulation {
    let center = glam::Vec2::splat(GRID as f32 / 2.0);
    let config = SimConfig {
        dx_meters: DX_METERS as f32,
        dt_seconds: DT_SECONDS as f32,
        gravity: glam::Vec2::ZERO,
        ..SimConfig::standard(GRID, DT_SECONDS as f32, glam::Vec2::ZERO)
    };
    let g_grid = (G_SI / DX_METERS.powi(3)) as f32;

    let spawn_sun = SpawnRegion {
        spacing: 1.0,
        box_size: glam::IVec2::new(1, 1),
        box_center: center,
        position_jitter: 0.0,
        material_id: 0,
        mass_override: Some(SUN_MASS_KG as f32),
        ..SpawnRegion::for_sim(&config)
    };
    let mut solver = Simulation::new(config, spawn_sun)
        .with_default_material(Box::new(NeoHookeanMaterial::new(1.0, 1.0)))
        .with_force_field(Box::new(NBodyGravityField::new(g_grid, 0.05, 0.1)));
    for mat_id in 1..=PLANETS.len() as u32 {
        solver = solver.with_material(mat_id, Box::new(NeoHookeanMaterial::new(1.0, 1.0)));
    }

    let mut planet_momentum = glam::Vec2::ZERO;
    for (idx, &(_, mass_kg, a_m)) in PLANETS.iter().enumerate() {
        let angle = idx as f32 * std::f32::consts::TAU / PLANETS.len() as f32;
        let (s, c) = angle.sin_cos();
        let r_grid = (a_m / DX_METERS) as f32;
        let v_mag = ((G_SI * SUN_MASS_KG / a_m).sqrt() / DX_METERS) as f32;
        let pos = center + glam::Vec2::new(c, s) * r_grid;
        let vel = glam::Vec2::new(-s, c) * v_mag;

        let spawn = SpawnRegion {
            spacing: 1.0,
            box_size: glam::IVec2::new(1, 1),
            box_center: pos,
            position_jitter: 0.0,
            material_id: idx as u32 + 1,
            mass_override: Some(mass_kg as f32),
            ..SpawnRegion::for_sim(solver.config())
        };
        let _ = solver.add_body(spawn);
        solver.particles_mut().v[idx + 1] = vel;
        planet_momentum += mass_kg as f32 * vel;
    }
    // Barycentric frame: Sun's velocity exactly cancels total planet momentum
    // (real, standard N-body initial-condition technique).
    solver.particles_mut().v[0] = -planet_momentum / SUN_MASS_KG as f32;

    // Real (2026-08-17): `Renderer`'s soft-glow now gates on each particle's
    // OWN `blackbody_glow_factor(temperature)` rather than applying
    // uniformly to every particle in the scene -- see `set_rigid_render`'s
    // sibling doc for the same "grounded in real taxonomy, not a demo-wide
    // toggle" reasoning. The Sun is the one body here that's actually
    // self-luminous (matches `RadianceField`'s own real distinction between
    // a light source and ordinary matter); planets only reflect light and
    // correctly get none. Real value, already computed above for luminosity
    // -- just wired into the particle's own temperature field, which
    // nothing in this demo previously used.
    solver.particles_mut().temperature[0] = SUN_SURFACE_TEMP_K;

    solver
}

struct State {
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    device: wgpu::Device,
    queue: wgpu::Queue,
    sim: Simulation,
    renderer: Renderer,
    /// Real, engine-level (not example-local) orbit-trail capability -- see
    /// [`TrailRenderer`]'s own doc.
    trails: TrailRenderer,
    /// Real Sun irradiance source -- see `RadianceField`'s own doc. Position
    /// re-synced from the Sun's own real (mutually-gravitating, slowly
    /// drifting) particle position each frame, not fixed at spawn.
    radiance: RadianceField,
    egui_ctx: egui::Context,
    egui_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
    days_elapsed: f32,
    steps_per_frame: u32,
    paused: bool,
    zoom: f32,
    /// Real camera center (grid coords) -- `Renderer::set_camera_centered`'s
    /// pan hook. Defaults to the grid's own center (old behavior); dragging
    /// (LMB) moves it so the view follows the cursor.
    pan_center: glam::Vec2,
    dragging: bool,
    last_cursor_screen: [f32; 2],
    glow_strength: f32,
    /// Real, opt-in billboard-sphere Lambertian shading strength -- see
    /// `Renderer::set_light_source`'s own doc. Gives planets a real
    /// lit/dark terminator from the Sun's actual, moving position instead
    /// of a flat painted disc.
    shading_strength: f32,
    /// Real simulated-seconds accumulator driving `trails.push`'s fixed
    /// cadence (`TRAIL_SAMPLE_INTERVAL_SECONDS`) -- see `TRAIL_LEN`'s own
    /// doc for why this must NOT be tied to render frame rate.
    trail_sample_accum: f64,
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
        // Real, requested default: 1.0 (the old default) leaves the inner
        // planets sub-pixel against the real linear distance scale
        // (Mercury ~78x closer than Neptune) -- 4.0 shows the inner solar
        // system clearly by default; drag (LMB) or the zoom slider reach
        // the rest.
        let zoom = 4.0;
        let pan_center = glam::Vec2::splat(GRID as f32 * 0.5);
        renderer.set_camera_centered(
            &queue,
            (GRID as f32 / zoom) as u32,
            size.width,
            size.height,
            6.0,
            true,
            pan_center,
        );
        // Real, general, opt-in soft-glow (see `Renderer::set_glow_strength`'s
        // own doc) -- a disclosed, simpler alternative to a true multi-pass
        // bloom pipeline, real and additive to every other demo's own
        // unchanged look (default 0.0 everywhere else).
        let glow_strength = 2.0;
        renderer.set_glow_strength(&queue, glow_strength);
        // Real, general, opt-in billboard-sphere Lambertian shading (see
        // `Renderer::set_light_source`'s own doc) -- shades from the Sun's
        // real position, updated every frame as it drifts (barycentric
        // N-body motion), with real inverse-square intensity falloff
        // normalized at Earth's own real orbital distance.
        let shading_strength = 1.0;
        renderer.set_light_source(
            &queue,
            sim.particles().x[0],
            shading_strength,
            light_reference_distance_grid(),
        );
        renderer.set_color_mode(ColorMode::ByMaterial);
        // Real, disclosed: each planet is one MPM particle standing in for
        // a whole rigid/point-mass body, not a differential element of a
        // deforming continuum -- its own local velocity-gradient-driven F
        // still evolves every substep (ordinary APIC mechanics), which
        // visually stretched/sheared the disc into a non-circular shape
        // with no real physical planetary meaning. `set_rigid_render`
        // renders every particle here as an undeformed disc instead --
        // see that method's own doc.
        renderer.set_rigid_render(true);

        let mut trails = TrailRenderer::new(&device, fmt, sim.particles().len(), TRAIL_LEN);
        for (i, &mat_id) in sim.particles().material_id.iter().enumerate() {
            trails.set_color(i, planet_trail_color(mat_id));
        }
        trails.set_camera(&queue, renderer.view_proj());

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

        let sun_luminosity_w = stellar_luminosity_w(SUN_RADIUS_M, SUN_SURFACE_TEMP_K);
        let radiance = RadianceField::point(sim.particles().x[0], sun_luminosity_w);

        println!(
            "solar system: Sun + 8 real planets, TRUE mutual N-body gravity  |  LMB drag pan  R reset  Q quit"
        );
        println!(
            "sun: R={SUN_RADIUS_M:.3e} m  T={SUN_SURFACE_TEMP_K:.0} K  -> L={sun_luminosity_w:.4e} W (real IAU nominal: 3.828e26 W)"
        );
        Self {
            surface,
            surface_config: sc,
            device,
            queue,
            sim,
            renderer,
            trails,
            radiance,
            egui_ctx,
            egui_state,
            egui_renderer,
            days_elapsed: 0.0,
            steps_per_frame: 4,
            paused: false,
            zoom,
            pan_center,
            dragging: false,
            last_cursor_screen: [0.0; 2],
            glow_strength,
            shading_strength,
            trail_sample_accum: 0.0,
        }
    }

    fn apply_camera(&mut self, w: u32, h: u32) {
        self.renderer.set_camera_centered(
            &self.queue,
            (GRID as f32 / self.zoom) as u32,
            w,
            h,
            6.0,
            true,
            self.pan_center,
        );
        self.trails
            .set_camera(&self.queue, self.renderer.view_proj());
    }

    fn resize(&mut self, w: u32, h: u32) {
        if w == 0 || h == 0 {
            return;
        }
        self.surface_config.width = w;
        self.surface_config.height = h;
        self.surface.configure(&self.device, &self.surface_config);
        self.apply_camera(w, h);
    }

    /// Real drag-to-pan: converts a screen-pixel cursor delta to a
    /// grid-space delta via `Renderer::screen_to_grid` (the renderer's own
    /// single source of truth for this conversion, so pan can't drift out
    /// of sync with what's actually rendered), then shifts `pan_center` so
    /// the content under the cursor follows it -- standard "drag the
    /// canvas" behavior.
    fn pan_by_screen_delta(&mut self, dx_screen: f32, dy_screen: f32, w: u32, h: u32) {
        let [last_x, last_y] = self.last_cursor_screen;
        let (gx0, gy0) = self.renderer.screen_to_grid(last_x, last_y, w, h);
        let (gx1, gy1) = self
            .renderer
            .screen_to_grid(last_x + dx_screen, last_y + dy_screen, w, h);
        self.pan_center -= glam::Vec2::new(gx1 - gx0, gy1 - gy0);
        self.apply_camera(w, h);
    }

    fn update_and_render(&mut self, window: &Window) {
        if !self.paused {
            for _ in 0..self.steps_per_frame {
                self.sim.step();
                self.days_elapsed += DT_SECONDS as f32 / 86400.0;
                // Fixed SIMULATED-TIME cadence, not per-frame -- see
                // `TRAIL_SAMPLE_INTERVAL_SECONDS`'s own doc. At high
                // `steps_per_frame` this correctly fires several times
                // within one rendered frame, keeping trail smoothness
                // independent of playback speed.
                self.trail_sample_accum += DT_SECONDS;
                if self.trail_sample_accum >= TRAIL_SAMPLE_INTERVAL_SECONDS {
                    self.trail_sample_accum -= TRAIL_SAMPLE_INTERVAL_SECONDS;
                    self.trails.push(&self.sim.particles().x);
                }
            }
        }
        // Real Sun position drifts slightly (barycentric frame -- the Sun
        // genuinely moves in response to real mutual gravity from every
        // planet), so the irradiance source must track it, not stay fixed
        // at spawn.
        self.radiance.sources[0].0 = self.sim.particles().x[0];
        self.renderer.set_light_source(
            &self.queue,
            self.sim.particles().x[0],
            self.shading_strength,
            light_reference_distance_grid(),
        );

        let output = match self.surface.get_current_texture() {
            Ok(t) => t,
            Err(_) => return,
        };
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        // Trails render first (clear + fading polylines), particles land on
        // top (`clear: false`, `LoadOp::Load`) so a planet's own bright dot
        // is never occluded by its own trail line.
        self.trails.render(&self.device, &self.queue, &view, true);
        self.renderer.render(
            &self.device,
            &self.queue,
            self.sim.particles(),
            &view,
            false,
        );

        // Real, live per-planet irradiance (W/m²) -- real inverse-square
        // law from the Sun's OWN just-synced position, at each planet's
        // CURRENT (not initial) orbital distance. Computed outside the
        // `egui_ctx.run` closure below (same reason `days`/`zoom` already
        // are: the closure can't hold a live borrow of `self`).
        let particles = self.sim.particles();
        let irradiance: Vec<(&str, f32)> = PLANETS
            .iter()
            .enumerate()
            .map(|(idx, &(name, _, _))| {
                (
                    name,
                    self.radiance
                        .irradiance_at(particles.x[idx + 1], DX_METERS as f32),
                )
            })
            .collect();
        // Real, direct sanity check: Earth's own live-computed irradiance
        // should sit near the real, independently-measured solar constant
        // (1361 W/m²), the same real number `transfer.rs`'s own test
        // verifies analytically -- here it comes out the other way, from
        // a live N-body scene's own real (slightly eccentric-drifting)
        // Earth-Sun distance, not a hand-picked 1 AU input.
        let earth_irradiance = irradiance
            .iter()
            .find(|&&(n, _)| n == "Earth")
            .map(|&(_, f)| f);

        let raw_input = self.egui_state.take_egui_input(window);
        let mut steps_per_frame = self.steps_per_frame as f32;
        let mut paused = self.paused;
        let mut zoom = self.zoom;
        let mut glow_strength = self.glow_strength;
        let days = self.days_elapsed;
        let mut reset = false;
        let (w, h) = (self.surface_config.width, self.surface_config.height);
        let full_output = self.egui_ctx.run(raw_input, |ctx| {
            egui::Window::new("Solar System (true N-body)")
                .default_pos([10.0, 10.0])
                .default_width(280.0)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.label(format!(
                        "day {days:.0}  ({:.2} years)  |  9 real bodies, mutual gravity",
                        days / 365.25
                    ));
                    ui.separator();
                    ui.checkbox(&mut paused, "Paused");
                    ui.label("Speed (real steps per rendered frame):");
                    ui.add(egui::Slider::new(&mut steps_per_frame, 1.0..=400.0).logarithmic(true));
                    ui.label(
                        "Zoom (real linear distance scale -- Mercury ~78x closer than Neptune):",
                    );
                    ui.add(egui::Slider::new(&mut zoom, 0.2..=20.0).logarithmic(true));
                    ui.label("Glow (soft-particle look, 0 = old flat disc):");
                    ui.add(egui::Slider::new(&mut glow_strength, 0.0..=6.0));
                    ui.separator();
                    ui.label(format!(
                        "Sun: R={SUN_RADIUS_M:.3e} m  T={SUN_SURFACE_TEMP_K:.0} K  L={:.3e} W",
                        stellar_luminosity_w(SUN_RADIUS_M, SUN_SURFACE_TEMP_K)
                    ));
                    if let Some(f) = earth_irradiance {
                        ui.label(format!(
                            "Earth irradiance (live, real inverse-square): {f:.0} W/m²  (real solar constant: 1361 W/m²)"
                        ));
                    }
                    egui::CollapsingHeader::new("Per-planet irradiance (live, W/m²)").show(
                        ui,
                        |ui| {
                            for &(name, flux) in &irradiance {
                                ui.label(format!("{name:<8} {flux:>10.2} W/m²"));
                            }
                        },
                    );
                    ui.separator();
                    ui.label("LMB drag: pan camera  |  R reset  Q quit");
                    if ui.button("Reset").clicked() {
                        reset = true;
                    }
                });
        });
        self.steps_per_frame = steps_per_frame.round().max(1.0) as u32;
        self.paused = paused;
        if (zoom - self.zoom).abs() > 1.0e-4 {
            self.zoom = zoom;
            self.apply_camera(w, h);
        }
        if (glow_strength - self.glow_strength).abs() > 1.0e-4 {
            self.glow_strength = glow_strength;
            self.renderer.set_glow_strength(&self.queue, glow_strength);
        }
        if reset {
            self.sim = make_sim();
            self.radiance = RadianceField::point(
                self.sim.particles().x[0],
                stellar_luminosity_w(SUN_RADIUS_M, SUN_SURFACE_TEMP_K),
            );
            self.days_elapsed = 0.0;
            self.pan_center = glam::Vec2::splat(GRID as f32 * 0.5);
            self.trails.clear();
            self.trail_sample_accum = 0.0;
            self.apply_camera(w, h);
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
                    .with_title("emerge -- True Solar System (N-body)")
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
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => match key {
                KeyCode::Escape | KeyCode::KeyQ => el.exit(),
                KeyCode::KeyR => {
                    s.sim = make_sim();
                    s.radiance = RadianceField::point(
                        s.sim.particles().x[0],
                        stellar_luminosity_w(SUN_RADIUS_M, SUN_SURFACE_TEMP_K),
                    );
                    s.days_elapsed = 0.0;
                    s.pan_center = glam::Vec2::splat(GRID as f32 * 0.5);
                    let (w, h) = (s.surface_config.width, s.surface_config.height);
                    s.apply_camera(w, h);
                    println!("reset");
                }
                _ => {}
            },
            WindowEvent::MouseInput {
                state: btn_state,
                button: MouseButton::Left,
                ..
            } => {
                s.dragging = btn_state == ElementState::Pressed;
            }
            WindowEvent::CursorMoved { position, .. } => {
                let (x, y) = (position.x as f32, position.y as f32);
                if s.dragging {
                    let (w, h) = (s.surface_config.width, s.surface_config.height);
                    s.pan_by_screen_delta(
                        x - s.last_cursor_screen[0],
                        y - s.last_cursor_screen[1],
                        w,
                        h,
                    );
                }
                s.last_cursor_screen = [x, y];
            }
            WindowEvent::MouseWheel { delta, .. } => {
                // Real trackpad pinch-to-zoom hook: winit's own native
                // `PinchGesture` event is macOS/iOS-only (confirmed against
                // winit 0.30's own source) -- on Windows, a real two-finger
                // pinch on a Precision Touchpad is synthesized by the OS
                // into wheel-scroll events (the same real mechanism
                // Chrome/Edge already rely on for pinch-zoom), so this is
                // the actual working hook on the platform this runs on,
                // not a Mac-only stub. Multiplicative, not additive, to
                // feel consistent across the egui slider's own log range.
                let scroll_y = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(pos) => (pos.y / 100.0) as f32,
                };
                s.zoom = (s.zoom * (1.0 + scroll_y * 0.08)).clamp(0.2, 20.0);
                let (w, h) = (s.surface_config.width, s.surface_config.height);
                s.apply_camera(w, h);
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

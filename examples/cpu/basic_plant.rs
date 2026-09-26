extern crate emerge_engine as emerge;

#[path = "../gui_common/coords.rs"]
mod gui_common;

/// Phase 1 of the basic-plant project: minimal, interactive, real-physics
/// stalk -- pinned root, real wind drag, real cursor push, real damped
/// elastic recovery. Pure composition of already-verified mechanisms
/// (`Particle::pinned`, `LinearDragField`, `apply_radial_impulse`,
/// `ViscoelasticMaterial`).
///
/// Generic viscoelastic (Kelvin-Voigt) stalk, not a literal-SI-accurate
/// species -- real PDE law, generic parameters; per-species tuning is future
/// work. Wind drag_coefficient chosen from `LinearDragField`'s own documented
/// real range ("start around 0.5-5.0").
///
/// Rendering: per-particle splat by default (`ColorMode::ByPhysics`); G
/// toggles grid-volume mode (`Renderer::render_grid_volume`), using
/// `fire_spread.rs`'s own CPU-`Simulation` bridging pattern (no GPU-resident
/// grid the way `GpuSimulation` has, so the bridge buffers are rebuilt from
/// the CPU solver each frame).
///
///   cargo run --example basic_plant --features render
use emerge::fields::LinearDragField;
use emerge::render::{ColorMode, GridVolumeSource, Renderer};
use emerge::{
    FrameLogger, SimConfig, Simulation, SlipBoundary, SpawnRegion, ViscoelasticMaterial,
    per_material_stats,
};
use glam::{IVec2, Vec2};
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

const GRID: usize = 64;
// The renderer's camera frames the full simulation grid with no pan -- this
// blade is short, so a separate, smaller DISPLAY_GRID gives a display-only
// zoom window decoupled from the real simulation domain (GRID stays
// untouched, physics unaffected). `cursor_grid` must use the same value so
// mouse push/pull stays aligned with what's on screen.
const DISPLAY_GRID: usize = 22;
const STALK_CENTER_X: f32 = 11.0;
const DT: f32 = 0.1;
const MAT_STALK: u32 = 0;
// Height bounded by real self-weight elastic buckling (Euler/Greenhill:
// h_crit = (7.8373*E*I/(rho*g*A))^(1/3), I ~ width^3) for a blade this thin
// (width~3), confirmed against Wikipedia "Self-buckling". Spawned near its
// own known equilibrium height rather than an unstressed taller one settling
// into it -- real plants grow adaptively under load (thigmomorphogenesis),
// they don't exist "unstressed" and then sag.
const STALK_HEIGHT: i32 = 4;
// Root: pin the bottom portion of the stalk (not a razor-thin single row --
// see `make_sim`'s own doc for why a knife-edge anchor concentrates the
// whole bending moment at one point). Matches the exact pin band used in
// every diagnostic test above (y <= stalk_bottom + 1.0).
const ROOT_HEIGHT: f32 = 10.0;
const WIND_DRAG_COEFFICIENT: f32 = 1.5;
// LinearDragField's acceleration is mass-independent (a = k*(target-v)) --
// tuned per-geometry: this thin/stiff blade needs a much smaller magnitude
// than a taller stalk to produce visible, bounded sway.
const WIND_SPEED: f32 = 0.015;
const WIND_GUST_PERIOD_SECONDS: f32 = 4.0;
// Tapered width (blade narrows toward tip). Tip kept at 4 particles wide
// (1.5 units), not 2 -- a 2-particle-wide tip can neck down and tear under
// bending.
const STALK_WIDTH_BASE: i32 = 3;
const STALK_WIDTH_TIP_UNITS: f32 = 1.5;

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
    lmb_just_pressed: bool,
    rmb_just_pressed: bool,
    frame: u64,
    fps_timer: std::time::Instant,
    fps_frames: u64,
    wind_enabled: bool,
    wind_time: f32,
    base_x: f32,
    last_tip_offset: f32,
    logger: FrameLogger,
    /// CPU-`Simulation` grid-volume render bridge (G toggles back to splats) --
    /// same pattern as `fire_spread.rs`'s bridge. Rebuilt from the CPU solver's
    /// `Grid`/`Particles` each frame; reuses the shared GPU render path/shader.
    grid_bridge_buf: wgpu::Buffer,
    material_mass_bridge_buf: wgpu::Buffer,
    grid_volume_mode: bool,
}

fn make_sim(wind_enabled: bool) -> Simulation {
    // max_substeps_per_step scales with stiffness under CFL (~ sqrt(stiffness/
    // density)) -- kept conservative rather than re-tuned down for headroom.
    //
    // Real, measured (2026-09-10): this demo's own real fps is 23-26, even
    // with the stalk sitting at rest (Jmin=Jmax=1.0, max_v~0.0004) -- a live
    // per-phase timing check confirmed the cost is `sim.step()` itself
    // (~38ms/frame), NOT rendering (acquire+render+present together stay
    // under 2ms). `min_dt=0.0007` against `DT=0.1` forces ~143 substeps
    // EVERY frame regardless of how visually calm the scene is, because the
    // real elastic stiffness (chosen for correct static-equilibrium physics,
    // see `eta`'s own doc below) drives a genuinely small CFL-safe dt --
    // the SAME real, disclosed stiffness-forces-many-substeps class already
    // found for `basic_sand.rs`/`basic_showcase.rs` tonight, not a render/UI
    // bug (an earlier hypothesis, now corrected). Real fix is the same
    // Stage 3 implicit-MPM plasticity work those two need -- not chased
    // further here; softening this material's own real stiffness would
    // break the physics this demo exists to show.
    let config = SimConfig {
        min_dt: 0.0007,
        max_substeps_per_step: 290,
        // Deliberately weak, NOT real IRL gravity (real g_grid ~= 981 via
        // SimConfig::earth) -- tuned down for a calmer, more legible demo at
        // this grid scale. Disclosed, deferred: basic_sand.rs's
        // gravity_fraction slider is the real-IRL-with-live-control
        // pattern, not yet ported to every plain example.
        gravity: Vec2::new(0.0, -0.3),
        ..SimConfig::earth(GRID, 0.01, DT)
    };

    // A taper built from multiple independently-spawned `SpawnRegion`s can leave
    // a seam gap wider than the MPM kernel's support radius -- pieces share no
    // grid node, zero momentum transfer, looks fine until it visibly detaches
    // under load. Spawn ONE continuous lattice instead and carve the taper
    // shape out with `Simulation::retain_particles`.
    let spawn_full = SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new(STALK_WIDTH_BASE, STALK_HEIGHT),
        box_center: Vec2::new(STALK_CENTER_X, 9.0 + STALK_HEIGHT as f32 / 2.0),
        material_id: MAT_STALK,
        precompute_initial_volumes: true,
        ..SpawnRegion::for_sim(&config)
    };

    // eta (3rd arg) only governs approach speed -- static equilibrium is set by
    // lambda/mu. Lower eta values can look flat for thousands of simulated
    // seconds before diverging into growing oscillation; this value is the
    // tested-stable floor.
    let stalk = ViscoelasticMaterial::new(8000.0, 12000.0, 1200.0);

    let mut solver = Simulation::new(config, spawn_full)
        .with_default_material(Box::new(stalk))
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));

    // Carve the linear taper: half-width shrinks from STALK_WIDTH_BASE/2 at
    // the base to STALK_WIDTH_TIP_UNITS/2 at the tip, same single lattice
    // throughout -- no seams, no detachment risk (see spawn_full's comment).
    let stalk_bottom = 9.0;
    let half_base = STALK_WIDTH_BASE as f32 / 2.0;
    let half_tip = STALK_WIDTH_TIP_UNITS / 2.0;
    solver.retain_particles(|p| {
        let t = ((p.x.y - stalk_bottom) / STALK_HEIGHT as f32).clamp(0.0, 1.0);
        let half_width = half_base + (half_tip - half_base) * t;
        (p.x.x - STALK_CENTER_X).abs() <= half_width
    });

    // Root: pin every particle at/below ROOT_HEIGHT -- a real Dirichlet
    // anchor (forces v=0, velocity_gradient=0 every substep in G2P), not a
    // scripted position lock.
    {
        let particles = solver.particles_mut();
        for i in 0..particles.len() {
            if particles.x[i].y <= ROOT_HEIGHT {
                particles.pinned[i] = 1;
            }
        }
    }

    // Wind is added fresh each frame in `update_and_render` (time-varying
    // gust), not here -- `wind_enabled` alone doesn't need an initial field.
    let _ = wind_enabled;

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
        let wind_enabled = true;
        let sim = make_sim(wind_enabled);
        let base_x = sim.particles().iter().map(|p| p.x.x).sum::<f32>()
            / sim.particles().len().max(1) as f32;
        let mut renderer = Renderer::new(&device, sim.particles().len(), fmt);
        renderer.set_camera(
            &queue,
            DISPLAY_GRID as u32,
            size.width,
            size.height,
            0.6,
            true,
        );
        renderer.set_color_mode(ColorMode::ByPhysics);
        println!(
            "basic_plant: {} particles  |  LMB push  RMB pull  W toggle wind  G toggle grid-volume/splat  R reset  Q quit",
            sim.particles().len()
        );
        println!(
            "wind: {} (drag={WIND_DRAG_COEFFICIENT}, speed={WIND_SPEED})",
            if wind_enabled { "on" } else { "off" }
        );
        // FrameLogger + per-material stats/SimSnapshot cover the generic
        // diagnostics; `extra` carries this demo's two scene-specific numbers
        // (tip_offset and its frame-to-frame delta).
        let log_path = std::env::temp_dir().join("emerge_basic_plant.ndjson");
        let logger = FrameLogger::open(&log_path).unwrap();
        println!("per-frame diagnostics log: {}", log_path.display());

        // Same sizing as `fire_spread.rs`'s own bridge buffers (real, verified
        // pattern) -- 16 material slots is far more than this single-material
        // scene needs, but matches the shared shader's expected layout exactly.
        const RENDER_MATERIAL_SLOTS: u64 = 16;
        let grid_bridge_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("basic_plant_grid_bridge"),
            size: (GRID * GRID * 4 * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let material_mass_bridge_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("basic_plant_material_mass_bridge"),
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
            lmb_just_pressed: false,
            rmb_just_pressed: false,
            frame: 0,
            fps_timer: std::time::Instant::now(),
            fps_frames: 0,
            wind_enabled,
            wind_time: 0.0,
            base_x,
            last_tip_offset: 0.0,
            logger,
            grid_bridge_buf,
            material_mass_bridge_buf,
            // grid_volume.wgsl indexes its density buffer by `params.grid_res`,
            // the same value `set_camera` uses for the display zoom window --
            // decoupling DISPLAY_GRID from the real GRID breaks that indexing
            // (renders nothing). Splat mode has no such coupling, so it's the
            // default.
            grid_volume_mode: false,
        }
    }

    /// Rebuilds `grid_bridge_buf`/`material_mass_bridge_buf` from the CPU
    /// solver's current state -- see those fields' own doc for the real,
    /// disclosed cost. Identical technique to `fire_spread.rs`'s own bridge.
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
        self.queue
            .write_buffer(&self.grid_bridge_buf, 0, bytemuck::cast_slice(&dense));

        let mut material_mass = vec![0f32; GRID * GRID * SLOTS];
        let particles = self.sim.particles();
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
            .set_camera(&self.queue, DISPLAY_GRID as u32, w, h, 0.6, true);
    }

    fn cursor_grid(&self) -> Vec2 {
        gui_common::cursor_to_grid(
            self.cursor_pos,
            self.surface_config.width,
            self.surface_config.height,
            DISPLAY_GRID,
        )
    }

    fn update_and_render(&mut self) {
        // Fires once on the press (rising edge), not every held frame:
        // repeatedly applying an impulse would deliberately stack unbounded
        // cumulative energy injection.
        if self.lmb_just_pressed {
            self.sim.apply_radial_impulse(self.cursor_grid(), 5.0, 2.0);
            self.lmb_just_pressed = false;
        }
        if self.rmb_just_pressed {
            self.sim.apply_radial_impulse(self.cursor_grid(), 5.0, -2.0);
            self.rmb_just_pressed = false;
        }
        if self.wind_enabled {
            self.wind_time += DT;
            let omega = std::f32::consts::TAU / WIND_GUST_PERIOD_SECONDS;
            let gust_speed = WIND_SPEED * (self.wind_time * omega).sin();
            self.sim.remove_force_field("wind");
            self.sim.add_named_force_field(
                "wind",
                Box::new(LinearDragField::new(
                    Vec2::new(gust_speed, 0.0),
                    WIND_DRAG_COEFFICIENT,
                    1 << MAT_STALK,
                )),
            );
        }
        self.sim.step();
        self.frame += 1;
        self.fps_frames += 1;

        // Real, direct evidence of bending + recovery: how far the stalk's
        // topmost particles have drifted horizontally from the root's own x
        // position, not just a visual impression.
        let particles = self.sim.particles();
        let top_y = particles.iter().map(|p| p.x.y).fold(f32::MIN, f32::max);
        let tip_band = top_y - 2.0;
        let tip_x: Vec<f32> = particles
            .iter()
            .filter(|p| p.x.y >= tip_band)
            .map(|p| p.x.x)
            .collect();
        let tip_avg_x = if tip_x.is_empty() {
            self.base_x
        } else {
            tip_x.iter().sum::<f32>() / tip_x.len() as f32
        };
        let tip_offset = tip_avg_x - self.base_x;
        let tip_offset_delta = tip_offset - self.last_tip_offset;
        self.last_tip_offset = tip_offset;

        // `max_pinned_particle_speed` is the "is the root actually anchored"
        // sanity check, read directly from SimSnapshot instead of recomputed
        // here.
        let snap = self.sim.diagnostics_snapshot();
        let stats = per_material_stats(particles);
        self.logger.log(
            self.frame,
            snap.effective_dt,
            &stats,
            &snap,
            &[(MAT_STALK, "stalk")],
            &[
                ("tip_offset", tip_offset),
                ("tip_offset_delta", tip_offset_delta),
            ],
        );

        if self.fps_timer.elapsed().as_secs_f32() >= 2.0 {
            let fps = self.fps_frames as f32 / self.fps_timer.elapsed().as_secs_f32();
            let (mut min_x, mut max_x, mut min_y, mut max_y) =
                (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
            for p in particles.iter() {
                min_x = min_x.min(p.x.x);
                max_x = max_x.max(p.x.x);
                min_y = min_y.min(p.x.y);
                max_y = max_y.max(p.x.y);
            }
            println!(
                "frame={} fps={:.0}  tip_offset={:.3} (d={:+.4})  Jmin={:.4} Jmax={:.4}  \
                 max_v={:.4}  KE={:.4}  root_v={:.2e}  wind={}  bbox=[{:.1},{:.1}]-[{:.1},{:.1}]  n={}",
                self.frame,
                fps,
                tip_offset,
                tip_offset_delta,
                snap.min_deformation_j,
                snap.max_deformation_j,
                snap.max_particle_speed,
                snap.total_kinetic_energy,
                snap.max_pinned_particle_speed,
                if self.wind_enabled { "on" } else { "off" },
                min_x,
                min_y,
                max_x,
                max_y,
                particles.len(),
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
                    .with_title("emerge -- Basic Plant [Phase 1: interactive stalk]")
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
        match event {
            WindowEvent::CloseRequested => el.exit(),
            WindowEvent::CursorMoved { position, .. } => {
                s.cursor_pos = [position.x as f32, position.y as f32];
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if state == ElementState::Pressed {
                    match button {
                        MouseButton::Left => s.lmb_just_pressed = true,
                        MouseButton::Right => s.rmb_just_pressed = true,
                        _ => {}
                    }
                }
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
                    s.sim = make_sim(s.wind_enabled);
                    s.frame = 0;
                    s.wind_time = 0.0;
                    println!("reset (wind={})", if s.wind_enabled { "on" } else { "off" });
                }
                KeyCode::KeyW => {
                    s.wind_enabled = !s.wind_enabled;
                    s.sim = make_sim(s.wind_enabled);
                    s.frame = 0;
                    s.wind_time = 0.0;
                    println!(
                        "wind: {} (reset to A/B cleanly)",
                        if s.wind_enabled { "on" } else { "off" }
                    );
                }
                KeyCode::KeyG => {
                    s.grid_volume_mode = !s.grid_volume_mode;
                    println!(
                        "render: {}",
                        if s.grid_volume_mode {
                            "grid-volume"
                        } else {
                            "splat"
                        }
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

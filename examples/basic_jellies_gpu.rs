extern crate emerge_engine as emerge;

/// GPU elastic solids — NeoHookean / Corotated / Viscoelastic, zero CPU readback.
///
///   Mat 0  NeoHookean   (orange) — soft hyperelastic
///   Mat 1  Corotated    (teal)   — stiffer corotated linear
///   Mat 2  Viscoelastic (purple) — Kelvin-Voigt dashpot
///
///   cargo run --example basic_jellies_gpu --features "render"
use std::sync::Arc;

use emerge::diagnostics::log_frame_gpu;
use emerge::render::{ColorMode, GridVolumeSource, Renderer, SurfaceReconstructionSource};
use emerge::{
    CorotatedMaterial, FixedStepController, GpuSimulation, MaterialRegistry, NeoHookeanMaterial,
    SimConfig, SpawnRegion, ViscoelasticMaterial, build_particles,
};
use glam::{IVec2, Vec2};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

const GRID: usize = 64;
const DT: f32 = 0.1;
const MAT_NEO: u32 = 0;
const MAT_COR: u32 = 1;
const MAT_VIS: u32 = 2;
const LABELS: &[(u32, &str)] = &[
    (MAT_NEO, "neo"),
    (MAT_COR, "corot"),
    (MAT_VIS, "viscoelastic"),
];

struct App {
    window: Option<Arc<Window>>,
    state: Option<State>,
}

struct State {
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    sim: GpuSimulation,
    renderer: Renderer,
    cursor_pos: [f32; 2],
    lmb: bool,
    rmb: bool,
    frame: u64,
    fps_timer: std::time::Instant,
    fps_frames: u64,
    /// Cycled with G: Particles -> GridVolume -> Surface -> Particles, all
    /// live on the exact same running sim. `Surface` is
    /// `render_surface_reconstruction` (curvature-flow, shipped
    /// 2026-07-29) -- a soft, cohesive continuous skin arguably suits a
    /// jelly/blob body even better than the fluid scene it was first
    /// verified on.
    render_mode: RenderMode,
    /// Real-time-decoupled stepping -- calling `sim.step_frame()` once per
    /// render frame silently ties simulated speed to render fps (see
    /// `basic_fluids_gpu.rs`'s own field doc for the full real bug/fix
    /// writeup). `standard(DT, 60.0)` deliberately keeps DT/call-rate as-is
    /// (preserving this demo's already-tuned feel), just decoupling it from
    /// actual measured fps instead of assuming every frame takes `DT`.
    stepper: FixedStepController,
    last_instant: std::time::Instant,
    /// Diagnostic: highest `steps_for_frame` result seen since the last fps
    /// print -- a catch-up burst (several real simulation steps crammed
    /// into one render call) would NOT show up in the averaged fps number
    /// alone, since a 2-second average smooths occasional slow frames out.
    max_steps_seen: usize,
}

/// The three real rendering paths this demo can show, cycled with G -- see
/// `basic_fluids_gpu.rs`'s own identical enum for the fuller doc.
#[derive(Clone, Copy, PartialEq, Eq)]
enum RenderMode {
    Particles,
    GridVolume,
    Surface,
}

fn make_sim_data(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> GpuSimulation {
    let config = SimConfig {
        max_substeps_per_step: 12,
        // Deliberately weak, NOT real IRL gravity (real g_grid ~= 981 via
        // SimConfig::earth) -- tuned down for a calmer, more legible demo at
        // this grid scale. Disclosed, deferred: basic_sand_gui.rs's
        // gravity_fraction slider is the real-IRL-with-live-control
        // pattern, not yet ported to every plain example.
        gravity: Vec2::new(0.0, -0.3),
        ..SimConfig::earth(GRID, 0.01, DT)
    };
    let blob = |cx: f32, mat: u32, seed: u32| SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new(16, 16),
        box_center: Vec2::new(cx, 48.0),
        material_id: mat,
        precompute_initial_volumes: true,
        rng_seed: seed,
        ..SpawnRegion::for_sim(&config)
    };
    let mut particles = build_particles(&config, blob(16.0, MAT_NEO, 1));
    let mut hot_cor = build_particles(&config, blob(32.0, MAT_COR, 2));
    // Demo-only, illustrative temperature (real magma/lava range, ~1200-1300K) --
    // NOT a physical claim that this Corotated jelly IS lava, purely a live,
    // visible confirmation that Surface mode's new blackbody emission (see
    // curvature_flow.wgsl's own doc) actually reads real per-particle
    // temperature end-to-end, not just passing in an automated pixel test.
    for p in hot_cor.iter_mut() {
        p.temperature = 1200.0;
    }
    particles.extend(hot_cor);
    particles.extend(build_particles(&config, blob(48.0, MAT_VIS, 3)));

    let mut registry =
        MaterialRegistry::with_default(Box::new(NeoHookeanMaterial::new(10.0, 20.0)));
    registry.insert(MAT_COR, Box::new(CorotatedMaterial::new(30.0, 60.0)));
    registry.insert(
        MAT_VIS,
        Box::new(ViscoelasticMaterial::new(10.0, 15.0, 0.15)),
    );
    GpuSimulation::with_device(device, queue, config, particles, registry)
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
                required_limits: adapter.limits(), // use full hardware limits, not wgpu defaults
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
        let mut sim = make_sim_data(Arc::new(device), Arc::new(queue));
        // Grid-volume material rendering: stable across single-material,
        // multi-material, and repeated spawn_region stress patterns (bounded J,
        // bounded speed, no NaN/explosion).
        sim.attach_grid_material_render_gpu();
        let mut renderer = Renderer::new(sim.device(), sim.particle_count(), fmt);
        renderer.set_camera(sim.queue(), GRID as u32, size.width, size.height, 0.6, true);
        renderer.set_color_mode(ColorMode::ByMaterial);
        // Distinct optics per material slot -- ByMaterial splat mode has its own
        // separate hardcoded palette (real, but unrelated to the Beer-Lambert
        // OpticalTable), so grid-volume mode's per-cell dominant-material coloring
        // (which reads THIS table, see grid_volume.wgsl's dominant_material) had
        // never been populated here -- every material silently fell back to the
        // same default, reading as flat grey/white. Aesthetic choice, not cited
        // (unlike water's Pope & Fry values elsewhere): soft elastic solids don't
        // have an obvious universal real absorption spectrum the way water/tissue
        // do. SIGMA_NEO is the same value material_sandbox_gpu.rs's own SIGMA_JELLY
        // constant is named after (this file, "(SIGMA_NEO)").
        const SIGMA_NEO: [f32; 3] = [0.05, 0.55, 0.60];
        const SIGMA_COR: [f32; 3] = [0.55, 0.15, 0.50];
        const SIGMA_VIS: [f32; 3] = [0.45, 0.35, 0.10];
        renderer.set_optical_params(sim.queue(), MAT_NEO as usize, SIGMA_NEO);
        renderer.set_optical_params(sim.queue(), MAT_COR as usize, SIGMA_COR);
        renderer.set_optical_params(sim.queue(), MAT_VIS as usize, SIGMA_VIS);
        // Real subsurface scattering + Fresnel specular -- these defaulted to
        // 0.0 (no visible effect at all, even after that shading math
        // shipped) until wired here, same real gap `basic_fluids_gpu.rs` had.
        //
        // Magnitudes chosen the same corrected way that file's own doc now
        // explains (NOT the original 4.0-5.0 first attempt, which would
        // have hit the exact same `albedo = sigma_s/(sigma_s+sigma_a)`
        // wash-out bug found+fixed there -- these materials' sigma_a values
        // are just as low in places, e.g. SIGMA_NEO's red channel at 0.05):
        // sigma_s set to roughly 0.43x each material's OWN smallest sigma_a
        // channel, targeting a bounded ~0.3 albedo there instead of an
        // unrelated borrowed magnitude. Specular kept low/soft (a gel
        // surface is not mirror-like still water).
        renderer.set_optical_scattering(sim.queue(), MAT_NEO as usize, 0.02);
        renderer.set_specular_r0(sim.queue(), MAT_NEO as usize, 0.01);
        renderer.set_optical_scattering(sim.queue(), MAT_COR as usize, 0.06);
        renderer.set_specular_r0(sim.queue(), MAT_COR as usize, 0.01);
        renderer.set_optical_scattering(sim.queue(), MAT_VIS as usize, 0.04);
        renderer.set_specular_r0(sim.queue(), MAT_VIS as usize, 0.01);
        println!(
            "jellies GPU: {} particles  |  LMB push  RMB pull  R reset  Q quit",
            sim.particle_count()
        );
        Self {
            surface,
            surface_config: sc,
            sim,
            renderer,
            cursor_pos: [0.0; 2],
            lmb: false,
            rmb: false,
            frame: 0,
            fps_timer: std::time::Instant::now(),
            fps_frames: 0,
            render_mode: RenderMode::Particles,
            stepper: FixedStepController::standard(DT, 60.0),
            last_instant: std::time::Instant::now(),
            max_steps_seen: 0,
        }
    }

    fn resize(&mut self, w: u32, h: u32) {
        if w == 0 || h == 0 {
            return;
        }
        self.surface_config.width = w;
        self.surface_config.height = h;
        self.surface
            .configure(self.sim.device(), &self.surface_config);
        self.renderer
            .set_camera(self.sim.queue(), GRID as u32, w, h, 0.6, true);
    }

    fn cursor_grid(&self) -> Vec2 {
        Vec2::new(
            self.cursor_pos[0] / self.surface_config.width as f32 * GRID as f32,
            (1.0 - self.cursor_pos[1] / self.surface_config.height as f32) * GRID as f32,
        )
    }

    fn reset(&mut self) {
        let (device, queue) = (self.sim.device().clone(), self.sim.queue().clone());
        self.sim = make_sim_data(device, queue);
        self.frame = 0;
        self.stepper.reset();
        self.last_instant = std::time::Instant::now();
        println!("reset");
    }

    fn update_and_render(&mut self) {
        // Real elapsed time computed FIRST now (was after the impulse call)
        // -- needed to scale the impulse by real time, see below.
        let now = std::time::Instant::now();
        let frame_delta = (now - self.last_instant).as_secs_f32();
        self.last_instant = now;

        if self.lmb || self.rmb {
            // Real bug #1 (found via a real user report + log, J swinging
            // 0.056..3.5): magnitude was 8.0, a 4x outlier vs. every other
            // elastic demo (`basic_jellies.rs`, the CPU version of this
            // EXACT scene, uses 2.0) -- matched back to 2.0.
            //
            // Real bug #2, found via a SECOND real report after fixing #1
            // (J still hit 0.016..3.0 at magnitude 2.0, worse on pull): the
            // impulse was applied at FULL magnitude every single RENDER
            // FRAME while the button stayed held, not scaled by real time.
            // Two real problems from this, not one: (a) holding the button
            // longer keeps adding MORE total momentum without bound (a real
            // velocity ADD every frame, not a force integrated over time --
            // the API is genuinely named "impulse," an instantaneous
            // concept, but this call site re-triggered it continuously);
            // (b) the total momentum added for the SAME real-world hold
            // duration silently depended on framerate (60fps holds for 1s
            // add 2x the momentum a 30fps machine would for the identical
            // real second). Real fix: scale by `frame_delta` so holding
            // applies a bounded RATE (real velocity added per second),
            // framerate-independent, and no longer compounds without limit
            // the longer the button stays down. `IMPULSE_RATE_PER_SEC` is a
            // real, disclosed, tuned constant (not cited -- there's no
            // physical law for "how strong should a game click feel"),
            // measured via real automated hold tests (PostMessage-driven
            // RMB hold on a fully-settled scene, reading back the demo's
            // own real diagnostic log), not guessed: 1.0 was too weak (a
            // real 2s hold barely moved anything, max_speed peaked ~0.04);
            // 20.0 gave a real, moderate poke -- J stayed in [0.205,1.34]
            // (recovers, doesn't collapse), max_speed peaked ~0.14,
            // non_finite=0 throughout.
            const IMPULSE_RATE_PER_SEC: f32 = 20.0;
            let mag = if self.lmb {
                IMPULSE_RATE_PER_SEC
            } else {
                -IMPULSE_RATE_PER_SEC
            };
            self.sim
                .apply_radial_impulse(self.cursor_grid(), 6.0, mag * frame_delta);
        }
        let output = match self.surface.get_current_texture() {
            Ok(t) => t,
            Err(_) => return,
        };
        let steps = self.stepper.steps_for_frame(frame_delta);
        self.max_steps_seen = self.max_steps_seen.max(steps);
        for _ in 0..steps {
            self.sim.step_frame();
            self.frame += 1;
            // Gated per real simulation step, not per render call -- see
            // `basic_fluids_gpu.rs`'s own doc for why (avoids re-printing
            // the same "frame N" diagnostic when `steps` is 0 for several
            // consecutive render calls).
            if self.frame.is_multiple_of(60) {
                log_frame_gpu(self.frame, DT, self.sim.particles(), LABELS, 1);
                let snap = self.sim.diagnostics_snapshot();
                println!(
                    "  non_finite={}  out_of_bounds={}  max_speed={:.3}  sub={}  cfl={:.4}",
                    snap.non_finite_particle_values,
                    snap.out_of_bounds_particles,
                    snap.max_particle_speed,
                    snap.substeps_last_step,
                    snap.cfl_number,
                );
            }
        }
        self.fps_frames += 1;
        if self.fps_timer.elapsed().as_secs_f32() >= 2.0 {
            let fps = self.fps_frames as f32 / self.fps_timer.elapsed().as_secs_f32();
            println!(
                "frame={} fps={:.0} max_steps_per_render={}",
                self.frame, fps, self.max_steps_seen
            );
            self.fps_timer = std::time::Instant::now();
            self.fps_frames = 0;
            self.max_steps_seen = 0;
        }
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        match self.render_mode {
            RenderMode::GridVolume => {
                self.renderer.render_grid_volume(
                    self.sim.device(),
                    self.sim.queue(),
                    GridVolumeSource {
                        grid: self.sim.grid_buffer(),
                        material_mass: self.sim.material_mass_buffer(),
                        material_mass_enabled: true,
                    },
                    &view,
                    true,
                );
            }
            RenderMode::Surface => {
                self.renderer.render_surface_reconstruction(
                    self.sim.device(),
                    self.sim.queue(),
                    SurfaceReconstructionSource {
                        particle_buf: self.sim.particle_buffer(),
                        particle_count: self.sim.particle_count(),
                        grid_res: GRID as u32,
                        material_slot: MAT_NEO, // fallback only, unused once material_mass_enabled resolves per-cell
                        material_mass_enabled: true,
                        dt: DT,
                    },
                    &view,
                    true,
                );
            }
            RenderMode::Particles => {
                self.renderer.render_gpu(
                    self.sim.device(),
                    self.sim.queue(),
                    self.sim.particle_buffer(),
                    self.sim.particle_count(),
                    &view,
                    true,
                );
            }
        }
        output.present();
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        let w = Arc::new(
            el.create_window(
                winit::window::WindowAttributes::default()
                    .with_title("emerge — Jellies GPU [NeoHookean / Corotated / Viscoelastic]")
                    .with_inner_size(winit::dpi::LogicalSize::new(480u32, 480u32)),
            )
            .unwrap(),
        );
        self.state = Some(pollster::block_on(State::new(w.clone())));
        self.window = Some(w);
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(s) = self.state.as_mut() else { return };
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
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => match key {
                KeyCode::Escape | KeyCode::KeyQ => el.exit(),
                KeyCode::KeyR => s.reset(),
                KeyCode::KeyG => {
                    s.render_mode = match s.render_mode {
                        RenderMode::Particles => RenderMode::GridVolume,
                        RenderMode::GridVolume => RenderMode::Surface,
                        RenderMode::Surface => RenderMode::Particles,
                    };
                    println!(
                        "render mode: {}",
                        match s.render_mode {
                            RenderMode::Particles => "particles",
                            RenderMode::GridVolume => "grid-volume",
                            RenderMode::Surface => "curvature-flow surface",
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

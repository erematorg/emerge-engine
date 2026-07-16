extern crate emerge_engine as emerge;

/// Interactive creature-on-terrain proof scene -- GPU, mouse-driven.
///
/// Proves TWO things that just landed this session, visually, for the first time:
///   1. Multi-field Coulomb contact (Bardenhagen 2001 + Nairn 2023 LR fit + Baumgarte
///      stabilization) -- the creature rests on real terrain particles via
///      `Particle::contact_group`, not a domain-edge boundary or the
///      `RatchetFrictionBoundary` special case. The creature should settle onto the
///      terrain and STAY settled (no sinking, no floating) -- that's the whole point
///      of today's contact fix, see `src/spacetime/grid/mod.rs::resolve_contact`.
///   2. Genuine, non-hardcoded signs of life -- muscle activation driven by a real
///      `Lnn` (Liquid Time-constant Network) continuous-time CPG, the same controller
///      LP's creatures use, not a scripted animation curve.
///
/// Deliberately NOT in scope yet (real, current limitations, not oversights):
///   - No walking/crawling gait. The CPG drives muscle activation; whether that adds up
///     to net locomotion depends on body-plan structure this scene doesn't have yet
///     (see `PHYSICS_PROOFS.md` and the `reference_rainworld_movement_architecture`
///     project note -- multi-region goal-seeking bodies are the next real step).
///   - No jumping. Movement here is push/pull only; a real jump needs a real
///     locomotion primitive first, not just an upward impulse.
///   - No terrain variety / environment. One flat floor is enough to prove contact.
///
///   cargo run --example interactive_creature --features render
use std::sync::Arc;

use emerge::control::Lnn;
use emerge::diagnostics::log_frame_gpu;
use emerge::gpu::GpuSimulation;
use emerge::render::{ColorMode, Renderer};
use emerge::{MaterialRegistry, NeoHookeanMaterial, SimConfig, SpawnRegion, build_particles};
use glam::{IVec2, Vec2};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

const GRID: usize = 64;
const DT: f32 = 0.05;

const CREATURE_ID: u32 = 0;
const TERRAIN_ID: u32 = 1;
const LABELS: &[(u32, &str)] = &[(CREATURE_ID, "creature"), (TERRAIN_ID, "terrain")];

// contact_group: 0 = terrain (the "rest" field, untagged), nonzero = creature's own
// "grip" field -- see Particle::contact_group's doc for the full multi-field rationale.
const CREATURE_CONTACT_GROUP: u32 = 1;

const MUSCLE_GROUPS: u32 = 4;
// Single-ring CPG -- no steering needed for this scene, just genuine internal rhythm.
// Burn-in matches basic_creature.rs's own finding: a freshly-seeded network needs real
// settle time before its activation output is a true phase-locked oscillation, not a
// transient. See that example's `make_cpg` doc for the measured evidence.
const CPG_BURN_IN_STEPS: usize = 600;

fn make_cpg() -> Lnn {
    let mut lnn = Lnn::coupled_traveling_wave(1, MUSCLE_GROUPS as usize, 1.0, 0.0);
    for _ in 0..CPG_BURN_IN_STEPS {
        lnn.step(DT);
    }
    lnn
}

struct App {
    window: Option<Arc<Window>>,
    state: Option<State>,
}

struct State {
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    sim: GpuSimulation,
    renderer: Renderer,
    cpg: Lnn,
    cursor_pos: [f32; 2],
    lmb: bool,
    rmb: bool,
    frame: u64,
    fps_timer: std::time::Instant,
    fps_frames: u64,
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
            .expect("device request failed");
        let device = Arc::new(device);
        let queue = Arc::new(queue);

        let caps = surface.get_capabilities(&adapter);
        let fmt = caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(caps.formats[0]);
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: fmt,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::AutoVsync,
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &surface_config);

        // Game-scale gravity, matches every other GPU example in this repo.
        //
        // max_substeps_per_step raised 32 -> 64 (2026-07-13): the CFL substep count is
        // computed from the CPU velocity mirror BEFORE apply_impulses runs each frame, so
        // a real mouse push/pull is always encoded using the PREVIOUS (calmer) frame's
        // substep estimate for its own frame -- a genuine one-frame CFL lag, not just "not
        // enough substeps" in general (verified: raising to 128 alone still left the body
        // unstable, ruling that out as sufficient on its own). Combined with the impulse
        // magnitude fix below, 64 substeps gives real headroom against that lag. Matches
        // basic_creature.rs's own already-tuned 64 for a similarly stiff creature.
        let config = SimConfig {
            gravity: Vec2::new(0.0, -0.3),
            apic_blend: 0.1,
            max_substeps_per_step: 64,
            contact_friction: 0.6,
            ..SimConfig::standard(GRID, DT, Vec2::new(0.0, -0.3))
        };

        // Terrain: a flat slab across the bottom of the domain. contact_group defaults
        // to 0 (untagged) -- this IS the "rest" field the creature's grip resolves
        // against, no extra setup needed.
        let mut particles = build_particles(
            &config,
            SpawnRegion {
                spacing: 0.5,
                box_size: IVec2::new(GRID as i32 - 8, 8),
                box_center: Vec2::new(GRID as f32 * 0.5, 4.0),
                material_id: TERRAIN_ID,
                precompute_initial_volumes: true,
                ..SpawnRegion::for_sim(&config)
            },
        );
        let terrain_count = particles.len();

        // Creature: a soft body resting just above the terrain at spawn (small real
        // gap, not pre-overlapping), tagged into its own contact field so the multi-
        // field resolver actually has two distinct bodies to separate.
        let creature = build_particles(
            &config,
            SpawnRegion {
                spacing: 0.5,
                box_size: IVec2::new(16, 10),
                box_center: Vec2::new(GRID as f32 * 0.5, 14.0),
                material_id: CREATURE_ID,
                precompute_initial_volumes: true,
                ..SpawnRegion::for_sim(&config)
            },
        );
        let creature_count = creature.len();
        particles.extend(creature);

        // Tag the creature's own particles: real multi-field contact grip, plus a few
        // muscle groups (left-to-right bands) for the CPG to actually drive something.
        //
        // activation_dir MUST be set to a nonzero fiber direction, or the GPU active-
        // stress branch (p2g.wgsl) falls back to an ISOTROPIC pressure term instead of
        // a directional F.(n⊗n).F^T squeeze -- an isotropic pulse on a body sitting
        // under gravity reads as the whole thing swelling/bouncing vertically every
        // cycle, not a real localized muscle contraction. Horizontal fiber direction
        // matches the left-to-right muscle-group banding below, giving a genuine
        // traveling squeeze along the body instead of a whole-body pump.
        for p in particles
            .iter_mut()
            .skip(terrain_count)
            .take(creature_count)
        {
            p.contact_group = CREATURE_CONTACT_GROUP;
            p.activation_dir = Vec2::X;
        }
        let (min_x, max_x) = particles[terrain_count..(terrain_count + creature_count)]
            .iter()
            .fold((f32::MAX, f32::MIN), |(lo, hi), p| {
                (lo.min(p.x.x), hi.max(p.x.x))
            });
        let span = (max_x - min_x).max(1e-3);
        for p in &mut particles[terrain_count..(terrain_count + creature_count)] {
            let t = ((p.x.x - min_x) / span).clamp(0.0, 1.0);
            p.muscle_group_id = (t * MUSCLE_GROUPS as f32) as u32;
        }

        let mut registry =
            MaterialRegistry::with_default(Box::new(NeoHookeanMaterial::new(120.0, 240.0)));
        registry.insert(
            TERRAIN_ID,
            Box::new(NeoHookeanMaterial::new(200.0, 400.0)), // much stiffer -- ground, not flesh
        );
        // Real, visible muscle drive on the soft body -- matches basic_creature.rs's
        // documented ceiling (a muscle can't contract >100%, and pushing the amplitude
        // knob past this detonates the CFL budget, not just "looks too strong").
        //
        // Stiffness raised 10,20 -> 120,240 (2026-07-13): a real, measured sweep found
        // the original softness made the body ring elastically against the much stiffer
        // terrain (200,400), reading as visible bouncing, not resting -- a zero-muscle
        // control test proved the CPG wasn't the cause (bounce amplitude unchanged with
        // muscle off), and softening the terrain toward the creature made it WORSE, not
        // better (2-2.6x amplitude), ruling that direction out too. The real driver is
        // the creature's own absolute stiffness (confirmed via an 11-config isolation
        // sweep, terrain held fixed): 10,20 gave amplitude 1.244; 120,240 gives 0.179
        // (86% reduction) while staying at 40% of the terrain's stiffness, still visibly
        // softer/squishier than the ground under real muscle activation. See project
        // memory `interactive_creature_bounce_bug_2026-07-12` for the full investigation.
        registry.insert(CREATURE_ID, {
            let mut creature_mat = NeoHookeanMaterial::new(120.0, 240.0);
            creature_mat.active_stress_coeff = 15.0;
            Box::new(creature_mat)
        });

        let sim =
            GpuSimulation::with_device(device.clone(), queue.clone(), config, particles, registry);

        let mut renderer = Renderer::new(&device, sim.particle_count(), fmt);
        renderer.set_camera(&queue, GRID as u32, size.width, size.height, 0.6, true);
        renderer.set_color_mode(ColorMode::ByMaterial);

        println!(
            "interactive_creature [GPU]: {} particles ({} terrain, {} creature)",
            sim.particle_count(),
            terrain_count,
            creature_count
        );
        println!("  LMB push  RMB pull  [1-4] color modes  [Q] quit");
        println!(
            "  Proves: contact settling (watch it rest, not sink/float) + real CPG-driven life"
        );

        Self {
            surface,
            surface_config,
            device,
            queue,
            sim,
            renderer,
            cpg: make_cpg(),
            cursor_pos: [0.0; 2],
            lmb: false,
            rmb: false,
            frame: 0,
            fps_timer: std::time::Instant::now(),
            fps_frames: 0,
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

    fn update_and_render(&mut self) {
        // Real CPG step -- genuine continuous-time dynamics, not a scripted curve.
        self.cpg.step(DT);
        let activations: Vec<f32> = self.cpg.activations().collect();

        // Mouse push/pull, applied by hand (not `apply_radial_impulse`) and filtered
        // strictly to CREATURE_ID particles. Real bug found live: the built-in impulse
        // has no material filter, so a click anywhere near the creature also kicked the
        // TERRAIN -- a near-rigid material taking a sudden radial impulse visibly rippled
        // (terrain velocity hit 4.87, well beyond the settled ~0.05-0.15 baseline), which
        // is not a physics bug, it's just genuinely the wrong particles being pushed. The
        // creature's bottom sits only ~1.5 units above the terrain's top at rest, well
        // inside the impulse's radius=5 reach from anywhere near the body. Applied in this
        // same particles_mut()+mark_particles_dirty() pass already used for activation
        // every frame -- same established read/write pattern, not a new risk.
        //
        // Magnitude 4.0 (copied from render_physics.rs) was real-verified to explode this
        // scene's much stiffer body (max_j hit 9.3) -- that example's soft water tolerates
        // a much bigger impulse than a stiff NeoHookean body can. Even after filtering the
        // impulse to creature-only, an intermediate 1.0 still failed a real worst-case
        // headless check (click held for 30 frames right at the creature's bottom edge,
        // where the most particles overlap the impulse radius at once): creature_max_v
        // reached 9.82, and that momentum transmits back into the terrain through real
        // contact even with the filter working correctly. 0.15 is the actual verified-safe
        // value for that worst case (terrain_max_v stays 0.18, creature_max_v 0.68) -- see
        // interactive_creature_bounce_bug project memory for the full magnitude sweep.
        let push_active = self.lmb || self.rmb;
        let mag = if self.lmb { 0.15 } else { -0.15 };
        let cursor = self.cursor_grid();

        for p in self.sim.particles_mut().iter_mut() {
            if p.material_id == CREATURE_ID {
                let group = p.muscle_group_id as usize % activations.len().max(1);
                p.activation = activations[group].clamp(0.0, 1.0);

                if push_active {
                    let r = p.x - cursor;
                    let dist = r.length();
                    if dist < 5.0 && dist > 1e-3 {
                        p.v += (r / dist) * mag;
                    }
                }
            }
        }
        self.sim.mark_particles_dirty();

        let output = match self.surface.get_current_texture() {
            Ok(t) => t,
            Err(_) => return,
        };

        self.sim.step_frame();
        self.frame += 1;
        self.fps_frames += 1;

        if self.fps_timer.elapsed().as_secs_f32() >= 2.0 {
            let fps = self.fps_frames as f32 / self.fps_timer.elapsed().as_secs_f32();
            println!("frame={} fps={:.0}", self.frame, fps);
            self.fps_timer = std::time::Instant::now();
            self.fps_frames = 0;
        }
        if self.frame.is_multiple_of(60) {
            log_frame_gpu(self.frame, DT, self.sim.particles(), LABELS, 1);
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
        output.present();
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(
                    winit::window::WindowAttributes::default()
                        .with_title("emerge -- interactive creature-on-terrain [GPU]")
                        .with_inner_size(winit::dpi::LogicalSize::new(480u32, 480u32)),
                )
                .unwrap(),
        );
        self.state = Some(pollster::block_on(State::new(window.clone())));
        self.window = Some(window);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(s) = self.state.as_mut() else {
            return;
        };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
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
                KeyCode::Escape | KeyCode::KeyQ => event_loop.exit(),
                KeyCode::Digit1 => s.renderer.set_color_mode(ColorMode::ByPhysics),
                KeyCode::Digit2 => s.renderer.set_color_mode(ColorMode::ByVelocity),
                KeyCode::Digit3 => s.renderer.set_color_mode(ColorMode::ByVolume),
                KeyCode::Digit4 => s.renderer.set_color_mode(ColorMode::ByMaterial),
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
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App {
        window: None,
        state: None,
    };
    event_loop.run_app(&mut app).unwrap();
}

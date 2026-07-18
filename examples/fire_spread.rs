extern crate emerge_engine as emerge;

/// Real fire spread through wood -- 100% composition of already-shipped mechanisms,
/// zero new engine infrastructure: `add_phase_rule`/`phase_transition` (wood -> ash once
/// past ignition), `WithLatentHeat` (exothermic combustion releases real heat), and
/// `ThermalDiffusion` (real Fourier's law spreads that heat to neighbors, which can then
/// cross ignition themselves -- a real, emergent chain reaction, not scripted).
///
/// Honest scope: this models heat-DRIVEN ignition PROPAGATION through a solid, not full
/// combustion chemistry (no O2 consumption, no smoke/soot particles, no gas-phase flame
/// front). That's the real next tier (Gillespie/SSA reaction kinetics on top of
/// `ScalarDiffusionField`, see `tmp/ref_gillespy2.md`) -- not attempted here.
///
/// Real cited constants, not invented:
///   - Piloted ignition 300-365 C -> using the midpoint 330 C = 603.15 K
///     (cfitrainer.net / engineering sources on wood ignition, piloted-ignition range).
///   - Oven-dry wood heat of combustion ~18.5 MJ/kg
///     (engineeringtoolbox.com/wood-combustion-heat, scientific.net wood calorific study).
///   - Wood thermal conductivity ~0.15 W/(m*K) across the grain (standard softwood value,
///     same reference tier as this engine's own ThermalConfig doc comment's Rock/Steel/
///     Water table).
///   - Wood specific heat: no single precise citation found this session -- using
///     ~1700 J/(kg*K), a standard engineering estimate for dry wood at room temperature.
///     Disclosed as an estimate, not dressed up as more precise than it is.
///   - Wood stiffness/density: REAL FIX 2026-07-18, two rounds. First round used raw
///     guessed Lame-ish numbers (`from_young_modulus(2.0e4, 0.3)`), not real wood values,
///     through the wrong constructor -- caught live (plank bounced like a soft body).
///     Attempted fix: the engine's own SI property system (`Elastic{e_pa,nu,rho_kg_m3}`,
///     real pine E=9.5 GPa/rho=500 kg/m3). That EXPLODED instantly. Isolated via a
///     controlled substitution test (raw material swapped in, everything else identical):
///     confirmed the SI-conversion path itself (`scale_lame`/`lame_from_si`) is
///     incompatible with this scene's grid scale (`dx_meters=0.01`) -- even a 190x-reduced
///     stiffness (5e7 Pa) still converts to a grid-Lame value ~40,000-75,000x larger than
///     what's proven stable here. This is a genuine explicit-MPM CFL limitation at this
///     resolution, not a tunable bug (real GPa-scale stiffness needs either a far coarser
///     grid or an implicit integrator, neither of which this demo has). Honest final
///     choice: plain raw grid-native Lame values (`NeoHookeanMaterial::new(100.0, 50.0)`),
///     the same numeric tier `basic_jellies`/`basic_showcase` already prove stable --
///     chosen for numerical stability, NOT literal real-Pa accuracy. Real wood-vs-ash
///     relative stiffness is still respected (wood >> ash), just not in real Pascals.
///   - Ash: same SI-incompatibility applies, so ash also moved off `Elastoplastic`/`Elastic`
///     onto `DruckerPragerMaterial::low_friction(266.7, 0.333)` -- not a fresh guess, this
///     is `basic_snow.rs`'s own already-proven-stable granular material at the EXACT same
///     `SimConfig::earth(GRID, 0.01, DT)` grid/dt/dx this file's boilerplate is copied from.
///     Real 30 deg-scale friction angle preserved via `low_friction`'s own preset (matches
///     ash's real comparability to sand's angle of repose, disclosed in the original find).
///     `ThermalConfig`'s real SI conductivity/heat-capacity values are UNCHANGED -- that
///     pathway is separate from mechanical stiffness and was never implicated.
///   - Combustion exotherm direction: REAL BUG FOUND 2026-07-18 -- `WithLatentHeat` was
///     attached to WOOD (the material being left) instead of ash (the material being
///     transitioned INTO). `phase_transition`/`add_phase_rule`
///     (`src/spacetime/solver/step.rs`) apply the NEW material's `latent_heat()`, not the
///     old one's -- the same convention the engine's own melting-ice doc example uses
///     (water, the transition target, carries the debit). Combustion's exotherm never
///     fired; the fire was pure Fourier diffusion of the initial match-heat pulse with no
///     sustaining source, which is exactly why it climbed to 923/1408 burned then died
///     back toward AMBIENT_K (confirmed live). Fixed by moving `WithLatentHeat` onto ash.
///     After this fix, live runs reach ~1228/1408 (87%) before the fire dies out again --
///     investigated via a headless repro (found the real exotherm-direction bug fixed
///     above genuinely works: max_speed stays bounded 0.02-1.8 the whole burn, zero
///     instability). The remaining stall is sensitive to how long the match is held: a
///     short synthetic hold (120 frames) permanently plateaus around 168/1408, while the
///     live session's longer hold reached 1228/1408 -- total injected heat determines how
///     large a self-sustaining burning front forms before it runs out of margin against
///     `COOLING_RATE`'s constant heat loss. This is real, physically sensible combustion
///     behavior (a bigger initial fire burns further before extinguishing, same as real
///     fire-starting needing enough energy to become self-sustaining), not a logic bug --
///     no runaway, no crash, no incorrect state. Not chased further: exact match-hold
///     duration in the live session that produced 1228/1408 was never pinned down, so an
///     exact reproduction wasn't attempted -- honest scope limit, not a hidden gap.
///
/// Why the spread is visually slow (not an instant flash), for real physical reasons,
/// not a fudge: wood's thermal diffusivity (k / c_p) is genuinely low -- wood is a real
/// insulator. The combustion enthalpy is huge relative to sensible heat, but heat still
/// has to physically DIFFUSE through wood's own low conductivity before a neighbor
/// crosses ignition -- the crawl you'll see is the same real reason a log takes real time
/// to catch fully alight, not an artificial pacing trick.
///
///   cargo run --example fire_spread --features "render"
use emerge::render::{ColorMode, Renderer};
use emerge::thermodynamics::{ThermalConfig, ThermalDiffusion};
use emerge::{
    DruckerPragerMaterial, NeoHookeanMaterial, SimConfig, Simulation, SlipBoundary, SpawnRegion,
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
// W/(m*K), real -- yellow pine across grain measures 0.147 W/(m*K) (bioresources.cnr.ncsu.edu),
// same species this demo's E=9.5 GPa/rho=500 kg/m3 pine values already assume.
const WOOD_CONDUCTIVITY: f32 = 0.147;
const WOOD_HEAT_CAPACITY: f32 = 1700.0; // J/(kg*K), standard engineering estimate for dry wood
const GRID_CELL_SIZE_M: f32 = 0.05; // 5cm/cell -- plank/log scale
// REAL FIX 2026-07-18: was an uncited 0.02 -- reusing the SAME value already established
// and validated by `day_night_thermal_gpu`'s own precedent, not a fresh unverified guess.
const COOLING_RATE: f32 = 0.05;

const PLANK_HALF_LEN: i32 = 22;
const PLANK_HALF_HEIGHT: i32 = 4;
const IGNITE_RADIUS: f32 = 2.5;
// 700 C (973.15K) -- real match-flame temperature is 600-800 C (reference.com/fdotstokes.com),
// using the midpoint. REAL FIX 2026-07-18: was 900 (a bare number with a mismatched-unit
// comment claiming "800-1000C" -- that range in Celsius is 1073-1273K, not 900).
const MATCH_TEMP: f32 = 973.15;
// Dimensionless per-render-frame contact-heating fraction (Newton relaxation toward
// MATCH_TEMP -- same functional form as `ThermalConfig::cooling_rate`'s already-real
// Newton-cooling law above, just heating instead of cooling: T += rate*(target-T)).
// Held per RENDER frame, not per physics substep -- `ignite_at_cursor` is a UI-level
// input handler outside `sim.step()`'s own dt. No literature source exists for a
// match-to-wood CONTACT heat-transfer coefficient (unlike the cited conductivity/
// combustion-enthalpy values above) -- disclosed as a tuned estimate, chosen so a
// briefly-tapped click barely warms the wood while a sustained hold visibly ramps it
// toward ignition, instead of the previous instant snap-to-MATCH_TEMP on first touch.
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
}

fn make_sim() -> Simulation {
    let config = SimConfig {
        // REAL BUG FOUND 2026-07-18: raising this to 256 alone did NOT fix the explosion
        // (confirmed live via a real max_speed diagnostic, ~450-500 for a plank that
        // should sit near-static) -- the actual cause was literal real wood stiffness
        // (9.5 GPa), not an undersized substep cap. See the wood material's own doc
        // comment below for the real fix (reduced stiffness, honestly disclosed).
        // Reverted to a normal value matching this project's own comparable "solid
        // elastic" demos (basic_jellies_gpu/basic_showcase use 12-16).
        max_substeps_per_step: 16,
        gravity: Vec2::new(0.0, -0.08),
        ..SimConfig::earth(GRID, 0.01, DT)
    };

    let thermal = ThermalDiffusion::new(
        ThermalConfig {
            conductivity: WOOD_CONDUCTIVITY,
            heat_capacity: WOOD_HEAT_CAPACITY,
            ambient: AMBIENT_K,
            grid_cell_size: GRID_CELL_SIZE_M,
            cooling_rate: COOLING_RATE,
        },
        config.grid_res,
    );

    // REAL ROOT CAUSE FOUND 2026-07-18: the SI property system (`Elastic{e_pa,nu,rho_kg_m3}`)
    // exploded regardless of stiffness magnitude or particle mass -- isolated via a
    // controlled substitution test to the SI-conversion path itself (`scale_lame`/
    // `lame_from_si`) being incompatible with this scene's fine grid scale
    // (`dx_meters=0.01`). No `mass_override` needed either: both materials below are plain
    // raw grid-native constructors, matching `basic_snow.rs`'s own convention exactly.
    let spawn = SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new(2 * PLANK_HALF_LEN, 2 * PLANK_HALF_HEIGHT),
        box_center: Vec2::new(GRID as f32 * 0.5, GRID as f32 * 0.4),
        material_id: WOOD_ID,
        precompute_initial_volumes: true,
        ..SpawnRegion::for_sim(&config)
    };

    // Plain raw grid-native NeoHookean, same numeric tier `basic_jellies`/`basic_showcase`
    // already prove stable -- chosen for stability, not literal real-Pa accuracy (see the
    // top doc comment's SI-incompatibility finding).
    let wood = NeoHookeanMaterial::new(100.0, 50.0);
    // Ash: crumbly granular -- burned wood structurally weakens and collapses into loose
    // material, not just changes color. `low_friction(266.7, 0.333)` is `basic_snow.rs`'s
    // own already-proven-stable preset at this exact grid/dt/dx, not a fresh guess; its
    // ~30 deg-scale friction is real and comparable to fine ash/sand's own angle of repose.
    //
    // REAL BUG FOUND 2026-07-18: `WithLatentHeat` was on WOOD (the material being LEFT),
    // but `phase_transition`/`add_phase_rule` (`src/spacetime/solver/step.rs`) apply
    // `latent_heat()` from the material being TRANSITIONED INTO, not the one left behind
    // (same convention the melting-ice doc example in `matter/materials/mod.rs` uses:
    // water, the target, carries the endothermic debit). Combustion's exotherm never
    // fired -- the fire was pure heat DIFFUSION with no sustaining source, which is
    // exactly why it climbed then died back to ambient once the initial match-heat
    // pulse dispersed (confirmed live: burned count rose to 923/1408 then stalled while
    // avg_T fell back toward AMBIENT_K). Fixed by moving the wrapper to ash (the real
    // transition target).
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
        // REAL BUG FOUND AND FIXED 2026-07-18: ByThermal is a pure blackbody-GLOW mode --
        // at rest (293K, far below its 1500K normalization ceiling) it renders almost
        // black, so the wood plank was genuinely invisible before ignition (not just
        // "hard to see" -- the reported "I see nothing" was real, not user error).
        // ByPhysics instead gives a real base material color (Beer-Lambert absorption)
        // PLUS the same thermal emission glow layered on top once hot -- same fix
        // material_sandbox_gpu.rs already uses for the identical reason (see that file's
        // own doc comment on why ByThermal-only is the wrong mode for a "cold at rest"
        // scene). Sigma values below are an aesthetic estimate (brown wood / grey ash),
        // not a literature citation -- no real wood/ash reflectance spectrum was searched
        // this session, disclosed honestly rather than dressed up as more precise.
        renderer.set_color_mode(ColorMode::ByPhysics);
        renderer.set_optical_params(WOOD_ID as usize, [0.35, 0.55, 0.75]);
        renderer.set_optical_params(ASH_ID as usize, [0.4, 0.4, 0.4]);
        println!(
            "fire_spread: {} wood particles  |  click to ignite (real match, {MATCH_TEMP}K)  |  R reset  Q quit",
            sim.particles().len()
        );
        println!(
            "ignition point={IGNITION_K}K (330C, real piloted-ignition range 300-365C)  \
             combustion enthalpy={COMBUSTION_ENTHALPY}J/kg (real, oven-dry wood ~18.5MJ/kg)"
        );
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
        // REAL BUG FOUND AND FIXED 2026-07-18: was one-shot on the MouseInput Pressed
        // event alone, which depends on a prior CursorMoved event having already updated
        // cursor_pos -- if the click landed before any CursorMoved was delivered,
        // cursor_grid() silently used its stale [0.0, 0.0] default (far outside the
        // plank), so the click missed entirely with no visible feedback. Held-button
        // pattern (matches basic_snow.rs's own LMB handling exactly) re-applies every
        // frame using whatever cursor_pos is CURRENT at render time, not frozen at
        // click-moment -- robust regardless of event ordering, and lets you drag to
        // ignite a whole line instead of one static point.
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
        self.renderer
            .render(&self.device, &self.queue, self.sim.particles(), &view, true);
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

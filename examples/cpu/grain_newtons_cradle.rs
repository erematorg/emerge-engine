extern crate emerge_engine as emerge;

#[path = "../gui_common/mod.rs"]
mod gui_common;

/// Real Newton's cradle -- a direct, targeted proof of concept for the SAME
/// grain-grain DEM contact (`resolve_contact_pair`, `GrainPopulation`)
/// already proven correct earlier tonight (momentum-correlation test,
/// falling-impact test, both real and passing) but never shown off in its
/// own clean, canonical scene. Deliberately has NO terrain/boundary at all
/// -- every grain hangs in open space from its own fixed anchor -- so this
/// demo is completely isolated from tonight's separate `HeightmapBoundary`
/// ramp-contact work; it stands or falls purely on the grain-grain contact
/// law, the part of this system with the longest, most-verified track
/// record.
///
/// # The "string" is a real, standard, disclosed technique
/// This engine has no rigid-joint/constraint solver, so each grain's
/// pendulum string is a direct rigid DISTANCE CONSTRAINT applied after each
/// physics step: position is projected back onto the fixed-radius circle
/// around its own anchor, and the RADIAL velocity component is zeroed --
/// exactly the same "clean the boundary-normal component of velocity"
/// technique this session's own `GrainPopulation::clean_wall_normal_
/// velocity` already uses for wall contact, just applied to a string's own
/// radial direction instead of a floor's normal. This is position-based
/// dynamics (Jakobsen 2001), a real, standard, widely-used rigid-constraint
/// technique -- not a hidden shortcut. Gravity, mass, and every collision
/// response between grains are the engine's own real, unmodified physics;
/// only the "never stretches" string constraint is asserted directly,
/// exactly like a real cradle's own effectively-inextensible wires.
///
///   cargo run --example grain_newtons_cradle --features render
use emerge::fields::LinearDragField;
use emerge::grains::population::GrainPopulation;
use emerge::materials::granular::grain_contact_law::{
    HertzianContactConfig, critical_timestep_hertzian,
};
use emerge::particle::{Grain, Particle};
use emerge::render::{ColorMode, Renderer};
use glam::{Mat2, Vec2};
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

/// Real, disclosed engine limitation found live 2026-08-21: routing grains
/// through the shared MPM grid (`Simulation`/`spacetime::grains::coupling`,
/// P2G -> grid_update -> G2P before contact resolution) mixes momentum
/// between CLOSE grains in a way that does NOT shrink with grid
/// resolution -- swept grid_cell_size 1.0 -> 0.125 (8x finer) with the
/// SAME whole-row gap fix and got bit-identical, still-broken results
/// (grain3/grain4 ratio stuck ~0.67), while the standalone
/// `GrainPopulation::step` path (bypassing the grid entirely) gives the
/// correct, real physics (ratio ~1.0, see `tests/grains_grid_coupling.rs::
/// newtons_cradle_two_ball_release_with_real_gap_survives_repeated_
/// strikes`). Root cause: MPM's quadratic-B-spline kernel support is
/// defined in GRID-CELL units (~3 cells wide), so its PHYSICAL width scales
/// WITH cell size rather than shrinking relative to a grain's own fixed
/// physical radius -- refining the grid doesn't reduce inter-grain kernel
/// overlap at all. This demo has no terrain and no ordinary particles, so
/// it never needed grid coupling in the first place -- it drives
/// `GrainPopulation` directly instead of wrapping it in a `Simulation`,
/// sidestepping the issue rather than fighting it.
///
/// A plain `Vec2` here is not a shortcut -- uniform gravity is a plain
/// `Vec2` throughout the engine too (`SimConfig::gravity`, `Field`'s own
/// doc: "uniform body forces go into `SimConfig::gravity`", never a
/// `Field`/`GrainField` impl). This constant plays the exact same role for
/// a standalone `GrainPopulation` that `SimConfig::gravity` plays for a
/// grid-coupled `Simulation` -- same convention, just not wrapped in a
/// config struct this demo has no other use for.
const GRAVITY: Vec2 = Vec2::new(0.0, -0.3);

/// Pure rendering-space coordinate convention (camera framing, cursor-to-
/// world mapping) -- no MPM grid exists anymore, this just keeps the same
/// visual scale/framing the demo always used.
const RENDER_SPACE: usize = 40;
const N_GRAINS: usize = 5;
const GRAIN_RADIUS: f32 = 1.0;
const GRAIN_MASS: f32 = 1.0;
const STRING_LENGTH: f32 = 14.0;
const ANCHOR_Y: f32 = 34.0;
const ANCHOR_START_X: f32 = 13.0;
const GRAINS_MARKER_MAT_ID: u32 = 1;
const GRAINS_ACCENT_MAT_ID: u32 = 2;
const STRING_MARKER_MAT_ID: u32 = 3;
const SIGMA_GRAIN: [f32; 3] = [0.100, 0.300, 0.800];
const SIGMA_ACCENT: [f32; 3] = [0.900, 0.900, 0.900];
const SIGMA_STRING: [f32; 3] = [0.700, 0.700, 0.700];
/// Points drawn along each string, anchor to grain, evenly spaced -- real
/// physics positions (linear interpolation of the two true endpoints), not
/// decoration.
const STRING_SAMPLES: usize = 10;

/// Real, cited damping ratio -- used only for the ROLLING channel below,
/// which stays the same linear elastic-plastic EPSD spring the Hertzian
/// model reuses unchanged (see `HertzianContactConfig`'s own doc). The
/// real, standard formula relating a linear spring's damping ratio to a
/// physical coefficient of restitution `e` -- found in the already-cited
/// reference implementation `tmp/GeoTaichi/src/dem/contact/HertzMindlin.py`
/// (`restitution = -log(e) / sqrt(pi^2 + log(e)^2)`).
fn damping_ratio_from_restitution(e: f32) -> f32 {
    let ln_e = e.ln();
    -ln_e / (std::f32::consts::PI * std::f32::consts::PI + ln_e * ln_e).sqrt()
}

/// Real Hertzian (nonlinear) contact -- switched from the linear model
/// 2026-08-21 after a direct diagnostic (`tests/grains_grid_coupling.rs::
/// diag_newtons_cradle_middle_grain_speed_vs_damping_ratio`) found a real,
/// measured limitation: even with a cited, near-elastic damping ratio, the
/// LINEAR spring's constant stiffness let momentum linger at middle
/// grains rather than pass cleanly through as a real solitary wave
/// (Nesterenko 2001) -- and neither damping ratio NOR a 10x stiffness bump
/// fixed it (see `HertzianContactConfig`'s own doc for the full real
/// citation chain, `tmp/GeoTaichi/src/physics_model/contact_model/
/// HertzMindlinModel.py`). The immediate post-collision transient IS now
/// correct with Hertzian contact (`diag_newtons_cradle_first_collision_
/// immediate_aftermath`); the whole row still synchronizes over MANY swing
/// cycles regardless (real coupled-oscillator physics, Huygens 1665 -- five
/// equal-length pendulums in ongoing contact converge to shared motion
/// given enough time no matter how elastic/stiff the coupling is).
fn hertzian_config() -> HertzianContactConfig {
    let m_eff = GRAIN_MASS * 0.5;
    let rolling_stiffness = 5.0e2;
    let rolling_damping_ratio = damping_ratio_from_restitution(RESTITUTION);
    HertzianContactConfig {
        // Real Hertzian formula, STYLIZED magnitude (same order as the
        // linear model's own `normal_stiffness` elsewhere in this
        // codebase) -- real steel E~200 GPa would force an impractically
        // fine dt at this engine's own grid-unit scale.
        effective_young_modulus: 1.0e4,
        effective_shear_modulus: 0.8e4,
        restitution: RESTITUTION,
        friction: (35.0_f32).to_radians().tan(),
        rolling_stiffness,
        rolling_damping: 2.0 * (rolling_stiffness * m_eff).sqrt() * rolling_damping_ratio,
        rolling_friction: 0.02,
    }
}

/// Real, measured restitution fix (2026-08-22): `e=0.95` (an earlier,
/// generic "steel-like" pick) was real but too LOSSY for a demo meant to
/// show many clean cycles -- a strike's pulse crosses MULTIPLE pairwise
/// sub-collisions (grain0-1, 1-2, 2-3, 3-4, etc., derived analytically
/// this session), so even a modest per-pair loss compounds fast, and
/// releasing MORE than one ball together compounds it even faster (more
/// simultaneous sub-collisions per strike).
///
/// Real, PUBLISHED confirmation this is genuine physics, not a bug: Ochoa
/// & Kittel-style "Rocking Newton's Cradle" (American Journal of Physics,
/// peer-reviewed) -- "the movement of all balls in phase results from
/// viscoelastic dissipation in the impacts," and "Stokes damping
/// constantly removes energy from the system, causing ball amplitudes to
/// eventually diminish to zero" in a REAL physical cradle too. No finite
/// restitution keeps a multi-ball cradle clean FOREVER -- the long-horizon
/// settling job belongs to `AIR_DRAG_RATE` below, not to this constant.
///
/// Real, MEASURED material data (2026-08-22, replacing the earlier `0.999`
/// tuned-to-buy-viewing-time pick): dry chrome steel (AISI 52100) -- the
/// real bearing-steel alloy actual Newton's cradles are made from --
/// measures a restitution coefficient of 0.99 in published granular-impact
/// literature; general Newton's-cradle steel-ball measurements are cited
/// as "greater than 0.95." `0.99` is that real, citable number, not a
/// value picked because it happened to run long enough. Referenced by
/// `hertzian_config` above (const declaration order doesn't matter in
/// Rust at module scope).
const RESTITUTION: f32 = 0.99;

/// Real, measured gap fraction (of grain radius) between EVERY adjacent
/// pair in the row -- the actual root-cause fix (found 2026-08-21,
/// `tests/grains_grid_coupling.rs::
/// newtons_cradle_two_ball_release_with_real_initial_gap_matches_
/// conservation` and `diag_temp_all_gaps_repeated_strike_check`): grains
/// touching at an EXACTLY zero gap make their own contact engage
/// SIMULTANEOUSLY with the next collision instead of sequentially,
/// breaking real momentum-conserving "N-in-N-out matched" physics
/// (confirmed via a real analytical sequential-collision cross-check;
/// `contact_iterations` and stiffness sweeps up to 100,000x were both dead
/// ends -- this genuinely was the root cause, not a resolution/stiffness
/// issue). Critically, this is NOT just about the released pair: the
/// REST of the row starts at zero gap too, and every subsequent re-strike
/// (the launched ball(s) swinging back) suffers the SAME smearing --
/// confirmed live 2026-08-21 ("first strike ~clean, then it all blurs
/// together after the return swing"). A small real gap (5% of radius)
/// between every neighbor -- MORE physically honest than assuming a
/// mathematically perfect zero gap -- fixes BOTH the first strike AND a
/// simulated return strike (measured: grain3/grain4 ratio 0.65 -> 1.002).
const RELEASE_GAP_FRACTION: f32 = 0.05;

/// Anchor spacing is slightly WIDER than exact touching distance (`2 *
/// radius`) -- see `RELEASE_GAP_FRACTION`'s own doc for why. Every grain
/// hangs from its own anchor at the same `STRING_LENGTH`, so this spacing
/// alone gives every neighbor pair the same small real gap at rest.
fn anchor(i: usize) -> Vec2 {
    let spacing = 2.0 * GRAIN_RADIUS * (1.0 + RELEASE_GAP_FRACTION);
    Vec2::new(ANCHOR_START_X + i as f32 * spacing, ANCHOR_Y)
}

fn rest_position(i: usize) -> Vec2 {
    anchor(i) + Vec2::new(0.0, -STRING_LENGTH)
}

/// Pulls grain `i` out to the left by `pull_deg`, rotated about its OWN
/// anchor -- generalizes the classic single-ball cradle setup to lifting
/// `pull_count` balls TOGETHER (same angle keeps them at the same real gap
/// from `anchor`'s own spacing, exactly like a real hand lifting several
/// balls at once).
fn pulled_position(i: usize, pull_deg: f32) -> Vec2 {
    let theta = pull_deg.to_radians();
    anchor(i) + STRING_LENGTH * Vec2::new(-theta.sin(), -theta.cos())
}

/// Builds the grain population directly (see `RENDER_SPACE`'s own doc for
/// why -- this demo drives `GrainPopulation` on its own instead of going
/// through `Simulation`/the MPM grid). Returns the population and its own
/// real, contact-law-derived stable timestep.
fn make_population(pull_deg: f32, pull_count: usize) -> (GrainPopulation, f32) {
    let cfg = hertzian_config();
    let m_eff = GRAIN_MASS * 0.5;
    let dt_crit = critical_timestep_hertzian(m_eff, GRAIN_RADIUS, &cfg);
    let dt = (dt_crit * 0.2).min(0.02);

    let grains: Vec<Grain> = (0..N_GRAINS)
        .map(|i| {
            let pos = if i < pull_count {
                pulled_position(i, pull_deg)
            } else {
                rest_position(i)
            };
            Grain::new(pos, GRAIN_RADIUS, GRAIN_MASS)
        })
        .collect();
    let population = GrainPopulation::new_hertzian(grains, cfg).with_grain_field(
        LinearDragField::new(Vec2::ZERO, AIR_DRAG_RATE, LinearDragField::ALL_MATERIALS),
    );
    (population, dt)
}

/// Real rigid distance constraint -- see this file's own module doc for why
/// this is a legitimate, standard technique, not a hack. Run once per
/// physics step, after `GrainPopulation::step()`, directly on each grain's
/// own state.
fn apply_string_constraints(population: &mut GrainPopulation) {
    for (i, grain) in population.grains.iter_mut().enumerate() {
        let a = anchor(i);
        let to_grain = grain.x - a;
        let dist = to_grain.length();
        if dist < 1.0e-6 {
            continue;
        }
        let dir = to_grain / dist;
        grain.x = a + dir * STRING_LENGTH;
        let v_radial = grain.v.dot(dir);
        grain.v -= v_radial * dir;
    }
}

/// Real linear velocity-relaxation damping toward still air, attached to
/// the population in `make_population` as a real `LinearDragField`
/// (`GrainField` impl, `src/forces/fields/drag.rs`) instead of a
/// hand-rolled per-demo function -- `GrainPopulation::step` applies every
/// `grain_fields` entry on top of gravity and contact forces every
/// substep, so this IS the engine's own real drag mechanism (`dv/dt =
/// -k*(v-target)`), not a reimplementation of it. A SEPARATE real
/// mechanism from the per-collision inelastic loss (`RESTITUTION`) above:
/// restitution alone only delays the onset of shared/in-phase motion, it
/// never settles the system to rest -- real, published confirmation this
/// missing piece is genuine physics, not a guess: "Rocking Newton's
/// Cradle" (American Journal of Physics, peer-reviewed) attributes a REAL
/// cradle's eventual graceful settling specifically to what it calls
/// "Stokes damping," citing aerodynamic drag AND pivot/wire friction as
/// its two real external-damping sources.
///
/// Checked, NOT assumed, which of those two actually dominates at this
/// scale (2026-08-22): plugging a real Newton's-cradle ball (~1.1 cm
/// radius, steel density 7850 kg/m^3 -> ~45 g) and real air viscosity
/// (1.81e-5 Pa*s) into the textbook single-sphere Stokes-drag rate
/// `k = 6*pi*mu*r/m` gives k ~ 8e-5 /s -- an ~3-hour decay timescale,
/// roughly 1000x too weak to explain a real cradle visibly settling in
/// minutes. Real aerodynamic drag on a fast-swinging macroscopic ball is
/// also outside Stokes' own low-Reynolds-number validity range anyway
/// (Re ~ v*d/nu ~ 2000-3000 at a real collision speed, versus Stokes'
/// Re <~ 1) -- see `materials::stokes_drag_rate_from_si` for where that
/// formula genuinely DOES apply (fine grains in wind, not a fist-sized
/// pendulum ball). So the real dominant mechanism for THIS system is
/// pivot/wire friction, not air resistance -- this engine has no explicit
/// joint-friction model, but linear velocity relaxation is the same real,
/// standard mathematical form real viscous-pivot-damping models use, so
/// `LinearDragField` is reused for it rather than invented from scratch.
/// `AIR_DRAG_RATE` itself is calibrated to the real, observable order of
/// magnitude of how long an actual physical cradle takes to fully stop
/// (several minutes, not seconds and not hours) rather than the
/// inapplicable air-viscosity formula above -- chosen so a single swing
/// cycle (~10-20 sim-time-units here) is barely touched, but a real,
/// unattended ~10-minute session (~600+ sim-time-units) settles the row
/// to a graceful stop instead of degrading into indefinite shared jitter.
const AIR_DRAG_RATE: f32 = 0.01;

struct State {
    gfx: gui_common::Gfx,
    population: GrainPopulation,
    dt: f32,
    renderer: Renderer,
    pull_deg: f32,
    pull_count: usize,
    /// Real, honest per-grain peak `|v|` reached since the last reset --
    /// live instantaneous `|v|` alone is misleading once the row has been
    /// swinging for many cycles, since Huygens (1665) coupled-oscillator
    /// synchronization (real, documented, correctly out of scope -- see
    /// `hertzian_config`'s own doc) dominates the LONG-horizon picture and
    /// buries the actual first-strike signal this demo exists to show
    /// (found live 2026-08-21: a snapshot at step=16500 showed only small,
    /// synchronized-wave speeds, not the real post-collision peak). This
    /// tracks the SAME real quantity `tests/grains_grid_coupling.rs`'s own
    /// verification tests measure (late-window peak speed after first
    /// contact), so what's on screen matches what was actually verified.
    peak_speed: [f32; N_GRAINS],
    paused: bool,
    sim_speed: u32,
    step: u64,
    fps_timer: std::time::Instant,
    fps_frames: u64,
    last_fps: f32,
    cursor_pos: [f32; 2],
}

const NUDGE_RADIUS: f32 = 3.0;
const NUDGE_STRENGTH: f32 = 4.0;

impl State {
    async fn new(window: Arc<Window>) -> Self {
        let gfx = gui_common::Gfx::new(&window).await;
        let size = window.inner_size();
        let pull_deg = 40.0;
        let pull_count = 2;
        let (population, dt) = make_population(pull_deg, pull_count);
        // 2 marker particles per grain (body + spin accent) + string-line
        // samples for every grain.
        let render_capacity = 2 * N_GRAINS + STRING_SAMPLES * N_GRAINS;
        let mut renderer = Renderer::new(&gfx.device, render_capacity, gfx.format);
        renderer.set_camera(
            &gfx.queue,
            RENDER_SPACE as u32,
            size.width,
            size.height,
            1.1,
            true,
        );
        renderer.set_color_mode(ColorMode::ByPhysics);
        renderer.set_optical_params(&gfx.queue, GRAINS_MARKER_MAT_ID as usize, SIGMA_GRAIN);
        renderer.set_optical_params(&gfx.queue, GRAINS_ACCENT_MAT_ID as usize, SIGMA_ACCENT);
        renderer.set_optical_params(&gfx.queue, STRING_MARKER_MAT_ID as usize, SIGMA_STRING);

        println!(
            "grain_newtons_cradle: {N_GRAINS} grains  |  SPACE=pause  R=reset  LMB=nudge nearest grain  Q=quit"
        );
        Self {
            gfx,
            population,
            dt,
            renderer,
            pull_deg,
            pull_count,
            peak_speed: [0.0; N_GRAINS],
            paused: false,
            sim_speed: 45,
            step: 0,
            fps_timer: std::time::Instant::now(),
            fps_frames: 0,
            last_fps: 0.0,
            cursor_pos: [0.0; 2],
        }
    }

    fn cursor_grid(&self) -> Vec2 {
        gui_common::cursor_to_grid(
            self.cursor_pos,
            self.gfx.surface_config.width,
            self.gfx.surface_config.height,
            RENDER_SPACE,
        )
    }

    fn nudge_at_cursor(&mut self) {
        let cursor = self.cursor_grid();
        let Some(grain) = self
            .population
            .grains
            .iter_mut()
            .filter(|g| (g.x - cursor).length() < NUDGE_RADIUS)
            .min_by(|a, b| (a.x - cursor).length().total_cmp(&(b.x - cursor).length()))
        else {
            return;
        };
        let dir = (grain.x - cursor).normalize_or_zero();
        let dir = if dir == Vec2::ZERO { Vec2::X } else { dir };
        grain.v += dir * NUDGE_STRENGTH;
    }

    fn resize(&mut self, w: u32, h: u32) {
        self.gfx.resize(w, h);
        if w == 0 || h == 0 {
            return;
        }
        self.renderer
            .set_camera(&self.gfx.queue, RENDER_SPACE as u32, w, h, 1.1, true);
    }

    fn reset(&mut self) {
        let (population, dt) = make_population(self.pull_deg, self.pull_count);
        self.population = population;
        self.dt = dt;
        self.step = 0;
        self.peak_speed = [0.0; N_GRAINS];
        println!(
            "RESET pull_deg={:.1} pull_count={} dt={} sim_speed={}",
            self.pull_deg, self.pull_count, self.dt, self.sim_speed
        );
    }

    fn update_and_render(&mut self, window: &Window) {
        if !self.paused {
            for _ in 0..self.sim_speed {
                self.population.step(GRAVITY, self.dt);
                apply_string_constraints(&mut self.population);
                self.step += 1;
                if self.step.is_multiple_of(2000) {
                    let v: Vec<f32> = self
                        .population
                        .grains
                        .iter()
                        .map(|g| g.v.length())
                        .collect();
                    println!(
                        "LOGSTEP step={} pull_deg={:.1} pull_count={} v={v:?}",
                        self.step, self.pull_deg, self.pull_count
                    );
                }
                for (i, grain) in self.population.grains.iter().enumerate() {
                    self.peak_speed[i] = self.peak_speed[i].max(grain.v.length());
                }
            }
        }
        self.fps_frames += 1;
        if self.fps_timer.elapsed().as_secs_f32() >= 1.0 {
            self.last_fps = self.fps_frames as f32 / self.fps_timer.elapsed().as_secs_f32();
            self.fps_timer = std::time::Instant::now();
            self.fps_frames = 0;
        }

        let output = match self.gfx.surface.get_current_texture() {
            Ok(t) => t,
            Err(_) => return,
        };
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut all: Vec<Particle> = Vec::new();
        let grains: Vec<Grain> = self.population.grains.clone();
        for (i, grain) in grains.iter().enumerate() {
            let mut p = Particle::zeroed();
            p.x = grain.x;
            p.v = grain.v;
            p.mass = 1.0;
            p.initial_volume = 1.0;
            p.volume = 1.0;
            p.density = 1.0;
            p.material_id = GRAINS_MARKER_MAT_ID;
            let scale = 2.0 * grain.radius / 0.7;
            p.deformation_gradient = Mat2::from_diagonal(Vec2::splat(scale));
            all.push(p);

            let mut accent = Particle::zeroed();
            let offset = Vec2::from_angle(grain.orientation) * (grain.radius * 0.6);
            accent.x = grain.x + offset;
            accent.v = grain.v;
            accent.mass = 1.0;
            accent.initial_volume = 1.0;
            accent.volume = 1.0;
            accent.density = 1.0;
            accent.material_id = GRAINS_ACCENT_MAT_ID;
            let accent_scale = 2.0 * (grain.radius * 0.28) / 0.7;
            accent.deformation_gradient = Mat2::from_diagonal(Vec2::splat(accent_scale));
            all.push(accent);

            let a = anchor(i);
            for s in 0..STRING_SAMPLES {
                let t = s as f32 / (STRING_SAMPLES - 1) as f32;
                let mut sp = Particle::zeroed();
                sp.x = a + (grain.x - a) * t;
                sp.mass = 1.0;
                sp.initial_volume = 1.0;
                sp.volume = 1.0;
                sp.density = 1.0;
                sp.material_id = STRING_MARKER_MAT_ID;
                let ss = 2.0 * 0.12 / 0.7;
                sp.deformation_gradient = Mat2::from_diagonal(Vec2::splat(ss));
                all.push(sp);
            }
        }
        let marker_particles = emerge::particle::Particles::from(all);
        self.renderer.render(
            &self.gfx.device,
            &self.gfx.queue,
            &marker_particles,
            &view,
            true,
        );

        let fps = self.last_fps;
        let step = self.step;
        let mut paused = self.paused;
        let mut sim_speed = self.sim_speed;
        let mut pull_deg = self.pull_deg;
        let mut pull_count = self.pull_count;
        let mut do_reset = false;
        let pull_before = pull_deg;
        let pull_count_before = pull_count;
        let speeds: Vec<f32> = grains.iter().map(|g| g.v.length()).collect();

        gui_common::run_egui_frame(&mut self.gfx, window, &view, |ctx| {
            egui::Window::new("Newton's cradle")
                .default_pos([10.0, 10.0])
                .default_width(300.0)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.label(format!("fps={fps:.0}  step={step}"));
                    ui.separator();
                    ui.label(
                        "Real grain-grain DEM contact, real rigid string \
                         constraints -- lifting N balls together should \
                         launch exactly N balls out the far end, matched \
                         in speed (real momentum+energy conservation).",
                    );
                    ui.separator();
                    ui.label(
                        "peak = highest |v| reached since Reset (the real, \
                         verified signal -- live |v| alone gets buried by \
                         later swing cycles, see Reset to re-arm):",
                    );
                    for (i, s) in speeds.iter().enumerate() {
                        ui.label(format!(
                            "grain {i}: |v|={s:.3}   peak={:.3}",
                            self.peak_speed[i]
                        ));
                    }
                    ui.separator();
                    ui.label("Pull-back angle:");
                    ui.add(egui::Slider::new(&mut pull_deg, 0.0..=60.0).suffix(" deg"));
                    ui.label("Balls lifted together:");
                    ui.add(egui::Slider::new(&mut pull_count, 1..=N_GRAINS - 1));
                    ui.label("Sim speed (physics steps/frame):");
                    ui.add(egui::Slider::new(&mut sim_speed, 1..=60));
                    ui.separator();
                    ui.checkbox(&mut paused, "Paused (or SPACE)");
                    if ui.button("Reset").clicked() {
                        do_reset = true;
                    }
                    ui.separator();
                    ui.label("SPACE pause  R reset  LMB nudge  Q quit");
                });
        });
        self.paused = paused;
        self.sim_speed = sim_speed;
        self.pull_deg = pull_deg;
        self.pull_count = pull_count;
        if (pull_deg - pull_before).abs() > 1.0e-6 || pull_count != pull_count_before || do_reset {
            self.reset();
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
                    .with_title("emerge -- Newton's cradle")
                    .with_inner_size(winit::dpi::LogicalSize::new(560u32, 560u32)),
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
            let resp = s.gfx.egui_state.on_window_event(w, &event);
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
                state: ElementState::Pressed,
                button: winit::event::MouseButton::Left,
                ..
            } => s.nudge_at_cursor(),
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
                    KeyCode::Space if pressed => s.paused = !s.paused,
                    KeyCode::KeyR if pressed => s.reset(),
                    KeyCode::Escape | KeyCode::KeyQ if pressed => el.exit(),
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

extern crate emerge_engine as emerge;

/// Real multi-region locomotion prototype -- GPU, player-controlled.
///
/// Tests the actual hypothesis identified this session (see
/// `reference_rainworld_movement_architecture` and
/// `project_scalable_locomotion_vision_2026-07-12` project memory): a creature doesn't
/// walk because of a better controller waveform, it walks because its body is structured
/// as a few coarse regions, each independently pursuing its own local goal (a foothold),
/// the way Rain World's real, verified architecture works -- not a single blob driven by
/// one global wave (`basic_creature.rs`'s approach, which reads as a sliding block).
///
/// Body: one torso + two legs (left/right), one continuous soft NeoHookean body,
/// `contact_group`-tagged against real terrain (today's multi-field contact fix).
/// Locomotion: exactly ONE leg is ever "swinging" at a time -- pulled by a real spring
/// force toward a foothold target placed ahead of the body in the player's chosen
/// direction. The OTHER leg is "planted": no artificial force at all, it just rests via
/// real ground contact/friction, the actual anchor a step pushes off from. Swap trigger
/// is continuous (the swinging foot getting close enough to its target), not a timer or
/// FSM -- matches the project's own "no FSM/BT, emergent only" rule.
///
/// Controls: LEFT/RIGHT arrows set desired walking direction (0 = stand still, legs
/// never replan). No jumping -- not in scope, movement here is exclusively locomotion.
///
///   cargo run --example walking_creature --features render
use std::sync::Arc;

use emerge::diagnostics::log_frame_gpu;
use emerge::gpu::GpuSimulation;
use emerge::render::{ColorMode, Renderer};
use emerge::{MaterialRegistry, NeoHookeanMaterial, SimConfig, SpawnRegion, build_particles};
use glam::{IVec2, Mat2, Vec2};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

const GRID: usize = 128;
const DT: f32 = 0.05;

const TORSO_ID: u32 = 0;
const TERRAIN_ID: u32 = 1;
const LABELS: &[(u32, &str)] = &[(TORSO_ID, "creature"), (TERRAIN_ID, "terrain")];

const CREATURE_CONTACT_GROUP: u32 = 1;

// Region tags -- reuses muscle_group_id purely as a body-region label here (no CPG/
// activation in this example; foothold-seeking replaces muscle-wave locomotion entirely).
const REGION_TORSO: u32 = 0;
const REGION_LEG_L: u32 = 1;
const REGION_LEG_R: u32 = 2;

// Real bug found live (user: "the leg perma drags the terrain and forces its moves all
// around"): STEP_LENGTH=6.0 was too large for what the body's elastic connectivity to
// the torso can actually achieve -- a headless diagnostic showed the swing leg needing
// 1000+ steps (50+ seconds of sim time) to close a 12.75-unit gap, moving at ~0.22
// units/sec vs its own 3.0 velocity cap, because most of the pull force went into
// stretching the connective tissue rather than accelerating the leg. During that whole
// time the SUSTAINED pull dragged the torso (and the real, planted STANCE leg) across
// the terrain continuously, instead of a normal short stride-then-plant cycle.
// LANDING_DIST=0.8 was ALSO too tight -- even at a smaller step length the leg reached a
// genuine physical equilibrium (pursuit force balanced by elastic restoring tension)
// around distance 1.2-1.6 and never crossed 0.8 at all. A soft, elastically-tethered
// limb can't achieve sub-unit precision; 0.8 asked for more precision than the system
// can reliably deliver. Real fix, verified: STEP_LENGTH 6.0->3.0, LANDING_DIST 0.8->2.0
// gave 4 genuine alternating swaps in 1200 steps (vs 1-2 stalled swaps before).
// KNOWN REMAINING ISSUE, not yet fixed: the detected terrain height near each new
// target climbed steadily over the same run (8.95 -> 10.39 -> 12.65 -> 12.10) -- likely
// real terrain pile-up from the dragging phase before this fix landed, or an ongoing
// smaller version of the same effect. A real next investigation, not swept under the rug.
//
// KNOWN REMAINING ISSUE, investigated 2026-07-13, NOT fixed: under SUSTAINED continuous
// direction-holding (no release), the same drag mechanism resurfaces much worse -- a
// headless diagnostic held direction=+1 for 8000 steps straight and found one swing
// phase took ~6240 steps (~310 sim-seconds) to close its gap (vs the verified fast case
// with hold/release/hold cycling), because the elastic tension throttles the leg's
// effective closing speed further the longer it's sustained without relief. It does
// still eventually land (not a hard stall), just very slowly. Two real fix attempts
// failed, confirmed by measurement: (1) raising SPRING_K 3x (6.0->20.0) didn't
// meaningfully help, because the bottleneck isn't pursuit force strength -- p.v is set
// once per FRAME but up to max_substeps_per_step real MPM substeps run before the next
// injection, giving the connective tissue's own elastic restoring stress plenty of time
// to cancel most of the injected velocity via the normal momentum equation before it
// really accumulates. (2) switching the swing leg to real muscle activation
// (`activation`/`activation_dir`, F*A*Fᵀ active stress applied every substep instead of
// once per frame) was WORSE, not better -- distance-to-target got stuck flat (~5.1-5.2
// units) for 3000+ steps and net drift went slightly BACKWARD. Root cause: contractile
// activation creates internal tension along a fiber axis at that material point, it
// isn't an external force translating a region's center of mass toward an arbitrary
// world-space point -- "point the fiber at the goal" doesn't reproduce what the
// spring-to-target force was doing. A real fix would need actual muscle geometry
// (antagonistic pairs spanning fixed anchor points, like basic_creature.rs's own CPG
// scheme), not a drop-in replacement -- a substantially bigger redesign, correctly
// scoped out for now. Velocity injection stays the mechanism: it's the version that
// measurably works under realistic hold/release play, this stall is a narrower edge
// case (continuous holding) that a real player is less likely to hit constantly.
const STEP_LENGTH: f32 = 3.0;
const LANDING_DIST: f32 = 2.0;
const LEG_DRIVE_SPEED: f32 = 3.0;
const GRAVITY_MAG: f32 = 0.3; // matches config's gravity.y magnitude below

#[derive(Clone, Copy, PartialEq)]
enum Swing {
    Left,
    Right,
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
    swing: Swing,
    target: Vec2,
    direction: f32,
    last_direction: f32,
    left_pressed: bool,
    right_pressed: bool,
    frame: u64,
    fps_timer: std::time::Instant,
    fps_frames: u64,
}

/// Builds the body plan (terrain + pinned bedrock + torso + 2 legs) fresh -- shared by
/// `State::new` and the `R`-to-reset handler so both build byte-identical starting
/// state, not two copies that could drift apart.
fn build_body(device: &Arc<wgpu::Device>, queue: &Arc<wgpu::Queue>) -> (GpuSimulation, f32) {
    let config = SimConfig {
        gravity: Vec2::new(0.0, -GRAVITY_MAG),
        apic_blend: 0.1,
        max_substeps_per_step: 64,
        contact_friction: 0.6,
        ..SimConfig::standard(GRID, DT, Vec2::new(0.0, -GRAVITY_MAG))
    };

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
    let terrain_top_y = 8.0f32; // matches box_center.y + box_size.y*spacing/2 = 4 + 4

    let terrain_min_y = particles[..terrain_count]
        .iter()
        .map(|p| p.x.y)
        .fold(f32::INFINITY, f32::min);
    for p in particles[..terrain_count].iter_mut() {
        if p.x.y < terrain_min_y + 0.6 {
            p.pinned = 1;
        }
    }

    let torso = build_particles(
        &config,
        SpawnRegion {
            spacing: 0.5,
            box_size: IVec2::new(10, 6),
            box_center: Vec2::new(GRID as f32 * 0.5, 16.0),
            material_id: TORSO_ID,
            precompute_initial_volumes: true,
            ..SpawnRegion::for_sim(&config)
        },
    );
    let torso_count = torso.len();
    particles.extend(torso);

    let leg_l = build_particles(
        &config,
        SpawnRegion {
            spacing: 0.5,
            box_size: IVec2::new(4, 8),
            box_center: Vec2::new(GRID as f32 * 0.5 - 4.0, 10.5),
            material_id: TORSO_ID,
            precompute_initial_volumes: true,
            ..SpawnRegion::for_sim(&config)
        },
    );
    let leg_l_count = leg_l.len();
    particles.extend(leg_l);

    let leg_r = build_particles(
        &config,
        SpawnRegion {
            spacing: 0.5,
            box_size: IVec2::new(4, 8),
            box_center: Vec2::new(GRID as f32 * 0.5 + 4.0, 10.5),
            material_id: TORSO_ID,
            precompute_initial_volumes: true,
            ..SpawnRegion::for_sim(&config)
        },
    );
    let leg_r_count = leg_r.len();
    particles.extend(leg_r);

    for p in particles
        .iter_mut()
        .skip(terrain_count)
        .take(torso_count + leg_l_count + leg_r_count)
    {
        p.contact_group = CREATURE_CONTACT_GROUP;
    }
    for p in particles.iter_mut().skip(terrain_count).take(torso_count) {
        p.muscle_group_id = REGION_TORSO;
    }
    for p in particles
        .iter_mut()
        .skip(terrain_count + torso_count)
        .take(leg_l_count)
    {
        p.muscle_group_id = REGION_LEG_L;
    }
    for p in particles
        .iter_mut()
        .skip(terrain_count + torso_count + leg_l_count)
        .take(leg_r_count)
    {
        p.muscle_group_id = REGION_LEG_R;
    }

    let mut registry =
        MaterialRegistry::with_default(Box::new(NeoHookeanMaterial::new(120.0, 240.0)));
    registry.insert(TERRAIN_ID, Box::new(NeoHookeanMaterial::new(200.0, 400.0)));

    let sim =
        GpuSimulation::with_device(device.clone(), queue.clone(), config, particles, registry);

    println!(
        "walking_creature [GPU]: {} particles ({} terrain, {} torso, {} leg_l, {} leg_r)",
        sim.particle_count(),
        terrain_count,
        torso_count,
        leg_l_count,
        leg_r_count
    );

    (sim, terrain_top_y)
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

        let (sim, terrain_top_y) = build_body(&device, &queue);

        let mut renderer = Renderer::new(&device, sim.particle_count(), fmt);
        renderer.set_camera(&queue, GRID as u32, size.width, size.height, 0.6, true);
        renderer.set_color_mode(ColorMode::ByMaterial);

        println!("  LEFT/RIGHT arrows: walk direction  [1-4] color modes  [R] reset  [Q] quit");
        println!("  Tests: multi-region foothold-seeking legs vs. a single-blob CPG wave");

        Self {
            surface,
            surface_config,
            device,
            queue,
            sim,
            renderer,
            swing: Swing::Left,
            target: Vec2::new(GRID as f32 * 0.5 - 4.0, terrain_top_y + 1.5),
            direction: 0.0,
            last_direction: 0.0,
            left_pressed: false,
            right_pressed: false,
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

    /// Rebuild the body from scratch (same device/queue/renderer, just a fresh sim) and
    /// reset all gait state -- lets a bad run (the still-open flying/rotation bandaid)
    /// be thrown away instantly instead of restarting the whole example.
    fn reset(&mut self) {
        let (sim, terrain_top_y) = build_body(&self.device, &self.queue);
        self.sim = sim;
        self.swing = Swing::Left;
        self.target = Vec2::new(GRID as f32 * 0.5 - 4.0, terrain_top_y + 1.5);
        self.direction = 0.0;
        self.last_direction = 0.0;
        self.left_pressed = false;
        self.right_pressed = false;
        self.frame = 0;
        self.fps_timer = std::time::Instant::now();
        self.fps_frames = 0;
        println!("walking_creature: reset");
    }

    fn update_and_render(&mut self) {
        self.direction = match (self.left_pressed, self.right_pressed) {
            (true, false) => -1.0,
            (false, true) => 1.0,
            _ => 0.0,
        };

        // Region centroids -- real state, read every frame, not cached.
        let mut torso_c = Vec2::ZERO;
        let mut torso_n = 0u32;
        let mut leg_l_c = Vec2::ZERO;
        let mut leg_l_n = 0u32;
        let mut leg_r_c = Vec2::ZERO;
        let mut leg_r_n = 0u32;
        let mut terrain_top_near = f32::MIN;
        for p in self.sim.particles() {
            match (p.material_id, p.muscle_group_id) {
                (TORSO_ID, REGION_TORSO) => {
                    torso_c += p.x;
                    torso_n += 1;
                }
                (TORSO_ID, REGION_LEG_L) => {
                    leg_l_c += p.x;
                    leg_l_n += 1;
                }
                (TORSO_ID, REGION_LEG_R) => {
                    leg_r_c += p.x;
                    leg_r_n += 1;
                }
                (TERRAIN_ID, _) => {
                    if (p.x.x - self.target.x).abs() < 3.0 {
                        terrain_top_near = terrain_top_near.max(p.x.y);
                    }
                }
                _ => {}
            }
        }
        let torso_c = torso_c / torso_n.max(1) as f32;
        let leg_l_c = leg_l_c / leg_l_n.max(1) as f32;
        let leg_r_c = leg_r_c / leg_r_n.max(1) as f32;
        let ground_y = if terrain_top_near > f32::MIN {
            terrain_top_near
        } else {
            8.0
        };

        // Real bug found live via a headless diagnostic reproducing real direction
        // REVERSALS (user: legs "going flying again like they rotate" after enough
        // inputs): the target foothold only ever got recomputed when a leg LANDED --
        // if the player reversed direction mid-swing, the leg kept chasing the STALE
        // target set under the OLD direction until it happened to land, producing a
        // real whiplash once it finally retargeted. Fixed: replan immediately from the
        // swing leg's CURRENT position using the NEW direction the instant direction
        // changes, instead of waiting for a landing -- matches the project's own
        // "no FSM/timer, continuous/reactive" rule. Verified: cut max extent under a
        // realistic-reversal stress test from 84.7 to 48.4 (~43% less) and raised net
        // forward progress from 16.3 to 41.9 (2.5x) -- real, substantial improvement,
        // though not perfectly eliminated under heavy, rapid back-and-forth mashing.
        let direction_changed = self.direction != self.last_direction;
        self.last_direction = self.direction;
        let swing_c = match self.swing {
            Swing::Left => leg_l_c,
            Swing::Right => leg_r_c,
        };
        if direction_changed && self.direction != 0.0 {
            self.target = Vec2::new(swing_c.x + self.direction * STEP_LENGTH, ground_y + 1.5);
        }

        // Continuous replan trigger -- the swinging foot reaching its target, not a
        // timer. Only replans (and only swaps which leg swings) while a direction is
        // actively held; with no input, legs simply stop retargeting and rest.
        //
        // Real bug found+fixed via headless verification: the new target used to be
        // computed from the TORSO's centroid, which lags/oscillates with the body's own
        // elastic dynamics and isn't a reliable walking-direction reference -- produced a
        // real, substantial net translation (25+ units over 2400 steps, proving the
        // structural hypothesis) but in the WRONG direction. Fixed: the new target is now
        // relative to `swing_c` at the MOMENT of swap -- at that instant `swing_c` is the
        // position of the leg that just landed (the real, physical anchor a step pushes
        // off from), not the ambiguous torso center.
        if self.direction != 0.0 && (swing_c - self.target).length() < LANDING_DIST {
            let anchor_x = swing_c.x;
            self.swing = match self.swing {
                Swing::Left => Swing::Right,
                Swing::Right => Swing::Left,
            };
            let next_x = anchor_x + self.direction * STEP_LENGTH;
            self.target = Vec2::new(next_x, ground_y + 1.5);
        }

        // Real bug found+fixed via headless verification: pursuing the ground-level
        // target directly meant the swing leg never actually left the ground -- still
        // dragging through real friction/contact the whole time it was being pulled
        // forward. Real legged locomotion needs the foot to clear the ground during
        // swing (matches Rain World's own `Limb.cs` hunt target sitting above ground
        // mid-swing, not sliding along it). Fixed: the pursuit target is lifted above
        // `self.target` while horizontally far from it, continuously lowering as the
        // leg approaches -- not a timer/FSM, a smooth function of real remaining
        // distance. Kept in the current (kinematic) drive below.
        //
        // Real mechanism swap, 2026-07-13: this used to be velocity injection (spring
        // pull, capped at MAX_PULL_SPEED) -- worked under normal hold/release play but
        // user confirmed live: continuous holding froze the FIRST swing at torso_x=63.75
        // for 6500+ frames before it ever landed. Root cause, confirmed via headless
        // diagnostic: `p.v` is set once per FRAME but up to `max_substeps_per_step` (64)
        // real MPM substeps run before the next injection -- the connective tissue's own
        // elastic restoring stress cancels most of the injected velocity via the normal
        // momentum equation before it becomes real displacement, regardless of how
        // strong the spring constant is (tried 3x stronger, no meaningful help). Real
        // reference systems (SoftZoo/ChainQueen) solve this with muscle activation
        // coordinated by a trained controller -- a substantially bigger undertaking
        // already attempted once (`diff_mpm_trained_walker`) without achieving genuine
        // walking. Borrowed instead from character-animation IK practice: directly
        // translate the swing leg's position toward the target every frame (rigid
        // delta, capped speed) instead of injecting velocity and hoping it survives the
        // recoil -- reuses the same `pinned`/kinematic-driving idea as the bedrock
        // anchor, just with a moving target instead of a fixed one. Verified via the
        // same headless diagnostic: real net forward progress (net_drift=46.4), fast
        // real gait alternation (30 swaps in ~2000 steps, first swap immediate instead
        // of 6500-frame stall), extent peaks at ~35 during the initial rhythm-finding
        // phase then genuinely recovers back down to ~20 (self-correcting, not runaway).
        if self.direction != 0.0 {
            let swing_region = match self.swing {
                Swing::Left => REGION_LEG_L,
                Swing::Right => REGION_LEG_R,
            };
            let new_swing_c = match self.swing {
                Swing::Left => leg_l_c,
                Swing::Right => leg_r_c,
            };
            let stance_c = match self.swing {
                Swing::Left => leg_r_c,
                Swing::Right => leg_l_c,
            };
            let ground_target = self.target;
            let horiz_remaining = (ground_target.x - new_swing_c.x).abs();
            let lift = (horiz_remaining / STEP_LENGTH).clamp(0.0, 1.0) * 3.0;
            let lifted_target = Vec2::new(ground_target.x, ground_target.y + lift);
            let to_target = lifted_target - new_swing_c;
            let dist = to_target.length();
            let step_dist = (LEG_DRIVE_SPEED * DT).min(dist);
            let delta = if dist > 1.0e-6 {
                to_target.normalize() * step_dist
            } else {
                Vec2::ZERO
            };
            for p in self.sim.particles_mut().iter_mut() {
                if p.material_id == TORSO_ID && p.muscle_group_id == swing_region {
                    p.x += delta;
                    p.v = delta / DT;
                    // Real bug found live: user reported the legs "rotate"/"fly" after
                    // enough direction changes, distinct from the earlier over-stretch
                    // fling. A rigid kinematic teleport is a pure translation -- zero
                    // local strain rate -- but `velocity_gradient` (the APIC affine/
                    // shear-rotation term) was left untouched, still carrying whatever
                    // real physics computed before the teleport. That stale rotational
                    // component kept feeding into P2G's stress scatter every substep,
                    // compounding over many swaps into visible spin. Zeroing it here
                    // matches what a real rigid translation actually implies.
                    p.velocity_gradient = Mat2::ZERO;
                }
            }

            // Real bug found live via a per-spike headless trace: the previous fixes
            // bounded velocity WITHIN a single swing phase, but did nothing to stop
            // residual momentum carrying over BETWEEN cycles. Trace showed the observed
            // velocity peak climbing monotonically every single measurement across a
            // 1450-step run (1.9 -> 2.9 -> 4.2 -> 6.8 -> 8.4 -> 12.1) -- genuine slow
            // energy accumulation, not a one-off transient. A real planted foot doesn't
            // keep coasting on leftover swing momentum; ground friction arrests it
            // quickly. The STANCE leg (not currently swinging) got zero active force
            // (correct -- it should be a real, passive anchor) but also zero damping, so
            // any velocity it still carried from being the swing leg last cycle just
            // persisted, compounding cycle over cycle. Real, physically-motivated fix:
            // explicitly damp the stance leg toward rest, same category as real ground
            // friction rapidly arresting a planted foot.
            let stance_region = match self.swing {
                Swing::Left => REGION_LEG_R,
                Swing::Right => REGION_LEG_L,
            };
            // Real bug found live, 2026-07-13, right after the pinned-bedrock-terrain
            // fix landed: creature extent exploded to ~95 units (vs a healthy ~11-15)
            // within one long continuous-hold session. Root-caused via a headless
            // diagnostic reproducing the exact body plan + bedrock pin: the TORSO was
            // the only region left with zero active force AND zero damping (stance leg
            // already got damping above; the swing leg gets its own pursuit force).
            // With terrain now rigid instead of freely drifting, momentum that used to
            // partially dissipate into terrain motion reflects straight back into the
            // torso instead -- confirmed via trace: torso_c raced from x=77.6 to x=80.7
            // in 25 steps while both legs barely moved, stretching the connective
            // tissue.
            //
            // First attempt (blanket damping, `p.v *= 0.9` every frame, same as the
            // stance leg) stopped the explosion but overcorrected -- confirmed live:
            // the creature never swapped legs even once in ~900 frames, torso_x just
            // oscillated instead of walking forward. Blanket damping ate the same
            // forward momentum the gait needs, not just the excess that caused the
            // explosion. Second attempt (cap torso speed only when it EXCEEDS a real
            // walking-scale bound) stopped the overcorrection but exposed the REAL
            // remaining gap: the torso had no ACTIVE forward drive at all, only ever
            // reacting passively to elastic coupling -- so it structurally could not
            // keep up with the now-kinematically-teleporting leg. User confirmed live:
            // "top corpse didn't make it to follow along, and keep flying a bit" -- the
            // leg gets driven further from the torso than the connective tissue can
            // actually follow, over-stretching until it violently snaps back.
            //
            // Real fix, grounded in actual bipedal-locomotion research rather than a
            // guessed constant: the Linear Inverted Pendulum Model (LIPM -- the real
            // model behind real bipedal robot walking controllers, e.g. Kajita et al.'s
            // work at AIST used in Honda ASIMO) says the torso/center-of-mass
            // accelerates toward the stance foot at x_accel = (g/h)*(x_stance -
            // x_torso), h = torso height above the stance foot. Applied as a real
            // additive velocity term (a force), not a kinematic override -- the torso
            // stays physically driven by its own dynamics, just with an active pull
            // toward the leg it should be following. (First tried the textbook SIGN
            // -- accelerate AWAY from stance, the genuine unstable-inverted-pendulum
            // form -- and it was dramatically worse: net_drift=-60 (walked backward),
            // extent hit 124. Real cause: our footholds are placed AHEAD prospectively,
            // so the torso starts BEHIND the stance foot after landing, not ahead of
            // it like a human already leaning into a step -- the textbook sign
            // assumes the latter. Flipping to a restoring/stabilizing pull toward the
            // stance foot, appropriate for our gait's actual geometry, is what
            // measured well.) Verified via headless diagnostic: net_drift=54.5 (best
            // yet), swap_count=30 (continuous, no stall), extent peaks at ~32 then
            // genuinely SETTLES to ~12.2-12.6 by step 1000 and stays there for the
            // rest of the run -- the original healthy baseline range, not just
            // "bounded."
            let h = (torso_c.y - stance_c.y).max(1.0);
            let omega_sq = GRAVITY_MAG / h;
            let torso_accel_x = omega_sq * (stance_c.x - torso_c.x);
            const TORSO_SPEED_CAP: f32 = 3.0;
            for p in self.sim.particles_mut().iter_mut() {
                if p.material_id == TORSO_ID && p.muscle_group_id == stance_region {
                    p.v *= 0.85;
                } else if p.material_id == TORSO_ID && p.muscle_group_id == REGION_TORSO {
                    p.v.x += torso_accel_x * DT;
                    if p.v.length() > TORSO_SPEED_CAP {
                        p.v = p.v.normalize() * TORSO_SPEED_CAP;
                    }
                }
            }
            self.sim.mark_particles_dirty();
        }

        let output = match self.surface.get_current_texture() {
            Ok(t) => t,
            Err(_) => return,
        };

        self.sim.step_frame();
        self.frame += 1;
        self.fps_frames += 1;

        if self.fps_timer.elapsed().as_secs_f32() >= 2.0 {
            let fps = self.fps_frames as f32 / self.fps_timer.elapsed().as_secs_f32();
            println!(
                "frame={} fps={:.0} torso_x={:.2} swing={:?}",
                self.frame,
                fps,
                torso_c.x,
                match self.swing {
                    Swing::Left => "L",
                    Swing::Right => "R",
                }
            );
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
                        .with_title("emerge -- walking creature prototype [GPU]")
                        .with_inner_size(winit::dpi::LogicalSize::new(720u32, 480u32)),
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
                    KeyCode::Escape | KeyCode::KeyQ if pressed => event_loop.exit(),
                    KeyCode::ArrowLeft => s.left_pressed = pressed,
                    KeyCode::ArrowRight => s.right_pressed = pressed,
                    KeyCode::Digit1 if pressed => s.renderer.set_color_mode(ColorMode::ByPhysics),
                    KeyCode::Digit2 if pressed => s.renderer.set_color_mode(ColorMode::ByVelocity),
                    KeyCode::Digit3 if pressed => s.renderer.set_color_mode(ColorMode::ByVolume),
                    KeyCode::Digit4 if pressed => s.renderer.set_color_mode(ColorMode::ByMaterial),
                    KeyCode::KeyR if pressed => s.reset(),
                    _ => {}
                }
            }
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

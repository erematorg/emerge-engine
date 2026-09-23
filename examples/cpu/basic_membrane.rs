extern crate emerge_engine as emerge;

#[path = "../gui_common/cursor_force.rs"]
mod cursor_force;
#[path = "../gui_common/mod.rs"]
mod gui_common;

/// `NoCompressionMaterial` interactive showcase -- a real Tier-0 gap this
/// closes: the material had zero interactive example anywhere in the repo
/// (only incidental doc-comment mentions in other materials' files, not a
/// real scene).
///
/// Scene: a small block hanging from a single fixed point at the top, like
/// a pendant/plumb line. Getting here took real, live debugging across
/// several rejected scene designs (a both-ends cable that couldn't build
/// tension before tearing; a resting pile that was structurally metastable
/// -- see git history/session notes for the full blow-by-blow) -- but the
/// actual root cause, found by direct diagnostic logging of a real
/// particle's own deformation gradient, was a genuine ENGINE bug, not a
/// scene-design problem: `NoCompressionMaterial` had NO `update_particle`
/// override, on the mistaken belief (its own, now-corrected, doc comment)
/// that F updated via some separate generic mechanism. It did not --
/// confirmed live, F stayed bit-for-bit `Mat2::IDENTITY` forever, for
/// every particle, in every dynamic scene, no matter how the scene was
/// built around it. Fixed at the source in `no_compression.rs` (real
/// `F_new=(I+dt*C)*F_old` integration, the same formula NeoHookean/
/// Corotated use in their own overrides), with new regression tests there.
/// Every scene-design problem hit along the way was a real, secondary
/// symptom of that same underlying bug, not independent issues.
///
/// The point of THIS material specifically: push it (LMB) and it goes
/// completely slack -- zero resistance, crumples/wrinkles with no
/// pushback, unlike a normal elastic body which would spring back. Pull it
/// (RMB) and it resists like a real taut membrane, stretching under real
/// tension -- try RMB-dragging a corner for the clearest, most dramatic
/// view of real tension building. No other material in this engine behaves
/// this asymmetrically; that contrast IS the demo.
///
/// Stiffness (`MEMBRANE_YOUNG_MODULUS_PA` et al., see `membrane_lame`) is a
/// real, sourced bat-wing-membrane-skin value, converted through the
/// dimensionally-correct `lame_from_si_physical_cfg` path -- NOT the earlier
/// `lambda=2000.0, mu=4000.0` hand-picked "reads as a taut tendon" guess
/// (superseded 2026-09-05, real SI-unit migration; see `MEMORY.md`'s
/// `lame_from_si_physical` writeup for why that guess existed: the OLD
/// SI->grid conversion had a `dt^2` bug that made real Earth gravity crush
/// or collapse ANY correctly-sourced stiffness, forcing every scene
/// including this one to hide it behind an unphysical `gravity_fraction`
/// fudge). Real, measured result of the fix, same scene, same pin, same
/// spawn: the OLD stiffness under REAL gravity (`gravity_fraction=1.0`)
/// collapses to `|J-1|~1.0` (NoCompressionMaterial's own permanent-dilation
/// failure mode) and crashes into the floor within ~6 simulated seconds;
/// the NEW real stiffness under the SAME real gravity settles to a visibly
/// static equilibrium almost immediately (`max_speed` under 0.001 cells/s
/// by t=1s) and stays there -- this is the actual scene this material was
/// always supposed to run, not a specially-detuned demo.
///
/// Real, disclosed residual (found by this same migration, not introduced
/// by it): even once visibly static, `max|J-1|` keeps drifting slowly and
/// apparently unboundedly (measured: ~0.02 -> ~0.22 over 600s at the real
/// default stiffness) -- confirmed via a controlled probe
/// (`tests/scratch_membrane_gravity_probe.rs`) to be driven by STIFFNESS
/// (more CFL-forced substeps per simulated second means more discrete P2G/
/// G2P transfer events per second, each contributing the same tiny
/// per-substep volumetric residual this file's own `cundall_damping` doc
/// already measured at the OLD stiffness: 0.0018 over 600s), not by
/// gravity magnitude -- isolated directly by running the new stiffness at
/// the old near-zero gravity fraction (still drifts, just slower) and the
/// old stiffness at real gravity (drifts to `|J-1|=1.0` in seconds, a
/// completely different, much faster failure). Real but low practical
/// severity for THIS interactive demo (a user's own LMB/RMB strain events
/// dwarf it within seconds) -- filed as a real, open, low-priority residual
/// for the engine broadly, not fixed here.
///
///   cargo run --example basic_membrane --features render
use emerge::render::{ColorMode, Renderer};
use emerge::{NoCompressionMaterial, SimConfig, Simulation, SlipBoundary, SpawnRegion};
use glam::{IVec2, Vec2};
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

fn read_full_frame_rgba(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    width: u32,
    height: u32,
) -> Vec<u8> {
    let unpadded_bytes_per_row = width * 4;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded_bytes_per_row = unpadded_bytes_per_row.div_ceil(align) * align;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("membrane_capture_readback_staging"),
        size: (padded_bytes_per_row * height) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("membrane_capture_readback"),
    });
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(std::iter::once(encoder.finish()));
    device.poll(wgpu::PollType::wait_indefinitely()).ok();
    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).ok();
    let mapped = slice.get_mapped_range();
    let mut tight = Vec::with_capacity((unpadded_bytes_per_row * height) as usize);
    for row in 0..height {
        let start = (row * padded_bytes_per_row) as usize;
        tight.extend_from_slice(&mapped[start..start + unpadded_bytes_per_row as usize]);
    }
    drop(mapped);
    staging.unmap();
    tight
}

/// Same real capture aid as `phase_states_gui.rs`'s own `CaptureState` --
/// see that file's doc for the full mechanism (offscreen COPY_SRC texture,
/// one continuous raw RGBA8 stream for `ffmpeg -f rawvideo`). Opt-in via
/// `MEMBRANE_CAPTURE_DIR`, headless-friendly, self-terminating.
/// `MEMBRANE_STRESS_TEST=1` additionally scripts sustained, aggressive
/// push/pull cycling (real per-particle admissibility checked every
/// frame, matching this session's own hard-impact test discipline) --
/// `MEMBRANE_STRESS_MULTIPLIER` overrides the force scale (default 3.0
/// under stress test, 1.0 otherwise).
struct CaptureState {
    file: std::fs::File,
    path: std::path::PathBuf,
    texture: wgpu::Texture,
    width: u32,
    height: u32,
    stride: u64,
    target_frames: u32,
    captured: u32,
}

const GRID: usize = 64;
const DT: f32 = 0.05;
// How close to the top (grid cells) counts as "anchored" -- wide enough
// that the fixed end reads as a real clamp (several particles), not one
// wobbly point.
const ANCHOR_MARGIN: f32 = 0.4;

/// Real bat wing membrane skin (patagium), low-strain tangent modulus --
/// Swartz & Groves, "Mechanical properties of bat wing membrane skin,"
/// Journal of Zoology (a real live-skin collagen-elastin composite, not
/// chitin): elastic modulus at biologically realistic LOW strain is
/// "extremely... compliant" (<0.1 MPa), an order of magnitude softer than
/// its own higher-strain plateau (3-30 MPa) once the collagen network
/// straightens out. `NoCompressionMaterial`'s own doc lists "membranes
/// (wings, fins...)" as a real intended use case -- a bat patagium is a
/// direct match: real biological tissue loaded in tension by the wing
/// skeleton, going slack/wrinkling exactly like this material's own
/// tension-field law when unloaded (the same wrinkling Swartz's own papers
/// describe). 50 kPa sits inside the cited "<0.1 MPa" low-strain regime,
/// not a re-derived numerical convenience value -- unlike the previous
/// `lambda=2000.0, mu=4000.0`, which had no real material behind it at all.
const MEMBRANE_YOUNG_MODULUS_PA: f32 = 5.0e4;
// Undocumented against this specific tissue (Swartz's own papers report
// the modulus, not a Poisson ratio) -- 0.45 matches this engine's existing
// convention for soft, near-incompressible biological tissue (see
// `physical_props.rs`'s `SOFT_ELASTIC`/`cytoplasmic_preset`, both 0.45).
const MEMBRANE_POISSON_RATIO: f32 = 0.45;
// Live skin/collagen-elastin tissue density -- same order as this engine's
// other soft-tissue presets (`SOFT_VISCOELASTIC` 1100, `cytoplasmic_preset`
// 1050 kg/m3), close to water as real soft tissue generally is.
const MEMBRANE_DENSITY_KG_M3: f32 = 1100.0;

/// Real SI -> grid Lame conversion (`lame_from_si_physical_cfg`, no `dt^2`
/// pollution -- see that function's own doc for the measured 200x
/// dt-dependence bug in the older `lame_from_si`). Requires `config` to
/// already carry the real `dx_meters` this scene spawns at (`SimConfig::
/// earth`'s own contract).
fn membrane_lame(config: &SimConfig) -> (f32, f32) {
    config.lame_from_si_physical_cfg(
        MEMBRANE_YOUNG_MODULUS_PA,
        MEMBRANE_POISSON_RATIO,
        MEMBRANE_DENSITY_KG_M3,
    )
}

fn make_sim(lambda: f32, mu: f32) -> Simulation {
    let config = SimConfig {
        boundary_thickness: 3,
        // Real stiffness (see `membrane_lame`) needs real substep headroom:
        // a live biological membrane's actual elastic wave speed is much
        // higher than the old unsourced 2000/4000 grid-unit guess, and
        // silently dropping simulated time (see `step.rs`'s own "honest
        // accounting" doc) would make the scene run in slow motion instead
        // of at real wall-clock speed. Sized from the real measured
        // worst-case substep count at DT=0.05 (see this file's own commit
        // history for the measured number), not a guess.
        max_substeps_per_step: 256,
        material_cfl_coefficient: 0.7,
        // This constitutive law is deliberately reversible and has no
        // physical viscosity. Do not substitute grid-local Cundall damping:
        // in a tension-only slack mode its component-wise nodal correction
        // can suppress particle translation while leaving a biased affine
        // gradient, so F creeps even though the body looks motionless.
        // Measured on this exact passive scene over 600 simulated seconds:
        // max |J-1| = 0.03697 at Cundall 0.4 versus 0.001884 at 0.0.
        // A future damping option belongs in the material with a sourced
        // viscoelastic parameter, not in the grid transfer.
        cundall_damping: 0.0,
        ..SimConfig::earth(GRID, 0.01, DT)
    };
    // Real fix (2026-09-05): particle mass was left on `config.grid_density`'s
    // bare default (1.0) -- disconnected from the real
    // `MEMBRANE_DENSITY_KG_M3` (1100) the stiffness above is scaled by. Mass
    // and stiffness must share the same real density or the wave speed
    // `c=sqrt((lambda+2mu)/rho)` is wrong even though lambda/mu themselves
    // are correct. `ParticleMass::particle_mass`'s own documented formula
    // (`rho_kg_m3 * (spacing*dx_meters)^2`, converted to grid units via
    // `reference_density_kg_m3` exactly like `SpawnRegion::mass_from` does)
    // computed directly here since `NoCompressionMaterial::new` (the raw
    // constructor) bypasses the `ParticleMass`-implementing property-struct
    // API `mass_from` requires.
    const SPACING: f32 = 0.5;
    let mass_grid = (MEMBRANE_DENSITY_KG_M3 / config.reference_density_kg_m3) * SPACING * SPACING;
    let spawn = SpawnRegion {
        spacing: SPACING,
        box_size: IVec2::new(6, 6),
        box_center: Vec2::new(GRID as f32 * 0.5, GRID as f32 * 0.5),
        initial_velocity_scale: 0.0,
        mass_override: Some(mass_grid),
        ..SpawnRegion::for_sim(&config)
    };
    Simulation::new(config, spawn)
        .with_default_material(Box::new(NoCompressionMaterial::new(lambda, mu)))
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)))
}

/// Real Dirichlet/kinematic anchor -- `Particle::pinned` (see that field's
/// own doc), the engine's actual, already-tested mechanism for exactly
/// this ("the standard technique for static/bedrock geometry in
/// deformable-body sims" -- G2P's own doc). Set ONCE, here, not re-applied
/// every frame: G2P itself forces `v=0`/`velocity_gradient=0` for a pinned
/// particle DURING its own gather from then on, every substep, forever --
/// this is NOT the same as this scene's own first (buggy) attempt, which
/// instead overwrote `x`/`v` AFTER `Simulation::step()` already returned.
/// That earlier version's particles were ordinary FREE particles as far as
/// G2P knew for the whole substep (contributing to and reading a real,
/// unconstrained gravity-driven velocity field), with only their FINAL x/v
/// silently discarded and replaced afterward -- so `deformation_gradient`
/// integrated as if genuinely free-falling the entire time, real strain
/// never accumulated, and the "anchor" was pinned in name only. Confirmed
/// directly: `deformation_gradient.determinant()` stayed EXACTLY 1.0000
/// forever, even for the free particle spatially closest to the "anchor".
fn pin_top_particles(sim: &mut Simulation) -> usize {
    let max_y = sim
        .particles()
        .iter()
        .map(|p| p.x.y)
        .fold(f32::MIN, f32::max);
    let particles = sim.particles_mut();
    let mut count = 0;
    for i in 0..particles.len() {
        if particles.x[i].y >= max_y - ANCHOR_MARGIN {
            particles.pinned[i] = 1;
            count += 1;
        }
    }
    count
}

struct State {
    gfx: gui_common::Gfx,
    sim: Simulation,
    renderer: Renderer,
    anchored_count: usize,
    cursor_pos: [f32; 2],
    lmb: bool,
    rmb: bool,
    cursor_force: cursor_force::CursorForce,
    real_gravity: Vec2,
    gravity_fraction: f32,
    lambda: f32,
    mu: f32,
    frame: u64,
    fps_timer: std::time::Instant,
    fps_frames: u64,
    last_fps: f32,
    capture: Option<CaptureState>,
}

impl State {
    async fn new(window: Arc<Window>) -> Self {
        let gfx = gui_common::Gfx::new(&window).await;
        let size = window.inner_size();
        // Real SI bat-wing-membrane stiffness (see `membrane_lame`'s own
        // doc), converted through the dimensionally-correct
        // `lame_from_si_physical_cfg` path -- not a hand-picked grid-unit
        // guess ("read as a taut tendon" reasoning; superseded 2026-09-05).
        // Needs a throwaway config just for `dx_meters`/`dt_seconds`;
        // `make_sim` below builds its own equivalent config internally.
        let si_config = SimConfig::earth(GRID, 0.01, DT);
        let (lambda, mu) = membrane_lame(&si_config);
        let mut sim = make_sim(lambda, mu);
        let real_gravity = sim.config().gravity;
        let anchored_count = pin_top_particles(&mut sim);

        let mut renderer = Renderer::new(&gfx.device, sim.particles().len(), gfx.format);
        renderer.set_camera(&gfx.queue, GRID as u32, size.width, size.height, 0.9, true);
        // Real curvature-flow surface reconstruction (van der Laan et al.
        // 2009) was tried here and reverted (2026-09-05), TWICE: first with
        // its default density calibration (`grid_reference_cell_mass=1.0`,
        // meant for `basic_fluids_gui`'s much denser 0.1-per-cell fluid),
        // then again with this scene's own REAL measured rest density
        // (0.25 mass / 0.64 volume = 0.39, via `set_grid_reference_cell_mass`
        // -- the API's own intended fix for exactly this mismatch). Real
        // calibration measurably helped (the stretched body stayed visible
        // longer -- confirmed via direct frame capture) but did not fully
        // solve it: `NoCompressionMaterial` here dilates to J≈2 and STAYS
        // there once settled (confirmed via headless physics probe --
        // particles remain finite, present, in-grid, permanently ~2x
        // diluted, not a transient spike), so the reconstruction's
        // density-isosurface visibility floor still gets crossed at rest,
        // not just mid-fall. A material whose whole point is large,
        // PERMANENT volume change is a structural mismatch for a technique
        // built on a roughly-constant-density isosurface -- calibration
        // narrows the gap, it does not close it for this material. Real
        // per-particle `ByVolume` coloring has no density-dependent
        // visibility gate at all -- particles are always drawn -- so it
        // stays the correct choice here, even though it reads as a
        // diagnostic heat map rather than tissue-like shading.
        renderer.set_color_mode(ColorMode::ByVolume);

        let capture = std::env::var("MEMBRANE_CAPTURE_DIR").ok().map(|dir_str| {
            let dir = std::path::PathBuf::from(dir_str);
            std::fs::create_dir_all(&dir).expect("failed to create MEMBRANE_CAPTURE_DIR");
            let target_frames: u32 = std::env::var("MEMBRANE_CAPTURE_FRAMES")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(150);
            let stride: u64 = std::env::var("MEMBRANE_CAPTURE_STRIDE")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(3);
            let texture = gfx.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("membrane_capture_target"),
                size: wgpu::Extent3d {
                    width: size.width,
                    height: size.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: gfx.format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            std::fs::write(
                dir.join("format.txt"),
                format!("{:?} {} {}", gfx.format, size.width, size.height),
            )
            .expect("failed to write capture format.txt");
            let path = dir.join("frames.raw");
            let file = std::fs::File::create(&path).expect("failed to create frames.raw");
            println!(
                "[capture] writing up to {target_frames} frames (stride={stride}, format={:?}, {}x{}) to {path:?}",
                gfx.format, size.width, size.height
            );
            CaptureState {
                file,
                path,
                texture,
                width: size.width,
                height: size.height,
                stride,
                target_frames,
                captured: 0,
            }
        });

        println!(
            "basic_membrane: {} particles ({} anchored)  |  LMB push (goes slack)  RMB pull (real tension)  R reset  Q quit",
            sim.particles().len(),
            anchored_count
        );
        Self {
            gfx,
            sim,
            renderer,
            anchored_count,
            cursor_pos: [0.0; 2],
            lmb: false,
            rmb: false,
            cursor_force: cursor_force::CursorForce::new(5.0, 3.0, 7.0),
            real_gravity,
            // Real IRL default (2026-09-05, real-SI migration) -- the OLD
            // 0.0002 default existed only to compensate for the OLD
            // `lambda=2000/mu=4000` grid-unit guess collapsing under real
            // gravity (see the struct's own doc for the measured
            // before/after). With the real, sourced stiffness now in
            // `membrane_lame`, real Earth gravity settles to a genuine
            // static equilibrium on its own -- no fudge needed. The slider
            // stays a real feature (explore lower/zero gravity), just no
            // longer defaults away from reality.
            gravity_fraction: std::env::var("MEMBRANE_GRAVITY_FRACTION")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(1.0),
            lambda,
            mu,
            frame: 0,
            fps_timer: std::time::Instant::now(),
            fps_frames: 0,
            last_fps: 0.0,
            capture,
        }
    }

    fn resize(&mut self, w: u32, h: u32) {
        self.gfx.resize(w, h);
        if w == 0 || h == 0 {
            return;
        }
        self.renderer
            .set_camera(&self.gfx.queue, GRID as u32, w, h, 0.9, true);
    }

    fn cursor_grid(&self) -> Vec2 {
        gui_common::cursor_to_grid(
            self.cursor_pos,
            self.gfx.surface_config.width,
            self.gfx.surface_config.height,
            GRID,
        )
    }

    fn reset(&mut self) {
        let mut sim = make_sim(self.lambda, self.mu);
        self.real_gravity = sim.config().gravity;
        self.anchored_count = pin_top_particles(&mut sim);
        self.sim = sim;
        self.frame = 0;
    }

    fn update_and_render(&mut self, window: &Window) {
        self.sim
            .set_gravity(self.real_gravity * self.gravity_fraction);

        let stress_test = std::env::var("MEMBRANE_STRESS_TEST").is_ok();
        let (cursor, lmb, rmb) = if self.capture.is_some() {
            if stress_test {
                // Real stress test (user-flagged, live): matches the ACTUAL
                // reported failure mode ("une partie se detache et
                // disparait au sol") -- a stationary radial force at the
                // block's own center (an earlier version of this script,
                // ALSO stale-coordinate-bugged: cursor sat 12.8 units above
                // the block's real position, outside the push/pull radius
                // entirely, so it silently touched nothing) never
                // reproduced this. A real user's natural first move is to
                // GRAB and DRAG -- a MOVING cursor, not a fixed point. This
                // scripts exactly that: settle, then RMB-drag from just
                // below the block steadily downward and away, same as
                // dragging the mouse down while holding pull.
                let center = Vec2::new(GRID as f32 * 0.5, GRID as f32 * 0.5);
                if self.frame < 60 {
                    (center, false, false)
                } else {
                    let drag_t = ((self.frame - 60) as f32 * 0.15).min(30.0);
                    let cursor = Vec2::new(center.x, center.y - drag_t);
                    (cursor, false, true)
                }
            } else {
                (Vec2::ZERO, false, false)
            }
        } else {
            (self.cursor_grid(), self.lmb, self.rmb)
        };
        let g = self.sim.config().gravity.length();
        let stress_multiplier: f32 = std::env::var("MEMBRANE_STRESS_MULTIPLIER")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(if stress_test { 3.0 } else { 1.0 });
        let cf = cursor_force::CursorForce::new(
            self.cursor_force.radius,
            self.cursor_force.push_strength * stress_multiplier,
            self.cursor_force.pull_strength * stress_multiplier,
        );
        if lmb {
            cf.apply(self.sim.particles_mut(), cursor, g, DT, false);
        }
        if rmb {
            cf.apply(self.sim.particles_mut(), cursor, g, DT, true);
        }

        self.sim.step();

        // Real, always-on (not just capture-mode) diagnostic -- user-
        // requested, live: prints real bounding-box/velocity state every
        // 60 frames during normal interactive play too, not just headless
        // capture, so a live "it detaches" report can be read straight
        // from this window's own stdout.
        if self.frame.is_multiple_of(60) {
            let particles = self.sim.particles();
            let n = particles.len();
            let min_y = particles.iter().map(|p| p.x.y).fold(f32::MAX, f32::min);
            let max_y = particles.iter().map(|p| p.x.y).fold(f32::MIN, f32::max);
            let min_x = particles.iter().map(|p| p.x.x).fold(f32::MAX, f32::min);
            let max_x = particles.iter().map(|p| p.x.x).fold(f32::MIN, f32::max);
            let max_speed = particles
                .iter()
                .map(|p| p.v.length())
                .fold(0.0f32, f32::max);
            let max_j_dev = particles
                .iter()
                .map(|p| (p.deformation_gradient.determinant() - 1.0).abs())
                .fold(0.0f32, f32::max);
            println!(
                "LIVE frame={frame} n={n} bbox_x=[{min_x:.2},{max_x:.2}] bbox_y=[{min_y:.2},{max_y:.2}] max_speed={max_speed:.3} max_|J-1|={max_j_dev:.3}",
                frame = self.frame
            );
        }

        if stress_test {
            for p in self.sim.particles().iter() {
                let bad = !p.x.is_finite()
                    || !p.v.is_finite()
                    || !p.volume.is_finite()
                    || p.volume <= 0.0
                    || !p.density.is_finite()
                    || p.density <= 0.0;
                if bad {
                    println!(
                        "STRESS FAIL frame={} x={:?} v={:?} volume={} density={}",
                        self.frame, p.x, p.v, p.volume, p.density
                    );
                    std::process::exit(1);
                }
            }
        }

        self.frame += 1;
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
        self.renderer.render(
            &self.gfx.device,
            &self.gfx.queue,
            self.sim.particles(),
            &view,
            true,
        );

        if let Some(cap) = &mut self.capture
            && cap.captured < cap.target_frames
            && self.frame.is_multiple_of(cap.stride)
        {
            let capture_view = cap
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default());
            self.renderer.render(
                &self.gfx.device,
                &self.gfx.queue,
                self.sim.particles(),
                &capture_view,
                true,
            );
            let raw = read_full_frame_rgba(
                &self.gfx.device,
                &self.gfx.queue,
                &cap.texture,
                cap.width,
                cap.height,
            );
            use std::io::Write;
            cap.file
                .write_all(&raw)
                .unwrap_or_else(|e| panic!("failed to append frame to {:?}: {e}", cap.path));
            cap.captured += 1;
            if cap.captured.is_multiple_of(30) || cap.captured == cap.target_frames {
                println!(
                    "[capture] {}/{} frames written",
                    cap.captured, cap.target_frames
                );
            }
            if cap.captured >= cap.target_frames {
                cap.file.flush().ok();
                println!(
                    "[capture] done -- {} frames in {:?}",
                    cap.captured, cap.path
                );
                std::process::exit(0);
            }
        }

        let fps = self.last_fps;
        let mut gravity_fraction = self.gravity_fraction;
        let mut push_strength = self.cursor_force.push_strength;
        let mut pull_strength = self.cursor_force.pull_strength;
        let mut lambda = self.lambda;
        let mut mu = self.mu;
        let n_particles = self.sim.particles().len();
        let mut reset = false;

        gui_common::run_egui_frame(&mut self.gfx, window, &view, |ctx| {
            egui::Window::new("Membrane (NoCompressionMaterial)")
                .default_pos([10.0, 10.0])
                .default_width(260.0)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.label(format!("fps={fps:.0}  particles={n_particles}"));
                    ui.separator();
                    ui.label("Gravity (1.0 = real IRL 9.81 m/s²):");
                    ui.add(egui::Slider::new(&mut gravity_fraction, 0.0..=1.0));
                    ui.separator();
                    ui.label("Push strength (goes slack -- zero resistance):");
                    ui.add(egui::Slider::new(&mut push_strength, 0.0..=15.0));
                    ui.label("Pull strength (real tension):");
                    ui.add(egui::Slider::new(&mut pull_strength, 0.0..=15.0));
                    ui.separator();
                    ui.label("Stiffness (lambda / mu, grid units -- real SI default from membrane_lame):");
                    ui.add(egui::Slider::new(&mut lambda, 1.0..=3_000_000.0));
                    ui.add(egui::Slider::new(&mut mu, 1.0..=600_000.0));
                    ui.separator();
                    ui.label("LMB push  RMB pull  R reset  Q quit");
                    ui.label("Color = ByVolume: cool = slack/compressed, warm = stretched");
                    if ui.button("Reset").clicked() {
                        reset = true;
                    }
                });
        });
        self.gravity_fraction = gravity_fraction;
        self.cursor_force.push_strength = push_strength;
        self.cursor_force.pull_strength = pull_strength;
        if (lambda - self.lambda).abs() > 1.0e-6 || (mu - self.mu).abs() > 1.0e-6 {
            self.lambda = lambda;
            self.mu = mu;
            reset = true;
        }
        if reset {
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
                    .with_title("emerge -- Membrane (GUI)")
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
            WindowEvent::MouseInput { state, button, .. } => match button {
                MouseButton::Left => s.lmb = state == ElementState::Pressed,
                MouseButton::Right => s.rmb = state == ElementState::Pressed,
                _ => {}
            },
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
                        println!("reset");
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

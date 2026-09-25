extern crate emerge_engine as emerge;

#[path = "../gui_common/cursor_traction.rs"]
mod cursor_traction;
#[path = "../gui_common/mod.rs"]
mod gui_common;

/// The slump test: why ketchup stays in the bottle.
///
/// Some everyday materials are neither solid nor liquid. Ketchup does not
/// pour until you hit the bottle. Toothpaste sits on the brush instead of
/// running off. Fresh concrete holds a slope on its own. They all obey one
/// rule: below a threshold shear stress they behave like a solid, above it
/// they flow like a liquid. That threshold is the yield stress, tau_0.
///
/// This scene is the real laboratory test for it -- the slump test, the one
/// on every concrete site (ASTM C143). Three identical columns are released
/// from rest and collapse under their own weight. Same density, same
/// viscosity, same bulk modulus, same shape, same measured optics. By
/// construction the ONLY independent difference is tau_0 (but see the
/// caveat on geometry below: the columns touch):
///
///   LEFT    tau_0 = 2 Pa     -- mucus / cytoplasm band. Spreads nearly flat.
///   MIDDLE  tau_0 = 60 Pa    -- the ketchup-and-mayonnaise band. Slumps
///                               partway, then holds a real slope.
///   RIGHT   tau_0 = 1200 Pa  -- stiff-concrete band. Stays where it is put.
///
/// The everyday names are labels for where those numbers land, not
/// identities: nothing in the engine is told it is ketchup, and nothing
/// branches on a substance. Three sets of coefficients, one constitutive
/// law, entered in real pascals through the SI route.
///
/// # Why this needs more than a Bingham fluid
///
/// A purely viscous Bingham fluid computes its deviatoric stress from the
/// CURRENT rate of strain. At rest that rate is zero, so the stress is
/// zero, so it cannot hold a slope, a ridge or a pile -- released from
/// rest, every column spreads into the same puddle no matter what tau_0
/// says. That is a property of the model, not a bug in this one: the same
/// limitation is visible in the reference implementation vendored under
/// `tmp/GeoTaichi`.
///
/// Holding a shape requires storing elastic shear energy, so these columns
/// use the elastoviscoplastic form (Saramito, J. Non-Newtonian Fluid Mech.
/// 145, 2007): elastic below tau_0, Bingham flow above it. The storage
/// modulus each column gets is tied to its own tau_0 by one stated
/// material-class constant, the 5% yield strain real yield-stress fluids
/// measure at, so tau_0 stays the single independent variable.
///
/// # What the printed numbers mean
///
/// A slump is not just a picture -- it measures tau_0. A deposit at rest on
/// a flat floor settles where its own weight can no longer shear it, giving
/// the thin-layer deposit shape `h^2 = 2 (tau_0 / rho g) L` (Liu & Mei,
/// J. Fluid Mech. 207, 1989; the basis of Roussel & Coussot's "fifty-cent
/// rheometer", J. Rheol. 49, 2005). Inverting it turns the measured pile
/// back into a yield stress:
///
///   tau_0_measured = rho * g * h^2 / (2 L)
///
/// The diagnostic prints that next to the tau_0 that went in. Measured
/// headless on this scene (`bingham_slump_probe`), from a 39 mm column,
/// two simulated seconds, with every column at rest:
///
/// ```text
///     tau_0 in   deposit h   half-width L   read back   standing shear
///        2 Pa      13.4 mm      28.7 mm        31 Pa    1.18 of its yield
///       60 Pa      22.5 mm      21.2 mm       117 Pa    1.00
///     1200 Pa      39.0 mm      10.0 mm       749 Pa    0.66
/// ```
///
/// The last column is the one that does not depend on any geometry, and it
/// is what actually proves the law: at rest a yield-stress fluid holds a
/// shear stress up to tau_0 and no further. The middle column sits at
/// exactly 1.00 of its own yield, which is a material at its limit holding
/// a slope. The right one sits at 0.66, below its yield, so it is elastic
/// and does not flow at all -- it ends where it started, 39 mm.
///
/// The inversion above it assumes a thin, wide deposit (h << L), which the
/// left column meets only loosely (h/L = 0.47), the middle one not at all
/// (1.06), and the right one does not either: it is designed not to
/// spread, so h exceeds L and the formula reads low. That
/// is a limit of the measurement, stated rather than hidden, and it is why
/// the standing-shear column is printed next to it.
///
/// An earlier version of this table was never produced by this scene at
/// all: it claimed a 31 mm column and a right-hand deposit of 20.5 mm
/// where the engine, at that very commit, gives 8.9 mm. What it was
/// describing was a column of 8 mm across standing 40 mm tall, five to
/// one, which does not demonstrate a yield stress: it TOPPLES, and the
/// fall makes the stress that makes it flow. The columns are 2 to 1 now.
///
/// The caveat on geometry, measured (`tests/scratch_bingham_column_symmetry.rs`):
/// the columns TOUCH, and the middle one's shape is constrained by its
/// neighbours. The left deposit reaches the wall and the middle column,
/// the middle deposit reaches the right column, and on this frictionless
/// floor the whole row slides right, 7.4, 15.7 and 9.3 mm over three
/// seconds, the last from a column that never yields. Alone in the tank the
/// 60 Pa column spreads to about 47 mm a side, against the 21 mm in the
/// table above. So the deposits here are not those of three independent
/// slumps, and neither is the tau_0 read back from them. Alone, and spawned
/// on a mirror line of the grid, each column collapses symmetrically to
/// within a micrometre; the lean seen here is the neighbours.
///
/// # Interaction
///
/// Pushing shows the other half of the behaviour. A gentle push moves the
/// left column and not the right one; a hard enough push makes even the
/// right one flow. That is the yield criterion being crossed, live. The
/// tau_0 slider rescales all three together, so the ordering can be swept
/// instead of taken on trust.
///
/// The playback slider sets how much simulated time one rendered frame
/// advances. It is a viewing choice, not a physics one: the cost of a frame
/// is dominated by the acoustic CFL condition, which fixes how many
/// substeps a millisecond of this material costs, so asking for less
/// simulated time per frame buys frame rate and slower motion at exactly
/// the same physical fidelity.
///
///   LMB push  RMB pull  V toggle real shear-stress field  R reset  Q quit
///   cargo run --release --example basic_bingham --features render
use emerge::render::{ColorMode, Renderer};
use emerge::{
    BinghamFluidMaterial, BinghamProps, FromSI, SimConfig, Simulation, SlipBoundary, SpawnRegion,
};
use glam::{IVec2, Vec2};
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

const GRID: usize = 64;

/// 2 mm cells -- a 12.8 cm tabletop tank, the scale a slump of a few
/// centilitres actually happens at. Deposit height scales as tau_0/(rho g),
/// millimetres for these materials, so a metre-scale domain would collapse
/// the whole effect into a single cell.
const DX_M: f32 = 0.002;

/// Simulated time advanced per rendered frame, live-adjustable in the
/// panel. A viewing choice, not a physics one: every material constant
/// stays real SI and each substep is identical whatever this is set to.
///
/// It does NOT change how fast the slump plays. The work per simulated
/// millisecond is fixed by the substep the physics needs, about 13 ms of
/// wall clock per simulated ms here, so halving this doubles the frame
/// rate and leaves the playback speed where it was. What it changes is
/// smoothness, and that is why the default is 1 ms and not 2.
///
/// The demo advances a FIXED slice of simulated time per frame, so a frame
/// that takes longer is a moment where the motion on screen slows down: a
/// stutter. At 2 ms not one frame fits the 16.7 ms a 60 Hz display allows.
/// At 1 ms nearly all of the collapse does. Measured per frame over three
/// simulated seconds, headless, release, physics only with no rendering, and
/// with the demo itself closed (a running copy competing for the cores made
/// an earlier release run read slower than the `quick` profile):
///
///   BINGHAM_PROBE_FRAMES=1500 cargo run --release --example bingham_cost_probe
///
/// ```text
///   per frame  substeps  mean     fps   collapse: over 16.7 ms, max   settled: over 16.7, over 33, max
///   1.0 ms      10.0    14.5 ms   69        2 of 500,   22 ms            139 of 1000,   51,   255 ms
///   2.0 ms      19.2    25.2 ms   40      250 of 250,   69 ms           1250 of 1250,   39,   189 ms
/// ```
///
/// Two things in that table are not understood and are recorded rather
/// than explained away. Release is no faster than the `quick` profile here
/// (13.6 and 24.8 ms at the same two steps). And the long frames sit in the
/// SETTLED phase, in both profiles, while the collapse is clean. It is not
/// particle sleep, which this scene leaves off. Sustained-load CPU
/// throttling and background load are the untested candidates, so read the
/// settled column as this machine, not as the physics. Rendering adds to
/// every row.
///
/// Real-time ratio, stated rather than left to be noticed: at 1 ms and
/// 69 fps this advances 0.069 s of slump per second of wall clock, 14.5
/// times slower than life, and 16.7 times if the display caps it at 60 Hz.
/// No setting of this reaches real time; the substep does, and it is set
/// by the sound speed, whose own derivation is below. The one fix still
/// open is to run the physics on its own thread and let the display
/// interpolate at 60 Hz between the last two states, which removes the
/// stutter without touching the physics and would serve every demo.
const DT_S_DEFAULT: f32 = 0.001;

/// Real yield stresses in pascals, spanning the three bands
/// `BinghamFluidMaterial`'s own doc lists (biological 1-50, mud 50-500,
/// lava 100-2000). Everything else about the three columns is identical.
const YIELD_STRESS_PA: [f32; 3] = [2.0, 60.0, 1200.0];
/// Position, not a baked-in yield stress: the panel's slider rescales all
/// three, so a label naming a pascal value would go stale the moment it
/// moves. The `in=` field in each readout carries the live value.
const COLUMN_LABEL: [&str; 3] = ["left", "mid", "right"];
const COLUMN_X: [f32; 3] = [12.0, 32.0, 52.0];

/// Water-based suspensions, so the density is water's.
const RHO_KG_M3: f32 = 1000.0;

/// Plastic (post-yield) viscosity, identical for all three columns -- this
/// scene varies exactly one parameter, and this is not it. Inside the
/// 0.1-5 Pa.s band the same doc gives for wet clay.
const ETA_PA_S: f32 = 0.5;

/// Yield strain, tau_0/G: how far a yield-stress fluid can be sheared
/// before it starts to flow. Real ones measure in the 1-10% band pretty
/// much regardless of what they are, so 5% is a stated property of the
/// material class rather than three unrelated numbers picked per column.
/// It is what ties each column's storage modulus to its own tau_0, so
/// tau_0 stays the single independent variable of the scene.
const YIELD_STRAIN: f32 = 0.05;

/// 20 mm across for 40 tall. The scene used to stand these columns at
/// 8 mm across, an aspect ratio of 5 to 1, and the stiffest one did not
/// demonstrate a yield stress at all: it stood while the soft ones spread,
/// then TOPPLED, and the fall generated the stress that made it flow.
/// Measured (`bingham_slump_probe`): its standing shear sat at 0.59 of its
/// own yield, crossed 1.05 at the instant it fell, and it ended flatter
/// than the column with a twentieth of its yield stress. At 2 to 1 it
/// stays where it is put, which is the behaviour this scene is about.
const COLUMN_CELLS: IVec2 = IVec2::new(10, 20);
/// Particle lattice spacing, in cells. Named because the cursor needs it
/// too: it fixes each particle's mass per unit depth, and with it the force
/// a push in pascals has to apply.
const SPACING: f32 = 0.5;
/// The cursor's default push and pull `P`, its net force over its diameter
/// (see `cursor_traction.rs`). Measured in zero gravity, a block pushed
/// this way starts keeping a permanent deformation between `P = 1` and `2`
/// times its own yield stress, at the same multiple for 2, 60 and 1200 Pa
/// to within one step of the measured grid
/// (`tests/scratch_bingham_cursor_yield.rs`), lower than the `2 tau_0` the
/// mean shear `P / 2` alone would suggest. So 300 Pa yields the middle
/// column and stays far under the right one's 1200. The slumped columns
/// sit on their own yield surface already, so in this scene a much weaker
/// push moves the left and middle ones.
const CURSOR_TRACTION_PA: f32 = 300.0;
const FLOOR_CELLS: f32 = 2.0;

/// Weakly-compressible sound-speed derating (Monaghan 1994): resolving
/// water's real 1483 m/s would cost roughly 15000 substeps per frame, so
/// the reference sound speed is 10x the fastest flow speed this scene can
/// produce, which holds density fluctuation under 1%. `v_max` comes from
/// free fall over the column's own height, not from a tuned number, and the
/// bulk modulus is then the definition `K = rho c^2`.
fn bulk_modulus_pa() -> f32 {
    let column_height_m = COLUMN_CELLS.y as f32 * DX_M;
    let v_max = (2.0 * 9.81 * column_height_m).sqrt();
    let c_ref = 10.0 * v_max;
    RHO_KG_M3 * c_ref * c_ref
}

fn make_config(gravity_fraction: f32) -> SimConfig {
    let mut config = SimConfig {
        // The acoustic CFL bound here is ~2e-4 s, below the 1e-3 s default
        // floor; leaving the default would clamp the substep above its own
        // stability limit.
        min_dt: 1.0e-5,
        // Real arithmetic, not a knob turned until it stopped complaining.
        // The acoustic CFL bound at this sound speed is ~74 us, so a 5 ms
        // frame genuinely needs ~68 substeps, and the 64 default is a
        // budget rather than a physics cap (see its own doc). 256 leaves
        // room for the compression transient at first contact, where the
        // Tait EOS raises the local sound speed above its rest value.
        max_substeps_per_step: 256,
        ..SimConfig::earth(GRID, DX_M, DT_S_DEFAULT)
    };
    config.gravity *= gravity_fraction;
    config
}

/// All three columns, built through the SI property route so every number
/// entered is a real pascal. The materials come back too: their grid-unit
/// `yield_stress` is what the shear-stress colour scale normalizes by.
fn make_sim(gravity_fraction: f32, yield_scale: f32) -> (Simulation, [BinghamFluidMaterial; 3]) {
    let config = make_config(gravity_fraction);
    let k_pa = bulk_modulus_pa();

    let build = |tau0_pa: f32| {
        let props = BinghamProps {
            rho_kg_m3: RHO_KG_M3,
            eta_pa_s: ETA_PA_S,
            bulk_modulus_pa: k_pa,
            yield_stress_pa: tau0_pa * yield_scale,
            // The storage modulus below the yield point. Without it this
            // material computes its deviatoric stress purely from the
            // CURRENT rate of strain, so at rest it has none and all three
            // columns collapse into identical puddles -- the model's own
            // limitation, not a bug, confirmed against the same model in
            // `tmp/GeoTaichi`. Holding a shape needs stored elastic shear
            // energy; this is it.
            shear_modulus_pa: tau0_pa * yield_scale / YIELD_STRAIN,
            cavitation_pressure_pa: BinghamProps::air_entrained_cavitation_pressure(),
        };
        let mut m = BinghamFluidMaterial::from_physical(&props, &config);
        // Measured coefficients, not a substance claim: the continuous
        // phase of all three is water, so all three get water's measured
        // absorption (Pope & Fry 1997). The suspended solids have their own
        // spectrum this engine holds no measurement for; that stays a named
        // gap rather than an invented tint. Identical across the three, so
        // nothing distinguishes them visually except how they move.
        m.optics = Some(emerge::materials::optical::pure_water());
        m.specific_heat_j_kg_k = 4182.0; // water, CRC Handbook
        (m, props)
    };

    let spawn = |slot: usize, props: &BinghamProps| {
        SpawnRegion {
            spacing: SPACING,
            box_size: COLUMN_CELLS,
            box_center: Vec2::new(COLUMN_X[slot], FLOOR_CELLS + COLUMN_CELLS.y as f32 * 0.5),
            material_id: slot as u32,
            initial_velocity_scale: 0.0,
            ..SpawnRegion::for_sim(&config)
        }
        // Real density -> real particle mass, rather than a hand-picked one.
        .mass_from(props, &config)
    };

    let (m0, p0) = build(YIELD_STRESS_PA[0]);
    let (m1, p1) = build(YIELD_STRESS_PA[1]);
    let (m2, p2) = build(YIELD_STRESS_PA[2]);

    let mut sim = Simulation::new(config, spawn(0, &p0))
        .with_default_material(Box::new(m0))
        .with_material(1, Box::new(m1))
        .with_material(2, Box::new(m2))
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));
    let _ = sim.add_body(spawn(1, &p1));
    let _ = sim.add_body(spawn(2, &p2));
    (sim, [m0, m1, m2])
}

/// Turn each deposit back into a yield stress and compare it with the one
/// that was entered -- see this file's header for the relation and its
/// stated limits.
fn print_slump(sim: &Simulation, elapsed_s: f32, yield_scale: f32) {
    const G: f32 = 9.81;
    let mut line = format!("SLUMP t={elapsed_s:.2}s");
    for slot in 0..3u32 {
        let (mut top, mut lo, mut hi, mut n) = (f32::MIN, f32::MAX, f32::MIN, 0u32);
        let mut speed = 0.0f32;
        for p in sim.particles().iter().filter(|p| p.material_id == slot) {
            top = top.max(p.x.y);
            lo = lo.min(p.x.x);
            hi = hi.max(p.x.x);
            speed = speed.max(p.v.length());
            n += 1;
        }
        if n == 0 {
            continue;
        }
        let h_m = (top - FLOOR_CELLS).max(0.0) * DX_M;
        let half_width_m = ((hi - lo) * 0.5).max(1.0e-6) * DX_M;
        let tau_measured = RHO_KG_M3 * G * h_m * h_m / (2.0 * half_width_m);
        line += &format!(
            "  {}[h={:.1}mm L={:.1}mm in={:.0} meas={:.0} vmax={:.3}]",
            COLUMN_LABEL[slot as usize],
            h_m * 1000.0,
            half_width_m * 1000.0,
            YIELD_STRESS_PA[slot as usize] * yield_scale,
            tau_measured,
            speed,
        );
    }
    println!("{line}");
}

struct State {
    gfx: gui_common::Gfx,
    sim: Simulation,
    materials: [BinghamFluidMaterial; 3],
    renderer: Renderer,
    cursor_pos: [f32; 2],
    lmb: bool,
    rmb: bool,
    /// Pushes in pascals through a force field the solver integrates every
    /// substep; see `cursor_traction.rs` for why not once per frame.
    cursor: cursor_traction::CursorTraction,
    gravity_fraction: f32,
    yield_scale: f32,
    /// Simulated seconds advanced per rendered frame -- see `DT_S_DEFAULT`.
    step_seconds: f32,
    /// Simulated time since the last reset, which the playback slider makes
    /// different from `frame * step_seconds`.
    elapsed_s: f32,
    frame: u64,
    fps_timer: std::time::Instant,
    fps_frames: u64,
    last_fps: f32,
    /// Real deviatoric (shear) stress field. For a Bingham fluid this IS
    /// the quantity the yield criterion tests, so the colour scale is
    /// normalized by the middle column's own tau_0: the map saturates
    /// exactly where that material starts to flow.
    show_stress: bool,
}

impl State {
    async fn new(window: Arc<Window>) -> Self {
        let gfx = gui_common::Gfx::new(&window).await;
        let size = window.inner_size();
        let gravity_fraction = 1.0;
        let yield_scale = 1.0;
        let (mut sim, materials) = make_sim(gravity_fraction, yield_scale);
        let cursor =
            cursor_traction::CursorTraction::new(5.0, CURSOR_TRACTION_PA, CURSOR_TRACTION_PA)
                .with_lattice(RHO_KG_M3, SPACING, DX_M);
        sim.add_force_field(Box::new(cursor.field()));

        let mut renderer = Renderer::new(&gfx.device, sim.particles().len(), gfx.format);
        renderer.set_camera(&gfx.queue, GRID as u32, size.width, size.height, 0.6, true);
        renderer.set_color_mode(ColorMode::ByPhysics);
        // Optical coefficients read straight off the materials, so nothing
        // about the colour is typed into this file.
        let declared = renderer.adopt_material_optics(&gfx.queue, sim.materials());

        println!(
            "basic_bingham: {} particles, 3 columns, {declared} carrying a measured absorption spectrum",
            sim.particles().len(),
        );
        println!(
            "  same rho={RHO_KG_M3} kg/m3, eta={ETA_PA_S} Pa.s, K={:.0} Pa -- only tau_0 differs: {YIELD_STRESS_PA:?} Pa",
            bulk_modulus_pa(),
        );
        println!("  LMB push  RMB pull  V shear-stress field  R reset  Q quit");

        Self {
            gfx,
            sim,
            materials,
            renderer,
            cursor_pos: [0.0; 2],
            lmb: false,
            rmb: false,
            cursor,
            gravity_fraction,
            yield_scale,
            step_seconds: DT_S_DEFAULT,
            elapsed_s: 0.0,
            frame: 0,
            fps_timer: std::time::Instant::now(),
            fps_frames: 0,
            last_fps: 0.0,
            show_stress: false,
        }
    }

    fn resize(&mut self, w: u32, h: u32) {
        self.gfx.resize(w, h);
        if w == 0 || h == 0 {
            return;
        }
        self.renderer
            .set_camera(&self.gfx.queue, GRID as u32, w, h, 0.6, true);
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
        let (mut sim, materials) = make_sim(self.gravity_fraction, self.yield_scale);
        sim.add_force_field(Box::new(self.cursor.field()));
        self.sim = sim;
        self.materials = materials;
        self.renderer
            .adopt_material_optics(&self.gfx.queue, self.sim.materials());
        self.frame = 0;
        self.elapsed_s = 0.0;
    }

    fn update_and_render(&mut self, window: &Window) {
        self.sim
            .set_gravity(make_config(self.gravity_fraction).gravity);
        self.sim.set_step_duration(self.step_seconds);

        // A traction in pascals, not a multiple of weight: the old push was
        // `k * m * g`, so the gravity slider at zero switched the cursor off.
        // The field reads this every substep; nothing is applied here.
        {
            let position = self.cursor_grid();
            let mut cursor = self.cursor.shared();
            cursor.position = position;
            cursor.pushing = self.lmb;
            cursor.pulling = self.rmb && !self.lmb;
        }

        self.sim.step();

        if self.frame.is_multiple_of(100) {
            print_slump(&self.sim, self.elapsed_s, self.yield_scale);
        }

        self.frame += 1;
        self.elapsed_s += self.step_seconds;
        self.fps_frames += 1;
        if self.fps_timer.elapsed().as_secs_f32() >= 1.0 {
            self.last_fps = self.fps_frames as f32 / self.fps_timer.elapsed().as_secs_f32();
            self.fps_timer = std::time::Instant::now();
            self.fps_frames = 0;
        }

        if self.show_stress {
            let stress = self
                .sim
                .materials()
                .von_mises_stress_field(self.sim.particles());
            self.renderer.set_stress_field(stress);
            self.renderer
                .set_stress_scale(1.0 / self.materials[1].yield_stress.max(1.0e-9));
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

        let fps = self.last_fps;
        let mut gravity_fraction = self.gravity_fraction;
        let mut yield_scale = self.yield_scale;
        let mut step_seconds = self.step_seconds;
        let (mut push_pa, mut pull_pa, contact) = {
            let cursor = self.cursor.shared();
            (cursor.push_pa, cursor.pull_pa, cursor.contact)
        };
        // The cursor's own P is fixed; what it actually exerts depends on the
        // contact it makes, and a contact narrower than the cursor exerts
        // more. Shown, not assumed.
        let contact_line = match contact {
            Some(c) => format!(
                "exerting {:.0} Pa over {:.1} mm ({} particles)",
                c.traction_pa(),
                c.width_m * 1000.0,
                c.particles
            ),
            None if self.lmb || self.rmb => "nothing under the cursor".to_string(),
            None => "LMB push, RMB pull".to_string(),
        };
        let n_particles = self.sim.particles().len();
        let mut reset = false;

        gui_common::run_egui_frame(&mut self.gfx, window, &view, |ctx| {
            egui::Window::new("Bingham -- the slump test")
                .default_pos([10.0, 10.0])
                .default_width(300.0)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.label(format!("fps={fps:.0}  particles={n_particles}"));
                    // The slow-motion factor is computed from the fps this
                    // run is ACTUALLY reaching, not from an assumed 60: the
                    // label used to divide by 60 while the line above it
                    // printed the measured rate, so the panel contradicted
                    // itself three lines apart (33 fps shown, 8x claimed,
                    // 15x real).
                    let simulated_per_second = step_seconds * fps;
                    ui.label(if simulated_per_second > 0.0 {
                        format!(
                            "{:.1} ms of physics per frame: {:.0}x slower than life at the {:.0} fps this is running at",
                            step_seconds * 1000.0,
                            1.0 / simulated_per_second,
                            fps
                        )
                    } else {
                        format!(
                            "{:.1} ms of physics per frame (waiting for a frame rate to measure)",
                            step_seconds * 1000.0
                        )
                    });
                    ui.add(
                        egui::Slider::new(&mut step_seconds, 0.0005..=0.005)
                            .logarithmic(true)
                            .text("s / frame"),
                    );
                    ui.separator();
                    ui.label("Yield stress tau_0 -- the only difference:");
                    ui.label(format!(
                        "left {:.0} Pa   middle {:.0} Pa   right {:.0} Pa",
                        YIELD_STRESS_PA[0] * yield_scale,
                        YIELD_STRESS_PA[1] * yield_scale,
                        YIELD_STRESS_PA[2] * yield_scale,
                    ));
                    if ui
                        .add(egui::Slider::new(&mut yield_scale, 0.01..=5.0).text("x tau_0"))
                        .drag_stopped()
                    {
                        reset = true;
                    }
                    ui.separator();
                    ui.label("Gravity (1.0 = real IRL 9.81 m/s2):");
                    ui.add(egui::Slider::new(&mut gravity_fraction, 0.0..=1.0));
                    ui.label("Push P, the cursor's force over its diameter (shears ~P/2):");
                    ui.add(
                        egui::Slider::new(&mut push_pa, 1.0..=5000.0)
                            .logarithmic(true)
                            .suffix(" Pa"),
                    );
                    ui.label("Pull P:");
                    ui.add(
                        egui::Slider::new(&mut pull_pa, 1.0..=5000.0)
                            .logarithmic(true)
                            .suffix(" Pa"),
                    );
                    ui.label(&contact_line);
                    ui.separator();
                    ui.label("Same density, viscosity, bulk modulus and shape.");
                    ui.label("Left spreads, middle holds a slope, right keeps its shape.");
                    ui.label("V = shear stress, saturating at the middle column's yield.");
                    ui.separator();
                    ui.label("LMB push  RMB pull  V stress  R reset  Q quit");
                    if ui.button("Reset").clicked() {
                        reset = true;
                    }
                });
        });
        self.gravity_fraction = gravity_fraction;
        self.yield_scale = yield_scale;
        self.step_seconds = step_seconds;
        {
            let mut cursor = self.cursor.shared();
            cursor.push_pa = push_pa;
            cursor.pull_pa = pull_pa;
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
                    .with_title("emerge -- Bingham slump test (GUI)")
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
                    KeyCode::KeyV if pressed => {
                        s.show_stress = !s.show_stress;
                        s.renderer.set_color_mode(if s.show_stress {
                            ColorMode::ByStress
                        } else {
                            ColorMode::ByPhysics
                        });
                        let on = if s.show_stress { "ON" } else { "off" };
                        println!("shear-stress field {on}");
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

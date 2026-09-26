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
/// viscosity, same bulk modulus, same shape, same measured optics, and they
/// stand far enough apart that their deposits never meet. The ONLY
/// independent difference is tau_0:
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
/// A slump is not just a picture -- it measures tau_0, but only through a
/// relation that holds for the shape the deposit ends in. The demo picks
/// the relation from what it sees, never from the tau_0 that went in:
/// choosing by the answer would make the reading circular.
///
/// - A THIN deposit (h/L at most 0.1, a declared bound) is read by a force
///   balance. At rest, the yield stress at the base holds up the hydrostatic
///   pressure gradient, `tau_0 = rho g h |dh/dx|`; integrated from the front,
///   where h = 0, to the centre, where it is h over a half-width L, that is
///   `tau_0 = rho g h^2 / (2 L)`. Exact only as h/L goes to zero.
/// - A deposit that SLUMPED without thinning is read by the planar law
///   Staron, Lagree, Ray and Popinet fitted to their own simulations of
///   two-dimensional Bingham columns on a no-slip base ("Scaling laws for
///   the slumping of a Bingham plastic fluid", J. Rheol. 57, 1265, 2013,
///   their eq. 13). Simulations, not experiments: "No experimental data were
///   available to test the results of the numerical simulations of Bingham
///   fluid", in their words. The law:
///   `H / R0 = 3.01 (tau_0 / (rho g R0))^0.66`, R0 the initial half-width,
///   fitted over `0.06 <= tau_0 / (rho g R0) <= 1.6` with a correlation of
///   0.95. Inverted, and only trusted inside that range.
/// - A column that HELD ITS SHAPE has not revealed tau_0, only a floor: by
///   the same paper's eq. 14 a column slumps only when `H0 / R0` exceeds
///   `3.01 (tau_0 / (rho g R0))^0.66`, so not slumping bounds it from below.
/// - Anything else says "no valid reading".
///
/// Measured headless on this scene (`bingham_slump_probe`, which builds it
/// from the same `bingham_slump_scene.rs`), eight simulated seconds, release:
///
/// ```text
///     tau_0 in   deposit h   half-width L   h/L    standing shear   read back
///        2 Pa       5.5 mm      96.0 mm     0.06       1.18         1.6 Pa, thin layer, creeping
///       60 Pa      20.9 mm      26.3 mm     0.79       0.98          56 Pa, planar law
///     1200 Pa      38.9 mm       9.5 mm     4.08       0.34        > 151 Pa, held its shape
/// ```
///
/// Each reading is what its relation is worth and no more. The 2 Pa deposit
/// is the one point squarely inside a relation, and it is still creeping at
/// 0.9 mm/s, so its 1.6 Pa keeps falling slowly; the panel marks it
/// provisional and shows the speed. The 60 Pa one is read by a
/// fitted law, not an exact one. The 1200 Pa one gives a bound, which it
/// satisfies.
///
/// The standing-shear column does not depend on any geometry, and it is
/// what actually tests the law: at rest a yield-stress fluid holds a shear
/// stress up to tau_0 and no further. The middle column sits at 0.98 of its
/// own yield, a material at its limit holding a slope. The right one sits at
/// 0.34, well below, so it is elastic and does not flow. The left one reads
/// 1.18, above its yield, because it has not stopped: it is still flowing.
///
/// The floor grips. On a frictionless one the thin-layer reading comes out
/// five to ten times low, because without basal shear the material spreads
/// too far to be a slump test at all (`tests/scratch_bingham_isolated_slump.rs`).
///
/// The grip leaves a mark of its own, at the floor. The lowest quarter of
/// each slumped deposit ends about one percent dilated, a volume ratio of
/// 1.010 under the 2 Pa deposit and 1.011 under the 60 Pa one, with a fifth
/// to a quarter of its particles at the cavitation pressure, the most
/// tension the law allows; every band above it sits at 1 within 1e-3. It is
/// the grip, not the wall: the 60 Pa column alone, its particles within a
/// cell and a half of the floor, ends at 1.016 on this floor and 0.9995 on a
/// slip one, and the dilation appears during the impact, in the first fifth
/// of a second, then stays frozen (`tests/scratch_bingham_isolated_slump.rs`).
/// So the mean volume ratio above one, 1.0044 and 1.0033, is that bottom
/// layer, not the deposits. In every band, the bottom one included, the
/// vertical stress carries the weight overhead to within 14 percent in the
/// middle deposit and 10 in the right; the left one, three cells thick, is
/// too thin to split that finely (`tests/scratch_bingham_deposit_state.rs`).
/// Why a gripping floor stretches the layer it grips is not established:
/// issue #44.
///
/// Two earlier versions of this table were wrong, and how is worth keeping.
/// The first was never produced by this scene: it described columns 8 mm
/// across and 40 mm tall, five to one, which do not demonstrate a yield
/// stress, they TOPPLE, and the fall makes the stress that makes them flow;
/// the columns became 2 to 1. The second was produced by this scene, on a
/// 64-cell tank and a frictionless floor, and its columns touched: the left
/// deposit reached the middle one, the middle one the right, and the whole
/// row slid. It read 117 Pa for the 60 Pa column. Alone on that same floor
/// the column reads 11.2 Pa; its neighbours had raised the reading tenfold.
/// The tank is now wide enough, and spaced, from each column's measured
/// spread alone (`bingham_slump_scene.rs`).
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
/// simulated time per frame buys frame rate and smoothness at exactly the
/// same physical fidelity. It does not slow the slump down: the frame rate
/// rises in the same proportion, and the playback speed stays where it was.
///
///   LMB push  RMB pull  V toggle shear / own yield  R reset  Q quit
///   cargo run --release --example basic_bingham --features render
#[path = "bingham_slump_scene.rs"]
mod bingham_slump_scene;
use bingham_slump_scene::*;
use emerge::render::{ColorMode, Renderer};
use emerge::{BinghamFluidMaterial, MaterialModel, Simulation};
use glam::Vec2;
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

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
/// stutter. At 2 ms not one frame fits the 16.7 ms a 60 Hz display allows;
/// at 1 ms most do. Measured per frame over three simulated seconds on this
/// scene, headless, release, physics only with no rendering, with the demo
/// itself closed (a running copy competing for the cores once made a
/// release run read slow):
///
///   BINGHAM_PROBE_FRAMES=1500 cargo run --release --example bingham_cost_probe
///
/// ```text
///   per frame  substeps  mean     fps   collapse: over 16.7 ms, max   settled: over 16.7, over 33, max
///   1.0 ms      10.0    16.2 ms   62       81 of 500,   43 ms            276 of 1000,   19,    73 ms
///   2.0 ms      19.0    31.2 ms   32      250 of 250,  114 ms           1250 of 1250,  219,   399 ms
/// ```
///
/// Read the timing as this machine, not as the physics: `bingham_slump_probe`
/// ran the same scene at 1 ms in the same build and averaged 12.8 ms a
/// frame, 27 percent under this table's 16.2. The substep counts are the
/// part that does not move. Release is no faster than the `quick` profile
/// here, which is expected rather than odd: `quick` inherits `release` and
/// only drops link-time optimisation and runs 16 codegen units instead of
/// one, which a loop bound by arithmetic barely notices. Why the long
/// frames gather in the settled phase is not established; it is not
/// particle sleep, which this scene leaves off, and the solver's own
/// per-phase timing, printed for the worst frames only, would say whether
/// it is the engine or the machine. Rendering adds to every row.
///
/// Real-time ratio, stated rather than left to be noticed: at 1 ms and
/// 62 to 78 fps this advances 0.062 to 0.078 s of slump per second of wall
/// clock, 13 to 16 times slower than life, and 16.7 times if the display
/// caps it at 60 Hz.
/// No setting of this reaches real time; the substep does, and it is set
/// by the sound speed, whose own derivation is below. The one fix still
/// open is to run the physics on its own thread and let the display
/// interpolate at 60 Hz between the last two states, which removes the
/// stutter without touching the physics and would serve every demo.
const DT_S_DEFAULT: f32 = 0.001;

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

/// Half-widths, in cells, that `COLUMN_X` leaves each deposit: what they
/// measure alone once settled, the softest rounded up because it still
/// creeps. See `COLUMN_X`.
const DEPOSIT_HALF_WIDTH_CELLS: [f32; 3] = [50.0, 13.2, 4.8];

/// The stage the slump ends on, in cells: every deposit at its settled
/// width, the columns' full height, the floor, and three cells of margin.
/// The camera starts on this rather than on the standing columns, so it
/// does not have to zoom out while the softest one spreads; on screen that
/// read as the particles shrinking.
fn stage() -> (Vec2, Vec2) {
    let left = COLUMN_X[0] - DEPOSIT_HALF_WIDTH_CELLS[0];
    let right = COLUMN_X[2] + DEPOSIT_HALF_WIDTH_CELLS[2];
    let top = FLOOR_CELLS + COLUMN_CELLS.y as f32;
    (
        Vec2::new(left - 3.0, 0.0),
        Vec2::new(right + 3.0, top + 3.0),
    )
}

/// Every particle, a margin of three cells, and the floor: what the camera
/// must still contain if a push carries material off the stage.
fn footprint(sim: &Simulation) -> (Vec2, Vec2) {
    let (mut lo, mut hi) = (Vec2::splat(f32::MAX), Vec2::splat(f32::MIN));
    for p in sim.particles().iter() {
        lo = lo.min(p.x);
        hi = hi.max(p.x);
    }
    (Vec2::new(lo.x - 3.0, 0.0), hi + Vec2::splat(3.0))
}

/// Each deposit's shape and what it says about its yield stress, next to
/// the one that went in -- see this file's header for the relations and
/// their limits. One line per column, for the panel; the console gets them
/// with the shape every hundred frames.
fn read_deposits(
    sim: &Simulation,
    watches: &mut [SlumpWatch; 3],
    yield_scale: f32,
    g: f32,
) -> [String; 3] {
    std::array::from_fn(|slot| {
        let tau_in = YIELD_STRESS_PA[slot] * yield_scale;
        match deposit(sim, slot as u32) {
            Some((h_m, half_width_m, speed)) => format!(
                "{} ({tau_in:.0} Pa in, h {:.1} mm, L {:.1} mm): {}",
                COLUMN_LABEL[slot],
                h_m * 1000.0,
                half_width_m * 1000.0,
                describe(&watches[slot].read(h_m, half_width_m, speed, g))
            ),
            None => format!("{} ({tau_in:.0} Pa in): gone", COLUMN_LABEL[slot]),
        }
    })
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
    /// Colour each particle by the shear its yield criterion tests, divided
    /// by its OWN column's tau_0: blue well below yield, red from the moment
    /// a material starts to flow, in all three columns at once. Pressure is
    /// left out on purpose. The engine's generic stress view,
    /// `von_mises_stress_field`, takes the plane-stress von Mises of the full
    /// stress, where a pure pressure reads as its own magnitude; on these
    /// deposits that painted their weight, not their shear
    /// (`tests/scratch_bingham_deposit_state.rs`).
    show_stress: bool,
    /// The region the camera frames: the stage the slump ends on, grown
    /// only if a push carries material off it, never shrunk; reset with the
    /// scene.
    view: (Vec2, Vec2),
    /// Whether each column's collapse is over, for its reading.
    watches: [SlumpWatch; 3],
}

impl State {
    async fn new(window: Arc<Window>) -> Self {
        let gfx = gui_common::Gfx::new(&window).await;
        let size = window.inner_size();
        let gravity_fraction = 1.0;
        let yield_scale = 1.0;
        let (mut sim, materials) = make_sim(gravity_fraction, yield_scale, DT_S_DEFAULT);
        let cursor =
            cursor_traction::CursorTraction::new(5.0, CURSOR_TRACTION_PA, CURSOR_TRACTION_PA)
                .with_lattice(RHO_KG_M3, SPACING, DX_M);
        sim.add_force_field(Box::new(cursor.field()));

        let mut renderer = Renderer::new(&gfx.device, sim.particles().len(), gfx.format);
        let view = stage();
        renderer.set_camera_region(&gfx.queue, view, size.width, size.height, 0.6, true);
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
        println!("  LMB push  RMB pull  V shear / own yield  R reset  Q quit");

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
            view,
            watches: Default::default(),
        }
    }

    fn resize(&mut self, w: u32, h: u32) {
        self.gfx.resize(w, h);
        if w == 0 || h == 0 {
            return;
        }
        self.renderer
            .set_camera_region(&self.gfx.queue, self.view, w, h, 0.6, true);
    }

    fn cursor_grid(&self) -> Vec2 {
        // The camera frames a region, not the whole grid, so the cursor is
        // read back through that region's projection.
        gui_common::cursor_to_grid(
            self.cursor_pos,
            self.gfx.surface_config.width,
            self.gfx.surface_config.height,
            self.view,
        )
    }

    fn reset(&mut self) {
        let (mut sim, materials) = make_sim(self.gravity_fraction, self.yield_scale, DT_S_DEFAULT);
        sim.add_force_field(Box::new(self.cursor.field()));
        self.sim = sim;
        self.materials = materials;
        self.renderer
            .adopt_material_optics(&self.gfx.queue, self.sim.materials());
        self.frame = 0;
        self.elapsed_s = 0.0;
        self.view = stage();
        self.watches = Default::default();
        let (w, h) = (
            self.gfx.surface_config.width,
            self.gfx.surface_config.height,
        );
        self.renderer
            .set_camera_region(&self.gfx.queue, self.view, w, h, 0.6, true);
    }

    fn update_and_render(&mut self, window: &Window) {
        self.sim
            .set_gravity(make_config(self.gravity_fraction, DT_S_DEFAULT).gravity);
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

        let g = 9.81 * self.gravity_fraction;
        let readings = read_deposits(&self.sim, &mut self.watches, self.yield_scale, g);
        if self.frame.is_multiple_of(100) {
            println!("SLUMP t={:.2}s  {}", self.elapsed_s, readings.join("  "));
        }
        let (lo, hi) = footprint(&self.sim);
        let grown = (self.view.0.min(lo), self.view.1.max(hi));
        if grown != self.view {
            self.view = grown;
            let (w, h) = (
                self.gfx.surface_config.width,
                self.gfx.surface_config.height,
            );
            self.renderer
                .set_camera_region(&self.gfx.queue, self.view, w, h, 0.6, true);
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
            let p = self.sim.particles();
            let shear: Vec<f32> = (0..p.len())
                .map(|i| {
                    let m = &self.materials[p.material_id[i] as usize];
                    deviatoric_shear(m.kirchhoff_stress(p, i)) / m.yield_stress.max(1.0e-12)
                })
                .collect();
            self.renderer.set_stress_field(shear);
            self.renderer.set_stress_scale(1.0);
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
                    ui.separator();
                    ui.label("What each deposit says about its own yield stress:");
                    for line in &readings {
                        ui.label(line);
                    }
                    ui.label("V = shear / its own yield stress: red at yield, blue well below.");
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

extern crate emerge_engine as emerge;

#[path = "../gui_common/cursor_force.rs"]
mod cursor_force;
#[path = "../gui_common/mod.rs"]
mod gui_common;
use cursor_force::CursorForce;

/// Real, live demo of moisture diffusion driving a genuine phase transition
/// into a real mixture material: water poured onto a loose sand pile
/// diffuses through the shared grid (`ScalarDiffusionField`, PIC/FLIP-
/// blended -- see that field's own doc), and each sand particle's own
/// saturation moves it through TWO real, distinct regimes as it wets:
///
/// 1. Low saturation (0 to `PENDULAR_REGIME_CEILING`): sand's
///    `cohesion_bonus_pa` hook applies real, cited apparent cohesion from
///    capillary bridging between grains (the "sandcastle effect" --
///    Hornbaker et al. 1997; Halsey & Levine 1998) -- damp sand holds a
///    shape better than bone-dry sand.
/// 2. Past that ceiling: a real phase transition (`add_phase_rule`, see
///    `make_mixture`'s own doc) converts the particle into
///    `GranularFluidMaterial` (Dunatunga & Kamrin 2015) -- capillary
///    bridges between separate grains merge and break down at real
///    saturation, and the material genuinely becomes a continuous
///    granular-fluid mixture, not "the same sand with capped cohesion."
///    This is the actual mixture material this scene exists to
///    demonstrate, not the scalar-diffusion-only approximation an earlier
///    version of this scene used.
///
/// The sand starts loose enough to genuinely slump under its own gravity;
/// damp regions should visibly hold together, saturated regions should
/// visibly flow like wet mud, dry regions keep flowing like dry sand.
///
/// A poured water particle's own moisture is set directly to 1.0 at the
/// moment it's spawned (see the pour handler) -- it IS water, a fact, not
/// something that ramps up toward "wet" over an invented per-second rate.
/// Diffusion alone then spreads that moisture into neighboring sand.
///
/// `ColorMode::ByScalarField` (already generic, not built for this demo)
/// renders each particle's own moisture level directly -- dry sand stays
/// its normal color, wet sand lights up, so the diffusion itself is
/// visible, not just its downstream mechanical effect.
///
///   cargo run --example sand_water_saturation --features render
use emerge::render::{ColorMode, Renderer};
use emerge::thermodynamics::{ScalarDiffusionConfig, ScalarDiffusionField};
use emerge::{
    DruckerPragerMaterial, GranularFluidMaterial, NewtonianFluidMaterial, SimConfig, Simulation,
    SlipBoundary, SpawnRegion,
};
use glam::{IVec2, Vec2};
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

const GRID: usize = 64;
// 100 Hz outer step, not 10 Hz. With REAL sand stiffness the elastic wave
// speed is genuinely c = sqrt(E/rho) = 96.8 m/s, so CFL demands
// dt <= 0.7*dx/c -- about 1400 substeps per 0.1s frame, far past any sane
// budget. This is not a tuning fudge: it is what resolving real elastic
// waves actually costs, and `dt` is only the OUTER granularity (the solver
// adaptively substeps inside it either way).
const DT: f32 = 0.01;
const MAT_SAND: u32 = 0;
const MAT_WATER: u32 = 1;
const MAT_MIXTURE: u32 = 2;
// Real saturation threshold shared between `make_sand`'s own
// `pendular_regime_ceiling` and the phase rule below -- one source of
// truth, not two literals that could drift apart. See `cohesion_bonus_pa`'s
// own doc in sand.rs: pendular-regime capillary cohesion is a REAL but
// EXPLICITLY BOUNDED model (Hornbaker et al. 1997; Halsey & Levine 1998),
// honestly disclosed there as not modeling what real wet sand actually does
// past this saturation -- capillary bridges between separate grains merge
// and break down, and the material genuinely becomes a continuous granular-
// fluid mixture (funicular/capillary/slurry regime), not "the same sand
// with capped cohesion." This is exactly the gap `GranularFluidMaterial`
// (Dunatunga & Kamrin 2015) exists to fill; the phase rule below hands off
// to it at the exact point the pendular model's own doc says it stops
// applying, not a separately guessed threshold.
const PENDULAR_REGIME_CEILING: f32 = 0.3;
// Real, disclosed cap on how much poured water can add beyond the initial
// sand pile -- same reason `basic_sand.rs`'s own POUR_BUDGET exists: the
// renderer's wgpu instance buffer is allocated once, not resizable live.
const POUR_BUDGET: usize = 800;
const POUR_SPACING: f32 = 0.5;
const POUR_BOX: IVec2 = IVec2::new(2, 1);

/// The CANONICAL MPM sand parameters, taken from the real reference
/// implementations this engine cross-checks against: sparkl/wgsparkl's own
/// `DruckerPragerPlasticity::new(E, nu)` demo values (`sparkl basic2`:
/// E = 1e5, nu = 0.2), the same family Klar et al. 2016 works in.
///
/// NOT the geotechnical bulk-soil figure (10-28 MPa for loose dry sand).
/// That distinction is real and deliberate, not a shortcut: in MPM granular
/// simulation the visible behaviour -- angle of repose, flow, yield,
/// collapse -- is governed by the DRUCKER-PRAGER PLASTIC response
/// (`friction_angle` and the return mapping, both kept exactly real here),
/// while the elastic modulus is a numerical stiffness parameter. Feeding
/// the geotechnical 15 MPa in makes the elastic wave speed
/// `c = sqrt(E/rho)` ~12x higher, which explicit integration must resolve
/// (`dt <= cfl*dx/c`), for an elastic strain of 0.02% nobody can see. The
/// published MPM implementations use ~1e5 for exactly this reason.
const SAND_YOUNG_MODULUS_PA: f32 = 1.0e5;
const SAND_POISSON_RATIO: f32 = 0.2;
const SAND_DENSITY_KG_M3: f32 = 1600.0;
// Loose packing (this scene's sand is deliberately "loose enough to slump"),
// void fraction e/(1+e) from the same cohesionless-soil void-ratio database
// `min_volume_jacobian` already uses: e_max=0.92 -> 0.92/1.92.
const SAND_POROSITY_LOOSE: f32 = 0.4792;

/// Built from REAL SI values through the dimensionally-correct conversion
/// (`lame_from_si_physical_cfg`), not raw grid numbers. That is what lets
/// this scene run at genuine 9.81 m/s^2 with no `gravity_fraction` fudge:
/// stiffness and gravity land in one consistent unit system, so the ratio
/// that actually decides whether a pile holds its shape (`rho*g*h/E`) comes
/// out physically correct on its own instead of being hand-tuned.
fn make_sand(config: &SimConfig) -> DruckerPragerMaterial {
    let (lambda, mu) = config.lame_from_si_physical_cfg(
        SAND_YOUNG_MODULUS_PA,
        SAND_POISSON_RATIO,
        SAND_DENSITY_KG_M3,
    );
    // Capillary cohesion from real grain-scale physics (Lian, Thornton &
    // Adams 1993 bridge force + Rumpf 1962 tensile-stress model -- see
    // `capillary_cohesion_stress_pa`'s own doc), not a hand-picked
    // magnitude: the previous flat 6.0e4 (grid units) corresponded to
    // ~9.6 kPa real, roughly 8.5x the derived value below -- wet sand was
    // dramatically stiffer than real capillary bridging can produce,
    // measured live as "tout se colle ensemble."
    let cohesion_pa = emerge::matter::materials::granular::sand::capillary_cohesion_stress_pa(
        emerge::matter::materials::granular::sand::GRAIN_DIAMETER_M,
        SAND_POROSITY_LOOSE,
        0.0, // water on clean quartz: fully wetting
    );
    let saturation_cohesion_coeff = config.stress_from_si_physical(cohesion_pa, SAND_DENSITY_KG_M3);
    // Real small-strain Kelvin-Voigt damping -- see
    // `small_strain_elastic_viscosity_pa_s`'s own doc (Seed & Idriss 1970 +
    // Darendeli 2001, zeta 0.5%-2% for clean sand). Below the Drucker-Prager
    // yield cone, this material has zero built-in dissipation on its own; a
    // firm push otherwise leaves kinetic energy ringing for 500+ substeps
    // with no mechanism to arrest it -- measured live as sand "springing
    // back" no matter how hard it's disturbed. Bottom of the cited range
    // used here -- measured 2026-08-25 that the top (1%) roughly doubles
    // substep count via the viscous CFL bound (`timestep_bound`), a real
    // interactive-fps cost this live demo actually pays.
    let shear_modulus_pa = SAND_YOUNG_MODULUS_PA / (2.0 * (1.0 + SAND_POISSON_RATIO));
    let elastic_viscosity_pa_s =
        emerge::matter::materials::granular::sand::small_strain_elastic_viscosity_pa_s(
            shear_modulus_pa,
            0.005,
        );
    let elastic_viscosity =
        config.visc_from_si_physical(elastic_viscosity_pa_s, SAND_DENSITY_KG_M3);
    DruckerPragerMaterial {
        // Real quartz critical-state friction angle (Bolton 1986, "The
        // strength and dilatancy of sands," Geotechnique 36(1):65-78) --
        // an intrinsic material property independent of density/dilatancy,
        // the correct floor for a genuinely LOOSE pile (near-zero
        // dilatancy). The previous 27 deg was below the real geotechnical
        // minimum for ANY sand condition (28-30 deg for loose sand,
        // web-confirmed 2026-08-26) -- not a real material state, picked
        // to force visible slumping.
        friction_angle: 33.0_f32.to_radians(),
        saturation_cohesion_coeff,
        pendular_regime_ceiling: PENDULAR_REGIME_CEILING,
        elastic_viscosity,
        ..DruckerPragerMaterial::new(lambda, mu)
    }
}

/// Real granular-fluid mixture (Dunatunga & Kamrin 2015 -- Tait EOS +
/// corotated elastic + SVD plasticity, see `GranularFluidMaterial`'s own
/// module doc) for sand that has crossed `PENDULAR_REGIME_CEILING`. This is
/// the actual mixture material this scene exists to demonstrate, replacing
/// the earlier version's scalar-diffusion-only approximation (moisture just
/// raised dry sand's apparent cohesion, with nothing modeling what happens
/// once it's genuinely saturated).
///
/// `saturated_loam`'s own doc HONESTLY DISCLOSES its shape parameters
/// (eos_stiffness, hardening_exponent, compression_limit) as real-law/
/// hand-tuned-values, not measured geotechnical loam data -- kept as-is
/// here rather than re-guessing new numbers, same standard the rest of this
/// codebase holds unsourced-but-disclosed constants to.
///
/// `rest_density` is the one field overridden from the preset: `saturated_
/// loam` hardcodes it to a scene-agnostic `1.0`, but this scene's other
/// materials (see `make_sim`'s water) are built from `config.grid_density`,
/// the solver's own real SI-derived reference -- using the preset's literal
/// `1.0` here would silently reintroduce the exact reference-density
/// mismatch class of bug this session's citation/render sweep spent all
/// night finding and fixing elsewhere. Corrected to the real, scene-
/// consistent value.
///
/// Elastic modulus halved from dry sand's own numerical `E` (Terzaghi's
/// effective-stress principle: pore water pressure carries part of the
/// total stress once saturated, so the load-bearing grain skeleton is
/// genuinely softer -- directionally real, not an independently measured
/// wet-sand modulus; disclosed as such).
///
/// Built as a struct literal rather than calling `saturated_loam(E, nu)`
/// directly: that constructor runs plain `lame_from_young` internally, with
/// no SI-to-grid conversion -- correct for a caller who's already in grid
/// units, but this scene's other materials (see `make_sand`) go through
/// `config.lame_from_si_physical_cfg`, the dimensionally-correct path. Real
/// SI here, `lame_from_si_physical_cfg`-converted like everything else in
/// this file, then the rest of `saturated_loam`'s own disclosed shape
/// values (eos_stiffness/hardening_exponent/compression_limit/etc, and the
/// anti-elastic-bounce viscosity terms scaled off THIS material's own
/// correctly-converted mu/eos_stiffness) copied over unchanged.
fn make_mixture(config: &SimConfig) -> GranularFluidMaterial {
    let (lambda, mu) = config.lame_from_si_physical_cfg(
        SAND_YOUNG_MODULUS_PA * 0.5,
        SAND_POISSON_RATIO,
        SAND_DENSITY_KG_M3,
    );
    const EOS_STIFFNESS: f32 = 200.0;
    GranularFluidMaterial {
        mu,
        lambda,
        rest_density: config.grid_density,
        eos_stiffness: EOS_STIFFNESS,
        eos_power: 2.0,
        hardening_exponent: 5.0,
        compression_limit: 0.4,
        stretch_limit: 0.01,
        min_plastic_jacobian: 0.2,
        max_plastic_jacobian: 3.0,
        pressure_floor: 0.0,
        dynamic_viscosity: 0.3 * mu,
        bulk_viscosity: 0.5 * EOS_STIFFNESS,
    }
}

fn make_sim() -> Simulation {
    let config = SimConfig {
        boundary_thickness: 3,
        // 12 (basic_sand.rs's own value) panics: that config was tuned for
        // sand alone, no fluid material in the scene. Strict WC-MPM water
        // has real, tighter CFL/stability requirements -- see
        // basic_fluids.rs's own identical fix, same real precedented value,
        // matching basic_fluids_gpu.rs's own.
        // Real headroom for genuine SI stiffness under real gravity.
        max_substeps_per_step: 400,
        material_cfl_coefficient: 0.7,
        ..SimConfig::earth(GRID, 0.01, DT)
    };
    let sand_spawn = SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new(30, 16),
        box_center: Vec2::new(32.0, 38.0),
        material_id: MAT_SAND,
        precompute_initial_volumes: true,
        initial_velocity_scale: 0.0,
        rng_seed: 11,
        position_jitter: 0.5,
        ..SpawnRegion::for_sim(&config)
    };
    Simulation::new(config, sand_spawn)
        .with_default_material(Box::new(make_sand(&config)))
        // rest_density is the density the SOLVER measures, which is a ratio
        // against `reference_density_kg_m3` -- so water at the reference sits at
        // `grid_density` exactly. Read it from the config rather than writing a
        // literal: the old hardcoded 4.0 was really `1/spacing^2` in disguise
        // and silently became wrong the moment the spawn was refined.
        .with_material(
            MAT_WATER,
            Box::new(NewtonianFluidMaterial::low_viscosity(
                config.grid_density,
                10.0,
            )),
        )
        .with_material(MAT_MIXTURE, Box::new(make_mixture(&config)))
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)))
        // The real mixture transition this scene exists to demonstrate --
        // see PENDULAR_REGIME_CEILING's own doc for why this exact
        // threshold, not a separately guessed one. Evaluated every substep
        // (`add_phase_rule`'s own contract).
        //
        // KNOWN, DISCLOSED, UNFIXED ISSUE (2026-08-26/27): `apply_phase_
        // transition` resets a transitioning particle's deformation_gradient
        // to IDENTITY, which drops GranularFluidMaterial's own EOS pressure
        // to exactly zero regardless of how much real compressive load that
        // particle was carrying as sand the substep before -- a genuine
        // stress discontinuity, confirmed and reproduced in a controlled
        // diagnostic (`diag_phase_transition_under_load_causes_stress_
        // discontinuity`, tests/physics_correctness.rs) and the direct
        // cause of a real crash after ~104,737 frames of live interactive
        // testing ("strict WC-MPM fluid could not advance the full
        // requested dt" -- the shock propagated into nearby water's own
        // strict CFL/retry check). A `GasMaterial`-style
        // `init_particle_from_transition` fix was tried and made the
        // measured spike WORSE, not better (see granular_fluid.rs's own
        // reverted-attempt comment on `GranularFluidMaterial` for the full
        // writeup) -- root cause not yet fully understood. This scene keeps
        // the real mixture transition (that's the actual point of it) but
        // a very long, heavy interactive session can still hit this crash.
        .with_phase_rule(|p| {
            if p.material_id == MAT_SAND && p.scalar_field > PENDULAR_REGIME_CEILING {
                Some(MAT_MIXTURE)
            } else {
                None
            }
        })
}

fn make_moisture_field(grid_res: usize) -> ScalarDiffusionField {
    let mut field = ScalarDiffusionField::new(
        ScalarDiffusionConfig {
            // Real cited sandy-soil moisture diffusivity, horizontal-
            // infiltration measurements span 1e-9..1.67e-4 m^2/s (two
            // independent sources, 2026-08-26: real D(theta) is highly
            // nonlinear, varying 4-5+ orders of magnitude between dry and
            // near-saturated water content -- pore-scale mechanism: large
            // pores empty first as soil dries, leaving fewer, smaller,
            // more tortuous conducting paths). This scene's water is
            // poured directly onto the pile -- a near-saturated wetting
            // front at the contact point, not the dry/low-moisture regime
            // -- so the physically correct point in that cited range is
            // near its TOP (1.67e-4), not a middle guess. The previous
            // 1e-6 (this scene's earlier fix, itself real but for the
            // WRONG end of the same cited range) measured completely
            // inert: 0/1920 sand particles ever reached the cohesion
            // ceiling after a realistic ~3s pour
            // (`diag_wet_sand_cohesion_spread_after_realistic_pour`,
            // `tests/physics_correctness.rs`) -- the diffusion LENGTH
            // `sqrt(D*t)` at 1e-6 over 3s doesn't even reach the nearest
            // sand particle. Converted to grid units: D/dx^2 = 1.67e-4/1e-4.
            // (The original bug this all traces back to: 0.5, picked by
            // feel, flooded the whole pile in seconds.)
            diffusivity: 1.67,
            decay_rate: 0.0,
            ambient: 0.0,
        },
        |p| p.scalar_field,
        |p, delta| p.scalar_field += delta,
        grid_res,
    );
    // No `field.source`: a water particle's moisture is set directly to
    // 1.0 the moment it's poured (see the pour handler) -- it IS water, a
    // fact, not a process that ramps up over an invented per-second rate.
    // Diffusion (above) is the only real transport left, spreading that
    // moisture into neighboring sand exactly as measured/cited.
    // Pure FLIP (1.0): the ONLY transport is the real Laplacian term, so
    // what is on screen is genuine diffusion. A PIC-leaning blend snaps each
    // particle most of the way toward its local grid average EVERY step,
    // which at this scene's real diffusivity is ~700x stronger than the
    // actual physics -- it reads as instant flooding, not propagation.
    field.blend = 1.0;
    field
}

struct State {
    gfx: gui_common::Gfx,
    sim: Simulation,
    renderer: Renderer,
    cursor_pos: [f32; 2],
    lmb: bool,
    rmb: bool,
    pouring: bool,
    poured_count: usize,
    /// Real, shared cursor force -- see `CursorForce`'s own doc for why
    /// push/pull are separate strengths, not one shared value (the real
    /// bug this scene originally shipped, then fixed, then extracted so
    /// the other ~20 examples with the same hand-rolled pattern have a
    /// correct shared implementation to migrate onto instead of repeating
    /// the same mistake independently).
    cursor_force: CursorForce,
    pour_seed: u32,
    // No gravity fudge field. This scene's materials come from real SI
    // through the dimensionally-correct conversion, so gravity stays at the
    // genuine 9.81 m/s^2 `SimConfig::earth` derives. Every OTHER interactive
    // example still carries a hand-tuned `gravity_fraction` (0.001-0.01, a
    // 10x spread) precisely because its materials are raw grid numbers with
    // no defined relationship to gravity -- see `make_sand`.
    frame: u64,
    fps_timer: std::time::Instant,
    fps_frames: u64,
    solve_micros: u64,
    last_fps: f32,
}

impl State {
    async fn new(window: Arc<Window>) -> Self {
        let gfx = gui_common::Gfx::new(&window).await;
        let size = window.inner_size();
        let mut sim = make_sim();
        sim.attach_scalar_field(make_moisture_field(GRID));
        let render_capacity = sim.particles().len() + POUR_BUDGET;
        let mut renderer = Renderer::new(&gfx.device, render_capacity, gfx.format);
        renderer.set_camera(&gfx.queue, GRID as u32, size.width, size.height, 0.9, true);
        // ByScalarField, not ByPhysics -- the whole point of this demo is
        // watching moisture actually spread, not just the material split.
        renderer.set_color_mode(ColorMode::ByScalarField);

        println!(
            "sand_water_saturation: {} particles  |  LMB push  RMB pull  hold P to pour water  R reset  Q quit",
            sim.particles().len()
        );
        println!(
            "Color = moisture level (ByScalarField): dry sand stays dark, wet sand lights up."
        );
        println!(
            "Past {PENDULAR_REGIME_CEILING:.1} saturation, sand really becomes a granular-fluid mixture (real phase transition, not just capped cohesion)."
        );
        Self {
            gfx,
            sim,
            renderer,
            cursor_pos: [0.0; 2],
            lmb: false,
            rmb: false,
            pouring: false,
            poured_count: 0,
            // radius=7, push=3.0 (retained_fraction=1.000 across every real
            // LMB usage pattern, verified 2026-08-26), pull=7.0 (RMB needs
            // to overcome a packed pile's own confinement -- 3.0 gave 0.097
            // cells of real lift, nothing; 7.0 gives 5.94, clearly real).
            cursor_force: CursorForce::new(7.0, 3.0, 7.0),
            pour_seed: 1000,
            frame: 0,
            fps_timer: std::time::Instant::now(),
            fps_frames: 0,
            solve_micros: 0,
            last_fps: 0.0,
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

    fn update_and_render(&mut self, window: &Window) {
        if self.lmb || self.rmb {
            // Real F=ma, not a velocity poke -- see `gui_common::
            // CursorForce`'s own doc for why (mass genuinely resisting
            // acceleration, same shape gravity itself takes) and for why
            // push/pull are separate strengths, not one shared value.
            let g = self.sim.config().gravity.length();
            let cursor = self.cursor_grid();
            self.cursor_force.apply(
                self.sim.particles_mut(),
                cursor,
                g,
                DT,
                self.rmb, // pulling
            );
        }

        if self.pouring && self.poured_count < POUR_BUDGET {
            let config = self.sim.config();
            let half = POUR_BOX.as_vec2() * 0.5;
            let domain_min = Vec2::splat(config.boundary_thickness as f32) + half;
            let domain_max =
                Vec2::splat((config.grid_res - config.boundary_thickness) as f32) - half;
            let cursor = self
                .cursor_grid()
                .clamp(domain_min, domain_max.max(domain_min));
            self.pour_seed += 1;
            let spawn = SpawnRegion {
                spacing: POUR_SPACING,
                box_size: POUR_BOX,
                box_center: cursor,
                material_id: MAT_WATER,
                precompute_initial_volumes: true,
                initial_velocity_scale: 0.0,
                rng_seed: self.pour_seed,
                position_jitter: 0.3,
                ..SpawnRegion::for_sim(self.sim.config())
            };
            let before = self.sim.particles().len();
            let _ = self.sim.add_body(spawn);
            // A newly poured particle IS water -- fully saturated by
            // definition, not something that ramps up to "wet" over time.
            // Setting this directly (not via an invented per-second
            // accumulation rate) removes the only unsourced constant left
            // in this scene's moisture coupling: see `capillary_cohesion_
            // stress_pa`'s and `small_strain_elastic_viscosity_pa_s`'s own
            // docs for why every OTHER magnitude here is real and cited --
            // this was the one that wasn't, and there was never a real
            // "infiltration delay" this scene needed to model.
            let p = self.sim.particles_mut();
            for i in before..p.len() {
                p.scalar_field[i] = 1.0;
            }
            self.poured_count += p.len() - before;
        }

        // Split the frame into solve vs everything-else so a perf claim about
        // this scene is measured rather than assumed -- physics and the render
        // path have very different fixes.
        let solve_start = std::time::Instant::now();
        self.sim.step();
        self.solve_micros += solve_start.elapsed().as_micros() as u64;
        self.frame += 1;
        self.fps_frames += 1;
        if self.fps_timer.elapsed().as_secs_f32() >= 1.0 {
            let elapsed = self.fps_timer.elapsed().as_secs_f32();
            self.last_fps = self.fps_frames as f32 / elapsed;
            let solve_ms = self.solve_micros as f32 / 1000.0 / self.fps_frames.max(1) as f32;
            let frame_ms = elapsed * 1000.0 / self.fps_frames.max(1) as f32;
            println!(
                "frame={} fps={:.1} particles={} | solve={:.1}ms rest={:.1}ms ({:.0}% solve) substeps={}",
                self.frame,
                self.last_fps,
                self.sim.particles().len(),
                solve_ms,
                frame_ms - solve_ms,
                100.0 * solve_ms / frame_ms,
                self.sim.last_substeps(),
            );
            self.fps_timer = std::time::Instant::now();
            self.fps_frames = 0;
            self.solve_micros = 0;
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
        let mut push_weights = self.cursor_force.push_strength;
        let mut pull_weights = self.cursor_force.pull_strength;
        let n_particles = self.sim.particles().len();
        let poured = self.poured_count;
        let mut reset = false;

        gui_common::run_egui_frame(&mut self.gfx, window, &view, |ctx| {
            egui::Window::new("Sand + Water Saturation")
                .default_pos([10.0, 10.0])
                .default_width(260.0)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.label(format!("fps={fps:.0}  particles={n_particles}"));
                    ui.separator();
                    ui.label("Gravity: real 9.81 m/s² (no fudge factor)");
                    ui.separator();
                    ui.label("LMB push force (x particle weight, 1.0 = cancels gravity):");
                    ui.add(egui::Slider::new(&mut push_weights, 0.0..=10.0));
                    ui.label("RMB pull/lift force (needs more to beat pile confinement):");
                    ui.add(egui::Slider::new(&mut pull_weights, 0.0..=15.0));
                    ui.separator();
                    ui.label(format!("Water poured: {poured}/{POUR_BUDGET}"));
                    ui.add(
                        egui::ProgressBar::new(poured as f32 / POUR_BUDGET as f32)
                            .desired_width(200.0),
                    );
                    ui.separator();
                    ui.label("Dry sand slumps freely. Pour water (P) on part of");
                    ui.label("the pile -- the wet region should hold its shape");
                    ui.label("while the dry region keeps flowing.");
                    ui.separator();
                    ui.label("LMB push  RMB pull  hold P to pour  R reset  Q quit");
                    if ui.button("Reset").clicked() {
                        reset = true;
                    }
                });
        });
        self.cursor_force.push_strength = push_weights;
        self.cursor_force.pull_strength = pull_weights;
        if reset {
            let mut sim = make_sim();
            sim.attach_scalar_field(make_moisture_field(GRID));
            self.sim = sim;
            self.frame = 0;
            self.poured_count = 0;
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
                    .with_title("emerge -- Sand + Water Saturation")
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
                    KeyCode::KeyP => s.pouring = pressed,
                    KeyCode::Escape | KeyCode::KeyQ if pressed => el.exit(),
                    KeyCode::KeyR if pressed => {
                        let mut sim = make_sim();
                        sim.attach_scalar_field(make_moisture_field(GRID));
                        s.sim = sim;
                        s.frame = 0;
                        s.poured_count = 0;
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

extern crate emerge_engine as emerge;

#[path = "../gui_common/render_mode.rs"]
mod render_mode;
#[path = "../gui_common/scripted.rs"]
mod scripted;

use egui_wgpu::ScreenDescriptor;
/// Newtonian water dam-break with a live egui panel, the pattern of
/// `basic_sand.rs`/`basic_snow.rs`: a gravity slider (1.0 = Earth's 9.81 m/s²),
/// push/pull, and directional-drag digging (the mechanism of `basic_sand.rs`: a
/// per-particle velocity nudge along the cursor's movement, no second body, no
/// contact_group tuning, mass-conserving by construction).
///
/// Phase range: water below 273 K freezes into ice (`NeoHookeanMaterial` wrapped in
/// `WithLatentHeat(-334_000.0)`, the exothermic value `latent_heat.rs` uses), with the
/// ambient set through `Simulation::thermal_config_mut()`. A Warm/Cold toggle rather
/// than a continuous slider: a two-state toggle shows the phase transition without a
/// hunt for the right value.
///
/// This explicit WC-MPM branch has no hidden Jacobian floor: an inadmissible
/// state is reported rather than replaced by a capped deformation. Use the
/// material's sound speed, viscosity, and CFL limit to choose a physically
/// resolved scene rather than treating the gravity slider as a stabilization
/// parameter.
///
///   cargo run --example basic_fluids --features render
///
/// `EMERGE_SCRIPT_LOG=<file>` runs a scripted hand instead of the mouse and
/// logs every step (`gui_common/scripted.rs`): the dam breaks for four
/// seconds, then the strongest push into the pile, on the front, and a
/// pull. Measured that way at this demo's gravity, the water's fastest
/// particle reaches 15.9 cells/s on its own and 25.7 under the hand, four
/// and six and a half times the 3.95 cells/s its stiffness is sized for;
/// at real gravity (`EMERGE_SCRIPT_GRAVITY=1`) the water is crushed to half
/// its volume (J at its 0.5 floor) and lies two rows deep.
use emerge::materials::MaterialModel;
use emerge::render::{ColorMode, CpuRenderBridge, Renderer};
use emerge::thermodynamics::{ThermalConfig, ThermalDiffusion};
use emerge::{
    FixedStepConfig, FixedStepController, NeoHookeanMaterial, NewtonianFluidMaterial, SimConfig,
    Simulation, SlipBoundary, SpawnRegion, WithLatentHeat,
};
use glam::{IVec2, Vec2};
use render_mode::RenderMode;
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

// No foam/spray (Ihmsen-simplified trapped-air potential with Spray/Foam secondary
// particles): its render and perf cost is not worth carrying while the core
// render/perf/physics work is unsettled.

const GRID: usize = 64;
const DT: f32 = 0.1;
// Requested fixed-physics-step rate for interactive playback.  This is a
// playback setting, not an EOS/CFL/material parameter.  The release audit
// measured only 35.7--38.0 rendered FPS at the old 60 Hz request; because
// this demo intentionally permits at most one outer step per rendered frame,
// 60 Hz made its simulated speed follow the fluctuating render rate.  30 Hz
// stays below the measured sustained capacity while retaining 3x playback at
// `DT=0.1`: the same A/B measured 49.8--60.0 FPS, with finite particles and
// bounded water J, for the initial un-interacted scene.
const PLAYBACK_STEP_RATE_HZ: f32 = 30.0;
/// The push slider's top, which a scripted run presses at.
const PUSH_STRENGTH_MAX: f32 = 20.0;
// Particle spacing, the particles-per-cell (PPC) sampling rate: a discretization
// parameter, not a cosmetic one. With grid_cell_size = 1.0, spacing s gives
// PPC = (1/s)^2, so 0.5 gives the 4 PPC of standard 2D MPM (particles seeded at
// dx/2, as in Hu et al.'s MLS-MPM and the reference implementations in tmp/). At
// 0.9 (1.23 PPC) each grid node is supported by barely one particle, which
// under-resolves the transfer, makes the free surface ragged and starves both
// density-based render paths. `box_size` is in cells, so the columns keep their
// dimensions and only the sampling density changes; particle mass is rho0 *
// SPACING^2, so it follows. A top-level const so `set_camera` calls can size
// `particle_scale` from it.
const SPACING: f32 = 0.5;
/// Rendered diameter of one particle, for `RenderMode::Particles`.
///
/// The quad spans `local_pos` in [-0.5, 0.5], so the drawn disc's DIAMETER is
/// exactly the `particle_scale` passed to `set_camera`. A diameter equal to
/// `SPACING` (the particle pitch) covers only pi/4 = 78.5% of a square lattice,
/// leaving 21.5% of the fluid as dark gaps at the diagonals, so the particle view
/// looks speckled rather than liquid.
///
/// A material point represents an area of `SPACING^2`, so the disc carrying
/// exactly that area has `pi*r^2 = SPACING^2`, i.e. diameter
/// `2/sqrt(pi) * SPACING ~= 1.128 * SPACING`. That is this constant: each
/// particle draws precisely the fluid area it actually stands for -- no
/// arbitrary fudge factor, and it stays correct automatically if SPACING
/// changes.
const PARTICLE_RENDER_DIAMETER: f32 = SPACING * std::f32::consts::FRAC_2_SQRT_PI;
const MAT_WATER: u32 = 0;
const MAT_ICE: u32 = 1;
const FREEZING_POINT: f32 = 273.0;
const ICE_LATENT_HEAT: f32 = -334_000.0; // exothermic: freezing releases energy (real water: 334 kJ/kg)
const WARM_AMBIENT: f32 = 300.0;
const COLD_AMBIENT: f32 = 250.0;
// 1/s, Newton cooling rate for the "Cold ambient" toggle -- models convective
// heat loss to surrounding cold air (a freezer), not bulk Fourier conduction
// through the water's own interior (too slow to visibly freeze this scene's
// real ~0.6m water column within any reasonable play session). Chosen
// empirically for a reasonable interactive wait (ice appears within ~2
// minutes).
const FREEZER_COOLING_RATE: f32 = 0.08;
// Radius of the directional dig nudge, grid cells -- matches basic_sand.rs.
const DIG_RADIUS: f32 = 4.0;

fn make_sim() -> Simulation {
    let config = SimConfig {
        min_dt: 1.0e-4,
        // Substep budget as in `basic_fluids_gpu.rs`. A frame-rate ladder for
        // this water-only scene (raise the cap in increments, keep a raise only
        // if live fps stays >= 45) has not been redone at this value. A cap the
        // CFL scan runs into drops simulated time instead of advancing it (a cap
        // of 8 once dropped 61% of each frame's time behind a comfortable fps).
        max_substeps_per_step: 150,
        // `spatial_sort_enabled` stays off: at this demo's ~1288 particles it
        // measured 46-59 fps -> 22-27 fps, and in a headless benchmark at 67,600
        // particles 52.9 ms/step unsorted against 79.0 ms/step sorted (+49%), with
        // the sort computed once per outer step. In debug builds the O(N log N)
        // sort costs more than the P2G cache-locality it buys. The feature stays
        // opt-in (default `false`, tested, see
        // `spatial_sort_order`/`scatter_particles_to_grid_sorted`) for a release
        // build or another access pattern.
        // 0.3: the CFL number C in the explicit acoustic condition
        // `dt <= C * dx / c_sound`, where C < 1 is the stability limit and solvers
        // run C = 0.2-0.4 for margin (Monaghan 1992/1994 uses 0.25-0.3 for SPH;
        // MLS-MPM commonly 0.3-0.5). A "cfl=0.5 panics on frame 1" failure came
        // from a ~100x too soft EOS (the column collapsed into the J clamp), which
        // no CFL number can save; with the stiffness derived (see the water
        // material), substeps/frame went 11 -> 54 (correct water is ~5x more work,
        // a higher sound speed being the point of a stiffer EOS), and C 0.1 -> 0.3
        // recovers ~3x of that from the safety margin rather than the physics.
        // This demo's only phase rule is the water->ice freeze predicate, a
        // thermodynamic test -- and temperature now advances once per step
        // (diffusion runs at its own stable rate), so it cannot change within
        // a frame. Opting in skips ~17 redundant O(N) scans per frame; see
        // SimConfig::phase_rules_once_per_step for why this is a caller's
        // choice rather than a silent default.
        phase_rules_once_per_step: true,
        material_cfl_coefficient: 0.3,
        cfl_include_affine_speed: false,
        // `fluid_near_wall_cfl_scale` stays at the engine default (1.0, off): the
        // water starts ~2 cells from a wall, so a large, sustained part of the
        // domain reads as near-wall, not only during brief contacts, and at 1000x
        // the scene freezes or lags severely.
        ..SimConfig::earth(GRID, 0.01, DT)
    };
    // Tait exponent 3, not water's 7 (Cole 1948): under a violent wall impact J drops
    // to ~0.3-0.4, and the acoustic term
    // `c2 = eos_stiffness*eos_power*ratio^(eos_power-1)/rest_density` grows as
    // ratio^6 at 7, becoming the dt-limiting term by two orders of magnitude over the
    // deformation-gradient and gravity terms (see basic_fluids_gpu.rs). Lower Tait
    // exponents (n=1..4) are an established real-time WCSPH trade for this reason
    // (Chorin's artificial compressibility uses n=1).
    //
    // Stiffness derived from the load, not hardcoded. With B=1.0, the hydrostatic load
    // at this column's base
    //     p = rho * g * h = 1000 kg/m^3 * (9.81 * 0.003) m/s^2 * 0.468 m = 13.77 Pa
    // solved through the Tait EOS for its equilibrium compression,
    //     p = B((rho/rho0)^gamma - 1),  gamma = 3
    //     B = 1.0  ->  r^3 = 14.77 -> r = 2.45 -> J = 0.408   (crushed)
    //     B = 104  ->  r^3 = 1.132 -> r = 1.04 -> J = 0.96
    // puts J below the material's [0.5, 2.0] clamp floor, pinning every base particle
    // at J=0.5; water under this load compresses by well under 1%.
    //
    // `c_ref = 10 * v_max`, the weakly compressible rule limiting density variation
    // to ~1% (Monaghan 1994; Becker & Teschner 2007 WCSPH), with v_max from
    // Torricelli for this column, sized with a derated gravity (0.3) for the acoustic
    // sizing. `basic_fluids_gpu.rs` now sizes from its scene's full gravity instead,
    // which removes a ~0.7 s acoustic "breathing" of the pool (see its doc).
    const WATER_EOS_POWER: f32 = 3.0;
    const COLUMN_HEIGHT_CELLS: f32 = 52.0 * SPACING;
    const DERATED_GRAVITY_FOR_ACOUSTIC_SIZING: f32 = 0.3;
    let v_max_grid = (2.0 * DERATED_GRAVITY_FOR_ACOUSTIC_SIZING * COLUMN_HEIGHT_CELLS).sqrt();
    let c_ref_m_s = 10.0 * v_max_grid * config.dx_meters;
    let water_tait_b_pa = 1000.0 * c_ref_m_s * c_ref_m_s / WATER_EOS_POWER;
    // Viscosity converted to grid units (`SimConfig::visc_from_si`,
    // `eta_SI/(rho*dx^2)`), the family `pressure_floor` below uses
    // (`stress_from_si`): `fluid.rs`'s stress law (`stress += eff_viscosity *
    // strain_dev`) needs grid units, and mixing raw and density-normalized
    // conventions in one stress tensor is wrong (see
    // `q_factor_elastic_viscosity_pa_s` for a ~917x instance of that mistake). The
    // raw 1.0e-3 would be ~10x too weak here (grid value 0.01).
    const WATER_DYNAMIC_VISCOSITY_PA_S: f32 = 1.0e-3;
    const WATER_RHO_SI_KG_M3_FOR_VISC: f32 = 1000.0;
    let water_dynamic_viscosity =
        config.visc_from_si(WATER_DYNAMIC_VISCOSITY_PA_S, WATER_RHO_SI_KG_M3_FOR_VISC);
    let mut water = NewtonianFluidMaterial::new(
        0.1,
        water_dynamic_viscosity,
        water_tait_b_pa,
        WATER_EOS_POWER,
    );
    // Cavitation floor converted to grid units, as in the GPU twin. The
    // constructor's `pressure_floor` default (-0.1) is a bare grid-unit constant;
    // water's practical cavitation onset (dissolved-gas nucleation) is ~-100,000 Pa
    // gauge, which through `stress_from_si` lands far below the EOS scale, so water
    // essentially never cavitates from ordinary splashing. The unit gap does not
    // depend on the backend.
    const REAL_CAVITATION_PRESSURE_PA: f32 = -100_000.0;
    const WATER_RHO_SI_KG_M3: f32 = 1000.0;
    water.pressure_floor = config.stress_from_si(REAL_CAVITATION_PRESSURE_PA, WATER_RHO_SI_KG_M3);
    let ice = WithLatentHeat::new(NeoHookeanMaterial::new(4.0, 8.0), ICE_LATENT_HEAT);
    let thermal = ThermalDiffusion::new(
        ThermalConfig {
            conductivity: 0.6,
            heat_capacity: 4182.0,
            density: 1000.0, // kg/m^3, real water -- see ThermalConfig::density's own doc
            ambient: WARM_AMBIENT,
            // Must match the sim's real dx_meters -- ThermalConfig::grid_cell_size
            // requires this, else alpha_grid() is mis-scaled by orders of
            // magnitude.
            grid_cell_size: config.dx_meters,
            ..Default::default()
        },
        config.grid_res,
    );
    // Choose m=rho0*spacing² so the particles' conserved reference volumes
    // fill the intended region.  The strict fluid initializer then sets
    // V0=m/rho0 and rho=rho0; it does not use a kernel-density estimate as
    // thermodynamic state.
    // SPACING (top-level const, see its doc) = 0.9, NOT the old 0.6 --
    // measured 45fps-debug-minimum fix: fewer, larger particles is a
    // disclosed RESOLUTION tradeoff, not a physics-accuracy one --
    // material constants below are untouched.
    const WATER_MASS: f32 = 0.1 * SPACING * SPACING;
    let spawn_water = SpawnRegion {
        spacing: SPACING,
        box_size: IVec2::new(14, 52),
        // x=20, not the old 11 -- matches basic_fluids_gpu.rs's own fix (see
        // that file's doc): at x=11 the column's left edge sat only 2 cells
        // past the near-wall threshold, permanently close to a wall-
        // contact regime rather than only during interaction.
        box_center: Vec2::new(20.0, 30.0),
        material_id: MAT_WATER,
        initial_velocity_scale: 0.0,
        mass_override: Some(WATER_MASS),
        ..SpawnRegion::for_sim(&config)
    };
    let mut solver = Simulation::new(config, spawn_water)
        .with_default_material(Box::new(water))
        .with_material(MAT_ICE, Box::new(ice))
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)))
        .with_thermal(thermal)
        .with_phase_rule(|p| {
            if p.material_id == MAT_WATER && p.temperature < FREEZING_POINT {
                Some(MAT_ICE)
            } else {
                None
            }
        });
    for t in solver.particles_mut().temperature.iter_mut() {
        *t = WARM_AMBIENT;
    }
    solver
}

struct State {
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    device: wgpu::Device,
    queue: wgpu::Queue,
    sim: Simulation,
    renderer: Renderer,
    egui_ctx: egui::Context,
    egui_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
    cursor_pos: [f32; 2],
    last_cursor_grid: Vec2,
    lmb: bool,
    rmb: bool,
    digging: bool,
    push_strength: f32,
    dig_strength: f32,
    real_gravity: Vec2,
    gravity_fraction: f32,
    cold: bool,
    frame: u64,
    /// `EMERGE_SCRIPT_LOG`: a scripted run, read from its log (see
    /// `gui_common/scripted.rs`), and where its hand is this frame.
    script: Option<scripted::Script>,
    scripted_at: Option<Vec2>,
    fps_timer: std::time::Instant,
    fps_frames: u64,
    last_fps: f32,
    // Worst single `Simulation::step()` call within the fps averaging window.
    // `last_fps` is a 1-second average, which hides a short spike felt as a
    // stutter during push/pull; this makes it visible and correlatable with what
    // is being done (pushing near a wall, digging). Headless, a scripted push
    // neither changed sim_time_dropped nor raised the step cost.
    worst_step_ms_this_window: f32,
    last_worst_step_ms: f32,
    fps_log_count: u32,
    // Steps `sim.step()` off elapsed time (`FixedStepController`,
    // `simulation_speed: 1.0` = real time), not once per render frame: at ~48 fps
    // with DT = 0.1 s, one step per frame advances 100 ms of simulated time every
    // ~21 ms, playing the scene ~4.8x too fast. Physics now steps ~10 Hz rather than
    // ~48 Hz, so the per-frame cost at a given substep cap is ~4.8x lower than when
    // that cap was last tuned.
    stepper: FixedStepController,
    // Render interpolation ("Fix Your Timestep", Gaffer 2004): every particle's
    // position from before the latest batch of physics steps, so `render_scene` can
    // blend it with the current position by `stepper.interpolation_alpha()`. Render
    // only: `self.sim`'s particles are swapped out and restored around one
    // `Renderer::render` call. Without it, motion speeds up and slows down whenever
    // the per-step cost varies, since the renderer draws the last completed step.
    prev_x: Vec<Vec2>,
    last_instant: std::time::Instant,
    /// Which render path draws the frame, cycled with G, in the order of
    /// `basic_fluids_gpu.rs`.
    render_mode: RenderMode,
    /// GPU buffers the grid-volume and surface modes read (the CPU solver
    /// keeps none), rebuilt on the frames those modes are shown.
    render_bridge: CpuRenderBridge,
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
        let real_gravity = sim.config().gravity;
        // A scripted run (see `gui_common/scripted.rs`): the dam breaks for
        // four seconds, then the hand pushes into the pile, pushes on the
        // spreading front and pulls, two seconds each with two to recover.
        let script = scripted::Script::from_env(
            vec![
                scripted::Press {
                    at: Vec2::new(20.0, 10.0),
                    from: 40,
                    to: 60,
                    pull: false,
                },
                scripted::Press {
                    at: Vec2::new(40.0, 8.0),
                    from: 80,
                    to: 100,
                    pull: false,
                },
                scripted::Press {
                    at: Vec2::new(30.0, 8.0),
                    from: 120,
                    to: 140,
                    pull: true,
                },
            ],
            160,
            (Vec2::ZERO, Vec2::splat(GRID as f32)),
        );
        let prev_x = sim.particles().x.clone();
        let mut renderer = Renderer::new(&device, sim.particles().len(), fmt);
        renderer.set_camera(
            &queue,
            GRID as u32,
            size.width,
            size.height,
            PARTICLE_RENDER_DIAMETER,
            true,
        );
        renderer.set_color_mode(ColorMode::ByMaterial);
        // The grid-volume and curvature-flow-surface paths threshold on absolute
        // cell mass (`mass_floor = 0.15`), written for scenes whose occupied cells
        // weigh "order 0.5-4". This scene uses water's `rho0 = 1000 kg/m^3 * dx^2 =
        // 0.1` grid units, so a full cell weighs 0.1, below that floor, and both
        // modes would discard every cell. Giving the renderer the scene's full-cell
        // mass makes the thresholds a fraction of a full cell.
        renderer.set_grid_reference_cell_mass(0.1);
        // Surface grid at 4x the physics grid, not the 6x default -- a real
        // sampling-statistics fix for the speckle/"white noise" on the
        // surface, not a quality cut.
        //
        // The reconstruction runs at `grid_res * multiplier`, so at 6x this
        // scene had 384x384 = 147k surface cells for only 2912 particles --
        // ~51 cells per particle. Each cell therefore samples a tiny, noisy
        // subset of the particle distribution, and since the shading normal
        // is a finite difference OF that density field, per-cell density noise
        // becomes per-pixel lighting noise. Coarsening to 4x gives 256x256 =
        // 65k cells (~23 per particle): each cell averages more than twice as
        // many particles, so the field -- and the normals taken from it -- are
        // measurably smoother.
        //
        // Cost scales with the SQUARE of the multiplier, so this is also 0.44x
        // the surface-pass work. Strictly better on both axes at this particle
        // count; raise it again if the particle count rises.
        renderer.set_surface_res_multiplier(4);
        // Splat width left at the default (1.0). `Renderer::set_particle_spacing_cells`
        // derives it from particle spacing (see its doc), but combined with a
        // boundary-truncation factor and this demo's raised `surface_res_multiplier`
        // it opens a gap at the free surface that grows with resolution; not used
        // until that interaction is diagnosed.
        // Optical properties for the volumetric render paths, the values of
        // `basic_fluids_gpu.rs`. Without them every slot keeps its 0.0 default, and
        // `grid-volume`/`surface` shade with no absorption, no subsurface scattering
        // and no Fresnel, so water renders flat grey.
        //
        // The absorption triple is physically meaningful, not a palette pick:
        // water's absorption coefficient rises steeply with wavelength, so red
        // is attenuated ~12x more strongly than blue. Encoding that as
        // sigma_a = [0.85, 0.25, 0.07] (R,G,B) makes transmitted light go blue
        // through depth for the Beer-Lambert reason, instead of being
        // tinted blue by hand.
        renderer.set_optical_params(&queue, MAT_WATER as usize, [0.85, 0.25, 0.07]);
        renderer.set_optical_scattering(&queue, MAT_WATER as usize, 0.03);
        renderer.set_specular_r0(&queue, MAT_WATER as usize, 0.02);
        // Derived (not a per-scene guess): a material's free surface
        // only propagates waves if it behaves like a real fluid --
        // `owns_deformation_volume_state()` IS that property (see
        // `Renderer::set_wave_force_coeff`'s doc), queried from the
        // actual material TYPE (result is parameter-independent, so a cheap
        // throwaway instance is a correct query, not a placeholder
        // value standing in for anything). `0.35` is this engine's own
        // tuned value for a real fluid's free-surface wave amplitude.
        if NewtonianFluidMaterial::low_viscosity(1.0, 1.0).owns_deformation_volume_state() {
            renderer.set_wave_force_coeff(0.35);
        }
        // Ice keeps its own distinct look, so its slot doesn't silently
        // inherit water's.
        // Ice: far less absorbing than liquid water (clear ice transmits
        // deeply) and much glossier -- r0 ~0.05 vs water's 0.02.
        renderer.set_optical_params(&queue, MAT_ICE as usize, [0.30, 0.12, 0.05]);
        renderer.set_optical_scattering(&queue, MAT_ICE as usize, 0.06);
        renderer.set_specular_r0(&queue, MAT_ICE as usize, 0.05);

        // GPU buffers for RenderMode::GridVolume/Surface -- see RenderMode's doc.
        let render_bridge = CpuRenderBridge::new(&device, GRID);

        let egui_ctx = egui::Context::default();
        let egui_state = egui_winit::State::new(
            egui_ctx.clone(),
            egui_ctx.viewport_id(),
            window.as_ref(),
            None,
            None,
            None,
        );
        let egui_renderer = egui_wgpu::Renderer::new(
            &device,
            fmt,
            egui_wgpu::RendererOptions {
                msaa_samples: 1,
                ..Default::default()
            },
        );

        println!(
            "basic_fluids: {} particles  |  LMB push  RMB pull  D toggle dig  G render mode  R reset  Q quit",
            sim.particles().len()
        );
        Self {
            surface,
            surface_config: sc,
            device,
            queue,
            sim,
            prev_x,
            renderer,
            egui_ctx,
            egui_state,
            egui_renderer,
            cursor_pos: [0.0; 2],
            last_cursor_grid: Vec2::ZERO,
            lmb: false,
            rmb: false,
            digging: false,
            push_strength: 5.0,
            dig_strength: 18.0,
            real_gravity,
            // 0.01, matching the already-validated sand/snow checkpoint at this
            // same grid scale -- not re-guessed live.
            gravity_fraction: script.as_ref().map_or(0.003, |s| s.gravity(0.003)),
            cold: false,
            frame: 0,
            script,
            scripted_at: None,
            fps_timer: std::time::Instant::now(),
            fps_frames: 0,
            last_fps: 0.0,
            worst_step_ms_this_window: 0.0,
            last_worst_step_ms: 0.0,
            fps_log_count: 0,
            // Each outer step is expensive enough that the old 60 Hz request
            // saturated this one-step-per-render cap.  At an observed 36 FPS,
            // that silently made playback 3.6x rather than the requested 6x,
            // so motion changed speed whenever rendering did.  The measured
            // 30 Hz request is sustainable on the initial scene; the cap of
            // one still deliberately prevents a slow frame from becoming a
            // burst of catch-up physics.  This changes playback only, never
            // the material law, force, internal CFL, or solver timestep.
            stepper: FixedStepController::new(FixedStepConfig {
                dt: DT,
                simulation_speed: PLAYBACK_STEP_RATE_HZ * DT,
                max_substeps_per_frame: 1,
                max_frame_delta: 1.0 / 15.0,
            }),
            last_instant: std::time::Instant::now(),
            render_mode: RenderMode::Particles,
            render_bridge,
        }
    }

    fn resize(&mut self, w: u32, h: u32) {
        if w == 0 || h == 0 {
            return;
        }
        self.surface_config.width = w;
        self.surface_config.height = h;
        self.surface.configure(&self.device, &self.surface_config);
        self.renderer.set_camera(
            &self.queue,
            GRID as u32,
            w,
            h,
            PARTICLE_RENDER_DIAMETER,
            true,
        );
    }

    fn cursor_grid(&self) -> Vec2 {
        if let Some(at) = self.scripted_at {
            return at;
        }
        let (gx, gy) = self.renderer.screen_to_grid(
            self.cursor_pos[0],
            self.cursor_pos[1],
            self.surface_config.width,
            self.surface_config.height,
        );
        Vec2::new(gx, gy)
    }

    /// Applies the live gravity slider + freeze/thaw ambient/cooling-rate
    /// toggle to the sim config. Split out of `update_and_render` purely for
    /// readability -- no behavior change.
    fn apply_gravity_and_thermal(&mut self) {
        self.sim
            .set_gravity(self.real_gravity * self.gravity_fraction);
        if let Some(cfg) = self.sim.thermal_config_mut() {
            cfg.ambient = if self.cold {
                COLD_AMBIENT
            } else {
                WARM_AMBIENT
            };
            // Convective (Newton) cooling, not bulk conduction -- see
            // FREEZER_COOLING_RATE's doc; bulk conduction alone is too slow
            // to ever visibly freeze in a play session.
            cfg.cooling_rate = if self.cold { FREEZER_COOLING_RATE } else { 0.0 };
        }
    }

    /// Applies the LMB/RMB radial push-pull impulse, and returns this
    /// frame's cursor position plus a digging direction (if actively
    /// digging and the cursor moved) for `step_physics` to apply once per
    /// real physics step below -- see that method's doc for why the
    /// direction is sampled here (render cadence) but applied there
    /// (physics cadence).
    fn apply_interaction_forces(&mut self) -> (Vec2, Option<Vec2>) {
        if self.lmb || self.rmb {
            let mag = if self.lmb {
                self.push_strength
            } else {
                -self.push_strength
            };
            self.sim.apply_radial_impulse(self.cursor_grid(), 5.0, mag);
        }
        let cursor = self.cursor_grid();
        let dig_dir = if self.digging {
            let delta = cursor - self.last_cursor_grid;
            (delta.length_squared() > 1.0e-8).then(|| delta.normalize())
        } else {
            None
        };
        self.last_cursor_grid = cursor;
        (cursor, dig_dir)
    }

    /// Temporary diagnostic: per-frame printing, to check a reported "hold shape
    /// ~0.2 s then sudden collapse" against data.
    fn log_early_frame_diagnostics(&self) {
        if self.frame > 20 {
            return;
        }
        let snap = self.sim.diagnostics_snapshot();
        let water_j = self
            .sim
            .particles()
            .deformation_gradient
            .iter()
            .zip(self.sim.particles().material_id.iter())
            .filter(|&(_, &m)| m == MAT_WATER)
            .map(|(f, _)| f.determinant())
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), j| {
                (lo.min(j), hi.max(j))
            });
        eprintln!(
            "frame={}  max_speed={:.3}  non_finite={}  water_j=[{:.3},{:.3}]  gravity_frac={:.3}",
            self.frame,
            snap.max_particle_speed,
            snap.non_finite_particle_values,
            water_j.0,
            water_j.1,
            self.gravity_fraction,
        );
    }

    /// Advances real simulated time by however many physics steps
    /// `FixedStepController` says real elapsed wall-clock time warrants
    /// (see that field's doc on `State` for the real time-dilation bug
    /// this fixes) -- 0 most render frames at this demo's playback speed,
    /// never more than `max_frame_delta` allows.
    fn step_physics(&mut self, cursor: Vec2, dig_dir: Option<Vec2>) {
        let now = std::time::Instant::now();
        // A scripted run paces by a fixed 1/60 s frame, not the wall clock,
        // so it does not depend on how fast the machine is.
        let frame_delta = if self.script.is_some() {
            1.0 / 60.0
        } else {
            (now - self.last_instant).as_secs_f32()
        };
        self.last_instant = now;
        let steps = self.stepper.steps_for_frame(frame_delta);
        // Snapshot the pre-step positions ONCE per batch (not zero -- most
        // render frames at this demo's playback speed, see `steps_for_
        // frame`'s doc), so `render_scene` can interpolate against them.
        // Deliberately skipped when `steps==0`: the last real snapshot stays
        // valid (nothing moved since it was taken), and re-cloning every
        // render frame regardless would add needless per-frame cost.
        if steps > 0 {
            self.prev_x.clone_from(&self.sim.particles().x);
        }
        for _ in 0..steps {
            if let Some(dir) = dig_dir {
                let particles = self.sim.particles_mut();
                for i in 0..particles.len() {
                    if (particles.x[i] - cursor).length() < DIG_RADIUS {
                        particles.v[i] += dir * self.dig_strength * DT;
                    }
                }
            }
            let step_start = std::time::Instant::now();
            self.sim.step();
            let step_ms = step_start.elapsed().as_secs_f32() * 1000.0;
            self.worst_step_ms_this_window = self.worst_step_ms_this_window.max(step_ms);
            // Low-cost tripwire (silent in normal operation): a slow step prints
            // with its phase timing and substep count, in case a periodic spike
            // recurs. The last one chased (~150 ms) was system noise, absent on a
            // quiet machine.
            if step_ms > 50.0 {
                let snap = self.sim.diagnostics_snapshot();
                let t = snap.timing;
                // Every phase timer `StepTiming` actually has -- the previous
                // line printed only 4 of them, leaving ~69% of a 178ms step
                // unattributed and making the cost impossible to find.
                // `grid_update_us` INCLUDES `pressure_us` (documented subset,
                // not additive); everything else is disjoint, so these should
                // sum to ~`total_us`.
                let accounted = t.p2g_us
                    + t.grid_update_us
                    + t.g2p_us
                    + t.fields_us
                    + t.thermal_us
                    + t.cfl_us
                    + t.spatial_hash_us
                    + t.phase_sleep_us
                    + t.project_us
                    + t.retry_snapshot_us;
                eprintln!(
                    "SPIKE frame={} step={:.1}ms subs={} cfl={:.3} | p2g={} grid_update={} (pressure={}) g2p={} cfl_sel={} project={} spatial_hash={} phase_sleep={} fields={} thermal={} retry_snap={} | accounted={} total={} MISSING={}",
                    self.frame,
                    step_ms,
                    snap.substeps_last_step,
                    snap.cfl_number,
                    t.p2g_us,
                    t.grid_update_us,
                    t.pressure_us,
                    t.g2p_us,
                    t.cfl_us,
                    t.project_us,
                    t.spatial_hash_us,
                    t.phase_sleep_us,
                    t.fields_us,
                    t.thermal_us,
                    t.retry_snapshot_us,
                    accounted,
                    t.total_us,
                    t.total_us.saturating_sub(accounted),
                );
            }
            if let Some(script) = &mut self.script {
                let done = script.record(
                    self.frame,
                    DT,
                    &self.sim,
                    &[(MAT_WATER, "water"), (MAT_ICE, "ice")],
                    &[
                        ("gravity", self.gravity_fraction),
                        ("push", self.push_strength),
                    ],
                );
                if done {
                    std::process::exit(0);
                }
            }
            self.frame += 1;
            self.log_early_frame_diagnostics();
        }
    }

    /// Updates the 1-second-averaged fps/worst-step-ms counters the panel
    /// (and the stderr diagnostic below) display.
    fn update_fps_counters(&mut self) {
        self.fps_frames += 1;
        if self.fps_timer.elapsed().as_secs_f32() >= 1.0 {
            self.last_fps = self.fps_frames as f32 / self.fps_timer.elapsed().as_secs_f32();
            self.fps_timer = std::time::Instant::now();
            self.fps_frames = 0;
            self.last_worst_step_ms = self.worst_step_ms_this_window;
            self.worst_step_ms_this_window = 0.0;
            // Temporary diagnostic: prints the numbers the on-screen panel shows,
            // so fps and perf can be read from stdout without a screenshot. Limited
            // to the first 20 seconds.
            if self.fps_log_count < 20 {
                self.fps_log_count += 1;
                eprintln!(
                    "sec={}  fps={:.1}  worst_step={:.2}ms  render={:?}",
                    self.fps_log_count, self.last_fps, self.last_worst_step_ms, self.render_mode,
                );
            }
        }
    }

    /// Dispatches to whichever of the 3 real render paths `G` last selected
    /// -- see `RenderMode`'s doc for what each one is and why the CPU
    /// solver needs a fresh bridge upload for the latter two.
    fn render_scene(&mut self, view: &wgpu::TextureView) {
        match self.render_mode {
            RenderMode::Particles => {
                // Render interpolation (see `prev_x`): blend `prev_x` with the
                // current position by how far time has advanced past the last
                // completed physics step, so motion stays smooth when the step
                // cadence varies. The blended positions are swapped in for the one
                // `render` call and the simulated positions swapped straight back.
                // `RenderMode::Particles` only: the other two modes build their own
                // grid-density bridge buffers from `self.sim.particles()` and do not
                // interpolate yet.
                let alpha = self.stepper.interpolation_alpha();
                if alpha > 0.0 && self.prev_x.len() == self.sim.particles().len() {
                    let blended: Vec<Vec2> = self
                        .prev_x
                        .iter()
                        .zip(self.sim.particles().x.iter())
                        .map(|(&prev, &now)| prev.lerp(now, alpha))
                        .collect();
                    let live = std::mem::replace(&mut self.sim.particles_mut().x, blended);
                    self.renderer.render(
                        &self.device,
                        &self.queue,
                        self.sim.particles(),
                        view,
                        true,
                    );
                    self.sim.particles_mut().x = live;
                } else {
                    self.renderer.render(
                        &self.device,
                        &self.queue,
                        self.sim.particles(),
                        view,
                        true,
                    );
                }
            }
            RenderMode::GridVolume => {
                self.render_bridge
                    .upload_grid(&self.queue, self.sim.particles(), self.sim.grid());
                self.renderer.render_grid_volume(
                    &self.device,
                    &self.queue,
                    self.render_bridge.grid_volume_source(),
                    view,
                    true,
                );
            }
            RenderMode::Surface => {
                self.render_bridge.upload_particles(
                    &self.device,
                    &self.queue,
                    self.sim.particles(),
                );
                // N-material per-cell coloring (`material_mass_enabled`), not
                // dual-phase: dual-phase handles exactly 2 materials, and ice and
                // water must stay distinct (each with its own `OpticalTable` slot).
                // `surface_material_mass` is built from each particle's own
                // `material_id` with the density splat's quadratic B-spline kernel
                // (finer than the nearest-cell approximation `CpuRenderBridge`
                // uses for `GridVolume`). The trade: one
                // shared density/smoothing field, not two independently smoothed
                // surfaces, so materials can blend slightly where they touch, the
                // reason dual-phase exists.
                self.renderer.render_surface_reconstruction(
                    &self.device,
                    &self.queue,
                    self.render_bridge
                        .surface_source(MAT_WATER, true, self.sim.mean_substep_dt()),
                    view,
                    true,
                );
            }
        }
    }

    fn update_and_render(&mut self, window: &Window) {
        if let Some(script) = &self.script {
            // The hand holds the cursor and the button; the push itself is
            // this demo's own, at its strongest setting.
            let hand = script.hand(self.frame);
            self.lmb = hand.is_some_and(|(_, pull)| !pull);
            self.rmb = hand.is_some_and(|(_, pull)| pull);
            self.scripted_at = hand.map(|(at, _)| at);
            self.push_strength = PUSH_STRENGTH_MAX;
        }
        self.apply_gravity_and_thermal();
        let (cursor, dig_dir) = self.apply_interaction_forces();
        self.step_physics(cursor, dig_dir);
        self.update_fps_counters();

        let output = match self.surface.get_current_texture() {
            Ok(t) => t,
            Err(_) => return,
        };
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        self.render_scene(&view);

        // --- egui panel ---
        let raw_input = self.egui_state.take_egui_input(window);
        let fps = self.last_fps;
        let worst_step_ms = self.last_worst_step_ms;
        let mut push_strength = self.push_strength;
        let mut dig_strength = self.dig_strength;
        let mut gravity_fraction = self.gravity_fraction;
        let mut digging = self.digging;
        let mut cold = self.cold;
        // Live render-solver dials, read back from the renderer so the widgets
        // always show the value actually in use.
        let mut curvature_iters = self.renderer.curvature_iterations();
        let mut surface_mult = self.renderer.surface_res_multiplier();
        let mut splat_width = self.renderer.splat_width_cells();
        let water_n = self
            .sim
            .particles()
            .iter()
            .filter(|p| p.material_id == MAT_WATER)
            .count();
        let ice_n = self
            .sim
            .particles()
            .iter()
            .filter(|p| p.material_id == MAT_ICE)
            .count();
        let mut reset = false;
        let render_mode_label = self.render_mode.label();
        let full_output = self.egui_ctx.run(raw_input, |ctx| {
            egui::Window::new("Fluids")
                .default_pos([10.0, 10.0])
                .default_width(260.0)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.label(format!(
                        "fps={fps:.0}  worst_step={worst_step_ms:.1}ms  render={render_mode_label}"
                    ));
                    ui.label(format!("water={water_n}  ice={ice_n}"));
                    ui.separator();
                    ui.label("Gravity (1.0 = real IRL 9.81 m/s², use --release above ~0.1):");
                    ui.add(egui::Slider::new(&mut gravity_fraction, 0.0..=2.0));
                    ui.separator();
                    ui.label("Push/pull strength:");
                    ui.add(egui::Slider::new(
                        &mut push_strength,
                        0.0..=PUSH_STRENGTH_MAX,
                    ));
                    ui.checkbox(&mut digging, "Digging/stirring active (or press D)");
                    ui.add(egui::Slider::new(&mut dig_strength, 0.0..=40.0).text("Dig strength"));
                    ui.separator();
                    ui.checkbox(&mut cold, "Cold ambient (water freezes below 273K)");
                    ui.separator();
                    ui.label("Renderer (surface/grid-volume modes):");
                    ui.add(
                        egui::Slider::new(&mut curvature_iters, 2..=32)
                            .text("Smoothing passes (higher = rounder)"),
                    );
                    ui.add(
                        egui::Slider::new(&mut surface_mult, 1..=10)
                            .text("Surface detail (cost = square!)"),
                    );
                    ui.add(
                        egui::Slider::new(&mut splat_width, 0.2..=2.0)
                            .text("Splat width, cells (lower = sharper)"),
                    );
                    ui.separator();
                    ui.label("LMB push  RMB pull  D toggle dig  G render mode  R reset  Q quit");
                    if ui.button("Reset").clicked() {
                        reset = true;
                    }
                });
        });
        self.push_strength = push_strength;
        self.dig_strength = dig_strength;
        self.gravity_fraction = gravity_fraction;
        self.digging = digging;
        self.cold = cold;
        // Only call the setters when the value actually moved -- the surface
        // multiplier forces a buffer realloc, so writing it every frame would
        // rebuild the surface buffers continuously.
        if curvature_iters != self.renderer.curvature_iterations() {
            self.renderer.set_curvature_iterations(curvature_iters);
        }
        if surface_mult != self.renderer.surface_res_multiplier() {
            self.renderer.set_surface_res_multiplier(surface_mult);
        }
        if (splat_width - self.renderer.splat_width_cells()).abs() > 1.0e-4 {
            self.renderer.set_splat_width_cells(splat_width);
        }
        if reset {
            let sim = make_sim();
            self.real_gravity = sim.config().gravity;
            self.prev_x = sim.particles().x.clone();
            self.sim = sim;
            self.frame = 0;
            self.stepper.reset();
            self.last_instant = std::time::Instant::now();
        }

        self.egui_state
            .handle_platform_output(window, full_output.platform_output);
        let tris = self
            .egui_ctx
            .tessellate(full_output.shapes, full_output.pixels_per_point);
        let sd = ScreenDescriptor {
            size_in_pixels: [self.surface_config.width, self.surface_config.height],
            pixels_per_point: full_output.pixels_per_point,
        };
        for (id, delta) in &full_output.textures_delta.set {
            self.egui_renderer
                .update_texture(&self.device, &self.queue, *id, delta);
        }
        let cmd = {
            let mut enc = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            self.egui_renderer
                .update_buffers(&self.device, &self.queue, &mut enc, &tris, &sd);
            let mut rp = enc
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    ..Default::default()
                })
                .forget_lifetime();
            self.egui_renderer.render(&mut rp, &tris, &sd);
            drop(rp);
            enc.finish()
        };
        self.queue.submit(std::iter::once(cmd));
        for id in &full_output.textures_delta.free {
            self.egui_renderer.free_texture(id);
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
                    .with_title("emerge -- Fluids (GUI)")
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
            let resp = s.egui_state.on_window_event(w, &event);
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
                    KeyCode::KeyD if pressed => s.digging = !s.digging,
                    KeyCode::KeyG if pressed => s.render_mode = s.render_mode.next(),
                    KeyCode::KeyR if pressed => {
                        let sim = make_sim();
                        s.real_gravity = sim.config().gravity;
                        s.prev_x = sim.particles().x.clone();
                        s.sim = sim;
                        s.frame = 0;
                        // Elapsed time since the last render frame (e.g. an idle
                        // window) must not be replayed as a burst of catch-up
                        // physics steps, as basic_fluids_gpu.rs does on reset.
                        s.stepper.reset();
                        s.last_instant = std::time::Instant::now();
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

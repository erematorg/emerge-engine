/// Two snowballs flying at each other — the canonical MPM snow demo.
///
/// IRL: thrown snowballs STICK and splat plastically at normal speeds. They do not
/// bounce like billiard balls — that would be elastic/rubber, not snow.
/// This is correct MPM behavior. Raise theta_c toward 0.5 to see more elastic bounce.
///
/// Ball A (blue): moving right from left side.
/// Ball B (white): moving left from right side.
/// 30-cell gap → ~1.5 real seconds of visible approach before impact.
use bevy::prelude::*;
use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass, egui};
use emerge::diagnostics::{
    MpmHealthThresholds, MpmReportingPolicy, MpmReportingState, MpmSnapshot, evaluate_mpm_health,
    update_mpm_reporting,
};
use emerge::runtime::fixed_step::{FixedStepConfig, FixedStepController};
use emerge::solver::{MpmSolver, SlipBoundary, SnowMaterial, SolverConfig, SpawnConfig};
use glam::{IVec2, Mat2, Vec2};

const BALL_A_ID: u32 = 0;
const BALL_B_ID: u32 = 1;

const GRID_RES: usize = 64;
// MPM2D reference: R_BALL = min(X,Y)*0.33 in a 200×100 grid → 33 cells.
// Scaled to our 64×64: 33/100*64 ≈ 21. We use 10 to fit two balls with room.
const BALL_RADIUS: f32 = 10.0;
// MPM2D ball layout proportions: A at (30%, 55%), B at (70%, 45%) of grid
// → In 64×64: A=(19, 35), B=(45, 29). Offset vertically for off-center collision.
const BALL_A_CENTER: Vec2 = Vec2::new(14.0, 36.0);
const BALL_B_CENTER: Vec2 = Vec2::new(50.0, 28.0);

// Cursor interaction (ported from bevy-mpm basic_fluids_gpu.rs)
// Left-click = push away, right-click = pull toward
const CURSOR_RADIUS: f32 = 6.0;    // cells — radius of influence
const CURSOR_STRENGTH: f32 = 50.0; // cells/s² impulse (scaled by dt each frame)
const CURSOR_MAX_VEL: f32 = 30.0;  // cells/s — cap after cursor to prevent CFL spiral

fn main() {
    let settings = BasicSnowSettings::default();
    let runtime_defaults = settings.runtime_defaults;

    App::new()
        .insert_resource(ClearColor(Color::srgb(0.05, 0.07, 0.10)))
        .insert_resource(settings)
        .insert_resource(runtime_defaults)
        .insert_resource(SnowRuntimeDefaults(runtime_defaults))
        .insert_resource(ResetSimulationRequested::default())
        .insert_resource(settings.runtime_ranges)
        .insert_resource(DiagnosticsRuntime::new(
            settings.diagnostics_thresholds,
            settings.diagnostics_report_interval_secs,
            settings.diagnostics_healthy_heartbeat_secs,
            settings.diagnostics_log_healthy,
        ))
        .insert_resource(Simulation::new(settings, runtime_defaults))
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "MLS-MPM Snow — Snowball Collision".to_string(),
                resolution: settings.window_resolution.into(),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(EguiPlugin::default())
        .add_systems(Startup, setup_scene)
        .add_systems(
            Update,
            (
                request_reset_on_keypress,
                handle_reset_request,
                apply_runtime_parameters,
                apply_cursor_force,
                step_simulation,
                report_diagnostics,
                sync_particle_visuals,
            )
                .chain(),
        )
        .add_systems(EguiPrimaryContextPass, runtime_controls_ui)
        .run();
}

#[derive(Resource, Clone, Copy, Debug)]
struct BasicSnowSettings {
    window_resolution: (u32, u32),
    pixels_per_cell: f32,
    particle_diameter: f32,
    snap_to_pixels: bool,
    max_substeps_per_frame: usize,
    max_frame_delta: f32,
    runtime_ranges: SnowUiRanges,
    solver_config: SolverConfig,
    spawn_config: SpawnConfig,
    runtime_defaults: SnowRuntimeParameters,
    diagnostics_report_interval_secs: f32,
    diagnostics_healthy_heartbeat_secs: f32,
    diagnostics_thresholds: MpmHealthThresholds,
    diagnostics_log_healthy: bool,
}

impl Default for BasicSnowSettings {
    fn default() -> Self {
        let solver_config = SolverConfig {
            grid_res: GRID_RES,
            grid_cell_size: 1.0,
            dt: 0.1,
            adaptive_timestep: true,
            cfl_include_affine_speed: true,
            cfl_coefficient: 0.9,
            material_cfl_coefficient: 0.5,
            viscous_timestep_coefficient: 0.5,
            min_dt: 1.0e-3,
            project_invalid_state: true,
            projection_min_density: 1.0e-6,
            projection_min_volume: 1.0e-6,
            projection_min_deformation_j: 1.0e-6,
            // Real gravity from MPM2D reference: G = (0, -9.81) in grid-cell units
            gravity: -9.81,
            boundary_thickness: 2,
            default_initial_volume: 1.0,
            recompute_density_each_step: false,
            particle_mass: 1.0,
            mls_d_inverse: 4.0,
            max_substeps_per_step: 64, // snow at lambda=38889: c_P≈197 → ~50 substeps per step
        };
        // Spawn box covering full usable grid — filtered to circles in Simulation::new
        let spawn_config = SpawnConfig {
            spacing: 0.5,
            box_size: IVec2::new(58, 58),
            box_center: Vec2::splat(32.0),
            initial_deformation_gradient: Mat2::IDENTITY,
            precompute_initial_volumes: false,
            initial_velocity_offset: Vec2::ZERO,
            initial_velocity_scale: 0.0,
            rng_seed: 7,
        };

        let runtime_defaults = SnowRuntimeParameters {
            // Each step() now loops sub-steps until config.dt is consumed → simulation_speed is honest.
            // simulation_speed = hz * dt = 5 * 0.1 = 0.5 physics/real-s
            // MPM2D launch speed = 40 cells/s (closing = 80). Gap ≈ 16 cells → 0.2 physics s to impact.
            // At sim_speed=0.5: user sees ~0.4 real seconds of approach, then violent scatter.
            target_solver_hz: 5.0,
            gravity: solver_config.gravity,
            // Exact MPM2D constants.h snow params — E=1.4e5, nu=0.2:
            // lambda = E*nu / ((1+nu)*(1-2*nu)) = 140000*0.2 / (1.2*0.6) = 38889
            // mu     = E / (2*(1+nu))            = 140000 / 2.4         = 58333
            // These are what produces the canonical "two snowballs smashing" behavior.
            elastic_lambda: 38889.0,
            elastic_mu: 58333.0,
            hardening_xi: 10.0,
            // MPM2D: THT_C=2.0e-2, THT_S=6.0e-3
            compression_limit: 0.02,
            stretch_limit: 0.006,
            min_plastic_jacobian: 0.6,
            max_plastic_jacobian: 1.05,
            // MPM2D launch speed = 40 cells/s
            launch_speed: 40.0,
        };

        let runtime_ranges = SnowUiRanges {
            target_solver_hz: RangeLimit::new(1.0, 60.0),
            gravity: RangeLimit::new(-20.0, 0.0),
            elastic_lambda: RangeLimit::new(100.0, 100000.0),
            elastic_mu: RangeLimit::new(100.0, 200000.0),
            hardening_xi: RangeLimit::new(0.0, 20.0),
            compression_limit: RangeLimit::new(0.001, 0.5),
            stretch_limit: RangeLimit::new(0.001, 0.1),
            launch_speed: RangeLimit::new(1.0, 80.0),
        };

        let pixels_per_cell = 14.0; // 64 cells × 14px = 896px ≈ window size
        let particle_diameter = pixels_per_cell * 0.5 * 1.6; // 60% overlap at spacing=0.5 → solid look

        Self {
            window_resolution: (900, 900),
            pixels_per_cell,
            particle_diameter,
            snap_to_pixels: true,
            max_substeps_per_frame: 64,
            max_frame_delta: 1.0 / 15.0,
            runtime_ranges,
            solver_config,
            spawn_config,
            runtime_defaults,
            diagnostics_report_interval_secs: 1.0,
            diagnostics_healthy_heartbeat_secs: 5.0,
            diagnostics_thresholds: MpmHealthThresholds::for_spacing(spawn_config.spacing),
            diagnostics_log_healthy: false,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct RangeLimit {
    min: f32,
    max: f32,
}

impl RangeLimit {
    const fn new(min: f32, max: f32) -> Self {
        Self { min, max }
    }
}

#[derive(Resource, Clone, Copy, Debug)]
struct SnowUiRanges {
    target_solver_hz: RangeLimit,
    gravity: RangeLimit,
    elastic_lambda: RangeLimit,
    elastic_mu: RangeLimit,
    hardening_xi: RangeLimit,
    compression_limit: RangeLimit,
    stretch_limit: RangeLimit,
    launch_speed: RangeLimit,
}

#[derive(Resource, Clone, Copy, Debug, PartialEq)]
struct SnowRuntimeParameters {
    target_solver_hz: f32,
    gravity: f32,
    elastic_lambda: f32,
    elastic_mu: f32,
    hardening_xi: f32,
    compression_limit: f32,
    stretch_limit: f32,
    min_plastic_jacobian: f32,
    max_plastic_jacobian: f32,
    launch_speed: f32,
}

#[derive(Resource, Clone, Copy, Debug)]
struct SnowRuntimeDefaults(SnowRuntimeParameters);

#[derive(Resource, Default)]
struct ResetSimulationRequested(bool);

#[derive(Resource)]
struct Simulation {
    solver: MpmSolver,
    stepper: FixedStepController,
}

impl Simulation {
    fn new(settings: BasicSnowSettings, params: SnowRuntimeParameters) -> Self {
        let config = settings.solver_config;
        let spawn = settings.spawn_config;
        let snow = snow_material(&params);

        let mut solver = MpmSolver::new(config, spawn)
            .with_material(BALL_A_ID, Box::new(snow))
            .with_material(BALL_B_ID, Box::new(snow))
            .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));

        // Keep only particles inside ball A or ball B, then assign launch velocities
        {
            let speed = params.launch_speed;
            let particles = solver.particles_mut();
            particles.retain(|p| {
                (p.x - BALL_A_CENTER).length() <= BALL_RADIUS
                    || (p.x - BALL_B_CENTER).length() <= BALL_RADIUS
            });
            for p in particles.iter_mut() {
                if (p.x - BALL_A_CENTER).length() <= BALL_RADIUS {
                    p.material_id = BALL_A_ID;
                    p.v = Vec2::new(speed, 0.0);
                } else {
                    p.material_id = BALL_B_ID;
                    p.v = Vec2::new(-speed, 0.0);
                }
            }
        }
        solver.recompute_initial_volumes();

        let stepper = FixedStepController::new(FixedStepConfig {
            dt: config.dt,
            simulation_speed: params.target_solver_hz * config.dt,
            max_substeps_per_frame: settings.max_substeps_per_frame,
            max_frame_delta: settings.max_frame_delta,
        });

        Self { solver, stepper }
    }
}

fn snow_material(params: &SnowRuntimeParameters) -> SnowMaterial {
    SnowMaterial::new(
        params.elastic_lambda,
        params.elastic_mu,
        params.hardening_xi,
        params.compression_limit,
        params.stretch_limit,
        params.min_plastic_jacobian,
        params.max_plastic_jacobian,
    )
}

#[derive(Resource, Clone, Debug)]
struct DiagnosticsRuntime {
    policy: MpmReportingPolicy,
    state: MpmReportingState,
}

impl DiagnosticsRuntime {
    fn new(
        thresholds: MpmHealthThresholds,
        report_interval_secs: f32,
        healthy_heartbeat_secs: f32,
        log_healthy: bool,
    ) -> Self {
        Self {
            policy: MpmReportingPolicy {
                thresholds,
                report_interval_secs,
                healthy_heartbeat_secs,
                issue_cooldown_secs: 3.0,
                log_healthy,
            },
            state: MpmReportingState::default(),
        }
    }
}

#[derive(Component)]
struct ParticleVisual {
    index: usize,
}

fn setup_scene(
    mut commands: Commands,
    sim: Res<Simulation>,
    settings: Res<BasicSnowSettings>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    commands.spawn(Camera2d);

    let mat_a = materials.add(Color::srgb(0.40, 0.70, 1.00)); // ball A — vivid blue
    let mat_b = materials.add(Color::srgb(0.95, 0.97, 1.00)); // ball B — near-white
    let mesh = meshes.add(Rectangle::from_length(1.0));
    let d = settings.particle_diameter;

    for (index, p) in sim.solver.particles().iter().enumerate() {
        let color = if p.material_id == BALL_A_ID { mat_a.clone() } else { mat_b.clone() };
        commands.spawn((
            Mesh2d(mesh.clone()),
            MeshMaterial2d(color),
            Transform {
                translation: to_world(p.x, GRID_RES, settings.pixels_per_cell, settings.snap_to_pixels),
                scale: Vec3::new(d, d, 1.0),
                ..default()
            },
            ParticleVisual { index },
        ));
    }
}

fn request_reset_on_keypress(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut reset_requested: ResMut<ResetSimulationRequested>,
) {
    if keyboard.just_pressed(KeyCode::KeyR) {
        reset_requested.0 = true;
    }
}

fn runtime_controls_ui(
    mut contexts: EguiContexts,
    ranges: Res<SnowUiRanges>,
    mut params: ResMut<SnowRuntimeParameters>,
    mut reset_requested: ResMut<ResetSimulationRequested>,
    sim: Res<Simulation>,
    diagnostics_runtime: Res<DiagnosticsRuntime>,
    time: Res<Time>,
    mut initialized: Local<bool>,
    mut diag_elapsed: Local<f32>,
    mut cached_diag: Local<Option<(MpmSnapshot, bool)>>,
) {
    if !*initialized {
        *initialized = true;
        return;
    }
    let Ok(ctx) = contexts.ctx_mut() else { return };

    *diag_elapsed += time.delta_secs();
    if cached_diag.is_none() || *diag_elapsed >= 0.25 {
        *diag_elapsed = 0.0;
        let snapshot = sim.solver.diagnostics_snapshot();
        let healthy =
            evaluate_mpm_health(&snapshot, &diagnostics_runtime.policy.thresholds).healthy();
        *cached_diag = Some((snapshot, healthy));
    }

    egui::Window::new("Snow Collision")
        .default_pos([10.0, 10.0])
        .default_width(310.0)
        .resizable(false)
        .show(ctx, |ui| {
            if let Some((snapshot, healthy)) = *cached_diag {
                let fps = if time.delta_secs() > 0.0 { 1.0 / time.delta_secs() } else { 0.0 };
                let color = if healthy { egui::Color32::LIGHT_GREEN } else { egui::Color32::LIGHT_RED };
                ui.colored_label(color, format!("fps={:.0}  substeps={}  health={}", fps, snapshot.substeps_last_step, healthy));
            } else {
                ui.label("fps=…  substeps=…");
            }
            ui.separator();

            ui.label("Simulation");
            ui.add(egui::Slider::new(&mut params.target_solver_hz, ranges.target_solver_hz.min..=ranges.target_solver_hz.max).text("solver_hz"));
            ui.add(egui::Slider::new(&mut params.gravity, ranges.gravity.min..=ranges.gravity.max).text("gravity"));
            ui.add(egui::Slider::new(&mut params.launch_speed, ranges.launch_speed.min..=ranges.launch_speed.max).text("launch speed (cells/s)"));

            ui.separator();
            ui.label("Elastic Stiffness");
            ui.add(egui::Slider::new(&mut params.elastic_lambda, ranges.elastic_lambda.min..=ranges.elastic_lambda.max).text("lambda"));
            ui.add(egui::Slider::new(&mut params.elastic_mu, ranges.elastic_mu.min..=ranges.elastic_mu.max).text("mu"));

            ui.separator();
            ui.label("Snow Plasticity (Stomakhin 2013)");
            ui.add(egui::Slider::new(&mut params.hardening_xi, ranges.hardening_xi.min..=ranges.hardening_xi.max).text("xi (hardening)"));
            ui.add(egui::Slider::new(&mut params.compression_limit, ranges.compression_limit.min..=ranges.compression_limit.max).logarithmic(true).text("theta_c (compress limit)"));
            ui.add(egui::Slider::new(&mut params.stretch_limit, ranges.stretch_limit.min..=ranges.stretch_limit.max).logarithmic(true).text("theta_s (stretch limit)"));

            ui.separator();
            if ui.button("Reset + Relaunch (R)").clicked() {
                reset_requested.0 = true;
            }

            ui.separator();
            ui.label("Diagnostics");
            if let Some((snapshot, _healthy)) = *cached_diag {
                ui.label(format!("cfl={:.3}  dt={:.4}/{:.4}", snapshot.cfl_number, snapshot.effective_dt, snapshot.configured_dt));
                ui.label(format!("mass_err={:.1e}  cells={}", snapshot.relative_mass_error, snapshot.active_grid_cells));
                ui.separator();
                ui.label("Plasticity (drops when snow compresses)");
                let jp_color = if snapshot.avg_plastic_jacobian < 0.95 {
                    egui::Color32::from_rgb(100, 180, 255)
                } else {
                    egui::Color32::GRAY
                };
                ui.colored_label(jp_color, format!("jp_avg={:.3}  jp_min={:.3}", snapshot.avg_plastic_jacobian, snapshot.min_plastic_jacobian));
                ui.label(format!("h_avg={:.3}  (1.0=no hardening)", snapshot.avg_elastic_hardening));
            }
        });
}

fn handle_reset_request(
    settings: Res<BasicSnowSettings>,
    defaults: Res<SnowRuntimeDefaults>,
    mut reset_requested: ResMut<ResetSimulationRequested>,
    mut params: ResMut<SnowRuntimeParameters>,
    mut sim: ResMut<Simulation>,
    mut diagnostics_runtime: ResMut<DiagnosticsRuntime>,
) {
    if !reset_requested.0 {
        return;
    }
    let reset_params = defaults.0;
    *params = reset_params;
    *sim = Simulation::new(*settings, reset_params);
    diagnostics_runtime.state = MpmReportingState::default();
    reset_requested.0 = false;
}

fn apply_runtime_parameters(
    params: Res<SnowRuntimeParameters>,
    mut sim: ResMut<Simulation>,
    time: Res<Time>,
    mut last_log_time: Local<f32>,
    mut last_applied: Local<Option<SnowRuntimeParameters>>,
) {
    let current = *params;
    if last_applied.is_some_and(|prev| prev == current) {
        return;
    }

    let configured_dt = sim.solver.config().dt;
    sim.solver.set_gravity(current.gravity);
    let snow = Box::new(snow_material(&current));
    sim.solver.set_material(BALL_A_ID, snow.clone());
    sim.solver.set_material(BALL_B_ID, snow);
    sim.stepper.set_simulation_speed(current.target_solver_hz * configured_dt);

    let now = time.elapsed_secs();
    if now - *last_log_time >= 0.75 || last_applied.is_none() {
        println!(
            "[runtime] hz={:.0} g={:.3} lambda={:.1} mu={:.1} xi={:.1} theta_c={:.4} theta_s={:.4} speed={:.1}",
            current.target_solver_hz, current.gravity, current.elastic_lambda, current.elastic_mu,
            current.hardening_xi, current.compression_limit, current.stretch_limit, current.launch_speed,
        );
        *last_log_time = now;
    }

    *last_applied = Some(current);
}

fn step_simulation(time: Res<Time>, mut sim: ResMut<Simulation>) {
    let steps = sim.stepper.steps_for_frame(time.delta_secs());
    sim.solver.step_n(steps);
}

fn report_diagnostics(
    time: Res<Time>,
    sim: Res<Simulation>,
    mut diagnostics_runtime: ResMut<DiagnosticsRuntime>,
) {
    let policy = diagnostics_runtime.policy.clone();
    let snapshot = sim.solver.diagnostics_snapshot();
    if let Some(report) = update_mpm_reporting(&mut diagnostics_runtime.state, time.delta_secs(), &snapshot, &policy) {
        if let Some(event_line) = report.event_line {
            println!("{}", event_line);
        }
        println!("{}", report.report_line);
    }
}

fn sync_particle_visuals(
    sim: Res<Simulation>,
    settings: Res<BasicSnowSettings>,
    mut query: Query<(&ParticleVisual, &mut Transform)>,
) {
    for (visual, mut transform) in &mut query {
        let p = &sim.solver.particles()[visual.index];
        transform.translation = to_world(p.x, GRID_RES, settings.pixels_per_cell, settings.snap_to_pixels);
    }
}

fn apply_cursor_force(
    windows: Query<&Window>,
    camera_query: Query<(&Camera, &GlobalTransform)>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    settings: Res<BasicSnowSettings>,
    mut sim: ResMut<Simulation>,
    time: Res<Time>,
) {
    if !mouse_buttons.pressed(MouseButton::Left) && !mouse_buttons.pressed(MouseButton::Right) {
        return;
    }
    let Ok(window) = windows.single() else { return };
    let Some(cursor_pos) = window.cursor_position() else { return };
    let Ok((camera, cam_transform)) = camera_query.single() else { return };
    let Ok(world_pos) = camera.viewport_to_world_2d(cam_transform, cursor_pos) else { return };

    // Invert to_world: world_pos → grid coordinates
    let half = GRID_RES as f32 * 0.5;
    let grid_pos = world_pos / settings.pixels_per_cell + Vec2::splat(half);

    // Right-click pulls in, left-click pushes out
    let sign = if mouse_buttons.pressed(MouseButton::Right) { -1.0 } else { 1.0 };
    let dt = time.delta_secs().min(settings.max_frame_delta);

    for p in sim.solver.particles_mut().iter_mut() {
        let diff = p.x - grid_pos;
        let dist = diff.length();
        if dist < CURSOR_RADIUS && dist > 1e-4 {
            let dir = diff / dist;
            let falloff = 1.0 - (dist / CURSOR_RADIUS);
            p.v += dir * sign * CURSOR_STRENGTH * falloff * dt;
            // Cap velocity so cursor can't spike CFL → spiral of expensive substeps
            let speed = p.v.length();
            if speed > CURSOR_MAX_VEL {
                p.v *= CURSOR_MAX_VEL / speed;
            }
        }
    }
}

fn to_world(grid_pos: Vec2, grid_res: usize, pixels_per_cell: f32, snap_to_pixels: bool) -> Vec3 {
    let half = grid_res as f32 * 0.5;
    let mut centered = (grid_pos - Vec2::splat(half)) * pixels_per_cell;
    if snap_to_pixels {
        centered = centered.round();
    }
    Vec3::new(centered.x, centered.y, 0.0)
}

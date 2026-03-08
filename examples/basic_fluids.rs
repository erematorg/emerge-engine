use bevy::prelude::*;
use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass, egui};
use emerge::diagnostics::{
    MpmHealthThresholds, MpmReportingPolicy, MpmReportingState, MpmSnapshot, evaluate_mpm_health,
    update_mpm_reporting,
};
use emerge::runtime::fixed_step::{FixedStepConfig, FixedStepController};
use emerge::solver::{
    MpmSolver, NewtonianFluidMaterial, PredictiveBoundary, SolverConfig, SpawnConfig,
};

fn main() {
    let settings = BasicFluidsSettings::default();
    let runtime_defaults = settings.runtime_defaults;

    App::new()
        .insert_resource(ClearColor(Color::srgb(0.05, 0.08, 0.11)))
        .insert_resource(settings)
        .insert_resource(runtime_defaults)
        .insert_resource(FluidRuntimeDefaults(runtime_defaults))
        .insert_resource(ResetSimulationRequested::default())
        .insert_resource(settings.runtime_limits)
        .insert_resource(DiagnosticsRuntime::new(
            settings.diagnostics_thresholds,
            settings.diagnostics_report_interval_secs,
            settings.diagnostics_healthy_heartbeat_secs,
            settings.diagnostics_log_healthy,
        ))
        .insert_resource(Simulation::new(settings, runtime_defaults))
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "MLS-MPM Basic Fluids".to_string(),
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
struct BasicFluidsSettings {
    window_resolution: (u32, u32),
    pixels_per_cell: f32,
    particle_diameter: f32,
    snap_to_pixels: bool,
    max_substeps_per_frame: usize,
    max_frame_delta: f32,
    solver_config: SolverConfig,
    spawn_config: SpawnConfig,
    runtime_defaults: FluidRuntimeParameters,
    runtime_limits: RuntimeParameterLimits,
    diagnostics_report_interval_secs: f32,
    diagnostics_healthy_heartbeat_secs: f32,
    diagnostics_thresholds: MpmHealthThresholds,
    diagnostics_log_healthy: bool,
}

impl Default for BasicFluidsSettings {
    fn default() -> Self {
        let solver_config = SolverConfig {
            grid_res: 64,
            grid_cell_size: 1.0,
            dt: 0.1,
            adaptive_timestep: true,
            cfl_include_affine_speed: true,
            cfl_coefficient: 0.9,
            material_cfl_coefficient: 0.5,
            viscous_timestep_coefficient: 0.5,
            min_dt: 0.01,
            project_invalid_state: true,
            projection_min_density: 1.0e-6,
            projection_min_volume: 1.0e-6,
            projection_min_deformation_j: 1.0e-6,
            gravity: -0.3,
            boundary_thickness: 2,
            default_initial_volume: 1.0,
            recompute_density_each_step: true,
            particle_mass: 1.0,
            mls_d_inverse: 4.0,
            max_substeps_per_step: 8,
        };
        let spawn_config = SpawnConfig {
            spacing: 0.5,
            box_size: IVec2::new(32, 32),
            box_center: Vec2::splat(32.0),
            initial_deformation_gradient: Mat2::IDENTITY,
            precompute_initial_volumes: false,
            initial_velocity_offset: Vec2::ZERO,
            initial_velocity_scale: 0.0,
            rng_seed: 1,
        };

        let runtime_defaults = FluidRuntimeParameters {
            target_solver_hz: 30.0,
            gravity: solver_config.gravity,
            rest_density: 4.0,
            dynamic_viscosity: 0.1,
            eos_stiffness: 10.0,
            eos_power: 4.0,
            pressure_floor: -0.1,
            predictive_wall_min: 3.0,
            velocity_damping: 1.0,
            affine_damping: 1.0,
        };

        let runtime_limits = RuntimeParameterLimits {
            target_solver_hz: ScalarLimit::new(5.0, 60.0),
            gravity: ScalarLimit::new(-2.0, 2.0),
            rest_density: ScalarLimit::new(1.0, 12.0),
            dynamic_viscosity: ScalarLimit::new(0.0, 20.0),
            eos_stiffness: ScalarLimit::new(1.0, 100.0),
            eos_power: ScalarLimit::new(1.0, 8.0),
            pressure_floor: ScalarLimit::new(-1.0, 0.0),
            predictive_wall_min: ScalarLimit::new(2.0, 6.0),
            velocity_damping: ScalarLimit::new(0.98, 1.0),
            affine_damping: ScalarLimit::new(0.98, 1.0),
        };

        let diagnostics_thresholds = MpmHealthThresholds::for_spacing(spawn_config.spacing);

        Self {
            window_resolution: (900, 900),
            pixels_per_cell: 10.0,
            particle_diameter: 1.0,
            snap_to_pixels: true,
            max_substeps_per_frame: 3,
            max_frame_delta: 1.0 / 15.0,
            solver_config,
            spawn_config,
            runtime_defaults,
            runtime_limits,
            diagnostics_report_interval_secs: 1.0,
            diagnostics_healthy_heartbeat_secs: 5.0,
            diagnostics_thresholds,
            diagnostics_log_healthy: false,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct ScalarLimit {
    min: f32,
    max: f32,
}

impl ScalarLimit {
    const fn new(min: f32, max: f32) -> Self {
        Self { min, max }
    }
}

#[derive(Resource, Clone, Copy, Debug)]
struct RuntimeParameterLimits {
    target_solver_hz: ScalarLimit,
    gravity: ScalarLimit,
    rest_density: ScalarLimit,
    dynamic_viscosity: ScalarLimit,
    eos_stiffness: ScalarLimit,
    eos_power: ScalarLimit,
    pressure_floor: ScalarLimit,
    predictive_wall_min: ScalarLimit,
    velocity_damping: ScalarLimit,
    affine_damping: ScalarLimit,
}

#[derive(Resource, Clone, Copy, Debug, PartialEq)]
struct FluidRuntimeParameters {
    target_solver_hz: f32,
    gravity: f32,
    rest_density: f32,
    dynamic_viscosity: f32,
    eos_stiffness: f32,
    eos_power: f32,
    pressure_floor: f32,
    predictive_wall_min: f32,
    velocity_damping: f32,
    affine_damping: f32,
}

#[derive(Resource, Clone, Copy, Debug)]
struct FluidRuntimeDefaults(FluidRuntimeParameters);

#[derive(Resource, Default)]
struct ResetSimulationRequested(bool);

#[derive(Resource)]
struct Simulation {
    solver: MpmSolver,
    stepper: FixedStepController,
}

impl Simulation {
    fn new(settings: BasicFluidsSettings, params: FluidRuntimeParameters) -> Self {
        let config = settings.solver_config;
        let spawn = settings.spawn_config;
        let solver = MpmSolver::new(config, spawn)
            .with_default_material(Box::new(build_fluid_material(params)))
            .with_boundary(Box::new(build_boundary(config.boundary_thickness, params)));
        let simulation_speed = params.target_solver_hz * config.dt;
        let stepper = FixedStepController::new(FixedStepConfig {
            dt: config.dt,
            simulation_speed,
            max_substeps_per_frame: settings.max_substeps_per_frame,
            max_frame_delta: settings.max_frame_delta,
        });

        Self {
            solver,
            stepper,
        }
    }
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

fn build_fluid_material(params: FluidRuntimeParameters) -> NewtonianFluidMaterial {
    let mut material = NewtonianFluidMaterial::new(
        params.rest_density,
        params.dynamic_viscosity,
        params.eos_stiffness,
        params.eos_power,
    );
    material.pressure_floor = params.pressure_floor;
    material.velocity_damping = params.velocity_damping;
    material.affine_damping = params.affine_damping;
    material
}

fn build_boundary(boundary_thickness: usize, params: FluidRuntimeParameters) -> PredictiveBoundary {
    PredictiveBoundary::new(boundary_thickness, params.predictive_wall_min)
}

#[derive(Component)]
struct ParticleVisual {
    index: usize,
}

fn setup_scene(mut commands: Commands, sim: Res<Simulation>, settings: Res<BasicFluidsSettings>) {
    commands.spawn(Camera2d);

    for (index, p) in sim.solver.particles().iter().enumerate() {
        commands.spawn((
            Sprite::from_color(Color::srgb(0.23, 0.66, 0.96), Vec2::ONE),
            Transform {
                translation: to_world(
                    p.x,
                    sim.solver.config().grid_res,
                    settings.pixels_per_cell,
                    settings.snap_to_pixels,
                ),
                scale: Vec3::new(settings.particle_diameter, settings.particle_diameter, 1.0),
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
    limits: Res<RuntimeParameterLimits>,
    mut params: ResMut<FluidRuntimeParameters>,
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

    let Ok(ctx) = contexts.ctx_mut() else {
        return;
    };

    *diag_elapsed += time.delta_secs();
    if cached_diag.is_none() || *diag_elapsed >= 0.25 {
        *diag_elapsed = 0.0;
        let snapshot = sim.solver.diagnostics_snapshot();
        let healthy = evaluate_mpm_health(&snapshot, &diagnostics_runtime.policy.thresholds).healthy();
        *cached_diag = Some((snapshot, healthy));
    }

    egui::Window::new("Basic Fluids Controls")
        .default_pos([10.0, 10.0])
        .default_width(320.0)
        .resizable(false)
        .show(ctx, |ui| {
            // Always-visible perf bar at the top.
            if let Some((snapshot, healthy)) = *cached_diag {
                let fps = if time.delta_secs() > 0.0 { 1.0 / time.delta_secs() } else { 0.0 };
                let color = if healthy { egui::Color32::LIGHT_GREEN } else { egui::Color32::LIGHT_RED };
                ui.colored_label(color, format!("fps={:.0}  substeps={}  health={}", fps, snapshot.substeps_last_step, healthy));
            } else {
                ui.label("fps=…  substeps=…");
            }
            ui.separator();

            ui.label("Simulation");
            ui.add(
                egui::Slider::new(
                    &mut params.target_solver_hz,
                    limits.target_solver_hz.min..=limits.target_solver_hz.max,
                )
                .text("solver_hz"),
            );
            ui.add(
                egui::Slider::new(&mut params.gravity, limits.gravity.min..=limits.gravity.max)
                    .text("gravity"),
            );

            ui.separator();
            ui.label("Fluid Material");
            ui.add(
                egui::Slider::new(
                    &mut params.rest_density,
                    limits.rest_density.min..=limits.rest_density.max,
                )
                .text("rest_density"),
            );
            ui.add(
                egui::Slider::new(
                    &mut params.dynamic_viscosity,
                    limits.dynamic_viscosity.min..=limits.dynamic_viscosity.max,
                )
                .text("viscosity"),
            );
            ui.add(
                egui::Slider::new(
                    &mut params.eos_stiffness,
                    limits.eos_stiffness.min..=limits.eos_stiffness.max,
                )
                .text("eos_stiffness"),
            );
            ui.add(
                egui::Slider::new(
                    &mut params.eos_power,
                    limits.eos_power.min..=limits.eos_power.max,
                )
                .text("eos_power"),
            );
            ui.add(
                egui::Slider::new(
                    &mut params.pressure_floor,
                    limits.pressure_floor.min..=limits.pressure_floor.max,
                )
                .text("pressure_floor"),
            );

            ui.separator();
            ui.label("Numerical Damping");
            ui.add(
                egui::Slider::new(
                    &mut params.velocity_damping,
                    limits.velocity_damping.min..=limits.velocity_damping.max,
                )
                .text("velocity_damping"),
            );
            ui.add(
                egui::Slider::new(
                    &mut params.affine_damping,
                    limits.affine_damping.min..=limits.affine_damping.max,
                )
                .text("affine_damping"),
            );
            ui.add(
                egui::Slider::new(
                    &mut params.predictive_wall_min,
                    limits.predictive_wall_min.min..=limits.predictive_wall_min.max,
                )
                .text("predictive_wall_min"),
            );

            ui.separator();
            if ui.button("Reset Simulation (R)").clicked() {
                reset_requested.0 = true;
            }

            ui.separator();
            ui.label("Diagnostics");
            if let Some((snapshot, _healthy)) = *cached_diag {
                ui.label(format!(
                    "cfl={:.3}  dt={:.4}/{:.4}",
                    snapshot.cfl_number,
                    snapshot.effective_dt,
                    snapshot.configured_dt,
                ));
                ui.label(format!(
                    "mass_err={:.1e}  mom_err={:.2e}",
                    snapshot.relative_mass_error,
                    snapshot.relative_momentum_error
                ));
                ui.label(format!(
                    "cells={}  p/cell={:.1}",
                    snapshot.active_grid_cells, snapshot.particles_per_active_cell
                ));
            }
        });
}

fn handle_reset_request(
    settings: Res<BasicFluidsSettings>,
    defaults: Res<FluidRuntimeDefaults>,
    mut reset_requested: ResMut<ResetSimulationRequested>,
    mut params: ResMut<FluidRuntimeParameters>,
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
    params: Res<FluidRuntimeParameters>,
    mut sim: ResMut<Simulation>,
    time: Res<Time>,
    mut last_log_time: Local<f32>,
    mut last_applied: Local<Option<FluidRuntimeParameters>>,
) {
    let current = *params;
    if last_applied.is_some_and(|previous| previous == current) {
        return;
    }

    let dt = sim.solver.config().dt;
    let boundary_thickness = sim.solver.config().boundary_thickness;

    sim.solver.set_gravity(current.gravity);
    sim.solver
        .set_default_material(Box::new(build_fluid_material(current)));
    sim.solver
        .set_boundary_condition(Box::new(build_boundary(boundary_thickness, current)));
    sim.stepper
        .set_simulation_speed(current.target_solver_hz * dt);

    // Throttle logging so interactive tuning does not flood stdout.
    let now = time.elapsed_secs();
    if now - *last_log_time >= 0.75 || last_applied.is_none() {
        println!(
            "[runtime] hz={:.0} g={:.3} rho0={:.2} mu={:.3} eos_k={:.1} eos_p={:.2}",
            current.target_solver_hz,
            current.gravity,
            current.rest_density,
            current.dynamic_viscosity,
            current.eos_stiffness,
            current.eos_power
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
    if let Some(report) = update_mpm_reporting(
        &mut diagnostics_runtime.state,
        time.delta_secs(),
        &snapshot,
        &policy,
    ) {
        if let Some(event_line) = report.event_line {
            println!("{}", event_line);
        }
        println!("{}", report.report_line);
    }
}

fn sync_particle_visuals(
    sim: Res<Simulation>,
    settings: Res<BasicFluidsSettings>,
    mut query: Query<(&ParticleVisual, &mut Transform)>,
) {
    for (visual, mut transform) in &mut query {
        let p = &sim.solver.particles()[visual.index];
        transform.translation = to_world(
            p.x,
            sim.solver.config().grid_res,
            settings.pixels_per_cell,
            settings.snap_to_pixels,
        );
    }
}

fn apply_cursor_force(
    windows: Query<&Window>,
    camera_query: Query<(&Camera, &GlobalTransform)>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    settings: Res<BasicFluidsSettings>,
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

    let grid_res = sim.solver.config().grid_res;
    let half = grid_res as f32 * 0.5;
    let grid_pos = world_pos / settings.pixels_per_cell + Vec2::splat(half);
    let sign = if mouse_buttons.pressed(MouseButton::Right) { -1.0 } else { 1.0 };
    let dt = time.delta_secs().min(settings.max_frame_delta);

    for p in sim.solver.particles_mut().iter_mut() {
        let diff = p.x - grid_pos;
        let dist = diff.length();
        if dist < 5.0 && dist > 1e-4 {
            p.v += (diff / dist) * sign * 40.0 * (1.0 - dist / 5.0) * dt;
            let speed = p.v.length();
            if speed > 20.0 { p.v *= 20.0 / speed; }
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

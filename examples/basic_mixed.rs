use bevy::prelude::*;
use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass, egui};
use emerge::diagnostics::{
    MpmHealthThresholds, MpmReportingPolicy, MpmReportingState, MpmSnapshot, evaluate_mpm_health,
    update_mpm_reporting,
};
use emerge::runtime::fixed_step::{FixedStepConfig, FixedStepController};
use emerge::solver::{
    MpmSolver, NeoHookeanMaterial, NewtonianFluidMaterial, SlipBoundary, SolverConfig, SpawnConfig,
};

const FLUID_MATERIAL_ID: u32 = 0;
const JELLY_MATERIAL_ID: u32 = 1;

fn main() {
    let settings = BasicMixedSettings::default();
    let runtime_defaults = settings.runtime_defaults;

    App::new()
        .insert_resource(ClearColor(Color::srgb(0.06, 0.06, 0.08)))
        .insert_resource(settings)
        .insert_resource(runtime_defaults)
        .insert_resource(MixedRuntimeDefaults(runtime_defaults))
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
                title: "MLS-MPM Basic Mixed Materials".to_string(),
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
struct BasicMixedSettings {
    window_resolution: (u32, u32),
    pixels_per_cell: f32,
    particle_diameter: f32,
    snap_to_pixels: bool,
    max_substeps_per_frame: usize,
    max_frame_delta: f32,
    solver_config: SolverConfig,
    spawn_config: SpawnConfig,
    runtime_defaults: MixedRuntimeParameters,
    runtime_limits: MixedRuntimeLimits,
    diagnostics_report_interval_secs: f32,
    diagnostics_healthy_heartbeat_secs: f32,
    diagnostics_thresholds: MpmHealthThresholds,
    diagnostics_log_healthy: bool,
}

impl Default for BasicMixedSettings {
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

        let runtime_defaults = MixedRuntimeParameters {
            target_solver_hz: 30.0,
            gravity: solver_config.gravity,
            split_x: spawn_config.box_center.x,
            fluid_rest_density: 4.0,
            fluid_viscosity: 0.1,
            fluid_eos_stiffness: 10.0,
            fluid_eos_power: 4.0,
            jelly_lambda: 10.0,
            jelly_mu: 20.0,
        };

        let runtime_limits = MixedRuntimeLimits {
            target_solver_hz: ScalarLimit::new(5.0, 60.0),
            gravity: ScalarLimit::new(-2.0, 2.0),
            split_x: ScalarLimit::new(8.0, 56.0),
            fluid_rest_density: ScalarLimit::new(1.0, 12.0),
            fluid_viscosity: ScalarLimit::new(0.0, 20.0),
            fluid_eos_stiffness: ScalarLimit::new(1.0, 100.0),
            fluid_eos_power: ScalarLimit::new(1.0, 8.0),
            jelly_lambda: ScalarLimit::new(1.0, 120.0),
            jelly_mu: ScalarLimit::new(1.0, 240.0),
        };

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
            diagnostics_thresholds: MpmHealthThresholds::for_spacing(spawn_config.spacing),
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
struct MixedRuntimeLimits {
    target_solver_hz: ScalarLimit,
    gravity: ScalarLimit,
    split_x: ScalarLimit,
    fluid_rest_density: ScalarLimit,
    fluid_viscosity: ScalarLimit,
    fluid_eos_stiffness: ScalarLimit,
    fluid_eos_power: ScalarLimit,
    jelly_lambda: ScalarLimit,
    jelly_mu: ScalarLimit,
}

#[derive(Resource, Clone, Copy, Debug, PartialEq)]
struct MixedRuntimeParameters {
    target_solver_hz: f32,
    gravity: f32,
    split_x: f32,
    fluid_rest_density: f32,
    fluid_viscosity: f32,
    fluid_eos_stiffness: f32,
    fluid_eos_power: f32,
    jelly_lambda: f32,
    jelly_mu: f32,
}

#[derive(Resource, Clone, Copy, Debug)]
struct MixedRuntimeDefaults(MixedRuntimeParameters);

#[derive(Resource, Default)]
struct ResetSimulationRequested(bool);

#[derive(Resource)]
struct Simulation {
    solver: MpmSolver,
    stepper: FixedStepController,
}

impl Simulation {
    fn new(settings: BasicMixedSettings, params: MixedRuntimeParameters) -> Self {
        let config = settings.solver_config;
        let spawn = settings.spawn_config;
        let solver = MpmSolver::new(config, spawn)
            .with_default_material(Box::new(build_fluid_material(params)))
            .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)))
            .with_material(
                JELLY_MATERIAL_ID,
                Box::new(build_jelly_material(params.jelly_lambda, params.jelly_mu)),
            )
            .with_particle_materials_by_position(|position| {
                if position.x < params.split_x {
                    FLUID_MATERIAL_ID
                } else {
                    JELLY_MATERIAL_ID
                }
            });

        let stepper = FixedStepController::new(FixedStepConfig {
            dt: config.dt,
            simulation_speed: params.target_solver_hz * config.dt,
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

fn build_fluid_material(params: MixedRuntimeParameters) -> NewtonianFluidMaterial {
    NewtonianFluidMaterial::new(
        params.fluid_rest_density,
        params.fluid_viscosity,
        params.fluid_eos_stiffness,
        params.fluid_eos_power,
    )
}

fn build_jelly_material(elastic_lambda: f32, elastic_mu: f32) -> NeoHookeanMaterial {
    NeoHookeanMaterial::new(elastic_lambda, elastic_mu)
}

fn particle_color(material_id: u32) -> Color {
    match material_id {
        FLUID_MATERIAL_ID => Color::srgb(0.22, 0.64, 0.95),
        JELLY_MATERIAL_ID => Color::srgb(0.95, 0.50, 0.22),
        _ => Color::WHITE,
    }
}

#[derive(Component)]
struct ParticleVisual {
    index: usize,
}

fn setup_scene(mut commands: Commands, sim: Res<Simulation>, settings: Res<BasicMixedSettings>) {
    commands.spawn(Camera2d);

    for (index, particle) in sim.solver.particles().iter().enumerate() {
        commands.spawn((
            Sprite::from_color(particle_color(particle.material_id), Vec2::ONE),
            Transform {
                translation: to_world(
                    particle.x,
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
    limits: Res<MixedRuntimeLimits>,
    mut params: ResMut<MixedRuntimeParameters>,
    mut reset_requested: ResMut<ResetSimulationRequested>,
    sim: Res<Simulation>,
    diagnostics_runtime: Res<DiagnosticsRuntime>,
    time: Res<Time>,
    mut initialized: Local<bool>,
    mut diag_elapsed: Local<f32>,
    mut cached_diag: Local<Option<(MpmSnapshot, bool, usize, usize)>>,
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
        let counts = sim.solver.material_particle_counts();
        let fluid_count = counts.get(&FLUID_MATERIAL_ID).copied().unwrap_or(0);
        let jelly_count = counts.get(&JELLY_MATERIAL_ID).copied().unwrap_or(0);
        *cached_diag = Some((snapshot, healthy, fluid_count, jelly_count));
    }

    egui::Window::new("Basic Mixed Controls")
        .default_pos([10.0, 10.0])
        .default_width(340.0)
        .resizable(false)
        .show(ctx, |ui| {
            if let Some((snapshot, healthy, _fc, _jc)) = *cached_diag {
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
            ui.add(
                egui::Slider::new(&mut params.split_x, limits.split_x.min..=limits.split_x.max)
                    .text("split_x"),
            );

            ui.separator();
            ui.label("Fluid");
            ui.add(
                egui::Slider::new(
                    &mut params.fluid_rest_density,
                    limits.fluid_rest_density.min..=limits.fluid_rest_density.max,
                )
                .text("rho0"),
            );
            ui.add(
                egui::Slider::new(
                    &mut params.fluid_viscosity,
                    limits.fluid_viscosity.min..=limits.fluid_viscosity.max,
                )
                .text("viscosity"),
            );
            ui.add(
                egui::Slider::new(
                    &mut params.fluid_eos_stiffness,
                    limits.fluid_eos_stiffness.min..=limits.fluid_eos_stiffness.max,
                )
                .text("eos_k"),
            );
            ui.add(
                egui::Slider::new(
                    &mut params.fluid_eos_power,
                    limits.fluid_eos_power.min..=limits.fluid_eos_power.max,
                )
                .text("eos_p"),
            );

            ui.separator();
            ui.label("Jelly");
            ui.add(
                egui::Slider::new(
                    &mut params.jelly_lambda,
                    limits.jelly_lambda.min..=limits.jelly_lambda.max,
                )
                .text("lambda"),
            );
            ui.add(
                egui::Slider::new(
                    &mut params.jelly_mu,
                    limits.jelly_mu.min..=limits.jelly_mu.max,
                )
                .text("mu"),
            );

            ui.separator();
            if ui.button("Reset Simulation (R)").clicked() {
                reset_requested.0 = true;
            }

            ui.separator();
            ui.label("Diagnostics");
            if let Some((snapshot, _healthy, fluid_count, jelly_count)) = *cached_diag {
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
                ui.label(format!("materials: fluid={}  jelly={}", fluid_count, jelly_count));
            }
        });
}

fn handle_reset_request(
    settings: Res<BasicMixedSettings>,
    defaults: Res<MixedRuntimeDefaults>,
    mut reset_requested: ResMut<ResetSimulationRequested>,
    mut params: ResMut<MixedRuntimeParameters>,
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
    params: Res<MixedRuntimeParameters>,
    mut sim: ResMut<Simulation>,
    time: Res<Time>,
    mut last_log_time: Local<f32>,
    mut last_applied: Local<Option<MixedRuntimeParameters>>,
) {
    let current = *params;
    if last_applied.is_some_and(|previous| previous == current) {
        return;
    }

    let configured_dt = sim.solver.config().dt;
    sim.solver.set_gravity(current.gravity);
    sim.solver
        .set_default_material(Box::new(build_fluid_material(current)));
    sim.solver.set_material(
        JELLY_MATERIAL_ID,
        Box::new(build_jelly_material(current.jelly_lambda, current.jelly_mu)),
    );
    sim.solver
        .assign_particle_materials_by_position(|position| {
            if position.x < current.split_x {
                FLUID_MATERIAL_ID
            } else {
                JELLY_MATERIAL_ID
            }
        });
    sim.stepper
        .set_simulation_speed(current.target_solver_hz * configured_dt);

    let now = time.elapsed_secs();
    if now - *last_log_time >= 0.75 || last_applied.is_none() {
        println!(
            "[runtime] hz={:.0} g={:.3} split={:.1} fluid(mu={:.2},rho0={:.2}) jelly(lambda={:.1},mu={:.1})",
            current.target_solver_hz,
            current.gravity,
            current.split_x,
            current.fluid_viscosity,
            current.fluid_rest_density,
            current.jelly_lambda,
            current.jelly_mu
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
    settings: Res<BasicMixedSettings>,
    mut query: Query<(&ParticleVisual, &mut Transform, &mut Sprite)>,
) {
    for (visual, mut transform, mut sprite) in &mut query {
        let particle = &sim.solver.particles()[visual.index];
        transform.translation = to_world(
            particle.x,
            sim.solver.config().grid_res,
            settings.pixels_per_cell,
            settings.snap_to_pixels,
        );
        sprite.color = particle_color(particle.material_id);
    }
}

fn apply_cursor_force(
    windows: Query<&Window>,
    camera_query: Query<(&Camera, &GlobalTransform)>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    settings: Res<BasicMixedSettings>,
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

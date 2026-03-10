use bevy::prelude::*;
use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass, egui};
use emerge::diagnostics::{
    MpmHealthThresholds, MpmReportingPolicy, MpmReportingState, MpmSnapshot, evaluate_mpm_health,
    update_mpm_reporting,
};
use emerge::runtime::fixed_step::{FixedStepConfig, FixedStepController};
use emerge::solver::{MpmSolver, SandMaterial, SlipBoundary, SolverConfig, SpawnConfig};

fn main() {
    let settings = SandSettings::default();
    let runtime_defaults = settings.runtime_defaults;

    App::new()
        .insert_resource(ClearColor(Color::srgb(0.08, 0.06, 0.04)))
        .insert_resource(settings)
        .insert_resource(runtime_defaults)
        .insert_resource(SandRuntimeDefaults(runtime_defaults))
        .insert_resource(ResetRequested::default())
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
                title: "MLS-MPM Sand — Drucker-Prager Elastoplasticity".to_string(),
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

// ─── Settings ────────────────────────────────────────────────────────────────

#[derive(Resource, Clone, Copy, Debug)]
struct SandSettings {
    window_resolution: (u32, u32),
    pixels_per_cell: f32,
    particle_diameter: f32,
    snap_to_pixels: bool,
    max_substeps_per_frame: usize,
    max_frame_delta: f32,
    solver_config: SolverConfig,
    spawn_config: SpawnConfig,
    runtime_defaults: SandRuntimeParameters,
    runtime_limits: SandRuntimeLimits,
    diagnostics_report_interval_secs: f32,
    diagnostics_healthy_heartbeat_secs: f32,
    diagnostics_thresholds: MpmHealthThresholds,
    diagnostics_log_healthy: bool,
}

impl Default for SandSettings {
    fn default() -> Self {
        let grid_res = 80;

        let solver_config = SolverConfig {
            grid_res,
            grid_cell_size: 1.0,
            dt: 0.05,
            adaptive_timestep: true,
            cfl_include_affine_speed: true,
            cfl_coefficient: 0.9,
            material_cfl_coefficient: 0.5,
            viscous_timestep_coefficient: 0.5,
            min_dt: 0.001,
            project_invalid_state: true,
            projection_min_density: 1.0e-6,
            projection_min_volume: 1.0e-6,
            projection_min_deformation_j: 1.0e-6,
            // Lower gravity (same as fluid/jelly examples) keeps the wave-speed budget
            // compatible with debug-CPU. Angle of repose is friction-controlled, not gravity.
            gravity: -0.3,
            boundary_thickness: 3,
            default_initial_volume: 1.0,
            recompute_density_each_step: true,
            particle_mass: 1.0,
            d_inverse: 4.0,
            max_substeps_per_step: 20,
        };

        // Sand block: 40×16 cells, centered horizontally, near the top.
        // Falls under gravity, piles at the bottom — angle of repose validates DP.
        let spawn_config = SpawnConfig {
            spacing: 0.5,
            box_size: IVec2::new(40, 16),
            box_center: Vec2::new(40.0, 64.0),
            initial_deformation_gradient: Mat2::IDENTITY,
            precompute_initial_volumes: false,
            initial_velocity_offset: Vec2::ZERO,
            initial_velocity_scale: 0.0,
            rng_seed: 42,
        };

        // Moderate stiffness: wave speed ≈ 45 cells/s → dt ≈ 0.011 → ~5 substeps.
        // J stays ≈ 0.997 under gravity=-0.3 overburden. Angle of repose is friction-controlled.
        let runtime_defaults = SandRuntimeParameters {
            target_solver_hz: 30.0,
            gravity: solver_config.gravity,
            lambda: 1000.0,
            mu: 500.0,
            friction_angle_deg: 35.0,
        };

        let runtime_limits = SandRuntimeLimits {
            target_solver_hz: (1.0, 60.0),
            gravity: (-5.0, 0.0),
            lambda: (100.0, 10000.0),
            mu: (50.0, 5000.0),
            friction_angle_deg: (1.0, 50.0),
        };

        Self {
            window_resolution: (800, 800),
            pixels_per_cell: 10.0,
            particle_diameter: 5.0,
            snap_to_pixels: true,
            max_substeps_per_frame: 4,
            max_frame_delta: 1.0 / 15.0,
            solver_config,
            spawn_config,
            runtime_defaults,
            runtime_limits,
            diagnostics_report_interval_secs: 2.0,
            diagnostics_healthy_heartbeat_secs: 10.0,
            diagnostics_thresholds: MpmHealthThresholds::for_spacing(spawn_config.spacing),
            diagnostics_log_healthy: false,
        }
    }
}

// ─── Runtime parameters ───────────────────────────────────────────────────────

#[derive(Resource, Clone, Copy, Debug, PartialEq)]
struct SandRuntimeParameters {
    target_solver_hz: f32,
    gravity: f32,
    lambda: f32,
    mu: f32,
    /// Internal friction angle φ in degrees. Controls angle of repose.
    /// Dry sand ≈ 30–35°. Wet sand ≈ 10–20°. Rock ≈ 40–50°.
    friction_angle_deg: f32,
}

#[derive(Resource, Clone, Copy, Debug)]
struct SandRuntimeDefaults(SandRuntimeParameters);

#[derive(Resource, Clone, Copy, Debug)]
struct SandRuntimeLimits {
    target_solver_hz: (f32, f32),
    gravity: (f32, f32),
    lambda: (f32, f32),
    mu: (f32, f32),
    friction_angle_deg: (f32, f32),
}

fn build_sand(params: SandRuntimeParameters) -> SandMaterial {
    let h0 = params.friction_angle_deg.to_radians();
    let mut m = SandMaterial::new(params.lambda, params.mu);
    m.h0 = h0;
    // h3 (residual angle) stays at default 10° — only initial angle is tunable here.
    m
}

// ─── Simulation resource ──────────────────────────────────────────────────────

#[derive(Resource)]
struct Simulation {
    solver: MpmSolver,
    stepper: FixedStepController,
}

impl Simulation {
    fn new(settings: SandSettings, params: SandRuntimeParameters) -> Self {
        let config = settings.solver_config;
        let spawn = settings.spawn_config;
        let solver = MpmSolver::new(config, spawn)
            .with_default_material(Box::new(build_sand(params)))
            .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));

        let stepper = FixedStepController::new(FixedStepConfig {
            dt: config.dt,
            simulation_speed: params.target_solver_hz * config.dt,
            max_substeps_per_frame: settings.max_substeps_per_frame,
            max_frame_delta: settings.max_frame_delta,
        });

        Self { solver, stepper }
    }
}

#[derive(Resource, Default)]
struct ResetRequested(bool);

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

// ─── Scene setup ──────────────────────────────────────────────────────────────

#[derive(Component)]
struct ParticleVisual {
    index: usize,
}

fn setup_scene(mut commands: Commands, sim: Res<Simulation>, settings: Res<SandSettings>) {
    commands.spawn(Camera2d);

    for (index, particle) in sim.solver.particles().iter().enumerate() {
        commands.spawn((
            Sprite::from_color(sand_color(particle.plastic_hardening), Vec2::ONE),
            Transform {
                translation: to_world(
                    particle.x,
                    sim.solver.config().grid_res,
                    settings.pixels_per_cell,
                    settings.snap_to_pixels,
                ),
                scale: Vec3::splat(settings.particle_diameter),
                ..default()
            },
            ParticleVisual { index },
        ));
    }
}

/// Maps accumulated plastic hardening q → a sandy color gradient.
/// Undisturbed (q≈0): dry sand gold. Heavily sheared (q large): darker/redder.
fn sand_color(q: f32) -> Color {
    let t = (q * 0.1).clamp(0.0, 1.0);
    Color::srgb(0.85 - t * 0.25, 0.72 - t * 0.30, 0.40 - t * 0.20)
}

// ─── Systems ──────────────────────────────────────────────────────────────────

fn request_reset_on_keypress(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut reset_requested: ResMut<ResetRequested>,
) {
    if keyboard.just_pressed(KeyCode::KeyR) {
        reset_requested.0 = true;
    }
}

fn handle_reset_request(
    settings: Res<SandSettings>,
    defaults: Res<SandRuntimeDefaults>,
    mut reset_requested: ResMut<ResetRequested>,
    mut params: ResMut<SandRuntimeParameters>,
    mut sim: ResMut<Simulation>,
    mut diagnostics_runtime: ResMut<DiagnosticsRuntime>,
) {
    if !reset_requested.0 {
        return;
    }
    *params = defaults.0;
    *sim = Simulation::new(*settings, *params);
    diagnostics_runtime.state = MpmReportingState::default();
    reset_requested.0 = false;
}

fn apply_runtime_parameters(
    params: Res<SandRuntimeParameters>,
    mut sim: ResMut<Simulation>,
    time: Res<Time>,
    mut last_log_time: Local<f32>,
    mut last_applied: Local<Option<SandRuntimeParameters>>,
) {
    let current = *params;
    if last_applied.is_some_and(|prev| prev == current) {
        return;
    }

    let configured_dt = sim.solver.config().dt;
    sim.solver.set_gravity(current.gravity);
    sim.solver
        .set_default_material(Box::new(build_sand(current)));
    sim.stepper
        .set_simulation_speed(current.target_solver_hz * configured_dt);

    let now = time.elapsed_secs();
    if now - *last_log_time >= 0.75 || last_applied.is_none() {
        println!(
            "[runtime] hz={:.0} g={:.2} lambda={:.0} mu={:.0} phi={:.1}°",
            current.target_solver_hz,
            current.gravity,
            current.lambda,
            current.mu,
            current.friction_angle_deg,
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
    settings: Res<SandSettings>,
    mut query: Query<(&ParticleVisual, &mut Transform, &mut Sprite)>,
) {
    for (visual, mut transform, mut sprite) in &mut query {
        let p = &sim.solver.particles()[visual.index];
        transform.translation = to_world(
            p.x,
            sim.solver.config().grid_res,
            settings.pixels_per_cell,
            settings.snap_to_pixels,
        );
        sprite.color = sand_color(p.plastic_hardening);
    }
}

fn apply_cursor_force(
    windows: Query<&Window>,
    camera_query: Query<(&Camera, &GlobalTransform)>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    settings: Res<SandSettings>,
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
        if dist < 6.0 && dist > 1e-4 {
            p.v += (diff / dist) * sign * 80.0 * (1.0 - dist / 6.0) * dt;
            let speed = p.v.length();
            if speed > 30.0 {
                p.v *= 30.0 / speed;
            }
        }
    }
}

// ─── UI ───────────────────────────────────────────────────────────────────────

fn runtime_controls_ui(
    mut contexts: EguiContexts,
    limits: Res<SandRuntimeLimits>,
    mut params: ResMut<SandRuntimeParameters>,
    mut reset_requested: ResMut<ResetRequested>,
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

    egui::Window::new("Sand Controls")
        .default_pos([10.0, 10.0])
        .default_width(300.0)
        .resizable(false)
        .show(ctx, |ui| {
            if let Some((snapshot, healthy)) = *cached_diag {
                let fps = if time.delta_secs() > 0.0 {
                    1.0 / time.delta_secs()
                } else {
                    0.0
                };
                let color = if healthy {
                    egui::Color32::LIGHT_GREEN
                } else {
                    egui::Color32::LIGHT_RED
                };
                ui.colored_label(
                    color,
                    format!(
                        "fps={:.0}  substeps={}  particles={}",
                        fps,
                        snapshot.substeps_last_step,
                        sim.solver.particles().len()
                    ),
                );
            }
            ui.separator();

            ui.label("Simulation");
            ui.add(
                egui::Slider::new(
                    &mut params.target_solver_hz,
                    limits.target_solver_hz.0..=limits.target_solver_hz.1,
                )
                .text("solver_hz"),
            );
            ui.add(
                egui::Slider::new(&mut params.gravity, limits.gravity.0..=limits.gravity.1)
                    .text("gravity"),
            );

            ui.separator();
            ui.label("Sand — Drucker-Prager");
            ui.add(
                egui::Slider::new(
                    &mut params.friction_angle_deg,
                    limits.friction_angle_deg.0..=limits.friction_angle_deg.1,
                )
                .text("friction φ (°)")
                .suffix("°"),
            );
            ui.label("↑ controls angle of repose. Dry sand ≈ 35°, rock ≈ 45°.");
            ui.add(
                egui::Slider::new(&mut params.lambda, limits.lambda.0..=limits.lambda.1)
                    .text("λ (bulk-like)"),
            );
            ui.add(
                egui::Slider::new(&mut params.mu, limits.mu.0..=limits.mu.1)
                    .text("µ (shear)"),
            );

            ui.separator();
            if let Some((snapshot, _)) = *cached_diag {
                ui.label(format!(
                    "cfl={:.3}  dt={:.4}/{:.4}",
                    snapshot.cfl_number, snapshot.effective_dt, snapshot.configured_dt,
                ));
                ui.label(format!(
                    "mass_err={:.1e}  mom_err={:.2e}",
                    snapshot.relative_mass_error, snapshot.relative_momentum_error,
                ));
            }

            ui.separator();
            ui.label("Left-click: push   Right-click: pull   R: reset");
            if ui.button("Reset (R)").clicked() {
                reset_requested.0 = true;
            }
        });
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

fn to_world(grid_pos: Vec2, grid_res: usize, pixels_per_cell: f32, snap_to_pixels: bool) -> Vec3 {
    let half = grid_res as f32 * 0.5;
    let mut centered = (grid_pos - Vec2::splat(half)) * pixels_per_cell;
    if snap_to_pixels {
        centered = centered.round();
    }
    Vec3::new(centered.x, centered.y, 0.0)
}

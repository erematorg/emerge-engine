/// Three-material showcase — sand terrain, fluid pool, and elastic blob — GPU compute.
///
/// Same scene as basic_showcase but runs on the GPU solver:
///   Mat 0  NeoHookean elastic  (green)  — creature body
///   Mat 1  Sand (Drucker-Prager) (gold)  — terrain
///   Mat 2  Newtonian fluid      (blue)  — water pool
///
/// Controls:
///   ↑ ↓ ← →   apply impulse to the elastic blob
///   LMB / RMB  push / pull all particles under cursor
///   R          reset scene
///
///   cargo run --example basic_showcase_gpu --features "bevy_examples,gpu"
use bevy::prelude::*;
use bevy::tasks::block_on;
use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass, egui};
use emerge::gpu::{GpuForceFieldEntry, GpuSolver};
use emerge::runtime::fixed_step::FixedStepController;
use emerge::{
    MaterialRegistry, NeoHookeanMaterial, NewtonianFluidMaterial, SandMaterial, SolverConfig,
    SpawnConfig, build_particles, log_frame_gpu,
};
use glam::{IVec2, Vec2};

const GRID: usize = 64;
const DT: f32 = 0.1;
const PPC: f32 = 10.0;
const MAX_DT: f32 = 1.0 / 15.0;

const ELASTIC_ID: u32 = 0;
const SAND_ID: u32 = 1;
const FLUID_ID: u32 = 2;

const LABELS: &[(u32, &str)] = &[
    (ELASTIC_ID, "elastic"),
    (SAND_ID, "sand"),
    (FLUID_ID, "fluid"),
];

const DRIVE_STRENGTH: f32 = 10.0;
const DRIVE_RADIUS: f32 = 12.0;

#[derive(Resource, Clone, Copy, PartialEq)]
struct Params {
    hz: f32,
    gravity: f32,
    e_lambda: f32,
    e_mu: f32,
    s_lambda: f32,
    s_mu: f32,
    f_density: f32,
    f_viscosity: f32,
    f_eos_k: f32,
    cursor_strength: f32,
    cursor_radius: f32,
}

const DEFAULTS: Params = Params {
    hz: 60.0,
    gravity: -0.3,
    e_lambda: 40.0,
    e_mu: 80.0,
    s_lambda: 400.0,
    s_mu: 200.0,
    f_density: 4.0,
    f_viscosity: 0.1,
    f_eos_k: 10.0,
    cursor_strength: 200.0,
    cursor_radius: 5.0,
};

#[derive(Resource)]
struct Sim {
    solver: GpuSolver,
    stepper: FixedStepController,
    prev: Params,
    // Elastic blob centroid — updated each frame for arrow-key drive.
    elastic_centroid: Vec2,
    physics_frame: u64,
}

impl Sim {
    fn new(p: Params) -> Self {
        let config = SolverConfig {
            min_dt: 0.005,
            max_substeps_per_step: 16,
            recompute_density_each_step: true,
            gravity: Vec2::new(0.0, p.gravity),
            ..SolverConfig::earth(GRID, 0.01, DT)
        };

        const SPACING: f32 = 0.7;

        let sand_spawn = SpawnConfig {
            spacing: SPACING,
            box_size: IVec2::new(22, 14),
            box_center: Vec2::new(19.0, 9.0),
            material_id: SAND_ID,
            initial_velocity_scale: 0.0,
            precompute_initial_volumes: true,
            ..SpawnConfig::for_solver(&config)
        };
        let fluid_spawn = SpawnConfig {
            spacing: SPACING,
            box_size: IVec2::new(22, 14),
            box_center: Vec2::new(45.0, 9.0),
            material_id: FLUID_ID,
            initial_velocity_scale: 0.0,
            precompute_initial_volumes: true,
            ..SpawnConfig::for_solver(&config)
        };
        let elastic_spawn = SpawnConfig {
            spacing: SPACING,
            box_size: IVec2::new(12, 12),
            box_center: Vec2::new(32.0, 46.0),
            material_id: ELASTIC_ID,
            initial_velocity_scale: 0.0,
            precompute_initial_volumes: true,
            ..SpawnConfig::for_solver(&config)
        };

        let mut particles = build_particles(&config, sand_spawn);
        particles.extend(build_particles(&config, fluid_spawn));
        particles.extend(build_particles(&config, elastic_spawn));

        let mut registry =
            MaterialRegistry::with_default(Box::new(NeoHookeanMaterial::new(p.e_lambda, p.e_mu)));
        registry.insert(SAND_ID, Box::new(SandMaterial::new(p.s_lambda, p.s_mu)));
        registry.insert(
            FLUID_ID,
            Box::new(NewtonianFluidMaterial::new(
                p.f_density,
                p.f_viscosity,
                p.f_eos_k,
                4.0,
            )),
        );

        let elastic_centroid = Vec2::new(32.0, 46.0);

        let solver = block_on(GpuSolver::new(config, particles, registry));

        println!("[showcase-gpu] total particles: {}", solver.particle_count());

        Self {
            solver,
            stepper: FixedStepController::standard(DT, p.hz),
            prev: p,
            elastic_centroid,
            physics_frame: 0,
        }
    }
}

fn mat_color(id: u32) -> Color {
    match id {
        SAND_ID => Color::srgb(0.80, 0.64, 0.22),
        FLUID_ID => Color::srgb(0.22, 0.62, 0.95),
        _ => Color::srgb(0.38, 0.80, 0.48),
    }
}

fn main() {
    App::new()
        .insert_resource(ClearColor(Color::srgb(0.06, 0.06, 0.09)))
        .insert_resource(DEFAULTS)
        .insert_resource(Sim::new(DEFAULTS))
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "emerge — Sand + Fluid + Elastic [GPU]  (arrows: move blob, R: reset)"
                    .into(),
                resolution: (900u32, 900u32).into(),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(EguiPlugin::default())
        .add_systems(Startup, setup)
        .add_systems(Update, (reset, cursor, drive, step, sync).chain())
        .add_systems(EguiPrimaryContextPass, ui)
        .run();
}

#[derive(Component)]
struct PVis(usize);

fn setup(mut commands: Commands, sim: Res<Sim>) {
    commands.spawn(Camera2d);
    for (i, p) in sim.solver.particles().iter().enumerate() {
        commands.spawn((
            PVis(i),
            Sprite {
                color: mat_color(p.material_id),
                custom_size: Some(Vec2::ONE),
                ..default()
            },
            Transform::from_translation(p2w(p.x)),
        ));
    }
}

fn reset(
    keys: Res<ButtonInput<KeyCode>>,
    mut sim: ResMut<Sim>,
    mut p: ResMut<Params>,
    mut commands: Commands,
    vis: Query<Entity, With<PVis>>,
) {
    if keys.just_pressed(KeyCode::KeyR) {
        *p = DEFAULTS;
        *sim = Sim::new(DEFAULTS);
        for e in &vis {
            commands.entity(e).despawn();
        }
        for (i, pt) in sim.solver.particles().iter().enumerate() {
            commands.spawn((
                PVis(i),
                Sprite {
                    color: mat_color(pt.material_id),
                    custom_size: Some(Vec2::ONE),
                    ..default()
                },
                Transform::from_translation(p2w(pt.x)),
            ));
        }
    }
}

fn cursor(
    windows: Query<&Window>,
    cam: Query<(&Camera, &GlobalTransform)>,
    mb: Res<ButtonInput<MouseButton>>,
    mut sim: ResMut<Sim>,
    params: Res<Params>,
) {
    sim.solver.clear_force_fields_gpu();
    if !mb.pressed(MouseButton::Left) && !mb.pressed(MouseButton::Right) {
        return;
    }
    let Ok(win) = windows.single() else { return };
    let Some(cp) = win.cursor_position() else {
        return;
    };
    let Ok((cam, ct)) = cam.single() else { return };
    let Ok(wp) = cam.viewport_to_world_2d(ct, cp) else {
        return;
    };
    let gp = wp / PPC + Vec2::splat(GRID as f32 * 0.5);
    let gm = if mb.pressed(MouseButton::Right) {
        params.cursor_strength
    } else {
        -params.cursor_strength
    };
    let r = params.cursor_radius;
    sim.solver
        .add_force_field_gpu(GpuForceFieldEntry::gravity_well(gp, gm, 4.0, r, r * 0.4));
}

fn drive(keys: Res<ButtonInput<KeyCode>>, mut sim: ResMut<Sim>, time: Res<Time>) {
    let dir = {
        let mut d = Vec2::ZERO;
        if keys.pressed(KeyCode::ArrowLeft) {
            d.x -= 1.0;
        }
        if keys.pressed(KeyCode::ArrowRight) {
            d.x += 1.0;
        }
        if keys.pressed(KeyCode::ArrowUp) {
            d.y += 1.0;
        }
        if keys.pressed(KeyCode::ArrowDown) {
            d.y -= 1.0;
        }
        d
    };
    if dir == Vec2::ZERO {
        return;
    }

    let impulse = dir.normalize() * DRIVE_STRENGTH * time.delta_secs().min(MAX_DT);
    let centroid = sim.elastic_centroid;
    for p in sim.solver.particles_mut() {
        if p.material_id != ELASTIC_ID {
            continue;
        }
        let dist = (p.x - centroid).length();
        if dist < DRIVE_RADIUS {
            p.v += impulse * (1.0 - dist / DRIVE_RADIUS);
        }
    }
}

fn step(time: Res<Time>, mut sim: ResMut<Sim>, params: Res<Params>) {
    sim.solver.set_gravity(Vec2::new(0.0, params.gravity));
    sim.stepper.set_simulation_speed(params.hz * DT);
    let n = sim.stepper.steps_for_frame(time.delta_secs());
    if n == 0 {
        return;
    }
    if sim.prev != *params {
        sim.solver
            .set_default_material(Box::new(NeoHookeanMaterial::new(
                params.e_lambda,
                params.e_mu,
            )));
        sim.solver
            .set_material(SAND_ID, Box::new(SandMaterial::new(params.s_lambda, params.s_mu)));
        sim.solver.set_material(
            FLUID_ID,
            Box::new(NewtonianFluidMaterial::new(
                params.f_density,
                params.f_viscosity,
                params.f_eos_k,
                4.0,
            )),
        );
        sim.prev = *params;
    }
    for _ in 0..n {
        sim.solver.step_frame();
        sim.physics_frame += 1;
    }
    sim.solver.sync_particles_blocking();
    log_frame_gpu(sim.physics_frame, DT, sim.solver.particles(), LABELS, 60);

    // Update elastic centroid from CPU mirror for arrow-key drive.
    let (sum, count) = sim
        .solver
        .particles()
        .iter()
        .filter(|p| p.material_id == ELASTIC_ID)
        .fold((Vec2::ZERO, 0usize), |(s, n), p| (s + p.x, n + 1));
    if count > 0 {
        sim.elastic_centroid = sum / count as f32;
    }
}

fn sync(sim: Res<Sim>, mut vis: Query<(&PVis, &mut Transform, &mut Sprite)>) {
    for (pv, mut t, mut s) in &mut vis {
        if let Some(p) = sim.solver.particles().get(pv.0) {
            t.translation = p2w(p.x);
            s.color = mat_color(p.material_id);
        }
    }
}

fn ui(mut ctx: EguiContexts, mut p: ResMut<Params>, mut sim: ResMut<Sim>, time: Res<Time>) {
    let Ok(ctx) = ctx.ctx_mut() else { return };
    egui::Window::new("Showcase [GPU]")
        .default_pos([10.0, 10.0])
        .default_width(260.0)
        .resizable(false)
        .show(ctx, |ui| {
            ui.label(format!(
                "fps={:.0}  n={}  [GPU]",
                time.delta_secs().recip(),
                sim.solver.particle_count(),
            ));
            ui.separator();
            ui.label("↑ ↓ ← →  move blob    R  reset");
            ui.label("LMB push  RMB pull (cursor field)");
            ui.separator();
            ui.add(egui::Slider::new(&mut p.hz, 5.0..=60.0).text("solver_hz"));
            ui.add(egui::Slider::new(&mut p.gravity, -3.0..=0.0).text("gravity"));
            ui.separator();
            ui.colored_label(egui::Color32::from_rgb(97, 204, 122), "Elastic blob (NeoHookean)");
            ui.add(egui::Slider::new(&mut p.e_lambda, 5.0..=200.0).text("λ"));
            ui.add(egui::Slider::new(&mut p.e_mu, 5.0..=400.0).text("µ"));
            ui.separator();
            ui.colored_label(egui::Color32::from_rgb(204, 163, 56), "Sand terrain (Drucker-Prager)");
            ui.add(egui::Slider::new(&mut p.s_lambda, 50.0..=2000.0).text("λ"));
            ui.add(egui::Slider::new(&mut p.s_mu, 25.0..=1000.0).text("µ"));
            ui.separator();
            ui.colored_label(egui::Color32::from_rgb(56, 158, 242), "Fluid pool (Tait EOS)");
            ui.add(egui::Slider::new(&mut p.f_density, 1.0..=10.0).text("rho0"));
            ui.add(egui::Slider::new(&mut p.f_viscosity, 0.0..=5.0).text("viscosity"));
            ui.add(egui::Slider::new(&mut p.f_eos_k, 1.0..=50.0).text("eos_k"));
            ui.separator();
            ui.add(
                egui::Slider::new(&mut p.cursor_strength, 10.0..=1000.0)
                    .text("cursor force")
                    .logarithmic(true),
            );
            ui.add(egui::Slider::new(&mut p.cursor_radius, 1.0..=15.0).text("cursor radius"));
            if ui.button("Reset (R)").clicked() {
                *p = DEFAULTS;
                *sim = Sim::new(DEFAULTS);
            }
        });
}

fn p2w(pos: Vec2) -> Vec3 {
    let c = (pos - Vec2::splat(GRID as f32 * 0.5)) * PPC;
    Vec3::new(c.x.round(), c.y.round(), 0.0)
}

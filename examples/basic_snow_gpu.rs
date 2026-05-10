/// Two snowballs colliding — GPU MLS-MPM, Stomakhin 2013 snow plasticity in WGSL.
///
/// Ball A (blue)  = soft powder  — low hardening, wide plastic limits.
/// Ball B (amber) = packed snow  — high hardening, tight limits.
/// Jp compression shown as red shift on impact.
///
///   cargo run --example basic_snow_gpu --features "bevy_examples,gpu"
use bevy::prelude::*;
use bevy::tasks::block_on;
use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass, egui};
use emerge::gpu::{GpuForceFieldEntry, GpuSolver};
use emerge::{MaterialRegistry, SnowMaterial, SolverConfig, SpawnConfig, build_particles, log_frame_gpu};
use emerge::runtime::fixed_step::FixedStepController;
use glam::Vec2;

const GRID: usize = 64;
const DT: f32 = 0.1;
const PPC: f32 = 10.0;

const BALL_R: f32 = 9.0;
const BALL_A: Vec2 = Vec2::new(16.0, 44.0);
const BALL_B: Vec2 = Vec2::new(48.0, 44.0);
const MAT_SOFT:   u32 = 0;
const MAT_PACKED: u32 = 1;
const LABELS: &[(u32, &str)] = &[(MAT_SOFT, "soft"), (MAT_PACKED, "packed")];

const COL_A: Color = Color::srgb(0.35, 0.65, 1.00); // blue
const COL_B: Color = Color::srgb(0.95, 0.80, 0.45); // amber

#[derive(Resource, Clone, Copy, PartialEq)]
struct Params {
    hz: f32,
    gravity: f32,
    speed: f32,
    lambda: f32,
    mu: f32,
    xi_a: f32,
    theta_c_a: f32,
    theta_s_a: f32,
    xi_b: f32,
    theta_c_b: f32,
    theta_s_b: f32,
    cursor_strength: f32,
    cursor_radius: f32,
}

const DEFAULTS: Params = Params {
    hz: 60.0,
    gravity: -0.08,
    speed: 8.0,
    lambda: 1389.0,
    mu: 2083.0,
    xi_a: 7.0,
    theta_c_a: 0.025,
    theta_s_a: 0.0075,
    xi_b: 10.0,
    theta_c_b: 0.012,
    theta_s_b: 0.004,
    cursor_strength: 300.0,
    cursor_radius: 6.0,
};

#[derive(Resource)]
struct Sim {
    solver: GpuSolver,
    stepper: FixedStepController,
    prev: Params,
}

impl Sim {
    fn new(p: Params) -> Self {
        let config = SolverConfig {
            max_substeps_per_step: 20,
            ..SolverConfig::standard(GRID, DT, Vec2::new(0.0, p.gravity))
        };
        // Two snowballs: each a disk, given opposing initial velocities post-spawn.
        let ball_spawn = |center: Vec2, mat: u32| {
            SpawnConfig::for_solver(&config)
                .at(center).disk(BALL_R).spacing(0.5).material(mat).rng_seed(7)
        };
        let mut ball_a = build_particles(&config, ball_spawn(BALL_A, MAT_SOFT));
        let mut ball_b = build_particles(&config, ball_spawn(BALL_B, MAT_PACKED));
        let speed = p.speed;
        for pt in &mut ball_a { pt.v = Vec2::new(speed, 0.0); }
        for pt in &mut ball_b { pt.v = Vec2::new(-speed, 0.0); }
        let mut particles = ball_a;
        particles.extend(ball_b);
        let mut registry = MaterialRegistry::with_default(Box::new(make_snow_a(&p)));
        registry.insert(MAT_PACKED, Box::new(make_snow_b(&p)));
        let solver = block_on(GpuSolver::new(config, particles, registry));
        Self {
            solver,
            stepper: FixedStepController::standard(DT, p.hz),
            prev: p,
        }
    }
}

fn make_snow_a(p: &Params) -> SnowMaterial {
    SnowMaterial::new(p.lambda, p.mu, p.xi_a, p.theta_c_a, p.theta_s_a, 0.6, 20.0)
}
fn make_snow_b(p: &Params) -> SnowMaterial {
    SnowMaterial::new(p.lambda, p.mu, p.xi_b, p.theta_c_b, p.theta_s_b, 0.6, 20.0)
        .with_cohesion(400.0)
}

fn main() {
    App::new()
        .insert_resource(ClearColor(Color::srgb(0.05, 0.07, 0.10)))
        .insert_resource(DEFAULTS)
        .insert_resource(Sim::new(DEFAULTS))
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "MLS-MPM Snow (GPU) — Powder vs Packed".into(),
                resolution: (900u32, 900u32).into(),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(EguiPlugin::default())
        .add_systems(Startup, setup)
        .add_systems(Update, (reset, cursor, step, sync).chain())
        .add_systems(EguiPrimaryContextPass, ui)
        .run();
}

#[derive(Component)]
struct PVis(usize);

fn setup(mut commands: Commands, sim: Res<Sim>) {
    commands.spawn(Camera2d);
    for (i, p) in sim.solver.particles().iter().enumerate() {
        let color = if p.material_id == MAT_SOFT { COL_A } else { COL_B };
        commands.spawn((
            PVis(i),
            Sprite { color, custom_size: Some(Vec2::ONE), ..default() },
            Transform::from_translation(p2w(p.x)),
        ));
    }
}

fn reset(keys: Res<ButtonInput<KeyCode>>, mut sim: ResMut<Sim>, mut p: ResMut<Params>,
         mut commands: Commands, vis: Query<Entity, With<PVis>>) {
    if keys.just_pressed(KeyCode::KeyR) {
        *p = DEFAULTS;
        *sim = Sim::new(DEFAULTS);
        for e in &vis { commands.entity(e).despawn(); }
        for (i, pt) in sim.solver.particles().iter().enumerate() {
            let color = if pt.material_id == MAT_SOFT { COL_A } else { COL_B };
            commands.spawn((
                PVis(i),
                Sprite { color, custom_size: Some(Vec2::ONE), ..default() },
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
    if !mb.pressed(MouseButton::Left) && !mb.pressed(MouseButton::Right) { return; }
    let Ok(win) = windows.single() else { return };
    let Some(cp) = win.cursor_position() else { return };
    let Ok((cam, ct)) = cam.single() else { return };
    let Ok(wp) = cam.viewport_to_world_2d(ct, cp) else { return };
    let gp = wp / PPC + Vec2::splat(GRID as f32 * 0.5);
    let gm = if mb.pressed(MouseButton::Right) { params.cursor_strength } else { -params.cursor_strength };
    let r = params.cursor_radius;
    sim.solver.add_force_field_gpu(GpuForceFieldEntry::gravity_well(gp, gm, 4.0, r, r * 0.4));
}

fn step(time: Res<Time>, mut sim: ResMut<Sim>, params: Res<Params>) {
    sim.solver.set_gravity(Vec2::new(0.0, params.gravity));
    sim.stepper.set_simulation_speed(params.hz * DT);
    let n = sim.stepper.steps_for_frame(time.delta_secs());
    if n == 0 { return; }
    if sim.prev != *params {
        sim.solver.set_default_material(Box::new(make_snow_a(&params)));
        sim.solver.set_material(MAT_PACKED, Box::new(make_snow_b(&params)));
        sim.prev = *params;
    }
    for _ in 0..n {
        sim.solver.step_frame();
    }
    sim.solver.sync_particles_blocking();
    static FRAME: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let f = FRAME.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
    log_frame_gpu(f, DT, sim.solver.particles(), LABELS, 60);
}

fn sync(sim: Res<Sim>, mut vis: Query<(&PVis, &mut Transform)>) {
    for (pv, mut t) in &mut vis {
        if let Some(p) = sim.solver.particles().get(pv.0) {
            t.translation = p2w(p.x);
        }
    }
}

fn ui(mut ctx: EguiContexts, mut p: ResMut<Params>, mut sim: ResMut<Sim>, time: Res<Time>) {
    let Ok(ctx) = ctx.ctx_mut() else { return };
    egui::Window::new("Snow (GPU)")
        .default_pos([10.0, 10.0])
        .default_width(280.0)
        .resizable(false)
        .show(ctx, |ui| {
            ui.label(format!(
                "fps={:.0}  n={}  [GPU]",
                time.delta_secs().recip(),
                sim.solver.particle_count()
            ));
            ui.separator();
            ui.add(egui::Slider::new(&mut p.hz, 1.0..=60.0).text("solver_hz"));
            ui.add(egui::Slider::new(&mut p.gravity, -2.0..=0.0).text("gravity"));
            ui.add(egui::Slider::new(&mut p.speed, 1.0..=30.0).text("speed (→ reset)"));
            ui.separator();
            ui.label("Shared stiffness");
            ui.add(egui::Slider::new(&mut p.lambda, 50.0..=5_000.0).text("λ"));
            ui.add(egui::Slider::new(&mut p.mu, 50.0..=10_000.0).text("µ"));
            ui.separator();
            ui.colored_label(egui::Color32::from_rgb(90, 165, 255), "Soft powder (blue)");
            ui.add(egui::Slider::new(&mut p.xi_a, 0.0..=20.0).text("ξ"));
            ui.add(egui::Slider::new(&mut p.theta_c_a, 0.001..=0.1).logarithmic(true).text("θ_c"));
            ui.add(egui::Slider::new(&mut p.theta_s_a, 0.001..=0.05).logarithmic(true).text("θ_s"));
            ui.separator();
            ui.colored_label(egui::Color32::from_rgb(242, 204, 115), "Packed snow (amber)");
            ui.add(egui::Slider::new(&mut p.xi_b, 0.0..=20.0).text("ξ"));
            ui.add(egui::Slider::new(&mut p.theta_c_b, 0.001..=0.1).logarithmic(true).text("θ_c"));
            ui.add(egui::Slider::new(&mut p.theta_s_b, 0.001..=0.05).logarithmic(true).text("θ_s"));
            ui.separator();
            ui.add(egui::Slider::new(&mut p.cursor_strength, 10.0..=1000.0).text("cursor force").logarithmic(true));
            ui.add(egui::Slider::new(&mut p.cursor_radius, 1.0..=15.0).text("cursor radius"));
            ui.label("LMB: push  RMB: pull  R: reset");
            if ui.button("Reset (R)").clicked() {
                *p = DEFAULTS;
                *sim = Sim::new(DEFAULTS);
            }
        });
}

fn p2w(pos: Vec2) -> Vec3 {
    let c = (pos - Vec2::splat(GRID as f32 * 0.5)) * PPC;
    Vec3::new(c.x, c.y, 0.0)
}

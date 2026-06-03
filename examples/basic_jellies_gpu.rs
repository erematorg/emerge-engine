/// GPU elastic solids — NeoHookean + Corotated + Viscoelastic, three-body comparison.
///
///   cargo run --example basic_jellies_gpu --features "bevy_examples,gpu"
use bevy::prelude::*;
use bevy::tasks::block_on;
use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass, egui};
use emerge::gpu::{GpuForceFieldEntry, GpuSolver};
use emerge::runtime::fixed_step::FixedStepController;
use emerge::{
    CorotatedMaterial, MaterialRegistry, NeoHookeanMaterial, SolverConfig, SpawnConfig,
    ViscoelasticMaterial, build_particles, log_frame_gpu,
};
use glam::Vec2;

const GRID: usize = 64;
const DT: f32 = 0.1;
const PPC: f32 = 10.0;
const LABELS: &[(u32, &str)] = &[(0, "neo"), (1, "cor"), (2, "vis")];

const MAT_NEO: u32 = 0;
const MAT_COR: u32 = 1;
const MAT_VIS: u32 = 2;

#[derive(Resource, Clone, Copy, PartialEq)]
struct Params {
    hz: f32,
    gravity: f32,
    neo_lambda: f32,
    neo_mu: f32,
    cor_lambda: f32,
    cor_mu: f32,
    vis_lambda: f32,
    vis_mu: f32,
    vis_viscosity: f32,
    cursor_strength: f32,
    cursor_radius: f32,
}

const DEFAULTS: Params = Params {
    hz: 60.0,
    gravity: -0.3,
    neo_lambda: 10.0,
    neo_mu: 20.0,
    cor_lambda: 30.0,
    cor_mu: 60.0,
    vis_lambda: 10.0,
    vis_mu: 15.0,
    vis_viscosity: 0.15,
    cursor_strength: 200.0,
    cursor_radius: 5.0,
};

#[derive(Resource)]
struct Sim {
    solver: GpuSolver,
    stepper: FixedStepController,
    prev: Params,
    physics_frame: u64,
}

impl Sim {
    fn new(p: Params) -> Self {
        let config = SolverConfig {
            min_dt: 0.01,
            max_substeps_per_step: 8,
            gravity: Vec2::new(0.0, p.gravity),
            ..SolverConfig::earth(GRID, 0.01, DT)
        };
        let spawn = |center: Vec2, mat: u32| {
            SpawnConfig::for_solver(&config)
                .at(center)
                .disk(7.0)
                .spacing(0.5)
                .material(mat)
                .precompute_volumes()
        };

        let mut particles = build_particles(&config, spawn(Vec2::new(14.0, 50.0), MAT_NEO));
        particles.extend(build_particles(&config, spawn(Vec2::new(32.0, 50.0), MAT_COR)));
        particles.extend(build_particles(&config, spawn(Vec2::new(50.0, 50.0), MAT_VIS)));

        let mut registry =
            MaterialRegistry::with_default(Box::new(NeoHookeanMaterial::new(p.neo_lambda, p.neo_mu)));
        registry.insert(MAT_COR, Box::new(CorotatedMaterial::new(p.cor_lambda, p.cor_mu)));
        registry.insert(MAT_VIS, Box::new(ViscoelasticMaterial::new(p.vis_lambda, p.vis_mu, p.vis_viscosity)));
        let solver = block_on(GpuSolver::new(config, particles, registry));

        Self { solver, stepper: FixedStepController::standard(DT, p.hz), prev: p, physics_frame: 0 }
    }
}

fn mat_color(mat: u32) -> Color {
    match mat {
        0 => Color::srgb(0.94, 0.52, 0.27),
        1 => Color::srgb(0.25, 0.78, 0.65),
        _ => Color::srgb(0.72, 0.40, 0.90),
    }
}

fn main() {
    App::new()
        .insert_resource(ClearColor(Color::srgb(0.07, 0.06, 0.08)))
        .insert_resource(DEFAULTS)
        .insert_resource(Sim::new(DEFAULTS))
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "MLS-MPM Jellies (GPU) — NeoHookean · Corotated · Viscoelastic".into(),
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
        commands.spawn((
            PVis(i),
            Sprite { color: mat_color(p.material_id), custom_size: Some(Vec2::ONE), ..default() },
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
        for e in &vis { commands.entity(e).despawn(); }
        for (i, pt) in sim.solver.particles().iter().enumerate() {
            commands.spawn((
                PVis(i),
                Sprite { color: mat_color(pt.material_id), custom_size: Some(Vec2::ONE), ..default() },
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
        sim.solver.set_default_material(Box::new(NeoHookeanMaterial::new(params.neo_lambda, params.neo_mu)));
        sim.solver.set_material(MAT_COR, Box::new(CorotatedMaterial::new(params.cor_lambda, params.cor_mu)));
        sim.solver.set_material(MAT_VIS, Box::new(ViscoelasticMaterial::new(params.vis_lambda, params.vis_mu, params.vis_viscosity)));
        sim.prev = *params;
    }
    for _ in 0..n {
        sim.solver.step_frame();
        sim.physics_frame += 1;
    }
    sim.solver.sync_particles_blocking();
    log_frame_gpu(sim.physics_frame, DT, sim.solver.particles(), LABELS, 60);
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
    egui::Window::new("Jellies (GPU)")
        .default_pos([10.0, 10.0])
        .default_width(300.0)
        .resizable(false)
        .show(ctx, |ui| {
            ui.label(format!("fps={:.0}  n={}  [GPU]", time.delta_secs().recip(), sim.solver.particle_count()));
            ui.separator();
            ui.add(egui::Slider::new(&mut p.hz, 5.0..=60.0).text("solver_hz"));
            ui.add(egui::Slider::new(&mut p.gravity, -3.0..=0.0).text("gravity"));
            ui.separator();
            ui.colored_label(egui::Color32::from_rgb(240, 133, 69), "NeoHookean (orange)");
            ui.add(egui::Slider::new(&mut p.neo_lambda, 1.0..=200.0).text("λ"));
            ui.add(egui::Slider::new(&mut p.neo_mu, 1.0..=400.0).text("µ"));
            ui.separator();
            ui.colored_label(egui::Color32::from_rgb(64, 199, 166), "Corotated (teal)");
            ui.add(egui::Slider::new(&mut p.cor_lambda, 1.0..=200.0).text("λ"));
            ui.add(egui::Slider::new(&mut p.cor_mu, 1.0..=400.0).text("µ"));
            ui.separator();
            ui.colored_label(egui::Color32::from_rgb(184, 102, 230), "Viscoelastic (purple)");
            ui.add(egui::Slider::new(&mut p.vis_lambda, 1.0..=200.0).text("λ"));
            ui.add(egui::Slider::new(&mut p.vis_mu, 1.0..=400.0).text("µ"));
            ui.add(egui::Slider::new(&mut p.vis_viscosity, 0.0..=5.0).text("viscosity"));
            ui.separator();
            ui.add(egui::Slider::new(&mut p.cursor_strength, 10.0..=1000.0).text("cursor force").logarithmic(true));
            ui.add(egui::Slider::new(&mut p.cursor_radius, 1.0..=15.0).text("cursor radius"));
            ui.label("LMB: push  RMB: pull  R: reset");
            if ui.button("Reset (R)").clicked() { *p = DEFAULTS; *sim = Sim::new(DEFAULTS); }
        });
}

fn p2w(pos: Vec2) -> Vec3 {
    let c = (pos - Vec2::splat(GRID as f32 * 0.5)) * PPC;
    Vec3::new(c.x, c.y, 0.0)
}

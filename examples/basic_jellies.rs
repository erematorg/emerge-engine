/// Elastic solids — three-body multi-material comparison.
///
/// Three blobs fall and interact:
///   Mat 0  NeoHookean   (orange) — Simo-Pister vol-dev split, standard elastic jelly
///   Mat 1  Corotated    (teal)   — linear corotated elastic, stiffer baseline
///   Mat 2  Viscoelastic (purple) — Kelvin-Voigt damped solid, soft tissue preset
///
/// Demonstrates: multi-material solver, material registry, per-body color, impulse cursor.
///   cargo run --example basic_jellies --features bevy_examples
use bevy::prelude::*;
use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass, egui};
use emerge::diagnostics::log_frame_full;
use emerge::{
    CorotatedMaterial, MpmSolver, NeoHookeanMaterial, SlipBoundary, SolverConfig, SpawnConfig,
    ViscoelasticMaterial,
};
use emerge::runtime::fixed_step::FixedStepController;
use glam::{IVec2, Vec2};

const GRID: usize = 64;
const DT: f32 = 0.1;
const PPC: f32 = 10.0;
const MAX_DT: f32 = 1.0 / 15.0;

const MAT_NEO: u32 = 0;
const MAT_COR: u32 = 1;
const MAT_VIS: u32 = 2;

#[derive(Resource, Clone, Copy, PartialEq)]
struct Params {
    hz: f32,
    gravity: f32,
    // NeoHookean
    neo_lambda: f32,
    neo_mu: f32,
    // Corotated
    cor_lambda: f32,
    cor_mu: f32,
    // Viscoelastic
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
    cursor_strength: 40.0,
    cursor_radius: 5.0,
};

#[derive(Resource)]
struct Sim {
    solver: MpmSolver,
    stepper: FixedStepController,
    prev: Params,
    frame: u64,
}

impl Sim {
    fn new(p: Params) -> Self {
        let config = SolverConfig {
            min_dt: 0.01,
            max_substeps_per_step: 8,
            ..SolverConfig::standard(GRID, DT, Vec2::new(0.0, p.gravity))
        };
        // Three blobs placed left / centre / right, same height → fall and settle.
        let spawn_neo = SpawnConfig {
            spacing: 0.5,
            box_size: IVec2::new(14, 14),
            box_center: Vec2::new(14.0, 50.0),
            material_id: MAT_NEO,
            precompute_initial_volumes: true,
            initial_velocity_scale: 0.0,
            ..SpawnConfig::for_solver(&config)
        };
        let spawn_cor = SpawnConfig {
            spacing: 0.5,
            box_size: IVec2::new(14, 14),
            box_center: Vec2::new(32.0, 50.0),
            material_id: MAT_COR,
            precompute_initial_volumes: true,
            initial_velocity_scale: 0.0,
            ..SpawnConfig::for_solver(&config)
        };
        let spawn_vis = SpawnConfig {
            spacing: 0.5,
            box_size: IVec2::new(14, 14),
            box_center: Vec2::new(50.0, 50.0),
            material_id: MAT_VIS,
            precompute_initial_volumes: true,
            initial_velocity_scale: 0.0,
            ..SpawnConfig::for_solver(&config)
        };

        let mut solver = MpmSolver::new(config, spawn_neo)
            .with_default_material(Box::new(NeoHookeanMaterial::new(p.neo_lambda, p.neo_mu)))
            .with_material(MAT_COR, Box::new(CorotatedMaterial::new(p.cor_lambda, p.cor_mu)))
            .with_material(MAT_VIS, Box::new(ViscoelasticMaterial::new(p.vis_lambda, p.vis_mu, p.vis_viscosity)))
            .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));
        let _ = solver.spawn_region(spawn_cor);
        let _ = solver.spawn_region(spawn_vis);

        Self {
            solver,
            stepper: FixedStepController::standard(DT, p.hz),
            prev: p,
            frame: 0,
        }
    }
}

fn main() {
    App::new()
        .insert_resource(ClearColor(Color::srgb(0.07, 0.06, 0.08)))
        .insert_resource(DEFAULTS)
        .insert_resource(Sim::new(DEFAULTS))
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "MLS-MPM Jellies — NeoHookean · Corotated · Viscoelastic".into(),
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

fn mat_color(id: u32) -> Color {
    match id {
        MAT_NEO => Color::srgb(0.94, 0.52, 0.27), // orange
        MAT_COR => Color::srgb(0.25, 0.78, 0.65), // teal
        MAT_VIS => Color::srgb(0.72, 0.40, 0.90), // purple
        _       => Color::WHITE,
    }
}

fn setup(mut commands: Commands, sim: Res<Sim>) {
    commands.spawn(Camera2d);
    for (i, p) in sim.solver.particles().iter().enumerate() {
        commands.spawn((
            Sprite::from_color(mat_color(p.material_id), Vec2::ONE),
            Transform { translation: p2w(p.x), ..default() },
            PVis(i),
        ));
    }
}

fn reset(keys: Res<ButtonInput<KeyCode>>, mut sim: ResMut<Sim>, mut p: ResMut<Params>) {
    if keys.just_pressed(KeyCode::KeyR) {
        *p = DEFAULTS;
        *sim = Sim::new(DEFAULTS);
    }
}

fn cursor(
    windows: Query<&Window>,
    cam: Query<(&Camera, &GlobalTransform)>,
    mb: Res<ButtonInput<MouseButton>>,
    mut sim: ResMut<Sim>,
    params: Res<Params>,
    time: Res<Time>,
) {
    if !mb.pressed(MouseButton::Left) && !mb.pressed(MouseButton::Right) {
        return;
    }
    let Ok(win) = windows.single() else { return };
    let Some(cp) = win.cursor_position() else { return };
    let Ok((cam, ct)) = cam.single() else { return };
    let Ok(wp) = cam.viewport_to_world_2d(ct, cp) else { return };
    let gp = wp / PPC + Vec2::splat(GRID as f32 * 0.5);
    let sign = if mb.pressed(MouseButton::Right) { -1.0 } else { 1.0 };
    let strength = params.cursor_strength;
    let radius = params.cursor_radius;
    let dt = time.delta_secs().min(MAX_DT);
    // apply_radial_impulse clamps to CFL limit — safe under any cursor_strength value.
    sim.solver.apply_radial_impulse(gp, radius, sign * strength * dt);
}

fn step(time: Res<Time>, mut sim: ResMut<Sim>, params: Res<Params>) {
    sim.solver.set_gravity(Vec2::new(0.0, params.gravity));
    sim.stepper.set_simulation_speed(params.hz * DT);
    let n = sim.stepper.steps_for_frame(time.delta_secs());
    if n == 0 { return; }
    if sim.prev != *params {
        sim.solver.set_default_material(Box::new(
            NeoHookeanMaterial::new(params.neo_lambda, params.neo_mu),
        ));
        sim.solver.set_material(MAT_COR, Box::new(
            CorotatedMaterial::new(params.cor_lambda, params.cor_mu),
        ));
        sim.solver.set_material(MAT_VIS, Box::new(
            ViscoelasticMaterial::new(params.vis_lambda, params.vis_mu, params.vis_viscosity),
        ));
        sim.prev = *params;
    }
    sim.solver.step_n(n);
    sim.frame += n as u64;
    let snap = sim.solver.diagnostics_snapshot();
    const LABELS: &[(u32, &str)] = &[(MAT_NEO, "neo"), (MAT_COR, "cor"), (MAT_VIS, "vis")];
    log_frame_full(sim.frame, DT, sim.solver.particles(), LABELS, &snap, 60);
}

fn sync(sim: Res<Sim>, mut q: Query<(&PVis, &mut Transform)>) {
    for (v, mut t) in &mut q {
        t.translation = p2w(sim.solver.particles().x[v.0]);
    }
}

fn ui(mut ctx: EguiContexts, mut p: ResMut<Params>, mut sim: ResMut<Sim>, time: Res<Time>) {
    let Ok(ctx) = ctx.ctx_mut() else { return };
    egui::Window::new("Jellies")
        .default_pos([10.0, 10.0])
        .default_width(300.0)
        .resizable(false)
        .show(ctx, |ui| {
            ui.label(format!(
                "fps={:.0}  n={}",
                time.delta_secs().recip(),
                sim.solver.particles().len(),
            ));
            ui.separator();
            ui.add(egui::Slider::new(&mut p.hz, 5.0..=60.0).text("solver_hz"));
            ui.add(egui::Slider::new(&mut p.gravity, -2.0..=2.0).text("gravity"));
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
            ui.add(egui::Slider::new(&mut p.cursor_strength, 5.0..=200.0).text("cursor force").logarithmic(true));
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

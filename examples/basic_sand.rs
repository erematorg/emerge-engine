/// Drucker-Prager sand — angle of repose comparison.
///
/// Two piles fall from the same height:
///   Mat 0  loose sand  (φ=20°, light yellow) — shallow repose angle
///   Mat 1  dense sand  (φ=40°, dark brown)   — steep repose angle
///
/// Demonstrates: Drucker-Prager plasticity, friction angle effect, multi-material,
///               color by friction hardening accumulator.
///   cargo run --example basic_sand --features bevy_examples
use bevy::prelude::*;
use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass, egui};
use emerge::diagnostics::log_frame_full;
use emerge::{MpmSolver, SandMaterial, SlipBoundary, SolverConfig, SpawnConfig};
use emerge::runtime::fixed_step::FixedStepController;
use glam::{IVec2, Vec2};

const GRID: usize = 64;
const DT: f32 = 0.1;
const PPC: f32 = 10.0;
const MAX_DT: f32 = 1.0 / 8.0;

const MAT_LOOSE: u32 = 0;
const MAT_DENSE: u32 = 1;

#[derive(Resource, Clone, Copy, PartialEq)]
struct Params {
    hz: f32,
    gravity: f32,
    loose_phi: f32,
    dense_phi: f32,
    lambda: f32,
    mu: f32,
    cursor_strength: f32,
    cursor_radius: f32,
}

const DEFAULTS: Params = Params {
    hz: 60.0,
    gravity: -0.3,
    loose_phi: 20.0,
    dense_phi: 40.0,
    lambda: 5000.0,
    mu: 3000.0,
    cursor_strength: 80.0,
    cursor_radius: 6.0,
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
            boundary_thickness: 3,
            max_substeps_per_step: 12,
            ..SolverConfig::standard(GRID, DT, Vec2::new(0.0, p.gravity))
        };
        // Left pile: loose sand
        let spawn_loose = SpawnConfig {
            spacing: 0.5,
            box_size: IVec2::new(18, 14),
            box_center: Vec2::new(17.0, 40.0),
            material_id: MAT_LOOSE,
            precompute_initial_volumes: true,
            initial_velocity_scale: 0.0,
            rng_seed: 11,
            ..SpawnConfig::for_solver(&config)
        };
        // Right pile: dense sand
        let spawn_dense = SpawnConfig {
            spacing: 0.5,
            box_size: IVec2::new(18, 14),
            box_center: Vec2::new(47.0, 40.0),
            material_id: MAT_DENSE,
            precompute_initial_volumes: true,
            initial_velocity_scale: 0.0,
            rng_seed: 22,
            ..SpawnConfig::for_solver(&config)
        };
        let mut solver = MpmSolver::new(config, spawn_loose)
            .with_default_material(Box::new(make_sand(p.lambda, p.mu, p.loose_phi)))
            .with_material(MAT_DENSE, Box::new(make_sand(p.lambda, p.mu, p.dense_phi)))
            .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));
        let _ = solver.spawn_region(spawn_dense);
        Self {
            solver,
            stepper: FixedStepController::standard(DT, p.hz),
            prev: p,
            frame: 0,
        }
    }
}

fn make_sand(lambda: f32, mu: f32, friction_deg: f32) -> SandMaterial {
    let mut m = SandMaterial::new(lambda, mu);
    m.friction_angle = friction_deg.to_radians();
    m
}

fn main() {
    App::new()
        .insert_resource(ClearColor(Color::srgb(0.08, 0.06, 0.04)))
        .insert_resource(DEFAULTS)
        .insert_resource(Sim::new(DEFAULTS))
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "MLS-MPM Sand — Angle of Repose".into(),
                resolution: (700u32, 700u32).into(),
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

fn sand_color(material_id: u32, q: f32) -> Color {
    let t = (q * 0.1).clamp(0.0, 1.0);
    if material_id == MAT_LOOSE {
        // light yellow → darker on stress
        Color::srgb(0.90 - t * 0.25, 0.78 - t * 0.28, 0.42 - t * 0.18)
    } else {
        // dark reddish-brown → deeper on stress
        Color::srgb(0.62 - t * 0.20, 0.42 - t * 0.18, 0.22 - t * 0.10)
    }
}

fn setup(mut commands: Commands, sim: Res<Sim>) {
    commands.spawn(Camera2d);
    for (i, p) in sim.solver.particles().iter().enumerate() {
        commands.spawn((
            Sprite::from_color(sand_color(p.material_id, p.friction_hardening), Vec2::ONE),
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
    let speed_cap = strength * 0.4;
    let dt = time.delta_secs().min(MAX_DT);
    sim.solver.particles_mut().for_each_mut(|p| {
        let d = p.x - gp;
        let dist = d.length();
        if dist < radius && dist > 1e-4 {
            p.v += (d / dist) * sign * strength * (1.0 - dist / radius) * dt;
            let s = p.v.length();
            if s > speed_cap { p.v *= speed_cap / s; }
        }
    });
}

fn step(time: Res<Time>, mut sim: ResMut<Sim>, params: Res<Params>) {
    sim.solver.set_gravity(Vec2::new(0.0, params.gravity));
    sim.stepper.set_simulation_speed(params.hz * DT);
    let n = sim.stepper.steps_for_frame(time.delta_secs());
    if n == 0 { return; }
    if sim.prev != *params {
        sim.solver.set_default_material(Box::new(
            make_sand(params.lambda, params.mu, params.loose_phi),
        ));
        sim.solver.set_material(MAT_DENSE, Box::new(
            make_sand(params.lambda, params.mu, params.dense_phi),
        ));
        sim.prev = *params;
    }
    sim.solver.step_n(n);
    sim.frame += n as u64;
    let snap = sim.solver.diagnostics_snapshot();
    const LABELS: &[(u32, &str)] = &[(MAT_LOOSE, "loose"), (MAT_DENSE, "dense")];
    log_frame_full(sim.frame, DT, sim.solver.particles(), LABELS, &snap, 60);
}

fn sync(sim: Res<Sim>, mut q: Query<(&PVis, &mut Transform, &mut Sprite)>) {
    for (v, mut t, mut s) in &mut q {
        let p = sim.solver.particles().get(v.0);
        t.translation = p2w(p.x);
        s.color = sand_color(p.material_id, p.friction_hardening);
    }
}

fn ui(mut ctx: EguiContexts, mut p: ResMut<Params>, mut sim: ResMut<Sim>, time: Res<Time>) {
    let Ok(ctx) = ctx.ctx_mut() else { return };
    egui::Window::new("Sand")
        .default_pos([10.0, 10.0])
        .default_width(280.0)
        .resizable(false)
        .show(ctx, |ui| {
            ui.label(format!(
                "fps={:.0}  n={}",
                time.delta_secs().recip(),
                sim.solver.particles().len(),
            ));
            ui.separator();
            ui.add(egui::Slider::new(&mut p.hz, 5.0..=60.0).text("solver_hz"));
            ui.add(egui::Slider::new(&mut p.gravity, -5.0..=0.0).text("gravity"));
            ui.separator();
            ui.label("Drucker-Prager friction angles");
            ui.add(
                egui::Slider::new(&mut p.loose_phi, 5.0..=60.0)
                    .text("loose φ (left)")
                    .suffix("°"),
            );
            ui.add(
                egui::Slider::new(&mut p.dense_phi, 5.0..=60.0)
                    .text("dense φ (right)")
                    .suffix("°"),
            );
            ui.label("↑ steeper φ → steeper pile slope");
            ui.separator();
            ui.label("Stiffness (shared)");
            ui.add(egui::Slider::new(&mut p.lambda, 1000.0..=100000.0).text("λ").logarithmic(true));
            ui.add(egui::Slider::new(&mut p.mu, 500.0..=80000.0).text("µ").logarithmic(true));
            ui.separator();
            ui.add(egui::Slider::new(&mut p.cursor_strength, 5.0..=500.0).text("cursor force").logarithmic(true));
            ui.add(egui::Slider::new(&mut p.cursor_radius, 1.0..=20.0).text("cursor radius"));
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

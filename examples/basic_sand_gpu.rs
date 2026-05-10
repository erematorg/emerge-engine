/// Drucker-Prager sand (GPU) — angle of repose comparison.
///
/// Two piles fall from the same height:
///   Mat 0  loose sand  (φ=20°, light yellow) — shallow repose angle
///   Mat 1  dense sand  (φ=40°, dark brown)   — steep repose angle
///
/// All plasticity runs in g2p.wgsl — no CPU roundtrip per substep.
///   cargo run --example basic_sand_gpu --features "bevy_examples,gpu"
use bevy::prelude::*;
use bevy::tasks::block_on;
use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass, egui};
use emerge::gpu::{GpuForceFieldEntry, GpuSolver};
use emerge::{MaterialRegistry, SandMaterial, SolverConfig, SpawnConfig, build_particles, log_frame_gpu};
use emerge::runtime::fixed_step::FixedStepController;
use glam::Vec2;

const GRID: usize = 96;
const DT: f32 = 0.05;
const PPC: f32 = 7.0;
const LABELS: &[(u32, &str)] = &[(0, "loose"), (1, "dense")];

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
    cursor_strength: 150.0,
    cursor_radius: 7.0,
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
            boundary_thickness: 3,
            max_substeps_per_step: 20,
            ..SolverConfig::standard(GRID, DT, Vec2::new(0.0, p.gravity))
        };
        let spawn = |center: Vec2, mat: u32| {
            SpawnConfig::for_solver(&config)
                .at(center).box_of(glam::IVec2::new(28, 20)).spacing(0.5).jitter(0.2).material(mat)
        };
        let mut particles = build_particles(&config, spawn(Vec2::new(26.0, 44.0), MAT_LOOSE));
        particles.extend(build_particles(&config, spawn(Vec2::new(70.0, 44.0), MAT_DENSE)));

        let mut registry = MaterialRegistry::with_default(Box::new(
            make_sand(p.lambda, p.mu, p.loose_phi),
        ));
        registry.insert(MAT_DENSE, Box::new(make_sand(p.lambda, p.mu, p.dense_phi)));

        let solver = block_on(GpuSolver::new(config, particles, registry));
        Self {
            solver,
            stepper: FixedStepController::standard(DT, p.hz),
            prev: p,
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
                title: "MLS-MPM Sand (GPU) — Angle of Repose".into(),
                resolution: (960u32, 700u32).into(),
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
        let color = sand_color(p.material_id);
        commands.spawn((
            PVis(i),
            Sprite { color, custom_size: Some(Vec2::ONE), ..default() },
            Transform::from_translation(p2w(p.x)),
        ));
    }
}

fn sand_color(mat: u32) -> Color {
    match mat {
        0 => Color::srgb(0.90, 0.82, 0.50), // loose — light yellow
        _ => Color::srgb(0.42, 0.28, 0.14), // dense — dark brown
    }
}

fn reset(keys: Res<ButtonInput<KeyCode>>, mut sim: ResMut<Sim>, mut p: ResMut<Params>,
         mut commands: Commands, vis: Query<Entity, With<PVis>>) {
    if keys.just_pressed(KeyCode::KeyR) {
        *p = DEFAULTS;
        *sim = Sim::new(DEFAULTS);
        for e in &vis { commands.entity(e).despawn(); }
        for (i, pt) in sim.solver.particles().iter().enumerate() {
            commands.spawn((
                PVis(i),
                Sprite { color: sand_color(pt.material_id), custom_size: Some(Vec2::ONE), ..default() },
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
        sim.solver.set_default_material(Box::new(make_sand(params.lambda, params.mu, params.loose_phi)));
        sim.solver.set_material(MAT_DENSE, Box::new(make_sand(params.lambda, params.mu, params.dense_phi)));
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
    egui::Window::new("Sand (GPU)")
        .default_pos([10.0, 10.0])
        .default_width(280.0)
        .resizable(false)
        .show(ctx, |ui| {
            ui.label(format!(
                "fps={:.0}  n={}  [GPU DP]",
                time.delta_secs().recip(),
                sim.solver.particle_count(),
            ));
            ui.separator();
            ui.add(egui::Slider::new(&mut p.hz, 5.0..=60.0).text("solver_hz"));
            ui.add(egui::Slider::new(&mut p.gravity, -5.0..=0.0).text("gravity"));
            ui.separator();
            ui.label("Drucker-Prager friction angles");
            ui.add(egui::Slider::new(&mut p.loose_phi, 5.0..=60.0).text("loose φ (left)").suffix("°"));
            ui.add(egui::Slider::new(&mut p.dense_phi, 5.0..=60.0).text("dense φ (right)").suffix("°"));
            ui.label("↑ steeper φ → steeper pile slope");
            ui.separator();
            ui.label("Stiffness (shared)");
            ui.add(egui::Slider::new(&mut p.lambda, 1000.0..=100000.0).text("λ").logarithmic(true));
            ui.add(egui::Slider::new(&mut p.mu, 500.0..=80000.0).text("µ").logarithmic(true));
            ui.separator();
            ui.add(egui::Slider::new(&mut p.cursor_strength, 10.0..=1000.0).text("cursor force").logarithmic(true));
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

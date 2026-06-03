/// Basic MPM creature — NeoHookean soft body with peristaltic muscle activation.
///
/// Traveling wave of vertical muscle contraction (ChainQueen / SoftZoo style):
///   segment activates → squats into floor → grips → neighboring segment slides forward.
/// Arrow keys steer / adjust wave speed.  Space pauses.  R resets.
///
///   cargo run --example basic_creature --features bevy_examples
use bevy::prelude::*;
use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass, egui};
use emerge::runtime::fixed_step::FixedStepController;
use emerge::{FrictionBoundary, MpmSolver, NeoHookeanMaterial, SolverConfig, SpawnConfig};
use glam::{IVec2, Vec2};
use std::f32::consts::TAU;

const GRID: usize = 64;
const DT: f32 = 0.1;
const PPC: f32 = 10.0;
const MAT_BODY: u32 = 0;
const MUSCLE_GROUPS: u32 = 8;

// Per-segment hue (HSV-ish) — vivid rainbow like SoftZoo
const GROUP_COLORS: [(f32, f32, f32); 8] = [
    (0.95, 0.30, 0.25), // red
    (0.95, 0.58, 0.15), // orange
    (0.90, 0.85, 0.10), // yellow
    (0.30, 0.85, 0.25), // green
    (0.15, 0.75, 0.90), // cyan
    (0.20, 0.35, 0.95), // blue
    (0.60, 0.20, 0.95), // violet
    (0.95, 0.20, 0.70), // magenta
];

fn group_color(group: u32, activation: f32) -> Color {
    let (r, g, b) = GROUP_COLORS[group.min(7) as usize];
    // Brighten when active — makes the muscle pulse visible
    let bright = 1.0 + 0.5 * activation;
    Color::srgb(
        (r * bright).min(1.0),
        (g * bright).min(1.0),
        (b * bright).min(1.0),
    )
}

#[derive(Resource, Clone, Copy)]
struct Params {
    hz: f32,
    wave_speed: f32,
    wave_amplitude: f32,
    steer: f32,
}

const DEFAULTS: Params = Params {
    hz: 60.0,
    wave_speed: 1.0,
    wave_amplitude: 0.9,
    steer: 0.0,
};

#[derive(Resource)]
struct Sim {
    solver: MpmSolver,
    body_range: std::ops::Range<usize>,
    time: f32,
    paused: bool,
    stepper: FixedStepController,
    frame: u64,
}

impl Sim {
    fn new(p: Params) -> Self {
        // Soft NeoHookean: lambda=5, mu=10 (same grid-unit scale as basic_jellies).
        // active_stress_coeff=25 = 2.5× mu → large visible deformation per activation.
        let mut body = NeoHookeanMaterial::new(5.0, 10.0);
        body.active_stress_coeff = 25.0;

        let config = SolverConfig {
            min_dt: 0.01,
            max_substeps_per_step: 8,
            ..SolverConfig::standard(GRID, DT, Vec2::new(0.0, -0.3))
        };

        let body_center = Vec2::new(32.0, 20.0);
        let spawn = SpawnConfig {
            spacing: 0.5,
            box_size: IVec2::new(24, 6),
            box_center: body_center,
            material_id: MAT_BODY,
            precompute_initial_volumes: true,
            ..SpawnConfig::for_solver(&config)
        };

        let mut solver = MpmSolver::new(config, spawn)
            .with_default_material(Box::new(body))
            .with_boundary(Box::new(FrictionBoundary::new(4, 0.65)));

        let body_range = 0..solver.particles().len();

        // Assign muscle groups along X axis + set activation direction to Y (vertical).
        // Vertical activation: each segment squats into the floor → grip → peristaltic crawl.
        let body_left = body_center.x - 12.0;
        {
            let particles = solver.particles_mut();
            for i in body_range.clone() {
                let t = ((particles.x[i].x - body_left) / 24.0).clamp(0.0, 1.0);
                particles.muscle_group_id[i] = (t * MUSCLE_GROUPS as f32) as u32;
                // Vertical activation (Y) → segment squats / lifts → ground interaction
                particles.activation_dir[i] = Vec2::Y;
            }
        }

        Sim {
            solver,
            body_range,
            time: 0.0,
            paused: false,
            stepper: FixedStepController::standard(DT, p.hz),
            frame: 0,
        }
    }
}

#[derive(Component)]
struct PVis(usize);

#[derive(Component)]
struct CreatureCam;

fn main() {
    App::new()
        .insert_resource(ClearColor(Color::srgb(0.07, 0.06, 0.08)))
        .insert_resource(DEFAULTS)
        .insert_resource(Sim::new(DEFAULTS))
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "emerge — basic creature".into(),
                resolution: (800u32, 600u32).into(),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(EguiPlugin::default())
        .add_systems(Startup, setup)
        .add_systems(Update, (input, step, sync).chain())
        .add_systems(EguiPrimaryContextPass, ui)
        .run();
}

fn setup(mut commands: Commands, sim: Res<Sim>) {
    commands.spawn((Camera2d, CreatureCam));
    for (i, p) in sim.solver.particles().iter().enumerate() {
        commands.spawn((
            Sprite::from_color(group_color(p.muscle_group_id, 0.0), Vec2::ONE),
            Transform {
                translation: p2w(p.x),
                ..default()
            },
            PVis(i),
        ));
    }
}

fn input(keys: Res<ButtonInput<KeyCode>>, mut sim: ResMut<Sim>, mut p: ResMut<Params>) {
    if keys.just_pressed(KeyCode::Space) {
        sim.paused = !sim.paused;
    }
    if keys.just_pressed(KeyCode::KeyR) {
        *sim = Sim::new(*p);
    }

    if keys.pressed(KeyCode::ArrowUp) {
        p.wave_speed = (p.wave_speed + 0.04).min(6.0);
    }
    if keys.pressed(KeyCode::ArrowDown) {
        p.wave_speed = (p.wave_speed - 0.04).max(0.1);
    }
    if keys.pressed(KeyCode::ArrowLeft) {
        p.steer = (p.steer - 0.05).max(-2.0);
    }
    if keys.pressed(KeyCode::ArrowRight) {
        p.steer = (p.steer + 0.05).min(2.0);
    }
    if !keys.pressed(KeyCode::ArrowLeft) && !keys.pressed(KeyCode::ArrowRight) {
        p.steer *= 0.88;
    }
}

fn step(mut sim: ResMut<Sim>, p: Res<Params>, time: Res<Time>) {
    if sim.paused {
        return;
    }
    let n = sim.stepper.steps_for_frame(time.delta_secs());
    if n == 0 {
        return;
    }

    for _ in 0..n {
        sim.time += DT;
        let t = sim.time;
        let body_range = sim.body_range.clone();
        let particles = sim.solver.particles_mut();
        for i in body_range {
            let group = particles.muscle_group_id[i] as f32;
            // Phase increases from group 0 (left/tail) to group N (right/head).
            // sin(ωt - phase): crest moves left→right → creature crawls right.
            // Steer shifts phase asymmetrically → turns.
            let phase = group / MUSCLE_GROUPS as f32 * TAU;
            let wave = (TAU * p.wave_speed * t - phase + p.steer).sin();
            // Map [-1,1] → [0,1]: activation is always non-negative (muscles only pull)
            particles.activation[i] = p.wave_amplitude * (wave * 0.5 + 0.5);
        }
        sim.solver.step();
    }

    sim.frame += n as u64;
}

fn sync(
    sim: Res<Sim>,
    mut sprites: Query<(&PVis, &mut Transform, &mut Sprite)>,
    mut cam: Query<&mut Transform, (With<CreatureCam>, Without<PVis>)>,
) {
    let p = sim.solver.particles();
    for (v, mut t, mut s) in &mut sprites {
        t.translation = p2w(p.x[v.0]);
        s.color = group_color(p.muscle_group_id[v.0], p.activation[v.0]);
    }

    // Camera follows creature centroid on X axis only
    let n = sim.body_range.len() as f32;
    let cx = sim.body_range.clone().fold(0.0_f32, |a, i| a + p.x[i].x) / n;
    if let Ok(mut cam_t) = cam.single_mut() {
        cam_t.translation.x = (cx - GRID as f32 * 0.5) * PPC;
    }
}

fn ui(mut ctx: EguiContexts, mut p: ResMut<Params>, sim: Res<Sim>, time: Res<Time>) {
    let Ok(ctx) = ctx.ctx_mut() else { return };

    let particles = sim.solver.particles();
    let n = sim.body_range.len() as f32;
    let centroid = sim
        .body_range
        .clone()
        .fold(Vec2::ZERO, |a, i| a + particles.x[i])
        / n;
    let vel = sim
        .body_range
        .clone()
        .fold(Vec2::ZERO, |a, i| a + particles.v[i])
        / n;

    egui::Window::new("Creature")
        .default_pos([10.0, 10.0])
        .default_width(260.0)
        .resizable(false)
        .show(ctx, |ui| {
            ui.label(format!(
                "fps={:.0}  n={}  {}",
                time.delta_secs().recip(),
                particles.len(),
                if sim.paused { "PAUSED" } else { "RUNNING" },
            ));
            ui.separator();
            ui.label(format!(
                "pos ({:.1}, {:.1})  vel ({:+.3}, {:+.3})",
                centroid.x, centroid.y, vel.x, vel.y
            ));
            ui.separator();
            ui.add(egui::Slider::new(&mut p.wave_speed, 0.1..=6.0).text("wave speed"));
            ui.add(egui::Slider::new(&mut p.wave_amplitude, 0.0..=1.0).text("amplitude"));
            ui.add(egui::Slider::new(&mut p.steer, -2.0..=2.0).text("steer"));
            ui.separator();
            ui.label("[↑↓] wave speed  [←→] steer  [space] pause  [r] reset");
        });
}

fn p2w(pos: Vec2) -> Vec3 {
    let c = (pos - Vec2::splat(GRID as f32 * 0.5)) * PPC;
    Vec3::new(c.x, c.y, 0.0)
}

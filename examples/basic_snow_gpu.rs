/// Two snowballs colliding — GPU MLS-MPM, SVD plasticity in g2p.wgsl.
///
/// Ball A (blue): launched right. Ball B (white): launched left.
/// Physics constants from MPM2D/constants.h — canonical snowball collision reference.
///
///   cargo run --example basic_snow_gpu --features "bevy_examples,gpu"
use bevy::prelude::*;
use bevy::tasks::block_on;
use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass, egui};
use emerge::gpu::GpuSolver;
use emerge::runtime::fixed_step::FixedStepController;
use emerge::solver::density::estimate_initial_particle_volumes;
use emerge::solver::{MaterialRegistry, SnowMaterial, SolverConfig, SpawnConfig};
use emerge::state::{grid::Grid, particle::Particle};
use glam::{IVec2, Mat2, Vec2};

const GRID: usize = 64;
const DT: f32 = 0.1;
const PPC: f32 = 14.0;
const PDIAM: f32 = 14.0 * 0.5 * 1.6; // 11.2 — solid circles at spacing=0.5
const MAX_DT: f32 = 1.0 / 15.0;

const BALL_R: f32 = 10.0;
const BALL_A: Vec2 = Vec2::new(14.0, 36.0);
const BALL_B: Vec2 = Vec2::new(50.0, 28.0);

#[derive(Resource, Clone, Copy, PartialEq)]
struct Params {
    hz: f32,
    gravity: f32,
    speed: f32,
    lambda: f32,
    mu: f32,
    xi: f32,
    theta_c: f32,
    theta_s: f32,
}
const DEFAULTS: Params = Params {
    hz: 5.0,
    gravity: -9.81,
    speed: 40.0,
    // MPM2D: E=1.4e5, nu=0.2 → lambda=38889, mu=58333
    lambda: 38889.0,
    mu: 58333.0,
    xi: 10.0,
    theta_c: 0.02,
    theta_s: 0.006,
};

#[derive(Resource)]
struct Sim {
    solver: GpuSolver,
    particles: Vec<Particle>,
    stepper: FixedStepController,
    prev: Params,
}

impl Sim {
    fn new(p: Params) -> Self {
        let config = SolverConfig::standard(GRID, DT, Vec2::new(0.0, p.gravity));
        let spawn = SpawnConfig {
            spacing: 0.5,
            box_size: IVec2::new(58, 58),
            initial_velocity_scale: 0.0,
            rng_seed: 7,
            ..SpawnConfig::for_solver(&config)
        };
        let mut particles = spawn_balls(&config, &spawn, p.speed);
        estimate_initial_particle_volumes(&mut particles, &mut Grid::new(GRID));
        let solver = block_on(GpuSolver::new(
            config,
            &particles,
            MaterialRegistry::with_default(Box::new(make_snow(&p))),
        ));
        Self {
            solver,
            particles,
            stepper: FixedStepController::standard(DT, p.hz),
            prev: p,
        }
    }
}

fn make_snow(p: &Params) -> SnowMaterial {
    SnowMaterial::new(p.lambda, p.mu, p.xi, p.theta_c, p.theta_s, 0.6, 1.05)
}

fn main() {
    App::new()
        .insert_resource(ClearColor(Color::srgb(0.05, 0.07, 0.10)))
        .insert_resource(DEFAULTS)
        .insert_resource(Sim::new(DEFAULTS))
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "MLS-MPM Snow (GPU) — SVD in WGSL".into(),
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
    for (i, p) in sim.particles.iter().enumerate() {
        let ball_a = (p.x - BALL_A).length() <= BALL_R + 0.5;
        let color = if ball_a {
            Color::srgb(0.40, 0.70, 1.00)
        } else {
            Color::srgb(0.95, 0.97, 1.00)
        };
        commands.spawn((
            Sprite::from_color(color, Vec2::splat(PDIAM)),
            Transform {
                translation: p2w(p.x),
                ..default()
            },
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
    time: Res<Time>,
) {
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
    let sign = if mb.pressed(MouseButton::Right) {
        -1.0
    } else {
        1.0
    };
    let dt = time.delta_secs().min(MAX_DT);
    for p in &mut sim.particles {
        let d = p.x - gp;
        let dist = d.length();
        if dist < 6.0 && dist > 1e-4 {
            p.v += (d / dist) * sign * 200.0 * (1.0 - dist / 6.0) * dt;
            let s = p.v.length();
            if s > 60.0 {
                p.v *= 60.0 / s;
            }
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
            .set_default_material(Box::new(make_snow(&params)));
        sim.prev = *params;
    }
    let sim = sim.as_mut();
    for _ in 0..n {
        sim.solver.step_frame(&mut sim.particles);
    }
}

fn sync(sim: Res<Sim>, mut q: Query<(&PVis, &mut Transform)>) {
    for (v, mut t) in &mut q {
        t.translation = p2w(sim.particles[v.0].x);
    }
}

fn ui(mut ctx: EguiContexts, mut p: ResMut<Params>, mut sim: ResMut<Sim>, time: Res<Time>) {
    let Ok(ctx) = ctx.ctx_mut() else { return };
    egui::Window::new("Snow (GPU)")
        .default_pos([10.0, 10.0])
        .default_width(300.0)
        .resizable(false)
        .show(ctx, |ui| {
            ui.label(format!(
                "fps={:.0}  n={}  [GPU SVD]",
                time.delta_secs().recip(),
                sim.particles.len()
            ));
            ui.separator();
            ui.add(egui::Slider::new(&mut p.hz, 1.0..=60.0).text("solver_hz"));
            ui.add(egui::Slider::new(&mut p.gravity, -20.0..=0.0).text("gravity"));
            ui.add(egui::Slider::new(&mut p.speed, 1.0..=80.0).text("launch speed (→ reset)"));
            ui.separator();
            ui.label("Stiffness (MPM2D: λ=38889 µ=58333)");
            ui.add(egui::Slider::new(&mut p.lambda, 100.0..=100_000.0).text("λ"));
            ui.add(egui::Slider::new(&mut p.mu, 100.0..=200_000.0).text("µ"));
            ui.separator();
            ui.label("Snow plasticity (Stomakhin 2013)");
            ui.add(egui::Slider::new(&mut p.xi, 0.0..=20.0).text("xi"));
            ui.add(
                egui::Slider::new(&mut p.theta_c, 0.001..=0.5)
                    .logarithmic(true)
                    .text("theta_c"),
            );
            ui.add(
                egui::Slider::new(&mut p.theta_s, 0.001..=0.1)
                    .logarithmic(true)
                    .text("theta_s"),
            );
            ui.separator();
            ui.label("LMB: push  RMB: pull  R: reset");
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

fn spawn_balls(config: &SolverConfig, spawn: &SpawnConfig, speed: f32) -> Vec<Particle> {
    let half = spawn.box_size.as_vec2() * 0.5;
    let (lo, hi) = (spawn.box_center - half, spawn.box_center + half);
    let mut s = spawn.rng_seed;
    let mut out = Vec::new();
    let mut i = lo.x;
    while i < hi.x {
        let mut j = lo.y;
        while j < hi.y {
            let pos = Vec2::new(i, j);
            lcg(&mut s);
            lcg(&mut s); // consume RNG deterministically
            let in_a = (pos - BALL_A).length() <= BALL_R;
            let in_b = (pos - BALL_B).length() <= BALL_R;
            if in_a || in_b {
                let vel = if in_a {
                    Vec2::new(speed, 0.0)
                } else {
                    Vec2::new(-speed, 0.0)
                };
                out.push(Particle {
                    x: pos,
                    v: vel,
                    affine: Mat2::ZERO,
                    deformation_gradient: Mat2::IDENTITY,
                    mass: config.particle_mass,
                    initial_volume: config.default_initial_volume,
                    volume: config.default_initial_volume,
                    density: config.particle_mass / config.default_initial_volume,
                    material_id: 0,
                    plastic_jacobian: 1.0,
                    elastic_hardening: 1.0,
                    plastic_hardening: 0.0,
                    log_vol_gain: 0.0,
                    _pad: [0.0; 3],
                });
            }
            j += spawn.spacing;
        }
        i += spawn.spacing;
    }
    out
}

fn lcg(s: &mut u32) -> f32 {
    *s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *s as f32 / (u32::MAX as f32 + 1.0)
}

/// Drucker-Prager sand — GPU MLS-MPM, plasticity in g2p.wgsl.
///   cargo run --example basic_sand_gpu --features "bevy_examples,gpu"
use bevy::prelude::*;
use bevy::tasks::block_on;
use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass, egui};
use emerge::gpu::GpuSolver;
use emerge::runtime::fixed_step::FixedStepController;
use emerge::solver::density::estimate_initial_particle_volumes;
use emerge::solver::{MaterialRegistry, SandMaterial, SolverConfig, SpawnConfig};
use emerge::state::{grid::Grid, particle::Particle};
use glam::{IVec2, Mat2, Vec2};

const GRID: usize = 80;
const DT: f32 = 0.05;
const PPC: f32 = 7.0;
const MAX_DT: f32 = 1.0 / 15.0;

#[derive(Resource, Clone, Copy, PartialEq)]
struct Params {
    hz: f32,
    gravity: f32,
    lambda: f32,
    mu: f32,
    friction_deg: f32,
}
const DEFAULTS: Params = Params {
    hz: 30.0,
    gravity: -0.3,
    lambda: 1000.0,
    mu: 500.0,
    friction_deg: 35.0,
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
        let config = SolverConfig {
            boundary_thickness: 3,
            max_substeps_per_step: 20,
            ..SolverConfig::standard(GRID, DT, Vec2::new(0.0, p.gravity))
        };
        let spawn = SpawnConfig {
            spacing: 0.5,
            box_size: IVec2::new(40, 20),
            box_center: Vec2::new(40.0, 66.0),
            initial_velocity_scale: 0.0,
            rng_seed: 42,
            ..SpawnConfig::default()
        };
        let mut particles = spawn_block(&config, &spawn);
        estimate_initial_particle_volumes(&mut particles, &mut Grid::new(GRID));
        let solver = block_on(GpuSolver::new(
            config,
            &particles,
            MaterialRegistry::with_default(Box::new(make_sand(&p))),
        ));
        Self {
            solver,
            particles,
            stepper: FixedStepController::standard(DT, p.hz),
            prev: p,
        }
    }
}

fn make_sand(p: &Params) -> SandMaterial {
    let mut m = SandMaterial::new(p.lambda, p.mu);
    m.friction_angle = p.friction_deg.to_radians();
    m
}

fn main() {
    App::new()
        .insert_resource(ClearColor(Color::srgb(0.08, 0.06, 0.04)))
        .insert_resource(DEFAULTS)
        .insert_resource(Sim::new(DEFAULTS))
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "MLS-MPM Sand (GPU) — Drucker-Prager".into(),
                resolution: (800u32, 800u32).into(),
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
        commands.spawn((
            Sprite::from_color(sand_color(p.plastic_hardening), Vec2::ONE),
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
            p.v += (d / dist) * sign * 80.0 * (1.0 - dist / 6.0) * dt;
            let s = p.v.length();
            if s > 30.0 {
                p.v *= 30.0 / s;
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
            .set_default_material(Box::new(make_sand(&params)));
        sim.prev = *params;
    }
    let sim = sim.as_mut();
    for _ in 0..n {
        sim.solver.step_frame(&mut sim.particles);
    }
}

fn sync(sim: Res<Sim>, mut q: Query<(&PVis, &mut Transform, &mut Sprite)>) {
    for (v, mut t, mut s) in &mut q {
        let p = &sim.particles[v.0];
        t.translation = p2w(p.x);
        s.color = sand_color(p.plastic_hardening);
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
                sim.particles.len()
            ));
            ui.separator();
            ui.add(egui::Slider::new(&mut p.hz, 5.0..=60.0).text("solver_hz"));
            ui.add(egui::Slider::new(&mut p.gravity, -5.0..=0.0).text("gravity"));
            ui.separator();
            ui.label("Drucker-Prager sand");
            ui.add(
                egui::Slider::new(&mut p.friction_deg, 1.0..=50.0)
                    .text("friction φ")
                    .suffix("°"),
            );
            ui.label("↑ angle of repose. Dry sand ≈ 35°");
            ui.add(egui::Slider::new(&mut p.lambda, 100.0..=10000.0).text("λ"));
            ui.add(egui::Slider::new(&mut p.mu, 50.0..=5000.0).text("µ"));
            ui.separator();
            ui.label("LMB: push  RMB: pull  R: reset");
            if ui.button("Reset (R)").clicked() {
                *p = DEFAULTS;
                *sim = Sim::new(DEFAULTS);
            }
        });
}

fn sand_color(q: f32) -> Color {
    let t = (q * 0.1).clamp(0.0, 1.0);
    Color::srgb(0.85 - t * 0.25, 0.72 - t * 0.30, 0.40 - t * 0.20)
}

fn p2w(pos: Vec2) -> Vec3 {
    let c = (pos - Vec2::splat(GRID as f32 * 0.5)) * PPC;
    Vec3::new(c.x.round(), c.y.round(), 0.0)
}

fn spawn_block(config: &SolverConfig, spawn: &SpawnConfig) -> Vec<Particle> {
    let half = spawn.box_size.as_vec2() * 0.5;
    let (lo, hi) = (spawn.box_center - half, spawn.box_center + half);
    let mut out = Vec::new();
    let mut i = lo.x;
    while i < hi.x {
        let mut j = lo.y;
        while j < hi.y {
            out.push(Particle {
                x: Vec2::new(i, j),
                v: Vec2::ZERO,
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
            j += spawn.spacing;
        }
        i += spawn.spacing;
    }
    out
}

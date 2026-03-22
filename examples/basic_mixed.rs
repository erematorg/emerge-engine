/// Fluid + jelly in one domain — CPU MLS-MPM, two materials split by x-position.
///   cargo run --example basic_mixed --features bevy_examples
use bevy::prelude::*;
use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass, egui};
use emerge::runtime::fixed_step::FixedStepController;
use emerge::solver::{
    MpmSolver, NeoHookeanMaterial, NewtonianFluidMaterial, SlipBoundary, SolverConfig, SpawnConfig,
};
use glam::{IVec2, Vec2};

const GRID: usize = 64;
const DT: f32 = 0.1;
const PPC: f32 = 10.0;
const MAX_DT: f32 = 1.0 / 15.0;
const JELLY_ID: u32 = 1;

#[derive(Resource, Clone, Copy, PartialEq)]
struct Params {
    hz: f32,
    gravity: f32,
    split_x: f32,
    f_density: f32,
    f_viscosity: f32,
    f_eos_k: f32,
    f_eos_p: f32,
    j_lambda: f32,
    j_mu: f32,
}
const DEFAULTS: Params = Params {
    hz: 30.0,
    gravity: -0.3,
    split_x: 32.0,
    f_density: 4.0,
    f_viscosity: 0.1,
    f_eos_k: 10.0,
    f_eos_p: 4.0,
    j_lambda: 10.0,
    j_mu: 20.0,
};

#[derive(Resource)]
struct Sim {
    solver: MpmSolver,
    stepper: FixedStepController,
    prev: Params,
}

impl Sim {
    fn new(p: Params) -> Self {
        let config = SolverConfig {
            min_dt: 0.01,
            max_substeps_per_step: 8,
            recompute_density_each_step: true,
            ..SolverConfig::standard(GRID, DT, Vec2::new(0.0, p.gravity))
        };
        let spawn = SpawnConfig {
            spacing: 0.5,
            box_size: IVec2::new(32, 32),
            initial_velocity_scale: 0.0,
            ..SpawnConfig::for_solver(&config)
        };
        let sx = p.split_x;
        let solver = MpmSolver::new(config, spawn)
            .with_default_material(Box::new(make_fluid(&p)))
            .with_material(
                JELLY_ID,
                Box::new(NeoHookeanMaterial::new(p.j_lambda, p.j_mu)),
            )
            .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)))
            .with_particle_materials_by_position(move |pos| if pos.x < sx { 0 } else { JELLY_ID });
        Self {
            solver,
            stepper: FixedStepController::standard(DT, p.hz),
            prev: p,
        }
    }
}

fn make_fluid(p: &Params) -> NewtonianFluidMaterial {
    NewtonianFluidMaterial::new(p.f_density, p.f_viscosity, p.f_eos_k, p.f_eos_p)
}

fn mat_color(id: u32) -> Color {
    if id == JELLY_ID {
        Color::srgb(0.95, 0.50, 0.22)
    } else {
        Color::srgb(0.22, 0.64, 0.95)
    }
}

fn main() {
    App::new()
        .insert_resource(ClearColor(Color::srgb(0.06, 0.06, 0.08)))
        .insert_resource(DEFAULTS)
        .insert_resource(Sim::new(DEFAULTS))
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "MLS-MPM Mixed — Fluid + Jelly".into(),
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
            Sprite::from_color(mat_color(p.material_id), Vec2::ONE),
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
    for p in sim.solver.particles_mut() {
        let d = p.x - gp;
        let dist = d.length();
        if dist < 5.0 && dist > 1e-4 {
            p.v += (d / dist) * sign * 40.0 * (1.0 - dist / 5.0) * dt;
            let s = p.v.length();
            if s > 20.0 {
                p.v *= 20.0 / s;
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
        let sx = params.split_x;
        sim.solver
            .set_default_material(Box::new(make_fluid(&params)));
        sim.solver.set_material(
            JELLY_ID,
            Box::new(NeoHookeanMaterial::new(params.j_lambda, params.j_mu)),
        );
        sim.solver.assign_particle_materials_by_position(
            move |pos| if pos.x < sx { 0 } else { JELLY_ID },
        );
        sim.prev = *params;
    }
    sim.solver.step_n(n);
}

fn sync(sim: Res<Sim>, mut q: Query<(&PVis, &mut Transform, &mut Sprite)>) {
    for (v, mut t, mut s) in &mut q {
        let p = &sim.solver.particles()[v.0];
        t.translation = p2w(p.x);
        s.color = mat_color(p.material_id);
    }
}

fn ui(mut ctx: EguiContexts, mut p: ResMut<Params>, mut sim: ResMut<Sim>, time: Res<Time>) {
    let Ok(ctx) = ctx.ctx_mut() else { return };
    egui::Window::new("Mixed")
        .default_pos([10.0, 10.0])
        .default_width(300.0)
        .resizable(false)
        .show(ctx, |ui| {
            ui.label(format!(
                "fps={:.0}  n={}",
                time.delta_secs().recip(),
                sim.solver.particles().len()
            ));
            ui.separator();
            ui.add(egui::Slider::new(&mut p.hz, 5.0..=60.0).text("solver_hz"));
            ui.add(egui::Slider::new(&mut p.gravity, -2.0..=2.0).text("gravity"));
            ui.add(egui::Slider::new(&mut p.split_x, 8.0..=56.0).text("split_x"));
            ui.separator();
            ui.label("Fluid (left)");
            ui.add(egui::Slider::new(&mut p.f_density, 1.0..=12.0).text("rho0"));
            ui.add(egui::Slider::new(&mut p.f_viscosity, 0.0..=20.0).text("viscosity"));
            ui.add(egui::Slider::new(&mut p.f_eos_k, 1.0..=100.0).text("eos_k"));
            ui.add(egui::Slider::new(&mut p.f_eos_p, 1.0..=8.0).text("eos_p"));
            ui.separator();
            ui.label("Jelly (right)");
            ui.add(egui::Slider::new(&mut p.j_lambda, 1.0..=120.0).text("λ"));
            ui.add(egui::Slider::new(&mut p.j_mu, 1.0..=240.0).text("µ"));
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

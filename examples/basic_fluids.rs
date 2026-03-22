/// Newtonian fluid — CPU MLS-MPM, Tait EOS + deviatoric viscosity.
///   cargo run --example basic_fluids --features bevy_examples
use bevy::prelude::*;
use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass, egui};
use emerge::runtime::fixed_step::FixedStepController;
use emerge::solver::{
    MpmSolver, NewtonianFluidMaterial, PredictiveBoundary, SolverConfig, SpawnConfig,
};
use glam::{IVec2, Vec2};

const GRID: usize = 64;
const DT: f32 = 0.1;
const PPC: f32 = 10.0;
const MAX_DT: f32 = 1.0 / 15.0;

#[derive(Resource, Clone, Copy, PartialEq)]
struct Params {
    hz: f32,
    gravity: f32,
    rest_density: f32,
    viscosity: f32,
    eos_stiffness: f32,
    eos_power: f32,
    pressure_floor: f32,
    wall_min: f32,
    vel_damp: f32,
    aff_damp: f32,
}
const DEFAULTS: Params = Params {
    hz: 30.0,
    gravity: -0.3,
    rest_density: 4.0,
    viscosity: 0.1,
    eos_stiffness: 10.0,
    eos_power: 4.0,
    pressure_floor: -0.1,
    wall_min: 3.0,
    vel_damp: 1.0,
    aff_damp: 1.0,
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
        let solver = MpmSolver::new(config, spawn)
            .with_default_material(Box::new(make_fluid(&p)))
            .with_boundary(Box::new(PredictiveBoundary::new(
                config.boundary_thickness,
                p.wall_min,
            )));
        Self {
            solver,
            stepper: FixedStepController::standard(DT, p.hz),
            prev: p,
        }
    }
}

fn make_fluid(p: &Params) -> NewtonianFluidMaterial {
    let mut m =
        NewtonianFluidMaterial::new(p.rest_density, p.viscosity, p.eos_stiffness, p.eos_power);
    m.pressure_floor = p.pressure_floor;
    m.velocity_damping = p.vel_damp;
    m.affine_damping = p.aff_damp;
    m
}

fn main() {
    App::new()
        .insert_resource(ClearColor(Color::srgb(0.05, 0.08, 0.11)))
        .insert_resource(DEFAULTS)
        .insert_resource(Sim::new(DEFAULTS))
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "MLS-MPM Fluids".into(),
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
            Sprite::from_color(Color::srgb(0.23, 0.66, 0.96), Vec2::ONE),
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
        sim.solver
            .set_default_material(Box::new(make_fluid(&params)));
        sim.solver
            .set_boundary_condition(Box::new(PredictiveBoundary::new(2, params.wall_min)));
        sim.prev = *params;
    }
    sim.solver.step_n(n);
}

fn sync(sim: Res<Sim>, mut q: Query<(&PVis, &mut Transform)>) {
    for (v, mut t) in &mut q {
        t.translation = p2w(sim.solver.particles()[v.0].x);
    }
}

fn ui(mut ctx: EguiContexts, mut p: ResMut<Params>, mut sim: ResMut<Sim>, time: Res<Time>) {
    let Ok(ctx) = ctx.ctx_mut() else { return };
    egui::Window::new("Fluids")
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
            ui.separator();
            ui.label("Tait EOS fluid");
            ui.add(egui::Slider::new(&mut p.rest_density, 1.0..=12.0).text("rho0"));
            ui.add(egui::Slider::new(&mut p.viscosity, 0.0..=20.0).text("viscosity"));
            ui.add(egui::Slider::new(&mut p.eos_stiffness, 1.0..=100.0).text("eos_k"));
            ui.add(egui::Slider::new(&mut p.eos_power, 1.0..=8.0).text("eos_p"));
            ui.add(egui::Slider::new(&mut p.pressure_floor, -1.0..=0.0).text("pressure_floor"));
            ui.separator();
            ui.label("Damping");
            ui.add(egui::Slider::new(&mut p.vel_damp, 0.98..=1.0).text("vel_damp"));
            ui.add(egui::Slider::new(&mut p.aff_damp, 0.98..=1.0).text("aff_damp"));
            ui.add(egui::Slider::new(&mut p.wall_min, 2.0..=6.0).text("wall_min"));
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

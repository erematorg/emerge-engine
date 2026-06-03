/// Viscoplastic fluids — Newtonian water + Bingham mud comparison.
///
/// Two separate fluid bodies:
///   Mat 0  Newtonian water (blue)  — Tait EOS + deviatoric viscosity, dam-break column
///   Mat 1  Bingham mud    (brown)  — viscoplastic with yield stress, compact blob
///
/// Demonstrates: multi-material fluid, Tait EOS, Bingham yield stress, surface tension.
///   cargo run --example basic_fluids --features bevy_examples
use bevy::prelude::*;
use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass, egui};
use emerge::diagnostics::log_frame_full;
use emerge::runtime::fixed_step::FixedStepController;
use emerge::{
    BinghamFluidMaterial, MpmSolver, NewtonianFluidMaterial, SlipBoundary, SolverConfig,
    SpawnConfig,
};
use glam::{IVec2, Vec2};

const GRID: usize = 64;
const DT: f32 = 0.1;
const PPC: f32 = 10.0;
const MAX_DT: f32 = 1.0 / 15.0;

const MAT_WATER: u32 = 0;
const MAT_MUD: u32 = 1;

// Water: tall left column — classic dam break.
const WATER_CENTER: Vec2 = Vec2::new(11.0, 30.0);
const WATER_SIZE: IVec2 = IVec2::new(14, 52);

// Mud: compact blob on the right — falls and barely flows (yield stress).
const MUD_CENTER: Vec2 = Vec2::new(50.0, 38.0);
const MUD_SIZE: IVec2 = IVec2::new(16, 18);

#[derive(Resource, Clone, Copy, PartialEq)]
struct Params {
    hz: f32,
    gravity: f32,
    // Water
    water_viscosity: f32,
    water_stiffness: f32,
    water_surface_tension: f32,
    // Mud
    mud_viscosity: f32,
    mud_yield_stress: f32,
    cursor_strength: f32,
    cursor_radius: f32,
}

const DEFAULTS: Params = Params {
    hz: 60.0,
    gravity: -0.3,
    water_viscosity: 0.1,
    water_stiffness: 10.0,
    water_surface_tension: 0.0,
    mud_viscosity: 8.0,
    mud_yield_stress: 4.0,
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
            min_dt: 1.0e-3,
            max_substeps_per_step: 8,
            recompute_density_each_step: true,
            cfl_include_affine_speed: false,
            gravity: Vec2::new(0.0, p.gravity),
            ..SolverConfig::earth(GRID, 0.01, DT)
        };
        let spawn_water = SpawnConfig {
            spacing: 0.6,
            box_size: WATER_SIZE,
            box_center: WATER_CENTER,
            material_id: MAT_WATER,
            initial_velocity_scale: 0.0,
            ..SpawnConfig::for_solver(&config)
        };
        let spawn_mud = SpawnConfig {
            spacing: 0.6,
            box_size: MUD_SIZE,
            box_center: MUD_CENTER,
            material_id: MAT_MUD,
            initial_velocity_scale: 0.0,
            ..SpawnConfig::for_solver(&config)
        };
        let mut solver = MpmSolver::new(config, spawn_water)
            .with_default_material(Box::new(make_water(&p)))
            .with_material(MAT_MUD, Box::new(make_mud(&p)))
            .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));
        let _ = solver.spawn_group(spawn_mud);
        Self {
            solver,
            stepper: FixedStepController::standard(DT, p.hz),
            prev: p,
            frame: 0,
        }
    }
}

fn make_water(p: &Params) -> NewtonianFluidMaterial {
    let mut m = NewtonianFluidMaterial::new(4.0, p.water_viscosity, p.water_stiffness, 3.0);
    m.surface_tension_coeff = p.water_surface_tension;
    m
}

fn make_mud(p: &Params) -> BinghamFluidMaterial {
    BinghamFluidMaterial::new(4.0, p.mud_viscosity, 5.0, 3.0, p.mud_yield_stress)
}

fn main() {
    App::new()
        .insert_resource(ClearColor(Color::srgb(0.05, 0.07, 0.09)))
        .insert_resource(DEFAULTS)
        .insert_resource(Sim::new(DEFAULTS))
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "MLS-MPM Fluids — Water · Bingham Mud".into(),
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

fn fluid_color(material_id: u32) -> Color {
    if material_id == MAT_WATER {
        Color::srgb(0.10, 0.55, 1.00) // vivid blue — water
    } else {
        Color::srgb(0.62, 0.38, 0.14) // earthy brown — mud
    }
}

fn setup(mut commands: Commands, sim: Res<Sim>) {
    commands.spawn(Camera2d);
    for (i, p) in sim.solver.particles().iter().enumerate() {
        commands.spawn((
            Sprite {
                color: fluid_color(p.material_id),
                custom_size: Some(Vec2::splat(8.0)),
                ..default()
            },
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
    params: Res<Params>,
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
    let strength = params.cursor_strength;
    let radius = params.cursor_radius;
    let dt = time.delta_secs().min(MAX_DT);
    sim.solver
        .apply_radial_impulse(gp, radius, sign * strength * dt);
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
            .set_default_material(Box::new(make_water(&params)));
        sim.solver
            .set_material(MAT_MUD, Box::new(make_mud(&params)));
        sim.solver
            .set_boundary_condition(Box::new(SlipBoundary::new(2)));
        sim.prev = *params;
    }
    sim.solver.step_n(n);
    sim.frame += n as u64;
    let snap = sim.solver.diagnostics_snapshot();
    const LABELS: &[(u32, &str)] = &[(MAT_WATER, "water"), (MAT_MUD, "mud")];
    log_frame_full(sim.frame, DT, sim.solver.particles(), LABELS, &snap, 60);
}

fn sync(sim: Res<Sim>, mut q: Query<(&PVis, &mut Transform, &mut Sprite)>) {
    let particles = sim.solver.particles();
    for (v, mut t, mut s) in &mut q {
        t.translation = p2w(particles.x[v.0]);
        s.color = fluid_color(particles.material_id[v.0]);
    }
}

fn ui(mut ctx: EguiContexts, mut p: ResMut<Params>, mut sim: ResMut<Sim>, time: Res<Time>) {
    let Ok(ctx) = ctx.ctx_mut() else { return };
    egui::Window::new("Fluids")
        .default_pos([10.0, 10.0])
        .default_width(270.0)
        .resizable(false)
        .show(ctx, |ui| {
            ui.label(format!(
                "fps={:.0}  n={}",
                time.delta_secs().recip(),
                sim.solver.particles().len(),
            ));
            ui.separator();
            ui.add(egui::Slider::new(&mut p.hz, 5.0..=60.0).text("solver_hz"));
            ui.add(egui::Slider::new(&mut p.gravity, -3.0..=0.0).text("gravity"));
            ui.separator();
            ui.colored_label(
                egui::Color32::from_rgb(59, 169, 245),
                "Newtonian water (blue)",
            );
            ui.add(egui::Slider::new(&mut p.water_viscosity, 0.0..=5.0).text("viscosity"));
            ui.add(egui::Slider::new(&mut p.water_stiffness, 1.0..=100.0).text("eos_k"));
            ui.add(
                egui::Slider::new(&mut p.water_surface_tension, 0.0..=2.0).text("surface tension"),
            );
            ui.separator();
            ui.colored_label(egui::Color32::from_rgb(133, 97, 56), "Bingham mud (brown)");
            ui.add(egui::Slider::new(&mut p.mud_viscosity, 0.5..=30.0).text("viscosity"));
            ui.add(egui::Slider::new(&mut p.mud_yield_stress, 0.0..=20.0).text("yield stress"));
            ui.label("↑ yield stress → mud resists flow until overcome");
            ui.separator();
            ui.add(
                egui::Slider::new(&mut p.cursor_strength, 5.0..=200.0)
                    .text("cursor force")
                    .logarithmic(true),
            );
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

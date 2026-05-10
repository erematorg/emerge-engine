/// Viscoplastic fluids — GPU MLS-MPM, Newtonian water + Bingham mud.
///
/// Mirrors basic_fluids (CPU) qualitatively:
///   Mat 0  Newtonian water (blue)  — Tait EOS dam-break column on the left
///   Mat 1  Bingham mud    (brown)  — viscoplastic blob on the right
///
///   cargo run --example basic_fluids_gpu --features "bevy_examples,gpu"
use bevy::prelude::*;
use bevy::tasks::block_on;
use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass, egui};
use emerge::gpu::{GpuSolver, GpuForceFieldEntry};
use emerge::{
    BinghamFluidMaterial, MaterialRegistry, NewtonianFluidMaterial,
    SolverConfig, SpawnConfig, build_particles, log_frame_gpu,
};
use emerge::runtime::fixed_step::FixedStepController;
use glam::{IVec2, Vec2};

const GRID: usize = 64;
const DT: f32 = 0.1;
const PPC: f32 = 10.0;
const LABELS: &[(u32, &str)] = &[(0, "water"), (1, "mud")];

const MAT_WATER: u32 = 0;
const MAT_MUD:   u32 = 1;

const WATER_CENTER: Vec2 = Vec2::new(11.0, 30.0);
const WATER_SIZE:   IVec2 = IVec2::new(14, 52);
const MUD_CENTER:   Vec2 = Vec2::new(50.0, 38.0);
const MUD_SIZE:     IVec2 = IVec2::new(16, 18);

#[derive(Resource, Clone, Copy, PartialEq)]
struct Params {
    hz: f32,
    gravity: f32,
    water_viscosity: f32,
    water_bulk_viscosity: f32,
    water_stiffness: f32,
    water_surface_tension: f32,
    mud_viscosity: f32,
    mud_yield_stress: f32,
    water_settling: f32,
    mud_settling: f32,
    cursor_strength: f32,
    cursor_radius: f32,
}

const DEFAULTS: Params = Params {
    hz: 60.0,
    gravity: -0.3,
    water_viscosity: 0.1,
    water_bulk_viscosity: 0.1,
    water_stiffness: 10.0,
    water_surface_tension: 0.0,
    mud_viscosity: 8.0,
    mud_yield_stress: 4.0,
    water_settling: 0.01,
    mud_settling: 0.02,
    cursor_strength: 30.0,
    cursor_radius: 5.0,
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
            ..SolverConfig::standard(GRID, DT, Vec2::new(0.0, p.gravity))
        };
        let spawn_water = SpawnConfig {
            spacing: 0.5,
            box_size: WATER_SIZE,
            box_center: WATER_CENTER,
            material_id: MAT_WATER,
            initial_velocity_scale: 0.0,
            ..SpawnConfig::for_solver(&config)
        };
        let spawn_mud = SpawnConfig {
            spacing: 0.5,
            box_size: MUD_SIZE,
            box_center: MUD_CENTER,
            material_id: MAT_MUD,
            initial_velocity_scale: 0.0,
            ..SpawnConfig::for_solver(&config)
        };

        let mut particles = build_particles(&config, spawn_water);
        particles.extend(build_particles(&config, spawn_mud));

        let mut registry = MaterialRegistry::with_default(Box::new(make_water(&p)));
        registry.insert(MAT_MUD, Box::new(make_mud(&p)));
        let solver = block_on(GpuSolver::new(config, particles, registry));

        Self {
            solver,
            stepper: FixedStepController::standard(DT, p.hz),
            prev: p,
        }
    }
}


fn make_water(p: &Params) -> NewtonianFluidMaterial {
    let mut m = NewtonianFluidMaterial::new(4.0, p.water_viscosity, p.water_stiffness, 3.0);
    m.bulk_viscosity = p.water_bulk_viscosity;
    m.surface_tension_coeff = p.water_surface_tension;
    m.pressure_floor = 0.0;
    m.settling_damping = p.water_settling;
    m
}

fn make_mud(p: &Params) -> BinghamFluidMaterial {
    let mut m = BinghamFluidMaterial::new(4.0, p.mud_viscosity, 15.0, 3.0, p.mud_yield_stress);
    m.settling_damping = p.mud_settling;
    m
}

fn fluid_color(mat: u32) -> Color {
    match mat {
        0 => Color::srgb(0.10, 0.55, 1.00), // vivid blue  — water
        _ => Color::srgb(0.62, 0.38, 0.14), // earthy brown — mud
    }
}

fn main() {
    App::new()
        .insert_resource(ClearColor(Color::srgb(0.05, 0.07, 0.09)))
        .insert_resource(DEFAULTS)
        .insert_resource(Sim::new(DEFAULTS))
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "MLS-MPM Fluids (GPU) — Water · Bingham Mud".into(),
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

fn setup(mut commands: Commands, sim: Res<Sim>) {
    commands.spawn(Camera2d);
    for (i, p) in sim.solver.particles().iter().enumerate() {
        commands.spawn((
            PVis(i),
            Sprite { color: fluid_color(p.material_id), custom_size: Some(Vec2::splat(4.0)), ..default() },
            Transform::from_translation(p2w(p.x)),
        ));
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
                Sprite { color: fluid_color(pt.material_id), custom_size: Some(Vec2::splat(4.0)), ..default() },
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
        sim.solver.set_default_material(Box::new(make_water(&params)));
        sim.solver.set_material(MAT_MUD, Box::new(make_mud(&params)));
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

fn sync(sim: Res<Sim>, mut vis: Query<(&PVis, &mut Transform, &mut Sprite)>) {
    let particles = sim.solver.particles();
    for (pv, mut t, mut s) in &mut vis {
        if let Some(p) = particles.get(pv.0) {
            t.translation = p2w(p.x);
            s.color = fluid_color(p.material_id);
        }
    }
}

fn ui(mut ctx: EguiContexts, mut p: ResMut<Params>, mut sim: ResMut<Sim>, time: Res<Time>) {
    let Ok(ctx) = ctx.ctx_mut() else { return };
    egui::Window::new("Fluids (GPU)")
        .default_pos([10.0, 10.0])
        .default_width(270.0)
        .resizable(false)
        .show(ctx, |ui| {
            ui.label(format!(
                "fps={:.0}  n={}  [GPU]",
                time.delta_secs().recip(),
                sim.solver.particle_count(),
            ));
            ui.separator();
            ui.add(egui::Slider::new(&mut p.hz, 5.0..=60.0).text("solver_hz"));
            ui.add(egui::Slider::new(&mut p.gravity, -2.0..=0.0).text("gravity"));
            ui.separator();
            ui.colored_label(egui::Color32::from_rgb(59, 169, 245), "Newtonian water (blue)");
            ui.add(egui::Slider::new(&mut p.water_viscosity, 0.0..=5.0).text("shear µ").logarithmic(true));
            ui.add(egui::Slider::new(&mut p.water_bulk_viscosity, 0.0..=5.0).text("bulk ζ (wave damp)").logarithmic(true));
            ui.add(egui::Slider::new(&mut p.water_settling, 0.0..=1.0).text("settling damp"));
            ui.add(egui::Slider::new(&mut p.water_stiffness, 1.0..=500.0).text("eos_k").logarithmic(true));
            ui.add(egui::Slider::new(&mut p.water_surface_tension, 0.0..=2.0).text("surface tension"));
            ui.separator();
            ui.colored_label(egui::Color32::from_rgb(133, 97, 56), "Bingham mud (brown)");
            ui.add(egui::Slider::new(&mut p.mud_viscosity, 0.5..=30.0).text("viscosity"));
            ui.add(egui::Slider::new(&mut p.mud_yield_stress, 0.0..=20.0).text("yield stress"));
            ui.add(egui::Slider::new(&mut p.mud_settling, 0.0..=2.0).text("settling damp"));
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

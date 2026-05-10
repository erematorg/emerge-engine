/// Three-material showcase — sand terrain, fluid pool, and elastic blob in one scene.
///
/// Proves emerge handles multi-material interaction: LP terrain (sand) + water (fluid)
/// + creature body (NeoHookean elastic) all running in the same MLS-MPM solver.
///
/// Controls:
///   ↑ ↓ ← →   apply impulse to the elastic blob
///   LMB / RMB  push / pull all particles under cursor
///   R          reset scene
///
///   cargo run --example basic_showcase --features bevy_examples
use bevy::prelude::*;
use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass, egui};
use emerge::diagnostics::log_frame_full;
use emerge::{
    MpmSolver, NeoHookeanMaterial, NewtonianFluidMaterial, SandMaterial, SlipBoundary, SolverConfig,
    SpawnConfig,
};
use emerge::runtime::fixed_step::FixedStepController;
use glam::{IVec2, Vec2};

const GRID: usize = 64;
const DT: f32 = 0.1;
const PPC: f32 = 10.0;
const MAX_DT: f32 = 1.0 / 15.0;

const ELASTIC_ID: u32 = 0;
const SAND_ID: u32 = 1;
const FLUID_ID: u32 = 2;

// Impulse applied to elastic blob via arrow keys.
const DRIVE_STRENGTH: f32 = 10.0;
const DRIVE_RADIUS: f32 = 12.0;

#[derive(Resource, Clone, Copy, PartialEq)]
struct Params {
    hz: f32,
    gravity: f32,
    // Elastic
    e_lambda: f32,
    e_mu: f32,
    // Sand
    s_lambda: f32,
    s_mu: f32,
    // Fluid
    f_density: f32,
    f_viscosity: f32,
    f_eos_k: f32,
}

const DEFAULTS: Params = Params {
    hz: 60.0,
    gravity: -0.5,
    e_lambda: 40.0,
    e_mu: 80.0,
    // lambda=400, mu=200 → c_P≈24, ~6 CFL substeps (vs ~9 at 1000/500).
    // Still forms realistic sand piles — friction angle controls repose, not stiffness.
    s_lambda: 400.0,
    s_mu: 200.0,
    f_density: 4.0,
    f_viscosity: 0.1,
    f_eos_k: 10.0,
};

#[derive(Resource)]
struct Sim {
    solver: MpmSolver,
    stepper: FixedStepController,
    prev: Params,
    frame: u64,
}

fn make_solver(p: Params) -> MpmSolver {
    let config = SolverConfig {
        min_dt: 0.005,
        max_substeps_per_step: 16,
        recompute_density_each_step: true,
        ..SolverConfig::standard(GRID, DT, Vec2::new(0.0, p.gravity))
    };

    // Spacing 0.7 keeps density realistic (~1500 particles total) while staying
    // interactive on CPU. At spacing=0.5 the scene hits ~3600 particles → 10 fps.
    const SPACING: f32 = 0.7;

    // Sand pile — bottom-left
    let sand_spawn = SpawnConfig {
        spacing: SPACING,
        box_size: IVec2::new(22, 14),
        box_center: Vec2::new(19.0, 9.0),
        material_id: SAND_ID,
        initial_velocity_scale: 0.0,
        precompute_initial_volumes: true,
        ..SpawnConfig::for_solver(&config)
    };

    let mut solver = MpmSolver::new(config, sand_spawn)
        .with_default_material(Box::new(NeoHookeanMaterial::new(p.e_lambda, p.e_mu)))
        .with_material(SAND_ID, Box::new(make_sand(p)))
        .with_material(FLUID_ID, Box::new(make_fluid(p)))
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));

    // Fluid pool — bottom-right
    let fluid_spawn = SpawnConfig {
        spacing: SPACING,
        box_size: IVec2::new(22, 14),
        box_center: Vec2::new(45.0, 9.0),
        material_id: FLUID_ID,
        initial_velocity_scale: 0.0,
        precompute_initial_volumes: true,
        ..SpawnConfig::for_solver(&config)
    };
    let _ = solver.spawn_region(fluid_spawn);

    // Elastic blob — center, high — falls onto terrain
    let elastic_spawn = SpawnConfig {
        spacing: SPACING,
        box_size: IVec2::new(12, 12),
        box_center: Vec2::new(32.0, 46.0),
        material_id: ELASTIC_ID,
        initial_velocity_scale: 0.0,
        precompute_initial_volumes: true,
        ..SpawnConfig::for_solver(&config)
    };
    let _ = solver.spawn_region(elastic_spawn);

    println!(
        "[showcase] total particles: {}",
        solver.particles().len()
    );
    solver
}

fn make_sand(p: Params) -> SandMaterial {
    SandMaterial::new(p.s_lambda, p.s_mu)
}

fn make_fluid(p: Params) -> NewtonianFluidMaterial {
    NewtonianFluidMaterial::new(p.f_density, p.f_viscosity, p.f_eos_k, 4.0)
}

impl Sim {
    fn new(p: Params) -> Self {
        let solver = make_solver(p);
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
        .insert_resource(ClearColor(Color::srgb(0.06, 0.06, 0.09)))
        .insert_resource(DEFAULTS)
        .insert_resource(Sim::new(DEFAULTS))
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "emerge — Sand + Fluid + Elastic  (arrows: move blob, R: reset)".into(),
                resolution: (900u32, 900u32).into(),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(EguiPlugin::default())
        .add_systems(Startup, setup)
        .add_systems(Update, (reset, cursor, drive, step, sync).chain())
        .add_systems(EguiPrimaryContextPass, ui)
        .run();
}

#[derive(Component)]
struct PVis(usize);

fn mat_color(id: u32) -> Color {
    match id {
        SAND_ID => Color::srgb(0.80, 0.64, 0.22), // golden sand
        FLUID_ID => Color::srgb(0.22, 0.62, 0.95), // blue water
        _ => Color::srgb(0.38, 0.80, 0.48),        // green elastic blob
    }
}

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
    let Some(cp) = win.cursor_position() else { return };
    let Ok((cam, ct)) = cam.single() else { return };
    let Ok(wp) = cam.viewport_to_world_2d(ct, cp) else { return };
    let gp = wp / PPC + Vec2::splat(GRID as f32 * 0.5);
    let sign = if mb.pressed(MouseButton::Right) { -1.0 } else { 1.0 };
    let dt = time.delta_secs().min(MAX_DT);
    sim.solver.particles_mut().for_each_mut(|p| {
        let d = p.x - gp;
        let dist = d.length();
        if dist < 5.0 && dist > 1e-4 {
            p.v += (d / dist) * sign * 40.0 * (1.0 - dist / 5.0) * dt;
            let s = p.v.length();
            if s > 20.0 {
                p.v *= 20.0 / s;
            }
        }
    });
}

/// Arrow keys apply an impulse to elastic blob particles only.
fn drive(keys: Res<ButtonInput<KeyCode>>, mut sim: ResMut<Sim>, time: Res<Time>) {
    let dir = {
        let mut d = Vec2::ZERO;
        if keys.pressed(KeyCode::ArrowLeft)  { d.x -= 1.0; }
        if keys.pressed(KeyCode::ArrowRight) { d.x += 1.0; }
        if keys.pressed(KeyCode::ArrowUp)    { d.y += 1.0; }
        if keys.pressed(KeyCode::ArrowDown)  { d.y -= 1.0; }
        d
    };
    if dir == Vec2::ZERO { return; }

    let centroid = {
        let particles = sim.solver.particles();
        let elastic: Vec<Vec2> = particles
            .iter()
            .filter(|p| p.material_id == ELASTIC_ID)
            .map(|p| p.x)
            .collect();
        if elastic.is_empty() { return; }
        elastic.iter().copied().fold(Vec2::ZERO, |a, x| a + x) / elastic.len() as f32
    };

    let impulse = dir.normalize() * DRIVE_STRENGTH * time.delta_secs().min(MAX_DT);
    sim.solver.particles_mut().for_each_mut(|p| {
        if p.material_id != ELASTIC_ID { return; }
        let dist = (p.x - centroid).length();
        if dist < DRIVE_RADIUS {
            p.v += impulse * (1.0 - dist / DRIVE_RADIUS);
        }
    });
}

fn step(time: Res<Time>, mut sim: ResMut<Sim>, params: Res<Params>) {
    sim.solver.set_gravity(Vec2::new(0.0, params.gravity));
    sim.stepper.set_simulation_speed(params.hz * DT);
    let n = sim.stepper.steps_for_frame(time.delta_secs());
    if n == 0 { return; }
    if sim.prev != *params {
        sim.solver.set_default_material(Box::new(NeoHookeanMaterial::new(params.e_lambda, params.e_mu)));
        sim.solver.set_material(SAND_ID, Box::new(make_sand(*params)));
        sim.solver.set_material(FLUID_ID, Box::new(make_fluid(*params)));
        sim.prev = *params;
    }
    sim.solver.step_n(n);
    sim.frame += n as u64;
    let snap = sim.solver.diagnostics_snapshot();
    const LABELS: &[(u32, &str)] = &[(ELASTIC_ID, "elastic"), (SAND_ID, "sand"), (FLUID_ID, "fluid")];
    log_frame_full(sim.frame, DT, sim.solver.particles(), LABELS, &snap, 60);
}

fn sync(sim: Res<Sim>, mut q: Query<(&PVis, &mut Transform, &mut Sprite)>) {
    for (v, mut t, mut s) in &mut q {
        let p = sim.solver.particles().get(v.0);
        t.translation = p2w(p.x);
        s.color = mat_color(p.material_id);
    }
}

fn ui(mut ctx: EguiContexts, mut p: ResMut<Params>, mut sim: ResMut<Sim>, time: Res<Time>) {
    let Ok(ctx) = ctx.ctx_mut() else { return };
    let n_elastic = sim.solver.particles().iter().filter(|p| p.material_id == ELASTIC_ID).count();
    let n_sand = sim.solver.particles().iter().filter(|p| p.material_id == SAND_ID).count();
    let n_fluid = sim.solver.particles().iter().filter(|p| p.material_id == FLUID_ID).count();
    egui::Window::new("Showcase")
        .default_pos([10.0, 10.0])
        .default_width(260.0)
        .resizable(false)
        .show(ctx, |ui| {
            ui.label(format!("fps={:.0}", time.delta_secs().recip()));
            ui.label(format!("elastic={n_elastic}  sand={n_sand}  fluid={n_fluid}"));
            ui.separator();
            ui.label("↑ ↓ ← →  move blob    R  reset");
            ui.label("LMB push  RMB pull (all materials)");
            ui.separator();
            ui.add(egui::Slider::new(&mut p.hz, 5.0..=60.0).text("solver_hz"));
            ui.add(egui::Slider::new(&mut p.gravity, -3.0..=0.0).text("gravity"));
            ui.separator();
            ui.label("Elastic blob (NeoHookean)");
            ui.add(egui::Slider::new(&mut p.e_lambda, 5.0..=200.0).text("λ"));
            ui.add(egui::Slider::new(&mut p.e_mu, 5.0..=400.0).text("µ"));
            ui.separator();
            ui.label("Sand terrain (Drucker-Prager)");
            ui.add(egui::Slider::new(&mut p.s_lambda, 50.0..=2000.0).text("λ"));
            ui.add(egui::Slider::new(&mut p.s_mu, 25.0..=1000.0).text("µ"));
            ui.separator();
            ui.label("Fluid pool (Tait EOS)");
            ui.add(egui::Slider::new(&mut p.f_density, 1.0..=10.0).text("rho0"));
            ui.add(egui::Slider::new(&mut p.f_viscosity, 0.0..=5.0).text("viscosity"));
            ui.add(egui::Slider::new(&mut p.f_eos_k, 1.0..=50.0).text("eos_k"));
            ui.separator();
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

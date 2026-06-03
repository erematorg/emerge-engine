/// Two snowballs colliding — CPU MLS-MPM, Stomakhin 2013 snow plasticity.
///
/// Ball A (blue)  = soft powder  — low hardening, wide plastic limits.
/// Ball B (amber) = packed snow  — high hardening, tight limits.
/// Jp compression shown as red shift on impact.
///
///   cargo run --example basic_snow --features bevy_examples
use bevy::prelude::*;
use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass, egui};
use emerge::diagnostics::log_frame_full;
use emerge::runtime::fixed_step::FixedStepController;
use emerge::{MpmSolver, SandMaterial, SlipBoundary, SnowMaterial, SolverConfig, SpawnConfig};
use glam::{IVec2, Vec2};

const GRID: usize = 64;
const DT: f32 = 0.1;
const PPC: f32 = 10.0;
const MAX_DT: f32 = 1.0 / 15.0;

const BALL_R: f32 = 9.0;
const BALL_A: Vec2 = Vec2::new(16.0, 44.0); // soft powder  → right
const BALL_B: Vec2 = Vec2::new(48.0, 44.0); // packed snow  ← left
const MAT_SOFT: u32 = 0;
const MAT_PACKED: u32 = 1;
/// Shattered fragments — packed snow that took a violent impact transitions to loose granular.
const MAT_SHATTER: u32 = 2;
const LABELS: &[(u32, &str)] = &[(MAT_SOFT, "soft"), (MAT_PACKED, "packed"), (MAT_SHATTER, "shatter")];

const COL_A: Color = Color::srgb(0.35, 0.65, 1.00); // blue
const COL_B: Color = Color::srgb(0.95, 0.80, 0.45); // amber

#[derive(Resource, Clone, Copy, PartialEq)]
struct Params {
    hz: f32,
    gravity: f32,
    speed: f32,
    lambda: f32,
    mu: f32,
    // soft powder
    xi_a: f32,
    theta_c_a: f32,
    theta_s_a: f32,
    // packed snow
    xi_b: f32,
    theta_c_b: f32,
    theta_s_b: f32,
    cursor_strength: f32,
    cursor_radius: f32,
}

const DEFAULTS: Params = Params {
    hz: 60.0,
    gravity: -0.08,
    speed: 15.0,
    // E=5000, ν=0.2 → λ≈1389, µ≈2083.  Matches Taichi MPM128 stiffness scale.
    // At h_max=5 (clamp), c_P(h=5)≈83 cells/s → sub_dt≈0.006 → ~17 substeps.
    lambda: 1389.0,
    mu: 2083.0,
    // soft powder — wider elastic range, lower hardening (Stomakhin §4 light snow)
    xi_a: 7.0,
    theta_c_a: 0.025,
    theta_s_a: 0.0075,
    // packed snow — tighter range, canonical ξ=10 (hits h_max at 20% compression vs soft's 28%)
    xi_b: 10.0,
    theta_c_b: 0.012,
    theta_s_b: 0.004,
    cursor_strength: 50.0,
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
            max_substeps_per_step: 20,
            gravity: Vec2::new(0.0, p.gravity),
            ..SolverConfig::earth(GRID, 0.01, DT)
        };
        let spawn = SpawnConfig {
            spacing: 0.5,
            box_size: IVec2::new(58, 58),
            initial_velocity_scale: 0.0,
            rng_seed: 7,
            ..SpawnConfig::for_solver(&config)
        };
        let mut solver = MpmSolver::new(config, spawn)
            .with_default_material(Box::new(make_snow_a(&p)))
            .with_material(MAT_PACKED, Box::new(make_snow_b(&p)))
            // Shatter material: loose granular, low friction → fragments scatter freely.
            .with_material(MAT_SHATTER, Box::new(SandMaterial::loose_sand(200.0, 100.0)))
            .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));
        {
            let speed = p.speed;
            solver.retain_particles(|pt| {
                (pt.x - BALL_A).length() <= BALL_R || (pt.x - BALL_B).length() <= BALL_R
            });
            solver.particles_mut().for_each_mut(|pt| {
                if (pt.x - BALL_A).length() <= BALL_R {
                    pt.material_id = MAT_SOFT;
                    pt.v = Vec2::new(speed, 0.0);
                } else {
                    pt.material_id = MAT_PACKED;
                    pt.v = Vec2::new(-speed, 0.0);
                }
            });
        }
        solver.recompute_initial_volumes();
        Self {
            solver,
            stepper: FixedStepController::standard(DT, p.hz),
            prev: p,
            frame: 0,
        }
    }
}

fn make_snow_a(p: &Params) -> SnowMaterial {
    SnowMaterial::new(p.lambda, p.mu, p.xi_a, p.theta_c_a, p.theta_s_a, 0.6, 20.0)
}
fn make_snow_b(p: &Params) -> SnowMaterial {
    SnowMaterial::new(p.lambda, p.mu, p.xi_b, p.theta_c_b, p.theta_s_b, 0.6, 20.0)
        .with_cohesion(400.0)
}

fn main() {
    App::new()
        .insert_resource(ClearColor(Color::srgb(0.05, 0.07, 0.10)))
        .insert_resource(DEFAULTS)
        .insert_resource(Sim::new(DEFAULTS))
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "MLS-MPM Snow — Powder vs Packed".into(),
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
        let color = if p.material_id == MAT_SOFT {
            COL_A
        } else {
            COL_B
        };
        commands.spawn((
            Sprite::from_color(color, Vec2::ONE),
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
    let speed_cap = strength * 0.6;
    let dt = time.delta_secs().min(MAX_DT);
    sim.solver.particles_mut().for_each_mut(|p| {
        let d = p.x - gp;
        let dist = d.length();
        if dist < radius && dist > 1e-4 {
            p.v += (d / dist) * sign * strength * (1.0 - dist / radius) * dt;
            let s = p.v.length();
            if s > speed_cap {
                p.v *= speed_cap / s;
            }
        }
    });
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
            .set_default_material(Box::new(make_snow_a(&params)));
        sim.solver
            .set_material(MAT_PACKED, Box::new(make_snow_b(&params)));
        sim.prev = *params;
    }
    sim.solver.step_n(n);
    let damp = 0.999_f32.powi((n * 17) as i32);
    sim.solver.particles_mut().for_each_mut(|p| p.v *= damp);

    // Fracture: packed snow hit hard → transitions to loose granular (shatter).
    // Granular material has no cohesion → fragments scatter visibly.
    sim.solver.phase_transition(
        |p| p.material_id == MAT_PACKED && p.v.length() > 5.0,
        MAT_SHATTER,
    );

    sim.frame += n as u64;
    let snap = sim.solver.diagnostics_snapshot();
    log_frame_full(sim.frame, DT, sim.solver.particles(), LABELS, &snap, 60);
}

fn sync(sim: Res<Sim>, mut q: Query<(&PVis, &mut Transform, &mut Sprite)>) {
    for (v, mut t, mut s) in &mut q {
        let p = sim.solver.particles().get(v.0);
        t.translation = p2w(p.x);
        // Shattered fragments → white/grey. Others: red shift on compression.
        if p.material_id == MAT_SHATTER {
            s.color = Color::srgb(0.90, 0.90, 0.95);
            continue;
        }
        let base = if p.material_id == MAT_SOFT { COL_A } else { COL_B }.to_srgba();
        let compress = (1.0 - p.plastic_volume_ratio).clamp(0.0, 1.0);
        s.color = Color::srgb(
            (base.red + compress * (1.0 - base.red)).min(1.0),
            (base.green - compress * base.green * 0.8).max(0.0),
            (base.blue - compress * base.blue).max(0.0),
        );
    }
}

fn ui(mut ctx: EguiContexts, mut p: ResMut<Params>, mut sim: ResMut<Sim>, time: Res<Time>) {
    let Ok(ctx) = ctx.ctx_mut() else { return };
    egui::Window::new("Snow")
        .default_pos([10.0, 10.0])
        .default_width(280.0)
        .resizable(false)
        .show(ctx, |ui| {
            ui.label(format!(
                "fps={:.0}  n={}",
                time.delta_secs().recip(),
                sim.solver.particles().len()
            ));
            ui.separator();
            ui.add(egui::Slider::new(&mut p.hz, 1.0..=60.0).text("solver_hz"));
            ui.add(egui::Slider::new(&mut p.gravity, -3.0..=0.0).text("gravity"));
            ui.add(egui::Slider::new(&mut p.speed, 1.0..=30.0).text("speed (→ reset)"));
            ui.separator();
            ui.label("Shared stiffness");
            ui.add(egui::Slider::new(&mut p.lambda, 50.0..=5_000.0).text("λ"));
            ui.add(egui::Slider::new(&mut p.mu, 50.0..=10_000.0).text("µ"));
            ui.separator();
            ui.colored_label(egui::Color32::from_rgb(90, 165, 255), "Soft powder (blue)");
            ui.add(egui::Slider::new(&mut p.xi_a, 0.0..=20.0).text("ξ"));
            ui.add(
                egui::Slider::new(&mut p.theta_c_a, 0.001..=0.1)
                    .logarithmic(true)
                    .text("θ_c"),
            );
            ui.add(
                egui::Slider::new(&mut p.theta_s_a, 0.001..=0.05)
                    .logarithmic(true)
                    .text("θ_s"),
            );
            ui.separator();
            ui.colored_label(
                egui::Color32::from_rgb(242, 204, 115),
                "Packed snow (amber)",
            );
            ui.add(egui::Slider::new(&mut p.xi_b, 0.0..=20.0).text("ξ"));
            ui.add(
                egui::Slider::new(&mut p.theta_c_b, 0.001..=0.1)
                    .logarithmic(true)
                    .text("θ_c"),
            );
            ui.add(
                egui::Slider::new(&mut p.theta_s_b, 0.001..=0.05)
                    .logarithmic(true)
                    .text("θ_s"),
            );
            ui.separator();
            ui.add(
                egui::Slider::new(&mut p.cursor_strength, 5.0..=500.0)
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

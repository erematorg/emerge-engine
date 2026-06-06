/// Material validation harness — headless, no Bevy, no window.
///
/// Modes (pass as first argument):
///   (none)     — basic: all 9 materials, 500 steps, reports settle/J/NaN
///   sweep      — param sweep: each material across soft/med/stiff/extreme E
///   perf       — particle budget: wall-clock ms/step at 100→20k particles
///   scenario   — LP gamedev scenarios: creature body, water pool, sand terrain
///
///   cargo run --example validate_materials
///   cargo run --example validate_materials -- sweep
///   cargo run --example validate_materials -- perf
///   cargo run --example validate_materials -- scenario
///
/// Physical convention: dimensionless, same as all emerge examples.
///   GRID=64, DT=0.1, GRAVITY=-0.3 cell/step². Fall from y=48 to floor ≈46 cells in ~55 steps.
use emerge::{
    BinghamFluidMaterial, FrictionBoundary, MpmSolver, NeoHookeanMaterial,
    NewtonianFluidMaterial, RankineMaterial, SandMaterial, SandMuIMaterial, SlipBoundary,
    SnowMaterial, SolverConfig, SpawnConfig, VonMisesMaterial, ViscoelasticMaterial,
    lame_from_young,
};
use glam::{IVec2, Vec2};
use std::time::Instant;

// ── Simulation constants ──────────────────────────────────────────────────────
const GRID: usize = 64;
const DT: f32 = 0.1;
const GRAVITY: f32 = -0.3;

fn gravity() -> Vec2 { Vec2::new(0.0, GRAVITY) }

fn config() -> SolverConfig {
    SolverConfig {
        max_substeps_per_step: 64,
        ..SolverConfig::standard(GRID, DT, gravity())
    }
}

fn spawn_at(center: Vec2, size: IVec2, spacing: f32, material_id: u32) -> SpawnConfig {
    SpawnConfig {
        spacing,
        box_size: size,
        box_center: center,
        material_id,
        precompute_initial_volumes: true,
        ..SpawnConfig::for_solver(&config())
    }
}

fn center_spawn(material_id: u32) -> SpawnConfig {
    spawn_at(Vec2::new(32.0, 48.0), IVec2::new(16, 16), 0.5, material_id)
}

// ── Shared report type ─────────────────────────────────────────────────────────

struct Report {
    name:        String,
    initial_y:   f32,
    min_y:       f32,
    _final_y:    f32,
    final_speed: f32,
    min_j:       f32,
    max_j:       f32,
    nans:        u32,
    settled:     bool,
    bouncing:    bool,
    exploded:    bool,
}

impl Report {
    fn from_solver(name: impl Into<String>, material_id: u32, mut solver: MpmSolver, steps: u64) -> Self {
        let initial_y = solver.material_state(material_id).centroid.y;
        let mut min_y = initial_y;

        for _ in 1..=steps {
            solver.step_n(1);
            let ms = solver.material_state(material_id);
            min_y = min_y.min(ms.centroid.y);
        }

        let snap = solver.diagnostics_snapshot();
        let ms   = solver.material_state(material_id);
        let max_fall   = initial_y - min_y;
        let final_drop = initial_y - ms.centroid.y;
        let bouncing = max_fall > 1.0 && final_drop < max_fall * 0.3 && ms.avg_speed > 0.02;
        let exploded = snap.non_finite_particle_values > 0
            || snap.min_deformation_j < 0.01
            || snap.max_deformation_j > 100.0;

        Report {
            name:        name.into(),
            initial_y,
            min_y,
            _final_y:    ms.centroid.y,
            final_speed: ms.avg_speed,
            min_j:       snap.min_deformation_j,
            max_j:       snap.max_deformation_j,
            nans:        snap.non_finite_particle_values as u32,
            settled:     ms.avg_speed < 0.01 && !exploded,
            bouncing,
            exploded,
        }
    }

    fn print_compact(&self) {
        let status = if self.exploded          { "💥 EXPLODED" }
                     else if self.nans > 0     { "❌ NaN     " }
                     else if self.bouncing     { "↕  bouncing" }
                     else if !self.settled     { "⚠  moving  " }
                     else                      { "✓  settled " };
        let max_fall = self.initial_y - self.min_y;
        println!(
            "  [{status}]  {name:<28}  fall={f:4.1}  spd={sp:6.3}  J=[{jl:.3},{jh:.3}]  nans={n}",
            name = self.name, f = max_fall, sp = self.final_speed,
            jl = self.min_j, jh = self.max_j, n = self.nans,
        );
    }
}

// ════════════════════════════════════════════════════════════════════════════════
// MODE 1 — Basic validation
// ════════════════════════════════════════════════════════════════════════════════

fn run_basic() {
    println!("emerge material validation — basic");
    println!("  grid={GRID}×{GRID}, dt={DT}, gravity={GRAVITY}");
    println!("  spawn y=48, floor at y≈2, expected fall ≈46 cells\n");

    let cases: &[(&str, Box<dyn Fn() -> MpmSolver>)] = &[
        ("water (Newtonian)",   Box::new(|| {
            let m = NewtonianFluidMaterial::water(1.0, 100.0);
            MpmSolver::new(config(), center_spawn(0)).with_default_material(Box::new(m)).with_boundary(Box::new(SlipBoundary::new(2)))
        })),
        ("mud (Bingham)",       Box::new(|| {
            let m = BinghamFluidMaterial::mud();
            MpmSolver::new(config(), center_spawn(0)).with_default_material(Box::new(m)).with_boundary(Box::new(SlipBoundary::new(2)))
        })),
        ("jelly (NeoHookean)",  Box::new(|| {
            let (l, u) = lame_from_young(5e4, 0.3);
            let m = NeoHookeanMaterial::new(l, u);
            MpmSolver::new(config(), center_spawn(0)).with_default_material(Box::new(m)).with_boundary(Box::new(SlipBoundary::new(2)))
        })),
        ("elastic (Corotated)", Box::new(|| {
            let (l, u) = lame_from_young(5e5, 0.3);
            let m = emerge::CorotatedMaterial::new(l, u);
            MpmSolver::new(config(), center_spawn(0)).with_default_material(Box::new(m)).with_boundary(Box::new(SlipBoundary::new(2)))
        })),
        ("snow",                Box::new(|| {
            let m = SnowMaterial::from_young_modulus(1.4e5, 0.2);
            MpmSolver::new(config(), center_spawn(0)).with_default_material(Box::new(m)).with_boundary(Box::new(SlipBoundary::new(2)))
        })),
        ("sand (DP)",           Box::new(|| {
            let m = SandMaterial::from_young_modulus(1e5, 0.2);
            MpmSolver::new(config(), center_spawn(0)).with_default_material(Box::new(m)).with_boundary(Box::new(SlipBoundary::new(2)))
        })),
        ("von Mises",           Box::new(|| {
            let (l, u) = lame_from_young(5e5, 0.3);
            let m = VonMisesMaterial::new(l, u, u * 0.01);
            MpmSolver::new(config(), center_spawn(0)).with_default_material(Box::new(m)).with_boundary(Box::new(SlipBoundary::new(2)))
        })),
        ("rankine (rock)",      Box::new(|| {
            let (l, u) = lame_from_young(5e5, 0.25);
            let m = RankineMaterial::rock(l, u);
            MpmSolver::new(config(), center_spawn(0)).with_default_material(Box::new(m)).with_boundary(Box::new(SlipBoundary::new(2)))
        })),
        ("viscoelastic",        Box::new(|| {
            let m = ViscoelasticMaterial::soft_tissue();
            MpmSolver::new(config(), center_spawn(0)).with_default_material(Box::new(m)).with_boundary(Box::new(SlipBoundary::new(2)))
        })),
        ("sand µ(I)",           Box::new(|| {
            let (l, u) = lame_from_young(1e5, 0.2);
            let m = SandMuIMaterial::fine_sand(l, u);
            MpmSolver::new(config(), center_spawn(0)).with_default_material(Box::new(m)).with_boundary(Box::new(FrictionBoundary::new(2, 0.6)))
        })),
    ];

    let mut reports = Vec::new();
    for (name, builder) in cases {
        let r = Report::from_solver(*name, 0, builder(), 500);
        r.print_compact();
        reports.push(r);
    }

    let nans   = reports.iter().filter(|r| r.nans > 0).count();
    let explod = reports.iter().filter(|r| r.exploded).count();
    println!("\n  NaN failures: {nans}  Explosions: {explod}");
    println!("  Note: bouncing elastic/rankine/viscoelastic is correct — APIC conserves energy.");
}

// ════════════════════════════════════════════════════════════════════════════════
// MODE 2 — Parameter sweep
// ════════════════════════════════════════════════════════════════════════════════

fn run_sweep() {
    println!("emerge parameter sweep — stability across E / yield / viscosity\n");

    // ── NeoHookean: E sweep ──────────────────────────────────────────────────
    println!("── NeoHookean (jelly), nu=0.3, sweep E ──");
    for &e in &[1e3f32, 1e4, 5e4, 1e5, 5e5, 1e6, 5e6] {
        let (l, u) = lame_from_young(e, 0.3);
        let m = NeoHookeanMaterial::new(l, u);
        let solver = MpmSolver::new(config(), center_spawn(0))
            .with_default_material(Box::new(m))
            .with_boundary(Box::new(SlipBoundary::new(2)));
        let r = Report::from_solver(format!("E={e:.0}"), 0, solver, 300);
        r.print_compact();
    }

    // ── NeoHookean: nu sweep ─────────────────────────────────────────────────
    println!("\n── NeoHookean (jelly), E=5e4, sweep nu ──");
    for &nu in &[0.0f32, 0.1, 0.2, 0.3, 0.4, 0.45, 0.49] {
        let (l, u) = lame_from_young(5e4, nu);
        let m = NeoHookeanMaterial::new(l, u);
        let solver = MpmSolver::new(config(), center_spawn(0))
            .with_default_material(Box::new(m))
            .with_boundary(Box::new(SlipBoundary::new(2)));
        let r = Report::from_solver(format!("nu={nu}"), 0, solver, 300);
        r.print_compact();
    }

    // ── VonMises: yield stress sweep ─────────────────────────────────────────
    println!("\n── VonMises, E=5e5, sweep yield (as fraction of mu) ──");
    let (lv, uv) = lame_from_young(5e5, 0.3);
    for &frac in &[0.001f32, 0.005, 0.01, 0.05, 0.1, 0.5, 2.0] {
        let m = VonMisesMaterial::new(lv, uv, uv * frac);
        let solver = MpmSolver::new(config(), center_spawn(0))
            .with_default_material(Box::new(m))
            .with_boundary(Box::new(SlipBoundary::new(2)));
        let r = Report::from_solver(format!("yield=mu*{frac}"), 0, solver, 300);
        r.print_compact();
    }

    // ── Sand: E sweep ────────────────────────────────────────────────────────
    println!("\n── Sand (DP), nu=0.2, sweep E ──");
    for &e in &[1e3f32, 1e4, 5e4, 1e5, 5e5, 1e6] {
        let m = SandMaterial::from_young_modulus(e, 0.2);
        let solver = MpmSolver::new(config(), center_spawn(0))
            .with_default_material(Box::new(m))
            .with_boundary(Box::new(SlipBoundary::new(2)));
        let r = Report::from_solver(format!("E={e:.0}"), 0, solver, 300);
        r.print_compact();
    }

    // ── Water: EOS stiffness sweep ───────────────────────────────────────────
    println!("\n── Newtonian fluid, rho=1.0, sweep EOS stiffness ──");
    for &k in &[10.0f32, 50.0, 100.0, 500.0, 1000.0, 5000.0] {
        let m = NewtonianFluidMaterial::water(1.0, k);
        let solver = MpmSolver::new(config(), center_spawn(0))
            .with_default_material(Box::new(m))
            .with_boundary(Box::new(SlipBoundary::new(2)));
        let r = Report::from_solver(format!("k={k}"), 0, solver, 300);
        r.print_compact();
    }

    println!("\n  Stability guide: J in [0.8, 1.5] = good. J < 0.1 or > 10 = unstable.");
    println!("  Exploded = NaN or J out of [0.01, 100] range.");
}

// ════════════════════════════════════════════════════════════════════════════════
// MODE 3 — Performance / particle budget
// ════════════════════════════════════════════════════════════════════════════════

fn run_perf() {
    println!("emerge performance — ms/step at various particle counts\n");
    println!("  Material: NeoHookean E=5e4. Grid=64×64. Steps=50 (warm-up 5 discarded).");
    println!("  Target: <16ms/step for 60fps gamedev.\n");

    // Fixed spacing=0.5, center=(32,32). Box side = ceil(sqrt(n)*0.5) cells, capped at 56
    // (stays inside boundary_thickness=2 on a 64-cell grid). Actual count ≈ n.
    let counts = [100usize, 500, 1000, 2500, 5000, 10000];

    for &n in &counts {
        let spacing = 0.5f32;
        let cells_per_side = ((n as f32).sqrt() * spacing).ceil() as i32;
        let cells_per_side = cells_per_side.min(56);

        let cfg = SolverConfig {
            max_substeps_per_step: 64,
            ..SolverConfig::standard(GRID, DT, gravity())
        };
        let spawn = SpawnConfig {
            spacing,
            box_size: IVec2::splat(cells_per_side),
            box_center: Vec2::splat(32.0),
            material_id: 0,
            precompute_initial_volumes: true,
            ..SpawnConfig::for_solver(&cfg)
        };

        let (l, u) = lame_from_young(5e4, 0.3);
        let m = NeoHookeanMaterial::new(l, u);
        let mut solver = MpmSolver::new(cfg, spawn)
            .with_default_material(Box::new(m))
            .with_boundary(Box::new(SlipBoundary::new(2)));

        let actual_n = solver.particles().len();

        // Warm-up
        for _ in 0..5 { solver.step_n(1); }

        // Timed run
        let t0 = Instant::now();
        for _ in 0..50 { solver.step_n(1); }
        let elapsed = t0.elapsed();
        let ms_per_step = elapsed.as_secs_f64() * 1000.0 / 50.0;

        let fps_headroom = 16.0 / ms_per_step;
        let feasible = if ms_per_step < 16.0 { "✓" } else { "✗" };

        println!(
            "  {feasible}  {actual_n:>6} particles  {ms_per_step:7.2}ms/step  headroom={fps_headroom:.1}x",
        );
    }

    println!("\n  Note: debug build. Release will be ~10–30× faster.");
    println!("  LP target: 2000–5000 particles for a creature-scale scene at 60fps release.");
}

// ════════════════════════════════════════════════════════════════════════════════
// MODE 4 — Gamedev scenarios
// ════════════════════════════════════════════════════════════════════════════════

fn run_scenarios() {
    println!("emerge gamedev scenarios — LP-relevant physics setups\n");

    // ── Scenario 1: Creature body (soft jelly) landing on sand floor ─────────
    println!("── Scenario 1: Soft body landing on sand floor ──");
    println!("   Jelly blob (material 0) spawned above, sand floor (material 1) at bottom.");
    {
        let cfg = SolverConfig {
            max_substeps_per_step: 64,
            ..SolverConfig::standard(GRID, DT, gravity())
        };

        let body_spawn = SpawnConfig {
            spacing: 0.5,
            box_size: IVec2::new(10, 10),
            box_center: Vec2::new(32.0, 50.0),
            material_id: 0,
            precompute_initial_volumes: true,
            ..SpawnConfig::for_solver(&cfg)
        };

        let (lj, uj) = lame_from_young(3e4, 0.45);
        let body_mat = NeoHookeanMaterial::new(lj, uj);

        let mut solver = MpmSolver::new(cfg, body_spawn)
            .with_default_material(Box::new(body_mat))
            .with_boundary(Box::new(FrictionBoundary::new(2, 0.6)));

        // Sand floor — spawn a flat bed at the bottom
        let floor_spawn = SpawnConfig {
            spacing: 0.5,
            box_size: IVec2::new(56, 6),
            box_center: Vec2::new(32.0, 6.0),
            material_id: 1,
            precompute_initial_volumes: true,
            ..SpawnConfig::for_solver(&cfg)
        };
        let sand_mat = SandMaterial::from_young_modulus(1e5, 0.2);
        solver.register_material(Box::new(sand_mat)); // registers as slot 1
        let _ = solver.spawn_group(floor_spawn);

        let t0 = Instant::now();
        for step in 1..=300u64 {
            solver.step_n(1);
            if step % 100 == 0 {
                let snap = solver.diagnostics_snapshot();
                let body = solver.material_state(0);
                let floor = solver.material_state(1);
                println!(
                    "    step={step:3}  body_y={:.2}  body_spd={:.3}  floor_y={:.2}  nans={}",
                    body.centroid.y, body.avg_speed, floor.centroid.y,
                    snap.non_finite_particle_values
                );
            }
        }
        println!("    wall={:.1}ms total  particles={}", t0.elapsed().as_millis(), solver.particles().len());
    }

    // ── Scenario 2: Water pool filling ──────────────────────────────────────
    println!("\n── Scenario 2: Water pool ──");
    println!("   Water blob falls into a basin formed by friction boundaries.");
    {
        let water_spawn = SpawnConfig {
            spacing: 0.5,
            box_size: IVec2::new(14, 14),
            box_center: Vec2::new(32.0, 50.0),
            material_id: 0,
            precompute_initial_volumes: true,
            ..SpawnConfig::for_solver(&config())
        };
        let water = NewtonianFluidMaterial::water(1.0, 400.0);
        let mut solver = MpmSolver::new(config(), water_spawn)
            .with_default_material(Box::new(water))
            .with_boundary(Box::new(FrictionBoundary::new(2, 0.0)));

        let t0 = Instant::now();
        for step in 1..=400u64 {
            solver.step_n(1);
            if step % 100 == 0 {
                let snap = solver.diagnostics_snapshot();
                let ms = solver.material_state(0);
                println!(
                    "    step={step:3}  y={:.2}  spd={:.3}  J=[{:.3},{:.3}]  nans={}",
                    ms.centroid.y, ms.avg_speed,
                    snap.min_deformation_j, snap.max_deformation_j,
                    snap.non_finite_particle_values
                );
            }
        }
        println!("    wall={:.1}ms total  particles={}", t0.elapsed().as_millis(), solver.particles().len());
    }

    // ── Scenario 3: Sand pile angle of repose ───────────────────────────────
    println!("\n── Scenario 3: Sand pile angle of repose ──");
    println!("   Tall column of sand collapses under gravity. Should form stable pile.");
    {
        let sand_col = SpawnConfig {
            spacing: 0.5,
            box_size: IVec2::new(6, 40),
            box_center: Vec2::new(32.0, 26.0),
            material_id: 0,
            precompute_initial_volumes: true,
            ..SpawnConfig::for_solver(&config())
        };
        let sand = SandMaterial::from_young_modulus(1e5, 0.2);
        let mut solver = MpmSolver::new(config(), sand_col)
            .with_default_material(Box::new(sand))
            .with_boundary(Box::new(FrictionBoundary::new(2, 0.7)));

        let t0 = Instant::now();
        for step in 1..=500u64 {
            solver.step_n(1);
            if step % 125 == 0 {
                let snap = solver.diagnostics_snapshot();
                let ms = solver.material_state(0);
                println!(
                    "    step={step:3}  centroid_y={:.2}  spd={:.4}  avg_J={:.3}  nans={}",
                    ms.centroid.y, ms.avg_speed, ms.avg_det_f,
                    snap.non_finite_particle_values
                );
            }
        }
        let ms = solver.material_state(0);
        println!("    wall={:.1}ms total  particles={}  final_centroid_y={:.2}",
            t0.elapsed().as_millis(), solver.particles().len(),
            ms.centroid.y
        );
        println!("    (dry sand angle of repose ≈25–35°)");
    }

    // ── Scenario 4: Mixed material — mud + elastic body ─────────────────────
    println!("\n── Scenario 4: Elastic creature body in mud ──");
    println!("   Rigid-ish body (E=2e5) sinking into mud. Tests multi-material interaction.");
    {
        let cfg = SolverConfig {
            max_substeps_per_step: 64,
            ..SolverConfig::standard(GRID, DT, gravity())
        };

        let mud_spawn = SpawnConfig {
            spacing: 0.5,
            box_size: IVec2::new(50, 20),
            box_center: Vec2::new(32.0, 12.0),
            material_id: 1,
            precompute_initial_volumes: true,
            ..SpawnConfig::for_solver(&cfg)
        };
        let mud = BinghamFluidMaterial::mud();

        let body_spawn = SpawnConfig {
            spacing: 0.5,
            box_size: IVec2::new(8, 8),
            box_center: Vec2::new(32.0, 50.0),
            material_id: 0,
            precompute_initial_volumes: true,
            ..SpawnConfig::for_solver(&cfg)
        };
        let (lb, ub) = lame_from_young(2e5, 0.3);
        let body = NeoHookeanMaterial::new(lb, ub);

        let mut solver = MpmSolver::new(cfg, body_spawn)
            .with_default_material(Box::new(body))
            .with_boundary(Box::new(FrictionBoundary::new(2, 0.5)));
        solver.register_material(Box::new(mud)); // registers as slot 1
        let _ = solver.spawn_group(mud_spawn);

        let t0 = Instant::now();
        for step in 1..=400u64 {
            solver.step_n(1);
            if step % 100 == 0 {
                let snap = solver.diagnostics_snapshot();
                let body_ms = solver.material_state(0);
                let mud_ms  = solver.material_state(1);
                println!(
                    "    step={step:3}  body_y={:.2}  body_spd={:.3}  mud_y={:.2}  nans={}",
                    body_ms.centroid.y, body_ms.avg_speed, mud_ms.centroid.y,
                    snap.non_finite_particle_values
                );
            }
        }
        println!("    wall={:.1}ms total  particles={}", t0.elapsed().as_millis(), solver.particles().len());
    }
}

// ════════════════════════════════════════════════════════════════════════════════
// Main
// ════════════════════════════════════════════════════════════════════════════════

fn main() {
    let mode = std::env::args().nth(1).unwrap_or_default();
    match mode.as_str() {
        "sweep"    => run_sweep(),
        "perf"     => run_perf(),
        "scenario" => run_scenarios(),
        _          => run_basic(),
    }
}

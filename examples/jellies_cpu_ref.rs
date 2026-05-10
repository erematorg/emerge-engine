/// Headless CPU reference for basic_jellies — same params as GPU example.
/// cargo run --example jellies_cpu_ref
use emerge::{
    CorotatedMaterial, MpmSolver, NeoHookeanMaterial, SolverConfig, SpawnConfig,
    ViscoelasticMaterial,
};
use glam::{IVec2, Vec2};

fn main() {
    let config = SolverConfig::standard(64, 0.1, Vec2::new(0.0, -0.3));

    let neo = NeoHookeanMaterial::new(10.0, 20.0);
    let cor = CorotatedMaterial::new(30.0, 60.0);
    let vis = ViscoelasticMaterial::new(10.0, 15.0, 0.15); // matches GPU DEFAULTS

    let spawn = |cx: f32, mat: u32| SpawnConfig {
        spacing: 0.5,
        box_size: IVec2::new(14, 14),
        box_center: Vec2::new(cx, 50.0),
        material_id: mat,
        precompute_initial_volumes: true,
        ..SpawnConfig::for_solver(&config)
    };

    let mut solver = MpmSolver::new(config, spawn(14.0, 0))
        .with_default_material(Box::new(neo))
        .with_material(1, Box::new(cor))
        .with_material(2, Box::new(vis));
    let _ = solver.spawn_region(spawn(32.0, 1));
    let _ = solver.spawn_region(spawn(50.0, 2));

    let log_frames = [60u32, 120, 180, 240, 300, 360, 420, 480, 540, 600];
    let mut frame = 0u32;

    // print sub_dt for a few key frames to understand CFL behavior
    for early in [1u32, 2, 3, 4, 5] {
        while frame < early { solver.step(); frame += 1; }
        let snap = solver.diagnostics_snapshot();
        println!("frame {:3}: substeps={} effective_dt={:.6}", frame, snap.substeps_last_step, snap.effective_dt);
    }
    for &target in &log_frames {
        while frame < target { solver.step(); frame += 1; }

        let ps = solver.particles();
        let snap = solver.diagnostics_snapshot();
        let (n0, cx0, cy0, j0_lo, j0_hi) = mat_stats(ps, 0);
        let (n1, cx1, cy1, j1_lo, j1_hi) = mat_stats(ps, 1);
        let (n2, cx2, cy2, j2_lo, j2_hi) = mat_stats(ps, 2);
        println!("── frame {:3}  [CPU]  J=[{:.3},{:.3}]  substeps={} ──", frame, snap.min_deformation_j, snap.max_deformation_j, snap.substeps_last_step);
        println!("  [mat 0 neo] n={}  cx=({:6.1},{:6.1})  J=[{:.3},{:.3}]", n0, cx0, cy0, j0_lo, j0_hi);
        println!("  [mat 1 cor] n={}  cx=({:6.1},{:6.1})  J=[{:.3},{:.3}]", n1, cx1, cy1, j1_lo, j1_hi);
        println!("  [mat 2 vis] n={}  cx=({:6.1},{:6.1})  J=[{:.3},{:.3}]", n2, cx2, cy2, j2_lo, j2_hi);
    }
}

fn mat_stats(ps: &emerge::particle::Particles, id: u32) -> (usize, f32, f32, f32, f32) {
    let it: Vec<_> = ps.iter().filter(|p| p.material_id == id).collect();
    let n = it.len();
    if n == 0 { return (0, 0.0, 0.0, 1.0, 1.0); }
    let cx = it.iter().map(|p| p.x.x).sum::<f32>() / n as f32;
    let cy = it.iter().map(|p| p.x.y).sum::<f32>() / n as f32;
    let j_lo = it.iter().map(|p| p.deformation_gradient.determinant()).fold(f32::INFINITY, f32::min);
    let j_hi = it.iter().map(|p| p.deformation_gradient.determinant()).fold(f32::NEG_INFINITY, f32::max);
    (n, cx, cy, j_lo, j_hi)
}

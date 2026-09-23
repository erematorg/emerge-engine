extern crate emerge_engine as emerge;

/// TEMP diagnostic (2026-09-18) -- decides "kernel vs schedule" for the GPU
/// fluid impact explosion, now that the CPU twin of the exact scene
/// (`examples/cpu/fragmentation_check_cpu.rs`) is proven stable with no
/// damping at all.
///
/// Every step: copy CPU's state into GPU, advance BOTH by exactly one
/// substep with the SAME fixed dt, compare. Divergence cannot accumulate
/// (both start every step from identical state), so any difference is a
/// real per-substep computational difference, not chaotic amplification of
/// rounding noise -- the flaw of a free-running lockstep.
///
/// CPU drives the trajectory (it is the stable reference), so the GPU kernel
/// gets tested on exactly the states CPU passes through, impact included.
///
/// - Kernels agree to rounding the whole run  -> the bug is in the GPU's
///   SCHEDULE (dt choice / missing per-substep adaptivity / missing retry).
/// - A real difference appears                -> the bug is in a GPU KERNEL;
///   the dump names the particle and which quantity diverges first.
///
/// REQUIRES the GPU shear damping bypassed in g2p.wgsl (CPU has none; with
/// it on, the damping itself is a known, deliberate difference).
///
///   cargo run --release --example cpu_gpu_single_step_kernel_diff --features gpu
use emerge::gpu::GpuSimulation;
use emerge::{
    GpuFieldEntry, LinearDragField, MaterialRegistry, NewtonianFluidMaterial, SimConfig,
    Simulation, SpawnRegion,
};
use glam::{IVec2, Vec2};
use pollster::block_on;
use wgpu::InstanceDescriptor;

const GRID: usize = 64;
const MAT_WATER: u32 = 0;
const SPACING: f32 = 0.5;
const WATER_RHO_GRID: f32 = 0.1;
// Small enough that CPU stays stable through impact at a FIXED dt (CPU's own
// adaptive run used ~2.6e-4 at CFL_DIVISOR=20 and stayed clean).
// FIXED_DT env overrides the default: the per-step CPU/GPU difference that
// matters turned out to grow with dt (negligible at 1e-4, visible at the
// scene's real ~5e-3), so it must be testable at the real substep size.
const DEFAULT_FIXED_DT: f32 = 1.0e-4;
const SIM_SECONDS: f32 = 2.0; // impact is at ~1.1-1.2 s

fn isolated_count(xs: &[Vec2]) -> usize {
    const CELL: f32 = SPACING * 3.0;
    const ISOLATION_THRESHOLD: f32 = SPACING * 4.0;
    let mut buckets: std::collections::HashMap<(i32, i32), Vec<usize>> = Default::default();
    for (i, x) in xs.iter().enumerate() {
        buckets
            .entry(((x.x / CELL).floor() as i32, (x.y / CELL).floor() as i32))
            .or_default()
            .push(i);
    }
    let mut count = 0;
    for (i, x) in xs.iter().enumerate() {
        let (bx, by) = ((x.x / CELL).floor() as i32, (x.y / CELL).floor() as i32);
        let mut best = f32::MAX;
        for dx in -1..=1 {
            for dy in -1..=1 {
                if let Some(b) = buckets.get(&(bx + dx, by + dy)) {
                    for &j in b {
                        if j != i {
                            best = best.min((xs[j] - *x).length());
                        }
                    }
                }
            }
        }
        if best > ISOLATION_THRESHOLD {
            count += 1;
        }
    }
    count
}

fn make_water(config: &SimConfig) -> NewtonianFluidMaterial {
    let v_max_grid = (2.0 * 0.3 * 52.0f32).sqrt();
    let c_ref_m_s = 10.0 * v_max_grid * config.dx_meters;
    let visc = config.visc_from_si_physical(1.0e-3, 1000.0);
    let mut water = NewtonianFluidMaterial::new(
        WATER_RHO_GRID,
        visc,
        1000.0 * c_ref_m_s * c_ref_m_s / 3.0,
        3.0,
    );
    water.bulk_viscosity = 3.0 * visc;
    water.pressure_floor = config.stress_from_si_physical(-100_000.0, 1000.0);
    water
}

fn main() {
    let fixed_dt: f32 = std::env::var("FIXED_DT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_FIXED_DT);
    let report_every = ((0.1 / fixed_dt).round() as u64).max(1);
    let config = SimConfig {
        adaptive_timestep: false,
        dt: fixed_dt,
        max_substeps_per_step: 1,
        min_dt: 1.0e-5,
        cfl_include_affine_speed: false,
        material_cfl_coefficient: 0.3,
        gravity: Vec2::new(0.0, -981.0 * 0.003),
        fluid_near_wall_cfl_scale: 20.0,
        ..SimConfig::earth(GRID, 0.01, fixed_dt)
    };
    let spawn = SpawnRegion {
        spacing: SPACING,
        box_size: IVec2::new(14, 52),
        box_center: Vec2::new(20.0, 30.0),
        material_id: MAT_WATER,
        mass_override: Some(WATER_RHO_GRID * SPACING * SPACING),
        ..SpawnRegion::for_sim(&config)
    };
    let mut cpu =
        Simulation::new(config, spawn).with_default_material(Box::new(make_water(&config)));
    // NO_DRAG: drop the drag field on BOTH sides -- isolates the GPU
    // force_fields pass (which re-writes the whole particle struct after
    // particles_update) from everything else.
    let no_drag = std::env::var("NO_DRAG").is_ok();
    if !no_drag {
        cpu.add_force_field(Box::new(LinearDragField::new(
            Vec2::ZERO,
            0.1,
            1 << MAT_WATER,
        )));
    }

    let instance = wgpu::Instance::new(&InstanceDescriptor {
        backend_options: wgpu::BackendOptions {
            dx12: wgpu::Dx12BackendOptions {
                shader_compiler: wgpu::Dx12Compiler::StaticDxc,
                ..Default::default()
            },
            ..Default::default()
        },
        ..Default::default()
    });
    let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .expect("adapter");
    println!("GPU adapter: {:?}", adapter.get_info());
    let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_limits: adapter.limits(),
        ..Default::default()
    }))
    .expect("device");
    let registry = MaterialRegistry::with_default(Box::new(make_water(&config)));
    let mut gpu = GpuSimulation::with_device(
        std::sync::Arc::new(device),
        std::sync::Arc::new(queue),
        config,
        cpu.particles().to_vec(),
        registry,
    );
    if !no_drag {
        gpu.add_force_field_gpu(GpuFieldEntry::linear_drag(Vec2::ZERO, 0.1, 1 << MAT_WATER));
    }
    println!(
        "drag field: {}",
        if no_drag { "OFF (both)" } else { "on (both)" }
    );

    let n = cpu.particles().x.len();
    let steps = (SIM_SECONDS / fixed_dt).round() as u64;
    println!("n={n}  fixed dt={fixed_dt}  steps={steps}  (state re-synced CPU->GPU every step)");

    let no_resync = std::env::var("NO_RESYNC").is_ok();
    println!(
        "mode: {}",
        if no_resync {
            "NO_RESYNC (GPU free-runs on its own state)"
        } else {
            "re-sync every step"
        }
    );
    let mut worst_rel_v_ever = 0.0f32;
    let mut dumped = false;
    let mut dumped_j = false;
    for step in 1..=steps {
        // Identical starting state for this one step -- unless NO_RESYNC is
        // set: then GPU free-runs on its OWN state at the SAME fixed dt that
        // is stable on CPU. Stable -> the bug is GPU's dt schedule. Explodes
        // -> the bug is GPU state persisting across steps (which the re-sync
        // was resetting every step and so could never see).
        if !no_resync {
            *gpu.particles_mut() = cpu.particles().to_vec();
            gpu.mark_particles_dirty();
        }

        // Inputs to this step (identical on both sides when re-synced).
        let j_in: Vec<f32> = cpu
            .particles()
            .deformation_gradient
            .iter()
            .map(|f| f.determinant())
            .collect();
        let x_in: Vec<Vec2> = cpu.particles().x.clone();
        let trc_in: Vec<f32> = cpu
            .particles()
            .velocity_gradient
            .iter()
            .map(|m| m.x_axis.x + m.y_axis.y)
            .collect();

        cpu.step_n(1);
        gpu.step_frame();
        gpu.sync_particles_blocking();

        let c = cpu.particles();
        let g = gpu.particles();
        let (mut worst_rel_v, mut worst_i) = (0.0f32, 0usize);
        let (mut max_dv, mut max_dc, mut max_dj, mut max_v) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        for (i, gp) in g.iter().enumerate().take(n) {
            let dv = (c.v[i] - gp.v).length();
            let dc = (c.velocity_gradient[i] - gp.velocity_gradient)
                .abs()
                .to_cols_array()
                .into_iter()
                .fold(0.0f32, f32::max);
            let dj = (c.deformation_gradient[i].determinant()
                - g[i].deformation_gradient.determinant())
            .abs();
            let rel_v = dv / (c.v[i].length() + 1.0);
            max_dv = max_dv.max(dv);
            max_dc = max_dc.max(dc);
            max_dj = max_dj.max(dj);
            max_v = max_v.max(c.v[i].length());
            if rel_v > worst_rel_v {
                worst_rel_v = rel_v;
                worst_i = i;
            }
        }
        worst_rel_v_ever = worst_rel_v_ever.max(worst_rel_v);

        // First step where J itself disagrees by > 1e-3: tell apart "the
        // gathered C differs" from "a later step (clamp / projection /
        // boundary) rewrote J" by checking each side against its OWN
        // continuity update J_in * exp(dt * tr(C)).
        if !no_resync && !dumped_j {
            let (mut worst_dj, mut wi) = (0.0f32, 0usize);
            for (i, gp) in g.iter().enumerate().take(n) {
                let dj = (c.deformation_gradient[i].determinant()
                    - gp.deformation_gradient.determinant())
                .abs();
                if dj > worst_dj {
                    worst_dj = dj;
                    wi = i;
                }
            }
            if worst_dj > 1.0e-3 {
                // Systematic check, not one particle: for every particle whose
                // GPU J moved at all, which combination of its own C entries
                // matches the divergence the GPU actually applied?
                let cands = [
                    "trace C00+C11",
                    "col0 C00+C10",
                    "col1 C01+C11",
                    "row0 C00+C01",
                    "row1 C10+C11",
                ];
                let mut hits = [0usize; 5];
                let mut checked = 0usize;
                for k in 0..n {
                    let jo = g[k].deformation_gradient.determinant();
                    if (jo / j_in[k] - 1.0).abs() < 1.0e-4 {
                        continue; // too small a change to resolve the divergence reliably
                    }
                    checked += 1;
                    let used = (jo / j_in[k]).ln() / fixed_dt;
                    let m = g[k].velocity_gradient;
                    let vals = [
                        m.x_axis.x + m.y_axis.y,
                        m.x_axis.x + m.x_axis.y,
                        m.y_axis.x + m.y_axis.y,
                        m.x_axis.x + m.y_axis.x,
                        m.x_axis.y + m.y_axis.y,
                    ];
                    for (h, v) in hits.iter_mut().zip(vals) {
                        if (used - v).abs() < 2.0e-3 * (1.0 + v.abs()) {
                            *h += 1;
                        }
                    }
                }
                // Shader-side truth (temporary instrumentation in
                // particles_update.wgsl): scalar_field = div_v it computed,
                // internal_pressure = the [1].y it read.
                let k = wi;
                let m = g[k].velocity_gradient;
                println!(
                    "\n### SHADER SAW (particle {k}): div_v={:.6}  [1].y={:.6}   READBACK C: [0].x={:.6} [0].y={:.6} [1].x={:.6} [1].y={:.6}  applied div={:.6}",
                    g[k].scalar_field,
                    g[k].internal_pressure,
                    m.x_axis.x,
                    m.x_axis.y,
                    m.y_axis.x,
                    m.y_axis.y,
                    (g[k].deformation_gradient.determinant() / j_in[k]).ln() / fixed_dt,
                );
                println!(
                    "\n### which C combination matches the GPU's applied divergence ({checked} particles checked):"
                );
                for (name, h) in cands.iter().zip(hits) {
                    println!("    {name:<14} matches {h:5} / {checked}");
                }
                dumped_j = true;
                let i = wi;
                let tr_c = c.velocity_gradient[i].x_axis.x + c.velocity_gradient[i].y_axis.y;
                let tr_g = g[i].velocity_gradient.x_axis.x + g[i].velocity_gradient.y_axis.y;
                println!(
                    "
### FIRST J DIFFERENCE > 1e-3 at t={:.4}s step={step}, particle {i}
                     input: x={:?}  J_in={:.6}  tr(C_in)={:.5}
                     div ACTUALLY used for J, ln(J_out/J_in)/dt:  CPU={:.5}  GPU={:.5}
                     OLD formula J_in*det(I+dt*C_out):  CPU={:.6}  GPU={:.6}   det(C_out): CPU={:.3}  GPU={:.3}
                     full F_out (cols): CPU={:?}  GPU={:?}
                     full C_out (cols): CPU={:?}  GPU={:?}
                     volume/density out: CPU={:.6e}/{:.6}  GPU={:.6e}/{:.6}  mass={:.6e}
                     CPU : x={:?} v={:?} tr(C)={tr_c:.5}  J_out={:.6}  J_in*exp(dt*trC)={:.6}
                     GPU : x={:?} v={:?} tr(C)={tr_g:.5}  J_out={:.6}  J_in*exp(dt*trC)={:.6}
",
                    step as f32 * fixed_dt,
                    x_in[i], j_in[i], trc_in[i],
                    (c.deformation_gradient[i].determinant() / j_in[i]).ln() / fixed_dt,
                    (g[i].deformation_gradient.determinant() / j_in[i]).ln() / fixed_dt,
                    j_in[i] * (glam::Mat2::IDENTITY + fixed_dt * c.velocity_gradient[i]).determinant(),
                    j_in[i] * (glam::Mat2::IDENTITY + fixed_dt * g[i].velocity_gradient).determinant(),
                    c.velocity_gradient[i].determinant(),
                    g[i].velocity_gradient.determinant(),
                    c.deformation_gradient[i].to_cols_array(),
                    g[i].deformation_gradient.to_cols_array(),
                    c.velocity_gradient[i].to_cols_array(),
                    g[i].velocity_gradient.to_cols_array(),
                    c.volume[i], c.density[i], g[i].volume, g[i].density, g[i].mass,
                    c.x[i], c.v[i], c.deformation_gradient[i].determinant(), j_in[i] * (fixed_dt * tr_c).exp(),
                    g[i].x, g[i].v, g[i].deformation_gradient.determinant(), j_in[i] * (fixed_dt * tr_g).exp(),
                );
            }
        }

        if no_resync && step % report_every == 0 {
            let gv = g.iter().map(|p| p.v.length()).fold(0.0f32, f32::max);
            let (mut gjmin, mut gjmax) = (f32::MAX, f32::MIN);
            for p in g {
                let j = p.deformation_gradient.determinant();
                gjmin = gjmin.min(j);
                gjmax = gjmax.max(j);
            }
            let gxs: Vec<Vec2> = g.iter().map(|p| p.x).collect();
            println!(
                "t={:.2}s  CPU max_speed={max_v:7.3} isolated={:3}   |   GPU max_speed={gv:7.3} J=[{gjmin:.3},{gjmax:.3}] isolated={:3}",
                step as f32 * fixed_dt,
                isolated_count(&c.x),
                isolated_count(&gxs),
            );
        }
        if !no_resync && step % report_every == 0 {
            println!(
                "t={:.2}s  cpu_max_speed={max_v:7.3}  max|dv|={max_dv:.2e}  max|dC|={max_dc:.2e}  max|dJ|={max_dj:.2e}  worst_rel_v={worst_rel_v:.2e} (p{worst_i})",
                step as f32 * fixed_dt
            );
        }
        // A per-step velocity disagreement above 1% of (|v|+1) is far beyond
        // f32 rounding for a single substep -- dump the first one in full.
        if !no_resync && !dumped && worst_rel_v > 1.0e-2 {
            dumped = true;
            let i = worst_i;
            let wall = [
                c.x[i].x,
                c.x[i].y,
                GRID as f32 - c.x[i].x,
                GRID as f32 - c.x[i].y,
            ]
            .into_iter()
            .fold(f32::MAX, f32::min);
            println!(
                "\n*** FIRST REAL PER-STEP DIFFERENCE at t={:.4}s step={step}, particle {i} (dist to nearest wall {wall:.2}) ***\n\
                 CPU: x={:?} v={:?} C={:?} J={:.5} vol={:.5e} dens={:.5}\n\
                 GPU: x={:?} v={:?} C={:?} J={:.5} vol={:.5e} dens={:.5}\n",
                step as f32 * fixed_dt,
                c.x[i],
                c.v[i],
                c.velocity_gradient[i].to_cols_array(),
                c.deformation_gradient[i].determinant(),
                c.volume[i],
                c.density[i],
                g[i].x,
                g[i].v,
                g[i].velocity_gradient.to_cols_array(),
                g[i].deformation_gradient.determinant(),
                g[i].volume,
                g[i].density,
            );
        }
    }
    println!(
        "\nworst per-step relative velocity difference over the whole run: {worst_rel_v_ever:.2e}"
    );
}

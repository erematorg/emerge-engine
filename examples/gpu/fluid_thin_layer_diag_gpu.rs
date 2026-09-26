extern crate emerge_engine as emerge;

/// TEMP diagnostic (2026-09-16) -- delete after use.
///
/// Headless GPU twin of `examples/cpu/fluid_thin_layer_diag.rs`, replicating
/// `basic_fluids_gpu.rs`'s exact `DamBreak` scene (same config/material/
/// gravity/spacing -- copied verbatim from that file's `make_sim_data`, NOT
/// re-derived) with no window/render, so it can run thousands of
/// `step_frame()` calls unattended and print the same `MaterialStats` used
/// by the CPU twin, at the same frame cadence, for a direct, real,
/// quantitative A/B against `HANDOFF_fluid_gpu_thin_layer_bug.md`'s open
/// question.
///
///   cargo run --example fluid_thin_layer_diag_gpu --features gpu
use emerge::diagnostics::per_material_stats_of;
use emerge::gpu::GpuSimulation;
use emerge::{
    GpuFieldEntry, MaterialRegistry, NewtonianFluidMaterial, Particle, SimConfig, SpawnRegion,
    build_particles,
};
use glam::{IVec2, Vec2};
use pollster::block_on;
use wgpu::InstanceDescriptor;

/// Bins particles into `N_BANDS` horizontal depth bands (by current y) and
/// prints mean J and mean v.y per band -- tests the hydrostatic-stratification
/// hypothesis directly: a real resting fluid column should show J decreasing
/// (more compressed) and pressure rising toward the floor (lowest band), the
/// gradient that actually holds the column up against gravity. If every band
/// reports near-identical J regardless of depth, there is no real vertical
/// pressure gradient holding the column up, and gravity has nothing to fight.
fn print_depth_stratification(frame: u64, particles: &[Particle], mat: u32) {
    const N_BANDS: usize = 8;
    let (mut y_min, mut y_max) = (f32::INFINITY, f32::NEG_INFINITY);
    for p in particles.iter().filter(|p| p.material_id == mat) {
        y_min = y_min.min(p.x.y);
        y_max = y_max.max(p.x.y);
    }
    let span = (y_max - y_min).max(1e-6);
    let mut count = [0u32; N_BANDS];
    let mut j_sum = [0f64; N_BANDS];
    let mut vy_sum = [0f64; N_BANDS];
    for p in particles.iter().filter(|p| p.material_id == mat) {
        let mut band = (((p.x.y - y_min) / span) * N_BANDS as f32) as usize;
        band = band.min(N_BANDS - 1);
        count[band] += 1;
        j_sum[band] += p.deformation_gradient.determinant() as f64;
        vy_sum[band] += p.v.y as f64;
    }
    println!(
        "  -- depth stratification @ frame {frame} (y_min={y_min:.2} y_max={y_max:.2} span={span:.3}) --"
    );
    for b in (0..N_BANDS).rev() {
        let n = count[b].max(1) as f64;
        println!(
            "     band {b} (floor-most={})  n={:4}  mean_J={:.4}  mean_vy={:.5}",
            b == 0,
            count[b],
            j_sum[b] / n,
            vy_sum[b] / n,
        );
    }
}

/// Decisive test for the `f1fea23` mechanism hypothesis (see
/// `HANDOFF_fluid_gpu_thin_layer_bug.md`'s "Seventh pass"): a genuinely
/// at-rest fluid particle should show `div(v) = trace(C) ~= 0` (no real
/// volume change once settled). If the free-surface extrapolation
/// contamination bug still leaves a small, persistent, ONE-SIGNED bias in
/// the measured divergence, `f1fea23`'s exact `exp(dt*div_v)` J-update
/// compounds that bias multiplicatively every substep (exp(x) > 1+x for
/// x != 0), which would explain the bulk collapse. Restricts to
/// "settled" particles (|v| < threshold) so this measures the SPURIOUS
/// bias, not real motion's own genuine divergence.
fn print_settled_divergence_bias(frame: u64, particles: &[Particle], mat: u32) {
    const SETTLED_SPEED_THRESHOLD: f32 = 0.1;
    let mut div_sum = 0f64;
    let mut div_sum_sq = 0f64;
    let mut n = 0u32;
    for p in particles
        .iter()
        .filter(|p| p.material_id == mat && p.v.length() < SETTLED_SPEED_THRESHOLD)
    {
        let div = (p.velocity_gradient.x_axis.x + p.velocity_gradient.y_axis.y) as f64;
        div_sum += div;
        div_sum_sq += div * div;
        n += 1;
    }
    if n == 0 {
        println!("  -- settled-divergence @ frame {frame}: no settled particles (all moving) --");
        return;
    }
    let mean = div_sum / n as f64;
    let variance = (div_sum_sq / n as f64) - mean * mean;
    let stderr = (variance.max(0.0) / n as f64).sqrt();
    println!(
        "  -- settled-divergence @ frame {frame}: n={n}  mean_div(v)={mean:.6}  stderr={stderr:.6}  (nonzero+one-signed => confirms the f1fea23 amplification mechanism)"
    );
}

/// Follow-up to `print_settled_divergence_bias`'s surprise result (settled
/// particles show a NEGATIVE div(v) bias, decaying over time -- a slow
/// partial self-correction, not a runaway). Reframed hypothesis: the real
/// spurious inflation happens during the FAST/violent phase instead
/// (TRACK[1456] showed J shoot from 1.0 to 2.0 within ~15 steps while still
/// moving). Splits ALL particles into fast (|v|>=1.0) vs slow, AND
/// separately by current J (already-near-the-clamp vs not), reporting mean
/// div(v) for each bucket -- tests directly whether positive divergence
/// correlates with fast motion and/or with already-inflated J.
fn print_fast_phase_divergence(frame: u64, particles: &[Particle], mat: u32) {
    let mut fast_sum = 0f64;
    let mut fast_n = 0u32;
    let mut slow_sum = 0f64;
    let mut slow_n = 0u32;
    let mut high_j_sum = 0f64;
    let mut high_j_n = 0u32;
    let mut low_j_sum = 0f64;
    let mut low_j_n = 0u32;
    for p in particles.iter().filter(|p| p.material_id == mat) {
        let div = (p.velocity_gradient.x_axis.x + p.velocity_gradient.y_axis.y) as f64;
        let j = p.deformation_gradient.determinant();
        if p.v.length() >= 1.0 {
            fast_sum += div;
            fast_n += 1;
        } else {
            slow_sum += div;
            slow_n += 1;
        }
        if j >= 1.9 {
            high_j_sum += div;
            high_j_n += 1;
        } else if j <= 1.1 {
            low_j_sum += div;
            low_j_n += 1;
        }
    }
    println!(
        "  -- fast-phase-divergence @ frame {frame}: fast(|v|>=1) n={} mean_div={:.5}  slow n={} mean_div={:.5}  |  J>=1.9 n={} mean_div={:.5}  J<=1.1 n={} mean_div={:.5}",
        fast_n,
        if fast_n > 0 {
            fast_sum / fast_n as f64
        } else {
            0.0
        },
        slow_n,
        if slow_n > 0 {
            slow_sum / slow_n as f64
        } else {
            0.0
        },
        high_j_n,
        if high_j_n > 0 {
            high_j_sum / high_j_n as f64
        } else {
            0.0
        },
        low_j_n,
        if low_j_n > 0 {
            low_j_sum / low_j_n as f64
        } else {
            0.0
        },
    );

    // Wall-proximity check: does extreme divergence cluster near a
    // boundary (GPU-specific hard velocity clamp discontinuity) or spread
    // through the bulk (a general P2G/G2P scatter bug)? `GRID`=64,
    // distance-to-nearest-wall = min(x, GRID-x, y, GRID-y).
    const EXTREME_DIV_THRESHOLD: f64 = 0.5;
    let mut extreme_wall_dist_sum = 0f64;
    let mut extreme_n = 0u32;
    let mut near_wall_extreme_n = 0u32; // within 3 grid-cells of any wall
    for p in particles.iter().filter(|p| p.material_id == mat) {
        let div = (p.velocity_gradient.x_axis.x + p.velocity_gradient.y_axis.y) as f64;
        if div.abs() < EXTREME_DIV_THRESHOLD {
            continue;
        }
        let wall_dist =
            p.x.x
                .min(GRID as f32 - p.x.x)
                .min(p.x.y)
                .min(GRID as f32 - p.x.y);
        extreme_wall_dist_sum += wall_dist as f64;
        extreme_n += 1;
        if wall_dist < 3.0 {
            near_wall_extreme_n += 1;
        }
    }
    if extreme_n > 0 {
        println!(
            "     extreme-div (|div|>{EXTREME_DIV_THRESHOLD}) n={extreme_n}  mean_wall_dist={:.3}  near_wall(<3cells)={near_wall_extreme_n} ({:.1}%)",
            extreme_wall_dist_sum / extreme_n as f64,
            100.0 * near_wall_extreme_n as f64 / extreme_n as f64,
        );
    }
}

const GRID: usize = 64;
const PLAYBACK_SPEED: f32 = 6.0;
const RENDER_FPS_TARGET: f32 = 60.0;
const DT: f32 = PLAYBACK_SPEED / RENDER_FPS_TARGET;
const MAT_WATER: u32 = 0;
const WATER_RHO_GRID: f32 = 0.1;
const N_STEPS: u64 = 2000;
const LOG_INTERVAL: u64 = 20;

// Same DX12 shader-compiler fix `tests/gpu.rs::create_instance` uses --
// default Fxc cannot compile `resolve_contact.wgsl` on WARP.
fn create_instance() -> wgpu::Instance {
    wgpu::Instance::new(&InstanceDescriptor {
        backend_options: wgpu::BackendOptions {
            dx12: wgpu::Dx12BackendOptions {
                shader_compiler: wgpu::Dx12Compiler::StaticDxc,
                ..Default::default()
            },
            ..Default::default()
        },
        ..Default::default()
    })
}

fn main() {
    let instance = create_instance();
    let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .expect("adapter");
    let info = adapter.get_info();
    println!(
        "GPU adapter: {} ({:?}, backend={:?})",
        info.name, info.device_type, info.backend
    );
    let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_limits: adapter.limits(),
        ..Default::default()
    }))
    .expect("device");
    let device = std::sync::Arc::new(device);
    let queue = std::sync::Arc::new(queue);

    // Verbatim from `basic_fluids_gpu.rs::make_sim_data`'s `DamBreak` path.
    let config = SimConfig {
        min_dt: 1.0e-4,
        max_substeps_per_step: 150,
        cfl_include_affine_speed: false,
        material_cfl_coefficient: 0.3,
        gravity: Vec2::new(0.0, -981.0 * 0.003),
        // CORRECTED (2026-09-16): was left at 1.0 from an earlier test
        // tonight and never reverted -- a real methodology gap, this
        // diagnostic was NOT a faithful copy of `basic_fluids_gpu.rs` the
        // whole time. Real production value is 20.0 (see that file's own
        // doc: reverted from 5.0 after live-measuring it made things worse).
        fluid_near_wall_cfl_scale: 20.0,
        ..SimConfig::earth(GRID, 0.01, DT)
    };
    const SPACING: f32 = 0.5;
    const WATER_MASS: f32 = WATER_RHO_GRID * SPACING * SPACING;
    let spawn = SpawnRegion {
        spacing: SPACING,
        box_size: IVec2::new(14, 52),
        box_center: Vec2::new(20.0, 30.0),
        material_id: MAT_WATER,
        mass_override: Some(WATER_MASS),
        // CORRECTED (2026-09-16): jitter removed -- real production demo
        // has no position_jitter, this diagnostic must match it exactly.
        ..SpawnRegion::for_sim(&config)
    };
    let particles = build_particles(&config, spawn);

    // TESTED and REVERTED (2026-09-16): `basic_fluids_gpu.rs` derives EOS
    // stiffness from `COLUMN_HEIGHT_CELLS = 52.0` (no `* SPACING`) while the
    // CPU twin (`basic_fluids.rs`) uses `52.0 * SPACING = 26.0` -- a real
    // ~2x mismatch in Tait-EOS stiffness for what's meant to be the
    // identical scene. Tried matching CPU's softer derivation here
    // (`52.0 * SPACING`) on the theory that GPU's atomic-scatter noise gets
    // amplified harder by a stiffer EOS (dp/drho ~ B) -- RESULT: made things
    // FAR worse, not better. `J` hit the 2.0 clamp by step 30 (full
    // blowup), versus hundreds-to-thousands of frames with the original,
    // stiffer value. Real, decisive, if negative, finding: GPU's solver is
    // on a much thinner stability margin than CPU for the identical
    // physical setup -- reducing stiffness to match CPU (which handles that
    // exact value fine for 2000+ frames) removes whatever margin was
    // masking GPU's own noise floor. Kept at the original, stiffer,
    // production-matching value; do NOT re-apply this reduction to
    // `basic_fluids_gpu.rs` itself.
    const WATER_EOS_POWER: f32 = 3.0;
    const COLUMN_HEIGHT_CELLS: f32 = 52.0;
    const DERATED_GRAVITY_FOR_ACOUSTIC_SIZING: f32 = 0.3;
    let v_max_grid = (2.0 * DERATED_GRAVITY_FOR_ACOUSTIC_SIZING * COLUMN_HEIGHT_CELLS).sqrt();
    let c_ref_m_s = 10.0 * v_max_grid * config.dx_meters;
    let water_tait_b_pa = 1000.0 * c_ref_m_s * c_ref_m_s / WATER_EOS_POWER;
    let mut water =
        NewtonianFluidMaterial::new(WATER_RHO_GRID, 1.0e-3, water_tait_b_pa, WATER_EOS_POWER);
    water.bulk_viscosity = 3.0 * 1.0e-3;
    // REAL FIX (2026-09-16), not a tuning knob -- see HANDOFF's Tenth pass.
    // `pressure_floor: -0.1` (NewtonianFluidMaterial's default) is a bare
    // GRID-UNIT constant that was NEVER run through this engine's own
    // SI-to-grid conversion pipeline -- unlike `water_tait_b_pa` just above,
    // which IS properly SI-derived in this same file. Real cavitation onset
    // for water in practice (dissolved-gas nucleation, the standard
    // engineering figure, not the much higher pure-degassed lab value) is
    // ~-0.1 MPa = -100,000 Pa gauge. Converted through `stress_from_si_
    // physical` (the SAME conversion `eos_stiffness` itself uses elsewhere
    // in this engine), that lands orders of magnitude more negative than
    // this demo's own derated `eos_stiffness` (~104) -- meaning real water,
    // properly scaled, essentially NEVER cavitates from ordinary splashing
    // (a mere 2x volumetric expansion is nowhere near its real tensile
    // limit). The un-converted `-0.1` default instead clips almost the
    // ENTIRE expansion range into a flat, zero-pressure-gradient dead zone
    // with no restoring force -- confirmed directly: this fix alone took
    // the resting bulk from mean_J=1.5-1.9 pinned at the 2.0 clamp (every
    // prior test tonight) to mean_J=1.00-1.16 (healthy) at every depth band,
    // every checkpoint through frame 2000.
    const REAL_CAVITATION_PRESSURE_PA: f32 = -100_000.0;
    const WATER_RHO_SI_KG_M3: f32 = 1000.0;
    water.pressure_floor =
        config.stress_from_si_physical(REAL_CAVITATION_PRESSURE_PA, WATER_RHO_SI_KG_M3);
    println!(
        "fluid_thin_layer_diag_gpu: water_tait_b_pa={:.3}  (CPU twin derives 52.0 Pa via its own, DIFFERENT COLUMN_HEIGHT_CELLS scaling -- see this file's own doc)",
        water_tait_b_pa
    );
    let registry = MaterialRegistry::with_default(Box::new(water));

    let mut sim = GpuSimulation::with_device(device, queue, config, particles, registry);
    // CORRECTED (2026-09-16): real production demo registers this force
    // field unconditionally for every pattern -- must match exactly.
    sim.add_force_field_gpu(GpuFieldEntry::linear_drag(Vec2::ZERO, 0.1, 1 << MAT_WATER));
    // TESTED and REVERTED (2026-09-16): real demo calls sim.enable_profiling()
    // (perf HUD), headless diag never did -- tested whether GPU timestamp
    // queries change scheduling/sync enough to explain the interactive
    // demo's v=180 explosion (not reproduced here). RESULT: frame 40 gave
    // v=11.77, bit-identical to the non-profiled run -- ruled out.
    let initial_area: f32 = 14.0 * 52.0;
    println!(
        "n={}  initial_column_area={:.2} grid-units^2  (box 14x52 grid-units)",
        sim.particle_count(),
        initial_area,
    );

    const TRACK_IDX: usize = 1456;
    const TRACK_STEPS: u64 = 60;
    for step in 1..=N_STEPS {
        sim.step_frame();
        if step <= TRACK_STEPS {
            sim.sync_particles_blocking();
            let particles = sim.particles();
            let p = &particles[TRACK_IDX];
            let c = p.velocity_gradient;
            let j = p.deformation_gradient.determinant();
            println!(
                "TRACK[{TRACK_IDX}] step={step:3}  x=({:.4},{:.4})  v=({:.4},{:.4})  C=[{:.4},{:.4};{:.4},{:.4}]  divC={:.5}  J={:.6}",
                p.x.x,
                p.x.y,
                p.v.x,
                p.v.y,
                c.x_axis.x,
                c.y_axis.x,
                c.x_axis.y,
                c.y_axis.y,
                c.x_axis.x + c.y_axis.y,
                j,
            );
        }
        if step % LOG_INTERVAL == 0 || step == N_STEPS {
            let snap = sim.diagnostics_snapshot();
            let stats = per_material_stats_of(sim.particles());
            for s in &stats {
                let ext = s.extent_max - s.extent_min;
                let implied_h = if ext.x > 1e-6 {
                    initial_area / ext.x
                } else {
                    f32::NAN
                };
                println!(
                    "frame={:5}  {}  implied_h_if_area_conserved={:.3}  non_finite={}  oob={}  y_bottom={:.4}  y_top={:.4}  sub={}",
                    step,
                    s.format(Some("water")),
                    implied_h,
                    snap.non_finite_particle_values,
                    snap.out_of_bounds_particles,
                    s.extent_min.y,
                    s.extent_max.y,
                    sim.last_substeps(),
                );
            }
        }
        if matches!(
            step,
            200 | 600 | 1000 | 1200 | 1400 | 1600 | 1800 | 2000 | 3000 | 6000
        ) {
            print_depth_stratification(step, sim.particles(), MAT_WATER);
        }
        if matches!(step, 400 | 800 | 1200 | 1600 | 2000) {
            print_settled_divergence_bias(step, sim.particles(), MAT_WATER);
        }
        if matches!(step, 20 | 40 | 60 | 80 | 100 | 150 | 200 | 300 | 400) {
            print_fast_phase_divergence(step, sim.particles(), MAT_WATER);
        }
    }
}

//! Temporary diagnostic: headless, fast-iteration verification of a real
//! whirlpool driven by the EXACT DCT pressure-projection fluid path
//! (`fluid_pressure_projection_gui.rs`'s own solver config: eos_stiffness=0,
//! incompressibility from `SimConfig::fluid_pressure_iterations`, not an
//! acoustic-CFL spring) instead of the GPU strict-fluid EOS path fought all
//! session. Real motivation: the GPU EOS approach can only approximate
//! cyclostrophic balance (dp/dr = rho*v^2/r); this solver resolves it
//! exactly, and this exact scene class (single-material water, no wall
//! contact) was just measured at a real 16fps even on this engine's hardest
//! known scene.
//!
//! STATUS (2026-08-15, unresolved -- read before trusting this scene):
//! live-measured a real, large initialization transient (max_speed
//! 100-240, well beyond anything else seen this session) on the very
//! FIRST `step()` call, REPRODUCIBLE WITH ZERO SEED VELOCITY (a plain
//! resting water block, no vortex mechanism involved at all) and
//! independent of pool height (tested both a 48-cell and a 16-cell-tall
//! pool -- both spike). Isolated by direct A/B testing, not assumed: this
//! is NOT the vortex seed, NOT a point-force/GravityWell interaction (also
//! tested removed), NOT specifically about pool aspect ratio. Real,
//! disclosed hypothesis, NOT yet confirmed: this exact solver
//! config's whole validation history (see `fluid_pressure_projection_gui.
//! rs`'s own doc) is against a water column ALREADY near/against a wall
//! (mostly Neumann boundary) -- a symmetric, wall-free resting pool (free
//! surface on every side, mostly Dirichlet per `project_fluid_
//! incompressibility`'s own free-surface handling) may be a genuinely
//! untested regime for this exact projection method. Bounded and
//! recoverable every time tested (no crash, no non-finite, settles to
//! near-stillness within ~100-150 frames) -- not a blocker for further
//! investigation, but NOT fixed, and real enough that a live GUI build of
//! this scene would likely show a violent, ugly opening flash before
//! settling. Do not claim this pattern "works" without solving this first.
//!
//! Vortex mechanism itself (once this transient is dealt with): same
//! real, proven approach as tonight's GPU work -- constant-angular-
//! momentum seed (NOT solid-body rotation, NOT a written velocity-profile
//! formula), smooth taper to zero at the seed edge (a hard cutoff was
//! tried and produced its own separate, smaller spike -- see git history).
//! NO GravityWell pull -- tried, live-rejected: pulling fluid toward a
//! point fights this solver's OWN strict incompressibility enforcement
//! directly, producing a real, violent reactive kick worse than not
//! having it at all. NO RadialConfinement -- live-measured tonight
//! (GPU side) that it forcibly reshapes ordinary resting water into a
//! circle; gravity + floor + SlipBoundary is what every other demo in
//! this engine already relies on.
//!
//! Run with --release; debug-mode timing is noise (matches this codebase's
//! own established convention for these diagnostics).
extern crate emerge_engine as emerge;

use emerge::{NewtonianFluidMaterial, SimConfig, Simulation, SlipBoundary, SpawnRegion};
use glam::{IVec2, Vec2};

const GRID: usize = 64;
const DT: f32 = 0.1;
const MAT_WATER: u32 = 0;

fn main() {
    let config = SimConfig {
        min_dt: 1.0e-4,
        max_substeps_per_step: 400,
        material_cfl_coefficient: 0.1,
        cfl_include_affine_speed: false,
        fluid_pressure_iterations: 1,
        fluid_near_wall_cfl_scale: 20.0,
        fluid_near_wall_compression_threshold: 0.0,
        ..SimConfig::earth(GRID, 0.01, DT)
    };

    // Same flat, wide, floor-resting pool shape as tonight's final GPU
    // vortex build -- live-confirmed to read as "a normal sea," not an
    // isolated blob.
    const POOL_BOX: IVec2 = IVec2::new(54, 48);
    let pool_center = Vec2::new(32.0, 26.0);
    let drain_center = pool_center;

    let water = NewtonianFluidMaterial::low_viscosity(0.1, 0.0);
    const WATER_MASS: f32 = 0.1 * 0.6 * 0.6;
    let spawn_water = SpawnRegion {
        spacing: 0.6,
        box_size: POOL_BOX,
        box_center: pool_center,
        material_id: MAT_WATER,
        initial_velocity_scale: 0.0,
        precompute_initial_volumes: true,
        mass_override: Some(WATER_MASS),
        ..SpawnRegion::for_sim(&config)
    };

    const DRAIN_EDGE_R: f32 = 17.0;
    const DRAIN_SOFTENING: f32 = 2.0;

    // NO GravityWellField this run -- live-measured (see this file's own
    // updated doc below) that pulling fluid toward a literal point fights
    // STRICT incompressibility (this solver enforces div(v)=0 almost
    // exactly, unlike the forgiving EOS path) and produced a real, violent
    // reactive kick (max_speed=163 at frame 0, one particle flung from
    // r=0.13 to r=24 in 20 frames). Testing Kelvin's circulation theorem
    // claim directly: does the seeded rotation alone sustain a real vortex
    // without any continuous inward force at all?
    let mut solver = Simulation::new(config, spawn_water)
        .with_default_material(Box::new(water))
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));

    // Same free-vortex (constant angular momentum) seed as tonight's proven
    // GPU build -- v = L/r, floored at DRAIN_SOFTENING, only within
    // SWIRL_SEED_RADIUS (the rest of the pool is the calm, undisturbed sea).
    const SWIRL_SEED_RADIUS: f32 = 22.0;
    const SEED_EDGE_SPEED: f32 = 1.0;
    let seed_l = SEED_EDGE_SPEED * DRAIN_EDGE_R;
    // Smooth taper to zero approaching SWIRL_SEED_RADIUS instead of a hard
    // cutoff -- live-measured (see this file's own doc): a sharp velocity
    // DISCONTINUITY at the seed edge (real speed just inside, exactly zero
    // just outside) is a badly-conditioned input for a pressure-projection
    // solve enforcing div(v)=0 -- the corrector reacted with a real,
    // violent, non-physical spike (max_speed up to 197 at frame 0) trying
    // to reconcile it. A real vortex's own edge is smooth too, not a wall.
    const TAPER_WIDTH: f32 = 6.0; // cells over which the seed fades to zero
    {
        let particles = solver.particles_mut();
        for i in 0..particles.len() {
            let r = particles.x[i] - drain_center;
            let dist = r.length();
            if dist < SWIRL_SEED_RADIUS {
                let d = dist.max(DRAIN_SOFTENING);
                let taper_start = SWIRL_SEED_RADIUS - TAPER_WIDTH;
                let taper = if dist > taper_start {
                    let t = (dist - taper_start) / TAPER_WIDTH;
                    1.0 - t * t * (3.0 - 2.0 * t) // smoothstep, C1-continuous fade to 0
                } else {
                    1.0
                };
                particles.v[i] = taper * (seed_l / (d * d)) * Vec2::new(-r.y, r.x);
            }
        }
    }

    let particle_count = solver.particles().len();
    println!("particle_count={particle_count}");

    // Fixed particle index, captured ONCE before stepping -- a genuine
    // Lagrangian trace (the earlier "closest particle each frame" version
    // could silently swap identity between frames if a different, unrelated
    // particle happened to pass near center, confounding the signal).
    let tracked_idx = {
        let particles = solver.particles();
        (0..particles.len())
            .map(|k| (k, (particles.x[k] - drain_center).length()))
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap()
            .0
    };

    let n = 300;
    let wall_start = std::time::Instant::now();
    let mut max_speed_ever = 0.0f32;
    let mut min_j_ever = f32::MAX;
    let mut max_j_ever = f32::MIN;
    for i in 0..n {
        solver.step();
        let snap = solver.diagnostics_snapshot();
        max_speed_ever = max_speed_ever.max(snap.max_particle_speed);
        min_j_ever = min_j_ever.min(snap.min_deformation_j);
        max_j_ever = max_j_ever.max(snap.max_deformation_j);
        if i % 20 == 0 || i == n - 1 {
            // Same fixed particle every frame now -- if it's genuinely
            // orbiting, its velocity DIRECTION should rotate over time
            // while its distance from center stays roughly bounded (not
            // monotonically escaping, not collapsing to zero).
            let particles = solver.particles();
            let idx = tracked_idx;
            let r = particles.x[idx] - drain_center;
            let v = particles.v[idx];
            println!(
                "  frame {i}: non_finite={} out_of_bounds={} max_speed={:.3} minJ={:.3} maxJ={:.3} | core_particle dist={:.2} v=({:.3},{:.3})",
                snap.non_finite_particle_values,
                snap.out_of_bounds_particles,
                snap.max_particle_speed,
                snap.min_deformation_j,
                snap.max_deformation_j,
                r.length(),
                v.x,
                v.y
            );
        }
    }
    let wall_elapsed = wall_start.elapsed();
    println!(
        "wall_time={:.2}s  ({:.2}fps over {n} frames)",
        wall_elapsed.as_secs_f64(),
        n as f64 / wall_elapsed.as_secs_f64()
    );
    println!(
        "max_speed_ever={max_speed_ever:.3} min_j_ever={min_j_ever:.3} max_j_ever={max_j_ever:.3}"
    );
}

// P2G — scatter particle mass, momentum and stress to the 3×3 grid neighbourhood.
// One thread per particle. Uses fixed-point atomicAdd (WebGPU has no atomic<f32>).

struct Particle {
    x:                    vec2<f32>,
    v:                    vec2<f32>,
    velocity_gradient:    mat2x2<f32>,
    deformation_gradient: mat2x2<f32>,
    mass:                 f32,
    initial_volume:       f32,
    volume:               f32,
    density:              f32,
    material_id:          u32,
    plastic_volume_ratio: f32,
    hardening_scale:      f32,
    friction_hardening:   f32,
    log_volume_strain:    f32,
    temperature:          f32,
    user_tag:             u32,
    activation:           f32,
    activation_dir:       vec2<f32>,
    muscle_group_id:      u32,
    contact_group:        u32,
    sleeping:             u32,
    pinned:               u32,
    scalar_field:         f32,
    _pad:                 u32,
}

struct MaterialParams {
    model:                   u32,
    lambda:                  f32,
    mu:                      f32,
    hardening_exponent:      f32, // Snow: ξ;  VonMises: yield_stress
    compression_limit:       f32, // Snow: θ_c;  DP: dilatancy ψ;  Bingham: yield_stress
    stretch_limit:           f32,
    rest_density:            f32,
    eos_stiffness:           f32,
    eos_power:               f32,
    dynamic_viscosity:       f32,
    volume_ratio_min:        f32,
    volume_ratio_max:        f32,
    dp_h0:                   f32,
    dp_h1:                   f32,
    dp_h2:                   f32,
    dp_h3:                   f32,
    active_stress_coeff:     f32,
    hardening_modulus:       f32,
    thermal_viscosity_coeff: f32,
    thermal_expansion:       f32,
    pressure_floor:          f32,
    bulk_viscosity:          f32,
    surface_tension_coeff:   f32,
    cohesion_coeff:          f32,
}

struct StepParams {
    grid_res:           u32,
    particle_count:     u32,
    dt:                 f32,
    kernel_d_inverse:   f32,
    gravity:            vec2<f32>,
    boundary_thickness: u32,
    vel_limit:          f32,
    sleep_threshold:    f32,
    _pad0:              u32,
    _pad1:              u32,
    _pad2:              u32,
}

const MAX_MATERIALS:        u32 = {{MAX_MATERIALS}}u;
const BSPLINE_INNER_LIMIT:  f32 = 0.5;
const BSPLINE_OUTER_LIMIT:  f32 = 1.5;
const BSPLINE_CENTER_COEFF: f32 = 0.75;
const BSPLINE_OUTER_SCALE:  f32 = 0.5;
const CELL_CENTER_OFFSET:   f32 = 0.5;
const NUM_FLOOR:            f32 = 1e-6;
// Fixed-point scales: mass and momentum use different scales to avoid i32 overflow.
// With 9 particles per cell: mass × 1e6 ≤ 9e6 (safe). Momentum at vel_limit=1000: 9×1000×1e5=9e8 (safe).
// MOM_ATOMIC_SCALE=1e5 gives 1e-5 precision — 100× better than 1e3, avoids overflow at min_dt=0.001.
const MASS_ATOMIC_SCALE:    f32 = 1000000.0;
const MOM_ATOMIC_SCALE:     f32 = 100000.0;
// Matches render::step_params::MAX_RENDER_MATERIAL_SLOTS exactly (Rust-side source of
// truth) -- render::OpticalTable's own real 16-slot cap, not MAX_MATERIALS' larger
// 64-material solver cap. material_id >= 16 collides into slot material_id % 16, same
// convention Renderer::set_optical_params already uses.
const MAX_RENDER_MATERIAL_SLOTS: u32 = 16u;
// Multi-field contact (GPU port, first slice) — must match
// `step_params::MAX_CONTACT_POINTS_PER_BLOCK` (Rust-side source of truth, sizes the
// `contact_points` buffer at construction) exactly, same duplicated-constant
// convention already used for MASS_ATOMIC_SCALE/MOM_ATOMIC_SCALE above. Bucketed per
// a DEDICATED finer contact-block partition, not per exact node — see that constant's
// own doc in step_params.rs for why (a first per-node version OOM'd at high grid_res;
// a 2026-07-18 re-partition then split this off from the coarser P2G-sort partition to
// fix a real scan-to-keep mismatch, see MAX_CONTACT_POINTS_PER_BLOCK's doc).
const MAX_POINTS_PER_BLOCK: u32 = 256u;
// override, not a hardcoded literal — must match resolve_contact.wgsl's
// NUM_CONTACT_BLOCKS_PER_DIM exactly, single Rust-side source of truth
// (src/gpu/step_params.rs). Needed here so gather_contact_points_main computes the SAME
// block index resolve_contact's gather_local_points reads. DEDICATED to contact-point
// bucketing — deliberately NOT the same override as particle_sort.wgsl's
// NUM_BLOCKS_PER_DIM (an unrelated partition, sort-permutation/active-block occupancy).
override NUM_CONTACT_BLOCKS_PER_DIM: u32;

@group(0) @binding(0) var<storage, read_write> particles:           array<Particle>;
@group(0) @binding(1) var<storage, read_write> grid_atomic:         array<atomic<i32>>;
@group(0) @binding(2) var<uniform>             materials:           array<MaterialParams, MAX_MATERIALS>;
@group(0) @binding(3) var<uniform>             step_params:         StepParams;
@group(0) @binding(5) var<storage, read_write> sorted_particle_ids: array<u32>;
// Multi-field contact (GPU port, first slice) — see buffers.rs doc. binding 12 is the
// SAME underlying buffer as grid_clear.wgsl's `grip_grid: array<Cell>` binding, viewed
// here as raw atomics for scatter (same dual-view convention already used for `grid`
// itself, bound as `array<Cell>` in grid_clear.wgsl and `array<atomic<i32>>` here).
@group(1) @binding(12) var<storage, read_write> grip_grid_atomic:     array<atomic<i32>>;
@group(1) @binding(13) var<storage, read_write> contact_points:       array<vec4<f32>>;
@group(1) @binding(14) var<storage, read_write> contact_point_counts: array<atomic<u32>>;

struct MaterialMassParams {
    enabled: u32,
    _pad0:   u32,
    _pad1:   u32,
    _pad2:   u32,
}
// `ColorMode::GridVolume`'s opt-in per-cell per-material mass accumulator -- see
// buffers.rs's `material_mass` doc. Shares group 1 purely for bind-group economy
// (same reason ASFLIP shares group 3), nothing to do with contact thematically.
@group(1) @binding(30) var<storage, read_write> material_mass_atomic: array<atomic<i32>>;
@group(1) @binding(31) var<uniform>              material_mass_params: MaterialMassParams;

// Exact copy of resolve_contact.wgsl's block_index_of — WGSL has no cross-file
// includes, so this is duplicated the same way MASS_ATOMIC_SCALE etc. already are
// across shader files. Must stay byte-for-byte identical: gather_contact_points_main
// needs the SAME contact-block a given cell belongs to as resolve_contact's
// gather_local_points scans by this same index.
fn contact_block_index(pos: vec2<f32>, grid_res: u32) -> u32 {
    let max_cell = grid_res - 1u;
    let cell_x = u32(clamp(pos.x, 0.0, f32(max_cell)));
    let cell_y = u32(clamp(pos.y, 0.0, f32(max_cell)));
    let block_size = (grid_res + NUM_CONTACT_BLOCKS_PER_DIM - 1u) / NUM_CONTACT_BLOCKS_PER_DIM;
    let block_x = min(cell_x / block_size, NUM_CONTACT_BLOCKS_PER_DIM - 1u);
    let block_y = min(cell_y / block_size, NUM_CONTACT_BLOCKS_PER_DIM - 1u);
    return block_y * NUM_CONTACT_BLOCKS_PER_DIM + block_x;
}

fn bspline_w(d: f32) -> f32 {
    let a = abs(d);
    if a < BSPLINE_INNER_LIMIT { return BSPLINE_CENTER_COEFF - a * a; }
    if a < BSPLINE_OUTER_LIMIT { let t = BSPLINE_OUTER_LIMIT - a; return BSPLINE_OUTER_SCALE * t * t; }
    return 0.0;
}

fn det2(m: mat2x2<f32>) -> f32 {
    return m[0][0] * m[1][1] - m[0][1] * m[1][0];
}

// 2D polar decomposition R — analytical, mirrors corotated.rs.
fn polar_r(f: mat2x2<f32>) -> mat2x2<f32> {
    let x = f[0][0] + f[1][1];
    let y = f[0][1] - f[1][0];
    let n = sqrt(x * x + y * y);
    if n < NUM_FLOOR { return mat2x2<f32>(vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0)); }
    let inv = 1.0 / n;
    return mat2x2<f32>(vec2<f32>(x, y) * inv, vec2<f32>(-y, x) * inv);
}

// Kirchhoff stress τ for all supported material models.
fn kirchhoff(p: Particle, mat: MaterialParams) -> mat2x2<f32> {
    let F = p.deformation_gradient;
    let J = max(det2(F), NUM_FLOOR);
    let h = p.hardening_scale;
    let I = mat2x2<f32>(vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0));

    var tau: mat2x2<f32>;
    switch mat.model {
        case 1u: { // Fluid — Tait EOS + Newtonian or Bingham deviatoric viscosity
            // Use J = det(F) for EOS density: ρ = ρ₀/J (sparkl canonical, no grid-lag).
            // F is reset to sqrt(J)·I in particles_update, so det(F) = J always for fluid.
            // This eliminates the one-step lag from grid-mass gather (p.density) and keeps
            // EOS consistent with the F-tracked volume ratio.
            let rho   = clamp(mat.rest_density / max(J, NUM_FLOOR), NUM_FLOOR, mat.rest_density * 2.0);
            let ratio = rho / max(mat.rest_density, NUM_FLOOR);
            let press = max(mat.eos_stiffness * (pow(ratio, mat.eos_power) - 1.0), mat.pressure_floor);
            var t     = -press * I;

            let sym  = p.velocity_gradient + transpose(p.velocity_gradient);
            let tr_s = sym[0][0] + sym[1][1];
            let dev  = sym - (tr_s * 0.5) * I;

            // Arrhenius thermal thinning: µ_eff = µ₀·exp(−k·T)
            let eff_visc = select(mat.dynamic_viscosity,
                mat.dynamic_viscosity * exp(-mat.thermal_viscosity_coeff * p.temperature),
                mat.thermal_viscosity_coeff > 0.0);

            let yield_s = mat.compression_limit; // Bingham τ₀; 0 for Newtonian
            if yield_s > 0.0 {
                // Bingham: apparent viscosity = τ₀/γ̇ + µ. Skip deviatoric below plug threshold.
                // γ̇ uses the deviatoric strain rate only — a yield criterion must not respond
                // to pure volumetric expansion/compression, which isn't shear.
                let dx = dev[0][0]; let dy = dev[1][1]; let dxy = dev[0][1];
                let shear_rate = sqrt(max(0.5 * (dx*dx + dy*dy + 2.0*dxy*dxy), 0.0));
                if shear_rate > 1e-4 {
                    let eta_app = yield_s / shear_rate + eff_visc;
                    t = t + dev * (eta_app * 0.5);
                }
            } else {
                t = t + eff_visc * dev;
            }

            // Bulk viscosity damps compression waves.
            if mat.bulk_viscosity > 0.0 {
                t = t + mat.bulk_viscosity * (tr_s * 0.5) * I;
            }

            // Surface tension: τ += γ·J·I
            if mat.surface_tension_coeff != 0.0 {
                t = t + mat.surface_tension_coeff * J * I;
            }

            return t;
        }
        case 2u: { // NeoHookean — Simo-Pister vol-dev split
            // CPU (elastic.rs) hard-zeroes stress for near-singular deformation
            // (raw det(F) <= MIN_J) instead of dividing by the clamped floor --
            // without this, GPU's shared `J = max(det2(F), NUM_FLOOR)` clamp let
            // `mu_e / J` blow up to mu_e * 1e6 at the floor instead of returning
            // zero like CPU does for the identical edge case.
            if (det2(F) <= NUM_FLOOR) {
                return mat2x2<f32>(vec2<f32>(0.0, 0.0), vec2<f32>(0.0, 0.0));
            }
            let t_scale = 1.0 + mat.thermal_expansion * p.temperature;
            // Damage softening: mu_eff = mu*exp(-rate*damage), same exponential form
            // RankineMaterial uses for tensile strength (continuum damage mechanics).
            // cohesion_coeff repurposed for damage_softening_rate (see elastic.rs
            // params() -- documented reusable padding, zero for other materials).
            let damage_scale = exp(-mat.cohesion_coeff * p.friction_hardening);
            let mu_e  = mat.mu * t_scale * damage_scale;
            let lam_e = mat.lambda * t_scale * damage_scale;
            let B     = F * transpose(F);
            let tr_B  = B[0][0] + B[1][1];
            let dev_B = B - (tr_B * 0.5) * I;
            // 2D plane-strain bulk modulus (k = lam_e + mu_e, not the 3D
            // relation this used to mirror -- see elastic.rs's Rust-side fix
            // for the full derivation; CPU and GPU must match exactly here).
            // Volumetric term: k*ln(J), NOT k/2*(J^2-1) (changed 2026-07-11,
            // mirroring elastic.rs's real fix -- the bounded (J^2-1) form has
            // only a finite compression ceiling and let a sustained driven
            // load ratchet a creature body into unrecoverable compaction; see
            // elastic.rs's kirchhoff_stress doc for the full derivation).
            let k     = lam_e + mu_e;
            // Kelvin-Voigt viscous term, same as case 9u's -- opt-in via
            // dynamic_viscosity (0.0 default, matches elastic.rs's `viscosity`
            // field added 2026-07-11; see that file's timestep_bound for the
            // matching CFL bound this term needs).
            let sym   = p.velocity_gradient + transpose(p.velocity_gradient);
            let d     = sym * 0.5;
            let tr_d  = d[0][0] + d[1][1];
            let d_dev = d - (tr_d * 0.5) * I;
            tau = (mu_e / J) * dev_B + (k * log(J)) * I + mat.dynamic_viscosity * d_dev;
        }
        case 3u, 4u, 5u, 6u, 7u, 8u: { // Corotated / Snow / DP / VonMises / Rankine / SandMuI
            let t_scale = 1.0 + mat.thermal_expansion * p.temperature;
            let R     = polar_r(F);
            let mu_e  = mat.mu * h * t_scale;
            let lam_e = mat.lambda * h * t_scale;
            tau = 2.0 * mu_e * (F - R) * transpose(F) + lam_e * (J - 1.0) * J * I;
        }
        case 11u: { // GranularFluid — Tait EOS pressure + corotated elastic deviatoric + SVD plasticity
            // EOS pressure: −k·((ρ/ρ₀)^γ − 1)·I
            let rho   = clamp(mat.rest_density / max(J, NUM_FLOOR), NUM_FLOOR, mat.rest_density * 4.0);
            let ratio = rho / max(mat.rest_density, NUM_FLOOR);
            let press = max(mat.eos_stiffness * (pow(ratio, mat.eos_power) - 1.0), mat.pressure_floor);
            // Corotated elastic deviatoric: 2µ·h·dev[(F−R)·Fᵀ]
            let h      = p.hardening_scale;
            let R      = polar_r(F);
            let mu_eff = mat.mu * h;
            let coro   = 2.0 * mu_eff * (F - R) * transpose(F);
            let tr_c   = coro[0][0] + coro[1][1];
            let dev_c  = coro - (tr_c * 0.5) * I;
            // Small elastic volumetric term from λ — prevents total collapse under EOS alone
            let lam_e  = mat.lambda * h;
            let lam_vol = lam_e * (J - 1.0) * J * I;
            tau = -press * I + dev_c + lam_vol;
        }
        case 9u: { // Viscoelastic (Kelvin-Voigt) — elastic NeoHookean + viscous dashpot
            let j_min   = max(mat.volume_ratio_min, NUM_FLOOR);
            let J_vis   = clamp(J, j_min, 1.0 / j_min);
            let B       = F * transpose(F);
            let lnJ     = log(J_vis);
            let t_scale = 1.0 + mat.thermal_expansion * p.temperature;
            let mu_e    = mat.mu * t_scale;
            let lam_e   = mat.lambda * t_scale;
            let elastic = mu_e * (B - I) + (lam_e * lnJ) * I;
            let sym     = p.velocity_gradient + transpose(p.velocity_gradient);
            let d       = sym * 0.5;
            let tr_d    = d[0][0] + d[1][1];
            let d_dev   = d - (tr_d * 0.5) * I;
            tau = elastic + mat.dynamic_viscosity * d_dev;
        }
        default: { return mat2x2<f32>(); }
    }

    // Snow cohesion: compacted snow resists re-expansion. Only fires when Jp < 1 and J > 1.
    if mat.model == 4u && mat.cohesion_coeff > 0.0 && p.plastic_volume_ratio < 1.0 && J > 1.0 {
        tau = tau + mat.cohesion_coeff * p.plastic_volume_ratio * (J - 1.0) * J * I;
    }

    // Active stress. Viscoelastic (9) uses isotropic form (matches CPU viscoelastic.rs).
    // All other elastic models use directional F·(n₀⊗n₀)·Fᵀ (follows fiber deformation).
    if mat.active_stress_coeff > 0.0 && p.activation > 0.0 {
        if mat.model == 9u {
            tau = tau + (p.activation * mat.active_stress_coeff) * I;
        } else {
            let n  = p.activation_dir;
            let ls = dot(n, n);
            if ls > NUM_FLOOR {
                let n0      = n / sqrt(ls);
                let n_outer = mat2x2<f32>(n0 * n0.x, n0 * n0.y);
                tau = tau + F * ((p.activation * mat.active_stress_coeff) * n_outer) * transpose(F);
            } else {
                tau = tau + (p.activation * mat.active_stress_coeff) * I;
            }
        }
    }
    return tau;
}

// stress_volume: fluids use initial_volume * J (= current volume, J from det(F)).
// J-based volume is consistent with the J-based EOS density above.
// Elastic models use initial (reference) volume — J accounted for in Kirchhoff stress.
fn sv(p: Particle, mat: MaterialParams) -> f32 {
    switch mat.model {
        case 1u: {
            // J = det(F); F is reset to sqrt(J)·I in particles_update.
            let J = max(det2(p.deformation_gradient), NUM_FLOOR);
            return max(p.initial_volume * J, NUM_FLOOR);
        }
        case 11u: {
            // GranularFluid: EOS is density-based — use current volume (tracks J each substep).
            return max(p.volume, NUM_FLOOR);
        }
        default: { return p.initial_volume; }
    }
}

fn atomic_addf_mass(idx: u32, val: f32) {
    atomicAdd(&grid_atomic[idx], i32(round(val * MASS_ATOMIC_SCALE)));
}
fn atomic_addf_mom(idx: u32, val: f32) {
    atomicAdd(&grid_atomic[idx], i32(round(val * MOM_ATOMIC_SCALE)));
}
fn grip_atomic_addf_mass(idx: u32, val: f32) {
    atomicAdd(&grip_grid_atomic[idx], i32(round(val * MASS_ATOMIC_SCALE)));
}
fn material_mass_atomic_addf(idx: u32, val: f32) {
    atomicAdd(&material_mass_atomic[idx], i32(round(val * MASS_ATOMIC_SCALE)));
}
fn grip_atomic_addf_mom(idx: u32, val: f32) {
    atomicAdd(&grip_grid_atomic[idx], i32(round(val * MOM_ATOMIC_SCALE)));
}

@compute @workgroup_size(64, 1, 1)
fn p2g_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= step_params.particle_count { return; }
    let p_idx = sorted_particle_ids[gid.x];

    let p   = particles[p_idx];
    let res = step_params.grid_res;
    let dt  = step_params.dt;
    let mat = materials[p.material_id];

    // Sleeping particles still scatter normally — their mass+stress is exactly what
    // provides support to anything resting on top of them. Skipping P2G for sleeping
    // particles (an earlier version of this code did) makes them invisible to the grid:
    // an awake neighbor stacked on a sleeping one would suddenly find no support beneath
    // it, generating permanent unresolvable jitter at every awake/asleep boundary — the
    // pile could never fully settle. Frozen (x, v, F) means the SAME scatter contribution
    // every substep, so this is deterministic, not wasted-but-harmless extra work: it's
    // the actual support mechanism. The real savings are in g2p/particles_update/
    // force_fields, which skip recomputing things that provably don't change for a
    // particle whose state is frozen — not in skipping the scatter itself.

    // NaN position would corrupt the i32 atomics — skip silently.
    if !(dot(p.x, p.x) >= 0.0) { return; }

    let tau   = kirchhoff(p, mat);
    let vol   = sv(p, mat);
    let scale = -vol * step_params.kernel_d_inverse * dt;

    let base = vec2<i32>(i32(p.x.x), i32(p.x.y));

    // Separable quadratic B-spline: only 3 distinct x-offsets and 3 distinct y-offsets
    // occur across the 9-cell neighborhood (di, dj each range over {-1,0,1}), so the
    // 1D weight only needs computing 3+3=6 times, not fresh for all 9 combinations
    // (18 calls) as a naive nested loop does. Matches the reference algorithm's own
    // technique -- Hu et al.'s mls-mpm88 (SIGGRAPH 2018) precomputes separable per-axis
    // weights the same way, cross-multiplying them per cell instead of recomputing the
    // 2D weight from scratch every iteration.
    var wx: array<f32, 3>;
    var wy: array<f32, 3>;
    var dx: array<f32, 3>;
    var dy: array<f32, 3>;
    for (var k: i32 = 0; k <= 2; k++) {
        let di = k - 1;
        dx[k] = f32(base.x + di) + CELL_CENTER_OFFSET - p.x.x;
        dy[k] = f32(base.y + di) + CELL_CENTER_OFFSET - p.x.y;
        wx[k] = bspline_w(dx[k]);
        wy[k] = bspline_w(dy[k]);
    }

    for (var ki: i32 = 0; ki <= 2; ki++) {
        let cx = base.x + ki - 1;
        if cx < 0 || cx >= i32(res) { continue; }
        for (var kj: i32 = 0; kj <= 2; kj++) {
            let cy = base.y + kj - 1;
            if cy < 0 || cy >= i32(res) { continue; }

            let cell_dist = vec2<f32>(dx[ki], dy[kj]);
            let w = wx[ki] * wy[kj];

            let apic_v    = p.v + p.velocity_gradient * cell_dist;
            let mass_w    = w * p.mass;
            let apic_mom  = mass_w * apic_v;
            let stress_mom = (scale * w) * (tau * cell_dist);

            let base4 = (u32(cy) * res + u32(cx)) * 4u;
            atomic_addf_mom(base4 + 0u, apic_mom.x + stress_mom.x);
            atomic_addf_mom(base4 + 1u, apic_mom.y + stress_mom.y);
            atomic_addf_mass(base4 + 2u, mass_w);

            // Multi-field contact (GPU port, first slice): additive second scatter for
            // the "grip" field (contact_group != 0), exactly mirroring the total-field
            // scatter above — same weights, same stress/APIC contributions — into the
            // separate grip_grid accumulator. No-op (branch not taken) for every
            // particle with contact_group == 0, matching CPU's zero-cost-when-unused
            // property (`scatter_particles_to_grid`'s own doc: "a no-op call for every
            // particle with contact_group == 0").
            if p.contact_group != 0u {
                grip_atomic_addf_mom(base4 + 0u, apic_mom.x + stress_mom.x);
                grip_atomic_addf_mom(base4 + 1u, apic_mom.y + stress_mom.y);
                grip_atomic_addf_mass(base4 + 2u, mass_w);
            }

            // `ColorMode::GridVolume`'s opt-in per-cell per-material mass scatter --
            // real, gated cost: skipped entirely (branch not taken) when disabled,
            // matching every other opt-in GPU subsystem's zero-cost-when-unused gate.
            if material_mass_params.enabled != 0u {
                let cell_idx = u32(cy) * res + u32(cx);
                let slot = p.material_id % MAX_RENDER_MATERIAL_SLOTS;
                material_mass_atomic_addf(cell_idx * MAX_RENDER_MATERIAL_SLOTS + slot, mass_w);
            }
        }
    }
}

// Multi-field contact (GPU port, first slice) — mirrors CPU's `gather_contact_point_cloud`
// (transfer.rs): a SECOND per-particle pass, run AFTER p2g_main has fully scattered grip
// mass (wgpu inserts the necessary barrier between separate compute dispatches
// automatically, same guarantee particle_sort's own multi-pass sequence already relies
// on). For each of a particle's 9 stencil nodes, if that node's grip mass (just written
// by p2g_main) is nonzero, atomically claims a slot in that node's point-cloud bucket and
// records this particle's (position, label). Labeling and the "only where grip already
// registered" gating exactly match CPU's `add_contact_point`/`gather_contact_point_cloud`
// semantics — see those functions' doc comments in `transfer.rs`/`grid/mod.rs` for the
// full rationale (this is what lets the LR normal fit ignore particles far from any real
// contact interface, not just cheaply skip the whole pass).
@compute @workgroup_size(64, 1, 1)
fn gather_contact_points_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= step_params.particle_count { return; }
    let p_idx = sorted_particle_ids[gid.x];
    let p = particles[p_idx];
    if !(dot(p.x, p.x) >= 0.0) { return; }

    let res = step_params.grid_res;
    let label = select(-1.0, 1.0, p.contact_group != 0u);

    // Real gate, not an optimization shortcut: only a particle whose OWN home cell
    // already has nonzero grip mass this substep is near a genuine contact interface
    // (matches CPU's `add_contact_point`, which only ever appends to an ALREADY-
    // existing `contact_cells` entry — the CPU equivalent of "grip mass already
    // registered here"). Checking one representative cell (not all 9 stencil cells)
    // is deliberate: bucketing is per dedicated contact-block now (see
    // MAX_POINTS_PER_BLOCK's doc), and resolve_contact's gather_local_points scans a
    // node's own block PLUS its neighbors, so a particle recorded once in its own
    // block is already visible to every node that could plausibly need it — recording
    // once per particle avoids redundantly writing the same particle into the same
    // block bucket up to 9 times. Note this partition is now finer than P2G's own
    // block_size (that's the whole point of the 2026-07-18 re-partition), so a
    // particle's 3×3 cell stencil CAN span multiple contact blocks — still correct:
    // gather_local_points's own 3×3 NEIGHBOR-BLOCK scan is exactly what covers that.
    let home_x = clamp(u32(p.x.x), 0u, res - 1u);
    let home_y = clamp(u32(p.x.y), 0u, res - 1u);
    let home_idx = home_y * res + home_x;
    let grip_mass_bits = atomicLoad(&grip_grid_atomic[home_idx * 4u + 2u]);
    if grip_mass_bits <= 0 { return; }

    let block = contact_block_index(p.x, res);
    // NOTE for any future reader of `contact_point_counts`: this counter keeps
    // incrementing past MAX_POINTS_PER_BLOCK even though writes beyond it are dropped
    // below (a real, honest overflow signal, not silently capped) — any consumer must
    // clamp its own iteration to `min(count, MAX_POINTS_PER_BLOCK)`, never trust the
    // raw count as the number of VALID slots in `contact_points`.
    let slot_in_block = atomicAdd(&contact_point_counts[block], 1u);
    if slot_in_block >= MAX_POINTS_PER_BLOCK { return; }
    let slot = block * MAX_POINTS_PER_BLOCK + slot_in_block;
    contact_points[slot] = vec4<f32>(p.x.x, p.x.y, label, 0.0);
}

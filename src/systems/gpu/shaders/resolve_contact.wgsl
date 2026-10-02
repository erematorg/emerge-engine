// Multi-field contact resolution (GPU port). Ports `Grid::resolve_contact`/
// `fit_contact_normal_lr` (src/spacetime/grid/mod.rs, CPU) to WGSL -- Bardenhagen 2001 +
// Nairn, Hammerquist and Smith 2020: LR normal fit, and contact only where the bodies
// approach and their particles' deformed edges touch (no Baumgarte term, issue #49).
//
// Point-cloud storage is bucketed per coarse BLOCK, not per exact grid node (per-node
// sizing scales as `grid_res² × capacity` and OOMs -- see `MAX_CONTACT_POINTS_PER_BLOCK`'s
// doc in step_params.rs). The fit here (`fit_contact_normal_lr`) gathers a node's
// candidate points from its own block PLUS its 8 neighbors (`gather_local_points`,
// mirroring the halo-expansion `particle_sort_compact_main` uses for occupancy) into a
// small fixed-size LOCAL array, filtered to actual kernel range (`|rel| < 1.5` cells, the
// 3x3 B-spline stencil reach P2G uses) -- only then does the Newton-Raphson iteration run,
// same as CPU's per-node exact list, just gathered differently underneath.
//
// `debug_fit_normal_main` runs the same fit against one whole block's raw points with no
// distance filtering, verified against CPU's own reference case
// (`gpu_debug_fit_normal_matches_cpu_clean_horizontal_interface`).

struct Cell {
    momentum: vec2<f32>,
    mass:     f32,
    _pad:     f32,
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
    contact_friction:   f32, // repurposes the first of GpuStepParams' 3 pad slots
    grid_cell_size:     f32, // repurposes the second pad slot -- SimConfig::grid_cell_size
    contact_active:     u32,
    cfl_coefficient:    f32,
    material_cfl_coefficient: f32, // the small-nodal-mass bound's fraction, see resolve_cell
    min_dt:             f32,
    dt_cap:             f32,
}

// Field order matches ContactDebugParams (Rust, step_params.rs) exactly -- node_pos
// first (8-byte alignment), then the two u32s.
struct ContactDebugParams {
    node_pos:     vec2<f32>,
    target_block: u32,
    point_count:  u32,
}

// Directional (setae-style) grip friction -- GPU mirror of `DirectionalContactGrip`
// (src/spacetime/grid/mod.rs). `mu_easy == mu_resist` (the default, uploaded whenever
// no directional bias is active) reduces this EXACTLY to plain symmetric Coulomb
// friction at `contact_friction` -- see `resolve_direction_aware` below for why one
// code path covers both cases instead of maintaining two.
struct DirectionalGripParams {
    easy_direction: vec2<f32>,
    mu_easy:        f32,
    mu_resist:      f32,
}

const MAX_POINTS_PER_BLOCK: u32 = 256u;
const MAX_LOCAL_POINTS:     u32 = 128u;
// Dedicated finer contact-point partition (see MAX_CONTACT_POINTS_PER_BLOCK's doc in
// step_params.rs) -- deliberately NOT the same override as this file's OWN
// NUM_BLOCKS_PER_DIM below (that one sizes active_block_ids/active_block_count, the
// unrelated sparse-MPM-dispatch partition).
override NUM_CONTACT_BLOCKS_PER_DIM: u32;
override NUM_BLOCKS_PER_DIM: u32;
const NUM_BLOCKS: u32 = 256u;
// Set at pipeline creation to min(16, cells per block side) -- same sizing as
// grid_update.wgsl's BLOCK_THREADS_PER_DIM (see its doc); the grid-stride loops below
// cover larger blocks.
override BLOCK_THREADS_PER_DIM: u32 = 16u;
// Division guard on grip/rest node mass, the value of the CPU `resolve_contact`.
const MIN_NODE_MASS: f32 = 1.0e-6;

@group(0) @binding(1)  var<storage, read_write> grid:                    array<Cell>;
@group(0) @binding(3)  var<uniform>             step_params:             StepParams;

// This substep's timestep and velocity cap, decided on the GPU at the end of the previous
// substep -- see `adaptive_cfl.wgsl`. `substep_dt()`/`vel_limit` are the CPU's
// frame-start values and are NOT authoritative any more (the GPU may only tighten them).
@group(2) @binding(37) var<storage, read_write> adaptive_dt: array<atomic<u32>, 5>;

// Cached per invocation: this is an atomic storage load, and reading it at every use
// site cost ~40% of a substep (measured: 0.29 -> 0.41ms per substep on the dam break).
var<private> substep_dt_cache: f32 = -1.0;

fn substep_dt() -> f32 {
    if substep_dt_cache < 0.0 {
        substep_dt_cache = bitcast<f32>(atomicLoad(&adaptive_dt[0]));
    }
    return substep_dt_cache;
}

// Raw per-block particle histogram for THIS substep -- see grid_update.wgsl's binding.
@group(0) @binding(6)  var<storage, read_write> block_counts:            array<atomic<u32>, NUM_BLOCKS>;
@group(0) @binding(8)  var<storage, read_write> active_block_ids:        array<u32, NUM_BLOCKS>;
@group(0) @binding(9)  var<storage, read_write> active_block_count:      atomic<u32>;
@group(0) @binding(10) var<storage, read_write> active_block_ids_prev:   array<u32, NUM_BLOCKS>;
@group(0) @binding(11) var<storage, read_write> active_block_count_prev: u32;
@group(1) @binding(12) var<storage, read_write> grip_grid:               array<Cell>;
@group(1) @binding(13) var<storage, read_write> contact_points:          array<vec4<f32>>;
@group(1) @binding(14) var<storage, read_write> contact_point_counts:    array<atomic<u32>>;
@group(1) @binding(15) var<uniform>             contact_debug_params:    ContactDebugParams;
@group(1) @binding(16) var<storage, read_write> contact_debug_output:   array<f32>;
@group(1) @binding(17) var<storage, read_write> resolved_grip_v:         array<vec2<f32>>;
@group(1) @binding(18) var<storage, read_write> resolved_rest_v:         array<vec2<f32>>;
@group(1) @binding(19) var<uniform>             grip_params:             DirectionalGripParams;

// Exact port of solve3x3 (src/spacetime/grid/mod.rs) -- Cramer's rule for a general 3x3
// linear system. `ok` is written false (leaving `out` untouched) when the system is
// singular (|det| <= epsilon), matching the Rust version's `Option::None` return.
//
// NOTE for the next reader: `mm[row][col] = rhs[row]` below looks transposed against
// CPU's `mm[row][col] = rhs[row]` written the other way in row-major terms -- it isn't
// a bug. `m` here is always the symmetric normal-equations matrix built in
// `fit_normal_from_local_points` (`sigma_sq * xp[i] * xp[j]`, symmetric by
// construction), and `det(M with column i replaced) == det(M with row i replaced)`
// whenever `M = Mᵀ` (a transpose leaves a symmetric matrix's determinant, and any
// single-row/column substitution's determinant, unchanged). Verified against CPU's
// own output on the same inputs (`gpu_debug_fit_normal_matches_cpu_...` test) rather
// than assumed from the algebra alone.
fn solve3x3(m: mat3x3<f32>, rhs: vec3<f32>, out: ptr<function, vec3<f32>>) -> bool {
    let det = determinant(m);
    // Relative to the diagonal product, as CPU's `solve3x3`: an absolute bound
    // stopped the fit before convergence on separable clouds (issue #49). The
    // second test is Rust's `!det.is_normal()`: zero, subnormal or not finite.
    let diagonal = abs(m[0][0] * m[1][1] * m[2][2]);
    if abs(det) <= 1.1920929e-7 * diagonal // f32::EPSILON
        || !(abs(det) >= 1.17549435e-38 && abs(det) <= 3.40282347e38) {
        return false;
    }
    var result: vec3<f32>;
    for (var col: u32 = 0u; col < 3u; col++) {
        var mm = m;
        mm[0][col] = rhs[0];
        mm[1][col] = rhs[1];
        mm[2][col] = rhs[2];
        result[col] = determinant(mm) / det;
    }
    *out = result;
    return true;
}

// Exact port of fit_contact_normal_lr's core Newton-Raphson NLLS iteration -- same
// numerics, same 15-iteration cap, same penalty, same z-clamp, same sign-consistency
// check against the actual labels. Operates on a caller-supplied LOCAL point list
// (`points[0..count)`), decoupling this core math from where the points came from
// (a single block's raw list for the debug entry point, or a distance-filtered
// multi-block gather for the resolve_contact pass below).
fn fit_normal_from_local_points(points: ptr<function, array<vec4<f32>, 128>>, count: u32, node_pos: vec2<f32>, grid_cell_size: f32) -> vec3<f32> {
    var has_grip = false;
    var has_rest = false;
    for (var i: u32 = 0u; i < count; i++) {
        let c = (*points)[i].z;
        if c > 0.0 { has_grip = true; }
        if c < 0.0 { has_rest = true; }
    }
    if !has_grip || !has_rest {
        return vec3<f32>(0.0, 0.0, 0.0);
    }

    let dx2 = grid_cell_size * grid_cell_size;
    let penalty = vec3<f32>(1.0e-7 * dx2, 1.0e-7 * dx2, 0.0);

    var beta = vec3<f32>(0.0, 0.0, 0.0);
    var prev_n = vec2<f32>(0.0, 0.0);
    var have_prev_n = false;

    for (var iter: u32 = 0u; iter < 15u; iter++) {
        var m = mat3x3<f32>(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0));
        var rhs = vec3<f32>(0.0, 0.0, 0.0);
        for (var i: u32 = 0u; i < count; i++) {
            let pt = (*points)[i];
            let rel = pt.xy - node_pos;
            let c = pt.z;
            let xp = vec3<f32>(rel.x, rel.y, 1.0);

            let z = clamp(dot(xp, beta), -40.0, 40.0);
            let ez = exp(-z);
            let denom = 1.0 + ez;
            let f = 2.0 / denom - 1.0;
            let sigma = 2.0 * ez / (denom * denom);
            let sigma_sq = sigma * sigma;
            m[0] += (sigma_sq * xp[0]) * xp;
            m[1] += (sigma_sq * xp[1]) * xp;
            m[2] += (sigma_sq * xp[2]) * xp;
            rhs += (sigma * (c - f)) * xp;
        }
        m[0][0] += penalty.x;
        m[1][1] += penalty.y;
        m[2][2] += penalty.z;
        rhs -= penalty * beta;

        var delta: vec3<f32>;
        if !solve3x3(m, rhs, &delta) {
            break;
        }
        if !all(delta == delta) { // NaN check: WGSL has no isnan(), NaN != NaN is portable
            break;
        }
        beta += delta;

        let normal_raw = beta.xy;
        let len_sq = dot(normal_raw, normal_raw);
        if len_sq <= 1.1920929e-7 || !all(normal_raw == normal_raw) {
            continue;
        }
        let n = normal_raw / sqrt(len_sq);
        if have_prev_n && (1.0 - dot(n, prev_n) < 1.0e-5) {
            prev_n = n;
            have_prev_n = true;
            break;
        }
        prev_n = n;
        have_prev_n = true;
    }

    if !have_prev_n {
        return vec3<f32>(0.0, 0.0, 0.0);
    }

    // Sign-consistency check against the actual labels -- see fit_contact_normal_lr's
    // own Rust doc for the full rationale (Newton can converge to a backwards normal).
    var grip_sum = 0.0;
    var grip_n = 0.0;
    var rest_sum = 0.0;
    var rest_n = 0.0;
    for (var i: u32 = 0u; i < count; i++) {
        let pt = (*points)[i];
        let proj = dot(pt.xy - node_pos, prev_n);
        if pt.z > 0.0 {
            grip_sum += proj;
            grip_n += 1.0;
        } else if pt.z < 0.0 {
            rest_sum += proj;
            rest_n += 1.0;
        }
    }
    let grip_mean = grip_sum / max(grip_n, 1.0);
    let rest_mean = rest_sum / max(rest_n, 1.0);
    var n_final = prev_n;
    if grip_mean < rest_mean {
        n_final = -prev_n;
    }
    return vec3<f32>(n_final.x, n_final.y, 1.0);
}

// `grip_grid` mass at (cx, cy), 0.0 if out of bounds -- mirrors CPU's `grip_mass_at`
// (src/spacetime/grid/mod.rs), which returns 0.0 for any OOB/untouched cell via its
// `flat_index` Option. `grip_grid` is already a dense grid_res^2 array on GPU (unlike
// CPU's sparse HashMap), so this is a direct bounds-checked index, no lookup needed.
fn grip_mass_at(cx: i32, cy: i32, res: u32) -> f32 {
    if cx < 0 || cy < 0 || cx >= i32(res) || cy >= i32(res) {
        return 0.0;
    }
    return grip_grid[u32(cy) * res + u32(cx)].mass;
}

// Fallback contact normal: Sobel-3x3 gradient of the grip field's own grid mass -- exact
// port of `grip_mass_gradient_normal` (CPU, src/spacetime/grid/mod.rs). Needed because a
// falling body's first touch has a shallow, one-sided point cloud where the LR fit often
// has no answer yet -- without this fallback, `resolve_cell` would skip correction
// outright and let the body free-fall straight through before tunneling deep and only
// then decelerating. Returns z<=0.0 (matching `fit_normal_from_local_points`'s own "no
// confident normal" convention) when there's no local gradient either.
fn grip_mass_gradient_normal(cx: u32, cy: u32, res: u32) -> vec3<f32> {
    let x = i32(cx);
    let y = i32(cy);
    let grad_x = (grip_mass_at(x + 1, y - 1, res) + 2.0 * grip_mass_at(x + 1, y, res) + grip_mass_at(x + 1, y + 1, res))
               - (grip_mass_at(x - 1, y - 1, res) + 2.0 * grip_mass_at(x - 1, y, res) + grip_mass_at(x - 1, y + 1, res));
    let grad_y = (grip_mass_at(x - 1, y + 1, res) + 2.0 * grip_mass_at(x, y + 1, res) + grip_mass_at(x + 1, y + 1, res))
               - (grip_mass_at(x - 1, y - 1, res) + 2.0 * grip_mass_at(x, y - 1, res) + grip_mass_at(x + 1, y - 1, res));
    let gradient = vec2<f32>(grad_x, grad_y);
    let len_sq = dot(gradient, gradient);
    if len_sq <= 1.1920929e-7 { // f32::EPSILON
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    let n = gradient / sqrt(len_sq);
    return vec3<f32>(n.x, n.y, 1.0);
}

// Contact-point bucket geometry -- uses the DEDICATED NUM_CONTACT_BLOCKS_PER_DIM
// partition, not this file's own NUM_BLOCKS_PER_DIM (that one is the sparse-MPM
// active-block partition resolve_contact_main iterates cells within, an unrelated
// purpose). Must stay byte-for-byte identical to p2g.wgsl's contact_block_index.
fn block_index_of(cell_x: u32, cell_y: u32, res: u32) -> vec2<u32> {
    let block_size = (res + NUM_CONTACT_BLOCKS_PER_DIM - 1u) / NUM_CONTACT_BLOCKS_PER_DIM;
    let bx = min(cell_x / block_size, NUM_CONTACT_BLOCKS_PER_DIM - 1u);
    let by = min(cell_y / block_size, NUM_CONTACT_BLOCKS_PER_DIM - 1u);
    return vec2<u32>(bx, by);
}

// Gathers this node's contact point cloud: scans its own block plus its 8 neighbors
// (same halo reasoning as particle_sort_compact_main: a point near a block boundary can
// belong to an adjacent block from this node's view), keeping points within the
// particle-to-node reach below. Stops at MAX_LOCAL_POINTS (128) even if more candidates
// exist, a bounded-memory tradeoff (see MAX_LOCAL_POINTS above).
fn gather_local_points(node_pos: vec2<f32>, res: u32, out_points: ptr<function, array<vec4<f32>, 128>>) -> u32 {
    let cell_x = u32(clamp(node_pos.x, 0.0, f32(res - 1u)));
    let cell_y = u32(clamp(node_pos.y, 0.0, f32(res - 1u)));
    let home_block = block_index_of(cell_x, cell_y, res);
    let bx = i32(home_block.x);
    let by = i32(home_block.y);

    var n: u32 = 0u;
    for (var dy: i32 = -1; dy <= 1; dy++) {
        let nby = by + dy;
        if nby < 0 || nby >= i32(NUM_CONTACT_BLOCKS_PER_DIM) { continue; }
        for (var dx: i32 = -1; dx <= 1; dx++) {
            let nbx = bx + dx;
            if nbx < 0 || nbx >= i32(NUM_CONTACT_BLOCKS_PER_DIM) { continue; }
            let block = u32(nby) * NUM_CONTACT_BLOCKS_PER_DIM + u32(nbx);
            let count = min(atomicLoad(&contact_point_counts[block]), MAX_POINTS_PER_BLOCK);
            let base = block * MAX_POINTS_PER_BLOCK;
            for (var i: u32 = 0u; i < count; i++) {
                if n >= MAX_LOCAL_POINTS { return n; }
                let slot = base + i;
                let head = contact_points[2u * slot];
                // The fit reads position and label; `w` carries the point's
                // slot so `resolve_cell` can read its edge data back.
                let pt = vec4<f32>(head.xyz, bitcast<f32>(slot));
                let rel = pt.xy - node_pos;
                // Same reach as CPU (`Grid::add_contact_point`, via
                // `gather_contact_point_cloud`): a particle is included at every
                // one of the 3x3 cells `base_cell + {-1,0,1}`, `base_cell =
                // floor(position)`, so its distance to an included node can
                // approach (not reach) 2.0 grid units (position at
                // `base_cell + 0.999`, node at `base_cell - 1`). A `< 1.5` bound
                // drops points in a spatially consistent pattern on a regular
                // lattice (e.g. `spacing=0.5`), tilting the fitted contact
                // normal (mean v_x went negative at friction=0 in
                // `gpu_multi_field_contact_produces_real_coulomb_slip_and_stick`).
                if abs(rel.x) < 2.0 && abs(rel.y) < 2.0 {
                    (*out_points)[n] = pt;
                    n++;
                }
            }
        }
    }
    return n;
}

// Debug-only entry point (1 thread) -- runs the fit against `gather_local_points`'s
// neighbor-expanded, distance-filtered point cloud around `contact_debug_params.
// node_pos`, i.e. the EXACT same input `resolve_cell` itself uses. `target_block`/
// `point_count` are unused (kept for struct layout compatibility only) -- a single
// un-expanded block's raw points would not represent what `resolve_cell` actually sees.
@compute @workgroup_size(1, 1, 1)
fn debug_fit_normal_main() {
    let node_pos = contact_debug_params.node_pos;
    var local_points: array<vec4<f32>, 128>;
    let n = gather_local_points(node_pos, step_params.grid_res, &local_points);
    let result = fit_normal_from_local_points(&local_points, n, node_pos, step_params.grid_cell_size);
    contact_debug_output[0] = result.x;
    contact_debug_output[1] = result.y;
    contact_debug_output[2] = result.z;
}

// Exact port of DirectionalContactGrip::resolve (src/spacetime/grid/mod.rs) --
// `mu_easy == mu_resist` (the uninvolved default) makes `mu` always that same value
// regardless of `aligned`, reducing exactly to plain symmetric Coulomb -- so this ONE
// path covers both CPU's `Some(grip) => grip.resolve(...)` and
// `None => apply_coulomb_wall(...)` branches without needing two separate code paths
// on GPU.
fn resolve_direction_aware(v_rel: vec2<f32>, n: vec2<f32>) -> vec2<f32> {
    let tangent = vec2<f32>(-n.y, n.x);
    let v_t = dot(v_rel, tangent);
    let easy_t = dot(grip_params.easy_direction, tangent);
    let aligned = v_t * easy_t >= 0.0;
    let mu = select(grip_params.mu_resist, grip_params.mu_easy, aligned);

    // apply_coulomb_wall (src/forces/boundary/mod.rs) exact port.
    let v_n_scalar = dot(v_rel, n);
    if v_n_scalar >= 0.0 {
        return v_rel;
    }
    let normal_speed = abs(v_n_scalar);
    let v_t_vec = v_rel - v_n_scalar * n;
    let v_t_len = length(v_t_vec);
    let friction_impulse = mu * normal_speed;
    if v_t_len > friction_impulse {
        return v_t_vec * ((v_t_len - friction_impulse) / v_t_len);
    }
    return vec2<f32>(0.0, 0.0);
}

// Exact port of Grid::resolve_contact (CPU): edge-touch detection, the Coulomb
// correction against the wall normal `-n`, and the small nodal mass bound.
// Dispatched the same active-block-bounded way as grid_update_main (one workgroup per
// block slot, grid-stride loop over the block's real cell range).
fn resolve_cell(cx: u32, cy: u32, res: u32) {
    let idx = cy * res + cx;
    let total = grid[idx];
    let grip = grip_grid[idx];
    let grip_mass = grip.mass;
    let rest_mass = total.mass - grip_mass;

    // Default: no real second field here -- both sides read the ordinary total
    // velocity, identical to CPU's grip_velocity_at/rest_velocity_at fallback.
    resolved_grip_v[idx] = total.momentum;
    resolved_rest_v[idx] = total.momentum;

    if grip_mass <= MIN_NODE_MASS || rest_mass <= MIN_NODE_MASS {
        return;
    }

    let v_cm = total.momentum;
    let v_grip = grip.momentum / grip_mass + step_params.gravity * substep_dt();

    let node_pos = vec2<f32>(f32(cx), f32(cy));
    var local_points: array<vec4<f32>, 128>;
    let n_local = gather_local_points(node_pos, res, &local_points);
    var fit = fit_normal_from_local_points(&local_points, n_local, node_pos, step_params.grid_cell_size);

    if fit.z <= 0.0 {
        // LR fit found no confident normal -- fall back to the Bardenhagen grid
        // mass-gradient normal rather than skipping correction outright.
        fit = grip_mass_gradient_normal(cx, cy, res);
    }

    if fit.z <= 0.0 {
        // Neither the LR fit nor the gradient fallback found a usable normal --
        // resolve nothing at this node (matches CPU's own "no confident normal"
        // branch: both fields keep their own velocities, total-momentum-consistent).
        resolved_grip_v[idx] = v_grip;
        resolved_rest_v[idx] = (v_cm * total.mass - v_grip * grip_mass) / rest_mass;
        return;
    }

    // `-` because the raw fit points toward increasing grip-label density; negating
    // matches CPU's "outward: away from grip" convention.
    let n = -fit.xy;
    let v_rel_before = v_grip - v_cm;
    var v_rel = v_rel_before;

    // Contact exists only where the bodies touch (Nairn, Hammerquist and
    // Smith 2020, eq. 15, 22-25), as CPU's `Grid::resolve_contact`: the
    // separation between their deformed edges along `n`,
    // `min_rest(X.n - R_p) - max_grip(X.n + R_p)`, is negative. `R_p` is the
    // point's undeformed half size over `|F_p^-1 n|` (eq. 25), both written by
    // `gather_contact_points_main`, read back through the slot in `w`.
    var max_grip_edge = -3.4e38;
    var min_rest_edge = 3.4e38;
    for (var i: u32 = 0u; i < n_local; i++) {
        let pt = local_points[i];
        let slot = bitcast<u32>(pt.w);
        let half_size = contact_points[2u * slot].w;
        let inverse_f4 = contact_points[2u * slot + 1u];
        let inverse_f = mat2x2<f32>(inverse_f4.xy, inverse_f4.zw);
        let reach = half_size / max(length(inverse_f * n), 1.17549435e-38);
        let proj = dot(pt.xy, n);
        if pt.z > 0.0 {
            max_grip_edge = max(max_grip_edge, proj + reach);
        } else if pt.z < 0.0 {
            min_rest_edge = min(min_rest_edge, proj - reach);
        }
    }
    let touching = min_rest_edge - max_grip_edge < 0.0;

    // Approaching (eq. 14) is the Coulomb correction's own test. It takes
    // the wall's outward normal, from the rest body into the grip body: `-n`.
    if touching {
        v_rel = resolve_direction_aware(v_rel_before, -n);
    }

    // Small nodal mass (Bardenhagen et al. 2001, eq. 17-22), exactly as CPU's
    // Grid::resolve_contact: the rest field's change is the grip field's times
    // -grip_mass / rest_mass, so both are scaled together when either would strain
    // a cell by more than material_cfl_coefficient in one substep. This replaces a
    // clamp of each field to one cell per substep, which discarded momentum.
    let grip_change = v_rel - v_rel_before;
    let rest_change = grip_change * (-grip_mass / rest_mass);
    let largest_change = max(max(abs(grip_change.x), abs(grip_change.y)),
                             max(abs(rest_change.x), abs(rest_change.y)));
    let strain_increment = largest_change * substep_dt() / step_params.grid_cell_size;
    var v_grip_new = v_cm + v_rel;
    if strain_increment > step_params.material_cfl_coefficient {
        v_grip_new = v_grip + grip_change * (step_params.material_cfl_coefficient / strain_increment);
    }
    let total_momentum = v_cm * total.mass;
    let v_rest_new = (total_momentum - v_grip_new * grip_mass) / rest_mass;

    resolved_grip_v[idx] = v_grip_new;
    resolved_rest_v[idx] = v_rest_new;
}

// Same occupancy rule as particle_sort_compact_main, which built this substep's
// active_block_ids from these exact counts.
fn in_current_active_list(block: u32) -> bool {
    let bx = i32(block % NUM_BLOCKS_PER_DIM);
    let by = i32(block / NUM_BLOCKS_PER_DIM);
    for (var dy: i32 = -1; dy <= 1; dy++) {
        let ny = by + dy;
        if ny < 0 || ny >= i32(NUM_BLOCKS_PER_DIM) { continue; }
        for (var dx: i32 = -1; dx <= 1; dx++) {
            let nx = bx + dx;
            if nx < 0 || nx >= i32(NUM_BLOCKS_PER_DIM) { continue; }
            if atomicLoad(&block_counts[u32(ny) * NUM_BLOCKS_PER_DIM + u32(nx)]) > 0u {
                return true;
            }
        }
    }
    return false;
}

@compute @workgroup_size(BLOCK_THREADS_PER_DIM, BLOCK_THREADS_PER_DIM, 1)
fn resolve_contact_main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    // The frame's time is already fully advanced -- this encoded substep is spare
    // capacity the CPU could not size exactly in advance (see adaptive_cfl.wgsl).
    if substep_dt() <= 0.0 { return; }
    var block: u32;
    if wg_id.x < NUM_BLOCKS {
        if wg_id.x >= atomicLoad(&active_block_count) { return; }
        block = active_block_ids[wg_id.x];
    } else {
        let slot = wg_id.x - NUM_BLOCKS;
        if slot >= active_block_count_prev { return; }
        block = active_block_ids_prev[slot];
        // Skip blocks also in the current list (its own workgroup handles them) --
        // O(9) occupancy check, see grid_update.wgsl's `in_current_active_list`.
        if in_current_active_list(block) { return; }
    }
    let res = step_params.grid_res;

    let block_size = (res + NUM_BLOCKS_PER_DIM - 1u) / NUM_BLOCKS_PER_DIM;
    let block_x = block % NUM_BLOCKS_PER_DIM;
    let block_y = block / NUM_BLOCKS_PER_DIM;
    let x_start = block_x * block_size;
    let y_start = block_y * block_size;
    let x_end = min(x_start + block_size, res);
    let y_end = min(y_start + block_size, res);

    var y = y_start + lid.y;
    loop {
        if y >= y_end { break; }
        var x = x_start + lid.x;
        loop {
            if x >= x_end { break; }
            resolve_cell(x, y, res);
            x += BLOCK_THREADS_PER_DIM;
        }
        y += BLOCK_THREADS_PER_DIM;
    }
}

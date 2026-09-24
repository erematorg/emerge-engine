extern crate emerge_engine as emerge;

/// PROBE, not a shipped feature -- headless, temporary, delete after use.
///
/// Real question: does `lame_from_si` (the SI-to-grid conversion for
/// elastic Lame parameters, used by nearly every SI-constructed solid
/// material via `Elastic::material()`/`scale_lame`) produce a GRID elastic
/// wave speed that matches the real, independently-known physical wave
/// speed converted to grid units the same way velocity already is
/// (c_grid = c_SI / dx_meters, since this solver keeps time in real
/// seconds and mass in real kg, only rescaling LENGTH by dx -- confirmed
/// directly from `gravity_from_si`'s own real, already-correct, dt-free
/// formula `g_grid = g_SI / dx_meters`).
///
/// `elastic_wave_dt` (utils.rs) already computes `c = sqrt((lambda+2mu)/rho)`
/// entirely in grid units for the real CFL bound -- this probe calls the
/// SAME formula on `lame_from_si`'s real output and compares against the
/// independently-expected `c_SI / dx_meters`, using real numbers, not
/// hand-algebra.
use emerge::materials::lame_from_si;

fn check(label: &str, e_pa: f32, nu: f32, rho_kg_m3: f32, dx_meters: f32, dt_seconds: f32) {
    let (lambda_grid, mu_grid) = lame_from_si(e_pa, nu, rho_kg_m3, dx_meters, dt_seconds);
    let rho_grid = rho_kg_m3 * dx_meters * dx_meters;

    let c_grid = ((lambda_grid + 2.0 * mu_grid) / rho_grid).sqrt();

    // Real, independent SI wave speed, then converted to grid units the
    // SAME way velocity already is (divide by dx_meters, no dt anywhere --
    // matches gravity_from_si's own already-correct, already-shipped formula).
    let lambda_si = e_pa * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    let mu_si = e_pa / (2.0 * (1.0 + nu));
    let c_si = ((lambda_si + 2.0 * mu_si) / rho_kg_m3).sqrt();
    let c_grid_expected = c_si / dx_meters;

    let ratio = c_grid / c_grid_expected;
    println!(
        "{label}: E={e_pa:.2e} nu={nu} rho={rho_kg_m3} dx={dx_meters} dt={dt_seconds}\n  \
         c_SI={c_si:.4} m/s -> c_grid_expected={c_grid_expected:.4}\n  \
         lame_from_si -> c_grid_actual={c_grid:.4}\n  \
         ratio actual/expected = {ratio:.6}  (1.0 = correct, anything else = real bug)"
    );
}

fn main() {
    // Real, standard rubber-like solid, one dx/dt pair.
    check("rubber, dx=0.01 dt=0.02", 1.0e6, 0.45, 1100.0, 0.01, 0.02);
    // SAME material, DIFFERENT dx/dt -- if the ratio above isn't 1.0 but
    // this ALSO differs from the first ratio, that's direct proof the
    // discrepancy is dt/dx-dependent (a real unit bug), not just a fixed
    // missing constant factor.
    check("rubber, dx=0.02 dt=0.01", 1.0e6, 0.45, 1100.0, 0.02, 0.01);
    check("rubber, dx=0.005 dt=0.05", 1.0e6, 0.45, 1100.0, 0.005, 0.05);
    // Real steel-like stiff solid, same dx/dt as the first case -- checks
    // the ratio doesn't depend on material stiffness (should be identical
    // to case 1 if the bug/non-bug is purely a dt/dx/rho scaling issue).
    check("steel, dx=0.01 dt=0.02", 200.0e9, 0.3, 7850.0, 0.01, 0.02);
}

//! Artificial viscosity: a numerical device that spreads a shock over a few
//! grid cells so the solver can step through it. It is not a property of the
//! material.
//!
//! Source: von Neumann & Richtmyer, "A method for the numerical calculation of
//! hydrodynamic shocks", J. Appl. Phys. 21 (1950) 232-237; Richtmyer, LA-671
//! (1948).

/// Von Neumann & Richtmyer 1950 (LA-671) artificial bulk viscosity, EOS-
/// agnostic core: `q = rho*(c0*h^2*(div v)^2 - c1*h*c_sound*div v)`, gated
/// to compression (`div v < 0`), where shocks form. `c0 = (gamma+1)/4` is
/// the Kurapatenko 1967 weak-shock coefficient; `c1 = 1.0` (Landshoff) is
/// the standard linear term. Each EOS supplies its own `c_sound` (its
/// `dp/drho` at the current state) and `gamma` (an ideal gas's adiabatic
/// index, or a Tait liquid's `eos_power` standing in, see
/// `fluid::artificial_bulk_viscosity`); this function owns only the shared
/// form, so the ideal gas reuses it without fake Tait parameters.
#[inline]
pub(crate) fn von_neumann_richtmyer_q(
    rest_density: f32,
    j: f32,
    div_v: f32,
    grid_cell_size: f32,
    c_sound: f32,
    weak_shock_gamma: f32,
) -> f32 {
    if div_v.is_nan() || div_v >= 0.0 {
        return 0.0;
    }
    let c0_quadratic = (weak_shock_gamma + 1.0) * 0.25;
    const C1_LINEAR: f32 = 1.0;
    let rho = rest_density / j;
    let h = grid_cell_size;
    let quadratic = c0_quadratic * h * h * div_v * div_v;
    let linear = C1_LINEAR * h * c_sound * div_v;
    let q = rho * (quadratic - linear);
    if q.is_finite() { q } else { 0.0 }
}

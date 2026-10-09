//! Three-branch barotropic equation of state for a cavitating weakly-
//! compressible liquid -- Lyu, Sun, Colagrossi & Zhang 2023, "A
//! consistent...cavitation model" (WCSPH; the same EOS class applies to
//! WCMPM, both explicit, particle-based, density-from-deformation methods).
//!
//! # Laying a scene down on this EOS: both halves, or neither
//!
//! Every material in this family reads its density from `det(F)` and
//! nothing else, while the grid reads it from how far apart the
//! particles sit. A scene that wants a body at rest at a density other
//! than the liquid reference has to set BOTH: the lattice spacing, so
//! the grid sees that density, and `SpawnRegion::initial_deformation_
//! gradient`, so the particle's own bookkeeping agrees. Either one alone
//! is a pressure shock, not a scene: measured on the boiling mixture at
//! a vapour quality of 0.19 (`examples/cpu/basic_boiling.rs`), spacing
//! without the gradient throws particles at 151 m/s on the first frame,
//! against 0.01 m/s once both are set. Nothing warns about it, because
//! each half on its own is a legal state.
//!
//! # Why not a pressure floor
//!
//! A flat floor (`NewtonianFluidMaterial::pressure_floor`) conflates a free
//! surface exposed to air (`p_gauge = 0` is right there) with cavitation in
//! the bulk liquid (`p_abs <= p_sat(T)`), and has zero stiffness once it
//! engages: in `phase_states_gui` (Tait `B ~ 4.6 MPa`) cavitation starts
//! at `J ~ 1.003`, so a floor removed the restoring force over the whole
//! `J` range (1.0 to 1.8) the scene reached. The vapour and mixture
//! branches keep one.
//!
//! Pressures are gauge, relative to an ambient `p_reference_abs_pa` (the
//! convention `IdealGasMaterial` uses), since the liquid branch is zero at
//! rest density. Both phases share the paper's one compressibility
//! constant (eq. 10), `B = c_l^2 rho_l_ref / gamma_l = c_v^2 rho_v_ref /
//! gamma_v`, so the liquid exponent `gamma_l` is a required input (Cole
//! 1948 gives 7.0 for water) and the vapour stiffness `b_v_pa` follows from
//! it.
//!
//! # The three branches (Lyu et al. eq. 4/9, gauge-pressure form)
//! ```text
//! p_gauge(rho) =
//!   c_l^2 * (rho - rho_l_ref),                                  rho > rho_m+
//!   p_v + (c_min^2 * rho_m^-  / 2) * asin((2*rho - rho_m^+) / rho_m^-),  rho_m- <= rho <= rho_m+
//!   B_v * ((rho / rho_v_ref)^gamma_v - 1),                       rho < rho_m-
//! ```
//! with `rho_m^+ = rho_m+ + rho_m-`, `rho_m^- = rho_m+ - rho_m-` (Lyu et
//! al.'s notation for the sum/difference of the two branch-boundary
//! densities), and `p_v` the saturation pressure in gauge terms
//! (`p_sat_abs(T) - p_reference_abs`); the paper's own simulations use
//! `p_infinity = 0`, the same gauge convention.
//!
//! # Parameters
//! `c_min` is the mixture's effective sound speed, not a fixed water
//! property: it depends on vapour fraction, phase-change lag, dissolved
//! non-condensable gas and timescale (Wood's relation gives the frozen-
//! mixture value; a mixture in phase equilibrium can be far softer). Lyu
//! et al.'s 0.1 m/s is their own model choice, so it is a required
//! constructor input with no default.
//!
//! `rho_m_plus_kg_m3`/`rho_m_minus_kg_m3` (the mixture band's two edges)
//! are not free inputs: they are coupled through `Delta_rho = rho_m+ -
//! rho_m-` and solved together by bisection on `Delta_rho`, from the
//! paper's relations:
//! ```text
//! p_v+ = p_v_gauge + pi*c_min^2*Delta_rho/4
//! p_v- = p_v_gauge - pi*c_min^2*Delta_rho/4
//! rho_m+ = rho_l_ref + p_v+ / c_l^2
//! rho_m- = rho_v_ref * (1 + p_v- / B_v)^(1/gamma_v)
//! ```
//! for the `Delta_rho` that makes `rho_m+ - rho_m- == Delta_rho` -- see
//! `solve_mixture_band`.
//!
//! # C^1 junctions
//! The branches are each smooth, but their slopes do not match at `rho_m+`
//! and `rho_m-` (the arcsin branch's `dp/drho` diverges at both edges), so
//! the raw EOS is only C^0 exactly where a particle cavitates or
//! recondenses. Each junction is bridged by a `C1Patch`: `dp/drho` built
//! as a piecewise-linear ramp between the two endpoint slopes and
//! integrated, exactly C^1 and bounded by the endpoint slopes. A cubic
//! Hermite does not work here: for these slope ratios its derivative
//! overshoots by an amount that does not shrink with width.
//! `build_junction_patch` sizes each patch's two half-widths.

use std::f32::consts::PI;

use crate::energy::thermodynamics::water_saturation::{
    WATER_SATURATION_MAX_VALID_K, water_saturation_pressure_pa,
};
use crate::matter::materials::gas::STANDARD_ATMOSPHERE_PA;

/// Saturation pressure in gauge terms at `temperature_k`, via the IAPWS-IF97
/// saturation curve, in `f64` so the dense verification sweep of the
/// temperature-dependent closure (`t_liquid_closure_max`) checks the `f32`
/// path against a precise reference.
fn p_v_gauge_pa_at_temperature_f64(temperature_k: f32) -> f64 {
    let p_v_abs_pa = water_saturation_pressure_pa(temperature_k) as f64;
    p_v_abs_pa - STANDARD_ATMOSPHERE_PA as f64
}

/// Bounded-derivative C^1 patch. A cubic Hermite cannot bridge an arbitrary
/// `(p0, m0)` to `(p1, m1)` while keeping its derivative inside
/// `[min(m0, m1), max(m0, m1)]`: at `m0 = 2048`, `m1 = 32400` it overshoots by
/// about 31 percent, and in the wide-patch limit that overshoot no longer
/// depends on width.
///
/// Instead `dp/drho` is built directly as a piecewise-linear ramp between
/// `m0` and `m1` (flat at one end, a linear transition of length `L` to the
/// other), sized so its integral reproduces `p1 - p0` exactly; `p(rho)` is
/// that integral. It stays inside `[min(m0, m1), max(m0, m1)]` everywhere
/// and is exactly C^1, though not C^2 at the one internal kink: the solver
/// only consumes `p` and `dp/drho`.
///
/// The patch spans `rho_left..rho_left+width`; `(p0, m0)` are the value and
/// slope at `rho_left`, `(p1, m1)` at `rho_left+width`.
#[derive(Debug, Clone, Copy)]
struct C1Patch {
    rho_left: f32,
    width: f32,
    p0: f32,
    m0: f32,
    p1: f32,
    m1: f32,
}

impl C1Patch {
    fn contains(&self, rho: f32) -> bool {
        rho >= self.rho_left && rho <= self.rho_left + self.width
    }

    /// Ramp length `L` and which end it sits against: `true` = the ramp
    /// occupies `[0, L]` (flat at `m1` after), `false` = `[width - L, width]`
    /// (flat at `m0` before). From matching the ramp's integral to the
    /// secant slope `d = (p1 - p0) / width`, both cases checked against the
    /// integral condition.
    fn ramp(&self) -> (f32, bool) {
        let w = self.width;
        let slope_range = self.m1 - self.m0;
        if slope_range.abs() < f32::EPSILON {
            return (0.0, true); // m0==m1: no real ramp needed, constant slope
        }
        let d = (self.p1 - self.p0) / w;
        let l_start = 2.0 * w * (self.m1 - d) / slope_range;
        if l_start >= 0.0 && l_start <= w {
            (l_start, true)
        } else {
            let l_end = 2.0 * w * (d - self.m0) / slope_range;
            (l_end.clamp(0.0, w), false)
        }
    }

    /// Value `p(rho)`: the exact integral of the ramp derivative below,
    /// `p0` at `rho_left`.
    fn value(&self, rho: f32) -> f32 {
        let x = (rho - self.rho_left).clamp(0.0, self.width);
        if (self.m1 - self.m0).abs() < f32::EPSILON {
            return self.p0 + self.m0 * x;
        }
        let (l, ramp_near_start) = self.ramp();
        let l_safe = l.max(f32::EPSILON);
        if ramp_near_start {
            if x <= l {
                self.p0 + self.m0 * x + (self.m1 - self.m0) * x * x / (2.0 * l_safe)
            } else {
                self.p0 + self.m0 * l + (self.m1 - self.m0) * l * 0.5 + self.m1 * (x - l)
            }
        } else {
            let flat_len = self.width - l;
            if x <= flat_len {
                self.p0 + self.m0 * x
            } else {
                let xr = x - flat_len;
                self.p0 + self.m0 * x + (self.m1 - self.m0) * xr * xr / (2.0 * l_safe)
            }
        }
    }

    /// Derivative `dp/drho`: the piecewise-linear ramp itself, exact (not a
    /// finite difference of `value`).
    fn derivative(&self, rho: f32) -> f32 {
        let x = (rho - self.rho_left).clamp(0.0, self.width);
        if (self.m1 - self.m0).abs() < f32::EPSILON {
            return self.m0;
        }
        let (l, ramp_near_start) = self.ramp();
        let l_safe = l.max(f32::EPSILON);
        if ramp_near_start {
            if x <= l {
                self.m0 + (self.m1 - self.m0) * x / l_safe
            } else {
                self.m1
            }
        } else {
            let flat_len = self.width - l;
            if x <= flat_len {
                self.m0
            } else {
                let xr = x - flat_len;
                self.m0 + (self.m1 - self.m0) * xr / l_safe
            }
        }
    }

    /// Min/max of `dp/drho` over the patch: always exactly
    /// `[min(m0, m1), max(m0, m1)]` by construction.
    fn derivative_extrema(&self) -> (f32, f32) {
        (self.m0.min(self.m1), self.m0.max(self.m1))
    }
}

/// Exact ULP (unit in the last place) at `x`: the gap to the next
/// representable `f32` above it. The patch-width search asks an
/// f32-representability question, so it uses the exact spacing, not
/// `x * f32::EPSILON`.
fn ulp_at(x: f32) -> f32 {
    let bits = x.to_bits();
    f32::from_bits(bits + 1) - x
}

/// Mixture-branch pressure (Lyu et al.'s arcsin closure, unregularized),
/// shared by `CavitatingEosParams::pressure_gauge_pa` and the patch search
/// (which runs before `Self` exists) so there is one copy of the formula.
fn mixture_pressure_gauge_raw(
    rho_kg_m3: f32,
    rho_m_plus_kg_m3: f32,
    rho_m_minus_kg_m3: f32,
    c_min_m_s: f32,
    p_v_gauge_pa: f32,
) -> f32 {
    let rho_plus = rho_m_plus_kg_m3 + rho_m_minus_kg_m3;
    let rho_minus = rho_m_plus_kg_m3 - rho_m_minus_kg_m3;
    let arg = ((2.0 * rho_kg_m3 - rho_plus) / rho_minus).clamp(-1.0, 1.0);
    p_v_gauge_pa + (c_min_m_s * c_min_m_s * rho_minus * 0.5) * arg.asin()
}

/// Mixture-branch `dp/drho`, unregularized: it diverges as `rho` approaches
/// either edge, a property of the arcsin closure. Shared like
/// `mixture_pressure_gauge_raw`.
fn mixture_derivative_raw(
    rho_kg_m3: f32,
    rho_m_plus_kg_m3: f32,
    rho_m_minus_kg_m3: f32,
    c_min_m_s: f32,
) -> f32 {
    let rho_plus = rho_m_plus_kg_m3 + rho_m_minus_kg_m3;
    let rho_minus = rho_m_plus_kg_m3 - rho_m_minus_kg_m3;
    let arg = (2.0 * rho_kg_m3 - rho_plus) / rho_minus;
    let one_minus_arg2 = (1.0 - arg * arg).max(1.0e-30);
    c_min_m_s * c_min_m_s / one_minus_arg2.sqrt()
}

/// Analytic cutoff: the mixture branch's slope is `c_min^2 / sqrt(1 - x^2)`,
/// with `x` its normalized position (`x = 1` at `rho_m+`, `x = -1` at
/// `rho_m-`). Setting it equal to a bound `s_max` gives the point where the
/// slope first exceeds `s_max` moving away from mid-band, `x_cut =
/// sqrt(1 - (c_min^2 / s_max)^2)`; since `d(rho)/dx = rho_minus / 2`, that is
/// the minimal patch half-width on the mixture side. Requires `s_max >
/// c_min`, else no cutoff exists and the whole branch would need patching.
///
/// Computed in `f64`: for real parameters `(c_min^2 / s_max)^2` falls below
/// f32 resolution next to 1.0, so in `f32` `1 - ratio^2` rounds to 1 and the
/// cutoff comes out exactly 0.
fn mixture_slope_cutoff_delta(rho_minus_width: f32, c_min_m_s: f32, s_max: f32) -> f64 {
    let ratio = (c_min_m_s as f64 * c_min_m_s as f64 / s_max as f64).clamp(0.0, 1.0);
    let x_cut = (1.0 - ratio * ratio).max(0.0).sqrt();
    rho_minus_width as f64 * (1.0 - x_cut) * 0.5
}

/// Patch-width search. `delta_mix` (the half-width into the mixture branch)
/// is the variable that controls the ramp length `L`: the mixture slope
/// `m0` shrinks as `delta_mix` grows away from the singular edge, widening
/// the gap the ramp bridges, so `L` grows roughly linearly with it. For a
/// fixed `delta_mix`, `L` does not depend on `delta_pure` at all (the
/// pure-side value is linear in `delta_pure` with a slope that cancels out
/// of `L`), so growing `delta_pure` only reaches representability through
/// rounding error: measured, it took `delta_pure ~ 16384 kg/m^3` and gave
/// `p1 ~ 5.3e8 Pa`.
///
/// So `delta_pure` is fixed at a small, always-representable step and
/// `delta_mix` is grown geometrically from its analytic start. On this
/// module's test parameters both junctions become representable at about 8
/// ULPs. `delta_mix` is capped at a small fraction of the mixture band
/// (`max_delta_mix`): a patch needing more would be replacing the mixture
/// branch, not joining it, and fails loudly.
const PATCH_MIN_RAMP_ULPS: f32 = 8.0;

/// The mixture-band edge densities and the two mixture-EOS constants
/// (`c_min_m_s`, `p_v_gauge_pa`) both junction-patch functions need.
#[derive(Debug, Clone, Copy)]
struct MixtureJunctionEdges {
    rho_m_plus_kg_m3: f32,
    rho_m_minus_kg_m3: f32,
    c_min_m_s: f32,
    p_v_gauge_pa: f32,
}

/// O(1) patch reconstruction from a known `delta_mix`: the construction
/// `build_junction_patch`'s search runs at each candidate width, shared
/// with the temperature-indexed table's runtime reconstruction so pressure,
/// `dp/drho` and CFL all use one implementation. `delta_pure` is always
/// exactly `PATCH_MIN_RAMP_ULPS` ULPs of `rho_junction`, so it is recomputed
/// here, never searched or stored.
fn reconstruct_junction_patch(
    rho_junction: f32,
    mix_sign: f32,
    delta_mix: f32,
    edges: MixtureJunctionEdges,
    pure_side_sign: f32,
    endpoint_at: impl Fn(f32) -> (f32, f32),
) -> C1Patch {
    let ulp = ulp_at(rho_junction);
    let delta_pure = (PATCH_MIN_RAMP_ULPS * ulp).max(1.0e-9);
    let rho_mix_edge = rho_junction + mix_sign * delta_mix;
    let p0 = mixture_pressure_gauge_raw(
        rho_mix_edge,
        edges.rho_m_plus_kg_m3,
        edges.rho_m_minus_kg_m3,
        edges.c_min_m_s,
        edges.p_v_gauge_pa,
    );
    let m0 = mixture_derivative_raw(
        rho_mix_edge,
        edges.rho_m_plus_kg_m3,
        edges.rho_m_minus_kg_m3,
        edges.c_min_m_s,
    );
    let (p_pure, m_pure) = endpoint_at(delta_pure);
    let rho_pure_edge = rho_junction + pure_side_sign * delta_pure;
    // `pure_side_sign>0`: the pure branch sits ABOVE `rho_junction` (the
    // liquid junction) -- `rho_mix_edge` is this patch's own LEFT side.
    // `pure_side_sign<0`: the pure branch sits BELOW (the vapor junction)
    // -- `rho_mix_edge` is this patch's own RIGHT side.
    if pure_side_sign > 0.0 {
        C1Patch {
            rho_left: rho_mix_edge,
            width: rho_pure_edge - rho_mix_edge,
            p0,
            m0,
            p1: p_pure,
            m1: m_pure,
        }
    } else {
        C1Patch {
            rho_left: rho_pure_edge,
            width: rho_mix_edge - rho_pure_edge,
            p0: p_pure,
            m0: m_pure,
            p1: p0,
            m1: m0,
        }
    }
}

fn build_junction_patch(
    rho_junction: f32,
    mix_sign: f32,
    analytic_delta_mix: f64,
    edges: MixtureJunctionEdges,
    pure_side_sign: f32,
    endpoint_at: impl Fn(f32) -> (f32, f32),
) -> C1Patch {
    let ulp = ulp_at(rho_junction);
    let analytic_delta_mix_f32 = analytic_delta_mix as f32;
    let mut delta_mix = if analytic_delta_mix_f32 >= ulp
        && rho_junction - mix_sign.signum() * analytic_delta_mix_f32 != rho_junction
    {
        analytic_delta_mix_f32
    } else {
        ulp
    };

    const MAX_MIX_GROWTHS: u32 = 48;
    let rho_minus_width = edges.rho_m_plus_kg_m3 - edges.rho_m_minus_kg_m3;
    let max_delta_mix = 0.1 * rho_minus_width;

    for _ in 0..MAX_MIX_GROWTHS {
        if delta_mix > max_delta_mix {
            break;
        }
        let patch = reconstruct_junction_patch(
            rho_junction,
            mix_sign,
            delta_mix,
            edges,
            pure_side_sign,
            &endpoint_at,
        );
        let (l, _) = patch.ramp();
        if l >= PATCH_MIN_RAMP_ULPS * ulp {
            return patch;
        }
        delta_mix *= 2.0;
    }
    panic!(
        "CavitatingEosParams: no real, representable C1 patch found for the junction at \
         rho={rho_junction} within a real, disclosed mixture-side growth bound \
         (max_delta_mix={max_delta_mix}, 10% of the mixture band's own width) -- real \
         numerical issue, not a parameter problem: this junction cannot be bridged by a \
         patch that stays local to it"
    );
}

/// Vapour-branch pressure (unregularized), shared with the patch search
/// like `mixture_pressure_gauge_raw`.
fn vapor_pressure_gauge_raw(rho_kg_m3: f32, b_pa: f32, gamma_v: f32, rho_v_ref_kg_m3: f32) -> f32 {
    b_pa * ((rho_kg_m3 / rho_v_ref_kg_m3).powf(gamma_v) - 1.0)
}

/// Vapour-branch `dp/drho`.
fn vapor_dp_drho_raw(rho_kg_m3: f32, b_pa: f32, gamma_v: f32, rho_v_ref_kg_m3: f32) -> f32 {
    b_pa * gamma_v / rho_v_ref_kg_m3 * (rho_kg_m3 / rho_v_ref_kg_m3).powf(gamma_v - 1.0)
}

/// Parameters of the three-branch cavitating EOS, gauge-pressure
/// convention. See the module doc for each field's status.
#[derive(Debug, Clone, Copy)]
pub struct CavitatingEosParams {
    /// Liquid branch reference density (kg/m^3) -- real rest density.
    pub rho_l_ref_kg_m3: f32,
    /// Liquid branch sound speed (m/s).
    pub c_l_m_s: f32,
    /// Liquid branch's own Tait-like exponent, used ONLY to derive the
    /// shared stiffness `B` this branch implies (see module doc) -- real,
    /// standard citable value for water is Cole 1948's `7.0`, NOT
    /// defaulted here.
    pub gamma_l: f32,
    /// Vapor branch reference density (kg/m^3).
    pub rho_v_ref_kg_m3: f32,
    /// Vapor branch polytropic exponent (dimensionless) -- real water
    /// vapor's own ratio of specific heats, ~1.33 (triatomic molecule).
    pub gamma_v: f32,
    /// Mixture-region minimum sound speed (m/s) -- see module doc, real
    /// disclosed model choice, no default.
    pub c_min_m_s: f32,
    /// Saturation (vapour) pressure at this material's operating
    /// temperature, in gauge terms (`p_sat_abs(T) - p_reference_abs`, Pa) --
    /// see `energy::thermodynamics::water_saturation`.
    pub p_v_gauge_pa: f32,
    /// DERIVED (not a free input): shared branch stiffness `B =
    /// c_l^2*rho_l_ref/gamma_l = c_v^2*rho_v_ref/gamma_v` (eq. 10).
    b_pa: f32,
    /// DERIVED: mixture band's liquid-side edge density (kg/m^3).
    rho_m_plus_kg_m3: f32,
    /// DERIVED: mixture band's vapor-side edge density (kg/m^3).
    rho_m_minus_kg_m3: f32,
    /// Derived: C^1 patch bridging the liquid and mixture branches around
    /// `rho_m_plus_kg_m3` -- see the module doc's "C^1 junctions".
    patch_liquid_junction: C1Patch,
    /// DERIVED: real C^1 patch bridging the mixture and vapor branches
    /// around `rho_m_minus_kg_m3`.
    patch_vapor_junction: C1Patch,
}

impl CavitatingEosParams {
    /// Constructs a continuity-guaranteed, gauge-pressure parameter
    /// set. Derives the shared stiffness `B` from the liquid branch
    /// (`B=c_l^2*rho_l_ref/gamma_l`), then solves for the mixture band's
    /// two edge densities via bisection on their difference `Delta_rho`
    /// -- see `solve_mixture_band`'s doc for the real root-finding
    /// procedure and its bracket.
    pub fn new(
        rho_l_ref_kg_m3: f32,
        c_l_m_s: f32,
        gamma_l: f32,
        rho_v_ref_kg_m3: f32,
        gamma_v: f32,
        c_min_m_s: f32,
        p_v_gauge_pa: f32,
    ) -> Self {
        assert!(
            rho_l_ref_kg_m3 > 0.0 && c_l_m_s > 0.0 && gamma_l > 0.0,
            "CavitatingEosParams: rho_l_ref/c_l/gamma_l must all be finite and positive"
        );
        assert!(
            rho_v_ref_kg_m3 > 0.0 && gamma_v > 0.0,
            "CavitatingEosParams: rho_v_ref/gamma_v must be finite and positive"
        );
        assert!(
            c_min_m_s > 0.0,
            "CavitatingEosParams: c_min must be finite and positive -- see module doc, no default exists"
        );
        // Shared branch stiffness (eq. 10): B = c_l^2*rho_l_ref/gamma_l =
        // c_v^2*rho_v_ref/gamma_v -- the same c^2=B*gamma/rho0 relation
        // `NewtonianFluidMaterial::rest_acoustic_c2` already uses in reverse.
        let b_pa = c_l_m_s * c_l_m_s * rho_l_ref_kg_m3 / gamma_l;
        assert!(
            b_pa.is_finite() && b_pa > 0.0,
            "CavitatingEosParams: derived shared stiffness B={b_pa} is not a real \
             positive stiffness -- check c_l/rho_l_ref/gamma_l inputs"
        );

        let (rho_m_plus_kg_m3, rho_m_minus_kg_m3) = solve_mixture_band(
            rho_l_ref_kg_m3,
            c_l_m_s,
            rho_v_ref_kg_m3,
            gamma_v,
            b_pa,
            c_min_m_s,
            p_v_gauge_pa,
        );
        assert!(
            rho_m_plus_kg_m3 > rho_m_minus_kg_m3
                && rho_m_minus_kg_m3 > 0.0
                && rho_m_plus_kg_m3 < rho_l_ref_kg_m3,
            "CavitatingEosParams: solved mixture band [{rho_m_minus_kg_m3}, \
             {rho_m_plus_kg_m3}] is not physically sane (must have \
             0 < rho_m- < rho_m+ < rho_l_ref) -- check inputs"
        );

        // C^1 patches (see the module doc's "C^1 junctions" and `C1Patch`):
        // each patch's derivative stays inside `[min(m0, m1), max(m0, m1)]`
        // by construction; `build_junction_patch` grows the mixture-side
        // half-width until the ramp is f32-representable.
        let rho_minus_width = rho_m_plus_kg_m3 - rho_m_minus_kg_m3;
        let c_l2 = c_l_m_s * c_l_m_s;
        let edges = MixtureJunctionEdges {
            rho_m_plus_kg_m3,
            rho_m_minus_kg_m3,
            c_min_m_s,
            p_v_gauge_pa,
        };

        // Liquid junction (around rho_m_plus): pure branch is the liquid
        // EOS, above rho_m_plus, with its own constant slope c_l^2.
        let analytic_delta_mix_liquid =
            mixture_slope_cutoff_delta(rho_minus_width, c_min_m_s, c_l2);
        let patch_liquid_junction = build_junction_patch(
            rho_m_plus_kg_m3,
            -1.0,
            analytic_delta_mix_liquid,
            edges,
            1.0,
            |delta_pure| {
                let rho = rho_m_plus_kg_m3 + delta_pure;
                (c_l2 * (rho - rho_l_ref_kg_m3), c_l2)
            },
        );

        // Vapor junction (around rho_m_minus): same real construction,
        // mirrored -- the pure branch here is the vapor power law, and its
        // own real slope AT rho_m_minus (not a constant) is the bound.
        let s_max_vapor = vapor_dp_drho_raw(rho_m_minus_kg_m3, b_pa, gamma_v, rho_v_ref_kg_m3);
        let analytic_delta_mix_vapor =
            mixture_slope_cutoff_delta(rho_minus_width, c_min_m_s, s_max_vapor);
        let patch_vapor_junction = build_junction_patch(
            rho_m_minus_kg_m3,
            1.0,
            analytic_delta_mix_vapor,
            edges,
            -1.0,
            |delta_pure| {
                let rho = rho_m_minus_kg_m3 - delta_pure;
                (
                    vapor_pressure_gauge_raw(rho, b_pa, gamma_v, rho_v_ref_kg_m3),
                    vapor_dp_drho_raw(rho, b_pa, gamma_v, rho_v_ref_kg_m3),
                )
            },
        );

        let params = Self {
            rho_l_ref_kg_m3,
            c_l_m_s,
            gamma_l,
            rho_v_ref_kg_m3,
            gamma_v,
            c_min_m_s,
            p_v_gauge_pa,
            b_pa,
            rho_m_plus_kg_m3,
            rho_m_minus_kg_m3,
            patch_liquid_junction,
            patch_vapor_junction,
        };
        debug_assert!(
            params.is_continuous(1.0),
            "CavitatingEosParams: solved mixture band does not actually produce a \
             continuous p(rho) to within 1 Pa -- this is a real bug in the solve \
             above, not a parameter problem"
        );
        debug_assert!(
            (params.pressure_gauge_pa(rho_l_ref_kg_m3) - 0.0).abs() < 1.0,
            "CavitatingEosParams: pressure at rest density must be exactly zero \
             gauge -- got {} Pa",
            params.pressure_gauge_pa(rho_l_ref_kg_m3)
        );
        // Both patches' derivative bounds must stay a finite, positive
        // interval: a bad mixture-side cutoff reaching the `1e-30`
        // division floor once produced `1e15` here.
        let (lo_l, hi_l) = params.patch_liquid_junction.derivative_extrema();
        let (lo_v, hi_v) = params.patch_vapor_junction.derivative_extrema();
        debug_assert!(
            lo_l > 0.0 && hi_l.is_finite() && lo_v > 0.0 && hi_v.is_finite(),
            "CavitatingEosParams: a real C^1 patch's own derivative bound is not a \
             genuine, finite, positive interval -- liquid junction [{lo_l},{hi_l}], \
             vapor junction [{lo_v},{hi_v}] -- real construction bug, not a \
             parameter problem"
        );
        params
    }

    /// Closure at a temperature: evaluates the IAPWS-IF97 saturation
    /// pressure at `temperature_k` (in `f64`, see
    /// `p_v_gauge_pa_at_temperature_f64`) and delegates to `new`.
    pub fn at_temperature(
        rho_l_ref_kg_m3: f32,
        c_l_m_s: f32,
        gamma_l: f32,
        rho_v_ref_kg_m3: f32,
        gamma_v: f32,
        c_min_m_s: f32,
        temperature_k: f32,
    ) -> Self {
        let p_v_gauge_pa = p_v_gauge_pa_at_temperature_f64(temperature_k) as f32;
        Self::new(
            rho_l_ref_kg_m3,
            c_l_m_s,
            gamma_l,
            rho_v_ref_kg_m3,
            gamma_v,
            c_min_m_s,
            p_v_gauge_pa,
        )
    }

    /// The liquid junction patch's outer (liquid-side) edge,
    /// `rho_m_plus_kg_m3 + delta_pure_liquid`, where it hands off to the plain
    /// liquid branch.
    /// `t_liquid_closure_max` checks it stays below `rho_l_ref_kg_m3`, the
    /// condition for "rest state is pure liquid".
    pub fn liquid_patch_outer_edge_kg_m3(&self) -> f32 {
        self.patch_liquid_junction.rho_left + self.patch_liquid_junction.width
    }

    /// Gauge pressure at `rho_kg_m3` (Pa, relative to the ambient
    /// `p_v_gauge_pa` was computed against): Lyu et al. 2023's three-branch
    /// EOS, with the two C^1 patches checked first.
    pub fn pressure_gauge_pa(&self, rho_kg_m3: f32) -> f32 {
        if self.patch_liquid_junction.contains(rho_kg_m3) {
            self.patch_liquid_junction.value(rho_kg_m3)
        } else if self.patch_vapor_junction.contains(rho_kg_m3) {
            self.patch_vapor_junction.value(rho_kg_m3)
        } else if rho_kg_m3 > self.rho_m_plus_kg_m3 {
            self.c_l_m_s * self.c_l_m_s * (rho_kg_m3 - self.rho_l_ref_kg_m3)
        } else if rho_kg_m3 < self.rho_m_minus_kg_m3 {
            vapor_pressure_gauge_raw(rho_kg_m3, self.b_pa, self.gamma_v, self.rho_v_ref_kg_m3)
        } else {
            mixture_pressure_gauge_raw(
                rho_kg_m3,
                self.rho_m_plus_kg_m3,
                self.rho_m_minus_kg_m3,
                self.c_min_m_s,
                self.p_v_gauge_pa,
            )
        }
    }

    /// Analytic `dp/drho` at this density (m^2/s^2, the local sound speed
    /// squared), following `pressure_gauge_pa`'s branch logic exactly,
    /// patches included: the derivative of the function actually
    /// integrated, which the CFL also reads.
    pub fn acoustic_c2_si(&self, rho_kg_m3: f32) -> f32 {
        if self.patch_liquid_junction.contains(rho_kg_m3) {
            self.patch_liquid_junction.derivative(rho_kg_m3)
        } else if self.patch_vapor_junction.contains(rho_kg_m3) {
            self.patch_vapor_junction.derivative(rho_kg_m3)
        } else if rho_kg_m3 > self.rho_m_plus_kg_m3 {
            self.c_l_m_s * self.c_l_m_s
        } else if rho_kg_m3 < self.rho_m_minus_kg_m3 {
            vapor_dp_drho_raw(rho_kg_m3, self.b_pa, self.gamma_v, self.rho_v_ref_kg_m3)
        } else {
            mixture_derivative_raw(
                rho_kg_m3,
                self.rho_m_plus_kg_m3,
                self.rho_m_minus_kg_m3,
                self.c_min_m_s,
            )
        }
    }

    /// Checks that `pressure_gauge_pa` agrees with each pure-branch formula
    /// at that branch's boundary, within `tol_pa`. The boundaries checked are
    /// the C^1 patches' outer edges, where they match the pure branches by
    /// construction.
    pub fn is_continuous(&self, tol_pa: f32) -> bool {
        let liquid_outer = self.patch_liquid_junction.rho_left + self.patch_liquid_junction.width;
        let p_liquid_at_outer = self.c_l_m_s * self.c_l_m_s * (liquid_outer - self.rho_l_ref_kg_m3);
        let p_patch_at_liquid_outer = self.patch_liquid_junction.value(liquid_outer);
        let vapor_outer = self.patch_vapor_junction.rho_left;
        let p_vapor_at_outer =
            vapor_pressure_gauge_raw(vapor_outer, self.b_pa, self.gamma_v, self.rho_v_ref_kg_m3);
        let p_patch_at_vapor_outer = self.patch_vapor_junction.value(vapor_outer);
        (p_liquid_at_outer - p_patch_at_liquid_outer).abs() < tol_pa
            && (p_vapor_at_outer - p_patch_at_vapor_outer).abs() < tol_pa
    }
}

/// Root-finding for the mixture band's two edge densities (the coupled
/// relations are in the module doc). `Delta_rho = rho_m+ - rho_m-` is the
/// single unknown; bisection on `f(Delta_rho) = trial_gap(Delta_rho) -
/// Delta_rho`, where `trial_gap` recomputes `rho_m+ - rho_m-` from the two
/// boundary formulas.
///
/// Bracket: at `Delta_rho = 0`, `trial_gap = (rho_l_ref + p_v_gauge/c_l^2) -
/// rho_v_ref (1 + p_v_gauge/B)^(1/gamma_v)`, large and positive for any
/// liquid/vapour pair (`rho_l_ref` orders of magnitude above `rho_v_ref`),
/// so `f(0) > 0`. The upper end is the largest `Delta_rho` keeping the
/// vapour formula real (`1 + p_v-/B >= 0`); there `rho_m- -> 0` and
/// `trial_gap` falls to `rho_m+`, below `Delta_rho`, so `f(upper) < 0`. No
/// sign change means the `c_min`/`p_v_gauge` pair is not physical for
/// these references, and this panics.
fn solve_mixture_band(
    rho_l_ref_kg_m3: f32,
    c_l_m_s: f32,
    rho_v_ref_kg_m3: f32,
    gamma_v: f32,
    b_pa: f32,
    c_min_m_s: f32,
    p_v_gauge_pa: f32,
) -> (f32, f32) {
    let c_l2 = c_l_m_s * c_l_m_s;
    let c_min2 = c_min_m_s * c_min_m_s;

    let trial_gap = |delta_rho: f32| -> f32 {
        let p_v_plus = p_v_gauge_pa + PI * c_min2 * delta_rho * 0.25;
        let p_v_minus = p_v_gauge_pa - PI * c_min2 * delta_rho * 0.25;
        let rho_m_plus = rho_l_ref_kg_m3 + p_v_plus / c_l2;
        let vapor_base = 1.0 + p_v_minus / b_pa;
        let rho_m_minus = if vapor_base > 0.0 {
            rho_v_ref_kg_m3 * vapor_base.powf(1.0 / gamma_v)
        } else {
            0.0 // vapor branch has fully collapsed to zero density -- real domain edge
        };
        rho_m_plus - rho_m_minus
    };
    let f = |delta_rho: f32| trial_gap(delta_rho) - delta_rho;

    // Upper bracket: the largest Delta_rho keeping `1+p_v-/B >= 0`.
    let delta_rho_max = 4.0 * (p_v_gauge_pa + b_pa) / (PI * c_min2);
    assert!(
        delta_rho_max.is_finite() && delta_rho_max > 0.0,
        "CavitatingEosParams: no physically valid mixture-band width exists for \
         these inputs (derived delta_rho_max={delta_rho_max}) -- check c_min/p_v_gauge/B"
    );

    let mut lo = 1.0e-6_f32;
    let mut hi = delta_rho_max * 0.999999;
    let f_lo = f(lo);
    let f_hi = f(hi);
    assert!(
        f_lo > 0.0 && f_hi < 0.0,
        "CavitatingEosParams: mixture-band bisection bracket does not change sign \
         (f(lo)={f_lo}, f(hi)={f_hi}) -- this c_min/p_v_gauge/liquid-vapor reference \
         combination has no real, physical mixture-band solution; adjust c_min or \
         the reference densities, do not force a solve"
    );

    // The tolerance sits under f32 resolution at these magnitudes, so the
    // bisection runs to machine precision within `MAX_ITERATIONS` (run only
    // when parameters or the temperature table are built).
    const MAX_ITERATIONS: u32 = 200;
    const TOLERANCE: f32 = 1.0e-9;
    for _ in 0..MAX_ITERATIONS {
        let mid = 0.5 * (lo + hi);
        let f_mid = f(mid);
        if f_mid.abs() < TOLERANCE || (hi - lo) < TOLERANCE {
            let p_v_plus = p_v_gauge_pa + PI * c_min2 * mid * 0.25;
            let rho_m_plus = rho_l_ref_kg_m3 + p_v_plus / c_l2;
            let rho_m_minus = rho_m_plus - mid;
            return (rho_m_plus, rho_m_minus);
        }
        if f_mid > 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let mid = 0.5 * (lo + hi);
    let p_v_plus = p_v_gauge_pa + PI * c_min2 * mid * 0.25;
    let rho_m_plus = rho_l_ref_kg_m3 + p_v_plus / c_l2;
    let rho_m_minus = rho_m_plus - mid;
    (rho_m_plus, rho_m_minus)
}

/// Non-panicking probe: the liquid junction patch's outer edge density at
/// `p_v_gauge_pa`, or `None` if the mixture band is not physical there
/// (`rho_m_plus` already past `rho_l_ref`, or no band solution). Only
/// `t_liquid_closure_max`'s bisection uses it, where an invalid band is
/// the expected signal "this temperature is past the closure", unlike a
/// direct `CavitatingEosParams::new` call, which asserts.
fn try_liquid_patch_outer_edge_kg_m3(
    rho_l_ref_kg_m3: f32,
    c_l_m_s: f32,
    gamma_l: f32,
    rho_v_ref_kg_m3: f32,
    gamma_v: f32,
    c_min_m_s: f32,
    p_v_gauge_pa: f32,
) -> Option<f32> {
    let b_pa = c_l_m_s * c_l_m_s * rho_l_ref_kg_m3 / gamma_l;
    if !(b_pa.is_finite() && b_pa > 0.0) {
        return None;
    }
    let delta_rho_max = 4.0 * (p_v_gauge_pa + b_pa) / (PI * c_min_m_s * c_min_m_s);
    if !(delta_rho_max.is_finite() && delta_rho_max > 0.0) {
        return None;
    }
    let c_l2 = c_l_m_s * c_l_m_s;
    let c_min2 = c_min_m_s * c_min_m_s;
    let trial_gap = |delta_rho: f32| -> f32 {
        let p_v_plus = p_v_gauge_pa + PI * c_min2 * delta_rho * 0.25;
        let p_v_minus = p_v_gauge_pa - PI * c_min2 * delta_rho * 0.25;
        let rho_m_plus = rho_l_ref_kg_m3 + p_v_plus / c_l2;
        let vapor_base = 1.0 + p_v_minus / b_pa;
        let rho_m_minus = if vapor_base > 0.0 {
            rho_v_ref_kg_m3 * vapor_base.powf(1.0 / gamma_v)
        } else {
            0.0
        };
        rho_m_plus - rho_m_minus
    };
    let f = |delta_rho: f32| trial_gap(delta_rho) - delta_rho;
    let lo0 = 1.0e-6_f32;
    let hi0 = delta_rho_max * 0.999999;
    if !(f(lo0) > 0.0 && f(hi0) < 0.0) {
        return None; // no real bracket -- no physical band solution here
    }
    let (rho_m_plus_kg_m3, rho_m_minus_kg_m3) = solve_mixture_band(
        rho_l_ref_kg_m3,
        c_l_m_s,
        rho_v_ref_kg_m3,
        gamma_v,
        b_pa,
        c_min_m_s,
        p_v_gauge_pa,
    );
    if !(rho_m_plus_kg_m3 > rho_m_minus_kg_m3
        && rho_m_minus_kg_m3 > 0.0
        && rho_m_plus_kg_m3 < rho_l_ref_kg_m3)
    {
        return None; // rho_m+ has already crossed rho_l_ref (or worse)
    }
    let rho_minus_width = rho_m_plus_kg_m3 - rho_m_minus_kg_m3;
    let analytic_delta_mix_liquid = mixture_slope_cutoff_delta(rho_minus_width, c_min_m_s, c_l2);
    let patch = build_junction_patch(
        rho_m_plus_kg_m3,
        -1.0,
        analytic_delta_mix_liquid,
        MixtureJunctionEdges {
            rho_m_plus_kg_m3,
            rho_m_minus_kg_m3,
            c_min_m_s,
            p_v_gauge_pa,
        },
        1.0,
        |delta_pure| {
            let rho = rho_m_plus_kg_m3 + delta_pure;
            (c_l2 * (rho - rho_l_ref_kg_m3), c_l2)
        },
    );
    Some(patch.rho_left + patch.width)
}

/// Bisection for the highest temperature at which the liquid junction
/// patch still keeps `rho_m+(T) + delta_rho_patch_liquid(T) + density_guard
/// <= rho_l_ref`, where "rest state is pure liquid" still holds. Above it,
/// `CavitatingEosParams::at_temperature`/`new` panic: `rho_m+` drifts past
/// `rho_l_ref` as `T` nears boiling, since the band's half-width term does
/// not vanish as `p_v_gauge -> 0`. This EOS covers a liquid, possibly under
/// tension or cavitating, not boiling; the enthalpy/latent-heat swap covers
/// the rest.
///
/// `density_guard_kg_m3`: the density change of one f32 ULP of pressure at
/// the liquid branch's full-scale stiffness `rho_l_ref c_l^2`, converted
/// back through the branch's constant slope `c_l^2`.
#[derive(Debug, Clone, Copy)]
struct EosBranchInputs {
    rho_l_ref_kg_m3: f32,
    c_l_m_s: f32,
    gamma_l: f32,
    rho_v_ref_kg_m3: f32,
    gamma_v: f32,
    c_min_m_s: f32,
}

/// Checked bracket: `t_min` must be a valid temperature
/// (`try_liquid_patch_outer_edge_kg_m3` returns `Some` with margin) and
/// `t_max` must not be; panics otherwise, like `solve_mixture_band`.
fn t_liquid_closure_max(inputs: EosBranchInputs, t_min_k: f32, t_max_k: f32) -> f32 {
    let EosBranchInputs {
        rho_l_ref_kg_m3,
        c_l_m_s,
        gamma_l,
        rho_v_ref_kg_m3,
        gamma_v,
        c_min_m_s,
    } = inputs;
    let density_guard_kg_m3 = {
        let pressure_scale = rho_l_ref_kg_m3 * c_l_m_s * c_l_m_s;
        ulp_at(pressure_scale) / (c_l_m_s * c_l_m_s)
    };
    let is_valid_at = |temperature_k: f32| -> bool {
        let p_v_gauge_pa = p_v_gauge_pa_at_temperature_f64(temperature_k) as f32;
        match try_liquid_patch_outer_edge_kg_m3(
            rho_l_ref_kg_m3,
            c_l_m_s,
            gamma_l,
            rho_v_ref_kg_m3,
            gamma_v,
            c_min_m_s,
            p_v_gauge_pa,
        ) {
            Some(outer_edge) => outer_edge + density_guard_kg_m3 <= rho_l_ref_kg_m3,
            None => false,
        }
    };
    assert!(
        is_valid_at(t_min_k),
        "CavitatingEosParams: t_liquid_closure_max's own t_min_k={t_min_k}K is not \
         itself a real, valid temperature for this material's liquid patch -- check \
         the real material constants, not a search-resolution problem"
    );
    assert!(
        !is_valid_at(t_max_k),
        "CavitatingEosParams: t_liquid_closure_max's own t_max_k={t_max_k}K is STILL \
         a valid temperature -- the real closure boundary lies above the supplied \
         search range, widen t_max_k rather than trusting this bisection to \
         extrapolate"
    );
    let mut lo = t_min_k;
    let mut hi = t_max_k;
    // 0.1 mK, far finer than the closure's accuracy: about 20 halvings of a
    // 100 K bracket, well inside `MAX_ITERATIONS`.
    const MAX_ITERATIONS: u32 = 60;
    const TOLERANCE_K: f32 = 1.0e-4;
    for _ in 0..MAX_ITERATIONS {
        if (hi - lo) < TOLERANCE_K {
            break;
        }
        let mid = 0.5 * (lo + hi);
        if is_valid_at(mid) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

/// The four temperature-dependent quantities that determine the closure at
/// `T`: the band edges `rho_m_plus`/`rho_m_minus` (`solve_mixture_band`) and
/// each junction's mixture-side half-width `delta_mix_liquid`/
/// `delta_mix_vapor` (`build_junction_patch`). Not patch coefficients:
/// interpolating those would break C^1, monotonicity and junction matching
/// between table nodes. `delta_pure` is a function of `rho_junction` alone
/// (`PATCH_MIN_RAMP_ULPS` ULPs), so it is never interpolated.
struct CavitatingEosPrimitives {
    rho_m_plus_kg_m3: f32,
    rho_m_minus_kg_m3: f32,
    delta_mix_liquid_kg_m3: f32,
    delta_mix_vapor_kg_m3: f32,
}

fn primitives_at_temperature(
    rho_l_ref_kg_m3: f32,
    c_l_m_s: f32,
    gamma_l: f32,
    rho_v_ref_kg_m3: f32,
    gamma_v: f32,
    c_min_m_s: f32,
    temperature_k: f32,
) -> CavitatingEosPrimitives {
    let p_v_gauge_pa = p_v_gauge_pa_at_temperature_f64(temperature_k) as f32;
    let b_pa = c_l_m_s * c_l_m_s * rho_l_ref_kg_m3 / gamma_l;
    let (rho_m_plus_kg_m3, rho_m_minus_kg_m3) = solve_mixture_band(
        rho_l_ref_kg_m3,
        c_l_m_s,
        rho_v_ref_kg_m3,
        gamma_v,
        b_pa,
        c_min_m_s,
        p_v_gauge_pa,
    );
    let rho_minus_width = rho_m_plus_kg_m3 - rho_m_minus_kg_m3;
    let c_l2 = c_l_m_s * c_l_m_s;
    let edges = MixtureJunctionEdges {
        rho_m_plus_kg_m3,
        rho_m_minus_kg_m3,
        c_min_m_s,
        p_v_gauge_pa,
    };

    let analytic_delta_mix_liquid = mixture_slope_cutoff_delta(rho_minus_width, c_min_m_s, c_l2);
    let patch_liquid = build_junction_patch(
        rho_m_plus_kg_m3,
        -1.0,
        analytic_delta_mix_liquid,
        edges,
        1.0,
        |delta_pure| {
            let rho = rho_m_plus_kg_m3 + delta_pure;
            (c_l2 * (rho - rho_l_ref_kg_m3), c_l2)
        },
    );
    // For the liquid junction the mixture edge is the patch's left side
    // (`mix_sign=-1`), so `delta_mix = rho_junction - patch.rho_left`.
    let delta_mix_liquid_kg_m3 = rho_m_plus_kg_m3 - patch_liquid.rho_left;

    let s_max_vapor = vapor_dp_drho_raw(rho_m_minus_kg_m3, b_pa, gamma_v, rho_v_ref_kg_m3);
    let analytic_delta_mix_vapor =
        mixture_slope_cutoff_delta(rho_minus_width, c_min_m_s, s_max_vapor);
    let patch_vapor = build_junction_patch(
        rho_m_minus_kg_m3,
        1.0,
        analytic_delta_mix_vapor,
        edges,
        -1.0,
        |delta_pure| {
            let rho = rho_m_minus_kg_m3 - delta_pure;
            (
                vapor_pressure_gauge_raw(rho, b_pa, gamma_v, rho_v_ref_kg_m3),
                vapor_dp_drho_raw(rho, b_pa, gamma_v, rho_v_ref_kg_m3),
            )
        },
    );
    // For the vapour junction the mixture edge is the patch's right side
    // (`mix_sign=+1`), so `delta_mix = (rho_left+width) - rho_junction`.
    let delta_mix_vapor_kg_m3 = (patch_vapor.rho_left + patch_vapor.width) - rho_m_minus_kg_m3;

    CavitatingEosPrimitives {
        rho_m_plus_kg_m3,
        rho_m_minus_kg_m3,
        delta_mix_liquid_kg_m3,
        delta_mix_vapor_kg_m3,
    }
}

/// Temperature-indexed table of `CavitatingEosPrimitives`, built once and
/// used for O(1) reconstruction of a full `CavitatingEosParams` at any `T`
/// in `[t_min_k, t_max_k]` through `reconstruct`, the same construction
/// `CavitatingEosParams::new` uses (`reconstruct_junction_patch`) fed an
/// interpolated `delta_mix`. So `pressure_gauge_pa`, `acoustic_c2_si` and
/// any CFL built on it read one EOS.
#[derive(Debug, Clone)]
pub struct CavitatingEosTable {
    /// Liquid branch reference density (kg/m^3) -- real rest density.
    /// Public, same convention `CavitatingEosParams`'s own free inputs
    /// use, since a `MaterialModel` built on this table needs it (e.g.
    /// deriving its own grid-unit rest density) same as it would from a
    /// fixed `CavitatingEosParams`.
    pub rho_l_ref_kg_m3: f32,
    /// Liquid branch sound speed (m/s).
    pub c_l_m_s: f32,
    /// Liquid branch's own Tait-like exponent -- see
    /// `CavitatingEosParams::gamma_l`'s doc.
    pub gamma_l: f32,
    /// Vapor branch reference density (kg/m^3).
    pub rho_v_ref_kg_m3: f32,
    /// Vapor branch polytropic exponent.
    pub gamma_v: f32,
    /// Mixture-region minimum sound speed (m/s).
    pub c_min_m_s: f32,
    t_min_k: f32,
    t_max_k: f32,
    rho_m_plus_kg_m3: Vec<f32>,
    rho_m_minus_kg_m3: Vec<f32>,
    delta_mix_liquid_kg_m3: Vec<f32>,
    delta_mix_vapor_kg_m3: Vec<f32>,
}

impl CavitatingEosTable {
    /// Resolution: starts at a small node count and doubles until the worst
    /// error, pressure and derivative, at the midpoints between nodes (where
    /// interpolation is worst; it is exact at the nodes) across a density
    /// sweep, against the direct `CavitatingEosParams::at_temperature`
    /// solve, falls under tolerance.
    pub fn build(
        rho_l_ref_kg_m3: f32,
        c_l_m_s: f32,
        gamma_l: f32,
        rho_v_ref_kg_m3: f32,
        gamma_v: f32,
        c_min_m_s: f32,
        t_min_k: f32,
    ) -> Self {
        let branch_inputs = EosBranchInputs {
            rho_l_ref_kg_m3,
            c_l_m_s,
            gamma_l,
            rho_v_ref_kg_m3,
            gamma_v,
            c_min_m_s,
        };
        let t_max_k = t_liquid_closure_max(branch_inputs, t_min_k, WATER_SATURATION_MAX_VALID_K);
        // 0.1% relative pressure error, 5% relative derivative error (the
        // derivative tolerance of `acoustic_c2_matches_finite_difference_
        // of_pressure_outside_the_c1_patches`); pressure is tighter since it
        // drives the P2G force directly.
        const PRESSURE_REL_TOL: f32 = 1.0e-3;
        const DERIVATIVE_REL_TOL: f32 = 0.05;
        const MAX_NODES: usize = 4096;
        let mut node_count = 4usize;
        loop {
            let table = Self::build_with_node_count(branch_inputs, t_min_k, t_max_k, node_count);
            let (p_err, d_err) = table.measure_worst_case_interpolation_error();
            if (p_err < PRESSURE_REL_TOL && d_err < DERIVATIVE_REL_TOL) || node_count >= MAX_NODES {
                return table;
            }
            node_count *= 2;
        }
    }

    fn build_with_node_count(
        inputs: EosBranchInputs,
        t_min_k: f32,
        t_max_k: f32,
        node_count: usize,
    ) -> Self {
        let EosBranchInputs {
            rho_l_ref_kg_m3,
            c_l_m_s,
            gamma_l,
            rho_v_ref_kg_m3,
            gamma_v,
            c_min_m_s,
        } = inputs;
        let mut rho_m_plus_kg_m3 = Vec::with_capacity(node_count);
        let mut rho_m_minus_kg_m3 = Vec::with_capacity(node_count);
        let mut delta_mix_liquid_kg_m3 = Vec::with_capacity(node_count);
        let mut delta_mix_vapor_kg_m3 = Vec::with_capacity(node_count);
        for i in 0..node_count {
            let t = t_min_k + (t_max_k - t_min_k) * (i as f32 / (node_count - 1) as f32);
            let p = primitives_at_temperature(
                rho_l_ref_kg_m3,
                c_l_m_s,
                gamma_l,
                rho_v_ref_kg_m3,
                gamma_v,
                c_min_m_s,
                t,
            );
            rho_m_plus_kg_m3.push(p.rho_m_plus_kg_m3);
            rho_m_minus_kg_m3.push(p.rho_m_minus_kg_m3);
            delta_mix_liquid_kg_m3.push(p.delta_mix_liquid_kg_m3);
            delta_mix_vapor_kg_m3.push(p.delta_mix_vapor_kg_m3);
        }
        Self {
            rho_l_ref_kg_m3,
            c_l_m_s,
            gamma_l,
            rho_v_ref_kg_m3,
            gamma_v,
            c_min_m_s,
            t_min_k,
            t_max_k,
            rho_m_plus_kg_m3,
            rho_m_minus_kg_m3,
            delta_mix_liquid_kg_m3,
            delta_mix_vapor_kg_m3,
        }
    }

    /// Samples the midpoint between every pair of adjacent nodes (where
    /// linear interpolation is worst) and compares `reconstruct(t)` with the
    /// direct `CavitatingEosParams::at_temperature(t)` across a density
    /// sweep from the vapour to the liquid branch. Returns the worst
    /// relative error, `(pressure, derivative)`.
    fn measure_worst_case_interpolation_error(&self) -> (f32, f32) {
        let mut worst_p_err = 0.0f32;
        let mut worst_d_err = 0.0f32;
        let n = self.rho_m_plus_kg_m3.len();
        if n < 2 {
            return (f32::INFINITY, f32::INFINITY); // can't interpolate at all yet
        }
        const DENSITY_SAMPLES: usize = 30;
        for i in 0..n - 1 {
            let t_lo = self.t_min_k + (self.t_max_k - self.t_min_k) * (i as f32 / (n - 1) as f32);
            let t_hi =
                self.t_min_k + (self.t_max_k - self.t_min_k) * ((i + 1) as f32 / (n - 1) as f32);
            let t_mid = 0.5 * (t_lo + t_hi);
            let via_table = self.reconstruct(t_mid);
            let via_direct = CavitatingEosParams::at_temperature(
                self.rho_l_ref_kg_m3,
                self.c_l_m_s,
                self.gamma_l,
                self.rho_v_ref_kg_m3,
                self.gamma_v,
                self.c_min_m_s,
                t_mid,
            );
            let rho_lo = self.rho_v_ref_kg_m3 * 0.5;
            let rho_hi = self.rho_l_ref_kg_m3 * 1.05;
            for j in 0..=DENSITY_SAMPLES {
                let rho = rho_lo + (rho_hi - rho_lo) * (j as f32 / DENSITY_SAMPLES as f32);
                let p_table = via_table.pressure_gauge_pa(rho);
                let p_direct = via_direct.pressure_gauge_pa(rho);
                let p_err = (p_table - p_direct).abs() / p_direct.abs().max(1.0);
                worst_p_err = worst_p_err.max(p_err);
                let d_table = via_table.acoustic_c2_si(rho);
                let d_direct = via_direct.acoustic_c2_si(rho);
                let d_err = (d_table - d_direct).abs() / d_direct.abs().max(1.0);
                worst_d_err = worst_d_err.max(d_err);
            }
        }
        (worst_p_err, worst_d_err)
    }

    fn interpolate_primitives_at_temperature(&self, temperature_k: f32) -> CavitatingEosPrimitives {
        let t = temperature_k.clamp(self.t_min_k, self.t_max_k);
        let n = self.rho_m_plus_kg_m3.len();
        let frac = if self.t_max_k > self.t_min_k {
            (t - self.t_min_k) / (self.t_max_k - self.t_min_k)
        } else {
            0.0
        };
        let pos = (frac * (n - 1) as f32).clamp(0.0, (n - 1) as f32);
        let i0 = (pos.floor() as usize).min(n - 2);
        let i1 = i0 + 1;
        let local = (pos - i0 as f32).clamp(0.0, 1.0);
        let lerp = |a: f32, b: f32| a + (b - a) * local;
        CavitatingEosPrimitives {
            rho_m_plus_kg_m3: lerp(self.rho_m_plus_kg_m3[i0], self.rho_m_plus_kg_m3[i1]),
            rho_m_minus_kg_m3: lerp(self.rho_m_minus_kg_m3[i0], self.rho_m_minus_kg_m3[i1]),
            delta_mix_liquid_kg_m3: lerp(
                self.delta_mix_liquid_kg_m3[i0],
                self.delta_mix_liquid_kg_m3[i1],
            ),
            delta_mix_vapor_kg_m3: lerp(
                self.delta_mix_vapor_kg_m3[i0],
                self.delta_mix_vapor_kg_m3[i1],
            ),
        }
    }

    /// O(1) reconstruction of a full `CavitatingEosParams` at `T` from the
    /// interpolated primitives; callers use its `pressure_gauge_pa`/
    /// `acoustic_c2_si` as for a fixed-`T` instance. The one reconstruction
    /// path for pressure, `dp/drho` and any CFL term.
    pub fn reconstruct(&self, temperature_k: f32) -> CavitatingEosParams {
        let p = self.interpolate_primitives_at_temperature(temperature_k);
        let p_v_gauge_pa = p_v_gauge_pa_at_temperature_f64(temperature_k) as f32;
        let c_l2 = self.c_l_m_s * self.c_l_m_s;
        let b_pa = c_l2 * self.rho_l_ref_kg_m3 / self.gamma_l;
        let rho_m_plus_kg_m3 = p.rho_m_plus_kg_m3;
        let rho_m_minus_kg_m3 = p.rho_m_minus_kg_m3;
        let rho_l_ref_kg_m3 = self.rho_l_ref_kg_m3;
        let edges = MixtureJunctionEdges {
            rho_m_plus_kg_m3,
            rho_m_minus_kg_m3,
            c_min_m_s: self.c_min_m_s,
            p_v_gauge_pa,
        };
        let patch_liquid_junction = reconstruct_junction_patch(
            rho_m_plus_kg_m3,
            -1.0,
            p.delta_mix_liquid_kg_m3,
            edges,
            1.0,
            |delta_pure| {
                let rho = rho_m_plus_kg_m3 + delta_pure;
                (c_l2 * (rho - rho_l_ref_kg_m3), c_l2)
            },
        );
        let patch_vapor_junction = reconstruct_junction_patch(
            rho_m_minus_kg_m3,
            1.0,
            p.delta_mix_vapor_kg_m3,
            edges,
            -1.0,
            |delta_pure| {
                let rho = rho_m_minus_kg_m3 - delta_pure;
                (
                    vapor_pressure_gauge_raw(rho, b_pa, self.gamma_v, self.rho_v_ref_kg_m3),
                    vapor_dp_drho_raw(rho, b_pa, self.gamma_v, self.rho_v_ref_kg_m3),
                )
            },
        );
        CavitatingEosParams {
            rho_l_ref_kg_m3,
            c_l_m_s: self.c_l_m_s,
            gamma_l: self.gamma_l,
            rho_v_ref_kg_m3: self.rho_v_ref_kg_m3,
            gamma_v: self.gamma_v,
            c_min_m_s: self.c_min_m_s,
            p_v_gauge_pa,
            b_pa,
            rho_m_plus_kg_m3,
            rho_m_minus_kg_m3,
            patch_liquid_junction,
            patch_vapor_junction,
        }
    }

    /// The node count the resolution search converged to (tests,
    /// diagnostics).
    pub fn node_count(&self) -> usize {
        self.rho_m_plus_kg_m3.len()
    }

    pub fn t_min_k(&self) -> f32 {
        self.t_min_k
    }

    pub fn t_max_k(&self) -> f32 {
        self.t_max_k
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::energy::thermodynamics::water_saturation::water_saturation_pressure_pa;
    use crate::materials::gas::STANDARD_ATMOSPHERE_PA;

    /// Test configuration: `rho_l_ref` water's rest density, `c_l` this
    /// engine's `WATER_C_REF_M_S` (`phase_states_gui.rs`), `gamma_l` 7.0
    /// (Cole 1948, as `weakly_compressible`), `rho_v_ref` the demo's
    /// `STEAM_RHO_KG_M3` (`WATER_RHO_KG_M3/6`), `gamma_v` 1.33 (water vapour
    /// specific-heat ratio), `p_v_gauge` the Antoine saturation pressure at
    /// 300 K minus one standard atmosphere. `c_min` is a model choice, not a
    /// production value (see module doc).
    fn real_test_params() -> CavitatingEosParams {
        let p_v_abs = water_saturation_pressure_pa(300.0);
        CavitatingEosParams::new(
            1000.0,                           // rho_l_ref_kg_m3
            180.0,                            // c_l_m_s
            7.0,                              // gamma_l (Cole 1948)
            1000.0 / 6.0,                     // rho_v_ref_kg_m3
            1.33,                             // gamma_v
            1.0,                              // c_min_m_s -- real model choice, exercised here
            p_v_abs - STANDARD_ATMOSPHERE_PA, // p_v_gauge_pa
        )
    }

    /// Constants shared by the temperature-dependent closure tests: those of
    /// `real_test_params` without the fixed `p_v_gauge_pa` (these vary `T`).
    const T_CLOSURE_RHO_L_REF_KG_M3: f32 = 1000.0;
    const T_CLOSURE_C_L_M_S: f32 = 180.0;
    const T_CLOSURE_GAMMA_L: f32 = 7.0;
    const T_CLOSURE_RHO_V_REF_KG_M3: f32 = 1000.0 / 6.0;
    const T_CLOSURE_GAMMA_V: f32 = 1.33;
    const T_CLOSURE_C_MIN_M_S: f32 = 1.0;

    /// `at_temperature` reproduces `new` exactly when fed the equivalent
    /// `p_v_gauge_pa`.
    #[test]
    fn at_temperature_matches_new_given_the_equivalent_gauge_pressure() {
        let p_v_gauge_pa = p_v_gauge_pa_at_temperature_f64(300.0) as f32;
        let via_new = CavitatingEosParams::new(
            T_CLOSURE_RHO_L_REF_KG_M3,
            T_CLOSURE_C_L_M_S,
            T_CLOSURE_GAMMA_L,
            T_CLOSURE_RHO_V_REF_KG_M3,
            T_CLOSURE_GAMMA_V,
            T_CLOSURE_C_MIN_M_S,
            p_v_gauge_pa,
        );
        let via_at_temperature = CavitatingEosParams::at_temperature(
            T_CLOSURE_RHO_L_REF_KG_M3,
            T_CLOSURE_C_L_M_S,
            T_CLOSURE_GAMMA_L,
            T_CLOSURE_RHO_V_REF_KG_M3,
            T_CLOSURE_GAMMA_V,
            T_CLOSURE_C_MIN_M_S,
            300.0,
        );
        assert_eq!(
            via_new.rho_m_plus_kg_m3,
            via_at_temperature.rho_m_plus_kg_m3
        );
        assert_eq!(
            via_new.rho_m_minus_kg_m3,
            via_at_temperature.rho_m_minus_kg_m3
        );
        assert_eq!(
            via_new.pressure_gauge_pa(999.0),
            via_at_temperature.pressure_gauge_pa(999.0)
        );
    }

    /// `t_liquid_closure_max` lands close to, and strictly below, the boiling
    /// point (373.15 K), since `rho_m+` drifts past `rho_l_ref` as `T` nears
    /// boiling. Within 1 K: the exact value follows from `c_min`/`c_l`.
    #[test]
    fn t_liquid_closure_max_lands_just_below_the_true_boiling_point() {
        let t_max = t_liquid_closure_max(
            EosBranchInputs {
                rho_l_ref_kg_m3: T_CLOSURE_RHO_L_REF_KG_M3,
                c_l_m_s: T_CLOSURE_C_L_M_S,
                gamma_l: T_CLOSURE_GAMMA_L,
                rho_v_ref_kg_m3: T_CLOSURE_RHO_V_REF_KG_M3,
                gamma_v: T_CLOSURE_GAMMA_V,
                c_min_m_s: T_CLOSURE_C_MIN_M_S,
            },
            273.15,
            373.15,
        );
        assert!(
            t_max < 373.15 && t_max > 372.0,
            "t_liquid_closure_max={t_max} should land within ~1K below the true \
             boiling point for these real material constants, not further off"
        );
        // Constructing right at the bisected boundary succeeds, and the
        // liquid patch stays strictly inside rho_l_ref.
        let params = CavitatingEosParams::at_temperature(
            T_CLOSURE_RHO_L_REF_KG_M3,
            T_CLOSURE_C_L_M_S,
            T_CLOSURE_GAMMA_L,
            T_CLOSURE_RHO_V_REF_KG_M3,
            T_CLOSURE_GAMMA_V,
            T_CLOSURE_C_MIN_M_S,
            t_max,
        );
        assert!(params.liquid_patch_outer_edge_kg_m3() < T_CLOSURE_RHO_L_REF_KG_M3);
    }

    /// Dense sweep over the whole closure range `[273.15,
    /// t_liquid_closure_max]`: at every sampled temperature the band
    /// exists, densities are ordered, pressure is monotonic, the C^1
    /// junctions are continuous and patch derivatives finite and positive.
    /// Direct construction at each `T`, not table lookup.
    #[test]
    fn dense_temperature_sweep_holds_every_real_invariant_up_to_the_closure_boundary() {
        let t_max = t_liquid_closure_max(
            EosBranchInputs {
                rho_l_ref_kg_m3: T_CLOSURE_RHO_L_REF_KG_M3,
                c_l_m_s: T_CLOSURE_C_L_M_S,
                gamma_l: T_CLOSURE_GAMMA_L,
                rho_v_ref_kg_m3: T_CLOSURE_RHO_V_REF_KG_M3,
                gamma_v: T_CLOSURE_GAMMA_V,
                c_min_m_s: T_CLOSURE_C_MIN_M_S,
            },
            273.15,
            373.15,
        );
        const N: usize = 500;
        for i in 0..=N {
            let t = 273.15 + (t_max - 273.15) * (i as f32 / N as f32);
            let params = CavitatingEosParams::at_temperature(
                T_CLOSURE_RHO_L_REF_KG_M3,
                T_CLOSURE_C_L_M_S,
                T_CLOSURE_GAMMA_L,
                T_CLOSURE_RHO_V_REF_KG_M3,
                T_CLOSURE_GAMMA_V,
                T_CLOSURE_C_MIN_M_S,
                t,
            );
            assert!(
                params.rho_m_minus_kg_m3 > 0.0
                    && params.rho_m_minus_kg_m3 < params.rho_m_plus_kg_m3
                    && params.rho_m_plus_kg_m3 < T_CLOSURE_RHO_L_REF_KG_M3,
                "T={t}K: real density ordering violated"
            );
            assert!(
                params.is_continuous(1.0),
                "T={t}K: real C^1 junction continuity violated"
            );
            let (lo_l, hi_l) = params.patch_liquid_junction.derivative_extrema();
            let (lo_v, hi_v) = params.patch_vapor_junction.derivative_extrema();
            assert!(
                lo_l > 0.0 && hi_l.is_finite() && lo_v > 0.0 && hi_v.is_finite(),
                "T={t}K: real patch derivative bound not finite/positive -- \
                 liquid=[{lo_l},{hi_l}] vapor=[{lo_v},{hi_v}]"
            );
            // Monotonic over a density sweep at this T, as
            // `pressure_is_monotonically_increasing_with_density_everywhere`
            // checks at one fixed T.
            let rho_lo = params.rho_v_ref_kg_m3 * 0.5;
            let rho_hi = T_CLOSURE_RHO_L_REF_KG_M3 * 1.05;
            const M: usize = 40;
            let mut prev_p = params.pressure_gauge_pa(rho_lo);
            for j in 1..=M {
                let rho = rho_lo + (rho_hi - rho_lo) * (j as f32 / M as f32);
                let p = params.pressure_gauge_pa(rho);
                assert!(
                    p >= prev_p,
                    "T={t}K: pressure must not decrease with density (rho={rho}, \
                     p={p} < prev_p={prev_p})"
                );
                prev_p = p;
            }
        }
    }

    /// The table's resolution search converges to a finite node count within
    /// its cap, and a modest one for a closure barely 100 K wide. Printed,
    /// not pinned: the count is a search outcome.
    #[test]
    fn table_build_converges_to_a_real_bounded_node_count() {
        let table = CavitatingEosTable::build(
            T_CLOSURE_RHO_L_REF_KG_M3,
            T_CLOSURE_C_L_M_S,
            T_CLOSURE_GAMMA_L,
            T_CLOSURE_RHO_V_REF_KG_M3,
            T_CLOSURE_GAMMA_V,
            T_CLOSURE_C_MIN_M_S,
            273.15,
        );
        println!(
            "[table] node_count={} t_min={} t_max={}",
            table.node_count(),
            table.t_min_k(),
            table.t_max_k()
        );
        assert!(
            table.node_count() >= 4 && table.node_count() < 4096,
            "table converged to node_count={}, expected a real, bounded search \
             outcome strictly under the search cap",
            table.node_count()
        );
        let (p_err, d_err) = table.measure_worst_case_interpolation_error();
        println!("[table] worst-case interpolation error: p_err={p_err} d_err={d_err}");
        assert!(p_err < 1.0e-3 && d_err < 0.05);
    }

    /// Reconstructing at a table node reproduces the direct `at_temperature`
    /// solve almost exactly: interpolation is exact at the nodes, so any
    /// discrepancy there is a reconstruction bug.
    #[test]
    fn reconstruct_at_a_table_node_matches_the_direct_solve() {
        let table = CavitatingEosTable::build(
            T_CLOSURE_RHO_L_REF_KG_M3,
            T_CLOSURE_C_L_M_S,
            T_CLOSURE_GAMMA_L,
            T_CLOSURE_RHO_V_REF_KG_M3,
            T_CLOSURE_GAMMA_V,
            T_CLOSURE_C_MIN_M_S,
            273.15,
        );
        let t_node = table.t_min_k();
        let via_table = table.reconstruct(t_node);
        let via_direct = CavitatingEosParams::at_temperature(
            T_CLOSURE_RHO_L_REF_KG_M3,
            T_CLOSURE_C_L_M_S,
            T_CLOSURE_GAMMA_L,
            T_CLOSURE_RHO_V_REF_KG_M3,
            T_CLOSURE_GAMMA_V,
            T_CLOSURE_C_MIN_M_S,
            t_node,
        );
        for rho in [200.0, 500.0, 999.0, 999.999] {
            let p_table = via_table.pressure_gauge_pa(rho);
            let p_direct = via_direct.pressure_gauge_pa(rho);
            assert!(
                (p_table - p_direct).abs() < 1.0,
                "rho={rho}: table={p_table} direct={p_direct} disagree by more \
                 than 1 Pa at a real table node"
            );
        }
    }

    /// Pressure is exactly zero gauge at the liquid's rest density (rest
    /// density must not fall inside the mixture band).
    #[test]
    fn pressure_is_exactly_zero_gauge_at_rest_density() {
        let params = real_test_params();
        let p = params.pressure_gauge_pa(params.rho_l_ref_kg_m3);
        assert!(
            p.abs() < 1.0,
            "rest density must give exactly zero gauge pressure, got {p} Pa"
        );
    }

    /// The rest density is classified as pure liquid (above the mixture
    /// band).
    #[test]
    fn rest_density_is_classified_as_pure_liquid() {
        let params = real_test_params();
        assert!(
            params.rho_l_ref_kg_m3 > params.rho_m_plus_kg_m3,
            "rest density ({}) must be ABOVE the mixture band's own liquid-side \
             edge ({}) -- otherwise rest state is wrongly inside the cavitating \
             region",
            params.rho_l_ref_kg_m3,
            params.rho_m_plus_kg_m3
        );
    }

    /// The shared stiffness `B` derived from the liquid branch matches
    /// `B = c_l^2 rho_l_ref / gamma_l`, the `c^2 = B gamma / rho0` identity
    /// `NewtonianFluidMaterial::rest_acoustic_c2` uses: eq. 10's cross-branch
    /// consistency.
    #[test]
    fn shared_stiffness_matches_the_real_c_squared_equals_b_gamma_over_rho_identity() {
        let params = real_test_params();
        let expected_b = params.c_l_m_s * params.c_l_m_s * params.rho_l_ref_kg_m3 / params.gamma_l;
        assert!(
            (params.b_pa - expected_b).abs() / expected_b < 1.0e-4,
            "derived B={} must match c_l^2*rho_l_ref/gamma_l={} exactly",
            params.b_pa,
            expected_b
        );
        // And the SAME B must reproduce a physically sane vapor
        // sound speed c_v = sqrt(B*gamma_v/rho_v_ref).
        let c_v = (params.b_pa * params.gamma_v / params.rho_v_ref_kg_m3).sqrt();
        assert!(
            c_v.is_finite() && c_v > 0.0,
            "derived vapor sound speed must be real, finite, and positive, got {c_v}"
        );
    }

    /// The liquid branch's constant slope is not a safe global bound for
    /// `acoustic_c2_si`: for these parameters the vapour branch's slope at
    /// the mixture boundary exceeds it, so a wrapper assuming `c_l_m_s^2` is
    /// conservative would pick too large a timestep.
    #[test]
    fn vapor_branch_acoustic_speed_can_exceed_the_liquid_branch_at_its_own_boundary() {
        let params = real_test_params();
        let c_l2 = params.c_l_m_s * params.c_l_m_s;
        let c_v2_at_boundary = params.acoustic_c2_si(params.rho_m_minus_kg_m3 - 1.0);
        assert!(
            c_v2_at_boundary > c_l2,
            "expected the vapor branch's own real acoustic speed squared \
             ({c_v2_at_boundary}) to exceed the liquid branch's constant \
             ({c_l2}) for these real test parameters -- if this no longer \
             holds, re-verify `timestep_bound`'s own real safety margin \
             assumption still applies"
        );
    }

    /// The mixture branch's analytic derivative diverges at its two edges
    /// (a property of the arcsin closure); `acoustic_c2_si` stays finite
    /// there.
    #[test]
    fn acoustic_c2_stays_finite_exactly_at_the_mixture_bands_own_edges() {
        let params = real_test_params();
        for rho in [params.rho_m_plus_kg_m3, params.rho_m_minus_kg_m3] {
            let c2 = params.acoustic_c2_si(rho);
            assert!(
                c2.is_finite() && c2 > 0.0,
                "acoustic_c2_si must stay finite and positive exactly at a \
                 mixture-band edge (rho={rho}), got {c2}"
            );
        }
    }

    /// Each C^1 patch's exact derivative (`C1Patch::derivative_extrema`,
    /// closed form) stays within `[c_min^2, s_max]` everywhere in the patch:
    /// the mixture band's minimum sound speed on one side, the neighbouring
    /// pure branch's slope on the other.
    #[test]
    fn c1_patches_keep_their_derivative_within_the_real_principled_bound() {
        let params = real_test_params();
        let c_min2 = params.c_min_m_s * params.c_min_m_s;
        let c_l2 = params.c_l_m_s * params.c_l_m_s;
        let s_max_vapor = vapor_dp_drho_raw(
            params.rho_m_minus_kg_m3,
            params.b_pa,
            params.gamma_v,
            params.rho_v_ref_kg_m3,
        );
        let (lo_liquid, hi_liquid) = params.patch_liquid_junction.derivative_extrema();
        assert!(
            lo_liquid >= c_min2 * 0.999 && hi_liquid <= c_l2 * 1.001,
            "liquid-junction patch derivative extrema [{lo_liquid}, {hi_liquid}] must \
             stay within [c_min^2={c_min2}, c_l^2={c_l2}]"
        );
        let (lo_vapor, hi_vapor) = params.patch_vapor_junction.derivative_extrema();
        assert!(
            lo_vapor >= c_min2 * 0.999 && hi_vapor <= s_max_vapor * 1.001,
            "vapor-junction patch derivative extrema [{lo_vapor}, {hi_vapor}] must \
             stay within [c_min^2={c_min2}, s_max_vapor={s_max_vapor}]"
        );
    }

    /// `acoustic_c2_si` (analytic) matches a finite difference of
    /// `pressure_gauge_pa` everywhere outside the two C^1 patches, which are
    /// too thin for one global `h` to resolve (their closed-form bound is
    /// covered by `c1_patches_keep_their_derivative_within_the_real_
    /// principled_bound`).
    #[test]
    fn acoustic_c2_matches_finite_difference_of_pressure_outside_the_c1_patches() {
        // The patches are a handful of f32 ULPs wide. One global finite-
        // difference step `h` either steps clean past a patch (truncation
        // mismatch) or drowns in f32 pressure quantization (pressures reach
        // ~1e6 Pa, rounding ~0.01-0.2 Pa), so patch interiors are skipped;
        // their closed-form bound is checked separately.
        let params = real_test_params();
        let rho_lo = params.rho_v_ref_kg_m3 * 0.5;
        let rho_hi = params.rho_l_ref_kg_m3 * 1.05;
        let n = 5000;
        // `h` must also clear the f32 rounding floor of the pressure itself
        // (~0.01-0.2 Pa at 1e5-1e6 Pa of gauge offset): where the slope is
        // small (mid-band, ~c_min^2 = 1) `h = 1e-3` gives a 0.002 Pa signal,
        // noise. `h = 0.1` gives ~0.2 Pa there and stays far below every
        // branch's curvature scale (hundreds of kg/m^3).
        let h = 0.1_f32;
        // A patch overlapping the PROBE INTERVAL `[rho-h,rho+h]` (not just
        // the center point `rho`) must also be skipped -- both real
        // patches are only a handful of ULPs wide, far thinner than `h`
        // itself, so a center point sitting just outside a patch can still
        // have `rho+h`/`rho-h` land on its far side, turning the "local"
        // finite difference into a coarse secant across the whole patch.
        let overlaps_patch = |lo: f32, hi: f32, patch: &C1Patch| -> bool {
            patch.rho_left <= hi && patch.rho_left + patch.width >= lo
        };
        // The raw mixture derivative diverges like `1/sqrt(distance to the
        // edge)`, so a fixed `h = 0.1` also fails a little past each patch
        // (0.15 kg/m^3 past the vapour patch: 7 percent truncation error).
        // Excludes 5 percent of the band's width around each edge, leaving
        // ~90 percent of the band and both pure branches; the excluded part
        // is covered by the patch-bound test and the raw formula's tests.
        let rho_minus_width = params.rho_m_plus_kg_m3 - params.rho_m_minus_kg_m3;
        let junction_exclusion = 0.05 * rho_minus_width;
        let near_junction = |rho: f32, junction: f32| (rho - junction).abs() < junction_exclusion;
        let mut max_rel_err = 0.0_f32;
        for i in 1..n {
            let rho = rho_lo + (rho_hi - rho_lo) * (i as f32 / n as f32);
            if overlaps_patch(rho - h, rho + h, &params.patch_liquid_junction)
                || overlaps_patch(rho - h, rho + h, &params.patch_vapor_junction)
                || near_junction(rho, params.rho_m_plus_kg_m3)
                || near_junction(rho, params.rho_m_minus_kg_m3)
            {
                continue;
            }
            let analytic = params.acoustic_c2_si(rho);
            let p_plus = params.pressure_gauge_pa(rho + h);
            let p_minus = params.pressure_gauge_pa(rho - h);
            let numeric = (p_plus - p_minus) / (2.0 * h);
            let rel_err = (analytic - numeric).abs() / analytic.abs().max(1.0);
            max_rel_err = max_rel_err.max(rel_err);
        }
        assert!(
            max_rel_err < 0.05,
            "acoustic_c2_si must match a real finite-difference derivative of \
             pressure_gauge_pa everywhere OUTSIDE both C^1 patches -- max relative \
             error over the real sweep was {max_rel_err}"
        );
    }

    /// The derivation's own direct self-check must hold.
    #[test]
    fn derived_parameters_produce_a_continuous_pressure_curve() {
        let params = real_test_params();
        assert!(
            params.is_continuous(1.0),
            "solved mixture band must produce p(rho) continuous to within 1 Pa \
             at both branch boundaries"
        );
    }

    /// Pressure strictly increases with density everywhere except exactly
    /// at the mixture band's edges (the arcsin-derivative singularity).
    #[test]
    fn pressure_is_monotonically_increasing_with_density_everywhere() {
        let params = real_test_params();
        let rho_lo = params.rho_v_ref_kg_m3 * 0.5;
        let rho_hi = params.rho_l_ref_kg_m3 * 1.2;
        let n = 2000;
        let mut prev_p = params.pressure_gauge_pa(rho_lo);
        for i in 1..=n {
            let rho = rho_lo + (rho_hi - rho_lo) * (i as f32 / n as f32);
            let p = params.pressure_gauge_pa(rho);
            assert!(
                p >= prev_p - 1.0e-3,
                "pressure must be monotonically non-decreasing with density: \
                 rho={rho} gave p={p}, previous sample gave {prev_p}"
            );
            prev_p = p;
        }
    }

    /// The mixture branch passes through the gauge saturation pressure at
    /// its midpoint density.
    #[test]
    fn mixture_branch_passes_through_saturation_pressure_at_its_own_midpoint() {
        let params = real_test_params();
        let midpoint = (params.rho_m_plus_kg_m3 + params.rho_m_minus_kg_m3) / 2.0;
        let p_mid = params.pressure_gauge_pa(midpoint);
        assert!(
            (p_mid - params.p_v_gauge_pa).abs() < 1.0,
            "mixture branch must pass through the real gauge saturation pressure \
             at its own midpoint density -- got {p_mid} Pa, expected {} Pa",
            params.p_v_gauge_pa
        );
    }

    /// An unphysical `c_min` (too large for the band to bridge the
    /// liquid/vapour pair) is rejected, not turned into a wrong band.
    #[test]
    #[should_panic]
    fn rejects_a_c_min_with_no_real_mixture_band_solution() {
        let p_v_abs = water_saturation_pressure_pa(300.0);
        CavitatingEosParams::new(
            1000.0,
            180.0,
            7.0,
            1000.0 / 6.0,
            1.33,
            1.0e6, // absurdly large c_min -- no real bracket should exist
            p_v_abs - STANDARD_ATMOSPHERE_PA,
        );
    }
}

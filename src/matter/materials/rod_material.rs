use crate::particle::RodPoints;

/// Real SI-unit rod material parameters — `EA`/`EI` use the SAME `E`
/// (Young's modulus) and `I` (second moment of area, `I ~ width^3` for a
/// rectangular section) used in the Greenhill self-buckling analysis
/// (`h_crit = (7.8373*EI/(linear_density*g))^(1/3)`) — direct continuity
/// with that formula, not a new concept.
///
/// Moved here from `spacetime::rod` 2026-08-05 -- not a `MaterialModel`
/// (no `constitutive_model`/`kirchhoff_stress` impl, doesn't plug into the
/// generic multi-material MPM dispatch the other 13 materials share), but
/// genuinely a constitutive law (EA/EI stiffness) same as
/// `grain_contact_law`'s own real precedent for this exact situation.
/// Originally kept in `spacetime::rod` because two of its own methods took
/// `&RodPoints` directly, which would have been a real backwards dependency
/// (matter depending on spacetime) at the time -- moot now that `RodPoints`
/// itself lives in `matter::particle`: both are Matter-domain kinematic
/// state/constitutive-law now, so this is a same-domain reference, not a
/// cross-domain one.
#[derive(Debug, Clone, Copy)]
pub struct RodMaterial {
    /// Axial (stretch) stiffness `E*A`, Newtons.
    pub ea: f32,
    /// Bending stiffness `E*I`, N·m².
    pub ei: f32,
    /// Kelvin-Voigt axial dashpot (mirrors `ViscoelasticMaterial`'s own
    /// `viscosity` field, 1D-projected onto each edge), N·s/m.
    pub axial_damping: f32,
    /// Rayleigh bending dissipation coefficient, N·m·s. Real, standard
    /// generalized-force construction (Rayleigh 1873) — disclosed as this
    /// plan's own composition of two separately-citable classical-mechanics
    /// results (Kelvin-Voigt + Rayleigh dissipation); Bergou et al. 2008
    /// itself does not define damping at all.
    pub bending_damping: f32,
}

impl RodMaterial {
    pub const fn new(ea: f32, ei: f32, axial_damping: f32, bending_damping: f32) -> Self {
        Self {
            ea,
            ei,
            axial_damping,
            bending_damping,
        }
    }

    /// Rectangular cross-section convenience: `A = width*thickness`,
    /// `I = width^3*thickness/12` (bending about the axis perpendicular to
    /// the simulation's own 2D plane — the same `I ~ width^3` relationship
    /// as Wikipedia's "Self-buckling" reference). `thickness_m` is the
    /// engine's own implicit out-of-plane depth (same convention
    /// `Elastic::particle_mass`'s areal
    /// density already assumes) — pass `1.0` unless modeling a real
    /// non-unit depth.
    pub fn from_young_modulus_rectangular(
        young_modulus_pa: f32,
        width_m: f32,
        thickness_m: f32,
        axial_damping: f32,
        bending_damping: f32,
    ) -> Self {
        let area = width_m * thickness_m;
        let i = width_m.powi(3) * thickness_m / 12.0;
        Self::new(
            young_modulus_pa * area,
            young_modulus_pa * i,
            axial_damping,
            bending_damping,
        )
    }

    /// Per-point, LOCAL critical damping `(axial_damping, bending_damping)`
    /// for a rod discretized with uniform segment length `l0_m` and
    /// per-point mass `point_mass_kg`.
    ///
    /// This is a LOCAL, single-segment reference (one point's mass against
    /// one segment's own stiffness) -- it is NOT the true GLOBAL modal
    /// critical damping for a whole rod's actual fundamental bending shape
    /// (many points moving together): for a 20-point cantilever blade it
    /// understates the real modal critical damping by roughly two to three
    /// orders of magnitude. Use `modal_critical_damping` below for natural
    /// whole-rod settling in a physically sensible time -- the common use
    /// case (grass blades, pushable branches, anything a player interacts
    /// with). This function's narrower, still-valid purpose is a per-segment
    /// numerical reference (e.g. bounding a single segment's own worst-case
    /// local stiffness/mass ratio), not a substitute for the true modal
    /// value.
    ///
    /// The two outputs are different kinds of quantities —
    /// `axial_damping` [N·s/m] is a real translational dashpot, while
    /// `bending_damping` [N·m·s] is conjugate to the dimensionless discrete
    /// curvature (see `forces::discrete_curvature`) — so naive `c=2*sqrt(k*m)`
    /// with the same translational stiffness is dimensionally wrong for the
    /// bending term. `bending_damping`'s generalized stiffness is `EI/l0`
    /// [N·m] (matching `compute_internal_forces`'s own `coeff`), its
    /// generalized mass `point_mass*l0²` [kg·m²] (from equating kinetic
    /// energies given curvature's own `|d(kappa)/dx| ~ O(1/l0)` gradient) —
    /// `c_crit=2*sqrt(k_gen*m_gen)` then comes out in the correct N·m·s.
    pub fn critical_damping(l0_m: f32, point_mass_kg: f32, ea: f32, ei: f32) -> (f32, f32) {
        let l0_m = l0_m.max(1.0e-9);
        let k_axial = ea / l0_m; // N/m, real translational edge stiffness
        let axial_damping = 2.0 * (k_axial * point_mass_kg).sqrt();

        let k_bend_generalized = ei / l0_m; // N*m, curvature-space stiffness
        let m_bend_generalized = point_mass_kg * l0_m * l0_m; // kg*m^2
        let bending_damping = 2.0 * (k_bend_generalized * m_bend_generalized).sqrt();

        (axial_damping, bending_damping)
    }

    /// Real, GLOBAL modal critical damping for a fixed-free (cantilever)
    /// rod's actual fundamental modes -- root-cause fix for
    /// `critical_damping`'s own disclosed local-reference gap above.
    ///
    /// # Real physics: bending
    /// Uses the SAME real, independently-confirmed fundamental cantilever
    /// eigenvalue `energy::acoustics::modal` uses (`beta_1*L = 1.8751`,
    /// verified via web search against Blevins 1979 / Rao's *Mechanical
    /// Vibrations*, not recalled from memory alone), and the rod's own real
    /// length/mass (`rest_edge_length`/`mass` sums) -- NOT a new,
    /// independently-invented constant. The mode shape itself is derived
    /// directly from the clamped-root/free-tip boundary value problem
    /// (`phi(0)=phi'(0)=0`, `phi''(L)=phi'''(L)=0`), giving the standard
    /// closed form `phi(x) = (cos(bx)-cosh(bx)) + sigma*(sinh(bx)-sin(bx))`,
    /// `sigma = (sinh(bL)-sin(bL))/(cosh(bL)+cos(bL))` -- re-derived here
    /// from the boundary conditions directly (not copied from an uncertain
    /// memory of a textbook constant), independently checked against the
    /// real characteristic equation `cos(bL)*cosh(bL) = -1`
    /// (cos(1.8751)*cosh(1.8751) ≈ -1.0009, confirms the eigenvalue). The
    /// true modal mass `integral(mu*phi(x)^2 dx)/phi(L)^2` is computed by
    /// real numerical integration (2000-sample composite trapezoidal rule)
    /// over the rod's own uniform-mass assumption (same disclosed
    /// simplification `energy::acoustics::modal` already makes) -- not a
    /// memorized modal-mass constant, so there is no new unverified number
    /// here, only re-derived real physics plus real numerical integration.
    /// `c_crit_bending = 2 * m_modal * omega_1`.
    ///
    /// # Real physics: axial
    /// Fixed-free rod longitudinal vibration has an exact elementary
    /// solution (no numerical integration needed): mode shape `sin(pi x /
    /// 2L)`, modal mass exactly `mu*L/2` (elementary integral of
    /// `sin^2(pi x/2L)` over `[0,L]`), `omega_1 = (pi/2L)*sqrt(EA/mu)`.
    /// `c_crit_axial = 2 * (mu*L/2) * omega_1`.
    ///
    /// # Scope: takes ONE `ea`/`ei`, not `RodPoints::ea`/`ei`
    /// The closed-form mode shape above is only exact for a UNIFORM rod.
    /// For a genuinely non-uniform rod (see `RodPoints::ei`'s own doc) this
    /// is a real, disclosed approximation — pass a representative (e.g.
    /// mean, or the caller's own base `RodMaterial`) value; a true
    /// non-uniform modal solution needs a different, not-yet-built method
    /// (e.g. a real Rayleigh-Ritz or FE eigenvalue solve), not attempted
    /// here.
    pub fn modal_critical_damping(points: &RodPoints, ea: f32, ei: f32) -> (f32, f32) {
        let n = points.len();
        if n < 2 {
            return (0.0, 0.0);
        }
        let length_m: f32 = points.rest_edge_length.iter().sum();
        let total_mass_kg: f32 = points.mass.iter().sum();
        if length_m <= 0.0 || total_mass_kg <= 0.0 {
            return (0.0, 0.0);
        }
        let mu = total_mass_kg / length_m; // kg/m, uniform-rod assumption (disclosed above)

        // ── Axial: exact elementary result ──
        let omega_axial = (std::f32::consts::PI / (2.0 * length_m)) * (ea / mu).sqrt();
        let m_modal_axial = mu * length_m / 2.0;
        let axial_damping = 2.0 * m_modal_axial * omega_axial;

        // ── Bending: real mode shape, numerically integrated ──
        const BETA_L: f32 = 1.8751; // same real constant energy::acoustics::modal uses
        let b = BETA_L / length_m;
        let sigma = (BETA_L.sinh() - BETA_L.sin()) / (BETA_L.cosh() + BETA_L.cos());
        let phi = |x: f32| -> f32 {
            let bx = b * x;
            (bx.cos() - bx.cosh()) + sigma * (bx.sinh() - bx.sin())
        };
        let phi_tip = phi(length_m);
        if phi_tip.abs() < 1.0e-9 {
            // Degenerate (shouldn't happen for the real beta_1*L root) --
            // real, disclosed fallback to the local reference above rather
            // than divide by ~zero.
            let l0 = length_m / (n - 1) as f32;
            let point_mass = total_mass_kg / n as f32;
            return Self::critical_damping(l0, point_mass, ea, ei);
        }

        const SAMPLES: usize = 2000;
        let dx = length_m / SAMPLES as f32;
        let mut integral = 0.0_f32;
        for i in 0..SAMPLES {
            let x0 = i as f32 * dx;
            let x1 = x0 + dx;
            let f0 = phi(x0).powi(2);
            let f1 = phi(x1).powi(2);
            integral += 0.5 * (f0 + f1) * dx; // composite trapezoidal rule
        }
        let m_modal_bending = mu * integral / phi_tip.powi(2);

        let omega_bending = (BETA_L * BETA_L) * (ei / (mu * length_m.powi(4))).sqrt();
        let bending_damping = 2.0 * m_modal_bending * omega_bending;

        (axial_damping, bending_damping)
    }

    /// Real fundamental bending-mode period (seconds) for a fixed-free rod —
    /// same real `beta_1*L=1.8751` eigenvalue and omega formula
    /// `modal_critical_damping` and `energy::acoustics::modal` both use, so
    /// this always agrees with them (single source of truth for this
    /// constant, not a second, possibly-drifting copy).
    ///
    /// Real use (root-cause fix): the rod sleep-scoring test
    /// (`step.rs`) used to require a FIXED 0.5s of sustained low velocity
    /// before sleeping, regardless of the rod's own natural period — for a
    /// soft, slow rod whose own period is comparable to or longer than that
    /// fixed window, a genuine, still-large-amplitude oscillation can dwell
    /// below the speed threshold near a swing peak for that whole window,
    /// freezing the rod mid-swing at a real, wrong, off-rest position (this
    /// is not hypothetical -- it was the direct, measured cause of a real
    /// user-reported bug). Scaling the settle-duration to a real
    /// multiple of THIS rod's own period fixes that at the root instead of
    /// picking a bigger fixed constant that would just move the same
    /// failure mode to an even slower rod.
    ///
    /// Same uniform-rod scope note as `modal_critical_damping` above: pass
    /// one representative `ei`, not a non-uniform rod's per-vertex array.
    pub fn fundamental_period_s(points: &RodPoints, ei: f32) -> f32 {
        let n = points.len();
        if n < 2 {
            return 0.0;
        }
        let length_m: f32 = points.rest_edge_length.iter().sum();
        let total_mass_kg: f32 = points.mass.iter().sum();
        if length_m <= 0.0 || total_mass_kg <= 0.0 || ei <= 0.0 {
            return 0.0;
        }
        let mu = total_mass_kg / length_m;
        const BETA_L: f32 = 1.8751; // same real constant used throughout this module
        let omega = (BETA_L * BETA_L) * (ei / (mu * length_m.powi(4))).sqrt();
        if omega <= 0.0 {
            return 0.0;
        }
        std::f32::consts::TAU / omega
    }

    /// Euler/Greenhill self-weight buckling critical height
    /// (`h_crit = (7.8373*EI/(mu*g))^(1/3)`, see `spacetime::rod`'s own
    /// top-level doc for the "direct continuity with the Greenhill
    /// self-buckling analysis" motivating this solver) -- `mu` here is the
    /// SAME linear mass density (kg/m) used everywhere else in this module.
    ///
    /// Closed-form result for a UNIFORM column (one `ei` for the whole
    /// height) -- a rod with per-vertex `RodPoints::ei` has no single exact
    /// non-uniform generalization here; `Rod::buckling_warning` covers that
    /// real case by calling this with the rod's own WEAKEST `ei` (the
    /// conservative bound — a non-uniform rod buckles first at its most
    /// slender point), not by extending this function itself.
    pub fn greenhill_critical_height_m(
        ei: f32,
        linear_density_kg_per_m: f32,
        gravity_m_s2: f32,
    ) -> f32 {
        if ei <= 0.0 || linear_density_kg_per_m <= 0.0 || gravity_m_s2 <= 0.0 {
            return f32::INFINITY; // no real self-weight buckling risk without all three real inputs
        }
        (7.8373 * ei / (linear_density_kg_per_m * gravity_m_s2)).powf(1.0 / 3.0)
    }
}

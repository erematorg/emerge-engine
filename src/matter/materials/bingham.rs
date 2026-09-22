use glam::{Mat2, Vec2};

use crate::materials::physical_props::{BinghamProps, FromSI, scale_stress, scale_visc};
use crate::materials::svd::svd2;
use crate::materials::utils::{
    LOG_CLAMP, MIN_J, deformation_increment_exp, elastic_wave_dt, hencky_strains, reconstruct_f,
};
use crate::materials::{ConstitutiveModel, MaterialModel, MaterialParams};
use crate::particle::{Particle, ParticleUpdateCtx, Particles};

/// Bingham viscoplastic fluid.
///
/// Below yield stress τ₀: rigid plug (no deviatoric flow).
/// Above yield stress τ₀: Newtonian with apparent viscosity η_app = τ₀/γ̇ + η.
///
/// Stress decomposition: σ = −p·I + τ_deviatoric
/// Pressure: Tait EOS -- p = k·((ρ/ρ₀)^γ − 1), same as NewtonianFluid.
/// Deviatoric:
///   γ̇ = √(2·D_dev:D_dev)   (scalar shear rate, D_dev = deviatoric part of D)
///   τ = (τ₀/γ̇ + η)·D_dev   if γ̇ > critical_shear_rate, else 0
///
/// Reference: Bingham 1916. MPM formulation: GeoTaichi BinghamModel (Taichi lang).
///
/// # Natural phenomena
/// - Mud / wet clay: τ₀ = 50–500 Pa, η = 0.1–5 Pa·s
/// - Lava (basaltic): τ₀ = 100–2000 Pa, η = 10–10000 Pa·s
/// - Biological cytoplasm: τ₀ ≈ 0.5–5 Pa, η ≈ 0.005–0.05 Pa·s
/// - Dense biological fluids (mucus, blood clot): τ₀ = 1–50 Pa, η = 0.01–1 Pa·s
#[derive(Debug, Clone, Copy)]
pub struct BinghamFluidMaterial {
    pub rest_density: f32,
    /// Dynamic viscosity η (Pa·s) -- slope of stress-rate curve above yield.
    pub dynamic_viscosity: f32,
    /// Tait EOS stiffness k (pressure scale factor).
    pub eos_stiffness: f32,
    /// Tait EOS exponent γ (7 for water-like, 1 for linear).
    pub eos_power: f32,
    /// Yield stress τ₀ -- shear stress required to initiate flow.
    /// Below this, deviatoric stress is zero (plug flow).
    pub yield_stress: f32,
    /// Minimum shear rate to avoid τ₀/γ̇ singularity.
    /// Particles below this rate are treated as rigid. Default: 1e-4.
    pub critical_shear_rate: f32,
    /// Lower bound on the Tait EOS pressure; negative values permit limited
    /// isotropic tension. Default: 0.0 (no tensile pressure).
    ///
    /// HONEST DISCLOSURE (audit 2026-08-15): 0.0 is a free numerical parameter,
    /// chosen empirically for this engine, not a physical consequence of Bingham
    /// rheology and not a literature-calibrated value for mud or another yield-
    /// stress material. The classical model separates total stress as
    /// `sigma_total = -p I + sigma_dev` and applies the Bingham yield condition
    /// to `sigma_dev` (Roquet & Saramito, J. Non-Newtonian Fluid Mech. 155,
    /// 2008, Eqs. 1-2, doi:10.1016/j.jnnfm.2007.12.008); it does not prescribe a
    /// tensile-pressure cutoff. Unlike `NewtonianFluidMaterial`'s `-0.1`, this
    /// default does not trace to the `tmp/sparkl` or `tmp/incremental_mpm`
    /// precedents. Keep it labelled as engine calibration unless tensile data
    /// for a specific viscoplastic material supplies a real value -- the same
    /// honesty convention used for `FORAGING_RECOVERY_RATE` elsewhere.
    pub pressure_floor: f32,
    pub min_density: f32,
    pub min_volume: f32,
    /// Surface tension coefficient γ -- adds γ·J·I to Kirchhoff stress.
    /// See `NewtonianFluidMaterial::surface_tension_coeff` for details.
    pub surface_tension_coeff: f32,
    /// Per-step velocity decay: v *= (1 − settling_damping · dt).
    /// Same as `NewtonianFluidMaterial::settling_damping`. 0.0 = off.
    pub settling_damping: f32,
    /// Physical second (bulk) viscosity ζ, same real term as
    /// `NewtonianFluidMaterial::bulk_viscosity` (see that field's own doc --
    /// Litovitz & Davis, ~3x shear viscosity for water-like liquids). NOT
    /// part of the original pre-`cac544b` file this struct was otherwise
    /// restored from (2026-08-13) -- added back here because it's a real,
    /// separately-sourced fix from later the same investigation
    /// (2026-08-12, acoustic/volumetric oscillation damping), not a
    /// guard-rail bandaid, and dropping it would lose real value.
    pub bulk_viscosity: f32,
    /// Elastic shear modulus G, the storage modulus a real yield-stress
    /// fluid has BELOW its yield point. `0.0` (the default) selects the
    /// classical purely-viscous Bingham fluid; any positive value selects
    /// the elastoviscoplastic form (Saramito 2007). See this type's own
    /// doc for what each branch can and cannot reproduce.
    ///
    /// This is a measurable quantity, not a numerical knob: oscillatory
    /// rheometry reports it as G' for exactly these materials (order 10^2
    /// Pa for sauces, 10^4 Pa for toothpaste-stiff pastes and fresh
    /// concrete). The yield strain it implies, tau_0/G, lands in the 1-10%
    /// band real yield-stress fluids are measured to have.
    pub shear_modulus: f32,
    /// Specific heat capacity `c_p`, J/(kg*K). 0 (the default) means
    /// undeclared -- see `MaterialModel::specific_heat_j_kg_k`. Same field,
    /// same convention and same post-construction assignment as
    /// `NewtonianFluidMaterial::specific_heat_j_kg_k`.
    pub specific_heat_j_kg_k: f32,
    /// Measured optical coefficients for this material, or `None` when the
    /// caller has not supplied any.
    ///
    /// Deliberately NOT a substance flag. A material model never decides
    /// that it "is water" because its numbers happen to look like water's;
    /// it carries whatever measured absorption and scattering the caller
    /// declares, and a name for the result is a label applied on top, never
    /// a branch inside the physics. `matter::materials::optical` holds the
    /// measured datasets to fill this with (`pure_water`, `dry_quartz_sand`,
    /// ...); anything else measured is equally valid here.
    ///
    /// `None` is the honest default: no spectrum was measured, so the
    /// renderer is told nothing rather than being handed an invented one.
    pub optics: Option<crate::energy::radiation::OpticalCoefficientsSi>,
}

impl BinghamFluidMaterial {
    /// Construct directly from grid-native parameters -- NOT SI units.
    /// Prefer [`Self::low_yield`]/[`Self::medium_yield`]/[`Self::high_yield`]
    /// for a real mud/viscoplastic-fluid preset.
    pub const fn new(
        rest_density: f32,
        dynamic_viscosity: f32,
        eos_stiffness: f32,
        eos_power: f32,
        yield_stress: f32,
    ) -> Self {
        Self {
            rest_density,
            dynamic_viscosity,
            eos_stiffness,
            eos_power,
            yield_stress,
            critical_shear_rate: 1.0e-4,
            pressure_floor: 0.0,
            min_density: 1.0e-6,
            min_volume: 1.0e-6,
            surface_tension_coeff: 0.0,
            settling_damping: 0.0,
            bulk_viscosity: 0.0,
            shear_modulus: 0.0,
            specific_heat_j_kg_k: 0.0,
            optics: None,
        }
    }

    /// High yield stress, low viscosity: τ₀=100 Pa, η=0.5 Pa·s. Wet mud regime.
    pub fn high_yield(rest_density: f32, eos_stiffness: f32) -> Self {
        Self::new(rest_density, 0.5, eos_stiffness, 7.0, 100.0)
    }

    /// High yield stress, high viscosity: τ₀=1000 Pa, η=500 Pa·s. Basaltic lava regime.
    pub fn viscous_high_yield(rest_density: f32, eos_stiffness: f32) -> Self {
        Self::new(rest_density, 500.0, eos_stiffness, 7.0, 1000.0)
    }

    /// Low yield stress, low viscosity: τ₀=1 Pa, η=0.01 Pa·s. Biological cytoplasm regime.
    pub fn low_yield(rest_density: f32, eos_stiffness: f32) -> Self {
        Self::new(rest_density, 0.01, eos_stiffness, 7.0, 1.0)
    }

    /// Medium yield stress, medium viscosity: τ₀=10 Pa, η=0.1 Pa·s. Dense biological fluid regime.
    pub fn medium_yield(rest_density: f32, eos_stiffness: f32) -> Self {
        Self::new(rest_density, 0.1, eos_stiffness, 7.0, 10.0)
    }

    /// Compute deviatoric Bingham stress from the APIC velocity gradient C.
    ///
    /// D = (C + Cᵀ)/2 (symmetric strain rate)
    /// γ̇ = √(2·D_dev:D_dev) (scalar shear rate -- deviatoric only: a yield criterion
    /// must not respond to pure volumetric expansion/compression, which isn't shear)
    /// Below yield: returns zero matrix (plug flow).
    /// Above yield: returns (τ₀/γ̇ + η)·D_dev.
    /// Deviatoric Kirchhoff stress of the elastic branch, from the stored
    /// elastic strain rather than from the current rate.
    ///
    /// Hencky (logarithmic) elasticity: `tau_dev = U diag(2 G dev(eps)) U^T`
    /// with `eps = log(sigma)` the principal log-strains of `F`. Written in
    /// the same principal basis the return mapping uses, so the yield check
    /// and the stress it limits are the same quantity expressed once.
    fn hencky_deviatoric_stress(&self, f: Mat2) -> Mat2 {
        if self.shear_modulus <= 0.0 {
            return Mat2::ZERO;
        }
        let (u, sigma, _) = svd2(f);
        let eps = hencky_strains(sigma);
        let dev = eps - Vec2::splat((eps.x + eps.y) * 0.5);
        u * Mat2::from_diagonal(2.0 * self.shear_modulus * dev) * u.transpose()
    }

    /// Equivalent deviatoric stress the yield criterion is written against,
    /// in the Frobenius measure this file's return mapping works in.
    ///
    /// Rheology states the Bingham criterion on the second invariant,
    /// `sqrt(tau_dev : tau_dev / 2) > tau_0`, while the principal-strain
    /// return mapping below works with `||tau_dev||`. The two differ by
    /// exactly `sqrt(2)`, so `yield_stress` keeps its plain rheological
    /// meaning and the conversion lives here, once.
    fn yield_in_frobenius_measure(&self) -> f32 {
        std::f32::consts::SQRT_2 * self.yield_stress
    }

    fn deviatoric_stress(&self, c: Mat2) -> Mat2 {
        // Symmetric strain rate D = (C + Cᵀ) / 2
        let sym = c + c.transpose();
        let d = sym * 0.5;

        // Deviatoric: remove isotropic part
        let trace = d.x_axis.x + d.y_axis.y;
        let d_dev = d - Mat2::from_diagonal(Vec2::splat(trace * 0.5));

        // Scalar shear rate γ̇ = √(2·D_dev:D_dev) -- Frobenius norm of deviatoric D, scaled.
        let d_xx = d_dev.x_axis.x;
        let d_yy = d_dev.y_axis.y;
        let d_xy = d_dev.x_axis.y; // = d_dev.y_axis.x for symmetric D
        let d_sq = d_xx * d_xx + d_yy * d_yy + 2.0 * d_xy * d_xy;
        let shear_rate = (2.0 * d_sq).sqrt();

        // Bi-viscosity regularization (O'Donovan & Tanner 1984; Beverly &
        // Tanner 1989): below the critical rate the apparent viscosity
        // SATURATES at tau_0/gamma_dot_c. This used to `return Mat2::ZERO`
        // instead, which is a different and wrong statement -- it discards
        // the yield stress at exactly the rates where a yield stress is the
        // only thing holding the material up, so a slow or resting body
        // reverted to an inviscid liquid. Confirmed against the same model
        // in `tmp/GeoTaichi/.../Bingham.py`, whose own cutoff is 1e-8, four
        // orders finer than the 1e-4 default here, and which gates on the
        // stress invariant rather than nulling the stress outright.
        //
        // Real, disclosed cost, not hidden: the saturated viscosity is what
        // `timestep_bound` must now respect, and for a large tau_0 that is
        // genuinely expensive. That cost is the reason the elastoviscoplastic
        // branch (`shear_modulus > 0`) exists and is the recommended route
        // for a material meant to hold its shape at rest.
        let regularized_rate = shear_rate.max(self.critical_shear_rate);

        // Apparent viscosity: Bingham formula η_app = τ₀/γ̇ + η
        let eta_app = self.yield_stress / regularized_rate + self.dynamic_viscosity;
        // Real, disclosed regression fix (external review): with D=(C+Cᵀ)/2
        // (the symmetric strain rate built above), the real tensorial
        // Bingham law is tau_dev = 2*eta_app*D_dev, not eta_app*D_dev --
        // the codebase's own real, cited source (Balmforth, Frigaard &
        // Ovarlez) writes the law in terms of the FULL shear-rate tensor
        // Delta = grad(v)+grad(v)^T = 2*D, which is exactly the factor of 2
        // missing here. Pre-fix, simple shear gave exactly HALF the real
        // stress (tau_xy = (tau_y+eta*gamma_dot)/2 instead of
        // tau_y+eta*gamma_dot).
        2.0 * eta_app * d_dev
    }
}

impl FromSI<BinghamProps> for BinghamFluidMaterial {
    fn from_physical(props: &BinghamProps, config: &crate::SimConfig) -> Self {
        // Tait EOS polytropic exponent -- Cole 1948, "Underwater Explosions"; standard
        // in SPH/MPM weakly-compressible fluid solvers (Monaghan 1994). Applies to the
        // volumetric/EOS part of a Bingham fluid same as any other weakly-compressible
        // liquid; the yield-stress physics (tau0 below) is separate and unaffected.
        const GAMMA: f32 = 7.0;
        let visc = scale_visc(props.eta_pa_s, props.rho_kg_m3, config);
        let tau0 = scale_stress(props.yield_stress_pa, props.rho_kg_m3, config);
        // Same conversion as `tau0`, and that is not a stylistic choice:
        // the return mapping compares `2 G |dev|` against `tau_0` directly,
        // so a modulus converted through a different family would compare
        // two different units and silently move the yield point.
        let shear_modulus = scale_stress(props.shear_modulus_pa, props.rho_kg_m3, config);
        let eos = scale_stress(props.bulk_modulus_pa / GAMMA, props.rho_kg_m3, config);
        // Real bug fix: this was `rho_kg_m3 * dx^2`, precisely the extra
        // `dx_meters^2` factor `NewtonianFluidMaterial::from_physical`'s own
        // doc says in as many words not to reintroduce. Both `scale_visc`
        // and `scale_stress` already divide by `rho * dx^2` (they return a
        // kinematic viscosity and a squared wave speed, both in cells), so
        // dividing by this density a second time inside `timestep_bound`
        // scaled the acoustic and viscous CFL bounds by `1/(rho * dx^2)`.
        // At 2 mm cells that is 250x, which is why a millimetre-scale
        // Bingham scene could not advance a 5 ms step inside 64 substeps.
        // The density the solver actually measures is a RATIO against the
        // scene reference density -- 1.0 for a fluid at that reference --
        // exactly as the Newtonian twin computes it.
        let rho_grid = props.rho_kg_m3 / config.reference_density_kg_m3;
        let mut material = Self::new(rho_grid, visc, eos, GAMMA, tau0);
        material.shear_modulus = shear_modulus;
        material
    }
}

impl BinghamFluidMaterial {
    /// Elastoviscoplastic update: elastic below the yield stress, Bingham
    /// flow above it (Saramito, *A new constitutive equation for
    /// elastoviscoplastic fluid flows*, J. Non-Newtonian Fluid Mech. 145,
    /// 2007), discretised as a von Mises radial return with a Perzyna
    /// viscous overstress (Perzyna 1966) -- structurally the same return
    /// mapping `VonMisesMaterial::update_particle` already uses and tests,
    /// with the hardening term replaced by the viscous one.
    ///
    /// Why this branch exists at all: a purely viscous Bingham fluid
    /// computes its deviatoric stress from the CURRENT rate of strain, so
    /// at rest it has none and cannot hold a slope, a pile or a ridge. That
    /// is a property of the model, not of this implementation -- the same
    /// limitation is visible in the reference implementation vendored at
    /// `tmp/GeoTaichi`. Holding a shape requires storing elastic shear
    /// energy, which is exactly what the elastic branch adds.
    ///
    /// The plastic multiplier comes from requiring the returned stress to
    /// sit on the Bingham line rather than on the dry yield surface:
    ///
    /// ```text
    ///   ||s_trial|| - 2 G dgamma = sqrt(2) tau_0 + 2 eta dgamma / dt
    ///   dgamma = (||s_trial|| - sqrt(2) tau_0) / (2 G + 2 eta / dt)
    /// ```
    ///
    /// `eta = 0` recovers perfect plasticity at exactly `tau_0`; a large
    /// `eta` recovers a purely elastic solid; steady simple shear recovers
    /// `tau = tau_0 + eta * gamma_dot`. Because `dt` appears in the
    /// denominator the viscous term is integrated implicitly, so unlike the
    /// purely viscous branch it imposes no viscous timestep restriction of
    /// its own.
    fn update_elastoviscoplastic(&self, ctx: &mut ParticleUpdateCtx, dt: f32) {
        let f_trial =
            deformation_increment_exp(dt * *ctx.velocity_gradient) * *ctx.deformation_gradient;
        let (u, sigma, vt) = svd2(f_trial);

        let eps = hencky_strains(sigma);
        let tr = eps.x + eps.y;
        let dev = eps - Vec2::splat(tr * 0.5);
        let dev_norm = dev.length();

        let trial_stress = 2.0 * self.shear_modulus * dev_norm;
        let yield_frobenius = self.yield_in_frobenius_measure();

        let sigma_new = if trial_stress > yield_frobenius && dev_norm > LOG_CLAMP {
            // `MIN_RETURN_DT` guards the division only; it is a floor on
            // time, so it must not borrow `LOG_CLAMP`, which is a floor on
            // strain. A substep this short never occurs in practice -- the
            // config's own `min_dt` is orders above it.
            const MIN_RETURN_DT: f32 = 1.0e-12;
            let denom =
                2.0 * self.shear_modulus + 2.0 * self.dynamic_viscosity / dt.max(MIN_RETURN_DT);
            let dgamma = (trial_stress - yield_frobenius) / denom;
            // Never relax past the yield surface: with a finite viscosity
            // the returned stress stays above `tau_0`, so a plastic step
            // that overshot it would mean the increment, not the material,
            // drove the sign.
            let relaxed = (trial_stress - 2.0 * self.shear_modulus * dgamma).max(yield_frobenius);
            let eps_projected = dev * (relaxed / trial_stress) + Vec2::splat(tr * 0.5);
            Vec2::new(eps_projected.x.exp(), eps_projected.y.exp())
        } else {
            sigma
        };

        *ctx.deformation_gradient = reconstruct_f(u, sigma_new, vt);
        if self.settling_damping > 0.0 {
            *ctx.v *= 1.0 - (self.settling_damping * dt).min(0.5);
        }
        let j = ctx.deformation_gradient.determinant().max(MIN_J);
        let volume = (ctx.initial_volume * j).max(1.0e-9);
        *ctx.volume = volume;
        *ctx.density = ctx.mass / volume;
    }
}

impl MaterialModel for BinghamFluidMaterial {
    fn constitutive_model(&self) -> ConstitutiveModel {
        ConstitutiveModel::Fluid
    }

    // Restored 2026-08-13; PERMANENT and required -- same
    // reasoning as `NewtonianFluidMaterial::init_particle`, see that
    // method's own doc for the full live-confirmed root cause.
    fn init_particle(&self, particle: &mut Particle) {
        let j = particle.deformation_gradient.determinant();
        particle.initial_volume = particle.mass / self.rest_density;
        particle.volume = particle.initial_volume * j;
        particle.density = self.rest_density / j;
    }

    /// Same restoration and reasoning as
    /// `NewtonianFluidMaterial::rest_acoustic_c2` -- this material shares the
    /// identical Tait EOS, so it needs the identical rest sound speed.
    fn rest_acoustic_c2(&self) -> Option<f32> {
        if self.eos_stiffness > 0.0 && self.rest_density > 0.0 {
            Some(self.eos_stiffness * self.eos_power / self.rest_density)
        } else {
            None
        }
    }

    fn kirchhoff_stress(&self, particles: &Particles, i: usize) -> Mat2 {
        // Pressure from Tait EOS (same as NewtonianFluid). Clamp density both
        // ways, matching `NewtonianFluidMaterial::kirchhoff_stress` exactly:
        // min prevents div-by-zero at low PPC, max (2x rho0) limits how far
        // the EOS pressure response saturates under impact overcompression.
        // Matches `NewtonianFluidMaterial::kirchhoff_stress`'s clamp exactly --
        // same Tait EOS, same unbounded-pressure-spike risk under violent
        // compression; see that file's doc for why 2x specifically.
        let density = particles.density[i]
            .max(self.min_density)
            .min(self.rest_density * 2.0);
        let pressure = (self.eos_stiffness
            * ((density / self.rest_density).powf(self.eos_power) - 1.0))
            .max(self.pressure_floor);

        let hydrostatic = Mat2::from_diagonal(Vec2::splat(-pressure));
        let deviatoric = if self.shear_modulus > 0.0 {
            // Elastoviscoplastic branch: the deviatoric stress comes from
            // STORED elastic strain, so it survives when the material stops
            // moving -- which is the whole point of a yield stress. The
            // viscous part lives inside the return mapping in
            // `update_particle`; adding it again here would count the same
            // dissipation twice.
            self.hencky_deviatoric_stress(particles.deformation_gradient[i])
        } else {
            self.deviatoric_stress(particles.velocity_gradient[i])
        };

        // Surface tension: τ += γ·J·I
        let surface = if self.surface_tension_coeff != 0.0 {
            let f = particles.deformation_gradient[i];
            let j = f.x_axis.x * f.y_axis.y - f.x_axis.y * f.y_axis.x;
            Mat2::from_diagonal(Vec2::splat(self.surface_tension_coeff * j))
        } else {
            Mat2::ZERO
        };

        // Bulk viscosity: τ += ζ·div(v)·I -- see field's own doc. Real,
        // disclosed regression fix (external review): this used to read
        // `div_v` from the RAW velocity gradient (`trace(C)`, already the
        // real divergence with no extra factor), then multiplied by an
        // extra `*0.5` on top -- a genuine 2x-too-small bulk viscosity
        // here specifically. `NewtonianFluidMaterial`/`IdealGasMaterial`/
        // `GranularFluidMaterial` all compute `div_v` from the SYMMETRIZED
        // strain (`C+Cᵀ`, i.e. `2*trace(C)`) instead, where the same `*0.5`
        // correctly recovers the real divergence -- matches the GPU
        // shader's own convention (`p2g.wgsl`'s `tr_s*0.5`) exactly. Fixed
        // by using the SAME symmetrized-strain convention here, not by
        // dropping the `*0.5` (which would have been correct ONLY for this
        // file's own now-removed raw-gradient basis, not the shared one).
        let bulk = if self.bulk_viscosity > 0.0 {
            let gradient = particles.velocity_gradient[i];
            let sym_strain = gradient + gradient.transpose();
            let div_v = sym_strain.x_axis.x + sym_strain.y_axis.y;
            Mat2::from_diagonal(Vec2::splat(self.bulk_viscosity * div_v * 0.5))
        } else {
            Mat2::ZERO
        };

        // Artificial (shock) viscosity -- same real PDE term, same citation,
        // as `NewtonianFluidMaterial::kirchhoff_stress` (see
        // `artificial_bulk_viscosity`'s own doc). This material shares the
        // identical Tait EOS, so it needs the identical shock-capturing term;
        // it was lost from the CPU path by the same wholesale revert.
        let gradient = particles.velocity_gradient[i];
        let div_v_true = gradient.x_axis.x + gradient.y_axis.y;
        let j_now = crate::materials::fluid::volume_j(
            particles.initial_volume[i],
            particles.volume[i],
            "BinghamFluidMaterial",
        );
        let q = crate::materials::fluid::artificial_bulk_viscosity(
            self.eos_stiffness,
            self.eos_power,
            self.rest_density,
            j_now,
            div_v_true,
            1.0,
        );
        let shock = Mat2::from_diagonal(Vec2::splat(-q));

        hydrostatic + deviatoric + surface + bulk + shock
    }

    fn stress_volume(&self, particles: &Particles, i: usize) -> f32 {
        particles.volume[i].max(self.min_volume)
    }

    fn update_particle(&self, ctx: &mut ParticleUpdateCtx, dt: f32) {
        // REAL BUG, found+fixed for real 2026-08-13: this material's comment
        // already said it "carries the identical dead-EOS-pressure bug" that
        // `NewtonianFluidMaterial::update_particle` was fixed for 2026-08-06
        // -- but the fix itself was never actually ported here, only noted
        // as still-outstanding. Live-confirmed consequence tonight: mud's
        // `deformation_gradient` updates normally (J drifts, e.g. 0.9989)
        // but `volume`/`density` never move from their spawn values (stuck
        // at exactly V/V0=1 forever) -- an internally inconsistent state
        // that trips `assert_owned_deformation_state`
        // ("det(F)=0.9989268, V/V0=1"). This is the user's own live
        // observation exactly: water settles correctly, mud does not.
        // Real, disclosed regression fixed 2026-08-30 -- same fix, same
        // root cause, as `NewtonianFluidMaterial::update_particle`'s own
        // doc: `det(I+dt*C)` is not rotation-invariant (a pure rigid
        // rotation should leave J exactly unchanged but this formula gives
        // a strictly positive O(dt^2) expansion every substep, baked in
        // permanently by isotropization). Fixed with the continuity
        // equation's own exact exponential solution, `J_{n+1}=J_n*exp(dt*
        // div(v))` -- restores the pre-`57b83dc` `fluid_state.rs` behavior.
        if self.shear_modulus > 0.0 {
            self.update_elastoviscoplastic(ctx, dt);
            return;
        }
        let old_j = ctx.deformation_gradient.determinant();
        let div_v = ctx.velocity_gradient.x_axis.x + ctx.velocity_gradient.y_axis.y;
        let j = (old_j * (dt * div_v).exp()).clamp(0.5, 2.0);
        let s = j.sqrt();
        *ctx.deformation_gradient =
            glam::Mat2::from_cols(glam::Vec2::new(s, 0.0), glam::Vec2::new(0.0, s));
        if self.settling_damping > 0.0 {
            *ctx.v *= 1.0 - (self.settling_damping * dt).min(0.5);
        }
        let density = (self.rest_density / j)
            .max(self.min_density)
            .min(self.rest_density * 2.0);
        *ctx.density = density;
        *ctx.volume = (ctx.mass / density).max(1.0e-9);
    }

    // Restored 2026-08-13; PERMANENT and required -- same
    // reasoning as `NewtonianFluidMaterial::owns_deformation_volume_state`,
    // see that method's own doc for the full live-confirmed root cause.
    /// True only for the purely viscous branch. The elastoviscoplastic
    /// branch keeps a real deviatoric elastic strain in `F`, so it is not a
    /// strict WC-MPM liquid any more and must not claim to be one: that
    /// claim is what `Simulation::assert_strict_fluid_mode_is_supported`
    /// keys off, and it is also the licence this material used to take to
    /// isotropise `F` every substep -- which is precisely what destroyed
    /// the stored shear strain a yield stress needs.
    fn owns_deformation_volume_state(&self) -> bool {
        self.shear_modulus <= 0.0
    }

    fn specific_heat_j_kg_k(&self) -> f32 {
        self.specific_heat_j_kg_k
    }

    /// Whatever the caller measured, verbatim. See the `optics` field: this
    /// model reports coefficients, it does not identify a substance.
    fn optical_properties(&self) -> Option<crate::energy::radiation::OpticalCoefficientsSi> {
        self.optics
    }

    /// The elastoviscoplastic branch has no WGSL counterpart: `params()`
    /// uploads `ConstitutiveModel::Fluid`, and `p2g.wgsl`'s fluid arm reads
    /// the rate of strain, not the stored elastic strain this branch keeps
    /// in `F`. On the GPU the CPU update does not rescue it: it runs once
    /// per frame with a single substep's dt, and re-uploading its copy makes
    /// the GPU simulation advance at half speed (a free fall reaches -49.0
    /// instead of -98.1 cells/s after 0.1 s). The stress feeding the
    /// transfer is still the fluid one. This branch is CPU-only until the
    /// shader gains a matching arm.
    fn needs_cpu_update(&self) -> bool {
        self.shear_modulus > 0.0
    }

    fn gpu_unsupported_reason(&self) -> Option<&'static str> {
        (self.shear_modulus > 0.0).then_some(
            "BinghamFluidMaterial with a shear modulus (the elastoviscoplastic branch) has no GPU stress path: the GPU runs the viscous fluid law and the per-frame CPU update advances only half the simulated time",
        )
    }

    fn params(&self) -> MaterialParams {
        MaterialParams {
            model: ConstitutiveModel::Fluid as u32,
            rest_density: self.rest_density,
            eos_stiffness: self.eos_stiffness,
            eos_power: self.eos_power,
            dynamic_viscosity: self.dynamic_viscosity,
            compression_limit: self.yield_stress,
            volume_ratio_max: 2.0,
            pressure_floor: self.pressure_floor,
            dp_h0: self.settling_damping,
            bulk_viscosity: self.bulk_viscosity,
            mu: self.shear_modulus,
            specific_heat_j_kg_k: self.specific_heat_j_kg_k,
            owns_deformation_volume_state: self.owns_deformation_volume_state() as u32,
            ..Default::default()
        }
    }

    fn timestep_bound(
        &self,
        density: f32,
        _hardening_scale: f32,
        cell_width: f32,
        material_cfl: f32,
        viscous_cfl: f32,
    ) -> f32 {
        let density = density.max(self.min_density);
        let ratio = (density / self.rest_density.max(self.min_density)).max(1.0e-6);

        let mut dt_bound = f32::INFINITY;

        // Acoustic bound from EOS
        let c2 = self.eos_stiffness * self.eos_power * ratio.powf(self.eos_power - 1.0)
            / self.rest_density.max(self.min_density);
        if c2.is_finite() && c2 > f32::EPSILON {
            dt_bound = dt_bound.min(material_cfl * cell_width / c2.sqrt());
        }

        // Elastic shear-wave bound, elastoviscoplastic branch only. Cheap
        // next to the acoustic term for any realistic storage modulus
        // (water-based pastes sit near G = 1e2..1e4 Pa against a derated
        // bulk modulus orders above that), which is what makes holding a
        // shape affordable here and not in the purely viscous branch.
        if self.shear_modulus > 0.0 {
            dt_bound = dt_bound.min(elastic_wave_dt(
                0.0,
                self.shear_modulus,
                1.0,
                density,
                self.min_density,
                cell_width,
                material_cfl,
            ));
        }

        // Viscous diffusion bound -- apparent viscosity is at least
        // dynamic_viscosity, combined with bulk_viscosity (real regression
        // fix, external review: same gap, same fix, as
        // `NewtonianFluidMaterial::timestep_bound` -- see that file's own
        // doc for the full explicit-integrator-instability reasoning this
        // closes).
        //
        // Real gap closed with the regularization fix in
        // `deviatoric_stress`: the purely viscous branch's apparent
        // viscosity SATURATES at `tau_0/gamma_dot_c + eta`, which can be
        // orders above `eta` alone, and this bound ignored it completely --
        // so the explicit step was never actually CFL-safe for the yield
        // term it was integrating. The elastoviscoplastic branch is exempt
        // because there the viscous term is solved implicitly inside the
        // return mapping, so it cannot destabilise the step.
        // `dynamic_viscosity` is dropped here for the elastoviscoplastic
        // branch, and that is not an oversight: there it appears ONLY in
        // the return mapping's denominator, i.e. integrated implicitly, so
        // it cannot destabilise an explicit step and must not be allowed to
        // shrink one. `bulk_viscosity` stays in either branch -- it is
        // applied explicitly in `kirchhoff_stress` regardless.
        let explicit_shear_viscosity = if self.shear_modulus > 0.0 {
            0.0
        } else if self.critical_shear_rate > 0.0 {
            self.dynamic_viscosity + self.yield_stress.max(0.0) / self.critical_shear_rate
        } else {
            self.dynamic_viscosity
        };
        let combined_viscosity = explicit_shear_viscosity + self.bulk_viscosity.max(0.0);
        if combined_viscosity > 0.0 {
            let kinematic_viscosity = combined_viscosity / density;
            if kinematic_viscosity > f32::EPSILON {
                dt_bound =
                    dt_bound.min(viscous_cfl * cell_width * cell_width / kinematic_viscosity);
            }
        }

        dt_bound
    }

    /// `false` -- same reasoning and same 2026-08-13 fix as
    /// `NewtonianFluidMaterial::needs_density_recompute`; see that method's
    /// own doc. This material owns `rho = rho0/J` through `init_particle` /
    /// `update_particle` and declares it via `owns_deformation_volume_state`,
    /// so the kernel-density gather was computed for it and then discarded.
    fn needs_density_recompute(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod analytical_validation_tests {
    use super::*;

    /// **Deviatoric stress must match the Bingham formula exactly** (Bingham
    /// 1916: below yield, rigid plug/zero stress; above yield, Newtonian with
    /// apparent viscosity `tau0/shear_rate + eta`). `BinghamFluidMaterial` had
    /// zero test comparing its deviatoric response to this analytical formula
    /// directly (only whole-simulation stability checks existed).
    #[test]
    fn below_critical_shear_rate_saturates_instead_of_discarding_the_yield_stress() {
        let mat = BinghamFluidMaterial::new(1000.0, 0.5, 5000.0, 7.0, 100.0);
        // A tiny, sub-critical shear rate: pure shear C with a small magnitude.
        let rate = 1.0e-6;
        let c = Mat2::from_cols(Vec2::new(0.0, rate), Vec2::new(rate, 0.0));
        let tau = mat.deviatoric_stress(c);

        // This used to assert exactly `Mat2::ZERO`, encoding the very
        // shortcut that made a resting Bingham fluid behave like an
        // inviscid liquid. The regularized law saturates the apparent
        // viscosity at `tau_0/gamma_dot_c + eta` instead, so the stress
        // stays proportional to the rate with that capped coefficient --
        // small, but present and pointing the right way.
        let expected_eta = mat.yield_stress / mat.critical_shear_rate + mat.dynamic_viscosity;
        // tau = 2 * eta_app * D_dev, and for this C the deviatoric strain
        // rate has off-diagonal `rate` and zero trace.
        let expected_xy = 2.0 * expected_eta * rate;
        assert!(
            (tau.x_axis.y - expected_xy).abs() < 1.0e-4 * expected_xy.abs().max(1.0),
            "sub-critical stress must follow the saturated viscosity: got {} expected {expected_xy}",
            tau.x_axis.y
        );
        assert!(
            tau.x_axis.y > 0.0,
            "a yield-stress fluid must still resist at a sub-critical rate"
        );
    }

    #[test]
    fn above_yield_matches_bingham_formula_exactly() {
        let mat = BinghamFluidMaterial::new(1000.0, 0.5, 5000.0, 7.0, 100.0);
        // Pure shear strain rate: C = [[0, g], [g, 0]] gives D=C (already symmetric),
        // D_dev=D (already traceless), d_xx=d_yy=0, d_xy=g, d_sq=2*g^2,
        // shear_rate=sqrt(2*2*g^2)=2*g (real, hand-derivable from the formula).
        let g = 5.0_f32;
        let c = Mat2::from_cols(Vec2::new(0.0, g), Vec2::new(g, 0.0));
        let tau = mat.deviatoric_stress(c);

        let shear_rate = 2.0 * g;
        let eta_app = mat.yield_stress / shear_rate + mat.dynamic_viscosity;
        let d_dev = Mat2::from_cols(Vec2::new(0.0, g), Vec2::new(g, 0.0)); // D_dev = D here
        // Real, disclosed fix: the real tensorial law is 2*eta_app*D_dev for
        // this D=(C+C^T)/2 convention (Balmforth, Frigaard & Ovarlez) -- see
        // `deviatoric_stress`'s own doc.
        let predicted = 2.0 * eta_app * d_dev;

        let diff = tau - predicted;
        let err = (diff.x_axis.length_squared() + diff.y_axis.length_squared()).sqrt();
        assert!(
            err < 1.0e-3,
            "above-yield deviatoric stress should match 2*(tau0/gamma_dot+eta)*D_dev exactly: \
             predicted={predicted:?} actual={tau:?}"
        );
    }

    /// Real regression guard (external review): as `yield_stress -> 0`,
    /// Bingham's own law must converge EXACTLY to plain Newtonian viscous
    /// stress (`2*eta*D_dev`) -- the same real tensorial form
    /// `NewtonianFluidMaterial`'s own bulk/shear terms use. A real,
    /// independent cross-check the factor-of-2 bug above could not have
    /// passed: pre-fix, this limit was still off by 2x.
    #[test]
    fn zero_yield_stress_limit_matches_newtonian_viscous_stress_exactly() {
        let eta = 0.5_f32;
        let mat = BinghamFluidMaterial::new(1000.0, eta, 5000.0, 7.0, 0.0);
        let g = 5.0_f32;
        let c = Mat2::from_cols(Vec2::new(0.0, g), Vec2::new(g, 0.0));
        let tau = mat.deviatoric_stress(c);

        let d_dev = c; // already symmetric and traceless for this C
        let predicted_newtonian = 2.0 * eta * d_dev;

        let diff = tau - predicted_newtonian;
        let err = (diff.x_axis.length_squared() + diff.y_axis.length_squared()).sqrt();
        assert!(
            err < 1.0e-3,
            "yield_stress=0 must reduce exactly to Newtonian viscous stress \
             2*eta*D_dev: predicted={predicted_newtonian:?} actual={tau:?}"
        );
    }

    /// Real regression guard (external review, P0 #3): bulk viscosity's own
    /// `div(v)` must be read from the SAME symmetrized-strain basis as
    /// `NewtonianFluidMaterial`'s (`sym=C+Cᵀ`, `div_v=trace(sym)`, then the
    /// `*0.5` in the stress formula recovers the true divergence) -- not
    /// from the raw gradient directly, which already IS the divergence and
    /// so must not be halved again. Isolated two ways at once: a
    /// pure-dilation `C=k*I` makes `D_dev=0` exactly (no deviatoric term to
    /// separate out), and `density=rest_density` makes the Tait EOS
    /// pressure exactly zero (no hydrostatic term either) -- density=1200
    /// was tried first and rejected: it gave a real but ~13000-magnitude
    /// pressure that swamped the ~0.1-magnitude bulk signal, so recovering
    /// it via subtraction hit f32's own ULP noise floor at that magnitude
    /// (catastrophic cancellation, not a code bug). With both zeroed,
    /// `kirchhoff_stress` reduces to the bulk term alone. Pre-fix, this
    /// measured exactly HALF of `zeta*div(v)*I`.
    #[test]
    fn bulk_viscosity_matches_real_divergence_term_exactly() {
        use crate::particle::{Particle, Particles};

        let zeta = 3.0_f32;
        let mut mat = BinghamFluidMaterial::new(1000.0, 0.5, 5000.0, 7.0, 100.0);
        mat.bulk_viscosity = zeta;

        let k = 0.02_f32; // pure dilation: C = k*I, D_dev = 0 (no shear)
        let mut p = Particle::zeroed();
        p.deformation_gradient = Mat2::IDENTITY;
        p.mass = 1.0;
        p.initial_volume = 1.0;
        p.volume = 1.0;
        p.density = 1000.0; // == rest_density -- Tait pressure is exactly zero
        p.velocity_gradient = Mat2::from_diagonal(Vec2::splat(k));
        let particles = Particles::from(vec![p]);

        let tau = mat.kirchhoff_stress(&particles, 0);

        let div_v = 2.0 * k; // trace(C) in 2D -- the real divergence
        let expected = Mat2::from_diagonal(Vec2::splat(zeta * div_v));

        let diff = tau - expected;
        let err = (diff.x_axis.length_squared() + diff.y_axis.length_squared()).sqrt();
        assert!(
            err < 1.0e-5,
            "bulk-viscosity contribution must equal zeta*div(v)*I exactly: \
             expected={expected:?} actual={tau:?}"
        );
    }

    /// Real, checkable monotonic claim: apparent viscosity (and thus deviatoric
    /// stress magnitude at a FIXED shear rate) must DECREASE as shear rate
    /// increases -- shear-thinning behavior intrinsic to the Bingham model
    /// (tau0/gamma_dot term shrinks as gamma_dot grows), not an assumption.
    #[test]
    fn apparent_viscosity_decreases_as_shear_rate_increases() {
        let mat = BinghamFluidMaterial::new(1000.0, 0.5, 5000.0, 7.0, 100.0);
        let tau_slow =
            mat.deviatoric_stress(Mat2::from_cols(Vec2::new(0.0, 1.0), Vec2::new(1.0, 0.0)));
        let tau_fast =
            mat.deviatoric_stress(Mat2::from_cols(Vec2::new(0.0, 10.0), Vec2::new(10.0, 0.0)));

        // Stress DOES grow with shear rate overall (more strain rate -> more
        // stress), but the EFFECTIVE viscosity (stress/shear_rate) must shrink --
        // check the ratio, not the raw magnitude.
        let eff_visc_slow = tau_slow.x_axis.y / 2.0; // shear_rate=2*g=2 here
        let eff_visc_fast = tau_fast.x_axis.y / 20.0; // shear_rate=2*g=20 here
        assert!(
            eff_visc_fast < eff_visc_slow,
            "apparent viscosity must decrease as shear rate increases (shear-thinning): \
             slow={eff_visc_slow:.4} fast={eff_visc_fast:.4}"
        );
    }
}

#[cfg(test)]
mod volume_integration_tests {
    use super::*;
    use crate::particle::{Particle, Particles};

    fn particle_with_f(f: Mat2) -> Particles {
        let mut p = Particle::zeroed();
        p.deformation_gradient = f;
        p.mass = 1.0;
        p.initial_volume = 1.0;
        p.volume = f.determinant();
        p.density = 1.0;
        Particles::from(vec![p])
    }

    /// Same audit-closing regression guard `NewtonianFluidMaterial` already
    /// carries for the identical mechanism (`rigid_rotation_leaves_j_
    /// exactly_unchanged`, `fluid.rs`'s own `volume_integration_tests`) --
    /// this file's own `update_particle` doc claims the SAME fix (the
    /// continuity equation's exact exponential solution,
    /// `J_new=J_old*exp(dt*div(v))`, replacing the non-rotation-invariant
    /// `det(I+dt*C)`), so it needs the same direct proof, not just a
    /// doc claim of parity with the sibling material.
    #[test]
    fn rigid_rotation_leaves_j_exactly_unchanged() {
        let mat = BinghamFluidMaterial::new(1.0, 0.0, 100.0, 7.0, 0.0);
        let mut particles = particle_with_f(Mat2::IDENTITY);
        let dt = 0.01;
        let omega = 5.0_f32; // deliberately large -- the old bug scales as omega^2
        {
            let mut ctx = particles.update_ctx(0);
            *ctx.velocity_gradient = Mat2::from_cols(Vec2::new(0.0, omega), Vec2::new(-omega, 0.0));
            for _ in 0..500 {
                mat.update_particle(&mut ctx, dt);
            }
        }
        let j = particles.deformation_gradient[0].determinant();
        assert!(
            (j - 1.0).abs() < 1.0e-5,
            "500 substeps of pure rotation (omega={omega}) must leave J exactly at 1.0, \
             got {j} -- the old det(I+dt*C) bug would give a real, measurable expansion here"
        );
    }

    /// Same audit-closing regression guard as `NewtonianFluidMaterial`'s own
    /// `constant_dilation_matches_exact_exponential_solution`: a constant,
    /// uniform dilation (`C=k*I`, `div(v)=2k` in 2D) must integrate to
    /// EXACTLY the continuity equation's own exponential solution,
    /// `J_new=J_old*exp(N*dt*2k)`, not the old per-step `det(I+dt*C)`
    /// approximation (which only agrees to first order and drifts at
    /// higher `dt*k`).
    #[test]
    fn constant_dilation_matches_exact_exponential_solution() {
        let mat = BinghamFluidMaterial::new(1.0, 0.0, 100.0, 7.0, 0.0);
        let mut particles = particle_with_f(Mat2::IDENTITY);
        let dt = 0.001;
        let k = 2.0_f32;
        const N: i32 = 50;
        {
            let mut ctx = particles.update_ctx(0);
            *ctx.velocity_gradient = Mat2::from_diagonal(Vec2::splat(k));
            for _ in 0..N {
                mat.update_particle(&mut ctx, dt);
            }
        }
        let j = particles.deformation_gradient[0].determinant();
        let expected = (N as f32 * dt * 2.0 * k).exp();
        assert!(
            (j - expected).abs() / expected < 1.0e-4,
            "constant dilation over {N} substeps must match J_old*exp(N*dt*2k) exactly \
             (continuity equation's own solution for constant C) -- got {j}, expected {expected}"
        );
    }
}

/// Does the elastoviscoplastic branch actually reproduce the Bingham line,
/// `tau = tau_0 + eta * gamma_dot`, rather than merely looking viscoplastic?
/// Driven directly through `update_particle` at a fixed shear rate until the
/// elastic strain stops growing, then read off the stress the material
/// really carries.
#[cfg(test)]
mod elastoviscoplastic_tests {
    use super::*;
    use crate::particle::Particles;

    fn material(yield_stress: f32, viscosity: f32, shear_modulus: f32) -> BinghamFluidMaterial {
        let mut m = BinghamFluidMaterial::new(1.0, viscosity, 1000.0, 7.0, yield_stress);
        m.shear_modulus = shear_modulus;
        m
    }

    /// Shear at a constant rate for long enough that the elastic strain
    /// saturates, and report the equivalent deviatoric stress in the plain
    /// rheological (second-invariant) measure the yield stress is stated in.
    fn steady_shear_stress(mat: &BinghamFluidMaterial, shear_rate: f32, dt: f32) -> f32 {
        // Simple shear: gamma_dot = sqrt(2 D_dev : D_dev) with
        // D = (C + C^T)/2, so C = [[0, gamma_dot], [0, 0]] gives exactly
        // this scalar rate.
        let c = Mat2::from_cols(Vec2::new(0.0, 0.0), Vec2::new(shear_rate, 0.0));
        let mut p = Particle::zeroed();
        p.deformation_gradient = Mat2::IDENTITY;
        p.velocity_gradient = c;
        p.mass = 1.0;
        p.initial_volume = 1.0;
        let mut particles = Particles::from(vec![p]);
        for _ in 0..20_000 {
            particles.velocity_gradient[0] = c;
            mat.update_particle(&mut particles.update_ctx(0), dt);
        }
        let tau_dev = mat.hencky_deviatoric_stress(particles.deformation_gradient[0]);
        // Second invariant: sqrt(tau_dev : tau_dev / 2).
        let frobenius_sq = tau_dev.x_axis.length_squared() + tau_dev.y_axis.length_squared();
        (0.5 * frobenius_sq).sqrt()
    }

    #[test]
    fn steady_simple_shear_lands_on_the_bingham_line() {
        let (tau0, eta) = (100.0, 2.0);
        let mat = material(tau0, eta, 5000.0);
        for shear_rate in [0.5f32, 2.0, 8.0] {
            let measured = steady_shear_stress(&mat, shear_rate, 1.0e-4);
            let predicted = tau0 + eta * shear_rate;
            let error = (measured - predicted).abs() / predicted;
            println!(
                "gamma_dot={shear_rate:>4}  measured={measured:8.3} Pa  bingham_line={predicted:8.3} Pa  err={:.2}%",
                error * 100.0
            );
            assert!(
                error < 0.05,
                "gamma_dot={shear_rate}: measured {measured:.3} Pa vs Bingham line {predicted:.3} Pa ({:.1}% off)",
                error * 100.0
            );
        }
    }

    #[test]
    fn zero_viscosity_recovers_perfect_plasticity_at_the_yield_stress() {
        let mat = material(100.0, 0.0, 5000.0);
        let measured = steady_shear_stress(&mat, 4.0, 1.0e-4);
        assert!(
            (measured - 100.0).abs() < 1.0,
            "with no viscosity the stress must sit on tau_0 itself, measured {measured:.3} Pa"
        );
    }

    /// The property the purely viscous branch structurally cannot have: a
    /// deviatoric stress that survives after the motion stops.
    #[test]
    fn stored_shear_stress_survives_when_the_motion_stops() {
        let mat = material(100.0, 2.0, 5000.0);
        let c = Mat2::from_cols(Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0));
        let mut p = Particle::zeroed();
        p.deformation_gradient = Mat2::IDENTITY;
        p.mass = 1.0;
        p.initial_volume = 1.0;
        let mut particles = Particles::from(vec![p]);
        for _ in 0..2_000 {
            particles.velocity_gradient[0] = c;
            mat.update_particle(&mut particles.update_ctx(0), 1.0e-4);
        }
        // Now stop dead and keep stepping: a rate-based law drops to zero
        // stress immediately, a stored-strain law does not.
        for _ in 0..2_000 {
            particles.velocity_gradient[0] = Mat2::ZERO;
            mat.update_particle(&mut particles.update_ctx(0), 1.0e-4);
        }
        let tau_dev = mat.hencky_deviatoric_stress(particles.deformation_gradient[0]);
        let frobenius_sq = tau_dev.x_axis.length_squared() + tau_dev.y_axis.length_squared();
        let held = (0.5 * frobenius_sq).sqrt();
        assert!(
            held > 90.0,
            "at rest the material must still carry close to its yield stress, held {held:.3} Pa"
        );
    }

    /// The purely viscous branch, for contrast: same yield stress, no
    /// storage modulus, and nothing left once the motion stops. Kept as a
    /// test rather than prose so the limitation stays measured.
    #[test]
    fn purely_viscous_branch_carries_nothing_at_rest() {
        let mat = material(100.0, 2.0, 0.0);
        let at_rest = mat.deviatoric_stress(Mat2::ZERO);
        assert_eq!(
            at_rest,
            Mat2::ZERO,
            "a rate-based Bingham fluid has no stress at zero rate; this is the model's own limit"
        );
        assert!(
            mat.owns_deformation_volume_state(),
            "the purely viscous branch stays a strict WC-MPM liquid"
        );
    }
}

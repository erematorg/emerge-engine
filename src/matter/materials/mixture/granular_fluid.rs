use glam::{Mat2, Vec2};

use crate::materials::svd::svd2;
use crate::materials::utils::{
    MIN_J, advance_deformation_gradient, elastic_wave_dt, lame_from_young, polar_decomposition_2d,
};
use crate::materials::{ConstitutiveModel, MaterialModel, MaterialParams};
use crate::particle::{Particle, ParticleUpdateCtx, Particles};

/// Granular-fluid mixture: Tait EOS bulk pressure + corotated elastic deviatoric + SVD plasticity.
///
/// Constitutive law (Dunatunga & Kamrin 2015, §3):
///   τ = τ_EOS + τ_corotated_dev
///   τ_EOS  = −k·((ρ/ρ₀)^γ − 1)·I                 -- weakly-compressible fluid bulk (Tait EOS)
///   τ_dev  = 2µ·h·dev[(F−R)·Fᵀ] + λ·h·(J−1)·J·I  -- corotated elastic (shape-restoring + vol)
///
/// Plasticity: SVD clamp on F singular values (Stomakhin 2013 §4 -- identical to StomakhinMaterial).
///   Jp accumulates plastic volume change. h = exp(ξ·(1−Jp)) hardens on compression.
///
/// This differs from:
///   - `BinghamFluidMaterial` (purely fluid, no elastic restoring force)
///   - `DruckerPragerMaterial` (no EOS pressure, rate-independent DP)
///   - `StomakhinMaterial` (no EOS, purely elastic+plastic)
///
/// Use for: wet terrain substrates, wet granular flows, biological cell matrices.
/// Ref: Kamrin 2015 granular-fluid; SoftZoo's own mud material (`mud.py`) independently
/// confirms the same fluid-EOS + corotated blend CONCEPT -- but its own specific
/// parameters (a single fixed set, linear not Tait EOS, θ_c=0.025) do NOT match this
/// file's three presets below; see each preset's own honest-disclosure doc comment.
#[derive(Debug, Clone, Copy)]
pub struct GranularFluidMaterial {
    /// Elastic shear modulus µ -- corotated deviatoric stiffness.
    pub mu: f32,
    /// Elastic first Lamé λ -- volumetric elastic contribution.
    pub lambda: f32,
    /// Rest density ρ₀. EOS pressure is zero when ρ = ρ₀.
    pub rest_density: f32,
    /// EOS bulk stiffness k. Tait EOS: p = k·((ρ/ρ₀)^γ − 1).
    pub eos_stiffness: f32,
    /// EOS exponent γ. 7 for near-incompressible; 1–3 for compressible granular flow.
    pub eos_power: f32,
    /// Hardening exponent ξ. h = exp(ξ·(1−Jp)). 0 = perfect plasticity.
    pub hardening_exponent: f32,
    /// Compression limit θ_c -- singular values clamped at (1−θ_c).
    pub compression_limit: f32,
    /// Stretch limit θ_s -- singular values clamped at (1+θ_s).
    pub stretch_limit: f32,
    /// Jp lower bound -- prevents h from exploding under sustained compression.
    pub min_plastic_jacobian: f32,
    /// Jp upper bound -- limits plastic volume expansion.
    pub max_plastic_jacobian: f32,
    /// Granular contact/no-tension pressure floor. This is a constitutive
    /// choice for the granular branch, not free-surface surface tension and
    /// not used by strict Newtonian/Bingham WC-MPM liquids.
    pub pressure_floor: f32,
    /// Dynamic shear viscosity η: wet granular materials (mud, clay,
    /// cytoplasm) dissipate on top of their elastic-plastic response. Same
    /// formula as `NewtonianFluidMaterial` (`τ += η·dev(D)`, D the symmetric
    /// velocity gradient). At 0.0 a hard impact bounces almost elastically
    /// (min_j ~0.43, the body rebounding past its drop height). 0.0 =
    /// disabled (`new()`'s default).
    pub dynamic_viscosity: f32,
    /// Bulk (volumetric) viscosity ζ. With shear viscosity alone, the two
    /// volumetric stiffness sources (Tait EOS pressure and the corotated
    /// `lambda*(J-1)*J` term) kept bouncing on impact (min_j 0.6-0.85), since
    /// `dynamic_viscosity` only damps the deviatoric part. Same formula as
    /// `NewtonianFluidMaterial::bulk_viscosity` (`τ += ζ·(∇·v)·I`); physically,
    /// pore-fluid drainage resists volume change. 0.0 = disabled.
    pub bulk_viscosity: f32,
}

impl GranularFluidMaterial {
    /// Raw field constructor (like every other material's `::new()`) with the
    /// physically meaningful parameters; the numerical-stability fields take
    /// the values every preset below (`saturated_loam`/`consolidated_clay`/
    /// `cytoplasmic`) uses: `eos_power` 2.0, in the granular-flow range the
    /// field requires (7.0 runs away under settling, see `saturated_loam`),
    /// and `pressure_floor: 0.0` (no tensile traction between grains).
    pub const fn new(
        lambda: f32,
        mu: f32,
        rest_density: f32,
        eos_stiffness: f32,
        hardening_exponent: f32,
        compression_limit: f32,
    ) -> Self {
        Self {
            mu,
            lambda,
            rest_density,
            eos_stiffness,
            eos_power: 2.0,
            hardening_exponent,
            compression_limit,
            stretch_limit: 0.01,
            min_plastic_jacobian: 0.2,
            max_plastic_jacobian: 3.0,
            pressure_floor: 0.0,
            dynamic_viscosity: 0.0,
            bulk_viscosity: 0.0,
        }
    }

    /// Saturated loam: eos_stiffness=200, ξ=5, θ_c=0.4 -- yields easily, flows under load.
    ///
    /// The constitutive law (Tait EOS + corotated elastic + Stomakhin SVD
    /// plasticity) is cited; these parameter values (eos_stiffness,
    /// hardening_exponent, compression_limit, stretch_limit, plastic-Jacobian
    /// bounds, rest_density) are hand-tuned. They do not trace to SoftZoo's
    /// `mud.py` (one parameter set, a linear EOS, θ_c = 0.025, 12-24x below
    /// this preset's 0.4) nor to Dunatunga & Kamrin 2015 (granular only).
    /// "Saturated loam" is illustrative, not a measured loam; it needs a
    /// geotechnical source before claiming otherwise.
    ///
    /// `eos_power` must stay in the granular-flow range (1-3, per the field's
    /// doc) -- the near-incompressible value 7.0 causes runaway pressure
    /// under gravity-settling compression, driving dilation that weakens
    /// `hardening_scale` toward its floor in an unbounded feedback loop.
    pub fn saturated_loam(young_modulus: f32, poisson_ratio: f32) -> Self {
        let (lambda, mu) = lame_from_young(young_modulus, poisson_ratio);
        Self {
            mu,
            lambda,
            rest_density: 1.0,
            eos_stiffness: 200.0,
            eos_power: 2.0,
            hardening_exponent: 5.0,
            compression_limit: 0.4,
            stretch_limit: 0.01,
            min_plastic_jacobian: 0.2,
            max_plastic_jacobian: 3.0,
            pressure_floor: 0.0,
            // 0.3*mu, order-of-mu damping (as `ViscoelasticMaterial`), enough
            // to stop a hard impact bouncing elastically (see
            // `dynamic_viscosity`).
            dynamic_viscosity: 0.3 * mu,
            // Scales with the volumetric stiffness (eos_stiffness), not mu:
            // bulk_viscosity damps div_v, the quantity the EOS pressure and the
            // `lam_vol` term act on (scaled by mu, consolidated_clay crept
            // without settling).
            bulk_viscosity: 0.5 * 200.0,
        }
    }

    /// Consolidated clay: eos_stiffness=500, ξ=3, θ_c=0.3 -- higher stiffness, slower creep.
    ///
    /// Same honest disclosure as `saturated_loam` above: real cited law, hand-tuned
    /// (not measured) shape parameters -- not yet verified against real consolidated-
    /// clay geotechnical data.
    ///
    /// Same `eos_power` constraint as `saturated_loam` above.
    pub fn consolidated_clay(young_modulus: f32, poisson_ratio: f32) -> Self {
        let (lambda, mu) = lame_from_young(young_modulus, poisson_ratio);
        Self {
            mu,
            lambda,
            rest_density: 1.2,
            eos_stiffness: 500.0,
            eos_power: 2.0,
            hardening_exponent: 3.0,
            compression_limit: 0.3,
            stretch_limit: 0.01,
            min_plastic_jacobian: 0.3,
            max_plastic_jacobian: 2.5,
            pressure_floor: 0.0,
            // As saturated_loam, with 0.5 instead of 0.3 for the stiffer,
            // slower-creeping preset.
            dynamic_viscosity: 0.5 * mu,
            // As saturated_loam: scales with this preset's eos_stiffness (500),
            // not mu.
            bulk_viscosity: 0.5 * 500.0,
        }
    }

    /// Cytoplasmic matrix: eos_stiffness=50, ξ=1, large yield surface.
    /// Use for biological cell interiors and soft tissue matrices.
    ///
    /// Same honest disclosure as `saturated_loam` above: real cited law, hand-tuned
    /// (not measured) shape parameters -- not yet verified against real cytoplasm
    /// rheology literature.
    ///
    /// Same `eos_power` constraint as `saturated_loam` above; low
    /// `eos_stiffness=50` makes this preset least sensitive to it, but the
    /// constraint still applies.
    pub fn cytoplasmic(young_modulus: f32, poisson_ratio: f32) -> Self {
        let (lambda, mu) = lame_from_young(young_modulus, poisson_ratio);
        Self {
            mu,
            lambda,
            rest_density: 1.0,
            eos_stiffness: 50.0,
            eos_power: 2.0,
            hardening_exponent: 1.0,
            compression_limit: 0.6,
            stretch_limit: 0.05,
            min_plastic_jacobian: 0.1,
            max_plastic_jacobian: 5.0,
            pressure_floor: 0.0,
            // As saturated_loam, with 0.15 instead of 0.3: cytoplasm is less
            // viscous relative to its stiffness than wet soil.
            dynamic_viscosity: 0.15 * mu,
            // As saturated_loam: scales with this preset's eos_stiffness (50).
            bulk_viscosity: 0.5 * 50.0,
        }
    }
}

impl MaterialModel for GranularFluidMaterial {
    fn constitutive_model(&self) -> ConstitutiveModel {
        ConstitutiveModel::GranularFluid
    }

    // Seeds `initial_volume`/`volume`/`density` from the conserved
    // `mass/rest_density` and owns that state (`owns_deformation_volume_state`),
    // like every sibling EOS-pressure material (`NewtonianFluidMaterial`,
    // `BinghamFluidMaterial`, `IdealGasMaterial`, `CavitatingFluidMaterial`,
    // `BoilingMixtureMaterial`). Otherwise density and volume come from
    // `SpawnRegion`'s kernel gather, biased at the free surface (see
    // `NewtonianFluidMaterial::init_particle`), and on the GPU `g2p.wgsl`
    // overwrites them from that gather every substep. `update_particle` then
    // derives `volume = initial_volume * j` and `density = mass / volume`.
    fn init_particle(&self, particle: &mut Particle) {
        particle.plastic_volume_ratio = 1.0;
        particle.hardening_scale = 1.0;
        let j = particle.deformation_gradient.determinant().max(MIN_J);
        particle.initial_volume = particle.mass / self.rest_density.max(1.0e-6);
        particle.volume = particle.initial_volume * j;
        particle.density = self.rest_density / j;
    }

    fn owns_deformation_volume_state(&self) -> bool {
        true
    }

    // No `init_particle_from_transition` override: the default resets F to
    // identity. A `gas.rs`-style override preserving J against `rest_density`
    // made `diag_phase_transition_under_load_causes_stress_discontinuity`
    // (`tests/physics_correctness.rs`) worse (speed delta 0.16 -> 2.93). With
    // the eos_power 2.0 the scene uses (`sand_water_saturation.rs`'s
    // `make_mixture`) the identity reset gives -0.0076 there, so it is not the
    // danger it looked like at 7.0. That scene's 104,737-frame crash may have a
    // separate, slower cause; a repeated-transition diagnostic is the next
    // step.

    fn kirchhoff_stress(&self, particles: &Particles, i: usize) -> Mat2 {
        let f = particles.deformation_gradient[i];
        let j = f.determinant().max(MIN_J);

        let density = (self.rest_density / j).max(1.0e-6);
        let ratio = (density / self.rest_density.max(1.0e-6)).max(1.0e-6);
        let pressure = (self.eos_stiffness
            * (crate::materials::utils::fast_pow(ratio, self.eos_power) - 1.0))
            .max(self.pressure_floor);

        let h = particles.hardening_scale[i];
        let r = polar_decomposition_2d(f);
        let mu_eff = self.mu * h;
        let coro = 2.0 * mu_eff * (f - r) * f.transpose();
        let tr = coro.x_axis.x + coro.y_axis.y;
        let dev_coro = coro - Mat2::from_diagonal(Vec2::splat(tr * 0.5));

        let lam_vol = self.lambda * h * (j - 1.0) * j * Mat2::IDENTITY;

        let mut stress = Mat2::from_diagonal(Vec2::splat(-pressure)) + dev_coro + lam_vol;

        // Viscous dissipation (see `dynamic_viscosity`/`bulk_viscosity`), the
        // formulas of `NewtonianFluidMaterial::kirchhoff_stress`
        // (τ += η·dev(D) + ζ·(∇·v)·I).
        if self.dynamic_viscosity > 0.0 || self.bulk_viscosity > 0.0 {
            let c = particles.velocity_gradient[i];
            let sym_strain = c + c.transpose();
            let div_v = sym_strain.x_axis.x + sym_strain.y_axis.y;
            if self.dynamic_viscosity > 0.0 {
                let strain_dev = sym_strain - Mat2::from_diagonal(Vec2::splat(div_v * 0.5));
                stress += self.dynamic_viscosity * strain_dev;
            }
            if self.bulk_viscosity > 0.0 {
                stress += Mat2::from_diagonal(Vec2::splat(self.bulk_viscosity * div_v * 0.5));
            }
        }

        stress
    }

    fn stress_volume(&self, particles: &Particles, i: usize) -> f32 {
        particles.volume[i].max(1.0e-6)
    }

    fn update_particle(&self, ctx: &mut ParticleUpdateCtx, dt: f32) {
        // This increment advances the single F read by both the EOS volume
        // response and the corotated/SVD branch. Exact constant-C integration
        // prevents Euler volume drift from becoming false EOS pressure or
        // permanent Jp/hardening.
        let f_trial =
            advance_deformation_gradient(*ctx.deformation_gradient, dt * *ctx.velocity_gradient);

        if self.compression_limit > 0.0 || self.stretch_limit > 0.0 {
            let (u, sigma, vt) = svd2(f_trial);
            let sigma_c = Vec2::new(
                sigma
                    .x
                    .clamp(1.0 - self.compression_limit, 1.0 + self.stretch_limit),
                sigma
                    .y
                    .clamp(1.0 - self.compression_limit, 1.0 + self.stretch_limit),
            );
            let jp_new = *ctx.plastic_volume_ratio * (sigma.x * sigma.y)
                / (sigma_c.x * sigma_c.y).max(1.0e-10);
            *ctx.plastic_volume_ratio =
                jp_new.clamp(self.min_plastic_jacobian, self.max_plastic_jacobian);
            // `hardening_scale` floor must stay at 1.0, not lower: `h` scales both
            // the deviatoric and volumetric elastic terms in `kirchhoff_stress`, so
            // h<1 under dilation (Jp>1) softens the material -- an unbounded
            // soften->dilate->soften feedback with nothing to stop it. Compression-
            // side hardening (h>1) is self-stabilizing; there's no equivalent
            // mechanism on the dilation side, so the floor clamps at baseline
            // stiffness (h=1) instead.
            *ctx.hardening_scale = (self.hardening_exponent * (1.0 - *ctx.plastic_volume_ratio))
                .exp()
                .clamp(1.0, 7.0);
            *ctx.deformation_gradient = u * Mat2::from_diagonal(sigma_c) * vt;
        } else {
            *ctx.deformation_gradient = f_trial;
        }

        let j = ctx.deformation_gradient.determinant().max(MIN_J);
        let v = (ctx.initial_volume * j).max(1.0e-6);
        *ctx.volume = v;
        *ctx.density = ctx.mass / v;
    }

    fn params(&self) -> MaterialParams {
        MaterialParams {
            model: ConstitutiveModel::GranularFluid as u32,
            mu: self.mu,
            lambda: self.lambda,
            rest_density: self.rest_density,
            eos_stiffness: self.eos_stiffness,
            eos_power: self.eos_power,
            hardening_exponent: self.hardening_exponent,
            compression_limit: self.compression_limit,
            stretch_limit: self.stretch_limit,
            volume_ratio_min: self.min_plastic_jacobian,
            volume_ratio_max: self.max_plastic_jacobian,
            pressure_floor: self.pressure_floor,
            dynamic_viscosity: self.dynamic_viscosity,
            bulk_viscosity: self.bulk_viscosity,
            // `MaterialParams::owns_deformation_volume_state` is not derived
            // from the trait method: each material's `params()` forwards it,
            // or `..Default::default()` leaves it false on the GPU.
            owns_deformation_volume_state: self.owns_deformation_volume_state() as u32,
            ..Default::default()
        }
    }

    fn timestep_bound(
        &self,
        density: f32,
        hardening_scale: f32,
        cell_width: f32,
        material_cfl: f32,
        viscous_cfl: f32,
    ) -> f32 {
        let mut dt_bound = elastic_wave_dt(
            self.lambda,
            self.mu,
            hardening_scale,
            density,
            MIN_J,
            cell_width,
            material_cfl,
        );

        // Viscous CFL bound for the explicit viscosity terms: too large a
        // dt*viscosity/mass makes an explicit damping term inject energy
        // (clay piles grew to 52+ units on impact). The formula of
        // `NewtonianFluidMaterial::timestep_bound`, extended to
        // bulk_viscosity, which acts on the same velocity gradient.
        let total_viscosity = self.dynamic_viscosity + self.bulk_viscosity;
        if total_viscosity > 0.0 {
            let density = density.max(1.0e-6);
            let kinematic_viscosity = total_viscosity / density;
            if kinematic_viscosity > f32::EPSILON {
                dt_bound =
                    dt_bound.min(viscous_cfl * cell_width * cell_width / kinematic_viscosity);
            }
        }

        dt_bound
    }
}

#[cfg(test)]
mod kinematic_projection_tests {
    use super::*;

    fn particle() -> Particles {
        let mut p = Particle::zeroed();
        p.deformation_gradient = Mat2::IDENTITY;
        p.mass = 1.0;
        p.initial_volume = 1.0;
        p.plastic_volume_ratio = 1.0;
        p.hardening_scale = 1.0;
        Particles::from(vec![p])
    }

    fn run_rate_step(mat: &GranularFluidMaterial, particles: &mut Particles, rate: Mat2, dt: f32) {
        let mut ctx = particles.update_ctx(0);
        *ctx.velocity_gradient = rate;
        mat.update_particle(&mut ctx, dt);
    }

    fn matrix_error(a: Mat2, b: Mat2) -> f32 {
        (a.x_axis - b.x_axis).length() + (a.y_axis - b.y_axis).length()
    }

    #[test]
    fn rigid_rotation_preserves_eos_volume_and_plastic_history() {
        let mat = GranularFluidMaterial::new(500.0, 200.0, 1.0, 100.0, 10.0, 0.025);
        let mut particles = particle();
        let omega = 2.1;
        let dt = 0.2;
        let spin = Mat2::from_cols(Vec2::new(0.0, omega), Vec2::new(-omega, 0.0));

        run_rate_step(&mat, &mut particles, spin, dt);

        let expected = Mat2::from_angle(omega * dt);
        assert!(matrix_error(particles.deformation_gradient[0], expected) < 3.0e-6);
        assert!((particles.deformation_gradient[0].determinant() - 1.0).abs() < 2.0e-6);
        assert_eq!(particles.plastic_volume_ratio[0], 1.0);
        assert_eq!(particles.hardening_scale[0], 1.0);
    }

    #[test]
    fn opposite_subclamp_rates_are_reversible_without_false_hardening() {
        let mat = GranularFluidMaterial::new(500.0, 200.0, 1.0, 100.0, 10.0, 0.025);
        let mut particles = particle();
        let rate = Mat2::from_diagonal(Vec2::new(-0.01, 0.005));
        let dt = 0.1;

        run_rate_step(&mat, &mut particles, rate, dt);
        assert_eq!(particles.plastic_volume_ratio[0], 1.0);
        assert_eq!(particles.hardening_scale[0], 1.0);
        run_rate_step(&mat, &mut particles, -rate, dt);

        assert!(matrix_error(particles.deformation_gradient[0], Mat2::IDENTITY) < 3.0e-6);
        assert_eq!(particles.plastic_volume_ratio[0], 1.0);
        assert_eq!(particles.hardening_scale[0], 1.0);
    }

    #[test]
    fn exponential_trial_respects_both_sides_of_compression_clamp() {
        let mat = GranularFluidMaterial::new(500.0, 200.0, 1.0, 100.0, 10.0, 0.025);
        let floor = 1.0 - mat.compression_limit;
        let dt = 0.1;

        let inside_sigma = floor + 1.0e-4;
        let mut inside = particle();
        run_rate_step(
            &mat,
            &mut inside,
            Mat2::from_diagonal(Vec2::new(inside_sigma.ln() / dt, 0.0)),
            dt,
        );
        assert!((inside.deformation_gradient[0].x_axis.x - inside_sigma).abs() < 2.0e-6);
        assert_eq!(inside.plastic_volume_ratio[0], 1.0);

        let outside_sigma = floor - 1.0e-4;
        let mut outside = particle();
        run_rate_step(
            &mat,
            &mut outside,
            Mat2::from_diagonal(Vec2::new(outside_sigma.ln() / dt, 0.0)),
            dt,
        );
        assert!((outside.deformation_gradient[0].x_axis.x - floor).abs() < 2.0e-6);
        let expected_jp = outside_sigma / floor;
        assert!((outside.plastic_volume_ratio[0] - expected_jp).abs() < 2.0e-6);
        assert!(outside.hardening_scale[0] > 1.0);
    }
}

/// A fresh particle gets its conserved volume and density from
/// `mass/rest_density`, like every sibling EOS-pressure material, and the
/// material declares it owns that state both through the trait method and
/// through `MaterialParams` for the GPU (the two are not linked).
#[cfg(test)]
mod spawn_state_tests {
    use super::*;

    #[test]
    fn init_particle_seeds_real_conserved_volume_and_density() {
        let mat = GranularFluidMaterial::new(500.0, 200.0, 4.0, 100.0, 10.0, 0.025);
        let mut p = Particle::zeroed();
        p.deformation_gradient = Mat2::IDENTITY;
        p.mass = 2.0;
        mat.init_particle(&mut p);

        let expected_volume = p.mass / mat.rest_density; // = 0.5
        assert!(
            (p.initial_volume - expected_volume).abs() < 1.0e-6,
            "initial_volume must be the real conserved mass/rest_density, \
             got {} expected {expected_volume}",
            p.initial_volume
        );
        assert!(
            (p.volume - expected_volume).abs() < 1.0e-6,
            "volume at spawn (J=1) must equal initial_volume exactly, got {} expected {expected_volume}",
            p.volume
        );
        assert!(
            (p.density - mat.rest_density).abs() < 1.0e-6,
            "density at spawn (J=1) must equal rest_density exactly, got {} expected {}",
            p.density,
            mat.rest_density
        );
    }

    #[test]
    fn init_particle_respects_real_prior_compression() {
        // A particle spawned with F already compressed (J=0.8, e.g. seeded
        // mid-scene by some other mechanism) must get a volume/density
        // consistent with THAT real J, not silently reset to J=1 -- matches
        // `NewtonianFluidMaterial::init_particle`'s own real contract.
        let mat = GranularFluidMaterial::new(500.0, 200.0, 4.0, 100.0, 10.0, 0.025);
        let mut p = Particle::zeroed();
        let s = 0.8_f32.sqrt();
        p.deformation_gradient = Mat2::from_diagonal(Vec2::splat(s));
        p.mass = 2.0;
        mat.init_particle(&mut p);

        let j = p.deformation_gradient.determinant();
        assert!(
            (j - 0.8).abs() < 1.0e-5,
            "test setup sanity: J should be 0.8, got {j}"
        );
        let expected_initial_volume = p.mass / mat.rest_density;
        assert!((p.initial_volume - expected_initial_volume).abs() < 1.0e-6);
        assert!((p.volume - expected_initial_volume * j).abs() < 1.0e-6);
        assert!((p.density - mat.rest_density / j).abs() < 1.0e-6);
    }

    #[test]
    fn owns_deformation_volume_state_is_true_and_reaches_gpu_params() {
        let mat = GranularFluidMaterial::new(500.0, 200.0, 4.0, 100.0, 10.0, 0.025);
        assert!(
            MaterialModel::owns_deformation_volume_state(&mat),
            "GranularFluidMaterial must own its volume/density state -- it \
             derives both from its own EOS+deformation-gradient, not a \
             kernel-mass gather"
        );
        // The trait method alone does not reach the GPU: `params()` must
        // forward it (see that function).
        assert_eq!(
            mat.params().owns_deformation_volume_state,
            1,
            "MaterialParams::owns_deformation_volume_state must be forwarded \
             from the trait method, not left at Default::default()'s 0"
        );
    }
}

//! Nonlocal Granular Fluidity (NGF) field -- lets granular flow "cooperate"
//! spatially instead of every point deciding to yield in total local
//! isolation. Published mechanism (Kamrin & Koval, PRL 2012; Henann &
//! Kamrin, several follow-ups); the MPM-specific numerical scheme below
//! matches a peer-reviewed reference read directly (not from its
//! abstract): Haeri & Skonieczny 2022, *"Three-dimensional granular flow
//! continuum modeling via material point method with hyperelastic nonlocal
//! granular fluidity"* (CMAME, arXiv:2111.01523).
//!
//! # Physics
//! `g` (granular fluidity) relates plastic shear strain rate to the stress
//! ratio: `γ̇ = g·μ`. Governed by (Kamrin & Henann 2015, Soft Matter 11,
//! 179-185, doi:10.1039/c4sm01838a; equations as numbered in the arXiv
//! preprint 1408.5205v1, read from its PDF), eq. 6, the dynamical form:
//! ```text
//! t0 · ∂g/∂t = A²d²·∇²g − (μs−μ)·g − b·μ·√(ρs·d²/P)·g²
//! ```
//! whose local steady state is eq. 4 with the linear `μ_loc(I) = μs + b·I`:
//! ```text
//! g_loc(μ, P) = (μ−μs)/(b·μ) · √(P/(ρs·d²))
//! ```
//! Fluidity vanishes with the confining pressure: no confinement, no flow
//! cooperation. (Issue #51: an earlier version had the square root inverted
//! and `μ` missing, so its `g_loc` diverged as `1/√P` at the free surface.)
//! `A` = nonlocal amplitude, `d` = grain diameter, `μs` = static friction
//! coefficient, `ρs` = grain density, `t0` = a microscopic grain-inertial
//! relaxation timescale, `b` a rate-dependence constant (same convention as
//! `MuIRheologyMaterial`'s own `b = (μ2−μs)/I0`).
//!
//! # Numerical scheme
//! Explicit finite-difference -- matches Haeri & Skonieczny's own verified
//! choice, not an invented shortcut. Same P2G→normalize→Laplacian→G2P shape
//! as [`super::scalar_field::ScalarDiffusionField`]/[`super::diffusion::ThermalDiffusion`]
//! (reuses the shared [`super::stencil::laplacian_step`]), plus a reaction
//! step for the two extra terms above. Quoted stability bound from
//! that same paper: `Δt < Δx²·t0 / (2·A²·d²)` -- unlike `ScalarDiffusionField`'s
//! bound (only ever documented, never enforced, since thermal diffusivity is
//! tiny relative to MPM's own elastic-wave CFL), this one is plausibly
//! binding and is exposed via [`GranularFluidityField::stability_dt`] for
//! callers to fold into their own adaptive substep choice.
//!
//! # `g` is grid-only state, not a particle field
//! Unlike temperature/pheromone, `g` has no natural per-particle home:
//! `Particle` is fixed at exactly 128 bytes with no spare padding
//! (`src/matter/particle/mod.rs`). This matches the real physics anyway --
//! nonlocal fluidity is fundamentally a spatial/grid quantity, not a
//! per-particle property. `g` therefore persists as this struct's own grid
//! state between calls to [`GranularFluidityField::apply`], and is
//! *gathered* (not stored) into a caller-supplied buffer each substep for
//! use in a yield check.
//!
//! # Deriving pressure and stress ratio (not stored fields either)
//! Pressure `P` and stress ratio `μ` are computed fresh each substep from a
//! particle's own `deformation_gradient`, reusing the exact same
//! Hencky-trace formula `MuIRheologyMaterial::update_particle`
//! (`src/matter/materials/sand_mui.rs`) already uses -- cited reuse,
//! not a new formula invented here. The caller supplies
//! `pressure_and_ratio: fn(&Particle) -> (f32, f32)` since only the coupled
//! material knows its own elastic Lamé parameters.

use glam::IVec2;

use crate::{grid::kernel::quadratic_weights, particle::Particles};

/// Physical parameters for the NGF PDE above.
#[derive(Clone, Copy, Debug)]
pub struct GranularFluidityConfig {
    /// Static friction coefficient μs (dimensionless) -- the same real value
    /// as the coupled material's own `tan(friction_angle)`.
    pub mu_s: f32,
    /// Grain diameter `d` \[m\] -- see e.g.
    /// `DruckerPragerMaterial::GRAIN_DIAMETER_M`.
    pub grain_diameter_m: f32,
    /// Grain density ρs \[kg/m³\].
    pub grain_density_kg_m3: f32,
    /// Nonlocal amplitude `A` (dimensionless) -- cited value 0.48
    /// (Henann & Kamrin 2013 glass beads; independently reconfirmed for
    /// real sand by Haeri & Skonieczny 2022, same value).
    pub nonlocal_amplitude: f32,
    /// Rate-dependence constant `b` (dimensionless), same convention as
    /// `MuIRheologyMaterial`'s own `b = (μ2−μs)/I0`.
    pub b: f32,
    /// Microscopic grain-inertial relaxation timescale `t0` \[s\]. Real,
    /// cited value 1e-4s (Haeri & Skonieczny 2022, Table 1).
    pub t0_s: f32,
}

impl GranularFluidityConfig {
    /// Von Neumann stability bound for the explicit scheme above, quoted
    /// from Haeri & Skonieczny 2022, §4: `Δt < Δx² · t0 / (2·A²·d²)`.
    ///
    /// `dx_m` is the physical grid cell size (`SimConfig::dx_meters`).
    pub fn stability_dt(&self, dx_m: f32) -> f32 {
        let denom = 2.0 * self.nonlocal_amplitude.powi(2) * self.grain_diameter_m.powi(2);
        dx_m * dx_m * self.t0_s / denom.max(1e-30)
    }
}

/// A persistent, grid-coupled granular fluidity field.
pub struct GranularFluidityField {
    pub config: GranularFluidityConfig,
    /// Reads a particle's current (pressure in Pa, stress ratio) pair. The
    /// pressure must be SI: the reaction term combines it with `ρs` in
    /// kg/m³ and `d` in m. The ratio is dimensionless.
    pub pressure_and_ratio: fn(&crate::particle::Particle) -> (f32, f32),

    grid_res: usize,
    grid_mass: Vec<f32>, // Σ(w·mass)              -- cleared each step
    grid_p: Vec<f32>,    // scattered pressure       -- cleared each step
    grid_mu: Vec<f32>,   // scattered stress ratio   -- cleared each step
    grid_g: Vec<f32>,    // persistent PDE state -- NOT cleared between steps
    grid_work: Vec<f32>, // scratch: diffusion/reaction output before it replaces grid_g
}

impl GranularFluidityField {
    pub fn new(
        config: GranularFluidityConfig,
        pressure_and_ratio: fn(&crate::particle::Particle) -> (f32, f32),
        grid_res: usize,
    ) -> Self {
        let n = grid_res * grid_res;
        Self {
            config,
            pressure_and_ratio,
            grid_res,
            grid_mass: vec![0.0; n],
            grid_p: vec![0.0; n],
            grid_mu: vec![0.0; n],
            grid_g: vec![0.0; n],
            grid_work: vec![0.0; n],
        }
    }

    /// Advance `g` by one substep and gather it into `out[pi]` for every
    /// particle in `particles`. Real per-substep sequence: P2G-scatter
    /// pressure/stress-ratio → normalize → diffuse (shared stencil) →
    /// react (the two extra NGF terms, explicit Euler) → G2P-gather `g`.
    ///
    /// `sub_dt` must respect [`GranularFluidityConfig::stability_dt`] --
    /// callers are responsible for folding that bound into their own
    /// adaptive substep choice (this field does not clamp `sub_dt` itself,
    /// matching how `ScalarDiffusionField`'s bound is also caller-enforced).
    ///
    /// `dx_meters`: real physical grid cell size (`SimConfig::dx_meters`).
    /// [`super::stencil::laplacian_step`]'s 4-neighbor-minus-center sum is a
    /// bare grid-INDEX-space finite difference (no implicit cell size) -- it
    /// must be scaled by `1/dx_meters²` to become the `∇²g` the PDE
    /// actually calls for, exactly the convention `ThermalConfig::alpha_grid`
    /// already documents ("Folding dx² in keeps the Laplacian formula
    /// dimensionless over grid indices") and `Cosserat` field's own `apply`
    /// call site already threads through. Without this dx-normalization, the diffusion term's magnitude doesn't
    /// depend on the cell size at all, so refining the grid (same `A`,
    /// `d`, `t0`) changes how many REAL METERS the same "diffusivity_dt"
    /// spreads `g` per step -- the direct cause of this module's own
    /// resolution-dependence bug (see `sand.rs`'s `ngf_lajeunesse_runout_
    /// resolution_independence` test history).
    pub fn apply(&mut self, particles: &Particles, sub_dt: f32, dx_meters: f32, out: &mut [f32]) {
        let n = self.grid_res * self.grid_res;
        let res = self.grid_res as i32;

        for i in 0..n {
            self.grid_p[i] = 0.0;
            self.grid_mu[i] = 0.0;
            self.grid_mass[i] = 0.0;
        }

        // --- P2G: scatter mass-weighted (P, μ) into grid_p / grid_mu ---
        for pi in 0..particles.len() {
            let p = particles.get(pi);
            let (pressure, mu) = (self.pressure_and_ratio)(&p);
            let w = quadratic_weights(p.x);
            for gx in 0i32..3 {
                for gy in 0i32..3 {
                    let weight = w.wx[gx as usize] * w.wy[gy as usize];
                    let cell = w.base_cell + IVec2::new(gx - 1, gy - 1);
                    if cell.x < 0 || cell.y < 0 || cell.x >= res || cell.y >= res {
                        continue;
                    }
                    let idx = (cell.x * res + cell.y) as usize;
                    let mw = weight * p.mass;
                    self.grid_p[idx] += mw * pressure;
                    self.grid_mu[idx] += mw * mu;
                    self.grid_mass[idx] += mw;
                }
            }
        }

        // --- Normalize: empty cells keep their last real g at rest (no
        // flow information there to update from) ---
        for i in 0..n {
            if self.grid_mass[i] > 1e-10 {
                self.grid_p[i] /= self.grid_mass[i];
                self.grid_mu[i] /= self.grid_mass[i];
            } else {
                self.grid_p[i] = 0.0;
                self.grid_mu[i] = self.config.mu_s;
            }
        }

        // --- Diffuse: shared explicit-Euler 5-point Laplacian, same
        // stencil ScalarDiffusionField/ThermalDiffusion already use ---
        // dx_meters^2 division: see this method's doc.
        let dx2 = (dx_meters * dx_meters).max(1e-30);
        let diffusivity_dt = (self.config.nonlocal_amplitude * self.config.grain_diameter_m)
            .powi(2)
            / (self.config.t0_s * dx2)
            * sub_dt;
        super::stencil::laplacian_step(
            &self.grid_g,
            &mut self.grid_work,
            self.grid_res,
            diffusivity_dt,
            0.0,
        );

        // --- React: the two local NGF terms of eq. 6, integrated exactly ---
        // t0*dg/dt = (mu-mu_s)*g - b*mu*sqrt(rho_s*d^2/P)*g^2 (already
        // diffused this step; this is the remaining reaction contribution).
        //
        // For g >= 0 this is the logistic equation `dg/dt = r*g*(1-g/g_loc)`
        // with `r = (mu-mu_s)/t0`, `c = b*mu*d*sqrt(rho_s/P)/t0` and
        // `g_loc = r/c` (eq. 4). The substitution `u = 1/g` turns it into the
        // linear `du/dt = -r*u + c`, solved exactly: `u(t) = c/r + (u0 -
        // c/r)*exp(-r*t)`. Exact for any `dt`: explicit Euler at the cited
        // `t0 = 1e-4 s` either lagged the growth or overshot it by orders of
        // magnitude.
        //
        // A cell with no material has no local rheology, so it only diffuses
        // (the fluidity is not forced to zero around the material; the paper
        // takes a zero gradient at a free surface). A material cell at zero
        // pressure has an infinite `c`: its fluidity relaxes to `g_loc = 0`.
        let cfg = self.config;
        const BOOTSTRAP_SEED: f32 = 1.0e-6; // still needed: u=1/g is singular at g=0
        for i in 0..n {
            let g_prev = self.grid_work[i];
            if self.grid_mass[i] <= 1e-10 {
                self.grid_work[i] = g_prev.max(0.0);
                continue;
            }
            let mu = self.grid_mu[i];
            let pressure = self.grid_p[i];
            if pressure <= 0.0 {
                self.grid_work[i] = 0.0;
                continue;
            }
            let linear_coeff = mu - cfg.mu_s; // >0 once locally past static friction

            // g=0 is a STABLE fixed point whenever linear_coeff<=0 --
            // leave it at exactly 0 (matches real physics: nothing to grow
            // from without a source). Only bootstrap the epsilon seed (the
            // `u=1/g` substitution below is singular at g=0) when there IS
            // a source to grow toward.
            if linear_coeff <= 0.0 && g_prev <= 0.0 {
                self.grid_work[i] = 0.0;
                continue;
            }
            let g = g_prev.max(BOOTSTRAP_SEED);

            let t0 = cfg.t0_s.max(1e-12);
            let r = linear_coeff / t0;
            let c = cfg.b * mu * cfg.grain_diameter_m * (cfg.grain_density_kg_m3 / pressure).sqrt()
                / t0;

            let u0 = 1.0 / g;
            let u_new = if r.abs() > 1e-9 {
                c / r + (u0 - c / r) * (-r * sub_dt).exp()
            } else {
                u0 + c * sub_dt // r=0 special case: du/dt=c exactly, linear in t
            };
            self.grid_work[i] = if u_new.is_finite() && u_new > 1e-12 {
                1.0 / u_new
            } else {
                0.0
            };
        }

        self.grid_g.copy_from_slice(&self.grid_work);

        // --- G2P: gather g back to particles (transient -- not stored) ---
        for pi in 0..particles.len().min(out.len()) {
            let p = particles.get(pi);
            let w = quadratic_weights(p.x);
            let mut g_sum = 0.0f32;
            let mut w_sum = 0.0f32;
            for gx in 0i32..3 {
                for gy in 0i32..3 {
                    let weight = w.wx[gx as usize] * w.wy[gy as usize];
                    let cell = w.base_cell + IVec2::new(gx - 1, gy - 1);
                    if cell.x < 0 || cell.y < 0 || cell.x >= res || cell.y >= res {
                        continue;
                    }
                    let idx = (cell.x * res + cell.y) as usize;
                    g_sum += weight * self.grid_g[idx];
                    w_sum += weight;
                }
            }
            out[pi] = if w_sum > 1e-10 { g_sum / w_sum } else { 0.0 };
        }
    }

    pub const fn grid_res(&self) -> usize {
        self.grid_res
    }

    /// (min, mean over nonzero, max, count nonzero) of the current `g` field,
    /// to check directly whether `g` stays small or narrow through a collapse
    /// (cooperation too slow or narrow relative to the moving flow front).
    pub fn g_stats(&self) -> (f32, f32, f32, usize) {
        let mut min = f32::INFINITY;
        let mut max = 0.0f32;
        let mut sum = 0.0f32;
        let mut count = 0usize;
        for &g in &self.grid_g {
            if g > 1e-9 {
                min = min.min(g);
                max = max.max(g);
                sum += g;
                count += 1;
            }
        }
        let mean = if count > 0 { sum / count as f32 } else { 0.0 };
        (if count > 0 { min } else { 0.0 }, mean, max, count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::particle::Particle;
    use glam::Vec2;

    fn test_config() -> GranularFluidityConfig {
        // Values from Haeri & Skonieczny 2022 Table 1 (Excavation case).
        GranularFluidityConfig {
            mu_s: 0.70,
            grain_diameter_m: 0.3e-3,
            grain_density_kg_m3: 2583.0,
            nonlocal_amplitude: 0.48,
            b: 0.278,
            t0_s: 1.0e-4,
        }
    }

    fn test_particle_at(pos: Vec2) -> Particle {
        let mut p = Particle::zeroed();
        p.x = pos;
        p.mass = 1.0;
        p.initial_volume = 1.0;
        p
    }

    #[test]
    fn stability_dt_matches_the_cited_formula_directly() {
        let cfg = test_config();
        let dx = 0.01f32;
        let expected = dx * dx * cfg.t0_s
            / (2.0 * cfg.nonlocal_amplitude.powi(2) * cfg.grain_diameter_m.powi(2));
        assert!((cfg.stability_dt(dx) - expected).abs() < 1e-12);
        assert!(cfg.stability_dt(dx) > 0.0);
    }

    #[test]
    fn g_stays_zero_with_no_particles() {
        let cfg = test_config();
        let mut field = GranularFluidityField::new(cfg, |_p| (0.0, 0.70), 16);
        let particles = Particles::from(vec![]);
        let mut out = vec![0.0; 0];
        field.apply(&particles, 1.0e-6, 0.01, &mut out);
        assert!(field.grid_g.iter().all(|&g| g == 0.0));
    }

    #[test]
    fn g_stays_zero_when_stress_ratio_never_exceeds_static_friction() {
        // At mu == mu_s everywhere, the linear relaxation term is exactly
        // zero and g has no source to grow from a zero start -- a real,
        // checkable invariant of the equation itself, not just "it runs."
        let cfg = test_config();
        let mut field = GranularFluidityField::new(cfg, |_p| (1000.0, 0.70), 8);
        let particles = Particles::from(vec![
            test_particle_at(Vec2::new(4.0, 4.0)),
            test_particle_at(Vec2::new(4.0, 4.0)),
            test_particle_at(Vec2::new(4.0, 4.0)),
            test_particle_at(Vec2::new(4.0, 4.0)),
        ]);
        let mut out = vec![0.0; 4];
        for _ in 0..50 {
            field.apply(&particles, 1.0e-6, 0.01, &mut out);
        }
        assert!(
            field.grid_g.iter().all(|&g| g.abs() < 1e-9),
            "g should stay ~0 when mu==mu_s everywhere"
        );
    }

    #[test]
    fn g_bootstraps_away_from_zero_once_mu_exceeds_mu_s() {
        // mu=0.90 > mu_s=0.70: g=0 is an unstable fixed point here, and a
        // zero start with no seed would stay at 0 forever. Confirm g grows
        // and converges toward eq. 4's local fluidity
        // g_loc = (mu-mu_s)/(b*mu) * sqrt(P/(rho_s*d^2)).
        let cfg = test_config();
        let pressure = 1.0e5f32;
        let mu = 0.90f32;
        let mut field = GranularFluidityField::new(cfg, |_p| (1.0e5, 0.90), 8);
        let particles = Particles::from(vec![
            test_particle_at(Vec2::new(4.0, 4.0)),
            test_particle_at(Vec2::new(4.0, 4.0)),
            test_particle_at(Vec2::new(4.0, 4.0)),
            test_particle_at(Vec2::new(4.0, 4.0)),
        ]);
        // Growing from the small seed takes time: near g ~ 0 the growth is
        // exponential at rate (mu-mu_s)/t0, so t_converge ~ t0/(mu-mu_s) *
        // ln(g_loc/seed) ~ 1e-4/0.2 * ln(1.7e4/1e-6) ~ 12 ms. Run 30 ms, well
        // inside the stability bound at this dx/t0/A/d (~0.24 s,
        // `stability_dt_matches_the_cited_formula_directly`).
        let mut out = vec![0.0; 4];
        for _ in 0..3000 {
            field.apply(&particles, 1.0e-5, 0.01, &mut out);
        }
        let g_eq = (mu - cfg.mu_s) / (cfg.b * mu)
            * (pressure / (cfg.grain_density_kg_m3 * cfg.grain_diameter_m.powi(2))).sqrt();
        assert!(out[0] > 0.0, "g should have bootstrapped away from zero");
        let relative_error = (out[0] - g_eq).abs() / g_eq;
        assert!(
            relative_error < 0.2,
            "g={} should have converged near g_eq={g_eq} (rel. error {relative_error})",
            out[0]
        );
    }

    /// A grid full of material at one pressure and stress ratio settles, in
    /// its interior, to eq. 4's local fluidity at every pressure, and the
    /// fluidity vanishes as the pressure does (issue #51: the inverted
    /// square root made it diverge as `1/sqrt(P)` instead). The pressure is
    /// carried on `internal_pressure` so one plain fn reads it.
    #[test]
    fn steady_local_fluidity_matches_eq_4_down_to_zero_pressure() {
        let cfg = test_config();
        let mu = 0.90f32;
        const RES: usize = 16;
        for pressure in [0.0f32, 1.0e-3, 1.0, 1.0e2, 1.0e5] {
            let mut field = GranularFluidityField::new(cfg, |p| (p.internal_pressure, 0.90), RES);
            let mut particles = Vec::new();
            for i in 0..2 * RES {
                for j in 0..2 * RES {
                    let mut p =
                        test_particle_at(Vec2::new(0.25 + 0.5 * i as f32, 0.25 + 0.5 * j as f32));
                    p.internal_pressure = pressure;
                    particles.push(p);
                }
            }
            let particles = Particles::from(particles);
            let mut out = vec![0.0; particles.len()];
            for _ in 0..3000 {
                field.apply(&particles, 1.0e-5, 0.01, &mut out);
            }
            let center = RES / 2 * RES + RES / 2;
            let g = field.grid_g[center];
            let g_loc = (mu - cfg.mu_s) / (cfg.b * mu)
                * (pressure / (cfg.grain_density_kg_m3 * cfg.grain_diameter_m.powi(2))).sqrt();
            println!("P = {pressure} Pa: g = {g}, eq. 4 g_loc = {g_loc}");
            if pressure == 0.0 {
                assert_eq!(g, 0.0, "no confinement, no fluidity");
            } else {
                let relative_error = (g - g_loc).abs() / g_loc;
                assert!(
                    relative_error < 1.0e-3,
                    "P = {pressure} Pa: g = {g} against eq. 4's {g_loc} (rel. error {relative_error})"
                );
            }
        }
    }
}

//! Gas-state materials — real compressible ideal-gas equation of state,
//! genuinely different from a liquid's Tait EOS: pressure vanishes as
//! density does (no rest-pressure offset), and real shock-capturing
//! matters far more than for a weakly-compressible liquid.
//!
//! `GasMaterial` (`ideal_gas.rs`) landed 2026-08-18: `p=ρRT`, real
//! adiabatic sound speed `c=√(γRT)`, von Neumann-Richtmyer shock
//! viscosity reusing the same shared q-formula
//! (`matter::materials::utils::von_neumann_richtmyer_q`) liquids use, fed
//! this material's own real γ instead of a Tait-derived stand-in. CPU
//! only — `p2g.wgsl`/`particles_update.wgsl` have no `case 13u` branch
//! yet, a real, disclosed limitation (see `GasMaterial`'s own doc and
//! `ConstitutiveModel::Gas`'s doc), not yet verified against Sod's shock
//! tube (Toro, *Riemann Solvers and Numerical Methods for Fluid
//! Dynamics* — the standard exact-analytical-solution benchmark for a
//! compressible-gas solver; needs an iterative Riemann solver, real
//! future work). See the design artifact for the full taxonomy plan:
//! <https://claude.ai/code/artifact/90290560-8992-4d7c-ae4b-11ede12a737f>

pub mod ideal_gas;
pub use ideal_gas::GasMaterial;

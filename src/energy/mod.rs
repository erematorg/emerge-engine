//! Energy domain: how it flows and transforms.
//!
//! `thermodynamics` — `ThermalDiffusion` (Fourier heat), `ScalarDiffusionField`
//! (generic reaction-diffusion: pheromone, nutrients, morphogen). `acoustics`
//! [feature = "experimental"] — `WaveEquation2D`, pressure-wave propagation.
//! `electromagnetics` [feature = "experimental"] — `ElectromagneticWave`,
//! optical `MaterialProperties` (refractive index, permittivity/permeability);
//! the point-charge force-application half lives in `forces::electromagnetics`
//! instead. `orbital` [feature = "experimental"] — real Earth rotation +
//! axial-tilt-driven sun direction (`OrbitalClock`), the real cause of a
//! day/night + seasonal cycle, first real step toward replacing the existing
//! arbitrary thermal day/night oscillation.
//!
//! Part of the emerge/LP domain taxonomy (matter/forces/energy/information/
//! spacetime/organism/systems) -- see `project_domain_taxonomy` design notes.
//! Re-exported at crate root (`pub use energy::thermodynamics;` etc in
//! `lib.rs`) so every existing `crate::thermodynamics::`/`crate::acoustics::`
//! path and every LP `emerge::thermodynamics::` path keeps resolving
//! unchanged -- this move only changes where the files physically live, not
//! any public API.

#[cfg(feature = "experimental")]
pub mod acoustics;
#[cfg(feature = "experimental")]
pub mod electromagnetics;
#[cfg(feature = "experimental")]
pub mod orbital;
pub mod thermodynamics;

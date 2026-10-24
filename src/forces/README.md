# Forces

**What acts on matter from outside, and the laws of walls and contact.**

The split that defines this folder is the momentum balance itself,
`div(sigma_internal) + f_external = rho a`: a material resisting its own
deformation is internal stress and belongs to `matter`; a body force applied
from outside, and the conditions a wall imposes, belong here. How a solver
enforces these laws on its discrete unknowns belongs to `spacetime`.

## Core API

- `Field` trait: a body force on particle `i`.
  - `acceleration`: an ordinary force per unit mass.
  - `gyration_rate`: a force that turns the velocity without doing work
    (`dv/dt = omega J v`). The solver integrates it with the Boris rotation so
    it cannot create energy (see `fields::gyroscopic`).
- Fields:
  - gravity: `NBodyGravityField` (Barnes-Hut, quadrupole), `GravityWellField`;
  - electromagnetism: `CoulombField`, `UniformElectricField` (F = qE),
    `UniformMagneticField` (magnetic Lorentz force, gyroscopic);
  - rotating frame: `CoriolisField` (gyroscopic);
  - surrounding medium: `BuoyancyField`, `LinearDragField`, `SpatialDragField`.
- `BoundaryCondition` trait: walls. `SlipBoundary`, `FrictionBoundary`,
  `RatchetFrictionBoundary`, `HeightmapBoundary`, `KinematicCircleBoundary`,
  `StaticBoxBoundary`. A wall removes the velocity that would enter it at the
  grid nodes inside it (grid node `i` sits at `i + 0.5`) and applies Coulomb
  friction to the tangential part. Each reports the energy its friction
  dissipates, which `energy::thermodynamics::frictional_heating` turns into heat.
- Friction laws: `apply_coulomb_wall` (isotropic) and
  `apply_anisotropic_coulomb_wall` (forward/backward coefficients, Hu et al.
  2009), shared by the walls and by `DirectionalContactGrip`, the directional
  law for body-to-body multi-field contact (resolved in `spacetime::grid::contact`).
- `interphase`: momentum exchange between the phases of a mixture.
  `darcy_exchange_coefficient` (Tampubolon et al. 2017, eq. 13) from a soil's
  measured porosity and hydraulic conductivity;
  `SimConfig::mixture_drag_from_darcy` converts it for the mixture solve.
- `electromagnetics` [feature = "experimental"]: point-charge and current
  field-query math. Not wired to particles.

## Scope & Limits

- Internal material stress is out of scope (`matter`).
- Field forces are applied to particles after the grid-to-particle transfer,
  after the walls, while `SimConfig::gravity` is applied on the grid before
  them. A field pushing into a wall is therefore not constrained by it within
  that substep. Moving field forces onto the grid momentum balance is open.
- The energy removed by the normal part of a wall correction (an inelastic
  impact at the node) is not reported anywhere; only the tangential friction
  work is. It is a discretization loss, not modelled physics.
- `electromagnetics.rs` uses the three-dimensional laws (1/r^2); a
  two-dimensional point charge has a 1/r field (McDonald, "Electrodynamics in 1
  and 2 Spatial Dimensions", eq. 9). Not reconciled yet; the module is unwired.
- `BuoyancyField` and `LinearDragField` describe a surrounding medium that is
  not simulated. If the fluid is simulated, buoyancy already comes from its
  pressure; adding the field counts it twice.

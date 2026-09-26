# Forces

**What pushes on matter from outside.**

Gravity, drag, buoyancy, magnetic attraction -- and walls, which push back
when something runs into them. The distinction that defines this folder: a
material resisting its own deformation is internal stress and belongs to
`matter`; anything acting on a body from outside belongs here. That is the
real split in the momentum balance, `div(sigma_internal) + f_external = rho*a`,
not a filing convenience.

Friction lives here too, and it now reports the energy it destroys so that
`energy` can turn it into heat -- a rubbing contact that silently deleted
kinetic energy was breaking the first law of thermodynamics.

What acts on matter from outside itself -- external/applied forces, distinct from the internal constitutive stress `matter` computes. Matches the real momentum-balance split (∇·σ_internal + f_external = ρa).

## Core API

- `Field` trait + 10 impls: `NBodyGravityField` (Barnes-Hut), `GravityWellField`, `CoulombField`, `UniformElectricField`, `AabbConfinementField`, `RadialConfinementField`, `BuoyancyField`, `ChemotaxisField` (Keller-Segel), `LinearDragField`, `SpatialDragField`.
- `BoundaryCondition` trait + 6 impls: `SlipBoundary`, `FrictionBoundary`, `GripFrictionBoundary`, `RatchetFrictionBoundary`, `HeightmapBoundary`, `KinematicCircleBoundary`. Each reports the specific kinetic energy its correction dissipates as friction, which `energy::thermodynamics::frictional_heating` turns into heat. Only the tangential (Coulomb) part is reported: the normal, no-penetration part of an impact is a modelling choice this engine does not make.
- `electromagnetics` [feature = "experimental"] -- point-charge/current field-query math (E/B at a point). The wave-propagation/optical half lives in `energy::electromagnetics` instead -- a deliberate split, not duplication.

## Scope & Limits

- Internal material stress (what a substance's own deformation produces) is out of scope here -- that's `matter::materials::MaterialModel::kirchhoff_stress`. This directory is exclusively forces applied from outside a body.
- `electromagnetics` is feature-gated experimental; not part of the LP-stable API surface.

## Status

Production-stable. Barnes-Hut N-body gravity is real (not brute-force O(N²)) and tested at scale.

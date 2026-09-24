# Forces

What acts on matter from outside itself — external/applied forces, distinct from the internal constitutive stress `matter` computes. Matches the real momentum-balance split (∇·σ_internal + f_external = ρa).

## Core API

- `Field` trait + 10 impls: `NBodyGravityField` (Barnes-Hut), `GravityWellField`, `CoulombField`, `UniformElectricField`, `AabbConfinementField`, `RadialConfinementField`, `BuoyancyField`, `ChemotaxisField` (Keller-Segel), `LinearDragField`, `SpatialDragField`.
- `BoundaryCondition` trait + 6 impls: `SlipBoundary`, `PredictiveBoundary`, `FrictionBoundary`, `GripFrictionBoundary`, `RatchetFrictionBoundary`, `HeightmapBoundary`.
- `electromagnetics` [feature = "experimental"] — point-charge/current field-query math (E/B at a point). The wave-propagation/optical half lives in `energy::electromagnetics` instead — a deliberate split, not duplication.

## Scope & Limits

- Internal material stress (what a substance's own deformation produces) is out of scope here — that's `matter::materials::MaterialModel::kirchhoff_stress`. This directory is exclusively forces applied from outside a body.
- `electromagnetics` is feature-gated experimental; not part of the LP-stable API surface.

## Status

Production-stable. Barnes-Hut N-body gravity is real (not brute-force O(N²)) and tested at scale.

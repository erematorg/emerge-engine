# Matter

Material phase states and transitions — bridge layer to the emerge physics engine.

## Role

Matter's only job is deciding *when* and *why* phase transitions happen (e.g. ice → water at T > 0°C),
then calling `emerge::MpmSolver::phase_transition()` to apply the change.
The solver, constitutive models, and density/viscosity all live in emerge.

## Scope & Limits

- Phase transition logic: LP-owned (thermodynamic conditions, material state machines)
- Constitutive models (fluid, elastic, snow, sand): emerge-owned, not duplicated here
- MPM blocker resolved: [emerge](https://github.com/erematorg/emerge) is stable and integrated

## Status

**VESTIGIAL** — nearly empty pending emerge wiring. Do not add new physics here.
New material behavior belongs in emerge's `MaterialModel` trait implementations.

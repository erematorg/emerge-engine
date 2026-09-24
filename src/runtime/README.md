# Runtime

How the simulation is driven over real wall-clock time — a genuinely distinct concern from the physics itself, small enough to stay a single module.

## Core API

- `FixedStepController` / `FixedStepConfig` — decouples real frame rate from a fixed physics `dt` via the standard accumulator pattern (accumulate real elapsed time, step the sim in fixed increments, carry remainder forward). Used across every GPU demo so physics behavior doesn't depend on display refresh rate.

## Scope & Limits

Frame-pacing only — no physics content. Does not decide *how much* physics to run per step (that's `SimConfig`/adaptive substeps), only *when*.

## Status

Production-stable, unchanged since rollout across all 9 GPU demos.

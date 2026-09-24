# Information

What senses, decides, and remembers **in** the simulated world — not tooling that observes it from outside (see `systems::diagnostics` for that).

## Core API

- `control::Lnn` — Liquid Time-constant Network locomotion controller. A genome/weight vector is itself information; this is a real in-world decision-making system, not an authoring tool.
- `measures` [feature = "experimental"] — Shannon entropy (discrete + continuous k-NN, Kraskov et al. 2004), mutual information (discrete + continuous k-NN, conditional), KL divergence. Applied to real simulated quantities (spatial/kinetic/phase distributions), O(N).

## Scope & Limits

- Strictly in-world: a `FrameLogger` or NDJSON stats collector has no physical law behind it and belongs in `systems`, not here, even though both could loosely be called "information."
- `measures` is feature-gated experimental.

## Status

`Lnn` is production — a real gradient-trained biped walker exists via the `spacetime::diff` adjoint chain. `measures` is experimental but functionally complete (no external dependencies, pure Rust).

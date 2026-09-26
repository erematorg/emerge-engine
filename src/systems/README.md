# Systems

**The machinery, not the physics.**

GPU compute, rendering, diagnostics, logging. This is the one folder with no
physical law behind it, and that exception is deliberate: everything without a
law lives here rather than being scattered into domains that would then be
pretending.

The rule that keeps it honest: `systems` **consumes** physics and never owns
any. The renderer applies Beer-Lambert, but the law lives in `energy` and the
coefficients live in `matter`.

Pure orchestration with no IRL counterpart of its own -- the deliberate exception to every other domain's IRL-grounding mandate, carved out on purpose. If it doesn't correspond to a physical law, it lives here, not scattered into a domain that pretends it does.

## Core API

- `diagnostics` -- health monitoring, NDJSON logging (`FrameLogger`, `log_frame*`), plugin-based stats collection (`DiagnosticsRegistry`, `RollingPlugin`, per-material stats), `MpmSnapshot`/`SimSnapshot`.
- `gpu` [feature = "gpu"] -- `GpuSimulation` + WGSL compute shaders (P2G/G2P/grid_clear/grid_update/force_fields/muscle), sparse-grid active-block dispatch on grid_clear+grid_update.
- `render` [feature = "render"] -- instanced particle renderer, N-material surface reconstruction, curvature flow, and the shader mirrors of `energy::radiation` (`blackbody.inc.wgsl`, `radiative_transfer.inc.wgsl`). A fragment shader cannot integrate a spectrum per pixel, so each mirror approximates a law from `energy` and a test holds it to within a stated tolerance of the exact form.

## Scope & Limits

Engineering plumbing only. A `FrameLogger` is categorically the same kind of thing as GPU/render backend code -- neither has a physical law behind it -- which is why both live here rather than being folded into `information` (which is strictly in-world) or scattered per-domain.

## Status

Production-stable. GPU path covers the full P2G→grid update→G2P cycle plus force fields and muscle activation; CPU remains the correctness reference (CPU correctness first, GPU port second, per standing project rule).

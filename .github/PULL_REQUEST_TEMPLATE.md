## What does this change?

<!-- One or two sentences: what it does and why. -->

## Type of change

- [ ] Bug fix
- [ ] New material / physics feature
- [ ] Refactor (no behavior change)
- [ ] Documentation
- [ ] Other

## Checklist

- [ ] `cargo test` passes
- [ ] `cargo clippy -- -D warnings` is clean
- [ ] No `#[allow(...)]` added to silence a warning
- [ ] If this adds or changes numerical physics: the source paper/reference is cited in the code, and a test in `tests/physics_correctness.rs` or `tests/accuracy.rs` checks it against a known limit case, conservation law, or measured value -- not just "doesn't crash"
- [ ] If this adds a material: all 4 steps in [CONTRIBUTING.md's "Adding a material model"](../CONTRIBUTING.md#adding-a-material-model) are done (trait impl, `ConstitutiveModel` variant, both WGSL shader cases)

## Anything a reviewer should know?

<!-- Design tradeoffs, things you're unsure about, what you deliberately left out. -->

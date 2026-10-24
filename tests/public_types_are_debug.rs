//! Compile-time check that public types implement `Debug` (Rust API
//! Guidelines C-DEBUG) and that public error types implement
//! `std::error::Error` (C-GOOD-ERR). The test body only names the types:
//! a missing impl is a build error, not a runtime failure.

extern crate emerge_engine as emerge;

fn assert_debug<T: std::fmt::Debug>() {}
fn assert_error<T: std::error::Error>() {}

#[test]
fn public_types_are_debug() {
    // Body force fields.
    assert_debug::<emerge::fields::CoulombField>();
    assert_debug::<emerge::fields::GravityWellField>();
    assert_debug::<emerge::fields::LinearDragField>();
    assert_debug::<emerge::fields::NBodyGravityField>();
    assert_debug::<emerge::fields::SpatialDragField>();
    assert_debug::<emerge::fields::UniformElectricField>();
    assert_debug::<emerge::fields::gyroscopic::UniformMagneticField>();

    // Thermal and scalar fields.
    assert_debug::<emerge::thermodynamics::CosseratField>();
    assert_debug::<emerge::thermodynamics::GranularFluidityField>();
    assert_debug::<emerge::thermodynamics::ScalarDiffusionField>();
    assert_debug::<emerge::thermodynamics::ThermalDiffusion>();

    // Particle storage.
    assert_debug::<emerge::particle::Particles>();
    assert_debug::<emerge::particle::ParticleUpdateCtx<'static>>();
    assert_debug::<emerge::particle::ParticlesIter<'static>>();

    // Differentiable stepping.
    assert_debug::<emerge::diff::BodyPlan>();
    assert_debug::<emerge::diff::DiffConfig>();
    assert_debug::<emerge::diff::DiffState>();
    assert_debug::<emerge::diff::FeedbackController>();
    assert_debug::<emerge::diff::GaitMetrics>();
    assert_debug::<emerge::diff::SinusoidController>();
    assert_debug::<emerge::diff::StepRecord>();
    assert_debug::<emerge::diff::StressEval>();

    // Grains, grid, rods, transfers.
    assert_debug::<emerge::grains::GrainPopulation>();
    assert_debug::<emerge::grid::FxU32BuildHasher>();
    assert_debug::<emerge::grid::FxU32Hasher>();
    assert_debug::<emerge::rod::RodForceParams>();
    assert_debug::<emerge::rod::RodRestState<'static>>();
    assert_debug::<emerge::transfer::G2PParams<'static>>();
    assert_debug::<emerge::transfer::P2GParticleState>();

    // Diagnostics.
    assert_debug::<emerge::diagnostics::ActivationStatsPlugin>();
    assert_debug::<emerge::diagnostics::DiagnosticsRegistry>();
    assert_debug::<emerge::diagnostics::FrameLogger>();
    assert_debug::<emerge::diagnostics::MaterialCountPlugin>();
    assert_debug::<emerge::diagnostics::PositionResolutionPlugin>();
    assert_debug::<emerge::diagnostics::RollingPlugin>();
    assert_debug::<emerge::diagnostics::ThermalStatsPlugin>();

    // Errors.
    assert_error::<emerge::materials::relaxation::RelaxationError>();
}

#[cfg(feature = "experimental")]
#[test]
fn experimental_public_types_are_debug() {
    assert_debug::<emerge::energy::electromagnetics::DielectricBreakdownLeader>();
    assert_debug::<emerge::energy::electromagnetics::ElectricPotentialField>();
    assert_debug::<emerge::energy::electromagnetics::ElectromagneticWave>();
    assert_debug::<emerge::measures::KLDivergence>();
    assert_debug::<emerge::grid::mac::Liquid<'static>>();
    assert_debug::<emerge::grid::mac::Walls<'static>>();
    assert_debug::<emerge::grid::mac::extrapolate::GhostPressure>();
    assert_debug::<emerge::grid::mac::multigrid::Multigrid>();
    assert_debug::<emerge::grid::mac::pressure::Surface<'static>>();
}

#[cfg(feature = "gpu")]
#[test]
fn gpu_public_types_are_debug() {
    assert_debug::<emerge::gpu::GpuImpulseEntry>();
    assert_debug::<emerge::gpu::GpuImpulseParams>();
    assert_debug::<emerge::gpu::GpuSimulation>();
    assert_debug::<emerge::gpu::GpuSleepWakeParams>();
    assert_debug::<emerge::gpu::buffers::GpuBuffers>();
    assert_debug::<emerge::gpu::pipeline::SimPipelines>();
}

#[cfg(feature = "render")]
#[test]
fn render_public_types_are_debug() {
    assert_debug::<emerge::render::CpuRenderBridge>();
    assert_debug::<emerge::render::DualPhaseSurfaceSource<'static>>();
    assert_debug::<emerge::render::GpuRenderParams<'static>>();
    assert_debug::<emerge::render::GridVolumeSource<'static>>();
    assert_debug::<emerge::render::Renderer>();
    assert_debug::<emerge::render::SurfaceReconstructionSource<'static>>();
}

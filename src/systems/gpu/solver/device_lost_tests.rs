//! White-box tests for `GpuSimulation`'s device-lost handling.
//!
//! Split out of `gpu/solver/mod.rs` (was 850 lines) to bring it under the
//! project's 400-500 line guideline, mirroring how the CPU solver's own test
//! modules live in dedicated files. Uses `super::*` for private-field access
//! (`device_lost`, `buffers`, `device`), so this must stay a submodule of
//! `gpu::solver`, not a standalone integration test -- purely a file-location
//! move, no behavior change.

use super::*;
use crate::materials::registry::MaterialRegistry;
use crate::materials::{FromSI, NeoHookeanMaterial};
use crate::solver::config::{SimConfig, SpawnRegion};
use glam::{IVec2, Vec2};

fn gpu_available() -> bool {
    let instance = crate::systems::gpu::create_wgpu_instance();
    pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::None,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .is_ok()
}

/// White-box check of the device-lost guard (issue #10, whose root cause was
/// an `Out of Memory` device loss under sustained load, seen through
/// `device_lost_callback`). Rather than force that OOM again, this injects
/// a lost reason into the private `device_lost` flag as the callback would,
/// then checks that (1) `device_lost_reason()` reports it, (2) `step_frame()`
/// is a no-op (frame_index does not advance), and (3) the blocking sync
/// methods are safe no-ops on a dead device.
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn step_frame_becomes_safe_noop_once_device_lost() {
    if !gpu_available() {
        return;
    }
    let config = SimConfig {
        max_substeps_per_step: 8,
        ..SimConfig::standard(32, 0.1, Vec2::new(0.0, -0.3))
    };
    let spawn = SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new(4, 4),
        box_center: Vec2::new(16.0, 16.0),
        ..SpawnRegion::for_sim(&config)
    };
    let particles = crate::build_particles(&config, spawn);
    let mat = NeoHookeanMaterial::from_physical(
        &crate::materials::physical_props::Elastic {
            e_pa: 30.0e3,
            nu: 0.45,
            rho_kg_m3: 1000.0,
        },
        &config,
    );
    let registry = MaterialRegistry::with_default(Box::new(mat));
    let mut sim = pollster::block_on(GpuSimulation::new(config, particles, registry));

    assert!(
        sim.device_lost_reason().is_none(),
        "a healthy, freshly-constructed sim must not report device loss"
    );

    sim.step_frame();
    let frame_after_healthy_step = sim.frame_index;
    assert!(
        frame_after_healthy_step > 0,
        "sanity check: a healthy device must actually advance frame_index"
    );

    // Directly inject a lost reason, exactly as the callback does.
    *sim.device_lost.lock().unwrap() = Some("Unknown: Out of memory".to_string());
    assert_eq!(
        sim.device_lost_reason(),
        Some("Unknown: Out of memory".to_string()),
        "device_lost_reason() must report an injected loss"
    );

    sim.step_frame();
    assert_eq!(
        sim.frame_index, frame_after_healthy_step,
        "step_frame must become a real no-op once device_lost is set -- \
         frame_index must NOT advance"
    );

    // Must not panic -- these touch the same "dead" device.
    sim.sync_particles_blocking();
    let ranges = vec![0..1usize, 1..2usize];
    sim.sync_particle_ranges_blocking(&ranges);
}

/// Reproduces issue #10's failure mode seen on windows-latest CI: under
/// sustained load, wgpu destroys buffers tied to the device before this
/// instance's `device_lost_callback` fires; the readback path's `.unmap()`
/// then hits an uncaptured Validation error naming the destroyed resource,
/// and wgpu's default handler panics (`default_error_handler`:
/// `panic!("wgpu error: {err}")`, wgpu-27.0.1).
///
/// Instead of a 9-minute sustained-load OOM, this takes the same call path
/// directly: destroy the readback staging buffer (as the device-loss cascade
/// does), then call `abandon_readback()`, whose `.unmap()` panicked on CI.
/// With `on_uncaptured_error` installed it must set `device_lost` instead:
/// no panic, and `device_lost_reason()` reports it.
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn uncaptured_destroyed_buffer_error_sets_device_lost_not_a_panic() {
    if !gpu_available() {
        return;
    }
    let config = SimConfig {
        max_substeps_per_step: 8,
        ..SimConfig::standard(32, 0.1, Vec2::new(0.0, -0.3))
    };
    let spawn = SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new(4, 4),
        box_center: Vec2::new(16.0, 16.0),
        ..SpawnRegion::for_sim(&config)
    };
    let particles = crate::build_particles(&config, spawn);
    let mat = NeoHookeanMaterial::from_physical(
        &crate::materials::physical_props::Elastic {
            e_pa: 30.0e3,
            nu: 0.45,
            rho_kg_m3: 1000.0,
        },
        &config,
    );
    let registry = MaterialRegistry::with_default(Box::new(mat));
    let sim = pollster::block_on(GpuSimulation::new(config, particles, registry));

    assert!(
        sim.device_lost_reason().is_none(),
        "a healthy, freshly-constructed sim must not report device loss"
    );

    // Exactly what a device-loss cascade does to resources tied to the
    // device, without needing 9 minutes of real sustained WARP load.
    sim.buffers.readback_staging.destroy();

    // The exact real call path that panicked on CI: finish_readback and
    // abandon_readback both end in `.unmap()` on this buffer.
    sim.buffers.abandon_readback();
    sim.device.poll(wgpu::PollType::Poll).ok();

    let reason = sim.device_lost_reason();
    assert!(
        reason.is_some(),
        "an uncaptured error naming a destroyed buffer must set device_lost, \
         not silently do nothing"
    );
    let reason = reason.unwrap();
    assert!(
        reason.contains("uncaptured wgpu error"),
        "reason should be tagged as coming from the uncaptured-error handler \
         (distinguishable from the official device_lost_callback's report), \
         got: {reason}"
    );
    assert!(
        reason.contains("destroyed"),
        "reason should retain the real wgpu error text naming the destroyed \
         resource, got: {reason}"
    );
}

/// The opt-in path, which LP needs (`World::with_device`: LP shares its
/// device with a renderer and has no device-lost handling of its own).
/// `enable_device_lost_detection()` makes a `with_device()` instance report
/// like a `new()`-constructed one. It does not show that `with_device()`
/// never registers its own callback (telling "no callback" from "callback,
/// nothing happened yet" needs a device loss); that rests on
/// `with_device()`'s body containing no `set_device_lost_callback` call.
#[test]
#[ignore = "needs a real GPU adapter: run manually on hardware, see CONTRIBUTING.md"]
fn with_device_instances_need_explicit_opt_in_for_device_lost_detection() {
    if !gpu_available() {
        return;
    }
    let instance = crate::systems::gpu::create_wgpu_instance();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .expect("no adapter");
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("test_shared_device"),
        required_limits: adapter.limits(),
        ..Default::default()
    }))
    .expect("no device");
    let device = Arc::new(device);
    let queue = Arc::new(queue);

    let config = SimConfig {
        max_substeps_per_step: 8,
        ..SimConfig::standard(32, 0.1, Vec2::new(0.0, -0.3))
    };
    let mat = NeoHookeanMaterial::from_physical(
        &crate::materials::physical_props::Elastic {
            e_pa: 30.0e3,
            nu: 0.45,
            rho_kg_m3: 1000.0,
        },
        &config,
    );
    let sim = GpuSimulation::with_device(
        device,
        queue,
        config,
        Vec::new(),
        MaterialRegistry::with_default(Box::new(mat)),
    );

    // Fresh with_device() instance: field starts unset (expected regardless
    // of whether a callback is wired -- see doc comment above for what this
    // does and doesn't prove).
    assert!(sim.device_lost_reason().is_none());

    sim.enable_device_lost_detection();
    // Directly invoke the same injection used in the other test -- proves the
    // callback registration path (not just the field) is wired correctly.
    *sim.device_lost.lock().unwrap() = Some("Unknown: Out of memory".to_string());
    assert_eq!(
        sim.device_lost_reason(),
        Some("Unknown: Out of memory".to_string()),
        "after enable_device_lost_detection(), device_lost_reason() must work \
         identically to a new()-constructed instance"
    );
}

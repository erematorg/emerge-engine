//! Particle-mutation API on `GpuSimulation`: material reassignment and
//! impulses.
//!
//! Split out of `gpu/solver/mod.rs` -- self-contained particle mutation,
//! not raw wgpu command encoding (impulses queue into `pending_impulses`,
//! applied by a dedicated compute pass the next `step_frame` call -- see
//! `step.rs`). Mirrors the CPU `solver::particles` split.

use crate::particle::Particle;

use super::super::step_params::{GpuImpulseEntry, MAX_GPU_IMPULSES};
use super::GpuSimulation;

impl GpuSimulation {
    /// Reassign material for all particles matching `predicate`. Marks dirty so GPU
    /// sees the change on the next `step_frame` call.
    ///
    /// If `new_material_id`'s `MaterialModel::latent_heat()` is non-zero and
    /// `attach_thermal_gpu` has been called, debits `temperature` by
    /// `latent_heat / heat_capacity` for every transitioned particle -- real energy
    /// conservation (an endothermic transition genuinely cools the particle, exothermic
    /// warms it), not a free material swap. Exact GPU parity with CPU's
    /// `Simulation::phase_transition` -- see `MaterialModel::latent_heat` for the sign
    /// convention. `None` heat capacity (no thermal model attached) skips the debit,
    /// matching CPU's identical `self.thermal.is_none()` gate.
    ///
    /// Calls `sync_particles_blocking` first: a predicate driven by live GPU state
    /// (temperature from the real diffusion PDE, position, velocity) needs the
    /// genuinely current mirror, not whatever the last readback happened to hold --
    /// the same staleness class of bug `remove_particles` guards against for the
    /// same reason (see its own doc).
    ///
    /// Uploads its own change to GPU immediately, not just `layout_dirty = true`
    /// deferred to the next `step_frame` -- a caller chaining multiple mutating calls
    /// back-to-back without an intervening `step_frame` (e.g. melt-check, then
    /// freeze-check, then `remove_particles` for evaporation) would otherwise have the
    /// SECOND call's own `sync_particles_blocking` re-download the GPU's still-stale
    /// pre-transition state, silently erasing the FIRST call's material_id change.
    /// `remove_particles` already uploads immediately (it has to, since it
    /// reallocates); this keeps `phase_transition` consistent with that contract.
    pub fn phase_transition<F>(&mut self, predicate: F, new_material_id: u32)
    where
        F: Fn(&Particle) -> bool,
    {
        assert!(
            self.registry.is_registered(new_material_id),
            "phase_transition: material_id {new_material_id} is not registered -- \
             call solver.set_material({new_material_id}, ...) first"
        );
        self.sync_particles_blocking();
        let new_material = self.registry.get(new_material_id);
        let heat_capacity = self.thermal_heat_capacity;
        for p in self.particles.iter_mut() {
            if predicate(p) {
                // Captured BEFORE being overwritten below -- same real,
                // general multi-source `latent_heat` extension the CPU
                // path uses (`spacetime::solver::particles::
                // apply_phase_transition`'s own doc), so a GPU-driven
                // transition also picks the right real energy for
                // whichever material this specific particle is actually
                // coming from, not one flat value for every source.
                let from_material_id = p.material_id;
                p.material_id = new_material_id;
                let latent_heat = new_material.latent_heat(from_material_id);
                if let (true, Some(cp)) = (latent_heat != 0.0, heat_capacity) {
                    p.temperature -= latent_heat / cp;
                }
            }
        }
        self.buffers.upload_particles(&self.queue, &self.particles);
        self.layout_dirty = true; // material_id changed -- sort order may differ
    }

    /// Add `force` to every particle within `radius` cells of `center`, scaled by proximity.
    /// Applied on the GPU at the start of the next step_frame -- reads LIVE GPU positions,
    /// avoiding any stale-CPU-mirror artifacts. No CPU particle scan.
    pub fn apply_impulse(&mut self, center: glam::Vec2, radius: f32, force: glam::Vec2) {
        if self.pending_impulses.len() < MAX_GPU_IMPULSES {
            self.pending_impulses.push(GpuImpulseEntry {
                center: center.to_array(),
                radius,
                strength: 0.0,
                force: force.to_array(),
                mode: 1,
                _pad: 0,
            });
        } else {
            eprintln!(
                "emerge: GPU impulse queue full ({MAX_GPU_IMPULSES}/frame max) -- impulse dropped"
            );
        }
    }

    /// Push every particle within `radius` cells outward from `center`.
    /// Applied on the GPU at the start of the next step_frame -- reads LIVE GPU positions.
    /// `strength` may be negative to pull. No CPU particle scan.
    pub fn apply_radial_impulse(&mut self, center: glam::Vec2, radius: f32, strength: f32) {
        if self.pending_impulses.len() < MAX_GPU_IMPULSES {
            self.pending_impulses.push(GpuImpulseEntry {
                center: center.to_array(),
                radius,
                strength,
                force: [0.0; 2],
                mode: 0,
                _pad: 0,
            });
        } else {
            eprintln!(
                "emerge: GPU impulse queue full ({MAX_GPU_IMPULSES}/frame max) -- impulse dropped"
            );
        }
    }
}

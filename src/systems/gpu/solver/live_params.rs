//! Live-adjustable GPU simulation parameters (force fields, directional grip,
//! thermal/resource attachment) -- split out of `step.rs` (was ~120 of its
//! ~1000 lines). These are simple setters/attachers with no timing-sensitive
//! device state, unlike `step_frame`/`encode_substep` which stay together in
//! `step.rs` per that file's own "highest-risk, done last and alone" doc.

use super::super::step_params::{
    GpuAsflipParams, GpuFieldEntry, GpuResourceParams, GpuThermalParams, MAX_FORCE_FIELDS,
};
use super::GpuSimulation;

impl GpuSimulation {
    /// Add a non-uniform body force field for the GPU path.
    /// Entries are uploaded and dispatched every substep until cleared.
    /// Panics if `MAX_FORCE_FIELDS` is exceeded.
    pub fn add_force_field_gpu(&mut self, entry: GpuFieldEntry) {
        assert!(
            self.force_field_entries.len() < MAX_FORCE_FIELDS,
            "add_force_field_gpu: MAX_FORCE_FIELDS ({MAX_FORCE_FIELDS}) exceeded"
        );
        self.force_field_entries.push(entry);
    }

    /// Remove all GPU force field entries.
    pub fn clear_force_fields_gpu(&mut self) {
        self.force_field_entries.clear();
    }

    /// Update the preferred grip direction live (e.g. player/AI steering input) --
    /// GPU counterpart to `DirectionalContactGrip::set_easy_direction`. Takes effect
    /// on the very next `step_frame` (re-uploaded fresh every frame, see that
    /// function's own `grip_params` upload). Only has a visible effect on particles
    /// with `contact_group != 0` and only once `set_grip_friction` has set
    /// `mu_easy != mu_resist` -- symmetric friction (the default) makes direction
    /// irrelevant, same "no bias without input" principle as `RatchetFrictionBoundary`.
    ///
    /// HONEST STATUS (2026-07-16): this API is real and correctly wired -- verified
    /// reaching `resolve_contact.wgsl`'s `grip_params` uniform. The RESULTING
    /// directional effect is correctly signed (the "easy" direction genuinely retains
    /// more speed) but its MAGNITUDE is measurably unstable run to run, unlike CPU's
    /// `DirectionalContactGrip` which is consistent -- see
    /// `gpu_directional_grip_is_direction_aware`'s `#[ignore]` reason in `tests/gpu.rs`
    /// for the real measured numbers and the likely (not confirmed) root cause. Usable
    /// for real, but don't present it as equivalent to CPU's steering yet.
    pub fn set_grip_direction(&mut self, direction: glam::Vec2) {
        self.grip_params.easy_direction = direction.normalize_or_zero();
    }

    /// Update the grip friction coefficients live -- GPU counterpart to
    /// `DirectionalContactGrip::set_friction`. Set `mu_easy == mu_resist` to disable
    /// directional asymmetry entirely (ordinary symmetric Coulomb friction) whenever
    /// there's no real steering intent.
    pub fn set_grip_friction(&mut self, mu_easy: f32, mu_resist: f32) {
        assert!(
            (0.0..=1.0).contains(&mu_easy) && (0.0..=1.0).contains(&mu_resist),
            "set_grip_friction: mu_easy and mu_resist must be in [0.0, 1.0]"
        );
        self.grip_params.mu_easy = mu_easy;
        self.grip_params.mu_resist = mu_resist;
    }

    /// Attach day-night/ambient thermal diffusion -- GPU counterpart to CPU's
    /// `Simulation::with_thermal`/`thermal_config_mut`. Real PDE (Fourier's law
    /// `∂T/∂t = α·∇²T` plus Newton cooling), see `GpuThermalParams`' own doc. Enables
    /// all 4 thermal passes starting next `step_frame`; call `set_thermal_ambient`
    /// afterward for live day-night oscillation.
    ///
    /// - `conductivity_w_m_k` / `heat_capacity_j_kg_k`: real SI material constants
    /// - `grid_cell_size_m`: physical cell size (pass `SimConfig::dx_meters`, NOT
    ///   `grid_cell_size` which is always 1.0 -- same trap `ThermalConfig`'s own doc
    ///   warns about)
    /// - `ambient`: background temperature; `cooling_rate`: Newton cooling k_c (1/s,
    ///   0.0 = none)
    pub fn attach_thermal_gpu(
        &mut self,
        conductivity_w_m_k: f32,
        heat_capacity_j_kg_k: f32,
        grid_cell_size_m: f32,
        ambient: f32,
        cooling_rate: f32,
    ) {
        let alpha =
            conductivity_w_m_k / (heat_capacity_j_kg_k * grid_cell_size_m * grid_cell_size_m);
        self.thermal_params = GpuThermalParams {
            alpha,
            ambient,
            cooling_rate,
            enabled: 1,
        };
        // Retained separately for phase_transition's latent-heat debit -- alpha already
        // folds heat_capacity in and can't be recovered back out of it.
        self.thermal_heat_capacity = Some(heat_capacity_j_kg_k);
    }

    /// Update the ambient/boundary temperature live (e.g. day-night oscillation) --
    /// GPU counterpart to CPU's `thermal_config_mut().ambient = ...`. No-op (with a
    /// debug assert) if thermal hasn't been attached yet.
    pub fn set_thermal_ambient(&mut self, ambient: f32) {
        debug_assert!(
            self.thermal_params.enabled != 0,
            "set_thermal_ambient: call attach_thermal_gpu first"
        );
        self.thermal_params.ambient = ambient;
    }

    /// Attach resource regrowth -- GPU counterpart to CPU's `ScalarDiffusionField` +
    /// a logistic-growth `source` (Verhulst 1838, `dφ/dt = r·φ·(1−φ/K)`), see
    /// `GpuResourceParams`' own doc for why this is baked in as the one real source
    /// term rather than staying generic (WGSL has no function pointers, same
    /// disclosed trade-off as `SpatialDragField`'s GPU port). State is carried in
    /// `particle.scalar_field`, NOT `particle.temperature` -- real fix, 2026-07-17:
    /// this and `attach_thermal_gpu` used to both hijack `temperature` as their
    /// carrier, meaning the two could never be attached in the same scene together.
    /// They now use separate fields and compose freely.
    ///
    /// - `diffusivity`: spatial spread rate D, grid-units²/s
    /// - `ambient`: value assigned to empty cells (no particle mass)
    /// - `resource_r`: logistic growth rate r, 1/s
    /// - `resource_k`: logistic carrying capacity K
    pub fn attach_resource_field_gpu(
        &mut self,
        diffusivity: f32,
        ambient: f32,
        resource_r: f32,
        resource_k: f32,
    ) {
        self.resource_params = GpuResourceParams {
            diffusivity,
            ambient,
            resource_r,
            resource_k,
            enabled: 1,
            _pad: [0; 3],
        };
    }

    /// Attach ASFLIP -- GPU counterpart to CPU's `SimConfig::asflip_blend`
    /// (Fei, Guo, Wu, Huang, Gao 2021). Takes effect starting the next `step_frame`:
    /// dispatches the fused `g2p_asflip_fused` pass instead of the ordinary split
    /// g2p/particles_update pair (see `SubstepGates::asflip_active`). `blend` is
    /// clamped to `[0.0, 1.0]` matching CPU's own field doc (~0.97 is the paper's own
    /// reference value, `nepluno/pyasflip`). `blend <= 0.0` behaves as fully disabled
    /// (same real gate as CPU's `asflip_blend > 0.0` check) -- byte-identical to never
    /// having called this at all.
    ///
    /// On first call with `blend > 0.0`, grows `buffers.asflip_snapshot` from its
    /// placeholder to real `grid_res²` size and rebuilds `resource_bind_group` to
    /// point at it -- see `GpuBuffers::asflip_snapshot`'s own doc for why this buffer
    /// is lazily grown instead of always allocated at full size (a real, measured OOM
    /// regression otherwise). Subsequent calls (e.g. re-tuning `blend`) are cheap --
    /// the buffer is already grown, no reallocation or bind-group rebuild happens again.
    pub fn attach_asflip_gpu(&mut self, blend: f32) {
        let blend = blend.clamp(0.0, 1.0);
        self.asflip_params = GpuAsflipParams {
            blend,
            enabled: if blend > 0.0 { 1 } else { 0 },
            _pad: [0; 2],
        };
        if blend > 0.0 && !self.buffers.asflip_snapshot_grown {
            self.buffers
                .grow_asflip_snapshot(&self.device, self.config.grid_res);
            self.resource_bind_group = self
                .pipelines
                .make_resource_bind_group(&self.device, &self.buffers);
        }
    }
}

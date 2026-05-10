use crate::materials::{ConstitutiveModel, MaterialModel, MaterialParams};

/// Maximum number of material slots — matches `MAX_MATERIALS` in WGSL shaders.
/// The GPU uniform buffer holds exactly this many `MaterialParams` entries.
/// CPU accepts any count up to this limit; exceeding it panics to catch silent GPU truncation.
pub const MAX_MATERIAL_SLOTS: usize = 64;

/// Maps material IDs to constitutive models.
/// IDs must be contiguous starting at 0 — index 0 is the default/fallback.
/// The GPU path binds this as a flat `array<MaterialParams>` indexed by material_id.
#[derive(Debug)]
pub struct MaterialRegistry {
    materials: Vec<Box<dyn MaterialModel>>,
}

impl MaterialRegistry {
    pub fn with_default(default_material: Box<dyn MaterialModel>) -> Self {
        Self {
            materials: vec![default_material],
        }
    }

    /// Set material at `material_id`, replacing it if already registered.
    ///
    /// For new IDs, insertion must still be contiguous (0, 1, 2…) — you cannot
    /// skip slots. Replacing an existing ID is always allowed (idempotent update).
    ///
    /// Panics if `material_id >= MAX_MATERIAL_SLOTS` — GPU uniform buffer is fixed-size.
    pub fn insert(&mut self, material_id: u32, material: Box<dyn MaterialModel>) {
        let idx = material_id as usize;
        assert!(
            idx < MAX_MATERIAL_SLOTS,
            "material_id {material_id} exceeds GPU limit of {MAX_MATERIAL_SLOTS} — \
             increase MAX_MATERIAL_SLOTS in material_registry.rs and WGSL shaders together"
        );
        if idx < self.materials.len() {
            self.materials[idx] = material; // replace existing
        } else {
            assert_eq!(
                idx,
                self.materials.len(),
                "material IDs must be registered contiguously starting at 0; \
                 expected id {}, got {}",
                self.materials.len(),
                material_id,
            );
            self.materials.push(material);
        }
    }

    /// Returns the next available material ID (= current count).
    /// Convenience for callers that auto-allocate IDs without tracking them manually.
    pub fn next_id(&self) -> u32 {
        self.materials.len() as u32
    }

    /// Replace the default material (ID 0).
    pub fn set_default(&mut self, material: Box<dyn MaterialModel>) {
        self.materials[0] = material;
    }

    /// Retrieve a material by ID. Falls back to material 0 for unknown IDs.
    ///
    /// In debug builds, triggers an assertion failure on out-of-range IDs so
    /// unregistered materials are caught at the spawn site, not silently muted.
    pub fn get(&self, material_id: u32) -> &dyn MaterialModel {
        debug_assert!(
            (material_id as usize) < self.materials.len(),
            "material_id {material_id} is not registered (only {} materials known)",
            self.materials.len()
        );
        self.materials
            .get(material_id as usize)
            .unwrap_or(&self.materials[0])
            .as_ref()
    }

    pub fn len(&self) -> usize {
        self.materials.len()
    }

    pub fn is_empty(&self) -> bool {
        self.materials.is_empty()
    }

    /// Returns true if `material_id` is a registered slot (not an out-of-range index).
    pub fn is_registered(&self, material_id: u32) -> bool {
        (material_id as usize) < self.materials.len()
    }

    /// Returns true if any registered material requires a CPU plasticity pass each substep.
    /// Used by the GPU solver to skip the download+update loop when all plasticity is on GPU.
    pub fn any_needs_cpu_update(&self) -> bool {
        self.materials.iter().any(|m| m.needs_cpu_update())
    }

    /// Returns true if any registered material requires per-substep density recompute.
    /// Fluid EOS materials need up-to-date density; elastic/plastic materials do not.
    pub fn any_needs_density_recompute(&self) -> bool {
        self.materials.iter().any(|m| m.needs_density_recompute())
    }

    /// Returns the constitutive model for the given material ID.
    pub fn constitutive_model_of(&self, material_id: u32) -> ConstitutiveModel {
        self.get(material_id).constitutive_model()
    }

    /// Returns flat parameters for all registered materials in ID order.
    /// Used to upload a `array<MaterialParams, N>` uniform buffer to the GPU.
    pub fn all_params(&self) -> Vec<MaterialParams> {
        self.materials.iter().map(|m| m.params()).collect()
    }
}

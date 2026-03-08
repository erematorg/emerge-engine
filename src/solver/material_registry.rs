use crate::solver::materials::{ConstitutiveModel, MaterialModel, MaterialParams};

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

    /// Register material at `material_id`.
    /// Panics if `material_id != materials.len()` — IDs must be registered in order (0, 1, 2…).
    pub fn insert(&mut self, material_id: u32, material: Box<dyn MaterialModel>) {
        assert_eq!(
            material_id as usize,
            self.materials.len(),
            "material IDs must be registered contiguously starting at 0; \
             expected id {}, got {}",
            self.materials.len(),
            material_id,
        );
        self.materials.push(material);
    }

    /// Replace the default material (ID 0).
    pub fn set_default(&mut self, material: Box<dyn MaterialModel>) {
        self.materials[0] = material;
    }

    /// Retrieve a material by ID. Falls back to material 0 for unknown IDs.
    pub fn get(&self, material_id: u32) -> &dyn MaterialModel {
        self.materials
            .get(material_id as usize)
            .unwrap_or(&self.materials[0])
            .as_ref()
    }

    pub fn len(&self) -> usize {
        self.materials.len()
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

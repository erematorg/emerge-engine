use std::collections::HashMap;

use crate::solver::materials::MaterialModel;

#[derive(Debug)]
pub struct MaterialRegistry {
    materials: HashMap<u32, Box<dyn MaterialModel>>,
    default_material_id: u32,
}

impl MaterialRegistry {
    pub fn with_default(default_material: Box<dyn MaterialModel>) -> Self {
        let default_material_id = 0u32;
        let mut materials = HashMap::new();
        materials.insert(default_material_id, default_material);
        Self {
            materials,
            default_material_id,
        }
    }

    pub fn insert(&mut self, material_id: u32, material: Box<dyn MaterialModel>) {
        self.materials.insert(material_id, material);
    }

    pub fn set_default(&mut self, default_material: Box<dyn MaterialModel>) {
        self.materials
            .insert(self.default_material_id, default_material);
    }

    pub fn get(&self, material_id: u32) -> &dyn MaterialModel {
        self.materials
            .get(&material_id)
            .or_else(|| self.materials.get(&self.default_material_id))
            .expect("MaterialRegistry must contain default material")
            .as_ref()
    }
}

use glam::Mat2;

use crate::materials::{
    BinghamFluidMaterial, ConstitutiveModel, CorotatedMaterial, DruckerPragerMaterial,
    GranularFluidMaterial, MaterialModel, MaterialParams, MuIRheologyMaterial, NaccMaterial,
    NeoHookeanMaterial, NewtonianFluidMaterial, NoCompressionMaterial, RankineMaterial,
    StomakhinMaterial, ViscoelasticMaterial, VonMisesMaterial,
};

/// Maximum number of material slots — matches `MAX_MATERIALS` in WGSL shaders.
/// The GPU uniform buffer holds exactly this many `MaterialParams` entries.
/// CPU accepts any count up to this limit; exceeding it panics to catch silent GPU truncation.
pub const MAX_MATERIAL_SLOTS: usize = 64;

/// A by-value copy of a registered material's concrete type, matched via
/// `MaterialModel::as_any` downcasting at registration time — see that
/// method's doc for why this exists. `Unknown` covers every wrapper
/// (`WithMixturePhase` etc.) and any external (e.g. LP-side) material: those
/// keep using the `Box<dyn MaterialModel>` vtable exactly as before, so this
/// enum can only ever add a fast path, never remove correctness.
#[derive(Debug, Clone, Copy)]
enum MaterialDispatch {
    NeoHookean(NeoHookeanMaterial),
    Corotated(CorotatedMaterial),
    Fluid(NewtonianFluidMaterial),
    Bingham(BinghamFluidMaterial),
    Snow(StomakhinMaterial),
    DruckerPrager(DruckerPragerMaterial),
    MuIRheology(MuIRheologyMaterial),
    VonMises(VonMisesMaterial),
    Rankine(RankineMaterial),
    Viscoelastic(ViscoelasticMaterial),
    Nacc(NaccMaterial),
    GranularFluid(GranularFluidMaterial),
    NoCompression(NoCompressionMaterial),
    Unknown,
}

/// Dispatches `$method` to whichever concrete material `MaterialDispatch`
/// holds -- a real static call per arm (the compiler knows the exact type),
/// not a vtable indirection. `Unknown` returns `None`, telling the caller to
/// fall back to the `&dyn MaterialModel` it always had. One macro for all
/// four hot-path methods so a 13-arm match isn't hand-duplicated four times
/// (and can't drift out of sync if a 14th material is added later).
macro_rules! dispatch {
    ($self:expr, $method:ident ( $($arg:expr),* )) => {
        Some(match $self {
            MaterialDispatch::NeoHookean(m) => m.$method($($arg),*),
            MaterialDispatch::Corotated(m) => m.$method($($arg),*),
            MaterialDispatch::Fluid(m) => m.$method($($arg),*),
            MaterialDispatch::Bingham(m) => m.$method($($arg),*),
            MaterialDispatch::Snow(m) => m.$method($($arg),*),
            MaterialDispatch::DruckerPrager(m) => m.$method($($arg),*),
            MaterialDispatch::MuIRheology(m) => m.$method($($arg),*),
            MaterialDispatch::VonMises(m) => m.$method($($arg),*),
            MaterialDispatch::Rankine(m) => m.$method($($arg),*),
            MaterialDispatch::Viscoelastic(m) => m.$method($($arg),*),
            MaterialDispatch::Nacc(m) => m.$method($($arg),*),
            MaterialDispatch::GranularFluid(m) => m.$method($($arg),*),
            MaterialDispatch::NoCompression(m) => m.$method($($arg),*),
            MaterialDispatch::Unknown => return None,
        })
    };
}

impl MaterialDispatch {
    fn from_model(material: &dyn MaterialModel) -> Self {
        let any = material.as_any();
        if let Some(m) = any.downcast_ref::<NeoHookeanMaterial>() {
            Self::NeoHookean(*m)
        } else if let Some(m) = any.downcast_ref::<CorotatedMaterial>() {
            Self::Corotated(*m)
        } else if let Some(m) = any.downcast_ref::<NewtonianFluidMaterial>() {
            Self::Fluid(*m)
        } else if let Some(m) = any.downcast_ref::<BinghamFluidMaterial>() {
            Self::Bingham(*m)
        } else if let Some(m) = any.downcast_ref::<StomakhinMaterial>() {
            Self::Snow(*m)
        } else if let Some(m) = any.downcast_ref::<DruckerPragerMaterial>() {
            Self::DruckerPrager(*m)
        } else if let Some(m) = any.downcast_ref::<MuIRheologyMaterial>() {
            Self::MuIRheology(*m)
        } else if let Some(m) = any.downcast_ref::<VonMisesMaterial>() {
            Self::VonMises(*m)
        } else if let Some(m) = any.downcast_ref::<RankineMaterial>() {
            Self::Rankine(*m)
        } else if let Some(m) = any.downcast_ref::<ViscoelasticMaterial>() {
            Self::Viscoelastic(*m)
        } else if let Some(m) = any.downcast_ref::<NaccMaterial>() {
            Self::Nacc(*m)
        } else if let Some(m) = any.downcast_ref::<GranularFluidMaterial>() {
            Self::GranularFluid(*m)
        } else if let Some(m) = any.downcast_ref::<NoCompressionMaterial>() {
            Self::NoCompression(*m)
        } else {
            Self::Unknown
        }
    }

    #[inline]
    fn kirchhoff_stress(&self, particles: &crate::particle::Particles, i: usize) -> Option<Mat2> {
        dispatch!(self, kirchhoff_stress(particles, i))
    }

    #[inline]
    fn stress_volume(&self, particles: &crate::particle::Particles, i: usize) -> Option<f32> {
        dispatch!(self, stress_volume(particles, i))
    }

    #[inline]
    fn owns_deformation_volume_state(&self) -> Option<bool> {
        dispatch!(self, owns_deformation_volume_state())
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn timestep_bound(
        &self,
        density: f32,
        hardening_scale: f32,
        cell_width: f32,
        material_cfl: f32,
        viscous_cfl: f32,
    ) -> Option<f32> {
        dispatch!(
            self,
            timestep_bound(
                density,
                hardening_scale,
                cell_width,
                material_cfl,
                viscous_cfl
            )
        )
    }

    /// `Some(())` = handled by the fast path, `None` = fall back to the
    /// vtable call (`Unknown` variant). G2P's per-particle plasticity update
    /// -- real work for fluids (integrates J) and every plastic material.
    #[inline]
    fn update_particle(&self, ctx: &mut crate::particle::ParticleUpdateCtx, dt: f32) -> Option<()> {
        dispatch!(self, update_particle(ctx, dt))
    }
}

/// Maps material IDs to constitutive models.
/// IDs must be contiguous starting at 0 — index 0 is the default/fallback.
/// The GPU path binds this as a flat `array<MaterialParams>` indexed by material_id.
#[derive(Debug)]
pub struct MaterialRegistry {
    materials: Vec<Box<dyn MaterialModel>>,
    // Parallel to `materials` — see `MaterialDispatch`'s own doc.
    dispatch: Vec<MaterialDispatch>,
}

impl MaterialRegistry {
    pub fn with_default(default_material: Box<dyn MaterialModel>) -> Self {
        let dispatch = MaterialDispatch::from_model(default_material.as_ref());
        Self {
            materials: vec![default_material],
            dispatch: vec![dispatch],
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
        let dispatch = MaterialDispatch::from_model(material.as_ref());
        if idx < self.materials.len() {
            self.materials[idx] = material; // replace existing
            self.dispatch[idx] = dispatch;
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
            self.dispatch.push(dispatch);
        }
    }

    /// Returns the next available material ID (= current count).
    /// Convenience for callers that auto-allocate IDs without tracking them manually.
    pub fn next_id(&self) -> u32 {
        self.materials.len() as u32
    }

    /// Replace the default material (ID 0).
    pub fn set_default(&mut self, material: Box<dyn MaterialModel>) {
        self.dispatch[0] = MaterialDispatch::from_model(material.as_ref());
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

    /// Real, static-dispatched `kirchhoff_stress` -- falls back to the
    /// `Box<dyn MaterialModel>` vtable call only for a material
    /// `MaterialDispatch::from_model` didn't recognise (see that type's
    /// doc). This is the hot per-particle P2G call; identical result to
    /// `self.get(material_id).kirchhoff_stress(...)` on every path, just
    /// without the vtable indirection when the fast path applies.
    pub(crate) fn kirchhoff_stress(
        &self,
        material_id: u32,
        particles: &crate::particle::Particles,
        i: usize,
    ) -> Mat2 {
        let idx = material_id as usize;
        match self
            .dispatch
            .get(idx)
            .and_then(|d| d.kirchhoff_stress(particles, i))
        {
            Some(tau) => tau,
            None => self.get(material_id).kirchhoff_stress(particles, i),
        }
    }

    /// Real, static-dispatched `timestep_bound` -- same fallback contract as
    /// `kirchhoff_stress` above. Hot per-particle CFL call.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn timestep_bound(
        &self,
        material_id: u32,
        density: f32,
        hardening_scale: f32,
        cell_width: f32,
        material_cfl: f32,
        viscous_cfl: f32,
    ) -> f32 {
        let idx = material_id as usize;
        match self.dispatch.get(idx).and_then(|d| {
            d.timestep_bound(
                density,
                hardening_scale,
                cell_width,
                material_cfl,
                viscous_cfl,
            )
        }) {
            Some(dt) => dt,
            None => self.get(material_id).timestep_bound(
                density,
                hardening_scale,
                cell_width,
                material_cfl,
                viscous_cfl,
            ),
        }
    }

    /// Real, static-dispatched `stress_volume` -- same fallback contract as
    /// `kirchhoff_stress` above. Hot per-particle P2G call.
    pub(crate) fn stress_volume(
        &self,
        material_id: u32,
        particles: &crate::particle::Particles,
        i: usize,
    ) -> f32 {
        let idx = material_id as usize;
        match self
            .dispatch
            .get(idx)
            .and_then(|d| d.stress_volume(particles, i))
        {
            Some(v) => v,
            None => self.get(material_id).stress_volume(particles, i),
        }
    }

    /// Real, static-dispatched `owns_deformation_volume_state` -- same
    /// fallback contract as `kirchhoff_stress` above. Hot per-particle P2G
    /// call (gates the strict WC-MPM finite-stress assertion).
    pub(crate) fn owns_deformation_volume_state(&self, material_id: u32) -> bool {
        let idx = material_id as usize;
        match self
            .dispatch
            .get(idx)
            .and_then(|d| d.owns_deformation_volume_state())
        {
            Some(v) => v,
            None => self.get(material_id).owns_deformation_volume_state(),
        }
    }

    /// Real, static-dispatched `update_particle` -- same fallback contract
    /// as `kirchhoff_stress` above. Hot per-particle G2P call.
    pub(crate) fn update_particle(
        &self,
        material_id: u32,
        ctx: &mut crate::particle::ParticleUpdateCtx,
        dt: f32,
    ) {
        let idx = material_id as usize;
        if self
            .dispatch
            .get(idx)
            .and_then(|d| d.update_particle(ctx, dt))
            .is_none()
        {
            self.get(material_id).update_particle(ctx, dt);
        }
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

    /// Returns true if any registered material consumes a per-substep
    /// kernel-density measurement. Strict WC-MPM liquids own `rho=rho0/J`
    /// instead and therefore return false.
    pub fn any_needs_density_recompute(&self) -> bool {
        self.materials.iter().any(|m| m.needs_density_recompute())
    }

    /// Returns true when a registered material owns a strict conservative
    /// volume/density state. The GPU backend uses this to turn numerical
    /// admissibility failures into a synchronous, observable failed step
    /// rather than allowing a bad scatter to continue unnoticed.
    pub fn any_owns_deformation_volume_state(&self) -> bool {
        self.materials
            .iter()
            .any(|m| m.owns_deformation_volume_state())
    }

    /// Plain vtable passthrough, not routed through `MaterialDispatch`'s fast
    /// path -- this is only ever called after `owns_deformation_volume_state`
    /// AND `is_near_wall` have both already short-circuited a per-particle
    /// `&&` chain (`cfl.rs`'s near-wall gate), so it only runs for the rare
    /// subset of particles that are both a strict fluid AND near a wall, not
    /// the hot per-particle path those two checks themselves are on.
    pub(crate) fn rest_acoustic_c2(&self, material_id: u32) -> Option<f32> {
        self.get(material_id).rest_acoustic_c2()
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

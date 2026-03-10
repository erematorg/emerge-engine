/// Compute pipeline setup for MLS-MPM GPU passes.
///
/// Three passes per substep, in order:
///   1. p2g         — scatter particles → grid (one workgroup per grid block)
///   2. grid_update — normalize momentum, apply gravity, enforce boundary
///   3. g2p         — gather grid → particles, update F and affine C matrix
///
/// Workgroup layout (2D, matching wgsparkl pattern):
///   - Grid blocks: 8×8 cells per workgroup
///   - P2G/G2P: one thread per cell in the block, particles loaded into shared memory
///
/// Material dispatch: ConstitutiveModel discriminant in particle data → WGSL switch
///   case 1 (Fluid):      Tait EOS pressure + deviatoric viscosity
///   case 2 (NeoHookean): µ(FFᵀ−I) + λ·ln(J)·I  (simplified form, no F⁻¹)
///   case 3 (Corotated):  2µ(F−R)Fᵀ + λ(J−1)J·I  (analytical 2D polar decomp)
///   Plasticity (Snow=4, DruckerPrager=5): return mapping stays CPU year 1.
use wgpu;

/// All compiled compute pipelines for one solver instance.
pub struct MpmPipelines {
    pub p2g: wgpu::ComputePipeline,
    pub grid_update: wgpu::ComputePipeline,
    pub g2p: wgpu::ComputePipeline,

    pub bind_group_layout: wgpu::BindGroupLayout,
}

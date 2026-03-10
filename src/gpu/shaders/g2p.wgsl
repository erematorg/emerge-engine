// G2P — Grid to Particle gather
// MLS-MPM, Hu et al. 2018 SIGGRAPH §4.
//
// One thread per particle. Reads 3×3 grid neighborhood.
// Updates: particle velocity, APIC affine C matrix, deformation gradient F.
//
// Steps per particle:
//   1. Gather: v_p = Σ w_i * v_i  (weighted sum of grid velocities)
//   2. B-matrix: B += w_i * v_i ⊗ (x_i - x_p)  (outer product, APIC)
//   3. C = B * D_inverse  (D_inverse = 4.0, quadratic B-spline)
//   4. F = (I + dt * C) * F  (left-multiply deformation gradient)
//   5. Constitutive update: kirchhoff stress, volume, density
//      (plasticity for snow/sand stays CPU year 1 — SVD in WGSL is complex)
//
// Material stress dispatch: switch on particle.material_id → ConstitutiveModel discriminant
//   1 = Fluid      → Tait EOS pressure + deviatoric viscosity
//   2 = NeoHookean → µ(FFᵀ − I) + λ·ln(J)·I
//   3 = Corotated  → 2µ(F−R)Fᵀ + λ(J−1)J·I  (analytical 2D polar decomp in WGSL)
//
// Reference: Hu et al. 2018, verified constants:
//   D_inverse = 4.0  (mls-mpm88-explained.cpp: `Dinv = 4 * inv_dx * inv_dx`, dx=1)
//   F update left-multiply: `F = (I + dt*C) * F`
//
// TODO: implement

// (shared struct definitions with p2g.wgsl — will be unified via wgsl module system)

// TODO: @compute @workgroup_size(64, 1, 1)
// fn g2p_main(@builtin(global_invocation_id) gid: vec3<u32>) { ... }

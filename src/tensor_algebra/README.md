# tensor_algebra

The mathematics the physics is written in: matrix and tensor operations.
Nothing here is a physical law.

| File | What it is |
|---|---|
| `svd.rs` | Singular value decomposition of a 2×2 matrix, and rebuilding a matrix from its singular values. |
| `frobenius.rs` | The Frobenius inner product: the dot product of two matrices. |
| `symmetric_eigen.rs` | Eigenvalues of a small symmetric matrix and the nearest positive semidefinite matrix (test oracles). |

Owns: pure mathematics used by several domains.
Does not own: anything with a physical meaning. The polar decomposition of a
deformation gradient, for example, describes a stretch and a rotation, so it
lives in `matter/kinematics/`.

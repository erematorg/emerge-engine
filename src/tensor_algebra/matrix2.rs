//! A 2 x 2 matrix and its scalar, whatever their precision: so a law is
//! written once and used at `f32` (the explicit solver) and at `f64` (the
//! total Lagrangian step, whose discrete gradient needs the energy's change
//! to `f64` round-off).
//!
//! Each operation delegates to glam's own (`Mat2`, `DMat2`), so a law
//! written through these traits computes, at `f32`, exactly what it
//! computed written on `Mat2` directly: same operations, same order.

use core::ops::{Add, Div, Mul, Neg, Sub};

use glam::{DMat2, DVec2, Mat2, Vec2};

/// A real number type: `f32` or `f64`.
pub trait Real:
    Copy
    + PartialOrd
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + From<f32>
{
    /// The type's machine epsilon.
    const EPSILON: Self;
    fn sqrt(self) -> Self;
}

impl Real for f32 {
    const EPSILON: Self = f32::EPSILON;
    fn sqrt(self) -> Self {
        f32::sqrt(self)
    }
}

impl Real for f64 {
    const EPSILON: Self = f64::EPSILON;
    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }
}

/// A 2 x 2 matrix over `Real`, by its columns.
pub trait Matrix2: Copy + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> {
    type Scalar: Real;
    const IDENTITY: Self;
    const ZERO: Self;
    fn determinant(self) -> Self::Scalar;
    fn transpose(self) -> Self;
    /// `s` times the matrix.
    fn scaled(self, s: Self::Scalar) -> Self;
    /// The columns, `[[x_axis.x, x_axis.y], [y_axis.x, y_axis.y]]`.
    fn columns(self) -> [[Self::Scalar; 2]; 2];
    fn from_columns(first: [Self::Scalar; 2], second: [Self::Scalar; 2]) -> Self;
}

impl Matrix2 for Mat2 {
    type Scalar = f32;
    const IDENTITY: Self = Mat2::IDENTITY;
    const ZERO: Self = Mat2::ZERO;
    fn determinant(self) -> f32 {
        Mat2::determinant(&self)
    }
    fn transpose(self) -> Self {
        Mat2::transpose(&self)
    }
    fn scaled(self, s: f32) -> Self {
        s * self
    }
    fn columns(self) -> [[f32; 2]; 2] {
        [self.x_axis.to_array(), self.y_axis.to_array()]
    }
    fn from_columns(first: [f32; 2], second: [f32; 2]) -> Self {
        Mat2::from_cols(Vec2::from_array(first), Vec2::from_array(second))
    }
}

impl Matrix2 for DMat2 {
    type Scalar = f64;
    const IDENTITY: Self = DMat2::IDENTITY;
    const ZERO: Self = DMat2::ZERO;
    fn determinant(self) -> f64 {
        DMat2::determinant(&self)
    }
    fn transpose(self) -> Self {
        DMat2::transpose(&self)
    }
    fn scaled(self, s: f64) -> Self {
        s * self
    }
    fn columns(self) -> [[f64; 2]; 2] {
        [self.x_axis.to_array(), self.y_axis.to_array()]
    }
    fn from_columns(first: [f64; 2], second: [f64; 2]) -> Self {
        DMat2::from_cols(DVec2::from_array(first), DVec2::from_array(second))
    }
}

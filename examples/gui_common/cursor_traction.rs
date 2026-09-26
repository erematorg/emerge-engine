//! A cursor push stated in pascals, delivered through the engine's own
//! force-field hook so that it acts on every substep.
//!
//! Its own file, next to `cursor_force.rs` rather than inside it, because
//! examples include these shared files with `#[path]`, one copy each: a
//! type an example does not use would be dead code in that example. A demo
//! that pushes in pascals includes this file, one that pushes in multiples
//! of weight includes the other.
//!
//! Why a force field and not a velocity change applied once per frame: the
//! first version did the latter, and a whole frame's push landing before
//! the frame's substeps ran was a single kick. Pushing a 1200 Pa column at
//! the tractions that yield it, that kick reached 32.7 m/s in one frame,
//! 3.7 times the material's sound speed, which is an impact and not a push.
//! A `Field` is integrated by the solver itself with the substep, so the
//! material's stress answers between one push increment and the next.

use crate::emerge::Field;
use crate::emerge::particle::Particles;
use glam::Vec2;
use std::sync::{Arc, Mutex, MutexGuard};

/// What the scene sets each frame, and what the field reports back.
pub struct CursorShared {
    /// Cursor position, in cells.
    pub position: Vec2,
    pub pushing: bool,
    pub pulling: bool,
    /// The cursor's push and pull, in pascals: see `CursorTraction`.
    pub push_pa: f32,
    pub pull_pa: f32,
    /// What the last substep the cursor acted in actually touched.
    pub contact: Option<CursorContact>,
}

/// The contact one push actually made, measured, for a panel to report.
#[derive(Clone, Copy)]
pub struct CursorContact {
    /// Particles inside the cursor's radius.
    pub particles: usize,
    /// Width of material touched, measured across the net force, metres.
    pub width_m: f32,
    /// Net force the push delivered, per metre of depth, newtons per metre.
    pub force_n_per_m: f32,
}

impl CursorContact {
    /// The traction the push actually exerts over the contact it made,
    /// net force over touched width. Larger than the cursor's own `P`
    /// whenever the contact is narrower than the cursor.
    pub fn traction_pa(&self) -> f32 {
        self.force_n_per_m / self.width_m
    }
}

/// A cursor whose push is stated in pascals.
///
/// `P` is a property of the cursor: the net force it delivers when half of
/// its disk is inside material, divided by its diameter `2r`. That is a
/// declared definition, chosen because it can be read against a stress.
/// Each particle within the radius is pushed radially with the linear
/// falloff `CursorForce` uses; the pushes across the push direction cancel
/// in pairs, so what moves the material as a whole is their resultant, and
/// over a half disk that resultant is `rho * a * r^2 / 3` per unit depth
/// (the projected integral of the falloff; summing the pushes' magnitudes
/// without projecting would give `pi r^2 / 6` instead, which is not the
/// force the body feels). Setting it to `P * 2r` gives one acceleration
/// for every touched particle, `a = 6 P / (rho r)`, fixed by `P` alone.
///
/// Fixed, and not re-solved for whatever the cursor touches at the moment:
/// holding a traction over a shrinking or partly cancelling contact means
/// dividing by a vanishing net, and that is how the first version reached
/// supersonic kicks. The price is that the traction really exerted differs
/// from `P` when the contact does, so the field measures it and reports it
/// (`CursorContact::traction_pa`) instead of assuming it.
///
/// A push `P` shears the material at about `P / 2`, so it yields a
/// material near `P = 2 tau_0`: `tests/scratch_bingham_cursor_yield.rs`.
pub struct CursorTraction {
    radius: f32,
    rho_kg_m3: f32,
    spacing: f32,
    dx_m: f32,
    shared: Arc<Mutex<CursorShared>>,
}

impl CursorTraction {
    pub fn new(radius: f32, push_pa: f32, pull_pa: f32) -> Self {
        Self {
            radius,
            rho_kg_m3: 1000.0,
            spacing: 0.5,
            dx_m: 0.01,
            shared: Arc::new(Mutex::new(CursorShared {
                position: Vec2::ZERO,
                pushing: false,
                pulling: false,
                push_pa,
                pull_pa,
                contact: None,
            })),
        }
    }

    /// The lattice the scene spawns on: density in kg/m3, spacing in cells,
    /// cell size in metres. It fixes each particle's mass per unit depth at
    /// `rho * (spacing * dx)^2`, which the pascals depend on. `new` assumes
    /// water on a 1 cm grid at spacing 0.5, which is only right for a scene
    /// that uses exactly that.
    pub fn with_lattice(mut self, rho_kg_m3: f32, spacing: f32, dx_m: f32) -> Self {
        self.rho_kg_m3 = rho_kg_m3;
        self.spacing = spacing;
        self.dx_m = dx_m;
        self
    }

    /// The force field to register with the simulation. It shares this
    /// cursor's state, so what the scene sets each frame reaches it.
    pub fn field(&self) -> CursorField {
        CursorField {
            radius: self.radius,
            rho_kg_m3: self.rho_kg_m3,
            spacing: self.spacing,
            dx_m: self.dx_m,
            shared: Arc::clone(&self.shared),
            at: Vec2::ZERO,
            accel_cells: 0.0,
            sign: 0.0,
        }
    }

    /// The shared state, locked: set the cursor each frame, read the
    /// contact back.
    pub fn shared(&self) -> MutexGuard<'_, CursorShared> {
        self.shared.lock().expect("cursor state poisoned")
    }
}

/// The cursor as the solver sees it. `prepare` copies the shared state once
/// per substep, so `acceleration`, called once per particle, takes no lock.
pub struct CursorField {
    radius: f32,
    rho_kg_m3: f32,
    spacing: f32,
    dx_m: f32,
    shared: Arc<Mutex<CursorShared>>,
    at: Vec2,
    accel_cells: f32,
    /// +1 pushing, -1 pulling, 0 idle.
    sign: f32,
}

impl Field for CursorField {
    fn prepare(&mut self, particles: &Particles) {
        let mut shared = self.shared.lock().expect("cursor state poisoned");
        let (traction_pa, sign) = if shared.pushing {
            (shared.push_pa, 1.0)
        } else if shared.pulling {
            (shared.pull_pa, -1.0)
        } else {
            (0.0, 0.0)
        };
        self.at = shared.position;
        self.sign = sign;
        let radius_m = self.radius * self.dx_m;
        // a = 6 P / (rho r): see `CursorTraction` for the derivation.
        self.accel_cells = 6.0 * traction_pa / (self.rho_kg_m3 * radius_m) / self.dx_m;
        if sign == 0.0 {
            shared.contact = None;
            return;
        }

        // Measure what this substep's push actually does, for the panel.
        let mass_per_depth = self.rho_kg_m3 * (self.spacing * self.dx_m).powi(2);
        let mut touched = Vec::new();
        let mut net = Vec2::ZERO;
        for i in 0..particles.len() {
            let a = self.acceleration(particles, i);
            if a != Vec2::ZERO {
                // Grid acceleration to SI, times each particle's mass.
                net += a * self.dx_m * mass_per_depth;
                touched.push(i);
            }
        }
        if touched.is_empty() {
            shared.contact = None;
            return;
        }
        let across = net.normalize_or_zero().perp();
        let (mut lo, mut hi) = (f32::MAX, f32::MIN);
        for &i in &touched {
            let s = particles.x[i].dot(across);
            lo = lo.min(s);
            hi = hi.max(s);
        }
        // One spacing added so a single row of particles counts for the
        // width it actually occupies, not zero.
        shared.contact = Some(CursorContact {
            particles: touched.len(),
            width_m: (hi - lo + self.spacing) * self.dx_m,
            force_n_per_m: net.length(),
        });
    }

    fn acceleration(&self, particles: &Particles, i: usize) -> Vec2 {
        if self.sign == 0.0 {
            return Vec2::ZERO;
        }
        let d = particles.x[i] - self.at;
        let dist = d.length();
        if dist <= 1.0e-4 || dist >= self.radius {
            return Vec2::ZERO;
        }
        let falloff = 1.0 - dist / self.radius;
        (d / dist) * (self.accel_cells * falloff * self.sign)
    }
}

//! Read-only diagnostics, tag-group, and spatial-query API on `Simulation`.
//!
//! Split out of `solver/mod.rs` -- everything here reads state (or, for the
//! tag-group setters, writes a small uniform slice of it) rather than
//! advancing the simulation. Distinct from `solver::body_state`, which holds
//! the free `BodyState`-aggregation functions these methods call into.

use glam::Vec2;

use super::Simulation;
use super::body_state::{self, BodyState, body_state_of};
use crate::diagnostics::{SimSnapshot, collect_rod_snapshot, collect_snapshot};

impl Simulation {
    pub fn diagnostics_snapshot(&self) -> SimSnapshot {
        let mut snap = collect_snapshot(
            self.frame_index,
            &self.particles,
            &self.grid,
            &self.config,
            self.last_step_dt,
            self.last_substeps,
        );
        snap.vel_clamp_count = self.last_vel_clamp_count;
        snap.j_projection_count = self.last_j_projection_count;
        snap.sim_time_dropped = self.last_sim_time_dropped;
        snap.active_count = self.active_count;
        snap.sleeping_count = self.particles.len().saturating_sub(self.active_count);
        snap.timing = self.last_timing;
        snap.rods = collect_rod_snapshot(self.rods());
        snap
    }

    // ── Tag-based group API ───────────────────────────────────────────────────

    /// Aggregate physics state for all particles with `tag`. O(group_size).
    pub fn group_state(&self, tag: u32) -> BodyState {
        let mut s = BodyState::default();
        if let Some(indices) = self.tag_index.get(&tag) {
            for &i in indices {
                s.accumulate(
                    self.particles.x[i],
                    self.particles.v[i].length(),
                    self.particles.plastic_volume_ratio[i],
                    self.particles.deformation_gradient[i].determinant(),
                    self.particles.density[i],
                );
            }
        }
        s.finalize();
        s
    }

    /// Center of mass for all particles with `tag`. O(group_size).
    pub fn group_centroid(&self, tag: u32) -> glam::Vec2 {
        let indices = match self.tag_index.get(&tag) {
            Some(s) if !s.is_empty() => s,
            _ => return glam::Vec2::ZERO,
        };
        let sum: glam::Vec2 = indices.iter().map(|&i| self.particles.x[i]).sum();
        sum / indices.len() as f32
    }

    /// Number of particles with `tag`. O(1).
    pub fn group_count(&self, tag: u32) -> usize {
        self.tag_index.get(&tag).map_or(0, |v| v.len())
    }

    /// Set `activation` uniformly on all particles with `tag`. O(group_size).
    pub fn set_group_activation(&mut self, tag: u32, value: f32) {
        if let Some(indices) = self.tag_index.get(&tag) {
            for &i in indices {
                self.particles.activation[i] = value.clamp(0.0, 1.0);
            }
        }
    }

    /// Set `activation` per particle using a spatial function. O(group_size).
    pub fn set_group_activation_fn(&mut self, tag: u32, f: impl Fn(glam::Vec2) -> f32) {
        if let Some(indices) = self.tag_index.get(&tag) {
            for &i in indices {
                self.particles.activation[i] = f(self.particles.x[i]).clamp(0.0, 1.0);
            }
        }
    }

    /// Set `temperature` uniformly on all particles with `tag`. O(group_size).
    pub fn set_group_temperature(&mut self, tag: u32, value: f32) {
        if let Some(indices) = self.tag_index.get(&tag) {
            for &i in indices {
                self.particles.temperature[i] = value;
            }
        }
    }

    /// Apply a velocity impulse to all particles with `tag`, with optional distance falloff. O(group_size).
    pub fn apply_group_impulse(
        &mut self,
        tag: u32,
        impulse: glam::Vec2,
        falloff_center: Option<glam::Vec2>,
    ) {
        let indices: Vec<usize> = match self.tag_index.get(&tag) {
            Some(s) => s.iter().copied().collect(),
            None => return,
        };
        // Linear falloff: full strength at center, zero at 10 cells.
        const FALLOFF_PER_CELL: f32 = 0.1;
        for i in indices {
            let scale = match falloff_center {
                None => 1.0,
                Some(c) => (1.0 - (self.particles.x[i] - c).length() * FALLOFF_PER_CELL).max(0.0),
            };
            self.particles.v[i] += impulse * scale;
        }
    }

    /// Directly overwrite `v` (not add, unlike `apply_group_impulse`) on every
    /// particle with `tag` — the real primitive for a KINEMATICALLY-driven body
    /// (position/velocity set by an external controller, e.g. player input,
    /// rather than by internal elastic/plastic forces). Combined with a real,
    /// large `mass` on that group and a nonzero `contact_group` (Bardenhagen
    /// 2001 multi-field contact), this lets an externally-driven tool genuinely
    /// PUSH/displace other real matter through real momentum exchange — no
    /// mass is created or destroyed, unlike deleting particles outright.
    /// O(group_size).
    pub fn set_group_velocity(&mut self, tag: u32, velocity: glam::Vec2) {
        if let Some(indices) = self.tag_index.get(&tag) {
            for &i in indices {
                self.particles.v[i] = velocity;
            }
        }
    }

    /// Set `contact_group` uniformly on all particles with `tag` — opts a
    /// body into its own real multi-field Coulomb contact (Bardenhagen 2001)
    /// against everything else, instead of MPM's default infinite-friction
    /// stick. O(group_size).
    pub fn set_group_contact_group(&mut self, tag: u32, contact_group: u32) {
        if let Some(indices) = self.tag_index.get(&tag) {
            for &i in indices {
                self.particles.contact_group[i] = contact_group;
            }
        }
    }

    /// Scale `mass` uniformly on all particles with `tag` — a real, disclosed
    /// way to make a kinematically-driven body act like a much heavier real
    /// object (e.g. a tool vs. the loose material it displaces) without
    /// changing its own internal material response. O(group_size).
    pub fn scale_group_mass(&mut self, tag: u32, multiplier: f32) {
        if let Some(indices) = self.tag_index.get(&tag) {
            for &i in indices {
                self.particles.mass[i] *= multiplier;
            }
        }
    }

    /// Lazily rebuilds the spatial hash if `step()` has run since the last
    /// rebuild (see `spatial_hash`'s own doc on `Simulation`). No-op when
    /// already fresh — at most one real rebuild per `step()` call no matter
    /// how many query methods get called before the next `step()`.
    fn ensure_spatial_hash_fresh(&self) {
        if self.spatial_hash_dirty.get() {
            self.spatial_hash
                .borrow_mut()
                .rebuild(&self.particles.x, self.active_count);
            self.spatial_hash_dirty.set(false);
        }
    }

    // ── Query & Transition API ────────────────────────────────────────────────

    /// Aggregate state for all particles of a given material.
    pub fn material_state(&self, material_id: u32) -> BodyState {
        body_state_of(&self.particles, material_id)
    }

    /// Aggregate state for all particles within `radius` grid-cells of `center`.
    pub fn region_state(&self, center: Vec2, radius: f32) -> BodyState {
        self.ensure_spatial_hash_fresh();
        let r2 = radius * radius;
        let mut s = body_state::BodyState::default();
        let hash = self.spatial_hash.borrow();
        for i in hash.query(center, radius) {
            if (self.particles.x[i] - center).length_squared() <= r2 {
                s.accumulate(
                    self.particles.x[i],
                    self.particles.v[i].length(),
                    self.particles.plastic_volume_ratio[i],
                    self.particles.deformation_gradient[i].determinant(),
                    self.particles.density[i],
                );
            }
        }
        s.finalize();
        s
    }

    /// Indices of active particles within `radius` grid-cells of `center`.
    ///
    /// Returns indices only — read particle data via `solver.particles().x[i]` etc.
    /// O(candidates) via spatial hash, not O(N). Collected eagerly into a `Vec`
    /// (not a lazy iterator, unlike an earlier version of this method) — the
    /// lazy spatial-hash rebuild below needs a `Ref` borrow that can't outlive
    /// this call, so results are gathered up front instead. Negligible extra
    /// cost relative to the O(candidates) work already being done.
    pub fn particles_near(&self, center: Vec2, radius: f32) -> Vec<usize> {
        self.ensure_spatial_hash_fresh();
        let r2 = radius * radius;
        let hash = self.spatial_hash.borrow();
        hash.query(center, radius)
            .filter(|&i| (self.particles.x[i] - center).length_squared() <= r2)
            .collect()
    }

    /// Count active particles of a given material within `radius` of `center`.
    /// O(candidates) via spatial hash, not O(N).
    pub fn count_near(&self, center: Vec2, radius: f32, material_id: u32) -> usize {
        self.ensure_spatial_hash_fresh();
        let r2 = radius * radius;
        let hash = self.spatial_hash.borrow();
        hash.query(center, radius)
            .filter(|&i| {
                self.particles.material_id[i] == material_id
                    && (self.particles.x[i] - center).length_squared() <= r2
            })
            .count()
    }

    /// Indices of the `k` active particles nearest to `center`, sorted by
    /// distance ascending -- a real topological neighbor rule (Ballerini et al.
    /// 2008, PNAS: real starling flocks track their ~6-7 nearest neighbors
    /// regardless of physical distance, not everything within a fixed radius).
    /// `particles_near`/`count_near` can't express this: a fixed radius pulls
    /// in more neighbors in dense regions and fewer in sparse ones, exactly the
    /// density-fragility topological neighbor rules are empirically shown to
    /// avoid (the same paper found this rule is what makes real flocks robust
    /// to predator-induced density changes).
    ///
    /// Exact, not approximate: expands the search radius geometrically until
    /// at least `k` candidates are found within it, then returns the true `k`
    /// closest by exact distance. Correctness relies on `SpatialHash::query`
    /// returning every particle in the queried box, never a sample -- once `k`
    /// candidates are confirmed within some radius R, the true k-nearest are
    /// guaranteed to already be included, so sorting and truncating what's
    /// been gathered is exact.
    ///
    /// Returns fewer than `k` if the active particle set itself has fewer.
    pub fn particles_knn(&self, center: Vec2, k: usize) -> Vec<usize> {
        if k == 0 || self.active_count == 0 {
            return Vec::new();
        }
        self.ensure_spatial_hash_fresh();
        let hash = self.spatial_hash.borrow();
        let domain_diag =
            self.config.grid_res as f32 * self.config.grid_cell_size * std::f32::consts::SQRT_2;
        let mut radius = self.config.grid_cell_size * (k as f32).sqrt().max(1.0);
        let mut candidates: Vec<(usize, f32)>;
        loop {
            let r2 = radius * radius;
            candidates = hash
                .query(center, radius)
                .map(|i| (i, (self.particles.x[i] - center).length_squared()))
                .filter(|&(_, d2)| d2 <= r2)
                .collect();
            if candidates.len() >= k || radius >= domain_diag {
                break;
            }
            radius *= 2.0;
        }
        // Partial selection, not a full sort: we only need the k nearest, not a total
        // ordering of every candidate. `select_nth_unstable_by` is quickselect -- O(n)
        // average to partition so the k smallest distances occupy [0, k), then we sort
        // only those k to honor the "ascending" contract. For the topological-neighbor
        // use case (k ~= 6-7, Ballerini 2008) against the dozens-to-hundreds of
        // candidates a dense region's radius pulls in, this is O(n + k log k) instead of
        // the old O(n log n) full sort of everything we were about to discard. Guarded on
        // `len > k` because select_nth panics on an out-of-range pivot; when len <= k we
        // keep everything and the final sort is already over just those elements.
        if candidates.len() > k {
            candidates.select_nth_unstable_by(k - 1, |a, b| a.1.total_cmp(&b.1));
            candidates.truncate(k);
        }
        candidates.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        candidates.into_iter().map(|(i, _)| i).collect()
    }
}

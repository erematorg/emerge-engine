//! Phase rules, impulses, and per-particle lifecycle: spawn (`add_body`),
//! removal, sleep/wake, and tag bookkeeping.
//!
//! Split out of `solver/mod.rs` -- the remaining piece after construction
//! (`solver::lifecycle`), the step loop (`solver::step`), and read-only
//! queries (`solver::queries`) were pulled out. Everything here still
//! mutates the particle buffer between `step()` calls, unlike `queries`.

use std::collections::{HashMap, HashSet};

use glam::{Mat2, Vec2};

use super::{LcgRng, Simulation, SpawnRegion, density, initialize_particles};
use crate::particle::Particle;
use crate::solver::density::estimate_particle_volumes;
use crate::thermodynamics::ScalarDiffusionField;

/// The equilibrium state of one particle at `depth` below its own free
/// surface: deformation gradient, volume and density, all three consistent.
///
/// Shared by the CPU and GPU paths so the physics exists once. They differ
/// only in how particles are stored, never in what equilibrium means.
///
/// `None` when the material has no equation of state to invert, or when the
/// particle is at the surface and already in equilibrium.
pub fn hydrostatic_state(
    material: &dyn crate::materials::MaterialModel,
    gravity_magnitude: f32,
    depth: f32,
    initial_volume: f32,
    mass: f32,
) -> Option<HydrostaticState> {
    if depth <= 0.0 || gravity_magnitude <= 0.0 {
        return None;
    }
    // Rest density is the right weight here: the correction is small by
    // construction, so iterating it would buy nothing.
    let pressure = material.params().rest_density * gravity_magnitude * depth;
    let j = material.hydrostatic_volume_ratio(pressure)?;
    let volume = initial_volume * j;
    Some(HydrostaticState {
        // det(s*I) = s^2 in 2D, so the isotropic compression with
        // det(F) = J is F = sqrt(J)*I.
        deformation_gradient: Mat2::from_diagonal(Vec2::splat(j.sqrt())),
        volume,
        // Volume and density are not independent of J -- V = V0*J and
        // rho = m/V. Leaving them behind puts the particle in a state the
        // solver's own strict-material consistency check rejects.
        density: if volume > 0.0 { mass / volume } else { 0.0 },
    })
}

/// One particle's hydrostatic equilibrium, as the three fields that must
/// change together. See [`hydrostatic_state`].
#[derive(Clone, Copy, Debug)]
pub struct HydrostaticState {
    pub deformation_gradient: Mat2,
    pub volume: f32,
    pub density: f32,
}

impl Simulation {
    /// Puts every particle into hydrostatic equilibrium under the current
    /// gravity, so a body spawned "at rest" starts at rest.
    ///
    /// Particles are created at uniform density, which is not an
    /// equilibrium state: a column of fluid is held up by the pressure
    /// gradient its own weight creates, `p = rho*g*h`. Spawned uniform, a
    /// pool has no internal pressure anywhere, collapses under its own
    /// weight, and rings -- visible as a "resting" pool twitching for its
    /// first frames before anything touches it. That is the solver being
    /// correct about a starting state that was never physical.
    ///
    /// This asks each material for the volume ratio that balances the
    /// pressure at each particle's own depth (`MaterialModel::
    /// hydrostatic_volume_ratio`), and writes it into the deformation
    /// gradient. Materials with no equation of state to invert are left
    /// untouched, so calling this on a scene of solids does nothing.
    ///
    /// Depth is measured per column against the highest particle of the
    /// same material in that column, so it handles a sloped or stepped free
    /// surface, not just a flat pool.
    ///
    /// Call it after spawning and before the first step.
    pub fn settle_hydrostatic(&mut self) {
        let gravity_magnitude = self.config.gravity.length();
        if gravity_magnitude <= 0.0 {
            return;
        }
        // Free surface per (material, column): the top of the matter whose
        // weight presses on everything below it in that column.
        let mut surface: HashMap<(u32, i32), f32> = HashMap::new();
        for i in 0..self.particles.len() {
            let key = (
                self.particles.material_id[i],
                self.particles.x[i].x.floor() as i32,
            );
            let top = surface.entry(key).or_insert(f32::NEG_INFINITY);
            *top = top.max(self.particles.x[i].y);
        }

        for i in 0..self.particles.len() {
            let material_id = self.particles.material_id[i];
            let material = self.materials.get(material_id);
            let key = (material_id, self.particles.x[i].x.floor() as i32);
            let Some(&top) = surface.get(&key) else {
                continue;
            };
            let depth = top - self.particles.x[i].y;
            if depth <= 0.0 {
                continue;
            }
            let Some(state) = hydrostatic_state(
                material,
                gravity_magnitude,
                depth,
                self.particles.initial_volume[i],
                self.particles.mass[i],
            ) else {
                continue;
            };
            self.particles.deformation_gradient[i] = state.deformation_gradient;
            self.particles.volume[i] = state.volume;
            self.particles.density[i] = state.density;
        }
    }

    /// Shared phase-transition logic: the one place where both
    /// `phase_transition` and the per-substep `add_phase_rule` evaluation
    /// (`solver::step`) change a particle's material, so the two cannot
    /// drift apart.
    ///
    /// Rebaselines the elastic reference to the particle's current shape. A
    /// fluid's `F` only encodes volume ratio (`sqrt(J)*I`, see
    /// `NewtonianFluidMaterial`), while a solid reads `F` as strain away from
    /// a rest configuration; a leftover `J != 1` would be read as elastic
    /// strain and oscillate as a spurious restoring stress instead of the
    /// particle freezing smoothly into place. A phase change alters the
    /// material's structure, so strain relative to the old phase's reference
    /// has no meaning in the new one: the new zero-strain reference is where
    /// the particle is at the instant of transition. Applied here, once, for
    /// every material rather than per solid.
    ///
    /// Volume and density stay continuous: `initial_volume` is rebaselined to
    /// the current volume, not the spawn volume, so only the elastic strain
    /// memory is cleared. A material whose `init_particle` then redefines
    /// volume/density from its own rest density (e.g.
    /// `NewtonianFluidMaterial` on melting) does so on purpose: most materials
    /// change density when they melt or freeze.
    ///
    /// ## Toward a Stefan-condition treatment
    ///
    /// Phase change here (and in every `add_phase_rule` predicate, e.g.
    /// "water freezes below 273K") is an instantaneous per-particle threshold,
    /// with no transformation rate. The governing condition for a moving phase
    /// boundary is the Stefan condition, `L * rho * v_interface = -[k *
    /// grad(T)]`: the interface moves at a speed set by the jump in heat flux
    /// across it. A threshold switch is its `v_interface -> infinity` limit.
    ///
    /// `L` (debited below from `MaterialModel::latent_heat()`) and
    /// `k`/`grad(T)` (computed every substep by `ThermalDiffusion`, see
    /// `energy::thermodynamics::diffusion`) already exist. Not done: using
    /// `grad(T)` at the moment of transition to rate-limit the transition
    /// instead of switching material once `L` has been debited.
    pub(super) fn apply_phase_transition(&mut self, i: usize, new_material_id: u32) {
        // Captured before being overwritten below: `MaterialModel::
        // latent_heat` needs the material this particle is transitioning
        // from, not only the one it arrives at.
        let from_material_id = self.particles.material_id[i];
        self.particles.material_id[i] = new_material_id;

        let current_volume = self.particles.volume[i];
        if current_volume.is_finite() && current_volume > 0.0 {
            self.particles.deformation_gradient[i] = Mat2::IDENTITY;
            self.particles.initial_volume[i] = current_volume;
            self.particles.density[i] = self.particles.mass[i] / current_volume;
        }

        let latent_heat = self
            .materials
            .get(new_material_id)
            .latent_heat(from_material_id);
        if latent_heat != 0.0
            && let Some(thermal) = &self.thermal
        {
            self.particles.temperature[i] -= latent_heat / thermal.config.heat_capacity;
        }

        // Defaults to `init_particle` for every material that does not
        // override it (see `MaterialModel::init_particle_from_transition`). A
        // material whose rest state differs sharply from what it transitions
        // from (water->steam, `IdealGasMaterial`) overrides it to keep the
        // continuous rebaseline above instead of recomputing from
        // `mass/rest_density`.
        let mut p = self.particles.get(i);
        self.materials
            .get(new_material_id)
            .init_particle_from_transition(&mut p);
        self.particles.set(i, p);
    }

    /// Switch material for every particle where `predicate` returns true.
    ///
    /// Rebaselines each transitioned particle's elastic reference state to
    /// its current physical configuration and calls the new material's own
    /// `init_particle` -- see `apply_phase_transition`'s doc for the
    /// real reasoning (this is not a cosmetic reset: without it, a solid
    /// material arriving from a fluid's leftover volumetric state visibly
    /// springs/oscillates instead of freezing smoothly). Otherwise a
    /// transitioned particle would also inherit stale material-specific
    /// plastic fields (`hardening_scale`, `friction_hardening`,
    /// `plastic_volume_ratio`, etc.) from its OLD material, reinterpreted
    /// under the new material's semantics (e.g. Rankine's damage
    /// accumulator read as Drucker-Prager's friction accumulator) --
    /// `init_particle` resets those to the new material's own defaults,
    /// exactly like `reinit_all_particle_state`/`add_body` do for every
    /// other material-assignment path.
    ///
    /// If `new_material_id`'s `MaterialModel::latent_heat()` is non-zero and a thermal
    /// model is configured (`with_thermal`/`set_thermal`), debits `temperature` by
    /// `latent_heat / heat_capacity` for every transitioned particle -- see
    /// `MaterialModel::latent_heat` for the sign convention.
    pub fn phase_transition<F>(&mut self, predicate: F, new_material_id: u32)
    where
        F: Fn(&Particle) -> bool,
    {
        assert!(
            self.materials.is_registered(new_material_id),
            "phase_transition: material_id {new_material_id} is not registered -- \
             call solver.with_material({new_material_id}, ...) first"
        );
        for i in 0..self.particles.len() {
            let p = self.particles.get(i);
            if predicate(&p) {
                self.apply_phase_transition(i, new_material_id);
            }
        }
    }

    /// Register an automatic phase transition rule, evaluated every substep.
    ///
    /// `rule` receives a particle and returns `Some(new_material_id)` to transition it,
    /// or `None` to leave it unchanged. Rules are checked in registration order;
    /// first match wins for each particle.
    ///
    /// All `new_material_id` values returned by the rule must be pre-registered via
    /// `solver.with_material(id, ...)` before any step is taken.
    ///
    /// Applies the same `latent_heat` energy debit as `phase_transition` -- see there.
    ///
    /// # Examples
    /// ```rust,ignore
    /// # extern crate emerge_engine as emerge;
    /// // Water freezes below 273 K
    /// solver.add_phase_rule(|p| {
    ///     if p.material_id == WATER_ID && p.temperature < 273.0 { Some(ICE_ID) } else { None }
    /// });
    /// // Rock melts above 1500 K
    /// solver.add_phase_rule(|p| {
    ///     if p.material_id == ROCK_ID && p.temperature > 1500.0 { Some(LAVA_ID) } else { None }
    /// });
    /// ```
    pub fn add_phase_rule<F>(&mut self, rule: F)
    where
        F: Fn(&Particle) -> Option<u32> + Send + Sync + 'static,
    {
        self.phase_rules.push(Box::new(rule));
    }

    /// Builder-style variant of `add_phase_rule`.
    pub fn with_phase_rule<F>(mut self, rule: F) -> Self
    where
        F: Fn(&Particle) -> Option<u32> + Send + Sync + 'static,
    {
        self.add_phase_rule(rule);
        self
    }

    /// Apply a velocity delta to all particles within `radius` of `center`, with linear falloff.
    /// `force` units: grid-cell/s (instantaneous velocity change).
    /// This applies the requested impulse exactly; it is not velocity-clamped.
    /// The next substep is selected from the resulting actual CFL state. Supply
    /// a physically meaningful impulse (or a resolved force history) rather
    /// than using this as a hidden settling/stability control.
    pub fn apply_impulse(&mut self, center: Vec2, radius: f32, force: Vec2) {
        let r2 = radius * radius;
        let mut to_wake = Vec::new();
        for i in 0..self.particles.len() {
            let d = self.particles.x[i] - center;
            let dist2 = d.length_squared();
            if dist2 <= r2 && dist2 > 1e-8 {
                if self.particles.sleeping[i] {
                    to_wake.push(i);
                }
                let falloff = 1.0 - (dist2 / r2).sqrt();
                self.particles.v[i] += force * falloff;
            }
        }
        for i in to_wake {
            self.wake_particle(i);
        }
    }

    /// Apply an outward radial velocity delta to particles within `radius`, with linear falloff.
    /// This applies the requested radial impulse exactly; the next substep
    /// uses the resulting actual CFL state.
    pub fn apply_radial_impulse(&mut self, center: Vec2, radius: f32, strength: f32) {
        let r2 = radius * radius;
        let mut to_wake = Vec::new();
        for i in 0..self.particles.len() {
            let d = self.particles.x[i] - center;
            let dist2 = d.length_squared();
            if dist2 <= r2 && dist2 > 1e-8 {
                if self.particles.sleeping[i] {
                    to_wake.push(i);
                }
                let dist = dist2.sqrt();
                let falloff = 1.0 - dist / radius;
                self.particles.v[i] += (d / dist) * strength * falloff;
            }
        }
        for i in to_wake {
            self.wake_particle(i);
        }
    }

    pub fn recompute_initial_volumes(&mut self) {
        estimate_particle_volumes(
            &mut self.particles,
            &mut self.grid,
            Some(&self.materials),
            self.active_count,
            true,
        );
    }

    /// Remove all particles where `predicate` returns true. Returns count removed.
    ///
    /// Uses stable retain (preserves order). O(N).
    ///
    /// LP pattern: tag particles with a sentinel before the step, then call
    /// `solver.remove_particles(|p| p.user_tag == DEAD)`. The tag-based group API
    /// remains valid after removal -- tag_index is rebuilt internally.
    pub fn remove_particles<F: Fn(&Particle) -> bool>(&mut self, predicate: F) -> usize {
        let before = self.particles.len();
        self.particles.retain(|p| !predicate(p));
        let removed = before - self.particles.len();
        if removed > 0 {
            // `retain()` compacts the array and shifts particle indices, but the
            // spatial hash caches indices from before the compaction and is
            // rebuilt only when `step()` marks it dirty (see
            // `ensure_spatial_hash_fresh`). Without marking it here, a
            // `particles_near`/`region_state`/`count_near` call after a
            // removal and before the next `step()` would read stale indices,
            // pointing at the wrong particle or past the shorter array (e.g.
            // several region removals or enrichments in a row).
            self.spatial_hash_dirty.set(true);
            // retain() compacted the array -- all physical indices in tag_index are stale.
            // Rebuild from scratch and re-establish the sleep partition.
            self.tag_index.clear();
            // Re-partition: move all sleeping particles to the back.
            let n = self.particles.len();
            let mut write = 0usize;
            for read in 0..n {
                if !self.particles.sleeping[read] {
                    if write != read {
                        self.particles.swap(write, read);
                    }
                    write += 1;
                }
            }
            self.active_count = write;
            // Rebuild tag_index over the freshly-partitioned array.
            for i in 0..n {
                self.tag_index
                    .entry(self.particles.user_tag[i])
                    .or_default()
                    .insert(i);
            }
        }
        removed
    }

    /// Oracle-triggered continuum -> discrete conversion, the "enrichment"
    /// direction of Hybrid Grains (Yue, Smith, Chen, Chantharayukhonthorn,
    /// Kamrin & Grinspun, ACM TOG 2018). Converts every active particle
    /// within `radius` of `center` matching `predicate` into one new discrete
    /// `Grain` in `grain_populations[population_idx]`, conserving summed mass
    /// and mass-weighted momentum (the grain's velocity) and placing it at the
    /// mass-weighted centre (a new body, unlike `grain_absorb_particles`,
    /// which keeps an existing grain's position). Radius from the 2D area,
    /// `area = sum(particle.volume)`, `radius = sqrt(area/pi)`. Consumed
    /// particles are removed with `remove_particles`. Returns the new grain's
    /// index in that population, or `None` if nothing in range matched.
    ///
    /// Only enrichment: `grains::oracle::needs_discrete_treatment` decides
    /// where it fires (typically a thin, low-packing-fraction free-surface
    /// cell, where continuum-only sand does not hold a repose angle). The
    /// reverse direction, homogenization (a settled grain back into continuum
    /// particles, via Christoffersen et al. 1981's discrete-to-continuum
    /// stress mapping as the paper does), is not implemented: grains never
    /// convert back.
    pub fn enrich_region_into_grain(
        &mut self,
        population_idx: usize,
        center: Vec2,
        radius: f32,
        predicate: impl Fn(&Particle) -> bool,
    ) -> Option<usize> {
        let nearby = self.particles_near(center, radius);

        // Reserved sentinel, not `Particle::user_tag` -- same real
        // justification `grain_absorb_particles`'s own precedent
        // established: that field is caller-defined (LP uses it for
        // creature ownership), stomping it even transiently is a real
        // correctness risk. `material_id` is safe here: set and the
        // particle removed within this single synchronous call, no
        // `step()` (the only place `material_id` is actually dispatched
        // on) running in between.
        const ENRICHED_SENTINEL: u32 = u32::MAX;
        let mut sum_mass = 0.0f32;
        let mut sum_momentum = Vec2::ZERO;
        let mut sum_area = 0.0f32;
        let mut weighted_center = Vec2::ZERO;
        let mut consumed = 0usize;
        for i in nearby {
            let p = self.particles.get(i);
            if !predicate(&p) {
                continue;
            }
            sum_mass += p.mass;
            sum_momentum += p.mass * p.v;
            sum_area += p.volume;
            weighted_center += p.mass * p.x;
            self.particles.material_id[i] = ENRICHED_SENTINEL;
            consumed += 1;
        }
        if consumed == 0 || sum_mass <= 0.0 {
            return None;
        }

        let grain_x = weighted_center / sum_mass;
        let grain_v = sum_momentum / sum_mass;
        let radius = (sum_area / std::f32::consts::PI).sqrt();
        let mut grain = crate::particle::Grain::new(grain_x, radius, sum_mass);
        grain.v = grain_v;

        self.remove_particles(|p| p.material_id == ENRICHED_SENTINEL);
        self.grain_populations[population_idx].grains.push(grain);
        Some(self.grain_populations[population_idx].grains.len() - 1)
    }

    /// Iterate physical indices of all particles with `tag`. O(group_size) via tag_index.
    ///
    /// Returns indices only -- read particle data via `solver.particles().x[i]` etc.
    /// This avoids cloning 112B per particle on every call.
    pub fn particles_with_tag(&self, tag: u32) -> impl Iterator<Item = usize> + '_ {
        self.tag_index
            .get(&tag)
            .into_iter()
            .flat_map(|s| s.iter())
            .copied()
    }

    // ── Sleep / wake ─────────────────────────────────────────────────────────

    /// Put particle at physical index `i` to sleep.
    ///
    /// Swaps it with the last active particle, decrementing `active_count`.
    /// Updates `tag_index` for both affected particles.
    /// No-op if already sleeping.
    pub(super) fn sleep_particle(&mut self, i: usize) {
        if self.particles.sleeping[i] || self.active_count == 0 {
            return;
        }
        self.particles.sleeping[i] = true;
        let last_active = self.active_count - 1;
        if i != last_active {
            let tag_i = self.particles.user_tag[i];
            let tag_j = self.particles.user_tag[last_active];
            // Same-tag swap: both indices stay in the same set -- no update needed.
            // Different-tag swap: each particle moves to the other's former position.
            if tag_i != tag_j {
                Self::tag_index_replace(&mut self.tag_index, tag_i, i, last_active);
                Self::tag_index_replace(&mut self.tag_index, tag_j, last_active, i);
            }
            self.particles.swap(i, last_active);
        }
        self.active_count -= 1;
    }

    /// Wake particle at physical index `i`.
    ///
    /// Swaps it with the first sleeping particle, incrementing `active_count`.
    /// Updates `tag_index` for both affected particles.
    /// No-op if already awake.
    pub(super) fn wake_particle(&mut self, i: usize) {
        if !self.particles.sleeping[i] {
            return;
        }
        self.particles.sleeping[i] = false;
        let first_sleeping = self.active_count;
        if i != first_sleeping {
            let tag_i = self.particles.user_tag[i];
            let tag_j = self.particles.user_tag[first_sleeping];
            if tag_i != tag_j {
                Self::tag_index_replace(&mut self.tag_index, tag_i, i, first_sleeping);
                Self::tag_index_replace(&mut self.tag_index, tag_j, first_sleeping, i);
            }
            self.particles.swap(i, first_sleeping);
        }
        self.active_count += 1;
    }

    /// Wake all particles belonging to `tag`. O(group_size · log group_size).
    pub fn wake_tag(&mut self, tag: u32) {
        // Snapshot sleeping indices, then wake ascending.
        // wake_particle(i) swaps i↔first_sleeping (low end of sleeping zone).
        // Ascending order: each displacement lands at an already-processed lower position,
        // so no sleeping tag particle is silently displaced to an unvisited index.
        let mut to_wake: Vec<usize> = self
            .tag_index
            .get(&tag)
            .map(|s| {
                s.iter()
                    .filter(|&&i| self.particles.sleeping[i])
                    .copied()
                    .collect()
            })
            .unwrap_or_default();
        to_wake.sort_unstable();
        for i in to_wake {
            self.wake_particle(i);
        }
    }

    /// Sleep all particles belonging to `tag`. O(group_size · log group_size).
    pub fn sleep_tag(&mut self, tag: u32) {
        // Snapshot active indices, then sleep descending.
        // sleep_particle(i) swaps i↔last_active (high end of active zone).
        // Descending order: each displacement lands at an already-processed higher position.
        let mut to_sleep: Vec<usize> = self
            .tag_index
            .get(&tag)
            .map(|s| {
                s.iter()
                    .filter(|&&i| !self.particles.sleeping[i])
                    .copied()
                    .collect()
            })
            .unwrap_or_default();
        to_sleep.sort_unstable_by(|a, b| b.cmp(a));
        for i in to_sleep {
            self.sleep_particle(i);
        }
    }

    /// Number of currently active (non-sleeping) particles.
    pub const fn active_count(&self) -> usize {
        self.active_count
    }

    fn tag_index_replace(
        tag_index: &mut HashMap<u32, HashSet<usize>>,
        tag: u32,
        old_idx: usize,
        new_idx: usize,
    ) {
        if let Some(s) = tag_index.get_mut(&tag) {
            s.remove(&old_idx);
            s.insert(new_idx);
        }
    }

    /// Collect all particles into a `Vec<Particle>` (for diagnostics or GPU upload).
    pub fn collect_particles(&self) -> Vec<Particle> {
        self.particles.to_vec()
    }

    /// Spawn particles, tag them, and return the stable tag.
    ///
    /// The returned `u32` tag is the only stable identity for this group -- physical
    /// indices change whenever particles sleep or wake. Pass it to `group_state`,
    /// `set_group_activation`, `group_centroid`, etc.
    ///
    /// All particles in the region are stamped with `user_tag = tag` (overrides
    /// any `user_tag` set in the SpawnRegion).
    ///
    /// ```rust,no_run
    /// # extern crate emerge_engine as emerge;
    /// # use emerge::solver::Simulation;
    /// # use emerge::{SimConfig, SpawnRegion};
    /// # let config = SimConfig::standard(64, 0.05, glam::Vec2::NEG_Y);
    /// # let mut solver = Simulation::empty(config);
    /// let creature = solver.add_body(SpawnRegion::for_sim(&config));
    /// solver.set_group_activation(creature, 1.0);
    /// let centroid = solver.group_centroid(creature);
    /// ```
    #[must_use = "store the tag -- it is the only stable identity for this group"]
    pub fn add_body(&mut self, spawn: SpawnRegion) -> u32 {
        let tag = self.next_tag;
        self.next_tag += 1;

        let old_active = self.active_count;
        let old_len = self.particles.len();
        // sleeping zone is [old_active..old_len] -- new particles must land before it.

        spawn.validate_for_sim(&self.config);
        debug_assert!(
            self.materials.is_registered(spawn.material_id),
            "add_body: material_id {} is not registered",
            spawn.material_id,
        );
        let mut rng = LcgRng::new(spawn.rng_seed);
        let new_particles = initialize_particles(&self.config, spawn, &mut rng);
        for p in new_particles {
            self.particles.push(p);
        }
        let new_len = self.particles.len();
        let new_count = new_len - old_len;
        let sleeping_count = old_len - old_active;

        // Stamp the tag (new particles still at [old_len..new_len]). The
        // material's `init_particle` runs after the volume estimate below,
        // not here: see the estimate's comment for why the order matters.
        let mat_id = spawn.material_id;
        for i in old_len..new_len {
            self.particles.user_tag[i] = tag;
        }

        // If sleeping particles sit between the active zone and the new particles, rotate new
        // particles before the sleeping zone so the partition invariant is maintained:
        //   before: [0..old_active] active | [old_active..old_len] sleeping | [old_len..new_len] new
        //   after:  [0..old_active] active | [old_active..old_active+new_count] new | [...] sleeping
        if sleeping_count > 0 {
            self.particles.rotate_range(old_active, old_len, new_len);
            // Sleeping particles moved from [old_active+k] → [old_active+new_count+k].
            // Update tag_index for each displaced sleeping particle.
            for k in 0..sleeping_count {
                let old_pos = old_active + k;
                let new_pos = old_active + new_count + k;
                let t = self.particles.user_tag[new_pos];
                Self::tag_index_replace(&mut self.tag_index, t, old_pos, new_pos);
            }
        }

        // New particles are at [old_active..old_active+new_count].
        let group_start = old_active;
        let group_end = old_active + new_count;
        self.tag_index
            .insert(tag, (group_start..group_end).collect::<HashSet<usize>>());
        self.active_count = group_end;

        // Scatter only particles in the spawn region + 3-cell margin.
        // O(active_count) scan but O(local × stencil) grid work -- fast for sparse spawns.
        density::estimate_particle_volumes_local(
            &mut self.particles,
            &mut self.grid,
            Some(&self.materials),
            self.active_count,
            group_start,
            true,
        );
        // THEN the material's own initialisation, never before it, because
        // `Simulation::new` does it in exactly this order and the two paths
        // must produce the same body. They did not. `new` estimates and is
        // then followed by `with_default_material`, which reinitialises
        // every particle, so a material that sets its own initial volume
        // has the last word there. If `init_particle` ran first here, the
        // estimate would have the last word instead, and a free-surface
        // particle's estimated volume is up to 2.56 times its packing
        // volume. Initial volume multiplies stress directly, so those
        // particles would push that much too hard.
        //
        // Measured on three IDENTICAL columns in one world
        // (`tests/probes/bingham_column_volume_loss.rs`): the body `new`
        // started with held mean J = 0.99907 while the two this function
        // added crushed to 0.94304, worst particle 0.603 against 0.986.
        // Six material families set their own volume this way and all six
        // are fluids or gases, so every fluid body added at runtime carried
        // it.
        if self.materials.is_registered(mat_id) {
            for i in group_start..group_end {
                let mut p = self.particles.get(i);
                self.materials.get(mat_id).init_particle(&mut p);
                self.particles.set(i, p);
            }
        }
        self.spatial_hash
            .borrow_mut()
            .rebuild(&self.particles.x, self.active_count);
        self.spatial_hash_dirty.set(false);
        tag
    }

    /// Attach a scalar diffusion field (pheromone, nutrients, morphogen).
    ///
    /// Attached fields are applied automatically every substep -- LP does not need
    /// to call `field.apply()` manually.
    ///
    /// ```rust,no_run
    /// # extern crate emerge_engine as emerge;
    /// # use emerge::solver::Simulation;
    /// # use emerge::{SimConfig, SpawnRegion, ScalarDiffusionConfig, ScalarDiffusionField};
    /// # let config = SimConfig::standard(64, 0.05, glam::Vec2::NEG_Y);
    /// # let mut solver = Simulation::new(config, SpawnRegion::default());
    /// let pheromone = ScalarDiffusionField::new(
    ///     ScalarDiffusionConfig { diffusivity: 0.5, decay_rate: 0.1, ambient: 0.0 },
    ///     |p| p.temperature,
    ///     |p, d| p.temperature += d,
    ///     64,
    /// );
    /// solver.attach_scalar_field(pheromone);
    /// // No manual field.apply() needed -- runs automatically in solver.step()
    /// ```
    pub fn attach_scalar_field(&mut self, field: ScalarDiffusionField) {
        self.scalar_fields.push(field);
    }

    /// Builder variant of `attach_scalar_field`.
    pub fn with_scalar_field(mut self, field: ScalarDiffusionField) -> Self {
        self.attach_scalar_field(field);
        self
    }
}

#[cfg(test)]
mod hydrostatic_tests {
    use crate::{NewtonianFluidMaterial, SimConfig, Simulation, SpawnRegion};
    use glam::{IVec2, Vec2};

    fn resting_pool(settle: bool) -> Simulation {
        let config = SimConfig {
            // A properly stiff fluid has a high sound speed and needs the
            // substeps to match; the cap is generous so the test measures
            // the initial condition, not the substep budget.
            max_substeps_per_step: 4000,
            ..SimConfig::earth(64, 0.01, 0.01)
        };
        let spacing = 0.5;
        let rest_density = 0.1;
        let pool = SpawnRegion {
            spacing,
            box_size: IVec2::new(40, 12),
            box_center: Vec2::new(32.0, 9.0),
            material_id: 0,
            mass_override: Some(rest_density * spacing * spacing),
            ..SpawnRegion::for_sim(&config)
        };
        // Stiffness sized so the pool is weakly compressible at this
        // depth: too soft and the "equilibrium" is a 37% squash, which is not
        // what the Tait equation is for.
        let water = NewtonianFluidMaterial::new(rest_density, 0.001, 1.0e5, 7.0);
        let mut sim = Simulation::new(config, pool).with_default_material(Box::new(water));
        if settle {
            sim.settle_hydrostatic();
        }
        sim
    }

    /// A pool spawned at uniform density is not at rest: it has no internal
    /// pressure to hold itself up, so it collapses and rings. Settled
    /// hydrostatically it starts in equilibrium and stays far quieter.
    ///
    /// This is the difference a viewer sees as a "resting" pool twitching
    /// before anything touches it.
    #[test]
    fn settling_hydrostatically_makes_a_resting_pool_actually_rest() {
        let mut uniform = resting_pool(false);
        let mut settled = resting_pool(true);
        for _ in 0..12 {
            uniform.step();
            settled.step();
        }
        let peak = |sim: &Simulation| {
            sim.particles()
                .v
                .iter()
                .fold(0.0f32, |acc, v| acc.max(v.length()))
        };
        let (noisy, quiet) = (peak(&uniform), peak(&settled));
        assert!(
            quiet < noisy,
            "a hydrostatically settled pool must be quieter than one spawned \
             at uniform density: settled peak {quiet:.4} vs uniform {noisy:.4}"
        );
    }

    /// The equilibrium is a gradient, not a uniform offset: deeper
    /// matter is more compressed, because it carries more weight.
    #[test]
    fn settled_pool_is_more_compressed_with_depth() {
        let sim = resting_pool(true);
        let det = |m: glam::Mat2| m.x_axis.x * m.y_axis.y - m.x_axis.y * m.y_axis.x;
        let particles = sim.particles();
        let (mut lowest, mut highest) = ((f32::MAX, 1.0f32), (f32::MIN, 1.0f32));
        for i in 0..particles.len() {
            let (y, j) = (particles.x[i].y, det(particles.deformation_gradient[i]));
            if y < lowest.0 {
                lowest = (y, j);
            }
            if y > highest.0 {
                highest = (y, j);
            }
        }
        assert!(
            lowest.1 < highest.1,
            "the bottom must be compressed more than the top: J={:.6} at y={:.2} \
             vs J={:.6} at y={:.2}",
            lowest.1,
            lowest.0,
            highest.1,
            highest.0
        );
        assert!(highest.1 <= 1.0, "the free surface must not be stretched");
    }
}

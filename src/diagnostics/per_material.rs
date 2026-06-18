use glam::Vec2;
use std::collections::BTreeMap;

use crate::diagnostics::rules::{StabilityThresholds, evaluate_stability};
use crate::diagnostics::snapshot::SimSnapshot;
use crate::particle::{Particle, Particles};

/// Per-material aggregate statistics — one entry per unique `material_id`.
///
/// Computed in a single pass over the particle slice.
/// Cheap enough to call every frame; only allocates one BTreeMap entry per material.
#[derive(Debug, Clone)]
pub struct MaterialStats {
    pub material_id: u32,
    pub count: usize,
    pub centroid: Vec2,
    /// Mean speed |v|.
    pub mean_speed: f32,
    /// Speed of fastest particle.
    pub max_speed: f32,
    /// [min, max] of det(F) — J < 1 = compressed, J > 1 = expanded.
    pub j_range: [f32; 2],
    /// Mean plastic volume ratio Jp = det(Fₚ). 1.0 = no plastic deformation.
    /// Drops below 1.0 when material compresses plastically (snow after impact).
    pub avg_plastic_volume_ratio: f32,
    /// Mean elastic hardening multiplier h = exp(ξ*(1−Jp)). 1.0 = no hardening.
    /// Rises above 1.0 for compacted snow.
    pub avg_hardening_scale: f32,
    /// Mean activation ∈ [0,1]. 0 if no particles have activation.
    pub mean_activation: f32,
    /// Max activation. Useful to confirm the oscillator is reaching full contraction.
    pub max_activation: f32,
    /// Mean damage (friction_hardening for Rankine; q accumulator for sand/VonMises).
    /// Meaning depends on material — use as a relative indicator.
    pub mean_damage: f32,
    /// Mean temperature.
    pub mean_temperature: f32,
}

impl MaterialStats {
    /// Format as a compact single line.
    ///
    /// Only prints non-zero optional fields to reduce noise.
    /// Example output:
    /// ```text
    /// [mat 0 snow  ] n=256  cx=(32.1,19.3)  v=0.023/0.041  J=[0.91,1.05]  Jp=0.982 h=1.027
    /// [mat 1 sand  ] n=512  cx=(40.0,12.0)  v=0.010/0.022  J=[0.98,1.01]  q=0.382
    /// [mat 2 jelly ] n=256  cx=(20.0,30.0)  v=1.200/2.100  J=[0.85,1.20]  act=0.80/1.00  T=36.50
    /// ```
    pub fn format(&self, label: Option<&str>) -> String {
        let tag = match label {
            Some(l) => format!("[mat {} {:<6}]", self.material_id, l),
            None => format!("[mat {:<8}]", self.material_id),
        };
        let mut s = format!(
            "{} n={:<5} cx=({:6.1},{:6.1})  v={:.3}/{:.3}  J=[{:.3},{:.3}]",
            tag,
            self.count,
            self.centroid.x,
            self.centroid.y,
            self.mean_speed,
            self.max_speed,
            self.j_range[0],
            self.j_range[1],
        );
        // Plastic state — only when deformed.
        if (self.avg_plastic_volume_ratio - 1.0).abs() > 1e-4 {
            s.push_str(&format!("  Jp={:.3}", self.avg_plastic_volume_ratio));
        }
        if (self.avg_hardening_scale - 1.0).abs() > 1e-4 {
            s.push_str(&format!("  h={:.3}", self.avg_hardening_scale));
        }
        // friction_hardening: DP q, VonMises κ, Rankine damage, SandMuI µ(I).
        // Only print when non-zero — name it generically as "q" (internal state).
        if self.mean_damage.abs() > 1e-4 {
            s.push_str(&format!("  q={:.3}", self.mean_damage));
        }
        // Activation — only for creature materials.
        if self.max_activation > 1e-4 {
            s.push_str(&format!(
                "  act={:.2}/{:.2}",
                self.mean_activation, self.max_activation
            ));
        }
        // Temperature — only when non-zero.
        if self.mean_temperature.abs() > 1e-4 {
            s.push_str(&format!("  T={:.2}", self.mean_temperature));
        }
        s
    }
}

/// Compute per-material stats in a single particle pass.
///
/// Returns one `MaterialStats` per unique `material_id`, sorted by id.
/// Only allocates a BTreeMap entry per material — O(n·log(m)) where m = number of materials.
pub fn per_material_stats(particles: &Particles) -> Vec<MaterialStats> {
    per_material_stats_iter(particles.iter())
}

/// Same as `per_material_stats` but accepts a `&[Particle]` slice (AOS layout).
/// Useful for GPU examples where the CPU mirror is a `Vec<Particle>`.
pub fn per_material_stats_of(particles: &[Particle]) -> Vec<MaterialStats> {
    per_material_stats_iter(particles.iter().copied())
}

fn per_material_stats_iter(iter: impl Iterator<Item = Particle>) -> Vec<MaterialStats> {
    struct Accum {
        count: usize,
        centroid_sum: Vec2,
        speed_sum: f32,
        max_speed: f32,
        j_min: f32,
        j_max: f32,
        plastic_volume_ratio_sum: f32,
        hardening_scale_sum: f32,
        activation_sum: f32,
        max_activation: f32,
        damage_sum: f32,
        temperature_sum: f32,
    }

    let mut map: BTreeMap<u32, Accum> = BTreeMap::new();

    for p in iter {
        let entry = map.entry(p.material_id).or_insert(Accum {
            count: 0,
            centroid_sum: Vec2::ZERO,
            speed_sum: 0.0,
            max_speed: 0.0,
            j_min: f32::INFINITY,
            j_max: f32::NEG_INFINITY,
            plastic_volume_ratio_sum: 0.0,
            hardening_scale_sum: 0.0,
            activation_sum: 0.0,
            max_activation: 0.0,
            damage_sum: 0.0,
            temperature_sum: 0.0,
        });

        entry.count += 1;
        entry.centroid_sum += p.x;

        let speed = p.v.length();
        entry.speed_sum += speed;
        entry.max_speed = entry.max_speed.max(speed);

        let j = p.deformation_gradient.determinant();
        if j.is_finite() {
            entry.j_min = entry.j_min.min(j);
            entry.j_max = entry.j_max.max(j);
        }

        if p.plastic_volume_ratio.is_finite() {
            entry.plastic_volume_ratio_sum += p.plastic_volume_ratio;
        }
        if p.hardening_scale.is_finite() {
            entry.hardening_scale_sum += p.hardening_scale;
        }

        entry.activation_sum += p.activation;
        entry.max_activation = entry.max_activation.max(p.activation);
        entry.damage_sum += p.friction_hardening;
        entry.temperature_sum += p.temperature;
    }

    map.into_iter()
        .map(|(id, a)| {
            let n = a.count as f32;
            MaterialStats {
                material_id: id,
                count: a.count,
                centroid: if a.count > 0 {
                    a.centroid_sum / n
                } else {
                    Vec2::ZERO
                },
                mean_speed: a.speed_sum / n,
                max_speed: a.max_speed,
                j_range: [
                    if a.j_min.is_infinite() { 1.0 } else { a.j_min },
                    if a.j_max.is_infinite() { 1.0 } else { a.j_max },
                ],
                avg_plastic_volume_ratio: a.plastic_volume_ratio_sum / n,
                avg_hardening_scale: a.hardening_scale_sum / n,
                mean_activation: a.activation_sum / n,
                max_activation: a.max_activation,
                mean_damage: a.damage_sum / n,
                mean_temperature: a.temperature_sum / n,
            }
        })
        .collect()
}

/// Print a clean per-frame summary: header + one line per material.
///
/// Pass `labels` to annotate material IDs with names (e.g. `&[(0, "snow"), (1, "sand")]`).
/// Only prints if `frame % interval == 0` — set `interval = 1` to print every frame.
pub fn log_frame(
    frame: u64,
    dt: f32,
    particles: &Particles,
    labels: &[(u32, &str)],
    interval: u64,
) {
    if interval > 0 && !frame.is_multiple_of(interval) {
        return;
    }
    let stats = per_material_stats(particles);
    println!("── frame {}  dt={:.4}  n={} ──", frame, dt, particles.len());
    for s in &stats {
        let label = labels
            .iter()
            .find(|(id, _)| *id == s.material_id)
            .map(|(_, l)| *l);
        println!("  {}", s.format(label));
    }
}

/// Print per-frame summary + global health status (CFL, mass/momentum conservation, NaN checks).
///
/// Header shows CFL and health — if unhealthy, the violated checks are listed.
/// Per-material lines follow (same format as `log_frame`).
///
/// Use `solver.diagnostics_snapshot()` to obtain the snapshot.
/// Only prints if `frame % interval == 0`.
///
/// Example output:
/// ```text
/// ── frame 120  dt=0.0500  n=256  cfl=0.421  J=[0.91,1.08]  health=OK ──
///   [mat 0 snow  ] n=256   cx=(32.1,19.3)  v=0.023/0.041  J=[0.91,1.05]  Jp=0.982 h=1.027  act=0.00/0.00  dmg=0.00  T=0.00
/// ```
pub fn log_frame_full(
    frame: u64,
    dt: f32,
    particles: &Particles,
    labels: &[(u32, &str)],
    snapshot: &SimSnapshot,
    interval: u64,
) {
    if interval > 0 && !frame.is_multiple_of(interval) {
        return;
    }
    let health = evaluate_stability(snapshot, &StabilityThresholds::default());
    let health_tag = if health.healthy() {
        "OK".to_string()
    } else {
        format!("WARN[{}]", health.issue_labels().join(","))
    };
    let mut extras = String::new();
    if snapshot.vel_clamp_count > 0 {
        extras.push_str(&format!("  vel_clamp={}", snapshot.vel_clamp_count));
    }
    if snapshot.j_projection_count > 0 {
        extras.push_str(&format!("  j_proj={}", snapshot.j_projection_count));
    }
    if snapshot.sim_time_dropped > 1e-6 {
        extras.push_str(&format!("  time_drop={:.4}", snapshot.sim_time_dropped));
    }
    // Show active/sleeping only when sleep system is in use.
    let particle_info = if snapshot.sleeping_count > 0 {
        format!(
            "n={}  active={}  sleep={}",
            particles.len(),
            snapshot.active_count,
            snapshot.sleeping_count,
        )
    } else {
        format!("n={}", particles.len())
    };
    let substep_info = if snapshot.substeps_last_step > 1 {
        format!("  sub={}", snapshot.substeps_last_step)
    } else {
        String::new()
    };
    println!(
        "── frame {}  dt={:.4}  {}  cfl={:.3}  J=[{:.3},{:.3}]{}  health={}{} ──",
        frame,
        dt,
        particle_info,
        snapshot.cfl_number,
        snapshot.min_deformation_j,
        snapshot.max_deformation_j,
        substep_info,
        health_tag,
        extras,
    );
    let stats = per_material_stats(particles);
    for s in &stats {
        let label = labels
            .iter()
            .find(|(id, _)| *id == s.material_id)
            .map(|(_, l)| *l);
        println!("  {}", s.format(label));
    }
}

/// GPU-compatible per-frame log. Takes a `&[Particle]` slice (CPU mirror from GpuSimulation).
///
/// No CFL or health check — those require grid data unavailable on the GPU path.
/// Shows global J range computed from the particle mirror (1-frame lag vs GPU state).
/// Format: `── frame N  dt=D  n=N  J=[min,max]  [GPU] ──` followed by per-material lines.
/// Only prints if `frame % interval == 0`.
pub fn log_frame_gpu(
    frame: u64,
    dt: f32,
    particles: &[Particle],
    labels: &[(u32, &str)],
    interval: u64,
) {
    if interval > 0 && !frame.is_multiple_of(interval) {
        return;
    }
    let stats = per_material_stats_of(particles);
    let (j_min, j_max) = stats
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), s| {
            (lo.min(s.j_range[0]), hi.max(s.j_range[1]))
        });
    let (j_min, j_max) = if j_min.is_infinite() {
        (1.0, 1.0)
    } else {
        (j_min, j_max)
    };
    println!(
        "── frame {}  dt={:.4}  n={}  J=[{:.3},{:.3}]  [GPU] ──",
        frame,
        dt,
        particles.len(),
        j_min,
        j_max,
    );
    for s in &stats {
        let label = labels
            .iter()
            .find(|(id, _)| *id == s.material_id)
            .map(|(_, l)| *l);
        println!("  {}", s.format(label));
    }
}

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::diagnostics::per_material::MaterialStats;
use crate::diagnostics::snapshot::MpmSnapshot;

/// NDJSON frame logger — one JSON object per line, one file per run.
///
/// Each `log()` call appends one line. The file is flushed immediately so
/// `tail -f run.ndjson | jq` gives live output during a simulation.
///
/// # Usage
/// ```ignore
/// let mut logger = FrameLogger::open("run.ndjson").unwrap();
/// // inside loop:
/// logger.log(frame, dt, &stats, &snap, labels);
/// ```
///
/// # Output format
/// ```json
/// {"frame":60,"dt":0.05,"active":2176,"sleeping":0,"substeps":4,"cfl":0.033,"j":[0.97,1.34],"health":"OK","materials":[...]}
/// ```
pub struct FrameLogger {
    writer: BufWriter<File>,
}

impl FrameLogger {
    /// Open (or create) an NDJSON log file. Truncates on open.
    pub fn open(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;
        Ok(Self {
            writer: BufWriter::new(file),
        })
    }

    /// Append one frame line. Labels map material_id → name (same as `log_frame_full`).
    pub fn log(
        &mut self,
        frame: u64,
        dt: f32,
        stats: &[MaterialStats],
        snap: &MpmSnapshot,
        labels: &[(u32, &str)],
    ) {
        let health = if snap.non_finite_particle_values > 0 || snap.invalid_physical_particle_values > 0 {
            "WARN"
        } else {
            "OK"
        };

        let mut line = format!(
            "{{\"frame\":{},\"dt\":{:.4},\"active\":{},\"sleeping\":{},\"substeps\":{},\"cfl\":{:.4},\"j\":[{:.4},{:.4}],\"health\":\"{}\"",
            frame,
            dt,
            snap.active_count,
            snap.sleeping_count,
            snap.substeps_last_step,
            snap.cfl_number,
            snap.min_deformation_j,
            snap.max_deformation_j,
            health,
        );

        // Optional warn fields — only when non-zero.
        if snap.vel_clamp_count > 0 {
            line.push_str(&format!(",\"vel_clamp\":{}", snap.vel_clamp_count));
        }
        if snap.j_projection_count > 0 {
            line.push_str(&format!(",\"j_proj\":{}", snap.j_projection_count));
        }
        if snap.non_finite_particle_values > 0 {
            line.push_str(&format!(",\"nan_particles\":{}", snap.non_finite_particle_values));
        }

        // Per-material array.
        line.push_str(",\"materials\":[");
        for (i, s) in stats.iter().enumerate() {
            if i > 0 {
                line.push(',');
            }
            let name = labels
                .iter()
                .find(|(id, _)| *id == s.material_id)
                .map(|(_, n)| *n)
                .unwrap_or("unknown");

            line.push_str(&format!(
                "{{\"id\":{},\"name\":\"{}\",\"n\":{},\"cx\":[{:.2},{:.2}],\"v_mean\":{:.4},\"v_max\":{:.4},\"j\":[{:.4},{:.4}]",
                s.material_id,
                name,
                s.count,
                s.centroid.x,
                s.centroid.y,
                s.mean_speed,
                s.max_speed,
                s.j_range[0],
                s.j_range[1],
            ));
            // Optional per-material fields.
            if (s.avg_plastic_volume_ratio - 1.0).abs() > 1e-4 {
                line.push_str(&format!(",\"jp\":{:.4}", s.avg_plastic_volume_ratio));
            }
            if (s.avg_hardening_scale - 1.0).abs() > 1e-4 {
                line.push_str(&format!(",\"h\":{:.4}", s.avg_hardening_scale));
            }
            if s.mean_damage.abs() > 1e-4 {
                line.push_str(&format!(",\"q\":{:.4}", s.mean_damage));
            }
            if s.max_activation > 1e-4 {
                line.push_str(&format!(",\"act_mean\":{:.4},\"act_max\":{:.4}", s.mean_activation, s.max_activation));
            }
            if s.mean_temperature.abs() > 1e-4 {
                line.push_str(&format!(",\"T\":{:.4}", s.mean_temperature));
            }
            line.push('}');
        }
        line.push_str("]}");

        let _ = writeln!(self.writer, "{}", line);
        let _ = self.writer.flush();
    }
}

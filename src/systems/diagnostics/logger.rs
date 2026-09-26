use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::diagnostics::per_material::MaterialStats;
use crate::diagnostics::snapshot::SimSnapshot;

/// NDJSON frame logger -- one JSON object per line, one file per run.
///
/// Each `log()` call appends one line. The underlying OS write is flushed
/// every `FLUSH_EVERY` calls (real, measured necessity, not a style choice
/// -- see this const's own doc), not every single call, so `tail -f
/// run.ndjson | jq` still gives live output during a simulation (well under
/// a second behind at any real frame rate) without paying a real disk-sync
/// cost every rendered frame. `BufWriter`'s own `Drop` impl flushes its
/// internal buffer on ordinary program exit regardless (so a normal quit
/// never loses data) -- the ONLY real, disclosed tradeoff of not flushing
/// every call is that a hard crash/panic can lose up to `FLUSH_EVERY-1`
/// frames of log data instead of zero, a real, acceptable cost for a
/// diagnostics/telemetry log, not gameplay-critical state.
///
/// # Usage
/// ```ignore
/// let mut logger = FrameLogger::open("run.ndjson").unwrap();
/// // inside loop:
/// logger.log(frame, dt, &stats, &snap, labels, &[]);
/// ```
///
/// # Output format
/// ```json
/// {"frame":60,"dt":0.05,"active":2176,"sleeping":0,"substeps":4,"cfl":0.033,"j":[0.97,1.34],"health":"OK","materials":[...]}
/// ```
pub struct FrameLogger {
    writer: BufWriter<File>,
    calls_since_flush: usize,
}

/// Real, measured choice (2026-09-10): flushing every single call was
/// found to be a genuine, real fps bottleneck completely independent of
/// particle count or grid resolution -- `basic_plant.rs` (37 particles)
/// measured 22-24fps in the same real audit that found `basic_sand.rs`'s
/// own real material-stiffness gap, and `Write::flush` on a `File` forces a
/// real OS-level write-through (often several ms on Windows, filesystem/AV
/// filters included) EVERY call. 30 calls (~0.5s at 60fps, ~1s at 30fps)
/// keeps `tail -f` feeling live to a human watching, while cutting the
/// real per-frame syscall cost by ~30x.
const FLUSH_EVERY: usize = 30;

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
            calls_since_flush: 0,
        })
    }

    /// Append one frame line. Labels map material_id → name (same as `log_frame_full`).
    ///
    /// `extra` is an optional list of app-defined scalar fields (e.g. a demo's
    /// live steer input or wave speed) merged into the top-level JSON object --
    /// context the engine has no name for, but that matters when replaying a
    /// run's telemetry (why did the body do that at frame N?).
    pub fn log(
        &mut self,
        frame: u64,
        dt: f32,
        stats: &[MaterialStats],
        snap: &SimSnapshot,
        labels: &[(u32, &str)],
        extra: &[(&str, f32)],
    ) {
        let health =
            if snap.non_finite_particle_values > 0 || snap.invalid_physical_particle_values > 0 {
                "WARN"
            } else {
                "OK"
            };

        let mut line = format!(
            "{{\"frame\":{},\"dt\":{:.4},\"active\":{},\"sleeping\":{},\"substeps\":{},\"cfl\":{:.4},\"j\":[{:.4},{:.4}],\"ke\":{:.4},\"health\":\"{}\"",
            frame,
            dt,
            snap.active_count,
            snap.sleeping_count,
            snap.substeps_last_step,
            snap.cfl_number,
            snap.min_deformation_j,
            snap.max_deformation_j,
            snap.total_kinetic_energy,
            health,
        );

        // Real, generic sanity check: any pinned/Dirichlet-anchored particle should
        // read exactly v=0 (see `SimSnapshot::max_pinned_particle_speed`'s own doc) --
        // only emitted when the scene actually uses `Particle::pinned` (nonzero here
        // means either real motion at an anchor -- a genuine engine bug -- or, more
        // often, that no particle is pinned at all, in which case this stays absent).
        if snap.max_pinned_particle_speed > 0.0 {
            line.push_str(&format!(
                ",\"pinned_v\":{:.6}",
                snap.max_pinned_particle_speed
            ));
        }

        // Optional warn fields -- only when non-zero.
        if snap.vel_clamp_count > 0 {
            line.push_str(&format!(",\"vel_clamp\":{}", snap.vel_clamp_count));
        }
        if snap.j_projection_count > 0 {
            line.push_str(&format!(",\"j_proj\":{}", snap.j_projection_count));
        }
        if snap.non_finite_particle_values > 0 {
            line.push_str(&format!(
                ",\"nan_particles\":{}",
                snap.non_finite_particle_values
            ));
        }

        // Rod-solver diagnostics -- only emitted when the scene actually has
        // rods (`snap.rods.count > 0`), so a particle-only scene's log stays
        // exactly as it was before this field existed.
        if snap.rods.count > 0 {
            line.push_str(&format!(
                ",\"rods\":{{\"n\":{},\"sleeping\":{},\"v_max\":{:.4},\"tips\":[",
                snap.rods.count, snap.rods.sleeping_count, snap.rods.max_speed,
            ));
            for (i, tip) in snap.rods.tip_positions.iter().enumerate() {
                if i > 0 {
                    line.push(',');
                }
                line.push_str(&format!("[{:.4},{:.4}]", tip.x, tip.y));
            }
            line.push_str("]}");
        }

        // App-defined scalar context (e.g. live steer input, wave speed).
        for (name, value) in extra {
            line.push_str(&format!(",\"{}\":{:.4}", name, value));
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
                "{{\"id\":{},\"name\":\"{}\",\"n\":{},\"cx\":[{:.2},{:.2}],\"extent\":[{:.2},{:.2}],\"v_mean\":{:.4},\"v_max\":{:.4},\"j\":[{:.4},{:.4}]",
                s.material_id,
                name,
                s.count,
                s.centroid.x,
                s.centroid.y,
                s.extent_max.x - s.extent_min.x,
                s.extent_max.y - s.extent_min.y,
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
                line.push_str(&format!(
                    ",\"act_mean\":{:.4},\"act_max\":{:.4}",
                    s.mean_activation, s.max_activation
                ));
            }
            if s.mean_temperature.abs() > 1e-4 {
                line.push_str(&format!(",\"T\":{:.4}", s.mean_temperature));
            }
            line.push('}');
        }
        line.push_str("]}");

        self.write_line(&line);
    }

    /// Append one text picture from [`crate::diagnostics::scene_map`] as
    /// its own line, `{"frame":N,"map":"<name>","rows":[...]}`, top row
    /// first, so the log shows where things are as well as how much. Read
    /// one back with
    /// `jq -r 'select(.map=="<name>" and .frame==N) | .rows[]' run.ndjson`.
    pub fn log_map(&mut self, frame: u64, name: &str, rows: &[String]) {
        let escape = |s: &str| s.replace('\\', "\\\\").replace('"', "\\\"");
        let mut line = format!(
            "{{\"frame\":{frame},\"map\":\"{}\",\"rows\":[",
            escape(name)
        );
        for (i, row) in rows.iter().enumerate() {
            if i > 0 {
                line.push(',');
            }
            line.push('"');
            line.push_str(&escape(row));
            line.push('"');
        }
        line.push_str("]}");
        self.write_line(&line);
    }

    fn write_line(&mut self, line: &str) {
        let _ = writeln!(self.writer, "{}", line);
        self.calls_since_flush += 1;
        if self.calls_since_flush >= FLUSH_EVERY {
            let _ = self.writer.flush();
            self.calls_since_flush = 0;
        }
    }
}

impl Drop for FrameLogger {
    /// Real, final flush on drop -- `BufWriter` itself already does this on
    /// its own `Drop`, but doing it explicitly here (and swallowing any
    /// error the same way `log`'s own periodic flush already does) makes
    /// the "a normal exit never loses buffered data" guarantee this
    /// struct's own doc promises a real, direct property of `FrameLogger`
    /// itself, not just an inherited side effect of what it happens to wrap.
    fn drop(&mut self) {
        let _ = self.writer.flush();
    }
}

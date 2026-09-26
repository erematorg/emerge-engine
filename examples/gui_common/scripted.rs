//! A robot hand for a demo, so a run can be read from its log instead of
//! watched.
//!
//! Set `EMERGE_SCRIPT_LOG=<file>` and the demo runs a fixed script: the hand
//! presses where the demo says, for as long as the demo says, at the demo's
//! own strongest push, through the demo's own push code (the script only
//! holds the cursor and the button). Every physics step goes to a
//! `FrameLogger` line, with a text picture of the scene (`scene_map`) every
//! `EMERGE_SCRIPT_MAP_EVERY` steps (10 by default), and the demo quits when
//! the script ends. Frames are a fixed 1/60 s instead of the wall clock, so
//! a run does not depend on how fast the machine is.
//!
//! Also read from the environment: `EMERGE_SCRIPT_STEPS` (the length, in
//! physics steps) and `EMERGE_SCRIPT_GRAVITY` (a gravity fraction to start
//! at instead of the demo's default, 1 for real gravity).

use crate::emerge::diagnostics::{OCCUPANCY_BANDS, scene_map};
use crate::emerge::{FrameLogger, Simulation, per_material_stats};
use glam::Vec2;

/// One press of the hand: where, in grid cells, over which physics steps
/// (from inclusive, to exclusive), and whether it pulls instead of pushing.
pub struct Press {
    pub at: Vec2,
    pub from: u64,
    pub to: u64,
    pub pull: bool,
}

pub struct Script {
    presses: Vec<Press>,
    steps: u64,
    map_every: u64,
    region: (Vec2, Vec2),
    gravity: Option<f32>,
    log: Option<FrameLogger>,
}

fn env_number(name: &str) -> Option<f32> {
    std::env::var(name).ok().and_then(|v| v.parse().ok())
}

impl Script {
    /// `None` unless `EMERGE_SCRIPT_LOG` is set, so a normal run is
    /// untouched. `presses` and `steps` are the demo's own script;
    /// `region` is what its camera frames, for the text pictures.
    pub fn from_env(presses: Vec<Press>, steps: u64, region: (Vec2, Vec2)) -> Option<Self> {
        let path = std::env::var("EMERGE_SCRIPT_LOG").ok()?;
        let log = FrameLogger::open(&path).expect("failed to open EMERGE_SCRIPT_LOG");
        Some(Self {
            presses,
            steps: env_number("EMERGE_SCRIPT_STEPS").map_or(steps, |s| s as u64),
            map_every: env_number("EMERGE_SCRIPT_MAP_EVERY").map_or(10, |s| (s as u64).max(1)),
            region,
            gravity: env_number("EMERGE_SCRIPT_GRAVITY"),
            log: Some(log),
        })
    }

    /// The gravity fraction to start at: `EMERGE_SCRIPT_GRAVITY` if set,
    /// the demo's own default otherwise.
    pub fn gravity(&self, default: f32) -> f32 {
        self.gravity.unwrap_or(default)
    }

    /// Where the hand is at this physics step, and whether it pulls;
    /// `None` when it is not pressing.
    pub fn hand(&self, step: u64) -> Option<(Vec2, bool)> {
        self.presses
            .iter()
            .find(|p| (p.from..p.to).contains(&step))
            .map(|p| (p.at, p.pull))
    }

    /// The wall-clock time a scripted frame stands for: a fixed 1/60 s.
    pub fn frame_seconds(&self) -> f32 {
        1.0 / 60.0
    }

    /// Logs one physics step: the frame statistics with `extra` and whether
    /// the hand is pressing, and a text picture every `map_every` steps.
    /// Returns `true` once the script is over; the log is closed (and
    /// flushed) by then, so the demo can quit straight away.
    pub fn record(
        &mut self,
        step: u64,
        dt: f32,
        sim: &Simulation,
        labels: &[(u32, &str)],
        extra: &[(&str, f32)],
    ) -> bool {
        let pressing = self
            .hand(step)
            .map_or(0.0, |(_, pull)| if pull { -1.0 } else { 1.0 });
        if let Some(log) = &mut self.log {
            let mut fields = extra.to_vec();
            fields.push(("hand", pressing));
            log.log(
                step,
                dt,
                &per_material_stats(sim.particles()),
                &sim.diagnostics_snapshot(),
                labels,
                &fields,
            );
            if step.is_multiple_of(self.map_every) {
                let rows = scene_map(
                    sim.particles(),
                    self.region,
                    64,
                    32,
                    |_| 1.0,
                    &OCCUPANCY_BANDS,
                );
                log.log_map(step, "occupancy", &rows);
            }
        }
        if step + 1 >= self.steps {
            // Dropping the logger flushes it.
            self.log = None;
            return true;
        }
        false
    }
}

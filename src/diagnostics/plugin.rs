//! Plugin system for diagnostics — drop-in stat collectors.
//!
//! Implement [`DiagnosticsPlugin`] and register with [`DiagnosticsRegistry`].
//! Each plugin collects key/value pairs once per frame.
//! The registry aggregates all plugins into a [`DiagnosticsFrame`].
//!
//! # Quick start — closure plugin
//! ```rust,no_run
//! # use emerge::{DiagnosticsRegistry, DiagnosticsFrame};
//! let mut registry = DiagnosticsRegistry::new()
//!     .with_fn("alive", |particles, _snap| {
//!         vec![("alive_n".into(), particles.iter().filter(|p| p.activation > 0.0).count() as f32)]
//!     });
//! ```
//!
//! # Stateful plugin (rolling average)
//! ```rust,no_run
//! # use emerge::{DiagnosticsPlugin, DiagnosticsRegistry};
//! # use emerge::diagnostics::MpmSnapshot;
//! # use emerge::particle::Particle;
//! struct EnergyPlugin { history: Vec<f32> }
//! impl DiagnosticsPlugin for EnergyPlugin {
//!     fn name(&self) -> &'static str { "energy" }
//!     fn collect(&mut self, particles: &[Particle], _snap: &MpmSnapshot) -> Vec<(String, f32)> {
//!         let ke: f32 = particles.iter().map(|p| 0.5 * p.mass * p.v.length_squared()).sum();
//!         self.history.push(ke);
//!         vec![("ke".into(), ke)]
//!     }
//! }
//! ```

use std::fmt;

use crate::diagnostics::snapshot::MpmSnapshot;
use crate::particle::Particle;

// ─── Trait ──────────────────────────────────────────────────────────────────

/// A drop-in diagnostics plugin.
///
/// Collect is `&mut self` so plugins can maintain state (rolling histories,
/// accumulators, per-step diffs) without external wrappers.
///
/// Keys should be short `snake_case`. Collisions are allowed — last writer wins.
pub trait DiagnosticsPlugin: Send + Sync {
    /// Short identifier for this plugin. Used only for documentation.
    fn name(&self) -> &'static str;

    /// Collect stats for this frame. Return `(key, value)` pairs.
    ///
    /// Called once per [`DiagnosticsRegistry::collect`] invocation.
    /// `&mut self` enables stateful plugins (see module-level example).
    fn collect(&mut self, particles: &[Particle], snapshot: &MpmSnapshot) -> Vec<(String, f32)>;
}

// ─── Registry ───────────────────────────────────────────────────────────────

/// Registry of diagnostics plugins.
///
/// Build via chained [`with`](Self::with) / [`with_fn`](Self::with_fn) calls,
/// or add plugins at runtime via [`register`](Self::register).
///
/// ```rust,no_run
/// # use emerge::{DiagnosticsRegistry, ActivationStatsPlugin, ThermalStatsPlugin};
/// let mut registry = DiagnosticsRegistry::new()
///     .with(Box::new(ActivationStatsPlugin))
///     .with(Box::new(ThermalStatsPlugin))
///     .with_fn("custom", |particles, _snap| {
///         vec![("n".into(), particles.len() as f32)]
///     });
/// ```
#[derive(Default)]
pub struct DiagnosticsRegistry {
    plugins: Vec<Box<dyn DiagnosticsPlugin>>,
}

impl DiagnosticsRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a plugin. Chainable builder form.
    pub fn with(mut self, plugin: Box<dyn DiagnosticsPlugin>) -> Self {
        self.plugins.push(plugin);
        self
    }

    /// Register a closure as a plugin. Chainable builder form.
    ///
    /// `name` is for identification only; `f` is called each frame.
    pub fn with_fn(
        mut self,
        name: &'static str,
        f: impl Fn(&[Particle], &MpmSnapshot) -> Vec<(String, f32)> + Send + Sync + 'static,
    ) -> Self {
        self.plugins.push(Box::new(FnPlugin { name, f: Box::new(f) }));
        self
    }

    /// Register a plugin (mutation form, for post-construction use).
    pub fn register(&mut self, plugin: Box<dyn DiagnosticsPlugin>) {
        self.plugins.push(plugin);
    }

    /// Register a closure as a plugin (mutation form).
    pub fn register_fn(
        &mut self,
        name: &'static str,
        f: impl Fn(&[Particle], &MpmSnapshot) -> Vec<(String, f32)> + Send + Sync + 'static,
    ) {
        self.plugins.push(Box::new(FnPlugin { name, f: Box::new(f) }));
    }

    /// Collect one frame of diagnostics from all registered plugins.
    ///
    /// Plugins are called in registration order. Duplicate keys are preserved
    /// (last writer wins when using `get`).
    pub fn collect(&mut self, particles: &[Particle], snapshot: &MpmSnapshot) -> DiagnosticsFrame {
        let mut stats = Vec::new();
        for plugin in &mut self.plugins {
            stats.extend(plugin.collect(particles, snapshot));
        }
        DiagnosticsFrame { stats }
    }

    /// Number of registered plugins.
    pub fn len(&self) -> usize {
        self.plugins.len()
    }

    /// True if no plugins registered.
    pub fn is_empty(&self) -> bool {
        self.plugins.is_empty()
    }
}

// ─── Frame ──────────────────────────────────────────────────────────────────

/// One frame's worth of collected diagnostics from all plugins.
///
/// Supports lookup by key, iteration, formatted output, and merging.
#[derive(Debug, Clone, Default)]
pub struct DiagnosticsFrame {
    /// All key-value pairs, in collection order.
    pub stats: Vec<(String, f32)>,
}

impl DiagnosticsFrame {
    /// Get a stat by key. `None` if not present this frame.
    pub fn get(&self, key: &str) -> Option<f32> {
        // Last writer wins — iterate in reverse to find the most recent.
        self.stats.iter().rfind(|(k, _)| k == key).map(|(_, v)| *v)
    }

    /// Get a stat by key, or `default` if absent.
    pub fn get_or(&self, key: &str, default: f32) -> f32 {
        self.get(key).unwrap_or(default)
    }

    /// Iterate over all (key, value) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&str, f32)> {
        self.stats.iter().map(|(k, v)| (k.as_str(), *v))
    }

    /// Merge another frame into this one (appends — duplicate keys allowed).
    pub fn merge(&mut self, other: DiagnosticsFrame) {
        self.stats.extend(other.stats);
    }

    /// Format as a compact log line: `key=val key=val ...`
    ///
    /// Integers (exact float, |v| < 1e6) are printed without decimals.
    /// Other values use 4 decimal places.
    pub fn format_line(&self) -> String {
        self.stats
            .iter()
            .map(|(k, v)| {
                if v.fract() == 0.0 && v.abs() < 1e6 {
                    format!("{k}={}", *v as i64)
                } else {
                    format!("{k}={v:.4}")
                }
            })
            .collect::<Vec<_>>()
            .join(" ")
    }
}

impl fmt::Display for DiagnosticsFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.format_line())
    }
}

// ─── Built-in plugins ───────────────────────────────────────────────────────

/// Per-particle activation summary: `act_mean`, `act_frac`.
///
/// - `act_mean` — mean activation across all particles.
/// - `act_frac` — fraction of particles with activation > 0.01 (active).
pub struct ActivationStatsPlugin;

impl DiagnosticsPlugin for ActivationStatsPlugin {
    fn name(&self) -> &'static str { "activation" }

    fn collect(&mut self, particles: &[Particle], _snap: &MpmSnapshot) -> Vec<(String, f32)> {
        if particles.is_empty() {
            return vec![("act_mean".into(), 0.0), ("act_frac".into(), 0.0)];
        }
        let mut sum = 0.0f32;
        let mut active = 0usize;
        for p in particles {
            sum += p.activation;
            if p.activation > 0.01 { active += 1; }
        }
        let n = particles.len() as f32;
        vec![
            ("act_mean".into(), sum / n),
            ("act_frac".into(), active as f32 / n),
        ]
    }
}

/// Temperature / thermal energy summary: `T_mean`, `T_max`.
pub struct ThermalStatsPlugin;

impl DiagnosticsPlugin for ThermalStatsPlugin {
    fn name(&self) -> &'static str { "thermal" }

    fn collect(&mut self, particles: &[Particle], _snap: &MpmSnapshot) -> Vec<(String, f32)> {
        if particles.is_empty() {
            return vec![("T_mean".into(), 0.0), ("T_max".into(), 0.0)];
        }
        let mut sum = 0.0f32;
        let mut max = f32::NEG_INFINITY;
        for p in particles {
            sum += p.temperature;
            if p.temperature > max { max = p.temperature; }
        }
        vec![
            ("T_mean".into(), sum / particles.len() as f32),
            ("T_max".into(), max),
        ]
    }
}

/// Per-material particle counts: `mat{id}_n` for each material found.
pub struct MaterialCountPlugin;

impl DiagnosticsPlugin for MaterialCountPlugin {
    fn name(&self) -> &'static str { "mat_counts" }

    fn collect(&mut self, particles: &[Particle], _snap: &MpmSnapshot) -> Vec<(String, f32)> {
        let mut counts: std::collections::BTreeMap<u32, usize> = std::collections::BTreeMap::new();
        for p in particles {
            *counts.entry(p.material_id).or_default() += 1;
        }
        counts.into_iter().map(|(id, n)| (format!("mat{id}_n"), n as f32)).collect()
    }
}

/// Rolling N-frame exponential moving average wrapper.
///
/// Wraps any plugin and smooths its scalar outputs using:
///   `ema = α·new + (1-α)·ema`
///
/// where `α = 2 / (window + 1)` (standard EMA formula).
///
/// Output keys are prefixed with `ema_`.
///
/// ```rust,no_run
/// # use emerge::{DiagnosticsRegistry, ActivationStatsPlugin};
/// # use emerge::diagnostics::plugin::RollingPlugin;
/// let registry = DiagnosticsRegistry::new()
///     .with(Box::new(RollingPlugin::new(Box::new(ActivationStatsPlugin), 30)));
/// ```
pub struct RollingPlugin {
    inner: Box<dyn DiagnosticsPlugin>,
    alpha: f32,
    ema: std::collections::HashMap<String, f32>,
}

impl RollingPlugin {
    /// Create a rolling EMA wrapper with a given window size (frames).
    ///
    /// `window = 1` → no smoothing. `window = 60` → ~1s smooth at 60fps.
    pub fn new(inner: Box<dyn DiagnosticsPlugin>, window: usize) -> Self {
        let window = window.max(1) as f32;
        Self {
            inner,
            alpha: 2.0 / (window + 1.0),
            ema: std::collections::HashMap::new(),
        }
    }
}

impl DiagnosticsPlugin for RollingPlugin {
    fn name(&self) -> &'static str { self.inner.name() }

    fn collect(&mut self, particles: &[Particle], snapshot: &MpmSnapshot) -> Vec<(String, f32)> {
        let raw = self.inner.collect(particles, snapshot);
        let alpha = self.alpha;
        raw.iter()
            .map(|(k, v)| {
                let smoothed = self.ema
                    .entry(k.clone())
                    .and_modify(|ema| *ema = alpha * v + (1.0 - alpha) * *ema)
                    .or_insert(*v);
                (format!("ema_{k}"), *smoothed)
            })
            .collect()
    }
}

// ─── Internal: fn pointer plugin ────────────────────────────────────────────

struct FnPlugin {
    name: &'static str,
    f: Box<dyn Fn(&[Particle], &MpmSnapshot) -> Vec<(String, f32)> + Send + Sync>,
}

impl DiagnosticsPlugin for FnPlugin {
    fn name(&self) -> &'static str { self.name }
    fn collect(&mut self, particles: &[Particle], snapshot: &MpmSnapshot) -> Vec<(String, f32)> {
        (self.f)(particles, snapshot)
    }
}

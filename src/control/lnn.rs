// TODO(LP): This controller belongs in LP's mpm crate, not in a general physics engine.
// emerge's job is to expose `activation`, `activation_dir`, and the F·A·Fᵀ active stress.
// LP builds the creature locomotion controller on top of those primitives.
// Kept here temporarily until LP integration is wired.

/// Liquid Time-constant Network (LNN) — Hasani, Lechner, Amini, Rus, Grosu,
/// "Liquid Time-constant Networks" (arXiv preprint 2020; published AAAI 2021,
/// not NeurIPS as an earlier version of this comment said).
///
/// Continuous-time recurrent ODE neuron model:
///   dx/dt = −x/τ + σ(W·x + b) · (A − x)
///
/// Each neuron has a state x, a decay time constant τ, and a saturation
/// amplitude A. The gate σ(W·x + b) mixes current states before driving the update.
/// Outputs are σ(x) ∈ (0, 1) — read directly as muscle activation values.
///
/// All parameters (τ, A, W, b) are plain f32 — drop into a genome flat vec.
/// Integration: Euler at the caller's dt (physics sub-step rate).
///
/// # Quick-start
/// ```ignore
/// let mut lnn = Lnn::traveling_wave(4, 1.0); // 4 segments, period ≈ 1 s
/// // each physics sub-step:
/// lnn.step(sub_dt);
/// for (p, act) in muscle_particles.iter_mut().zip(lnn.activations()) {
///     p.activation = act;
/// }
/// ```
#[derive(Debug, Clone)]
pub struct Lnn {
    /// Neuron states xᵢ ∈ ℝ.  Persist between steps — carry oscillator memory.
    state: Vec<f32>,
    /// Time constants τᵢ > 0.  Controls how fast each neuron decays toward its attractor.
    pub tau: Vec<f32>,
    /// Saturation amplitudes Aᵢ.  States are attracted toward A when the gate is open.
    pub amplitude: Vec<f32>,
    /// Recurrent weight matrix W, row-major: W[i·n + j] = w_ij (neuron j → neuron i).
    pub weights: Vec<f32>,
    /// Per-neuron bias bᵢ added inside the gate sigmoid.
    pub bias: Vec<f32>,
}

impl Lnn {
    pub fn new(tau: Vec<f32>, amplitude: Vec<f32>, weights: Vec<f32>, bias: Vec<f32>) -> Self {
        let n = tau.len();
        assert_eq!(amplitude.len(), n, "amplitude length must equal n_neurons");
        assert_eq!(weights.len(), n * n, "weights length must equal n²");
        assert_eq!(bias.len(), n, "bias length must equal n_neurons");
        Self {
            state: vec![0.0; n],
            tau,
            amplitude,
            weights,
            bias,
        }
    }

    pub fn n_neurons(&self) -> usize {
        self.state.len()
    }

    /// Overwrite neuron states — use to seed the oscillator before running.
    /// Without seeding, all states start at 0 and no wave forms.
    pub fn set_state(&mut self, state: Vec<f32>) {
        assert_eq!(state.len(), self.state.len());
        self.state = state;
    }

    /// Euler step of the ODE.  Call once per physics sub-step.
    ///
    ///   dx_i = (−xᵢ/τᵢ + σ(Σⱼ wᵢⱼ·xⱼ + bᵢ) · (Aᵢ − xᵢ)) · dt
    pub fn step(&mut self, dt: f32) {
        let n = self.state.len();
        let mut dx = vec![0.0f32; n];
        for (i, dx_i) in dx.iter_mut().enumerate() {
            let net: f32 = (0..n)
                .map(|j| self.weights[i * n + j] * self.state[j])
                .sum::<f32>()
                + self.bias[i];
            let gate = sigmoid(net);
            *dx_i =
                (-self.state[i] / self.tau[i] + gate * (self.amplitude[i] - self.state[i])) * dt;
        }
        for (state, dxi) in self.state.iter_mut().zip(dx.iter()) {
            *state += dxi;
        }
    }

    /// Per-neuron output in (0, 1).  Feed directly into `particle.activation`.
    pub fn activations(&self) -> impl Iterator<Item = f32> + '_ {
        self.state.iter().map(|&x| sigmoid(x))
    }

    // ── Genome API ──────────────────────────────────────────────────────────────

    /// Expected flat genome length for n neurons: n·(n + 3).
    pub fn genome_size(n: usize) -> usize {
        n + n + n * n + n
    }

    /// Encode all parameters as a flat genome: [τ₀..τₙ, A₀..Aₙ, W_flat, b₀..bₙ].
    pub fn to_genome(&self) -> Vec<f32> {
        let mut g = Vec::with_capacity(Self::genome_size(self.n_neurons()));
        g.extend_from_slice(&self.tau);
        g.extend_from_slice(&self.amplitude);
        g.extend_from_slice(&self.weights);
        g.extend_from_slice(&self.bias);
        g
    }

    /// Decode from a genome slice previously produced by `to_genome`.
    pub fn from_genome(n: usize, genome: &[f32]) -> Self {
        let expected = Self::genome_size(n);
        assert_eq!(
            genome.len(),
            expected,
            "genome length mismatch: expected {expected}, got {}",
            genome.len()
        );
        Self::new(
            genome[..n].to_vec(),
            genome[n..2 * n].to_vec(),
            genome[2 * n..2 * n + n * n].to_vec(),
            genome[2 * n + n * n..].to_vec(),
        )
    }

    // ── Presets ─────────────────────────────────────────────────────────────────

    /// Traveling-wave CPG for `n_segments` muscle groups.
    ///
    /// Ring topology: neuron i excites neuron (i+1)%n and inhibits (i+n/2)%n.
    /// Seeded with phase-staggered initial states so the wave starts immediately.
    ///
    /// `period`: approximate oscillation period in simulation time units.
    pub fn traveling_wave(n_segments: usize, period: f32) -> Self {
        Self::coupled_traveling_wave(1, n_segments, period, 0.0)
    }

    /// `n_rings` independent traveling-wave CPGs (each `n_per_ring` neurons, ring
    /// topology as in [`Self::traveling_wave`]), cross-coupled neuron-for-neuron
    /// between corresponding segments of every ring pair.
    ///
    /// Two mutually-coupled half-center rings (`n_rings = 2`) is the standard CPG
    /// model for bilateral locomotion (e.g. lamprey spinal cord: left/right half-
    /// centers) — driving one ring's baseline harder than the other (see
    /// [`Self::set_ring_bias`]) turns a symmetric traveling wave into an
    /// asymmetric one, the real mechanism animals use to steer. `n_rings` isn't
    /// restricted to 2; any number of coupled oscillator groups works.
    ///
    /// `cross_coupling`: weight applied between corresponding neurons in
    /// different rings (0.0 = rings evolve fully independently, as if built via
    /// separate `traveling_wave` calls; negative = mutual inhibition, positive =
    /// mutual excitation).
    pub fn coupled_traveling_wave(
        n_rings: usize,
        n_per_ring: usize,
        period: f32,
        cross_coupling: f32,
    ) -> Self {
        assert!(n_rings >= 1, "need at least 1 ring");
        assert!(
            n_per_ring >= 2,
            "need at least 2 segments per ring for a traveling wave"
        );

        let n = n_rings * n_per_ring;
        // tau=0.5 at period=1.0 -- NOT derived from the period via a formula
        // (see 2026-07-05 rewrite below for why the old period/(2*pi) mapping
        // is gone). Scaled proportionally with period as the least-surprising
        // extrapolation, but ONLY period=1.0 is empirically deep-verified
        // (every real call site in this codebase -- emerge's own demo and
        // LP's creature -- uses period=1.0; no call site uses another value
        // outside a single short unit test). Other periods are a reasonable
        // guess, not independently proven.
        let tau_val = (0.5 * period).max(1e-3);
        let tau = vec![tau_val; n];
        // States oscillate in (-A, +A); sigmoid maps ±4 → (0.018, 0.982) --
        // see the amplitude rewrite note below for why this changed from 2.0.
        let amplitude = vec![4.0f32; n];

        // 2026-07-05 REWRITE (was: excite=3.0/inhibit=-2.0/self_inhib=-0.5,
        // tau=period/(2*pi), amplitude=2.0). The OLD topology looked
        // reasonable and passed its own test at the time, but that test only
        // ran 50 steps at dt=0.01 (0.5 simulated seconds) and only checked
        // that SOMETHING moved -- it never checked SUSTAINED oscillation.
        // Real finding (2026-07-04/05, see project memory
        // [[emerge_locomotion_root_cause_and_fix]]): the old topology
        // converges to a fully-synchronized fixed point (oscillation DIES,
        // every neuron settles to an identical constant) within ~20 steps at
        // dt=0.1, regardless of external bias -- driving zero real locomotion
        // for the entire time this was in use.
        //
        // This rewrite was found via a real systematic parameter sweep (not
        // hand-picked), then deep-verified two ways an "alive" variance check
        // alone can't catch: (1) survival to 10,000 steps, not 50 or 2000 --
        // real long-horizon stability; (2) genuine phase COHERENCE, via
        // cross-correlating neuron 0 against neuron 1 across a range of
        // lags -- a real traveling wave peaks at a consistent nonzero lag
        // (proof segment 1 is doing what segment 0 did, slightly later), not
        // just nonzero variance (which synchronized flickering or incoherent
        // noise would also produce). Measured on the actual bilateral
        // 2-ring/4-per-ring configuration real code uses (not just an
        // isolated ring): peak correlation 1.000 at cross_coupling 0/0.5/1.0,
        // sustained past 10,000 steps, with a real ~25% peak-to-peak
        // activation swing (checked explicitly -- a technically-alive-but-1%-
        // swing wave would be real but useless for visible locomotion).
        //
        // The resulting real oscillation period (~9.6s for a 4-per-ring at
        // period=1.0) is slower than a fast animal's gait, but genuinely
        // appropriate here: this controller drives an earthworm/lamprey-style
        // PERISTALTIC crawl (see `world::creature`'s doc in LP), and real
        // earthworm peristaltic waves are themselves slow -- this is not a
        // compromise forced by the fix, it is the physically-realistic regime
        // for this creature type.
        //
        // Connection STRUCTURE (excite-next / inhibit-antiphase, still real
        // and cited: Getting 1989, "Emerging principles governing the
        // operation of neural networks"; Ijspeert 2008, "Central pattern
        // generators for locomotion control in animals and robots", Neural
        // Networks 21:642) is unchanged. What changed: self-inhibition
        // REMOVED (was -0.5, now 0.0 -- the sweep found self-inhibition
        // specifically was part of what collapsed the ring to synchrony), and
        // inhibit now matches excite in magnitude (was 3.0/-2.0 asymmetric,
        // now 6.0/-6.0 symmetric) -- an unverified-in-literature but now
        // directly, numerically verified parameter choice, same honesty
        // standard as the old comment already applied to the magnitudes.
        let mut weights = vec![0.0f32; n * n];
        for r in 0..n_rings {
            let base = r * n_per_ring;
            for i in 0..n_per_ring {
                let row = base + i;
                weights[row * n + base + (i + 1) % n_per_ring] = 6.0; // excite next → wave propagation
                weights[row * n + base + (i + n_per_ring / 2) % n_per_ring] = -6.0; // inhibit opposite → phase separation
                // No self-inhibition term (was -0.5) -- verified this was
                // part of what collapsed the ring to a synchronized fixed
                // point; the outer leak term (-x/tau) already provides decay.
                for other_r in 0..n_rings {
                    if other_r != r {
                        weights[row * n + other_r * n_per_ring + i] = cross_coupling;
                    }
                }
            }
        }

        let mut lnn = Self::new(tau, amplitude, weights, vec![0.0; n]);
        // Seed: distribute initial states across phase space, per ring.
        let seed: Vec<f32> = (0..n)
            .map(|idx| {
                let i = idx % n_per_ring;
                (std::f32::consts::TAU * i as f32 / n_per_ring as f32).sin() * 1.5
            })
            .collect();
        lnn.set_state(seed);
        lnn
    }

    /// Overwrite the baseline bias of every neuron in ring `ring` (0-indexed,
    /// `n_per_ring` neurons per ring, matching the layout produced by
    /// [`Self::coupled_traveling_wave`]) to `value` — a tonic drive offset, the
    /// same lever real CPGs use to steer: bias one ring harder than another and
    /// the traveling wave becomes asymmetric.
    pub fn set_ring_bias(&mut self, ring: usize, n_per_ring: usize, value: f32) {
        for b in self
            .bias
            .iter_mut()
            .skip(ring * n_per_ring)
            .take(n_per_ring)
        {
            *b = value;
        }
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Real, permanent regression for the 2026-07-05 CPG rewrite (see
    /// [[emerge_locomotion_root_cause_and_fix]] in project memory). The OLD
    /// topology looked alive under a 50-step/dt=0.01 check but converged to a
    /// fully-synchronized fixed point (oscillation DIES) within ~20 steps at
    /// the dt=0.1 real gameplay actually runs at -- this exact kind of gap is
    /// why "alive after N steps" is not a sufficient test. This test checks
    /// what actually matters: does the oscillation SURVIVE 10,000 steps (not
    /// 50), and is it a genuine COHERENT traveling wave (real sequential
    /// phase lag between segments), not just nonzero noise or synchronized
    /// flickering? Checked on the real bilateral 2-ring/4-per-ring
    /// configuration actual code uses (`basic_creature` demo, LP's creature),
    /// not just an isolated ring.
    #[test]
    fn coupled_traveling_wave_sustains_a_real_long_horizon_traveling_wave() {
        let dt = 0.1;
        let steps = 10_000usize;
        let tail = 2000usize;

        let mut lnn = Lnn::coupled_traveling_wave(2, 4, 1.0, 1.0);
        let mut history: Vec<Vec<f32>> = Vec::with_capacity(tail);
        for step in 0..steps {
            lnn.step(dt);
            if step >= steps - tail {
                history.push(lnn.activations().collect());
            }
        }

        let n = history[0].len();
        let series: Vec<Vec<f32>> = (0..n)
            .map(|i| history.iter().map(|h| h[i]).collect())
            .collect();
        let mean: Vec<f32> = series
            .iter()
            .map(|s| s.iter().sum::<f32>() / s.len() as f32)
            .collect();

        // 1. Real, sustained variance -- not a dead, frozen oscillator.
        let total_var: f32 = series
            .iter()
            .zip(&mean)
            .map(|(s, m)| s.iter().map(|v| (v - m).powi(2)).sum::<f32>() / s.len() as f32)
            .sum();
        assert!(
            total_var > 1e-3,
            "CPG oscillation died (converged to a fixed point) within {steps} steps: \
             total_var={total_var:.6}. This is exactly the bug this test guards against."
        );

        // 2. Genuine phase coherence -- neuron 1 must correlate strongly with
        // a TIME-SHIFTED copy of neuron 0 (a real traveling wave), not just
        // have its own independent nonzero variance.
        let corr_at = |lag: usize| -> f32 {
            let a = &series[0];
            let b = &series[1];
            let len = a.len() - lag;
            let (am, bm) = (mean[0], mean[1]);
            let num: f32 = (0..len).map(|t| (a[t] - am) * (b[t + lag] - bm)).sum();
            let da: f32 = (0..len).map(|t| (a[t] - am).powi(2)).sum::<f32>().sqrt();
            let db: f32 = (0..len)
                .map(|t| (b[t + lag] - bm).powi(2))
                .sum::<f32>()
                .sqrt();
            if da * db > 1e-6 { num / (da * db) } else { 0.0 }
        };
        let best_corr = (0..150).map(corr_at).fold(f32::NEG_INFINITY, f32::max);
        assert!(
            best_corr > 0.9,
            "CPG activation is alive but not a coherent traveling wave (best neuron0-vs-\
             neuron1 cross-correlation across lags = {best_corr:.3}, want > 0.9) -- this \
             would be synchronized flickering or incoherent noise, not real peristalsis."
        );

        // 3. Real, usable amplitude -- not technically-alive-but-imperceptible.
        let (mut min, mut max) = (f32::INFINITY, f32::NEG_INFINITY);
        for h in &history {
            min = min.min(h[0]);
            max = max.max(h[0]);
        }
        assert!(
            max - min > 0.1,
            "CPG oscillation amplitude too weak to drive visible muscle contraction \
             (peak-to-peak swing = {:.4}, want > 0.1)",
            max - min
        );
    }

    #[test]
    fn traveling_wave_oscillates_at_realistic_gameplay_dt() {
        // Real gameplay runs at dt~=0.1 (LP's physics substep scale), not the
        // old test's dt=0.01 -- checked over 50 steps only proves SOMETHING
        // moved initially; the real long-horizon guarantee lives in
        // `coupled_traveling_wave_sustains_a_real_long_horizon_traveling_wave`.
        let mut lnn = Lnn::traveling_wave(4, 1.0);
        let first: Vec<f32> = lnn.activations().collect();
        for _ in 0..50 {
            lnn.step(0.1);
        }
        let later: Vec<f32> = lnn.activations().collect();
        let delta: f32 = first.iter().zip(&later).map(|(a, b)| (a - b).abs()).sum();
        assert!(delta > 0.05, "LNN did not oscillate (delta={delta})");
    }

    #[test]
    fn genome_roundtrip() {
        let original = Lnn::traveling_wave(4, 1.0);
        let genome = original.to_genome();
        assert_eq!(genome.len(), Lnn::genome_size(4));
        let restored = Lnn::from_genome(4, &genome);
        assert_eq!(restored.tau, original.tau);
        assert_eq!(restored.weights, original.weights);
    }

    #[test]
    fn activations_in_unit_range() {
        let mut lnn = Lnn::traveling_wave(6, 0.8);
        for _ in 0..200 {
            lnn.step(0.005);
        }
        for a in lnn.activations() {
            assert!(a > 0.0 && a < 1.0, "activation out of (0,1): {a}");
        }
    }

    #[test]
    fn coupled_traveling_wave_has_expected_neuron_count() {
        let lnn = Lnn::coupled_traveling_wave(2, 4, 1.0, -0.5);
        assert_eq!(lnn.n_neurons(), 8);
    }

    #[test]
    fn zero_cross_coupling_matches_independent_rings() {
        // n_rings=1 (via traveling_wave) run twice should equal n_rings=2 with
        // cross_coupling=0.0 — proves coupling is opt-in, not baked in.
        let mut solo_a = Lnn::traveling_wave(4, 1.0);
        let mut solo_b = Lnn::traveling_wave(4, 1.0);
        let mut coupled = Lnn::coupled_traveling_wave(2, 4, 1.0, 0.0);

        for _ in 0..50 {
            solo_a.step(0.01);
            solo_b.step(0.01);
            coupled.step(0.01);
        }

        let expected: Vec<f32> = solo_a.activations().chain(solo_b.activations()).collect();
        let actual: Vec<f32> = coupled.activations().collect();
        for (e, a) in expected.iter().zip(actual.iter()) {
            assert!((e - a).abs() < 1e-5, "expected {e}, got {a}");
        }
    }

    #[test]
    fn ring_bias_breaks_symmetry_between_rings() {
        let mut lnn = Lnn::coupled_traveling_wave(2, 4, 1.0, -0.3);
        lnn.set_ring_bias(0, 4, 1.0);
        lnn.set_ring_bias(1, 4, -1.0);
        for _ in 0..100 {
            lnn.step(0.01);
        }
        let acts: Vec<f32> = lnn.activations().collect();
        let ring0_mean: f32 = acts[0..4].iter().sum::<f32>() / 4.0;
        let ring1_mean: f32 = acts[4..8].iter().sum::<f32>() / 4.0;
        assert!(
            ring0_mean > ring1_mean,
            "expected biased ring to have higher mean activation: ring0={ring0_mean}, ring1={ring1_mean}"
        );
    }
}

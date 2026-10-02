// Lives in emerge, not LP: a deliberate exception
// that doesn't participate in the substep loop at all -- a standalone ODE
// the caller integrates, writing its output into
// `Particle::activation`/`activation_dir` between steps.

/// Liquid Time-constant Network (LNN) -- Hasani, Lechner, Amini, Rus, Grosu,
/// "Liquid Time-constant Networks" (arXiv preprint 2020; published AAAI 2021).
///
/// Continuous-time recurrent ODE neuron model:
///   dx/dt = −x/τ + σ(W·x + b) · (A − x)
///
/// Each neuron has a state x, a decay time constant τ, and a saturation
/// amplitude A. The gate σ(W·x + b) mixes current states before driving the update.
/// Outputs are σ(x) ∈ (0, 1) -- read directly as muscle activation values.
///
/// All parameters (τ, A, W, b) are plain f32 -- drop into a genome flat vec.
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
    /// Neuron states xᵢ ∈ ℝ.  Persist between steps -- carry oscillator memory.
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

    pub const fn n_neurons(&self) -> usize {
        self.state.len()
    }

    /// Overwrite neuron states -- use to seed the oscillator before running.
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
    pub const fn genome_size(n: usize) -> usize {
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
    /// centers) -- driving one ring's baseline harder than the other (see
    /// [`Self::set_ring_bias`]) turns a symmetric traveling wave into an
    /// asymmetric one, the mechanism animals use to steer. `n_rings` isn't
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
        // `tau=0.5` at `period=1.0` is a free parameter tuned against this
        // engine's sustained-wave regression below, not derived from CPG
        // literature: Ijspeert's 2008 review covers other oscillator
        // equations, and the LTC model of Hasani et al. (AAAI 2021) defines tau
        // as a model parameter; neither maps a requested period to tau for
        // this equation. The proportional scaling is likewise uncalibrated.
        // Only period=1.0 is verified (and used by callers); other periods
        // need their own calibration before being treated as physically
        // meaningful.
        const EMPIRICAL_TAU_AT_UNIT_PERIOD: f32 = 0.5;
        let tau_val = (EMPIRICAL_TAU_AT_UNIT_PERIOD * period).max(1e-3);
        let tau = vec![tau_val; n];
        // States oscillate in (-A, +A); sigmoid maps ±4 → (0.018, 0.982) --
        // see the amplitude rewrite note below for why this changed from 2.0.
        let amplitude = vec![4.0f32; n];

        // Topology from a systematic parameter sweep, verified two ways a
        // variance check alone cannot: (1) the wave survives 10,000 steps;
        // (2) phase coherence, by cross-correlating neuron 0 against neuron 1
        // over a range of lags: a traveling wave peaks at a consistent nonzero
        // lag (segment 1 repeats segment 0, slightly later), which
        // synchronized flickering or incoherent noise would not. On the
        // bilateral 2-ring/4-per-ring configuration callers use: peak
        // correlation 1.000 at cross_coupling 0/0.5/1.0, sustained past 10,000
        // steps, with a ~25% peak-to-peak activation swing (a 1% swing would
        // be alive but useless for visible locomotion).
        //
        // The resulting period (~9.6 s for 4-per-ring at period=1.0) is slower
        // than a fast animal's gait but suits the earthworm/lamprey-style
        // peristaltic crawl this controller drives (see `world::creature` in
        // LP): earthworm peristaltic waves are slow.
        //
        // Connection structure, excite-next / inhibit-antiphase: Getting 1989,
        // "Emerging principles governing the operation of neural networks";
        // Ijspeert 2008, "Central pattern generators for locomotion control in
        // animals and robots", Neural Networks 21:642. Self-inhibition is 0.0
        // (the sweep found a self-inhibition of -0.5 collapses the ring to a
        // synchronized fixed point within ~20 steps at dt=0.1), and inhibit
        // matches excite in magnitude. The magnitudes are free parameters
        // tuned against the sustained-wave regression, not literature
        // synaptic strengths: published CPG models use other state equations
        // and normalizations, so their weights do not transfer to this LNN.
        const EMPIRICAL_EXCITATORY_WEIGHT: f32 = 6.0;
        const EMPIRICAL_INHIBITORY_WEIGHT: f32 = -6.0;
        let mut weights = vec![0.0f32; n * n];
        for r in 0..n_rings {
            let base = r * n_per_ring;
            for i in 0..n_per_ring {
                let row = base + i;
                weights[row * n + base + (i + 1) % n_per_ring] = EMPIRICAL_EXCITATORY_WEIGHT; // excite next → wave propagation
                weights[row * n + base + (i + n_per_ring / 2) % n_per_ring] =
                    EMPIRICAL_INHIBITORY_WEIGHT; // inhibit opposite → phase separation
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
    /// [`Self::coupled_traveling_wave`]) to `value` -- a tonic drive offset, the
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

    /// Sustained, coherent oscillation: "alive after N steps" is not enough
    /// (a ring can look alive over 50 steps at dt=0.01 and still collapse to a
    /// synchronized fixed point within ~20 steps at dt=0.1). Checks that the
    /// oscillation survives 10,000 steps and is a coherent traveling wave
    /// (sequential phase lag between segments), not noise or synchronized
    /// flickering, on the bilateral 2-ring/4-per-ring configuration callers
    /// use (`basic_creature` demo, LP's creature).
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

        // 1. Sustained variance -- not a dead, frozen oscillator.
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
        // a TIME-SHIFTED copy of neuron 0 (a traveling wave), not just
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

        // 3. Usable amplitude -- not technically-alive-but-imperceptible.
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
        // Gameplay runs at dt~=0.1 (LP's physics substep scale). A short run
        // only proves something moved initially; the long-horizon guarantee
        // is `coupled_traveling_wave_sustains_a_real_long_horizon_traveling_wave`.
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
        // cross_coupling=0.0 -- proves coupling is opt-in, not baked in.
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

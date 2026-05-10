/// Liquid Time-constant Network (LNN) — Hasani et al. 2020 (NeurIPS).
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
        Self { state: vec![0.0; n], tau, amplitude, weights, bias }
    }

    pub fn n_neurons(&self) -> usize { self.state.len() }

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
        for i in 0..n {
            let net: f32 = (0..n)
                .map(|j| self.weights[i * n + j] * self.state[j])
                .sum::<f32>()
                + self.bias[i];
            let gate = sigmoid(net);
            dx[i] = (-self.state[i] / self.tau[i]
                + gate * (self.amplitude[i] - self.state[i]))
                * dt;
        }
        for i in 0..n {
            self.state[i] += dx[i];
        }
    }

    /// Per-neuron output in (0, 1).  Feed directly into `particle.activation`.
    pub fn activations(&self) -> impl Iterator<Item = f32> + '_ {
        self.state.iter().map(|&x| sigmoid(x))
    }

    // ── Genome API ──────────────────────────────────────────────────────────────

    /// Expected flat genome length for n neurons: n·(n + 3).
    pub fn genome_size(n: usize) -> usize { n + n + n * n + n }

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
            genome.len(), expected,
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
        assert!(n_segments >= 2, "need at least 2 segments for a traveling wave");

        let tau_val = (period / std::f32::consts::TAU).max(1e-3);
        let tau = vec![tau_val; n_segments];
        // States oscillate in (-A, +A); sigmoid maps ±2 → (0.12, 0.88).
        let amplitude = vec![2.0f32; n_segments];

        let n = n_segments;
        let mut weights = vec![0.0f32; n * n];
        for i in 0..n {
            weights[i * n + (i + 1) % n]       =  3.0; // excite next → wave propagation
            weights[i * n + (i + n / 2) % n]   = -2.0; // inhibit opposite → phase separation
            weights[i * n + i]                  = -0.5; // weak self-inhibition → no saturation
        }

        let mut lnn = Self::new(tau, amplitude, weights, vec![0.0; n]);
        // Seed: distribute initial states across phase space.
        let seed: Vec<f32> = (0..n)
            .map(|i| (std::f32::consts::TAU * i as f32 / n as f32).sin() * 1.5)
            .collect();
        lnn.set_state(seed);
        lnn
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn traveling_wave_oscillates() {
        let mut lnn = Lnn::traveling_wave(4, 1.0);
        let first: Vec<f32> = lnn.activations().collect();
        for _ in 0..50 { lnn.step(0.01); }
        let later: Vec<f32> = lnn.activations().collect();
        let delta: f32 = first.iter().zip(&later).map(|(a, b)| (a - b).abs()).sum();
        assert!(delta > 0.1, "LNN did not oscillate (delta={delta})");
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
        for _ in 0..200 { lnn.step(0.005); }
        for a in lnn.activations() {
            assert!(a > 0.0 && a < 1.0, "activation out of (0,1): {a}");
        }
    }
}

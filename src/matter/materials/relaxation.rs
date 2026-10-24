//! A solid's measured relaxation: the generalised Zener spectrum that
//! makes its stress lag its strain and turns part of every vibration into
//! heat, as a material record (data), attached to the solid's own elastic
//! law.
//!
//! The law (Blanc, Komatitsch, Chaljub, Lombard and Xie, GJI 205 (2016)
//! 427-439, eqs. 10 and 21-23, read): `N` relaxation mechanisms of
//! frequencies `theta_l > 0` and weights `kappa_l > 0`, the modulus `M(omega)
//! = M_r (1 + sum_l i omega kappa_l / (theta_l + i omega))` about the relaxed
//! modulus `M_r`. Their Property 1: with every weight and frequency positive
//! the total energy can only decrease. In two dimensions (Lombard and
//! Piraux, arXiv 1012.2669, eqs. 9-11, read) the P-wave modulus `pi = lambda
//! + 2 mu` and the shear modulus `mu` relax through the same frequencies with
//! weights of their own, `kappa^p_l` and `kappa^s_l`. The elastic law's
//! moduli are the relaxed ones: a solid with a spectrum is stiffer at high
//! frequency, `M_u = M_r (1 + sum kappa)` (Blanc et al. eq. 8).
//!
//! Each mechanism acts on each part of the elastic law's stored energy as a
//! Maxwell branch (Liu and Guan, arXiv 2305.15812, eqs. 2.2-2.8, their MIPC
//! branch with strain-energy factor 1, read): an internal strain `Gamma`,
//! its conjugate `Q = S_part(C) - mu_b (Gamma - I)`, the branch energy
//! `|Q|^2 / (4 mu_b)` and the flow `eta_b Gamma' = Q`. A part of modulus `G`
//! (the shear part `mu`, the volume part `lambda`) relaxing with weight
//! `kappa` and frequency `theta` takes `mu_b = G / kappa` and `eta_b = G /
//! (kappa theta)`. Adaptation, declared and derived on the page: in the
//! linear limit a branch's stress `sigma_b` then obeys `sigma_b' = kappa G
//! eps' - theta sigma_b`, Blanc et al.'s memory variable, so the mapping is
//! exact there; beyond it the finite-strain branch is Liu and Guan's
//! (their Part I), unverified here past the record's strain limit.
//!
//! The part weights, from Lombard and Piraux's linear law `sigma = (psi_pi -
//! 2 psi_mu) * tr(eps') I + 2 psi_mu * eps'` (derived on the page): the
//! shear part relaxes with `kappa^s`, the volume part `lambda` with `(pi
//! kappa^p - 2 mu kappa^s) / lambda`. A record with `kappa^s = kappa^p`
//! relaxes both parts alike (proportional damping).

use std::fmt;

use crate::matter::materials::physical_props::Elastic;

/// The most mechanisms a record holds: a bound on each particle's state,
/// not a physical number. Three already match Plasticine's loss factor
/// within 0.83 % over its measured band (`plasticine_crandall_1970`).
pub const MAX_MECHANISMS: usize = 5;

/// One relaxation mechanism: its frequency (rad/s) and its weights on the
/// P-wave and shear moduli.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mechanism {
    pub theta: f64,
    pub kappa_shear: f64,
    pub kappa_p: f64,
}

/// A solid's relaxation spectrum: its mechanisms, and the largest strain
/// the measurement it comes from holds linear (the record's own validity
/// limit, with its source).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RelaxationSpectrum {
    mechanisms: [Mechanism; MAX_MECHANISMS],
    count: usize,
    /// The largest strain at which the measured spectrum is linear: beyond
    /// it the record is applied outside its data, which the step reports
    /// (`StepReport::relaxation_beyond`), never silently.
    pub strain_limit: f64,
}

/// Why a spectrum or its part weights were refused.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RelaxationError {
    /// No mechanism, or more than `MAX_MECHANISMS`.
    Count(usize),
    /// A frequency or a weight not strictly positive (Blanc et al.
    /// Property 1 needs both).
    NotPositive(usize),
    /// A part's weight not positive for these moduli: the P-wave weight is
    /// too small against the shear one for the volume part to relax (or the
    /// law has no volume part to carry a P-wave relaxation that differs
    /// from the shear's).
    PartWeight(usize),
}

impl fmt::Display for RelaxationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Count(n) => write!(f, "{n} relaxation mechanisms (1 to {MAX_MECHANISMS})"),
            Self::NotPositive(l) => write!(f, "mechanism {l}: a frequency or weight not positive"),
            Self::PartWeight(l) => write!(
                f,
                "mechanism {l}: its P-wave and shear weights give the volume part a weight not positive"
            ),
        }
    }
}

impl std::error::Error for RelaxationError {}

impl RelaxationSpectrum {
    /// The spectrum of these mechanisms, linear up to `strain_limit`.
    pub fn new(mechanisms: &[Mechanism], strain_limit: f64) -> Result<Self, RelaxationError> {
        if mechanisms.is_empty() || mechanisms.len() > MAX_MECHANISMS {
            return Err(RelaxationError::Count(mechanisms.len()));
        }
        for (l, m) in mechanisms.iter().enumerate() {
            if !(m.theta > 0.0 && m.kappa_shear > 0.0 && m.kappa_p > 0.0) {
                return Err(RelaxationError::NotPositive(l));
            }
        }
        let empty = Mechanism {
            theta: 0.0,
            kappa_shear: 0.0,
            kappa_p: 0.0,
        };
        let mut all = [empty; MAX_MECHANISMS];
        all[..mechanisms.len()].copy_from_slice(mechanisms);
        Ok(Self {
            mechanisms: all,
            count: mechanisms.len(),
            strain_limit,
        })
    }

    pub fn mechanisms(&self) -> &[Mechanism] {
        &self.mechanisms[..self.count]
    }

    /// Each mechanism's weights on the corotated law's two parts (shear,
    /// volume) for the relaxed Lame moduli `(lambda, mu)`. A part whose
    /// modulus is zero takes no weight (it stores nothing); a positive part
    /// whose weight would not be positive refuses the spectrum.
    pub fn part_weights(
        &self,
        lambda: f64,
        mu: f64,
    ) -> Result<[(f64, f64); MAX_MECHANISMS], RelaxationError> {
        let mut weights = [(0.0, 0.0); MAX_MECHANISMS];
        let pi = lambda + 2.0 * mu;
        for (l, m) in self.mechanisms().iter().enumerate() {
            let shear = if mu > 0.0 { m.kappa_shear } else { 0.0 };
            // The volume part's relaxation, `pi kappa^p - 2 mu kappa^s`, is
            // carried by `lambda`: with no volume part it must vanish.
            let carried = pi * m.kappa_p - 2.0 * mu * m.kappa_shear;
            let volume = if lambda > 0.0 {
                carried / lambda
            } else if carried.abs() <= 1e-12 * pi * m.kappa_p {
                0.0
            } else {
                return Err(RelaxationError::PartWeight(l));
            };
            if lambda > 0.0 && volume <= 0.0 {
                return Err(RelaxationError::PartWeight(l));
            }
            weights[l] = (shear, volume);
        }
        Ok(weights)
    }

    /// Harbutt's Plasticine at 75 F (Crandall, Kurzweil, Nigam and
    /// Remington, *Dynamic properties of modelling clay*, NASA CR-66953 /
    /// MIT 76205-3, 1970).
    ///
    /// - Their measured loss factor, the same for shear and dilatation in
    ///   their model (their Eqs. 1-2): 0.225 in shear (Sec. 5), 0.222 in
    ///   rods, 0.234 in the sandwich tests, over about 100-3000 Hz. The
    ///   weights below are equal for shear and P waves accordingly.
    /// - Three mechanisms fitted to a constant loss factor 0.225 over
    ///   100-3000 Hz by Blanc et al.'s eq. 28 (minimax, positive weights,
    ///   their eq. 33): the largest error in the loss factor over the band is
    ///   0.83 %, below the measurements' own repeatability (2-6 % within a
    ///   specimen).
    /// - Linear below a strain of 1e-5: their Sec. 10 finds the threshold of
    ///   nonlinearity near an rms longitudinal strain of 1e-5 (rods) and an
    ///   rms shear strain of 3e-5, the smaller kept.
    ///
    /// Declared gaps, each with its removal criterion:
    /// - below 100 Hz and above 3 kHz the spectrum is extrapolated: a loss
    ///   factor measured there;
    /// - beyond the strain limit it is applied outside its data (reported
    ///   per step): an amplitude-dependent modulus and loss factor of the same
    ///   material, read;
    /// - the relaxed moduli are not part of this spectrum: a scene using it
    ///   sets them from Crandall et al.'s measured wave speed at a reference
    ///   frequency (Blanc et al. eq. 25) first.
    pub fn plasticine_crandall_1970() -> Self {
        let mechanism = |theta: f64, kappa: f64| Mechanism {
            theta,
            kappa_shear: kappa,
            kappa_p: kappa,
        };
        Self::new(
            &[
                mechanism(654.876_730_621_607_9, 0.380_744_010_597_488_7),
                mechanism(3_865.458_262_006_148, 0.361_264_618_173_421_7),
                mechanism(25_438.050_835_431_713, 0.753_998_274_291_466_7),
            ],
            1e-5,
        )
        .expect("a positive spectrum")
    }
}

impl RelaxationSpectrum {
    /// The phase speed over the relaxed one at angular frequency `omega`,
    /// shear weights (Blanc et al. eqs. 24-26: `c = sqrt(M_r / rho) F`, `F =
    /// 1 / Re[(M / M_r)^-1/2]`). With `M / M_r = a + i b`, `Re(m^-1/2) =
    /// Re(sqrt m) / |m|` and `Re(sqrt m) = sqrt((|m| + a) / 2)` (the
    /// principal root, `a > 0`), so `F = |m| / sqrt((|m| + a) / 2)` (derived
    /// on the page).
    pub fn phase_speed_factor(&self, omega: f64) -> f64 {
        let (a, b) = self.mechanisms().iter().fold((1.0, 0.0), |(a, b), m| {
            let d = m.theta * m.theta + omega * omega;
            (
                a + m.kappa_shear * omega * omega / d,
                b + m.kappa_shear * omega * m.theta / d,
            )
        });
        let size = a.hypot(b);
        size / ((size + a) / 2.0).sqrt()
    }
}

/// A measured speed at its frequency.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpeedAt {
    pub frequency_hz: f64,
    pub speed_m_s: f64,
}

/// A viscoelastic solid as measured in one state: its relaxed moduli (in
/// SI), its relaxation spectrum, and the measurements they come from. A
/// scene using a record uses all of it.
#[derive(Clone, Copy, Debug)]
pub struct ViscoelasticRecord {
    /// The relaxed Young's modulus, Poisson's ratio and density.
    pub relaxed: Elastic,
    pub spectrum: RelaxationSpectrum,
    /// The specimen's state as its source names it, and its age, hours.
    pub state: &'static str,
    pub age_hours: f64,
    /// The shear speed the relaxed modulus is set from (Blanc et al. eq.
    /// 25), and a second measurement of the same state, the check.
    pub reference: SpeedAt,
    pub check: SpeedAt,
}

impl ViscoelasticRecord {
    /// The record whose relaxed shear modulus makes `spectrum` reproduce
    /// the shear speed `reference`: `mu_r = rho c^2 / F^2` (Blanc et al.
    /// eq. 25 from eq. 26, `phase_speed_factor`), `E_r = 2 mu_r (1 + nu)`
    /// (isotropic, every modulus relaxing alike: the spectrum must be
    /// proportional).
    fn from_shear_speed(
        spectrum: RelaxationSpectrum,
        (rho, nu): (f64, f64),
        (reference, check): (SpeedAt, SpeedAt),
        (state, age_hours): (&'static str, f64),
    ) -> Self {
        assert!(
            spectrum
                .mechanisms()
                .iter()
                .all(|m| m.kappa_shear == m.kappa_p),
            "a record set from a shear speed with one Poisson's ratio needs a proportional spectrum"
        );
        let omega = 2.0 * std::f64::consts::PI * reference.frequency_hz;
        let mu = rho * (reference.speed_m_s / spectrum.phase_speed_factor(omega)).powi(2);
        Self {
            relaxed: Elastic {
                e_pa: (2.0 * mu * (1.0 + nu)) as f32,
                nu: nu as f32,
                rho_kg_m3: rho as f32,
            },
            spectrum,
            state,
            age_hours,
            reference,
            check,
        }
    }

    /// The record's shear phase speed at `frequency_hz`, m/s.
    pub fn shear_speed(&self, frequency_hz: f64) -> f64 {
        let (e, nu, rho) = (
            f64::from(self.relaxed.e_pa),
            f64::from(self.relaxed.nu),
            f64::from(self.relaxed.rho_kg_m3),
        );
        let mu = e / (2.0 * (1.0 + nu));
        (mu / rho).sqrt()
            * self
                .spectrum
                .phase_speed_factor(2.0 * std::f64::consts::PI * frequency_hz)
    }

    /// Crandall et al.'s Plasticine (NASA CR-66953, 1970) as measured on one
    /// specimen just after forming: the first shear antiresonance, 258 ft/s
    /// at 738 Hz, the reference; the first resonance, 266 ft/s at 943 Hz, the
    /// check (their sec. 12, the start of the 120-hour series). Their density
    /// "119 lbs/cu. ft." (p. 27), their Poisson's ratio 0.434, frequency
    /// independent over 700-2800 Hz (p. 29), and their loss factor as
    /// `RelaxationSpectrum::plasticine_crandall_1970`. Relaxed `mu_r` about
    /// 7.05 MPa, `E_r` 20.2 MPa; the check differs from the record's own
    /// speed there by 2.7 % (the fit's dispersion against the measured one).
    pub fn plasticine_crandall_1970_fresh() -> Self {
        Self::plasticine(
            (feet(258.0), 738.0),
            (feet(266.0), 943.0),
            (
                "freshly formed (the start of Crandall et al.'s 120-hour series)",
                0.0,
            ),
        )
    }

    /// The same specimen 120 hours after forming (the end of the series):
    /// the antiresonance 319 ft/s at 996 Hz, the reference; the resonance
    /// 327 ft/s at 1220 Hz, the check. Relaxed `mu_r` about 10.3 MPa, `E_r`
    /// 29.6 MPa, 45 % above the fresh state's: thixotropic stiffening with
    /// time after forming (their Secs. 11-12), a different state of the
    /// material, not scatter. The check differs by 2.2 %.
    pub fn plasticine_crandall_1970_aged_120_h() -> Self {
        Self::plasticine(
            (feet(319.0), 996.0),
            (feet(327.0), 1220.0),
            (
                "aged 120 h (the end of Crandall et al.'s 120-hour series)",
                120.0,
            ),
        )
    }

    fn plasticine(
        (reference_speed, reference_hz): (f64, f64),
        (check_speed, check_hz): (f64, f64),
        state: (&'static str, f64),
    ) -> Self {
        // 119 lb/ft^3: the pound and the foot exact by definition.
        let rho = 119.0 * 0.453_592_37 / 0.3048f64.powi(3);
        Self::from_shear_speed(
            RelaxationSpectrum::plasticine_crandall_1970(),
            (rho, 0.434),
            (
                SpeedAt {
                    frequency_hz: reference_hz,
                    speed_m_s: reference_speed,
                },
                SpeedAt {
                    frequency_hz: check_hz,
                    speed_m_s: check_speed,
                },
            ),
            state,
        )
    }
}

/// Feet to metres, the foot exact by definition.
fn feet(ft: f64) -> f64 {
    ft * 0.3048
}

/// A Maxwell branch on one part of the elastic energy: its spring modulus
/// `mu_b` and viscosity `eta_b`, for a part of modulus `G` relaxing with
/// weight `kappa` at frequency `theta`: `mu_b = G / kappa`, `eta_b = G /
/// (kappa theta)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Branch {
    pub modulus: f64,
    pub viscosity: f64,
}

impl Branch {
    pub fn of_part(part_modulus: f64, kappa: f64, theta: f64) -> Self {
        Self {
            modulus: part_modulus / kappa,
            viscosity: part_modulus / (kappa * theta),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each Plasticine record reproduces its reference speed (to the f32 its
    /// relaxed modulus is stored in), its check within the 2-3 % its fit's
    /// dispersion leaves, and its relaxed moduli are those derived on the
    /// page (`tmp` derivation: `mu_r` 7.052 and 10.335 MPa).
    #[test]
    fn the_plasticine_records_reproduce_their_measured_shear_speeds() {
        for (record, mu_derived) in [
            (
                ViscoelasticRecord::plasticine_crandall_1970_fresh(),
                7.052e6,
            ),
            (
                ViscoelasticRecord::plasticine_crandall_1970_aged_120_h(),
                10.335e6,
            ),
        ] {
            let at = record.shear_speed(record.reference.frequency_hz);
            assert!(
                (at / record.reference.speed_m_s - 1.0).abs() < 1e-6,
                "{}: {at} m/s against {}",
                record.state,
                record.reference.speed_m_s
            );
            let check = record.shear_speed(record.check.frequency_hz) / record.check.speed_m_s;
            assert!(
                (check - 1.0).abs() < 0.03,
                "{}: check {check}",
                record.state
            );
            let mu = f64::from(record.relaxed.e_pa) / (2.0 * (1.0 + f64::from(record.relaxed.nu)));
            assert!(
                (mu / mu_derived - 1.0).abs() < 1e-3,
                "{}: mu_r {mu}",
                record.state
            );
            assert!((f64::from(record.relaxed.rho_kg_m3) - 1906.2).abs() < 0.1);
        }
    }

    #[test]
    fn a_proportional_record_relaxes_both_parts_alike() {
        let record = RelaxationSpectrum::plasticine_crandall_1970();
        let weights = record.part_weights(2.0e6, 1.3e6).expect("positive");
        for (m, &(shear, volume)) in record.mechanisms().iter().zip(&weights) {
            assert!((shear - m.kappa_shear).abs() <= 1e-15 * shear);
            assert!((volume - m.kappa_p).abs() <= 1e-14 * volume);
        }
    }

    #[test]
    fn unequal_weights_pass_the_p_wave_relaxation_to_the_volume_part() {
        // Test data, not a material: the P weight twice the shear's.
        let spectrum = RelaxationSpectrum::new(
            &[Mechanism {
                theta: 1000.0,
                kappa_shear: 0.2,
                kappa_p: 0.4,
            }],
            1e-5,
        )
        .expect("positive");
        let (lambda, mu) = (2.0e6, 1.3e6);
        let (shear, volume) = spectrum.part_weights(lambda, mu).expect("positive")[0];
        // The P-wave modulus relaxes as `lambda w_v + 2 mu w_s = pi kappa^p`.
        let pi = lambda + 2.0 * mu;
        assert!((lambda * volume + 2.0 * mu * shear - pi * 0.4).abs() <= 1e-9 * pi);
        // Too small a P weight leaves the volume part a negative one: refused.
        let low = RelaxationSpectrum::new(
            &[Mechanism {
                theta: 1000.0,
                kappa_shear: 0.4,
                kappa_p: 0.1,
            }],
            1e-5,
        )
        .expect("positive");
        assert_eq!(
            low.part_weights(lambda, mu),
            Err(RelaxationError::PartWeight(0))
        );
        // A law with no volume part (`lambda = 0`) cannot carry a P-wave
        // relaxation that differs from the shear's: refused; a proportional
        // one it carries on its shear part alone.
        assert_eq!(
            spectrum.part_weights(0.0, mu),
            Err(RelaxationError::PartWeight(0))
        );
        let proportional = RelaxationSpectrum::plasticine_crandall_1970();
        let weights = proportional.part_weights(0.0, mu).expect("proportional");
        assert!(
            weights[..3]
                .iter()
                .all(|&(shear, volume)| shear > 0.0 && volume == 0.0)
        );
    }
}

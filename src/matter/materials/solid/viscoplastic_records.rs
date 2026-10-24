//! Measured viscoplastic records: Norton's power law (`j2_hencky::Norton`)
//! with the elastic moduli it was measured with, the strain rates it covers
//! and where it comes from. A record is data; a scene builds its material
//! from one (`ViscoplasticRecord::material`).
//!
//! Two plasticines are never blended: Sapozhnikov and Ignatova's technical
//! plasticine and Chijiiwa, Hatamura and Hasegawa's (Trans. ISIJ 21, 1981)
//! differ by a factor of about 4 at 1/s, and the latter's law has a strain
//! hardening and a temperature term this law does not carry; it would be a
//! record of its own law.

use super::j2_hencky::Norton;
use super::von_mises::VonMisesMaterial;
use crate::materials::FromSI;
use crate::materials::physical_props::{DuctileProps, Elastic};

/// A measured Norton law and its elastic moduli.
#[derive(Clone, Copy, Debug)]
pub struct ViscoplasticRecord {
    pub law: Norton,
    /// Young's modulus and its stated uncertainty, Pa; Poisson's ratio.
    pub e_pa: f64,
    pub e_uncertainty_pa: f64,
    pub nu: f64,
    /// The material, its state and the source, page by page.
    pub material: &'static str,
    pub source: &'static str,
}

impl ViscoplasticRecord {
    /// Technical plasticine at 20 C, Sapozhnikov and Ignatova, PNRPU Mech.
    /// Bull. 2014 no. 2, 200-219 (doi 10.15593/perm.mech/2014.2.09): the flow
    /// stress `sigma = 6.37e5 eps_dot^0.266` Pa in tension, compression and
    /// indentation over 4e-4 to 80 1/s (eq. 7, p. 212), read as a Norton law
    /// with `rate_0 = 1/s` (p. 213: "elastic and viscous elements in
    /// series"); `E = 30 +- 5 MPa`, `nu = 0.30` initially (pp. 204-205).
    /// Read through the reading agent's report (`VISCOPLAST_SOURCES_REPORT`
    /// rows 22, sec. 4 and 6, the Russian text); to be checked on the page
    /// before a presented result rests on it.
    ///
    /// Declared gaps, each with its removal criterion in that report: no
    /// density is given (the scene states its own, `material`); engineering
    /// stress and nominal rate, while the law here is in Kirchhoff stress and
    /// log rate (about 15 % apart at 15 % strain); compression-dominated
    /// states only (tension damages it beyond 3-5 %; counted,
    /// `Stepped::outside`); Poisson's ratio rises to 0.45 at 3 % strain; the
    /// shear law (eq. 8, another exponent) is not this one.
    pub fn technical_plasticine_sapozhnikov_2014() -> Self {
        Self {
            law: Norton {
                k: 6.37e5,
                m: 0.266,
                rate_0: 1.0,
                valid_rates: (4.0e-4, 80.0),
            },
            e_pa: 30.0e6,
            e_uncertainty_pa: 5.0e6,
            nu: 0.30,
            material: "technical plasticine, 20 C",
            source: "Sapozhnikov and Ignatova 2014, eq. 7 p. 212, pp. 204-205, 213",
        }
    }

    /// The ductile material of this record for a scene, with Young's modulus
    /// `e_pa` (the record's or a value inside its uncertainty) and the
    /// density `rho_kg_m3` the scene states with its own source: the record
    /// gives none, and no default is taken.
    pub fn material(
        &self,
        (e_pa, rho_kg_m3): (f64, f64),
        config: &crate::SimConfig,
    ) -> VonMisesMaterial {
        let props = DuctileProps {
            elastic: Elastic {
                e_pa: e_pa as f32,
                nu: self.nu as f32,
                rho_kg_m3: rho_kg_m3 as f32,
            },
            yield_stress_pa: 0.0,
        };
        let mut material = VonMisesMaterial::from_physical(&props, config);
        material.norton = Some(self.law);
        material
    }
}

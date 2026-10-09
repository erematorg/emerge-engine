//! The light the demos that render through the SI optics contract declare.
//!
//! A physical render needs the scene to say what light reaches the slab and
//! what lies behind it; the materials only say how they absorb and scatter.
//! These demos share one declaration so their water reads the same: an
//! overcast sky above, normalised so it displays as white, and a darker
//! backdrop behind at a quarter of it. Equal sky and backdrop would make the
//! slab's reflection and scattering cancel out of view. The values are the
//! ones `basic_fluids_gpu` first declared, kept rather than re-tuned.

use emerge::SimConfig;
use emerge::render::{PhysicalRenderContract, PhysicalRenderContractParams};

/// Sky radiance above the slab, in units of the display's white.
const SKY_RADIANCE: f32 = 1.0;
/// Backdrop radiance behind the slab, relative to the sky.
const BACKDROP_RADIANCE: f32 = 0.25;

/// The contract for `config`'s scene under the shared overcast light: its
/// cell size, its stated slice thickness (`SimConfig::slice_thickness_m`,
/// required), a camera looking straight through the slab and light from
/// above and in front.
pub fn overcast_contract(config: &SimConfig) -> PhysicalRenderContract {
    PhysicalRenderContract::new(PhysicalRenderContractParams {
        dx_meters: config.dx_meters,
        slice_thickness_m: config.require_slice_thickness_m("the physical render contract"),
        incident_radiance_w_m2_sr: [SKY_RADIANCE; 3],
        background_radiance_w_m2_sr: [BACKDROP_RADIANCE; 3],
        display_white_radiance_w_m2_sr: [SKY_RADIANCE; 3],
        camera_direction: glam::Vec3::new(0.0, 0.0, -1.0),
        light_direction: glam::Vec3::new(0.0, 1.0, -1.0),
    })
    .expect("the declared light is a valid physical render contract")
}

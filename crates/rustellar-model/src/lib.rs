//! Transit light curve model for circular orbits.
//!
//! # Accuracy
//!
//! The light curve agrees with batman 2.5.1 (Kreidberg 2015) to `2.1e-8`
//! (0.02 ppm) at worst, over 840 transits with quadratic limb darkening:
//! radius ratio `k` from 0.005 to 0.29, impact parameter `b` from 0 to 1.05,
//! and scaled semi-major axis `a_rs` from 1.5 to 100. The comparison is part
//! of the test suite in the source repository. See
//! [`TransitParams::flux_at`] for why the remaining difference is batman's.
mod error;
mod numeric;
mod params;
mod uniform;

#[cfg(test)]
mod tests;

pub use error::{LimbDarkeningError, ParamError};
pub use numeric::flux_ring_sum;
pub use params::{
    Contacts, ImpactParameter, LimbDarkening, RadiusRatio, ScaledSemiMajorAxis, TransitParams,
};
pub use uniform::flux_uniform;

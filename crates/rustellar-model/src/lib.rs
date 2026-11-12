//! Transit light curve model for circular orbits.
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

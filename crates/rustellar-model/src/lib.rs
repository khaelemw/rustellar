//! Transit light curve model for circular orbits.
mod error;
mod params;

pub use error::{LimbDarkeningError, ParamError};
pub use params::{ImpactParameter, LimbDarkening, RadiusRatio, ScaledSemiMajorAxis, TransitParams};

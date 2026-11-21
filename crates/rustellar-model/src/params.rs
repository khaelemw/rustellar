// The parameter types, split by kind: the three scalar parameters, the
// limb-darkening law, the full set of transit parameters, and the contact
// points they produce.
mod contacts;
mod limb_darkening;
mod orbit;
mod scalars;
mod transit;

pub use contacts::Contacts;
pub use limb_darkening::LimbDarkening;
#[cfg(test)]
pub(crate) use orbit::{kepler, sin_cos_deg};
pub use scalars::{ImpactParameter, RadiusRatio, ScaledSemiMajorAxis};
#[cfg(test)]
pub(crate) use transit::Averaging;
pub use transit::TransitParams;
pub(crate) use transit::check_lengths;

// The parameter types, split by kind: the three scalar parameters, the
// limb-darkening law, the full set of transit parameters, and the contact
// points they produce.
mod contacts;
mod limb_darkening;
mod scalars;
mod transit;

pub use contacts::Contacts;
pub use limb_darkening::LimbDarkening;
pub use scalars::{ImpactParameter, RadiusRatio, ScaledSemiMajorAxis};
pub use transit::TransitParams;

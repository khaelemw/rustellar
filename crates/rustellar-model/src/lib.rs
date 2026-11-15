//! Transit light curves: how a star dims as a planet on a circular orbit
//! passes in front of it.
//!
//! The model is the analytic one of Mandel & Agol (2002) for quadratic limb
//! darkening (the star is darker towards its edge), in pure Rust with no
//! dependencies beyond `rustellar-core`.
//!
//! # Quick start
//!
//! Describe the planet with [`TransitParams`], then ask for the flux at any
//! times:
//!
//! ```
//! use rustellar_core::{Epoch, TimeDelta};
//! use rustellar_model::{
//!     ImpactParameter, LimbDarkening, RadiusRatio, ScaledSemiMajorAxis, TransitParams,
//! };
//!
//! let params = TransitParams::new(
//!     Epoch::from_btjd(1000.0)?,           // mid-transit time
//!     TimeDelta::from_days(3.0)?,          // orbital period
//!     RadiusRatio::new(0.1)?,              // planet radius / star radius
//!     ScaledSemiMajorAxis::new(10.0)?,     // orbit radius / star radius
//!     ImpactParameter::new(0.3)?,          // closest approach to the star's centre
//!     LimbDarkening::quadratic(0.3, 0.25)?,
//! )?;
//!
//! let times: Vec<Epoch> = (-50..=50)
//!     .map(|i| Epoch::from_btjd(1000.0 + 0.002 * f64::from(i)))
//!     .collect::<Result<_, _>>()?;
//! let flux = params.light_curve(&times);
//!
//! assert_eq!(flux.len(), times.len());
//! assert!(flux.iter().all(|&f| (0.98..=1.0).contains(&f)));
//!
//! // Durations and contact points come from the same geometry.
//! let t14 = params.t14().expect("this planet transits");
//! assert!((t14.as_hours() - 2.431).abs() < 1e-3);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # What is here
//!
//! - [`TransitParams`]: the planet and orbit, checked once when built. Gives
//!   the flux at a time ([`flux_at`](TransitParams::flux_at),
//!   [`light_curve`](TransitParams::light_curve)), the orbit geometry
//!   ([`separation`](TransitParams::separation),
//!   [`in_front`](TransitParams::in_front)), and the transit's timing
//!   ([`t14`](TransitParams::t14), [`t23`](TransitParams::t23),
//!   [`contacts`](TransitParams::contacts),
//!   [`transit_times`](TransitParams::transit_times)).
//! - [`TransitModel`]: a fixed set of times, and optionally an [`Exposure`],
//!   for evaluating many parameter sets without allocating, as in a fit.
//! - [`Exposure`]: the mean flux over each measurement's exposure, for long
//!   cadences such as Kepler's (Kipping 2010); see also
//!   [`binned_flux_at`](TransitParams::binned_flux_at).
//! - [`flux`]: the flux for a given planet-star separation, with no orbit.
//! - [`flux_uniform`]: the same for a star with no limb darkening.
//! - [`flux_ring_sum`]: a slow numerical integration of the same quantity,
//!   kept as an independent check.
//!
//! # Conventions
//!
//! - Lengths are in stellar radii: the separation `z`, the radius ratio `k`,
//!   the impact parameter `b`, and the semi-major axis `a_rs`.
//! - Times are [`Epoch`](rustellar_core::Epoch)s (BJD_TDB) and
//!   [`TimeDelta`](rustellar_core::TimeDelta)s, so units cannot be mixed up.
//! - Flux is relative to the star's flux out of transit: 1 out of transit,
//!   and never outside `[0, 1]`.
//! - Every input is checked by its type's constructor, which returns an error
//!   rather than panicking. After that, nothing in the model can fail.
//!
//! # Limits
//!
//! Circular orbits only, with no light from the planet itself. The radius
//! ratio is at most [`RadiusRatio::MAX`] (100), above which the analytic flux
//! loses accuracy.
//!
//! # Accuracy
//!
//! [`flux`] is within about `1e-10` of the exact value for `k` up to 20, and
//! `2e-9` at 100. Against Agol, Luger & Foreman-Mackey (2020), the most
//! precise published method, at about 28,000 random points over the whole
//! valid range, the worst difference is `1e-13` for `k` below 1, `1.4e-12`
//! below 10, `3.3e-11` below 30 and `1.5e-9` up to 100. It also stays accurate
//! on the lines where the formula changes case, where batman, PyTransit and
//! that method's reference code each fail somewhere.
//!
//! The light curve agrees with batman 2.5.1 (Kreidberg 2015) to `2.1e-8`
//! (0.02 ppm) at worst, over 840 transits with quadratic limb darkening:
//! radius ratio `k` from 0.005 to 0.29, impact parameter `b` from 0 to 1.05,
//! and scaled semi-major axis `a_rs` from 1.5 to 100. The comparison is part
//! of the test suite in the source repository. See
//! [`TransitParams::flux_at`] for why the remaining difference is batman's.
mod analytic;
mod elliptic;
mod error;
mod exposure;
mod model;
mod numeric;
mod params;
mod uniform;

#[cfg(test)]
mod tests;

pub use analytic::flux;
pub use error::{LengthMismatch, LimbDarkeningError, ParamError};
pub use exposure::Exposure;
pub use model::TransitModel;
pub use numeric::flux_ring_sum;
pub use params::{
    Contacts, ImpactParameter, LimbDarkening, RadiusRatio, ScaledSemiMajorAxis, TransitParams,
};
pub use uniform::flux_uniform;

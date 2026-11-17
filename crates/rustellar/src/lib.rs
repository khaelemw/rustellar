//! Exoplanet and astronomy time-series tools in pure Rust.
//!
//! Each tool is a separate crate, re-exported here as a module behind a
//! feature of the same name, so a project compiles only the tools it turns
//! on. The shared types (light curves and times) are always available at the
//! top level.
//!
//! | Feature | Module | What it does |
//! |---|---|---|
//! | `bls` | [`bls`] | Box least squares transit search (in development) |
//! | `detrend` | [`detrend`] | Detrending: a sliding biweight, median or mean, to full precision (in development) |
//! | `model` | [`model`] | Transit light curves: Mandel & Agol (2002) with quadratic limb darkening, exposure averaging |
//! | `full` | | Every tool |
//!
//! No feature is on by default. In `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! rustellar = { version = "0.0", features = ["model"] }
//! ```
//!
//! Each tool can also be used on its own (for example `rustellar-model`); the
//! types are the same either way.
//!
//! # Examples
//!
//! ```
//! # #[cfg(feature = "model")] {
//! use rustellar::model::{
//!     ImpactParameter, LimbDarkening, RadiusRatio, ScaledSemiMajorAxis, TransitParams,
//! };
//! use rustellar::{Epoch, TimeDelta};
//!
//! let params = TransitParams::new(
//!     Epoch::from_btjd(1000.0)?,
//!     TimeDelta::from_days(3.0)?,
//!     RadiusRatio::new(0.1)?,
//!     ScaledSemiMajorAxis::new(10.0)?,
//!     ImpactParameter::new(0.3)?,
//!     LimbDarkening::quadratic(0.3, 0.25)?,
//! )?;
//! assert!(params.flux_at(Epoch::from_btjd(1000.0)?) < 0.99);
//! # }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

#![cfg_attr(docsrs, feature(doc_cfg))]

pub use rustellar_core::{
    Epoch, FoldedLightCurve, LightCurve, LightCurveError, TimeDelta, TimeError,
};

/// Box least squares transit search, from [`rustellar-bls`](rustellar_bls).
#[cfg(feature = "bls")]
#[cfg_attr(docsrs, doc(cfg(feature = "bls")))]
pub use rustellar_bls as bls;

/// Detrending light curves, from [`rustellar-detrend`](rustellar_detrend).
#[cfg(feature = "detrend")]
#[cfg_attr(docsrs, doc(cfg(feature = "detrend")))]
pub use rustellar_detrend as detrend;

/// Transit light curves, from [`rustellar-model`](rustellar_model).
#[cfg(feature = "model")]
#[cfg_attr(docsrs, doc(cfg(feature = "model")))]
pub use rustellar_model as model;

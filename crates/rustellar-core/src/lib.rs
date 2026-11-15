//! Light curves and time types shared by the rustellar crates.
//!
//! - [`Epoch`]: a point in time on the barycentric (BJD_TDB) time scale,
//!   stored as BTJD.
//! - [`TimeDelta`]: a length of time, such as a period or a duration.
//! - [`LightCurve`]: matching time, flux and flux-error arrays, checked once
//!   when built, which can be [folded](LightCurve::fold) on a period.
//!
//! Constructors check their input and return an error instead of panicking;
//! once a value exists it is valid. There are no dependencies.
//!
//! # Examples
//!
//! ```
//! use rustellar_core::{Epoch, LightCurve, TimeDelta};
//!
//! let lc = LightCurve::from_btjd(
//!     vec![0.0, 0.5, 1.0, 1.5],
//!     vec![1.0, 0.99, 1.0, 0.99],
//!     vec![0.001; 4],
//! )?;
//!
//! // Fold on a 1-day period with phase 0 at BTJD 0.5: the two dips line up.
//! let folded = lc.fold(TimeDelta::from_days(1.0)?, Epoch::from_btjd(0.5)?)?;
//! assert_eq!(folded.phase(), &[-0.5, -0.5, 0.0, 0.0]);
//! assert_eq!(folded.flux(), &[1.0, 1.0, 0.99, 0.99]);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

mod error;
mod lightcurve;
mod time;

pub use error::{LightCurveError, TimeError};
pub use lightcurve::{FoldedLightCurve, LightCurve};
pub use time::{Epoch, TimeDelta};

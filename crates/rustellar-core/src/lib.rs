//! Light curves and time types shared by the rustellar crates.
//!
//! - [`Epoch`]: a point in time, in days on the data's own time scale
//!   (BJD_TDB, or an offset form such as BTJD or BKJD).
//! - [`TimeDelta`]: a length of time, such as a period or a duration.
//! - [`LightCurve`]: matching time, flux and flux-error arrays, checked once
//!   when built, which can be [folded](LightCurve::fold) on a period.
//! - [`Periods`] and [`OfirPeriod`]: the trial periods of a transit search,
//!   any list or Ofir's (2014) optimal grid.
//! - [`Device`]: where a search runs, the CPU or a GPU.
//!
//! Constructors check their input and return an error instead of panicking;
//! once a value exists it is valid. There are no dependencies.
//!
//! # Examples
//!
//! ```
//! use rustellar_core::{Epoch, LightCurve, TimeDelta};
//!
//! let lc = LightCurve::from_days(
//!     vec![0.0, 0.5, 1.0, 1.5],
//!     vec![1.0, 0.99, 1.0, 0.99],
//!     vec![0.001; 4],
//! )?;
//!
//! // Fold on a 1-day period with phase 0 at day 0.5: the two dips line up.
//! let folded = lc.fold(TimeDelta::from_days(1.0)?, Epoch::from_days(0.5)?)?;
//! assert_eq!(folded.phase(), &[-0.5, -0.5, 0.0, 0.0]);
//! assert_eq!(folded.flux(), &[1.0, 1.0, 0.99, 0.99]);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

mod device;
mod error;
mod grid;
mod lightcurve;
mod time;

pub use device::Device;
pub use error::{GridError, LightCurveError, TimeError};
pub use grid::{OfirPeriod, Periods};
pub use lightcurve::{FoldedLightCurve, LightCurve};
pub use time::{Epoch, TimeDelta};

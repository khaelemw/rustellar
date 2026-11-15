use std::num::NonZeroUsize;

use rustellar_core::TimeDelta;

use crate::error::ParamError;

/// How long each measurement integrates light for, and how many instants to
/// average over to model that.
///
/// A detector records the mean flux over its exposure, not the flux at one
/// instant. For long exposures, such as Kepler's 29.4-minute long cadence,
/// this smooths ingress and egress and makes the transit look shallower and
/// longer; fitting instantaneous models to such data biases `a_rs`, `b` and
/// `k` (Kipping 2010, "Binning is sinning").
///
/// The model averages the flux at `samples` instants spread over the
/// exposure, each in the middle of an equal slice of it: offsets of
/// `duration * ((i + 1/2) / samples - 1/2)` from the time stamp, for
/// `i = 0..samples`. That is the midpoint rule, Kipping's resampling, and the
/// scheme of PyTransit and of exoplanet's default. batman instead spreads its
/// samples from edge to edge of the exposure, which converges to the same
/// answer more slowly.
///
/// Time stamps are taken to be the middle of each exposure. With one sample,
/// the result is exactly the flux at the time stamp.
///
/// # Choosing `samples`
///
/// The error of the average falls as `1 / samples^2`. Kipping (2010)
/// recommends enough samples that the error is well below the noise. For
/// Kepler long cadence and a hot Jupiter (`k = 0.1`), the worst error is
/// about `5e-6` with 15 samples and `1e-6` with 30. The error is largest in
/// ingress and egress and is zero where the flux is flat.
///
/// # Examples
///
/// ```
/// use std::num::NonZeroUsize;
///
/// use rustellar_core::TimeDelta;
/// use rustellar_model::Exposure;
///
/// // Kepler long cadence: 29.4 minutes, averaged over 15 instants.
/// let exposure = Exposure::new(
///     TimeDelta::from_hours(29.4 / 60.0)?,
///     NonZeroUsize::new(15).unwrap(),
/// )?;
/// assert_eq!(exposure.samples().get(), 15);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Exposure {
    duration: TimeDelta,
    samples: NonZeroUsize,
}

impl Exposure {
    /// Makes an exposure of the given duration, averaged over `samples`
    /// instants.
    ///
    /// # Errors
    ///
    /// [`ParamError::ExposureNotPositive`] if `duration` is zero or negative.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::num::NonZeroUsize;
    ///
    /// use rustellar_core::TimeDelta;
    /// use rustellar_model::{Exposure, ParamError};
    ///
    /// let samples = NonZeroUsize::new(5).unwrap();
    /// let tess_2min = Exposure::new(TimeDelta::from_hours(2.0 / 60.0)?, samples)?;
    /// assert_eq!(tess_2min.samples(), samples);
    ///
    /// let none = Exposure::new(TimeDelta::from_days(0.0)?, samples);
    /// assert_eq!(none, Err(ParamError::ExposureNotPositive));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(duration: TimeDelta, samples: NonZeroUsize) -> Result<Self, ParamError> {
        if duration.as_days() <= 0.0 {
            return Err(ParamError::ExposureNotPositive);
        }

        Ok(Self { duration, samples })
    }

    /// How long each measurement integrates for.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::num::NonZeroUsize;
    ///
    /// use rustellar_core::TimeDelta;
    /// use rustellar_model::Exposure;
    ///
    /// let exposure = Exposure::new(TimeDelta::from_hours(0.5)?, NonZeroUsize::new(15).unwrap())?;
    /// assert_eq!(exposure.duration().as_hours(), 0.5);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[must_use]
    pub fn duration(self) -> TimeDelta {
        self.duration
    }

    /// How many instants each measurement is averaged over.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::num::NonZeroUsize;
    ///
    /// use rustellar_core::TimeDelta;
    /// use rustellar_model::Exposure;
    ///
    /// let exposure = Exposure::new(TimeDelta::from_hours(0.5)?, NonZeroUsize::new(15).unwrap())?;
    /// assert_eq!(exposure.samples().get(), 15);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[must_use]
    pub fn samples(self) -> NonZeroUsize {
        self.samples
    }

    /// The offsets from the time stamp of each instant averaged over, in
    /// time order: the middle of each of `samples` equal slices of the
    /// exposure.
    ///
    /// Exactly symmetric about zero, so the averaging adds no asymmetry of
    /// its own to the light curve.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::num::NonZeroUsize;
    ///
    /// use rustellar_core::TimeDelta;
    /// use rustellar_model::Exposure;
    ///
    /// // Four samples over one day: the middles of four six-hour slices.
    /// let exposure = Exposure::new(TimeDelta::from_days(1.0)?, NonZeroUsize::new(4).unwrap())?;
    /// let hours: Vec<f64> = exposure.offsets().map(|dt| dt.as_hours()).collect();
    /// assert_eq!(hours, [-9.0, -3.0, 3.0, 9.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn offsets(self) -> impl Iterator<Item = TimeDelta> {
        let n = self.samples.get() as f64;
        // (i + 1/2) / n - 1/2 as (2i + 1 - n) / 2n: the numerator is an exact
        // integer, so offsets i and n - 1 - i are exact negatives.
        (0..self.samples.get())
            .map(move |i| self.duration * ((2.0 * i as f64 + 1.0 - n) / (2.0 * n)))
    }
}

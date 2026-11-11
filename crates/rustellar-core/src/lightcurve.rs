use crate::{Duration, Epoch, LightCurveError};

/// A validated light curve: matching arrays of time, flux, and flux error.
///
/// Every `LightCurve` has at least one point, all values finite, every flux
/// error greater than zero, and time strictly increasing. These are checked once
/// in [`new`](Self::new); the fields are private, so they stay true afterwards.
///
/// Times are [`Epoch`]s, so their time system is fixed. Flux and flux error can
/// be in any unit, as long as both use the same one.
#[derive(Debug, Clone, PartialEq)]
pub struct LightCurve {
    time: Vec<Epoch>,
    flux: Vec<f64>,
    flux_err: Vec<f64>,
}

impl LightCurve {
    /// Builds a light curve, checking the inputs.
    ///
    /// Time must be strictly increasing: a repeated time is rejected, since in
    /// practice it usually means rows were duplicated upstream.
    /// Flux may be zero or negative (for example after detrending), but each
    /// flux error must be greater than zero.
    ///
    /// # Errors
    ///
    /// The checks run in this order, and the first one that fails is returned:
    ///
    /// 1. [`LengthMismatch`](LightCurveError::LengthMismatch) if the three
    ///    arrays differ in length.
    /// 2. [`Empty`](LightCurveError::Empty) if they have no points.
    /// 3. [`NonFinite`](LightCurveError::NonFinite) if any flux or flux error
    ///    is NaN or infinite.
    /// 4. [`NonPositiveError`](LightCurveError::NonPositiveError) if any flux
    ///    error is zero or negative.
    /// 5. [`NotSorted`](LightCurveError::NotSorted) if a time is not greater
    ///    than the one before it.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustellar_core::{Epoch, LightCurve, LightCurveError};
    ///
    /// let time = vec![Epoch::from_btjd(1.0)?, Epoch::from_btjd(2.0)?];
    /// let lc = LightCurve::new(time, vec![1.0, 0.99], vec![0.01, 0.01])?;
    /// assert_eq!(lc.len(), 2);
    ///
    /// let unsorted = vec![Epoch::from_btjd(2.0)?, Epoch::from_btjd(1.0)?];
    /// let result = LightCurve::new(unsorted, vec![1.0, 1.0], vec![0.1, 0.1]);
    /// assert_eq!(result, Err(LightCurveError::NotSorted));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(
        time: Vec<Epoch>,
        flux: Vec<f64>,
        flux_err: Vec<f64>,
    ) -> Result<Self, LightCurveError> {
        if time.len() != flux.len() || flux.len() != flux_err.len() {
            return Err(LightCurveError::LengthMismatch);
        }

        if time.is_empty() {
            return Err(LightCurveError::Empty);
        }

        let flux_finite = flux.iter().all(|f| f.is_finite());
        let flux_err_finite = flux_err.iter().all(|fe| fe.is_finite());

        if !flux_finite || !flux_err_finite {
            return Err(LightCurveError::NonFinite);
        }

        let flux_err_positive = flux_err.iter().all(|&fe| fe > 0.0);

        if !flux_err_positive {
            return Err(LightCurveError::NonPositiveError);
        }

        let time_increasing = time.windows(2).all(|w| w[0] < w[1]);

        if !time_increasing {
            return Err(LightCurveError::NotSorted);
        }

        Ok(Self {
            time,
            flux,
            flux_err,
        })
    }

    /// Builds a light curve from times given as BTJD values.
    ///
    /// The same as [`new`](Self::new), but takes plain numbers for time, as
    /// read from a TESS light curve file.
    ///
    /// # Errors
    ///
    /// [`NonFinite`](LightCurveError::NonFinite) if any time is NaN or
    /// infinite. This is checked first, before the checks listed on
    /// [`new`](Self::new).
    ///
    /// # Examples
    ///
    /// ```
    /// use rustellar_core::LightCurve;
    ///
    /// let lc = LightCurve::from_btjd(vec![1.0, 2.0], vec![1.0, 0.99], vec![0.01, 0.01])?;
    /// assert_eq!(lc.time()[1].as_btjd(), 2.0);
    /// # Ok::<(), rustellar_core::LightCurveError>(())
    /// ```
    pub fn from_btjd(
        time: Vec<f64>,
        flux: Vec<f64>,
        flux_err: Vec<f64>,
    ) -> Result<Self, LightCurveError> {
        let time = time
            .into_iter()
            .map(Epoch::from_btjd)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| LightCurveError::NonFinite)?;

        Self::new(time, flux, flux_err)
    }

    /// Number of points. Never zero, because [`new`](Self::new) rejects empty
    /// input.
    #[expect(
        clippy::len_without_is_empty,
        reason = "new() rejects empty input, so a LightCurve is never empty"
    )]
    pub fn len(&self) -> usize {
        self.time.len()
    }

    /// Observation times, strictly increasing.
    pub fn time(&self) -> &[Epoch] {
        &self.time
    }

    /// Flux values, one per time.
    pub fn flux(&self) -> &[f64] {
        &self.flux
    }

    /// Flux errors, one per time, all greater than zero.
    pub fn flux_err(&self) -> &[f64] {
        &self.flux_err
    }

    /// Folds the light curve on `period`, with phase 0 at time `t0`.
    ///
    /// Each phase is `((t - t0) / period + 0.5).rem_euclid(1.0) - 0.5`, which
    /// lies in `[-0.5, 0.5)`: a point half a period from `t0` gets `-0.5`,
    /// never `+0.5`. The result is sorted by phase, with each flux and flux
    /// error moved along with its phase. Points with equal phase (from
    /// different orbits) keep their time order.
    ///
    /// # Errors
    ///
    /// [`InvalidPeriod`](LightCurveError::InvalidPeriod) if `period` is zero or
    /// negative.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustellar_core::{Duration, Epoch, LightCurve};
    ///
    /// let lc = LightCurve::from_btjd(
    ///     vec![0.0, 0.5, 1.0, 1.5],
    ///     vec![10.0, 11.0, 12.0, 13.0],
    ///     vec![0.1, 0.1, 0.1, 0.1],
    /// )?;
    /// let folded = lc.fold(Duration::from_days(2.0)?, Epoch::from_btjd(0.0)?)?;
    /// assert_eq!(folded.phase(), &[-0.5, -0.25, 0.0, 0.25]);
    /// assert_eq!(folded.flux(), &[12.0, 13.0, 10.0, 11.0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn fold(&self, period: Duration, t0: Epoch) -> Result<FoldedLightCurve, LightCurveError> {
        if period.as_days() <= 0.0 {
            return Err(LightCurveError::InvalidPeriod);
        }

        let phase: Vec<f64> = self
            .time
            .iter()
            .map(|&t| ((t - t0) / period + 0.5).rem_euclid(1.0) - 0.5)
            .collect();

        // Sort row indices by phase, then rebuild all three arrays from them in
        // ascending phase order.
        let mut order: Vec<usize> = (0..phase.len()).collect();
        // `sort_by` leaves points with equal phase in their original time order.
        order.sort_by(|&a, &b| phase[a].total_cmp(&phase[b]));

        Ok(FoldedLightCurve {
            phase: order.iter().map(|&i| phase[i]).collect(),
            flux: order.iter().map(|&i| self.flux[i]).collect(),
            flux_err: order.iter().map(|&i| self.flux_err[i]).collect(),
        })
    }
}

/// A light curve folded on a period, from [`LightCurve::fold`].
///
/// Rows are sorted by phase. Row `i` of [`phase`](Self::phase),
/// [`flux`](Self::flux), and [`flux_err`](Self::flux_err) is the same
/// measurement.
#[derive(Debug, Clone, PartialEq)]
pub struct FoldedLightCurve {
    phase: Vec<f64>,
    flux: Vec<f64>,
    flux_err: Vec<f64>,
}

impl FoldedLightCurve {
    /// Phases in `[-0.5, 0.5)`, in increasing order, with 0 at the fold's `t0`.
    pub fn phase(&self) -> &[f64] {
        &self.phase
    }

    /// Flux values from the original light curve, in phase order.
    pub fn flux(&self) -> &[f64] {
        &self.flux
    }

    /// Flux errors from the original light curve, in phase order.
    pub fn flux_err(&self) -> &[f64] {
        &self.flux_err
    }
}

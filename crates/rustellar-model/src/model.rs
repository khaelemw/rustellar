use rustellar_core::Epoch;

use crate::{Exposure, TransitParams, error::LengthMismatch, params::check_lengths};

/// A light-curve model for a fixed set of times, for evaluating many
/// parameter sets, as in a fit or a sampler.
///
/// It owns the times, the exposure if there is one, and a buffer for the
/// result, so [`light_curve`](Self::light_curve) does no allocation after the
/// model is built. This mirrors batman's `TransitModel(params, t)` and
/// PyTransit's `set_data`: the data are fixed once, and only the parameters
/// change between calls.
///
/// With an [`Exposure`], each flux is the mean over the exposure
/// ([`TransitParams::binned_flux_at`]); without one, the flux at the time
/// stamp ([`TransitParams::flux_at`]).
///
/// # Examples
///
/// ```
/// use std::num::NonZeroUsize;
///
/// use rustellar_core::{Epoch, TimeDelta};
/// use rustellar_model::{
///     Exposure, ImpactParameter, LimbDarkening, RadiusRatio, ScaledSemiMajorAxis,
///     TransitModel, TransitParams,
/// };
///
/// // Kepler long-cadence time stamps across one transit.
/// let cadence = 29.4 / 60.0 / 24.0;
/// let times: Vec<Epoch> = (-10..=10)
///     .map(|i| Epoch::from_btjd(f64::from(i) * cadence).unwrap())
///     .collect();
/// let exposure = Exposure::new(TimeDelta::from_days(cadence)?, NonZeroUsize::new(15).unwrap())?;
/// let mut model = TransitModel::with_exposure(times, exposure);
///
/// for k in [0.09, 0.1, 0.11] {
///     let params = TransitParams::new(
///         Epoch::from_btjd(0.0)?,
///         TimeDelta::from_days(3.0)?,
///         RadiusRatio::new(k)?,
///         ScaledSemiMajorAxis::new(10.0)?,
///         ImpactParameter::new(0.3)?,
///         LimbDarkening::quadratic(0.3, 0.25)?,
///     )?;
///     let flux = model.light_curve(params);
///     assert_eq!(flux.len(), 21);
///     assert!(flux[10] < 1.0 - k * k * 0.9);
/// }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct TransitModel {
    times: Vec<Epoch>,
    exposure: Option<Exposure>,
    flux: Vec<f64>,
}

impl TransitModel {
    /// A model of the flux at each of `times`, with no exposure averaging.
    ///
    /// The times do not need to be sorted. For a
    /// [`LightCurve`](rustellar_core::LightCurve), pass `lc.time().to_vec()`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use rustellar_core::Epoch;
    /// # use rustellar_model::TransitModel;
    /// # let times: Vec<Epoch> = [-0.1, 0.0, 0.1]
    /// #     .into_iter()
    /// #     .map(|t| Epoch::from_btjd(t).unwrap())
    /// #     .collect();
    /// # use rustellar_core::TimeDelta;
    /// # use rustellar_model::{
    /// #     ImpactParameter, LimbDarkening, RadiusRatio, ScaledSemiMajorAxis, TransitParams,
    /// # };
    /// # // A hot Jupiter: t0 = BTJD 0, P = 3 d, k = 0.1, a_rs = 10, b = 0.3.
    /// # let params = TransitParams::new(
    /// #     Epoch::from_btjd(0.0)?,
    /// #     TimeDelta::from_days(3.0)?,
    /// #     RadiusRatio::new(0.1)?,
    /// #     ScaledSemiMajorAxis::new(10.0)?,
    /// #     ImpactParameter::new(0.3)?,
    /// #     LimbDarkening::quadratic(0.3, 0.25)?,
    /// # )?;
    /// let mut model = TransitModel::new(times);
    /// let flux = model.light_curve(params);
    /// assert!(flux[1] < flux[0]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[must_use]
    pub fn new(times: Vec<Epoch>) -> Self {
        let flux = vec![1.0; times.len()];
        Self {
            times,
            exposure: None,
            flux,
        }
    }

    /// A model of the mean flux over an exposure centred on each of `times`.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::num::NonZeroUsize;
    /// # use rustellar_core::Epoch;
    /// # use rustellar_model::TransitModel;
    /// # let times: Vec<Epoch> = [-0.1, 0.0, 0.1]
    /// #     .into_iter()
    /// #     .map(|t| Epoch::from_btjd(t).unwrap())
    /// #     .collect();
    /// # use rustellar_core::TimeDelta;
    /// use rustellar_model::Exposure;
    ///
    /// // TESS 200-second full-frame images, averaged over 5 samples.
    /// let exposure = Exposure::new(
    ///     TimeDelta::from_hours(200.0 / 3600.0)?,
    ///     NonZeroUsize::new(5).unwrap(),
    /// )?;
    /// let model = TransitModel::with_exposure(times, exposure);
    /// assert_eq!(model.exposure(), Some(exposure));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[must_use]
    pub fn with_exposure(times: Vec<Epoch>, exposure: Exposure) -> Self {
        Self {
            exposure: Some(exposure),
            ..Self::new(times)
        }
    }

    /// The times modelled, in the order given.
    ///
    /// # Examples
    ///
    /// ```
    /// # use rustellar_core::Epoch;
    /// # use rustellar_model::TransitModel;
    /// # let times: Vec<Epoch> = [-0.1, 0.0, 0.1]
    /// #     .into_iter()
    /// #     .map(|t| Epoch::from_btjd(t).unwrap())
    /// #     .collect();
    /// let model = TransitModel::new(times.clone());
    /// assert_eq!(model.times(), times.as_slice());
    /// # Ok::<(), rustellar_core::TimeError>(())
    /// ```
    #[must_use]
    pub fn times(&self) -> &[Epoch] {
        &self.times
    }

    /// The exposure averaged over, if any.
    ///
    /// # Examples
    ///
    /// ```
    /// # use rustellar_core::Epoch;
    /// # use rustellar_model::TransitModel;
    /// # let times: Vec<Epoch> = [-0.1, 0.0, 0.1]
    /// #     .into_iter()
    /// #     .map(|t| Epoch::from_btjd(t).unwrap())
    /// #     .collect();
    /// assert_eq!(TransitModel::new(times).exposure(), None);
    /// # Ok::<(), rustellar_core::TimeError>(())
    /// ```
    #[must_use]
    pub fn exposure(&self) -> Option<Exposure> {
        self.exposure
    }

    /// The flux at each time for `params`, in the order of
    /// [`times`](Self::times).
    ///
    /// Written into a buffer the model owns, so this does not allocate. The
    /// slice is overwritten by the next call.
    ///
    /// # Examples
    ///
    /// ```
    /// # use rustellar_core::Epoch;
    /// # use rustellar_model::TransitModel;
    /// # let times: Vec<Epoch> = [-0.1, 0.0, 0.1]
    /// #     .into_iter()
    /// #     .map(|t| Epoch::from_btjd(t).unwrap())
    /// #     .collect();
    /// # use rustellar_core::TimeDelta;
    /// # use rustellar_model::{
    /// #     ImpactParameter, LimbDarkening, RadiusRatio, ScaledSemiMajorAxis, TransitParams,
    /// # };
    /// # // A hot Jupiter: t0 = BTJD 0, P = 3 d, k = 0.1, a_rs = 10, b = 0.3.
    /// # let params = TransitParams::new(
    /// #     Epoch::from_btjd(0.0)?,
    /// #     TimeDelta::from_days(3.0)?,
    /// #     RadiusRatio::new(0.1)?,
    /// #     ScaledSemiMajorAxis::new(10.0)?,
    /// #     ImpactParameter::new(0.3)?,
    /// #     LimbDarkening::quadratic(0.3, 0.25)?,
    /// # )?;
    /// let mut model = TransitModel::new(times);
    /// let flux = model.light_curve(params);
    /// assert_eq!(flux.len(), 3);
    /// assert_eq!(flux[0], flux[2]); // symmetric about mid-transit
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn light_curve(&mut self, params: TransitParams) -> &[f64] {
        fill(&self.times, self.exposure, params, &mut self.flux);
        &self.flux
    }

    /// As [`light_curve`](Self::light_curve), but into a buffer the caller
    /// owns, so one model can be shared between threads, each with its own
    /// buffer.
    ///
    /// # Errors
    ///
    /// [`LengthMismatch`] if `out` is not the same length as the times; `out`
    /// is then left unchanged.
    ///
    /// # Examples
    ///
    /// ```
    /// # use rustellar_core::Epoch;
    /// # use rustellar_model::TransitModel;
    /// # let times: Vec<Epoch> = [-0.1, 0.0, 0.1]
    /// #     .into_iter()
    /// #     .map(|t| Epoch::from_btjd(t).unwrap())
    /// #     .collect();
    /// # use rustellar_core::TimeDelta;
    /// # use rustellar_model::{
    /// #     ImpactParameter, LimbDarkening, RadiusRatio, ScaledSemiMajorAxis, TransitParams,
    /// # };
    /// # // A hot Jupiter: t0 = BTJD 0, P = 3 d, k = 0.1, a_rs = 10, b = 0.3.
    /// # let params = TransitParams::new(
    /// #     Epoch::from_btjd(0.0)?,
    /// #     TimeDelta::from_days(3.0)?,
    /// #     RadiusRatio::new(0.1)?,
    /// #     ScaledSemiMajorAxis::new(10.0)?,
    /// #     ImpactParameter::new(0.3)?,
    /// #     LimbDarkening::quadratic(0.3, 0.25)?,
    /// # )?;
    /// let model = TransitModel::new(times);
    ///
    /// // Each thread can keep its own buffer and share the model.
    /// let mut out = vec![0.0; model.times().len()];
    /// model.light_curve_into(params, &mut out)?;
    /// assert!(out[1] < 1.0);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn light_curve_into(
        &self,
        params: TransitParams,
        out: &mut [f64],
    ) -> Result<(), LengthMismatch> {
        check_lengths(&self.times, out)?;
        fill(&self.times, self.exposure, params, out);
        Ok(())
    }
}

// Callers have checked that `out` is as long as `times`.
fn fill(times: &[Epoch], exposure: Option<Exposure>, params: TransitParams, out: &mut [f64]) {
    match exposure {
        None => {
            let clear = params.clear();
            for (f, &t) in out.iter_mut().zip(times) {
                *f = params.flux_beyond(t, clear);
            }
        }
        Some(exposure) => {
            let averaging = params.averaging(exposure);
            for (f, &t) in out.iter_mut().zip(times) {
                *f = params.binned_flux_with(t, &averaging);
            }
        }
    }
}

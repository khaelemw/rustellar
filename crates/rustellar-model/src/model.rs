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
    #[must_use]
    pub fn with_exposure(times: Vec<Epoch>, exposure: Exposure) -> Self {
        Self {
            exposure: Some(exposure),
            ..Self::new(times)
        }
    }

    /// The times modelled, in the order given.
    #[must_use]
    pub fn times(&self) -> &[Epoch] {
        &self.times
    }

    /// The exposure averaged over, if any.
    #[must_use]
    pub fn exposure(&self) -> Option<Exposure> {
        self.exposure
    }

    /// The flux at each time for `params`, in the order of
    /// [`times`](Self::times).
    ///
    /// Written into a buffer the model owns, so this does not allocate. The
    /// slice is overwritten by the next call.
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
            for (f, &t) in out.iter_mut().zip(times) {
                *f = params.flux_at(t);
            }
        }
        Some(exposure) => {
            for (f, &t) in out.iter_mut().zip(times) {
                *f = params.binned_flux_at(t, exposure);
            }
        }
    }
}

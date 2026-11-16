//! Running totals of the weights and centred fluxes, and the box statistic
//! from them: the inner loop of both searches.

use crate::{
    BlsError, Objective,
    sums::{Sum, between},
};

/// Compensated sums of the weights and of the centred fluxes, over the points
/// (or bins) up to one, in phase order.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Total {
    pub(crate) weight: Sum,
    pub(crate) flux: Sum,
}

/// The weight and centred flux in a box, between the totals at its start and
/// end, and out of it, between its end and its start one orbit on: the sums
/// [`BoxFit::from_sums`](crate::BoxFit) takes, for the best box's fit.
pub(crate) fn box_sums(first: &Total, end: &Total, wrap: &Total) -> [[f64; 2]; 2] {
    [
        [
            between(first.weight, end.weight),
            between(first.flux, end.flux),
        ],
        [
            between(end.weight, wrap.weight),
            between(end.flux, wrap.flux),
        ],
    ]
}

/// The sums over one orbit, for the statistic of every box.
pub(crate) struct Orbit {
    weight: f64,
    flux: f64,
    inverse_weight: f64,
    likelihood: bool,
}

impl Orbit {
    /// From the totals at the start and one orbit on, for boxes ranked by
    /// `objective`.
    pub(crate) fn new(first: &Total, wrap: &Total, objective: Objective) -> Self {
        let weight = between(first.weight, wrap.weight);
        Self {
            weight,
            flux: between(first.flux, wrap.flux),
            inverse_weight: 1.0 / weight,
            likelihood: objective == Objective::AstropyLikelihood,
        }
    }

    /// The orbit's summed weight `W`.
    pub(crate) fn weight(&self) -> f64 {
        self.weight
    }

    /// The orbit's summed centred flux `S`.
    pub(crate) fn flux(&self) -> f64 {
        self.flux
    }

    /// Whether boxes are ranked by astropy's likelihood.
    pub(crate) fn likelihood(&self) -> bool {
        self.likelihood
    }

    /// The box's score if it has a dip, else zero, with one division (this is
    /// the inner loop of the search): its `delta_chi2`, or for
    /// [`Objective::AstropyLikelihood`] its likelihood,
    /// `delta_chi2 W / (2 W_out)`. (The snr ranks boxes as `delta_chi2`, its
    /// square, does.) It differs from
    /// [`BoxFit::from_sums`](crate::BoxFit::from_sums) only in rounding, and the best box's fit is
    /// built with that.
    ///
    /// With `W = W_in + W_out` and `S = S_in + S_out` the orbit's sums (`S`
    /// is near zero, the fluxes being centred on their weighted mean),
    ///
    /// ```text
    /// depth      = S_out / W_out - S_in / W_in = -u W / (W_in W_out),  u = S_in - S W_in / W
    /// delta_chi2 = W_in W_out / W depth^2      = u^2 / (W_in W_out / W)
    /// ```
    ///
    /// so there is a dip when `u < 0`. `W_out = W - W_in` is as accurate as
    /// `W_out` itself while the box holds at most half the weight; past that,
    /// it is taken from the running totals.
    pub(crate) fn statistic(
        &self,
        first: &Total,
        end: &Total,
        wrap: &Total,
    ) -> Result<f64, BlsError> {
        let weight_in = between(first.weight, end.weight);
        let weight_out = if weight_in <= 0.5 * self.weight {
            self.weight - weight_in
        } else {
            between(end.weight, wrap.weight)
        };
        // A box with no point in it, or none out of it. (The difference of
        // two equal totals is exactly zero.)
        if weight_in == 0.0 || weight_out == 0.0 {
            return Ok(0.0);
        }
        let u = between(first.flux, end.flux) - self.flux * (weight_in * self.inverse_weight);
        // W_in (W_out / W) first: W_in W_out alone can overflow.
        let delta_chi2 = u * (u / (weight_in * (weight_out * self.inverse_weight)));
        let score = if self.likelihood {
            delta_chi2 * (0.5 * self.weight / weight_out)
        } else {
            delta_chi2
        };
        if !score.is_finite() {
            return Err(BlsError::OutOfRange);
        }
        Ok(if u < 0.0 { score } else { 0.0 })
    }

    /// [`statistic`](Self::statistic) of every box `length` long starting at
    /// `0..scores.len()`, into `scores`, with the same arithmetic box for box,
    /// so each score is the same to the last bit; with no branches in the
    /// loop, so the compiler can run several boxes at once. `totals` runs
    /// twice round the orbit.
    pub(crate) fn statistics(
        &self,
        totals: &[Total],
        length: usize,
        scores: &mut [f64],
    ) -> Result<(), BlsError> {
        if self.likelihood {
            self.statistics_for::<true>(totals, length, scores)
        } else {
            self.statistics_for::<false>(totals, length, scores)
        }
    }

    fn statistics_for<const LIKELIHOOD: bool>(
        &self,
        totals: &[Total],
        length: usize,
        scores: &mut [f64],
    ) -> Result<(), BlsError> {
        let m = scores.len();
        let half = 0.5 * self.weight;
        let mut bad = false;
        let boxes = totals[..m]
            .iter()
            .zip(&totals[length..length + m])
            .zip(&totals[m..2 * m]);
        for (score, ((first, end), wrap)) in scores.iter_mut().zip(boxes) {
            let weight_in = between(first.weight, end.weight);
            let rest = between(end.weight, wrap.weight);
            let weight_out = if weight_in <= half {
                self.weight - weight_in
            } else {
                rest
            };
            let empty = weight_in == 0.0 || weight_out == 0.0;
            let u = between(first.flux, end.flux) - self.flux * (weight_in * self.inverse_weight);
            let delta_chi2 = u * (u / (weight_in * (weight_out * self.inverse_weight)));
            let value = if LIKELIHOOD {
                delta_chi2 * (half / weight_out)
            } else {
                delta_chi2
            };
            bad |= !empty & !value.is_finite();
            *score = if empty | (u >= 0.0) | u.is_nan() {
                0.0
            } else {
                value
            };
        }
        if bad {
            return Err(BlsError::OutOfRange);
        }
        Ok(())
    }
}

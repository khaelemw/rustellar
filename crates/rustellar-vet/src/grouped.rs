//! One baseline shared by every point and a depth for each group of
//! transits: `flux = alpha + sum_k beta_k * shape` where each point in
//! transit belongs to one group `k`. Each point's shape enters one group
//! only, so the normal equations are an arrow (the baseline row and column
//! full, the rest diagonal) and solve exactly in one pass over the groups:
//!
//! - `beta_k = (C_k - B_k alpha) / A_k`, with `A_k = sum w s^2`,
//!   `B_k = sum w s` and `C_k = sum w s y` over the group's points;
//! - `alpha (W - sum B_k^2 / A_k) = Y - sum B_k C_k / A_k`, with `W` and
//!   `Y` the sums of `w` and `w y` over every point.
//!
//! The covariance follows from the same Schur complement
//! `S = W - sum B_k^2 / A_k`: `var(alpha) = 1 / S`, `cov(alpha, beta_k) =
//! -(B_k / A_k) / S`, `cov(beta_j, beta_k) = (B_j / A_j)(B_k / A_k) / S`
//! plus `1 / A_k` on the diagonal.

use crate::VetError;
use crate::sum::Sum;

/// The fit: the baseline, each group's coefficient, and their covariance.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Grouped {
    pub(crate) alpha: f64,
    pub(crate) var_alpha: f64,
    /// Each group's `beta`, or `None` where it has no point in transit.
    pub(crate) beta: Vec<Option<f64>>,
    /// Each group's `B / A`, the slope of its `beta` on `alpha`.
    ratio: Vec<f64>,
    /// Each group's `1 / A`.
    inverse: Vec<f64>,
    /// Points in transit in each group.
    pub(crate) points: Vec<usize>,
    /// The fit's chi-squared.
    pub(crate) chi2: f64,
}

impl Grouped {
    /// The covariance of groups `j` and `k`'s coefficients.
    pub(crate) fn cov_beta(&self, j: usize, k: usize) -> f64 {
        let shared = self.ratio[j] * self.ratio[k] * self.var_alpha;
        if j == k {
            shared + self.inverse[k]
        } else {
            shared
        }
    }

    /// The covariance of the baseline and group `k`'s coefficient.
    pub(crate) fn cov_alpha_beta(&self, k: usize) -> f64 {
        -self.ratio[k] * self.var_alpha
    }

    /// Group `k`'s fractional depth, `-beta / alpha`, if it has points.
    pub(crate) fn depth(&self, k: usize) -> Option<f64> {
        self.beta[k].map(|b| -b / self.alpha)
    }

    /// The covariance of groups `j` and `k`'s depths, by the delta method
    /// on `-beta / alpha`.
    pub(crate) fn cov_depth(&self, j: usize, k: usize) -> Option<f64> {
        let (bj, bk) = (self.beta[j]?, self.beta[k]?);
        let a = self.alpha;
        // Each depth's gradient: d/d alpha = beta / alpha^2, d/d beta =
        // -1 / alpha.
        let (gaj, gak, gb) = (bj / (a * a), bk / (a * a), -1.0 / a);
        Some(
            gaj * gak * self.var_alpha
                + gaj * gb * self.cov_alpha_beta(k)
                + gb * gak * self.cov_alpha_beta(j)
                + gb * gb * self.cov_beta(j, k),
        )
    }
}

/// The fit of `flux` (errors `flux_err`) with each point's `shape` in group
/// `group` (`None` out of transit, or where the shape is 0), `groups`
/// groups in all.
pub(crate) fn solve(
    flux: &[f64],
    flux_err: &[f64],
    shapes: &[f64],
    group: &[Option<usize>],
    groups: usize,
) -> Result<Grouped, VetError> {
    let (mut w_all, mut wy_all) = (Sum::default(), Sum::default());
    let mut a = vec![Sum::default(); groups];
    let mut b = vec![Sum::default(); groups];
    let mut c = vec![Sum::default(); groups];
    let mut points = vec![0; groups];
    let mut outside = 0usize;
    for (((&y, &e), &s), &g) in flux.iter().zip(flux_err).zip(shapes).zip(group) {
        let w = 1.0 / (e * e);
        w_all.add(w);
        wy_all.add(w * y);
        match g {
            Some(k) if s > 0.0 => {
                a[k].add(w * s * s);
                b[k].add(w * s);
                c[k].add(w * s * y);
                points[k] += 1;
            }
            _ => outside += 1,
        }
    }
    if points.iter().all(|&p| p == 0) {
        return Err(VetError::NoPointsInTransit);
    }
    if outside == 0 {
        return Err(VetError::NoPointsOutOfTransit);
    }
    let (mut schur, mut rhs) = (Sum::default(), Sum::default());
    schur.add(w_all.get());
    rhs.add(wy_all.get());
    let mut ratio = vec![0.0; groups];
    let mut inverse = vec![0.0; groups];
    for k in 0..groups {
        if points[k] == 0 {
            continue;
        }
        let (ak, bk, ck) = (a[k].get(), b[k].get(), c[k].get());
        ratio[k] = bk / ak;
        inverse[k] = 1.0 / ak;
        schur.add(-bk * ratio[k]);
        rhs.add(-ck * ratio[k]);
    }
    let var_alpha = 1.0 / schur.get();
    let alpha = rhs.get() * var_alpha;
    if alpha <= 0.0 {
        return Err(VetError::BaselineNotPositive);
    }
    let beta: Vec<Option<f64>> = (0..groups)
        .map(|k| (points[k] > 0).then(|| (c[k].get() - b[k].get() * alpha) / a[k].get()))
        .collect();
    // The chi-squared from the residuals themselves, not from the normal
    // equations' identity, which cancels.
    let mut chi2 = Sum::default();
    for (((&y, &e), &s), &g) in flux.iter().zip(flux_err).zip(shapes).zip(group) {
        let model = match g.and_then(|k| beta[k]) {
            Some(b) => alpha + b * s,
            None => alpha,
        };
        let r = (y - model) / e;
        chi2.add(r * r);
    }
    Ok(Grouped {
        alpha,
        var_alpha,
        beta,
        ratio,
        inverse,
        points,
        chi2: chi2.get(),
    })
}

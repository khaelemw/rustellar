use rustellar_core::LightCurve;

use crate::alias::Depth;
use crate::evidence::{Layout, layout};
use crate::fit::shapes;
use crate::grouped::{Grouped, solve};
use crate::{Candidate, Shape, VetError};

/// The candidate's transits in `n` groups, transit `e` in group `e mod n`,
/// each group's depth fitted with one shared baseline.
///
/// If the true period is `m` times the candidate's (`m` dividing `n`), only
/// every `m`-th group holds the transits and the rest are empty; for a
/// planet at the candidate's period all `n` are equal; an eclipsing binary
/// at twice the period alternates deep and shallow.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Multiple {
    /// The number of groups.
    pub n: u16,
    /// Each group's depth (`None` where it has no points in transit).
    pub groups: Vec<Option<Depth>>,
    /// How much better a depth per group fits than one depth for all: the
    /// drop in chi-squared, distributed as chi-squared with
    /// [`dof`](Self::dof) degrees of freedom when the groups share one
    /// depth.
    pub chi2: f64,
    /// The groups with points less one.
    pub dof: usize,
}

impl Multiple {
    /// The groups whose depth is more than `sigmas` standard errors above
    /// zero.
    #[must_use]
    pub fn significant(&self, sigmas: f64) -> Vec<u16> {
        (0..self.n)
            .filter(|&k| self.groups[usize::from(k)].is_some_and(|d| d.sigma() > sigmas))
            .collect()
    }

    /// The true period's multiple of the candidate's that the groups point
    /// to, and the group its first transit is in: the smallest `m` dividing
    /// [`n`](Self::n) for which the groups more than `sigmas` errors deep
    /// are exactly those of one class `r` mod `m`, among the groups with
    /// points. `(1, 0)` when every group with points is deep; `None` when
    /// none is, or no `m` fits.
    #[must_use]
    pub fn implied(&self, sigmas: f64) -> Option<(u16, u16)> {
        let deep = self.significant(sigmas);
        if deep.is_empty() {
            return None;
        }
        let with_points: Vec<u16> = (0..self.n)
            .filter(|&k| self.groups[usize::from(k)].is_some())
            .collect();
        (1..=self.n).filter(|m| self.n % m == 0).find_map(|m| {
            let r = deep[0] % m;
            let class: Vec<u16> = with_points.iter().copied().filter(|k| k % m == r).collect();
            (class == deep).then_some((m, r))
        })
    }
}

/// The candidate's transits split `n` ways for every `n` from 2 to
/// `up_to`, each group's depth fitted: the depths at phase `k / n` of `n`
/// times the period.
///
/// # Errors
///
/// As [`fit_depth`](crate::fit_depth).
///
/// # Examples
///
/// ```
/// use rustellar_vet::{Candidate, Epoch, LightCurve, Shape, TimeDelta, multiples};
///
/// // A 0.3% box dip every 6 days, searched at 3 days.
/// let time: Vec<f64> = (0..3456).map(|i| f64::from(i) / 144.0).collect();
/// let flux = time
///     .iter()
///     .map(|&t| {
///         let dip = ((t - 0.75 + 3.0).rem_euclid(6.0) - 3.0).abs() < 0.05;
///         1.0 - if dip { 3e-3 } else { 0.0 }
///     })
///     .collect();
/// let lc = LightCurve::from_days(time, flux, vec![1e-3; 3456])?;
/// let candidate = Candidate::new(
///     TimeDelta::from_days(3.0)?,
///     Epoch::from_days(0.75)?,
///     TimeDelta::from_days(0.1)?,
/// )?;
/// let split = multiples(&lc, &candidate, Shape::BOX, 4)?;
/// // In two groups and in four, every second group is empty: twice the
/// // period, from group 0.
/// assert_eq!(split[0].implied(3.0), Some((2, 0)));
/// assert_eq!(split[2].implied(3.0), Some((2, 0)));
/// // In three, all equal.
/// assert_eq!(split[1].implied(3.0), Some((1, 0)));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn multiples(
    lc: &LightCurve,
    candidate: &Candidate,
    shape: Shape,
    up_to: u16,
) -> Result<Vec<Multiple>, VetError> {
    let period = candidate.period().as_days();
    let duration = candidate.duration().as_days();
    let profile = shape.profile(period, duration)?;
    let s = shapes(lc, candidate, &profile);
    let times: Vec<f64> = lc.time().iter().map(|t| t.as_days()).collect();
    let Layout { epochs, .. } = layout(&times, candidate);
    let (flux, err) = (lc.flux(), lc.flux_err());
    let one = solve(flux, err, &s, &vec![Some(0); s.len()], 1)?;
    (2..=up_to)
        .map(|n| {
            let group: Vec<Option<usize>> = epochs
                .iter()
                .map(|&e| {
                    #[expect(clippy::cast_possible_truncation, reason = "a remainder below n")]
                    let k = e.rem_euclid(i64::from(n)) as usize;
                    Some(k)
                })
                .collect();
            let fit = solve(flux, err, &s, &group, usize::from(n))?;
            Ok(multiple(n, &fit, one.chi2))
        })
        .collect()
}

/// The groups' depths and the chi-squared test from a fit in `n` groups,
/// against one depth's chi-squared `chi2`.
fn multiple(n: u16, fit: &Grouped, chi2: f64) -> Multiple {
    let groups: Vec<Option<Depth>> = (0..usize::from(n))
        .map(|k| {
            Some(Depth {
                depth: fit.depth(k)?,
                depth_err: fit.cov_depth(k, k)?.sqrt(),
                points: fit.points[k],
            })
        })
        .collect();
    let with_points = groups.iter().filter(|g| g.is_some()).count();
    Multiple {
        n,
        groups,
        chi2: (chi2 - fit.chi2).max(0.0),
        dof: with_points - 1,
    }
}

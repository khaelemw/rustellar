//! One walker's slice-sampling update along a direction (Neal 2003, as the
//! ensemble slice sampler of Karamanis & Beutler 2021).

use crate::LogProbability;
use crate::rng::Rng;

/// The most expansions of a slice, both ends together (zeus's 10^4),
/// split at random between the ends so the update stays reversible (Neal
/// 2003, section 4.1).
const MAX_EXPANSIONS: usize = 10_000;

/// Where a slice update left the walker, and what it cost.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SliceUpdate {
    pub(crate) to: Vec<f64>,
    pub(crate) log_probability: f64,
    pub(crate) evaluations: usize,
    pub(crate) expansions: usize,
    pub(crate) contractions: usize,
}

/// The walker at `x` (log-probability `lp`) moved along `direction` by
/// slice sampling: a height under `lp`, an interval of unit width placed
/// at random around `x` and stepped out until both ends are below the
/// height, then shrunk towards `x` until a point above it is drawn.
///
/// Where the interval shrinks so far that the point drawn is `x` itself
/// (a height rounded to `lp`), the walker stays.
pub(crate) fn update<T: LogProbability + ?Sized>(
    target: &T,
    x: &[f64],
    lp: f64,
    direction: &[f64],
    rng: &mut Rng,
) -> SliceUpdate {
    let at = |t: f64| -> Vec<f64> {
        x.iter()
            .zip(direction)
            .map(|(&xi, &di)| t.mul_add(di, xi))
            .collect()
    };
    let log_p = |y: &[f64]| {
        let v = target.log_probability(y);
        if v.is_nan() { f64::NEG_INFINITY } else { v }
    };
    let mut evaluations = 0;
    let height = lp + rng.open_unit().ln();
    let mut left = -rng.unit();
    let mut right = left + 1.0;
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    let mut left_budget = (MAX_EXPANSIONS as f64 * rng.unit()) as usize;
    let mut right_budget = MAX_EXPANSIONS - 1 - left_budget;
    let mut expansions = 0;
    while left_budget > 0 && {
        evaluations += 1;
        log_p(&at(left)) > height
    } {
        left -= 1.0;
        left_budget -= 1;
        expansions += 1;
    }
    while right_budget > 0 && {
        evaluations += 1;
        log_p(&at(right)) > height
    } {
        right += 1.0;
        right_budget -= 1;
        expansions += 1;
    }
    let mut contractions = 0;
    loop {
        let t = rng.unit().mul_add(right - left, left);
        let y = at(t);
        if y == x {
            return SliceUpdate {
                to: y,
                log_probability: lp,
                evaluations,
                expansions,
                contractions,
            };
        }
        evaluations += 1;
        let lp_y = log_p(&y);
        if lp_y > height {
            return SliceUpdate {
                to: y,
                log_probability: lp_y,
                evaluations,
                expansions,
                contractions,
            };
        }
        if t < 0.0 {
            left = t;
        } else {
            right = t;
        }
        contractions += 1;
    }
}

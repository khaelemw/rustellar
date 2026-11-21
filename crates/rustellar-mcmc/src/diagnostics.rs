//! Convergence diagnostics: the integrated autocorrelation time (Sokal
//! 1997, as emcee), and the rank-normalised split-R-hat and the bulk and
//! tail effective sample sizes (Vehtari et al. 2021, as ArviZ).
//!
//! Each takes one parameter's draws as `chains`: one series per chain (for
//! an ensemble, per walker), all the same length.

use crate::fft::autocovariance_sums;
use crate::normal;

/// An integrated autocorrelation time and the window it was summed over.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct IntegratedTime {
    /// The estimate, in steps: `1 + 2 sum_{t=1}^{M} rho(t)`.
    pub tau: f64,
    /// The window `M`, the first lag at least `c` times the estimate up
    /// to it.
    pub window: usize,
    /// The steps in each chain.
    pub steps: usize,
}

impl IntegratedTime {
    /// Whether the chains are at least `tol` autocorrelation times long,
    /// so the estimate can be trusted (emcee asks for 50).
    #[must_use]
    pub fn trusted(&self, tol: f64) -> bool {
        #[allow(clippy::cast_precision_loss)]
        let steps = self.steps as f64;
        tol * self.tau <= steps
    }
}

/// The integrated autocorrelation time of `chains`, as emcee 3's
/// `integrated_time`: each chain's normalised autocorrelation function
/// (by FFT), averaged over the chains, summed with Sokal's automatic window
/// for the step `c` (emcee's default is 5).
///
/// NaN where a chain is constant, the window then the last lag.
///
/// # Panics
///
/// If the chains are empty, have no steps or differ in length.
///
/// # Examples
///
/// ```
/// use rustellar_mcmc::diagnostics::integrated_time;
///
/// // Independent draws: tau near 1.
/// let mut s = 1u64;
/// let mut next = || {
///     s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
///     (s >> 11) as f64 / (1u64 << 53) as f64
/// };
/// let chains: Vec<Vec<f64>> = (0..8).map(|_| (0..2000).map(|_| next()).collect()).collect();
/// let it = integrated_time(&chains, 5.0);
/// assert!((it.tau - 1.0).abs() < 0.2);
/// assert!(it.trusted(50.0));
/// ```
#[must_use]
pub fn integrated_time(chains: &[Vec<f64>], c: f64) -> IntegratedTime {
    let n = steps_of(chains);
    assert!(n > 0, "no steps");
    let mut f = vec![0.0; n];
    for chain in chains {
        let acf = autocovariance_sums(chain);
        let first = acf[0];
        for (fi, a) in f.iter_mut().zip(acf) {
            *fi += a / first;
        }
    }
    #[allow(clippy::cast_precision_loss)]
    let walkers = chains.len() as f64;
    let mut sum = 0.0;
    let taus: Vec<f64> = f
        .iter()
        .map(|&fi| {
            sum += fi / walkers;
            2.0f64.mul_add(sum, -1.0)
        })
        .collect();
    #[allow(clippy::cast_precision_loss)]
    let window = taus
        .iter()
        .enumerate()
        .position(|(m, &tau)| (m as f64) >= c * tau)
        .unwrap_or(n - 1);
    IntegratedTime {
        tau: taus[window],
        window,
        steps: n,
    }
}

/// The rank-normalised split-R-hat of `chains`, as ArviZ's `rhat` (method
/// `rank`): the larger of the bulk's (split chains, ranks to normal
/// scores) and the tail's (the same on the distance from the median).
/// Near 1 when the chains agree; above 1.01 they should run longer.
///
/// NaN with fewer than two chains or four draws each, or with any NaN.
///
/// # Panics
///
/// If the chains are empty or differ in length.
#[must_use]
pub fn rhat(chains: &[Vec<f64>]) -> f64 {
    if !valid(chains, 2) {
        return f64::NAN;
    }
    let split = split_chains(chains);
    let bulk = plain_rhat(&z_scale(&split));
    let median = median(&split);
    let folded: Vec<Vec<f64>> = split
        .iter()
        .map(|c| c.iter().map(|v| (v - median).abs()).collect())
        .collect();
    let tail = plain_rhat(&z_scale(&folded));
    bulk.max(tail)
}

/// The bulk effective sample size of `chains`, as ArviZ's `ess` (method
/// `bulk`): Geyer's initial monotone sequence on the split chains' normal
/// scores.
///
/// NaN with fewer than four draws, or with any NaN.
///
/// # Panics
///
/// If the chains are empty or differ in length.
#[must_use]
pub fn ess_bulk(chains: &[Vec<f64>]) -> f64 {
    if !valid(chains, 1) {
        return f64::NAN;
    }
    ess(&z_scale(&split_chains(chains)))
}

/// The tail effective sample size of `chains`, as ArviZ's `ess` (method
/// `tail`): the smaller of the effective sample sizes of the indicators
/// of the 5% and 95% quantiles.
///
/// NaN with fewer than four draws, or with any NaN.
///
/// # Panics
///
/// If the chains are empty or differ in length.
#[must_use]
pub fn ess_tail(chains: &[Vec<f64>]) -> f64 {
    if !valid(chains, 1) {
        return f64::NAN;
    }
    let mut all: Vec<f64> = chains.iter().flatten().copied().collect();
    all.sort_by(f64::total_cmp);
    let at = |prob: f64| {
        let q = quantile_sorted(&all, prob);
        let indicator: Vec<Vec<f64>> = chains
            .iter()
            .map(|c| c.iter().map(|&v| f64::from(u8::from(v <= q))).collect())
            .collect();
        ess(&split_chains(&indicator))
    };
    at(0.05).min(at(0.95))
}

/// The common length of the chains.
fn steps_of(chains: &[Vec<f64>]) -> usize {
    assert!(!chains.is_empty(), "no chains");
    let n = chains[0].len();
    assert!(
        chains.iter().all(|c| c.len() == n),
        "chains differ in length"
    );
    n
}

/// ArviZ's checks: at least `min_chains` chains of four draws, no NaN.
fn valid(chains: &[Vec<f64>], min_chains: usize) -> bool {
    let n = steps_of(chains);
    n >= 4 && chains.len() >= min_chains && chains.iter().flatten().all(|v| !v.is_nan())
}

/// Each chain's first and last halves as two chains (the middle draw of
/// an odd number left out).
fn split_chains(chains: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = chains[0].len();
    let half = n / 2;
    let mut out: Vec<Vec<f64>> = chains.iter().map(|c| c[..half].to_vec()).collect();
    out.extend(chains.iter().map(|c| c[n - half..].to_vec()));
    out
}

/// The median of split chains' values, as numpy's: split chains hold an
/// even number, so the mean of the middle two.
fn median(chains: &[Vec<f64>]) -> f64 {
    let mut all: Vec<f64> = chains.iter().flatten().copied().collect();
    all.sort_by(f64::total_cmp);
    let n = all.len();
    (all[n / 2 - 1] + all[n / 2]) / 2.0
}

/// numpy's `quantile` (method `linear`) of sorted values, with its
/// interpolation from whichever end is nearer.
#[allow(clippy::suboptimal_flops)]
fn quantile_sorted(sorted: &[f64], prob: f64) -> f64 {
    #[allow(clippy::cast_precision_loss)]
    let n = sorted.len() as f64;
    // numpy's order of operations, for the same rounding.
    let index = (n * prob + (1.0 - prob)) - 1.0;
    let lo = index.floor().clamp(0.0, n - 1.0);
    let t = index - lo;
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let i = lo as usize;
    let (a, b) = (sorted[i], sorted[(i + 1).min(sorted.len() - 1)]);
    let diff = b - a;
    if t >= 0.5 {
        b - diff * (1.0 - t)
    } else {
        a + diff * t
    }
}

/// Ranks to normal scores over every value, ties given their average rank,
/// with Blom's offset: `Phi^-1((rank - 3/8) / (n + 1/4))`.
fn z_scale(chains: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let all: Vec<f64> = chains.iter().flatten().copied().collect();
    let n = all.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| all[i].total_cmp(&all[j]));
    let mut rank = vec![0.0; n];
    let mut start = 0;
    while start < n {
        let mut end = start + 1;
        while end < n && all[order[end]] == all[order[start]] {
            end += 1;
        }
        // Ranks start + 1 to end, averaged.
        #[allow(clippy::cast_precision_loss)]
        let average = (start + 1 + end) as f64 / 2.0;
        for &i in &order[start..end] {
            rank[i] = average;
        }
        start = end;
    }
    #[allow(clippy::cast_precision_loss)]
    let size = n as f64;
    let z: Vec<f64> = rank
        .iter()
        .map(|&r| normal::quantile((r - 0.375) / (size + 0.25)))
        .collect();
    let m = chains[0].len();
    z.chunks_exact(m).map(<[f64]>::to_vec).collect()
}

fn mean(x: &[f64]) -> f64 {
    #[allow(clippy::cast_precision_loss)]
    let n = x.len() as f64;
    x.iter().sum::<f64>() / n
}

/// The variance over `n - 1`.
fn variance(x: &[f64]) -> f64 {
    let m = mean(x);
    #[allow(clippy::cast_precision_loss)]
    let n = x.len() as f64;
    x.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / (n - 1.0)
}

/// Gelman-Rubin's R-hat of equal-length chains, as ArviZ's `_rhat`.
fn plain_rhat(chains: &[Vec<f64>]) -> f64 {
    #[allow(clippy::cast_precision_loss)]
    let n = chains[0].len() as f64;
    let means: Vec<f64> = chains.iter().map(|c| mean(c)).collect();
    let within = mean(&chains.iter().map(|c| variance(c)).collect::<Vec<_>>());
    let between = n * variance(&means);
    ((between / within + n - 1.0) / n).sqrt()
}

/// The effective sample size of equal-length chains, as ArviZ's `_ess`:
/// Geyer's initial positive then initial monotone sequence of the
/// autocorrelations combined over the chains.
#[allow(clippy::cast_precision_loss)]
fn ess(chains: &[Vec<f64>]) -> f64 {
    let (lo, hi) = chains
        .iter()
        .flatten()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| {
            (a.min(v), b.max(v))
        });
    let n_chain = chains.len();
    let n_draw = chains[0].len();
    let size = (n_chain * n_draw) as f64;
    // numpy's float resolution, 1e-15.
    if hi - lo < 1e-15 {
        return size;
    }
    let draws = n_draw as f64;
    let acov: Vec<Vec<f64>> = chains
        .iter()
        .map(|c| autocovariance_sums(c).iter().map(|s| s / draws).collect())
        .collect();
    let lag_mean = |t: usize| acov.iter().map(|a| a[t]).sum::<f64>() / n_chain as f64;
    let mean_var = lag_mean(0) * draws / (draws - 1.0);
    // Split chains are at least two, so the between-chain term is always
    // there.
    let means: Vec<f64> = chains.iter().map(|c| mean(c)).collect();
    let var_plus = mean_var * (draws - 1.0) / draws + variance(&means);
    let rho_at = |t: usize| 1.0 - (mean_var - lag_mean(t)) / var_plus;
    let mut rho = vec![0.0; n_draw];
    let mut rho_even = 1.0;
    rho[0] = rho_even;
    let mut rho_odd = rho_at(1);
    rho[1] = rho_odd;
    // Initial positive sequence: pairs summed while positive.
    let mut t = 1;
    while t + 3 < n_draw && rho_even + rho_odd > 0.0 {
        rho_even = rho_at(t + 1);
        rho_odd = rho_at(t + 2);
        if rho_even + rho_odd >= 0.0 {
            rho[t + 1] = rho_even;
            rho[t + 2] = rho_odd;
        }
        t += 2;
    }
    // ArviZ's max_t is t - 2, at least -1; `last` is max_t + 1.
    let last = t - 1;
    if rho_even > 0.0 {
        rho[last] = rho_even;
    }
    // Initial monotone sequence: each pair no larger than the one before.
    let mut t = 1;
    while t + 3 <= last {
        if rho[t + 1] + rho[t + 2] > rho[t - 1] + rho[t] {
            rho[t + 1] = (rho[t - 1] + rho[t]) / 2.0;
            rho[t + 2] = rho[t + 1];
        }
        t += 2;
    }
    let tau = 2.0f64.mul_add(rho[..last].iter().sum::<f64>(), -1.0) + rho[last];
    size / tau.max(size.log10().recip())
}

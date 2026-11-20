//! The statistics of a comparison.

/// The median of `values` (at least one).
///
/// # Panics
///
/// If `values` is empty or holds a NaN.
#[must_use]
pub fn median(values: &[f64]) -> f64 {
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).expect("no NaN"));
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

/// The two-sided sign test's p-value for `wins` of `n` pairs: how often a
/// split at least this uneven comes from a fair coin.
#[must_use]
pub fn sign_test(wins: usize, n: usize) -> f64 {
    let k = wins.min(n - wins);
    // The binomial probabilities, built up term by term in f64: exact
    // enough for any count a comparison has.
    let mut term = 0.5_f64.powi(i32::try_from(n).expect("a count"));
    let mut tail = term;
    for i in 0..k {
        #[expect(clippy::cast_precision_loss, reason = "small counts")]
        let ratio = (n - i) as f64 / (i + 1) as f64;
        term *= ratio;
        tail += term;
    }
    (2.0 * tail).min(1.0)
}

/// The mean error, in minutes, of the mid-transit times a candidate of
/// `period` and `t0` predicts for each injected transit (`true_period`,
/// `true_t0`) up to `span` days.
#[must_use]
pub fn timing(true_period: f64, true_t0: f64, period: f64, t0: f64, span: f64) -> f64 {
    let (mut sum, mut count) = (0.0, 0.0);
    let mut n = (-true_t0 / true_period).ceil();
    while true_t0 + n * true_period < span {
        let truth = true_t0 + n * true_period;
        let predicted = t0 + ((truth - t0) / period).round() * period;
        sum += (predicted - truth).abs() * 1440.0;
        count += 1.0;
        n += 1.0;
    }
    sum / count
}

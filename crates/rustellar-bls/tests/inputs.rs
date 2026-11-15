// The checked inputs: each bad input gives its own error, and good inputs
// are kept exactly as given.

use rstest::rstest;
use rustellar_bls::{BlsError, Durations, Periods};
use rustellar_core::TimeDelta;

// ==========================
// Helper Functions
// ==========================

fn days(values: &[f64]) -> Vec<TimeDelta> {
    values
        .iter()
        .map(|&d| TimeDelta::from_days(d).unwrap())
        .collect()
}

// ==========================
// Durations
// ==========================

#[test]
fn durations_are_kept_in_order() {
    let durations = Durations::new(days(&[0.2, 0.05, 0.1, 0.1])).unwrap();
    assert_eq!(
        durations.as_slice(),
        days(&[0.2, 0.05, 0.1, 0.1]).as_slice()
    );
}

#[test]
fn durations_min_and_max() {
    let durations = Durations::from_days(&[0.2, 0.05, 0.3, 0.1]).unwrap();
    assert_eq!(durations.min().as_days(), 0.05);
    assert_eq!(durations.max().as_days(), 0.3);
}

#[test]
fn one_duration_is_its_own_min_and_max() {
    let durations = Durations::from_days(&[0.1]).unwrap();
    assert_eq!(durations.min().as_days(), 0.1);
    assert_eq!(durations.max().as_days(), 0.1);
}

#[test]
fn from_days_matches_new() {
    assert_eq!(
        Durations::from_days(&[0.05, 0.1]),
        Durations::new(days(&[0.05, 0.1]))
    );
}

#[test]
fn no_durations() {
    assert_eq!(Durations::new(vec![]), Err(BlsError::NoDurations));
    assert_eq!(Durations::from_days(&[]), Err(BlsError::NoDurations));
}

#[rstest]
#[case::zero(0.0)]
#[case::negative_zero(-0.0)]
#[case::negative(-0.1)]
fn duration_not_positive(#[case] bad: f64) {
    assert_eq!(
        Durations::from_days(&[0.1, bad, 0.2]),
        Err(BlsError::DurationNotPositive)
    );
}

#[rstest]
#[case::nan(f64::NAN)]
#[case::infinite(f64::INFINITY)]
#[case::negative_infinite(f64::NEG_INFINITY)]
fn duration_not_finite(#[case] bad: f64) {
    assert_eq!(
        Durations::from_days(&[0.1, bad]),
        Err(BlsError::NonFinite { name: "duration" })
    );
}

// A NaN is reported before a negative value that comes after it: from_days
// converts every value before checking signs.
#[test]
fn non_finite_is_found_before_sign() {
    assert_eq!(
        Durations::from_days(&[f64::NAN, -1.0]),
        Err(BlsError::NonFinite { name: "duration" })
    );
}

// ==========================
// Periods
// ==========================

#[test]
fn periods_are_kept_in_order() {
    let periods = Periods::new(days(&[3.0, 0.5, 10.0])).unwrap();
    assert_eq!(periods.as_slice(), days(&[3.0, 0.5, 10.0]).as_slice());
}

#[test]
fn periods_min() {
    assert_eq!(
        Periods::from_days(&[3.0, 0.5, 10.0])
            .unwrap()
            .min()
            .as_days(),
        0.5
    );
}

#[test]
fn no_periods() {
    assert_eq!(Periods::new(vec![]), Err(BlsError::NoPeriods));
    assert_eq!(Periods::from_days(&[]), Err(BlsError::NoPeriods));
}

#[rstest]
#[case::zero(0.0)]
#[case::negative(-3.0)]
fn period_not_positive(#[case] bad: f64) {
    assert_eq!(
        Periods::from_days(&[1.0, bad]),
        Err(BlsError::PeriodNotPositive)
    );
}

#[test]
fn period_not_finite() {
    assert_eq!(
        Periods::from_days(&[f64::NAN]),
        Err(BlsError::NonFinite { name: "period" })
    );
}

// ==========================
// Messages
// ==========================

#[rstest]
#[case(BlsError::NonFinite { name: "period" }, "period must be finite")]
#[case(BlsError::NoDurations, "at least one duration is needed")]
#[case(BlsError::DurationNotPositive, "durations must be greater than zero")]
#[case(BlsError::NoPeriods, "at least one period is needed")]
#[case(BlsError::PeriodNotPositive, "periods must be greater than zero")]
#[case(
    BlsError::DurationNotShorterThanPeriod,
    "the longest duration must be shorter than the shortest period"
)]
#[case(BlsError::NoBaseline, "the light curve must span more than one time")]
#[case(BlsError::TooFewTransits, "minimum_n_transit must be at least 2")]
#[case(
    BlsError::FrequencyFactorNotPositive,
    "frequency_factor must be greater than zero"
)]
#[case(
    BlsError::PeriodRangeEmpty,
    "the minimum period must not be above the maximum period"
)]
#[case(
    BlsError::GridTooCoarse,
    "frequency_factor is too large for the period range: the grid reaches zero frequency"
)]
#[case(
    BlsError::NotPositive { name: "stellar_mass" },
    "stellar_mass must be greater than zero"
)]
#[case(BlsError::NoPointsInTransit, "the box holds none of the points")]
#[case(
    BlsError::NoPointsOutOfTransit,
    "the box holds every point, leaving none out of transit"
)]
#[case(
    BlsError::OutOfRange,
    "the weights 1 / flux_err^2, or sums built from them, are out of range for an f64"
)]
#[case(
    BlsError::GridTooLarge,
    "the period grid would hold more than 100 million periods"
)]
fn error_messages(#[case] error: BlsError, #[case] message: &str) {
    assert_eq!(error.to_string(), message);
}

#[test]
fn errors_work_with_question_mark() {
    fn build() -> Result<Periods, Box<dyn std::error::Error>> {
        Ok(Periods::from_days(&[-1.0])?)
    }
    assert_eq!(
        build().unwrap_err().to_string(),
        "periods must be greater than zero"
    );
}

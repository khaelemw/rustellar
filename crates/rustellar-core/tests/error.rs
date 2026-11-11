use rstest::rstest;
use rustellar_core::{LightCurveError, TimeError};

#[rstest]
#[case::empty(LightCurveError::Empty, "light curve has no points")]
#[case::length_mismatch(
    LightCurveError::LengthMismatch,
    "time, flux, and flux_err must have the same length"
)]
#[case::not_sorted(LightCurveError::NotSorted, "time values must be strictly increasing")]
#[case::non_finite(
    LightCurveError::NonFinite,
    "time, flux, and flux_err must all be finite"
)]
#[case::non_positive_error(
    LightCurveError::NonPositiveError,
    "every flux_err value must be greater than zero"
)]
#[case::invalid_period(
    LightCurveError::InvalidPeriod,
    "fold period must be greater than zero"
)]
fn lightcurve_error_display_message(#[case] error: LightCurveError, #[case] expected: &str) {
    assert_eq!(error.to_string(), expected);
}

#[rstest]
#[case::non_finite(TimeError::NonFinite, "time value must be finite")]
fn time_error_display_message(#[case] error: TimeError, #[case] expected: &str) {
    assert_eq!(error.to_string(), expected);
}

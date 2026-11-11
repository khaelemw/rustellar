use rstest::rstest;
use rustellar_core::{Duration, Epoch, TimeError};

#[rstest]
#[case::positive_days(10.0, 240.0)]
#[case::fractional_days(5.7, 136.8)]
#[case::negative_days(-5.0, -120.0)]
fn duration_accessors_return_correct_value(#[case] days: f64, #[case] expected_hours: f64) {
    let duration = Duration::from_days(days).unwrap();
    assert_eq!(duration.as_days(), days);
    assert_eq!(duration.as_hours(), expected_hours);
}

#[rstest]
fn duration_constructors_reject_non_finite(
    #[values(Duration::from_days, Duration::from_hours)] constructor_function: fn(
        f64,
    ) -> Result<
        Duration,
        TimeError,
    >,
    #[values(f64::NAN, f64::INFINITY, f64::NEG_INFINITY)] value: f64,
) {
    assert_eq!(constructor_function(value), Err(TimeError::NonFinite));
}

#[rstest]
#[case::zero_btjd(0.0, 2_457_000.0)]
#[case::positive_btjd(10.0, 2_457_010.0)]
#[case::fractional_btjd(5.7, 2_457_005.7)]
#[case::negative_btjd(-5.0, 2_456_995.0)]
#[case::typical_tess_btjd(3000.123456, 2_460_000.123456)]
fn epoch_accessors_return_correct_value(#[case] btjd: f64, #[case] expected_bjd: f64) {
    let epoch = Epoch::from_btjd(btjd).unwrap();
    assert_eq!(epoch.as_btjd(), btjd);
    assert_eq!(epoch.as_bjd(), expected_bjd);
}

#[rstest]
fn epoch_constructors_reject_non_finite(
    #[values(Epoch::from_btjd, Epoch::from_bjd)] constructor_function: fn(
        f64,
    ) -> Result<
        Epoch,
        TimeError,
    >,
    #[values(f64::NAN, f64::INFINITY, f64::NEG_INFINITY)] value: f64,
) {
    assert_eq!(constructor_function(value), Err(TimeError::NonFinite));
}

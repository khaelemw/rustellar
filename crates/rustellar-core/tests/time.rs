use rstest::rstest;
use rustellar_core::{Epoch, TimeDelta, TimeError};

// ==========================
// Helper Functions
// ==========================

fn days(x: f64) -> TimeDelta {
    TimeDelta::from_days(x).unwrap()
}

fn at(x: f64) -> Epoch {
    Epoch::from_days(x).unwrap()
}

// ==========================
// Tests
// ==========================

#[rstest]
#[case::positive_days(10.0, 240.0)]
#[case::fractional_days(5.7, 136.8)]
#[case::negative_days(-5.0, -120.0)]
fn time_delta_accessors_return_correct_value(#[case] days: f64, #[case] expected_hours: f64) {
    let duration = TimeDelta::from_days(days).unwrap();
    assert_eq!(duration.as_days(), days);
    assert_eq!(duration.as_hours(), expected_hours);
}

#[rstest]
fn time_delta_constructors_reject_non_finite(
    #[values(TimeDelta::from_days, TimeDelta::from_hours)] constructor_function: fn(
        f64,
    ) -> Result<
        TimeDelta,
        TimeError,
    >,
    #[values(f64::NAN, f64::INFINITY, f64::NEG_INFINITY)] value: f64,
) {
    assert_eq!(constructor_function(value), Err(TimeError::NonFinite));
}

#[test]
fn time_delta_hours_round_trip() {
    for i in -1000..=1000 {
        let hours = f64::from(i) * 0.37;
        let back = TimeDelta::from_hours(hours).unwrap().as_hours();

        assert!(
            (back - hours).abs() <= 1e-12 * hours.abs(),
            "{hours} hours came back as {back}"
        );
    }
}

#[rstest]
#[case::zero(0.0)]
#[case::fractional(5.7)]
#[case::negative(-5.0)]
#[case::offset_scale(3000.123456)]
#[case::full_bjd(2_460_000.123456)]
fn epoch_keeps_its_days(#[case] value: f64) {
    assert_eq!(Epoch::from_days(value).unwrap().as_days(), value);
}

#[rstest]
fn epoch_rejects_non_finite(#[values(f64::NAN, f64::INFINITY, f64::NEG_INFINITY)] value: f64) {
    assert_eq!(Epoch::from_days(value), Err(TimeError::NonFinite));
}

#[rstest]
#[case::time_delta_add(|a, b| (days(a) + days(b)).as_days(), 1.0, 0.5, 1.5)]
#[case::time_delta_sub(|a, b| (days(a) - days(b)).as_days(), 0.5, 1.0, -0.5)]
#[case::time_delta_div(|a, b| days(a) / days(b), 3.0, 1.5, 2.0)]
#[case::time_delta_mul(|a, b| (days(a) * b).as_days(), 2.0, 4.0, 8.0)]
#[case::time_delta_neg(|a, _| (-days(a)).as_days(), 2.0, 0.0, -2.0)]
#[case::epoch_minus_epoch(|a, b| (at(a) -  at(b)).as_days(), 10.0, 4.0, 6.0)]
#[case::epoch_plus_time_delta(|a, b| (at(a) + days(b)).as_days(), 3000.0, 0.25, 3000.25)]
#[case::epoch_minus_time_delta(|a, b| (at(a) - days(b)).as_days(), 3000.0, 0.25, 2999.75)]
fn time_ops_give_expected_value(
    #[case] op: fn(f64, f64) -> f64,
    #[case] a: f64,
    #[case] b: f64,
    #[case] expected: f64,
) {
    assert_eq!(op(a, b), expected);
}

#[test]
fn epoch_and_time_delta_ops_undo_each_other() {
    let e = at(3000.5);
    let d = days(0.25);

    assert_eq!((e + d) - e, d);
    assert_eq!((e - d) + d, e);
    assert_eq!(e - (e + d), -d);
}

#[test]
fn days_and_hours_combine() {
    let total = days(1.0) + TimeDelta::from_hours(12.0).unwrap();

    assert_eq!(total.as_days(), 1.5);
    assert_eq!(total.as_hours(), 36.0);
}

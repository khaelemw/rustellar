use rstest::rstest;
use rustellar_core::{Epoch, TimeDelta, TimeError};

// ==========================
// Helper Functions
// ==========================

fn days(x: f64) -> TimeDelta {
    TimeDelta::from_days(x).unwrap()
}

fn btjd(x: f64) -> Epoch {
    Epoch::from_btjd(x).unwrap()
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

#[rstest]
#[case::typical_tess_bjd(2_460_000.123456, 3000.123456)]
#[case::tess_reference_bjd(2_457_000.0, 0.0)]
#[case::before_tess_bjd(2_455_765.25, -1234.75)]
fn epoch_from_bjd_round_trips(#[case] bjd: f64, #[case] expected_btjd: f64) {
    let epoch = Epoch::from_bjd(bjd).unwrap();

    assert!(
        (epoch.as_btjd() - expected_btjd).abs() < 1e-9,
        "got {} BTJD, expected {expected_btjd}",
        epoch.as_btjd()
    );
    assert_eq!(epoch.as_bjd(), bjd);
}

#[rstest]
#[case::time_delta_add(|a, b| (days(a) + days(b)).as_days(), 1.0, 0.5, 1.5)]
#[case::time_delta_sub(|a, b| (days(a) - days(b)).as_days(), 0.5, 1.0, -0.5)]
#[case::time_delta_div(|a, b| days(a) / days(b), 3.0, 1.5, 2.0)]
#[case::time_delta_neg(|a, _| (-days(a)).as_days(), 2.0, 0.0, -2.0)]
#[case::epoch_minus_epoch(|a, b| (btjd(a) - btjd(b)).as_days(), 10.0, 4.0, 6.0)]
#[case::epoch_plus_time_delta(|a, b| (btjd(a) + days(b)).as_btjd(), 3000.0, 0.25, 3000.25)]
#[case::epoch_minus_time_delta(|a, b| (btjd(a) - days(b)).as_btjd(), 3000.0, 0.25, 2999.75)]
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
    let e = btjd(3000.5);
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

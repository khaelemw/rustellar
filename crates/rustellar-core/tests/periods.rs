// The checked period list: each bad input gives its own error, and good
// periods are kept exactly as given.

use rstest::rstest;
use rustellar_core::{GridError, Periods, TimeDelta};

fn days(values: &[f64]) -> Vec<TimeDelta> {
    values
        .iter()
        .map(|&d| TimeDelta::from_days(d).unwrap())
        .collect()
}

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
    assert_eq!(Periods::new(vec![]), Err(GridError::NoPeriods));
    assert_eq!(Periods::from_days(&[]), Err(GridError::NoPeriods));
}

#[rstest]
#[case::zero(0.0)]
#[case::negative(-3.0)]
fn period_not_positive(#[case] bad: f64) {
    assert_eq!(
        Periods::from_days(&[1.0, bad]),
        Err(GridError::PeriodNotPositive)
    );
}

#[test]
fn period_not_finite() {
    assert_eq!(
        Periods::from_days(&[f64::NAN]),
        Err(GridError::NonFinite { name: "period" })
    );
}

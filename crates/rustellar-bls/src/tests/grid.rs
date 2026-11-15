use rstest::rstest;

use crate::{BlsError, Durations, Periods, grid::check_durations_fit};

fn periods(days: &[f64]) -> Periods {
    Periods::from_days(days).unwrap()
}

fn durations(days: &[f64]) -> Durations {
    Durations::from_days(days).unwrap()
}

#[rstest]
#[case::one_each(&[1.0], &[0.1])]
#[case::longest_just_shorter(&[1.0, 5.0], &[0.1, 0.999_999])]
// Order does not matter: the shortest period and longest duration are found.
#[case::unsorted(&[5.0, 0.5, 2.0], &[0.3, 0.05, 0.4])]
fn boxes_that_fit(#[case] p: &[f64], #[case] d: &[f64]) {
    assert_eq!(check_durations_fit(&periods(p), &durations(d)), Ok(()));
}

#[rstest]
#[case::equal(&[1.0], &[1.0])]
#[case::longer(&[1.0], &[1.5])]
// Only the shortest period and the longest duration decide it.
#[case::one_pair_fails(&[0.5, 10.0], &[0.1, 0.6])]
fn boxes_that_do_not_fit(#[case] p: &[f64], #[case] d: &[f64]) {
    assert_eq!(
        check_durations_fit(&periods(p), &durations(d)),
        Err(BlsError::DurationNotShorterThanPeriod)
    );
}

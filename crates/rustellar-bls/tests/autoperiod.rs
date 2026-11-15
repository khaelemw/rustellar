// astropy's automatic period grid: the same periods to the last bit, on
// reference grids from astropy itself, and an error wherever astropy quietly
// carries on with a bad input.

use rstest::rstest;
use rustellar_bls::{AutoPeriod, BlsError, Durations, Periods};
use rustellar_core::{LightCurve, TimeDelta};

// ==========================
// Helper Functions
// ==========================

const CASES: &str = include_str!("reference/autoperiod_cases.csv");
const GRIDS: &str = include_str!("reference/autoperiod_grids.csv");

struct Case {
    t_min: f64,
    t_max: f64,
    durations: Durations,
    settings: AutoPeriod,
    count: usize,
}

fn rows(csv: &str) -> impl Iterator<Item = Vec<&str>> {
    csv.lines()
        .filter(|line| !line.starts_with('#'))
        .skip(1)
        .map(|line| line.split(',').collect())
}

fn optional_days(field: &str) -> Option<TimeDelta> {
    (!field.is_empty()).then(|| TimeDelta::from_days(field.parse().unwrap()).unwrap())
}

fn cases() -> Vec<Case> {
    rows(CASES)
        .map(|v| {
            assert_eq!(v.len(), 9, "{v:?}");
            let durations: Vec<f64> = v[3].split(';').map(|d| d.parse().unwrap()).collect();
            Case {
                t_min: v[1].parse().unwrap(),
                t_max: v[2].parse().unwrap(),
                durations: Durations::from_days(&durations).unwrap(),
                settings: AutoPeriod {
                    minimum_period: optional_days(v[4]),
                    maximum_period: optional_days(v[5]),
                    minimum_n_transit: v[6].parse().unwrap(),
                    frequency_factor: v[7].parse().unwrap(),
                },
                count: v[8].parse().unwrap(),
            }
        })
        .collect()
}

// (case, index, period) for the sampled periods.
fn sampled_periods() -> Vec<(usize, usize, f64)> {
    rows(GRIDS)
        .map(|v| {
            (
                v[0].parse().unwrap(),
                v[1].parse().unwrap(),
                v[2].parse().unwrap(),
            )
        })
        .collect()
}

// The grid depends only on the first and last times, as in astropy; the
// middle point is the one the reference generator used.
fn light_curve(t_min: f64, t_max: f64) -> LightCurve {
    let time = vec![t_min, 0.5 * (t_min + t_max), t_max];
    LightCurve::from_btjd(time, vec![1.0; 3], vec![1e-3; 3]).unwrap()
}

fn grid(case: &Case) -> Periods {
    case.settings
        .periods(&light_curve(case.t_min, case.t_max), &case.durations)
        .unwrap()
}

fn one_day_light_curve() -> LightCurve {
    light_curve(0.0, 1.0)
}

fn durations(days: &[f64]) -> Durations {
    Durations::from_days(days).unwrap()
}

fn days(value: f64) -> Option<TimeDelta> {
    Some(TimeDelta::from_days(value).unwrap())
}

// ==========================
// Agreement with astropy
// ==========================

#[test]
fn reference_files_are_complete() {
    let cases = cases();
    assert_eq!(cases.len(), 13);
    let sampled = sampled_periods();
    assert_eq!(sampled.len(), 11_938);
    // Every case has its first and last period sampled.
    for (i, case) in cases.iter().enumerate() {
        for index in [0, case.count - 1] {
            assert!(
                sampled.iter().any(|&(c, j, _)| c == i && j == index),
                "case {i}, index {index}"
            );
        }
    }
}

#[test]
fn same_number_of_periods_as_astropy() {
    for (i, case) in cases().iter().enumerate() {
        assert_eq!(grid(case).as_slice().len(), case.count, "case {i}");
    }
}

#[test]
fn same_periods_as_astropy_to_the_last_bit() {
    let grids: Vec<Periods> = cases().iter().map(grid).collect();
    for (case, index, period) in sampled_periods() {
        let ours = grids[case].as_slice()[index].as_days();
        assert_eq!(
            ours.to_bits(),
            period.to_bits(),
            "case {case}, index {index}: {ours} vs {period}"
        );
    }
}

// 1,000 even times over 27 days; the values are astropy's.
#[test]
fn worked_example() {
    let lc = LightCurve::from_btjd(
        (0..1000).map(|i| 27.0 * f64::from(i) / 999.0).collect(),
        vec![1.0; 1000],
        vec![1e-3; 1000],
    )
    .unwrap();
    let periods = AutoPeriod::default()
        .periods(&lc, &durations(&[0.05, 0.1, 0.2]))
        .unwrap();
    let periods = periods.as_slice();
    assert_eq!(periods.len(), 35_371);
    assert_eq!(periods[0].as_days(), 0.4);
    assert_eq!(periods[1].as_days(), 0.400_010_974_237_976_4);
    assert_eq!(periods[35_370].as_days(), 13.500_000_000_000_021);
}

// (4 - 0.25) / 1.5 = 2.5 steps exactly. numpy rounds the tie to the even 2,
// giving 3 periods; rounding away from zero would give 4.
#[test]
fn rounding_ties_go_to_even() {
    let settings = AutoPeriod {
        minimum_period: days(0.25),
        maximum_period: days(4.0),
        frequency_factor: 6.0,
        ..AutoPeriod::default()
    };
    let periods = settings
        .periods(&one_day_light_curve(), &durations(&[0.25]))
        .unwrap();
    let periods: Vec<f64> = periods.as_slice().iter().map(|p| p.as_days()).collect();
    assert_eq!(periods, [0.25, 0.4, 1.0]);
}

#[test]
fn shortest_period_first_and_increasing() {
    for case in cases() {
        let periods = grid(&case);
        assert!(
            periods.as_slice().windows(2).all(|pair| pair[0] < pair[1]),
            "not increasing"
        );
        assert_eq!(periods.min(), periods.as_slice()[0]);
    }
}

#[test]
fn defaults_are_astropys() {
    assert_eq!(
        AutoPeriod::default(),
        AutoPeriod {
            minimum_period: None,
            maximum_period: None,
            minimum_n_transit: 3,
            frequency_factor: 1.0,
        }
    );
}

// Unset, the range runs from twice the longest duration to
// baseline / (minimum_n_transit - 1).
#[test]
fn default_range() {
    let lc = light_curve(1325.0, 1345.0);
    let periods = AutoPeriod::default()
        .periods(&lc, &durations(&[0.1, 0.25]))
        .unwrap();
    let periods = periods.as_slice();
    assert_eq!(periods[0].as_days(), 0.5);
    let last = periods[periods.len() - 1].as_days();
    assert!((last - 10.0).abs() < 1e-3, "{last}");
}

// ==========================
// Errors where astropy carries on
// ==========================

#[rstest]
#[case(0)]
#[case(1)]
fn too_few_transits(#[case] minimum_n_transit: u32) {
    let settings = AutoPeriod {
        minimum_n_transit,
        ..AutoPeriod::default()
    };
    assert_eq!(
        settings.periods(&one_day_light_curve(), &durations(&[0.1])),
        Err(BlsError::TooFewTransits)
    );
}

#[rstest]
#[case::nan(f64::NAN, BlsError::NonFinite { name: "frequency_factor" })]
#[case::infinite(f64::INFINITY, BlsError::NonFinite { name: "frequency_factor" })]
#[case::zero(0.0, BlsError::FrequencyFactorNotPositive)]
#[case::negative(-1.0, BlsError::FrequencyFactorNotPositive)]
fn bad_frequency_factor(#[case] frequency_factor: f64, #[case] error: BlsError) {
    let settings = AutoPeriod {
        frequency_factor,
        ..AutoPeriod::default()
    };
    assert_eq!(
        settings.periods(&one_day_light_curve(), &durations(&[0.1])),
        Err(error)
    );
}

#[rstest]
#[case::minimum_zero(days(0.0), None)]
#[case::minimum_negative(days(-1.0), None)]
#[case::maximum_zero(None, days(0.0))]
#[case::maximum_negative(None, days(-2.0))]
fn set_period_not_positive(
    #[case] minimum_period: Option<TimeDelta>,
    #[case] maximum_period: Option<TimeDelta>,
) {
    let settings = AutoPeriod {
        minimum_period,
        maximum_period,
        ..AutoPeriod::default()
    };
    assert_eq!(
        settings.periods(&one_day_light_curve(), &durations(&[0.1])),
        Err(BlsError::PeriodNotPositive)
    );
}

#[test]
fn single_time_has_no_baseline() {
    let lc = LightCurve::from_btjd(vec![1325.0], vec![1.0], vec![1e-3]).unwrap();
    assert_eq!(
        AutoPeriod::default().periods(&lc, &durations(&[0.1])),
        Err(BlsError::NoBaseline)
    );
}

// astropy swaps a reversed range; here it is an error, whether the range was
// set or follows from the defaults (a one-day light curve cannot show three
// transits of a planet whose period must be at least 1.2 days).
#[rstest]
#[case::set(days(5.0), days(2.0), &[0.1])]
#[case::from_defaults(None, None, &[0.6])]
fn reversed_range(
    #[case] minimum_period: Option<TimeDelta>,
    #[case] maximum_period: Option<TimeDelta>,
    #[case] duration_days: &[f64],
) {
    let settings = AutoPeriod {
        minimum_period,
        maximum_period,
        ..AutoPeriod::default()
    };
    assert_eq!(
        settings.periods(&one_day_light_curve(), &durations(duration_days)),
        Err(BlsError::PeriodRangeEmpty)
    );
}

// Frequencies 1 to 0.25 in steps of 1: one step reaches frequency 0, where
// astropy would return an infinite period.
#[test]
fn grid_too_coarse() {
    let settings = AutoPeriod {
        minimum_period: days(1.0),
        maximum_period: days(4.0),
        frequency_factor: 4.0,
        ..AutoPeriod::default()
    };
    assert_eq!(
        settings.periods(&one_day_light_curve(), &durations(&[0.25])),
        Err(BlsError::GridTooCoarse)
    );
}

// A range of a single period gives just that period.
#[test]
fn equal_minimum_and_maximum() {
    let settings = AutoPeriod {
        minimum_period: days(0.7),
        maximum_period: days(0.7),
        ..AutoPeriod::default()
    };
    let periods = settings
        .periods(&one_day_light_curve(), &durations(&[0.1]))
        .unwrap();
    assert_eq!(periods.as_slice(), [TimeDelta::from_days(0.7).unwrap()]);
}

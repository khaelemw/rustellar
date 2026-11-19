// Ofir's (2014) optimal period grid: the same periods as the reference grids
// (to rounding), the step the paper asks for, far fewer periods than astropy's
// grid, and an error wherever TLS quietly changes a setting.

use std::f64::consts::{PI, TAU};

use rstest::rstest;
use rustellar_core::{GridError, LightCurve, OfirPeriod, Periods, TimeDelta};

// ==========================
// Helper Functions
// ==========================

const CASES: &str = include_str!("reference/ofir_cases.csv");
const GRIDS: &str = include_str!("reference/ofir_grids.csv");

// IAU 2015 nominal values, for checks that work the grid out again.
const GM_SUN: f64 = 1.327_124_4e20;
const R_SUN: f64 = 6.957e8;
const DAY: f64 = 86_400.0;

struct Case {
    t_min: f64,
    t_max: f64,
    settings: OfirPeriod,
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
            assert_eq!(v.len(), 10, "{v:?}");
            Case {
                t_min: v[1].parse().unwrap(),
                t_max: v[2].parse().unwrap(),
                settings: OfirPeriod {
                    stellar_radius: v[3].parse().unwrap(),
                    stellar_mass: v[4].parse().unwrap(),
                    minimum_period: optional_days(v[5]),
                    maximum_period: optional_days(v[6]),
                    oversampling_factor: v[7].parse().unwrap(),
                    minimum_n_transit: v[8].parse().unwrap(),
                },
                count: v[9].parse().unwrap(),
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

// The grid depends only on the first and last times.
fn light_curve(t_min: f64, t_max: f64) -> LightCurve {
    let time = vec![t_min, 0.5 * (t_min + t_max), t_max];
    LightCurve::from_btjd(time, vec![1.0; 3], vec![1e-3; 3]).unwrap()
}

fn sector() -> LightCurve {
    light_curve(1325.293, 1353.177)
}

fn grid(case: &Case) -> Periods {
    case.settings
        .periods(&light_curve(case.t_min, case.t_max))
        .unwrap()
}

fn days(value: f64) -> Option<TimeDelta> {
    Some(TimeDelta::from_days(value).unwrap())
}

fn as_days(periods: &Periods) -> Vec<f64> {
    periods.as_slice().iter().map(|p| p.as_days()).collect()
}

// ==========================
// Agreement with the reference
// ==========================

#[test]
fn reference_files_are_complete() {
    let cases = cases();
    assert_eq!(cases.len(), 11);
    let sampled = sampled_periods();
    assert_eq!(sampled.len(), 10_278);
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
fn same_number_of_periods_as_the_reference() {
    for (i, case) in cases().iter().enumerate() {
        assert_eq!(grid(case).as_slice().len(), case.count, "case {i}");
    }
}

// Not to the last bit: `cbrt` comes from each platform's maths library, and
// IEEE 754 does not require it to be correctly rounded. A cube root one
// rounding off moves A by that much, and the cube in f(x) triples it: the
// reference (numpy on Windows) and Linux and macOS differ by up to 7 roundings
// (1.6e-15). 1e-14 leaves room for other libraries and nothing more.
#[test]
fn same_periods_as_the_reference() {
    let grids: Vec<Periods> = cases().iter().map(grid).collect();
    for (case, index, period) in sampled_periods() {
        let ours = grids[case].as_slice()[index].as_days();
        let error = (ours / period - 1.0).abs();
        assert!(
            error < 1e-14,
            "case {case}, index {index}: {ours} vs {period}"
        );
    }
}

// ==========================
// What the paper asks for
// ==========================

// Eq. 5: each step in frequency is A f^(2/3), so a transit's phase drifts by
// a third of its duration from one period to the next.
#[test]
fn step_follows_the_duty_cycle() {
    let baseline = (1353.177 - 1325.293) * DAY;
    let a = (2.0 * PI).powf(2.0 / 3.0) / PI * R_SUN / GM_SUN.powf(1.0 / 3.0) / (baseline * 3.0);
    let frequencies: Vec<f64> = as_days(&OfirPeriod::default().periods(&sector()).unwrap())
        .iter()
        .map(|p| 1.0 / (p * DAY))
        .collect();
    for pair in frequencies.windows(2) {
        let step = pair[0] - pair[1];
        // Taken at the step's middle in f^(1/3), the grid's own variable, the
        // finite step differs from eq. 5 by a factor 1 + (A/3)^2 / (12 f^(2/3)),
        // at most 1 + 9e-7 here, at the longest periods.
        let middle = (0.5 * (pair[0].cbrt() + pair[1].cbrt())).powi(3);
        let expected = a * middle.powf(2.0 / 3.0);
        assert!((step / expected - 1.0).abs() < 1e-6, "{step} vs {expected}");
    }
}

// Unset, the grid runs from the Roche limit, a = 3 R, to baseline / 2.
#[test]
fn default_range() {
    let periods = as_days(&OfirPeriod::default().periods(&sector()).unwrap());
    let roche = TAU * ((3.0 * R_SUN).powi(3) / GM_SUN).sqrt() / DAY;
    let shortest = periods[0];
    assert!(
        shortest >= roche,
        "{shortest} below the Roche limit {roche}"
    );
    assert!(shortest / roche - 1.0 < 1e-3, "{shortest} vs {roche}");
    let longest = periods[periods.len() - 1];
    let half_baseline = (1353.177 - 1325.293) / 2.0;
    assert!((longest / half_baseline - 1.0).abs() < 1e-12, "{longest}");
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

// A set range is the grid's edges, not a cut of the default grid.
#[test]
fn set_range_moves_the_edges() {
    let settings = OfirPeriod {
        minimum_period: days(0.5),
        maximum_period: days(20.0),
        ..OfirPeriod::default()
    };
    let periods = as_days(&settings.periods(&sector()).unwrap());
    assert!(
        periods[0] >= 0.5 && periods[0] - 0.5 < 1e-3,
        "{}",
        periods[0]
    );
    let longest = periods[periods.len() - 1];
    assert!((longest - 20.0).abs() < 1e-12, "{longest}");
}

#[test]
fn defaults_are_ofirs() {
    assert_eq!(
        OfirPeriod::default(),
        OfirPeriod {
            stellar_radius: 1.0,
            stellar_mass: 1.0,
            minimum_period: None,
            maximum_period: None,
            oversampling_factor: 3.0,
            minimum_n_transit: 2,
        }
    );
}

// A range of a single period gives just that period.
#[test]
fn equal_minimum_and_maximum() {
    let settings = OfirPeriod {
        minimum_period: days(0.7),
        maximum_period: days(0.7),
        ..OfirPeriod::default()
    };
    let periods = as_days(&settings.periods(&sector()).unwrap());
    assert_eq!(periods.len(), 1);
    assert!((periods[0] - 0.7).abs() < 1e-15, "{}", periods[0]);
}

// ==========================
// Errors where TLS carries on
// ==========================

#[rstest]
#[case(0)]
#[case(1)]
fn too_few_transits(#[case] minimum_n_transit: u32) {
    let settings = OfirPeriod {
        minimum_n_transit,
        ..OfirPeriod::default()
    };
    assert_eq!(settings.periods(&sector()), Err(GridError::TooFewTransits));
}

fn with(name: &str, value: f64) -> OfirPeriod {
    let mut settings = OfirPeriod::default();
    match name {
        "stellar_radius" => settings.stellar_radius = value,
        "stellar_mass" => settings.stellar_mass = value,
        _ => settings.oversampling_factor = value,
    }
    settings
}

#[rstest]
fn bad_setting(
    #[values("stellar_radius", "stellar_mass", "oversampling_factor")] name: &'static str,
    #[values(f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0, -1.0)] value: f64,
) {
    let error = if value.is_finite() {
        GridError::NotPositive { name }
    } else {
        GridError::NonFinite { name }
    };
    assert_eq!(with(name, value).periods(&sector()), Err(error));
}

// TLS clamps a radius below 0.01 to 0.1 and a mass below 0.01 to 0.01; here
// a small star gives its own grid.
#[test]
fn small_star_is_not_clamped() {
    let small = OfirPeriod {
        stellar_radius: 0.008,
        stellar_mass: 0.6,
        maximum_period: days(1.0),
        ..OfirPeriod::default()
    };
    let clamped = OfirPeriod {
        stellar_radius: 0.1,
        ..small
    };
    assert_ne!(
        small.periods(&sector()).unwrap(),
        clamped.periods(&sector()).unwrap()
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
    let settings = OfirPeriod {
        minimum_period,
        maximum_period,
        ..OfirPeriod::default()
    };
    assert_eq!(
        settings.periods(&sector()),
        Err(GridError::PeriodNotPositive)
    );
}

#[test]
fn single_time_has_no_baseline() {
    let lc = LightCurve::from_btjd(vec![1325.0], vec![1.0], vec![1e-3]).unwrap();
    assert_eq!(
        OfirPeriod::default().periods(&lc),
        Err(GridError::NoBaseline)
    );
}

// Set, or from the defaults: a one-day light curve shows two transits only
// below 0.5 days, inside the Sun's Roche limit of 0.6 days. TLS would fall
// back to a five-day baseline instead.
#[rstest]
#[case::set(days(5.0), days(2.0), (1325.0, 1335.0))]
#[case::from_defaults(None, None, (1325.0, 1326.0))]
fn reversed_range(
    #[case] minimum_period: Option<TimeDelta>,
    #[case] maximum_period: Option<TimeDelta>,
    #[case] (t_min, t_max): (f64, f64),
) {
    let settings = OfirPeriod {
        minimum_period,
        maximum_period,
        ..OfirPeriod::default()
    };
    assert_eq!(
        settings.periods(&light_curve(t_min, t_max)),
        Err(GridError::PeriodRangeEmpty)
    );
}

// A star a millionth of the Sun's radius has its Roche limit at seconds.
#[test]
fn grid_too_large() {
    let settings = OfirPeriod {
        stellar_radius: 1e-6,
        ..OfirPeriod::default()
    };
    assert_eq!(settings.periods(&sector()), Err(GridError::GridTooLarge));
}

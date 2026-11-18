// The fast search: its table, its sums against the exact fit at its own
// trial times, against the oracle, transits across the fold's ends, the
// durations it tries, the depth floor, and threads.

use std::num::NonZeroUsize;

use super::{Random, light_curve};
use crate::durations::{Constants, Stars};
use crate::exact::best;
use crate::fast::{PeriodBest, Scratch, Search, Settings, Table};
use crate::fit::{Points, Position, fit};
use crate::template::Template;

const SAMPLES: usize = 4096;

fn settings(steps: usize) -> Settings {
    Settings {
        steps,
        min_depth: 0.0,
        stars: Stars::TLS,
        constants: Constants::IAU,
    }
}

/// A search over exactly the durations `fractions` (stars wide enough to
/// allow them all).
fn search<'a>(
    points: &'a Points,
    table: &'a Table,
    fractions: &'a [f64],
    steps: usize,
) -> Search<'a> {
    let stars = Stars {
        r_min: 1e-6,
        r_max: 1e6,
        m_min: 1.0,
        m_max: 1.0,
    };
    let origin = points.time[0];
    Search {
        points,
        table,
        fractions,
        settings: Settings {
            stars,
            ..settings(steps)
        },
        flat_chi2: points.flat_chi2(),
        origin,
        span: points.time[points.time.len() - 1] - origin,
    }
}

#[test]
fn the_table_is_the_template_at_its_samples_and_close_between() {
    for template in [
        Template::tls_default(),
        Template::tls_grazing(),
        Template::tls_box(),
    ] {
        let table = Table::new(&template, SAMPLES);
        for j in 0..=SAMPLES {
            let x = j as f64 / SAMPLES as f64 - 0.5;
            assert_eq!(
                table.at(x),
                if j == 0 || j == SAMPLES {
                    0.0
                } else {
                    template.shape(x)
                }
            );
        }
        for x in [-0.5, 0.5, 0.7, -3.0, f64::NAN, f64::INFINITY] {
            assert_eq!(table.at(x), 0.0, "{x}");
        }
        let mut worst: f64 = 0.0;
        for i in 0..100_000 {
            let x = (f64::from(i) + 0.5) / 100_000.0 - 0.5;
            worst = worst.max((table.at(x) - template.shape(x)).abs());
        }
        assert!(worst < 2e-4, "{worst}");
    }
}

#[test]
fn its_sums_are_the_exact_fits_at_its_own_trial_times() {
    let template = Template::tls_default();
    let table = Table::new(&template, SAMPLES);
    let mut random = Random::new(21);
    for _ in 0..10 {
        let period = random.between(0.8, 5.0);
        let duration = random.between(0.02, 0.1) * period;
        let position = Position {
            period,
            duration,
            t0: random.between(0.0, 10.0),
        };
        let points = light_curve(
            &mut random,
            &template,
            position,
            2e-3,
            1e-3,
            (0.0, 20.0, 1.0 / 144.0),
        );
        let fractions = [duration / period];
        let found = search(&points, &table, &fractions, 25).period(period, &mut Scratch::default());
        let exact = fit(
            &points,
            &template,
            Position {
                period,
                duration: found.duration,
                t0: found.t0,
            },
        );
        let gain = points.flat_chi2() - exact.chi2;
        assert!(
            (found.chi2 - exact.chi2).abs() < 1e-3 * gain,
            "{found:?} {exact:?}"
        );
        assert!(
            (found.depth / exact.depth - 1.0).abs() < 1e-3,
            "{found:?} {exact:?}"
        );
    }
}

// The trial times are a grid, so the fast search loses a little of the
// best gain, less the finer the grid: on these 10 noisy transits at most
// 2.8%, 1.6%, 0.31% and 0.11% with 10, 25, 50 and 100 trial times per
// duration. The peaks are refined exactly later.
#[test]
fn close_to_the_oracle() {
    let template = Template::tls_default();
    let table = Table::new(&template, SAMPLES);
    for (steps, bound) in [(10, 0.03), (25, 0.02), (50, 0.004), (100, 0.0015)] {
        let mut random = Random::new(22);
        let mut worst: f64 = 0.0;
        for _ in 0..10 {
            let period = random.between(0.8, 5.0);
            let duration = random.between(0.02, 0.1) * period;
            let position = Position {
                period,
                duration,
                t0: random.between(0.0, 10.0),
            };
            let points = light_curve(
                &mut random,
                &template,
                position,
                2e-3,
                1e-3,
                (0.0, 20.0, 1.0 / 144.0),
            );
            let fractions = [duration / period];
            let found =
                search(&points, &table, &fractions, steps).period(period, &mut Scratch::default());
            let oracle = best(&points, &template, period, duration, 20);
            let gain = points.flat_chi2() - oracle.fit.chi2;
            worst = worst.max((found.chi2 - oracle.fit.chi2) / gain);
            assert!(
                found.chi2 >= oracle.fit.chi2 - 1e-6 * gain,
                "{found:?} {oracle:?}"
            );
        }
        assert!(worst <= bound, "{steps}: {worst}");
    }
}

#[test]
fn a_noiseless_transit_across_the_folds_ends() {
    let template = Template::tls_default();
    let table = Table::new(&template, SAMPLES);
    let mut random = Random::new(23);
    for t0 in [0.0, 1e-7, 1.999_999_9, 0.5] {
        let position = Position {
            period: 2.0,
            duration: 0.1,
            t0,
        };
        let points = light_curve(
            &mut random,
            &template,
            position,
            0.01,
            0.0,
            (0.0, 27.0, 1.0 / 720.0),
        );
        let fractions = [0.05];
        let found = search(&points, &table, &fractions, 25).period(2.0, &mut Scratch::default());
        let step = 0.1 / 25.0;
        let apart = (found.t0 - t0)
            .rem_euclid(2.0)
            .min((t0 - found.t0).rem_euclid(2.0));
        assert!(apart <= 0.5 * step + 1e-12, "{t0}: {found:?}");
        assert!((found.depth / 0.01 - 1.0).abs() < 0.05, "{found:?}");
    }
}

#[test]
fn nothing_to_fit_leaves_the_flat_model() {
    let template = Template::tls_default();
    let table = Table::new(&template, SAMPLES);
    let time: Vec<f64> = (0..2000).map(|i| f64::from(i) * 0.01).collect();
    let points = Points {
        residual: vec![-1e-3; time.len()],
        weight: vec![1.0; time.len()],
        time,
    };
    let fractions = [0.01, 0.02, 0.04];
    let found = search(&points, &table, &fractions, 25).period(3.0, &mut Scratch::default());
    assert_eq!(
        found,
        PeriodBest {
            period: 3.0,
            chi2: points.flat_chi2(),
            duration: 0.03,
            t0: 0.0,
            depth: 0.0,
        }
    );
}

#[test]
fn fits_shallower_than_the_floor_do_not_count() {
    let template = Template::tls_default();
    let table = Table::new(&template, SAMPLES);
    let mut random = Random::new(24);
    let position = Position {
        period: 2.0,
        duration: 0.1,
        t0: 0.7,
    };
    let points = light_curve(
        &mut random,
        &template,
        position,
        1e-3,
        0.0,
        (0.0, 20.0, 1.0 / 144.0),
    );
    let fractions = [0.05];
    let mut s = search(&points, &table, &fractions, 25);
    assert!(s.period(2.0, &mut Scratch::default()).depth > 9e-4);
    s.settings.min_depth = 1.1e-3;
    let found = s.period(2.0, &mut Scratch::default());
    assert!(found.depth == 0.0 || found.depth > 1.1e-3, "{found:?}");
    assert!(found.chi2 > points.flat_chi2() * 0.5);
}

#[test]
fn the_durations_at_a_period_are_the_stars() {
    let points = Points {
        time: vec![0.0, 27.0],
        residual: vec![0.0; 2],
        weight: vec![1.0; 2],
    };
    let table = Table::new(&Template::tls_box(), 64);
    let fractions: Vec<f64> = (0..200).map(|i| 1e-4 * 1.05_f64.powi(i)).collect();
    let s = Search {
        points: &points,
        table: &table,
        fractions: &fractions,
        settings: settings(25),
        flat_chi2: 0.0,
        origin: 0.0,
        span: 27.0,
    };
    let stars = Stars::TLS;
    for period in [0.3, 1.0, 5.0, 13.0] {
        let shortest = stars.shortest(Constants::IAU, period);
        let orbits = 27.0 / period;
        let longest = stars.longest(Constants::IAU, period) * (orbits + 1.0) / orbits;
        let kept: Vec<f64> = s.durations(period).collect();
        let expected: Vec<f64> = fractions
            .iter()
            .filter(|&&q| q >= shortest && q <= longest)
            .map(|q| q * period)
            .collect();
        assert_eq!(kept, expected);
        assert!(!kept.is_empty());
    }
    // A period far longer than the light curve: the longest transit is
    // widened by (n + 1) / n for n = 0.027 orbits, past the period itself,
    // but a duration as long as the period is never tried.
    let whole = [0.5, 1.0, 1.5];
    let wide = search(&points, &table, &whole, 25);
    assert_eq!(wide.durations(1000.0).collect::<Vec<_>>(), vec![500.0]);
}

#[test]
fn the_same_results_on_any_number_of_threads() {
    let template = Template::tls_default();
    let table = Table::new(&template, 1024);
    let mut random = Random::new(25);
    let position = Position {
        period: 3.1,
        duration: 0.12,
        t0: 1.0,
    };
    let points = light_curve(
        &mut random,
        &template,
        position,
        2e-3,
        1e-3,
        (0.0, 15.0, 1.0 / 48.0),
    );
    let fractions = [0.01, 0.02, 0.04, 0.08];
    let s = search(&points, &table, &fractions, 10);
    let periods: Vec<f64> = (0..37).map(|i| 1.0 + 0.11 * f64::from(i)).collect();
    let mut scratch = Scratch::default();
    let serial: Vec<PeriodBest> = periods.iter().map(|&p| s.period(p, &mut scratch)).collect();
    for threads in [1, 2, 3, 8, 64] {
        assert_eq!(
            s.periods(&periods, NonZeroUsize::new(threads).unwrap()),
            serial,
            "{threads}"
        );
    }
    assert!(s.periods(&[], NonZeroUsize::new(4).unwrap()).is_empty());
}

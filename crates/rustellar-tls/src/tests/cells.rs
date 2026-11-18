// The search turned round: its cells against the exact fit of one transit,
// against the oracle and the fold, transits at the light curve's ends, the
// durations it tries, the depth floor, and threads.

use std::num::NonZeroUsize;

use super::{Random, light_curve};
use crate::cells::{CELLS_PER_STEP, Cells, Search};
use crate::durations::{Constants, Stars};
use crate::exact::best;
use crate::fast::{self, PeriodBest, Scratch, Settings, Table};
use crate::fit::{Points, Position, fit};
use crate::template::Template;

const SAMPLES: usize = 4096;

/// Stars wide enough to allow every duration tried here.
const WIDE: Stars = Stars {
    r_min: 1e-6,
    r_max: 1e6,
    m_min: 1.0,
    m_max: 1.0,
};

fn settings(steps: usize) -> Settings {
    Settings {
        steps,
        min_depth: 0.0,
        stars: WIDE,
        constants: Constants::IAU,
    }
}

fn search<'a>(
    points: &'a Points,
    table: &'a Table,
    durations: &'a [f64],
    steps: usize,
) -> Search<'a> {
    let origin = points.time[0];
    Search {
        points,
        table,
        durations,
        cells_per_step: CELLS_PER_STEP,
        settings: settings(steps),
        flat_chi2: points.flat_chi2(),
        origin,
        span: points.time[points.time.len() - 1] - origin,
    }
}

/// The search at one period.
fn one(s: &Search<'_>, period: f64) -> PeriodBest {
    s.periods(&[period], NonZeroUsize::MIN)[0]
}

#[test]
fn each_cell_holds_the_sums_of_one_transit_centred_there() {
    let template = Template::tls_default();
    let table = Table::new(&template, SAMPLES);
    let mut random = Random::new(31);
    let points = light_curve(
        &mut random,
        &template,
        Position {
            period: 2.3,
            duration: 0.1,
            t0: 0.4,
        },
        2e-3,
        1e-3,
        (0.0, 10.0, 1.0 / 144.0),
    );
    let duration = 0.13;
    let cells = Cells::new(&points, &table, duration, 20, NonZeroUsize::MIN);
    for threads in [2, 3, 7] {
        assert_eq!(
            Cells::new(
                &points,
                &table,
                duration,
                20,
                NonZeroUsize::new(threads).unwrap()
            ),
            cells
        );
    }
    assert_eq!(cells.step, duration / 20.0);
    assert!(cells.start <= points.time[0] - 0.5 * duration - cells.step);
    let end = cells.start + cells.step * (cells.rs.len() - 1) as f64;
    assert!(end >= points.time[points.time.len() - 1] + 0.5 * duration + cells.step);
    for m in (0..cells.rs.len()).step_by(7) {
        let centre = cells.start + cells.step * m as f64;
        // One transit: a period longer than the light curve.
        let (mut rs, mut ss) = (0.0, 0.0);
        for ((&t, &r), &w) in points.time.iter().zip(&points.residual).zip(&points.weight) {
            let s = template.shape(
                Position {
                    period: 1e6,
                    duration,
                    t0: centre,
                }
                .x(t),
            );
            rs += w * r * s;
            ss += w * s * s;
        }
        let scale = ss.max(1.0);
        assert!(
            (cells.rs[m] - rs).abs() < 1e-3 * scale.sqrt() * rs.abs().max(1.0),
            "{m}"
        );
        assert!((cells.ss[m] - ss).abs() < 1e-3 * scale, "{m}");
    }
}

// Each transit's sums are interpolated between cells a quarter of the trial
// times' spacing apart: on the fold's 10 noisy transits the gain lost
// against the oracle is at most 3.07%, 1.61%, 0.34% and 0.10% with 10, 25,
// 50 and 100 trial times per duration (the fold: 2.8%, 1.6%, 0.31%,
// 0.11%). Moved to the nearest cell instead, the cells could fit a little
// better than any periodic position.
#[test]
fn close_to_the_oracle() {
    let template = Template::tls_default();
    let table = Table::new(&template, SAMPLES);
    for (steps, bound) in [(10, 0.032), (25, 0.017), (50, 0.0035), (100, 0.0011)] {
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
            let durations = [duration];
            let found = one(&search(&points, &table, &durations, steps), period);
            let oracle = best(&points, &template, period, duration, 20);
            let gain = points.flat_chi2() - oracle.fit.chi2;
            worst = worst.max((found.chi2 - oracle.fit.chi2) / gain);
            assert!(
                found.chi2 >= oracle.fit.chi2 - 1e-3 * gain,
                "{found:?} {oracle:?}"
            );
        }
        assert!(worst <= bound, "{steps}: {worst}");
    }
}

// Against the fold on the same trial times: close, and the same period's
// best within a trial time.
#[test]
fn close_to_the_fold() {
    let template = Template::tls_default();
    let table = Table::new(&template, SAMPLES);
    let mut random = Random::new(32);
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
            (0.0, 20.0, 1.0 / 720.0),
        );
        let durations = [duration];
        let ours = one(&search(&points, &table, &durations, 50), period);
        let fractions = [duration / period];
        let origin = points.time[0];
        let fold = fast::Search {
            points: &points,
            table: &table,
            fractions: &fractions,
            settings: settings(50),
            flat_chi2: points.flat_chi2(),
            origin,
            span: points.time[points.time.len() - 1] - origin,
        }
        .period(period, &mut Scratch::default());
        let gain = points.flat_chi2() - fold.chi2;
        assert!(
            (ours.chi2 - fold.chi2).abs() < 3e-3 * gain,
            "{ours:?} {fold:?}"
        );
        let apart = (ours.t0 - fold.t0).rem_euclid(period);
        assert!(
            apart.min(period - apart) <= 2.0 * duration / 50.0,
            "{ours:?} {fold:?}"
        );
        // The fit at the cells' best place, exactly, is close to what the
        // cells say.
        let exact = fit(
            &points,
            &template,
            Position {
                period,
                duration,
                t0: ours.t0,
            },
        );
        assert!(
            (ours.chi2 - exact.chi2).abs() < 3e-3 * gain,
            "{ours:?} {exact:?}"
        );
    }
}

#[test]
fn transits_at_the_light_curves_ends() {
    let template = Template::tls_default();
    let table = Table::new(&template, SAMPLES);
    let mut random = Random::new(33);
    // Transits centred just before the first time, at it, just after the
    // last, and across the period's end in phase.
    for (t0, period) in [
        (-0.04, 2.0),
        (0.0, 2.0),
        (1.999_999_9, 2.0),
        (0.5, 2.0),
        (26.98, 3.0),
    ] {
        let position = Position {
            period,
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
        let durations = [0.1];
        let found = one(&search(&points, &table, &durations, 25), period);
        let apart = (found.t0 - t0)
            .rem_euclid(period)
            .min((t0 - found.t0).rem_euclid(period));
        assert!(apart <= 0.1 / 25.0, "{t0}: {found:?}");
        assert!((found.depth / 0.01 - 1.0).abs() < 0.05, "{t0}: {found:?}");
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
    let durations = [0.03, 0.06, 0.12];
    let found = one(&search(&points, &table, &durations, 25), 3.0);
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
    // No duration allowed at this period at all, and no durations at all.
    let tiny = [1e-6];
    let mut none = search(&points, &table, &tiny, 25);
    none.settings.stars = Stars::TLS;
    let flat = one(&none, 3.0);
    assert_eq!(
        (flat.chi2, flat.depth, flat.duration),
        (points.flat_chi2(), 0.0, 1e-6)
    );
    none.durations = &[];
    assert_eq!(one(&none, 3.0).duration, 0.0);
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
    let durations = [0.1];
    let mut s = search(&points, &table, &durations, 25);
    assert!(one(&s, 2.0).depth > 9e-4);
    s.settings.min_depth = 1.1e-3;
    let found = one(&s, 2.0);
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
    let durations: Vec<f64> = (0..200).map(|i| 1e-4 * 1.05_f64.powi(i)).collect();
    let mut s = search(&points, &table, &durations, 1);
    s.settings.stars = Stars::TLS;
    let stars = Stars::TLS;
    for period in [0.3, 1.0, 5.0, 13.0] {
        let shortest = stars.shortest(Constants::IAU, period) * period;
        let orbits = 27.0 / period;
        let longest = stars.longest(Constants::IAU, period) * period * (orbits + 1.0) / orbits;
        let kept: Vec<f64> = s.durations_at(period).collect();
        let expected: Vec<f64> = durations
            .iter()
            .copied()
            .filter(|&d| d >= shortest && d <= longest)
            .collect();
        assert_eq!(kept, expected);
        assert!(!kept.is_empty());
    }
    // Never a duration as long as the period.
    s.settings.stars = WIDE;
    assert!(s.durations_at(1.0).all(|d| d < 1.0));
    assert!(s.durations_at(1.0).count() > 0);
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
    let durations = [0.02, 0.04, 0.08, 0.16];
    let s = search(&points, &table, &durations, 10);
    let periods: Vec<f64> = (0..37).map(|i| 1.0 + 0.11 * f64::from(i)).collect();
    let serial: Vec<PeriodBest> = periods.iter().map(|&p| one(&s, p)).collect();
    for threads in [1, 2, 3, 8, 64] {
        assert_eq!(
            s.periods(&periods, NonZeroUsize::new(threads).unwrap()),
            serial,
            "{threads}"
        );
    }
    assert!(s.periods(&[], NonZeroUsize::new(4).unwrap()).is_empty());
}

// Refinement: never worse than where it starts, noiseless transits found
// to rounding from a coarse start, noisy ones closer to the truth than the
// grid, and the exact gain against the fit.

use super::{Random, light_curve};
use crate::durations::{Constants, Stars};
use crate::exact::best;
use crate::fast::{Binning, Scratch, Search, Settings, Table};
use crate::fit::{Points, Position, fit, gain};
use crate::refine::{Reach, refine};
use crate::template::Template;

fn apart(a: f64, b: f64, period: f64) -> f64 {
    let d = (a - b).rem_euclid(period);
    d.min(period - d)
}

#[test]
fn the_gain_is_the_flat_chi2_less_the_fits() {
    let template = Template::tls_default();
    let mut random = Random::new(31);
    for _ in 0..50 {
        let position = Position {
            period: random.between(0.5, 4.0),
            duration: random.between(0.03, 0.2),
            t0: random.between(0.0, 5.0),
        };
        let points = light_curve(
            &mut random,
            &template,
            position,
            1e-3,
            1e-3,
            (0.0, 10.0, 0.01),
        );
        let g = gain(&points, &template, position);
        let f = fit(&points, &template, position);
        assert!(
            (points.flat_chi2() - f.chi2 - g).abs() <= 1e-9 * points.flat_chi2(),
            "{g} {f:?}"
        );
    }
    // Nothing below 1 in transit: no gain.
    let points = Points {
        time: vec![0.0, 0.5, 1.0],
        residual: vec![-1.0; 3],
        weight: vec![1.0; 3],
    };
    let position = Position {
        period: 1.0,
        duration: 0.2,
        t0: 0.0,
    };
    assert_eq!(gain(&points, &template, position), 0.0);
}

#[test]
fn a_noiseless_transit_from_a_coarse_start() {
    let template = Template::tls_default();
    let mut random = Random::new(32);
    for (period, duration, t0) in [(3.7, 0.13, 1.3), (1.1, 0.06, 0.02), (6.3, 0.25, 4.0)] {
        let truth = Position {
            period,
            duration,
            t0,
        };
        let points = light_curve(
            &mut random,
            &template,
            truth,
            0.01,
            0.0,
            (0.0, 27.0, 1.0 / 720.0),
        );
        let start = Position {
            period: period * (1.0 + 2e-4),
            duration: duration * 1.04,
            t0: t0 + 0.02 * duration,
        };
        let reach = Reach {
            t0: 0.05 * duration,
            duration: 1.1,
            period: 5e-4 * period,
        };
        let refined = refine(&points, &template, start, reach);
        let got = refined.position;
        assert!((got.period / period - 1.0).abs() < 1e-8, "{got:?}");
        assert!((got.duration / duration - 1.0).abs() < 1e-6, "{got:?}");
        assert!(apart(got.t0, t0, period) < 1e-7, "{got:?}");
        assert!((refined.fit.depth / 0.01 - 1.0).abs() < 1e-6, "{refined:?}");
        // The mid-transit time is the one nearest the first point.
        assert!((got.t0 - points.time[0]).abs() <= 0.5 * period, "{got:?}");
    }
}

// With noise the best fit is not the truth (on these curves both the
// grid's and the refined mid-transit times are about 1% of a duration off
// it), so what refinement promises is the fit: never worse than its start,
// nor than the oracle's at the start's duration and period.
#[test]
fn never_worse_than_its_start_or_the_oracle() {
    let template = Template::tls_default();
    let table = Table::new(&template, 4096);
    let mut random = Random::new(33);
    for _ in 0..12 {
        let period = random.between(1.0, 5.0);
        let duration = random.between(0.03, 0.08) * period;
        let truth = Position {
            period,
            duration,
            t0: random.between(0.0, 5.0),
        };
        let points = light_curve(
            &mut random,
            &template,
            truth,
            2e-3,
            1e-3,
            (0.0, 27.0, 1.0 / 144.0),
        );
        // The fast search on a grid of durations 10% apart, at the true
        // period.
        let fractions: Vec<f64> = (0..40).map(|i| 0.005 * 1.1_f64.powi(i)).collect();
        let search = Search {
            points: &points,
            table: &table,
            fractions: &fractions,
            settings: Settings {
                steps: 25,
                min_depth: 0.0,
                stars: Stars {
                    r_min: 1e-6,
                    r_max: 1e6,
                    m_min: 1.0,
                    m_max: 1.0,
                },
                constants: Constants::IAU,
            },
            binning: Binning::Auto,
            flat_chi2: points.flat_chi2(),
            origin: points.time[0],
            span: 27.0,
        };
        let coarse = search.period(period, &mut Scratch::default());
        let start = Position {
            period,
            duration: coarse.duration,
            t0: coarse.t0,
        };
        let reach = Reach {
            t0: coarse.duration / 25.0,
            duration: 1.1,
            period: 1e-3 * period,
        };
        let refined = refine(&points, &template, start, reach);
        let start_chi2 = fit(&points, &template, start).chi2;
        assert!(refined.fit.chi2 <= start_chi2, "{refined:?} {start_chi2}");
        let oracle = best(&points, &template, period, coarse.duration, 20);
        assert!(
            refined.fit.chi2 <= oracle.fit.chi2 * (1.0 + 1e-12),
            "{refined:?} {oracle:?}"
        );
        assert!(apart(refined.position.t0, truth.t0, period) < 0.05 * duration);
        assert!((refined.fit.depth / 2e-3 - 1.0).abs() < 0.15, "{refined:?}");
    }
}

// The best fit's standard errors against the scatter of fits to many noisy
// light curves, each transit's own depth, and odd against even transits.

use super::{Random, light_curve};
use crate::errors::{OddEven, Transit, errors, odd_even, transits};
use crate::fit::{Points, Position, fit};
use crate::refine::{Reach, refine};
use crate::template::{SAMPLED_INTERVALS, Sampled, Template};

const TRUTH: Position = Position {
    period: 3.1,
    duration: 0.12,
    t0: 1.3,
};

/// 27 days of points every `cadence` days with the template's transit of
/// `depth` at `TRUTH`, and noise `noise`.
fn curve(random: &mut Random, depth: f64, noise: f64, cadence: f64) -> Points {
    light_curve(
        random,
        &Template::tls_default(),
        TRUTH,
        depth,
        noise,
        (0.0, 27.0, cadence),
    )
}

/// Over 300 light curves made with the template itself (so the model is
/// right), each fitted from the truth: for period, T0, duration and depth,
/// the share of fits within their standard error of the truth, and the
/// fits' root mean square distance from it over the errors'.
fn scatter(noise: f64, cadence: f64) -> [(f64, f64); 4] {
    let template = Template::tls_default();
    let sampled = Sampled::new(&template, SAMPLED_INTERVALS);
    let depth = 2e-3;
    let mut random = Random::new(41);
    let count = 300;
    let (mut within, mut squares, mut variance) = ([0_u32; 4], [0.0; 4], [0.0; 4]);
    for _ in 0..count {
        let points = curve(&mut random, depth, noise, cadence);
        let reach = Reach {
            t0: 0.02,
            duration: 1.3,
            period: 5e-3,
        };
        let p = refine(&points, &sampled, TRUTH, reach).position;
        let fitted = fit(&points, &template, p);
        let e = errors(&points, &template, p, fitted.depth).unwrap();
        let off = [
            p.period - TRUTH.period,
            p.t0 - TRUTH.t0,
            p.duration - TRUTH.duration,
            fitted.depth - depth,
        ];
        let sigma = [e.period, e.t0, e.duration, e.depth];
        for (((within, squares), variance), (off, sigma)) in within
            .iter_mut()
            .zip(&mut squares)
            .zip(&mut variance)
            .zip(off.iter().zip(sigma))
        {
            *within += u32::from(off.abs() <= sigma);
            *squares += off * off;
            *variance += sigma * sigma;
        }
    }
    let mut out = [(0.0, 0.0); 4];
    for (o, ((&w, &s), &v)) in out
        .iter_mut()
        .zip(within.iter().zip(&squares).zip(&variance))
    {
        *o = (f64::from(w) / f64::from(count), (s / v).sqrt());
    }
    out
}

// Where the signal is strong (about 100 times the noise in all, at 10
// minutes), the chi-squared is a quadratic over the fits' range and its
// curvature gives the errors: 65% to 74% of fits within 1 sigma (68% has a
// binomial spread of 2.7% over 300), scatter 0.95 to 1.06 of the errors.
#[test]
fn the_errors_are_the_scatter_of_the_fits() {
    for (i, (within, ratio)) in scatter(2e-4, 1.0 / 144.0).into_iter().enumerate() {
        assert!((within - 0.683).abs() < 0.08, "{i}: {within}");
        assert!((ratio - 1.0).abs() < 0.1, "{i}: {ratio}");
    }
}

// Where it is weak (about 12 times the noise, 30 minutes, six points a
// transit), the errors are a lower bound, as curvature errors are: the fits
// scatter up to 22% more, with tails (95% within 3 sigma, not 2), as points
// crossing the template's edges make the chi-squared lumpy in time.
#[test]
fn at_low_signal_the_errors_are_a_lower_bound() {
    for (i, (within, ratio)) in scatter(1e-3, 1.0 / 48.0).into_iter().enumerate() {
        assert!(within > 0.6 && within < 0.75, "{i}: {within}");
        assert!(ratio > 1.0 && ratio < 1.3, "{i}: {ratio}");
    }
}

// No points in transit: nothing fixes the parameters.
#[test]
fn no_points_in_transit_no_errors() {
    let template = Template::tls_default();
    let mut random = Random::new(42);
    let points = curve(&mut random, 2e-3, 1e-3, 1.0 / 48.0);
    let empty = Position {
        period: 100.0,
        duration: 0.1,
        t0: 60.0,
    };
    assert_eq!(errors(&points, &template, empty, 2e-3), None);
}

// Without noise every transit's depth is the injected one, its error the
// fit's with its points alone, and the transits' depths weighted by their
// errors are the whole fit's.
#[test]
fn each_transits_depth() {
    let template = Template::tls_default();
    let mut random = Random::new(43);
    let mut points = curve(&mut random, 2e-3, 1e-3, 1.0 / 48.0);
    for (r, &t) in points.residual.iter_mut().zip(&points.time) {
        *r = 2e-3 * template.shape(TRUTH.x(t));
    }
    let found = transits(&points, &template, TRUTH);
    // Transits at 1.3 + 3.1 n for n = 0..8, all inside 27 days.
    assert_eq!(found.len(), 9);
    let (mut sum, mut weights) = (0.0, 0.0);
    for (n, t) in found.iter().enumerate() {
        assert_eq!(t.number, i64::try_from(n).unwrap());
        assert!((t.time - TRUTH.period.mul_add(t.number as f64, TRUTH.t0)).abs() < 1e-12);
        assert!((t.depth / 2e-3 - 1.0).abs() < 1e-12, "{t:?}");
        // About 0.12 days at 30 minutes, less the ends where s is 0.
        assert!((4..=6).contains(&t.points), "{t:?}");
        let w = t.error.powi(-2);
        sum += w * t.depth;
        weights += w;
    }
    let whole = fit(&points, &template, TRUTH);
    assert!((sum / weights / whole.depth - 1.0).abs() < 1e-12);
    // The errors: 1 / sqrt(sum w s²), w = 1 / noise².
    let one = &found[0];
    let ss: f64 = points
        .time
        .iter()
        .map(|&t| template.shape(TRUTH.x(t)))
        .filter(|&s| s > 0.0)
        .take(one.points)
        .map(|s| s * s / 1e-6)
        .sum();
    assert!((one.error * ss.sqrt() - 1.0).abs() < 1e-12);
}

fn transit(number: i64, depth: f64, error: f64) -> Transit {
    Transit {
        number,
        time: 0.0,
        depth,
        error,
        points: 5,
    }
}

// Odd and even transits: each set's depth weighted by the errors, and how
// many combined errors apart.
#[test]
fn odd_and_even_transits() {
    let same: Vec<Transit> = (0..6).map(|n| transit(n, 1e-3, 2e-4)).collect();
    let OddEven { sigma, .. } = odd_even(&same).unwrap();
    assert!(sigma < 1e-12);
    // Even 2e-3, odd 1e-3, three of each with error 2e-4: each set's error
    // 2e-4 / sqrt(3), so the difference is 1e-3 / (2e-4 sqrt(2/3)).
    let alternating: Vec<Transit> = (0..6)
        .map(|n| transit(n, if n % 2 == 0 { 2e-3 } else { 1e-3 }, 2e-4))
        .collect();
    let o = odd_even(&alternating).unwrap();
    assert!((o.even - 2e-3).abs() < 1e-15 && (o.odd - 1e-3).abs() < 1e-15);
    assert!((o.odd_error - 2e-4 / 3_f64.sqrt()).abs() < 1e-15);
    let expected = 1e-3 / (2e-4 * (2.0_f64 / 3.0).sqrt());
    assert!((o.sigma / expected - 1.0).abs() < 1e-12, "{o:?}");
    // Weighted: one precise even transit outweighs a poor one.
    let weighted = [
        transit(0, 1e-3, 1e-4),
        transit(2, 3e-3, 1e-3),
        transit(1, 1e-3, 1e-4),
    ];
    let o = odd_even(&weighted).unwrap();
    let expected_even = (1e-3 / 1e-8 + 3e-3 / 1e-6) / (1.0 / 1e-8 + 1.0 / 1e-6);
    assert!((o.even / expected_even - 1.0).abs() < 1e-12);
    // Only even transits, or only odd: nothing to compare.
    assert_eq!(
        odd_even(&[transit(0, 1e-3, 1e-4), transit(2, 1e-3, 1e-4)]),
        None
    );
    assert_eq!(
        odd_even(&[transit(1, 1e-3, 1e-4), transit(3, 1e-3, 1e-4)]),
        None
    );
    assert_eq!(odd_even(&[]), None);
}

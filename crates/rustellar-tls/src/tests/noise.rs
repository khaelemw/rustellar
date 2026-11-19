// The best fit's signal to noise in white noise, and in noise correlated
// over the transit's duration.

use super::{Random, light_curve};
use crate::fit::{Points, Position};
use crate::noise::snr;
use crate::template::Template;

const TRUTH: Position = Position {
    period: 3.1,
    duration: 0.12,
    t0: 1.3,
};

/// 27 days at 10 minutes with a transit of 1e-3 at `TRUTH` and white noise
/// of 1e-3, on a level of 0.01 the depth is measured from.
fn curve(seed: u64) -> Points {
    let mut random = Random::new(seed);
    let mut points = light_curve(
        &mut random,
        &Template::tls_default(),
        TRUTH,
        1e-3,
        1e-3,
        (0.0, 27.0, 1.0 / 144.0),
    );
    points.residual.iter_mut().for_each(|r| *r += 0.01);
    points
}

// In white noise the binned scatter is the white noise's over the root of
// the points a bin, so the red noise is near 0 and both signals to noise
// are the depth over the white noise times the root of the points in
// transit.
#[test]
fn in_white_noise_the_two_agree() {
    let template = Template::tls_default();
    let (mut white, mut pink, mut red) = (0.0, 0.0, 0.0);
    for seed in 0..50 {
        let s = snr(&curve(seed), &template, TRUTH).unwrap();
        assert!((s.white_noise / 1e-3 - 1.0).abs() < 0.05, "{s:?}");
        white += s.white;
        pink += s.pink;
        red += s.red_noise;
    }
    // The template's mean shape at the points in transit, over the noise,
    // times their root.
    let shapes: Vec<f64> = curve(0)
        .time
        .iter()
        .map(|&t| template.shape(TRUTH.x(t)))
        .filter(|&s| s > 0.0)
        .collect();
    let n = shapes.len() as f64;
    let expected = shapes.iter().sum::<f64>() / n * n.sqrt();
    let (white, pink, red) = (white / 50.0, pink / 50.0, red / 50.0);
    assert!((white / expected - 1.0).abs() < 0.05, "{white} {expected}");
    // The red noise a small share of the white noise's 1e-3 / sqrt(17) a
    // bin, and so the pink signal to noise within a few percent.
    assert!(red < 1e-4, "{red}");
    assert!(pink / white > 0.9 && pink <= white, "{pink} {white}");
}

// With a level that wanders by 5e-4 from one duration to the next, the red
// noise is found, and the pink signal to noise is well below the white,
// which takes all 135 points as independent.
#[test]
fn red_noise_is_found() {
    let template = Template::tls_default();
    let mut random = Random::new(7);
    let mut points = curve(7);
    let mut level = 0.0;
    let mut block = -1.0;
    for (r, &t) in points.residual.iter_mut().zip(&points.time) {
        let b = (t / 0.12).floor();
        if b != block {
            block = b;
            level = (random.uniform() - 0.5) * 3_f64.sqrt() * 2.0 * 5e-4;
        }
        *r += level;
    }
    let s = snr(&points, &template, TRUTH).unwrap();
    assert!(s.red_noise > 3e-4 && s.red_noise < 6e-4, "{s:?}");
    assert!(s.pink < 0.6 * s.white, "{s:?}");
    // With 9 transits: about 0.75e-3 / sqrt(1e-6 / 135 + (5e-4)² / 9).
    assert!(s.pink > 2.5 && s.pink < 6.5, "{s:?}");
}

// Nothing in transit, or nothing out of it to measure the noise against.
#[test]
fn nothing_to_measure() {
    let template = Template::tls_default();
    let points = curve(1);
    let empty = Position {
        period: 100.0,
        duration: 0.1,
        t0: 60.0,
    };
    assert_eq!(snr(&points, &template, empty), None);
    // Every point in transit.
    let all = Points {
        time: vec![0.0, 0.01, 0.02],
        residual: vec![1e-3; 3],
        weight: vec![1.0; 3],
    };
    let wide = Position {
        period: 10.0,
        duration: 1.0,
        t0: 0.01,
    };
    assert_eq!(snr(&all, &template, wide), None);
    // Two points a transit, and out of transit two points more than a
    // duration apart: no full run of two.
    let apart = Points {
        time: vec![0.0, 0.5, 0.55, 1.0],
        residual: vec![0.0, 1e-3, 1e-3, 0.0],
        weight: vec![1.0; 4],
    };
    let middle = Position {
        period: 10.0,
        duration: 0.2,
        t0: 0.525,
    };
    assert_eq!(snr(&apart, &template, middle), None);
}

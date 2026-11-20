// The search for a second eclipse: each box the same as a box fit with
// the transit's points removed; a circular binary's second eclipse at half
// the period and an eccentric one elsewhere found; in white noise the
// significance at half the period a standard normal and the scan's spread
// near 1; and too short a period.

use rustellar_vet::{Candidate, Epoch, LightCurve, Shape, TimeDelta, fit_depth, secondary};

fn days(d: f64) -> TimeDelta {
    TimeDelta::from_days(d).unwrap()
}

fn at(t: f64) -> Epoch {
    Epoch::from_days(t).unwrap()
}

struct Random(u64);

impl Random {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut x = self.0;
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        x ^ (x >> 31)
    }

    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        let (u, v) = (1.0 - self.unit(), self.unit());
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }
}

const P: f64 = 3.7;
const T0: f64 = 1501.3;
const DURATION: f64 = 0.1;

// Days from the nearest of the boxes every `P` from `T0 + offset * P`.
fn from(t: f64, offset: f64) -> f64 {
    let c = T0 + offset * P;
    (t - c + 0.5 * P).rem_euclid(P) - 0.5 * P
}

// 27 days every 10 minutes from day 1500: a 3e-3 box transit, a box
// eclipse `second` deep at `offset` of the period, and white noise
// `noise` with uneven errors.
fn curve(second: f64, offset: f64, noise: f64, seed: u64) -> LightCurve {
    let mut r = Random(seed);
    let time: Vec<f64> = (0..3888).map(|i| 1500.0 + f64::from(i) / 144.0).collect();
    let err: Vec<f64> = time
        .iter()
        .map(|_| noise * (0.8 + 0.4 * r.unit()))
        .collect();
    let flux = time
        .iter()
        .zip(&err)
        .map(|(&t, &e)| {
            let transit = if from(t, 0.0).abs() < 0.5 * DURATION {
                3e-3
            } else {
                0.0
            };
            let eclipse = if from(t, offset).abs() < 0.5 * DURATION {
                second
            } else {
                0.0
            };
            1.0 - transit - eclipse + e * r.normal()
        })
        .collect();
    LightCurve::from_days(time, flux, err).unwrap()
}

fn candidate() -> Candidate {
    Candidate::new(days(P), at(T0), days(DURATION)).unwrap()
}

// Each box is a box fit at its phase to the points at least a duration
// from mid-transit.
#[test]
fn each_box_is_a_box_fit() {
    let lc = curve(1e-3, 0.5, 5e-4, 1);
    let s = secondary(&lc, &candidate());
    let keep: Vec<usize> = (0..lc.len())
        .filter(|&i| from(lc.time()[i].as_days(), 0.0).abs() >= DURATION)
        .collect();
    let reduced = LightCurve::from_days(
        keep.iter().map(|&i| lc.time()[i].as_days()).collect(),
        keep.iter().map(|&i| lc.flux()[i]).collect(),
        keep.iter().map(|&i| lc.flux_err()[i]).collect(),
    )
    .unwrap();
    for dip in [s.at_half, s.deepest, s.brightest] {
        let dip = dip.unwrap();
        let c = Candidate::new(days(P), at(T0 + dip.phase * P), days(DURATION)).unwrap();
        let fit = fit_depth(&reduced, &c, Shape::BOX).unwrap();
        assert!((dip.depth - fit.depth).abs() < 1e-12, "{dip:?} {fit:?}");
        assert!((dip.depth_err / fit.depth_err - 1.0).abs() < 1e-9);
        assert_eq!(dip.points, fit.points_in_transit);
    }
}

// A circular binary's second eclipse, 1e-3 deep at half the period: found
// there, deepest of the scan, and its depth back within its error.
#[test]
fn a_second_eclipse_at_half_the_period() {
    let s = secondary(&curve(1e-3, 0.5, 5e-4, 2), &candidate());
    let half = s.at_half.unwrap();
    assert!((half.depth - 1e-3).abs() < 4.0 * half.depth_err, "{half:?}");
    assert!(half.sigma() > 10.0);
    let deepest = s.deepest.unwrap();
    assert!((deepest.phase - 0.5).abs() < DURATION / P, "{deepest:?}");
    assert!(deepest.sigma() / s.spread > 5.0);
    assert!(half.upper_limit(3.0) > half.depth);
    // The primary on the same terms: 3e-3, a third of which at half.
    let primary = s.primary.unwrap();
    assert_eq!(primary.phase, 0.0);
    assert!((primary.depth - 3e-3).abs() < 4.0 * primary.depth_err);
    assert!((s.depth_ratio().unwrap() - 1.0 / 3.0).abs() < 0.05);
}

// An eccentric binary's second eclipse at 0.37 of the period: found there,
// and nothing at half the period.
#[test]
fn an_eccentric_second_eclipse() {
    let s = secondary(&curve(1e-3, 0.37, 5e-4, 3), &candidate());
    let deepest = s.deepest.unwrap();
    assert!((deepest.phase - 0.37).abs() < DURATION / P, "{deepest:?}");
    let half = s.at_half.unwrap();
    assert!(half.sigma().abs() < 4.0, "{half:?}");
    // A brightening limits the depth to the errors alone.
    let b = s.brightest.unwrap();
    assert!(b.depth < 0.0);
    assert_eq!(b.upper_limit(3.0), 3.0 * b.depth_err);
}

// In white noise over 400 draws: the significance at half the period has
// mean 0 and spread 1, the scan's spread is near 1, the deepest and
// brightest are each a few sigma, and the trials are the scan's length
// over the duration.
#[test]
fn white_noise_is_calibrated() {
    let (mut half, mut spread) = (vec![], vec![]);
    let mut s = secondary(&curve(0.0, 0.5, 5e-4, 0), &candidate());
    for seed in 0..400 {
        s = secondary(&curve(0.0, 0.5, 5e-4, 100 + seed), &candidate());
        half.push(s.at_half.unwrap().sigma());
        spread.push(s.spread);
        assert!(s.deepest.unwrap().sigma() < 5.0);
        assert!(s.brightest.unwrap().sigma() > -5.0);
    }
    let n = 400.0;
    let mean = half.iter().sum::<f64>() / n;
    let sd = (half.iter().map(|h| (h - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
    assert!(mean.abs() < 0.15, "{mean}");
    assert!((sd - 1.0).abs() < 0.1, "{sd}");
    let spread = spread.iter().sum::<f64>() / n;
    assert!((spread - 1.0).abs() < 0.1, "{spread}");
    assert_eq!(s.trials, 36);
}

// A period under two durations leaves no room; a light curve with every
// point near mid-transit has no points to scan; a box with every point in
// it no baseline.
#[test]
fn no_room_or_no_points() {
    let lc = curve(0.0, 0.5, 5e-4, 4);
    let short = Candidate::new(days(0.15), at(T0), days(DURATION)).unwrap();
    let s = secondary(&lc, &short);
    assert_eq!(
        (s.at_half, s.deepest, s.brightest, s.trials),
        (None, None, None, 0)
    );
    assert!(s.spread.is_nan());
    assert!(s.primary.is_some());
    assert_eq!(s.depth_ratio(), None);
    let near = LightCurve::from_days(vec![T0, T0 + 0.01], vec![1.0; 2], vec![1e-3; 2]).unwrap();
    let s = secondary(&near, &candidate());
    assert_eq!((s.deepest, s.primary), (None, None));
    // One point away from the transit, in the box at half the period.
    let one = LightCurve::from_days(vec![T0, T0 + 0.5 * P], vec![1.0; 2], vec![1e-3; 2]).unwrap();
    let s = secondary(&one, &candidate());
    assert_eq!((s.at_half, s.deepest), (None, None));
    // Two points: one box each; one sigma, no spread.
    let two = LightCurve::from_days(
        vec![T0 + 0.5 * P, T0 + 0.5 * P + 0.3],
        vec![0.999, 1.0],
        vec![1e-3; 2],
    )
    .unwrap();
    let s = secondary(&two, &candidate());
    let half = s.at_half.unwrap();
    assert!((half.depth - 1e-3).abs() < 1e-12);
    assert!(s.spread > 0.0);
}

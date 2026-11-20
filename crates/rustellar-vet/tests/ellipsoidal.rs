// Sinusoids out of eclipse: injected terms recovered exactly beside deep
// eclipses (left out), the errors calibrated on noise draws, and too few
// points or phases.

use std::f64::consts::TAU;

use rustellar_vet::{Candidate, Epoch, LightCurve, TimeDelta, ellipsoidal};

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
        (-2.0 * u.ln()).sqrt() * (TAU * v).cos()
    }
}

const P: f64 = 2.5;
const T0: f64 = 0.4;

fn candidate() -> Candidate {
    Candidate::new(
        TimeDelta::from_days(P).unwrap(),
        Epoch::from_btjd(T0).unwrap(),
        TimeDelta::from_days(0.12).unwrap(),
    )
    .unwrap()
}

// 27 days every 10 minutes: a baseline of 2.0 (in counts), the terms
// `terms` (cos 1, sin 1, cos 2, sin 2, fractions of it), 5% and 1% eclipses
// at 0 and 0.5, and white noise `noise` of the baseline.
fn curve(terms: [f64; 4], noise: f64, seed: u64) -> LightCurve {
    let mut r = Random(seed);
    let time: Vec<f64> = (0..3888).map(|i| f64::from(i) / 144.0).collect();
    let flux = time
        .iter()
        .map(|&t| {
            let x = TAU * (t - T0) / P;
            let ph = (t - T0).rem_euclid(P);
            let eclipse = if ph < 0.06 || ph > P - 0.06 {
                0.05
            } else if (ph - 0.5 * P).abs() < 0.06 {
                0.01
            } else {
                0.0
            };
            let wave = terms[0] * x.cos()
                + terms[1] * x.sin()
                + terms[2] * (2.0 * x).cos()
                + terms[3] * (2.0 * x).sin();
            2.0 * (1.0 + wave - eclipse + noise * r.normal())
        })
        .collect();
    LightCurve::from_btjd(time, flux, vec![2.0 * noise.max(1e-6); 3888]).unwrap()
}

#[test]
fn injected_terms_come_back() {
    let terms = [-1e-4, 3e-5, -6e-4, 2e-5];
    let e = ellipsoidal(&curve(terms, 0.0, 0), &candidate()).unwrap();
    let got = [e.once.cos, e.once.sin, e.twice.cos, e.twice.sin];
    for (g, t) in got.iter().zip(terms) {
        assert!((g - t).abs() < 1e-12, "{got:?}");
    }
    assert!((e.twice.amplitude() - 6e-4f64.hypot(2e-5)).abs() < 1e-12);
    // Both eclipses left out: a duration either side of each.
    let out = 3888.0 * (1.0 - 4.0 * 0.12 / P);
    assert!(
        (f64::from(u32::try_from(e.points).unwrap()) - out).abs() < 30.0,
        "{}",
        e.points
    );
}

// In white noise of 1e-3, over 300 draws: each term over its error is a
// standard normal, and twice-an-orbit's sigma squared averages 2.
#[test]
fn the_errors_are_calibrated() {
    let (mut z, mut s2) = (vec![], 0.0);
    for seed in 0..300 {
        let e = ellipsoidal(&curve([0.0; 4], 1e-3, 10 + seed), &candidate()).unwrap();
        z.push(e.twice.cos / e.twice.cos_err);
        z.push(e.once.sin / e.once.sin_err);
        s2 += e.twice.sigma().powi(2);
    }
    let n = f64::from(u32::try_from(z.len()).unwrap());
    let mean = z.iter().sum::<f64>() / n;
    let sd = (z.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
    assert!(mean.abs() < 0.1, "{mean}");
    assert!((sd - 1.0).abs() < 0.07, "{sd}");
    assert!((s2 / 300.0 - 2.0).abs() < 0.3, "{}", s2 / 300.0);
}

#[test]
fn too_few_points_or_phases() {
    let few = LightCurve::from_btjd(vec![1.0, 1.2, 1.4], vec![1.0; 3], vec![1e-3; 3]).unwrap();
    assert_eq!(ellipsoidal(&few, &candidate()), None);
    // Ten points at one phase, an orbit apart: nothing tells the terms
    // apart.
    let time: Vec<f64> = (0..10).map(|k| T0 + 0.7 + P * f64::from(k)).collect();
    let same = LightCurve::from_btjd(time, vec![1.0; 10], vec![1e-3; 10]).unwrap();
    assert_eq!(ellipsoidal(&same, &candidate()), None);
}

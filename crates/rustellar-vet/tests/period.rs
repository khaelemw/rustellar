// The period searched near a candidate's: the true period found from one
// off by up to the width, the spectrum's peak there, and the limits.

use rustellar_vet::{Candidate, Epoch, LightCurve, Shape, TimeDelta, VetError, search_period};

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

// 54 days every 10 minutes, a 2e-3 box `sign`-ward (1 a dip) 0.1 days
// long every `period` from 1.3, white noise 4e-4.
fn curve(period: f64, sign: f64, seed: u64) -> LightCurve {
    let mut r = Random(seed);
    let time: Vec<f64> = (0..7776).map(|i| f64::from(i) / 144.0).collect();
    let flux = time
        .iter()
        .map(|&t| {
            let inside = ((t - 1.3 + 0.5 * period).rem_euclid(period) - 0.5 * period).abs() < 0.05;
            1.0 - if inside { sign * 2e-3 } else { 0.0 } + 4e-4 * r.normal()
        })
        .collect();
    LightCurve::from_days(time, flux, vec![4e-4; 7776]).unwrap()
}

fn candidate(period: f64, t0: f64) -> Candidate {
    Candidate::new(
        TimeDelta::from_days(period).unwrap(),
        Epoch::from_days(t0).unwrap(),
        TimeDelta::from_days(0.1).unwrap(),
    )
    .unwrap()
}

// A candidate 0.5% off in period and 0.02 days off in time: the true
// period within a step, and the spectrum peaking there.
#[test]
fn the_true_period_from_one_off() {
    for seed in 1..=3 {
        let truth = 3.7;
        let lc = curve(truth, 1.0, seed);
        let c = candidate(truth * 1.005, 1.32);
        let found = search_period(&lc, &c, Shape::BOX, 0.01).unwrap();
        let step = 0.1 * c.period().as_days() / (4.0 * (lc.time()[7775].as_days()));
        let p = found.best.period().as_days();
        assert!((p - truth).abs() < step, "{p}");
        assert_eq!(found.best.depth(), Some(found.fit.depth));
        assert!((found.fit.depth - 2e-3).abs() < 5.0 * found.fit.depth_err);
        assert_eq!(found.periods.len(), found.delta_chi2.len());
        assert_eq!(found.periods.len() % 2, 1);
        let top = found
            .delta_chi2
            .iter()
            .copied()
            .filter(|g| !g.is_nan())
            .fold(f64::NEG_INFINITY, f64::max);
        assert_eq!(top, found.fit.delta_chi2);
        // The best time is a transit's.
        let t = found.best.transit_time().as_days();
        let off = ((t - 1.3 + 0.5 * p).rem_euclid(p) - 0.5 * p).abs();
        assert!(off < 0.0126, "{off}");
    }
}

#[test]
fn the_limits() {
    let lc = curve(3.7, 1.0, 9);
    let c = candidate(3.7, 1.3);
    for width in [0.0, -0.1, 0.6, f64::NAN] {
        assert_eq!(
            search_period(&lc, &c, Shape::BOX, width),
            Err(VetError::WidthOutOfRange)
        );
    }
    // A brightening: no dip at any period or time.
    let bump = curve(3.7, -1.0, 9);
    assert_eq!(
        search_period(&bump, &c, Shape::BOX, 0.001),
        Err(VetError::NoPointsInTransit)
    );
    // The widest search allowed runs.
    let short = LightCurve::from_days(
        (0..200).map(|i| f64::from(i) * 0.05).collect(),
        (0..200)
            .map(|i| if i % 37 == 26 { 0.99 } else { 1.0 })
            .collect(),
        vec![1e-3; 200],
    )
    .unwrap();
    let found = search_period(&short, &candidate(1.85, 1.3), Shape::BOX, 0.5).unwrap();
    assert!((found.best.period().as_days() - 1.85).abs() < 0.01);
}

// The evidence transit by transit: a planet's equal depths, transits in
// gaps and off the edge, an eclipsing binary's alternating depths, a
// candidate at half the true period, and the odd-even significance
// calibrated over many noise draws.

use rustellar_model::{ScaledSemiMajorAxis, TransitParams};
use rustellar_vet::{
    Candidate, Epoch, ImpactParameter, LightCurve, LimbDarkening, RadiusRatio, Shape, TimeDelta,
    VetError, evidence,
};

fn days(d: f64) -> TimeDelta {
    TimeDelta::from_days(d).unwrap()
}

fn at(t: f64) -> Epoch {
    Epoch::from_btjd(t).unwrap()
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

// k 0.05, b 0.5, a/R* 15, P 3.7 d, first transit at 1501.3: the default
// shape's planet, 1.74 hours long.
fn planet(period: f64) -> TransitParams {
    TransitParams::new(
        at(1501.3),
        days(period),
        RadiusRatio::new(0.05).unwrap(),
        ScaledSemiMajorAxis::new(15.0).unwrap(),
        ImpactParameter::new(0.5).unwrap(),
        LimbDarkening::quadratic(0.4804, 0.1867).unwrap(),
    )
    .unwrap()
}

// 27 days every 10 minutes from BTJD 1500, without the times `gap` holds.
fn times(gap: impl Fn(f64) -> bool) -> Vec<f64> {
    (0..3888)
        .map(|i| 1500.0 + f64::from(i) / 144.0)
        .filter(|&t| !gap(t))
        .collect()
}

fn curve(time: Vec<f64>, flux: impl Fn(f64) -> f64, noise: f64, seed: u64) -> LightCurve {
    let mut r = Random(seed);
    let f: Vec<f64> = time.iter().map(|&t| flux(t) + noise * r.normal()).collect();
    let n = time.len();
    LightCurve::from_btjd(time, f, vec![noise.max(1e-6); n]).unwrap()
}

fn candidate(period: f64, p: &TransitParams) -> Candidate {
    Candidate::new(days(period), at(1501.3), p.t14().unwrap()).unwrap()
}

// Every transit's own depth is the planet's, and the odd and even agree;
// a transit in a gap has no points and no depth; one cut by a gap edge has
// fewer than expected.
#[test]
fn a_planet_transit_by_transit() {
    let p = planet(3.7);
    // A gap over the third transit (1508.7), and one ending mid-way
    // through the fifth (1516.1).
    let t = times(|t| (1508.0..1509.5).contains(&t) || (1515.5..1516.1).contains(&t));
    let lc = curve(t, |t| p.flux_at(at(t)), 0.0, 0);
    let e = evidence(&lc, &candidate(3.7, &p), Shape::default()).unwrap();
    let centre = 1.0 - p.flux_at(at(1501.3));
    let epochs: Vec<i64> = e.transits.iter().map(|t| t.epoch).collect();
    assert_eq!(epochs, [0, 1, 2, 3, 4, 5, 6]);
    assert_eq!(e.with_data(), 6);
    assert!((e.cadence.unwrap().as_days() - 1.0 / 144.0).abs() < 1e-12);
    for t in &e.transits {
        let expected = t.expected.unwrap();
        assert!((expected - 10.45).abs() < 0.01, "{expected}");
        if t.epoch == 2 {
            assert_eq!((t.points, t.depth, t.depth_err), (0, None, None));
            assert_eq!(t.coverage(), Some(0.0));
            continue;
        }
        assert!((t.depth.unwrap() / centre - 1.0).abs() < 1e-10, "{t:?}");
        assert!(t.depth_err.unwrap() > 0.0);
        let coverage = t.coverage().unwrap();
        if t.epoch == 4 {
            assert!((0.3..0.7).contains(&coverage), "{coverage}");
        } else {
            assert!((0.9..1.1).contains(&coverage), "{coverage}");
        }
    }
    let oe = e.odd_even.unwrap();
    assert!(oe.difference.abs() < 1e-10 * centre);
    assert_eq!((oe.odd_transits, oe.even_transits), (3, 3));
    assert!((e.transits[1].time.as_btjd() - 1505.0).abs() < 1e-12);
}

// A transit before the first time but overlapping it is predicted; the
// candidate's epochs counted from its own time, negative before it.
#[test]
fn transits_at_the_edges_and_before_the_epoch() {
    let p = planet(3.7);
    // From 1501.3, mid-way through the first transit, to 1527.
    let t = times(|t| t < 1501.3);
    let lc = curve(t, |t| p.flux_at(at(t)), 0.0, 0);
    // The candidate's time a few periods on: epochs -2 to 4.
    let c = Candidate::new(days(3.7), at(1501.3 + 2.0 * 3.7), p.t14().unwrap()).unwrap();
    let e = evidence(&lc, &c, Shape::default()).unwrap();
    let epochs: Vec<i64> = e.transits.iter().map(|t| t.epoch).collect();
    assert_eq!(epochs, [-2, -1, 0, 1, 2, 3, 4]);
    let first = e.transits[0].coverage().unwrap();
    assert!((0.4..0.65).contains(&first), "{first}");
    // Epochs -2, 0, 2, 4 are even.
    let oe = e.odd_even.unwrap();
    assert_eq!((oe.odd_transits, oe.even_transits), (3, 4));
}

// An eclipsing binary at twice the period: depths alternate, and the
// odd-even difference stands out by many sigma, its sign the odd's.
#[test]
fn an_eclipsing_binary() {
    let t = times(|_| false);
    let flux = |t: f64| {
        let n = ((t - 1501.3) / 3.7).round();
        let depth = if n.rem_euclid(2.0) == 1.0 { 3e-3 } else { 2e-3 };
        if (t - 1501.3 - 3.7 * n).abs() < 0.05 {
            1.0 - depth
        } else {
            1.0
        }
    };
    let lc = curve(t, flux, 5e-4, 1);
    let c = Candidate::new(days(3.7), at(1501.3), days(0.1)).unwrap();
    let e = evidence(&lc, &c, Shape::BOX).unwrap();
    let oe = e.odd_even.unwrap();
    assert!((oe.odd - 3e-3).abs() < 4.0 * oe.odd_err);
    assert!((oe.even - 2e-3).abs() < 4.0 * oe.even_err);
    assert!(oe.sigma() > 10.0, "{}", oe.sigma());
    for t in &e.transits {
        let want = if t.epoch % 2 == 0 { 2e-3 } else { 3e-3 };
        assert!((t.depth.unwrap() - want).abs() < 5.0 * t.depth_err.unwrap());
    }
}

// At half the true period every other predicted transit is empty: the odd
// depths near zero, the even the planet's, and many sigma apart.
#[test]
fn half_the_true_period() {
    let p = planet(7.4);
    let lc = curve(times(|_| false), |t| p.flux_at(at(t)), 2e-4, 2);
    let c = Candidate::new(days(3.7), at(1501.3), p.t14().unwrap()).unwrap();
    let e = evidence(&lc, &c, Shape::default()).unwrap();
    let oe = e.odd_even.unwrap();
    let centre = 1.0 - p.flux_at(at(1501.3));
    assert!(oe.odd.abs() < 4.0 * oe.odd_err, "{oe:?}");
    assert!((oe.even - centre).abs() < 4.0 * oe.even_err, "{oe:?}");
    assert!(oe.sigma() < -10.0, "{}", oe.sigma());
}

// For a planet (equal depths) the odd-even significance is a standard
// normal: over 1,000 noise draws its mean is near 0 and its spread near 1,
// and each transit's depth error matches its scatter.
#[test]
fn the_significance_is_calibrated() {
    let p = planet(3.7);
    let t = times(|_| false);
    let c = candidate(3.7, &p);
    let draws = 1000;
    let (mut sigmas, mut first, mut first_err) = (vec![], vec![], 0.0);
    for seed in 0..draws {
        let lc = curve(t.clone(), |t| p.flux_at(at(t)), 1e-3, 100 + seed);
        let e = evidence(&lc, &c, Shape::default()).unwrap();
        sigmas.push(e.odd_even.unwrap().sigma());
        first.push(e.transits[0].depth.unwrap());
        first_err += e.transits[0].depth_err.unwrap();
    }
    let n = f64::from(u32::try_from(draws).unwrap());
    let mean = sigmas.iter().sum::<f64>() / n;
    let spread = (sigmas.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
    assert!(mean.abs() < 0.1, "{mean}");
    assert!((spread - 1.0).abs() < 0.07, "{spread}");
    let mean = first.iter().sum::<f64>() / n;
    let scatter = (first.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
    assert!((scatter / (first_err / n) - 1.0).abs() < 0.07, "{scatter}");
}

#[test]
fn what_cannot_be_measured() {
    let lc = LightCurve::from_btjd(vec![0.0, 1.0], vec![1.0, 1.0], vec![1e-3; 2]).unwrap();
    let late = Candidate::new(days(10.0), at(5.0), days(0.1)).unwrap();
    assert_eq!(
        evidence(&lc, &late, Shape::BOX),
        Err(VetError::NoPointsInTransit)
    );
    let wide = Candidate::new(days(1.0), at(0.0), days(0.6)).unwrap();
    assert_eq!(
        evidence(&lc, &wide, Shape::default()),
        Err(VetError::DurationTooLongForShape)
    );
    // One point, in transit: no baseline.
    let one = LightCurve::from_btjd(vec![0.0], vec![0.99], vec![1e-3]).unwrap();
    let c = Candidate::new(days(10.0), at(0.0), days(0.1)).unwrap();
    assert_eq!(
        evidence(&one, &c, Shape::BOX),
        Err(VetError::NoPointsOutOfTransit)
    );
    // Two: one transit, no odd and even.
    let two = LightCurve::from_btjd(vec![0.0, 0.5], vec![0.99, 1.0], vec![1e-3; 2]).unwrap();
    let e = evidence(&two, &c, Shape::BOX).unwrap();
    assert_eq!(e.odd_even, None);
    assert_eq!(e.transits.len(), 1);
    assert!(e.cadence.is_some());
}

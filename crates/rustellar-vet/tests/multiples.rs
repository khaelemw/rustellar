// The transits split n ways: a planet at the candidate's period equal in
// every split; at two and three times it, every second or third group
// empty and the multiple named; an irregular pattern named by none; the
// equality test calibrated on noise draws; groups in gaps; errors.

use rustellar_vet::{Candidate, Epoch, LightCurve, Shape, TimeDelta, VetError, multiples};

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

const P: f64 = 2.0;
const T0: f64 = 0.5;

// 54 days every 10 minutes, without the times `gap` holds; a 2e-3 box
// transit 0.1 days long at each epoch `deep` holds, every 2 days from
// 0.5; white noise `noise`.
fn curve(
    deep: impl Fn(i64) -> bool,
    gap: impl Fn(f64) -> bool,
    noise: f64,
    seed: u64,
) -> LightCurve {
    let mut r = Random(seed);
    let time: Vec<f64> = (0..7776)
        .map(|i| f64::from(i) / 144.0)
        .filter(|&t| !gap(t))
        .collect();
    let flux = time
        .iter()
        .map(|&t| {
            #[expect(clippy::cast_possible_truncation, reason = "a transit number")]
            let e = ((t - T0) / P).round() as i64;
            let inside = (t - T0 - P * e as f64).abs() < 0.05;
            1.0 - if inside && deep(e) { 2e-3 } else { 0.0 } + noise * r.normal()
        })
        .collect();
    let n = time.len();
    LightCurve::from_days(time, flux, vec![noise.max(1e-6); n]).unwrap()
}

fn candidate() -> Candidate {
    Candidate::new(
        TimeDelta::from_days(P).unwrap(),
        Epoch::from_days(T0).unwrap(),
        TimeDelta::from_days(0.1).unwrap(),
    )
    .unwrap()
}

#[test]
fn a_planet_is_equal_in_every_split() {
    let lc = curve(|_| true, |_| false, 3e-4, 1);
    let split = multiples(&lc, &candidate(), Shape::BOX, 8).unwrap();
    assert_eq!(split.len(), 7);
    for (m, n) in split.iter().zip(2u16..) {
        assert_eq!(m.n, n);
        assert_eq!(m.groups.len(), usize::from(n));
        assert_eq!(m.dof, usize::from(n) - 1);
        assert_eq!(m.implied(3.0), Some((1, 0)), "{m:?}");
        assert_eq!(m.significant(3.0).len(), usize::from(n));
        assert!(m.chi2 < 30.0, "{m:?}");
        for g in m.groups.iter().flatten() {
            assert!((g.depth - 2e-3).abs() < 5.0 * g.depth_err);
            assert!(g.points > 0);
        }
    }
}

// Twice the period from epoch 1 and three times from epoch 2: the right
// multiple and class in the splits it divides, none empty in the others.
#[test]
fn twice_and_three_times_the_period() {
    let lc = curve(|e| e.rem_euclid(2) == 1, |_| false, 3e-4, 2);
    let split = multiples(&lc, &candidate(), Shape::BOX, 8).unwrap();
    for m in &split {
        let want = if m.n % 2 == 0 { (2, 1) } else { (1, 0) };
        assert_eq!(m.implied(3.0), Some(want), "{m:?}");
        // Even splits separate the full transits from the empty.
        assert!(m.n % 2 == 1 || m.chi2 > 1000.0, "{m:?}");
    }
    let lc = curve(|e| e.rem_euclid(3) == 2, |_| false, 3e-4, 3);
    let split = multiples(&lc, &candidate(), Shape::BOX, 6).unwrap();
    assert_eq!(split[1].implied(3.0), Some((3, 2)));
    assert_eq!(split[4].implied(3.0), Some((3, 2)));
    assert_eq!(split[0].implied(3.0), Some((1, 0)));
}

// Deep in groups 0 and 1 of 4: no single class, so no multiple; nothing
// deep at all, likewise.
#[test]
fn patterns_no_multiple_explains() {
    let lc = curve(|e| e.rem_euclid(4) < 2, |_| false, 3e-4, 4);
    let split = multiples(&lc, &candidate(), Shape::BOX, 4).unwrap();
    assert_eq!(split[2].significant(3.0), [0, 1]);
    assert_eq!(split[2].implied(3.0), None);
    let flat = curve(|_| false, |_| false, 3e-4, 5);
    let split = multiples(&flat, &candidate(), Shape::BOX, 3).unwrap();
    assert!(split.iter().all(|m| m.implied(3.0).is_none()));
}

// A group with no points (every epoch 3 mod 4 in a gap) is left out: the
// rest decide, and the test has a degree of freedom fewer.
#[test]
fn a_group_in_gaps() {
    let gap = |t: f64| {
        let e = ((t - T0) / P).round();
        e.rem_euclid(4.0) == 3.0 && (t - T0 - P * e).abs() < 0.3
    };
    let lc = curve(|e| e.rem_euclid(2) == 0, gap, 3e-4, 6);
    let split = multiples(&lc, &candidate(), Shape::BOX, 4).unwrap();
    let four = &split[2];
    assert!(four.groups[3].is_none());
    assert_eq!(four.dof, 2);
    assert_eq!(four.implied(3.0), Some((2, 0)));
}

// For a planet the equality test is chi-squared: over 300 draws, its mean
// near its degrees of freedom.
#[test]
fn the_equality_test_is_calibrated() {
    let mut sums = [0.0; 3];
    for seed in 0..300 {
        let lc = curve(|_| true, |_| false, 1e-3, 100 + seed);
        let split = multiples(&lc, &candidate(), Shape::BOX, 4).unwrap();
        for (s, m) in sums.iter_mut().zip(&split) {
            *s += m.chi2;
        }
    }
    for (s, dof) in sums.iter().zip([1.0, 2.0, 3.0]) {
        let mean = s / 300.0;
        assert!((mean - dof).abs() < 0.3 * dof.max(1.0), "{mean} {dof}");
    }
}

#[test]
fn none_or_errors() {
    let lc = curve(|_| true, |_| false, 3e-4, 7);
    assert!(
        multiples(&lc, &candidate(), Shape::BOX, 1)
            .unwrap()
            .is_empty()
    );
    let empty = LightCurve::from_days(vec![0.2, 0.3], vec![1.0; 2], vec![1e-3; 2]).unwrap();
    assert_eq!(
        multiples(&empty, &candidate(), Shape::BOX, 3),
        Err(VetError::NoPointsInTransit)
    );
    let long = Candidate::new(
        TimeDelta::from_days(P).unwrap(),
        Epoch::from_days(T0).unwrap(),
        TimeDelta::from_days(1.5).unwrap(),
    )
    .unwrap();
    assert_eq!(
        multiples(&lc, &long, Shape::default(), 3),
        Err(VetError::DurationTooLongForShape)
    );
}

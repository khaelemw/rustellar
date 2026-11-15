// One box, fitted exactly: the worked example, the identities between the
// statistics, what a shifted flux or scaled weights should and should not
// change, the in-transit test against exact arithmetic, and the errors.

use rstest::rstest;
use rustellar_bls::{BlsError, BoxFit, TransitBox};
use rustellar_core::{Epoch, LightCurve, TimeDelta};

// ==========================
// Helper Functions
// ==========================

const IN_TRANSIT: &str = include_str!("reference/in_transit.csv");

// SplitMix64: a small, seeded generator, so the tests need no dependency.
struct Random(u64);

impl Random {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut x = self.0;
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        x ^ (x >> 31)
    }

    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        let unit = (self.next_u64() >> 11) as f64 / (1_u64 << 53) as f64;
        lo + (hi - lo) * unit
    }
}

fn days(value: f64) -> TimeDelta {
    TimeDelta::from_days(value).unwrap()
}

fn btjd(value: f64) -> Epoch {
    Epoch::from_btjd(value).unwrap()
}

fn transit(period: f64, duration: f64, transit_time: f64) -> TransitBox {
    TransitBox::new(days(period), days(duration), btjd(transit_time)).unwrap()
}

// Eight points a day apart, the third and fourth one lower, all weights 1.
fn worked_example() -> (LightCurve, TransitBox) {
    let lc = LightCurve::from_btjd(
        (0..8).map(f64::from).collect(),
        vec![0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
        vec![1.0; 8],
    )
    .unwrap();
    (lc, transit(100.0, 2.0, 2.5))
}

// A month of 2-minute-ish cadence with noise, a 3-day orbit and a box at a
// random place: flux near 1, errors near 1e-3.
fn random_light_curve(rng: &mut Random, n: usize) -> LightCurve {
    let mut time = Vec::with_capacity(n);
    let mut t = 1325.0;
    for _ in 0..n {
        t += rng.uniform(0.5, 1.5) * 27.0 / n as f64;
        time.push(t);
    }
    let flux = (0..n).map(|_| 1.0 + rng.uniform(-3e-3, 3e-3)).collect();
    let flux_err = (0..n).map(|_| rng.uniform(5e-4, 2e-3)).collect();
    LightCurve::from_btjd(time, flux, flux_err).unwrap()
}

fn random_transit(rng: &mut Random) -> TransitBox {
    transit(
        rng.uniform(1.0, 5.0),
        rng.uniform(0.05, 0.3),
        rng.uniform(1325.0, 1330.0),
    )
}

fn relative(a: f64, b: f64) -> f64 {
    (a / b - 1.0).abs()
}

// ==========================
// The worked example
// ==========================

#[test]
fn worked_example_exactly() {
    let (lc, transit) = worked_example();
    let fit = transit.fit(&lc).unwrap();
    assert_eq!(fit.n_in, 2);
    assert_eq!(fit.flux_in, -1.0);
    assert_eq!(fit.flux_out, 0.0);
    assert_eq!(fit.depth, 1.0);
    assert_eq!(fit.weight_in, 2.0);
    assert_eq!(fit.weight_out, 6.0);
    assert_eq!(fit.depth_err, 0.816_496_580_927_726);
    assert_eq!(fit.snr, 1.224_744_871_391_589);
    assert_eq!(fit.delta_chi2, 1.5);
    assert_eq!(fit.likelihood, 1.0);
}

// ==========================
// Identities on random data
// ==========================

// snr^2 = delta_chi2, and astropy's likelihood = delta_chi2 W / (2 W_out).
#[test]
fn identities_hold_on_random_data() {
    let mut rng = Random(2);
    for _ in 0..200 {
        let lc = random_light_curve(&mut rng, 2_000);
        let fit = random_transit(&mut rng).fit(&lc).unwrap();
        let weight = fit.weight_in + fit.weight_out;
        assert!(
            relative(fit.snr * fit.snr, fit.delta_chi2) < 1e-15,
            "{fit:?}"
        );
        let likelihood = 0.5 * fit.delta_chi2 * weight / fit.weight_out;
        assert!(relative(fit.likelihood, likelihood) < 1e-15, "{fit:?}");
    }
}

// The fall in chi-squared, worked out the long way: chi-squared about the
// overall weighted mean, less chi-squared about the two box levels.
#[test]
fn delta_chi2_is_the_fall_in_chi_squared() {
    let mut rng = Random(3);
    for _ in 0..50 {
        let lc = random_light_curve(&mut rng, 500);
        let transit = random_transit(&mut rng);
        let fit = transit.fit(&lc).unwrap();
        let points = || {
            lc.time()
                .iter()
                .zip(lc.flux())
                .zip(lc.flux_err())
                .map(|((&t, &y), &e)| (t, y, 1.0 / (e * e)))
        };
        let weight: f64 = points().map(|(_, _, w)| w).sum();
        let mean = points().map(|(_, y, w)| w * y).sum::<f64>() / weight;
        let flat: f64 = points().map(|(_, y, w)| w * (y - mean).powi(2)).sum();
        let boxed: f64 = points()
            .map(|(t, y, w)| {
                let level = if transit.contains(t) {
                    fit.flux_in
                } else {
                    fit.flux_out
                };
                w * (y - level).powi(2)
            })
            .sum();
        // The long way subtracts two sums of about `n` each, so it is good
        // only to a few roundings of `flat`; delta_chi2 can be far smaller.
        assert!(
            (flat - boxed - fit.delta_chi2).abs() < 1e-12 * flat,
            "{fit:?}"
        );
    }
}

// ==========================
// What should and should not change
// ==========================

// Adding a constant moves both levels by it and nothing else.
#[rstest]
#[case(1.0)]
#[case(-0.75)]
#[case(1e4)]
fn shifted_flux(#[case] shift: f64) {
    let mut rng = Random(4);
    let lc = random_light_curve(&mut rng, 3_000);
    let transit = random_transit(&mut rng);
    let shifted = LightCurve::new(
        lc.time().to_vec(),
        lc.flux().iter().map(|y| y + shift).collect(),
        lc.flux_err().to_vec(),
    )
    .unwrap();
    let (a, b) = (transit.fit(&lc).unwrap(), transit.fit(&shifted).unwrap());

    // The shifted flux is itself rounded, at the size of `shift`, so the
    // depth can move by about that much.
    let tolerance = 1e-15 * shift.abs().max(1.0) / a.depth.abs();
    assert!(relative(b.depth, a.depth) < tolerance, "{a:?}\n{b:?}");
    assert!(relative(b.delta_chi2, a.delta_chi2) < 2.0 * tolerance);
    assert!((b.flux_in - (a.flux_in + shift)).abs() < 1e-15 * shift.abs().max(1.0) * 4.0);
    assert!((b.flux_out - (a.flux_out + shift)).abs() < 1e-15 * shift.abs().max(1.0) * 4.0);
    assert_eq!(
        (b.weight_in, b.weight_out, b.depth_err),
        (a.weight_in, a.weight_out, a.depth_err)
    );
    assert_eq!(b.n_in, a.n_in);
}

// Halving every error multiplies every weight by 4, exactly: the levels and
// depth stay, the error halves, snr doubles, and delta_chi2 and the
// likelihood go up 4 times.
#[test]
fn scaled_weights() {
    let mut rng = Random(5);
    let lc = random_light_curve(&mut rng, 3_000);
    let transit = random_transit(&mut rng);
    let halved = LightCurve::new(
        lc.time().to_vec(),
        lc.flux().to_vec(),
        lc.flux_err().iter().map(|e| 0.5 * e).collect(),
    )
    .unwrap();
    let (a, b) = (transit.fit(&lc).unwrap(), transit.fit(&halved).unwrap());
    assert_eq!(b.flux_in, a.flux_in);
    assert_eq!(b.flux_out, a.flux_out);
    assert_eq!(b.depth, a.depth);
    assert_eq!(b.depth_err, 0.5 * a.depth_err);
    assert_eq!(b.snr, 2.0 * a.snr);
    assert_eq!(b.delta_chi2, 4.0 * a.delta_chi2);
    assert_eq!(b.likelihood, 4.0 * a.likelihood);
    assert_eq!(b.weight_in, 4.0 * a.weight_in);
}

// A shallow box on flux near 1: centring on the mean first keeps the depth's
// digits. Here the depth is 1e-6 exactly, in and out of transit.
#[test]
fn shallow_depth_keeps_its_digits() {
    let n = 1_000;
    let time: Vec<f64> = (0..n).map(|i| 1325.0 + 0.02 * f64::from(i)).collect();
    let transit = transit(3.0, 0.2, 1326.0);
    let flux = time
        .iter()
        .map(|&t| {
            if transit.contains(btjd(t)) {
                0.999_999
            } else {
                1.0
            }
        })
        .collect();
    let lc = LightCurve::from_btjd(time, flux, vec![1e-3; n as usize]).unwrap();
    let fit = transit.fit(&lc).unwrap();
    let exact = 1.0 - 0.999_999; // the depth between the two floats
    assert!(
        relative(fit.depth, exact) < 1e-15,
        "{} vs {exact}",
        fit.depth
    );
}

// ==========================
// In transit, against exact arithmetic
// ==========================

// (period, duration, transit_time, time, exact, astropy) per row.
fn in_transit_rows() -> Vec<(f64, f64, f64, f64, bool, bool)> {
    IN_TRANSIT
        .lines()
        .filter(|line| !line.starts_with('#'))
        .skip(1)
        .map(|line| {
            let v: Vec<&str> = line.split(',').collect();
            (
                v[1].parse().unwrap(),
                v[2].parse().unwrap(),
                v[3].parse().unwrap(),
                v[4].parse().unwrap(),
                v[5] == "1",
                v[6] == "1",
            )
        })
        .collect()
}

// Times within 1e-12 days (86 ns) of a box edge, 1,000 to 3,000 orbits from
// the mid-transit time, near BTJD 3000.
#[test]
fn in_transit_agrees_with_exact_arithmetic() {
    let rows = in_transit_rows();
    assert_eq!(rows.len(), 2_000);
    for (period, duration, transit_time, time, exact, _) in rows {
        let ours = transit(period, duration, transit_time).contains(btjd(time));
        assert_eq!(
            ours, exact,
            "P {period}, D {duration}, t0 {transit_time}, t {time}"
        );
    }
}

// astropy's mask, `abs((t - t0 + P/2) % P - P/2) < D/2`, written out here
// with Python's `%` (`rem_euclid` for a positive period), gets 85 of them
// wrong.
#[test]
fn astropys_mask_does_not() {
    let mut wrong = 0;
    for (period, duration, transit_time, time, exact, astropy) in in_transit_rows() {
        let half = 0.5 * period;
        let theirs =
            ((time - transit_time + half).rem_euclid(period) - half).abs() < 0.5 * duration;
        assert_eq!(theirs, astropy, "the formula is astropy's");
        wrong += usize::from(theirs != exact);
    }
    assert_eq!(wrong, 85);
}

// The edge itself is out of transit, as in astropy.
#[test]
fn edges_are_outside() {
    let transit = transit(2.0, 0.5, 10.0);
    assert!(!transit.contains(btjd(9.75)));
    assert!(!transit.contains(btjd(10.25)));
    assert!(transit.contains(btjd(10.249_999_999_999_998)));
    assert!(!transit.contains(btjd(12.25)));
    assert!(transit.contains(btjd(6.0)));
}

// ==========================
// Errors
// ==========================

#[rstest]
#[case::period_zero(0.0, 0.1, BlsError::PeriodNotPositive)]
#[case::period_negative(-1.0, 0.1, BlsError::PeriodNotPositive)]
#[case::duration_zero(1.0, 0.0, BlsError::DurationNotPositive)]
#[case::duration_negative(1.0, -0.1, BlsError::DurationNotPositive)]
#[case::duration_equal(1.0, 1.0, BlsError::DurationNotShorterThanPeriod)]
#[case::duration_longer(1.0, 1.5, BlsError::DurationNotShorterThanPeriod)]
fn bad_box(#[case] period: f64, #[case] duration: f64, #[case] error: BlsError) {
    assert_eq!(
        TransitBox::new(days(period), days(duration), btjd(0.0)),
        Err(error)
    );
}

#[test]
fn accessors_return_what_was_given() {
    let transit = transit(3.0, 0.1, 1325.5);
    assert_eq!(transit.period(), days(3.0));
    assert_eq!(transit.duration(), days(0.1));
    assert_eq!(transit.transit_time(), btjd(1325.5));
}

#[test]
fn box_with_no_points() {
    let (lc, _) = worked_example();
    assert_eq!(
        transit(100.0, 0.5, 50.0).fit(&lc),
        Err(BlsError::NoPointsInTransit)
    );
}

#[test]
fn box_with_every_point() {
    let (lc, _) = worked_example();
    assert_eq!(
        transit(100.0, 20.0, 3.5).fit(&lc),
        Err(BlsError::NoPointsOutOfTransit)
    );
}

// An error of 1e-160 gives a weight of 1e320, beyond an f64; one of 1e155
// gives 1e-310, which has lost digits; one of 1e200 gives 0. Each is an
// error, wherever the box is.
#[rstest]
fn weights_out_of_range(
    #[values(1e-160, 1e-155, 1e155, 1e200)] bad: f64,
    #[values(0.0, 1.0)] transit_time: f64,
) {
    let lc = LightCurve::from_btjd(
        vec![0.0, 1.0, 2.0],
        vec![1.0, 0.9, 1.0],
        vec![bad, 1.0, 1.0],
    )
    .unwrap();
    assert_eq!(
        transit(10.0, 0.5, transit_time).fit(&lc),
        Err(BlsError::OutOfRange)
    );
}

// The largest and smallest errors whose weights are normal floats still fit.
#[rstest]
#[case(1e-153)]
#[case(1e153)]
fn weights_at_the_edge_of_range(#[case] err: f64) {
    let lc = LightCurve::from_btjd(
        vec![0.0, 1.0, 2.0, 3.0],
        vec![1.0, 0.9, 1.0, 1.0],
        vec![err; 4],
    )
    .unwrap();
    let fit = transit(10.0, 0.5, 1.0).fit(&lc).unwrap();
    assert!((fit.depth - 0.1).abs() < 1e-15, "{fit:?}");
}

// The fit is plain data, so it can be copied and compared.
#[test]
fn fit_is_plain_data() {
    let (lc, transit) = worked_example();
    let fit: BoxFit = transit.fit(&lc).unwrap();
    let copy = fit;
    assert_eq!(copy, fit);
}

// Weights of 1e300 and a dip of 1e5: the likelihood, W_in depth^2 / 2, is
// about 1e310.
#[test]
fn statistics_that_overflow() {
    let lc =
        LightCurve::from_btjd(vec![0.0, 1.0, 2.0], vec![1e5, 0.0, 1e5], vec![1e-150; 3]).unwrap();
    assert_eq!(transit(10.0, 0.5, 1.0).fit(&lc), Err(BlsError::OutOfRange));
}

// Four weights of 1e308, each a normal float, sum past the largest f64.
#[test]
fn weights_that_sum_out_of_range() {
    let lc = LightCurve::from_btjd(
        vec![0.0, 1.0, 2.0, 3.0],
        vec![1.0, 0.9, 1.0, 1.0],
        vec![1e-154; 4],
    )
    .unwrap();
    assert_eq!(transit(10.0, 0.5, 1.0).fit(&lc), Err(BlsError::OutOfRange));
}

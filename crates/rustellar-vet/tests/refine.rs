// The refinement of mid-transit time and duration: against a dense grid of
// plain fits (the box's must match or beat every grid point, being exact),
// and the truth recovered from a candidate that is off.

use rustellar_model::{ScaledSemiMajorAxis, TransitParams};
use rustellar_vet::{
    Candidate, Epoch, ImpactParameter, LightCurve, LimbDarkening, RadiusRatio, Shape, TimeDelta,
    VetError, fit_depth, refine,
};

fn days(d: f64) -> TimeDelta {
    TimeDelta::from_days(d).unwrap()
}

fn at(t: f64) -> Epoch {
    Epoch::from_btjd(t).unwrap()
}

// A small deterministic generator (SplitMix64) and normal draws.
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

// The planet of the fit tests: k 0.05, b 0.5, a/R* 15, P 3.7 d.
fn planet() -> TransitParams {
    TransitParams::new(
        at(1501.3),
        days(3.7),
        RadiusRatio::new(0.05).unwrap(),
        ScaledSemiMajorAxis::new(15.0).unwrap(),
        ImpactParameter::new(0.5).unwrap(),
        LimbDarkening::quadratic(0.4804, 0.1867).unwrap(),
    )
    .unwrap()
}

// 12 days every 30 minutes from BTJD 1500, the planet's transits in white
// noise of `noise`.
fn light_curve(seed: u64, noise: f64) -> LightCurve {
    let p = planet();
    let mut r = Random(seed);
    let time: Vec<f64> = (0..576).map(|i| 1500.0 + f64::from(i) / 48.0).collect();
    let flux = time
        .iter()
        .map(|&t| p.flux_at(at(t)) + noise * r.normal())
        .collect();
    LightCurve::from_btjd(time, flux, vec![noise.max(1e-6); 576]).unwrap()
}

// The best drop in chi-squared on a dense grid of mid-transit times (within
// half a duration) and durations (half to twice), by plain fits.
fn dense(lc: &LightCurve, c: &Candidate, shape: Shape, steps: u32) -> f64 {
    let (t0, d) = (c.transit_time().as_btjd(), c.duration().as_days());
    let mut best = f64::NEG_INFINITY;
    for a in 0..=steps {
        let shift = d * (f64::from(a) / f64::from(steps) - 0.5);
        for b in 0..=steps {
            let length = d * 0.5 * 4f64.powf(f64::from(b) / f64::from(steps));
            let Ok(trial) = Candidate::new(c.period(), at(t0 + shift), days(length)) else {
                continue;
            };
            if let Ok(fit) = fit_depth(lc, &trial, shape) {
                if fit.depth > 0.0 {
                    best = best.max(fit.delta_chi2);
                }
            }
        }
    }
    best
}

// The box's refinement is exact: no grid point does better, on noisy
// curves from candidates a little off.
#[test]
fn the_box_beats_every_grid_point() {
    for seed in 1..=4_u32 {
        let lc = light_curve(u64::from(seed), 4e-4);
        let c = Candidate::new(days(3.7), at(1501.3 + 0.01 * f64::from(seed)), days(0.08)).unwrap();
        let refined = refine(&lc, &c, Shape::BOX).unwrap();
        let grid = dense(&lc, &c, Shape::BOX, 300);
        assert!(
            refined.fit.delta_chi2 >= grid - 1e-9 * grid,
            "{seed}: {} {grid}",
            refined.fit.delta_chi2
        );
        // Within the ranges.
        let shift = refined.candidate.transit_time().as_btjd() - c.transit_time().as_btjd();
        let length = refined.candidate.duration().as_days();
        assert!(shift.abs() <= 0.04 && (0.04..=0.16).contains(&length));
        assert_eq!(refined.candidate.depth(), Some(refined.fit.depth));
    }
}

// The limb-darkened refinement finds the dense grid's best, or a little
// better between its points.
#[test]
fn the_transit_matches_a_dense_grid() {
    for seed in 1..=2_u64 {
        let lc = light_curve(seed, 4e-4);
        let c = Candidate::new(days(3.7), at(1501.31), days(0.06)).unwrap();
        let refined = refine(&lc, &c, Shape::default()).unwrap();
        let grid = dense(&lc, &c, Shape::default(), 120);
        assert!(
            refined.fit.delta_chi2 >= grid - 1e-6 * grid,
            "{seed}: {} {grid}",
            refined.fit.delta_chi2
        );
    }
}

// A limb-darkened transit almost a third of its period long, whose trial
// transits wrap past half the period, with a gap over its middle that
// leaves the shortest trials no points: the refinement still matches a
// dense grid.
#[test]
fn a_long_transit_over_a_gap() {
    let mut r = Random(9);
    let time: Vec<f64> = (0..1440)
        .map(|i| f64::from(i) / 144.0)
        .filter(|&t| ((t - 0.5 + 0.5).rem_euclid(1.0) - 0.5).abs() >= 0.06)
        .collect();
    let flux: Vec<f64> = time
        .iter()
        .map(|&t| {
            let dip = ((t - 0.5 + 0.5).rem_euclid(1.0) - 0.5).abs() < 0.12;
            1.0 - if dip { 3e-3 } else { 0.0 } + 4e-4 * r.normal()
        })
        .collect();
    let n = time.len();
    let lc = LightCurve::from_btjd(time, flux, vec![4e-4; n]).unwrap();
    let c = Candidate::new(days(1.0), at(0.5), days(0.3)).unwrap();
    let refined = refine(&lc, &c, Shape::default()).unwrap();
    let grid = dense(&lc, &c, Shape::default(), 80);
    assert!(
        refined.fit.delta_chi2 >= grid - 1e-6 * grid,
        "{} {grid}",
        refined.fit.delta_chi2
    );
}

// Without noise the refinement walks back to the planet's own time and
// duration, and its depth.
#[test]
fn the_truth_from_a_candidate_off() {
    let p = planet();
    let lc = light_curve(1, 0.0);
    let truth = p.t14().unwrap().as_days();
    let c = Candidate::new(days(3.7), at(1501.3 + 0.3 * truth), days(1.4 * truth)).unwrap();
    let refined = refine(&lc, &c, Shape::default()).unwrap();
    let t0 = refined.candidate.transit_time().as_btjd();
    let length = refined.candidate.duration().as_days();
    assert!((t0 - 1501.3).abs() < 2e-5, "{t0}");
    assert!((length / truth - 1.0).abs() < 1e-3, "{length} {truth}");
    let centre = 1.0 - p.flux_at(at(1501.3));
    assert!((refined.fit.depth / centre - 1.0).abs() < 1e-3);
}

#[test]
fn no_dip_to_find() {
    let time: Vec<f64> = (0..200).map(|i| f64::from(i) / 48.0).collect();
    let flat = LightCurve::from_btjd(time.clone(), vec![1.0; 200], vec![1e-3; 200]).unwrap();
    let c = Candidate::new(days(1.0), at(0.5), days(0.1)).unwrap();
    for shape in [Shape::BOX, Shape::default()] {
        assert_eq!(refine(&flat, &c, shape), Err(VetError::NoPointsInTransit));
    }
    // A candidate with nothing in transit fails as the plain fit does.
    let late = Candidate::new(days(100.0), at(50.0), days(0.1)).unwrap();
    assert_eq!(
        refine(&flat, &late, Shape::BOX),
        Err(VetError::NoPointsInTransit)
    );
}

// A box longer than half the period can hold every point; that run leaves
// no baseline and is never chosen, whatever the rounding.
#[test]
fn a_run_of_every_point_is_no_box() {
    // Half a step off the hour, so every point is within half a period.
    let time: Vec<f64> = (0..48).map(|i| (f64::from(i) + 0.5) / 48.0).collect();
    let flux: Vec<f64> = time
        .iter()
        .map(|&t| if (t - 0.5).abs() < 0.3 { 0.99 } else { 1.0 })
        .collect();
    let lc = LightCurve::from_btjd(time, flux, vec![1e-3; 48]).unwrap();
    let c = Candidate::new(days(1.0), at(0.5), days(0.6)).unwrap();
    let refined = refine(&lc, &c, Shape::BOX).unwrap();
    assert!(refined.fit.points_in_transit < 48);
    assert!(
        (refined.fit.depth - 0.01).abs() < 1e-12,
        "{}",
        refined.fit.depth
    );
}

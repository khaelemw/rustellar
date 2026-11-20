// Properties that must hold for every valid input, checked at millions of
// random points over the whole valid range. The points come
// from a fixed-seed generator, so every run checks the same points and a
// failure can be reproduced.
//
// The ranges match tests/flux_random.rs: k log-uniform from 1e-6 to 100,
// separations across and beyond the transit, and limb darkening from
// Kipping's (2013) physical prior plus the rest of the allowed range.

use std::num::NonZeroUsize;

use rustellar_core::{Epoch, TimeDelta};
use rustellar_model::{
    Exposure, ImpactParameter, LimbDarkening, RadiusRatio, ScaledSemiMajorAxis, TransitParams,
    flux, flux_ring_sum,
};

// ==========================
// Helper Functions
// ==========================

// SplitMix64 (Steele, Lea & Flood 2014): small, fast, and good enough to
// spread test points; not for anything statistical.
struct Random(u64);

impl Random {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut x = self.0;
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        x ^ (x >> 31)
    }

    // Uniform in [0, 1).
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1_u64 << 53) as f64
    }

    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }

    fn log_uniform(&mut self, lo: f64, hi: f64) -> f64 {
        self.uniform(lo.ln(), hi.ln()).exp()
    }

    fn radius_ratio(&mut self) -> f64 {
        self.log_uniform(1e-6, 100.0)
    }

    // A separation from the centre to beyond the transit.
    fn separation(&mut self, k: f64) -> f64 {
        self.uniform(0.0, 1.2 * (1.0 + k))
    }

    // 70% Kipping (2013): q1, q2 uniform; the rest any allowed law in
    // [-1, 3] x [-2, 2].
    fn limb_darkening(&mut self) -> LimbDarkening {
        if self.unit() < 0.7 {
            return self.darkened();
        }
        loop {
            let (u1, u2) = (self.uniform(-1.0, 3.0), self.uniform(-2.0, 2.0));
            if let Ok(ld) = LimbDarkening::quadratic(u1, u2) {
                return ld;
            }
        }
    }

    // Kipping's (2013) prior: every law whose intensity falls from centre to
    // limb (u1 >= 0, u1 + 2 u2 >= 0, u1 + u2 <= 1).
    fn darkened(&mut self) -> LimbDarkening {
        let (q1, q2) = (self.unit(), self.unit());
        let root = q1.sqrt();
        LimbDarkening::quadratic(2.0 * root * q2, root * (1.0 - 2.0 * q2)).unwrap()
    }
}

fn k(k: f64) -> RadiusRatio {
    RadiusRatio::new(k).unwrap()
}

// This crate's measured error against exoplanet-core in each range of k
// (tests/flux_random.rs), used as the rounding allowed in the properties.
fn error_bound(k: f64) -> f64 {
    match k {
        k if k < 1.0 => 3e-13,
        k if k < 10.0 => 5e-12,
        k if k < 20.0 => 5e-11,
        k if k < 30.0 => 1e-10,
        _ => 3e-9,
    }
}

// A random transit: any valid orbit, with t0 anywhere from the Kepler era to
// well past TESS, and b up to just past a miss.
fn random_transit(rng: &mut Random) -> TransitParams {
    let radius_ratio = rng.log_uniform(1e-3, 10.0);
    let a_rs = rng.log_uniform(1.5, 300.0);
    let b = rng.uniform(0.0, (1.1 + radius_ratio).min(0.99 * a_rs));
    TransitParams::new(
        Epoch::from_days(rng.uniform(-2000.0, 5000.0)).unwrap(),
        TimeDelta::from_days(rng.log_uniform(0.2, 1000.0)).unwrap(),
        k(radius_ratio),
        ScaledSemiMajorAxis::new(a_rs).unwrap(),
        ImpactParameter::new(b).unwrap(),
        rng.limb_darkening(),
    )
    .unwrap()
}

// A time near one of the transits within 100 orbits of t0.
fn time_near_transit(rng: &mut Random, params: TransitParams) -> Epoch {
    let orbits = (rng.uniform(-100.0, 100.0)).round();
    let period = params.period().as_days();
    let half = params
        .t14()
        .map_or(0.05 * period, |t14| 0.6 * t14.as_days());
    params.t0() + params.period() * orbits + TimeDelta::from_days(rng.uniform(-half, half)).unwrap()
}

// ==========================
// Tests: flux
// ==========================

#[test]
fn flux_is_between_zero_and_one_and_even_in_z() {
    let mut rng = Random::new(1);
    for _ in 0..4_000_000 {
        let radius_ratio = rng.radius_ratio();
        let z = rng.separation(radius_ratio);
        let ld = rng.limb_darkening();
        let f = flux(z, k(radius_ratio), ld);
        assert!(
            (0.0..=1.0).contains(&f),
            "k {radius_ratio}, z {z}, {ld:?}: {f}"
        );
        assert_eq!(flux(-z, k(radius_ratio), ld), f);
    }
}

#[test]
fn flux_is_exactly_one_clear_of_the_star_and_zero_under_full_cover() {
    let mut rng = Random::new(2);
    for _ in 0..1_000_000 {
        let radius_ratio = rng.radius_ratio();
        let ld = rng.limb_darkening();
        let clear = (1.0 + radius_ratio) * rng.uniform(1.0, 10.0);
        assert_eq!(flux(clear, k(radius_ratio), ld), 1.0);
        if radius_ratio > 1.0 {
            let covered = (radius_ratio - 1.0) * rng.unit();
            assert_eq!(flux(covered, k(radius_ratio), ld), 0.0);
        }
    }
}

// The blocked light, (1 - F) times the disk integral, is linear in (u1, u2).
#[test]
fn blocked_light_is_linear_in_the_coefficients() {
    let mut rng = Random::new(3);
    let blocked = |z: f64, r: f64, u1: f64, u2: f64| {
        let ld = LimbDarkening::quadratic(u1, u2).unwrap();
        (1.0 - flux(z, k(r), ld)) * ld.disk_integral()
    };
    for _ in 0..1_000_000 {
        let radius_ratio = rng.radius_ratio();
        let z = rng.separation(radius_ratio);
        let (u1, u2) = rng.limb_darkening().quadratic_coefficients();
        let base = blocked(z, radius_ratio, 0.0, 0.0);
        let expected = base
            + u1 * (blocked(z, radius_ratio, 1.0, 0.0) - base)
            + u2 * (blocked(z, radius_ratio, 0.0, 1.0) - base);
        let actual = blocked(z, radius_ratio, u1, u2);
        let tolerance = 10.0 * error_bound(radius_ratio) * (1.0 + u1.abs() + u2.abs());
        assert!(
            (actual - expected).abs() < tolerance,
            "k {radius_ratio}, z {z}, u ({u1}, {u2}): {actual} vs {expected}"
        );
    }
}

// A larger planet at the same separation never lets more light through.
#[test]
fn larger_planet_never_brightens() {
    let mut rng = Random::new(4);
    for _ in 0..1_000_000 {
        let small = rng.radius_ratio();
        let large = (small * rng.log_uniform(1.0, 2.0)).min(100.0);
        let z = rng.separation(large);
        let ld = rng.limb_darkening();
        let (f_small, f_large) = (flux(z, k(small), ld), flux(z, k(large), ld));
        assert!(
            f_large <= f_small + 2.0 * error_bound(large),
            "z {z}, {ld:?}: k {small} gives {f_small}, k {large} gives {f_large}"
        );
    }
}

// With intensity falling from centre to limb, moving the planet outwards
// never dims the star (Anderson's theorem).
#[test]
fn moving_outwards_never_dims_a_darkened_star() {
    let mut rng = Random::new(5);
    for _ in 0..1_000_000 {
        let radius_ratio = rng.radius_ratio();
        let ld = rng.darkened();
        let inner = rng.separation(radius_ratio);
        let outer = inner + rng.uniform(0.0, 0.1) * (1.0 + radius_ratio);
        let (f_inner, f_outer) = (
            flux(inner, k(radius_ratio), ld),
            flux(outer, k(radius_ratio), ld),
        );
        assert!(
            f_outer >= f_inner - 2.0 * error_bound(radius_ratio),
            "k {radius_ratio}, {ld:?}: z {inner} gives {f_inner}, z {outer} gives {f_outer}"
        );
    }
}

// Against brute-force integration over 20,000 rings at 10,000 points, for k
// up to 2 (beyond
// that the ring sum's own error grows).
#[test]
fn flux_matches_ring_sum() {
    let mut rng = Random::new(6);
    let rings = NonZeroUsize::new(20_000).unwrap();
    for _ in 0..10_000 {
        let radius_ratio = rng.log_uniform(1e-4, 2.0);
        let z = rng.separation(radius_ratio);
        let ld = rng.limb_darkening();
        let analytic = flux(z, k(radius_ratio), ld);
        let ring_sum = flux_ring_sum(z, k(radius_ratio), ld, rings);
        assert!(
            (analytic - ring_sum).abs() < 1e-9,
            "k {radius_ratio}, z {z}, {ld:?}: {analytic} vs {ring_sum}"
        );
    }
}

// ==========================
// Tests: light curves and exposures
// ==========================

#[test]
fn light_curve_is_between_zero_and_one() {
    let mut rng = Random::new(7);
    for _ in 0..100_000 {
        let params = random_transit(&mut rng);
        for _ in 0..20 {
            let t = time_near_transit(&mut rng, params);
            let f = params.flux_at(t);
            assert!((0.0..=1.0).contains(&f), "{params:?} at {t:?}: {f}");
        }
    }
}

// The same moment of the orbit gives the same flux in any orbit, up to the
// rounding of times of that size.
#[test]
fn light_curve_repeats_every_orbit() {
    let mut rng = Random::new(8);
    for _ in 0..200_000 {
        let params = random_transit(&mut rng);
        let t = time_near_transit(&mut rng, params);
        let orbits = rng.uniform(-50.0, 50.0).round();
        let later = t + params.period() * orbits;
        let (f, f_later) = (params.flux_at(t), params.flux_at(later));
        // Each time is rounded to about 1e-12 days; times the fastest the
        // flux can change (a deep ingress of a few minutes).
        assert!(
            (f - f_later).abs() < 1e-8,
            "{params:?}: {f} at {t:?}, {f_later} {orbits} orbits later"
        );
    }
}

#[test]
fn light_curve_is_symmetric_about_mid_transit() {
    let mut rng = Random::new(9);
    for _ in 0..200_000 {
        let params = random_transit(&mut rng);
        let dt = TimeDelta::from_days(rng.uniform(0.0, 0.5) * params.period().as_days()).unwrap();
        let (before, after) = (
            params.flux_at(params.t0() - dt),
            params.flux_at(params.t0() + dt),
        );
        assert!(
            (before - after).abs() < 1e-8,
            "{params:?} at +-{dt:?}: {before} vs {after}"
        );
    }
}

fn random_exposure(rng: &mut Random) -> Exposure {
    Exposure::new(
        TimeDelta::from_days(rng.log_uniform(10.0, 3600.0) / 86_400.0).unwrap(),
        NonZeroUsize::new(rng.uniform(1.0, 40.0) as usize).unwrap(),
    )
    .unwrap()
}

// The average lies between the smallest and largest flux it averages, so in
// [0, 1]; with one sample it is the flux itself.
#[test]
fn binned_flux_is_an_average_of_its_samples() {
    let mut rng = Random::new(10);
    for _ in 0..100_000 {
        let params = random_transit(&mut rng);
        let exposure = random_exposure(&mut rng);
        let t = time_near_transit(&mut rng, params);
        let samples: Vec<f64> = exposure.offsets().map(|o| params.flux_at(t + o)).collect();
        let lo = samples.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = samples.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let binned = params.binned_flux_at(t, exposure);
        assert!(
            lo - 1e-15 <= binned && binned <= hi + 1e-15,
            "{params:?}, {exposure:?} at {t:?}: {binned} outside [{lo}, {hi}]"
        );
        let one = Exposure::new(exposure.duration(), NonZeroUsize::MIN).unwrap();
        assert_eq!(params.binned_flux_at(t, one), params.flux_at(t));
    }
}

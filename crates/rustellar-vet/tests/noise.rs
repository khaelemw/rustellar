// The signal to noise: in white noise the red equals the white and the
// one-depth test follows its chi-squared; in correlated noise the red
// error matches the depth's real scatter where the white falls short; one
// deeper transit and a single event stand out.

use rustellar_model::{ScaledSemiMajorAxis, TransitParams};
use rustellar_vet::{
    Candidate, Epoch, ImpactParameter, LightCurve, LimbDarkening, RadiusRatio, Shape, TimeDelta,
    VetError, signal_to_noise,
};

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

// The default shape's planet: k 0.05, b 0.5, a/R* 15, P 3.7 d, first
// transit at 1501.3, 1.74 hours long.
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

// 27 days every 10 minutes from day 1500.
fn times() -> Vec<f64> {
    (0..3888).map(|i| 1500.0 + f64::from(i) / 144.0).collect()
}

// `flux` plus white noise `white`, and red noise of deviation `red`: an
// Ornstein-Uhlenbeck process with a time scale of 0.2 days, so each point
// keeps exp(-step / 0.2) of the last one's. (Fixed sinusoids would repeat
// from transit to transit rather than average down.)
fn curve(flux: impl Fn(f64) -> f64, white: f64, red: f64, seed: u64) -> LightCurve {
    let mut r = Random(seed);
    let keep = (-1.0f64 / 144.0 / 0.2).exp();
    let fresh = (1.0 - keep * keep).sqrt();
    let mut wiggle = red * r.normal();
    let time = times();
    let f: Vec<f64> = time
        .iter()
        .map(|&t| {
            wiggle = keep * wiggle + fresh * red * r.normal();
            flux(t) + wiggle + white * r.normal()
        })
        .collect();
    let n = time.len();
    LightCurve::from_days(time, f, vec![white; n]).unwrap()
}

fn candidate() -> Candidate {
    Candidate::new(days(3.7), at(1501.3), planet().t14().unwrap()).unwrap()
}

fn mean_and_variance(values: &[f64]) -> (f64, f64) {
    let n = f64::from(u32::try_from(values.len()).unwrap());
    let mean = values.iter().sum::<f64>() / n;
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
    (mean, var)
}

// Noiseless: the depth the planet's, every transit alike, no red noise and
// nothing to tell the transits apart.
#[test]
fn a_noiseless_planet() {
    let p = planet();
    let time = times();
    let flux: Vec<f64> = time.iter().map(|&t| p.flux_at(at(t))).collect();
    let lc = LightCurve::from_days(time, flux, vec![1e-3; 3888]).unwrap();
    let s = signal_to_noise(&lc, &candidate(), Shape::default()).unwrap();
    let centre = 1.0 - p.flux_at(at(1501.3));
    assert!((s.depth / centre - 1.0).abs() < 1e-10);
    assert_eq!(s.noise.white, 0.0);
    assert_eq!(s.noise.red, 0.0);
    assert_eq!(s.white, s.red);
    assert_eq!(s.transits.len(), 7);
    assert_eq!(s.consistency_dof, 6);
    assert!(s.consistency_chi2 < 1e-12, "{}", s.consistency_chi2);
    // A seventh, give or take the transits sampled at slightly different
    // phases.
    assert!((s.largest_share - 1.0 / 7.0).abs() < 5e-3, "{s:?}");
    let epochs: Vec<i64> = s.transits.iter().map(|t| t.epoch).collect();
    assert_eq!(epochs, [0, 1, 2, 3, 4, 5, 6]);
    for t in &s.transits {
        assert_eq!(t.white, t.red);
        // Each transit carries about a seventh of the information.
        assert!((t.white / s.white - 1.0 / 7.0f64.sqrt()).abs() < 0.02);
    }
}

// In white noise, over 1,000 draws: the white noise measured is the noise
// put in, the red signal to noise within a few percent of the white, the
// one-depth chi-squared distributed as chi-squared with 6 degrees of
// freedom (mean 6, variance 12), and the largest share near a seventh.
#[test]
fn white_noise_is_calibrated() {
    let p = planet();
    let c = candidate();
    let (mut chi2, mut ratio, mut share, mut white) = (vec![], vec![], vec![], vec![]);
    for seed in 0..1000 {
        let lc = curve(|t| p.flux_at(at(t)), 1e-3, 0.0, 300 + seed);
        let s = signal_to_noise(&lc, &c, Shape::default()).unwrap();
        chi2.push(s.consistency_chi2);
        ratio.push(s.red / s.white);
        share.push(s.largest_share);
        white.push(s.noise.white);
    }
    let (mean, var) = mean_and_variance(&chi2);
    assert!((mean - 6.0).abs() < 0.35, "{mean}");
    assert!((var - 12.0).abs() < 2.5, "{var}");
    let (mean, _) = mean_and_variance(&ratio);
    assert!(mean > 0.95, "{mean}");
    assert!(ratio.iter().all(|&r| r > 0.85 && r <= 1.0));
    let (mean, _) = mean_and_variance(&white);
    assert!((mean / 1e-3 - 1.0).abs() < 0.01, "{mean}");
    let (mean, _) = mean_and_variance(&share);
    assert!(mean > 1.0 / 7.0 && mean < 0.35, "{mean}");
}

// In red noise as large as the white, over 300 draws: the depth scatters
// far more than its white error says, and about as much as its red error
// says.
#[test]
fn red_noise_is_measured() {
    let p = planet();
    let c = candidate();
    let (mut depth, mut white_err, mut red_err, mut red) = (vec![], 0.0, 0.0, vec![]);
    for seed in 0..300 {
        let lc = curve(|t| p.flux_at(at(t)), 1e-3, 1e-3, 5000 + seed);
        let s = signal_to_noise(&lc, &c, Shape::default()).unwrap();
        depth.push(s.depth);
        white_err += s.depth / s.white;
        red_err += s.depth / s.red;
        red.push(s.noise.red);
    }
    let (_, var) = mean_and_variance(&depth);
    let scatter = var.sqrt();
    let (white_err, red_err) = (white_err / 300.0, red_err / 300.0);
    assert!(scatter / white_err > 2.0, "{scatter} {white_err}");
    assert!(
        (scatter / red_err - 1.0).abs() < 0.25,
        "{scatter} {red_err}"
    );
    let (mean, _) = mean_and_variance(&red);
    assert!((mean / 1e-3 - 1.0).abs() < 0.2, "{mean}");
}

// One transit twice as deep: the one-depth test fails by far, and that
// transit carries the largest share and signal to noise.
#[test]
fn one_deeper_transit() {
    let p = planet();
    let deeper = |t: f64| {
        let f = p.flux_at(at(t));
        if (t - (1501.3 + 3.0 * 3.7)).abs() < 0.5 {
            1.0 - 2.0 * (1.0 - f)
        } else {
            f
        }
    };
    let lc = curve(deeper, 2e-4, 0.0, 7);
    let s = signal_to_noise(&lc, &candidate(), Shape::default()).unwrap();
    assert!(s.consistency_chi2 > 100.0, "{}", s.consistency_chi2);
    let best = s
        .transits
        .iter()
        .max_by(|a, b| a.white.total_cmp(&b.white))
        .unwrap();
    assert_eq!(best.epoch, 3);
    assert!(
        (s.largest_share - 2.0 / 8.0).abs() < 0.03,
        "{}",
        s.largest_share
    );
}

// A single event among seven predicted transits: its share near 1.
#[test]
fn a_single_event() {
    let p = planet();
    let single = |t: f64| {
        if (t - (1501.3 + 3.0 * 3.7)).abs() < 0.5 {
            p.flux_at(at(t))
        } else {
            1.0
        }
    };
    let lc = curve(single, 2e-4, 0.0, 8);
    let s = signal_to_noise(&lc, &candidate(), Shape::default()).unwrap();
    assert!(s.largest_share > 0.8, "{}", s.largest_share);
    assert!(s.consistency_chi2 > 100.0);
}

#[test]
fn what_cannot_be_measured() {
    let lc = LightCurve::from_days(vec![0.0, 1.0], vec![1.0, 1.0], vec![1e-3; 2]).unwrap();
    let late = Candidate::new(days(10.0), at(5.0), days(0.1)).unwrap();
    assert_eq!(
        signal_to_noise(&lc, &late, Shape::BOX),
        Err(VetError::NoPointsInTransit)
    );
    let wide = Candidate::new(days(1.0), at(0.0), days(0.6)).unwrap();
    assert_eq!(
        signal_to_noise(&lc, &wide, Shape::default()),
        Err(VetError::DurationTooLongForShape)
    );
    // Two points, one in transit: one transit, nothing to compare.
    let two = LightCurve::from_days(vec![0.0, 0.5], vec![0.99, 1.0], vec![1e-3; 2]).unwrap();
    let c = Candidate::new(days(10.0), at(0.0), days(0.1)).unwrap();
    let s = signal_to_noise(&two, &c, Shape::BOX).unwrap();
    assert_eq!((s.transits.len(), s.consistency_dof), (1, 0));
    assert!((s.largest_share - 1.0).abs() < 1e-12);
    assert!(s.noise.white.is_nan());
    assert_eq!(s.white, s.red);
}

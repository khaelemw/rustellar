// Standard errors, covariance and derived quantities: exact where the fit
// is linear, and, on simulated noise, pulls of mean 0 and spread 1.

use rustellar_fit::{
    Epoch, Estimate, ImpactParameter, LightCurve, LimbDarkening, Parameter, RadiusRatio,
    ScaledSemiMajorAxis, TimeDelta, TransitFit, TransitParams,
};

fn params(t0: f64, p: f64, k: f64, a: f64, b: f64, u1: f64, u2: f64) -> TransitParams {
    TransitParams::new(
        Epoch::from_days(t0).unwrap(),
        TimeDelta::from_days(p).unwrap(),
        RadiusRatio::new(k).unwrap(),
        ScaledSemiMajorAxis::new(a).unwrap(),
        ImpactParameter::new(b).unwrap(),
        LimbDarkening::quadratic(u1, u2).unwrap(),
    )
    .unwrap()
}

/// SplitMix64 (Steele, Lea & Flood 2014) and Box-Muller: the same normal
/// deviates on every platform.
struct Normal(u64);

impl Normal {
    fn uniform(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        // 53 bits in (0, 1].
        ((z >> 11) + 1) as f64 / (1u64 << 53) as f64
    }

    fn next(&mut self) -> f64 {
        let (u, v) = (self.uniform(), self.uniform());
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }
}

fn hot_jupiter() -> TransitParams {
    params(1.3, 3.0, 0.1, 10.0, 0.3, 0.4, 0.25)
}

/// Two transits at five-minute cadence, with Gaussian noise of `sigma`.
fn noisy(truth: TransitParams, sigma: f64, noise: &mut Normal) -> LightCurve {
    let time: Vec<f64> = (0..1440).map(|i| f64::from(i) * 5.0 / 1440.0).collect();
    let flux = time
        .iter()
        .map(|&t| sigma.mul_add(noise.next(), truth.flux_at(Epoch::from_days(t).unwrap())))
        .collect();
    LightCurve::from_days(time, flux, vec![sigma; 1440]).unwrap()
}

#[test]
fn baseline_alone_is_exact() {
    // The model is linear in the baseline: its variance is
    // 1 / sum((F / sigma)^2) exactly.
    let truth = hot_jupiter();
    let lc = noisy(truth, 1e-3, &mut Normal(7));
    let mut fit = TransitFit::new(&lc, truth);
    for p in Parameter::ALL {
        if p != Parameter::Baseline {
            fit = fit.with_fixed(p);
        }
    }
    let result = fit.run().unwrap();
    let information: f64 = lc
        .time()
        .iter()
        .map(|&t| (truth.flux_at(t) / 1e-3).powi(2))
        .sum();
    let e = result.estimate(Parameter::Baseline);
    assert!((e.error() - information.recip().sqrt()).abs() < 1e-12 * e.error());
    // The fixed have none.
    assert_eq!(result.estimate(Parameter::RadiusRatio).error(), 0.0);
    let c = result.covariance().unwrap();
    assert_eq!(c.parameters(), [Parameter::Baseline]);
    assert_eq!(c.get(Parameter::Baseline, Parameter::Period), 0.0);
    assert_eq!(c.correlation(Parameter::Baseline, Parameter::Period), 0.0);
    assert!((c.correlation(Parameter::Baseline, Parameter::Baseline) - 1.0).abs() < 1e-15);
    let scaled = c.scaled(4.0);
    assert!(
        (scaled.standard_error(Parameter::Baseline) - 2.0 * e.error()).abs() < 1e-15 * e.error()
    );
}

#[test]
fn derived_quantities() {
    // At b = 0.5, where b is well measured: nearer 0 the shape hardly
    // changes with it once a and limb darkening adjust.
    let truth = params(1.3, 3.0, 0.1, 10.0, 0.5, 0.4, 0.25);
    let lc = noisy(truth, 2e-4, &mut Normal(11));
    let result = TransitFit::new(&lc, truth).run().unwrap();
    let best = result.params();
    let c = result.covariance().unwrap();

    // The density by hand: 3 pi a^3 / (G P^2), and its error from
    // d ln rho = 3 d ln a - 2 d ln P.
    let (a, p) = (best.a_rs().get(), best.period().as_days() * 86_400.0);
    let rho = 3.0 * std::f64::consts::PI * a.powi(3) / (6.674_30e-11 * p * p) / 1000.0;
    let density = result.stellar_density();
    assert!((density.value() - rho).abs() < 1e-14 * rho);
    let (sa, sp) = (
        c.standard_error(Parameter::SemiMajorAxis) / a,
        c.standard_error(Parameter::Period) * 86_400.0 / p,
    );
    let cov = c.get(Parameter::SemiMajorAxis, Parameter::Period) * 86_400.0 / (a * p);
    let relative = (9.0 * sa * sa + 4.0 * sp * sp - 12.0 * cov).sqrt();
    assert!((density.error() / rho - relative).abs() < 1e-6 * relative);

    // Inclination from b and a; durations and depth from the model.
    let i = result.inclination_deg();
    assert!((i.value() - (best.b().get() / a).acos().to_degrees()).abs() < 1e-12);
    assert!(i.error() > 0.0 && i.error() < 0.5);
    let t14 = result.t14().unwrap();
    let t23 = result.t23().unwrap();
    assert_eq!(t14.value(), best.t14().unwrap().as_days());
    assert_eq!(t23.value(), best.t23().unwrap().as_days());
    assert!(t14.error() > 0.0 && t23.error() > 0.0);
    assert_eq!(
        format!("{t14:.1}"),
        format!("{:.1} ± {:.1}", t14.value(), t14.error())
    );
    let depth = result.depth();
    assert_eq!(depth.value(), 1.0 - best.flux_at(best.t0()));
    // The depth is far better measured than k^2 alone suggests.
    assert!(depth.error() < 1e-4 && depth.error() > 1e-6);
}

#[test]
fn grazing_has_no_full_duration() {
    let truth = params(1.0, 2.0, 0.12, 6.0, 1.02, 0.4, 0.25);
    let lc = noisy(truth, 1e-4, &mut Normal(3));
    let result = TransitFit::new(&lc, truth)
        .with_fixed(Parameter::U1)
        .with_fixed(Parameter::U2)
        .run()
        .unwrap();
    assert!(result.t23().is_none());
    assert!(result.t14().is_some());
}

/// Pulls `(fit - truth) / error` over many noise realisations: mean 0 and
/// standard deviation 1 if the errors are right. The errors are to first
/// order, so the test is where the fit is close to linear over them: at
/// lower signal to noise, b and limb darkening trade off far beyond it.
#[test]
fn pulls_on_simulated_noise() {
    const RUNS: u32 = 300;
    let truth = params(1.3, 3.0, 0.1, 10.0, 0.5, 0.4, 0.25);
    let want = [1.3, 3.0, 0.1, 10.0, 0.5, 0.4, 0.25, 0.0, 0.0, 1.0];
    let mut noise = Normal(2026);
    let mut sums = [[0.0; 2]; 10];
    let fitted = TransitFit::new(&noisy(truth, 3e-4, &mut Normal(1)), truth).free();
    let mut chi2 = 0.0;
    for _ in 0..RUNS {
        let lc = noisy(truth, 3e-4, &mut noise);
        let result = TransitFit::new(&lc, truth).run().unwrap();
        for p in &fitted {
            let s = &mut sums[p.index()];
            let e = result.estimate(*p);
            let pull = (e.value() - want[p.index()]) / e.error();
            s[0] += pull;
            s[1] += pull * pull;
        }
        chi2 += result.reduced_chi_squared();
    }
    let runs = f64::from(RUNS);
    for p in &fitted {
        let [sum, squares] = sums[p.index()];
        let mean = sum / runs;
        let sd = (squares / runs - mean * mean).sqrt();
        // Within about three standard errors of 0 and 1.
        assert!(mean.abs() < 0.2, "{p}: mean pull {mean}");
        assert!((sd - 1.0).abs() < 0.15, "{p}: pull spread {sd}");
    }
    assert!((chi2 / runs - 1.0).abs() < 0.01);
}

#[test]
fn estimates_print_with_their_errors() {
    let e = Estimate::new(0.1, 0.002);
    assert_eq!(e.to_string(), "0.1 ± 0.002");
    assert_eq!(format!("{e:.3}"), "0.100 ± 0.002");
}

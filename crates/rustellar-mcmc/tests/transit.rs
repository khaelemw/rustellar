// The transit posterior: sampled from the fit's walkers, its means and
// spreads match the fit's estimates where the posterior is near Gaussian; a
// density prior narrows a / R*; every way of freeing the limb darkening and
// eccentricity; the checks. Against emcee with batman on the same light
// curve: the percentiles agree.

use rustellar_fit::{
    Epoch, ImpactParameter, LightCurve, LimbDarkening, Parameter, RadiusRatio, ScaledSemiMajorAxis,
    TimeDelta, TransitFit, TransitParams,
};
use rustellar_mcmc::transit::{Coordinate, Prior, TransitPosterior};
use rustellar_mcmc::{Ensemble, LogProbability, McmcError, Move};

fn truth() -> TransitParams {
    TransitParams::new(
        Epoch::from_days(0.0).unwrap(),
        TimeDelta::from_days(3.0).unwrap(),
        RadiusRatio::new(0.1).unwrap(),
        ScaledSemiMajorAxis::new(10.0).unwrap(),
        ImpactParameter::new(0.3).unwrap(),
        LimbDarkening::quadratic(0.4, 0.25).unwrap(),
    )
    .unwrap()
}

/// `n` points 0.0006 d apart about mid-transit, with white noise `sigma`.
fn noisy(params: TransitParams, n: u32, sigma: f64, seed: u64) -> LightCurve {
    let mut s = seed;
    let mut unit = || {
        s = s.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    };
    let time: Vec<f64> = (0..n)
        .map(|i| 0.0006 * (f64::from(i) - 0.5 * f64::from(n - 1)))
        .collect();
    let flux = time
        .iter()
        .map(|&t| {
            let (u1, u2) = (1.0 - unit(), unit());
            let z = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
            params.flux_at(Epoch::from_days(t).unwrap()) + sigma * z
        })
        .collect();
    LightCurve::from_days(time, flux, vec![sigma; n as usize]).unwrap()
}

fn mean_sd(chain: &rustellar_mcmc::Chain, discard: usize, i: usize) -> (f64, f64) {
    let (m, c) = (chain.mean(discard), chain.covariance(discard));
    let d = chain.dimensions();
    (m[i], c[i * d + i].sqrt())
}

#[test]
fn matches_the_fit_where_near_gaussian() {
    // A deep, well-sampled transit: the posterior close to the fit's normal
    // in t0, k and the baseline.
    let lc = noisy(truth(), 500, 2e-4, 1);
    let setup = TransitFit::new(&lc, truth()).with_fixed(Parameter::Period);
    let result = setup.run().unwrap();
    let posterior = TransitPosterior::new(&setup).unwrap();
    let coords = posterior.coordinates().to_vec();
    assert_eq!(
        coords,
        [
            Coordinate::MidTime,
            Coordinate::RadiusRatio,
            Coordinate::SemiMajorAxis,
            Coordinate::ImpactParameter,
            Coordinate::Q1,
            Coordinate::Q2,
            Coordinate::Baseline
        ]
    );
    let start = posterior.walkers(&result, 32, 7).unwrap();
    assert_eq!(start.len(), 32);
    assert!(
        start
            .iter()
            .all(|x| posterior.log_probability(x).is_finite())
    );
    let chain = Ensemble::new(start)
        .unwrap()
        .with_seed(3)
        .with_move(Move::differential_evolution(None, 1e-5).unwrap())
        .run(&posterior, 4000)
        .unwrap();
    // The spreads within 10% of the fit's errors, the means within a tenth
    // of them; k's mean is pulled down by a third of its error, as b's
    // bound at zero skews the posterior along the k-b valley.
    for (i, p, gap) in [
        (0, Parameter::MidTime, 0.1),
        (1, Parameter::RadiusRatio, 0.5),
        (6, Parameter::Baseline, 0.1),
    ] {
        let (m, sd) = mean_sd(&chain, 1000, i);
        let fit = result.estimate(p);
        assert!(
            (m - fit.value()).abs() < gap * fit.error(),
            "{p}: {m} vs {fit:?}"
        );
        assert!((sd / fit.error() - 1.0).abs() < 0.1, "{p}: {sd} vs {fit:?}");
    }
    let (params, baseline) = posterior.params_at(chain.position(3999, 0)).unwrap();
    assert!(baseline > 0.99 && params.k().get() > 0.09);
    assert!(
        posterior
            .params_at(&[0.0, 0.1, 10.0, 2.0, 0.4, 0.3, 1.0])
            .is_none()
    );
}

#[test]
fn a_density_prior_narrows_the_orbit() {
    let lc = noisy(truth(), 300, 1e-3, 2);
    let setup = TransitFit::new(&lc, truth()).with_fixed(Parameter::Period);
    let result = setup.run().unwrap();
    let run = |posterior: &TransitPosterior| {
        let start = posterior.walkers(&result, 32, 1).unwrap();
        Ensemble::new(start)
            .unwrap()
            .with_seed(5)
            .run(posterior, 3000)
            .unwrap()
    };
    let free = TransitPosterior::new(&setup).unwrap();
    // The truth's density, 2.1 g/cm^3, known to 2%.
    let x = free.point(truth(), 1.0);
    let rho = free.stellar_density(&x).unwrap();
    assert!((rho - 2.1018).abs() < 1e-3);
    let held = free.clone().with_stellar_density(rho, 0.02 * rho).unwrap();
    let (_, wide) = mean_sd(&run(&free), 1000, 2);
    let (m, narrow) = mean_sd(&run(&held), 1000, 2);
    // a / R* goes as the cube root of the density: 2% / 3 of 10.
    assert!(narrow < 0.5 * wide, "{narrow} {wide}");
    assert!((m - 10.0).abs() < 3.0 * 0.067, "{m}");
    assert!((narrow - 0.067).abs() < 0.025, "{narrow}");
}

#[test]
fn coordinates_for_each_freedom() {
    let lc = noisy(truth(), 200, 1e-3, 3);
    let base = TransitFit::new(&lc, truth());
    let coords = |fit: TransitFit| TransitPosterior::new(&fit).unwrap().coordinates().to_vec();
    let only = |fit: TransitFit| {
        let c = coords(fit);
        c[c.len() - 2..].to_vec()
    };
    assert_eq!(
        only(base.clone().with_fixed(Parameter::U2)),
        [Coordinate::U1, Coordinate::Baseline]
    );
    assert_eq!(
        only(base.clone().with_fixed(Parameter::U1)),
        [Coordinate::U2, Coordinate::Baseline]
    );
    assert_eq!(
        only(base.clone().with_free(Parameter::ECosW)),
        [Coordinate::ECosW, Coordinate::Baseline]
    );
    assert_eq!(
        only(base.clone().with_free(Parameter::ESinW)),
        [Coordinate::ESinW, Coordinate::Baseline]
    );
    let both = base
        .clone()
        .with_free(Parameter::ECosW)
        .with_free(Parameter::ESinW);
    let posterior = TransitPosterior::new(&both).unwrap();
    assert_eq!(
        posterior.coordinates()[7..],
        [
            Coordinate::SqrtECosW,
            Coordinate::SqrtESinW,
            Coordinate::Baseline
        ]
    );
    // An eccentric orbit: the point and back.
    let eccentric = truth().with_eccentricity(0.2, 60.0).unwrap();
    let x = posterior.point(eccentric, 1.0);
    let (back, _) = posterior.params_at(&x).unwrap();
    assert!((back.eccentricity() - 0.2).abs() < 1e-14);
    assert!((back.periastron_deg() - 60.0).abs() < 1e-11);
    assert!((x[7].hypot(x[8]) - 0.2f64.sqrt()).abs() < 1e-15);
    // Every other coordinate fixed, the period free: one coordinate.
    let mut period_only = base.clone();
    for p in Parameter::ALL {
        period_only = period_only.with_fixed(p);
    }
    assert_eq!(
        TransitPosterior::new(&period_only).unwrap_err(),
        McmcError::NoDimensions
    );
    assert_eq!(
        coords(period_only.with_free(Parameter::Period)),
        [Coordinate::Period]
    );
    let names: Vec<&str> = coords(base).iter().map(|c| c.name()).collect();
    assert_eq!(
        names,
        ["t0", "period", "k", "a_rs", "b", "q1", "q2", "baseline"]
    );
    for (c, name) in [
        (Coordinate::U1, "u1"),
        (Coordinate::U2, "u2"),
        (Coordinate::ECosW, "e_cos_w"),
        (Coordinate::ESinW, "e_sin_w"),
        (Coordinate::SqrtECosW, "sqrt_e_cos_w"),
        (Coordinate::SqrtESinW, "sqrt_e_sin_w"),
    ] {
        assert_eq!(c.name(), name);
    }
}

#[test]
fn priors_and_checks() {
    let lc = noisy(truth(), 200, 1e-3, 4);
    let setup = TransitFit::new(&lc, truth()).with_fixed(Parameter::Period);
    let posterior = TransitPosterior::new(&setup).unwrap();
    let x = posterior.point(truth(), 1.0);
    let bare = posterior.log_probability(&x);
    // A normal prior adds its log-density; a log-uniform too.
    let normal = Prior::normal(0.1, 0.01).unwrap();
    let with = posterior
        .clone()
        .with_prior(Coordinate::RadiusRatio, normal)
        .unwrap()
        .with_prior(
            Coordinate::SemiMajorAxis,
            Prior::log_uniform(1.0, 100.0).unwrap(),
        )
        .unwrap();
    let want = bare + normal.log_density(0.1) - 10f64.ln() - 100f64.ln();
    assert!((with.log_probability(&x) - want).abs() < 1e-9);
    // Outside a prior, the range or the model: minus infinity.
    let narrow = posterior
        .clone()
        .with_prior(Coordinate::RadiusRatio, Prior::uniform(0.5, 0.6).unwrap())
        .unwrap();
    assert_eq!(narrow.log_probability(&x), f64::NEG_INFINITY);
    let mut far = x.clone();
    far[3] = 1.2;
    assert_eq!(posterior.log_probability(&far), f64::NEG_INFINITY);
    let mut inside = x.clone();
    inside[2] = 0.5;
    assert_eq!(posterior.log_probability(&inside), f64::NEG_INFINITY);
    // Walkers that cannot start.
    let result = setup.run().unwrap();
    assert_eq!(
        narrow.walkers(&result, 8, 1).unwrap_err(),
        McmcError::OutsideSupport { walker: 0 }
    );
    let other = TransitPosterior::new(&TransitFit::new(&lc, truth())).unwrap();
    assert_eq!(
        other.walkers(&result, 8, 1).unwrap_err(),
        McmcError::NoCovariance
    );
    // The priors' checks.
    let bad = |name| Err(McmcError::InvalidPrior { name });
    assert_eq!(Prior::uniform(1.0, 1.0), bad("uniform"));
    assert_eq!(Prior::uniform(f64::NAN, 1.0), bad("uniform"));
    assert_eq!(Prior::normal(0.0, 0.0), bad("normal"));
    assert_eq!(Prior::normal(f64::INFINITY, 1.0), bad("normal"));
    assert_eq!(Prior::log_uniform(0.0, 1.0), bad("log-uniform"));
    assert_eq!(Prior::log_uniform(2.0, 1.0), bad("log-uniform"));
    assert_eq!(
        Prior::uniform(0.0, 2.0).unwrap().log_density(1.0),
        -(2f64.ln())
    );
    assert_eq!(
        Prior::log_uniform(1.0, 10.0).unwrap().log_density(20.0),
        f64::NEG_INFINITY
    );
    assert_eq!(
        posterior
            .clone()
            .with_prior(Coordinate::Period, normal)
            .unwrap_err(),
        McmcError::InvalidPrior { name: "period" }
    );
    for (m, s) in [(1.0, 0.0), (f64::NAN, 1.0)] {
        assert_eq!(
            posterior.clone().with_stellar_density(m, s).unwrap_err(),
            McmcError::InvalidPrior {
                name: "stellar density"
            }
        );
    }
    assert_eq!(
        McmcError::NoCovariance.to_string(),
        "the fit has no covariance over these parameters"
    );
    assert_eq!(
        McmcError::InvalidPrior { name: "q1" }.to_string(),
        "the q1 prior is out of range"
    );
}

/// The `q` percentile of `x` (sorted), linear between ranks as numpy's.
fn percentile(sorted: &[f64], q: f64) -> f64 {
    let at = q / 100.0 * (sorted.len() - 1) as f64;
    let (i, t) = (at.floor() as usize, at.fract());
    if i + 1 < sorted.len() {
        sorted[i] + t * (sorted[i + 1] - sorted[i])
    } else {
        sorted[i]
    }
}

#[test]
fn matches_emcee_with_batman() {
    // The light curve and emcee's percentiles from the reference tool.
    let rows = |text: &str| -> Vec<Vec<String>> {
        text.lines()
            .skip(1)
            .map(|l| l.split(',').map(str::to_owned).collect())
            .collect()
    };
    let points = rows(include_str!("reference/transit_points.csv"));
    let num = |s: &String| s.parse::<f64>().unwrap();
    let column = |i: usize| points.iter().map(|r| num(&r[i])).collect::<Vec<f64>>();
    let lc = LightCurve::from_days(column(0), column(1), column(2)).unwrap();
    let setup = TransitFit::new(&lc, truth()).with_fixed(Parameter::Period);
    let result = setup.run().unwrap();
    let posterior = TransitPosterior::new(&setup).unwrap();
    let start = posterior.walkers(&result, 40, 11).unwrap();
    let chain = Ensemble::new(start)
        .unwrap()
        .with_seed(12)
        .with_move(Move::differential_evolution(None, 1e-5).unwrap())
        .run(&posterior, 30_000)
        .unwrap();
    let discard = 5000;
    let mut columns: Vec<Vec<f64>> = vec![Vec::new(); 8];
    for x in chain.samples(discard) {
        for (c, v) in columns.iter_mut().zip(x) {
            c.push(*v);
        }
        columns[7].push(posterior.stellar_density(x).unwrap());
    }
    let reference = rows(include_str!("reference/transit_posterior.csv"));
    let names = ["t0", "k", "a_rs", "b", "q1", "q2", "baseline", "density"];
    let (mut worst_centre, mut worst_tail) = (0.0f64, 0.0f64);
    for ((row, name), mut ours) in reference.iter().zip(names).zip(columns) {
        assert_eq!(row[0], name);
        ours.sort_by(f64::total_cmp);
        let theirs: Vec<f64> = row[1..].iter().map(num).collect();
        let sigma = (theirs[3] - theirs[1]) / 2.0;
        for (j, (q, gap)) in [
            (2.5, 0.3),
            (16.0, 0.15),
            (50.0, 0.15),
            (84.0, 0.15),
            (97.5, 0.3),
        ]
        .into_iter()
        .enumerate()
        {
            let mine = percentile(&ours, q);
            let gap_sigma = ((mine - theirs[j]) / sigma).abs();
            if j == 0 || j == 4 {
                worst_tail = worst_tail.max(gap_sigma);
            } else {
                worst_centre = worst_centre.max(gap_sigma);
            }
            assert!(
                (mine - theirs[j]).abs() < gap * sigma,
                "{name} p{q}: {mine} vs {} ({:.3} sigma)",
                theirs[j],
                (mine - theirs[j]) / sigma
            );
        }
    }
    eprintln!("worst gaps in sigma: 16-84th {worst_centre:.3}, 2.5th and 97.5th {worst_tail:.3}");
}

#[test]
fn single_coordinates_and_exposure() {
    use std::num::NonZeroUsize;

    use rustellar_fit::Exposure;
    let lc = noisy(truth(), 200, 1e-3, 5);
    let eccentric = truth().with_eccentricity(0.1, 40.0).unwrap();
    // Each lone coordinate: the point and back to the same parameters.
    for free in [
        Parameter::U1,
        Parameter::U2,
        Parameter::ECosW,
        Parameter::ESinW,
    ] {
        let mut setup = TransitFit::new(&lc, eccentric);
        for p in Parameter::ALL {
            setup = setup.with_fixed(p);
        }
        let posterior = TransitPosterior::new(&setup.with_free(free)).unwrap();
        let x = posterior.point(eccentric, 1.0);
        let (back, baseline) = posterior.params_at(&x).unwrap();
        assert_eq!(baseline, 1.0);
        let (u1, u2) = back.limb_darkening().quadratic_coefficients();
        assert!(
            (u1 - 0.4).abs() < 1e-15 && (u2 - 0.25).abs() < 1e-15,
            "{free}"
        );
        assert!((back.eccentricity() - 0.1).abs() < 1e-15, "{free}");
        assert!(posterior.log_probability(&x).is_finite());
    }
    // With an exposure, the likelihood is the binned model's.
    let exposure = Exposure::new(
        TimeDelta::from_days(0.02).unwrap(),
        NonZeroUsize::new(7).unwrap(),
    )
    .unwrap();
    let setup = TransitFit::new(&lc, truth())
        .with_fixed(Parameter::Period)
        .with_exposure(exposure);
    let posterior = TransitPosterior::new(&setup).unwrap();
    let x = posterior.point(truth(), 1.0);
    let chi2: f64 = lc
        .time()
        .iter()
        .zip(lc.flux())
        .map(|(&t, &y)| {
            let r = (truth().binned_flux_at(t, exposure) - y) / 1e-3;
            r * r
        })
        .sum();
    assert!((posterior.log_probability(&x) + 0.5 * chi2).abs() < 1e-9 * chi2);
}

#[test]
fn eccentric_walkers() {
    // A deep transit with the eccentricity free: walkers from the fit, in
    // sqrt(e) cos w and sqrt(e) sin w.
    let eccentric = truth().with_eccentricity(0.3, 70.0).unwrap();
    let lc = noisy(eccentric, 500, 2e-4, 6);
    let setup = TransitFit::new(&lc, eccentric)
        .with_fixed(Parameter::Period)
        .with_free(Parameter::ECosW)
        .with_free(Parameter::ESinW);
    let result = setup.run().unwrap();
    let posterior = TransitPosterior::new(&setup).unwrap();
    let start = posterior.walkers(&result, 24, 3).unwrap();
    assert!(
        start
            .iter()
            .all(|x| posterior.log_probability(x).is_finite())
    );
    assert_eq!(
        posterior.coordinates()[6..8],
        [Coordinate::SqrtECosW, Coordinate::SqrtESinW]
    );
}

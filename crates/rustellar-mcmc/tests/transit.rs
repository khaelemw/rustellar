// The transit posterior: sampled from the fit's walkers, its means and
// spreads match the fit's estimates where the posterior is near Gaussian; a
// density prior narrows a / R*; every way of freeing the limb darkening and
// eccentricity; the checks. Against emcee with batman on the same light
// curve, and zeus with batman on a grazing transit and an eccentric orbit:
// the percentiles agree.

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

/// A posterior against a reference sampler's: the case's light curve from
/// the reference tool, our chain (40 walkers from the fit, `seed`) and the
/// worst gaps of the percentiles in the reference's sigma (16th to 84th,
/// 2.5th and 97.5th), less the reference's own uncertainty there where it
/// gives one.
struct Case {
    points: &'static str,
    reference: &'static str,
    /// How far the reference's own chains strayed from its pooled
    /// percentiles, in its sigma.
    spread: Option<&'static str>,
    truth: TransitParams,
    names: &'static [&'static str],
    steps: usize,
    discard: usize,
    /// Differential evolution alone, or with the snooker move a fifth of
    /// the time (which keeps walkers from sticking in a narrow ridge).
    snooker: bool,
}

fn rows(text: &str) -> Vec<Vec<f64>> {
    text.lines()
        .skip(1)
        .map(|l| l.split(',').filter_map(|v| v.parse().ok()).collect())
        .collect()
}

fn against_reference(
    case: &Case,
    free: impl Fn(TransitFit<'_>) -> TransitFit<'_>,
    priors: impl Fn(TransitPosterior<'_>) -> TransitPosterior<'_>,
    seed: u64,
    threads: usize,
) -> (f64, f64) {
    let points = rows(case.points);
    let column = |i: usize| points.iter().map(|r| r[i]).collect::<Vec<f64>>();
    let lc = LightCurve::from_days(column(0), column(1), column(2)).unwrap();
    let setup = free(TransitFit::new(&lc, case.truth).with_fixed(Parameter::Period));
    let result = setup.run().unwrap();
    let posterior = priors(TransitPosterior::new(&setup).unwrap());
    let names: Vec<&str> = posterior.coordinates().iter().map(|c| c.name()).collect();
    assert_eq!(names, case.names[..case.names.len() - 1]);
    let start = posterior.walkers(&result, 40, seed).unwrap();
    let ensemble = Ensemble::new(start)
        .unwrap()
        .with_seed(seed + 1)
        .with_threads(std::num::NonZeroUsize::new(threads).unwrap())
        .with_move(Move::differential_evolution(None, 1e-5).unwrap());
    let ensemble = if case.snooker {
        ensemble
            .with_moves(vec![
                (Move::differential_evolution(None, 1e-5).unwrap(), 0.8),
                (Move::snooker(1.7).unwrap(), 0.2),
            ])
            .unwrap()
    } else {
        ensemble
    };
    let chain = ensemble.run(&posterior, case.steps).unwrap();
    let d = names.len();
    let mut columns: Vec<Vec<f64>> = vec![Vec::new(); d + 1];
    for x in chain.samples(case.discard) {
        for (c, v) in columns.iter_mut().zip(x) {
            c.push(*v);
        }
        columns[d].push(posterior.stellar_density(x).unwrap());
    }
    let reference = rows(case.reference);
    let names_in_file: Vec<&str> = case
        .reference
        .lines()
        .skip(1)
        .map(|l| l.split(',').next().unwrap())
        .collect();
    assert_eq!(names_in_file, case.names);
    let spread = case
        .spread
        .map_or_else(|| vec![vec![0.0; 5]; reference.len()], rows);
    assert_eq!(spread.len(), reference.len());
    let (mut centre, mut tail) = (0.0f64, 0.0f64);
    for (i, (theirs, mut ours)) in reference.iter().zip(columns).enumerate() {
        ours.sort_by(f64::total_cmp);
        let sigma = (theirs[3] - theirs[1]) / 2.0;
        for (j, q) in [2.5, 16.0, 50.0, 84.0, 97.5].into_iter().enumerate() {
            let gap = (((percentile(&ours, q) - theirs[j]) / sigma).abs() - spread[i][j]).max(0.0);
            if gap > 0.3 {
                eprintln!(
                    "  {} p{q}: {} against {} ({gap:.2} sigma beyond the reference's spread)",
                    case.names[i],
                    percentile(&ours, q),
                    theirs[j]
                );
            }
            if j == 0 || j == 4 {
                tail = tail.max(gap);
            } else {
                centre = centre.max(gap);
            }
        }
    }
    (centre, tail)
}

const HOT: [&str; 8] = ["t0", "k", "a_rs", "b", "q1", "q2", "baseline", "density"];

fn hot() -> Case {
    Case {
        points: include_str!("reference/transit_points.csv"),
        reference: include_str!("reference/transit_posterior.csv"),
        spread: None,
        truth: truth(),
        names: &HOT,
        steps: 30_000,
        discard: 5000,
        snooker: false,
    }
}

fn grazing() -> Case {
    Case {
        points: include_str!("reference/grazing_points.csv"),
        reference: include_str!("reference/grazing_posterior.csv"),
        spread: Some(include_str!("reference/grazing_spread.csv")),
        truth: TransitParams::new(
            Epoch::from_days(0.0).unwrap(),
            TimeDelta::from_days(3.0).unwrap(),
            RadiusRatio::new(0.1).unwrap(),
            ScaledSemiMajorAxis::new(10.0).unwrap(),
            ImpactParameter::new(1.0).unwrap(),
            LimbDarkening::quadratic(0.4, 0.25).unwrap(),
        )
        .unwrap(),
        names: &HOT,
        steps: GRAZING_STEPS,
        discard: GRAZING_STEPS / 5,
        snooker: true,
    }
}

const ECCENTRIC: [&str; 10] = [
    "t0",
    "k",
    "a_rs",
    "b",
    "q1",
    "q2",
    "sqrt_e_cos_w",
    "sqrt_e_sin_w",
    "baseline",
    "density",
];

fn eccentric() -> Case {
    Case {
        points: include_str!("reference/eccentric_points.csv"),
        reference: include_str!("reference/eccentric_posterior.csv"),
        spread: Some(include_str!("reference/eccentric_spread.csv")),
        truth: truth().with_eccentricity(0.3, 60.0).unwrap(),
        names: &ECCENTRIC,
        steps: ECCENTRIC_STEPS,
        discard: ECCENTRIC_STEPS / 5,
        snooker: true,
    }
}

const GRAZING_STEPS: usize = 200_000;
const ECCENTRIC_STEPS: usize = 100_000;

/// The star's density at a / R* = 10 and a 3-day period, in g/cm^3.
fn density_at_ten() -> f64 {
    let period = 3.0 * 86400.0;
    3.0 * std::f64::consts::PI * 1000.0 / (6.674_30e-11 * period * period) / 1000.0
}

fn grazing_priors(p: TransitPosterior<'_>) -> TransitPosterior<'_> {
    p.with_prior(Coordinate::RadiusRatio, Prior::uniform(0.0, 0.3).unwrap())
        .unwrap()
}

fn eccentric_free(f: TransitFit<'_>) -> TransitFit<'_> {
    f.with_free(Parameter::ECosW).with_free(Parameter::ESinW)
}

fn eccentric_priors(p: TransitPosterior<'_>) -> TransitPosterior<'_> {
    let rho = density_at_ten();
    p.with_stellar_density(rho, 0.1 * rho).unwrap()
}

#[test]
fn matches_emcee_with_batman() {
    // Tolerances from a sweep of 30 seeds (see the README).
    let (centre, tail) = against_reference(&hot(), |f| f, |p| p, 11, 1);
    eprintln!("worst gaps in sigma: 16-84th {centre:.3}, 2.5th and 97.5th {tail:.3}");
    assert!(centre < 0.15 && tail < 0.3, "{centre} {tail}");
}

#[test]
#[ignore = "heavy: minutes, the grazing posterior against zeus"]
fn a_grazing_transit_matches_zeus() {
    // In 2 of 30 seeds (and 1 of zeus's 4 chains) one walker spends the
    // whole chain in the full-transit ridge (a / R* near 16), which moves
    // the 97.5th percentiles of a / R* and the density by several sigma;
    // the median of three seeds is held to the tolerances.
    let mut gaps: Vec<(f64, f64)> = [11, 13, 15]
        .map(|seed| against_reference(&grazing(), |f| f, grazing_priors, seed, 8))
        .to_vec();
    gaps.sort_by(|a, b| a.1.total_cmp(&b.1));
    let (centre, tail) = gaps[1];
    eprintln!("grazing, worst gaps in sigma: 16-84th {centre:.3}, 2.5th and 97.5th {tail:.3}");
    assert!(
        centre < GRAZING_GAPS.0 && tail < GRAZING_GAPS.1,
        "{centre} {tail}"
    );
}

#[test]
#[ignore = "heavy: minutes, the eccentric posterior against zeus"]
fn an_eccentric_orbit_with_a_density_prior_matches_zeus() {
    let (centre, tail) = against_reference(&eccentric(), eccentric_free, eccentric_priors, 11, 8);
    eprintln!("eccentric, worst gaps in sigma: 16-84th {centre:.3}, 2.5th and 97.5th {tail:.3}");
    assert!(
        centre < ECCENTRIC_GAPS.0 && tail < ECCENTRIC_GAPS.1,
        "{centre} {tail}"
    );
}

/// Over 30 seeds (the ridge aside) the worst gaps were 0.081 and 0.03.
const GRAZING_GAPS: (f64, f64) = (0.15, 0.3);
/// Over 30 seeds the worst gaps were 0.37 and 0.74 (medians 0.19 and
/// 0.34): t0 and e sin w mix slowly, in zeus's chains as in ours.
const ECCENTRIC_GAPS: (f64, f64) = (0.5, 1.0);

/// The worst gaps over 30 seeds, from which the tolerances above are set:
/// `SWEEP=grazing cargo test --release -p rustellar-mcmc --test transit --
/// --ignored --nocapture seed_sweep` (`hot`, `grazing`, `eccentric` or
/// `all`); without `SWEEP` it does nothing.
#[test]
#[ignore = "heavy: with SWEEP set, the worst gaps over 30 seeds"]
fn seed_sweep() {
    let Ok(which) = std::env::var("SWEEP") else {
        return;
    };
    for (name, case) in [
        ("hot", hot()),
        ("grazing", grazing()),
        ("eccentric", eccentric()),
    ] {
        if which != "all" && which != name {
            continue;
        }
        let one = |s: u64| match name {
            "grazing" => against_reference(&case, |f| f, grazing_priors, 100 + 2 * s, 1),
            "eccentric" => {
                against_reference(&case, eccentric_free, eccentric_priors, 100 + 2 * s, 1)
            }
            _ => against_reference(&case, |f| f, |p| p, 100 + 2 * s, 1),
        };
        let gaps: Vec<(f64, f64)> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..30).map(|s| scope.spawn(move || one(s))).collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        for (s, g) in gaps.iter().enumerate() {
            if g.1 > 1.0 {
                eprintln!("{name}: seed {} tail gap {:.2}", 100 + 2 * s, g.1);
            }
        }
        let mut centre: Vec<f64> = gaps.iter().map(|g| g.0).collect();
        let mut tail: Vec<f64> = gaps.iter().map(|g| g.1).collect();
        centre.sort_by(f64::total_cmp);
        tail.sort_by(f64::total_cmp);
        eprintln!(
            "{name}: 16-84th median {:.3} worst {:.3}; tails median {:.3} worst {:.3}",
            centre[15], centre[29], tail[15], tail[29]
        );
        eprintln!("{name} tails: {:.2?}", tail);
    }
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

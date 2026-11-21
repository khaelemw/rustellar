// Noiseless light curves recovered from offset starts: the minimum is the
// truth, so every parameter should come back to the fit's precision.

use std::num::NonZeroUsize;

use rustellar_fit::{
    Epoch, Exposure, FitError, FitResult, ImpactParameter, LightCurve, LimbDarkening, Parameter,
    RadiusRatio, ScaledSemiMajorAxis, Termination, TimeDelta, TransitFit, TransitParams,
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

/// `n` points every `step` days from `start`: the model times `baseline`,
/// each point the mean over `exposure` if given.
fn noiseless(
    truth: TransitParams,
    baseline: f64,
    (start, step, n): (f64, f64, u32),
    exposure: Option<Exposure>,
) -> LightCurve {
    let time: Vec<f64> = (0..n).map(|i| step.mul_add(f64::from(i), start)).collect();
    let flux = time
        .iter()
        .map(|&t| {
            let t = Epoch::from_days(t).unwrap();
            baseline * exposure.map_or_else(|| truth.flux_at(t), |e| truth.binned_flux_at(t, e))
        })
        .collect();
    LightCurve::from_days(time, flux, vec![1e-3; n as usize]).unwrap()
}

fn values(p: TransitParams, baseline: f64) -> [f64; 10] {
    let (u1, u2) = p.limb_darkening().quadratic_coefficients();
    let (w_sin, w_cos) = p.periastron_deg().to_radians().sin_cos();
    [
        p.t0().as_days(),
        p.period().as_days(),
        p.k().get(),
        p.a_rs().get(),
        p.b().get(),
        u1,
        u2,
        p.eccentricity() * w_cos,
        p.eccentricity() * w_sin,
        baseline,
    ]
}

/// Each parameter within `tol` of the truth, relative to its size (to 1 for
/// `b`, `u1` and `u2`, to the period for `t0`, which may be reported at
/// another transit); `b` within `b_tol`.
fn assert_recovered(result: &FitResult, truth: TransitParams, baseline: f64, tol: f64, b_tol: f64) {
    assert_eq!(result.termination(), Termination::Converged);
    let have = values(result.params(), result.baseline());
    let want = values(truth, baseline);
    let cycles = (have[0] - want[0]) / want[1];
    let errors = [
        (cycles - cycles.round()).abs(),
        (have[1] - want[1]).abs() / want[1],
        (have[2] - want[2]).abs() / want[2],
        (have[3] - want[3]).abs() / want[3],
        (have[4] - want[4]).abs() * tol / b_tol,
        (have[5] - want[5]).abs(),
        (have[6] - want[6]).abs(),
        (have[7] - want[7]).abs(),
        (have[8] - want[8]).abs(),
        (have[9] - want[9]).abs() / want[9],
    ];
    for (p, e) in Parameter::ALL.iter().zip(errors) {
        assert!(
            e <= tol,
            "{p}: {} against {}",
            have[p.index()],
            want[p.index()]
        );
    }
}

fn hot_jupiter() -> TransitParams {
    params(1.3, 3.0, 0.1, 10.0, 0.3, 0.4, 0.25)
}

#[test]
fn hot_jupiter_every_circular_parameter_free() {
    // Three transits at two-minute cadence.
    let truth = hot_jupiter();
    let lc = noiseless(truth, 1.002, (0.0, 2.0 / 1440.0, 5040), None);
    let start = params(1.302, 3.0005, 0.105, 9.0, 0.4, 0.3, 0.2);
    for accelerate in [true, false] {
        let fit = TransitFit::new(&lc, start).with_acceleration(accelerate);
        let result = fit.run().unwrap();
        assert_recovered(&result, truth, 1.002, 1e-9, 1e-9);
        assert!(result.chi_squared() < 1e-15);
        assert_eq!(result.free(), TransitFit::new(&lc, start).free());
        assert_eq!(result.free().len(), 8);
        assert_eq!(result.points(), 5040);
        assert_eq!(result.jacobians(), result.iterations() + 1);
        assert!(result.evaluations() <= 4 * result.iterations() + 4);
        // Reported at the transit nearest the data's middle (3.5 days).
        assert!((result.value(Parameter::MidTime) - 4.3).abs() < 1e-9);
    }
}

#[test]
fn epochs_far_from_zero() {
    let truth = params(2_459_001.3, 3.0, 0.1, 10.0, 0.3, 0.4, 0.25);
    let lc = noiseless(truth, 1.0, (2_459_000.0, 2.0 / 1440.0, 5040), None);
    let start = params(2_459_001.31, 3.001, 0.09, 12.0, 0.1, 0.5, 0.1);
    let result = TransitFit::new(&lc, start).run().unwrap();
    assert_recovered(&result, truth, 1.0, 1e-9, 1e-9);
}

#[test]
fn grazing() {
    let truth = params(1.0, 2.0, 0.12, 6.0, 1.02, 0.4, 0.25);
    let lc = noiseless(truth, 1.0, (0.0, 2.0 / 1440.0, 2880), None);
    let start = params(1.0, 2.0, 0.15, 6.5, 0.95, 0.4, 0.25);
    let result = TransitFit::new(&lc, start).run().unwrap();
    // Only the limb is crossed, so limb darkening is all but free (standard
    // errors of hundreds): the fit stops within 1e-10 of them, here a few
    // times 1e-9.
    assert_recovered(&result, truth, 1.0, 1e-7, 1e-7);
}

#[test]
fn small_planet() {
    // A 400 ppm transit under 1000 ppm errors: limb darkening is barely
    // constrained, so the minimum is shallow along it.
    let truth = params(5.0, 11.0, 0.02, 25.0, 0.5, 0.45, 0.2);
    let lc = noiseless(truth, 1.0, (0.0, 10.0 / 1440.0, 27 * 144), None);
    let start = params(5.02, 11.01, 0.025, 20.0, 0.2, 0.3, 0.3);
    let result = TransitFit::new(&lc, start).run().unwrap();
    assert_recovered(&result, truth, 1.0, 1e-5, 1e-5);
}

#[test]
fn central_transit_from_near_and_far() {
    // At b = 0 the flux changes with b^2, so b is found only to the square
    // root of the precision.
    let truth = params(1.0, 2.0, 0.1, 8.0, 0.0, 0.4, 0.25);
    let lc = noiseless(truth, 1.0, (0.0, 2.0 / 1440.0, 2880), None);
    for start in [
        params(1.0, 2.0, 0.11, 7.0, 0.3, 0.4, 0.25),
        params(1.01, 2.0, 0.07, 5.0, 0.6, 0.2, 0.1),
    ] {
        let result = TransitFit::new(&lc, start).run().unwrap();
        assert_recovered(&result, truth, 1.0, 1e-8, 1e-4);
    }
}

#[test]
fn long_cadence() {
    // Half-hour exposures over 30 days, the model averaged to match.
    let exposure = Exposure::new(
        TimeDelta::from_days(30.0 / 1440.0).unwrap(),
        NonZeroUsize::new(15).unwrap(),
    )
    .unwrap();
    let truth = params(3.0, 4.5123, 0.08, 12.0, 0.4, 0.4, 0.25);
    let lc = noiseless(truth, 1.0, (0.0, 30.0 / 1440.0, 1440), Some(exposure));
    let start = params(3.01, 4.514, 0.09, 10.0, 0.2, 0.3, 0.2);
    let result = TransitFit::new(&lc, start)
        .with_exposure(exposure)
        .run()
        .unwrap();
    assert_recovered(&result, truth, 1.0, 1e-9, 1e-9);
}

#[test]
fn fixed_parameters_stay_put() {
    // The mid-time held at the first transit while the period is fitted:
    // it is not moved to the data's middle.
    let truth = hot_jupiter();
    let lc = noiseless(truth, 1.0, (0.0, 2.0 / 1440.0, 5040), None);
    let start = params(1.3, 3.0004, 0.105, 9.5, 0.35, 0.4, 0.25);
    let result = TransitFit::new(&lc, start)
        .with_fixed(Parameter::MidTime)
        .with_fixed(Parameter::U1)
        .with_fixed(Parameter::U2)
        .with_fixed(Parameter::Baseline)
        .run()
        .unwrap();
    assert_recovered(&result, truth, 1.0, 1e-9, 1e-9);
    assert_eq!(result.value(Parameter::MidTime), 1.3);
    assert_eq!(result.free().len(), 4);
    assert!(result.reduced_chi_squared() < 1e-15);
}

#[test]
fn stops_at_the_iteration_limit() {
    let truth = hot_jupiter();
    let lc = noiseless(truth, 1.0, (0.0, 2.0 / 1440.0, 5040), None);
    let start = params(1.302, 3.0005, 0.105, 9.0, 0.4, 0.3, 0.2);
    let result = TransitFit::new(&lc, start)
        .with_max_iterations(2)
        .run()
        .unwrap();
    assert_eq!(result.termination(), Termination::MaxIterations);
    assert_eq!(result.iterations(), 2);
    assert!(result.chi_squared() > 1e-6);
}

#[test]
fn the_set_up_is_checked_first() {
    let lc = noiseless(hot_jupiter(), 1.0, (0.0, 0.01, 50), None);
    let mut fit = TransitFit::new(&lc, hot_jupiter());
    for p in Parameter::ALL {
        fit = fit.with_fixed(p);
    }
    assert_eq!(fit.run().unwrap_err(), FitError::NothingToFit);
}

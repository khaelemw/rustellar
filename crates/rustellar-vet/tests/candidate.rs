// A candidate's checks, its errors' messages, and the conversions from the
// searches' results (with the `bls` and `tls` features).

use rustellar_vet::{Candidate, Epoch, LightCurve, TimeDelta, VetError};

fn days(d: f64) -> TimeDelta {
    TimeDelta::from_days(d).unwrap()
}

fn at(t: f64) -> Epoch {
    Epoch::from_days(t).unwrap()
}

#[test]
fn a_candidate_keeps_what_it_is_given() {
    let c = Candidate::new(days(3.0), at(1000.5), days(0.1)).unwrap();
    assert_eq!(c.period(), days(3.0));
    assert_eq!(c.transit_time(), at(1000.5));
    assert_eq!(c.duration(), days(0.1));
    assert_eq!(c.depth(), None);
    assert!((c.duty_cycle() - 0.1 / 3.0).abs() < 1e-15);
    let deep = c.with_depth(2e-3).unwrap();
    assert_eq!(deep.depth(), Some(2e-3));
    assert_eq!(deep.period(), c.period());
    // A negative depth (a bump) is a number like any other.
    assert_eq!(c.with_depth(-1e-4).unwrap().depth(), Some(-1e-4));
}

#[test]
fn a_candidate_is_checked() {
    let new = |p, d| Candidate::new(days(p), at(0.0), days(d));
    assert_eq!(new(0.0, 0.1), Err(VetError::NotPositive { name: "period" }));
    assert_eq!(
        new(-1.0, 0.1),
        Err(VetError::NotPositive { name: "period" })
    );
    assert_eq!(
        new(1.0, 0.0),
        Err(VetError::NotPositive { name: "duration" })
    );
    assert_eq!(
        new(1.0, -0.1),
        Err(VetError::NotPositive { name: "duration" })
    );
    assert_eq!(new(1.0, 1.0), Err(VetError::DurationNotShorterThanPeriod));
    assert_eq!(new(1.0, 2.0), Err(VetError::DurationNotShorterThanPeriod));
    assert!(new(1.0, 0.999).is_ok());
    let c = new(1.0, 0.1).unwrap();
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(c.with_depth(bad), Err(VetError::DepthNotFinite));
    }
}

#[test]
fn the_errors_say_what_is_wrong() {
    let messages = [
        (
            VetError::NotPositive { name: "period" },
            "period must be greater than zero",
        ),
        (
            VetError::DurationNotShorterThanPeriod,
            "the duration must be shorter than the period",
        ),
        (
            VetError::DepthNotFinite,
            "the depth must be a finite number",
        ),
    ];
    for (err, text) in messages {
        assert_eq!(err.to_string(), text);
        let boxed: Box<dyn std::error::Error> = Box::new(err);
        assert_eq!(boxed.to_string(), text);
    }
}

// Every 30 minutes for 12 days, a dip of 0.005 lasting 0.1 days every 3
// days from 0.7.
fn dips() -> LightCurve {
    let time: Vec<f64> = (0..576).map(|i| f64::from(i) / 48.0).collect();
    let flux = time
        .iter()
        .map(|&t| {
            if (t - 0.7).rem_euclid(3.0) < 0.1 {
                0.995
            } else {
                1.0
            }
        })
        .collect();
    LightCurve::from_days(time, flux, vec![1e-3; 576]).unwrap()
}

#[cfg(feature = "bls")]
#[test]
fn from_a_bls_search() {
    use rustellar_bls::{Bls, Durations, Periods, SearchOptions};

    let lc = dips();
    let periods = Periods::from_days(&[2.0, 2.5, 3.0, 3.5]).unwrap();
    let durations = Durations::from_days(&[0.05, 0.1]).unwrap();
    let result = Bls::new(&lc)
        .unwrap()
        .search(&periods, &durations, &SearchOptions::default())
        .unwrap();
    let best = result.best().unwrap();
    let c = Candidate::try_from(best).unwrap();
    assert_eq!(c.period(), best.transit.period());
    assert_eq!(c.transit_time(), best.transit.transit_time());
    assert_eq!(c.duration(), best.transit.duration());
    assert_eq!(c.depth(), Some(best.fit.depth));
    assert!((c.period().as_days() - 3.0).abs() < 1e-12);
}

#[cfg(feature = "tls")]
#[test]
fn from_a_tls_search() {
    use rustellar_tls::{OfirPeriod, SearchOptions, Tls};

    let lc = dips();
    let periods = OfirPeriod::default().periods(&lc).unwrap();
    let result = Tls::new(&lc)
        .unwrap()
        .search(&periods, &SearchOptions::default().with_bootstrap(0))
        .unwrap();
    let best = result.best().unwrap();
    let c = Candidate::try_from(best).unwrap();
    assert_eq!(c.period(), best.period);
    assert_eq!(c.transit_time(), best.transit_time);
    assert_eq!(c.duration(), best.duration);
    assert_eq!(c.depth(), Some(best.depth));
    assert!((c.period().as_days() - 3.0).abs() < 0.01);
}

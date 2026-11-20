//! Iterative transit masking against its definition (flatten, search, mask,
//! flatten again, written out plainly) and on injected transits.

use std::cell::{Cell, RefCell};
use std::f64::consts::PI;
use std::num::NonZeroUsize;

use rustellar_core::{Epoch, LightCurve, TimeDelta};
use rustellar_detrend::{
    Candidate, Detrend, DetrendError, MASK_PER_DURATION, MASKED_WINDOW_PER_DURATION, Masking,
    Method, transit_mask,
};

fn days(d: f64) -> TimeDelta {
    TimeDelta::from_days(d).unwrap()
}

fn candidate(period: f64, t0: f64, duration: f64) -> Candidate {
    Candidate {
        period: days(period),
        t0: Epoch::from_days(t0).unwrap(),
        duration: days(duration),
    }
}

/// Every 10 minutes for 8 days: a star varying by 1% over 1.5 days, and a
/// 1% box transit 0.15 days long every 2.3 days from day 0.7, in noise of
/// 1e-4 (a fixed sequence).
fn star() -> LightCurve {
    let time: Vec<f64> = (0..1152).map(|i| f64::from(i) / 144.0).collect();
    let flux = time
        .iter()
        .enumerate()
        .map(|(i, &t)| {
            let dip = ((t - 0.7) / 2.3 + 0.5).rem_euclid(1.0) - 0.5;
            let transit = if (dip * 2.3).abs() < 0.075 { 0.99 } else { 1.0 };
            #[expect(clippy::cast_precision_loss, reason = "small")]
            let noise = 1e-4 * ((i as f64 * 0.618_034).fract() - 0.5);
            (1.0 + 0.01 * (2.0 * PI * t / 1.5).sin()) * transit + noise
        })
        .collect();
    LightCurve::from_days(time, flux, vec![1e-4; 1152]).unwrap()
}

fn five() -> NonZeroUsize {
    NonZeroUsize::new(5).unwrap()
}

const TRUE: (f64, f64, f64) = (2.3, 0.7, 0.15);

fn truth() -> Candidate {
    candidate(TRUE.0, TRUE.1, TRUE.2)
}

/// The depth the flattened `lc` keeps: the mean dimming in transit (the
/// true transits' points) over 1%.
fn depth(lc: &LightCurve) -> f64 {
    let inside = transit_mask(
        lc,
        days(TRUE.0),
        Epoch::from_days(TRUE.1).unwrap(),
        days(0.14),
    )
    .unwrap();
    let dims: Vec<f64> = lc
        .flux()
        .iter()
        .zip(&inside)
        .filter(|(_, m)| **m)
        .map(|(f, _)| 1.0 - f)
        .collect();
    #[expect(clippy::cast_precision_loss, reason = "small")]
    let mean = dims.iter().sum::<f64>() / dims.len() as f64;
    mean / 0.01
}

#[test]
fn masking_settings() {
    let m = Masking::default();
    assert_eq!(m.mask_per_duration(), MASK_PER_DURATION);
    assert_eq!(m.window_per_duration(), MASKED_WINDOW_PER_DURATION);
    assert_eq!(m.rounds().get(), 1);
    const { assert!(MASK_PER_DURATION > 1.0 && MASK_PER_DURATION < MASKED_WINDOW_PER_DURATION) };
    let m = m
        .with_mask_per_duration(1.7)
        .with_window_per_duration(4.0)
        .with_rounds(NonZeroUsize::new(9).unwrap());
    assert_eq!(
        (
            m.mask_per_duration(),
            m.window_per_duration(),
            m.rounds().get()
        ),
        (1.7, 4.0, 9)
    );
}

#[test]
fn mask_must_fit_in_window() {
    let lc = star();
    let calls = Cell::new(0);
    let search = |_: &LightCurve| {
        calls.set(calls.get() + 1);
        Some(truth())
    };
    for (mask, window) in [
        (0.0, 2.0),
        (-1.0, 2.0),
        (2.0, 2.0),
        (3.0, 2.0),
        (f64::NAN, 2.0),
        (1.0, f64::NAN),
    ] {
        let masking = Masking::default()
            .with_mask_per_duration(mask)
            .with_window_per_duration(window);
        let err = Detrend::default()
            .flatten_around(&lc, &masking, search)
            .unwrap_err();
        assert_eq!(err, DetrendError::MaskNotInWindow, "{mask} {window}");
    }
    // Checked before anything else: even before the settings' own checks.
    let bad = Detrend::default().with_window(days(0.0));
    let masking = Masking::default().with_mask_per_duration(0.0);
    assert_eq!(
        bad.flatten_around(&lc, &masking, search).unwrap_err(),
        DetrendError::MaskNotInWindow
    );
    assert_eq!(calls.get(), 0);
    assert_eq!(
        DetrendError::MaskNotInWindow.to_string(),
        "the mask must be longer than zero and shorter than the window"
    );
}

#[test]
fn errors_pass_through() {
    let lc = star();
    let bad = Detrend::default().with_window(days(0.0));
    assert_eq!(
        bad.flatten_around(&lc, &Masking::default(), |_| None)
            .unwrap_err(),
        DetrendError::NotPositive { name: "window" }
    );
    // A candidate without a duration cannot be masked.
    let err = Detrend::default()
        .flatten_around(&lc, &Masking::default(), |_| Some(candidate(2.3, 0.7, 0.0)))
        .unwrap_err();
    assert_eq!(err, DetrendError::NotPositive { name: "duration" });
    let err = Detrend::default()
        .flatten_around(&lc, &Masking::default(), |_| {
            Some(candidate(-1.0, 0.7, 0.1))
        })
        .unwrap_err();
    assert_eq!(err, DetrendError::NotPositive { name: "period" });
}

#[test]
fn nothing_found_is_the_blind_flattening() {
    let lc = star();
    let detrend = Detrend::default().with_window(days(0.4));
    let done = detrend
        .flatten_around(&lc, &Masking::default(), |_| None)
        .unwrap();
    assert_eq!(done.flattened(), &detrend.flatten(&lc, None).unwrap());
    assert_eq!(done.candidate(), None);
    assert_eq!(done.mask(), None);
    assert_eq!(done.searches(), 1);
    assert!(done.converged());
}

#[test]
fn the_same_candidate_twice_settles() {
    let lc = star();
    let detrend = Detrend::default().with_window(days(0.4));
    let masking = Masking::default().with_rounds(five());
    let seen = RefCell::new(vec![]);
    let done = detrend
        .flatten_around(&lc, &masking, |flat| {
            seen.borrow_mut().push(flat.clone());
            Some(truth())
        })
        .unwrap();
    // By definition: the default mask, then the default windows.
    let mask = transit_mask(
        &lc,
        days(TRUE.0),
        Epoch::from_days(TRUE.1).unwrap(),
        days(TRUE.2 * MASK_PER_DURATION),
    )
    .unwrap();
    let expected = detrend
        .with_window(days(TRUE.2 * MASKED_WINDOW_PER_DURATION))
        .flatten(&lc, Some(&mask))
        .unwrap();
    assert_eq!(done.flattened(), &expected);
    assert_eq!(done.mask(), Some(mask.as_slice()));
    assert_eq!(done.candidate(), Some(truth()));
    assert_eq!(done.searches(), 2);
    assert!(done.converged());
    // The first search saw the blind flattening, the second the masked one.
    let seen = seen.into_inner();
    assert_eq!(seen[0], *detrend.flatten(&lc, None).unwrap().light_curve());
    assert_eq!(seen[1], *expected.light_curve());
    assert_eq!(done.clone().into_flattened(), expected);
}

#[test]
fn a_moving_candidate_runs_out_of_rounds() {
    let lc = star();
    let detrend = Detrend::default().with_window(days(0.4));
    let masking = Masking::default().with_rounds(NonZeroUsize::new(3).unwrap());
    let round = Cell::new(0.0);
    let done = detrend
        .flatten_around(&lc, &masking, |_| {
            round.set(round.get() + 1.0);
            Some(candidate(2.3, 0.5 + 0.1 * round.get(), 0.15))
        })
        .unwrap();
    assert_eq!(done.searches(), 3);
    assert!(!done.converged());
    // The last candidate's mask is the one applied.
    assert_eq!(done.candidate(), Some(candidate(2.3, 0.8, 0.15)));
    let mask = transit_mask(
        &lc,
        days(2.3),
        Epoch::from_days(0.8).unwrap(),
        days(0.15 * MASK_PER_DURATION),
    )
    .unwrap();
    assert_eq!(done.mask(), Some(mask.as_slice()));
}

#[test]
fn a_search_that_stops_finding_keeps_the_last_mask() {
    let lc = star();
    let detrend = Detrend::default().with_window(days(0.4));
    let calls = Cell::new(0);
    let done = detrend
        .flatten_around(&lc, &Masking::default().with_rounds(five()), |_| {
            calls.set(calls.get() + 1);
            (calls.get() == 1).then(truth)
        })
        .unwrap();
    assert_eq!(done.searches(), 2);
    assert!(done.converged());
    assert_eq!(done.candidate(), Some(truth()));
    assert!(done.mask().is_some());
}

#[test]
fn a_mask_over_everything_keeps_the_last_flattening() {
    let lc = star();
    let detrend = Detrend::default().with_window(days(0.4));
    let calls = Cell::new(0);
    // Then a candidate whose mask is longer than its period.
    let done = detrend
        .flatten_around(&lc, &Masking::default().with_rounds(five()), |_| {
            calls.set(calls.get() + 1);
            Some(if calls.get() == 1 {
                truth()
            } else {
                candidate(0.6, 0.0, 0.7)
            })
        })
        .unwrap();
    assert_eq!(done.searches(), 2);
    assert!(!done.converged());
    assert_eq!(done.candidate(), Some(truth()));
    let once = detrend
        .flatten_around(
            &lc,
            &Masking::default().with_rounds(NonZeroUsize::MIN),
            |_| Some(truth()),
        )
        .unwrap();
    assert_eq!(done.flattened(), once.flattened());
    // And at once: the blind flattening stands.
    let done = detrend
        .flatten_around(&lc, &Masking::default(), |_| Some(candidate(0.6, 0.0, 0.7)))
        .unwrap();
    assert_eq!(done.flattened(), &detrend.flatten(&lc, None).unwrap());
    assert_eq!(done.candidate(), None);
    assert!(!done.converged());
}

#[test]
fn masking_keeps_the_depth_a_short_window_takes() {
    let lc = star();
    // Windows of 2.5 durations: the blind trend dips into the transit.
    for method in [Method::default(), Method::Median, Method::Spline] {
        let detrend = Detrend::default()
            .with_method(method)
            .with_window_for_transit(days(TRUE.2));
        let blind = depth(detrend.flatten(&lc, None).unwrap().light_curve());
        let done = detrend
            .flatten_around(&lc, &Masking::default(), |_| Some(truth()))
            .unwrap();
        let masked = depth(done.flattened().light_curve());
        assert!(blind < 0.9, "{method:?} {blind}");
        assert!((masked - 1.0).abs() < 0.02, "{method:?} {masked}");
    }
}

#[test]
fn a_masked_flattening_that_fails_is_an_error() {
    // So short a candidate that its windows split every point into its own
    // segment, which the edge cutoff then removes.
    let lc = star();
    let detrend = Detrend::default()
        .with_window(days(0.4))
        .with_edge_cutoff(days(0.01));
    assert!(detrend.flatten(&lc, None).is_ok());
    let err = detrend
        .flatten_around(&lc, &Masking::default(), |_| {
            Some(candidate(2.3, 0.7, 1e-4))
        })
        .unwrap_err();
    assert_eq!(err, DetrendError::NoPointsLeft);
}

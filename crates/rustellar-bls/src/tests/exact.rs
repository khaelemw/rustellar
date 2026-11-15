// The exact search over several durations at one fold: the best of the exact
// search run at each duration in turn.

use rustellar_core::{LightCurve, TimeDelta};

use crate::{TransitBox, box_fit::Weighted, exact::Folded};

fn dip() -> LightCurve {
    LightCurve::from_btjd(
        vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        vec![1.0, 0.9, 0.9, 1.0, 1.0, 1.0],
        vec![1e-3; 6],
    )
    .unwrap()
}

// The 3-day box is the one that holds just the dip at 1 and 2, and only one
// mid-transit time gives it: the edges sit on the points at 0 and 3.
#[test]
fn best_of_the_durations() {
    let lc = dip();
    let weighted = Weighted::new(&lc).unwrap();
    let folded = Folded::new(&lc, &weighted, 10.0);
    let (duration, centre, fit) = folded
        .best_of(&weighted, &[0.5, 3.0, 12.0])
        .unwrap()
        .unwrap();
    assert_eq!((duration, centre, fit.n_in), (1, 1.5, 2));
    for (k, d) in [0.5, 3.0].into_iter().enumerate() {
        let one = TransitBox::best(
            &lc,
            TimeDelta::from_days(10.0).unwrap(),
            TimeDelta::from_days(d).unwrap(),
        )
        .unwrap()
        .unwrap();
        assert!(one.fit.delta_chi2 <= fit.delta_chi2, "duration {k}");
    }
}

// Durations not shorter than the period are skipped, so there may be none.
#[test]
fn every_duration_too_long() {
    let lc = dip();
    let weighted = Weighted::new(&lc).unwrap();
    let folded = Folded::new(&lc, &weighted, 2.5);
    assert_eq!(folded.best_of(&weighted, &[2.5, 3.0]).unwrap(), None);
}

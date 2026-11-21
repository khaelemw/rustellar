// The slice update's edge cases: a height rounded to the walker's own
// log-probability, a target flat forever, and the width's tuning factor.

use crate::rng::Rng;
use crate::run::tuning_factor;
use crate::slice::update;

#[test]
fn stays_where_the_slice_is_a_point() {
    // A log-probability so large that the height rounds to it: no point
    // but the walker's own is above, and the interval shrinks onto it.
    let target = |_: &[f64]| 1e20;
    let mut rng = Rng::new(1);
    let u = update(&target, &[1.0, 2.0], 1e20, &[0.5, 0.5], &mut rng);
    assert_eq!(u.to, vec![1.0, 2.0]);
    assert_eq!(u.log_probability, 1e20);
    assert_eq!(u.expansions, 0);
    assert!(u.contractions > 10);
    assert_eq!(u.evaluations, 2 + u.contractions);
}

#[test]
fn expansions_end_on_a_flat_target() {
    let target = |_: &[f64]| 0.0;
    let mut rng = Rng::new(2);
    let u = update(&target, &[0.0], 0.0, &[1.0], &mut rng);
    assert_eq!(u.expansions, 9_999);
    assert_eq!(u.contractions, 0);
    assert_eq!(u.evaluations, 10_000);
    // A NaN beyond the walker counts as outside.
    let nan = |x: &[f64]| if x[0].abs() < 1.0 { 0.0 } else { f64::NAN };
    let u = update(&nan, &[0.0], 0.0, &[0.1], &mut rng);
    assert!(u.to[0].abs() < 1.0);
}

#[test]
fn tuning() {
    assert_eq!(tuning_factor(0, 0), 1.0);
    assert_eq!(tuning_factor(5, 5), 1.0);
    assert_eq!(tuning_factor(0, 9), 0.5);
    assert_eq!(tuning_factor(9, 0), 2.0);
    assert_eq!(tuning_factor(3, 1), 1.5);
}

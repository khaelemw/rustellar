use crate::Termination;
use crate::lm::{Residuals, Settings, solve};

/// Residuals from a closure, feasible where `feasible` says, projected by
/// `project`.
struct Test<F, G, H> {
    m: usize,
    f: F,
    feasible: G,
    project: H,
}

impl<F, G, H> Residuals for Test<F, G, H>
where
    F: FnMut(&[f64], &mut [f64]),
    G: Fn(&[f64]) -> bool,
    H: Fn(&mut [f64]),
{
    fn len(&self) -> usize {
        self.m
    }

    fn eval(&mut self, x: &[f64], out: &mut [f64]) -> bool {
        if !(self.feasible)(x) {
            return false;
        }
        (self.f)(x, out);
        true
    }

    fn project(&self, x: &mut [f64]) {
        (self.project)(x);
    }
}

const fn settings(max_iterations: usize, accelerate: bool) -> Settings {
    Settings {
        max_iterations,
        accelerate,
    }
}

fn rosenbrock() -> impl Residuals + use<> {
    Test {
        m: 2,
        f: |x: &[f64], out: &mut [f64]| {
            out[0] = 10.0 * x[0].mul_add(-x[0], x[1]);
            out[1] = 1.0 - x[0];
        },
        feasible: |_: &[f64]| true,
        project: |_: &mut [f64]| {},
    }
}

#[test]
fn rosenbrock_with_and_without_acceleration() {
    for accelerate in [true, false] {
        let mut p = rosenbrock();
        let s = solve(
            &mut p,
            vec![-1.2, 1.0],
            &[1e-6, 1e-6],
            settings(200, accelerate),
        );
        assert_eq!(s.termination, Termination::Converged);
        assert!((s.x[0] - 1.0).abs() < 1e-10 && (s.x[1] - 1.0).abs() < 1e-10);
        assert!(s.iterations > 5);
        // One evaluation to start, two per Jacobian, one or two per trial.
        assert!(s.evaluations > 1 + 3 * s.iterations);
    }
}

#[test]
fn stops_at_the_iteration_limit() {
    let mut p = rosenbrock();
    let s = solve(&mut p, vec![-1.2, 1.0], &[1e-6, 1e-6], settings(1, true));
    assert_eq!(s.termination, Termination::MaxIterations);
    assert_eq!(s.iterations, 1);
}

#[test]
fn already_at_the_minimum() {
    let mut p = Test {
        m: 3,
        f: |x: &[f64], out: &mut [f64]| {
            out[0] = x[0] - 1.0;
            out[1] = x[1] - 2.0;
            out[2] = x[0] - x[1] + 1.0;
        },
        feasible: |_: &[f64]| true,
        project: |_: &mut [f64]| {},
    };
    let s = solve(&mut p, vec![1.0, 2.0], &[1e-3, 1e-3], settings(10, true));
    assert_eq!(s.termination, Termination::Converged);
    assert_eq!((s.iterations, s.evaluations), (0, 5));
    // R of [[1, 0], [0, 1], [1, -1]]: det(R)^2 = det(J^T J) = 3.
    let det = s.r[0] * s.r[3];
    assert!((det.abs() - 3f64.sqrt()).abs() < 1e-9);
    assert_eq!(s.r[2], 0.0);
}

#[test]
fn uncoupled_parameters() {
    // A diagonal R: the damping rows meet zeros beyond their own column.
    let mut p = Test {
        m: 2,
        f: |x: &[f64], out: &mut [f64]| {
            out[0] = x[0] - 1.0;
            out[1] = 3.0 * (x[1] - 2.0);
        },
        feasible: |_: &[f64]| true,
        project: |_: &mut [f64]| {},
    };
    let s = solve(&mut p, vec![0.0, 0.0], &[1e-3, 1e-3], settings(50, false));
    assert_eq!(s.termination, Termination::Converged);
    assert!((s.x[0] - 1.0).abs() < 1e-12 && (s.x[1] - 2.0).abs() < 1e-12);
}

#[test]
fn slides_to_a_lower_bound() {
    // The minimum of (x + 1)^2 over x >= 0 is on the bound, where the
    // Jacobian is one-sided and every step is projected back to x = 0.
    let mut p = Test {
        m: 1,
        f: |x: &[f64], out: &mut [f64]| out[0] = x[0] + 1.0,
        feasible: |x: &[f64]| x[0] >= 0.0,
        project: |x: &mut [f64]| x[0] = x[0].max(0.0),
    };
    let s = solve(&mut p, vec![1.0], &[1e-6], settings(100, true));
    assert_eq!(s.termination, Termination::PrecisionLimit);
    assert_eq!(s.x, [0.0]);
    assert!((s.f[0] - 1.0).abs() < 1e-15);
}

#[test]
fn stops_at_an_upper_bound_it_cannot_cross() {
    // The minimum at x = 1 is beyond the bound x <= 0.5, which the
    // projection does not know of: steps across it are rejected, the
    // acceleration's probe across it too, until the steps vanish.
    let mut p = Test {
        m: 1,
        f: |x: &[f64], out: &mut [f64]| out[0] = x[0] - 1.0,
        feasible: |x: &[f64]| x[0] <= 0.5,
        project: |_: &mut [f64]| {},
    };
    let s = solve(&mut p, vec![0.0], &[1e-6], settings(1000, true));
    assert_eq!(s.termination, Termination::PrecisionLimit);
    assert!(s.x[0] <= 0.5 && s.x[0] > 0.5 - 1e-9);
}

#[test]
fn a_parameter_with_no_feasible_neighbour() {
    // Only x1 = 0 is feasible: its column is zero and it stays put.
    let mut p = Test {
        m: 2,
        f: |x: &[f64], out: &mut [f64]| {
            out[0] = x[0] - 1.0;
            out[1] = x[1];
        },
        feasible: |x: &[f64]| x[1] == 0.0,
        project: |x: &mut [f64]| x[1] = 0.0,
    };
    let s = solve(&mut p, vec![0.0, 0.0], &[1e-6, 1e-6], settings(50, true));
    assert_eq!(s.termination, Termination::Converged);
    assert!((s.x[0] - 1.0).abs() < 1e-9);
    assert_eq!(s.x[1], 0.0);
    assert_eq!(s.r[3], 0.0);
}

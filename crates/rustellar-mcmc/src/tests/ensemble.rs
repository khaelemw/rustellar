// The rank of the walkers' spread: full, on a line, on a plane in three
// dimensions, to rounding, and with repeated points.

use crate::ensemble::rank;
use crate::rng::Rng;
use crate::{Ensemble, Move};

#[test]
fn ranks() {
    let square = vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 1.0],
    ];
    assert_eq!(rank(&square), 2);
    let line = vec![
        vec![0.0, 0.0],
        vec![1.0, 2.0],
        vec![2.0, 4.0],
        vec![3.0, 6.0],
    ];
    assert_eq!(rank(&line), 1);
    // A plane in three dimensions, its points far from the origin.
    let plane: Vec<Vec<f64>> = (0..6)
        .map(|i| {
            let (a, b) = (f64::from(i), f64::from(i * i % 5));
            vec![1e6 + a, 1e6 + b, 1e6 + 0.1 * a + 0.3 * b]
        })
        .collect();
    assert_eq!(rank(&plane), 2);
    // Every point the same, or repeated.
    assert_eq!(rank(&[vec![1.0, 2.0], vec![1.0, 2.0]]), 0);
    let twice = vec![
        vec![0.0, 1.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 0.0],
    ];
    assert_eq!(rank(&twice), 1);
}

#[test]
fn moves_picked_by_weight() {
    let (narrow, wide) = (Move::stretch(1.5).unwrap(), Move::stretch(3.0).unwrap());
    let ensemble = Ensemble::new(vec![vec![0.0], vec![1.0]])
        .unwrap()
        .with_moves(vec![(narrow, 1.0), (wide, 3.0)])
        .unwrap();
    let mut rng = Rng::new(4);
    let n = 40_000;
    let narrow_count = (0..n).filter(|_| ensemble.pick(&mut rng) == narrow).count();
    // A quarter, within five standard deviations (87).
    assert!(
        (narrow_count as f64 - 1e4).abs() < 5.0 * 86.6,
        "{narrow_count}"
    );
    // One move: no random number drawn.
    let single = Ensemble::new(vec![vec![0.0], vec![1.0]]).unwrap();
    let (mut a, b) = (Rng::new(5), Rng::new(5));
    assert_eq!(single.pick(&mut a), Move::default());
    assert_eq!(a, b);
}

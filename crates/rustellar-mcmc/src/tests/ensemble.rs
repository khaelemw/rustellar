// The rank of the walkers' spread: full, on a line, on a plane in three
// dimensions, to rounding, and with repeated points.

use crate::ensemble::rank;

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

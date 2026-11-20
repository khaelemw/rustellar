// The noise measured out of transit: white noise that averages away in
// each bin, offsets held for a whole bin, partial bins dropped, and too
// few points or bins.

use crate::noise::measure;

// 10 bins of 0.1 days, 10 points each, the residuals `offset(bin) +/-
// white` alternating within each bin (so each bin's mean is its offset).
fn binned(offset: impl Fn(usize) -> f64, white: f64) -> (Vec<f64>, Vec<f64>) {
    let mut times = vec![];
    let mut flux = vec![];
    for b in 0..10u8 {
        for i in 0..10u8 {
            // Clear of the bin edges, which run from the first time.
            let (bf, fi) = (f64::from(b), f64::from(i));
            let start = if b + i == 0 { 0.0 } else { 0.001 };
            times.push(bf * 0.1 + start + fi * 0.0095);
            let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
            flux.push(2.0 * (1.0 + offset(usize::from(b)) + sign * white));
        }
    }
    (times, flux)
}

#[test]
fn white_noise_averages_away() {
    let (times, flux) = binned(|_| 0.0, 1e-3);
    let n = measure(&times, &flux, &vec![2e-3; 100], &vec![0.0; 100], 2.0, 0.1);
    // 100 values of +/- 1e-3: the sample deviation is 1e-3 sqrt(100 / 99).
    assert!((n.white - 1e-3 * (100.0f64 / 99.0).sqrt()).abs() < 1e-15);
    // The errors given are 2e-3 in flux, 1e-3 of the baseline: right.
    assert!((n.error_scale - (100.0f64 / 99.0).sqrt()).abs() < 1e-12);
    assert!(n.binned.abs() < 1e-15);
    assert_eq!((n.red, n.bins), (0.0, 10));
}

#[test]
fn offsets_held_for_a_bin_are_red() {
    // Offsets of +/- 2e-3 bin by bin: the binned scatter is theirs, and the
    // red noise is what the white noise does not explain.
    let (times, flux) = binned(|b| if b % 2 == 0 { 2e-3 } else { -2e-3 }, 1e-3);
    let n = measure(&times, &flux, &vec![2e-3; 100], &vec![0.0; 100], 2.0, 0.1);
    let binned = 2e-3 * (10.0f64 / 9.0).sqrt();
    assert!((n.binned - binned).abs() < 1e-15, "{n:?}");
    let red = (binned * binned - n.white * n.white / 10.0).sqrt();
    assert!((n.red - red).abs() < 1e-15, "{n:?}");
}

#[test]
fn points_in_transit_and_partial_bins_are_left_out() {
    let (mut times, mut flux) = binned(|_| 0.0, 1e-3);
    // In transit: excluded however far off.
    let mut shapes = vec![0.0; 100];
    shapes[5] = 1.0;
    flux[5] = 0.0;
    // A partial bin, 3 points of 10, with a large offset: dropped.
    for i in 0..3 {
        times.push(1.001 + f64::from(i) * 0.0095);
        flux.push(2.1);
        shapes.push(0.0);
    }
    let n = measure(&times, &flux, &vec![2e-3; 103], &shapes, 2.0, 0.1);
    assert_eq!(n.bins, 10);
    assert_eq!(n.red, 0.0);
}

#[test]
fn too_few_points_or_bins() {
    // One bin: no binned scatter, and no red noise.
    let n = measure(
        &[0.0, 0.01, 0.02],
        &[1.0, 1.1, 0.9],
        &[0.1; 3],
        &[0.0; 3],
        1.0,
        1.0,
    );
    assert!((n.white - 0.1).abs() < 1e-12);
    assert!((n.error_scale - 1.0).abs() < 1e-12);
    assert!(n.binned.is_nan());
    assert_eq!((n.red, n.bins), (0.0, 1));
    // One point: no scatter at all, and no bin of two.
    let n = measure(&[0.0], &[1.0], &[1.0], &[0.0], 1.0, 1.0);
    assert!(n.white.is_nan() && n.error_scale.is_nan());
    assert_eq!((n.red, n.bins), (0.0, 0));
    // None out of transit.
    let n = measure(&[0.0], &[1.0], &[1.0], &[1.0], 1.0, 1.0);
    assert!(n.white.is_nan());
    assert_eq!(n.bins, 0);
}

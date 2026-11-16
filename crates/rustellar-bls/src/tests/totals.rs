// The box statistic of every box at once against one box at a time: the same
// to the last bit, empty boxes included.

use rstest::rstest;

use crate::{
    BlsError, Objective,
    totals::{Orbit, Total},
};

// Bins with weights and centred fluxes, some empty, twice round the orbit.
fn totals(m: usize) -> Vec<Total> {
    let mut state = 12_345_u64;
    let mut uniform = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    let bins: Vec<(f64, f64)> = (0..m)
        .map(|k| {
            if k % 5 == 3 {
                (0.0, 0.0)
            } else {
                let w = 1e6 * (0.5 + uniform());
                (w, w * 1e-3 * (uniform() - 0.5))
            }
        })
        .collect();
    let mut total = Total::default();
    let mut out = vec![total];
    for &(w, f) in bins.iter().chain(&bins) {
        total.weight.add(w);
        total.flux.add(f);
        out.push(total);
    }
    out
}

#[rstest]
fn every_box_as_one_box(
    #[values(Objective::DeltaChi2, Objective::Snr, Objective::AstropyLikelihood)]
    objective: Objective,
) {
    let m = 40;
    let totals = totals(m);
    let orbit = Orbit::new(&totals[0], &totals[m], objective);
    let mut empty = 0;
    for length in 0..m {
        let mut scores = vec![f64::NAN; m];
        orbit.statistics(&totals, length, &mut scores).unwrap();
        for (k, score) in scores.iter().enumerate() {
            let one = orbit
                .statistic(&totals[k], &totals[k + length], &totals[k + m])
                .unwrap();
            assert_eq!(score.to_bits(), one.to_bits(), "length {length}, box {k}");
            empty += usize::from(length == 0 && one == 0.0);
        }
    }
    assert_eq!(empty, m);
}

// Two bins of fluxes 1e200 and -1e200: a box of either one's delta_chi2
// overflows, an error from both, for every objective.
#[rstest]
fn overflow_in_either(
    #[values(Objective::DeltaChi2, Objective::Snr, Objective::AstropyLikelihood)]
    objective: Objective,
) {
    let mut total = Total::default();
    let mut totals = vec![total];
    for f in [1e200, -1e200, 1e200, -1e200] {
        total.weight.add(1.0);
        total.flux.add(f);
        totals.push(total);
    }
    let orbit = Orbit::new(&totals[0], &totals[2], objective);
    let mut scores = [0.0; 2];
    assert_eq!(
        orbit.statistics(&totals, 1, &mut scores),
        Err(BlsError::OutOfRange)
    );
    assert_eq!(
        orbit.statistic(&totals[1], &totals[2], &totals[3]),
        Err(BlsError::OutOfRange)
    );
}

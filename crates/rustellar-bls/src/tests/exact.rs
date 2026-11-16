// The exact search over several durations at one fold: the best of the exact
// search run at each duration in turn.

use rustellar_core::{LightCurve, TimeDelta};

use rstest::rstest;

use crate::{
    BoxFit, Objective, TransitBox,
    box_fit::{Weighted, offset_from_transit},
    exact::Folded,
};

fn btjd(lc: &LightCurve) -> Vec<f64> {
    lc.time().iter().map(|t| t.as_btjd()).collect()
}

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
    let folded = Folded::new(&btjd(&lc), &weighted, 10.0);
    let (duration, centre, fit) = folded
        .best_of(&weighted, &[0.5, 3.0, 12.0], Objective::DeltaChi2)
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
    let folded = Folded::new(&btjd(&lc), &weighted, 2.5);
    assert_eq!(
        folded
            .best_of(&weighted, &[2.5, 3.0], Objective::DeltaChi2)
            .unwrap(),
        None
    );
}

// Dense points and a range of durations: many possible ends for each first
// point, so the search skips blocks of ends by their bound. It must find the
// run that scoring every run finds. One point has almost no weight, so with
// the shortest duration short enough the blocks of runs from it alone have
// infinite bounds, and there are enough ends for two levels of blocks.
fn dense(seed: u64, depth: f64) -> LightCurve {
    let mut state = seed;
    let mut uniform = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    let n = 3000;
    let mut time: Vec<f64> = (0..n).map(|_| 30.0 * uniform()).collect();
    time.sort_by(f64::total_cmp);
    let flux = time
        .iter()
        .map(|&t| {
            let noise = 1e-3 * (uniform() + uniform() + uniform() - 1.5);
            let phase = (t - 1.1).rem_euclid(3.0);
            1.0 + noise - if phase < 0.12 { depth } else { 0.0 }
        })
        .collect();
    let mut err = vec![1e-3; n];
    err[n / 2] = 1e10;
    LightCurve::from_btjd(time, flux, err).unwrap()
}

// Every run that a box from `shortest` to `longest` days holds, scored from
// plain sums: the best by `objective`, as (n_in, score).
fn every_run(
    lc: &LightCurve,
    period: f64,
    shortest: f64,
    longest: f64,
    objective: Objective,
) -> (usize, f64) {
    let times = btjd(lc);
    let weighted = Weighted::new(lc).unwrap();
    let n = times.len();
    let mut order: Vec<(f64, usize)> = times
        .iter()
        .enumerate()
        .map(|(k, &t)| (offset_from_transit(t, times[0], period), k))
        .collect();
    order.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    let phase: Vec<f64> = (0..2 * n)
        .map(|k| order[k % n].0 + if k < n { 0.0 } else { period })
        .collect();
    let (mut weight, mut flux) = (vec![0.0; 2 * n + 1], vec![0.0; 2 * n + 1]);
    for k in 0..2 * n {
        let point = order[k % n].1;
        weight[k + 1] = weight[k] + weighted.weights[point];
        flux[k + 1] = flux[k] + weighted.centred[point];
    }
    let mut best = (0, 0.0);
    for i in 0..n {
        let before = if i == 0 {
            phase[n - 1] - period
        } else {
            phase[i - 1]
        };
        for j in i + 1..i + n {
            if phase[j - 1] - phase[i] >= longest {
                break;
            }
            if phase[j] - before < shortest {
                continue;
            }
            let inside = [weight[j] - weight[i], flux[j] - flux[i]];
            let outside = [weight[i + n] - weight[j], flux[i + n] - flux[j]];
            // The point with almost no weight alone: its weight is lost in
            // plain sums, and its score is near zero anyway.
            let Ok(fit) = BoxFit::from_sums(weighted.mean, inside, outside, j - i) else {
                continue;
            };
            if fit.depth > 0.0 && objective.of(&fit) > best.1 {
                best = (j - i, objective.of(&fit));
            }
        }
    }
    best
}

#[rstest]
fn skipping_blocks_finds_the_best_run(
    #[values(1, 2, 3)] seed: u64,
    #[values(0.0, 1e-3)] depth: f64,
    #[values(1e-4, 0.02)] shortest: f64,
    #[values(Objective::DeltaChi2, Objective::Snr, Objective::AstropyLikelihood)]
    objective: Objective,
) {
    let lc = dense(seed, depth);
    let weighted = Weighted::new(&lc).unwrap();
    let folded = Folded::new(&btjd(&lc), &weighted, 3.0);
    let (_, _, fit) = folded
        .best_between(&weighted, shortest, 0.4, objective)
        .unwrap()
        .unwrap();
    let (n_in, score) = every_run(&lc, 3.0, shortest, 0.4, objective);
    assert_eq!(fit.n_in, n_in);
    assert!((objective.of(&fit) - score).abs() <= 1e-9 * score);
}

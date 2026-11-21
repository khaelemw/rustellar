// The diagnostics against emcee 3.1.6 (`integrated_time`) and ArviZ 1.0
// (rank-normalised split-R-hat, bulk and tail ESS), on chains rebuilt from
// the same SplitMix64 stream with only +, *, / and floor, so bit-identical
// to the reference's.

use rustellar_mcmc::diagnostics::{ess_bulk, ess_tail, integrated_time, rhat};

struct SplitMix(u64);

impl SplitMix {
    fn unit(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 * 2f64.powi(-53)
    }
}

fn chains(
    kind: &str,
    n_chain: usize,
    n_draw: usize,
    seed: u64,
    phi: f64,
    offset: f64,
) -> Vec<Vec<f64>> {
    (0..n_chain)
        .map(|k| {
            let mut g = SplitMix(seed * 1000 + k as u64);
            let mut y = 0.0;
            (0..n_draw)
                .map(|t| {
                    let u = g.unit();
                    match kind {
                        "ar" => {
                            y = if t > 0 { phi * y + (u - 0.5) } else { u - 0.5 };
                            y + offset * k as f64
                        }
                        "ties" => (5.0 * u).floor(),
                        "heavy" => 1.0 / u,
                        "constant" => 1.0,
                        "trend" => t as f64 / n_draw as f64 + 0.01 * (u - 0.5),
                        _ => panic!("{kind}"),
                    }
                })
                .collect()
        })
        .collect()
}

fn close(case: &str, what: &str, got: f64, want: f64, rel: f64) {
    if want.is_nan() {
        assert!(got.is_nan(), "{case} {what}: {got} not NaN");
    } else {
        let err = (got - want).abs() / want.abs().max(1.0);
        assert!(err <= rel, "{case} {what}: {got} vs {want} ({err:e})");
    }
}

#[test]
fn match_emcee_and_arviz() {
    let text = include_str!("reference/diagnostics.csv");
    let mut cases = 0;
    let (mut worst_tau, mut worst_rhat, mut worst_ess) = (0.0f64, 0.0f64, 0.0f64);
    for line in text.lines().skip(1) {
        let f: Vec<&str> = line.split(',').collect();
        let num = |i: usize| f[i].parse::<f64>().unwrap();
        let x = chains(
            f[1],
            f[2].parse().unwrap(),
            f[3].parse().unwrap(),
            f[4].parse().unwrap(),
            num(5),
            num(6),
        );
        let it = integrated_time(&x, 5.0);
        assert_eq!(it.window, f[8].parse::<usize>().unwrap(), "{}", f[0]);
        close(f[0], "tau", it.tau, num(7), 1e-13);
        close(f[0], "rhat", rhat(&x), num(9), 1e-14);
        close(f[0], "ess_bulk", ess_bulk(&x), num(10), 1e-13);
        close(f[0], "ess_tail", ess_tail(&x), num(11), 1e-13);
        let rel = |a: f64, b: f64| {
            if b.is_nan() {
                0.0
            } else {
                (a - b).abs() / b.abs().max(1.0)
            }
        };
        worst_tau = worst_tau.max(rel(it.tau, num(7)));
        worst_rhat = worst_rhat.max(rel(rhat(&x), num(9)));
        worst_ess = worst_ess
            .max(rel(ess_bulk(&x), num(10)))
            .max(rel(ess_tail(&x), num(11)));
        cases += 1;
    }
    eprintln!("worst relative: tau {worst_tau:e}, rhat {worst_rhat:e}, ess {worst_ess:e}");
    assert_eq!(cases, 14);
}

#[test]
fn chain_helpers_take_walkers_as_chains() {
    use rustellar_mcmc::Ensemble;
    let target = |x: &[f64]| -0.5 * (x[0] * x[0] + 4.0 * x[1] * x[1]);
    let start = (0..10)
        .map(|i| vec![0.1 * f64::from(i) - 0.45, 0.05 * f64::from(i % 3)])
        .collect();
    let chain = Ensemble::new(start)
        .unwrap()
        .with_seed(8)
        .run(&target, 3000)
        .unwrap();
    let times = chain.autocorrelation_times(500);
    let (r, bulk, tail) = (chain.rhat(500), chain.ess_bulk(500), chain.ess_tail(500));
    for d in 0..2 {
        let traces = chain.traces(d, 500);
        assert_eq!((traces.len(), traces[0].len()), (10, 2500));
        assert_eq!(traces[3], chain.trace(3, d)[500..]);
        assert_eq!(times[d], integrated_time(&traces, 5.0));
        assert_eq!(r[d], rhat(&traces));
        assert_eq!(bulk[d], ess_bulk(&traces));
        assert_eq!(tail[d], ess_tail(&traces));
        // A well-mixed run: tau of a few tens of steps, trusted.
        assert!(times[d].tau > 2.0 && times[d].tau < 60.0, "{:?}", times[d]);
        assert!(times[d].trusted(40.0) && !times[d].trusted(1e4));
        assert!(r[d] < 1.05, "{}", r[d]);
        assert!(bulk[d] > 300.0 && tail[d] > 300.0);
    }
    // Discarding everything: no draws, so NaN.
    assert!(chain.rhat(5000)[0].is_nan() && chain.ess_bulk(3000)[1].is_nan());
}

#[test]
fn too_few_draws_or_chains_or_any_nan() {
    let two = vec![vec![0.1, 0.4, 0.2, 0.9], vec![0.3, 0.5, 0.8, 0.6]];
    assert!(rhat(&two[..1]).is_nan());
    assert!(rhat(&two).is_finite() && ess_bulk(&two[..1]).is_finite());
    let short: Vec<Vec<f64>> = two.iter().map(|c| c[..3].to_vec()).collect();
    assert!(rhat(&short).is_nan() && ess_bulk(&short).is_nan() && ess_tail(&short).is_nan());
    let mut nan = two.clone();
    nan[1][2] = f64::NAN;
    assert!(rhat(&nan).is_nan() && ess_bulk(&nan).is_nan() && ess_tail(&nan).is_nan());
}

#[test]
#[should_panic(expected = "no chains")]
fn no_chains() {
    let _ = rhat(&[]);
}

#[test]
#[should_panic(expected = "differ in length")]
fn ragged_chains() {
    let _ = ess_bulk(&[vec![1.0, 2.0, 3.0, 4.0], vec![1.0, 2.0, 3.0]]);
}

#[test]
#[should_panic(expected = "no steps")]
fn no_steps() {
    let _ = integrated_time(&[vec![], vec![]], 5.0);
}

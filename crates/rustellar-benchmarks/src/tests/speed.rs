// The speed tables: the reference files read, the times formatted, each
// crate's cases built as its README describes and timed on small inputs.

use std::num::NonZeroUsize;

use crate::speed::{
    Reference, best, bls, detrend, ls, model, package, references, seconds, short, significant,
    tls, two_figures,
};

const TWO: NonZeroUsize = NonZeroUsize::new(2).unwrap();

fn table(lines: &[&str]) -> Vec<Reference> {
    references(&format!("case,package,seconds\n{}\n", lines.join("\n")))
}

#[test]
fn the_reference_files() {
    let rows = table(&["curve 100,batman 2.5.1,7.5e-06", "", "flux 100,exo 1,2"]);
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].case, "curve 100");
    assert_eq!(package(&rows, "batman"), "batman 2.5.1");
    assert_eq!(package(&rows, "astropy"), "astropy");
    assert_eq!(seconds(&rows, "batman", "curve 100"), Some(7.5e-6));
    assert_eq!(seconds(&rows, "batman", "flux 100"), None);
    // Every file in the crate reads, with a package and version on each row.
    for csv in [
        model::reference(),
        detrend::reference(),
        bls::reference(),
        tls::reference(),
        ls::reference(),
    ] {
        let rows = references(csv);
        assert!(!rows.is_empty());
        assert!(
            rows.iter()
                .all(|r| r.package.contains(' ') && r.seconds > 0.0)
        );
    }
}

#[test]
#[should_panic(expected = "a row of three fields")]
fn a_short_row() {
    let _ = table(&["curve 100,batman"]);
}

#[test]
fn the_figures() {
    assert_eq!(two_figures(4.0), "4.0");
    assert_eq!(two_figures(40.4), "40");
    assert_eq!(two_figures(0.644), "0.64");
    assert_eq!(two_figures(9.96), "10");
    assert_eq!(two_figures(123.0), "123");
    assert_eq!(significant(0.010_44, 3), "0.0104");
    assert_eq!(significant(10.64, 3), "10.6");
    assert_eq!(significant(0.5, 0), "0.5");
    assert_eq!(short(Some(4.0e-6)), "4.0 µs");
    assert_eq!(short(Some(6.4e-4)), "0.64 ms");
    assert_eq!(short(Some(0.25)), "0.25 s");
    assert_eq!(short(None), "-");
    assert_eq!(bls::thousands(166_108.0), "166,108");
    assert_eq!(bls::thousands(999.0), "999");
    assert_eq!(bls::thousands(1_000_000.0), "1,000,000");
}

#[test]
fn the_best_of_the_repeats() {
    let mut calls = 0;
    let s = best(3, 4, || calls += 1);
    assert_eq!(calls, 12);
    assert!(s >= 0.0 && s.is_finite());
}

#[test]
fn the_model_table() {
    let timings = model::run(1);
    assert_eq!(timings.len(), 2 * model::SIZES.len() + model::SAMPLES.len());
    let refs = references(model::reference());
    let out = model::report(&timings, &refs);
    assert!(out.contains("| batman 2.5.1 | 7.5 µs |"), "{out}");
    assert!(out.contains("| exoplanet-core 0.4.0, flux only (no orbit) |"));
    assert!(out.contains("| PyTransit 2.6.18 | 0.20 ms | 0.59 ms | 1.2 ms |"));
    // A missing time is a dash.
    let out = model::report(&[], &[]);
    assert!(out.contains("| rustellar-model | - | - | - | - |"), "{out}");
    assert!(out.contains("| batman | - | - | - |"), "{out}");
}

#[test]
fn the_detrend_table() {
    let sets = detrend::light_curves();
    let points: Vec<usize> = sets.iter().map(|s| s.lc.len()).collect();
    assert_eq!(points, [4265, 18139, 108_878]);
    let timings = detrend::run(1, TWO, &sets[..1], 0);
    assert_eq!(timings.len(), 2 + detrend::METHODS.len());
    assert!(timings.iter().any(|(c, _)| c == "biweight kepler 2"));
    let refs = table(&[
        "biweight kepler,wotan 1.10,0.0104",
        "method mean,wotan 1.10,0.0099",
    ]);
    let out = detrend::report(&timings, &refs, TWO);
    assert!(
        out.contains(
            "| Biweight | wotan 1.10 | rustellar-detrend, 1 thread | rustellar-detrend, 2 threads |"
        ),
        "{out}"
    );
    assert!(out.contains("| Kepler quarter | 0.0104 | "), "{out}");
    assert!(out.contains("| TESS, 20 seconds | - | - | - |"), "{out}");
    assert!(out.contains("| Mean | 0.00990 | "), "{out}");
}

#[test]
fn the_bls_table() {
    let cases = bls::cases();
    let periods: Vec<usize> = cases.iter().map(|c| c.periods.as_slice().len()).collect();
    assert_eq!(periods, [10_000, 2_353, 42_692, 166_108]);
    let few = rustellar_core::Periods::from_days(&[2.0, 3.0, 4.7]).unwrap();
    let small = [bls::Case {
        name: bls::NAMES[1].0,
        lc: cases[1].lc.clone(),
        periods: few,
    }];
    let timings = bls::run(&small, TWO);
    assert_eq!(bls::repeats(18_000, 2_353), 3);
    assert_eq!(bls::repeats(66_253, 166_108), 1);
    assert_eq!(timings.len(), 3);
    let refs = table(&["1 sector Ofir,astropy 7.2.0,0.76"]);
    let out = bls::report(&timings, &refs, TWO);
    assert!(
        out.contains(
            "| | Periods | astropy 7.2.0 | rustellar-bls, 1 thread | rustellar-bls, 2 threads |"
        ),
        "{out}"
    );
    assert!(out.contains("| 1 sector, Ofir | 3 | 0.760 | "), "{out}");
    assert!(out.contains("| Kepler, Ofir | - | - | - | - |"), "{out}");
}

#[test]
fn the_tls_table() {
    let cases = tls::cases();
    let sizes: Vec<(usize, usize)> = cases
        .iter()
        .map(|c| (c.lc.len(), c.periods.as_slice().len()))
        .collect();
    assert_eq!(sizes, [(1240, 2436), (18_000, 2349), (36_000, 5411)]);
    let few = rustellar_core::Periods::from_days(&[3.0, 4.7]).unwrap();
    let small = [tls::Case {
        name: tls::NAMES[0].0,
        lc: cases[0].lc.clone(),
        periods: few,
        repeats: 1,
    }];
    let timings = tls::run(&small, TWO);
    assert!(timings.iter().any(|(c, _)| c == "1 sector 30 min fold"));
    let refs = table(&["1 sector 30 min,transitleastsquares 1.32,8.41"]);
    let out = tls::report(&timings, &refs, TWO);
    assert!(out.contains("Every search on 2 threads."), "{out}");
    assert!(
        out.contains("| 1 sector, 30 min | 1,240 | 2 | 8.41 | "),
        "{out}"
    );
    assert!(
        out.contains("| 2 sectors, 2 min | - | - | - | - | - | - |"),
        "{out}"
    );
    assert!(
        out.contains("`Method::Fold`, transitleastsquares' own way: "),
        "{out}"
    );
}

#[test]
fn the_ls_table() {
    let lc = ls::curve(1000);
    assert_eq!(lc.len(), 1000);
    let days: Vec<f64> = lc.time().iter().map(|t| t.as_days()).collect();
    assert!(days.windows(2).all(|w| w[0] < w[1]) && days[999] < 30.0);
    let timings = ls::run(TWO, &[(100, 50)], &[(100, 64)]);
    assert_eq!(timings.len(), 4);
    assert_eq!(timings[0].0, "exact 100 x 50 1");
    assert_eq!(timings[3].0, "fast 100 x 64 1e-12");
    let refs = table(&[
        "slow 1000 x 10000,astropy 7.2.0,0.54",
        "fast 1000 x 10000,astropy 7.2.0,0.0074",
    ]);
    let out = ls::report(&timings, &refs, TWO);
    assert!(
        out.contains("| Points x frequencies | exact, 1 thread | exact, 2 threads | astropy 7.2.0 slow | astropy 7.2.0 cython |"),
        "{out}"
    );
    assert!(
        out.contains("| 1,000 x 10,000 | - | - | 0.54 s | - |"),
        "{out}"
    );
    assert!(out.contains("| 1,000 x 10,000 | - | - | 7.4 ms |"), "{out}");
    assert!(
        out.contains("| 1,000,000 x 1,000,000 | - | - | - |"),
        "{out}"
    );
}

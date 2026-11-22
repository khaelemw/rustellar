// The speed tables: the reference files read, the times formatted, each
// crate's cases built as its README describes and timed on small inputs.

use std::num::NonZeroUsize;

use crate::speed::{
    Reference, best, bls, detrend, fit, ls, mcmc, milliseconds, model, package, references,
    seconds, short, significant, tls, two_figures, vet,
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
        fit::reference(),
        mcmc::reference(),
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
    assert_eq!(milliseconds(Some(0.38)), "380 ms");
    assert_eq!(milliseconds(Some(1.3e-3)), "1.3 ms");
    assert_eq!(milliseconds(None), "-");
    assert_eq!(bls::thousands(166_108.0), "166,108");
    assert_eq!(bls::thousands(999.0), "999");
    assert_eq!(bls::thousands(1_000_000.0), "1,000,000");
}

#[test]
fn the_best_of_the_repeats() {
    let mut calls = 0;
    let s = best(3, 4, || calls += 1);
    // One call to warm up, then three repeats of four.
    assert_eq!(calls, 13);
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
    // Many curves: 64 Kepler quarters, one after another and at once.
    let many = detrend::many(TWO, &sets[..1]);
    let names: Vec<&str> = many.iter().map(|(c, _)| c.as_str()).collect();
    assert_eq!(names, ["many kepler loop", "many kepler many"]);
    let out = detrend::report(&many, &refs, TWO);
    assert!(out.contains("seconds a light curve on 2 threads:"), "{out}");
    assert!(out.contains("| Kepler quarter | 64 | "), "{out}");
    assert!(out.contains("| TESS, 20 seconds | 8 | - | - |"), "{out}");
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
    // The many-curve table (`tls::many` takes minutes, so its times are
    // given here).
    let many = [
        ("many 30 min loop cpu".to_owned(), 0.75),
        ("many 30 min many gpu".to_owned(), 0.125),
    ];
    let out = tls::report(&many, &refs, TWO);
    assert!(
        out.contains("| 16 curves, 1 sector at 30 min | 0.750 | - | - | 0.125 |"),
        "{out}"
    );
    assert!(
        out.contains("| 16 curves, 1 sector at 2 min | - | - | - | - |"),
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

#[test]
fn the_fit_table() {
    let cases = fit::cases();
    assert_eq!(cases.len(), 6);
    // Each case's set-up as the reference test's: exposure and fixed limb
    // darkening where the case has them.
    let long = cases.iter().find(|c| c.name == "long_cadence").unwrap();
    assert!(long.exposure.is_some() && long.fixed.len() == 2);
    let timings = fit::run(&cases[..1], 1, 0.0);
    assert_eq!(timings.len(), 1);
    let refs = references(fit::reference());
    let out = fit::report(&timings, &refs);
    assert!(out.contains("| batman 2.5.1 lm 2-point |"), "{out}");
    assert!(out.contains("| PyTransit 2.6.18 trf 3-point |"), "{out}");
    assert!(out.contains("| bjd times | - |"), "{out}");
    // A missing time is a dash.
    let out = fit::report(&[], &[]);
    assert!(out.contains("| grazing | - | - | - | - | - | - |"), "{out}");
}

#[test]
fn the_vet_table() {
    let cases = vec![vet::case("short", 6.0, 10.0 / 1440.0)];
    let timings = vet::run(&cases, 1, 1e-6);
    // Two shapes for each step but the secondary eclipse.
    assert_eq!(timings.len(), 2 * vet::STEPS.len() - 1);
    assert!(timings.iter().all(|t| t.3 > 0.0));
    let table = vet::report(&cases, &timings);
    assert!(
        table.starts_with("| Step | short, limb-darkened | short, box |"),
        "{table}"
    );
    assert!(table.contains("| `refine` |"), "{table}");
    // The secondary eclipse has no box column.
    assert!(
        table.contains(
            " | - |
"
        ),
        "{table}"
    );
    assert_eq!(vet::cases().len(), 2);
}

#[test]
fn the_mcmc_table() {
    let timings = mcmc::run(&mcmc::light_curve(), TWO, (16, 400, 100));
    assert_eq!(timings.len(), 2 * mcmc::MOVES.len());
    assert!(timings.iter().all(|t| t.2 > 0.0 && t.3 > 0.0));
    let one = mcmc::run(&mcmc::light_curve(), NonZeroUsize::MIN, (16, 200, 50));
    assert_eq!(one.len(), mcmc::MOVES.len());
    let refs = references(mcmc::reference());
    let table = mcmc::report(&timings, &refs);
    assert!(
        table.contains("| emcee 3.1.6 + batman 2.5.1 | stretch | 552 | 39 ms |"),
        "{table}"
    );
    assert!(
        table.contains("| zeus 2.5.4 + batman 2.5.1 | slice | 128 |"),
        "{table}"
    );
    assert!(
        table.contains("| rustellar-mcmc, 2 threads | de |"),
        "{table}"
    );
    assert!(
        table.contains("| rustellar-mcmc, 1 thread | slice |"),
        "{table}"
    );
    // No reference row for a move without both its time and its tau.
    assert!(
        !table.contains("zeus 2.5.4 + batman 2.5.1 | stretch"),
        "{table}"
    );
}

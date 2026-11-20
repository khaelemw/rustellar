// Injection-recovery: box transits injected into survey light curves (TESS
// 30-minute and 2-minute sectors, thinned and whole, two consecutive sectors,
// nights from the ground, a Kepler-like quarter), searched blind over the
// same Ofir grid by this crate, astropy's fast search and, on a subset,
// transitleastsquares. How often each finds the injected period, and how far
// its period, mid-transit time and depth are from the injected ones.
//
// The grid's periods go through `cbrt`, which is not rounded the same on
// every platform, so a near-tie between two boxes may resolve differently
// elsewhere. Every claim below therefore holds with any one case changed:
// counts have one case of slack, and medians are compared through the values
// one rank either side.

#[expect(dead_code, reason = "the recovery tests use only some of the recipes")]
#[path = "../src/tests/recipe.rs"]
mod recipe;

use std::num::NonZeroUsize;
use std::thread;

use recipe::{Survey, check_samples};
use rustellar_bls::{Bls, Durations, Objective, OfirPeriod, Periods, SearchOptions};
use rustellar_core::{LightCurve, TimeDelta};

// ==========================
// Helper Functions
// ==========================

// A reference set: its families, its cases (with astropy's best box) and
// transitleastsquares' best box on some of them.
struct Set {
    families: &'static str,
    cases: &'static str,
    tls: &'static str,
}

const FIRST: Set = Set {
    families: include_str!("reference/recovery_families.csv"),
    cases: include_str!("reference/recovery_cases.csv"),
    tls: include_str!("reference/recovery_tls.csv"),
};

// A second set, independent of the first (other seeds) and wider: two more
// families (a whole 2-minute sector, a Kepler-like quarter) and snr from 5.
const WIDE: Set = Set {
    families: include_str!("reference/recovery_wide_families.csv"),
    cases: include_str!("reference/recovery_wide_cases.csv"),
    tls: include_str!("reference/recovery_wide_tls.csv"),
};

const DURATIONS: [f64; 6] = [0.04, 0.06, 0.08, 0.11, 0.15, 0.2];

fn rows(csv: &str) -> impl Iterator<Item = Vec<&str>> {
    csv.lines()
        .filter(|line| !line.starts_with('#'))
        .skip(1)
        .map(|line| line.split(',').collect())
}

struct Family {
    name: String,
    segments: Vec<(f64, u32)>,
    cadence: f64,
    noise: f64,
    keep: f64,
}

// A box found by a search: period, mid-transit time, depth.
#[derive(Debug, Clone, Copy)]
struct Found {
    period: f64,
    t0: f64,
    depth: f64,
}

struct Case {
    index: usize,
    family: usize,
    lc: LightCurve,
    period: f64,
    tc: f64,
    depth: f64,
    snr: f64,
    grid: usize,
    astropy: Found,
    tls: Option<Found>,
}

fn families(set: &Set) -> Vec<Family> {
    rows(set.families)
        .map(|v| Family {
            name: v[1].to_string(),
            segments: v[2]
                .split('|')
                .map(|s| {
                    let (start, n) = s.split_once(':').unwrap();
                    (start.parse().unwrap(), n.parse().unwrap())
                })
                .collect(),
            cadence: v[3].parse().unwrap(),
            noise: v[4].parse().unwrap(),
            keep: v[5].parse().unwrap(),
        })
        .collect()
}

fn cases(set: &Set, families: &[Family]) -> Vec<Case> {
    let mut tls: Vec<(usize, Found)> = rows(set.tls)
        .map(|v| {
            let f = |i: usize| v[i].parse::<f64>().unwrap();
            (
                v[0].parse().unwrap(),
                Found {
                    period: f(1),
                    t0: f(2),
                    depth: f(3),
                },
            )
        })
        .collect();
    tls.reverse();
    let cases: Vec<Case> = rows(set.cases)
        .enumerate()
        .map(|(index, v)| {
            let f = |i: usize| v[i].parse::<f64>().unwrap();
            let case: usize = v[0].parse().unwrap();
            assert_eq!(case, index);
            let family: usize = v[1].parse().unwrap();
            let fam = &families[family];
            let lc = Survey {
                seed: v[2].parse().unwrap(),
                segments: fam.segments.clone(),
                cadence: fam.cadence,
                noise: fam.noise,
                keep: fam.keep,
                period: f(3),
                tc: f(4),
                duration: f(5),
                depth: f(6),
            }
            .light_curve();
            check_samples(&lc, v[15]);
            let tls = if tls.last().is_some_and(|t| t.0 == case) {
                tls.pop().map(|t| t.1)
            } else {
                None
            };
            Case {
                index,
                family,
                lc,
                period: f(3),
                tc: f(4),
                depth: f(6),
                snr: f(7),
                grid: v[9].parse().unwrap(),
                astropy: Found {
                    period: f(10),
                    t0: f(11),
                    depth: f(12),
                },
                tls,
            }
        })
        .collect();
    assert!(tls.is_empty(), "every transitleastsquares row has its case");
    cases
}

// The Ofir grid the reference data used: a Sun-like star, oversampling 3, from
// 0.6 d to half the baseline.
fn grid(lc: &LightCurve) -> Periods {
    OfirPeriod {
        minimum_period: Some(TimeDelta::from_days(0.6).unwrap()),
        ..OfirPeriod::default()
    }
    .periods(lc)
    .unwrap()
}

// Our best box over the grid: objective snr, as astropy's; the default
// refinement (the five highest peaks).
fn ours(case: &Case) -> Found {
    let periods = grid(&case.lc);
    assert_eq!(periods.as_slice().len(), case.grid);
    let result = Bls::new(&case.lc)
        .unwrap()
        .search(
            &periods,
            &Durations::from_days(&DURATIONS).unwrap(),
            &SearchOptions::default()
                .with_objective(Objective::Snr)
                .with_threads(NonZeroUsize::MIN),
        )
        .unwrap();
    let best = result.best().expect("a box with a dip");
    Found {
        period: best.transit.period().as_days(),
        t0: best.transit.transit_time().as_btjd(),
        depth: best.fit.depth,
    }
}

// Each case searched on one thread, many cases at a time.
fn search_all(cases: &[Case]) -> Vec<Found> {
    let threads = thread::available_parallelism().map_or(1, NonZeroUsize::get);
    let chunk = cases.len().div_ceil(threads);
    thread::scope(|scope| {
        let handles: Vec<_> = cases
            .chunks(chunk)
            .map(|chunk| scope.spawn(move || chunk.iter().map(ours).collect::<Vec<_>>()))
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().unwrap())
            .collect()
    })
}

// How far a found box is from the injected one, if it found the period
// (within 1%): the period, relative; the mid-transit time at the transit
// nearest the middle of the data, days; the depth, relative.
fn errors(case: &Case, found: Found) -> Option<[f64; 3]> {
    if (found.period / case.period - 1.0).abs() >= 0.01 {
        return None;
    }
    let time = case.lc.time();
    let middle = 0.5 * (time[0].as_btjd() + time[time.len() - 1].as_btjd());
    let near = found.t0 + ((middle - found.t0) / found.period).round() * found.period;
    let r = (near - case.tc) / case.period;
    Some([
        (found.period / case.period - 1.0).abs(),
        (r - r.round()).abs() * case.period,
        (found.depth / case.depth - 1.0).abs(),
    ])
}

// What one search did on some cases: how many it recovered, and each of the
// three errors over those, sorted.
struct Tally {
    recovered: usize,
    errors: [Vec<f64>; 3],
}

impl Tally {
    fn new(cases: &[&Case], found: impl Fn(&Case) -> Found) -> Self {
        let mut errors: [Vec<f64>; 3] = Default::default();
        for case in cases {
            if let Some(e) = self::errors(case, found(case)) {
                for (list, e) in errors.iter_mut().zip(e) {
                    list.push(e);
                }
            }
        }
        for list in &mut errors {
            list.sort_by(f64::total_cmp);
        }
        Tally {
            recovered: errors[0].len(),
            errors,
        }
    }

    fn median(&self, j: usize) -> f64 {
        let x = &self.errors[j];
        0.5 * (x[(x.len() - 1) / 2] + x[x.len() / 2])
    }

    // The values one rank below and above the median: changing, adding or
    // removing any one value keeps the median between them.
    fn below(&self, j: usize) -> f64 {
        self.errors[j][(self.errors[j].len() - 1) / 2 - 1]
    }

    fn above(&self, j: usize) -> f64 {
        self.errors[j][self.errors[j].len() / 2 + 1]
    }

    // Our median errors (0 period, 1 mid-transit time, 2 depth) below the
    // other's, with any one case changed in either.
    fn beats(&self, other: &Tally, errors: &[usize]) -> bool {
        errors.iter().all(|&j| self.above(j) < other.below(j))
    }

    fn show(&self, name: &str, n: usize) -> String {
        format!(
            "{name} {}/{n} P {:.2e} t0 {:.2e} depth {:.3}",
            self.recovered,
            self.median(0),
            self.median(1),
            self.median(2)
        )
    }
}

// One-sided sign test: the chance of `wins` or more of `wins + losses`
// cases going one way if each were a fair coin.
fn sign_test(wins: usize, losses: usize) -> f64 {
    let n = wins + losses;
    let mut term = 0.5_f64.powi(i32::try_from(n).unwrap());
    let mut p = 0.0;
    for k in 0..=n {
        if k >= wins {
            p += term;
        }
        // C(n, k + 1) / C(n, k) = (n - k) / (k + 1).
        term *= (n - k) as f64 / (k + 1) as f64;
    }
    p
}

// The study on one set: its claims asserted, its table printed (see it with
// `--nocapture`). Returns the cases only we recovered and those only astropy
// did.
fn study(set: &Set) -> (usize, usize) {
    let families = families(set);
    let cases = cases(set, &families);
    let found = search_all(&cases);
    let ours = |case: &Case| found[case.index];

    let mut groups: Vec<(String, Vec<&Case>)> = families
        .iter()
        .enumerate()
        .map(|(k, f)| {
            let picked = cases.iter().filter(|c| c.family == k).collect();
            (f.name.clone(), picked)
        })
        .collect();
    groups.push((
        "snr below 20".into(),
        cases.iter().filter(|c| c.snr < 20.0).collect(),
    ));
    groups.push((
        "snr 20 to 100".into(),
        cases.iter().filter(|c| c.snr >= 20.0).collect(),
    ));
    groups.push(("all".into(), cases.iter().collect()));
    for (name, picked) in &groups {
        let o = Tally::new(picked, ours);
        let a = Tally::new(picked, |c| c.astropy);
        println!(
            "{name:24} {} | {}",
            o.show("ours", picked.len()),
            a.show("astropy", picked.len())
        );
        assert!(
            o.beats(&a, &[0, 1, 2]),
            "{name}: median errors below astropy's"
        );
    }

    // Recovery, case by case against astropy.
    let recovered = |c: &Case, f: Found| errors(c, f).is_some();
    let only_ours = cases
        .iter()
        .filter(|c| recovered(c, ours(c)) && !recovered(c, c.astropy))
        .count();
    let only_astropy = cases
        .iter()
        .filter(|c| !recovered(c, ours(c)) && recovered(c, c.astropy))
        .count();
    println!("recovered only by us {only_ours}, only by astropy {only_astropy}");
    // At least as many: the two recover about the same (a sign test over
    // both sets does not show a difference at 1%), so no more is claimed.
    assert!(
        only_ours > only_astropy,
        "at least as many recovered as astropy"
    );

    // transitleastsquares, on its subset.
    let subset: Vec<&Case> = cases.iter().filter(|c| c.tls.is_some()).collect();
    let o = Tally::new(&subset, ours);
    let t = Tally::new(&subset, |c| c.tls.unwrap());
    let n = subset.len();
    println!(
        "{:24} {} | {}",
        "tls subset",
        o.show("ours", n),
        t.show("tls", n)
    );
    // As many recovered (equal on both sets), within the one case of slack;
    // the mid-transit time and depth closer. The period is as close as
    // transitleastsquares' (medians within a few percent), so not claimed.
    assert!(
        o.recovered + 1 >= t.recovered,
        "recovers as many as transitleastsquares"
    );
    assert!(
        o.beats(&t, &[1, 2]),
        "median errors below transitleastsquares'"
    );
    (only_ours, only_astropy)
}

// ==========================
// The study
// ==========================

// A study over thousands of searches, minutes long: run with `--ignored`.
#[test]
#[ignore = "heavy study: cargo test --release -p rustellar-bls -- --ignored"]
fn injection_recovery() {
    let mut only = [0, 0];
    for (name, set) in [("first set", &FIRST), ("wide set", &WIDE)] {
        println!("{name}");
        let (ours, astropy) = study(set);
        only = [only[0] + ours, only[1] + astropy];
    }
    // How likely so many more recovered only by us would be by chance
    // (one-sided sign test): printed, not asserted, as it is not small
    // enough to claim more recoveries than astropy.
    let p = sign_test(only[0], only[1]);
    println!(
        "both sets: recovered only by us {}, only by astropy {}; p = {p:.1e}",
        only[0], only[1]
    );
}

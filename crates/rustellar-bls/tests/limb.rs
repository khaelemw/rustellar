// Injection-recovery with limb-darkened transits (the Mandel & Agol model,
// quadratic limb darkening, a Sun-like star, impact parameters from 0 to 1)
// in the survey families' noise, and as many curves of noise alone. A search
// detects a transit when the signal detection efficiency (SDE) of its power
// spectrum, the peak's height above the mean in standard deviations, is above
// the level that 1% of its own noise-only curves reach: each search at the
// same false-alarm rate. It recovers the transit when it detects it at the
// injected period (within 1%).
//
// Boxes favour any BLS; real transits are rounded, which transitleastsquares
// models (Hippke & Heller 2019). This is the comparison on those.
//
// The same caveat as for the box injections: the grid goes through `cbrt`,
// so every claim holds with any one case changed.

#[expect(
    dead_code,
    reason = "the limb-darkened tests use only some of the recipes"
)]
#[path = "../src/tests/recipe.rs"]
mod recipe;

use std::num::NonZeroUsize;
use std::thread;

use recipe::Survey;
use rustellar_bls::{Bls, Durations, Objective, OfirPeriod, Periods, SearchOptions};
use rustellar_core::{Epoch, LightCurve, TimeDelta};
use rustellar_model::{
    ImpactParameter, LimbDarkening, RadiusRatio, ScaledSemiMajorAxis, TransitParams,
};

const FAMILIES: &str = include_str!("reference/limb_families.csv");
const CASES: &str = include_str!("reference/limb_cases.csv");
const TLS: &str = include_str!("reference/limb_tls.csv");

const DURATIONS: [f64; 6] = [0.04, 0.06, 0.08, 0.11, 0.15, 0.2];
const LIMB_DARKENING: (f64, f64) = (0.4, 0.26);

fn rows(csv: &str) -> impl Iterator<Item = Vec<&str>> {
    csv.lines()
        .filter(|line| !line.starts_with('#'))
        .skip(1)
        .map(|line| line.split(',').collect())
}

struct Family {
    segments: Vec<(f64, u32)>,
    cadence: f64,
    noise: f64,
    keep: f64,
}

// What a search reported: its SDE, and its best box's period and
// mid-transit time.
#[derive(Debug, Clone, Copy)]
struct Found {
    sde: f64,
    period: f64,
    t0: f64,
}

struct Case {
    index: usize,
    family: usize,
    injected: bool,
    lc: LightCurve,
    period: f64,
    tc: f64,
    snr: f64,
    grid: usize,
    astropy: Found,
    tls: Option<Found>,
}

fn families() -> Vec<Family> {
    rows(FAMILIES)
        .map(|v| Family {
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

// The noise of the survey recipe, plus the transit model.
fn light_curve(family: &Family, seed: u64, transit: Option<TransitParams>) -> LightCurve {
    let noise = Survey {
        seed,
        segments: family.segments.clone(),
        cadence: family.cadence,
        noise: family.noise,
        keep: family.keep,
        period: 1.0,
        tc: 0.0,
        duration: 0.0,
        depth: 0.0,
    }
    .light_curve();
    let Some(transit) = transit else {
        return noise;
    };
    let model = transit.light_curve(noise.time());
    let flux = noise
        .flux()
        .iter()
        .zip(&model)
        .map(|(y, m)| y + (m - 1.0))
        .collect();
    LightCurve::from_days(
        noise.time().iter().map(|t| t.as_days()).collect(),
        flux,
        noise.flux_err().to_vec(),
    )
    .unwrap()
}

// The reference data's points: times and errors to the bit; fluxes to within
// batman's error (the reference data's model), which rustellar-model documents as
// 2.1e-8 at worst against batman 2.5.1.
fn check_samples(lc: &LightCurve, samples: &str) {
    for sample in samples.split(';') {
        let v: Vec<f64> = sample.split(':').map(|x| x.parse().unwrap()).collect();
        let i = v[0] as usize;
        assert_eq!(lc.time()[i].as_days().to_bits(), v[1].to_bits(), "time {i}");
        assert_eq!(lc.flux_err()[i].to_bits(), v[3].to_bits(), "error {i}");
        assert!((lc.flux()[i] - v[2]).abs() <= MODEL_TOLERANCE, "flux {i}");
    }
}

const MODEL_TOLERANCE: f64 = 2.1e-8;

fn cases() -> Vec<Case> {
    let families = families();
    let mut tls: Vec<(usize, Found)> = rows(TLS)
        .map(|v| {
            let f = |i: usize| v[i].parse::<f64>().unwrap();
            (
                v[0].parse().unwrap(),
                Found {
                    sde: f(5),
                    period: f(1),
                    t0: f(2),
                },
            )
        })
        .collect();
    tls.reverse();
    let cases: Vec<Case> = rows(CASES)
        .enumerate()
        .map(|(index, v)| {
            let f = |i: usize| v[i].parse::<f64>().unwrap();
            assert_eq!(v[0].parse::<usize>().unwrap(), index);
            let family: usize = v[1].parse().unwrap();
            let injected = v[3] == "1";
            let (period, tc) = (f(4), f(5));
            let transit = injected.then(|| {
                TransitParams::new(
                    Epoch::from_days(tc).unwrap(),
                    TimeDelta::from_days(period).unwrap(),
                    RadiusRatio::new(f(6)).unwrap(),
                    ScaledSemiMajorAxis::new(f(7)).unwrap(),
                    ImpactParameter::new(f(8)).unwrap(),
                    LimbDarkening::quadratic(LIMB_DARKENING.0, LIMB_DARKENING.1).unwrap(),
                )
                .unwrap()
            });
            let lc = light_curve(&families[family], v[2].parse().unwrap(), transit);
            check_samples(&lc, v[15]);
            let tls = if tls.last().is_some_and(|t| t.0 == index) {
                tls.pop().map(|t| t.1)
            } else {
                None
            };
            Case {
                index,
                family,
                injected,
                lc,
                period,
                tc,
                snr: f(9),
                grid: v[10].parse().unwrap(),
                astropy: Found {
                    sde: f(11),
                    period: f(12),
                    t0: f(13),
                },
                tls,
            }
        })
        .collect();
    assert!(tls.is_empty(), "every transitleastsquares row has its case");
    cases
}

// The SDE of a power spectrum: the peak's height above the mean, in standard
// deviations.
fn sde(power: &[f64]) -> f64 {
    let n = power.len() as f64;
    let mean = power.iter().sum::<f64>() / n;
    let variance = power.iter().map(|p| (p - mean) * (p - mean)).sum::<f64>() / n;
    let peak = power.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    (peak - mean) / variance.sqrt()
}

// Our search over the reference data's grid, as in the box injections: objective
// snr, the default refinement.
fn ours(case: &Case) -> Found {
    let periods: Periods = OfirPeriod {
        minimum_period: Some(TimeDelta::from_days(0.6).unwrap()),
        ..OfirPeriod::default()
    }
    .periods(&case.lc)
    .unwrap();
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
        sde: sde(result.power()),
        period: best.transit.period().as_days(),
        t0: best.transit.transit_time().as_days(),
    }
}

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

// The SDE a search must beat to detect: above all but 1% of its noise-only
// curves (the highest 1% are its false alarms).
fn threshold(noise: &[f64]) -> f64 {
    let mut sorted = noise.to_vec();
    sorted.sort_by(|a, b| b.total_cmp(a));
    sorted[sorted.len() / 100]
}

// The period's relative error and the mid-transit time's error (at the
// transit nearest the middle of the data, days), if the search detected the
// transit at its period.
fn errors(case: &Case, found: Found, threshold: f64) -> Option<[f64; 2]> {
    if found.sde <= threshold || (found.period / case.period - 1.0).abs() >= 0.01 {
        return None;
    }
    let time = case.lc.time();
    let middle = 0.5 * (time[0].as_days() + time[time.len() - 1].as_days());
    let near = found.t0 + ((middle - found.t0) / found.period).round() * found.period;
    let r = (near - case.tc) / case.period;
    Some([
        (found.period / case.period - 1.0).abs(),
        (r - r.round()).abs() * case.period,
    ])
}

struct Tally {
    recovered: usize,
    errors: [Vec<f64>; 2],
}

impl Tally {
    fn new(cases: &[&Case], found: impl Fn(&Case) -> Found, threshold: f64) -> Self {
        let mut errors: [Vec<f64>; 2] = Default::default();
        for case in cases {
            if let Some(e) = self::errors(case, found(case), threshold) {
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

    fn below(&self, j: usize) -> f64 {
        self.errors[j][(self.errors[j].len() - 1) / 2 - 1]
    }

    fn above(&self, j: usize) -> f64 {
        self.errors[j][self.errors[j].len() / 2 + 1]
    }

    fn beats(&self, other: &Tally, errors: &[usize]) -> bool {
        errors.iter().all(|&j| self.above(j) < other.below(j))
    }

    fn show(&self, name: &str, n: usize) -> String {
        format!(
            "{name} {}/{n} P {:.2e} t0 {:.2e}",
            self.recovered,
            self.median(0),
            self.median(1)
        )
    }
}

// A study over thousands of searches, minutes long: run with `--ignored`.
#[test]
#[ignore = "heavy study: cargo test --release -p rustellar-bls -- --ignored"]
fn limb_darkened_injection_recovery() {
    let cases = cases();
    let found = search_all(&cases);
    let ours = |case: &Case| found[case.index];
    let injected: Vec<&Case> = cases.iter().filter(|c| c.injected).collect();
    let noise: Vec<&Case> = cases.iter().filter(|c| !c.injected).collect();

    let ours_threshold = threshold(&noise.iter().map(|c| ours(c).sde).collect::<Vec<_>>());
    let astropy_threshold = threshold(&noise.iter().map(|c| c.astropy.sde).collect::<Vec<_>>());
    println!("SDE at 1% false alarms: ours {ours_threshold:.2}, astropy {astropy_threshold:.2}");

    // Each group, and the errors whose medians are claimed below astropy's
    // (0 the period, 1 the mid-transit time). Below an snr of 10 ours are
    // lower too, but not by more than the slack; so is the period in the
    // first family.
    let families = families();
    let mut groups: Vec<(String, Vec<&Case>, &[usize])> = (0..families.len())
        .map(|k| {
            let picked = injected.iter().copied().filter(|c| c.family == k).collect();
            let claims: &[usize] = if k == 0 { &[1] } else { &[0, 1] };
            (format!("family {k}"), picked, claims)
        })
        .collect();
    groups.push((
        "snr below 10".into(),
        injected.iter().copied().filter(|c| c.snr < 10.0).collect(),
        &[],
    ));
    groups.push((
        "snr 10 to 20".into(),
        injected
            .iter()
            .copied()
            .filter(|c| (10.0..20.0).contains(&c.snr))
            .collect(),
        &[0, 1],
    ));
    groups.push((
        "snr 20 to 100".into(),
        injected.iter().copied().filter(|c| c.snr >= 20.0).collect(),
        &[0, 1],
    ));
    groups.push(("all".into(), injected.clone(), &[0, 1]));
    for (name, picked, claims) in &groups {
        let o = Tally::new(picked, ours, ours_threshold);
        let a = Tally::new(picked, |c| c.astropy, astropy_threshold);
        println!(
            "{name:16} {} | {}",
            o.show("ours", picked.len()),
            a.show("astropy", picked.len())
        );
        assert!(o.beats(&a, claims), "{name}: median errors below astropy's");
        if name == "all" {
            // As many detected and recovered at the same false-alarm rate,
            // within the one case of slack. The cases only one of the two
            // recovers sit at either threshold, but for one: at three times
            // the period on the ground-based nights, where the box holds
            // nearly the same points as at the period itself and fits better
            // by one part in a million.
            assert!(
                o.recovered + 1 >= a.recovered,
                "recovers as many as astropy"
            );
        }
    }

    // transitleastsquares, on its subset, each at its own threshold from the
    // subset's noise-only curves. It recovers more of these: it fits a
    // limb-darkened transit, not a box (Hippke & Heller 2019). The period is
    // as close as its; the mid-transit time closer.
    let subset: Vec<&Case> = cases.iter().filter(|c| c.tls.is_some()).collect();
    let noise: Vec<&Case> = subset.iter().copied().filter(|c| !c.injected).collect();
    let tls_threshold = threshold(&noise.iter().map(|c| c.tls.unwrap().sde).collect::<Vec<_>>());
    let ours_subset = threshold(&noise.iter().map(|c| ours(c).sde).collect::<Vec<_>>());
    let injected: Vec<&Case> = subset.iter().copied().filter(|c| c.injected).collect();
    let o = Tally::new(&injected, ours, ours_subset);
    let t = Tally::new(&injected, |c| c.tls.unwrap(), tls_threshold);
    println!(
        "tls subset (SDE {ours_subset:.2} / {tls_threshold:.2}) {} | {}",
        o.show("ours", injected.len()),
        t.show("tls", injected.len())
    );
    assert!(
        o.beats(&t, &[1]),
        "median mid-transit time error below transitleastsquares'"
    );
}

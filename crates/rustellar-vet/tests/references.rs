// Against LEO-Vetter 1.2.0 and exovetter 0.0.16 on the same 620 simulated
// light curves: odd and even depths, the secondary eclipse, ellipsoidal
// variation, the depth's error in red noise and the transits with data.
// The curves are made here exactly as the reference values were (the same
// generator, seeds and arithmetic); the references' values are read from
// the file and this crate's computed alongside them.

use std::collections::BTreeMap;

use rustellar_vet::{
    Candidate, Epoch, LightCurve, Shape, TimeDelta, ellipsoidal, evidence, secondary,
    signal_to_noise,
};

const REFERENCES: &str = include_str!("reference/vet_references.csv");

const N: u32 = 19_440;
const STEP: f64 = 2.0 / 1440.0;
const P: f64 = 3.0;
const T0: f64 = 1.5;
const DUR: f64 = 0.1;
const SIGMA: f64 = 1e-3;
const SCENARIOS: [&str; 5] = [
    "planet",
    "binary_odd_even",
    "binary_secondary",
    "red_noise",
    "gaps",
];

struct Random(u64);

impl Random {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut x = self.0;
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        x ^ (x >> 31)
    }

    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        let (u, v) = (1.0 - self.unit(), self.unit());
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }
}

/// Python's `round`: halves to even.
fn round_even(x: f64) -> f64 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 {
        2.0 * (0.5 * x).round()
    } else {
        r
    }
}

/// The light curve of `scenario`'s draw `draw`, as the reference tool made
/// it.
fn curve(scenario: &str, draw: u64) -> LightCurve {
    curve_with(scenario, draw, 1.0)
}

/// The same with the noise times `noise`.
fn curve_with(scenario: &str, draw: u64, noise: f64) -> LightCurve {
    let index = SCENARIOS.iter().position(|&s| s == scenario).unwrap() as u64;
    let mut rng = Random(1_000_003 * index + draw);
    let phi = (-STEP / 0.05).exp();
    let root = (1.0 - phi * phi).sqrt();
    let mut red = 0.0;
    let (mut time, mut flux) = (Vec::new(), Vec::new());
    for i in 0..N {
        let t = f64::from(i) * STEP;
        let cycles = (t - T0) / P;
        let epoch = round_even(cycles);
        let dt = (cycles - epoch) * P;
        let mut f = 1.0;
        if dt.abs() < 0.5 * DUR {
            if scenario == "binary_odd_even" && epoch.rem_euclid(2.0) != 0.0 {
                f -= 2.7e-3;
            } else {
                f -= 3e-3;
            }
        }
        if scenario == "binary_secondary" {
            let half = (cycles - 0.5 - round_even(cycles - 0.5)) * P;
            if half.abs() < 0.5 * DUR {
                f -= 1e-3;
            }
            f += -2e-4 * (4.0 * std::f64::consts::PI * cycles).cos();
        }
        f += noise * SIGMA * rng.normal();
        if scenario == "red_noise" {
            let n = rng.normal();
            red = if i == 0 { n } else { phi * red + root * n };
            f += noise * SIGMA * red;
        }
        time.push(t);
        flux.push(f);
    }
    if scenario == "gaps" {
        let mut keep = vec![true; time.len()];
        for _ in 0..3 {
            let start = 27.0 * rng.unit();
            let length = 2.0 * rng.unit();
            for (k, &t) in keep.iter_mut().zip(&time) {
                *k &= !(t >= start && t < start + length);
            }
        }
        let kept = |v: &[f64]| -> Vec<f64> {
            v.iter()
                .zip(&keep)
                .filter(|&(_, &k)| k)
                .map(|(&x, _)| x)
                .collect()
        };
        (time, flux) = (kept(&time), kept(&flux));
    }
    let n = time.len();
    LightCurve::from_days(time, flux, vec![SIGMA; n]).unwrap()
}

fn candidate() -> Candidate {
    Candidate::new(
        TimeDelta::from_days(P).unwrap(),
        Epoch::from_days(T0).unwrap(),
        TimeDelta::from_days(DUR).unwrap(),
    )
    .unwrap()
}

/// One draw and its reference values by column.
type Row = (u64, BTreeMap<String, f64>);

/// Each row's reference values by column, grouped by scenario.
fn references() -> BTreeMap<String, Vec<Row>> {
    let mut lines = REFERENCES.lines();
    let header: Vec<&str> = lines.next().unwrap().split(',').collect();
    let mut out: BTreeMap<String, Vec<_>> = BTreeMap::new();
    for line in lines {
        let f: Vec<&str> = line.split(',').collect();
        let values = header[2..]
            .iter()
            .zip(&f[2..])
            .map(|(&k, v)| (k.to_owned(), v.parse().unwrap()))
            .collect();
        out.entry(f[0].to_owned())
            .or_default()
            .push((f[1].parse().unwrap(), values));
    }
    out
}

fn mean(x: &[f64]) -> f64 {
    x.iter().sum::<f64>() / x.len() as f64
}

fn spread(x: &[f64]) -> f64 {
    let m = mean(x);
    (x.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / (x.len() - 1) as f64).sqrt()
}

#[test]
fn against_leo_vetter_and_exovetter() {
    let refs = references();
    let candidate = candidate();
    let column = |scenario: &str, key: &str| -> Vec<f64> {
        refs[scenario].iter().map(|(_, r)| r[key]).collect()
    };

    // Odd and even depths: a planet's significance should be a standard
    // normal, a binary's (3.8 standard errors apart) should stand out.
    let odd_even = |scenario: &str| -> Vec<f64> {
        refs[scenario]
            .iter()
            .map(|&(draw, _)| {
                let lc = curve(scenario, draw);
                evidence(&lc, &candidate, Shape::BOX)
                    .unwrap()
                    .odd_even
                    .unwrap()
                    .sigma()
            })
            .collect()
    };
    let leo_signed = |scenario: &str| -> Vec<f64> {
        refs[scenario]
            .iter()
            .map(|(_, r)| {
                (r["leo_odd"] - r["leo_even"]) / r["leo_odd_err"].hypot(r["leo_even_err"])
            })
            .collect()
    };
    let ours = odd_even("planet");
    let leo = leo_signed("planet");
    let exo = column("planet", "exo_oe_sigma");
    let abs = |x: &[f64]| x.iter().map(|v| v.abs()).collect::<Vec<_>>();
    eprintln!("odd-even, planets: mean |sigma| (0.80) and spread (1)");
    eprintln!("  ours {:.2} {:.2}", mean(&abs(&ours)), spread(&ours));
    eprintln!("  LEO  {:.2} {:.2}", mean(&abs(&leo)), spread(&leo));
    eprintln!("  exo  {:.3} (unsigned)", mean(&abs(&exo)));
    assert!((mean(&abs(&ours)) - 0.80).abs() < 0.1);
    assert!((spread(&ours) - 1.0).abs() < 0.12);
    let ours_eb = odd_even("binary_odd_even");
    let leo_eb = leo_signed("binary_odd_even");
    let exo_eb = column("binary_odd_even", "exo_oe_sigma");
    eprintln!(
        "odd-even, binaries: mean sigma ours {:.2}, LEO {:.2}, exo {:.2}",
        mean(&ours_eb),
        mean(&leo_eb),
        mean(&exo_eb)
    );
    assert!(mean(&ours_eb) < -3.0);

    // The secondary eclipse and the ellipsoidal variation of a binary.
    let mut found = 0;
    let (mut sig, mut depth, mut twice) = (Vec::new(), Vec::new(), Vec::new());
    for &(draw, _) in &refs["binary_secondary"] {
        let lc = curve("binary_secondary", draw);
        let s = secondary(&lc, &candidate);
        let deepest = s.deepest.unwrap();
        if (deepest.phase - 0.5).abs() < 0.5 * DUR / P {
            found += 1;
        }
        sig.push(deepest.sigma());
        depth.push(s.at_half.unwrap().depth);
        twice.push(ellipsoidal(&lc, &candidate).unwrap().twice.amplitude());
    }
    let near_half = |key: &str| {
        column("binary_secondary", key)
            .iter()
            .filter(|&&p| (p - 0.5).abs() < 0.5 * DUR / P)
            .count()
    };
    let draws = refs["binary_secondary"].len();
    eprintln!("secondary (1e-3 at half the period): found at half, mean sigma, depth");
    eprintln!(
        "  ours {found}/{draws} {:.1} {:.2e} ± {:.1e}",
        mean(&sig),
        mean(&depth),
        spread(&depth)
    );
    let leo_depth = column("binary_secondary", "leo_dep_sec");
    eprintln!(
        "  LEO  {}/{draws} {:.1} {:.2e} ± {:.1e}",
        near_half("leo_phs_sec"),
        mean(&column("binary_secondary", "leo_sig_sec")),
        mean(&leo_depth),
        spread(&leo_depth)
    );
    eprintln!(
        "  exo  {}/{draws} {:.1}",
        near_half("exo_ms_phase_sec"),
        mean(&column("binary_secondary", "exo_ms_sigma_sec"))
    );
    assert_eq!(found, draws);
    // The dip against the mean flux out of eclipse, ellipsoidal variation
    // included, as measured without noise.
    let clean = secondary(&curve_with("binary_secondary", 0, 0.0), &candidate);
    let truth = clean.at_half.unwrap().depth;
    eprintln!("  (without noise {truth:.3e})");
    assert!((mean(&depth) - truth).abs() < 3.0 * spread(&depth) / (draws as f64).sqrt());
    let leo_amp = column("binary_secondary", "leo_sine_amp");
    let exo_amp = column("binary_secondary", "exo_sweet_half_amp");
    eprintln!("ellipsoidal amplitude (2e-4): mean ± spread");
    eprintln!("  ours {:.3e} ± {:.1e}", mean(&twice), spread(&twice));
    eprintln!("  LEO  {:.3e} ± {:.1e}", mean(&leo_amp), spread(&leo_amp));
    eprintln!("  exo  {:.3e} ± {:.1e}", mean(&exo_amp), spread(&exo_amp));
    assert!((mean(&twice) - 2e-4).abs() < 3.0 * spread(&twice) / (draws as f64).sqrt());

    // A planet's ellipsoidal significance: chi with two degrees of freedom,
    // above 3 in 1.1% of light curves.
    let mut quiet = Vec::new();
    for &(draw, _) in &refs["planet"] {
        let lc = curve("planet", draw);
        quiet.push(ellipsoidal(&lc, &candidate).unwrap().twice.sigma());
    }
    let above = |x: &[f64]| x.iter().filter(|&&v| v > 3.0).count();
    eprintln!(
        "planets above 3 sigma in ellipsoidal variation: ours {}, LEO {}, exo {} of {}",
        above(&quiet),
        above(&column("planet", "leo_sine_sig")),
        above(&column("planet", "exo_sweet_half_snr")),
        quiet.len()
    );
    assert!((mean(&quiet) - (std::f64::consts::PI / 2.0).sqrt()).abs() < 0.15);

    // The depth's stated error against its scatter, white and red noise.
    for scenario in ["planet", "red_noise"] {
        let (mut depths, mut errors) = (Vec::new(), Vec::new());
        for &(draw, _) in &refs[scenario] {
            let lc = curve(scenario, draw);
            let snr = signal_to_noise(&lc, &candidate, Shape::BOX).unwrap();
            depths.push(snr.depth);
            errors.push(snr.depth / snr.red);
        }
        let ratio = spread(&depths) / mean(&errors);
        let leo = spread(&column(scenario, "leo_dep")) / mean(&column(scenario, "leo_err"));
        eprintln!("{scenario}: depth scatter over stated error (1): ours {ratio:.2}, LEO {leo:.2}");
        assert!((ratio - 1.0).abs() < 0.15, "{scenario}");
    }

    // Transits with data, in light curves with gaps.
    for &(draw, ref r) in &refs["gaps"] {
        let lc = curve("gaps", draw);
        let e = evidence(&lc, &candidate, Shape::BOX).unwrap();
        assert_eq!(e.with_data() as f64, r["leo_n_transit"], "draw {draw}");
    }
}

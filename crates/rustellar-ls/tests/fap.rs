//! False-alarm probabilities and levels against the same formulas at 40
//! digits (mpmath; Baluev 2008), on the periodogram reference's six cases
//! and 100,000 even points, with every mean model's degrees of freedom,
//! normalisation and method, for probabilities from 0.5 to 1e-40;
//! astropy 7.2's beside (with a floating mean; it refuses data not
//! centred).

use std::collections::BTreeMap;

use rustellar_ls::{FalseAlarm, LightCurve, LombScargle, LsError, Mean, Normalization};

const POINTS: &str = include_str!("reference/ls_points.csv");
const FAP: &str = include_str!("reference/ls_fap.csv");

fn curves() -> BTreeMap<String, LightCurve> {
    let mut columns: BTreeMap<String, [Vec<f64>; 3]> = BTreeMap::new();
    for line in POINTS.lines().skip(1) {
        let mut f = line.split(',');
        let case = f.next().unwrap().to_string();
        let entry = columns.entry(case).or_default();
        for column in entry.iter_mut() {
            column.push(f.next().unwrap().parse().unwrap());
        }
    }
    let mut out: BTreeMap<String, LightCurve> = columns
        .into_iter()
        .map(|(case, [t, y, dy])| (case, LightCurve::from_days(t, y, dy).unwrap()))
        .collect();
    // 100,000 even times 0.01 days apart, unit errors (the flux does not
    // enter).
    let time = (0..100_000).map(|i| 0.01 * f64::from(i)).collect();
    let flux = (0..100_000).map(|i| f64::from(i % 2)).collect();
    out.insert(
        "large".into(),
        LightCurve::from_days(time, flux, vec![1.0; 100_000]).unwrap(),
    );
    out
}

fn normalization(name: &str) -> Normalization {
    match name {
        "standard" => Normalization::Standard,
        "model" => Normalization::Model,
        "log" => Normalization::Log,
        _ => Normalization::Psd,
    }
}

fn method(name: &str) -> FalseAlarm {
    match name {
        "baluev" => FalseAlarm::Baluev,
        "davies" => FalseAlarm::Davies,
        "naive" => FalseAlarm::Naive,
        _ => FalseAlarm::Single,
    }
}

fn relative(a: f64, b: f64) -> f64 {
    if a == b { 0.0 } else { (a - b).abs() / b.abs() }
}

#[test]
fn probabilities_and_levels_against_forty_digits() {
    let curves = curves();
    let mut setups: BTreeMap<(String, String), LombScargle> = BTreeMap::new();
    // Worst relative errors by method: ours (probability, level), astropy's.
    let mut worst: BTreeMap<String, [f64; 4]> = BTreeMap::new();
    let mut astropy_failed = 0;
    for line in FAP.lines().skip(1) {
        let f: Vec<&str> = line.split(',').collect();
        let num = |k: usize| f[k].parse::<f64>().unwrap();
        let (case, mean) = (f[0].to_string(), f[1].to_string());
        let ls = setups
            .entry((case.clone(), mean.clone()))
            .or_insert_with(|| {
                let m = if mean == "floating" {
                    Mean::Floating
                } else {
                    Mean::Zero
                };
                LombScargle::new(&curves[&case], m).unwrap()
            });
        let (n, m, fmax, p) = (normalization(f[2]), method(f[3]), num(4), num(5));
        let (z_exact, z, fap_exact) = (num(6), num(7), num(8));
        let fap = ls.false_alarm_probability(z, n, fmax, m).unwrap();
        let level = ls.false_alarm_level(p, n, fmax, m).unwrap();
        let errors = [relative(fap, fap_exact), relative(level, z_exact)];
        let w = worst.entry(f[3].to_string()).or_default();
        w[0] = w[0].max(errors[0]);
        w[1] = w[1].max(errors[1]);
        assert!(
            errors[0] <= 1e-12 && errors[1] <= 1e-13,
            "{line}: {fap:e} {level:e} {errors:?}"
        );
        if mean == "floating" {
            let (af, al) = (num(9), num(10));
            if af.is_nan() || al.is_nan() || relative(al, z_exact) > 1e-3 {
                astropy_failed += 1;
            } else {
                w[2] = w[2].max(relative(af, fap_exact));
                w[3] = w[3].max(relative(al, z_exact));
            }
        }
    }
    println!("relative error (ours: probability, level; astropy: probability, level):");
    for (method, w) in &worst {
        println!(
            "  {method:>7}: {:.0e} {:.0e}   {:.0e} {:.0e}",
            w[0], w[1], w[2], w[3]
        );
    }
    println!("astropy levels failed or off by more than 1e-3: {astropy_failed}");
}

#[test]
fn bad_inputs() {
    let lc = &curves()["few"];
    let ls = LombScargle::new(lc, Mean::Floating).unwrap();
    let (s, b) = (Normalization::Standard, FalseAlarm::Baluev);
    assert_eq!(
        ls.false_alarm_probability(f64::NAN, s, 1.0, b).unwrap_err(),
        LsError::NonFinite { name: "power" }
    );
    for p in [0.0, 1.0, f64::NAN, -0.5] {
        assert_eq!(
            ls.false_alarm_level(p, s, 1.0, b).unwrap_err(),
            LsError::NotProbability
        );
    }
    assert_eq!(
        LsError::NotProbability.to_string(),
        "the probability must be above 0 and below 1"
    );
    assert_eq!(
        ls.false_alarm_probability(0.5, s, f64::INFINITY, b)
            .unwrap_err(),
        LsError::NonFinite {
            name: "maximum_frequency"
        }
    );
    assert_eq!(
        ls.false_alarm_level(0.5, s, 0.0, b).unwrap_err(),
        LsError::NotPositive {
            name: "maximum_frequency"
        }
    );
    // Three points leave no degree of freedom with a mean and a sinusoid.
    let three =
        LightCurve::from_days(vec![0.0, 1.0, 2.5], vec![1.0, 2.0, 1.5], vec![0.1; 3]).unwrap();
    let error = LombScargle::new(&three, Mean::Floating)
        .unwrap()
        .false_alarm_probability(0.5, s, 1.0, b)
        .unwrap_err();
    assert_eq!(error, LsError::TooFewPoints { needed: 4 });
    assert_eq!(error.to_string(), "at least 4 points are needed");
    // Without a mean, three are enough.
    assert!(
        LombScargle::new(&three, Mean::Zero)
            .unwrap()
            .false_alarm_probability(0.5, s, 1.0, b)
            .is_ok()
    );
}

#[test]
fn powers_at_the_ends() {
    let ls = LombScargle::new(&curves()["uneven"], Mean::Floating).unwrap();
    for n in [
        Normalization::Standard,
        Normalization::Model,
        Normalization::Log,
        Normalization::Psd,
    ] {
        for m in [
            FalseAlarm::Baluev,
            FalseAlarm::Davies,
            FalseAlarm::Naive,
            FalseAlarm::Single,
        ] {
            assert_eq!(ls.false_alarm_probability(0.0, n, 3.0, m).unwrap(), 1.0);
            assert_eq!(ls.false_alarm_probability(-1.0, n, 3.0, m).unwrap(), 1.0);
        }
    }
    // A standard power of 1 or more fits the noise exactly: never by chance.
    let s = Normalization::Standard;
    assert_eq!(
        ls.false_alarm_probability(1.0, s, 3.0, FalseAlarm::Baluev)
            .unwrap(),
        0.0
    );
    // The Davies bound is at most 1.
    assert_eq!(
        ls.false_alarm_probability(1e-3, s, 3.0, FalseAlarm::Davies)
            .unwrap(),
        1.0
    );
}

/// A seeded generator: xorshift64* and the Box-Muller transform.
struct Normal(u64);

impl Normal {
    fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let bits = self.0.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 11;
        #[expect(clippy::cast_precision_loss, reason = "53 bits")]
        let u = (bits as f64 + 0.5) / (1_u64 << 53) as f64;
        u
    }

    fn next(&mut self) -> f64 {
        let (u, v) = (self.uniform(), self.uniform());
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }
}

#[test]
fn levels_against_simulated_noise() {
    // 60 uneven points over 20 days, unit errors; frequencies to 3 per
    // day at a fifth of 1 / T.
    let mut rng = Normal(0x5eed_1234_abcd_ef01);
    let mut time: Vec<f64> = (0..60).map(|_| 20.0 * rng.uniform()).collect();
    time.sort_by(f64::total_cmp);
    let grid = rustellar_ls::Frequencies::uniform(0.005, 0.01, 300).unwrap();
    let n = Normalization::Standard;
    let runs = 4000;
    let mut highest = Vec::with_capacity(runs);
    let mut first = None;
    for _ in 0..runs {
        let flux = (0..60).map(|_| rng.next()).collect();
        let lc = LightCurve::from_days(time.clone(), flux, vec![1.0; 60]).unwrap();
        let ls = LombScargle::new(&lc, Mean::Floating).unwrap();
        highest.push(ls.power(&grid, n).best().1);
        first.get_or_insert(ls);
    }
    let ls = first.unwrap();
    #[expect(clippy::cast_precision_loss, reason = "a few thousand")]
    let runs = runs as f64;
    for (p, m) in [
        (0.5, FalseAlarm::Baluev),
        (0.1, FalseAlarm::Baluev),
        (0.01, FalseAlarm::Baluev),
        (0.1, FalseAlarm::Davies),
        (0.1, FalseAlarm::Naive),
    ] {
        let level = ls.false_alarm_level(p, n, 3.0, m).unwrap();
        #[expect(clippy::cast_precision_loss, reason = "a few thousand")]
        let seen = highest.iter().filter(|&&h| h >= level).count() as f64 / runs;
        let sigma = (p * (1.0 - p) / runs).sqrt();
        println!("{m:?} p {p}: seen {seen:.4} (sigma {sigma:.4})");
        // An upper bound, close for small probabilities.
        if m != FalseAlarm::Naive {
            assert!(
                seen <= p + 3.0 * sigma && seen >= 0.5 * p,
                "{m:?} {p}: {seen}"
            );
        }
    }
}

// The alias table: with the true period a third, half, two thirds, one,
// three halves, two or three times the candidate's, the best row is the
// true one and the split shows why; and the ratios, the table's text and
// what cannot be measured.

use rustellar_model::{ScaledSemiMajorAxis, TransitParams};
use rustellar_vet::{
    AliasRow, Candidate, Epoch, ImpactParameter, LightCurve, LimbDarkening, RadiusRatio, Ratio,
    Shape, TimeDelta, VetError, aliases, aliases_at,
};

fn days(d: f64) -> TimeDelta {
    TimeDelta::from_days(d).unwrap()
}

fn at(t: f64) -> Epoch {
    Epoch::from_btjd(t).unwrap()
}

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

// A planet of period `period` (k 0.05, b 0.5, a/R* 15, first transit at
// 1501.3) in 54 days every 10 minutes with white noise of 3e-4.
fn light_curve(period: f64, seed: u64) -> (LightCurve, TimeDelta) {
    let p = TransitParams::new(
        at(1501.3),
        days(period),
        RadiusRatio::new(0.05).unwrap(),
        ScaledSemiMajorAxis::new(15.0).unwrap(),
        ImpactParameter::new(0.5).unwrap(),
        LimbDarkening::quadratic(0.4804, 0.1867).unwrap(),
    )
    .unwrap();
    let mut r = Random(seed);
    let time: Vec<f64> = (0..7776).map(|i| 1500.0 + f64::from(i) / 144.0).collect();
    let flux = time
        .iter()
        .map(|&t| p.flux_at(at(t)) + 3e-4 * r.normal())
        .collect();
    let lc = LightCurve::from_btjd(time, flux, vec![3e-4; 7776]).unwrap();
    (lc, p.t14().unwrap())
}

fn row(rows: &[AliasRow], p: u16, q: u16) -> &AliasRow {
    let r = Ratio::new(p, q).unwrap();
    rows.iter().find(|row| row.ratio == r).unwrap()
}

// For each true ratio, a planet at 4.2 days times it found at 4.2 days:
// the best row is the true ratio, every other fits worse by far, and the
// true row's split has the depth at its own transits and none at the
// candidate's alone.
#[test]
fn the_true_period_is_the_best_row() {
    for (seed, &truth) in Ratio::DEFAULT.iter().enumerate() {
        let (lc, duration) = light_curve(4.2 * truth.value(), 10 + seed as u64);
        let candidate = Candidate::new(days(4.2), at(1501.3), duration).unwrap();
        let table = aliases(&lc, &candidate, Shape::default()).unwrap();
        let best = table.best().unwrap();
        assert_eq!(best.ratio, truth, "{table}");
        let full = best.alias.as_ref().unwrap().fit.depth;
        for r in &table.rows {
            let Ok(a) = &r.alias else { continue };
            if r.ratio != truth {
                assert!(
                    a.delta_chi2 - best.alias.as_ref().unwrap().delta_chi2 > 50.0,
                    "{truth} {}\n{table}",
                    r.ratio
                );
            }
        }
        let split = best.alias.as_ref().unwrap().split.unwrap();
        if truth.numerator() > 1 {
            // Some of the candidate's transits are not the planet's.
            let c = split.candidate_only.unwrap();
            assert!(c.sigma().abs() < 4.0, "{truth} {c:?}");
        }
        if truth.denominator() > 1 {
            // The planet has transits the candidate does not predict.
            let a = split.alias_only.unwrap();
            assert!(
                (a.depth / full - 1.0).abs() < 5.0 * a.depth_err / full,
                "{truth} {a:?}"
            );
        }
        let both = split.both.unwrap();
        assert!((both.depth / full - 1.0).abs() < 5.0 * both.depth_err / full);
        // In the candidate's own row the split is the candidate's alone.
        let own = row(&table.rows, 1, 1).alias.as_ref().unwrap();
        let own_split = own.split.unwrap();
        assert!(own_split.candidate_only.is_none() && own_split.alias_only.is_none());
        assert_eq!(own.delta_chi2, 0.0);
        assert_eq!(own.fit, table.fit);
    }
}

// Two thirds: the alias's transits every 2 steps of P / 3, the candidate's
// every 3, so steps 1 and 5 of each 6 are neither's, with no depth there.
#[test]
fn the_times_neither_predicts() {
    let (lc, duration) = light_curve(4.2 * 2.0 / 3.0, 3);
    let candidate = Candidate::new(days(4.2), at(1501.3), duration).unwrap();
    let table = aliases_at(
        &lc,
        &candidate,
        Shape::default(),
        &[Ratio::new(2, 3).unwrap()],
    )
    .unwrap();
    let split = table.rows[0].alias.as_ref().unwrap().split.unwrap();
    let neither = split.neither.unwrap();
    assert!(neither.sigma().abs() < 4.0, "{neither:?}");
    assert!(neither.points > 0);
}

#[test]
fn ratios() {
    let r = Ratio::new(6, 4).unwrap();
    assert_eq!((r.numerator(), r.denominator(), r.value()), (3, 2, 1.5));
    assert_eq!(format!("[{r:>5}]"), "[  3/2]");
    assert_eq!(Ratio::new(3, 3).unwrap().to_string(), "1");
    assert_eq!(Ratio::new(0, 1), Err(VetError::RatioZero));
    assert_eq!(Ratio::new(1, 0), Err(VetError::RatioZero));
    let names: Vec<String> = Ratio::DEFAULT.iter().map(ToString::to_string).collect();
    assert_eq!(names, ["1/3", "1/2", "2/3", "1", "3/2", "2", "3"]);
}

#[test]
fn the_table_as_text() {
    let (lc, _) = light_curve(4.2, 1);
    // A box 1.5 days long: longer than a third of the period, so no row
    // there, and no split at two thirds (whose finer spacing is P / 3).
    let long = Candidate::new(days(4.2), at(1501.3), days(1.5)).unwrap();
    let table = aliases(&lc, &long, Shape::BOX).unwrap();
    assert_eq!(
        row(&table.rows, 1, 3).alias,
        Err(VetError::DurationNotShorterThanPeriod)
    );
    assert_eq!(row(&table.rows, 2, 3).alias.as_ref().unwrap().split, None);
    let text = table.to_string();
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 8, "{text}");
    assert!(lines[0].contains("ratio") && lines[0].contains("delta chi2"));
    assert!(lines[1].contains("1/3") && lines[1].contains("shorter than the period"));
    assert!(lines[4].trim_start().starts_with("1 "), "{text}");
    // A dash where there is no split depth.
    assert!(lines[3].contains(" - "), "{text}");
    // Twenty times the period, 84 days: one transit in 54 days, so no odd
    // and even to compare.
    let lone = aliases_at(&lc, &long, Shape::BOX, &[Ratio::new(20, 1).unwrap()]).unwrap();
    let text = lone.to_string();
    let fields: Vec<&str> = text.lines().nth(1).unwrap().split_whitespace().collect();
    assert_eq!(fields[7..9], ["1/1", "-"], "{text}");
    // At a third of the period a limb-darkened shape 0.75 days long is
    // impossible.
    let candidate = Candidate::new(days(4.2), at(1501.3), days(0.75)).unwrap();
    let table = aliases(&lc, &candidate, Shape::default()).unwrap();
    assert_eq!(
        row(&table.rows, 1, 3).alias,
        Err(VetError::DurationTooLongForShape)
    );
}

#[test]
fn what_cannot_be_measured() {
    let lc = LightCurve::from_btjd(vec![0.0, 1.0], vec![1.0, 1.0], vec![1e-3; 2]).unwrap();
    let late = Candidate::new(days(10.0), at(5.0), days(0.1)).unwrap();
    assert_eq!(
        aliases(&lc, &late, Shape::BOX),
        Err(VetError::NoPointsInTransit)
    );
    // An alias period beyond the largest number.
    let (lc, _) = light_curve(4.2, 1);
    let huge = Candidate::new(days(1e305), at(1501.3), days(0.1)).unwrap();
    let table = aliases_at(&lc, &huge, Shape::BOX, &[Ratio::new(60000, 1).unwrap()]).unwrap();
    assert_eq!(table.rows[0].alias, Err(VetError::AliasPeriodNotFinite));
    assert!(table.best().is_none());
    // A third of a period shorter than the duration.
    let short = Candidate::new(days(0.25), at(1501.3), days(0.1)).unwrap();
    let table = aliases_at(&lc, &short, Shape::BOX, &[Ratio::new(1, 3).unwrap()]).unwrap();
    assert_eq!(
        table.rows[0].alias,
        Err(VetError::DurationNotShorterThanPeriod)
    );
}

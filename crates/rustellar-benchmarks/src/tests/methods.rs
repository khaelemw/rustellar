// The methods study on a few curves: every method run and searched, the
// table's rows (with wotan's recorded ones for the same seeds), and the
// export's rows.

use std::num::NonZeroUsize;

use crate::methods::{TRIALS, export, measure_all, report, search_flattened};
use crate::window::cases;

#[test]
fn every_method_on_a_few_curves() {
    let drawn = cases(3);
    let results = measure_all(&drawn, NonZeroUsize::new(2).unwrap());
    assert_eq!(results.len(), 3);
    for outcomes in &results {
        assert_eq!(outcomes.len(), TRIALS.len());
        for o in outcomes {
            assert!(o.kept.snr.is_finite() && o.kept.depth.is_finite());
            assert!(o.found.sde.is_finite());
        }
    }
    let table = report(&drawn, &results);
    let rows = table.lines().filter(|l| l.starts_with("| ")).count();
    // The header, ours, and wotan's seven.
    assert_eq!(rows, 1 + TRIALS.len() + 7, "{table}");
    assert!(table.contains("| wotan hspline | 4 x T14 |"), "{table}");
}

#[test]
fn the_export() {
    let text = export(&cases(1));
    let mut lines = text.lines();
    assert_eq!(lines.next(), Some("seed,t14,time,flux,shape"));
    let rows: Vec<Vec<f64>> = lines
        .map(|l| l.split(',').map(|v| v.parse().unwrap()).collect())
        .collect();
    assert_eq!(rows.len(), 3802);
    assert!(rows.iter().all(|r| r[0] == 1.0 && r[1] > 0.0));
    assert!(rows.iter().any(|r| r[4] > 0.0));
}

// A recorded curve searched as the crate's own: the export's rows of seed
// 1 as if flattened by a method (raw, the star left in), twice under two
// names.
#[test]
fn searching_recorded_curves() {
    let text = export(&cases(1));
    let mut lines = vec!["seed,method,factor,time,flux".to_owned()];
    for name in ["a", "b"] {
        for l in text.lines().skip(1) {
            let r: Vec<&str> = l.split(',').collect();
            lines.push(format!("{},{name},2.5,{},{}", r[0], r[2], r[3]));
        }
    }
    let out = search_flattened(lines.into_iter(), NonZeroUsize::new(2).unwrap());
    let rows: Vec<&str> = out.lines().collect();
    assert_eq!(rows[0], "seed,method,factor,right,sde");
    assert_eq!(rows.len(), 3);
    assert!(rows[1].starts_with("1,a,2.5,") && rows[2].starts_with("1,b,2.5,"));
}

// The window study on a few curves: the cases as drawn, a masked window
// keeping the depth, and both tables' shapes.

use std::num::NonZeroUsize;

use crate::stats::median;
use crate::window::{WINDOWS, cases, measure_all, report, search_all, search_report};

#[test]
fn the_cases_are_transits_on_spotted_stars() {
    let drawn = cases(20);
    assert_eq!(drawn.len(), 20);
    for case in &drawn {
        let period = case.transit.period().as_days();
        let t14 = case.transit.t14().unwrap().as_hours();
        assert!((1.0..9.0).contains(&period), "period {period}");
        // The duration drawn, to rounding.
        assert!((1.0 - 1e-9..7.0 + 1e-9).contains(&t14), "duration {t14}");
        for (p, a, _) in case.waves {
            assert!((0.5..10.0).contains(&p) && (1e-4..5e-3).contains(&a));
        }
    }
    // The same every run.
    assert_eq!(cases(3)[2].transit, drawn[2].transit);
}

#[test]
fn masked_windows_keep_the_depth() {
    let drawn = cases(4);
    let results = measure_all(&drawn, NonZeroUsize::new(2).unwrap());
    assert_eq!(results.len(), 4);
    let mut masked = vec![];
    for kept in &results {
        assert_eq!(kept.len(), WINDOWS.len());
        for (w, k) in WINDOWS.iter().zip(kept) {
            if w.masked {
                masked.push(k.depth);
            }
            assert!(k.ideal > 0.0 && k.snr.is_finite());
        }
    }
    // Each depth is noisy (a shallow transit), their median is not.
    let depth = median(&masked);
    assert!((depth - 1.0).abs() < 0.1, "median depth kept {depth}");
    let table = report(&drawn, &results);
    assert!(table.starts_with("4 curves, "));
    assert_eq!(
        table.lines().filter(|l| l.starts_with("| ")).count(),
        WINDOWS.len() + 1
    );
}

#[test]
fn the_blind_search_table() {
    let drawn = cases(2);
    let found = search_all(&drawn, NonZeroUsize::new(2).unwrap());
    let blind = WINDOWS.iter().filter(|w| !w.masked).count();
    assert!(found.iter().all(|f| f.len() == blind));
    assert!(found.iter().flatten().all(|f| f.sde.is_finite()));
    let table = search_report(&drawn, &found);
    assert_eq!(table.lines().count(), blind + 2);
}

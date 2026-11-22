// The comparison on a few curves: ours found where TLS finds it (the
// third too faint for either, both on the same false peak), and the table
// says so.

use rustellar_tls::{Device, SearchOptions};

use crate::Kind;
use crate::tls::{figures, report, run, run_around};

#[test]
fn a_few_curves() {
    let options = SearchOptions::default()
        .with_bootstrap(0)
        .with_device(Device::Cpu);
    let curves = run(Kind::Many, 3, &options);
    assert_eq!(curves.len(), 3);
    for c in &curves {
        let ours = c.ours.unwrap();
        assert!((ours.period / c.row.tls.period - 1.0).abs() < 1e-3, "{c:?}");
        assert!(c.depth > 0.0 && c.duration > 0.0 && c.span > 26.9);
    }
    let table = report(Kind::Many.name(), &curves);
    assert!(
        table.contains("rustellar-tls 2, TLS 1.32 2, both 2"),
        "{table}"
    );
    assert!(table.contains("| period, relative |"), "{table}");
    // No curve both find: the counts only.
    let none = report(Kind::Many.name(), &[]);
    assert!(
        none.ends_with(
            "both 0.

"
        ),
        "{none}"
    );
}

// Detrended again around the candidate: the same rows, each searched on
// its new flattening, which keeps more of the transit than the first.
#[test]
fn around_the_candidate() {
    let options = SearchOptions::default()
        .with_bootstrap(0)
        .with_device(Device::Cpu);
    let blind = run(Kind::Variable, 3, &options);
    let around = run_around(3, &options);
    assert_eq!(around.len(), 3);
    for (a, b) in around.iter().zip(&blind) {
        assert_eq!(a.row, b.row);
        assert_eq!((a.depth, a.duration), (b.depth, b.duration));
    }
    let depth = |c: &[crate::tls::Curve]| {
        c.iter()
            .filter_map(|c| c.ours)
            .map(|f| f.depth)
            .sum::<f64>()
    };
    assert!(depth(&around) > depth(&blind), "{around:?} {blind:?}");
}

#[test]
fn three_significant_figures() {
    assert_eq!(figures(3.05e-4), "3.05e-4");
    assert_eq!(figures(0.06547), "0.0655");
    assert_eq!(figures(5.303), "5.30");
    assert_eq!(figures(1234.5), "1234");
}

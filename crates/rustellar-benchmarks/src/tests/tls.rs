// The comparison on a few curves: ours found where TLS finds it (the
// third too faint for either, both on the same false peak), and the table
// says so.

use rustellar_tls::{Device, SearchOptions};

use crate::Kind;
use crate::tls::{figures, report, run};

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
    let table = report(Kind::Many, &curves);
    assert!(
        table.contains("rustellar-tls 2, TLS 1.32 2, both 2"),
        "{table}"
    );
    assert!(table.contains("| period, relative |"), "{table}");
    // No curve both find: the counts only.
    let none = report(Kind::Many, &[]);
    assert!(
        none.ends_with(
            "both 0.

"
        ),
        "{none}"
    );
}

#[test]
fn three_significant_figures() {
    assert_eq!(figures(3.05e-4), "3.05e-4");
    assert_eq!(figures(0.06547), "0.0655");
    assert_eq!(figures(5.303), "5.30");
    assert_eq!(figures(1234.5), "1234");
}

// The compare binary: a table for each set asked for, and the usage for
// anything else.

use std::process::Command;

fn compare(args: &[&str]) -> (bool, String, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_compare"))
        .args(args)
        .output()
        .unwrap();
    (
        out.status.success(),
        String::from_utf8(out.stdout).unwrap(),
        String::from_utf8(out.stderr).unwrap(),
    )
}

#[test]
fn a_table_for_each_set() {
    let (ok, table, _) = compare(&["many", "short", "--limit", "1", "--threads", "2", "--cpu"]);
    assert!(ok);
    assert!(table.contains("### many (1 curves)"), "{table}");
    assert!(table.contains("### short (1 curves)"), "{table}");
}

#[test]
fn the_limb_darkened_study() {
    let (ok, table, _) = compare(&["limb", "--step", "250", "--threads", "4", "--cpu"]);
    assert!(ok);
    assert!(
        table.contains("rustellar-bls against rustellar-tls"),
        "{table}"
    );
    assert!(table.contains("(8 curves,"), "{table}");
    assert!(table.contains("| transitleastsquares 1.32 |"), "{table}");
}

#[test]
fn every_set_by_default() {
    let (ok, table, _) = compare(&["--limit", "0", "--step", "250", "--cpu"]);
    assert!(ok);
    for set in ["many", "mixed", "varying", "short", "sectors"] {
        assert!(table.contains(&format!("### {set} (0 curves)")), "{table}");
    }
    assert!(table.contains("(8 curves,"), "{table}");
    // The window study capped by --limit too, here to none, and the alias
    // study.
    assert!(table.contains("(0 curves,"), "{table}");
    assert!(table.contains("(0 candidates,"), "{table}");
}

#[test]
fn the_window_study_on_a_few_curves() {
    let (ok, table, _) = compare(&["window", "--limit", "2", "--threads", "2"]);
    assert!(ok);
    assert!(table.contains("2 curves, "), "{table}");
    assert!(table.contains("### A blind BLS search"), "{table}");
}

#[test]
fn the_alias_study() {
    let (ok, table, _) = compare(&["vet", "--limit", "3", "--threads", "2"]);
    assert!(ok);
    assert!(table.contains("alias table"), "{table}");
    assert!(table.contains("| 1/2 × true | TLS 1.32 | 1 |"), "{table}");
}

#[test]
fn the_usage_otherwise() {
    for args in [
        &["none"][..],
        &["--limit"],
        &["--limit", "x"],
        &["--threads", "0"],
        &["--step", "0"],
    ] {
        let (ok, _, usage) = compare(args);
        assert!(!ok, "{args:?}");
        assert!(usage.starts_with("compare [SET...]"), "{usage}");
    }
}

#[test]
fn a_speed_table_and_an_unknown_one() {
    let (ok, table, _) = compare(&["speed-model"]);
    assert!(ok);
    assert!(table.contains("## rustellar-model speed"), "{table}");
    assert!(table.contains("| batman 2.5.1 |"), "{table}");
    let (ok, _, usage) = compare(&["speed-nothing"]);
    assert!(!ok);
    assert!(usage.contains("speed-CRATE"), "{usage}");
}

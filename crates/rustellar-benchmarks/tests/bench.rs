// The rustellar-bench binary: the list, the help, the suites a command
// picks and the usage for anything else.

use std::process::Command;

fn bench(args: &[&str]) -> (bool, String, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_rustellar-bench"))
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
fn the_list_of_suites() {
    let (ok, list, _) = bench(&["list"]);
    assert!(ok);
    assert!(list.starts_with("| Suite | Crate |"), "{list}");
    for row in [
        "| many | tls | accuracy |",
        "| mcmc | mcmc | speed |",
        "| vet | vet | speed |",
    ] {
        assert!(list.contains(row), "{list}");
    }
}

#[test]
fn the_help() {
    for args in [&["help"][..], &["--help"], &["list", "-h"]] {
        let (ok, help, _) = bench(args);
        assert!(ok, "{args:?}");
        assert!(help.starts_with("rustellar-bench COMMAND"), "{help}");
    }
}

#[test]
fn the_suites_each_command_picks() {
    let picked = |args: &[&str]| {
        let mut all = vec![];
        all.extend_from_slice(args);
        all.push("--dry-run");
        let (ok, out, err) = bench(&all);
        assert!(ok, "{args:?}");
        assert!(err.contains("suites, about"), "{err}");
        out.lines().map(str::to_owned).collect::<Vec<String>>()
    };
    assert_eq!(
        picked(&["accuracy", "tls"]),
        [
            "many (tls)",
            "mixed (tls)",
            "varying (tls)",
            "short (tls)",
            "sectors (tls)",
            "long (tls)",
            "ultrashort (tls)",
            "deep (tls)",
            "grazing (tls)",
            "eccentric (tls)",
            "fast (tls)",
            "gaps (tls)",
            "flagged (tls)",
            "variable (tls)",
            "around (tls)"
        ]
    );
    // A suite by name, a crate and the same suite again: once each.
    assert_eq!(
        picked(&["accuracy", "limb", "bls", "window"]),
        ["limb (bls)", "window (detrend)"]
    );
    assert_eq!(
        picked(&["speed", "fit", "mcmc"]),
        ["fit (fit)", "mcmc (mcmc)"]
    );
    // A crate with no suite of that measure: none.
    assert!(picked(&["accuracy", "model"]).is_empty());
    assert_eq!(picked(&["accuracy"]).len(), 21);
    assert_eq!(picked(&["speed", "all"]).len(), 8);
    assert_eq!(picked(&["all", "--quick"]).len(), 29);
}

#[test]
fn a_suite_run() {
    let (ok, table, err) = bench(&[
        "accuracy",
        "many",
        "--limit",
        "1",
        "--threads",
        "2",
        "--cpu",
    ]);
    assert!(ok);
    assert!(err.contains("running many (tls)"), "{err}");
    assert!(table.contains("### many (1 curves)"), "{table}");
    let (ok, table, _) = bench(&[
        "accuracy",
        "limb",
        "--quick",
        "--step",
        "250",
        "--threads",
        "4",
    ]);
    assert!(ok);
    assert!(table.contains("(8 curves,"), "{table}");
}

#[test]
fn the_usage_otherwise() {
    for args in [
        &[][..],
        &["nothing"],
        &["accuracy", "nothing"],
        &["speed", "many"],
        &["all", "tls"],
        &["accuracy", "--limit"],
        &["accuracy", "--limit", "x"],
        &["accuracy", "--threads", "0"],
        &["accuracy", "--step", "0"],
        &["accuracy", "--fast"],
    ] {
        let (ok, _, usage) = bench(args);
        assert!(!ok, "{args:?}");
        assert!(usage.starts_with("rustellar-bench COMMAND"), "{usage}");
    }
}

fn folder(name: &str) -> std::path::PathBuf {
    let f = std::env::temp_dir().join(format!("rustellar-bench-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&f);
    f
}

#[test]
fn runs_saved_reported_and_compared() {
    let (a, b) = (folder("a"), folder("b"));
    let (a_str, b_str) = (a.to_str().unwrap(), b.to_str().unwrap());
    let (ok, _, err) = bench(&["accuracy", "many", "--limit", "1", "--cpu", "--to", a_str]);
    assert!(ok, "{err}");
    assert!(err.contains("saved in"), "{err}");
    let (ok, _, _) = bench(&["speed", "fit", "--repeat", "2", "--to", a_str]);
    assert!(ok);
    for file in [
        "machine.md",
        "accuracy-many.md",
        "speed-fit.md",
        "speed-fit.spread.csv",
    ] {
        assert!(a.join(file).is_file(), "{file}");
    }
    let card = std::fs::read_to_string(a.join("machine.md")).unwrap();
    assert!(
        card.contains("| Run | accuracy many |\n| Run | speed fit |"),
        "{card}"
    );
    let (ok, report, _) = bench(&["report", a_str]);
    assert!(ok);
    assert!(report.starts_with("# Machine"), "{report}");
    assert!(report.contains("## rustellar-fit speed"), "{report}");
    // The same run against itself, and against one with a slower fit.
    let (ok, same, err) = bench(&["diff", a_str, a_str]);
    assert!(ok);
    assert!(err.contains("0 cells differ"), "{err}");
    assert!(same.contains("No cell differs."), "{same}");
    std::fs::create_dir_all(&b).unwrap();
    let fit = std::fs::read_to_string(a.join("speed-fit.md")).unwrap();
    let slower: String = fit
        .lines()
        .map(|l| {
            if l.starts_with("| hot jupiter |") {
                "| hot jupiter | 99 s | 12 ms | 11 ms | 19 ms | 31 ms | 9.6 ms |".to_owned()
            } else {
                l.to_owned()
            }
        })
        .collect::<Vec<String>>()
        .join("\n");
    std::fs::write(b.join("speed-fit.md"), slower).unwrap();
    std::fs::write(b.join("accuracy-limb.md"), "## x\n").unwrap();
    let (ok, changed, err) = bench(&["diff", a_str, b_str, "--tolerance", "0.5"]);
    assert!(ok);
    assert!(err.contains("1 cells differ"), "{err}");
    assert!(changed.contains("% slower |"), "{changed}");
    assert!(changed.contains("Only in"), "{changed}");
    // Missing runs fail.
    let missing = folder("missing");
    let m = missing.to_str().unwrap();
    for args in [&["report", m][..], &["diff", m, a_str], &["diff", a_str, m]] {
        let (ok, _, _) = bench(args);
        assert!(!ok, "{args:?}");
    }
    let empty = folder("empty");
    std::fs::create_dir_all(&empty).unwrap();
    let (ok, _, err) = bench(&["report", empty.to_str().unwrap()]);
    assert!(!ok);
    assert!(err.contains("no saved results"), "{err}");
    // A run by version: one never saved.
    let (ok, _, _) = bench(&["report", "0.0.0-never"]);
    assert!(!ok);
    for f in [a, b, empty] {
        std::fs::remove_dir_all(f).unwrap();
    }
}

#[test]
fn the_new_options_checked() {
    for args in [
        &["speed", "--repeat", "0"][..],
        &["speed", "--repeat", "x"],
        &["diff", "a", "b", "--tolerance", "-1"],
        &["diff", "a", "b", "--tolerance", "x"],
        &["speed", "--to"],
        &["diff", "only-one"],
        &["report", "a", "b"],
    ] {
        let (ok, _, usage) = bench(args);
        assert!(!ok, "{args:?}");
        assert!(usage.starts_with("rustellar-bench COMMAND"), "{usage}");
    }
}

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
            "sectors (tls)"
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
    assert_eq!(picked(&["accuracy"]).len(), 11);
    assert_eq!(picked(&["speed", "all"]).len(), 8);
    assert_eq!(picked(&["all", "--quick"]).len(), 19);
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

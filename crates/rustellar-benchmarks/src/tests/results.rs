// Saved results: tables read back, times parsed, repeats merged, two runs
// compared, the machine's card and the files.

use std::path::PathBuf;

use crate::results::{
    Change, card, default_folder, diff, diff_report, file_name, load, machine, merge, read_spreads,
    save, seconds, spreads_csv, steadiness, tables,
};

const RUN: &str = "## rustellar-fit speed

| Case | rustellar-fit | batman 2.5.1 |
|---|--:|--:|
| hot jupiter | 1.3 ms | 12 ms |
| grazing | 14 ms | 0.38 s |

(8 s)

### counts

| Set | found | p |
|---|---|---|
| many | 1/2 | 0.50 |
";

fn with(rust: &str, grazing: &str) -> String {
    RUN.replace("| 1.3 ms |", &format!("| {rust} |"))
        .replace("| 14 ms |", &format!("| {grazing} |"))
}

#[test]
fn the_tables() {
    let t = tables(RUN);
    assert_eq!(t.len(), 2);
    assert_eq!(t[0].headings, ["rustellar-fit speed"]);
    assert_eq!(t[0].header, ["Case", "rustellar-fit", "batman 2.5.1"]);
    assert_eq!(t[0].rows[1], ["grazing", "14 ms", "0.38 s"]);
    // A deeper heading nests; one at the same level replaces it.
    assert_eq!(t[1].headings, ["rustellar-fit speed", "counts"]);
    let other = tables("# A\n\n## B\n\n## C\n\n| x |\n|---|\n| 1 |\n");
    assert_eq!(other[0].headings, ["A", "C"]);
    // A line of pipes with no rule under it is no table.
    assert!(tables("| a | b |\nplain\n").is_empty());
}

#[test]
fn the_times() {
    assert_eq!(seconds("4.0 µs"), Some(4.0e-6));
    assert_eq!(seconds("0.64 ms"), Some(6.4e-4));
    assert_eq!(seconds("1,200 s"), Some(1200.0));
    assert_eq!(seconds("3.0 ms†"), Some(3.0e-3));
    for not in ["1/2", "0.50", "-", "fast", "x ms"] {
        assert_eq!(seconds(not), None, "{not}");
    }
}

#[test]
fn repeats_merged() {
    let runs = [
        with("1.5 ms", "14 ms"),
        with("1.3 ms", "15 ms"),
        with("1.4 ms", "16 ms"),
    ];
    let (merged, spreads) = merge(&runs);
    // Each time its best; text, counts and the reference column as they were.
    assert!(
        merged.contains("| hot jupiter | 1.3 ms | 12 ms |"),
        "{merged}"
    );
    assert!(merged.contains("| grazing | 14 ms | 0.38 s |"), "{merged}");
    assert!(merged.contains("| many | 1/2 | 0.50 |"), "{merged}");
    assert!(merged.contains("(8 s)"), "{merged}");
    let spread = |cell: &str| spreads.iter().find(|(c, _)| c == cell).map(|s| s.1);
    let s = spread("rustellar-fit speed / hot jupiter / rustellar-fit").unwrap();
    assert!((s - 0.1 / 1.3).abs() < 1e-12, "{s}");
    assert_eq!(
        spread("rustellar-fit speed / grazing / batman 2.5.1"),
        Some(0.0)
    );
    // Counts are not times, so have no spread.
    assert_eq!(spread("rustellar-fit speed / counts / many / found"), None);
    // One run is its own merge.
    let (one, _) = merge(&[RUN.to_owned()]);
    assert_eq!(tables(&one), tables(RUN));
}

#[test]
#[should_panic(expected = "a run")]
fn no_runs_to_merge() {
    let _ = merge(&[]);
}

#[test]
fn two_runs_compared() {
    let ten = |_: &str| 0.1;
    assert!(diff(RUN, RUN, &ten).is_empty());
    // Within the tolerance, unchanged; beyond it, a change with its ratio.
    assert!(diff(RUN, &with("1.4 ms", "14 ms"), &ten).is_empty());
    let changes = diff(RUN, &with("2.6 ms", "14 ms"), &ten);
    assert_eq!(changes.len(), 1);
    assert_eq!(
        changes[0].cell,
        "rustellar-fit speed / hot jupiter / rustellar-fit"
    );
    assert!((changes[0].ratio.unwrap() - 2.0).abs() < 1e-12);
    // A wider tolerance for that cell alone lets it pass.
    let wide = |cell: &str| {
        if cell.contains("hot jupiter") {
            1.5
        } else {
            0.1
        }
    };
    assert!(diff(RUN, &with("2.6 ms", "14 ms"), &wide).is_empty());
    // Text differs at all; the same number written another way does not.
    let counted = RUN.replace("| 1/2 |", "| 2/2 |");
    let changes = diff(RUN, &counted, &ten);
    assert_eq!((changes.len(), changes[0].ratio), (1, None));
    assert!(diff(RUN, &RUN.replace("| 0.50 |", "| 0.5 |"), &ten).is_empty());
    // A row only one side has.
    let more = RUN.replace(
        "| many | 1/2 | 0.50 |",
        "| many | 1/2 | 0.50 |\n| few | 0/1 | 1.00 |",
    );
    let added = diff(RUN, &more, &ten);
    assert_eq!(added.len(), 2);
    assert!(added.iter().all(|c| c.old.is_empty()));
    let removed = diff(&more, RUN, &ten);
    assert!(removed.iter().all(|c| c.new.is_empty()));
}

#[test]
fn the_difference_as_a_table() {
    assert_eq!(diff_report(&[]), "No cell differs.\n");
    let change = |ratio| Change {
        cell: "a / b / c".to_owned(),
        old: "1 ms".to_owned(),
        new: "2 ms".to_owned(),
        ratio,
    };
    let report = diff_report(&[change(Some(2.0)), change(Some(0.5)), change(None)]);
    assert!(
        report.contains("| a / b / c | 1 ms | 2 ms | 100% slower |"),
        "{report}"
    );
    assert!(report.contains("| 50% faster |"), "{report}");
    assert!(report.contains("| differs |"), "{report}");
}

#[test]
fn spreads_as_a_file() {
    let spreads = vec![("x / 1,000 / y".to_owned(), 0.0312), ("z".to_owned(), 0.0)];
    let csv = spreads_csv(&spreads);
    assert!(csv.starts_with("cell,spread\n"));
    assert_eq!(
        read_spreads(&csv),
        [("x / 1,000 / y".to_owned(), 0.0312), ("z".to_owned(), 0.0)]
    );
    assert!(read_spreads("cell,spread\nno comma\nz,x\n").is_empty());
}

#[test]
fn the_machine() {
    let fresh = machine(8, true, "speed fit");
    for row in [
        "| Threads | 8 |",
        "| GPU searches | built in (cuda) |",
        "| Run | speed fit |",
    ] {
        assert!(fresh.contains(row), "{fresh}");
    }
    assert!(fresh.starts_with("# Machine"));
    assert!(machine(1, false, "x").contains("| GPU searches | off |"));
    // A second run adds its line to the card already saved.
    assert_eq!(card(None, &fresh, "later"), fresh);
    assert_eq!(card(Some("junk"), &fresh, "later"), fresh);
    let both = card(Some(&fresh), "unused", "accuracy tls");
    assert!(
        both.ends_with("| Run | speed fit |\n| Run | accuracy tls |\n"),
        "{both}"
    );
    assert!(steadiness(3) >= 0.0);
    assert!(default_folder().ends_with(env!("CARGO_PKG_VERSION")));
    assert_eq!(file_name("speed", "fit"), "speed-fit.md");
}

#[test]
fn files_saved_and_loaded() {
    let folder: PathBuf =
        std::env::temp_dir().join(format!("rustellar-results-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&folder);
    save(&folder, "speed-fit.md", RUN).unwrap();
    save(&folder, "machine.md", "# Machine\n").unwrap();
    save(&folder, "speed-fit.spread.csv", "cell,spread\n").unwrap();
    let files = load(&folder).unwrap();
    // The card first, then the tables; other files left out.
    let names: Vec<&str> = files.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(names, ["machine.md", "speed-fit.md"]);
    assert_eq!(files[1].1, RUN);
    std::fs::remove_dir_all(&folder).unwrap();
    assert!(load(&folder).is_err());
}

//! Saved results: each suite's Markdown kept in a folder per release with
//! a card of the machine it ran on; repeated speed runs merged into their
//! best with each time's spread; and two saved runs compared, table cell
//! by table cell.
//!
//! The tables are the suites' own Markdown, so the saved files read as
//! the READMEs do. A cell is a time when it is a number with a unit
//! (`4.0 µs`, `0.64 ms`, `1.2 s`), a number otherwise; any other text
//! (names, counts like `1/2`, dashes) is compared as text.

use std::fmt::Write;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::speed::short;

/// One Markdown table: the headings above it, its header and its rows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Table {
    pub headings: Vec<String>,
    pub header: Vec<String>,
    pub rows: Vec<Vec<String>>,
}

fn cells(line: &str) -> Vec<String> {
    let inner = line.trim().trim_start_matches('|').trim_end_matches('|');
    inner.split('|').map(|c| c.trim().to_owned()).collect()
}

fn is_rule(line: &str) -> bool {
    let t = line.trim();
    t.starts_with('|') && t.chars().all(|c| matches!(c, '|' | '-' | ':' | ' '))
}

/// Every table in `markdown`, with the `#` headings in force above each
/// (outermost first).
#[must_use]
pub fn tables(markdown: &str) -> Vec<Table> {
    let mut out = Vec::new();
    let mut headings: Vec<(usize, String)> = Vec::new();
    let lines: Vec<&str> = markdown.lines().collect();
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i].trim();
        if let Some(rest) = line.strip_prefix('#') {
            let level = 1 + rest.chars().take_while(|&c| c == '#').count();
            let title = rest.trim_start_matches('#').trim().to_owned();
            headings.retain(|(l, _)| *l < level);
            headings.push((level, title));
            i += 1;
        } else if line.starts_with('|') && i + 1 < lines.len() && is_rule(lines[i + 1]) {
            let header = cells(line);
            let mut rows = Vec::new();
            i += 2;
            while i < lines.len() && lines[i].trim().starts_with('|') {
                rows.push(cells(lines[i]));
                i += 1;
            }
            out.push(Table {
                headings: headings.iter().map(|(_, h)| h.clone()).collect(),
                header,
                rows,
            });
        } else {
            i += 1;
        }
    }
    out
}

/// A cell's time in seconds, if it is a number with a unit (a trailing
/// `†` or `*` allowed).
#[must_use]
pub fn seconds(cell: &str) -> Option<f64> {
    let t = cell.trim().trim_end_matches(['†', '*']).trim();
    let (number, scale) = [("µs", 1e-6), ("ms", 1e-3), (" s", 1.0)]
        .into_iter()
        .find_map(|(unit, scale)| t.strip_suffix(unit).map(|n| (n, scale)))?;
    number
        .trim()
        .replace(',', "")
        .parse::<f64>()
        .ok()
        .map(|v| v * scale)
}

fn number(cell: &str) -> Option<f64> {
    cell.trim()
        .trim_end_matches(['†', '*', '%'])
        .replace(',', "")
        .parse::<f64>()
        .ok()
}

/// Runs of one speed suite merged: each time cell the best of the runs,
/// and the spread of every time cell, `(median - best) / best`, keyed as
/// [`diff`] keys cells. The runs must have the same tables, as runs of
/// one suite do; text outside the tables is the first run's.
///
/// # Panics
///
/// On no runs.
#[must_use]
pub fn merge(runs: &[String]) -> (String, Vec<(String, f64)>) {
    let first = runs.first().expect("a run");
    let parsed: Vec<Vec<Table>> = runs.iter().map(|r| tables(r)).collect();
    let mut spreads = Vec::new();
    let mut merged = String::new();
    let mut table_index = 0;
    let lines: Vec<&str> = first.lines().collect();
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i];
        if line.trim().starts_with('|') && i + 1 < lines.len() && is_rule(lines[i + 1]) {
            let base = &parsed[0][table_index];
            writeln!(merged, "{line}\n{}", lines[i + 1]).expect("a string");
            for (r, row) in base.rows.iter().enumerate() {
                let mut out_row = Vec::new();
                for (c, cell) in row.iter().enumerate() {
                    let times: Vec<f64> = parsed
                        .iter()
                        .filter_map(|t| t.get(table_index)?.rows.get(r)?.get(c))
                        .filter_map(|s| seconds(s))
                        .collect();
                    if times.len() == runs.len() && seconds(cell).is_some() {
                        let mut sorted = times.clone();
                        sorted.sort_by(f64::total_cmp);
                        let best = sorted[0];
                        let median = sorted[sorted.len() / 2];
                        spreads.push((key(base, r, c), (median - best) / best));
                        out_row.push(short(Some(best)));
                    } else {
                        out_row.push(cell.clone());
                    }
                }
                writeln!(merged, "| {} |", out_row.join(" | ")).expect("a string");
            }
            i += 2 + base.rows.len();
            table_index += 1;
        } else {
            writeln!(merged, "{line}").expect("a string");
            i += 1;
        }
    }
    (merged, spreads)
}

/// A cell's name: its headings, its row's first cell and its column's
/// header.
fn key(table: &Table, row: usize, column: usize) -> String {
    format!(
        "{} / {} / {}",
        table.headings.join(" / "),
        table.rows[row].first().map_or("", String::as_str),
        table.header.get(column).map_or("", String::as_str)
    )
}

/// One cell that differs between two saved runs.
#[derive(Debug, Clone, PartialEq)]
pub struct Change {
    pub cell: String,
    pub old: String,
    pub new: String,
    /// The new time over the old, for a time; `None` otherwise.
    pub ratio: Option<f64>,
}

/// Every cell of `new`'s tables that differs from `old`'s at the same
/// place: a time by more than `tolerance(cell)` (relative), any other
/// number or text at all. Cells only one side has are listed with the
/// other side empty.
#[must_use]
pub fn diff(old: &str, new: &str, tolerance: &dyn Fn(&str) -> f64) -> Vec<Change> {
    let index = |md: &str| {
        let mut cells = Vec::new();
        for t in tables(md) {
            for (r, row) in t.rows.iter().enumerate() {
                for (c, cell) in row.iter().enumerate().skip(1) {
                    cells.push((key(&t, r, c), cell.clone()));
                }
            }
        }
        cells
    };
    let (a, b) = (index(old), index(new));
    let mut out = Vec::new();
    for (k, new_cell) in &b {
        match a.iter().find(|(ka, _)| ka == k) {
            Some((_, old_cell)) => {
                if let (Some(x), Some(y)) = (seconds(old_cell), seconds(new_cell)) {
                    let ratio = y / x;
                    if (ratio - 1.0).abs() > tolerance(k) {
                        out.push(Change {
                            cell: k.clone(),
                            old: old_cell.clone(),
                            new: new_cell.clone(),
                            ratio: Some(ratio),
                        });
                    }
                } else if old_cell != new_cell
                    && !matches!((number(old_cell), number(new_cell)), (Some(x), Some(y)) if x == y)
                {
                    out.push(Change {
                        cell: k.clone(),
                        old: old_cell.clone(),
                        new: new_cell.clone(),
                        ratio: None,
                    });
                }
            }
            None => out.push(Change {
                cell: k.clone(),
                old: String::new(),
                new: new_cell.clone(),
                ratio: None,
            }),
        }
    }
    for (k, old_cell) in &a {
        if !b.iter().any(|(kb, _)| kb == k) {
            out.push(Change {
                cell: k.clone(),
                old: old_cell.clone(),
                new: String::new(),
                ratio: None,
            });
        }
    }
    out
}

/// `changes` as a Markdown table, slower times first marked.
#[must_use]
pub fn diff_report(changes: &[Change]) -> String {
    if changes.is_empty() {
        return "No cell differs.\n".to_owned();
    }
    let mut out = String::from("| Cell | Old | New | Change |\n|---|---|---|---|\n");
    for c in changes {
        let change = match c.ratio {
            Some(r) if r > 1.0 => format!("{:.0}% slower", (r - 1.0) * 100.0),
            Some(r) => format!("{:.0}% faster", (1.0 - r) * 100.0),
            None => "differs".to_owned(),
        };
        writeln!(out, "| {} | {} | {} | {change} |", c.cell, c.old, c.new).expect("a string");
    }
    out
}

fn command(program: &str, args: &[&str]) -> Option<String> {
    let out = Command::new(program).args(args).output().ok()?;
    let text = String::from_utf8(out.stdout).ok()?;
    let line = text.lines().map(str::trim).find(|l| !l.is_empty())?;
    out.status.success().then(|| line.to_owned())
}

fn processor() -> String {
    if let Ok(text) = fs::read_to_string("/proc/cpuinfo") {
        if let Some(name) = text.lines().find_map(|l| {
            l.strip_prefix("model name")
                .map(|r| r.trim_start_matches([' ', '\t', ':']))
        }) {
            return name.trim().to_owned();
        }
    }
    command("sysctl", &["-n", "machdep.cpu.brand_string"])
        .or_else(windows_processor)
        .or_else(|| std::env::var("PROCESSOR_IDENTIFIER").ok())
        .unwrap_or_else(|| "unknown".to_owned())
}

/// The processor's name in the Windows registry, read with `reg query`.
fn windows_processor() -> Option<String> {
    let out = Command::new("reg")
        .args([
            "query",
            r"HKLM\HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            "/v",
            "ProcessorNameString",
        ])
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    text.lines()
        .find_map(|l| {
            l.split_once("REG_SZ")
                .map(|(_, name)| name.trim().to_owned())
        })
        .filter(|name| !name.is_empty())
}

/// The card `machine` made for a run, or, when `existing` is a card
/// already saved, that card with this run's line added.
#[must_use]
pub fn card(existing: Option<&str>, fresh: &str, when: &str) -> String {
    match existing {
        Some(card) if card.starts_with("# Machine") => {
            format!("{}\n| Run | {when} |\n", card.trim_end())
        }
        _ => fresh.to_owned(),
    }
}

/// The machine a run is made on, as Markdown: the processor, threads,
/// operating system, compiler, commit, crate version and whether the GPU
/// searches were built in.
#[must_use]
pub fn machine(threads: usize, cuda: bool, when: &str) -> String {
    let rustc = command("rustc", &["--version"]).unwrap_or_else(|| "unknown".to_owned());
    let commit =
        command("git", &["rev-parse", "--short", "HEAD"]).unwrap_or_else(|| "unknown".to_owned());
    let dirty = command("git", &["status", "--porcelain"]).is_some();
    format!(
        "# Machine\n\n| | |\n|---|---|\n| Processor | {} |\n| Threads | {threads} |\n| System | {} {} |\n| Compiler | {rustc} |\n| Commit | {commit}{} |\n| Version | {} |\n| GPU searches | {} |\n| Run | {when} |\n",
        processor(),
        std::env::consts::OS,
        std::env::consts::ARCH,
        if dirty { " (uncommitted changes)" } else { "" },
        env!("CARGO_PKG_VERSION"),
        if cuda { "built in (cuda)" } else { "off" },
    )
}

/// The folder results are saved in by default: `results/<version>` in
/// this crate.
#[must_use]
pub fn default_folder() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("results")
        .join(env!("CARGO_PKG_VERSION"))
}

/// The file a suite's results are saved to: `accuracy-<name>.md` or
/// `speed-<name>.md`.
#[must_use]
pub fn file_name(measure: &str, name: &str) -> String {
    format!("{measure}-{name}.md")
}

/// `text` written to `folder/name`, the folder made if need be.
///
/// # Errors
///
/// When the folder or file cannot be written.
pub fn save(folder: &Path, name: &str, text: &str) -> io::Result<()> {
    fs::create_dir_all(folder)?;
    fs::write(folder.join(name), text)
}

/// Every `.md` file of a saved run, by name, the machine card first.
///
/// # Errors
///
/// When the folder cannot be read.
pub fn load(folder: &Path) -> io::Result<Vec<(String, String)>> {
    let mut files: Vec<(String, String)> = Vec::new();
    for entry in fs::read_dir(folder)? {
        let path = entry?.path();
        if path.extension().is_some_and(|e| e == "md") {
            let name = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("")
                .to_owned();
            files.push((name, fs::read_to_string(&path)?));
        }
    }
    files.sort_by(|a, b| (a.0 != "machine.md", &a.0).cmp(&(b.0 != "machine.md", &b.0)));
    Ok(files)
}

/// Spreads as a file: one `cell,spread` row each (cells hold no commas
/// but in numbers, so the spread is the last field).
#[must_use]
pub fn spreads_csv(spreads: &[(String, f64)]) -> String {
    let mut out = String::from(
        "cell,spread
",
    );
    for (cell, spread) in spreads {
        writeln!(out, "{cell},{spread:.4}").expect("a string");
    }
    out
}

/// The spreads [`spreads_csv`] wrote; rows that do not read are skipped.
#[must_use]
pub fn read_spreads(csv: &str) -> Vec<(String, f64)> {
    csv.lines()
        .skip(1)
        .filter_map(|l| {
            let (cell, spread) = l.rsplit_once(',')?;
            Some((cell.to_owned(), spread.parse().ok()?))
        })
        .collect()
}

/// How steady the machine is: a fixed loop timed `runs` times, the spread
/// `(median - best) / best` of its times. Above a few per cent, something
/// else is running and speed tables will be noisy.
#[must_use]
pub fn steadiness(runs: usize) -> f64 {
    let mut times: Vec<f64> = (0..runs.max(1))
        .map(|_| {
            let start = std::time::Instant::now();
            let mut x = 1.0f64;
            for i in 0..2_000_000u32 {
                x = std::hint::black_box(x.mul_add(1.000_000_1, f64::from(i & 7)));
            }
            std::hint::black_box(x);
            start.elapsed().as_secs_f64()
        })
        .collect();
    times.sort_by(f64::total_cmp);
    (times[times.len() / 2] - times[0]) / times[0]
}

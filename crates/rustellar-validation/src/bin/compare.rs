//! Prints the comparisons as Markdown tables.
//!
//! ```text
//! cargo run --release -p rustellar-validation --bin compare -- [SET...] [--limit N] [--threads N] [--cpu]
//! ```
//!
//! SET is many, mixed, varying, short or sectors (default all). `--limit`
//! searches only each set's first N curves; `--threads` sets the threads per
//! search (default all cores); `--cpu` keeps the search off the GPU with the
//! `cuda` feature.

use std::num::NonZeroUsize;
use std::process::ExitCode;
use std::time::Instant;

use rustellar_tls::{Device, SearchOptions};
use rustellar_validation::{Kind, tls};

const HELP: &str = "compare [SET...] [--limit N] [--threads N] [--cpu]
  SET: many, mixed, varying, short or sectors (default all)";

fn main() -> ExitCode {
    let mut kinds = vec![];
    let mut limit = usize::MAX;
    let mut options = SearchOptions::default().with_bootstrap(0);
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut number = || args.next().and_then(|n| n.parse::<usize>().ok());
        match arg.as_str() {
            "--limit" => match number() {
                Some(n) => limit = n,
                None => return usage(),
            },
            "--threads" => match number().and_then(NonZeroUsize::new) {
                Some(n) => options = options.with_threads(n),
                None => return usage(),
            },
            "--cpu" => options = options.with_device(Device::Cpu),
            name => match Kind::from_name(name) {
                Some(kind) => kinds.push(kind),
                None => return usage(),
            },
        }
    }
    if kinds.is_empty() {
        kinds = Kind::ALL.to_vec();
    }
    println!("## rustellar-tls against transitleastsquares 1.32\n");
    for kind in kinds {
        let start = Instant::now();
        let curves = tls::run(kind, limit, &options);
        print!("{}", tls::report(kind, &curves));
        println!("\n({:.0} s)\n", start.elapsed().as_secs_f64());
    }
    ExitCode::SUCCESS
}

fn usage() -> ExitCode {
    eprintln!("{HELP}");
    ExitCode::FAILURE
}

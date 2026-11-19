// The limb-darkened study: its curves built again, both searches on one,
// and the report's tables on hits made up to land each way.

use std::num::NonZeroUsize;

use rustellar_tls::{Device, SearchOptions};

use crate::limb::{Hit, cases, report, search_all};

fn options() -> SearchOptions {
    SearchOptions::default()
        .with_bootstrap(0)
        .with_device(Device::Cpu)
}

// The first block: each family with a transit, then without, the samples
// checked against the generator's, and transitleastsquares 1.32 on it.
#[test]
fn the_first_block() {
    let block = cases(250);
    assert_eq!(block.len(), 8);
    for (i, c) in block.iter().enumerate() {
        assert_eq!((c.family, c.injected), (i % 4, i < 4));
        assert!(c.tls.is_some());
    }
    // Every second block is noise only, and every fourth pair is in the
    // subset.
    assert_eq!(cases(1).len(), 2000);
}

// Both searches find the first transit, TLS with the higher SDE.
#[test]
fn both_find_the_first_transit() {
    let block = cases(250);
    let hits = search_all(&block[..1], &options(), NonZeroUsize::new(2).unwrap());
    let c = &block[0];
    for hit in hits[0] {
        assert!((hit.period / c.period - 1.0).abs() < 0.01, "{hit:?}");
    }
    assert!(hits[0][1].sde > hits[0][0].sde, "{hits:?}");
}

// Made-up hits: every transit found at its period by BLS, none by TLS (its
// periods wrong); the noise curves' SDEs set the thresholds.
#[test]
fn the_tables() {
    let block = cases(250);
    let hits: Vec<[Hit; 2]> = block
        .iter()
        .map(|c| {
            let at = |period: f64| Hit {
                sde: if c.injected { 10.0 } else { 3.0 },
                period,
                t0: c.tc,
            };
            [at(c.period), at(2.1 * c.period)]
        })
        .collect();
    let table = report(&block, &hits);
    assert!(
        table.contains("rustellar-bls 3.00, rustellar-tls 3.00"),
        "{table}"
    );
    assert!(
        table.contains("| all | 4/4 | 0.00e0 | 0.00 | 0/4 | |  |"),
        "{table}"
    );
    assert!(table.contains("| transitleastsquares 1.32 |"), "{table}");
    // Without the subset's curves: no second table.
    let mut outside = block.clone();
    for c in &mut outside {
        c.tls = None;
    }
    let table = report(&outside, &hits);
    assert!(!table.contains("subset"), "{table}");
}

// The masking study on a few curves: every variant run and searched, and
// the table's rows.

use std::num::NonZeroUsize;

use crate::masking::{VARIANTS, Variant, measure_all, report};
use crate::window::cases;

#[test]
fn every_variant_on_a_few_curves() {
    let drawn = cases(3);
    let results = measure_all(&drawn, NonZeroUsize::new(2).unwrap());
    assert_eq!(results.len(), 3);
    for outcomes in &results {
        assert_eq!(outcomes.len(), VARIANTS.len());
        for (o, v) in outcomes.iter().zip(&VARIANTS) {
            assert!(o.kept.snr.is_finite() && o.kept.depth.is_finite());
            assert!(o.found.sde.is_finite());
            match *v {
                Variant::Blind => assert_eq!(o.searches, 1),
                Variant::Iterative { rounds, .. } => {
                    assert!((1..=rounds).contains(&o.searches));
                }
                Variant::Truth { .. } => assert_eq!(o.searches, 0),
            }
        }
    }
    let table = report(&drawn, &results);
    let rows: Vec<&str> = table.lines().filter(|l| l.starts_with('|')).collect();
    // The header, its rule, and each variant.
    assert_eq!(rows.len(), 2 + VARIANTS.len(), "{table}");
    let columns = |l: &str| l.matches('|').count();
    assert!(
        rows.iter().all(|r| columns(r) == columns(rows[0])),
        "{table}"
    );
    assert!(
        table.contains("| masked once | 1.5 x duration |"),
        "{table}"
    );
    assert!(
        table.contains("| true transits masked | 1.2 x T14 |"),
        "{table}"
    );
}

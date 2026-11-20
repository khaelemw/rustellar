// The whitening study on a few curves: every noise and order run and
// searched, and the table's rows.

use std::num::NonZeroUsize;

use crate::whitening::{NOISES, ORDERS, measure_all, report};
use crate::window::cases;

#[test]
fn every_noise_on_a_few_curves() {
    let drawn = cases(2);
    let results = measure_all(&drawn, NonZeroUsize::new(2).unwrap());
    assert_eq!(results.len(), 2);
    for outcomes in &results {
        assert_eq!(outcomes.len(), NOISES.len());
        for o in outcomes {
            for s in [&o.transit, &o.empty] {
                assert!(s.flattened.sde.is_finite());
                assert_eq!(s.whitened.len(), ORDERS.len());
                for (&(found, order, gain), &most) in s.whitened.iter().zip(&ORDERS) {
                    assert!(found.sde.is_finite() && gain.is_finite());
                    assert!(order <= most);
                }
            }
        }
    }
    let table = report(&results);
    let rows: Vec<&str> = table.lines().filter(|l| l.starts_with('|')).collect();
    // The header, its rule, and per noise the flattened row and one per order.
    assert_eq!(rows.len(), 2 + NOISES.len() * (1 + ORDERS.len()), "{table}");
    let columns = |l: &str| l.matches('|').count();
    assert!(
        rows.iter().all(|r| columns(r) == columns(rows[0])),
        "{table}"
    );
    assert!(
        table.contains("| AR(1) 0.8 | whitened, AIC up to 30 |"),
        "{table}"
    );
}

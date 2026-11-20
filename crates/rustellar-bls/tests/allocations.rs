// search_into with a result of the right size allocates the same few buffers
// whatever the number of periods: the bins, running totals, scores and
// bounds once per thread, sized for the longest period, and nothing per
// period. The only
// test in its file, so no other test allocates while it counts.

use std::alloc::{GlobalAlloc, Layout, System};
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicUsize, Ordering};

use rustellar_bls::{Bls, BlsResult, Device, Durations, Periods, SearchOptions};
use rustellar_core::LightCurve;

struct Counting;

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

// SAFETY: every call is passed straight to the system allocator.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: as the caller's contract.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: as the caller's contract.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: as the caller's contract.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

#[test]
fn allocations_do_not_grow_with_the_periods() {
    let time: Vec<f64> = (0..3000).map(|i| 0.01 * f64::from(i)).collect();
    let flux = time
        .iter()
        .map(|t| if t % 2.7 < 0.1 { 0.995 } else { 1.0 })
        .collect();
    let lc = LightCurve::from_days(time, flux, vec![1e-3; 3000]).unwrap();
    let bls = Bls::new(&lc).unwrap();
    let durations = Durations::from_days(&[0.05, 0.1, 0.2]).unwrap();
    // The same longest period, so the same bins.
    let grid = |count: u32| {
        let days: Vec<f64> = (0..count)
            .map(|k| 10.0 - 9.0 * f64::from(k) / f64::from(count))
            .collect();
        Periods::from_days(&days).unwrap()
    };
    let (few, many) = (grid(50), grid(2000));
    for threads in [1, 4] {
        let options = SearchOptions::default()
            .with_device(Device::Cpu)
            .with_refine_peaks(0)
            .with_threads(NonZeroUsize::new(threads).unwrap());
        let mut result = BlsResult::default();
        bls.search_into(&many, &durations, &options, &mut result)
            .unwrap();
        let mut count = |periods: &Periods| {
            let before = ALLOCATIONS.load(Ordering::Relaxed);
            bls.search_into(periods, &durations, &options, &mut result)
                .unwrap();
            ALLOCATIONS.load(Ordering::Relaxed) - before
        };
        let (a, b) = (count(&few), count(&many));
        assert_eq!(a, b, "{threads} threads");
        // The durations and the split into chunks, and per thread its ten
        // buffers and the thread's own bookkeeping (more when the test
        // harness captures its output).
        assert!(b <= 6 + 16 * threads, "{threads} threads: {b}");
    }
}

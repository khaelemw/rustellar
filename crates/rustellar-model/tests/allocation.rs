// Checks that evaluating a light curve into an existing buffer allocates
// nothing, by counting allocations on the test's own thread.

use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
    num::NonZeroUsize,
};

use rustellar_core::{Epoch, TimeDelta};
use rustellar_model::{
    Exposure, ImpactParameter, LimbDarkening, RadiusRatio, ScaledSemiMajorAxis, TransitModel,
    TransitParams,
};

// ==========================
// Helper Functions
// ==========================

struct Counting;

thread_local! {
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

// SAFETY: every call is passed straight to the system allocator; the
// counter only reads and writes a thread-local `Cell`, which does not
// allocate.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.with(|count| count.set(count.get() + 1));
        // SAFETY: the caller's contract for `alloc` is passed on unchanged.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: the caller's contract for `dealloc` is passed on unchanged.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATIONS.with(|count| count.set(count.get() + 1));
        // SAFETY: the caller's contract for `realloc` is passed on unchanged.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

// Allocations made on this thread while running `f`.
fn allocations_in(f: impl FnOnce()) -> usize {
    let before = ALLOCATIONS.with(Cell::get);
    f();
    ALLOCATIONS.with(Cell::get) - before
}

fn params(k: f64) -> TransitParams {
    TransitParams::new(
        Epoch::from_days(0.0).unwrap(),
        TimeDelta::from_days(3.0).unwrap(),
        RadiusRatio::new(k).unwrap(),
        ScaledSemiMajorAxis::new(10.0).unwrap(),
        ImpactParameter::new(0.3).unwrap(),
        LimbDarkening::quadratic(0.4, 0.25).unwrap(),
    )
    .unwrap()
}

fn times() -> Vec<Epoch> {
    (-500..=500)
        .map(|i| Epoch::from_days(f64::from(i) * 1e-4).unwrap())
        .collect()
}

// ==========================
// Tests
// ==========================

#[test]
fn counter_sees_allocations() {
    assert_eq!(allocations_in(|| drop(vec![1.0_f64; 10])), 1);
}

#[test]
fn model_light_curve_does_not_allocate() {
    let exposure = Exposure::new(
        TimeDelta::from_hours(0.5).unwrap(),
        NonZeroUsize::new(15).unwrap(),
    )
    .unwrap();
    let mut plain = TransitModel::new(times());
    let mut binned = TransitModel::with_exposure(times(), exposure);
    let mut out = vec![0.0; times().len()];

    let count = allocations_in(|| {
        for k in [0.05, 0.1, 0.15] {
            let _ = plain.light_curve(params(k));
            let _ = binned.light_curve(params(k));
            binned.light_curve_into(params(k), &mut out).unwrap();
        }
    });
    assert_eq!(count, 0);
}

#[test]
fn params_light_curve_into_does_not_allocate() {
    let times = times();
    let mut out = vec![0.0; times.len()];
    let count = allocations_in(|| params(0.1).light_curve_into(&times, &mut out).unwrap());
    assert_eq!(count, 0);
}

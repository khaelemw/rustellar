// The offset from the nearest mid-transit: exact where the answer is a
// representable number, and never more than half a period.

use rstest::rstest;

use crate::box_fit::{AtPeriod, Cells, nearest, offset_from_transit};

// 2^-1000 and 2^950.
const TINY: f64 = 9.332_636_185_032_189e-302;
const HUGE: f64 = 9.516_908_214_257_812e285;

#[test]
fn tiny_and_huge_are_powers_of_two() {
    assert_eq!(TINY.to_bits(), (1023_u64 - 1000) << 52);
    assert_eq!(HUGE.to_bits(), (1023_u64 + 950) << 52);
}

#[rstest]
#[case::at_t0(1.5, 1.5, 2.0, 0.0)]
#[case::orbits_later(3001.5, 1.5, 2.0, 0.0)]
#[case::orbits_earlier(-2.5, 1.5, 2.0, 0.0)]
#[case::after(3001.75, 1.5, 2.0, 0.25)]
#[case::before(3001.25, 1.5, 2.0, -0.25)]
#[case::before_t0(1.25, 1.5, 2.0, -0.25)]
// Beyond 2^26 orbits, and for periods past 2^-900 or 2^900, the product of
// the orbits and the period comes from a fused multiply-add instead.
#[case::many_orbits(100_000_000.25, 0.0, 1.0, 0.25)]
#[case::tiny_period(1.25 * TINY, 0.0, TINY, 0.25 * TINY)]
#[case::huge_period(3.25 * HUGE, 0.0, HUGE, 0.25 * HUGE)]
fn exact_offsets(#[case] t: f64, #[case] t0: f64, #[case] p: f64, #[case] offset: f64) {
    assert_eq!(offset_from_transit(t, t0, p), offset);
}

// 0.1 is not a representable number, so 3000.1 - 0.1 is not 3000 exactly;
// the offset is the exact difference of the two floats, about -9e-14 days,
// which plain subtraction rounds away.
#[test]
fn keeps_what_plain_subtraction_loses() {
    let (t, t0, p) = (3000.1, 0.1, 1.0);
    let ours = offset_from_transit(t, t0, p);
    // 3000.1 is 3000.09999999999990905052982270717620849609375 and 0.1 is
    // 0.1000000000000000055511151231257827021181583404541015625.
    // Their difference less 3000, worked out in rational arithmetic, is this
    // float exactly.
    assert_eq!(ours, -9.095_502_129_241_595e-14);
    assert_eq!(t - t0 - 3000.0, 0.0);
}

#[test]
fn never_more_than_half_a_period() {
    for i in 0..10_000 {
        let t = 1325.0 + 0.037 * f64::from(i);
        let offset = offset_from_transit(t, 1.3, 0.731);
        assert!(offset.abs() <= 0.5 * 0.731, "{t}: {offset}");
    }
}

// Halves go to the even neighbour; from 2^52 up, every double is whole.
#[rstest]
#[case::down(2.4, 2.0)]
#[case::up(2.6, 3.0)]
#[case::half_to_even_below(2.5, 2.0)]
#[case::half_to_even_above(3.5, 4.0)]
#[case::negative(-2.5, -2.0)]
#[case::small_negative(-0.4, -0.0)]
#[case::just_below_2_52(4_503_599_627_370_495.5, 4_503_599_627_370_496.0)]
#[case::large(1e17, 1e17)]
#[case::large_negative(-1e17, -1e17)]
fn nearest_whole_number(#[case] x: f64, #[case] expected: f64) {
    assert_eq!(nearest(x).to_bits(), expected.to_bits());
}

// ==========================
// The offset by multiplication
// ==========================

// The offset as it was computed before the per-period constants: the whole
// number of orbits by division, then the same exact products. The fast paths
// must give these bits.
fn by_division(s: f64, e: f64, p: f64) -> f64 {
    let n = nearest(s / p);
    let (hi, lo) = if n.abs() < 67_108_864.0 && (1e-270..1e270).contains(&p.abs()) {
        let c = 134_217_729.0 * p;
        let p_hi = c - (c - p);
        (n * p_hi, n * (p - p_hi))
    } else {
        let hi = n * p;
        (hi, n.mul_add(p, -hi))
    };
    ((s - hi) - lo) + e
}

// Each path, where it applies, against the division.
fn every_path(s: f64, e: f64, p: f64) {
    let at = AtPeriod::new(p);
    let expected = by_division(s, e, p).to_bits();
    assert_eq!(at.offset_after(s, e).to_bits(), expected, "{s} {e} {p}");
    if s >= 0.0 {
        assert_eq!(
            at.offset_after_start(s, e).to_bits(),
            expected,
            "{s} {e} {p}"
        );
        if at.short_up_to(s) {
            assert_eq!(
                at.offset_after_start_short(s, e).to_bits(),
                expected,
                "{s} {e} {p}"
            );
        }
    }
}

// SplitMix64, for reproducible random inputs.
fn uniform(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 11) as f64 / (1u64 << 53) as f64
}

#[test]
fn multiplication_gives_the_division_bits_at_random() {
    let mut state = 17;
    for _ in 0..200_000 {
        let s = 4000.0 * uniform(&mut state);
        let p = 0.05 + 500.0 * uniform(&mut state);
        let e = (uniform(&mut state) - 0.5) * 1e-13;
        every_path(s, e, p);
        every_path(-s, e, p);
    }
}

// Quotients within a few rounding steps of a half, where the product and
// the quotient can round to different whole numbers: the margin sends these
// to the division.
#[test]
fn multiplication_gives_the_division_bits_next_to_a_half() {
    let mut state = 18;
    for _ in 0..20_000 {
        let p = 0.05 + 500.0 * uniform(&mut state);
        let n = (1.0 + 1e6 * uniform(&mut state)).floor();
        // Positive, so neighbouring doubles have neighbouring bits.
        let half = ((n + 0.5) * p).to_bits();
        for bits in half - 6..half + 6 {
            every_path(f64::from_bits(bits), 0.0, p);
        }
    }
}

// Many orbits (past the exact split, and past 2^52 where every double is
// whole), and periods too small or large to split.
#[rstest]
#[case::past_the_split(1e9, 3.7)]
#[case::past_2_52(1e17, 1.3)]
#[case::far_past_2_52(1e300, 1e-5)]
#[case::tiny_period(1.0, TINY)]
#[case::huge_period(1.0, HUGE)]
#[case::zero_elapsed(0.0, 2.5)]
fn multiplication_gives_the_division_bits_at_the_ends(#[case] s: f64, #[case] p: f64) {
    for k in [-3.0, 0.0, 3.0] {
        let s = s + k * s * f64::EPSILON;
        every_path(s, 0.0, p);
        every_path(s, 1e-20, p);
    }
}

// The cells find the first point not below any time, as a binary search of
// them all does: with gaps, runs of equal elapsed times, times at and next
// to the cells' edges and every point's own time, before the first and past
// the last, and with no cells at all.
#[rstest]
#[case::gaps(&[0.0, 0.1, 0.2, 5.0, 5.05, 5.1, 9.0, 20.0, 20.5, 21.0, 21.5, 22.0, 30.0])]
#[case::equal_times(&[0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0])]
#[case::one_point(&[0.0])]
#[case::all_at_the_first(&[0.0, 0.0, 0.0, 0.0, 0.0])]
#[case::two_points(&[0.0, 1e-9])]
fn cells_find_the_first_point_from_a_time(#[case] times: &[f64]) {
    let mut state = 4;
    let mut elapsed: Vec<(f64, f64)> = times.iter().map(|&s| (s, 0.0)).collect();
    // And a long irregular run.
    if times.len() > 10 {
        let mut s = 31.0;
        for _ in 0..2000 {
            s += 0.02 * uniform(&mut state);
            elapsed.push((s, 0.0));
        }
    }
    let cells = Cells::new(&elapsed);
    let last = elapsed[elapsed.len() - 1].0;
    let mut xs = vec![-1.0, 0.0, -0.0, f64::MIN_POSITIVE, last, last * 2.0 + 1.0];
    for &(s, _) in &elapsed {
        xs.extend(neighbours(s));
    }
    for k in 0..=elapsed.len() {
        // Cell edges as `Cells` makes them, when there are cells.
        let width = last / (elapsed.len() / 4).max(1) as f64;
        let edge = k as f64 * width;
        xs.extend(neighbours(edge));
    }
    for _ in 0..1000 {
        xs.push((last + 1.0) * uniform(&mut state));
    }
    for x in xs {
        assert_eq!(
            cells.first_from(&elapsed, x),
            elapsed.partition_point(|&(s, _)| s < x),
            "{x}"
        );
    }
}

// The doubles either side of `x`, by their bits (`next_up` needs a newer
// Rust than the crate's oldest); `x` itself unless it is positive and finite.
fn neighbours(x: f64) -> [f64; 3] {
    if x > 0.0 && x.is_finite() {
        [
            x,
            f64::from_bits(x.to_bits() + 1),
            f64::from_bits(x.to_bits() - 1),
        ]
    } else {
        [x; 3]
    }
}

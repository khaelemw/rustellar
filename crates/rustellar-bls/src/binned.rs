//! The fast binned search at one period.
//!
//! Fold the light curve at the period and drop the points into `m` equal
//! phase bins that tile the orbit exactly, as in Kovács, Zucker & Mazeh
//! (2002), from the phase of the first point. `m` is the fewest bins no wider
//! than `min(D) / oversample`, so a bin is `b = P / m`. Each duration becomes
//! a whole number of bins, `k = round(D / b)`, and a box is `k` neighbouring
//! bins. Running totals of the weight and the weighted flux, over the bins
//! twice round the orbit, give any box's sums in and out of transit as
//! differences of two totals, so every start bin and every duration costs
//! one step each.
//!
//! Each bin's own sums are plain sums of its few points, rounding at the size
//! of the bin; the running totals over the bins are compensated, so a box's
//! sums are about as accurate as adding up its own points, however many
//! points the light curve has. (astropy's plain running totals round at the
//! size of the whole light curve's sums.)
//!
//! astropy instead uses bins of exactly `min(D) / oversample`, so its last bin
//! is a part bin, and a box through it is shorter than `k b`. Here every box
//! is exactly `k b` long, wherever it starts, so it holds a set of points that
//! a box of that duration can hold. astropy also pads only `oversample` bins
//! past the end, on top of the last real bin; going twice round the orbit, a
//! box of any length can start in any bin, and no bin is lost.

use crate::{
    BlsError, BoxFit, Objective,
    box_fit::{AtPeriod, Weighted},
    totals::{Orbit, Total, box_sums},
};

/// The best box of a binned search at one period.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct BinnedBox {
    /// Its fit, from the bins' sums.
    pub(crate) fit: BoxFit,
    /// The bin width, days: the period over the number of bins.
    pub(crate) bin: f64,
    /// The first bin, counted from the bin holding the first point.
    pub(crate) start: usize,
    /// How many bins long.
    pub(crate) bins: usize,
    /// Which of the durations searched.
    pub(crate) duration: usize,
    /// The middle of the box, days after the first time: the middle of the
    /// range of times at which a box `bins` bins long holds the same points.
    pub(crate) centre: f64,
}

/// The bins and running totals of a binned search, kept between periods so
/// that a search allocates nothing per period.
#[derive(Debug, Clone, Default)]
pub(crate) struct Scratch {
    bins: Vec<Bin>,
    /// Each bin's lowest and highest phase, kept only when the orbits hold
    /// few points each (see [`LAZY_POINTS`]).
    ranges: Vec<(f64, f64)>,
    /// A window's running totals from its first start, and one orbit on.
    near: Vec<Total>,
    far: Vec<Total>,
    scores: Vec<f64>,
    bounds: Bounds,
    /// Each duration's length in bins (none if a box of it holds every bin)
    /// and its best box so far: reused, so no period allocates.
    lengths: Vec<Option<usize>>,
    bests: Vec<Option<(f64, usize, [Total; 3])>>,
}

impl Scratch {
    /// Room for up to `bins` bins and `durations` durations, so no period
    /// that needs no more grows it.
    pub(crate) fn with_capacity(bins: usize, durations: usize) -> Self {
        let window = WINDOW.min(bins);
        // Twice round, or a window and the longest box, and the first total.
        let near = window + bins + 1;
        Self {
            bins: Vec::with_capacity(bins),
            ranges: Vec::with_capacity(bins),
            near: Vec::with_capacity(near),
            far: Vec::with_capacity(window),
            scores: Vec::with_capacity(window),
            bounds: Bounds {
                g: Vec::with_capacity(near),
                weight: Vec::with_capacity(near),
                least: Vec::with_capacity(near.div_ceil(GRAIN)),
                greatest: Vec::with_capacity(near.div_ceil(GRAIN)),
                ..Bounds::default()
            },
            lengths: Vec::with_capacity(durations),
            bests: Vec::with_capacity(durations),
        }
    }
}

/// The best box by `objective` at `period` days over `durations` days, with
/// bins no wider than `min(durations) / oversample`. The first time is the
/// bins' phase zero, as for `weighted.elapsed`.
pub(crate) fn best_binned(
    weighted: &Weighted,
    period: f64,
    durations: &[f64],
    oversample: usize,
    objective: Objective,
    scratch: &mut Scratch,
) -> Result<Option<BinnedBox>, BlsError> {
    let m = (bin_count(period, durations, oversample) as usize).max(1);
    let bin = period / m as f64;

    let Scratch {
        bins,
        ranges,
        near,
        far,
        scores,
        bounds,
        lengths,
        bests,
    } = scratch;
    bins.clear();
    bins.resize(m, Bin::default());
    let at = AtPeriod::new(period);
    let last = weighted.elapsed.last().map_or(0.0, |&(s, _)| s);
    let short = at.short_up_to(last);
    // Few points an orbit, each bin keeps its phase range as it fills. (So
    // does the longer path, which takes more than 2^25 orbits, or a period
    // so long that every point is in one bin.)
    let eager = !short || (last / period + 1.0) * LAZY_POINTS > weighted.elapsed.len() as f64;
    ranges.clear();
    if eager {
        ranges.resize(m, (f64::INFINITY, f64::NEG_INFINITY));
    }
    let offset = |s, e| at.offset_after_start_short(s, e);
    match (short, eager) {
        (true, true) => fill::<true>(bins, ranges, weighted, period, bin, offset),
        (true, false) => fill::<false>(bins, ranges, weighted, period, bin, offset),
        _ => fill::<true>(bins, ranges, weighted, period, bin, |s, e| {
            at.offset_after_start(s, e)
        }),
    }

    // The running totals over the bins, twice round the orbit, `totals[k]`
    // over the first `k`: every box's sums are differences of two. They are
    // made a window of starts at a time, and one orbit on, by two running
    // totals, so no more than a window's are held: at the longest periods
    // all of them would not fit in the cache. Each is the same sequence of
    // additions as over them all, so the same to the bit.
    let add = |total: &mut Total, b: &Bin| {
        total.weight.add(b.weight);
        total.flux.add(b.flux);
    };
    // A box of every bin holds every point. (A box of no bins holds none,
    // and scores zero.)
    lengths.clear();
    lengths.extend(
        durations
            .iter()
            .map(|&d| Some((d / bin).round() as usize).filter(|&length| length < m)),
    );
    let longest = lengths.iter().flatten().copied().max().unwrap_or(0);

    // Every duration in a window before the next. Each duration's best is
    // the first of its highest in start order, with its totals at its start,
    // end and one orbit on.
    bests.clear();
    bests.resize(durations.len(), None);
    // The best score so far over every duration: a block of boxes whose
    // bound is below it is skipped. (Strictly below, so that no box equal
    // to the best is skipped, and the first of equal ones is kept.)
    let mut highest = 0.0_f64;
    let mut score = |whole: &Orbit, from: usize, near: &[Total], far: &[Total]| {
        let window = far.len().min(m - from);
        scores.clear();
        scores.resize(window, 0.0);
        bounds.prepare(near, whole);
        for (length, best) in lengths.iter().zip(bests.iter_mut()) {
            let Some(length) = *length else {
                continue;
            };
            // Runs of blocks not skipped, each scored at once.
            let mut run = 0;
            let mut i = 0;
            while run < window {
                let end = (i + STARTS).min(window);
                let skip = i < window && bounds.bound(i, end, length) < highest;
                if i < window && !skip {
                    i = end;
                    continue;
                }
                if run < i {
                    whole.statistics(&near[run..], length, &far[run..], &mut scores[run..i])?;
                    for (k, &score) in (run..i).zip(&scores[run..i]) {
                        if score > best.map_or(0.0, |b| b.0) {
                            *best = Some((score, from + k, [near[k], near[k + length], far[k]]));
                            highest = highest.max(score);
                        }
                    }
                }
                i = end;
                run = end;
            }
        }
        Ok::<(), BlsError>(())
    };
    if m <= WINDOW {
        // All of them at once, twice round.
        near.clear();
        let mut total = Total::default();
        near.push(total);
        for b in bins.iter().chain(bins.iter()) {
            add(&mut total, b);
            near.push(total);
        }
        let whole = Orbit::new(&near[0], &near[m], objective);
        score(&whole, 0, near, &near[m..2 * m])?;
    } else {
        let mut orbit = Total::default();
        for b in bins.iter() {
            add(&mut orbit, b);
        }
        let whole = Orbit::new(&Total::default(), &orbit, objective);
        let (mut at_start, mut at_wrap) = (Total::default(), orbit);
        for from in (0..m).step_by(WINDOW) {
            let window = WINDOW.min(m - from);
            // `near[i]` is `totals[from + i]`, for the window's starts and
            // the longest box's ends; `far[i]` is `totals[from + m + i]`.
            near.clear();
            let mut total = at_start;
            for k in from..from + window + longest {
                near.push(total);
                add(&mut total, &bins[k % m]);
            }
            at_start = near[window];
            far.clear();
            let mut total = at_wrap;
            for b in &bins[from..from + window] {
                far.push(total);
                add(&mut total, b);
            }
            at_wrap = total;
            score(&whole, from, near, far)?;
        }
    }

    // The first of the highest in duration order: the first of the highest
    // over the durations and then the starts. Its fit is built once.
    let mut best: Option<(f64, usize, usize, usize, [Total; 3])> = None;
    for (duration, (found, length)) in bests.iter().zip(lengths.iter()).enumerate() {
        if let (Some((score, start, totals)), Some(length)) = (*found, *length) {
            if score > best.map_or(0.0, |b| b.0) {
                best = Some((score, length, start, duration, totals));
            }
        }
    }
    let Some((_, length, start, duration, [first, end, wrap])) = best else {
        return Ok(None);
    };
    let n_in = (start..start + length).map(|k| bins[k % m].count).sum();
    let [inside, outside] = box_sums(&first, &end, &wrap);
    let fit = BoxFit::from_sums(weighted.mean, inside, outside, n_in)?;

    // The fit is finite, so there are points in the box and out of it: the
    // box's first and last bins with points, and the nearest outside it
    // either way round, all exist. Indices start one orbit on, so the bins
    // before the box are not below 0.
    let full = |j: &usize| bins[j % m].count > 0;
    let from = start + m;
    let edges = [
        (from..from + length).find(full),
        (from..from + length).rev().find(full),
        (from + length..from + m).find(full),
        (from + length - m..from).rev().find(full),
    ]
    .map(|j| j.expect("points in and out of transit"));
    let phases = if eager {
        let [first, last, next, previous] = edges.map(|j| ranges[j % m]);
        [first.0, last.1, next.0, previous.1]
    } else {
        phases(&edges, weighted, period, bin, m, offset)
    };
    Ok(Some(BinnedBox {
        fit,
        bin,
        start,
        bins: length,
        duration,
        centre: centre(&edges, &phases, length as f64 * bin, period, m),
    }))
}

/// Fewer points an orbit than this, and the fill keeps every bin's phase
/// range; more, and the four bins' that the centre needs are found
/// afterwards by [`phases`]. Keeping them costs about half a nanosecond a
/// point; finding them, about four searches of the elapsed times an orbit,
/// a hundred nanoseconds (measured on 13 TESS sectors and four years of
/// Kepler: the two are equal near 220 points an orbit).
const LAZY_POINTS: f64 = 256.0;

/// Each point into its bin, its phase from `offset`: the bins' sums and
/// counts, and with `RANGES` their phase ranges.
// Not inlined: each of its forms is then compiled on its own (inlined into
// the search, they ran 10 to 30% slower).
#[inline(never)]
fn fill<const RANGES: bool>(
    bins: &mut [Bin],
    ranges: &mut [(f64, f64)],
    weighted: &Weighted,
    period: f64,
    bin: f64,
    offset: impl Fn(f64, f64) -> f64,
) {
    let last = bins.len() - 1;
    let points = weighted
        .elapsed
        .iter()
        .zip(&weighted.weights)
        .zip(&weighted.centred);
    for ((&(s, e), &weight), &centred) in points {
        let (k, phase) = bin_of(offset(s, e), period, bin, last);
        let slot = &mut bins[k];
        slot.weight += weight;
        slot.flux += centred;
        slot.count += 1;
        if RANGES {
            // Plain comparisons: a phase is never NaN.
            let (low, high) = &mut ranges[k];
            *low = if phase < *low { phase } else { *low };
            *high = if phase > *high { phase } else { *high };
        }
    }
}

/// A point's bin and its phase in the orbit, from its offset after the
/// first point's phase. The same arithmetic for the fill and for
/// [`phases`], so both put every point in the same bin.
#[inline(always)]
pub(crate) fn bin_of(offset: f64, period: f64, bin: f64, last: usize) -> (usize, f64) {
    let phase = if offset < 0.0 {
        offset + period
    } else {
        offset
    };
    // A phase that rounds up to m bins belongs in the last. (Through `u32`,
    // whose conversion is shorter: there are at most a million.)
    (((phase / bin) as u32 as usize).min(last), phase)
}

/// The phases the centre needs, of the four bins with points around the
/// best box's edges, `[first, last, next, previous]` (indices run on round
/// the orbit from `start + m`): the box's first and last, and the nearest
/// outside it after and before. The lowest phase in `first` and `next`, the
/// highest in `last` and `previous`.
///
/// Bin `j` holds the points about `(j - m) b .. (j - m + 1) b` after the
/// start of an orbit, off by a few roundings of the times and the orbits. So
/// in each orbit only the points in that range, widened by those roundings
/// either way, are looked at, found from the elapsed times' cells; each is
/// put in its bin by [`bin_of`] with the fill's `offset`, so it is in the bin
/// the fill put it in, and only the four bins' points count. A point looked
/// at twice changes nothing: the phases are a least and a greatest. Used
/// only with at least [`LAZY_POINTS`] points an orbit, so the orbits are few.
// Not inlined: it runs once a period, and inlined it changes how the fill's
// loop is compiled.
#[inline(never)]
fn phases(
    edges: &[usize; 4],
    weighted: &Weighted,
    period: f64,
    bin: f64,
    m: usize,
    offset: impl Fn(f64, f64) -> f64,
) -> [f64; 4] {
    let elapsed = &weighted.elapsed;
    let last_time = elapsed.last().map_or(0.0, |&(s, _)| s);
    let margin = 64.0 * f64::EPSILON * (last_time + 4.0 * period);
    let ranges = edges.map(|j| {
        (
            (j as f64 - m as f64) * bin - margin,
            (j as f64 + 1.0 - m as f64) * bin + margin,
        )
    });
    // Every orbit in which one of the ranges meets the times.
    let (low, high) = (ranges[3].0, ranges[2].1);
    let orbits =
        (-high / period).floor() as i64 - 1..=((last_time - low) / period).ceil() as i64 + 1;
    let targets = edges.map(|j| j % m);
    let mut phases = [
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ];
    let mut look = |&(s, e): &(f64, f64)| {
        let (k, phase) = bin_of(offset(s, e), period, bin, m - 1);
        // Plain comparisons: a phase is never NaN.
        for (edge, (&target, kept)) in targets.iter().zip(&mut phases).enumerate() {
            let better = if edge % 2 == 0 {
                phase < *kept
            } else {
                phase > *kept
            };
            if k == target && better {
                *kept = phase;
            }
        }
    };
    for orbit in orbits {
        let at = orbit as f64 * period;
        for &(from, to) in &ranges {
            elapsed[weighted.cells.first_from(elapsed, at + from)..]
                .iter()
                .take_while(|&&(s, _)| s < at + to)
                .for_each(&mut look);
        }
    }
    phases
}

/// The middle of the range of times at which a box `length` bins long holds
/// the points of bins `start..start + length`, in days after the first time.
///
/// Its left edge `a` holds them, with the box open at both ends, when
/// `previous <= a < first` and `last < a + D <= next`, `first` and `last`
/// being the box's first and last points and `previous` and `next` its
/// neighbours outside. The bins themselves hold a point on the box's left
/// edge, so the box at the bins' own edges might not: the first point always
/// sits on the left edge of a box that starts at bin 0. The range is never
/// empty: `previous` and `last - D` are below the bins' left edge, and
/// `first` and `next - D` not below it. A box with a dip holds some points and
/// leaves some out, so all four exist.
pub(crate) fn centre(
    edges: &[usize; 4],
    phases: &[f64; 4],
    span: f64,
    period: f64,
    m: usize,
) -> f64 {
    // Each phase continued round the orbit to its bin's index, in days.
    let [first, last, next, previous] =
        [0, 1, 2, 3].map(|k| phases[k] + (edges[k] / m) as f64 * period);
    let lower = previous.max(last - span);
    let upper = first.min(next - span);
    0.5 * (lower + upper) + 0.5 * span - period
}

/// Running totals over the points in time order, `totals[i]` over the first
/// `i`: the sums of any run of points are the difference of two, for
/// [`box_at`].
#[cfg(any(feature = "cuda", test))]
pub(crate) fn point_totals(weighted: &Weighted) -> Vec<Total> {
    let mut total = Total::default();
    let mut totals = Vec::with_capacity(weighted.weights.len() + 1);
    totals.push(total);
    for (&weight, &centred) in weighted.weights.iter().zip(&weighted.centred) {
        total.weight.add(weight);
        total.flux.add(centred);
        totals.push(total);
    }
    totals
}

/// The box of [`best_binned`]'s bins at `period` that starts at bin `start`
/// and is the `duration`th duration long, built from the points it holds
/// rather than from every bin: in each orbit its points, and those between
/// it and the next orbit's, are a run of points in time order, whose ends
/// are found near the box's edges. `points` are [`point_totals`].
///
/// The box holds the points `best_binned` would put in its bins, and its
/// middle is the same to the bit; its fit's sums are over the same points,
/// added in another order. `None` where this is not done: off the short
/// path, with fewer than 8 bins, or a box holding no point or every point.
#[cfg(any(feature = "cuda", test))]
pub(crate) fn box_at(
    weighted: &Weighted,
    points: &[Total],
    period: f64,
    durations: &[f64],
    oversample: usize,
    duration: usize,
    start: usize,
) -> Result<Option<BinnedBox>, BlsError> {
    let m = (bin_count(period, durations, oversample) as usize).max(1);
    let bin = period / m as f64;
    let length = (durations[duration] / bin).round() as usize;
    let elapsed = &weighted.elapsed;
    let n = elapsed.len();
    let last = elapsed.last().map_or(0.0, |&(s, _)| s);
    let at = AtPeriod::new(period);
    if m < 8 || start >= m || length == 0 || length >= m || !at.short_up_to(last) {
        return Ok(None);
    }
    let bin_at = |i: usize| {
        let (s, e) = elapsed[i];
        bin_of(at.offset_after_start_short(s, e), period, bin, m - 1)
    };
    // Points further than this from a time are on its side whatever the
    // roundings of their phases ([`phases`]' margin).
    let margin = 64.0 * f64::EPSILON * (last + 4.0 * period);
    // The first point at or past bin `edge` (counted on round the orbits)
    // in orbit `orbit`. Only points within the margin of the edge's time
    // are put in their bins, and theirs are next to it, so less than half
    // an orbit on from it is past it.
    let first_at = |orbit: i64, edge: usize| {
        let x = orbit as f64 * period + edge as f64 * bin;
        let mut i = weighted.cells.first_from(elapsed, x - margin);
        while i < n && elapsed[i].0 < x + margin && (bin_at(i).0 + m - edge % m) % m >= m / 2 {
            i += 1;
        }
        i
    };

    // Every orbit whose box or the gap after it holds a point: from before
    // the first time (a box ends less than two orbits on from the start of
    // its orbit) to past the last.
    let orbits = -3..=(last / period).ceil() as i64 + 1;
    use crate::sums::{Sum, between};
    let (mut inside, mut outside) = ([Sum::default(); 2], [Sum::default(); 2]);
    let add = |sums: &mut [Sum; 2], from: usize, to: usize| {
        sums[0].add(between(points[from].weight, points[to].weight));
        sums[1].add(between(points[from].flux, points[to].flux));
    };
    let mut n_in = 0;
    // Each edge bin (counted from `start`) and its phase: the box's first
    // and last bins with points, and the nearest with points after it and
    // before it.
    let mut first: Option<(usize, f64)> = None;
    let mut last_in: Option<(usize, f64)> = None;
    let mut next: Option<(usize, f64)> = None;
    let mut previous: Option<(usize, f64)> = None;
    let from_start = |i: usize| {
        let (k, phase) = bin_at(i);
        ((k + m - start) % m, phase)
    };
    let lowest = |kept: &mut Option<(usize, f64)>, found: (usize, f64)| {
        if kept.is_none_or(|k| found.0 < k.0 || (found.0 == k.0 && found.1 < k.1)) {
            *kept = Some(found);
        }
    };
    let highest = |kept: &mut Option<(usize, f64)>, found: (usize, f64)| {
        if kept.is_none_or(|k| found.0 > k.0 || (found.0 == k.0 && found.1 > k.1)) {
            *kept = Some(found);
        }
    };
    let mut after = 0;
    for orbit in orbits {
        let (begin, end) = (first_at(orbit, start), first_at(orbit, start + length));
        // The gap before this box, after the last.
        if after < begin {
            add(&mut outside, after, begin);
            lowest(&mut next, from_start(after));
            highest(&mut previous, from_start(begin - 1));
        }
        if begin < end {
            add(&mut inside, begin, end);
            n_in += end - begin;
            lowest(&mut first, from_start(begin));
            highest(&mut last_in, from_start(end - 1));
        }
        after = end;
    }
    if after < n {
        add(&mut outside, after, n);
        lowest(&mut next, from_start(after));
        highest(&mut previous, from_start(n - 1));
    }
    let (Some(first), Some(last_in), Some(next), Some(previous)) = (first, last_in, next, previous)
    else {
        return Ok(None);
    };
    let fit = BoxFit::from_sums(
        weighted.mean,
        inside.map(Sum::value),
        outside.map(Sum::value),
        n_in,
    )?;
    let from = start + m;
    let edges = [
        from + first.0,
        from + last_in.0,
        from + next.0,
        from + previous.0 - m,
    ];
    let phases = [first.1, last_in.1, next.1, previous.1];
    Ok(Some(BinnedBox {
        fit,
        bin,
        start,
        bins: length,
        duration,
        centre: centre(&edges, &phases, length as f64 * bin, period, m),
    }))
}

/// How many starts are scored at every duration before the next: their
/// totals, those one orbit on, and the longest box's past them fit in a
/// core's cache (32 bytes each).
pub(crate) const WINDOW: usize = 4096;

/// Box starts to a block, for the bounds.
const STARTS: usize = 32;
/// Totals to a grain of the bounds' least and greatest `g`.
const GRAIN: usize = 8;

/// Upper bounds on the scores of blocks of boxes in a window, as in the
/// exact search: with `F` and `V` the running totals of centred flux and
/// weight, and `S` and `W` the orbit's, a box from `i` to `k` has `u = g_k -
/// g_i`, `g = F - (S / W) V`, and its score is `u^2 / D`, `D = W_in (W -
/// W_in) / W`, when `u < 0` (times `W / (2 W_out)` for astropy's
/// likelihood). Over a block of starts at one length, `-u` is at most the
/// greatest `g` at the starts less the least at the ends, and `W_in` lies
/// between the least end's total less the greatest start's and the greatest
/// end's less the least start's (the totals only grow); `D` is concave in
/// `W_in`, so its least is at one of the two. The margins are far above the
/// rounding of `g`, `W_in` and the statistic, so a bound is never below a
/// score the statistic computes.
#[derive(Debug, Clone, Default)]
pub(crate) struct Bounds {
    g: Vec<f64>,
    weight: Vec<f64>,
    /// The least and greatest `g` of each grain.
    least: Vec<f64>,
    greatest: Vec<f64>,
    margin_u: f64,
    margin_w: f64,
    whole: f64,
    likelihood: bool,
}

impl Bounds {
    /// For the boxes starting in `near` (the running totals from a window's
    /// first start), with the orbit's sums `whole`.
    pub(crate) fn prepare(&mut self, near: &[Total], whole: &Orbit) {
        let w = whole.weight();
        let ratio = whole.flux() / w;
        self.g.clear();
        self.weight.clear();
        let mut largest_flux = 0.0_f64;
        for total in near {
            let (v, f) = (total.weight.value(), total.flux.value());
            self.weight.push(v);
            self.g.push(f - ratio * v);
            largest_flux = largest_flux.max(f.abs());
        }
        self.least.clear();
        self.greatest.clear();
        for grain in self.g.chunks(GRAIN) {
            self.least
                .push(grain.iter().copied().fold(f64::INFINITY, f64::min));
            self.greatest
                .push(grain.iter().copied().fold(f64::NEG_INFINITY, f64::max));
        }
        let largest_weight = self.weight.last().copied().unwrap_or(0.0);
        self.margin_u = 64.0
            * f64::EPSILON
            * (largest_flux + ratio.abs() * largest_weight + whole.flux().abs());
        self.margin_w = 64.0 * f64::EPSILON * largest_weight;
        self.whole = w;
        self.likelihood = whole.likelihood();
    }

    /// An upper bound on the scores of the boxes `length` long starting at
    /// `from..to`. Infinite (never skipped) when the weight in a box could
    /// reach zero or the whole orbit, and NaN on overflow, which is never
    /// skipped either.
    pub(crate) fn bound(&self, from: usize, to: usize, length: usize) -> f64 {
        let over = |values: &[f64], a: usize, b: usize, pick: fn(f64, f64) -> f64, start: f64| {
            values[a / GRAIN..=(b - 1) / GRAIN]
                .iter()
                .copied()
                .fold(start, pick)
        };
        let starts = over(&self.greatest, from, to, f64::max, f64::NEG_INFINITY);
        let ends = over(
            &self.least,
            from + length,
            to + length,
            f64::min,
            f64::INFINITY,
        );
        let fall = starts - ends + self.margin_u;
        if fall <= 0.0 {
            return 0.0;
        }
        let w = self.whole;
        let shortest = (self.weight[from + length] - self.weight[to - 1]) - self.margin_w;
        let longest = (self.weight[to - 1 + length] - self.weight[from]) + self.margin_w;
        if shortest <= 0.0 || longest >= w {
            return f64::INFINITY;
        }
        let d = |x: f64| x * ((w - x) / w);
        let least_d = d(shortest).min(d(longest)) * (1.0 - 64.0 * f64::EPSILON);
        let mut bound = fall * (fall / least_d);
        if self.likelihood {
            bound *= 0.5 * w / (w - longest);
        }
        bound * (1.0 + 64.0 * f64::EPSILON)
    }
}

/// The number of bins at `period` days: the fewest no wider than
/// `min(durations) / oversample`.
pub(crate) fn bin_count(period: f64, durations: &[f64], oversample: usize) -> f64 {
    let shortest = durations.iter().copied().fold(f64::INFINITY, f64::min);
    (period / (shortest / oversample as f64)).ceil()
}

/// One bin's plain sums of the weights and centred fluxes, and its number of
/// points. (24 bytes: at 16, filling bins about twenty apart, as Kepler's
/// long cadence does at the shortest durations, ran 70% slower.)
#[derive(Debug, Clone, Copy, Default)]
struct Bin {
    weight: f64,
    flux: f64,
    count: usize,
}

//! The search with its binned step on an NVIDIA GPU: [`CudaBls`].

use std::{
    fmt,
    sync::{Arc, mpsc},
    thread,
};

use cudarc::{
    driver::{
        CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
        sys::{CUdevice_attribute_enum, CUfunction_attribute_enum},
    },
    nvrtc::{CompileOptions, compile_ptx_with_opts},
};

use crate::{
    Bls, BlsError, BlsResult, Durations, Objective, Periods, SearchOptions,
    binned::{BinnedBox, Scratch, best_binned, bin_count, box_at, point_totals},
    box_fit::AtPeriod,
    parallel::on_threads,
    search::prepare,
    top::make_top_exact,
    totals::Total,
};

/// The kernel's source, compiled by NVRTC when a [`CudaBls`] is made.
const SOURCE: &str = include_str!("cuda/binned.cu");
/// Threads a block, one block a period.
const THREADS: u32 = 256;
/// Bins of a tile, unless the longest box needs more.
const TILE: usize = 2048;
/// A launch's work, in points binned (about a tenth of a second): Windows
/// stops a display GPU's kernel that runs for two.
const LAUNCH_WORK: f64 = 2e9;
/// The narrowest bin, in days, whose points the integers put in their bins:
/// times are counted in 2^-40 days, so a point's place in a bin is then
/// known to within 2^-26 bins, inside the 2^-24 within which the CPU's
/// arithmetic decides.
const NARROWEST_BIN: f64 = 1.0 / 16384.0;
/// The latest elapsed time, in days, held in units of 2^-40 days by a `u64`
/// with room to spare (2^22 days, 11,000 years).
const LATEST: f64 = 4_194_304.0;

/// An error from a search on the GPU.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum CudaError {
    /// The search's own error, as [`Bls::search`] gives.
    Search(BlsError),
    /// No GPU, its driver failed, or the kernel did not compile: the
    /// driver's or compiler's message.
    Device(String),
}

impl fmt::Display for CudaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Search(error) => error.fmt(f),
            Self::Device(message) => write!(f, "CUDA: {message}"),
        }
    }
}

impl std::error::Error for CudaError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Search(error) => Some(error),
            Self::Device(_) => None,
        }
    }
}

impl From<BlsError> for CudaError {
    fn from(error: BlsError) -> Self {
        Self::Search(error)
    }
}

fn device(error: impl fmt::Display) -> CudaError {
    CudaError::Device(error.to_string())
}

/// A light curve ready to search with the binned step on an NVIDIA GPU: the
/// compiled kernel and the points, on the GPU, made once.
///
/// The results are [`Bls::search`]'s, with the binned step's power and
/// boxes known to within a rounding error of the sums at most periods:
///
/// - The GPU finds each period's best box from the same bins as the CPU
///   (every point is put in the bin the CPU puts it in), with the bins'
///   sums held exactly in 64-bit fixed point. The box is then built on the
///   CPU from the points it holds: its middle is the CPU's to the bit, and
///   its fit's sums are over the same points, added in another order.
/// - At every period whose power could be among the highest peaks or the
///   highest power, and the periods next to them, the binned step is run on
///   the CPU, so the peaks, their refinement and [`BlsResult::best`] are
///   [`Bls::search`]'s to the bit.
///
/// The power elsewhere differs from the CPU's by about 1e-14 of itself
/// (4e-14 at most, measured on 1 and 13 TESS sectors and 4 years of Kepler),
/// and where two boxes score within that of each other the other may be
/// reported. Periods the GPU does not take (bins narrower than 2^-14 days,
/// more than 2^25 orbits, or a box longer than its shared memory holds) are
/// searched on the CPU.
///
/// # Examples
///
/// ```no_run
/// use rustellar_bls::{Bls, CudaBls, Durations, Periods, SearchOptions};
/// use rustellar_core::LightCurve;
///
/// let time: Vec<f64> = (0..960).map(|i| f64::from(i) / 48.0).collect();
/// let flux = time
///     .iter()
///     .map(|t| if (t % 3.0 - 1.0).abs() < 0.075 { 0.995 } else { 1.0 })
///     .collect();
/// let lc = LightCurve::from_btjd(time, flux, vec![1e-3; 960])?;
///
/// let gpu = CudaBls::new(&Bls::new(&lc)?, 0)?;
/// let periods: Vec<f64> = (0..200).map(|k| 2.0 + 0.01 * f64::from(k)).collect();
/// let result = gpu.search(
///     &Periods::from_days(&periods)?,
///     &Durations::from_days(&[0.1, 0.15, 0.2])?,
///     &SearchOptions::default(),
/// )?;
/// let best = result.best().expect("there is a dip");
/// assert!((best.transit.period().as_days() - 3.0).abs() < 0.006);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct CudaBls {
    bls: Bls,
    /// Running totals over the points, for the boxes built on the CPU.
    points: Vec<Total>,
    stream: Arc<CudaStream>,
    function: CudaFunction,
    /// The most bins a tile can hold.
    most_bins: usize,
    /// Whether the times fit the integers (see [`LATEST`]).
    fits: bool,
    elapsed: [CudaSlice<f64>; 2],
    fixed_times: CudaSlice<u64>,
    weights: CudaSlice<i64>,
    fluxes: CudaSlice<i64>,
    quanta: [f64; 2],
    totals: [i64; 2],
}

impl fmt::Debug for CudaBls {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CudaBls")
            .field("points", &self.points.len().saturating_sub(1))
            .field("most_bins", &self.most_bins)
            .finish_non_exhaustive()
    }
}

/// The power of two by which `values`, as whole numbers of it, add up to
/// less than 2^62 in absolute value, however they are grouped.
fn quantum(values: &[f64]) -> f64 {
    let total: f64 = values.iter().map(|v| v.abs()).sum();
    if total.is_nan() || total <= 0.0 {
        return 1.0;
    }
    // A hundredth to spare for the sum's rounding and each value's.
    2f64.powi((1.01 * total).log2().ceil() as i32 - 62)
}

/// `G` and its shift for period `p`: `2^c / p` rounded, below 2^64, so that
/// bits `shift..shift + 64` of `S G`, `S` a time in units of 2^-40 days, are
/// the fraction of an orbit. `None` if the shift is out of range.
fn reciprocal(p: f64) -> Option<(u64, i32)> {
    let bits = p.to_bits();
    let exponent = ((bits >> 52) & 0x7ff) as i32;
    if exponent == 0 || exponent == 0x7ff {
        return None;
    }
    // p = mantissa 2^power.
    let mantissa = u128::from((bits & ((1 << 52) - 1)) | (1 << 52));
    let power = exponent - 1075;
    for k in [116, 115] {
        let g = (1u128 << k) / mantissa + u128::from(2 * ((1u128 << k) % mantissa) >= mantissa);
        if g < 1 << 64 {
            let shift = k + power - 24;
            return (1..64).contains(&shift).then_some((g as u64, shift));
        }
    }
    None
}

/// `work` on each item, item `k` on thread `k % threads`, the outcomes in
/// the items' order. A box takes work in proportion to the orbits, so
/// neighbouring periods cost about the same and a contiguous chunk of the
/// shortest would take far longer than the rest.
fn interleaved<I: Sync, T: Send>(
    items: &[I],
    threads: usize,
    work: &(impl Fn(&I) -> T + Sync),
) -> Vec<T> {
    let lanes = threads.clamp(1, items.len().max(1));
    let mut parts: Vec<std::vec::IntoIter<T>> = on_threads(0..lanes, &|lane: usize| {
        items
            .iter()
            .skip(lane)
            .step_by(lanes)
            .map(work)
            .collect::<Vec<T>>()
    })
    .into_iter()
    .map(Vec::into_iter)
    .collect();
    (0..items.len())
        .map(|k| parts[k % lanes].next().expect("every item worked on"))
        .collect()
}

/// One period's work for the GPU.
struct Job {
    period: usize,
    bins: usize,
    constants: [f64; 5],
    reciprocal: u64,
    shift: i32,
    lengths: Vec<i32>,
    work: f64,
}

impl CudaBls {
    /// The kernel compiled for GPU `device` (0 for the first) and `bls`'s
    /// points copied to it.
    ///
    /// # Errors
    ///
    /// [`CudaError::Device`] if there is no such GPU, the CUDA driver or
    /// NVRTC library cannot be loaded, or the kernel does not compile.
    pub fn new(bls: &Bls, device: usize) -> Result<Self, CudaError> {
        let context = CudaContext::new(device).map_err(self::device)?;
        let options = CompileOptions {
            options: vec!["--fmad=false".into(), format!("-DTHREADS={THREADS}")],
            ..CompileOptions::default()
        };
        let ptx = compile_ptx_with_opts(SOURCE, options).map_err(self::device)?;
        let module = context.load_module(ptx).map_err(self::device)?;
        let function = module.load_function("binned").map_err(self::device)?;
        let shared = context
            .attribute(
                CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
            )
            .map_err(self::device)?;
        let fixed = function
            .get_attribute(CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES)
            .map_err(self::device)?;
        let dynamic = (shared - fixed).max(0);
        function
            .set_attribute(
                CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                dynamic,
            )
            .map_err(self::device)?;
        let most_bins = (dynamic as usize / 16).saturating_sub(1);

        let weighted = &bls.weighted;
        let (s, e): (Vec<f64>, Vec<f64>) = weighted.elapsed.iter().copied().unzip();
        let last = s.last().copied().unwrap_or(0.0);
        const UNIT: f64 = 1_099_511_627_776.0; // 2^40
        let fixed_times: Vec<u64> = s
            .iter()
            .zip(&e)
            .map(|(&s, &e)| ((s * UNIT).round() + (e * UNIT).round()).max(0.0) as u64)
            .collect();
        let quanta = [quantum(&weighted.weights), quantum(&weighted.centred)];
        let whole = |values: &[f64], q: f64| -> Vec<i64> {
            values.iter().map(|v| (v / q).round() as i64).collect()
        };
        let (weights, fluxes) = (
            whole(&weighted.weights, quanta[0]),
            whole(&weighted.centred, quanta[1]),
        );
        let totals = [weights.iter().sum(), fluxes.iter().sum()];
        let stream = context.default_stream();
        Ok(Self {
            bls: bls.clone(),
            points: point_totals(weighted),
            most_bins,
            fits: last < LATEST,
            elapsed: [
                stream.clone_htod(&s).map_err(self::device)?,
                stream.clone_htod(&e).map_err(self::device)?,
            ],
            fixed_times: stream.clone_htod(&fixed_times).map_err(self::device)?,
            weights: stream.clone_htod(&weights).map_err(self::device)?,
            fluxes: stream.clone_htod(&fluxes).map_err(self::device)?,
            quanta,
            totals,
            function,
            stream,
        })
    }

    /// As [`Bls::search`], with the binned step on the GPU.
    ///
    /// # Errors
    ///
    /// [`CudaError::Search`] for [`Bls::search`]'s errors, and
    /// [`CudaError::Device`] if the GPU fails.
    pub fn search(
        &self,
        periods: &Periods,
        durations: &Durations,
        options: &SearchOptions,
    ) -> Result<BlsResult, CudaError> {
        let mut result = BlsResult::default();
        self.search_into(periods, durations, options, &mut result)?;
        Ok(result)
    }

    /// As [`Bls::search_into`], with the binned step on the GPU.
    ///
    /// # Errors
    ///
    /// As [`CudaBls::search`].
    pub fn search_into(
        &self,
        periods: &Periods,
        durations: &Durations,
        options: &SearchOptions,
        result: &mut BlsResult,
    ) -> Result<(), CudaError> {
        let durations = prepare(periods, durations, options, result)?;
        let oversample = options.oversample.get();
        let objective = options.objective;
        let threads = options.threads.get();
        let periods = periods.as_slice();
        let days: Vec<f64> = periods.iter().map(|p| p.as_days()).collect();
        let n = days.len();
        let points = self.bls.weighted.elapsed.len() as f64;
        let last = self.bls.weighted.elapsed.last().map_or(0.0, |&(s, _)| s);

        // Each period's bins and lengths; those the GPU does not take are
        // searched on the CPU, and so known exactly.
        let mut jobs = Vec::new();
        let mut known = vec![false; n];
        let lengths_at = |m: usize, bin: f64| -> Vec<i32> {
            durations
                .iter()
                .map(|&d| {
                    let length = (d / bin).round() as usize;
                    if length < m { length as i32 } else { -1 }
                })
                .collect()
        };
        let longest_box = days
            .iter()
            .map(|&p| {
                let m = (bin_count(p, &durations, oversample) as usize).max(1);
                lengths_at(m, p / m as f64)
                    .into_iter()
                    .max()
                    .unwrap_or(0)
                    .max(0) as usize
            })
            .max()
            .unwrap_or(0);
        let cap = TILE.max(longest_box + 1024).min(self.most_bins);
        for (period, &p) in days.iter().enumerate() {
            let m = (bin_count(p, &durations, oversample) as usize).max(1);
            let bin = p / m as f64;
            let at = AtPeriod::new(p);
            let lengths = lengths_at(m, bin);
            let longest = lengths.iter().copied().max().unwrap_or(0).max(0) as usize;
            let taken = self.fits
                && at.short_up_to(last)
                && bin >= NARROWEST_BIN
                && m >= 8
                && m < i32::MAX as usize / 2
                && longest + 64 <= cap;
            match reciprocal(p).filter(|_| taken) {
                Some((reciprocal, shift)) => {
                    let (p_hi, p_lo) = at.halves();
                    let tile = if m + longest <= cap { m } else { cap - longest };
                    let visits = points * (m.div_ceil(tile) * (tile + longest)) as f64 / m as f64;
                    jobs.push(Job {
                        period,
                        bins: m,
                        constants: [p, 1.0 / p, p_hi, p_lo, bin],
                        reciprocal,
                        shift,
                        lengths,
                        work: visits.min(points * m.div_ceil(tile) as f64)
                            + (m * durations.len()) as f64 / 4.0,
                    });
                }
                None => known[period] = true,
            }
        }

        // The GPU's best box at each of its periods, built on the CPU.
        // The GPU's best box at each of its periods, a launch at a time on a
        // thread of its own, each launch's built on the CPU from its points
        // while the GPU runs the next.
        let build = |&(job, &(duration, start)): &(&Job, &(i32, i32))| {
            if duration < 0 {
                return Ok(Some(None));
            }
            box_at(
                &self.bls.weighted,
                &self.points,
                days[job.period],
                &durations,
                oversample,
                duration as usize,
                start as usize,
            )
            .map(|b| b.map(Some))
        };
        let (sender, receiver) = mpsc::channel::<(usize, Vec<(i32, i32)>)>();
        let mut built = Vec::with_capacity(jobs.len());
        thread::scope(|scope| {
            // The sender goes with the GPU's thread, so the channel closes
            // when it ends.
            let gpu = scope.spawn(|| {
                let sender = sender;
                self.launch(&jobs, durations.len(), cap, objective, &sender)
            });
            for (from, found) in &receiver {
                let pairs: Vec<(&Job, &(i32, i32))> = jobs[from..].iter().zip(&found).collect();
                built.extend(interleaved(&pairs, threads, &build));
            }
            gpu.join().expect("the GPU's thread does not panic")
        })?;
        let BlsResult { power, boxes, .. } = result;
        let mut put = |period: usize, found: Option<BinnedBox>| {
            if let Some(found) = found {
                power[period] = objective.of(&found.fit);
                boxes[period] = Some(self.bls.best_box(
                    periods[period],
                    found.bin * found.bins as f64,
                    found.centre,
                    found.fit,
                ));
            }
        };
        let mut first_error = None;
        for (job, outcome) in jobs.iter().zip(built) {
            match outcome {
                Ok(Some(found)) => put(job.period, found),
                // Not built from its points: on the CPU.
                Ok(None) => known[job.period] = true,
                Err(error) => {
                    first_error.get_or_insert((job.period, error));
                }
            }
        }
        let cpu: Vec<usize> = (0..n).filter(|&k| known[k]).collect();
        let exact =
            |todo: &[usize]| self.exact_at(todo, &days, &durations, oversample, objective, threads);
        for (&period, outcome) in cpu.iter().zip(exact(&cpu)) {
            match outcome {
                Ok(found) => put(period, found),
                Err(error) => {
                    if first_error.is_none_or(|(at, _)| period < at) {
                        first_error = Some((period, error));
                    }
                }
            }
        }
        if let Some((_, error)) = first_error {
            return Err(error.into());
        }

        // The top made exact, so the peaks and best box are the CPU's.
        make_top_exact(power, &known, options.refine_peaks, &mut |todo, power| {
            for (&period, outcome) in todo.iter().zip(exact(todo)) {
                let found = outcome?;
                power[period] = 0.0;
                boxes[period] = None;
                if let Some(found) = found {
                    power[period] = objective.of(&found.fit);
                    boxes[period] = Some(self.bls.best_box(
                        periods[period],
                        found.bin * found.bins as f64,
                        found.centre,
                        found.fit,
                    ));
                }
            }
            Ok(())
        })?;
        Ok(self.bls.finish(periods, &durations, options, result)?)
    }

    /// The CPU's binned search at each of `todo`, on up to `threads`
    /// threads.
    fn exact_at(
        &self,
        todo: &[usize],
        days: &[f64],
        durations: &[f64],
        oversample: usize,
        objective: Objective,
        threads: usize,
    ) -> Vec<Result<Option<BinnedBox>, BlsError>> {
        if todo.is_empty() {
            return Vec::new();
        }
        let most = todo
            .iter()
            .map(|&k| bin_count(days[k], durations, oversample) as usize)
            .max()
            .unwrap_or(1);
        let chunk = todo.len().div_ceil(threads.clamp(1, todo.len()));
        on_threads(todo.chunks(chunk), &|chunk: &[usize]| {
            let mut scratch = Scratch::with_capacity(most, durations.len());
            chunk
                .iter()
                .map(|&k| {
                    best_binned(
                        &self.bls.weighted,
                        days[k],
                        durations,
                        oversample,
                        objective,
                        &mut scratch,
                    )
                })
                .collect::<Vec<_>>()
        })
        .into_iter()
        .flatten()
        .collect()
    }

    /// The GPU's best box at each job's period, `(duration, start)`, sent a
    /// launch at a time with the index of its first job; duration -1 where
    /// no box has a dip.
    fn launch(
        &self,
        jobs: &[Job],
        durations: usize,
        cap: usize,
        objective: Objective,
        sender: &mpsc::Sender<(usize, Vec<(i32, i32)>)>,
    ) -> Result<(), CudaError> {
        let stream = &self.stream;
        let n = self.points.len() as i32 - 1;
        let (nd, cap_i, likelihood) = (
            durations as i32,
            cap as i32,
            i32::from(objective == Objective::AstropyLikelihood),
        );
        let shared = (2 * (cap + 1) * 8) as u32;
        let mut from = 0;
        while from < jobs.len() {
            let mut to = from;
            let mut work = 0.0;
            while to < jobs.len() && (to == from || work + jobs[to].work <= LAUNCH_WORK) {
                work += jobs[to].work;
                to += 1;
            }
            let batch = &jobs[from..to];
            let constants: Vec<f64> = batch.iter().flat_map(|j| j.constants).collect();
            let bins: Vec<i32> = batch.iter().map(|j| j.bins as i32).collect();
            let reciprocals: Vec<u64> = batch.iter().map(|j| j.reciprocal).collect();
            let shifts: Vec<i32> = batch.iter().map(|j| j.shift).collect();
            let lengths: Vec<i32> = batch
                .iter()
                .flat_map(|j| j.lengths.iter().copied())
                .collect();
            let upload = |e| device(e);
            let constants = stream.clone_htod(&constants).map_err(upload)?;
            let bins = stream.clone_htod(&bins).map_err(upload)?;
            let reciprocals = stream.clone_htod(&reciprocals).map_err(upload)?;
            let shifts = stream.clone_htod(&shifts).map_err(upload)?;
            let lengths = stream.clone_htod(&lengths).map_err(upload)?;
            let mut scores = stream.alloc_zeros::<f64>(batch.len()).map_err(upload)?;
            let mut chosen = stream.alloc_zeros::<i32>(batch.len()).map_err(upload)?;
            let mut starts = stream.alloc_zeros::<i32>(batch.len()).map_err(upload)?;
            let config = LaunchConfig {
                grid_dim: (batch.len() as u32, 1, 1),
                block_dim: (THREADS, 1, 1),
                shared_mem_bytes: shared,
            };
            let mut launch = stream.launch_builder(&self.function);
            launch
                .arg(&self.elapsed[0])
                .arg(&self.elapsed[1])
                .arg(&self.fixed_times)
                .arg(&self.weights)
                .arg(&self.fluxes)
                .arg(&n)
                .arg(&self.quanta[0])
                .arg(&self.quanta[1])
                .arg(&self.totals[0])
                .arg(&self.totals[1])
                .arg(&constants)
                .arg(&bins)
                .arg(&reciprocals)
                .arg(&shifts)
                .arg(&lengths)
                .arg(&nd)
                .arg(&cap_i)
                .arg(&likelihood)
                .arg(&mut scores)
                .arg(&mut chosen)
                .arg(&mut starts);
            // SAFETY: the arguments are the kernel's, in its order and of its
            // types; every array holds what the kernel reads of it (a period's
            // five constants and its lengths for each of the `nd` durations,
            // one entry a period otherwise, and the `n` points), and the
            // outputs one entry for each of the launch's blocks.
            unsafe { launch.launch(config) }.map_err(device)?;
            let chosen = stream.clone_dtoh(&chosen).map_err(device)?;
            let starts = stream.clone_dtoh(&starts).map_err(device)?;
            // The receiver is only gone once the search has stopped.
            let _ = sender.send((from, chosen.into_iter().zip(starts).collect()));
            from = to;
        }
        Ok(())
    }
}

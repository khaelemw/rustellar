//! The search with its binned step and its refinement of the peaks on an
//! NVIDIA GPU: [`Gpu`], chosen by [`Device`](crate::Device).

use std::{
    fmt,
    sync::{Arc, Mutex, OnceLock, PoisonError, mpsc},
    thread,
};

use cudarc::{
    driver::{
        CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DriverError, LaunchConfig,
        PushKernelArg,
        sys::{CUdevice_attribute_enum, CUfunction_attribute_enum},
    },
    nvrtc::{CompileOptions, compile_ptx_with_opts},
};

use crate::{
    Bls, BlsError, BlsResult, BoxFit, Durations, Objective, Periods, SearchOptions,
    binned::{BinnedBox, Scratch, best_binned, bin_count, box_at, point_totals},
    box_fit::AtPeriod,
    exact::Run,
    parallel::on_threads,
    refine::{BestBetween, BestOf, Folds},
    search::prepare,
    sums::Sum,
    top::make_top_exact,
    totals::Total,
};

/// The kernels' source, compiled by NVRTC when the first [`Gpu`] is made on a
/// GPU: the
/// binned step's, and the folds' for refining the peaks.
const SOURCE: &str = concat!(
    include_str!("cuda/binned.cu"),
    include_str!("cuda/refine.cu")
);
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
#[derive(Debug)]
pub(crate) enum CudaError {
    /// The search's own error, as [`Bls::search`] gives.
    Search(BlsError),
    /// No GPU, its driver failed, or the kernel did not compile: the
    /// driver's or compiler's message.
    Device(String),
}

impl From<BlsError> for CudaError {
    fn from(error: BlsError) -> Self {
        Self::Search(error)
    }
}

/// The GPUs a [`Bls`] has set up, by index; `None` for one that cannot be.
pub(crate) type GpuCache = Arc<Mutex<Vec<(usize, Option<Arc<Gpu>>)>>>;

/// The kernels loaded on each GPU, by index, once a program: the driver
/// translates them for the GPU as they load, which takes longer than a
/// small search.
static KERNELS: Mutex<Vec<(usize, Result<Kernels, String>)>> = Mutex::new(Vec::new());

/// A GPU's context and kernels.
#[derive(Clone)]
struct Kernels {
    context: Arc<CudaContext>,
    binned: CudaFunction,
    refine: RefineKernels,
    /// The most bins a tile of the binned step can hold.
    most_bins: usize,
}

impl Kernels {
    /// GPU `device`'s, loaded at the first call.
    fn on(device: usize) -> Result<Self, CudaError> {
        let mut loaded = KERNELS.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some((_, kernels)) = loaded.iter().find(|(at, _)| *at == device) {
            return kernels.clone().map_err(CudaError::Device);
        }
        let kernels = Self::load(device).map_err(|error| match error {
            CudaError::Device(why) => why,
            CudaError::Search(error) => error.to_string(),
        });
        loaded.push((device, kernels.clone()));
        kernels.map_err(CudaError::Device)
    }

    fn load(device: usize) -> Result<Self, CudaError> {
        let context = CudaContext::new(device).map_err(self::device)?;
        let options = CompileOptions {
            options: vec!["--fmad=false".into(), format!("-DTHREADS={THREADS}")],
            ..CompileOptions::default()
        };
        let ptx = compile_ptx_with_opts(SOURCE, options).map_err(self::device)?;
        let module = context.load_module(ptx).map_err(self::device)?;
        let binned = module.load_function("binned").map_err(self::device)?;
        let refine = RefineKernels::new(&module).map_err(self::device)?;
        let shared = context
            .attribute(
                CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
            )
            .map_err(self::device)?;
        let fixed = binned
            .get_attribute(CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES)
            .map_err(self::device)?;
        let dynamic = (shared - fixed).max(0);
        binned
            .set_attribute(
                CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                dynamic,
            )
            .map_err(self::device)?;
        Ok(Self {
            context,
            binned,
            refine,
            most_bins: (dynamic as usize / 16).saturating_sub(1),
        })
    }
}

#[cfg(test)]
thread_local! {
    /// For the tests: the next search on the GPU from this thread fails as a
    /// failing GPU would.
    pub(crate) static FAIL_NEXT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

fn device(error: impl fmt::Display) -> CudaError {
    CudaError::Device(error.to_string())
}

/// A light curve ready to search with the binned step and the refinement of
/// the peaks on an NVIDIA GPU: the compiled kernels and the points, on the
/// GPU, made once. See [`Device`](crate::Device) for how its results
/// compare with the CPU's.
pub(crate) struct Gpu {
    /// The GPU's index.
    index: usize,
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
    /// The folds, for refining the peaks.
    pub(crate) folds: GpuFolds,
}

impl fmt::Debug for Gpu {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Gpu")
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

impl Gpu {
    /// The kernel compiled for GPU `device` (0 for the first) and `bls`'s
    /// points copied to it.
    ///
    /// # Errors
    ///
    /// [`CudaError::Device`] if there is no such GPU, the CUDA driver or
    /// NVRTC library is not installed or cannot be loaded, or the kernel
    /// does not compile: never a panic, so a program can search on the CPU
    /// instead.
    pub(crate) fn new(bls: &Bls, device: usize) -> Result<Self, CudaError> {
        // cudarc panics where a library is missing, so look for both first,
        // once a program: loading them takes longer than a small search.
        static PRESENT: OnceLock<bool> = OnceLock::new();
        // SAFETY: this only tries to load the libraries, as cudarc does.
        let present = *PRESENT.get_or_init(|| unsafe {
            cudarc::driver::sys::is_culib_present() && cudarc::nvrtc::sys::is_culib_present()
        });
        if !present {
            return Err(CudaError::Device(
                "the CUDA driver or the NVRTC library is not installed".into(),
            ));
        }
        // A driver too old for a function cudarc calls panics too.
        std::panic::catch_unwind(|| Self::on(bls, device)).unwrap_or_else(|panic| {
            let why = panic
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| panic.downcast_ref::<&str>().copied())
                .unwrap_or("the CUDA library failed");
            Err(CudaError::Device(why.to_owned()))
        })
    }

    /// [`Gpu::new`], the libraries present.
    fn on(bls: &Bls, device: usize) -> Result<Self, CudaError> {
        let Kernels {
            context,
            binned: function,
            refine: kernels,
            most_bins,
        } = Kernels::on(device)?;
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
        // Streams of its own, so searches of other light curves run beside
        // its.
        let stream = context.new_stream().map_err(self::device)?;
        let folds = context.new_stream().map_err(self::device)?;
        let gpu = Self {
            index: device,
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
            folds: GpuFolds {
                kernels,
                points: [
                    folds.clone_htod(&s).map_err(self::device)?,
                    folds.clone_htod(&e).map_err(self::device)?,
                    folds.clone_htod(&weighted.weights).map_err(self::device)?,
                    folds.clone_htod(&weighted.centred).map_err(self::device)?,
                ],
                stream: folds,
                n: s.len(),
                last,
                mean: weighted.mean,
                least_weight: weighted
                    .weights
                    .iter()
                    .copied()
                    .fold(f64::INFINITY, f64::min),
                lock: Mutex::new(()),
            },
            stream,
        };
        // Every copy made before the first search.
        gpu.stream.synchronize().map_err(self::device)?;
        gpu.folds.stream.synchronize().map_err(self::device)?;
        Ok(gpu)
    }

    /// As [`Bls::search_into`], on the GPU.
    ///
    /// # Errors
    ///
    /// [`CudaError::Search`] for [`Bls::search`]'s errors, and
    /// [`CudaError::Device`] if the GPU fails.
    pub(crate) fn search_into(
        &self,
        bls: &Bls,
        periods: &Periods,
        durations: &Durations,
        options: &SearchOptions,
        result: &mut BlsResult,
    ) -> Result<(), CudaError> {
        let durations = prepare(periods, durations, options, result)?;
        #[cfg(test)]
        if FAIL_NEXT.with(|fail| fail.replace(false)) {
            return Err(CudaError::Device("a failure made by a test".into()));
        }
        let oversample = options.oversample.get();
        let objective = options.objective;
        let threads = options.threads.get();
        let periods = periods.as_slice();
        let days: Vec<f64> = periods.iter().map(|p| p.as_days()).collect();
        let n = days.len();
        let points = bls.weighted.elapsed.len() as f64;
        let last = bls.weighted.elapsed.last().map_or(0.0, |&(s, _)| s);

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

        // The GPU's best box at each of its periods, a launch at a time on a
        // thread of its own, each launch's built on the CPU from its points
        // while the GPU runs the next.
        let build = |&(job, &(duration, start)): &(&Job, &(i32, i32))| {
            if duration < 0 {
                return Ok(Some(None));
            }
            box_at(
                &bls.weighted,
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
                boxes[period] = Some(bls.best_box(
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
        let exact = |todo: &[usize]| {
            Self::exact_at(bls, todo, &days, &durations, oversample, objective, threads)
        };
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
                    boxes[period] = Some(bls.best_box(
                        periods[period],
                        found.bin * found.bins as f64,
                        found.centre,
                        found.fit,
                    ));
                }
            }
            Ok(())
        })?;
        bls.finish(periods, &durations, options, result, Some(&self.folds))?;
        result.gpu = Some(self.index);
        Ok(())
    }

    /// The CPU's binned search at each of `todo`, on up to `threads`
    /// threads.
    fn exact_at(
        bls: &Bls,
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
                        &bls.weighted,
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

/// Refining the peaks with the folds made and searched on the GPU, a batch
/// of them at once (see `cuda/refine.cu`); the CPU builds each best box
/// from its run.
pub(crate) struct GpuFolds {
    stream: Arc<CudaStream>,
    kernels: RefineKernels,
    /// The points' elapsed times (two parts), weights and centred fluxes.
    points: [CudaSlice<f64>; 4],
    n: usize,
    /// The last elapsed time.
    last: f64,
    /// The fluxes' weighted mean, and the least weight of a point.
    mean: f64,
    least_weight: f64,
    /// One batch at a time.
    lock: Mutex<()>,
}

/// The kernels of `cuda/refine.cu`.
#[derive(Clone)]
struct RefineKernels {
    fold_keys: CudaFunction,
    bitonic_step: CudaFunction,
    bitonic_local: CudaFunction,
    fold_phases: CudaFunction,
    fold_gather: CudaFunction,
    fold_sums: CudaFunction,
    fold_errors: CudaFunction,
    fold_compensations: CudaFunction,
    fold_orbit: CudaFunction,
    run_values: CudaFunction,
    run_mins: CudaFunction,
    runs: CudaFunction,
    run_finish: CudaFunction,
}

impl RefineKernels {
    fn new(module: &Arc<CudaModule>) -> Result<Self, DriverError> {
        let load = |name: &str| module.load_function(name);
        Ok(Self {
            fold_keys: load("fold_keys")?,
            bitonic_step: load("bitonic_step")?,
            bitonic_local: load("bitonic_local")?,
            fold_phases: load("fold_phases")?,
            fold_gather: load("fold_gather")?,
            fold_sums: load("fold_sums")?,
            fold_errors: load("fold_errors")?,
            fold_compensations: load("fold_compensations")?,
            fold_orbit: load("fold_orbit")?,
            run_values: load("run_values")?,
            run_mins: load("run_mins")?,
            runs: load("runs")?,
            run_finish: load("run_finish")?,
        })
    }
}

/// A fold to search: its period, and each search's shortest and longest
/// durations and the score to beat.
struct FoldJob {
    period: f64,
    searches: Vec<[f64; 3]>,
}

/// Each search's run, or `None` where no run beats its score; `None` for a
/// fold to search on the CPU.
type FoldRuns = Option<Vec<Option<Run>>>;

/// Points the bitonic sort orders in shared memory (`LOCAL` in the
/// kernels).
const LOCAL: usize = 2048;
/// Ends to a block of the bounds, at each level (`RUN_BLOCK`).
const RUN_BLOCK: usize = 32;
/// Numbers a search's run comes back as (`RUN_VALUES`).
const RUN_VALUES: usize = 27;
/// The share of the GPU's free memory a batch of folds may take.
const MEMORY_SHARE: f64 = 0.5;

impl GpuFolds {
    /// Each fold's runs, a batch at a time; `None` for those the GPU does not
    /// take (the CPU's fold then differs, or a score could overflow) and for
    /// all from a batch where the GPU fails.
    fn search(&self, jobs: &[FoldJob], objective: Objective) -> Vec<FoldRuns> {
        let mut found: Vec<FoldRuns> = (0..jobs.len()).map(|_| None).collect();
        let n = self.n;
        if !(2..1 << 28).contains(&n) {
            return found;
        }
        let take: Vec<usize> = (0..jobs.len())
            .filter(|&k| {
                !jobs[k].searches.is_empty() && AtPeriod::new(jobs[k].period).short_up_to(self.last)
            })
            .collect();
        if take.is_empty() {
            return found;
        }
        let Ok(_one) = self.lock.lock() else {
            return found;
        };
        // Bytes a fold: keys and indices to sort, phases, totals, weights and
        // g, and the blocks' least g.
        let width = n.next_power_of_two();
        let per_fold = 12 * width + 16 * n + 32 * (2 * n + 1) + 32 * n + n;
        let free = self
            .stream
            .context()
            .bind_to_thread()
            .and_then(|()| cudarc::driver::result::mem_get_info())
            .map_or(0, |(free, _)| free);
        let batch = ((free as f64 * MEMORY_SHARE) as usize / per_fold).max(1);
        for chunk in take.chunks(batch) {
            let Ok(runs) = self.batch(jobs, chunk, objective) else {
                break;
            };
            for (&k, runs) in chunk.iter().zip(runs) {
                found[k] = runs;
            }
        }
        found
    }

    /// The runs of the folds `which` of `jobs`, at once.
    fn batch(
        &self,
        jobs: &[FoldJob],
        which: &[usize],
        objective: Objective,
    ) -> Result<Vec<FoldRuns>, DriverError> {
        let stream = &self.stream;
        let k = &self.kernels;
        let n = self.n;
        let folds = which.len();
        let width = n.next_power_of_two();
        let periods: Vec<f64> = which
            .iter()
            .flat_map(|&j| {
                let p = jobs[j].period;
                let (hi, lo) = AtPeriod::new(p).halves();
                [p, 1.0 / p, hi, lo]
            })
            .collect();
        let most = which
            .iter()
            .map(|&j| jobs[j].searches.len())
            .max()
            .unwrap_or(0);
        let mut searches = vec![0.0; folds * most * 3];
        let mut counts = vec![0_i32; folds];
        for (f, &j) in which.iter().enumerate() {
            counts[f] = jobs[j].searches.len() as i32;
            for (s, values) in jobs[j].searches.iter().enumerate() {
                searches[3 * (f * most + s)..3 * (f * most + s + 1)].copy_from_slice(values);
            }
        }
        let periods = stream.clone_htod(&periods)?;
        let searches = stream.clone_htod(&searches)?;
        let counts = stream.clone_htod(&counts)?;

        let (n_i, width_i, folds_i, most_i) = (n as i32, width as i32, folds as i32, most as i32);
        let grid = |items: usize, rows: usize| LaunchConfig {
            grid_dim: (items.div_ceil(THREADS as usize) as u32, rows as u32, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let [s, e, weights, centred] = &self.points;

        // The folds: keys, sorted, phases and totals.
        // SAFETY (each launch below): the arguments are the kernel's, in its
        // order and of its types, and every array holds what the kernel
        // reads or writes of it for `folds` folds of `n` points (`width`
        // keys and indices, `2n` phases, `2n + 1` totals of four, `most`
        // searches of three numbers, and one output a block of each).
        let mut keys = stream.alloc_zeros::<u64>(folds * width)?;
        let mut index = stream.alloc_zeros::<u32>(folds * width)?;
        let mut launch = stream.launch_builder(&k.fold_keys);
        launch
            .arg(s)
            .arg(e)
            .arg(&n_i)
            .arg(&periods)
            .arg(&width_i)
            .arg(&mut keys)
            .arg(&mut index);
        unsafe { launch.launch(grid(width, folds)) }?;
        // The steps between points less than `LOCAL` apart, in shared memory.
        let local = |keys: &mut CudaSlice<u64>,
                     index: &mut CudaSlice<u32>,
                     kk: i32|
         -> Result<(), DriverError> {
            let config = LaunchConfig {
                grid_dim: (width.div_ceil(LOCAL) as u32, folds as u32, 1),
                block_dim: (1024, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut launch = stream.launch_builder(&k.bitonic_local);
            launch.arg(keys).arg(index).arg(&width_i).arg(&kk);
            unsafe { launch.launch(config) }?;
            Ok(())
        };
        local(&mut keys, &mut index, 0)?;
        let mut length = 2 * LOCAL;
        while length <= width {
            let mut j = length / 2;
            while j >= LOCAL {
                let (kk, jj) = (length as i32, j as i32);
                let mut launch = stream.launch_builder(&k.bitonic_step);
                launch
                    .arg(&mut keys)
                    .arg(&mut index)
                    .arg(&width_i)
                    .arg(&kk)
                    .arg(&jj);
                unsafe { launch.launch(grid(width, folds)) }?;
                j /= 2;
            }
            local(&mut keys, &mut index, length as i32)?;
            length *= 2;
        }
        let mut phase = stream.alloc_zeros::<f64>(folds * 2 * n)?;
        let mut launch = stream.launch_builder(&k.fold_phases);
        launch
            .arg(&keys)
            .arg(&n_i)
            .arg(&width_i)
            .arg(&periods)
            .arg(&mut phase);
        unsafe { launch.launch(grid(n, folds)) }?;
        drop(keys);
        let mut sorted = stream.alloc_zeros::<f64>(folds * 2 * n)?;
        let mut launch = stream.launch_builder(&k.fold_gather);
        launch
            .arg(&index)
            .arg(weights)
            .arg(centred)
            .arg(&n_i)
            .arg(&width_i)
            .arg(&mut sorted);
        unsafe { launch.launch(grid(n, folds)) }?;
        drop(index);
        let mut totals = stream.alloc_zeros::<f64>(folds * 4 * (2 * n + 1))?;
        let chains = LaunchConfig {
            grid_dim: (folds as u32, 1, 1),
            block_dim: (2, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch = stream.launch_builder(&k.fold_sums);
        launch.arg(&sorted).arg(&n_i).arg(&mut totals);
        unsafe { launch.launch(chains) }?;
        let mut launch = stream.launch_builder(&k.fold_errors);
        launch.arg(&sorted).arg(&n_i).arg(&mut totals);
        unsafe { launch.launch(grid(2 * n, folds)) }?;
        let mut launch = stream.launch_builder(&k.fold_compensations);
        launch.arg(&n_i).arg(&mut totals);
        unsafe { launch.launch(chains) }?;
        drop(sorted);

        // The searches.
        let mut orbit = stream.alloc_zeros::<f64>(folds * 4)?;
        let mut ordered = stream.alloc_zeros::<i32>(folds)?;
        let mut launch = stream.launch_builder(&k.fold_orbit);
        launch
            .arg(&totals)
            .arg(&phase)
            .arg(&n_i)
            .arg(&periods)
            .arg(&folds_i)
            .arg(&mut orbit)
            .arg(&mut ordered);
        unsafe { launch.launch(grid(folds, 1)) }?;
        let mut weight = stream.alloc_zeros::<f64>(folds * 2 * n)?;
        let mut g = stream.alloc_zeros::<f64>(folds * 2 * n)?;
        let mut largest = stream.alloc_zeros::<u64>(folds)?;
        let mut launch = stream.launch_builder(&k.run_values);
        launch
            .arg(&totals)
            .arg(&n_i)
            .arg(&orbit)
            .arg(&mut weight)
            .arg(&mut g)
            .arg(&mut largest);
        unsafe { launch.launch(grid(2 * n, folds)) }?;
        let count0 = (2 * n).div_ceil(RUN_BLOCK);
        let count1 = count0.div_ceil(RUN_BLOCK);
        let (count_i, count0_i, count1_i) = ((2 * n) as i32, count0 as i32, count1 as i32);
        let mut mins0 = stream.alloc_zeros::<f64>(folds * count0)?;
        let mut mins1 = stream.alloc_zeros::<f64>(folds * count1)?;
        let mut launch = stream.launch_builder(&k.run_mins);
        launch.arg(&g).arg(&count_i).arg(&count0_i).arg(&mut mins0);
        unsafe { launch.launch(grid(count0, folds)) }?;
        let mut launch = stream.launch_builder(&k.run_mins);
        launch
            .arg(&mins0)
            .arg(&count0_i)
            .arg(&count1_i)
            .arg(&mut mins1);
        unsafe { launch.launch(grid(count1, folds)) }?;

        // Enough blocks a search to fill the GPU, each thread taking first
        // points in turn.
        let blocks = n
            .div_ceil(THREADS as usize)
            .min((4096 / (folds * most).max(1)).max(8));
        let outputs = folds * most * blocks;
        let mut highest = stream.alloc_zeros::<u64>(folds * most)?;
        let mut scores = stream.alloc_zeros::<f64>(outputs)?;
        let mut firsts = stream.alloc_zeros::<i32>(outputs)?;
        let mut ends = stream.alloc_zeros::<i32>(outputs)?;
        let mut bad = stream.alloc_zeros::<i32>(folds)?;
        let likelihood = i32::from(objective == Objective::AstropyLikelihood);
        let mut launch = stream.launch_builder(&k.runs);
        launch
            .arg(&phase)
            .arg(&totals)
            .arg(&n_i)
            .arg(&periods)
            .arg(&searches)
            .arg(&counts)
            .arg(&most_i)
            .arg(&orbit)
            .arg(&ordered)
            .arg(&likelihood)
            .arg(&g)
            .arg(&weight)
            .arg(&mins0)
            .arg(&mins1)
            .arg(&count0_i)
            .arg(&count1_i)
            .arg(&largest)
            .arg(&mut highest)
            .arg(&mut scores)
            .arg(&mut firsts)
            .arg(&mut ends)
            .arg(&mut bad);
        let config = LaunchConfig {
            grid_dim: (blocks as u32, most as u32, folds as u32),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { launch.launch(config) }?;
        let mut out = stream.alloc_zeros::<f64>(folds * most * RUN_VALUES)?;
        let blocks_i = blocks as i32;
        let mut launch = stream.launch_builder(&k.run_finish);
        launch
            .arg(&phase)
            .arg(&totals)
            .arg(&n_i)
            .arg(&periods)
            .arg(&counts)
            .arg(&most_i)
            .arg(&folds_i)
            .arg(&blocks_i)
            .arg(&scores)
            .arg(&firsts)
            .arg(&ends)
            .arg(&mut out);
        unsafe { launch.launch(grid(folds * most, 1)) }?;
        let out = stream.clone_dtoh(&out)?;
        let bad = stream.clone_dtoh(&bad)?;
        let ordered = stream.clone_dtoh(&ordered)?;
        let largest = stream.clone_dtoh(&largest)?;
        let orbit = stream.clone_dtoh(&orbit)?;

        Ok((0..folds)
            .map(|f| {
                if bad[f] != 0 || ordered[f] == 0 {
                    return None;
                }
                // No score can overflow, so none is an error, wherever the
                // CPU's search would skip it: the fall in flux is at most
                // `2F + |S|`, and the box and the rest each hold a point's
                // weight, to a rounding.
                let (whole, flux) = (orbit[4 * f], orbit[4 * f + 1]);
                let fall = 2.0 * f64::from_bits(largest[f]) + flux.abs();
                let least = self.least_weight * (self.least_weight / whole) * 0.5;
                let mut most_score = fall * (fall / least);
                if objective == Objective::AstropyLikelihood {
                    most_score *= whole / self.least_weight;
                }
                if most_score.is_nan() || most_score >= 1e300 {
                    return None;
                }
                let count = jobs[which[f]].searches.len();
                Some(
                    (0..count)
                        .map(|s| {
                            let o = &out[RUN_VALUES * (f * most + s)..][..RUN_VALUES];
                            (o[0] != 0.0).then(|| {
                                let total = |w: usize| {
                                    let t = &o[7 + 4 * w..][..4];
                                    Total {
                                        weight: Sum::from_parts([t[0], t[1]]),
                                        flux: Sum::from_parts([t[2], t[3]]),
                                    }
                                };
                                Run {
                                    points: (o[2] - o[1]) as usize,
                                    phases: [o[3], o[4], o[5], o[6]],
                                    totals: [total(0), total(1), total(2), total(3), total(4)],
                                }
                            })
                        })
                        .collect(),
                )
            })
            .collect())
    }
}

impl Folds for GpuFolds {
    fn best_of(
        &self,
        periods: &[f64],
        durations: &[f64],
        objective: Objective,
    ) -> Vec<Option<BestOf>> {
        // Each duration's first highest run with a dip: of the runs beating
        // the best so far, `Folded::best_of` finds that one whenever it is
        // among them, and none otherwise.
        let jobs: Vec<FoldJob> = periods
            .iter()
            .map(|&p| FoldJob {
                period: p,
                searches: durations
                    .iter()
                    .filter(|&&d| d < p)
                    .map(|&d| [d, d, 0.0])
                    .collect(),
            })
            .collect();
        let found = self.search(&jobs, objective);
        found
            .into_iter()
            .zip(periods)
            .map(|(runs, &p)| {
                let runs = runs?;
                let mut runs = runs.into_iter();
                let mut best = || -> BestOf {
                    let mut best: Option<(usize, f64, BoxFit)> = None;
                    for (k, &d) in durations.iter().enumerate() {
                        if d >= p {
                            continue;
                        }
                        let beat = best.map_or(0.0, |b| objective.beat(objective.of(&b.2)));
                        let Some(run) = runs.next().expect("a run a duration") else {
                            continue;
                        };
                        if run.score(objective)? > beat {
                            let (centre, _, fit) = run.best_box(self.mean, d, d)?;
                            if best.is_none_or(|b| objective.of(&fit) > objective.of(&b.2)) {
                                best = Some((k, centre, fit));
                            }
                        }
                    }
                    Ok(best)
                };
                Some(best())
            })
            .collect()
    }

    fn best_between(&self, jobs: &[[f64; 4]], objective: Objective) -> Vec<Option<BestBetween>> {
        let folds: Vec<FoldJob> = jobs
            .iter()
            .map(|&[p, shortest, longest, beat]| FoldJob {
                period: p,
                searches: if shortest < p {
                    vec![[shortest, longest, beat]]
                } else {
                    Vec::new()
                },
            })
            .collect();
        self.search(&folds, objective)
            .into_iter()
            .zip(jobs)
            .map(|(runs, &[_, shortest, longest, _])| {
                let run = runs?[0];
                Some(
                    run.map(|run| run.best_box(self.mean, shortest, longest))
                        .transpose(),
                )
            })
            .collect()
    }
}

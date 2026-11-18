//! The cells search on an NVIDIA GPU: [`Gpu`].
//!
//! The points and the template's table are copied to the GPU once; each
//! duration's cells are made there, one thread a cell, and every period it
//! is searched at runs there, one block a period. Both kernels make the
//! CPU's operations in the CPU's order, so each period's best is the CPU's
//! to the bit.

use std::{
    fmt,
    sync::{Arc, Mutex, OnceLock, PoisonError},
};

use cudarc::{
    driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg},
    nvrtc::{CompileOptions, compile_ptx_with_opts},
};

use crate::cells::{Cells, Search, Trials};
use crate::fast::Table;
use crate::fit::Points;

/// The kernels' source, compiled by NVRTC when the first [`Gpu`] is made on
/// a GPU.
const SOURCE: &str = concat!(
    include_str!("cuda/cells.cu"),
    include_str!("cuda/periodic.cu")
);
/// Threads a block.
const THREADS: u32 = 256;

/// The kernels loaded on each GPU, by index, once a program.
static KERNELS: Mutex<Vec<(usize, Result<Kernels, String>)>> = Mutex::new(Vec::new());

/// A GPU's context and kernels.
#[derive(Clone)]
struct Kernels {
    context: Arc<CudaContext>,
    cells: CudaFunction,
    periodic: CudaFunction,
}

impl Kernels {
    /// GPU `device`'s, loaded at the first call.
    fn on(device: usize) -> Result<Self, String> {
        let mut loaded = KERNELS.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some((_, kernels)) = loaded.iter().find(|(at, _)| *at == device) {
            return kernels.clone();
        }
        let kernels = Self::load(device);
        loaded.push((device, kernels.clone()));
        kernels
    }

    fn load(device: usize) -> Result<Self, String> {
        let context = CudaContext::new(device).map_err(why)?;
        let options = CompileOptions {
            options: vec!["--fmad=false".into(), format!("-DTHREADS={THREADS}")],
            ..CompileOptions::default()
        };
        let ptx = compile_ptx_with_opts(SOURCE, options).map_err(why)?;
        let module = context.load_module(ptx).map_err(why)?;
        Ok(Self {
            context,
            cells: module.load_function("cells").map_err(why)?,
            periodic: module.load_function("periodic").map_err(why)?,
        })
    }
}

fn why(error: impl fmt::Display) -> String {
    error.to_string()
}

#[cfg(test)]
thread_local! {
    /// For the tests: the next search on the GPU from this thread fails as a
    /// failing GPU would.
    pub(crate) static FAIL_NEXT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// A GPU ready to search: its kernels, compiled once a program, and a
/// stream of its own.
pub(crate) struct Gpu {
    stream: Arc<CudaStream>,
    cells: CudaFunction,
    periodic: CudaFunction,
}

impl fmt::Debug for Gpu {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Gpu").finish_non_exhaustive()
    }
}

/// A period's best on the GPU: its trial time, gain and depth, or `None`
/// where nothing fitted.
pub(crate) type Found = Option<(usize, f64, f64)>;

impl Gpu {
    /// GPU `device` (0 for the first), or why it cannot be used: no such
    /// GPU, the CUDA driver or NVRTC library not installed, or the kernels
    /// not compiled. Never a panic, so the search can stay on the CPU.
    pub(crate) fn new(device: usize) -> Result<Self, String> {
        // cudarc panics where a library is missing, so look for both first,
        // once a program.
        static PRESENT: OnceLock<bool> = OnceLock::new();
        // SAFETY: this only tries to load the libraries, as cudarc does.
        let present = *PRESENT.get_or_init(|| unsafe {
            cudarc::driver::sys::is_culib_present() && cudarc::nvrtc::sys::is_culib_present()
        });
        if !present {
            return Err("the CUDA driver or the NVRTC library is not installed".into());
        }
        // A driver too old for a function cudarc calls panics too.
        std::panic::catch_unwind(|| Self::on(device)).unwrap_or_else(|panic| {
            Err(panic
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| panic.downcast_ref::<&str>().copied())
                .unwrap_or("the CUDA library failed")
                .to_owned())
        })
    }

    fn on(device: usize) -> Result<Self, String> {
        let Kernels {
            context,
            cells,
            periodic,
        } = Kernels::on(device)?;
        Ok(Self {
            stream: context.new_stream().map_err(why)?,
            cells,
            periodic,
        })
    }

    /// `points` and `table` copied to the GPU, for a search.
    pub(crate) fn load(&self, points: &Points, table: &Table) -> Result<Loaded<'_>, String> {
        let stream = &self.stream;
        Ok(Loaded {
            gpu: self,
            time: stream.clone_htod(&points.time).map_err(why)?,
            residual: stream.clone_htod(&points.residual).map_err(why)?,
            weight: stream.clone_htod(&points.weight).map_err(why)?,
            table: stream.clone_htod(table.values()).map_err(why)?,
            intervals: table.intervals(),
        })
    }
}

/// The points and the template's table on a GPU.
pub(crate) struct Loaded<'g> {
    gpu: &'g Gpu,
    time: CudaSlice<f64>,
    residual: CudaSlice<f64>,
    weight: CudaSlice<f64>,
    table: CudaSlice<f64>,
    intervals: f64,
}

impl Loaded<'_> {
    /// The cells of `duration`, `per_duration` to a duration, made on the
    /// GPU, and each of `jobs` (a period and its trial times) searched with
    /// them: its best trial time, gain and depth, as `search`'s scan finds
    /// them.
    pub(crate) fn search(
        &self,
        search: &Search<'_>,
        duration: f64,
        per_duration: usize,
        jobs: &[(f64, Trials)],
    ) -> Result<Vec<Found>, String> {
        #[cfg(test)]
        if FAIL_NEXT.with(|fail| fail.replace(false)) {
            return Err("failed for the test".into());
        }
        if jobs.is_empty() {
            return Ok(Vec::new());
        }
        let stream = &self.gpu.stream;
        let (start, step, count) = Cells::grid(&search.points.time, duration, per_duration);
        #[expect(clippy::cast_possible_wrap, reason = "counts far below 2^63")]
        let (points, cells) = (search.points.time.len() as i64, count as i64);
        let mut rs = stream.alloc_zeros::<f64>(count).map_err(why)?;
        let mut ss = stream.alloc_zeros::<f64>(count).map_err(why)?;
        #[expect(
            clippy::cast_possible_truncation,
            reason = "far below 2^31 blocks of cells"
        )]
        let config = LaunchConfig {
            grid_dim: (count.div_ceil(THREADS as usize) as u32, 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch = stream.launch_builder(&self.gpu.cells);
        launch
            .arg(&self.time)
            .arg(&self.residual)
            .arg(&self.weight)
            .arg(&points)
            .arg(&self.table)
            .arg(&self.intervals)
            .arg(&duration)
            .arg(&start)
            .arg(&step)
            .arg(&cells)
            .arg(&mut rs)
            .arg(&mut ss);
        // SAFETY: the arguments are the kernel's, in its order; each thread
        // writes its own cell.
        unsafe { launch.launch(config) }.map_err(why)?;

        let (from, to) = search.window(duration);
        let periods: Vec<f64> = jobs.iter().map(|&(p, _)| p).collect();
        #[expect(clippy::cast_possible_wrap, reason = "counts far below 2^63")]
        let (deltas, counts): (Vec<i64>, Vec<i64>) = jobs
            .iter()
            .map(|&(_, t)| (t.delta as i64, t.count as i64))
            .unzip();
        let periods = stream.clone_htod(&periods).map_err(why)?;
        let deltas = stream.clone_htod(&deltas).map_err(why)?;
        let counts = stream.clone_htod(&counts).map_err(why)?;
        let mut best_k = stream.alloc_zeros::<i64>(jobs.len()).map_err(why)?;
        let mut best_gain = stream.alloc_zeros::<f64>(jobs.len()).map_err(why)?;
        let mut best_depth = stream.alloc_zeros::<f64>(jobs.len()).map_err(why)?;
        #[expect(clippy::cast_possible_truncation, reason = "periods far below 2^31")]
        let config = LaunchConfig {
            grid_dim: (jobs.len() as u32, 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let min_depth = search.settings.min_depth;
        let mut launch = stream.launch_builder(&self.gpu.periodic);
        launch
            .arg(&rs)
            .arg(&ss)
            .arg(&start)
            .arg(&step)
            .arg(&periods)
            .arg(&deltas)
            .arg(&counts)
            .arg(&search.origin)
            .arg(&from)
            .arg(&to)
            .arg(&min_depth)
            .arg(&mut best_k)
            .arg(&mut best_gain)
            .arg(&mut best_depth);
        // SAFETY: the arguments are the kernel's, in its order, and each
        // block reads only the cells its trial times' transits lie in.
        unsafe { launch.launch(config) }.map_err(why)?;
        let k = stream.clone_dtoh(&best_k).map_err(why)?;
        let gain = stream.clone_dtoh(&best_gain).map_err(why)?;
        let depth = stream.clone_dtoh(&best_depth).map_err(why)?;
        Ok(k.into_iter()
            .zip(gain)
            .zip(depth)
            .map(|((k, gain), depth)| usize::try_from(k).ok().map(|k| (k, gain, depth)))
            .collect())
    }
}

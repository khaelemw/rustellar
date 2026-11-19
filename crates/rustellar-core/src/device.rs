//! Where a search runs.

/// Where a search runs: the CPU or an NVIDIA GPU.
///
/// The GPU is used only by searches built with their `cuda` feature; without
/// it every search runs on the CPU. Each search documents what it runs on
/// the GPU and whether its results there match the CPU's to the bit.
///
/// # Examples
///
/// ```
/// use rustellar_core::Device;
///
/// // The default: a GPU where one can be used, otherwise the CPU.
/// assert_eq!(Device::default(), Device::Auto);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum Device {
    /// The first GPU where the `cuda` feature is on and the GPU can be
    /// used, and otherwise the CPU, with no error: a GPU that cannot be set
    /// up, or fails during the search, leaves the search to the CPU.
    #[default]
    Auto,
    /// The CPU.
    Cpu,
    /// The GPU of this index (0 for the first), or the search's
    /// `GpuUnavailable` error where it cannot be used.
    Gpu(usize),
}

# rustellar-tls

Transit least squares in pure Rust: a search for a repeating transit with a
limb-darkened template instead of a box (Hippke & Heller 2019). At each trial
period and duration the template is fitted at every mid-transit time, its
depth by weighted least squares, and the best fit's chi-squared gives the
spectrum.

Part of [rustellar](../../README.md). Use it on its own, or through the
`rustellar` crate with the `tls` feature.

**Status:** in development, not yet released.

Rust 1.85 or later; the `cuda` feature (the search on an NVIDIA GPU, with
the CPU's results to the bit) needs Rust 1.88 or later, an NVIDIA driver
and the CUDA NVRTC library at run time.

## References

- Hippke, M. & Heller, R. 2019, A&A, 623, A39
- Mandel, K. & Agol, E. 2002, ApJ, 580, L171
- Ofir, A. 2014, A&A, 561, A138

## License

MIT or Apache-2.0, at your option.

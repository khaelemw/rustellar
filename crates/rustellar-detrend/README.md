# rustellar-detrend

Detrending light curves in pure Rust: the slow variation of a star and of
the instrument estimated and divided out before a transit search. The method
is a time-windowed sliding filter, by default Tukey's biweight (Hippke et al.
2019), solved to full double precision, with every point in the windows
that cover it and the same results on any number of threads.

Part of [rustellar](../../README.md). Use it on its own, or through the
`rustellar` crate with the `detrend` feature.

**Status:** in development, not yet released.

## References

- Hippke, M., David, T. J., Mulders, G. D. & Heller, R. 2019, AJ, 158, 143

## License

MIT or Apache-2.0, at your option.

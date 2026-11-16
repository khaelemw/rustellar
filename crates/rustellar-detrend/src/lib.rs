//! Detrending light curves: the slow variation of a star (spots, rotation)
//! and of the instrument (drift, scattered light) estimated and divided out,
//! so that a transit search sees a flat light curve.
//!
//! The method will be a time-windowed sliding filter: for each point, a
//! robust location of the fluxes within half a window either side, as the
//! trend, and the flux divided by it. The default location is Tukey's
//! biweight (Hippke et al. 2019), solved to full double precision.
//!
//! **Status:** in development. Nothing is public yet.

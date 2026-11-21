//! The non-uniform fast Fourier transform of type 1 in one dimension, with
//! the "exponential of semicircle" kernel of Barnett, Magland & af
//! Klinteberg (2019, SIAM J. Sci. Comput. 41, C479):
//! `F(m) = sum_j c_j e^{i m x_j}` for every whole `m` in `[-M/2, M/2)`,
//! each to about `tolerance` times `sum_j |c_j|`.
//!
//! Each point's strength is spread onto a grid twice as fine as the modes
//! with the kernel `phi(z) = exp(beta (sqrt(1 - z^2) - 1))`, `w` grid points
//! wide; the grid is transformed; and each mode divided by the kernel's
//! Fourier transform, from Gauss-Legendre quadrature.

use crate::fft::{Complex, Fft, turn};
use crate::parallel::{run_all, threads_for};

/// A transform for one number of modes and tolerance.
#[derive(Debug, Clone)]
pub(crate) struct Nufft {
    /// The modes, even.
    modes: usize,
    fft: Fft,
    /// The kernel's width in grid points.
    width: usize,
    beta: f64,
    /// `1 / (w I(m))` for `m` from 0 to `modes / 2`, `I` the kernel's
    /// transform over `[0, 1]`.
    correction: Vec<f64>,
}

impl Nufft {
    /// For `modes` modes (made even) to `tolerance` (from 1e-15 to 0.1),
    /// set up on up to `threads` threads.
    pub(crate) fn new(modes: usize, tolerance: f64, threads: usize) -> Self {
        let modes = modes + modes % 2;
        let tolerance = tolerance.clamp(1e-15, 0.1);
        // Barnett et al. (2019), section 3: w = ceil(log10(1 / eps)) + 1
        // points for an upsampling of 2, beta = 2.30 w.
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "from 2 to 16"
        )]
        let width = ((1.0 / tolerance).log10().ceil() as usize + 1).clamp(2, 16);
        #[expect(clippy::cast_precision_loss, reason = "at most 16")]
        let beta = 2.30 * width as f64;
        let fft = Fft::new((2 * modes).max(2 * width).next_power_of_two());
        let correction = correction(modes, width, beta, fft.len(), threads);
        Self {
            modes,
            fft,
            width,
            beta,
            correction,
        }
    }

    /// The modes (even).
    pub(crate) const fn modes(&self) -> usize {
        self.modes
    }

    /// The kernel's width in grid points.
    #[cfg(test)]
    pub(crate) const fn width(&self) -> usize {
        self.width
    }

    /// `F_s(m) = sum_j strengths[s][j] e^{2 pi i m cycles[j]}` for each set
    /// of strengths `s` and `m` from `-modes / 2` (at index 0) up, with
    /// `cycles` in `[-0.5, 0.5]`, on up to `threads` threads (the same
    /// bits on any number).
    pub(crate) fn run<const S: usize>(
        &self,
        cycles: &[f64],
        strengths: [&[Complex]; S],
        threads: usize,
    ) -> [Vec<Complex>; S] {
        let n = self.fft.len();
        let threads = threads.max(1);
        let grids = self.spread(cycles, strengths, threads);
        let half_modes = self.modes / 2;
        grids.map(|mut grid| {
            self.fft.run(&mut grid, threads);
            (0..self.modes)
                .map(|index| {
                    // Mode m = index - modes / 2, at grid index m mod n.
                    let (cell, c) = if index < half_modes {
                        (
                            n - (half_modes - index),
                            self.correction[half_modes - index],
                        )
                    } else {
                        (index - half_modes, self.correction[index - half_modes])
                    };
                    let b = grid[cell];
                    Complex::new(b.re * c, b.im * c)
                })
                .collect()
        })
    }

    /// The strengths spread onto the grid: the grid cut into segments of a
    /// fixed size, each point's kernel spread into a private copy of the
    /// segment its first grid point falls in (and the width after it), the
    /// segments shared out among the threads, then the copies added.
    fn spread<const S: usize>(
        &self,
        cycles: &[f64],
        strengths: [&[Complex]; S],
        threads: usize,
    ) -> [Vec<Complex>; S] {
        let n = self.fft.len();
        let width = self.width;
        // Segments of at least 4 widths, at most 256 of them: fixed by the
        // grid alone, so the sums' order is too.
        let segment = (n / 256).max(4 * width).next_power_of_two().min(n);
        let segments = n / segment;
        // Each point's first grid point (wrapped) and its kernel's offset.
        let half = 0.5 * to_f64(width);
        let mut place = Vec::with_capacity(cycles.len());
        let mut counts = vec![0_usize; segments + 1];
        for &r in cycles {
            let u = r * to_f64(n);
            let first = (u - half).ceil();
            #[expect(
                clippy::cast_possible_truncation,
                reason = "rem_euclid is from 0 to below n"
            )]
            let start = (first as i64).rem_euclid(to_i64(n)) as usize;
            place.push((start, first - u));
            counts[start / segment + 1] += 1;
        }
        // The points by segment, in their order within each.
        for s in 0..segments {
            counts[s + 1] += counts[s];
        }
        let mut order = vec![0; cycles.len()];
        let mut next = counts.clone();
        for (j, &(start, _)) in place.iter().enumerate() {
            let s = start / segment;
            order[next[s]] = j;
            next[s] += 1;
        }
        // Each segment's copy, `segment + width` cells; a thread to some
        // thousands of points at least.
        let threads = threads_for(threads, cycles.len(), 1 << 12);
        let per_thread = segments.div_ceil(threads);
        let (place, order, counts) = (&place, &order, &counts);
        let copies: Vec<[Vec<Complex>; S]> =
            run_all((0..segments).step_by(per_thread).map(|first_segment| {
                move || {
                    let mut kernel = vec![0.0; width];
                    (first_segment..(first_segment + per_thread).min(segments))
                        .map(|s| {
                            let mut copy: [Vec<Complex>; S] =
                                std::array::from_fn(|_| vec![Complex::default(); segment + width]);
                            for &j in &order[counts[s]..counts[s + 1]] {
                                let (start, offset) = place[j];
                                for (k, value) in kernel.iter_mut().enumerate() {
                                    *value = self.phi((offset + to_f64(k)) / half);
                                }
                                let at = start - s * segment;
                                for (grid, set) in copy.iter_mut().zip(&strengths) {
                                    let c = set[j];
                                    for (cell, &v) in grid[at..at + width].iter_mut().zip(&kernel) {
                                        cell.re += c.re * v;
                                        cell.im += c.im * v;
                                    }
                                }
                            }
                            copy
                        })
                        .collect::<Vec<_>>()
                }
            }))
            .into_iter()
            .flatten()
            .collect();
        // Each segment's own cells, then the width past it added to the
        // next segment's (wrapping), in segment order.
        std::array::from_fn(|set| {
            let mut grid = vec![Complex::default(); n];
            for (s, copy) in copies.iter().enumerate() {
                grid[s * segment..(s + 1) * segment].copy_from_slice(&copy[set][..segment]);
            }
            for (s, copy) in copies.iter().enumerate() {
                for (k, v) in copy[set][segment..].iter().enumerate() {
                    let cell = &mut grid[((s + 1) * segment + k) & (n - 1)];
                    cell.re += v.re;
                    cell.im += v.im;
                }
            }
            grid
        })
    }

    /// The kernel at `z`, for `|z|` at most 1 (as every spread point is).
    #[inline]
    fn phi(&self, z: f64) -> f64 {
        (self.beta * ((1.0 - z * z).max(0.0).sqrt() - 1.0)).exp()
    }
}

/// `1 / (w I(m))` for `m` from 0 to `modes / 2`, with
/// `I(m) = integral_0^1 phi(z) cos(pi w m z / n) dz` by Gauss-Legendre
/// quadrature, the cosines turned from mode to mode and set exactly every
/// 256 modes; blocks of modes shared out among `threads` threads (the same
/// bits on any number).
fn correction(modes: usize, width: usize, beta: f64, n: usize, threads: usize) -> Vec<f64> {
    let (nodes, weights) = gauss_legendre(2 + 2 * width);
    let w = to_f64(width);
    let count = modes / 2 + 1;
    let threads = threads_for(threads, count * nodes.len(), 1 << 16);
    let block = count.div_ceil(threads).div_ceil(256) * 256;
    let mut out = vec![0.0; count];
    let (nodes, weights) = (&nodes, &weights);
    run_all(out.chunks_mut(block).enumerate().map(|(b, part)| {
        move || {
            let mut integral = vec![0.0; part.len()];
            for (&z, &weight) in nodes.iter().zip(weights) {
                let value = weight * (beta * ((1.0 - z * z).sqrt() - 1.0)).exp();
                // cos(2 pi m step) with step = w z / (2 n) cycles.
                let step = w * z / (2.0 * to_f64(n));
                let rotate = turn(step);
                let mut current = Complex::new(1.0, 0.0);
                for (i, sum) in integral.iter_mut().enumerate() {
                    let m = b * block + i;
                    if m % 256 == 0 {
                        current = turn(step * to_f64(m));
                    }
                    *sum += value * current.re;
                    current = current.mul(rotate);
                }
            }
            for (o, i) in part.iter_mut().zip(integral) {
                *o = 1.0 / (w * i);
            }
        }
    }));
    out
}

/// The nodes and weights of Gauss-Legendre quadrature with `q` nodes on
/// `[0, 1]`, by Newton's method on the Legendre polynomial.
pub(crate) fn gauss_legendre(q: usize) -> (Vec<f64>, Vec<f64>) {
    let mut nodes = Vec::with_capacity(q);
    let mut weights = Vec::with_capacity(q);
    let qf = to_f64(q);
    for i in 1..=q {
        // A first guess at the i-th root of P_q on [-1, 1].
        let mut x = (std::f64::consts::PI * (to_f64(i) - 0.25) / (qf + 0.5)).cos();
        for _ in 0..100 {
            let (p, dp) = legendre(q, x);
            let dx = p / dp;
            x -= dx;
            if dx.abs() <= 1e-17 {
                break;
            }
        }
        let (_, derivative) = legendre(q, x);
        nodes.push(0.5 * (1.0 - x));
        weights.push(1.0 / ((1.0 - x * x) * derivative * derivative));
    }
    (nodes, weights)
}

/// `P_q(x)` and its derivative for `q` at least 1, by the three-term
/// recurrence.
fn legendre(q: usize, x: f64) -> (f64, f64) {
    let (mut p0, mut p1) = (1.0, x);
    for k in 2..=q {
        let k = to_f64(k);
        let p2 = ((2.0 * k - 1.0) * x * p1 - (k - 1.0) * p0) / k;
        p0 = p1;
        p1 = p2;
    }
    (p1, to_f64(q) * (x * p1 - p0) / (x * x - 1.0))
}

#[inline]
fn to_f64(n: usize) -> f64 {
    #[expect(clippy::cast_precision_loss, reason = "small counts")]
    let x = n as f64;
    x
}

#[inline]
fn to_i64(n: usize) -> i64 {
    i64::try_from(n).expect("a grid of fewer than 2^63 points")
}

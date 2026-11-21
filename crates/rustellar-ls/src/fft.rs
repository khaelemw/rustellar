//! A complex fast Fourier transform of a power-of-two length: iterative
//! radix 2, twiddles from exact phases.

use crate::parallel::{run_all, threads_for};
use crate::sums::{phase, split};

/// A complex number.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct Complex {
    pub(crate) re: f64,
    pub(crate) im: f64,
}

impl Complex {
    #[inline]
    pub(crate) const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    #[inline]
    pub(crate) fn mul(self, o: Self) -> Self {
        Self::new(
            self.re * o.re - self.im * o.im,
            self.re * o.im + self.im * o.re,
        )
    }
}

/// `e^{2 pi i x}`, from the exact phase.
pub(crate) fn turn(x: f64) -> Complex {
    let ((cb, cx), (sb, sx)) = phase(x, split(x), 1.0, split(1.0));
    Complex::new(cb + cx, sb + sx)
}

/// The transform `X_m = sum_l x_l e^{2 pi i m l / n}` (the positive sign,
/// unscaled) for one length `n`, a power of two.
#[derive(Debug, Clone)]
pub(crate) struct Fft {
    n: usize,
    /// Each stage's twiddles in turn: `e^{2 pi i k / (2 h)}` at `h + k`,
    /// for each half `h` below `n` and `k < h`.
    twiddles: Vec<Complex>,
}

impl Fft {
    /// For length `n`, a power of two.
    pub(crate) fn new(n: usize) -> Self {
        assert!(n.is_power_of_two(), "length {n}");
        #[expect(clippy::cast_precision_loss, reason = "a power of two")]
        let inverse = 1.0 / n as f64;
        let mut twiddles = vec![Complex::default(); n.max(2)];
        let last = n / 2;
        for k in 0..last {
            #[expect(clippy::cast_precision_loss, reason = "below n")]
            let x = k as f64;
            // k / n exactly, n a power of two.
            twiddles[last + k] = turn(x * inverse);
        }
        // The smaller stages' are every so many of the last's.
        let mut half = last / 2;
        while half > 0 {
            let stride = last / half;
            for k in 0..half {
                twiddles[half + k] = twiddles[last + k * stride];
            }
            half /= 2;
        }
        Self { n, twiddles }
    }

    /// The length.
    pub(crate) const fn len(&self) -> usize {
        self.n
    }

    /// Transforms `data` (of the length) in place, on up to `threads`
    /// threads (one to a block of work): first every stage within blocks
    /// that fit in a core's cache, a block to a thread, then the larger
    /// stages with each block's butterflies shared out.
    pub(crate) fn run(&self, data: &mut [Complex], threads: usize) {
        let n = self.n;
        assert_eq!(data.len(), n, "data of length {n}");
        let block = BLOCK.min(n);
        let threads = threads_for(threads, n, block);
        // Bit-reversed order, out of place.
        let bits = n.trailing_zeros();
        if bits > 0 {
            let source = data.to_vec();
            let source = &source;
            let chunk = n.div_ceil(threads);
            run_all(data.chunks_mut(chunk).enumerate().map(|(c, out)| {
                move || {
                    for (i, value) in out.iter_mut().enumerate() {
                        let j = (c * chunk + i).reverse_bits() >> (usize::BITS - bits);
                        *value = source[j];
                    }
                }
            }));
        }
        // The small stages, a block at a time.
        let per_thread = (n / block).div_ceil(threads) * block;
        run_all(data.chunks_mut(per_thread).map(|part| {
            move || {
                for b in part.chunks_mut(block) {
                    let mut half = 1;
                    while half < block {
                        self.stage(b, half);
                        half *= 2;
                    }
                }
            }
        }));
        // The large stages: whole pairs of halves to each thread while
        // there are enough of them, then each pair's butterflies shared out.
        let mut half = block;
        while half < n {
            let pairs = n / (2 * half);
            if pairs >= threads {
                run_all(
                    data.chunks_mut(pairs.div_ceil(threads) * 2 * half)
                        .map(|part| move || self.stage(part, half)),
                );
            } else {
                let share = half.div_ceil(threads / pairs);
                run_all(data.chunks_mut(2 * half).flat_map(|pair| {
                    let (lower, upper) = pair.split_at_mut(half);
                    lower
                        .chunks_mut(share)
                        .zip(upper.chunks_mut(share))
                        .enumerate()
                        .map(move |(k, (lo, hi))| move || self.butterflies(lo, hi, half, k * share))
                }));
            }
            half *= 2;
        }
    }

    /// Every butterfly of the stage of `half` within `data`.
    fn stage(&self, data: &mut [Complex], half: usize) {
        for pair in data.chunks_mut(2 * half) {
            let (lo, hi) = pair.split_at_mut(half);
            self.butterflies(lo, hi, half, 0);
        }
    }

    /// The butterflies of the stage of `half` pairing `lo[i]` with `hi[i]`,
    /// the first at offset `first` within its block.
    #[inline]
    fn butterflies(&self, lo: &mut [Complex], hi: &mut [Complex], half: usize, first: usize) {
        let twiddles = &self.twiddles[half + first..half + first + lo.len()];
        for ((a, b), &w) in lo.iter_mut().zip(hi.iter_mut()).zip(twiddles) {
            let t = b.mul(w);
            let x = *a;
            *a = Complex::new(x.re + t.re, x.im + t.im);
            *b = Complex::new(x.re - t.re, x.im - t.im);
        }
    }
}

/// The largest block whose stages run on one thread: 2^14 values, 256 KB.
const BLOCK: usize = 1 << 14;

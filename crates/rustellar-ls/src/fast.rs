//! The periodogram on an even grid by non-uniform FFTs, to a tolerance:
//! each frequency's sums from three transforms, a bound on each power's
//! error from the transforms', and the exact fit wherever the bound is not
//! met.

use crate::fft::Complex;
use crate::nufft::Nufft;
use crate::parallel::{run_all, threads_for};
use crate::periodogram::{Buffer, Fit, GRAIN, has_avx2, solve};
use crate::sums::{phase, split, two_product_split};
use crate::{Frequencies, LombScargle, LsError, Mean, Normalization, Periodogram};

/// The smallest tolerance asked of the transforms, near their limit.
const SMALLEST: f64 = 1e-14;

impl LombScargle {
    /// The power at each frequency of an even grid, each within `tolerance`
    /// of the exact power in the standard normalisation (a fraction of the
    /// reference chi-squared), in `O(n + m log m)` for `n` points and `m`
    /// frequencies.
    ///
    /// The sums of the fit at every frequency come from three non-uniform
    /// FFTs (Leroy 2012) with the kernel of Barnett, Magland & af
    /// Klinteberg (2019), each point's phase exact; each power's error is
    /// bounded from the transforms' error, and where the bound exceeds
    /// `tolerance` (the lowest frequencies, where the fit's terms nearly
    /// coincide, and degenerate ones) the power is computed exactly
    /// instead, as [`power`](Self::power). So every power is within
    /// `tolerance`, however the data fall. The other normalisations come
    /// from the same sums: the model and log ones divide by the chi-squared
    /// left, and are as accurate as that.
    ///
    /// # Errors
    ///
    /// - [`LsError::NotUniform`] unless `frequencies` is an even grid
    ///   ([`Frequencies::uniform`]).
    /// - [`LsError::NotPositive`] or [`LsError::NonFinite`] if `tolerance`
    ///   is not above zero and finite.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustellar_ls::{Frequencies, LightCurve, LombScargle, Mean, Normalization};
    ///
    /// let time: Vec<f64> = (0..2000).map(|i| 0.015 * f64::from(i) + 1e-4 * f64::from(i % 3)).collect();
    /// let flux = time.iter().map(|t| (std::f64::consts::TAU * 1.3 * t).sin() + 0.1 * (t * 7.7).sin()).collect();
    /// let lc = LightCurve::from_days(time, flux, vec![0.1; 2000])?;
    /// let ls = LombScargle::new(&lc, Mean::Floating)?;
    /// let grid = Frequencies::uniform(0.01, 0.002, 5000)?;
    ///
    /// let fast = ls.power_fast(&grid, Normalization::Standard, 1e-10)?;
    /// let exact = ls.power(&grid, Normalization::Standard);
    /// for (a, b) in fast.power().iter().zip(exact.power()) {
    ///     assert!((a - b).abs() <= 1e-10);
    /// }
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn power_fast(
        &self,
        frequencies: &Frequencies,
        normalization: Normalization,
        tolerance: f64,
    ) -> Result<Periodogram, LsError> {
        if !tolerance.is_finite() {
            return Err(LsError::NonFinite { name: "tolerance" });
        }
        if tolerance <= 0.0 {
            return Err(LsError::NotPositive { name: "tolerance" });
        }
        let Some(step) = frequencies.uniform_step() else {
            return Err(LsError::NotUniform);
        };
        let count = frequencies.len();
        let start = frequencies.min();
        // The transforms to a hundredth of the tolerance: a power's error
        // is a few times theirs where the fit is well conditioned.
        let accuracy = (0.01 * tolerance).max(SMALLEST);
        let threads = self.threads.get();
        let plan = Nufft::new(count, accuracy, threads);
        let half = plan.modes() / 2;
        // Frequencies `centre + m step`, m from -half: f_k at m = k - half.
        #[expect(clippy::cast_precision_loss, reason = "at most a billion")]
        let centre = start + half as f64 * step;
        let sums = self.transforms(&plan, centre, step);
        let bound = 2.0 * accuracy;
        let mut power = vec![f64::NAN; count];
        let mut exact = Vec::new();
        for (k, p) in power.iter_mut().enumerate() {
            match self.fit_from_sums(&sums, k, bound, tolerance) {
                Some(fit) => *p = self.normalized(fit, normalization),
                None => exact.push(k),
            }
        }
        self.fill_exact(frequencies, normalization, &exact, &mut power);
        Ok(Periodogram {
            frequencies: frequencies.clone(),
            power,
            normalization,
        })
    }

    /// The three transforms' sums at each mode: `sum w e`, `sum w y e` at
    /// the frequencies and `sum w e` at twice them, `e` the turn of each
    /// point at the frequency.
    fn transforms(&self, plan: &Nufft, centre: f64, step: f64) -> Sums {
        let n = self.time.len();
        let mut once = Vec::with_capacity(n);
        let mut twice = Vec::with_capacity(n);
        let mut w = Vec::with_capacity(n);
        let mut wy = Vec::with_capacity(n);
        let mut w2 = Vec::with_capacity(n);
        let (step_halves, centre_halves) = (split(step), split(centre));
        let centre2 = 2.0 * centre;
        let centre2_halves = split(centre2);
        for ((&t, &th), (&wi, &y)) in self
            .time
            .iter()
            .zip(&self.halves)
            .zip(self.weight.iter().zip(&self.flux))
        {
            // Each point's place on the transforms' circle, in cycles.
            once.push(cycles(step, step_halves, t, th));
            twice.push(cycles(
                2.0 * step,
                (2.0 * step_halves.0, 2.0 * step_halves.1),
                t,
                th,
            ));
            let e = value(phase(centre, centre_halves, t, th));
            let e2 = value(phase(centre2, centre2_halves, t, th));
            w.push(Complex::new(wi * e.re, wi * e.im));
            wy.push(Complex::new(wi * y * e.re, wi * y * e.im));
            w2.push(Complex::new(wi * e2.re, wi * e2.im));
        }
        let threads = self.threads.get();
        let [w, wy] = plan.run(&once, [&w, &wy], threads);
        let [w2] = plan.run(&twice, [&w2], threads);
        let weight_sum = self.weight.iter().sum();
        let flux_sum = self
            .weight
            .iter()
            .zip(&self.flux)
            .map(|(w, y)| w * y.abs())
            .sum();
        Sums {
            w,
            wy,
            w2,
            weight_sum,
            flux_sum,
        }
    }

    /// The fit at mode `k` from the sums, or `None` if its bound on the
    /// standard power's error exceeds `tolerance` (with the sums' error
    /// `bound` times the strengths' sizes).
    fn fit_from_sums(&self, sums: &Sums, k: usize, bound: f64, tolerance: f64) -> Option<Fit> {
        let (a, b, c) = (sums.w[k], sums.wy[k], sums.w2[k]);
        let w = sums.weight_sum;
        let (cc, ss, cs, yc, ys, d_matrix);
        // The sums' errors: the transforms' times their strengths' sizes.
        let (dw, dy) = (bound * w, bound * sums.flux_sum);
        if self.mean == Mean::Floating {
            let (mc, ms) = (a.re / w, a.im / w);
            cc = 0.5 * (w + c.re) - a.re * mc;
            ss = 0.5 * (w - c.re) - a.im * ms;
            cs = 0.5 * c.im - a.re * ms;
            d_matrix = 0.5 * dw + 2.0 * (mc.abs() + ms.abs()) * dw + dw * dw / w;
        } else {
            cc = 0.5 * (w + c.re);
            ss = 0.5 * (w - c.re);
            cs = 0.5 * c.im;
            d_matrix = 0.5 * dw;
        }
        (yc, ys) = (b.re, b.im);
        let trace = cc + ss;
        let det = cc * ss - cs * cs;
        // The fit is known only if the matrix's smaller eigenvalue is well
        // above its error.
        if det.is_nan() || det <= 0.0 || det / trace <= 4.0 * d_matrix {
            return None;
        }
        #[expect(clippy::cast_precision_loss, reason = "a point count")]
        let (beta, gamma) = solve(cc, cs, ss, yc, ys, self.time.len() as f64);
        let reduction = (beta * yc + gamma * ys).max(0.0);
        let size = beta.abs() + gamma.abs();
        // First order in the errors, doubled: 2 x . db + x . dA x.
        let error = 2.0 * (2.0 * size * dy + size * size * d_matrix);
        if error > tolerance * self.chi2_ref {
            return None;
        }
        Some(Fit {
            reduction,
            chi2: self.chi2_ref - reduction,
        })
    }

    /// The exact powers at the grid's `indices`, on all cores.
    fn fill_exact(
        &self,
        frequencies: &Frequencies,
        normalization: Normalization,
        indices: &[usize],
        power: &mut [f64],
    ) {
        if indices.is_empty() {
            return;
        }
        let residual = matches!(normalization, Normalization::Model | Normalization::Log);
        let wide = has_avx2();
        let threads = threads_for(self.threads.get(), indices.len() * self.time.len(), GRAIN);
        let chunk = indices.len().div_ceil(threads);
        let results = run_all(indices.chunks(chunk).map(|part| {
            move || {
                let mut buffer = Buffer::new(self.time.len());
                part.iter()
                    .map(|&k| {
                        let fit = self.fit(frequencies.get(k), residual, wide, &mut buffer);
                        self.normalized(fit, normalization)
                    })
                    .collect::<Vec<_>>()
            }
        }));
        for (&k, p) in indices.iter().zip(results.into_iter().flatten()) {
            power[k] = p;
        }
    }
}

/// The transforms' sums at each mode, and the strengths' sizes.
struct Sums {
    w: Vec<Complex>,
    wy: Vec<Complex>,
    w2: Vec<Complex>,
    weight_sum: f64,
    flux_sum: f64,
}

/// `f t` in cycles less the nearest whole number, from the exact product.
fn cycles(f: f64, f_halves: (f64, f64), t: f64, t_halves: (f64, f64)) -> f64 {
    let (hi, lo) = two_product_split(f, f_halves, t, t_halves);
    (hi - hi.round()) + lo
}

/// The turn from its terms' bases and parts.
fn value(((cb, cx), (sb, sx)): ((f64, f64), (f64, f64))) -> Complex {
    Complex::new(cb + cx, sb + sx)
}

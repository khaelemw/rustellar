//! Whitening a flattened light curve with an autoregressive model (as ARPS,
//! Caceres et al. 2019): each flux predicted from the few before it, and
//! the light curve replaced by what the prediction misses, so noise that is
//! correlated over a few cadences (which a sliding filter leaves) becomes
//! white.

use rustellar_core::LightCurve;

use crate::DetrendError;
use crate::double::{Double, accumulate, two_product};

/// The highest order [`Order::Aic`] tries by default.
pub const MAX_ORDER: usize = 30;

/// Refinement steps after the first least-squares solve.
const REFINEMENTS: usize = 2;

/// A diagonal of the triangular factor at or below this fraction of the
/// largest marks its column as dependent on the others: its coefficient is
/// left at zero.
const DEPENDENT: f64 = 1e-13;

/// How the model's order (the number of earlier fluxes each is predicted
/// from) is chosen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Order {
    /// This order.
    Fixed(usize),
    /// The order from 0 to this one with the least AIC, as statsmodels'
    /// `ar_select_order(ic="aic", trend="c")`: every order fitted to the
    /// same points (those with this many before them), then the one chosen
    /// fitted again to all the points it can predict.
    Aic(usize),
}

impl Default for Order {
    /// [`Order::Aic`] up to [`MAX_ORDER`].
    fn default() -> Self {
        Self::Aic(MAX_ORDER)
    }
}

/// An autoregressive model fitted by least squares,
///
/// `x_t = c + phi_1 x_(t-1) + ... + phi_p x_(t-p) + e_t`,
///
/// with the light curve split where its cadence breaks: a point is
/// predicted only from the `p` points before it in its run of regular
/// cadence (a step within half a cadence of the median step).
///
/// # Examples
///
/// ```
/// use rustellar_detrend::{Autoregressive, LightCurve, Order};
///
/// // Each flux half the one before plus noise (a fixed generator).
/// let mut state = 1_u64;
/// let mut noise = || {
///     state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
///     1e-3 * ((state >> 11) as f64 / (1_u64 << 53) as f64 - 0.5)
/// };
/// let mut flux = vec![1.0];
/// for i in 1..400 {
///     flux.push(0.5 + 0.5 * flux[i - 1] + noise());
/// }
/// let time: Vec<f64> = (0..400).map(|i| f64::from(i) / 720.0).collect();
/// let lc = LightCurve::from_btjd(time, flux, vec![1e-3; 400])?;
///
/// let white = Autoregressive::default().with_order(Order::Fixed(1)).whiten(&lc)?;
/// assert_eq!(white.order(), 1);
/// assert!((white.coefficients()[0] - 0.5).abs() < 0.05);
/// // The first point has nothing before it to be predicted from.
/// assert_eq!(white.indices()[0], 1);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Autoregressive {
    order: Order,
}

impl Autoregressive {
    /// With the order chosen by `order`.
    #[must_use]
    pub fn with_order(mut self, order: Order) -> Self {
        self.order = order;
        self
    }

    /// How the order is chosen.
    #[must_use]
    pub fn order(&self) -> Order {
        self.order
    }

    /// The model fitted to `lc` (a flattened light curve), and `lc`
    /// replaced by what it misses: `1 + e_t` at each point it predicts.
    ///
    /// A transit's dip of depth `d` comes out about `d (1 - phi_1 - ... -
    /// phi_p)` deep ([`gain`](Whitened::gain)) once inside it, with a
    /// deeper first point at ingress; a search finds it in white noise. The
    /// flux errors are the light curve's own.
    ///
    /// # Errors
    ///
    /// [`DetrendError::TooFewPoints`] if fewer points can be predicted
    /// than the model has coefficients (the constant and one per order),
    /// for the highest order tried.
    pub fn whiten(&self, lc: &LightCurve) -> Result<Whitened, DetrendError> {
        let flux = lc.flux();
        let starts = runs(lc);
        let enough = |p: usize| {
            let rows = predictable(&starts, p);
            if rows.len() > p + 1 {
                Ok(rows)
            } else {
                Err(DetrendError::TooFewPoints)
            }
        };
        let (order, rows) = match self.order {
            Order::Fixed(p) => (p, enough(p)?),
            // Every order tried on the points the highest can predict,
            // then the chosen one on all it can.
            Order::Aic(most) => {
                let p = aic_order(flux, &enough(most)?, most);
                (p, predictable(&starts, p))
            }
        };
        let beta = least_squares(flux, &rows, order);
        let (mut time, mut out, mut err) = (vec![], vec![], vec![]);
        let mut sum_squares = 0.0;
        for &t in &rows {
            let e = residual(flux, t, &beta);
            sum_squares += e * e;
            time.push(lc.time()[t]);
            out.push(1.0 + e);
            err.push(lc.flux_err()[t]);
        }
        #[expect(clippy::cast_precision_loss, reason = "a count of points")]
        let variance = sum_squares / rows.len() as f64;
        // The input's times, a subset, and its errors, with finite fluxes.
        let light_curve = LightCurve::new(time, out, err).expect("a valid light curve");
        Ok(Whitened {
            light_curve,
            indices: rows,
            constant: beta[0],
            coefficients: beta[1..].to_vec(),
            variance,
        })
    }
}

/// A light curve whitened by [`Autoregressive::whiten`], with its model.
#[derive(Debug, Clone, PartialEq)]
pub struct Whitened {
    light_curve: LightCurve,
    indices: Vec<usize>,
    constant: f64,
    coefficients: Vec<f64>,
    variance: f64,
}

impl Whitened {
    /// `1 + e_t` at each point the model predicts.
    #[must_use]
    pub fn light_curve(&self) -> &LightCurve {
        &self.light_curve
    }

    /// The light curve, taken out.
    #[must_use]
    pub fn into_light_curve(self) -> LightCurve {
        self.light_curve
    }

    /// For each point of the whitened light curve, its index in the
    /// original.
    #[must_use]
    pub fn indices(&self) -> &[usize] {
        &self.indices
    }

    /// The model's order `p`.
    #[must_use]
    pub fn order(&self) -> usize {
        self.coefficients.len()
    }

    /// The constant `c`.
    #[must_use]
    pub fn constant(&self) -> f64 {
        self.constant
    }

    /// `phi_1` to `phi_p`.
    #[must_use]
    pub fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }

    /// The mean square of `e_t` (statsmodels' `sigma2`).
    #[must_use]
    pub fn variance(&self) -> f64 {
        self.variance
    }

    /// `1 - phi_1 - ... - phi_p`: the share of a long, flat dip the
    /// whitened light curve keeps.
    #[must_use]
    pub fn gain(&self) -> f64 {
        1.0 - self.coefficients.iter().sum::<f64>()
    }
}

/// The index where each run of regular cadence starts, and the length as
/// the last: a point starts a run where its step from the one before is
/// more than half a cadence (the median step) from it.
fn runs(lc: &LightCurve) -> Vec<usize> {
    let time = lc.time();
    let mut steps: Vec<f64> = time.windows(2).map(|w| (w[1] - w[0]).as_days()).collect();
    let mut starts = vec![0];
    if !steps.is_empty() {
        let mid = steps.len() / 2;
        let (_, &mut cadence, _) = steps.select_nth_unstable_by(mid, f64::total_cmp);
        for (i, w) in time.windows(2).enumerate() {
            if ((w[1] - w[0]).as_days() - cadence).abs() > 0.5 * cadence {
                starts.push(i + 1);
            }
        }
    }
    starts.push(time.len());
    starts
}

/// The points with at least `order` points before them in their run.
fn predictable(starts: &[usize], order: usize) -> Vec<usize> {
    starts
        .windows(2)
        .flat_map(|w| (w[0] + order).min(w[1])..w[1])
        .collect()
}

/// `x_t - c - phi_1 x_(t-1) - ... - phi_p x_(t-p)` for `beta = [c, phi_1,
/// ..]`, its products exact and summed in double-double.
fn residual(flux: &[f64], t: usize, beta: &[f64]) -> f64 {
    let mut sum: Double = (flux[t], 0.0);
    sum = accumulate(sum, (-beta[0], 0.0));
    for (k, &phi) in beta[1..].iter().enumerate() {
        let (hi, lo) = two_product(phi, flux[t - 1 - k]);
        sum = accumulate(sum, (-hi, -lo));
    }
    sum.0 + sum.1
}

/// The row of the design at point `t`: `[1, x_(t-1), .., x_(t-p)]`.
fn design(flux: &[f64], t: usize, order: usize, row: &mut [f64]) {
    row[0] = 1.0;
    for k in 0..order {
        row[k + 1] = flux[t - 1 - k];
    }
}

/// The upper-triangular factor of the design (columns: the constant, then
/// the lags up to `order`) and of the targets, by Givens rotations row by
/// row, and the sum of squares the rotations leave over.
struct Triangle {
    size: usize,
    /// Row-major, `size` by `size + 1`: the factor and the rotated
    /// targets in the last column.
    r: Vec<f64>,
    rest: f64,
}

impl Triangle {
    fn new(size: usize) -> Self {
        Self {
            size,
            r: vec![0.0; size * (size + 1)],
            rest: 0.0,
        }
    }

    /// Rotates `row` (`size` design values and the target) in.
    fn add(&mut self, row: &mut [f64]) {
        let width = self.size + 1;
        for j in 0..self.size {
            let x = row[j];
            if x == 0.0 {
                continue;
            }
            let d = self.r[j * width + j];
            let h = d.hypot(x);
            let (c, s) = (d / h, x / h);
            self.r[j * width + j] = h;
            row[j] = 0.0;
            let rest = &mut self.r[j * width + j + 1..(j + 1) * width];
            for (a, b) in rest.iter_mut().zip(&mut row[j + 1..width]) {
                (*a, *b) = (c * *a + s * *b, c * *b - s * *a);
            }
        }
        self.rest += row[self.size] * row[self.size];
    }

    fn at(&self, i: usize, j: usize) -> f64 {
        self.r[i * (self.size + 1) + j]
    }

    /// The sum of squared residuals of the fit with the first `columns`
    /// columns only.
    fn sum_squares(&self, columns: usize) -> f64 {
        let tail: f64 = (columns..self.size)
            .map(|j| self.at(j, self.size).powi(2))
            .sum();
        self.rest + tail
    }

    /// Which columns depend on those before them.
    fn dependent(&self) -> Vec<bool> {
        let largest = (0..self.size)
            .map(|j| self.at(j, j).abs())
            .fold(0.0, f64::max);
        (0..self.size)
            .map(|j| self.at(j, j).abs() <= DEPENDENT * largest)
            .collect()
    }

    /// `R x = b` by back substitution, dependent columns left at zero.
    fn back(&self, b: &mut [f64], dependent: &[bool]) {
        for j in (0..self.size).rev() {
            if dependent[j] {
                b[j] = 0.0;
                continue;
            }
            let row = &self.r[j * (self.size + 1)..];
            let v = (j + 1..self.size).fold(b[j], |v, k| v - row[k] * b[k]);
            b[j] = v / self.at(j, j);
        }
    }

    /// `R^T x = b` by forward substitution, dependent columns left at zero.
    fn forward(&self, b: &mut [f64], dependent: &[bool]) {
        for j in 0..self.size {
            if dependent[j] {
                b[j] = 0.0;
                continue;
            }
            let v = b[..j]
                .iter()
                .enumerate()
                .fold(b[j], |v, (k, &bk)| v - self.at(k, j) * bk);
            b[j] = v / self.at(j, j);
        }
    }
}

/// The triangle of the fit of order `order` on `rows`.
fn triangle(flux: &[f64], rows: &[usize], order: usize) -> Triangle {
    let mut tri = Triangle::new(order + 1);
    let mut row = vec![0.0; order + 2];
    for &t in rows {
        design(flux, t, order, &mut row);
        row[order + 1] = flux[t];
        tri.add(&mut row);
    }
    tri
}

/// The order from 0 to `most` with the least AIC, every order fitted to
/// `rows`: `n ln(ssr / n) + 2 (p + 2)` (statsmodels' `-2 llf + 2 (k + 1)`
/// less a constant), the lowest order on a tie.
fn aic_order(flux: &[f64], rows: &[usize], most: usize) -> usize {
    let tri = triangle(flux, rows, most);
    #[expect(clippy::cast_precision_loss, reason = "counts")]
    let n = rows.len() as f64;
    let mut best = (f64::INFINITY, 0);
    for p in 0..=most {
        #[expect(clippy::cast_precision_loss, reason = "a small order")]
        let aic = n * (tri.sum_squares(p + 1) / n).ln() + 2.0 * (p as f64 + 2.0);
        if aic < best.0 {
            best = (aic, p);
        }
    }
    best.1
}

/// `[c, phi_1, .., phi_p]` by least squares on `rows`: QR by Givens
/// rotations, then refinement steps on the residuals (exact products,
/// summed in double-double) by the corrected semi-normal equations.
fn least_squares(flux: &[f64], rows: &[usize], order: usize) -> Vec<f64> {
    let tri = triangle(flux, rows, order);
    let dependent = tri.dependent();
    let mut beta: Vec<f64> = (0..=order).map(|j| tri.at(j, order + 1)).collect();
    tri.back(&mut beta, &dependent);
    let mut row = vec![0.0; order + 1];
    for _ in 0..REFINEMENTS {
        let mut gradient = vec![(0.0, 0.0); order + 1];
        for &t in rows {
            let r = residual(flux, t, &beta);
            design(flux, t, order, &mut row);
            for (g, &a) in gradient.iter_mut().zip(&row) {
                *g = accumulate(*g, two_product(a, r));
            }
        }
        let mut step: Vec<f64> = gradient.iter().map(|g| g.0 + g.1).collect();
        tri.forward(&mut step, &dependent);
        tri.back(&mut step, &dependent);
        for (b, s) in beta.iter_mut().zip(&step) {
            *b += s;
        }
    }
    beta
}

//! The transit fit's Jacobian from the geometry.
//!
//! The flux at a point depends on the parameters only through the
//! separation `z`, the radius ratio `k` and the limb darkening. So:
//!
//! - `t0`, `P`, `a / R*` and `b` by the chain rule through `z`, from
//!   `z^2 = (a sin phi)^2 + (b cos phi)^2` and `phi = 2 pi (t - t0) / P`;
//! - `u1` and `u2` exactly, since the flux is linear in the three parts of
//!   the blocked light ([`blocked_light`]) with weights set by them;
//! - the baseline exactly: the model itself;
//! - only `dF/dz` and `dF/dk` by central differences, of the blocked
//!   light's parts.
//!
//! Five evaluations of the blocked light per point in transit, against two
//! light curves per free parameter by differencing them, and every
//! derivative but two free of differencing error. With an exposure each
//! sample's derivatives are averaged, as its flux is.

use std::f64::consts::TAU;
use std::thread;

use rustellar_model::{RadiusRatio, blocked_light};

use crate::Parameter;
use crate::parameter::{BASE, COUNT};
use crate::problem::{BLOCK, Problem, STEP};

/// What every row's derivatives need at one set of values.
struct Geometry {
    t0: f64,
    period: f64,
    a: f64,
    b: f64,
    u2: f64,
    omega: f64,
    c2: f64,
    hz: f64,
    hk: f64,
    rk: RadiusRatio,
    up: RadiusRatio,
    down: RadiusRatio,
    reach: f64,
    clear: f64,
    spread: f64,
    samples: f64,
}

impl Geometry {
    /// `None` where `k` is too close to its limits to be differenced.
    fn new(values: &[f64; COUNT], offsets: &[f64]) -> Option<Self> {
        let [t0, period, k, a, b, u1, u2, ..] = *values;
        let (hz, hk) = (STEP * k, STEP * k);
        let (Ok(rk), Ok(up), Ok(down)) = (
            RadiusRatio::new(k),
            RadiusRatio::new(k + hk),
            RadiusRatio::new(k - hk),
        ) else {
            return None;
        };
        let reach = 1.0 + k + hz + hk;
        #[allow(clippy::cast_precision_loss)]
        let samples = offsets.len() as f64;
        Some(Self {
            t0,
            period,
            a,
            b,
            u2,
            omega: 1.0 - u1 / 3.0 - u2 / 6.0,
            c2: u1 + 2.0 * u2,
            hz,
            hk,
            rk,
            up,
            down,
            reach,
            // Out of transit, and its derivatives zero, beyond this phase
            // angle: the separation is at least a |sin phi|.
            clear: if reach < a {
                (reach / a).asin()
            } else {
                TAU / 4.0
            },
            // The furthest a sample is from its point, as a phase angle.
            spread: offsets.iter().fold(0.0f64, |s, o| s.max(o.abs())) * TAU / period,
            samples,
        })
    }

    /// The blocked light times the disc's integral, for this law.
    fn light(&self, [b0, b1, b2]: [f64; 3]) -> f64 {
        (1.0 - self.c2).mul_add(b0, self.c2.mul_add(b1, self.u2 * b2))
    }

    /// dF/d(t0, P, k, a, b, u1, u2), zero for the eccentricity, and F, summed over the samples of the
    /// point at `time`.
    fn row(&self, time: f64, offsets: &[f64]) -> [f64; COUNT] {
        let Self {
            t0,
            period,
            a,
            b,
            omega,
            hz,
            hk,
            rk,
            up,
            down,
            reach,
            clear,
            spread,
            samples,
            ..
        } = *self;
        let mut d = [0.0; COUNT];
        let middle = TAU * (((time - t0) / period + 0.5).rem_euclid(1.0) - 0.5);
        if middle.abs() - spread > clear {
            d[BASE] = samples;
            return d;
        }
        for &offset in offsets {
            let cycles = (time + offset - t0) / period;
            let phi = TAU * ((cycles + 0.5).rem_euclid(1.0) - 0.5);
            if phi.abs() > clear {
                d[BASE] += 1.0;
                continue;
            }
            let (sin, cos) = phi.sin_cos();
            let (x, y) = (a * sin, b * cos);
            let z = x.mul_add(x, y * y).sqrt();
            if cos <= 0.0 || z >= reach {
                d[BASE] += 1.0;
                continue;
            }
            let parts = blocked_light(z, rk);
            let l = self.light(parts);
            d[BASE] += 1.0 - l / omega;
            let dl_dz = (self.light(blocked_light(z + hz, rk))
                - self.light(blocked_light(z - hz, rk)))
                / (2.0 * hz);
            let dl_dk = (self.light(blocked_light(z, up)) - self.light(blocked_light(z, down)))
                / (2.0 * hk);
            let df_dz = -dl_dz / omega;
            d[2] -= dl_dk / omega;
            // d(l / omega) with d omega / d u1 = -1/3, d omega / d u2 = -1/6.
            let slope = parts[1] - parts[0];
            d[5] -= slope.mul_add(omega, l / 3.0) / (omega * omega);
            d[6] -= 2.0f64.mul_add(slope, parts[2]).mul_add(omega, l / 6.0) / (omega * omega);
            if z > 0.0 {
                let per_z = df_dz / z;
                let dz_dphi = (a - b) * (a + b) * sin * cos;
                d[0] -= per_z * dz_dphi * TAU / period;
                d[1] -= per_z * dz_dphi * TAU * cycles / period;
                d[3] += per_z * a * sin * sin;
                d[4] += per_z * b * cos * cos;
            }
        }
        d
    }
}

/// One thread's rows: those of blocks `t`, `t + threads`, ... of `BLOCK`,
/// the baseline's column written into `base` (that thread's blocks of it,
/// if the baseline is free), each row with shape derivatives pushed to
/// `out`.
#[allow(clippy::too_many_arguments)]
fn fill(
    g: &Geometry,
    times: &[f64],
    offsets: &[f64],
    weight: &[f64],
    t: usize,
    threads: usize,
    mut base: Vec<&mut [f64]>,
    out: &mut Vec<(usize, [f64; COUNT])>,
) {
    out.clear();
    let blocks = (t * BLOCK..times.len()).step_by(threads * BLOCK);
    for (b, start) in blocks.enumerate() {
        let end = (start + BLOCK).min(times.len());
        for i in start..end {
            let d = g.row(times[i], offsets);
            if let Some(block) = base.get_mut(b) {
                block[i - start] = d[BASE] * (weight[i] / g.samples);
            }
            if d[..BASE].iter().any(|&x| x != 0.0) {
                out.push((i, d));
            }
        }
    }
}

/// The Jacobian of `problem`'s residuals at `values` into `jac` (its free
/// columns, column-major), or `false` where `k` is too close to its limits
/// to be differenced. Out of transit a row is zero but for the baseline,
/// so only the rows in transit are written, after clearing the last call's
/// (every row, after a Jacobian from elsewhere). Each point's row is the
/// same on any number of threads.
pub(crate) fn jacobian(problem: &mut Problem<'_>, values: &[f64; COUNT], jac: &mut [f64]) -> bool {
    // Circular orbits only: an eccentric one's is differenced.
    let eccentric = [Parameter::ECosW, Parameter::ESinW]
        .iter()
        .any(|p| values[p.index()] != 0.0 || problem.free.contains(&p.index()));
    if eccentric {
        return false;
    }
    let Some(g) = Geometry::new(values, &problem.offsets) else {
        return false;
    };
    let baseline = values[BASE];
    let m = problem.times.len();
    let Problem {
        times,
        offsets,
        weight,
        free,
        rows_out,
        active,
        written,
        ..
    } = problem;
    let threads = rows_out.len();
    let shape: Vec<(usize, usize)> = free
        .iter()
        .enumerate()
        .filter(|&(_, &p)| p != BASE)
        .map(|(c, &p)| (c, p))
        .collect();
    if *written {
        for &(c, _) in &shape {
            for &i in active.iter() {
                jac[c * m + i] = 0.0;
            }
        }
    } else {
        for &(c, _) in &shape {
            jac[c * m..(c + 1) * m].fill(0.0);
        }
    }
    // The baseline is the last parameter, so its column, if free, is the
    // last: the shape columns are all before it.
    let (shape_columns, base_column) = jac.split_at_mut(shape.len() * m);
    let mut base_part: Vec<Vec<&mut [f64]>> = (0..threads).map(|_| Vec::new()).collect();
    for (b, block) in base_column.chunks_mut(BLOCK).enumerate() {
        base_part[b % threads].push(block);
    }
    if threads == 1 {
        let base = base_part.into_iter().next().unwrap_or_default();
        fill(&g, times, offsets, weight, 0, 1, base, &mut rows_out[0]);
    } else {
        thread::scope(|scope| {
            for ((t, out), base) in rows_out.iter_mut().enumerate().zip(base_part) {
                let (g, times, offsets, weight) = (&g, &*times, &*offsets, &*weight);
                scope.spawn(move || fill(g, times, offsets, weight, t, threads, base, out));
            }
        });
    }
    active.clear();
    for out in rows_out.iter() {
        for &(i, d) in out {
            let w = weight[i] / g.samples;
            for &(c, p) in &shape {
                shape_columns[c * m + i] = baseline * d[p] * w;
            }
            active.push(i);
        }
    }
    *written = true;
    true
}

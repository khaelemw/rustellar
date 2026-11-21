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

use rustellar_model::{RadiusRatio, blocked_light};

use crate::problem::{Problem, STEP};

/// The Jacobian of `problem`'s residuals at `values` into `jac` (its free
/// columns, column-major), or `false` where `k` is too close to its limits
/// to be differenced.
pub(crate) fn jacobian(problem: &Problem<'_>, values: &[f64; 8], jac: &mut [f64]) -> bool {
    let [t0, period, k, a, b, u1, u2, baseline] = *values;
    let (hz, hk) = (STEP * k, STEP * k);
    let (Ok(rk), Ok(up), Ok(down)) = (
        RadiusRatio::new(k),
        RadiusRatio::new(k + hk),
        RadiusRatio::new(k - hk),
    ) else {
        return false;
    };
    let omega = 1.0 - u1 / 3.0 - u2 / 6.0;
    let c2 = u1 + 2.0 * u2;
    // The blocked light times the disc's integral, for this law.
    let light = |[b0, b1, b2]: [f64; 3]| (1.0 - c2).mul_add(b0, c2.mul_add(b1, u2 * b2));
    let reach = 1.0 + k + hz + hk;
    // Out of transit, and its derivatives zero, beyond this phase angle:
    // the separation is at least a |sin phi|.
    let clear = if reach < a {
        (reach / a).asin()
    } else {
        TAU / 4.0
    };
    // The furthest a sample is from its point, as a phase angle.
    let spread = problem.offsets.iter().fold(0.0f64, |s, o| s.max(o.abs())) * TAU / period;
    let m = problem.times.len();
    #[allow(clippy::cast_precision_loss)]
    let samples = problem.offsets.len() as f64;

    for (i, &time) in problem.times.iter().enumerate() {
        // dF/d(t0, P, k, a, b, u1, u2), and F, summed over the samples.
        let mut d = [0.0; 8];
        let middle = TAU * (((time - t0) / period + 0.5).rem_euclid(1.0) - 0.5);
        if middle.abs() - spread > clear {
            d[7] = samples;
        }
        for &offset in &problem.offsets {
            if d[7] == samples {
                break;
            }
            let cycles = (time + offset - t0) / period;
            let phi = TAU * ((cycles + 0.5).rem_euclid(1.0) - 0.5);
            if phi.abs() > clear {
                d[7] += 1.0;
                continue;
            }
            let (sin, cos) = phi.sin_cos();
            let (x, y) = (a * sin, b * cos);
            let z = x.mul_add(x, y * y).sqrt();
            if cos <= 0.0 || z >= reach {
                d[7] += 1.0;
                continue;
            }
            let parts = blocked_light(z, rk);
            let l = light(parts);
            d[7] += 1.0 - l / omega;
            let dl_dz =
                (light(blocked_light(z + hz, rk)) - light(blocked_light(z - hz, rk))) / (2.0 * hz);
            let dl_dk = (light(blocked_light(z, up)) - light(blocked_light(z, down))) / (2.0 * hk);
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
        let w = problem.weight[i] / samples;
        for (column, &p) in problem.free.iter().enumerate() {
            jac[column * m + i] = if p == 7 {
                d[7] * w
            } else {
                baseline * d[p] * w
            };
        }
    }
    true
}

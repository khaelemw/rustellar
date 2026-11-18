// The survey light curves' recipe, shared bit for bit with the reference
// generator: only + - * / and floor, which IEEE 754 rounds the same
// everywhere. Used by the survey tests and the benchmarks.

use rustellar_core::{LightCurve, TimeDelta};
use rustellar_detrend::Detrend;

// SplitMix64.
pub(crate) struct Random(pub(crate) u64);

impl Random {
    pub(crate) fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut x = self.0;
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        x ^ (x >> 31)
    }

    // A multiple of 2^-53 in [0, 1), exactly.
    pub(crate) fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1_u64 << 53) as f64
    }
}

/// One curve's recipe, with the generator's fingerprint of it.
#[derive(Clone)]
pub(crate) struct Curve {
    pub(crate) name: String,
    pub(crate) seed: u64,
    pub(crate) n: u32,
    pub(crate) t_start: f64,
    pub(crate) cadence: f64,
    pub(crate) keep: f64,
    pub(crate) gap_start: f64,
    pub(crate) gap: f64,
    pub(crate) scale: f64,
    pub(crate) rotation: f64,
    pub(crate) amplitude: f64,
    pub(crate) drift: f64,
    pub(crate) noise: f64,
    pub(crate) flare_rate: f64,
    pub(crate) flare: f64,
    pub(crate) tau: f64,
    pub(crate) period: f64,
    pub(crate) tc: f64,
    pub(crate) duration: f64,
    pub(crate) depth: f64,
    pub(crate) window: f64,
    pub(crate) points: usize,
    pub(crate) fingerprint: u64,
}

/// The kept cadences' times, fluxes and noise-free model.
pub(crate) struct Built {
    pub(crate) time: Vec<f64>,
    pub(crate) flux: Vec<f64>,
    pub(crate) model: Vec<f64>,
}

impl Curve {
    pub(crate) fn in_box(&self, x: f64) -> bool {
        let k = ((x - self.tc) / self.period + 0.5).floor();
        let r = (x - self.tc) - k * self.period;
        r.abs() < self.duration / 2.0
    }

    // Spots: a Catmull-Rom spline through random knots a quarter rotation
    // apart, and a linear drift; noise, the sum of four uniforms; flares
    // decaying as 1 / (1 + dt / tau)^2; box transits; cadences dropped at
    // random and in a gap. Written as the generator writes it, operation
    // for operation.
    pub(crate) fn build(&self) -> Built {
        let mut knots_rng = Random(self.seed ^ 0x5eed_5eed_5eed_5eed);
        let span = self.cadence * f64::from(self.n);
        let h = self.rotation / 4.0;
        let knots: Vec<f64> = (0..(span / h).floor() as usize + 4)
            .map(|_| self.amplitude * (2.0 * knots_rng.unit() - 1.0))
            .collect();
        let mut rng = Random(self.seed);
        let mut flares: Vec<(f64, f64)> = Vec::new();
        let (mut time, mut flux, mut model) = (vec![], vec![], vec![]);
        for i in 0..self.n {
            let mut u = [0.0; 7];
            for x in &mut u {
                *x = rng.unit();
            }
            let [u_keep, n1, n2, n3, n4, u_flare, u_amp] = u;
            let x = self.cadence * f64::from(i);
            if u_flare < self.flare_rate {
                flares.push((x, self.flare * (0.2 + u_amp)));
            }
            if u_keep >= self.keep || (self.gap_start <= x && x < self.gap_start + self.gap) {
                continue;
            }
            let s = x / h + 1.0;
            let k = s.floor();
            let f = s - k;
            let k = k as usize;
            let (p0, p1, p2, p3) = (knots[k - 1], knots[k], knots[k + 1], knots[k + 2]);
            let spline = 0.5
                * (2.0 * p1
                    + f * ((p2 - p0)
                        + f * ((2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3)
                            + f * (3.0 * (p1 - p2) + p3 - p0))));
            let m = self.scale * (1.0 + spline + self.drift * (x / span));
            let mut flare = 0.0;
            for &(x0, a) in &flares {
                if x - x0 < 30.0 * self.tau {
                    let q = 1.0 + (x - x0) / self.tau;
                    flare += a / (q * q);
                }
            }
            let noise = self.noise * (((n1 + n2) + (n3 + n4)) - 2.0);
            let dip = if self.in_box(x) { self.depth } else { 0.0 };
            time.push(self.t_start + x);
            flux.push(m * (((1.0 + noise) - dip) + flare));
            model.push(m);
        }
        Built { time, flux, model }
    }

    pub(crate) fn light_curve(&self, time: &[f64], flux: &[f64]) -> LightCurve {
        let error = vec![self.scale * self.noise; time.len()];
        LightCurve::from_btjd(time.to_vec(), flux.to_vec(), error).unwrap()
    }

    pub(crate) fn detrend(&self) -> Detrend {
        Detrend::default().with_window(TimeDelta::from_days(self.window).unwrap())
    }
}

pub(crate) fn curves() -> Vec<Curve> {
    include_str!("../reference/survey_curves.csv")
        .lines()
        .filter(|l| !l.starts_with('#'))
        .skip(1)
        .map(|line| {
            let f: Vec<&str> = line.split(',').collect();
            let x = |k: usize| f[k].parse::<f64>().unwrap();
            Curve {
                name: f[0].to_string(),
                seed: f[1].parse().unwrap(),
                n: f[2].parse().unwrap(),
                t_start: x(3),
                cadence: x(4),
                keep: x(5),
                gap_start: x(6),
                gap: x(7),
                scale: x(8),
                rotation: x(9),
                amplitude: x(10),
                drift: x(11),
                noise: x(12),
                flare_rate: x(13),
                flare: x(14),
                tau: x(15),
                period: x(16),
                tc: x(17),
                duration: x(18),
                depth: x(19),
                window: x(20),
                points: f[22].parse().unwrap(),
                fingerprint: f[24].parse().unwrap(),
            }
        })
        .collect()
}

/// FNV-1a over the bits of every time and flux.
pub(crate) fn fingerprint(time: &[f64], flux: &[f64]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for (t, f) in time.iter().zip(flux) {
        for v in [t, f] {
            h = (h ^ v.to_bits()).wrapping_mul(0x0100_0000_01b3);
        }
    }
    h
}

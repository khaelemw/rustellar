/// A compensated sum (Neumaier's improvement of Kahan's): the rounding error
/// of each addition is kept and added back at the end, so a long sum is as
/// accurate as if it were done in twice the precision, whatever the order
/// of the terms' sizes.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Sum {
    sum: f64,
    error: f64,
}

impl Sum {
    pub(crate) fn add(&mut self, x: f64) {
        let t = self.sum + x;
        self.error += if self.sum.abs() >= x.abs() {
            (self.sum - t) + x
        } else {
            (x - t) + self.sum
        };
        self.sum = t;
    }

    pub(crate) fn get(self) -> f64 {
        self.sum + self.error
    }
}

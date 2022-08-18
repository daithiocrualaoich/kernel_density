//! Normal kernel density estimation functions.

use density::Density;
use std::f64::consts::PI;

/** https://en.wikipedia.org/wiki/Error_function#Numerical_approximations */
fn erf_compute(z: f64) -> f64 {
    if z > 9.231948545 {
        return 1.0;
    } else if z < -9.231948545 {
        return -1.0;
    }
    let a1 = 0.0705230784;
    let a2 = 0.0422820123;
    let a3 = 0.0092705272;
    let a4 = 0.0001520143;
    let a5 = 0.0002765672;
    let a6 = 0.0000430638;
    let denom = (1.0
        + a1 * z
        + a2 * z.powf(2.0)
        + a3 * z.powf(3.0)
        + a4 * z.powf(4.0)
        + a5 * z.powf(5.0)
        + a6 * z.powf(6.0))
    .powf(16.0);
    1.0 - 1.0 / denom
}

fn erf(z: f64) -> f64 {
    if z < 0.0 {
        -erf_compute(-z)
    } else {
        erf_compute(z)
    }
}

fn norm(x: f64) -> f64 {
    let z = x / (2.0_f64).sqrt();
    (1.0 + erf(z)) / 2.0
}

pub struct NormalDensity {
    pub mean: f64,
    pub variance: f64,
}

impl Density for NormalDensity {
    /// Calculate a value of the normal density function for a given value.
    ///
    /// # Examples
    ///
    /// ```
    /// extern crate kernel_density;
    ///
    /// let mean = 0.0;
    /// let variance = 1.0;
    /// let normal = kernel_density::density::normal(mean, variance);
    ///
    /// assert_eq!(normal.density(0.0), 0.3989422804014327);
    /// ```
    fn density(&self, x: f64) -> f64 {
        let coefficient = 1.0 / (2.0 * PI * self.variance).sqrt();
        let exponent = -(x - self.mean).powi(2) / (2.0 * self.variance);

        coefficient * exponent.exp()
    }

    /// Calculate a value of the cumulative density function for this normal
    /// density.
    ///
    /// # Examples
    ///
    /// ```
    /// extern crate kernel_density;
    ///
    /// let mean = 0.0;
    /// let variance = 1.0;
    /// let normal = kernel_density::density::normal(mean, variance);
    ///
    /// assert_eq!(normal.cdf(0.0), 0.5);
    /// ```
    fn cdf(&self, x: f64) -> f64 {
        let z: f64 = (x - self.mean) / self.variance.sqrt();
        norm(z)
    }
}

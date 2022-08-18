//! Normal kernel density estimation functions.

use density::Density;
use std::f64::consts::PI;

/** https://en.wikipedia.org/wiki/Error_function#Numerical_approximations */
fn erf(z: f64) -> f64 {
    assert!(z >= 0.0);
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

fn norm(x: f64) -> f64 {
    let z = x / 2f64.sqrt();
    (1.0 + erf(z)) / 2.0
}

pub struct NormalKernelDensityEstimation {
    pub samples: Vec<f64>,
    pub bandwidth: f64,
}

impl Density for NormalKernelDensityEstimation {
    /// Calculate a value of the kernel density function for a given value.
    ///
    /// # Examples
    ///
    /// ```
    /// extern crate kernel_density;
    ///
    /// let samples = vec!(9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0);
    /// let bandwidth = 0.1;
    /// let kde = kernel_density::kde::normal(&samples, bandwidth);
    ///
    /// assert_eq!(kde.density(4.0), 0.3989422804014327);
    /// ```
    fn density(&self, x: f64) -> f64 {
        let length = self.samples.len();

        let mut sum = 0.0;
        for sample in &self.samples {
            let rescaled: f64 = (x - sample) / self.bandwidth;
            sum += (-0.5 * rescaled.powi(2)).exp()
        }

        let sqrt_2pi = (2.0 * PI).sqrt();
        sum / (sqrt_2pi * length as f64 * self.bandwidth)
    }

    /// Calculate a value of the cumulative density function for this kernel
    /// density estimation.
    ///
    /// # Examples
    ///
    /// ```
    /// extern crate kernel_density;
    ///
    /// let samples = vec!(9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0);
    /// let bandwidth = 0.1;
    /// let kde = kernel_density::kde::normal(&samples, bandwidth);
    ///
    /// assert_eq!(kde.cdf(0.1), 0.08413447460685429);
    /// ```
    fn cdf(&self, x: f64) -> f64 {
        let length = self.samples.len();

        let mut sum = 0.0;
        for sample in &self.samples {
            let rescaled: f64 = (x - sample) / self.bandwidth;
            sum += norm(rescaled);
        }

        sum / length as f64
    }
}

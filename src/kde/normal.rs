//! Normal kernel density estimation functions.

extern crate special_fun;

use density::Density;
use self::special_fun::FloatSpecial;
use std::f64::consts::PI;

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
            sum += rescaled.norm();
        }

        sum / length as f64
    }
}

//! Normal kernel density estimation functions.

extern crate special_fun;

use self::special_fun::FloatSpecial;
use std::f64::consts;

pub struct NormalKernelDensityEstimation {
    samples: Vec<f64>,
}

impl NormalKernelDensityEstimation {
    /// Construct a kernel density estimation for a given sample. Uses the
    /// Normal kernel.
    ///
    /// k(x) = exp(0.5 * x^2)/(2*pi)
    ///
    /// # Panics
    ///
    /// The sample set must be non-empty.
    ///
    /// # Examples
    ///
    /// ```
    /// extern crate kernel_density;
    ///
    /// let samples = vec!(9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0);
    /// let kde = kernel_density::kde::normal::NormalKernelDensityEstimation::new(&samples);
    /// ```
    pub fn new(samples: &[f64]) -> NormalKernelDensityEstimation {
        let length = samples.len();
        assert!(length > 0);

        NormalKernelDensityEstimation { samples: samples.to_vec() }
    }

    /// Calculate a value of the kernel density function for a given value.
    ///
    /// # Panics
    ///
    /// The bandwidth should be > 0.
    ///
    /// # Examples
    ///
    /// ```
    /// extern crate kernel_density;
    ///
    /// let samples = vec!(9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0);
    /// let kde = kernel_density::kde::normal::NormalKernelDensityEstimation::new(&samples);
    /// assert_eq!(kde.value(4.0, 0.1), 0.3989422804014327);
    /// ```
    pub fn value(&self, x: f64, bandwidth: f64) -> f64 {
        assert!(bandwidth > 0.0);
        let length = self.samples.len();

        let mut sum = 0.0;
        for sample in &self.samples {
            let rescaled: f64 = (x - sample) / bandwidth;
            sum += (-0.5 * rescaled.powi(2)).exp()
        }

        let sqrt_2pi = (2.0 * consts::PI).sqrt();
        sum / (sqrt_2pi * length as f64 * bandwidth)
    }

    /// Calculate a value of the cumulative density function for this kernel
    /// density estimation.
    ///
    /// # Panics
    ///
    /// The bandwidth should be > 0.
    ///
    /// # Examples
    ///
    /// ```
    /// extern crate kernel_density;
    ///
    /// let samples = vec!(9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0);
    /// let kde = kernel_density::kde::normal::NormalKernelDensityEstimation::new(&samples);
    /// assert_eq!(kde.cdf(0.1, 0.1), 0.08413447460685429);
    /// ```
    pub fn cdf(&self, x: f64, bandwidth: f64) -> f64 {
        assert!(bandwidth > 0.0);
        let length = self.samples.len();

        let mut sum = 0.0;
        for sample in &self.samples {
            let rescaled: f64 = (x - sample) / bandwidth;
            sum += rescaled.norm();
        }

        sum / length as f64
    }
}

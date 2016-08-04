//! Uniform kernel density estimation functions.

use density::Density;

pub struct UniformKernelDensityEstimation {
    pub samples: Vec<f64>,
    pub bandwidth: f64,
}

impl Density for UniformKernelDensityEstimation {
    /// Calculate a value of the kernel density function for a given value.
    ///
    /// # Examples
    ///
    /// ```
    /// extern crate kernel_density;
    ///
    /// let samples = vec!(9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0);
    /// let bandwidth = 0.1;
    /// let kde = kernel_density::kde::uniform(&samples, bandwidth);
    ///
    /// assert_eq!(kde.density(4.0), 0.5);
    /// ```
    fn density(&self, x: f64) -> f64 {
        let length = self.samples.len();

        let mut sum = 0.0;
        for sample in &self.samples {
            if (x - sample).abs() / self.bandwidth <= 1.0 {
                sum += 0.5
            }
        }

        sum / (length as f64 * self.bandwidth)
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
    /// let kde = kernel_density::kde::uniform(&samples, bandwidth);
    ///
    /// assert_eq!(kde.cdf(0.1), 0.1);
    /// ```
    fn cdf(&self, x: f64) -> f64 {
        let length = self.samples.len();

        let mut sum = 0.0;
        for sample in &self.samples {
            let rescaled: f64 = (x - sample) / self.bandwidth;
            if rescaled >= 1.0 {
                sum += 1.0;
            } else if rescaled > -1.0 {
                sum += 0.5 * (rescaled + 1.0);
            }
        }

        sum / length as f64
    }
}

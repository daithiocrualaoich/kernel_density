//! Epanechnikov kernel density estimation functions.

use Density;

pub struct EpanechnikovKernelDensityEstimation {
    samples: Vec<f64>,
    bandwidth: f64,
}

impl EpanechnikovKernelDensityEstimation {
    /// Construct a kernel density estimation for a given sample. Uses the
    /// Epanenchnikov kernel.
    ///
    /// k(x) = 3 * (1 - x^2) / 4 for abs(x) <= 1 and 0 otherwise.
    ///
    /// # Panics
    ///
    /// Bandwidth must be greater than zero and the sample set must be
    /// non-empty.
    ///
    /// # Examples
    ///
    /// ```
    /// extern crate kernel_density;
    ///
    /// let samples = vec!(9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0);
    /// let bandwidth = 0.1;
    /// let kde = kernel_density::kde::epanechnikov::EpanechnikovKernelDensityEstimation::new(
    ///     &samples, bandwidth
    /// );
    /// ```
    pub fn new(samples: &[f64], bandwidth: f64) -> EpanechnikovKernelDensityEstimation {
        assert!(bandwidth > 0.0);

        let length = samples.len();
        assert!(length > 0);

        EpanechnikovKernelDensityEstimation {
            samples: samples.to_vec(),
            bandwidth: bandwidth,
        }
    }
}

impl Density for EpanechnikovKernelDensityEstimation {
    /// Calculate a value of the kernel density function for a given value.
    ///
    /// # Examples
    ///
    /// ```
    /// extern crate kernel_density;
    /// use self::kernel_density::Density;
    /// use self::kernel_density::kde::epanechnikov::EpanechnikovKernelDensityEstimation;
    ///
    /// fn main() {
    ///     let samples = vec!(9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0);
    ///     let bandwidth = 0.1;
    ///     let kde = EpanechnikovKernelDensityEstimation::new(&samples, bandwidth);
    ///
    ///     assert_eq!(kde.density(4.0), 0.75);
    /// }
    /// ```
    fn density(&self, x: f64) -> f64 {
        let length = self.samples.len();

        let mut sum = 0.0;
        for sample in &self.samples {
            let rescaled: f64 = (x - sample) / self.bandwidth;
            if rescaled.abs() <= 1.0 {
                sum += 1.0 - rescaled.powi(2);
            }
        }

        0.75 * sum / (length as f64 * self.bandwidth)
    }

    /// Calculate a value of the cumulative density function for this kernel
    /// density estimation.
    ///
    /// # Examples
    ///
    /// ```
    /// extern crate kernel_density;
    /// use self::kernel_density::Density;
    /// use self::kernel_density::kde::epanechnikov::EpanechnikovKernelDensityEstimation;
    ///
    /// fn main() {
    ///     let samples = vec!(9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0);
    ///     let bandwidth = 0.1;
    ///     let kde = EpanechnikovKernelDensityEstimation::new(&samples, bandwidth);
    ///
    ///     assert_eq!(kde.cdf(0.1), 0.1);
    /// }
    /// ```
    fn cdf(&self, x: f64) -> f64 {
        let length = self.samples.len();

        let mut sum = 0.0;
        for sample in &self.samples {
            let rescaled: f64 = (x - sample) / self.bandwidth;
            if rescaled >= 1.0 {
                sum += 1.0;
            } else if rescaled > -1.0 {
                sum += 0.5 + 0.25 * (3.0 * rescaled - rescaled.powi(3));
            }
        }

        sum / length as f64
    }
}

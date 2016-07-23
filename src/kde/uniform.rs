//! Uniform kernel density estimation functions.

pub struct UniformKernelDensityEstimation {
    samples: Vec<f64>,
}

impl UniformKernelDensityEstimation {
    /// Construct a kernel density estimation for a given sample. Uses the
    /// Uniform kernel.
    ///
    /// k(x) = 0.5 for abs(x) <= 1 and 0 otherwise.
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
    /// let kde = kernel_density::kde::uniform::UniformKernelDensityEstimation::new(&samples);
    /// ```
    pub fn new(samples: &[f64]) -> UniformKernelDensityEstimation {
        let length = samples.len();
        assert!(length > 0);

        UniformKernelDensityEstimation { samples: samples.to_vec() }
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
    /// let kde = kernel_density::kde::uniform::UniformKernelDensityEstimation::new(&samples);
    /// assert_eq!(kde.value(4.0, 0.1), 0.5);
    /// ```
    pub fn value(&self, x: f64, bandwidth: f64) -> f64 {
        assert!(bandwidth > 0.0);
        let length = self.samples.len();

        let mut sum = 0.0;
        for sample in &self.samples {
            if (x - sample).abs() / bandwidth <= 1.0 {
                sum += 0.5
            }
        }

        sum / (length as f64 * bandwidth)
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
    /// let kde = kernel_density::kde::uniform::UniformKernelDensityEstimation::new(&samples);
    /// assert_eq!(kde.cdf(0.1, 0.1), 0.1);
    /// ```
    pub fn cdf(&self, x: f64, bandwidth: f64) -> f64 {
        assert!(bandwidth > 0.0);
        let length = self.samples.len();

        let mut sum = 0.0;
        for sample in &self.samples {
            let rescaled: f64 = (x - sample) / bandwidth;
            if rescaled >= 1.0 {
                sum += 1.0;
            } else if rescaled > -1.0 {
                sum += 0.5 * (rescaled + 1.0);
            }
        }

        sum / length as f64
    }
}

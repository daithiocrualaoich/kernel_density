//! Kernel Density Estimation functions.

mod epanechnikov;
mod normal;
mod uniform;

use density::Density;

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
/// let kde = kernel_density::kde::epanechnikov(&samples, bandwidth);
/// ```
pub fn epanechnikov(samples: &[f64], bandwidth: f64) -> Box<dyn Density> {
    assert!(bandwidth > 0.0);

    let length = samples.len();
    assert!(length > 0);

    Box::new(epanechnikov::EpanechnikovKernelDensityEstimation {
        samples: samples.to_vec(),
        bandwidth: bandwidth,
    })
}

/// Construct a kernel density estimation for a given sample. Uses the
/// Normal kernel.
///
/// k(x) = exp(0.5 * x^2)/(2*pi)
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
/// let kde = kernel_density::kde::normal(&samples, bandwidth);
/// ```
pub fn normal(samples: &[f64], bandwidth: f64) -> Box<dyn Density> {
    assert!(bandwidth > 0.0);

    let length = samples.len();
    assert!(length > 0);

    Box::new(normal::NormalKernelDensityEstimation {
        samples: samples.to_vec(),
        bandwidth: bandwidth,
    })
}

/// Construct a kernel density estimation for a given sample. Uses the
/// Uniform kernel.
///
/// k(x) = 0.5 for abs(x) <= 1 and 0 otherwise.
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
/// let kde = kernel_density::kde::uniform(&samples, bandwidth);
/// ```
pub fn uniform(samples: &[f64], bandwidth: f64) -> Box<dyn Density> {
    assert!(bandwidth > 0.0);

    let length = samples.len();
    assert!(length > 0);

    Box::new(uniform::UniformKernelDensityEstimation {
        samples: samples.to_vec(),
        bandwidth: bandwidth,
    })
}

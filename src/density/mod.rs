//! Density function definitions and examples.

pub trait Density {
    fn cdf(&self, x: f64) -> f64;
    fn density(&self, x: f64) -> f64;
}

mod ecdf;
pub use self::ecdf::{ecdf, p, percentile, rank, Ecdf};

mod normal;

/// Construct a normal density for given mean and variance.
///
/// # Panics
///
/// Variance must be greater than zero.
///
/// # Examples
///
/// ```
/// extern crate kernel_density;
///
/// let mean = 0.0;
/// let variance = 1.0;
/// kernel_density::density::normal(mean, variance);
/// ```
pub fn normal(mean: f64, variance: f64) -> Box<dyn Density> {
    assert!(variance > 0.0);

    Box::new(normal::NormalDensity {
        mean: mean,
        variance: variance,
    })
}

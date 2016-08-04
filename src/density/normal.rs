//! Normal kernel density estimation functions.

extern crate special_fun;

use density::Density;
use self::special_fun::FloatSpecial;
use std::f64::consts::PI;

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
        z.norm()
    }
}

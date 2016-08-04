//! Density function definitions and examples.

pub trait Density {
    fn cdf(&self, x: f64) -> f64;
    fn density(&self, x: f64) -> f64;
}

mod ecdf;
pub use self::ecdf::{Ecdf, ecdf, percentile, p, rank};

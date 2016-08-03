pub mod ecdf;
pub mod kde;
pub mod kolmogorov_smirnov;

pub trait Density {
    fn density(&self, x: f64) -> f64;
    fn cdf(&self, x: f64) -> f64;
}

mod common;

extern crate kernel_density;
extern crate rand;
extern crate quickcheck;

use kernel_density::Density;
use kernel_density::kde::normal::NormalKernelDensityEstimation;
use common::{check, SamplesF64, PositiveF64};
use std::f64;

#[test]
#[should_panic(expected="assertion failed: length > 0")]
fn new_normal_kde_panics_on_empty_samples_set() {
    let xs: Vec<f64> = vec![];
    NormalKernelDensityEstimation::new(&xs, 1.0);
}

#[test]
#[should_panic(expected="assertion failed: bandwidth > 0.0")]
fn normal_kde_panics_on_zero_bandwidth() {
    let xs: Vec<f64> = vec![0.0];
    NormalKernelDensityEstimation::new(&xs, 0.0);
}

#[test]
fn normal_kde_between_zero_and_one() {
    fn prop(xs: SamplesF64, x: f64, bandwidth: PositiveF64) -> bool {
        let kde = NormalKernelDensityEstimation::new(&xs.vec, bandwidth.val);
        let actual = kde.density(x);

        0.0 <= actual && actual <= 1.0
    }

    check(prop as fn(SamplesF64, f64, PositiveF64) -> bool);
}

#[test]
fn normal_kde_cdf_between_zero_and_one() {
    fn prop(xs: SamplesF64, x: f64, bandwidth: PositiveF64) -> bool {
        let kde = NormalKernelDensityEstimation::new(&xs.vec, bandwidth.val);
        let actual = kde.cdf(x);

        0.0 <= actual && actual <= 1.0
    }

    check(prop as fn(SamplesF64, f64, PositiveF64) -> bool);
}

#[test]
fn normal_kde_cdf_is_an_increasing_function() {
    fn prop(xs: SamplesF64, x: f64, bandwidth: PositiveF64) -> bool {
        let kde = NormalKernelDensityEstimation::new(&xs.vec, bandwidth.val);
        let actual = kde.cdf(x);

        kde.cdf(x - 0.01) <= actual && actual <= kde.cdf(x + 0.01)
    }

    check(prop as fn(SamplesF64, f64, PositiveF64) -> bool);
}

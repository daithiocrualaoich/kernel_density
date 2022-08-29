mod common;

extern crate kernel_density;
extern crate quickcheck;
extern crate rand;

use common::{check, PositiveF64};
use kernel_density::density;
use std::f64;

#[test]
#[should_panic(expected = "assertion failed: variance > 0.0")]
fn new_normal_density_panics_on_zero_variance() {
    density::normal(0.0, 0.0);
}

#[test]
#[should_panic(expected = "assertion failed: variance > 0.0")]
fn new_normal_density_panics_on_negative_variance() {
    density::normal(0.0, -1.0);
}

#[test]
fn normal_density_between_zero_and_one() {
    fn prop(mean: f64, variance: PositiveF64, x: f64) -> bool {
        let normal = density::normal(mean, variance.val);
        let actual = normal.density(x);

        0.0 <= actual && actual <= 1.0
    }

    check(prop as fn(f64, PositiveF64, f64) -> bool);
}

#[test]
fn normal_density_is_symmetric_around_mean() {
    fn prop(mean: f64, variance: PositiveF64, delta: PositiveF64) -> bool {
        let normal = density::normal(mean, variance.val);

        normal.density(mean + delta.val) == normal.density(mean - delta.val)
    }

    check(prop as fn(f64, PositiveF64, PositiveF64) -> bool);
}

#[test]
fn normal_density_cdf_between_zero_and_one() {
    fn prop(mean: f64, variance: PositiveF64, x: f64) -> bool {
        let normal = density::normal(mean, variance.val);
        let actual = normal.cdf(x);

        // Non NaN value => between 0 and 1.
        actual.is_nan() || (0.0 <= actual && actual <= 1.0)
    }

    assert!(prop(0.0, PositiveF64 { val: 2.0 }, 139327.0));
    check(prop as fn(f64, PositiveF64, f64) -> bool);
}

#[test]
fn normal_density_cdf_is_an_increasing_function() {
    fn prop(mean: f64, variance: PositiveF64, x: f64) -> bool {
        let normal = density::normal(mean, variance.val);
        let left = normal.cdf(x - 0.01);
        let actual = normal.cdf(x);
        let right = normal.cdf(x + 0.01);

        let nan_value = left.is_nan() || actual.is_nan() || right.is_nan();

        // Non NaN values => (left <= actual <= right).
        nan_value || (left <= actual && actual <= right)
    }

    assert!(prop(0.0, PositiveF64 { val: 1.0 }, 162237.0));
    check(prop as fn(f64, PositiveF64, f64) -> bool);
}

#[test]
fn normal_density_cdf_is_equal_weight_around_mean() {
    fn prop(mean: f64, variance: PositiveF64) -> bool {
        let normal = density::normal(mean, variance.val);
        normal.cdf(mean) == 0.5
    }

    check(prop as fn(f64, PositiveF64) -> bool);
}

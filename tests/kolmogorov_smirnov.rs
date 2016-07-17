extern crate kernel_density;
extern crate quickcheck;
extern crate rand;

use quickcheck::{Arbitrary, Gen, QuickCheck, Testable, StdGen};
use rand::Rng;
use std::{cmp, usize};

use kernel_density::kolmogorov_smirnov::test;
use kernel_density::ecdf::Ecdf;

const EPSILON: f64 = 1e-10;

fn check<A: Testable>(f: A) {
    // Need -1 to ensure space for creating non-overlapping samples.
    let g = StdGen::new(rand::thread_rng(), usize::MAX - 1);
    QuickCheck::new().gen(g).quickcheck(f);
}

/// Wrapper for generating sample data with QuickCheck.
///
/// Samples must be sequences of u64 values with more than 7 elements.
#[derive(Debug, Clone)]
struct Samples {
    vec: Vec<u64>,
}

impl Samples {
    fn min(&self) -> u64 {
        let &min = self.vec.iter().min().unwrap();
        min
    }

    fn max(&self) -> u64 {
        let &max = self.vec.iter().max().unwrap();
        max
    }

    fn shuffle(&mut self) {
        let mut rng = rand::thread_rng();
        rng.shuffle(&mut self.vec);
    }
}

impl Arbitrary for Samples {
    fn arbitrary<G: Gen>(g: &mut G) -> Samples {
        // Limit size of generated sample set to 1024
        let max = cmp::min(g.size(), 1024);

        let size = g.gen_range(8, max);
        let vec = (0..size).map(|_| u64::arbitrary(g)).collect();

        Samples { vec: vec }
    }

    fn shrink(&self) -> Box<Iterator<Item = Samples>> {
        let vec: Vec<u64> = self.vec.clone();
        let shrunk: Box<Iterator<Item = Vec<u64>>> = vec.shrink();

        Box::new(shrunk.filter(|v| v.len() > 7).map(|v| Samples { vec: v }))
    }
}

#[test]
#[should_panic(expected="assertion failed: xs.len() > 7 && ys.len() > 7")]
fn test_panics_on_empty_samples_set() {
    let xs: Vec<u64> = vec![];
    let ys: Vec<u64> = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    test(&xs, &ys, 0.95);
}

#[test]
#[should_panic(expected="assertion failed: xs.len() > 7 && ys.len() > 7")]
fn test_panics_on_empty_other_samples_set() {
    let xs: Vec<u64> = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    let ys: Vec<u64> = vec![];
    test(&xs, &ys, 0.95);
}

#[test]
#[should_panic(expected="assertion failed: 0.0 < confidence && confidence < 1.0")]
fn test_panics_on_confidence_leq_zero() {
    let xs: Vec<u64> = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    let ys: Vec<u64> = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    test(&xs, &ys, 0.0);
}

#[test]
#[should_panic(expected="assertion failed: 0.0 < confidence && confidence < 1.0")]
fn test_panics_on_confidence_geq_one() {
    let xs: Vec<u64> = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    let ys: Vec<u64> = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    test(&xs, &ys, 1.0);
}

/// Alternative calculation for the test statistic for the two sample
/// Kolmogorov-Smirnov test. This simple implementation is used as a
/// verification check against actual calculation used.
fn calculate_statistic_alt<T: Ord + Clone>(xs: &[T], ys: &[T]) -> f64 {
    assert!(xs.len() > 0 && ys.len() > 0);

    let ecdf_xs = Ecdf::new(xs);
    let ecdf_ys = Ecdf::new(ys);

    let mut statistic = 0.0;

    for x in xs.iter() {
        let diff = (ecdf_xs.value(x.clone()) - ecdf_ys.value(x.clone())).abs();
        if diff > statistic {
            statistic = diff;
        }
    }

    for y in ys.iter() {
        let diff = (ecdf_xs.value(y.clone()) - ecdf_ys.value(y.clone())).abs();
        if diff > statistic {
            statistic = diff;
        }
    }

    statistic
}

#[test]
fn test_calculate_statistic() {
    fn prop(xs: Samples, ys: Samples) -> bool {
        let result = test(&xs.vec, &ys.vec, 0.95);
        let actual = result.statistic;
        let expected = calculate_statistic_alt(&xs.vec, &ys.vec);

        actual == expected
    }

    check(prop as fn(Samples, Samples) -> bool);
}

#[test]
fn test_statistic_is_between_zero_and_one() {
    fn prop(xs: Samples, ys: Samples) -> bool {
        let result = test(&xs.vec, &ys.vec, 0.95);
        let actual = result.statistic;

        0.0 <= actual && actual <= 1.0
    }

    check(prop as fn(Samples, Samples) -> bool);
}

#[test]
fn test_statistic_is_zero_for_identical_samples() {
    fn prop(xs: Samples) -> bool {
        let ys = xs.clone();

        let result = test(&xs.vec, &ys.vec, 0.95);

        result.statistic == 0.0
    }

    check(prop as fn(Samples) -> bool);
}

#[test]
fn test_statistic_is_zero_for_permuted_sample() {
    fn prop(xs: Samples) -> bool {
        let mut ys = xs.clone();
        ys.shuffle();

        let result = test(&xs.vec, &ys.vec, 0.95);

        result.statistic == 0.0
    }

    check(prop as fn(Samples) -> bool);
}

#[test]
fn test_statistic_is_one_for_samples_with_no_overlap_in_support() {
    fn prop(xs: Samples) -> bool {
        let mut ys = xs.clone();

        // Shift ys so that ys.min > xs.max.
        let ys_min = xs.max() + 1;
        ys.vec = ys.vec.iter().map(|&y| cmp::max(y, ys_min)).collect();

        let result = test(&xs.vec, &ys.vec, 0.95);

        result.statistic == 1.0
    }

    check(prop as fn(Samples) -> bool);
}

#[test]
fn test_statistic_is_one_half_for_sample_with_non_overlapping_in_support_replicate_added() {
    fn prop(xs: Samples) -> bool {
        let mut ys = xs.clone();

        // Shift ys so that ys.min > xs.max.
        let ys_min = xs.max() + 1;
        ys.vec = ys.vec.iter().map(|&y| cmp::max(y, ys_min)).collect();

        // Add all the original items back too.
        for &x in xs.vec.iter() {
            ys.vec.push(x);
        }

        let result = test(&xs.vec, &ys.vec, 0.95);

        result.statistic == 0.5
    }

    check(prop as fn(Samples) -> bool);
}

#[test]
fn test_statistic_is_one_div_length_for_sample_with_additional_low_value() {
    fn prop(xs: Samples) -> bool {
        // Add a extra sample of early weight to ys.
        let min = xs.min();
        let mut ys = xs.clone();
        ys.vec.push(min - 1);

        let result = test(&xs.vec, &ys.vec, 0.95);
        let expected = 1.0 / ys.vec.len() as f64;

        result.statistic == expected
    }

    check(prop as fn(Samples) -> bool);
}

#[test]
fn test_statistic_is_one_div_length_for_sample_with_additional_high_value() {
    fn prop(xs: Samples) -> bool {
        // Add a extra sample of late weight to ys.
        let max = xs.max();
        let mut ys = xs.clone();
        ys.vec.push(max + 1);

        let result = test(&xs.vec, &ys.vec, 0.95);
        let expected = 1.0 / ys.vec.len() as f64;

        (result.statistic - expected).abs() < EPSILON
    }

    check(prop as fn(Samples) -> bool);
}

#[test]
fn test_statistic_is_one_div_length_for_sample_with_additional_low_and_high_values() {
    fn prop(xs: Samples) -> bool {
        // Add a extra sample of late weight to ys.
        let min = xs.min();
        let max = xs.max();

        let mut ys = xs.clone();

        ys.vec.push(min - 1);
        ys.vec.push(max + 1);

        let result = test(&xs.vec, &ys.vec, 0.95);
        let expected = 1.0 / ys.vec.len() as f64;

        (result.statistic - expected).abs() < EPSILON
    }

    check(prop as fn(Samples) -> bool);
}

#[test]
fn test_statistic_is_n_div_length_for_sample_with_additional_n_low_values() {
    fn prop(xs: Samples, n: u8) -> bool {
        // Add extra sample of early weight to ys.
        let min = xs.min();
        let mut ys = xs.clone();
        for j in 0..n {
            ys.vec.push(min - (j as u64) - 1);
        }

        let result = test(&xs.vec, &ys.vec, 0.95);
        let expected = n as f64 / ys.vec.len() as f64;

        result.statistic == expected
    }

    check(prop as fn(Samples, u8) -> bool);
}

#[test]
fn test_statistic_is_n_div_length_for_sample_with_additional_n_high_values() {
    fn prop(xs: Samples, n: u8) -> bool {
        // Add extra sample of early weight to ys.
        let max = xs.max();
        let mut ys = xs.clone();
        for j in 0..n {
            ys.vec.push(max + (j as u64) + 1);
        }

        let result = test(&xs.vec, &ys.vec, 0.95);
        let expected = n as f64 / ys.vec.len() as f64;

        (result.statistic - expected).abs() < EPSILON
    }

    check(prop as fn(Samples, u8) -> bool);
}

#[test]
fn test_statistic_is_n_div_length_for_sample_with_additional_n_low_and_high_values() {
    fn prop(xs: Samples, n: u8) -> bool {
        // Add extra sample of early weight to ys.
        let min = xs.min();
        let max = xs.max();
        let mut ys = xs.clone();
        for j in 0..n {
            ys.vec.push(min - (j as u64) - 1);
            ys.vec.push(max + (j as u64) + 1);
        }

        let result = test(&xs.vec, &ys.vec, 0.95);
        let expected = n as f64 / ys.vec.len() as f64;

        (result.statistic - expected).abs() < EPSILON
    }

    check(prop as fn(Samples, u8) -> bool);
}

#[test]
fn test_statistic_is_n_or_m_div_length_for_sample_with_additional_n_low_and_m_high_values() {
    fn prop(xs: Samples, n: u8, m: u8) -> bool {
        // Add extra sample of early weight to ys.
        let min = xs.min();
        let max = xs.max();
        let mut ys = xs.clone();
        for j in 0..n {
            ys.vec.push(min - (j as u64) - 1);
        }

        for j in 0..m {
            ys.vec.push(max + (j as u64) + 1);
        }

        let result = test(&xs.vec, &ys.vec, 0.95);
        let expected = cmp::max(n, m) as f64 / ys.vec.len() as f64;

        (result.statistic - expected).abs() < EPSILON
    }

    check(prop as fn(Samples, u8, u8) -> bool);
}

#[test]
fn test_is_rejected_if_reject_probability_greater_than_confidence() {
    fn prop(xs: Samples, ys: Samples) -> bool {
        let result = test(&xs.vec, &ys.vec, 0.95);

        if result.is_rejected {
            result.reject_probability > 0.95
        } else {
            result.reject_probability <= 0.95
        }
    }

    check(prop as fn(Samples, Samples) -> bool);
}

#[test]
fn test_reject_probability_is_zero_for_identical_samples() {
    fn prop(xs: Samples) -> bool {
        let ys = xs.clone();

        let result = test(&xs.vec, &ys.vec, 0.95);

        result.reject_probability == 0.0
    }

    check(prop as fn(Samples) -> bool);
}

#[test]
fn test_reject_probability_is_zero_for_permuted_sample() {
    fn prop(xs: Samples) -> bool {
        let mut ys = xs.clone();
        ys.shuffle();

        let result = test(&xs.vec, &ys.vec, 0.95);

        result.reject_probability == 0.0
    }

    check(prop as fn(Samples) -> bool);
}

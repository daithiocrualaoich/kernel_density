extern crate kernel_density;
extern crate quickcheck;
extern crate rand;

use quickcheck::{Arbitrary, Gen, QuickCheck, Testable, TestResult, StdGen};
use std::{cmp, usize};

use kernel_density::ecdf::{Ecdf, ecdf, percentile, p, rank};

fn check<A: Testable>(f: A) {
    let g = StdGen::new(rand::thread_rng(), usize::MAX);
    QuickCheck::new().gen(g).quickcheck(f);
}

/// Wrapper for generating sample data with QuickCheck.
///
/// Samples must be non-empty sequences of u64 values.
#[derive(Debug, Clone)]
struct Samples {
    vec: Vec<u64>,
}

impl Arbitrary for Samples {
    fn arbitrary<G: Gen>(g: &mut G) -> Samples {
        // Limit size of generated sample set to 1024
        let max = cmp::min(g.size(), 1024);

        let size = g.gen_range(1, max);
        let vec = (0..size).map(|_| u64::arbitrary(g)).collect();

        Samples { vec: vec }
    }

    fn shrink(&self) -> Box<Iterator<Item = Samples>> {
        let vec: Vec<u64> = self.vec.clone();
        let shrunk: Box<Iterator<Item = Vec<u64>>> = vec.shrink();

        Box::new(shrunk.filter(|v| v.len() > 0).map(|v| Samples { vec: v }))
    }
}

/// Wrapper for generating percentile query value data with QuickCheck.
///
/// Percentile must be f64 greater than 0.0 and less than or equal to 100.
/// In particular, there is no 0-percentile.
#[derive(Debug, Clone)]
struct Percentile {
    val: f64,
}

impl Arbitrary for Percentile {
    fn arbitrary<G: Gen>(g: &mut G) -> Percentile {
        let val: f64 = g.gen_range(0.0, 100.0);

        Percentile { val: val }
    }

    fn shrink(&self) -> Box<Iterator<Item = Percentile>> {
        let shrunk: Box<Iterator<Item = f64>> = self.val.shrink();

        Box::new(shrunk.filter(|&v| 0.0 < v && v <= 100.0).map(|v| Percentile { val: v }))
    }
}

/// Wrapper for generating proportion query value data with QuickCheck.
///
/// Proportion must be f64 greater than 0.0 and less than or equal to 1.0.
/// In particular, there is no 0-proportion.
#[derive(Debug, Clone)]
struct Proportion {
    val: f64,
}

impl Arbitrary for Proportion {
    fn arbitrary<G: Gen>(g: &mut G) -> Proportion {
        let val: f64 = g.gen_range(0.0, 1.0);

        Proportion { val: val }
    }

    fn shrink(&self) -> Box<Iterator<Item = Proportion>> {
        let shrunk: Box<Iterator<Item = f64>> = self.val.shrink();

        Box::new(shrunk.filter(|&v| 0.0 < v && v <= 1.0).map(|v| Proportion { val: v }))
    }
}

#[test]
#[should_panic(expected="assertion failed: length > 0")]
fn single_use_ecdf_panics_on_empty_samples_set() {
    let xs: Vec<u64> = vec![];
    ecdf(&xs, 0);
}

#[test]
#[should_panic(expected="assertion failed: length > 0")]
fn multiple_use_ecdf_panics_on_empty_samples_set() {
    let xs: Vec<u64> = vec![];
    Ecdf::new(&xs);
}

#[test]
fn single_use_ecdf_between_zero_and_one() {
    fn prop(xs: Samples, val: u64) -> bool {
        let actual = ecdf(&xs.vec, val);

        0.0 <= actual && actual <= 1.0
    }

    check(prop as fn(Samples, u64) -> bool);
}

#[test]
fn multiple_use_ecdf_between_zero_and_one() {
    fn prop(xs: Samples, val: u64) -> bool {
        let ecdf = Ecdf::new(&xs.vec);
        let actual = ecdf.value(val);

        0.0 <= actual && actual <= 1.0
    }

    check(prop as fn(Samples, u64) -> bool);
}

#[test]
fn single_use_ecdf_is_an_increasing_function() {
    fn prop(xs: Samples, val: u64) -> bool {
        let actual = ecdf(&xs.vec, val);

        ecdf(&xs.vec, val - 1) <= actual && actual <= ecdf(&xs.vec, val + 1)
    }

    check(prop as fn(Samples, u64) -> bool);
}

#[test]
fn multiple_use_ecdf_is_an_increasing_function() {
    fn prop(xs: Samples, val: u64) -> bool {
        let ecdf = Ecdf::new(&xs.vec);
        let actual = ecdf.value(val);

        ecdf.value(val - 1) <= actual && actual <= ecdf.value(val + 1)
    }

    check(prop as fn(Samples, u64) -> bool);
}

#[test]
fn single_use_ecdf_sample_min_minus_one_is_zero() {
    fn prop(xs: Samples) -> bool {
        let &min = xs.vec.iter().min().unwrap();

        ecdf(&xs.vec, min - 1) == 0.0
    }

    check(prop as fn(Samples) -> bool);
}

#[test]
fn multiple_use_ecdf_sample_min_minus_one_is_zero() {
    fn prop(xs: Samples) -> bool {
        let &min = xs.vec.iter().min().unwrap();
        let ecdf = Ecdf::new(&xs.vec);

        ecdf.value(min - 1) == 0.0
    }

    check(prop as fn(Samples) -> bool);
}

#[test]
fn single_use_ecdf_sample_max_is_one() {
    fn prop(xs: Samples) -> bool {
        let &max = xs.vec.iter().max().unwrap();

        ecdf(&xs.vec, max) == 1.0
    }

    check(prop as fn(Samples) -> bool);
}

#[test]
fn multiple_use_ecdf_sample_max_is_one() {
    fn prop(xs: Samples) -> bool {
        let &max = xs.vec.iter().max().unwrap();
        let ecdf = Ecdf::new(&xs.vec);

        ecdf.value(max) == 1.0
    }

    check(prop as fn(Samples) -> bool);
}

#[test]
fn single_use_ecdf_sample_val_is_num_samples_leq_val_div_length() {
    fn prop(xs: Samples) -> bool {
        let &val = xs.vec.first().unwrap();
        let num_samples = xs.vec
            .iter()
            .filter(|&&x| x <= val)
            .count();
        let expected = num_samples as f64 / xs.vec.len() as f64;

        ecdf(&xs.vec, val) == expected
    }

    check(prop as fn(Samples) -> bool);
}

#[test]
fn multiple_use_ecdf_sample_val_is_num_samples_leq_val_div_length() {
    fn prop(xs: Samples) -> bool {
        let &val = xs.vec.first().unwrap();
        let num_samples = xs.vec
            .iter()
            .filter(|&&x| x <= val)
            .count();
        let expected = num_samples as f64 / xs.vec.len() as f64;

        let ecdf = Ecdf::new(&xs.vec);

        ecdf.value(val) == expected
    }

    check(prop as fn(Samples) -> bool);
}

#[test]
fn single_use_ecdf_non_sample_val_is_num_samples_leq_val_div_length() {
    fn prop(xs: Samples, val: u64) -> TestResult {
        let length = xs.vec.len();

        if xs.vec.iter().any(|&x| x == val) {
            // Discard Vec containing val.
            return TestResult::discard();
        }

        let num_samples = xs.vec
            .iter()
            .filter(|&&x| x <= val)
            .count();
        let expected = num_samples as f64 / length as f64;

        let actual = ecdf(&xs.vec, val);

        TestResult::from_bool(actual == expected)
    }

    check(prop as fn(Samples, u64) -> TestResult);
}

#[test]
fn multiple_use_ecdf_non_sample_val_is_num_samples_leq_val_div_length() {
    fn prop(xs: Samples, val: u64) -> TestResult {
        let length = xs.vec.len();

        if xs.vec.iter().any(|&x| x == val) {
            // Discard Vec containing val.
            return TestResult::discard();
        }

        let num_samples = xs.vec
            .iter()
            .filter(|&&x| x <= val)
            .count();
        let expected = num_samples as f64 / length as f64;

        let ecdf = Ecdf::new(&xs.vec);

        TestResult::from_bool(ecdf.value(val) == expected)
    }

    check(prop as fn(Samples, u64) -> TestResult);
}

#[test]
fn single_and_multiple_use_ecdf_agree() {
    fn prop(xs: Samples, val: u64) -> bool {
        let multiple_use = Ecdf::new(&xs.vec);

        multiple_use.value(val) == ecdf(&xs.vec, val)
    }

    check(prop as fn(Samples, u64) -> bool);
}

#[test]
#[should_panic(expected="assertion failed: 0.0 < percentile && percentile <= 100.0")]
fn single_use_percentile_panics_on_zero_percentile() {
    let xs: Vec<u64> = vec![0];

    percentile(&xs, 0.0);
}

#[test]
#[should_panic(expected="assertion failed: 0.0 < percentile && percentile <= 100.0")]
fn single_use_percentile_panics_on_greater_than_100_percentile() {
    let xs: Vec<u64> = vec![0];

    percentile(&xs, 100.1);
}

#[test]
#[should_panic(expected="assertion failed: 0.0 < percentile && percentile <= 100.0")]
fn multiple_use_percentile_panics_on_zero_percentile() {
    let xs: Vec<u64> = vec![0];
    let ecdf = Ecdf::new(&xs);

    ecdf.percentile(0.0);
}

#[test]
#[should_panic(expected="assertion failed: 0.0 < percentile && percentile <= 100.0")]
fn multiple_use_percentile_panics_on_greater_than_100_percentile() {
    let xs: Vec<u64> = vec![0];
    let ecdf = Ecdf::new(&xs);

    ecdf.percentile(100.1);
}

#[test]
fn single_use_percentile_between_samples_min_and_max() {
    fn prop(xs: Samples, p: Percentile) -> bool {
        let &min = xs.vec.iter().min().unwrap();
        let &max = xs.vec.iter().max().unwrap();

        let actual = percentile(&xs.vec, p.val);

        min <= actual && actual <= max
    }

    check(prop as fn(Samples, Percentile) -> bool);
}

#[test]
fn single_use_percentile_is_an_increasing_function() {
    fn prop(xs: Samples, p: Percentile) -> bool {
        let smaller = (p.val - 1.0).max(0.1);
        let larger = (p.val + 1.0).min(100.0);

        let actual = percentile(&xs.vec, p.val);

        percentile(&xs.vec, smaller) <= actual && actual <= percentile(&xs.vec, larger)
    }

    check(prop as fn(Samples, Percentile) -> bool);
}

#[test]
fn single_use_percentile_100_is_sample_max() {
    fn prop(xs: Samples) -> bool {
        let &max = xs.vec.iter().max().unwrap();

        percentile(&xs.vec, 100.0) == max
    }

    check(prop as fn(Samples) -> bool);
}

#[test]
fn multiple_use_percentile_between_samples_min_and_max() {
    fn prop(xs: Samples, p: Percentile) -> bool {
        let &min = xs.vec.iter().min().unwrap();
        let &max = xs.vec.iter().max().unwrap();

        let ecdf = Ecdf::new(&xs.vec);
        let actual = ecdf.percentile(p.val);

        min <= actual && actual <= max
    }

    check(prop as fn(Samples, Percentile) -> bool);
}

#[test]
fn multiple_use_percentile_is_an_increasing_function() {
    fn prop(xs: Samples, p: Percentile) -> bool {
        let smaller = (p.val - 1.0).max(0.1);
        let larger = (p.val + 1.0).min(100.0);

        let ecdf = Ecdf::new(&xs.vec);
        let actual = ecdf.percentile(p.val);

        ecdf.percentile(smaller) <= actual && actual <= ecdf.percentile(larger)
    }

    check(prop as fn(Samples, Percentile) -> bool);
}

#[test]
fn multiple_use_percentile_100_is_sample_max() {
    fn prop(xs: Samples) -> bool {
        let &max = xs.vec.iter().max().unwrap();
        let ecdf = Ecdf::new(&xs.vec);

        ecdf.percentile(100.0) == max
    }

    check(prop as fn(Samples) -> bool);
}

#[test]
fn single_use_percentile_followed_by_single_use_ecdf_is_geq_original_value() {
    fn prop(xs: Samples, p: Percentile) -> bool {
        let actual = percentile(&xs.vec, p.val);

        // Not equal because e.g. 1- through 50-percentiles of [0, 1] are 0.
        // So original value of 1 gives percentile == 0 and ecdf(0) == 0.5.
        p.val / 100.0 <= ecdf(&xs.vec, actual)
    }

    check(prop as fn(Samples, Percentile) -> bool);
}

#[test]
fn single_use_percentile_followed_by_multiple_use_ecdf_is_geq_original_value() {
    fn prop(xs: Samples, p: Percentile) -> bool {
        let actual = percentile(&xs.vec, p.val);

        let ecdf = Ecdf::new(&xs.vec);

        // Not equal because e.g. 1- through 50-percentiles of [0, 1] are 0.
        // So original value of 1 gives percentile == 0 and ecdf(0) == 0.5.
        p.val / 100.0 <= ecdf.value(actual)
    }

    check(prop as fn(Samples, Percentile) -> bool);
}

#[test]
fn multiple_use_percentile_followed_by_single_use_ecdf_is_geq_original_value() {
    fn prop(xs: Samples, p: Percentile) -> bool {
        let multiple_use = Ecdf::new(&xs.vec);
        let actual = multiple_use.percentile(p.val);

        // Not equal because e.g. 1- through 50-percentiles of [0, 1] are 0.
        // So original value of 1 gives percentile == 0 and ecdf(0) == 0.5.
        p.val / 100.0 <= ecdf(&xs.vec, actual)
    }

    check(prop as fn(Samples, Percentile) -> bool);
}

#[test]
fn multiple_use_percentile_followed_by_multiple_use_ecdf_is_geq_original_value() {
    fn prop(xs: Samples, p: Percentile) -> bool {
        let ecdf = Ecdf::new(&xs.vec);
        let actual = ecdf.percentile(p.val);

        // Not equal because e.g. 1- through 50-percentiles of [0, 1] are 0.
        // So original value of 1 gives percentile == 0 and ecdf(0) == 0.5.
        p.val / 100.0 <= ecdf.value(actual)
    }

    check(prop as fn(Samples, Percentile) -> bool);
}

#[test]
fn single_and_multiple_use_percentile_agree() {
    fn prop(xs: Samples, p: Percentile) -> bool {
        let multiple_use = Ecdf::new(&xs.vec);

        multiple_use.percentile(p.val) == percentile(&xs.vec, p.val)
    }

    check(prop as fn(Samples, Percentile) -> bool);
}

#[test]
#[should_panic(expected="assertion failed: 0.0 < proportion && proportion <= 1.0")]
fn single_use_p_panics_on_zero_p() {
  let xs: Vec<u64> = vec![0];

  p(&xs, 0.0);
}

#[test]
#[should_panic(expected="assertion failed: 0.0 < proportion && proportion <= 1.0")]
fn single_use_p_panics_on_greater_than_1_p() {
  let xs: Vec<u64> = vec![0];

  p(&xs, 1.01);
}

#[test]
#[should_panic(expected="assertion failed: 0.0 < proportion && proportion <= 1.0")]
fn multiple_use_p_panics_on_zero_p() {
  let xs: Vec<u64> = vec![0];
  let ecdf = Ecdf::new(&xs);

  ecdf.p(0.0);
}

#[test]
#[should_panic(expected="assertion failed: 0.0 < proportion && proportion <= 1.0")]
fn multiple_use_p_panics_on_greater_than_1_p() {
  let xs: Vec<u64> = vec![0];
  let ecdf = Ecdf::new(&xs);

  ecdf.p(1.01);
}

#[test]
fn single_use_p_between_samples_min_and_max() {
  fn prop(xs: Samples, proportion: Proportion) -> bool {
    let &min = xs.vec.iter().min().unwrap();
    let &max = xs.vec.iter().max().unwrap();

    let actual = p(&xs.vec, proportion.val);

    min <= actual && actual <= max
  }

  check(prop as fn(Samples, Proportion) -> bool);
}

#[test]
fn single_use_p_is_an_increasing_function() {
  fn prop(xs: Samples, proportion: Proportion) -> bool {
    let proportion = proportion.val.max(0.01);

    let smaller = (proportion - 0.01).max(0.01);
    let larger = (proportion + 0.01).min(1.0);

    let actual = p(&xs.vec, proportion);

    p(&xs.vec, smaller) <= actual && actual <= p(&xs.vec, larger)
  }

  check(prop as fn(Samples, Proportion) -> bool);
}

#[test]
fn single_use_p_1_is_sample_max() {
  fn prop(xs: Samples) -> bool {
    let &max = xs.vec.iter().max().unwrap();

    p(&xs.vec, 1.0) == max
  }

  check(prop as fn(Samples) -> bool);
}

#[test]
fn multiple_use_p_between_samples_min_and_max() {
  fn prop(xs: Samples, proportion: Proportion) -> bool {
    let &min = xs.vec.iter().min().unwrap();
    let &max = xs.vec.iter().max().unwrap();

    let ecdf = Ecdf::new(&xs.vec);
    let actual = ecdf.p(proportion.val);

    min <= actual && actual <= max
  }

  check(prop as fn(Samples, Proportion) -> bool);
}

#[test]
fn multiple_use_p_is_an_increasing_function() {
  fn prop(xs: Samples, proportion: Proportion) -> bool {
    let proportion = proportion.val.max(0.01);

    let smaller = (proportion - 0.01).max(0.01);
    let larger = (proportion + 0.01).min(1.0);

    let ecdf = Ecdf::new(&xs.vec);
    let actual = ecdf.p(proportion);

    ecdf.p(smaller) <= actual && actual <= ecdf.p(larger)
  }

  check(prop as fn(Samples, Proportion) -> bool);
}

#[test]
fn multiple_use_p_1_is_sample_max() {
  fn prop(xs: Samples) -> bool {
    let &max = xs.vec.iter().max().unwrap();
    let ecdf = Ecdf::new(&xs.vec);

    ecdf.p(1.0) == max
  }

  check(prop as fn(Samples) -> bool);
}

#[test]
fn single_use_p_followed_by_single_use_ecdf_is_geq_original_value() {
  fn prop(xs: Samples, proportion: Proportion) -> bool {
    let actual = p(&xs.vec, proportion.val);

    // Not equal because e.g. 0 though .5-ps of [0, 1] are 0.
    // So original value of 1 gives p == 0 and ecdf(0) == 0.5.
    proportion.val <= ecdf(&xs.vec, actual)
  }

  check(prop as fn(Samples, Proportion) -> bool);
}

#[test]
fn single_use_p_followed_by_multiple_use_ecdf_is_geq_original_value() {
  fn prop(xs: Samples, proportion: Proportion) -> bool {
    let actual = p(&xs.vec, proportion.val);

    let ecdf = Ecdf::new(&xs.vec);

    // Not equal because e.g. 0 though .5-ps of [0, 1] are 0.
    // So original value of 1 gives p == 0 and ecdf(0) == 0.5.
    proportion.val <= ecdf.value(actual)
  }

  check(prop as fn(Samples, Proportion) -> bool);
}

#[test]
fn multiple_use_p_followed_by_single_use_ecdf_is_geq_original_value() {
  fn prop(xs: Samples, proportion: Proportion) -> bool {
    let multiple_use = Ecdf::new(&xs.vec);
    let actual = multiple_use.p(proportion.val);

    // Not equal because e.g. 0 though .5-ps of [0, 1] are 0.
    // So original value of 1 gives p == 0 and ecdf(0) == 0.5.
    proportion.val <= ecdf(&xs.vec, actual)
  }

  check(prop as fn(Samples, Proportion) -> bool);
}

#[test]
fn multiple_use_p_followed_by_multiple_use_ecdf_is_geq_original_value() {
  fn prop(xs: Samples, proportion: Proportion) -> bool {
    let ecdf = Ecdf::new(&xs.vec);
    let actual = ecdf.p(proportion.val);

    // Not equal because e.g. 0 though .5-ps of [0, 1] are 0.
    // So original value of 1 gives p == 0 and ecdf(0) == 0.5.
    proportion.val <= ecdf.value(actual)
  }

  check(prop as fn(Samples, Proportion) -> bool);
}

#[test]
fn single_and_multiple_use_p_agree() {
  fn prop(xs: Samples, proportion: Proportion) -> bool {
    let multiple_use = Ecdf::new(&xs.vec);

    multiple_use.p(proportion.val) == p(&xs.vec, proportion.val)
  }

  check(prop as fn(Samples, Proportion) -> bool);
}

#[test]
#[should_panic(expected="assertion failed: 0 < rank && rank <= length")]
fn single_use_rank_panics_on_zero_rank() {
    let xs: Vec<u64> = vec![0];

    rank(&xs, 0);
}

#[test]
#[should_panic(expected="assertion failed: 0 < rank && rank <= length")]
fn single_use_rank_panics_on_too_large_rank() {
    let xs: Vec<u64> = vec![0];

    rank(&xs, 2);
}

#[test]
#[should_panic(expected="assertion failed: 0 < rank && rank <= length")]
fn multiple_use_rank_panics_on_zero_rank() {
    let xs: Vec<u64> = vec![0];
    let ecdf = Ecdf::new(&xs);

    ecdf.rank(0);
}

#[test]
#[should_panic(expected="assertion failed: 0 < rank && rank <= length")]
fn multiple_use_rank_panics_on_too_large_rank() {
    let xs: Vec<u64> = vec![0];
    let ecdf = Ecdf::new(&xs);

    ecdf.rank(2);
}

#[test]
fn single_use_rank_between_samples_min_and_max() {
    fn prop(xs: Samples, r: usize) -> bool {
        let length = xs.vec.len();
        let &min = xs.vec.iter().min().unwrap();
        let &max = xs.vec.iter().max().unwrap();

        let x = r % length + 1;
        let actual = rank(&xs.vec, x);
        min <= actual && actual <= max
    }

    check(prop as fn(Samples, usize) -> bool);
}

#[test]
fn single_use_rank_is_an_increasing_function() {
    fn prop(xs: Samples, r: usize) -> bool {
        let length = xs.vec.len();
        let x = r % length + 1;

        let smaller = cmp::max(x - 1, 1);
        let larger = cmp::min(x + 1, length);

        let actual = rank(&xs.vec, x);

        rank(&xs.vec, smaller) <= actual && actual <= rank(&xs.vec, larger)
    }

    check(prop as fn(Samples, usize) -> bool);
}

#[test]
fn single_use_rank_1_is_sample_min() {
    fn prop(xs: Samples) -> bool {
        let &min = xs.vec.iter().min().unwrap();

        rank(&xs.vec, 1) == min
    }

    check(prop as fn(Samples) -> bool);
}

#[test]
fn single_use_rank_length_is_sample_max() {
    fn prop(xs: Samples) -> bool {
        let &max = xs.vec.iter().max().unwrap();

        rank(&xs.vec, xs.vec.len()) == max
    }

    check(prop as fn(Samples) -> bool);
}

#[test]
fn multiple_use_rank_between_samples_min_and_max() {
    fn prop(xs: Samples, r: usize) -> bool {
        let length = xs.vec.len();
        let &min = xs.vec.iter().min().unwrap();
        let &max = xs.vec.iter().max().unwrap();

        let ecdf = Ecdf::new(&xs.vec);

        let x = r % length + 1;
        let actual = ecdf.rank(x);
        min <= actual && actual <= max
    }

    check(prop as fn(Samples, usize) -> bool);
}

#[test]
fn multiple_use_rank_is_an_increasing_function() {
    fn prop(xs: Samples, r: usize) -> bool {
        let length = xs.vec.len();
        let x = r % length + 1;

        let smaller = cmp::max(x - 1, 1);
        let larger = cmp::min(x + 1, length);

        let ecdf = Ecdf::new(&xs.vec);
        let actual = ecdf.rank(x);

        ecdf.rank(smaller) <= actual && actual <= ecdf.rank(larger)
    }

    check(prop as fn(Samples, usize) -> bool);
}

#[test]
fn multiple_use_rank_1_is_sample_min() {
    fn prop(xs: Samples) -> bool {
        let &min = xs.vec.iter().min().unwrap();
        let ecdf = Ecdf::new(&xs.vec);

        ecdf.rank(1) == min
    }

    check(prop as fn(Samples) -> bool);
}

#[test]
fn multiple_use_rank_length_is_sample_max() {
    fn prop(xs: Samples) -> bool {
        let &max = xs.vec.iter().max().unwrap();
        let ecdf = Ecdf::new(&xs.vec);

        ecdf.rank(xs.vec.len()) == max
    }

    check(prop as fn(Samples) -> bool);
}

#[test]
fn single_use_ecdf_followed_by_single_use_rank_is_leq_original_value() {
    fn prop(xs: Samples, val: u64) -> TestResult {
        let length = xs.vec.len();
        let actual = ecdf(&xs.vec, val);

        let p = (actual * length as f64).floor() as usize;

        match p {
            0 => {
                // val is below the first rank threshold. Can't calculate
                // 0-rank value so discard.
                TestResult::discard()
            }
            _ => {
                // Not equal because e.g. all ranks of [0] are 0. So value
                // of 1 gives ecdf == 1.0 and rank(1) == 0.
                let single_use = rank(&xs.vec, p);
                TestResult::from_bool(single_use <= val)
            }
        }
    }

    check(prop as fn(Samples, u64) -> TestResult);
}

#[test]
fn single_use_ecdf_followed_by_multiple_use_rank_is_leq_original_value() {
    fn prop(xs: Samples, val: u64) -> TestResult {
        let length = xs.vec.len();
        let actual = ecdf(&xs.vec, val);

        let p = (actual * length as f64).floor() as usize;

        match p {
            0 => {
                // val is below the first rank threshold. Can't calculate
                // 0-rank value so discard.
                TestResult::discard()
            }
            _ => {
                // Not equal because e.g. all ranks of [0] are 0. So value
                // of 1 gives ecdf == 1.0 and rank(1) == 0.
                let multiple_use = Ecdf::new(&xs.vec);
                TestResult::from_bool(multiple_use.rank(p) <= val)
            }
        }
    }

    check(prop as fn(Samples, u64) -> TestResult);
}

#[test]
fn multiple_use_ecdf_followed_by_single_use_rank_is_leq_original_value() {
    fn prop(xs: Samples, val: u64) -> TestResult {
        let length = xs.vec.len();
        let ecdf = Ecdf::new(&xs.vec);
        let actual = ecdf.value(val);

        let p = (actual * length as f64).floor() as usize;

        match p {
            0 => {
                // val is below the first rank threshold. Can't calculate
                // 0-rank value so discard.
                TestResult::discard()
            }
            _ => {
                // Not equal because e.g. all ranks of [0] are 0. So value
                // of 1 gives ecdf == 1.0 and rank(1) == 0.
                let single_use = rank(&xs.vec, p);
                TestResult::from_bool(single_use <= val)
            }
        }
    }

    check(prop as fn(Samples, u64) -> TestResult);
}

#[test]
fn multiple_use_ecdf_followed_by_multiple_use_rank_is_leq_original_value() {
    fn prop(xs: Samples, val: u64) -> TestResult {
        let length = xs.vec.len();
        let ecdf = Ecdf::new(&xs.vec);
        let actual = ecdf.value(val);

        let p = (actual * length as f64).floor() as usize;

        match p {
            0 => {
                // val is below the first rank threshold. Can't calculate
                // 0-rank value so discard.
                TestResult::discard()
            }
            _ => {
                // Not equal because e.g. all ranks of [0] are 0. So value
                // of 1 gives ecdf == 1.0 and rank(1) == 0.
                TestResult::from_bool(ecdf.rank(p) <= val)
            }
        }
    }

    check(prop as fn(Samples, u64) -> TestResult);
}

#[test]
fn single_use_rank_followed_by_single_use_ecdf_is_geq_original_value() {
    fn prop(xs: Samples, r: usize) -> bool {
        let length = xs.vec.len();
        let x = r % length + 1;

        let actual = rank(&xs.vec, x);

        // Not equal because e.g. all ranks of [0, 0] are 0. So
        // rank(1) == 0 and value of 0 gives ecdf == 1.0.
        (x as f64 / length as f64) <= ecdf(&xs.vec, actual)
    }

    assert!(prop(Samples { vec: vec![0, 0] }, 0));
    check(prop as fn(Samples, usize) -> bool);
}

#[test]
fn single_use_rank_followed_by_multiple_use_ecdf_is_geq_original_value() {
    fn prop(xs: Samples, r: usize) -> bool {
        let length = xs.vec.len();
        let x = r % length + 1;

        let actual = rank(&xs.vec, x);

        let ecdf = Ecdf::new(&xs.vec);

        // Not equal because e.g. all ranks of [0, 0] are 0. So
        // rank(1) == 0 and value of 0 gives ecdf == 1.0.
        (x as f64 / length as f64) <= ecdf.value(actual)
    }

    assert!(prop(Samples { vec: vec![0, 0] }, 0));
    check(prop as fn(Samples, usize) -> bool);
}

#[test]
fn multiple_use_rank_followed_by_single_use_ecdf_is_geq_original_value() {
    fn prop(xs: Samples, r: usize) -> bool {
        let length = xs.vec.len();
        let x = r % length + 1;

        let multiple_use = Ecdf::new(&xs.vec);
        let actual = multiple_use.rank(x);

        // Not equal because e.g. all ranks of [0, 0] are 0. So
        // rank(1) == 0 and value of 0 gives ecdf == 1.0.
        (x as f64 / length as f64) <= ecdf(&xs.vec, actual)
    }

    assert!(prop(Samples { vec: vec![0, 0] }, 0));
    check(prop as fn(Samples, usize) -> bool);
}

#[test]
fn multiple_use_rank_followed_by_multiple_use_ecdf_is_geq_original_value() {
    fn prop(xs: Samples, r: usize) -> bool {
        let length = xs.vec.len();
        let x = r % length + 1;

        let ecdf = Ecdf::new(&xs.vec);
        let actual = ecdf.rank(x);

        // Not equal because e.g. all ranks of [0, 0] are 0. So
        // rank(1) == 0 and value of 0 gives ecdf == 1.0.
        (x as f64 / length as f64) <= ecdf.value(actual)
    }

    assert!(prop(Samples { vec: vec![0, 0] }, 0));
    check(prop as fn(Samples, usize) -> bool);
}

#[test]
fn single_and_multiple_use_rank_agree() {
    fn prop(xs: Samples, r: usize) -> bool {
        let length = xs.vec.len();
        let multiple_use = Ecdf::new(&xs.vec);

        let x = r % length + 1;
        multiple_use.rank(x) == rank(&xs.vec, x)
    }

    check(prop as fn(Samples, usize) -> bool);
}

#[test]
fn min_is_leq_all_samples() {
    fn prop(xs: Samples) -> bool {
        let multiple_use = Ecdf::new(&xs.vec);
        let actual = multiple_use.min();

        xs.vec.iter().all(|&x| actual <= x)
    }

    check(prop as fn(Samples) -> bool);
}

#[test]
fn max_is_geq_all_samples() {
    fn prop(xs: Samples) -> bool {
        let multiple_use = Ecdf::new(&xs.vec);
        let actual = multiple_use.max();

        xs.vec.iter().all(|&x| actual >= x)
    }

    check(prop as fn(Samples) -> bool);
}

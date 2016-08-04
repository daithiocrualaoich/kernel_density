mod common;

extern crate kernel_density;
extern crate quickcheck;
extern crate rand;

use kernel_density::density::{Ecdf, ecdf, percentile, p, rank};
use common::{check, SamplesF64, Percentile, Proportion};

use quickcheck::TestResult;
use std::{cmp, usize};

#[test]
#[should_panic(expected="assertion failed: length > 0")]
fn single_use_ecdf_panics_on_empty_samples_set() {
    let xs: Vec<f64> = vec![];
    ecdf(&xs, 0.0);
}

#[test]
#[should_panic(expected="assertion failed: length > 0")]
fn multiple_use_ecdf_panics_on_empty_samples_set() {
    let xs: Vec<f64> = vec![];
    Ecdf::new(&xs);
}

#[test]
fn single_use_ecdf_between_zero_and_one() {
    fn prop(xs: SamplesF64, val: f64) -> bool {
        let actual = ecdf(&xs.vec, val);

        0.0 <= actual && actual <= 1.0
    }

    check(prop as fn(SamplesF64, f64) -> bool);
}

#[test]
fn multiple_use_ecdf_between_zero_and_one() {
    fn prop(xs: SamplesF64, val: f64) -> bool {
        let ecdf = Ecdf::new(&xs.vec);
        let actual = ecdf.value(val);

        0.0 <= actual && actual <= 1.0
    }

    check(prop as fn(SamplesF64, f64) -> bool);
}

#[test]
fn single_use_ecdf_is_an_increasing_function() {
    fn prop(xs: SamplesF64, val: f64) -> bool {
        let actual = ecdf(&xs.vec, val);

        ecdf(&xs.vec, val - 1.0) <= actual && actual <= ecdf(&xs.vec, val + 1.0)
    }

    check(prop as fn(SamplesF64, f64) -> bool);
}

#[test]
fn multiple_use_ecdf_is_an_increasing_function() {
    fn prop(xs: SamplesF64, val: f64) -> bool {
        let ecdf = Ecdf::new(&xs.vec);
        let actual = ecdf.value(val);

        ecdf.value(val - 1.0) <= actual && actual <= ecdf.value(val + 1.0)
    }

    check(prop as fn(SamplesF64, f64) -> bool);
}

#[test]
fn single_use_ecdf_sample_min_minus_one_is_zero() {
    fn prop(xs: SamplesF64) -> bool {
        ecdf(&xs.vec, xs.min() - 1.0) == 0.0
    }

    check(prop as fn(SamplesF64) -> bool);
}

#[test]
fn multiple_use_ecdf_sample_min_minus_one_is_zero() {
    fn prop(xs: SamplesF64) -> bool {
        let ecdf = Ecdf::new(&xs.vec);

        ecdf.value(xs.min() - 1.0) == 0.0
    }

    check(prop as fn(SamplesF64) -> bool);
}

#[test]
fn single_use_ecdf_sample_max_is_one() {
    fn prop(xs: SamplesF64) -> bool {
        ecdf(&xs.vec, xs.max()) == 1.0
    }

    check(prop as fn(SamplesF64) -> bool);
}

#[test]
fn multiple_use_ecdf_sample_max_is_one() {
    fn prop(xs: SamplesF64) -> bool {
        let ecdf = Ecdf::new(&xs.vec);

        ecdf.value(xs.max()) == 1.0
    }

    check(prop as fn(SamplesF64) -> bool);
}

#[test]
fn single_use_ecdf_sample_val_is_num_samples_leq_val_div_length() {
    fn prop(xs: SamplesF64) -> bool {
        let &val = xs.vec.first().unwrap();
        let num_samples = xs.vec
            .iter()
            .filter(|&&x| x <= val)
            .count();
        let expected = num_samples as f64 / xs.vec.len() as f64;

        ecdf(&xs.vec, val) == expected
    }

    check(prop as fn(SamplesF64) -> bool);
}

#[test]
fn multiple_use_ecdf_sample_val_is_num_samples_leq_val_div_length() {
    fn prop(xs: SamplesF64) -> bool {
        let &val = xs.vec.first().unwrap();
        let num_samples = xs.vec
            .iter()
            .filter(|&&x| x <= val)
            .count();
        let expected = num_samples as f64 / xs.vec.len() as f64;

        let ecdf = Ecdf::new(&xs.vec);

        ecdf.value(val) == expected
    }

    check(prop as fn(SamplesF64) -> bool);
}

#[test]
fn single_use_ecdf_non_sample_val_is_num_samples_leq_val_div_length() {
    fn prop(xs: SamplesF64, val: f64) -> TestResult {
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

    check(prop as fn(SamplesF64, f64) -> TestResult);
}

#[test]
fn multiple_use_ecdf_non_sample_val_is_num_samples_leq_val_div_length() {
    fn prop(xs: SamplesF64, val: f64) -> TestResult {
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

    check(prop as fn(SamplesF64, f64) -> TestResult);
}

#[test]
fn single_and_multiple_use_ecdf_agree() {
    fn prop(xs: SamplesF64, val: f64) -> bool {
        let multiple_use = Ecdf::new(&xs.vec);

        multiple_use.value(val) == ecdf(&xs.vec, val)
    }

    check(prop as fn(SamplesF64, f64) -> bool);
}

#[test]
#[should_panic(expected="assertion failed: 0.0 < percentile && percentile <= 100.0")]
fn single_use_percentile_panics_on_zero_percentile() {
    let xs: Vec<f64> = vec![0.0];

    percentile(&xs, 0.0);
}

#[test]
#[should_panic(expected="assertion failed: 0.0 < percentile && percentile <= 100.0")]
fn single_use_percentile_panics_on_greater_than_100_percentile() {
    let xs: Vec<f64> = vec![0.0];

    percentile(&xs, 100.01);
}

#[test]
#[should_panic(expected="assertion failed: 0.0 < percentile && percentile <= 100.0")]
fn multiple_use_percentile_panics_on_zero_percentile() {
    let xs: Vec<f64> = vec![0.0];
    let ecdf = Ecdf::new(&xs);

    ecdf.percentile(0.0);
}

#[test]
#[should_panic(expected="assertion failed: 0.0 < percentile && percentile <= 100.0")]
fn multiple_use_percentile_panics_on_greater_than_100_percentile() {
    let xs: Vec<f64> = vec![0.0];
    let ecdf = Ecdf::new(&xs);

    ecdf.percentile(100.01);
}

#[test]
fn single_use_percentile_between_samples_min_and_max() {
    fn prop(xs: SamplesF64, p: Percentile) -> bool {
        let actual = percentile(&xs.vec, p.val);

        xs.min() <= actual && actual <= xs.max()
    }

    check(prop as fn(SamplesF64, Percentile) -> bool);
}

#[test]
fn single_use_percentile_is_an_increasing_function() {
    fn prop(xs: SamplesF64, p: Percentile) -> bool {
        let smaller = (p.val - 1.0).max(0.01);
        let larger = (p.val + 1.0).min(100.0);

        let actual = percentile(&xs.vec, p.val);

        percentile(&xs.vec, smaller) <= actual && actual <= percentile(&xs.vec, larger)
    }

    check(prop as fn(SamplesF64, Percentile) -> bool);
}

#[test]
fn single_use_percentile_100_is_sample_max() {
    fn prop(xs: SamplesF64) -> bool {
        percentile(&xs.vec, 100.0) == xs.max()
    }

    check(prop as fn(SamplesF64) -> bool);
}

#[test]
fn multiple_use_percentile_between_samples_min_and_max() {
    fn prop(xs: SamplesF64, p: Percentile) -> bool {
        let ecdf = Ecdf::new(&xs.vec);
        let actual = ecdf.percentile(p.val);

        xs.min() <= actual && actual <= xs.max()
    }

    check(prop as fn(SamplesF64, Percentile) -> bool);
}

#[test]
fn multiple_use_percentile_is_an_increasing_function() {
    fn prop(xs: SamplesF64, p: Percentile) -> bool {
        let smaller = (p.val - 1.0).max(0.01);
        let larger = (p.val + 1.0).min(100.0);

        let ecdf = Ecdf::new(&xs.vec);
        let actual = ecdf.percentile(p.val);

        ecdf.percentile(smaller) <= actual && actual <= ecdf.percentile(larger)
    }

    check(prop as fn(SamplesF64, Percentile) -> bool);
}

#[test]
fn multiple_use_percentile_100_is_sample_max() {
    fn prop(xs: SamplesF64) -> bool {
        let ecdf = Ecdf::new(&xs.vec);

        ecdf.percentile(100.0) == xs.max()
    }

    check(prop as fn(SamplesF64) -> bool);
}

#[test]
fn single_use_percentile_followed_by_single_use_ecdf_is_geq_original_value() {
    fn prop(xs: SamplesF64, p: Percentile) -> bool {
        let actual = percentile(&xs.vec, p.val);

        // Not equal because e.g. 1- through 50-percentiles of [0, 1] are 0.
        // So original value of 1 gives percentile == 0 and ecdf(0) == 0.5.
        p.val / 100.0 <= ecdf(&xs.vec, actual)
    }

    check(prop as fn(SamplesF64, Percentile) -> bool);
}

#[test]
fn single_use_percentile_followed_by_multiple_use_ecdf_is_geq_original_value() {
    fn prop(xs: SamplesF64, p: Percentile) -> bool {
        let actual = percentile(&xs.vec, p.val);

        let ecdf = Ecdf::new(&xs.vec);

        // Not equal because e.g. 1- through 50-percentiles of [0, 1] are 0.
        // So original value of 1 gives percentile == 0 and ecdf(0) == 0.5.
        p.val / 100.0 <= ecdf.value(actual)
    }

    check(prop as fn(SamplesF64, Percentile) -> bool);
}

#[test]
fn multiple_use_percentile_followed_by_single_use_ecdf_is_geq_original_value() {
    fn prop(xs: SamplesF64, p: Percentile) -> bool {
        let multiple_use = Ecdf::new(&xs.vec);
        let actual = multiple_use.percentile(p.val);

        // Not equal because e.g. 1- through 50-percentiles of [0, 1] are 0.
        // So original value of 1 gives percentile == 0 and ecdf(0) == 0.5.
        p.val / 100.0 <= ecdf(&xs.vec, actual)
    }

    check(prop as fn(SamplesF64, Percentile) -> bool);
}

#[test]
fn multiple_use_percentile_followed_by_multiple_use_ecdf_is_geq_original_value() {
    fn prop(xs: SamplesF64, p: Percentile) -> bool {
        let ecdf = Ecdf::new(&xs.vec);
        let actual = ecdf.percentile(p.val);

        // Not equal because e.g. 1- through 50-percentiles of [0, 1] are 0.
        // So original value of 1 gives percentile == 0 and ecdf(0) == 0.5.
        p.val / 100.0 <= ecdf.value(actual)
    }

    check(prop as fn(SamplesF64, Percentile) -> bool);
}

#[test]
fn single_and_multiple_use_percentile_agree() {
    fn prop(xs: SamplesF64, p: Percentile) -> bool {
        let multiple_use = Ecdf::new(&xs.vec);

        multiple_use.percentile(p.val) == percentile(&xs.vec, p.val)
    }

    check(prop as fn(SamplesF64, Percentile) -> bool);
}

#[test]
#[should_panic(expected="assertion failed: 0.0 < proportion && proportion <= 1.0")]
fn single_use_p_panics_on_zero_p() {
    let xs: Vec<f64> = vec![0.0];

    p(&xs, 0.0);
}

#[test]
#[should_panic(expected="assertion failed: 0.0 < proportion && proportion <= 1.0")]
fn single_use_p_panics_on_greater_than_1_p() {
    let xs: Vec<f64> = vec![0.0];

    p(&xs, 1.01);
}

#[test]
#[should_panic(expected="assertion failed: 0.0 < proportion && proportion <= 1.0")]
fn multiple_use_p_panics_on_zero_p() {
    let xs: Vec<f64> = vec![0.0];
    let ecdf = Ecdf::new(&xs);

    ecdf.p(0.0);
}

#[test]
#[should_panic(expected="assertion failed: 0.0 < proportion && proportion <= 1.0")]
fn multiple_use_p_panics_on_greater_than_1_p() {
    let xs: Vec<f64> = vec![0.0];
    let ecdf = Ecdf::new(&xs);

    ecdf.p(1.01);
}

#[test]
fn single_use_p_between_samples_min_and_max() {
    fn prop(xs: SamplesF64, proportion: Proportion) -> bool {
        let actual = p(&xs.vec, proportion.val);

        xs.min() <= actual && actual <= xs.max()
    }

    check(prop as fn(SamplesF64, Proportion) -> bool);
}

#[test]
fn single_use_p_is_an_increasing_function() {
    fn prop(xs: SamplesF64, proportion: Proportion) -> bool {
        let proportion = proportion.val.max(0.01);

        let smaller = (proportion - 0.01).max(0.01);
        let larger = (proportion + 0.01).min(1.0);

        let actual = p(&xs.vec, proportion);

        p(&xs.vec, smaller) <= actual && actual <= p(&xs.vec, larger)
    }

    check(prop as fn(SamplesF64, Proportion) -> bool);
}

#[test]
fn single_use_p_1_is_sample_max() {
    fn prop(xs: SamplesF64) -> bool {
        p(&xs.vec, 1.0) == xs.max()
    }

    check(prop as fn(SamplesF64) -> bool);
}

#[test]
fn multiple_use_p_between_samples_min_and_max() {
    fn prop(xs: SamplesF64, proportion: Proportion) -> bool {
        let ecdf = Ecdf::new(&xs.vec);
        let actual = ecdf.p(proportion.val);

        xs.min() <= actual && actual <= xs.max()
    }

    check(prop as fn(SamplesF64, Proportion) -> bool);
}

#[test]
fn multiple_use_p_is_an_increasing_function() {
    fn prop(xs: SamplesF64, proportion: Proportion) -> bool {
        let proportion = proportion.val.max(0.01);

        let smaller = (proportion - 0.01).max(0.01);
        let larger = (proportion + 0.01).min(1.0);

        let ecdf = Ecdf::new(&xs.vec);
        let actual = ecdf.p(proportion);

        ecdf.p(smaller) <= actual && actual <= ecdf.p(larger)
    }

    check(prop as fn(SamplesF64, Proportion) -> bool);
}

#[test]
fn multiple_use_p_1_is_sample_max() {
    fn prop(xs: SamplesF64) -> bool {
        let ecdf = Ecdf::new(&xs.vec);

        ecdf.p(1.0) == xs.max()
    }

    check(prop as fn(SamplesF64) -> bool);
}

#[test]
fn single_use_p_followed_by_single_use_ecdf_is_geq_original_value() {
    fn prop(xs: SamplesF64, proportion: Proportion) -> bool {
        let actual = p(&xs.vec, proportion.val);

        // Not equal because e.g. 0 though .5-ps of [0, 1] are 0.
        // So original value of 1 gives p == 0 and ecdf(0) == 0.5.
        proportion.val <= ecdf(&xs.vec, actual)
    }

    check(prop as fn(SamplesF64, Proportion) -> bool);
}

#[test]
fn single_use_p_followed_by_multiple_use_ecdf_is_geq_original_value() {
    fn prop(xs: SamplesF64, proportion: Proportion) -> bool {
        let actual = p(&xs.vec, proportion.val);

        let ecdf = Ecdf::new(&xs.vec);

        // Not equal because e.g. 0 though .5-ps of [0, 1] are 0.
        // So original value of 1 gives p == 0 and ecdf(0) == 0.5.
        proportion.val <= ecdf.value(actual)
    }

    check(prop as fn(SamplesF64, Proportion) -> bool);
}

#[test]
fn multiple_use_p_followed_by_single_use_ecdf_is_geq_original_value() {
    fn prop(xs: SamplesF64, proportion: Proportion) -> bool {
        let multiple_use = Ecdf::new(&xs.vec);
        let actual = multiple_use.p(proportion.val);

        // Not equal because e.g. 0 though .5-ps of [0, 1] are 0.
        // So original value of 1 gives p == 0 and ecdf(0) == 0.5.
        proportion.val <= ecdf(&xs.vec, actual)
    }

    check(prop as fn(SamplesF64, Proportion) -> bool);
}

#[test]
fn multiple_use_p_followed_by_multiple_use_ecdf_is_geq_original_value() {
    fn prop(xs: SamplesF64, proportion: Proportion) -> bool {
        let ecdf = Ecdf::new(&xs.vec);
        let actual = ecdf.p(proportion.val);

        // Not equal because e.g. 0 though .5-ps of [0, 1] are 0.
        // So original value of 1 gives p == 0 and ecdf(0) == 0.5.
        proportion.val <= ecdf.value(actual)
    }

    check(prop as fn(SamplesF64, Proportion) -> bool);
}

#[test]
fn single_and_multiple_use_p_agree() {
    fn prop(xs: SamplesF64, proportion: Proportion) -> bool {
        let multiple_use = Ecdf::new(&xs.vec);

        multiple_use.p(proportion.val) == p(&xs.vec, proportion.val)
    }

    check(prop as fn(SamplesF64, Proportion) -> bool);
}

#[test]
#[should_panic(expected="assertion failed: 0 < rank && rank <= length")]
fn single_use_rank_panics_on_zero_rank() {
    let xs: Vec<f64> = vec![0.0];

    rank(&xs, 0);
}

#[test]
#[should_panic(expected="assertion failed: 0 < rank && rank <= length")]
fn single_use_rank_panics_on_too_large_rank() {
    let xs: Vec<f64> = vec![0.0];

    rank(&xs, 2);
}

#[test]
#[should_panic(expected="assertion failed: 0 < rank && rank <= length")]
fn multiple_use_rank_panics_on_zero_rank() {
    let xs: Vec<f64> = vec![0.0];
    let ecdf = Ecdf::new(&xs);

    ecdf.rank(0);
}

#[test]
#[should_panic(expected="assertion failed: 0 < rank && rank <= length")]
fn multiple_use_rank_panics_on_too_large_rank() {
    let xs: Vec<f64> = vec![0.0];
    let ecdf = Ecdf::new(&xs);

    ecdf.rank(2);
}

#[test]
fn single_use_rank_between_samples_min_and_max() {
    fn prop(xs: SamplesF64, r: usize) -> bool {
        let length = xs.vec.len();
        let x = r % length + 1;
        let actual = rank(&xs.vec, x);

        xs.min() <= actual && actual <= xs.max()
    }

    check(prop as fn(SamplesF64, usize) -> bool);
}

#[test]
fn single_use_rank_is_an_increasing_function() {
    fn prop(xs: SamplesF64, r: usize) -> bool {
        let length = xs.vec.len();
        let x = r % length + 1;

        let smaller = cmp::max(x - 1, 1);
        let larger = cmp::min(x + 1, length);

        let actual = rank(&xs.vec, x);

        rank(&xs.vec, smaller) <= actual && actual <= rank(&xs.vec, larger)
    }

    check(prop as fn(SamplesF64, usize) -> bool);
}

#[test]
fn single_use_rank_1_is_sample_min() {
    fn prop(xs: SamplesF64) -> bool {
        rank(&xs.vec, 1) == xs.min()
    }

    check(prop as fn(SamplesF64) -> bool);
}

#[test]
fn single_use_rank_length_is_sample_max() {
    fn prop(xs: SamplesF64) -> bool {
        rank(&xs.vec, xs.vec.len()) == xs.max()
    }

    check(prop as fn(SamplesF64) -> bool);
}

#[test]
fn multiple_use_rank_between_samples_min_and_max() {
    fn prop(xs: SamplesF64, r: usize) -> bool {
        let length = xs.vec.len();
        let ecdf = Ecdf::new(&xs.vec);

        let x = r % length + 1;
        let actual = ecdf.rank(x);
        xs.min() <= actual && actual <= xs.max()
    }

    check(prop as fn(SamplesF64, usize) -> bool);
}

#[test]
fn multiple_use_rank_is_an_increasing_function() {
    fn prop(xs: SamplesF64, r: usize) -> bool {
        let length = xs.vec.len();
        let x = r % length + 1;

        let smaller = cmp::max(x - 1, 1);
        let larger = cmp::min(x + 1, length);

        let ecdf = Ecdf::new(&xs.vec);
        let actual = ecdf.rank(x);

        ecdf.rank(smaller) <= actual && actual <= ecdf.rank(larger)
    }

    check(prop as fn(SamplesF64, usize) -> bool);
}

#[test]
fn multiple_use_rank_1_is_sample_min() {
    fn prop(xs: SamplesF64) -> bool {
        let ecdf = Ecdf::new(&xs.vec);

        ecdf.rank(1) == xs.min()
    }

    check(prop as fn(SamplesF64) -> bool);
}

#[test]
fn multiple_use_rank_length_is_sample_max() {
    fn prop(xs: SamplesF64) -> bool {
        let ecdf = Ecdf::new(&xs.vec);

        ecdf.rank(xs.vec.len()) == xs.max()
    }

    check(prop as fn(SamplesF64) -> bool);
}

#[test]
fn single_use_ecdf_followed_by_single_use_rank_is_leq_original_value() {
    fn prop(xs: SamplesF64, val: f64) -> TestResult {
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

    check(prop as fn(SamplesF64, f64) -> TestResult);
}

#[test]
fn single_use_ecdf_followed_by_multiple_use_rank_is_leq_original_value() {
    fn prop(xs: SamplesF64, val: f64) -> TestResult {
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

    check(prop as fn(SamplesF64, f64) -> TestResult);
}

#[test]
fn multiple_use_ecdf_followed_by_single_use_rank_is_leq_original_value() {
    fn prop(xs: SamplesF64, val: f64) -> TestResult {
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

    check(prop as fn(SamplesF64, f64) -> TestResult);
}

#[test]
fn multiple_use_ecdf_followed_by_multiple_use_rank_is_leq_original_value() {
    fn prop(xs: SamplesF64, val: f64) -> TestResult {
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

    check(prop as fn(SamplesF64, f64) -> TestResult);
}

#[test]
fn single_use_rank_followed_by_single_use_ecdf_is_geq_original_value() {
    fn prop(xs: SamplesF64, r: usize) -> bool {
        let length = xs.vec.len();
        let x = r % length + 1;

        let actual = rank(&xs.vec, x);

        // Not equal because e.g. all ranks of [0, 0] are 0. So
        // rank(1) == 0 and value of 0 gives ecdf == 1.0.
        (x as f64 / length as f64) <= ecdf(&xs.vec, actual)
    }

    assert!(prop(SamplesF64 { vec: vec![0.0, 0.0] }, 0));
    check(prop as fn(SamplesF64, usize) -> bool);
}

#[test]
fn single_use_rank_followed_by_multiple_use_ecdf_is_geq_original_value() {
    fn prop(xs: SamplesF64, r: usize) -> bool {
        let length = xs.vec.len();
        let x = r % length + 1;

        let actual = rank(&xs.vec, x);

        let ecdf = Ecdf::new(&xs.vec);

        // Not equal because e.g. all ranks of [0, 0] are 0. So
        // rank(1) == 0 and value of 0 gives ecdf == 1.0.
        (x as f64 / length as f64) <= ecdf.value(actual)
    }

    assert!(prop(SamplesF64 { vec: vec![0.0, 0.0] }, 0));
    check(prop as fn(SamplesF64, usize) -> bool);
}

#[test]
fn multiple_use_rank_followed_by_single_use_ecdf_is_geq_original_value() {
    fn prop(xs: SamplesF64, r: usize) -> bool {
        let length = xs.vec.len();
        let x = r % length + 1;

        let multiple_use = Ecdf::new(&xs.vec);
        let actual = multiple_use.rank(x);

        // Not equal because e.g. all ranks of [0, 0] are 0. So
        // rank(1) == 0 and value of 0 gives ecdf == 1.0.
        (x as f64 / length as f64) <= ecdf(&xs.vec, actual)
    }

    assert!(prop(SamplesF64 { vec: vec![0.0, 0.0] }, 0));
    check(prop as fn(SamplesF64, usize) -> bool);
}

#[test]
fn multiple_use_rank_followed_by_multiple_use_ecdf_is_geq_original_value() {
    fn prop(xs: SamplesF64, r: usize) -> bool {
        let length = xs.vec.len();
        let x = r % length + 1;

        let ecdf = Ecdf::new(&xs.vec);
        let actual = ecdf.rank(x);

        // Not equal because e.g. all ranks of [0, 0] are 0. So
        // rank(1) == 0 and value of 0 gives ecdf == 1.0.
        (x as f64 / length as f64) <= ecdf.value(actual)
    }

    assert!(prop(SamplesF64 { vec: vec![0.0, 0.0] }, 0));
    check(prop as fn(SamplesF64, usize) -> bool);
}

#[test]
fn single_and_multiple_use_rank_agree() {
    fn prop(xs: SamplesF64, r: usize) -> bool {
        let length = xs.vec.len();
        let multiple_use = Ecdf::new(&xs.vec);

        let x = r % length + 1;
        multiple_use.rank(x) == rank(&xs.vec, x)
    }

    check(prop as fn(SamplesF64, usize) -> bool);
}

#[test]
fn min_is_leq_all_samples() {
    fn prop(xs: SamplesF64) -> bool {
        let multiple_use = Ecdf::new(&xs.vec);
        let actual = multiple_use.min();

        xs.vec.iter().all(|&x| actual <= x)
    }

    check(prop as fn(SamplesF64) -> bool);
}

#[test]
fn max_is_geq_all_samples() {
    fn prop(xs: SamplesF64) -> bool {
        let multiple_use = Ecdf::new(&xs.vec);
        let actual = multiple_use.max();

        xs.vec.iter().all(|&x| actual >= x)
    }

    check(prop as fn(SamplesF64) -> bool);
}

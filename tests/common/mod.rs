extern crate quickcheck;
extern crate rand;

use quickcheck::{Arbitrary, Gen, QuickCheck, StdGen, Testable};
use rand::Rng;
use std::{cmp, f64};

#[allow(dead_code)]
pub const EPSILON: f64 = 1e-10;

pub fn check<A: Testable>(f: A) {
    // Adjust the size of the generator to be larger than the default size of
    // 100. The problem is this restricts generated f64 values to be in the
    // range -100 to 100 where we would like to test a wider space.

    // But increasing the size too high means the generated f64 values are
    // large integers instead of decimals because the closest f64 to most of
    // the available range are floats with large positive exponents, i.e
    // integers. This also causes some tests to break by underflowing.

    let g = StdGen::new(rand::thread_rng(), 1_000_000_000_000);
    QuickCheck::new().gen(g).quickcheck(f);
}

/// Wrapper for generating sample data with QuickCheck.
///
/// Samples must be non-empty sequences of f64 values.
#[derive(Debug, Clone)]
pub struct SamplesF64 {
    pub vec: Vec<f64>,
}

#[allow(dead_code)]
impl SamplesF64 {
    pub fn min(&self) -> f64 {
        let mut min = f64::MAX;

        for &x in self.vec.iter() {
            if x < min {
                min = x
            }
        }

        min
    }

    pub fn max(&self) -> f64 {
        let mut max = f64::MIN;

        for &x in self.vec.iter() {
            if x > max {
                max = x
            }
        }

        max
    }
}

impl Arbitrary for SamplesF64 {
    fn arbitrary<G: Gen>(g: &mut G) -> SamplesF64 {
        // Limit size of generated sample set to 1024
        let max = cmp::min(g.size(), 1024);

        let size = g.gen_range(1, max);
        let vec = (0..size).map(|_| f64::arbitrary(g)).collect();

        SamplesF64 { vec: vec }
    }

    fn shrink(&self) -> Box<dyn Iterator<Item = SamplesF64>> {
        let vec: Vec<f64> = self.vec.clone();
        let shrunk: Box<dyn Iterator<Item = Vec<f64>>> = vec.shrink();

        Box::new(
            shrunk
                .filter(|v| v.len() > 0)
                .map(|v| SamplesF64 { vec: v }),
        )
    }
}

/// Wrapper for generating sample data with QuickCheck.
///
/// Samples must be sequences of f64 values with more than 7 elements.
#[derive(Debug, Clone)]
pub struct MoreThanSevenSamplesF64 {
    pub vec: Vec<f64>,
}

#[allow(dead_code)]
impl MoreThanSevenSamplesF64 {
    pub fn min(&self) -> f64 {
        let mut min = f64::MAX;

        for &x in self.vec.iter() {
            if x < min {
                min = x
            }
        }

        min
    }

    pub fn max(&self) -> f64 {
        let mut max = f64::MIN;

        for &x in self.vec.iter() {
            if x > max {
                max = x
            }
        }

        max
    }

    pub fn shuffle(&mut self) {
        let mut rng = rand::thread_rng();
        rng.shuffle(&mut self.vec);
    }
}

impl Arbitrary for MoreThanSevenSamplesF64 {
    fn arbitrary<G: Gen>(g: &mut G) -> MoreThanSevenSamplesF64 {
        // Limit size of generated sample set to 1024
        let max = cmp::min(g.size(), 1024);

        let size = g.gen_range(8, max);
        let vec = (0..size).map(|_| f64::arbitrary(g)).collect();

        MoreThanSevenSamplesF64 { vec: vec }
    }

    fn shrink(&self) -> Box<dyn Iterator<Item = MoreThanSevenSamplesF64>> {
        let vec: Vec<f64> = self.vec.clone();
        let shrunk: Box<dyn Iterator<Item = Vec<f64>>> = vec.shrink();

        Box::new(
            shrunk
                .filter(|v| v.len() > 7)
                .map(|v| MoreThanSevenSamplesF64 { vec: v }),
        )
    }
}

/// Wrapper for generating positive floating point numbers with QuickCheck.
#[derive(Debug, Clone)]
pub struct PositiveF64 {
    pub val: f64,
}

impl Arbitrary for PositiveF64 {
    fn arbitrary<G: Gen>(g: &mut G) -> PositiveF64 {
        let mut val: f64 = f64::arbitrary(g).abs();

        // Retry until non-zero
        while val == 0.0 {
            val = f64::arbitrary(g).abs();
        }

        PositiveF64 { val: val }
    }

    fn shrink(&self) -> Box<dyn Iterator<Item = PositiveF64>> {
        let shrunk: Box<dyn Iterator<Item = f64>> = self.val.shrink();

        Box::new(shrunk.filter(|&v| v > 0.0).map(|v| PositiveF64 { val: v }))
    }
}

/// Wrapper for generating non-positive floating point numbers with QuickCheck.
#[derive(Debug, Clone)]
pub struct NonPositiveF64 {
    pub val: f64,
}

impl Arbitrary for NonPositiveF64 {
    fn arbitrary<G: Gen>(g: &mut G) -> NonPositiveF64 {
        let val: f64 = -f64::arbitrary(g).abs();

        NonPositiveF64 { val: val }
    }

    fn shrink(&self) -> Box<dyn Iterator<Item = NonPositiveF64>> {
        let shrunk: Box<dyn Iterator<Item = f64>> = self.val.shrink();

        Box::new(
            shrunk
                .filter(|&v| v <= 0.0)
                .map(|v| NonPositiveF64 { val: v }),
        )
    }
}

/// Wrapper for generating percentile query value data with QuickCheck.
///
/// Percentile must be f64 greater than 0.0 and less than or equal to 100.
/// In particular, there is no 0-percentile.
#[derive(Debug, Clone)]
pub struct Percentile {
    pub val: f64,
}

impl Arbitrary for Percentile {
    fn arbitrary<G: Gen>(g: &mut G) -> Percentile {
        let val: f64 = g.gen_range(0.0, 100.0);

        Percentile { val: val }
    }

    fn shrink(&self) -> Box<dyn Iterator<Item = Percentile>> {
        let shrunk: Box<dyn Iterator<Item = f64>> = self.val.shrink();

        Box::new(
            shrunk
                .filter(|&v| 0.0 < v && v <= 100.0)
                .map(|v| Percentile { val: v }),
        )
    }
}

/// Wrapper for generating proportion query value data with QuickCheck.
///
/// Proportion must be f64 greater than 0.0 and less than or equal to 1.0.
/// In particular, there is no 0-proportion.
#[derive(Debug, Clone)]
pub struct Proportion {
    pub val: f64,
}

impl Arbitrary for Proportion {
    fn arbitrary<G: Gen>(g: &mut G) -> Proportion {
        let val: f64 = g.gen_range(0.0, 1.0);
        Proportion { val: val }
    }

    fn shrink(&self) -> Box<dyn Iterator<Item = Proportion>> {
        let shrunk: Box<dyn Iterator<Item = f64>> = self.val.shrink();

        Box::new(
            shrunk
                .filter(|&v| 0.0 < v && v <= 1.0)
                .map(|v| Proportion { val: v }),
        )
    }
}

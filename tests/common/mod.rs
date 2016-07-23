extern crate rand;
extern crate quickcheck;

use rand::Rng;
use quickcheck::{Arbitrary, Gen, QuickCheck, Testable, StdGen};
use std::{cmp, usize};

#[allow(dead_code)]
pub const EPSILON: f64 = 1e-10;

pub fn check<A: Testable>(f: A) {
    // Need -1 to ensure space for creating non-overlapping samples
    // in kolmogorov_smirnov tests.
    let g = StdGen::new(rand::thread_rng(), usize::MAX - 1);
    QuickCheck::new().gen(g).quickcheck(f);
}

/// Wrapper for generating sample data with QuickCheck.
///
/// Samples must be non-empty sequences of u64 values.
#[derive(Debug, Clone)]
pub struct SamplesU64 {
    pub vec: Vec<u64>,
}

impl Arbitrary for SamplesU64 {
    fn arbitrary<G: Gen>(g: &mut G) -> SamplesU64 {
        // Limit size of generated sample set to 1024
        let max = cmp::min(g.size(), 1024);

        let size = g.gen_range(1, max);
        let vec = (0..size).map(|_| u64::arbitrary(g)).collect();

        SamplesU64 { vec: vec }
    }

    fn shrink(&self) -> Box<Iterator<Item = SamplesU64>> {
        let vec: Vec<u64> = self.vec.clone();
        let shrunk: Box<Iterator<Item = Vec<u64>>> = vec.shrink();

        Box::new(shrunk.filter(|v| v.len() > 0).map(|v| SamplesU64 { vec: v }))
    }
}

/// Wrapper for generating sample data with QuickCheck.
///
/// Samples must be non-empty sequences of f64 values.
#[derive(Debug, Clone)]
pub struct SamplesF64 {
    pub vec: Vec<f64>,
}

impl Arbitrary for SamplesF64 {
    fn arbitrary<G: Gen>(g: &mut G) -> SamplesF64 {
        // Limit size of generated sample set to 1024
        let max = cmp::min(g.size(), 1024);

        let size = g.gen_range(1, max);
        let vec = (0..size).map(|_| f64::arbitrary(g)).collect();

        SamplesF64 { vec: vec }
    }

    fn shrink(&self) -> Box<Iterator<Item = SamplesF64>> {
        let vec: Vec<f64> = self.vec.clone();
        let shrunk: Box<Iterator<Item = Vec<f64>>> = vec.shrink();

        Box::new(shrunk.filter(|v| v.len() > 0).map(|v| SamplesF64 { vec: v }))
    }
}

/// Wrapper for generating sample data with QuickCheck.
///
/// Samples must be sequences of u64 values with more than 7 elements.
#[derive(Debug, Clone)]
pub struct MoreThanSevenSamplesU64 {
    pub vec: Vec<u64>,
}

#[allow(dead_code)]
impl MoreThanSevenSamplesU64 {
    pub fn min(&self) -> u64 {
        let &min = self.vec.iter().min().unwrap();
        min
    }

    pub fn max(&self) -> u64 {
        let &max = self.vec.iter().max().unwrap();
        max
    }

    pub fn shuffle(&mut self) {
        let mut rng = rand::thread_rng();
        rng.shuffle(&mut self.vec);
    }
}

impl Arbitrary for MoreThanSevenSamplesU64 {
    fn arbitrary<G: Gen>(g: &mut G) -> MoreThanSevenSamplesU64 {
        // Limit size of generated sample set to 1024
        let max = cmp::min(g.size(), 1024);

        let size = g.gen_range(8, max);
        let vec = (0..size).map(|_| u64::arbitrary(g)).collect();

        MoreThanSevenSamplesU64 { vec: vec }
    }

    fn shrink(&self) -> Box<Iterator<Item = MoreThanSevenSamplesU64>> {
        let vec: Vec<u64> = self.vec.clone();
        let shrunk: Box<Iterator<Item = Vec<u64>>> = vec.shrink();

        Box::new(shrunk.filter(|v| v.len() > 7).map(|v| MoreThanSevenSamplesU64 { vec: v }))
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

    fn shrink(&self) -> Box<Iterator<Item = PositiveF64>> {
        let shrunk: Box<Iterator<Item = f64>> = self.val.shrink();

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

    fn shrink(&self) -> Box<Iterator<Item = NonPositiveF64>> {
        let shrunk: Box<Iterator<Item = f64>> = self.val.shrink();

        Box::new(shrunk.filter(|&v| v <= 0.0).map(|v| NonPositiveF64 { val: v }))
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
pub struct Proportion {
    pub val: f64,
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

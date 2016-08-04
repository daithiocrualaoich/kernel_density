//! Empirical cumulative distribution function.

pub struct Ecdf {
    samples: Vec<f64>,
}

impl Ecdf {
    /// Construct a new representation of a cumulative distribution function
    /// for a given sample.
    ///
    /// The construction will involve computing a sorted clone of the given
    /// sample and may be inefficient or completely prohibitive for large
    /// samples. This computation is amortized significantly if there is heavy
    /// use of the value function.
    ///
    /// # Panics
    ///
    /// The sample set must be non-empty.
    ///
    /// # Examples
    ///
    /// ```
    /// extern crate kernel_density;
    ///
    /// let samples = vec!(9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0);
    /// let ecdf = kernel_density::density::Ecdf::new(&samples);
    /// ```
    pub fn new(samples: &[f64]) -> Ecdf {
        let length = samples.len();
        assert!(length > 0);

        // Sort a copied sample for binary searching.
        let mut sorted = samples.to_vec();
        sorted.sort_by(|x_1, x_2| x_1.partial_cmp(x_2).unwrap());

        Ecdf { samples: sorted }
    }

    /// Calculate a value of the empirical cumulative distribution function for
    /// a given sample.
    ///
    /// # Examples
    ///
    /// ```
    /// extern crate kernel_density;
    ///
    /// let samples = vec!(9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0);
    /// let ecdf = kernel_density::density::Ecdf::new(&samples);
    /// assert_eq!(ecdf.value(4.0), 0.5);
    /// ```
    pub fn value(&self, x: f64) -> f64 {
        let length = self.samples.len();
        let binary_search_x = self.samples.binary_search_by(|x_1| x_1.partial_cmp(&x).unwrap());

        let num_samples_leq_x = match binary_search_x {
            Ok(mut index) => {
                // At least one sample is a t and we have the index of it. Need
                // to walk down the sorted samples until at last that == x.
                while index + 1 < length && self.samples[index + 1] == x {
                    index += 1;
                }

                // Compensate for 0-based indexing.
                index + 1
            }
            Err(index) => {
                // No sample is a t but if we had to put one in it would go at
                // index. This means all indices to the left have samples < x
                // and should be counted in the cdf proportion. We must take
                // one from index to get the last included sample but then we
                // just have to add one again to account for 0-based indexing.
                index
            }
        };

        num_samples_leq_x as f64 / length as f64
    }

    /// Calculate a p-proportion for the sample using the Nearest Rank method.
    ///
    /// Note, the p-proportion of an ECDF is the _least_ number, n, for which
    /// at least ratio p of the values in the ECDF are less than or equal to n.
    ///
    /// This definition means the p-proportion for p between 0 and the ecdf of
    /// the lowest value in the sample exists and is the lowest value itself.
    /// For instance in [0, 1] all the p-proportions between 0 and .5 are 0
    /// even though 0 is greater than or equal to a larger proportion of the
    /// values than each p in (0, 50).
    ///
    /// # Panics
    ///
    /// The proportion requested must be greater than 0 and less than or equal
    /// 1. In particular, there is no 0-proportion value.
    ///
    /// # Examples
    ///
    /// ```
    /// extern crate kernel_density;
    ///
    /// let samples = vec!(9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0);
    /// let ecdf = kernel_density::density::Ecdf::new(&samples);
    /// assert_eq!(ecdf.p(0.5), 4.0);
    /// assert_eq!(ecdf.p(0.05), 0.0);
    /// ```
    pub fn p(&self, proportion: f64) -> f64 {
        assert!(0.0 < proportion && proportion <= 1.0);

        let length = self.samples.len();
        let rank = (proportion * length as f64).ceil() as usize;
        self.samples[rank - 1]
    }

    /// Calculate a percentile for the sample using the Nearest Rank method.
    ///
    /// Note, the p-percentile of an ECDF is the _least_ number, n, for which
    /// at least p% of the values in the ECDF are less than or equal to n.
    ///
    /// This definition means the p-percentiles for p between 0 and the ecdf of
    /// the lowest value in the sample exists and is the lowest value itself.
    /// For instance in [0, 1] all the percentiles between 0 and 50% are 0 even
    /// though 0 is greater than or equal to a larger percentile of the values
    /// than each p in (0, 50).
    ///
    /// # Panics
    ///
    /// The percentile requested must be greater than 0 and less than or equal
    /// 100. In particular, there is no 0-percentile.
    ///
    /// # Examples
    ///
    /// ```
    /// extern crate kernel_density;
    ///
    /// let samples = vec!(9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0);
    /// let ecdf = kernel_density::density::Ecdf::new(&samples);
    /// assert_eq!(ecdf.percentile(50.0), 4.0);
    /// assert_eq!(ecdf.percentile(5.0), 0.0);
    /// ```
    pub fn percentile(&self, percentile: f64) -> f64 {
        assert!(0.0 < percentile && percentile <= 100.0);
        self.p(percentile / 100.0)
    }

    /// Calculate a rank element for the sample.
    ///
    /// # Panics
    ///
    /// The rank requested must be between 1 and the sample length inclusive.
    /// In particular, there is no 0-rank.
    ///
    /// # Examples
    ///
    /// ```
    /// extern crate kernel_density;
    ///
    /// let samples = vec!(9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0);
    /// let ecdf = kernel_density::density::Ecdf::new(&samples);
    /// assert_eq!(ecdf.rank(5), 4.0);
    /// ```
    pub fn rank(&self, rank: usize) -> f64 {
        let length = self.samples.len();
        assert!(0 < rank && rank <= length);
        self.samples[rank - 1]
    }

    /// Return the minimal element of the samples.
    ///
    /// # Examples
    ///
    /// ```
    /// extern crate kernel_density;
    ///
    /// let samples = vec!(9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0);
    /// let ecdf = kernel_density::density::Ecdf::new(&samples);
    /// assert_eq!(ecdf.min(), 0.0);
    /// ```
    pub fn min(&self) -> f64 {
        self.samples[0]
    }

    /// Return the maximal element of the samples.
    ///
    /// # Examples
    ///
    /// ```
    /// extern crate kernel_density;
    ///
    /// let samples = vec!(9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0);
    /// let ecdf = kernel_density::density::Ecdf::new(&samples);
    /// assert_eq!(ecdf.max(), 9.0);
    /// ```
    pub fn max(&self) -> f64 {
        let length = self.samples.len();
        self.samples[length - 1]
    }
}

/// Calculate a one-time value of the empirical cumulative distribution
/// function for a given sample.
///
/// Computational running time of this function is O(n) but does not amortize
/// across multiple calls like `Ecdf::value`. This function should only be
/// used in the case that a small number of ECDF values are required for the
/// sample. Otherwise, `Ecdf::new` should be used to create a structure that
/// takes the upfront O(n log n) sort cost but calculates values in O(log n).
///
/// # Panics
///
/// The sample set must be non-empty.
///
/// # Examples
///
/// ```
/// extern crate kernel_density;
///
/// let samples = vec!(9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0);
/// let value = kernel_density::density::ecdf(&samples, 4.0);
/// assert_eq!(value, 0.5);
/// ```
pub fn ecdf(samples: &[f64], x: f64) -> f64 {
    let mut num_samples_leq_x = 0;
    let mut length = 0;

    for sample in samples.iter() {
        length += 1;
        if *sample <= x {
            num_samples_leq_x += 1;
        }
    }

    assert!(length > 0);

    num_samples_leq_x as f64 / length as f64
}

/// Calculate a one-time proportion for a given sample using the Nearest Rank
/// method and Quick Select.
///
/// Note, the p-proportion of an ECDF is the _least_ number, n, for which at
/// least ratio p of the values in the ECDF are less than or equal to n.
///
/// This definition means the p-proportion for p between 0 and the ecdf of the
/// lowest value in the sample exists and is the lowest value itself. For
/// instance in [0, 1] all the p-proportions between 0 and .5 are 0 even though
/// 0 is greater than or equal to a larger proportion of the values than each p
/// in (0, 50).
///
/// Computational running time of this function is O(n) but does not amortize
/// across multiple calls like `Ecdf::percentile`. This function should only
/// becused in the case that a small number of percentiles are required for the
/// sample. Otherwise, `Ecdf::new` should be used to create a structure that
/// takes the upfront O(n log n) sort cost but calculates percentiles in O(1).
///
/// # Panics
///
/// The sample set must be non-empty.
///
/// The proportion requested must be greater than 0 and less than or equal 1.
/// In particular, there is no 0-proportion.
///
/// # Examples
///
/// ```
/// extern crate kernel_density;
///
/// let samples = vec!(9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0);
/// let percentile = kernel_density::density::p(&samples, 0.5);
/// assert_eq!(percentile, 4.0);
///
/// let percentile = kernel_density::density::p(&samples, 0.05);
/// assert_eq!(percentile, 0.0);
/// ```
pub fn p(samples: &[f64], proportion: f64) -> f64 {
    assert!(0.0 < proportion && proportion <= 1.0);

    let length = samples.len();
    assert!(length > 0);

    let r = (proportion * length as f64).ceil() as usize;

    rank(samples, r)
}

/// Calculate a one-time percentile for a given sample using the Nearest Rank
/// method and Quick Select.
///
/// Note, the p-percentile of an ECDF is the _least_ number, n, for which at
/// least p% of the values in the ECDF are less than or equal to n.
///
/// This definition means the p-percentiles for p between 0 and the ecdf of the
/// lowest value in the sample exists and is the lowest value itself. For
/// instance in [0, 1] all the percentiles between 0 and 50% are 0 even though
/// 0 is greater than or equal to a larger percentile of the values than each p
/// in (0, 50).
///
/// Computational running time of this function is O(n) but does not amortize
/// across multiple calls like `Ecdf::percentile`. This function should only
/// becused in the case that a small number of percentiles are required for the
/// sample. Otherwise, `Ecdf::new` should be used to create a structure that
/// takes the upfront O(n log n) sort cost but calculates percentiles in O(1).
///
/// # Panics
///
/// The sample set must be non-empty.
///
/// The percentile requested must be greater than 0 and less than or equal 100.
/// In particular, there is no 0-percentile.
///
/// # Examples
///
/// ```
/// extern crate kernel_density;
///
/// let samples = vec!(9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0);
/// let percentile = kernel_density::density::percentile(&samples, 50.0);
/// assert_eq!(percentile, 4.0);
///
/// let percentile = kernel_density::density::percentile(&samples, 5.0);
/// assert_eq!(percentile, 0.0);
/// ```
pub fn percentile(samples: &[f64], percentile: f64) -> f64 {
    assert!(0.0 < percentile && percentile <= 100.0);
    p(samples, percentile / 100.0)
}

/// Calculate a one-time rank for a given sample using Quick Select.
///
/// Computational running time of this function is O(n) and does not amortize
/// across multiple calls. This function should only be used in the case that a
/// small number of ranks are required for the sample.
///
/// # Panics
///
/// The sample set must be non-empty.
///
/// The rank requested must be between 1 and the sample length inclusive. In
/// particular, there is no 0-rank.
///
/// # Examples
///
/// ```
/// extern crate kernel_density;
///
/// let samples = vec!(9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0);
/// let rank = kernel_density::density::rank(&samples, 5);
/// assert_eq!(rank, 4.0);
/// ```
pub fn rank(samples: &[f64], rank: usize) -> f64 {
    let length = samples.len();
    assert!(length > 0);
    assert!(0 < rank && rank <= length);

    // Quick Select the element at rank.

    let mut samples: Vec<f64> = samples.to_vec();
    let mut low = 0;
    let mut high = length;

    loop {
        assert!(low < high);

        let pivot = samples[low].clone();

        if low >= high - 1 {
            return pivot;
        }

        // First determine if the rank item is less than the pivot.

        // Organise samples so that all items less than pivot are to the left,
        // `bottom` is the number of items less than pivot.

        let mut bottom = low;
        let mut top = high - 1;

        while bottom < top {
            while bottom < top && samples[bottom] < pivot {
                bottom += 1;
            }
            while bottom < top && samples[top] >= pivot {
                top -= 1;
            }

            if bottom < top {
                samples.swap(bottom, top);
            }
        }

        if rank <= bottom {
            // Rank item is less than pivot. Exclude pivot and larger items.
            high = bottom;
        } else {
            // Rank item is pivot or in the larger set. Exclude smaller items.
            low = bottom;

            // Next, determine if the pivot is the rank item.

            // Organise samples so that all items less than or equal to pivot
            // are to the left, `bottom` is the number of items less than or
            // equal to pivot. Since the left is already less than the pivot,
            // this just requires moving the pivots left also.

            let mut bottom = low;
            let mut top = high - 1;

            while bottom < top {
                while bottom < top && samples[bottom] == pivot {
                    bottom += 1;
                }
                while bottom < top && samples[top] != pivot {
                    top -= 1;
                }

                if bottom < top {
                    samples.swap(bottom, top);
                }
            }

            // Is pivot the rank item?

            if rank <= bottom {
                return pivot;
            }

            // Rank item is greater than pivot. Exclude pivot, smaller items.
            low = bottom;
        }
    }
}

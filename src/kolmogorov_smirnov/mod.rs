//! Two sample Kolmogorov-Smirnov test.

/// Two sample test result.
pub struct TestResult {
    pub is_rejected: bool,
    pub statistic: f64,
    pub reject_probability: f64,
    pub critical_value: f64,
    pub confidence: f64,
}

/// Perform a two sample Kolmogorov-Smirnov test on given samples.
///
/// The samples must have length > 7 elements for the test to be valid.
///
/// # Panics
///
/// There are assertion panics if either sequence has <= 7 elements or
/// if the requested confidence level is not between 0 and 1.
///
/// # Examples
///
/// ```
/// extern crate kernel_density;
///
/// let xs = vec!(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0);
/// let ys = vec!(12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0);
/// let confidence = 0.95;
///
/// let result = kernel_density::kolmogorov_smirnov::test(&xs, &ys, confidence);
///
/// if result.is_rejected {
///     println!("{:?} and {:?} are not from the same distribution with probability {}.",
///       xs, ys, result.reject_probability);
/// }
/// ```
pub fn test(xs: &[f64], ys: &[f64], confidence: f64) -> TestResult {
    assert!(0.0 < confidence && confidence < 1.0);

    // Only supports samples of size > 7.
    assert!(xs.len() > 7 && ys.len() > 7);

    let statistic = calculate_statistic(xs, ys);
    let critical_value = calculate_critical_value(xs.len(), ys.len(), confidence);

    let reject_probability = calculate_reject_probability(statistic, xs.len(), ys.len());
    let is_rejected = reject_probability > confidence;

    TestResult {
        is_rejected: is_rejected,
        statistic: statistic,
        reject_probability: reject_probability,
        critical_value: critical_value,
        confidence: confidence,
    }
}


/// Calculate the test statistic for the two sample Kolmogorov-Smirnov test.
///
/// The test statistic is the maximum vertical distance between the ECDFs of
/// the two samples.
fn calculate_statistic(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len();
    let m = ys.len();

    assert!(n > 0 && m > 0);

    let mut xs = xs.to_vec();
    let mut ys = ys.to_vec();

    // xs and ys must be sorted for the stepwise ECDF calculations to work.
    xs.sort_by(|x_1, x_2| x_1.partial_cmp(x_2).unwrap());
    ys.sort_by(|y_1, y_2| y_1.partial_cmp(y_2).unwrap());

    // The current value testing for ECDF difference. Sweeps up through elements
    // present in xs and ys.
    let mut current: f64;

    // i, j index the first values in xs and ys that are greater than current.
    let mut i = 0;
    let mut j = 0;

    // ecdf_xs, ecdf_ys always hold the ECDF(current) of xs and ys.
    let mut ecdf_xs = 0.0;
    let mut ecdf_ys = 0.0;

    // The test statistic value computed over values <= current.
    let mut statistic = 0.0;

    while i < n && j < m {
        // Advance i through duplicate samples in xs.
        let x_i = xs[i];
        while i + 1 < n && x_i == xs[i + 1] {
            i += 1;
        }

        // Advance j through duplicate samples in ys.
        let y_j = ys[j];
        while j + 1 < m && y_j == ys[j + 1] {
            j += 1;
        }

        // Step to the next sample value in the ECDF sweep from low to high.
        current = x_i.min(y_j);

        // Update invariant conditions for i, j, ecdf_xs, and ecdf_ys.
        if current == x_i {
            ecdf_xs = (i + 1) as f64 / n as f64;
            i += 1;
        }
        if current == y_j {
            ecdf_ys = (j + 1) as f64 / m as f64;
            j += 1;
        }

        // Update invariant conditions for the test statistic.
        let diff = (ecdf_xs - ecdf_ys).abs();
        if diff > statistic {
            statistic = diff;
        }
    }

    // Don't need to walk the rest of the samples because one of the ecdfs is
    // already one and the other will be increasing up to one. This means the
    // difference will be monotonically decreasing, so we have our test
    // statistic value already.

    statistic
}

/// Calculate the probability that the null hypothesis is false for a two sample
/// Kolmogorov-Smirnov test. Can only reject the null hypothesis if this
/// evidence exceeds the confidence level required.
fn calculate_reject_probability(statistic: f64, n1: usize, n2: usize) -> f64 {
    // Only supports samples of size > 7.
    assert!(n1 > 7 && n2 > 7);

    let n1 = n1 as f64;
    let n2 = n2 as f64;

    let factor = ((n1 * n2) / (n1 + n2)).sqrt();
    let term = (factor + 0.12 + 0.11 / factor) * statistic;

    let reject_probability = 1.0 - probability_kolmogorov_smirnov(term);

    assert!(0.0 <= reject_probability && reject_probability <= 1.0);
    reject_probability
}

/// Calculate the critical value for the two sample Kolmogorov-Smirnov test.
///
/// # Panics
///
/// There are assertion panics if either sequence size is <= 7 or if the
/// requested confidence level is not between 0 and 1.
///
/// No convergence panic if the binary search does not locate the critical
/// value in less than 200 iterations.
///
/// # Examples
///
/// ```
/// extern crate kernel_density;
///
/// let critical_value = kernel_density::kolmogorov_smirnov::calculate_critical_value(
///       256, 256, 0.95);
/// println!("Critical value at 95% confidence for samples of size 256 is {}",
///       critical_value);
/// ```
pub fn calculate_critical_value(n1: usize, n2: usize, confidence: f64) -> f64 {
    assert!(0.0 < confidence && confidence < 1.0);

    // Only supports samples of size > 7.
    assert!(n1 > 7 && n2 > 7);

    // The test statistic is between zero and one so can binary search quickly
    // for the critical value.
    let mut low = 0.0;
    let mut high = 1.0;

    for _ in 1..200 {
        if low + 1e-8 >= high {
            return high;
        }

        let mid = low + (high - low) / 2.0;
        let reject_probability = calculate_reject_probability(mid, n1, n2);

        if reject_probability > confidence {
            // Maintain invariant that reject_probability(high) > confidence.
            high = mid;
        } else {
            // Maintain invariant that reject_probability(low) <= confidence.
            low = mid;
        }
    }

    panic!("No convergence in calculate_critical_value({}, {}, {}).",
           n1,
           n2,
           confidence);
}

/// Calculate the Kolmogorov-Smirnov probability function.
fn probability_kolmogorov_smirnov(lambda: f64) -> f64 {
    if lambda == 0.0 {
        return 1.0;
    }

    let minus_two_lambda_squared = -2.0 * lambda * lambda;
    let mut q_ks = 0.0;

    for j in 1..200 {
        let sign = if j % 2 == 1 {
            1.0
        } else {
            -1.0
        };

        let j = j as f64;
        let term = sign * 2.0 * (minus_two_lambda_squared * j * j).exp();

        q_ks += term;

        if term.abs() < 1e-8 {
            // Trim results that exceed 1.
            return q_ks.min(1.0);
        }
    }

    panic!("No convergence in probability_kolmogorov_smirnov({}).",
           lambda);
}

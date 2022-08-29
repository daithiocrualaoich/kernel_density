extern crate kernel_density;

use kernel_density::density;

use std::env;

fn parse_float(s: String) -> f64 {
    s.parse::<f64>().expect("Not a floating point number.")
}

/// Calculate Normal density values.
///
/// # Examples
///
/// ```bash
/// cargo run --bin normal_density <min> <max> <mean> <variance>
/// ```
///
/// This will print the values of the Normal density with given mean and
/// variance for values between min and max using 0.01 as step size.
///
/// `<variance>` must be a floating point number strictly greater than zero.
/// `<min>` and `<max>` must be floating point numbers with `<min>` less than
/// `<max>`. `<mean>` can be any floating point number.
fn main() {
    let args: Vec<String> = env::args().collect();

    let min: f64 = parse_float(args[1].clone());
    let max: f64 = parse_float(args[2].clone());
    let mean: f64 = parse_float(args[3].clone());
    let variance: f64 = parse_float(args[4].clone());

    assert!(variance > 0.0);
    assert!(min <= max);

    let density = density::normal(mean, variance);

    println!("x\tdensity\tcdf");
    println!("{}\t{}\t{}", min, density.density(min), density.cdf(min));

    // Iterate using fixed point arithmetic over a 0.01 grid resolution.
    let mut x_fixed: i64 = (min * 100.0).floor() as i64 + 1;
    let mut x_f64: f64 = x_fixed as f64 / 100.0;

    while x_f64 < max {
        println!(
            "{}\t{}\t{}",
            x_f64,
            density.density(x_f64),
            density.cdf(x_f64)
        );

        x_fixed += 1;
        x_f64 = x_fixed as f64 / 100.0;
    }

    println!("{}\t{}\t{}", max, density.density(max), density.cdf(max));
}

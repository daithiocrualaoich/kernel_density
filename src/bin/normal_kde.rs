extern crate kernel_density;

use kernel_density::kde;

use std::env;
use std::io::{BufReader, BufRead};
use std::fs::File;
use std::path::Path;

fn parse_float(s: String) -> f64 {
    s.parse::<f64>().expect("Not a floating point number.")
}

/// Calculate Normal Kernel Density Estimation values.
///
/// Input files must be single-column headerless data files.
///
/// # Examples
///
/// ```bash
/// cargo run --bin uniform_kde <min> <max> <step> <bandwidth> <file>
/// ```
///
/// This will print the values of the Normal KDE for values between min and
/// max using 0.01 as step size.
///
/// `<bandwidth>` must be a floating point number strictly greater than zero.
/// `<min>` and `<max>` must be floating point numbers with `<min>` less than
/// `<max>`.
fn main() {
    let args: Vec<String> = env::args().collect();

    let min: f64 = parse_float(args[1].clone());
    let max: f64 = parse_float(args[2].clone());
    let bandwidth: f64 = parse_float(args[3].clone());

    assert!(bandwidth > 0.0);
    assert!(min <= max);

    let path = Path::new(&args[4]);
    let file = BufReader::new(File::open(&path).unwrap());
    let lines = file.lines().map(|line| line.unwrap());

    let xs: Vec<f64> = lines.map(parse_float).collect();

    let kde = kde::normal(&xs, bandwidth);

    println!("x\tkde\tcdf");
    println!("{}\t{}\t{}", min, kde.density(min), kde.cdf(min));

    // Iterate using fixed point arithmetic over a 0.01 grid resolution.
    let mut x_fixed: i64 = (min * 100.0).floor() as i64 + 1;
    let mut x_f64: f64 = x_fixed as f64 / 100.0;

    while x_f64 < max {
        println!("{}\t{}\t{}", x_f64, kde.density(x_f64), kde.cdf(x_f64));

        x_fixed += 1;
        x_f64 = x_fixed as f64 / 100.0;
    }

    println!("{}\t{}\t{}", max, kde.density(max), kde.cdf(max));
}

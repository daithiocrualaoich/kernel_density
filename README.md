I don't know if the original author is ever going to respond to my PR, but I basically factored out dependence on `cephes` so it could work with `wasm`. You can use my fork in `Cargo.toml` like the following

``` toml
[dependencies]
kernel_density = { git = "https://github.com/quinn-dougherty/kernel_density", version = "0.0.3" }
```

[![Build Status](https://travis-ci.org/daithiocrualaoich/kernel_density.svg?branch=master)](https://travis-ci.org/daithiocrualaoich/kernel_density)
[![Crates.io](https://img.shields.io/crates/v/kernel_density.svg)](https://crates.io/crates/kernel_density)
[![License](https://img.shields.io/crates/l/kernel_density.svg)](https://github.com/daithiocrualaoich/kernel_density/blob/master/LICENSE)
[![Cargo test CI](https://github.com/quinn-dougherty/kernel_density/actions/workflows/ci.yml/badge.svg)](https://github.com/quinn-dougherty/kernel_density/actions/workflows/ci.yml)

Implementation of Kernel Density Estimation as a Rust library.

Getting Started
---------------
The Kernel Density Estimation library is available as a crate, so it is easy to
incorporate into your programs. Add the dependency to your `Cargo.toml` file.

    [dependencies]
    kernel_density = "0.0.3"

Information about the latest published crate is available on
[crates.io](https://crates.io/crates/kernel_density).


Developing Kernel Density
-------------------------
Install the [Rust] development tools on your system if they are not already
available. Then build and test the library using:

    cargo test

[Rust]: https://www.rust-lang.org


Docker
------
A [Docker] container definition is provided with installations of the tools
used to develop the software. To use the container, first install Docker if not
already available and start a Docker terminal. Then create the container by
running the following build at the top level of the repository source tree:

    docker build --rm=true -t statistics .

[Docker]: http://docker.io

Once built, an interactive shell can be run in the container using:

    docker run -it -v "$(pwd):/statistics" --workdir=/statistics statistics /bin/bash

The current working directory from the host machine is available as the current
directory in the container so it is possible to build and test the library as
described earlier.

    cargo test


Publishing on crates.io
-----------------------
Instructions for uploading to the crate repository at crates.io are
[here](http://doc.crates.io/crates-io.html#publishing-crates). First login to
the site using:

    cargo login <token>

Token can be found from [crates.io/me](https://crates.io/me). To make a release,
first clean and build the package:

    git stash
    cargo clean
    cargo package

Examine the built package under `target/package/kernel_density-<version>`.
And when happy to publish:

    cargo publish

And check out the new update at
[crates.io/crates/kernel_density](https://crates.io/crates/kernel_density).


License
-------

    Copyright [2016] [Daithi O Crualaoich]

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

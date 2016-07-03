Implementation of Kernel Density Estimation as a Rust library.


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
directory in the container.


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

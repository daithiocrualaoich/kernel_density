FROM        ubuntu:xenial
MAINTAINER  Daithi O Crualaoich


################################################################################
# Basic Development Tools
################################################################################

RUN     apt-get update -qq
RUN     apt-get upgrade -qq

RUN     apt-get install -qq wget
RUN     apt-get install -qq build-essential gcc


################################################################################
# LaTeX
################################################################################

RUN     apt-get install -qq texlive
RUN     apt-get install -qq texlive-latex-extra dvipng


################################################################################
# Pandoc
################################################################################

RUN     apt-get install -qq pandoc


################################################################################
# Sphinx
################################################################################

RUN     apt-get install -qq python2.7 python2.7-dev python-pip
RUN     pip install Sphinx sphinxcontrib-googleanalytics
RUN     pip install cloud_sptheme


################################################################################
# R
################################################################################

RUN     apt-get install -qq libjpeg62 libcairo2-dev libcurl4-openssl-dev \
            libxml2-dev libxt-dev libxaw7-dev libssl-dev

RUN     gpg --keyserver keyserver.ubuntu.com --recv-key E084DAB9
RUN     gpg -a --export E084DAB9 | apt-key add -
RUN     echo 'deb http://cran.r-project.org/bin/linux/ubuntu xenial/' > /etc/apt/sources.list.d/cran.list

RUN     apt-get install -qq r-base

RUN     echo 'update.packages(ask = FALSE, repos="http://cran.r-project.org")' | R --vanilla
RUN     echo 'install.packages(c("markdown"), repos="http://cran.r-project.org", dependencies=TRUE)' | R --vanilla
RUN     echo 'install.packages(c("knitr"), repos="http://cran.r-project.org", dependencies=TRUE)' | R --vanilla
RUN     echo 'install.packages(c("Cairo"), repos="http://cran.r-project.org", dependencies=TRUE)' | R --vanilla
RUN     echo 'install.packages(c("ggplot2"), repos="http://cran.r-project.org", dependencies=TRUE)' | R --vanilla
RUN     echo 'install.packages(c("scales"), repos="http://cran.r-project.org", dependencies=TRUE)' | R --vanilla


################################################################################
# Rust
################################################################################

RUN    apt-get install -qq curl graphviz

RUN    curl https://sh.rustup.rs -sSf | sh -s -- -y
RUN    echo '\nexport PATH=$PATH:/root/.cargo/bin\n' >> /root/.bashrc

RUN    /root/.cargo/bin/rustup install nightly

RUN    /root/.cargo/bin/cargo install rustfmt
RUN    /root/.cargo/bin/cargo install cargo-watch
RUN    /root/.cargo/bin/cargo install cargo-outdated
RUN    /root/.cargo/bin/cargo install cargo-graph
RUN    /root/.cargo/bin/cargo install cargo-count

RUN    /root/.cargo/bin/rustup default nightly
RUN    /root/.cargo/bin/cargo install clippy 
RUN    /root/.cargo/bin/rustup default stable

{
  description = "Kernel Density CI";

  inputs = {
    nixpkgs.url = "nixpkgs/nixos-22.05";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    flake-utils = {
      url = "github:numtide/flake-utils";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, rust-overlay, flake-utils, ... }:
    let
      overlays = [ rust-overlay.overlays.default ];
      rust-fn = pkgs:
        pkgs.rust-bin.fromRustupToolchainFile ./nix/rust-toolchain.toml;
      build-fn = pkgs:
        pkgs.rustPlatform.buildRustPackage {
          pname = "kernel_density";
          version = "0.0.3";
          src = ./.;
          nativeBuildInputs = [ (rust-fn pkgs) pkgs.wasm-pack ];
          buildPhase = ''
            cargo build
          '';
          installPhase = "cargo test && mkdir -p $out";
          cargoLock.lockFile = ./Cargo.lock;
        };
      lint-fn = pkgs:
        pkgs.stdenv.mkDerivation {
          name = "kernel_density-lint";
          src = ./.;
          buildInputs = with pkgs; [ rustfmt cargo ];
          buildPhase = "cargo fmt --check";
          installPhase = "mkdir -p $out";
        };
      # local machines
      localFlake = { pkgs, ... }: {
        # validating
        checks = flake-utils.lib.flattenTree { lint = lint-fn pkgs; };
        # building
        packages = flake-utils.lib.flattenTree { default = build-fn pkgs; };

        # developing
        devShells = flake-utils.lib.flattenTree {
          default = pkgs.mkShell {
            name = "kernel_density-devshell";
            buildInputs = with pkgs; [
              wasm-pack
              cargo
              rustup
              libressl
              nixfmt
              rustfmt
              binaryen
              wasm-bindgen-cli
            ];
          };
        };
      };

      # ci
      herc = let
        hciSystem = "x86_64-linux";
        hciPkgs = import nixpkgs {
          system = hciSystem;
          overlays = overlays;
        };
      in {
        herculesCI = {
          ciSystems = [ hciSystem ];
          onPush = {
            kernel_density.outputs = {
              lint = lint-fn hciPkgs;
              build = build-fn hciPkgs;
            };
          };
        };
      };
    in flake-utils.lib.eachDefaultSystem (system:
      let
        # globals
        pkgs = import nixpkgs {
          inherit system;
          overlays = overlays;
        };

      in localFlake pkgs) // herc;
}

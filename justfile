set positional-arguments

default:
    @just --list

build:
    cargo build --workspace

check:
    cargo check --workspace
    cargo fmt --all -- --check
    cargo clippy --all-targets --features candle -- -D warnings
    cargo audit
    cargo deny check all
    cargo test --workspace
    cargo test --workspace --features candle
    cargo test --workspace --no-default-features

install:
    cargo install --path . --bin open-stt-server

run *args:
    cargo run --bin open-stt-server -- {{args}}

test:
    cargo test --workspace

pre-commit:
    ./scripts/scan-staged-secrets.sh
    cargo fmt --all
    cargo clippy --all-targets -- -D warnings
    cargo audit
    cargo deny check all
    cargo test

release *args:
    git pull --rebase
    cargo release {{args}}

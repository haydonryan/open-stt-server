set positional-arguments

default:
    @just --list

build:
    cargo build --workspace

check:
    cargo check --workspace
    cargo fmt --all -- --check
    cargo clippy --all-targets --features candle -- -D warnings -W clippy::pedantic -W clippy::nursery
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
    #!/usr/bin/env bash
    set -euo pipefail
    ./scripts/scan-staged-secrets.sh
    before_fmt_diff="$(mktemp)"
    after_fmt_diff="$(mktemp)"
    trap 'rm -f "$before_fmt_diff" "$after_fmt_diff"' EXIT
    git diff --name-only -- . >"$before_fmt_diff"
    cargo fmt --all
    git diff --name-only -- . >"$after_fmt_diff"
    if ! cmp -s "$before_fmt_diff" "$after_fmt_diff"; then
      echo "cargo fmt updated files. Review and stage the formatting changes, then commit again." >&2
      exit 1
    fi
    cargo clippy --all-targets --features candle -- -D warnings -W clippy::pedantic -W clippy::nursery
    cargo audit
    cargo deny check all
    cargo test

release *args:
    git pull --rebase
    cargo release {{args}}

bump version:
    cargo release version {{version}} --execute --no-confirm

docker-build:
    ./scripts/docker-build.sh

deploy:
    just docker-build
    docker compose up -d --force-recreate --remove-orphans

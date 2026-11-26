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

install:
    cargo install --path . --bin open-stt-server

run *args:
    cargo run --bin open-stt-server -- {{args}}

test:
    #!/usr/bin/env bash
    set -euo pipefail
    if command -v cargo-nextest >/dev/null 2>&1; then
      cargo nextest run --workspace --no-fail-fast
    else
      cargo test --workspace
    fi

pre-commit:
    #!/usr/bin/env bash
    set -euo pipefail
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
    ./scripts/scan-staged-secrets.sh

release *args:
    git pull --rebase
    just check && cargo release {{args}}

bump version:
    cargo release version {{version}} --execute --no-confirm

docker-build:
    ./scripts/docker-build.sh

deploy:
    just docker-build
    docker compose up -d --force-recreate --remove-orphans

nemotron-build:
    docker build -f Dockerfile.nemotron -t ${NEMOTRON_IMAGE:-open-stt-server:nemotron} .

nemotron-deploy:
    docker compose --profile nemotron up -d --force-recreate

# Sweep stale Cargo target artifacts while keeping artifacts resolved by build/test.
gc-target:
    cargo mark-sweep --cmd "build --workspace" --cmd "test --no-run --workspace"

# Preview the same sweep without deleting anything.
gc-target-dry:
    cargo mark-sweep --dry-run --cmd "build --workspace" --cmd "test --no-run --workspace"

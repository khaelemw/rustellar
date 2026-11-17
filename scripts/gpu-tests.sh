#!/bin/sh
# The GPU tests, on this machine's NVIDIA GPU: CI has none, so run this before
# pushing changes to the GPU code. A missing GPU is a failure here, not a skip.
#
#   scripts/gpu-tests.sh             every rustellar-bls test with the feature
#                                    cuda (searches on the GPU), the heavy GPU
#                                    comparisons too, but not the studies
#   scripts/gpu-tests.sh --coverage  the same but the heavy ones, under
#                                    cargo-llvm-cov: rustellar-bls's line
#                                    coverage, its GPU code included
set -eu
cd "$(dirname "$0")/.."
export RUSTELLAR_REQUIRE_GPU=1

# Every test target but the two studies (recovery and limb), which the
# heavy CI workflow runs.
targets="--lib --test allocations --test autoperiod --test box_fit --test cuda
    --test exact --test inputs --test ofir --test precision --test search
    --test survey"

if [ "${1:-}" = "--coverage" ]; then
    cargo llvm-cov clean -p rustellar-bls
    # shellcheck disable=SC2086
    cargo llvm-cov --release -p rustellar-bls --features cuda $targets --summary-only
else
    # shellcheck disable=SC2086
    cargo test --release -p rustellar-bls --features cuda $targets
    cargo test --release -p rustellar-bls --features cuda --test cuda -- --ignored
fi

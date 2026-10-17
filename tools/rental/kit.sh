#!/usr/bin/env bash
# Builds the rental kit on the trusted machine: one tarball holding the
# committed tree's rig, entries/combined, the vectors and these scripts, plus
# prebuilt fat binaries (gfx90a, gfx942, gfx950) of every build in builds.txt
# and of the microbenchmark, compiled offline in the ROCm 7.2.4 container from
# the same committed sources. No keys or results go in.
#
# Usage: tools/rental/kit.sh [OUT.tar.gz]   (default: dist/rental-kit-<commit>.tar.gz)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
IMAGE=rocm/dev-ubuntu-24.04:7.2.4
ARCHS=(gfx90a gfx942 gfx950)
# entries/baseline/src: the microbenchmark borrows its field code.
PATHS=(pyproject.toml uv.lock README.md LICENSE src vectors entries/combined entries/baseline/src
       tools/microbench tools/rental tools/sol/setup.sh tools/sol/env.sh)

cd "$ROOT"
commit=$(git rev-parse --short HEAD)
if [ -n "$(git status --porcelain -- "${PATHS[@]}")" ]; then
    echo "warning: uncommitted changes are not in the kit; it packs HEAD ($commit)" >&2
fi
out=$(realpath -m "${1:-dist/rental-kit-$commit.tar.gz}")
mkdir -p "$(dirname "$out")"
stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT
mkdir "$stage/kit"
git archive HEAD "${PATHS[@]}" | tar -x -C "$stage/kit"

docker run --rm --network none --user "$(id -u):$(id -g)" -e HOME=/tmp \
    -v "$stage/kit:/kit" "$IMAGE" bash /kit/tools/rental/prebuild.sh /kit "$commit" "${ARCHS[@]}"
tar -C "$stage" -czf "$out" kit
echo "wrote $out ($(du -h "$out" | cut -f1)); next: tools/rental/README.md"

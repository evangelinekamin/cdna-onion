#!/usr/bin/env bash
# One-time setup on a shared cluster's login node: a pinned, checksum-verified uv,
# a uv-managed Python, and the locked environment, all inside this repo (.uv/ and
# .venv/). No shell startup file changes, so deleting the repo removes every trace.
#
# Usage: tools/sol/setup.sh        (then: sbatch tools/sol/mi210-bench.sbatch)
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
UV_VERSION=0.10.9
UV_ASSET=uv-x86_64-unknown-linux-musl.tar.gz  # static, so any glibc works
UV_SHA256=433e56874739e92c7cfd661ba9e5f287b376ca612c08c8194a41a98a13158aea
PYTHON_VERSION=3.14                           # what the development machine runs

if [ ! -x "$ROOT/.uv/bin/uv" ]; then
    TMP=$(mktemp -d)
    trap 'rm -rf "$TMP"' EXIT
    curl -fsSL --retry 2 -o "$TMP/$UV_ASSET" \
        "https://github.com/astral-sh/uv/releases/download/$UV_VERSION/$UV_ASSET"
    echo "$UV_SHA256  $TMP/$UV_ASSET" | sha256sum -c --quiet
    mkdir -p "$ROOT/.uv/bin"
    tar -xzf "$TMP/$UV_ASSET" -C "$ROOT/.uv/bin" --strip-components=1
fi

# shellcheck source=env.sh
. "$ROOT/tools/sol/env.sh"
uv --version
uv sync --project "$ROOT" --locked --python "$PYTHON_VERSION"
uv run --project "$ROOT" onionrig vectors --check

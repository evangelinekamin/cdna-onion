#!/usr/bin/env bash
# Set up one competitor box from the organizer's machine: provision it as root,
# then give the `agent` user a workspace (a fresh clone of main, the AMD
# reference PDFs, the rig's Python environment) and prove the toolchain builds
# the baseline for MI210 and MI355X. An existing workspace is never touched, so
# this is safe to run again.
#
# Usage: tools/arena/setup-box.sh SSH_HOST LABEL     e.g. arena-gemini gemini
set -euo pipefail
[ $# -eq 2 ] || { echo "usage: $0 SSH_HOST LABEL" >&2; exit 2; }
HOST=$1
LABEL=$2
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
AMD_DOCS=(instinct-mi200-cdna2-instruction-set-architecture amd-instinct-cdna4-instruction-set-architecture
          amd-instinct-cdna5-instruction-set-architecture amd-cdna2-white-paper amd-cdna5-whitepaper)

echo "== provisioning $HOST"
ssh "root@$HOST" 'bash -s' < "$ROOT/tools/arena/provision.sh"

echo "== workspace"
if ssh "agent@$HOST" 'test -e ~/onionrig'; then
    echo "the agent's onionrig workspace already exists on $HOST; leaving it as it is"
else
    # A bundle of main carries tracked files only: no internal notes, local
    # results, or organizer memory can travel, and the clone has no remote.
    git -C "$ROOT" bundle create - HEAD main 2> /dev/null | ssh "agent@$HOST" "set -e
        cat > ~/.onionrig.bundle
        git clone -q ~/.onionrig.bundle ~/onionrig
        rm ~/.onionrig.bundle
        cd ~/onionrig
        git remote remove origin
        git config user.name 'arena-$LABEL'
        git config user.email 'arena-$LABEL@localhost'"
    pdfs=()
    for doc in "${AMD_DOCS[@]}"; do pdfs+=("$ROOT/docs/$doc.pdf"); done
    rsync -a "${pdfs[@]}" "agent@$HOST:onionrig/docs/"
fi

echo "== rig environment and toolchain check"
ssh "agent@$HOST" 'set -eo pipefail
    cd ~/onionrig
    tools/sol/setup.sh | tail -1
    tools/fetch-amd-docs.sh
    make -s -C entries/baseline hip HIP_ARCHS="gfx90a gfx950"
    make -s -C tools/microbench hip HIP_ARCHS="gfx90a gfx950"
    make -s -C entries/baseline resource-usage HIP_ARCHS="gfx90a gfx950" | grep -E "^==|VGPRs:|Occupancy"
    echo "commit $(git log -1 --format=%h), clean: $(git status --porcelain | wc -l) changes"'
echo "setup-box: $HOST OK"

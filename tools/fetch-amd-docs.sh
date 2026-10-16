#!/usr/bin/env bash
# Download the AMD documents that docs/design cites into docs/. They are AMD's
# copyrighted publications, so this repo links to them instead of shipping them.
# Each file is checked against the SHA-256 of the edition the notes were written
# from; any other edition is kept, with a warning that cited sections may have moved.
#
# Usage: tools/fetch-amd-docs.sh
set -euo pipefail

DOCS=$(cd "$(dirname "$0")/../docs" && pwd)
AMD=https://www.amd.com/content/dam/amd/en/documents
# amd.com rejects curl's default HTTP/2 client; HTTP/1.1 with a generic UA works.
CURL=(curl -fsSL --http1.1 -A Mozilla/5.0 --retry 2 --max-time 300)
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

check() {  # check FILE SHA256: say whether FILE is the cited edition
    if echo "$2  $1" | sha256sum -c --status; then echo "ok       ${1##*/}"
    else echo "ok       ${1##*/} (not the edition the notes cite; sections may have moved)"; fi
}

fetch() {  # fetch PATH SHA256
    local name=${1##*/}
    if [ -f "$DOCS/$name" ]; then check "$DOCS/$name" "$2"; return; fi
    "${CURL[@]}" -o "$TMP/$name" "$AMD/$1" || { echo "FAILED   $name (download)" >&2; return 1; }
    [ "$(head -c 5 "$TMP/$name")" = "%PDF-" ] || { echo "FAILED   $name (not a PDF)" >&2; return 1; }
    mv "$TMP/$name" "$DOCS/$name"
    check "$DOCS/$name" "$2"
}

status=0
ISA=instinct-tech-docs/instruction-set-architectures
WP=instinct-business-docs/white-papers
fetch $ISA/instinct-mi200-cdna2-instruction-set-architecture.pdf \
    6c8b6e86fddb489437765c1d4c290c223e7668576bbd2ea9c2ebb4d45e3e2212 || status=1
fetch $ISA/amd-instinct-cdna4-instruction-set-architecture.pdf \
    c459fb8db759006da52df57b5d7b913686ed7a68b52dc76e3f557f553a5d667f || status=1
fetch $ISA/amd-instinct-cdna5-instruction-set-architecture.pdf \
    5fae9822405c04f3e380c38d55b27df5b19469b78d5dfdd87c319a712f0239a8 || status=1
fetch $WP/amd-cdna2-white-paper.pdf \
    62125a4a3f00e653556ca98c178712496b241e5c3d8a4edadee7578944c2cc6d || status=1

# No stable public URL was found for this one; get it from AMD's Instinct pages.
CDNA5_WP=2381d60185f79989d3d5e4260c86f72504fcab256b40bb85fe4a6dd782afb3ef
if [ -f "$DOCS/amd-cdna5-whitepaper.pdf" ]; then check "$DOCS/amd-cdna5-whitepaper.pdf" "$CDNA5_WP"
else echo "MANUAL   amd-cdna5-whitepaper.pdf: download the CDNA 5 white paper from amd.com into docs/ (cited edition sha256 $CDNA5_WP)"; fi
exit $status

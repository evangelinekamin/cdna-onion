#!/bin/bash
# Builds an A/B probe directory for `submit check`.
#   mkprobe.sh OUTDIR NAME=SRCDIR[:DEFS] ...
# SRCDIR is an entry directory (with Makefile and src/). Each variant builds as
# OUTDIR/NAME/entry with `make entry DEFS=...`; OUTDIR/entry runs them in
# interleaved rounds (probe/run.sh). Knobs: OUTDIR/ROUNDS, SECONDS, ARGS.
set -euo pipefail
out=$1; shift
here=$(cd "$(dirname "$0")" && pwd)
rm -rf "$out"; mkdir -p "$out"
names=()
for spec in "$@"; do
    name=${spec%%=*}; rest=${spec#*=}
    src=${rest%%:*}; defs=""
    [[ "$rest" == *:* ]] && defs=${rest#*:}
    mkdir -p "$out/$name"
    cp -r "$src/src" "$src/Makefile" "$out/$name/"
    echo "$defs" > "$out/$name/DEFS"
    names+=("$name")
done
echo "${names[@]}" > "$out/VARIANTS"
cp "$here/run.sh" "$out/run.sh"
cat > "$out/Makefile" <<'MK'
GPU_ARCH ?= gfx90a
VARIANTS := $(shell cat VARIANTS)
# One target per variant, so `make -j` builds them in parallel.
entry: $(addsuffix /entry,$(VARIANTS))
	cp run.sh entry && chmod +x entry
%/entry:
	$(MAKE) -C $* entry GPU_ARCH=$(GPU_ARCH) DEFS="$$(cat $*/DEFS)"
MK
echo "probe in $out: ${names[*]}"

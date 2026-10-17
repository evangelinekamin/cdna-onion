#!/bin/sh
# A/B probe launcher (see mkprobe.sh): installed as ./entry in a probe directory.
# --verify times every variant in interleaved rounds. The rig shows only the
# last 200 characters of a verify stdout that is not a JSON summary, so the
# result is one compact line: per variant, the mean rate in G/s and the spread
# (max - min) over rounds.
cd "$(dirname "$0")" || exit 2
case "$1" in
--info)
    echo '{"contract": "onionrig-grind/1", "name": "onion-walk-probe", "version": "0", "model": "Claude Opus 5.5"}'
    ;;
--verify)
    ROUNDS=$(cat ROUNDS 2>/dev/null || echo 3)
    SECS=$(cat SECONDS 2>/dev/null || echo 6)
    ARGS=$(cat ARGS 2>/dev/null || echo "4 5")
    all=""
    prof=""
    r=1
    while [ "$r" -le "$ROUNDS" ]; do
        for v in $(cat VARIANTS); do
            out=$(./"$v"/entry --bench "$SECS" $ARGS 2>/dev/null)
            # Each cell's own JSON line, as it lands, on stderr (a Slurm log
            # shows it at once; the rig keeps stdout for the summary).
            echo "$out" | tail -n 1 | sed "s/^{/{\"round\": $r, \"variant\": \"$v\", /" >&2
            rate=$(echo "$out" | sed -n 's/.*"rate": \([0-9.e+]*\).*/\1/p')
            all="$all $v:${rate:-0}"
            # Profile builds: keep "clock MHz / alive fraction" per variant.
            p=$(echo "$out" | sed -n 's/.*"prof": \[\(.*\)\].*/\1/p' | tr -d ' ' |
                awk -F, '{ if ($3 > 0) printf "%d/%.2f", $1, $2 / $3 }')
            [ -n "$p" ] && prof="$prof $v:$p"
        done
        r=$((r + 1))
    done
    echo "$all" | tr ' ' '\n' | awk -F: 'NF == 2 {
        n[$1]++; s[$1] += $2; if (!($1 in lo) || $2 < lo[$1]) lo[$1] = $2; if ($2 > hi[$1]) hi[$1] = $2
        if (!($1 in seen)) { seen[$1] = 1; order[++k] = $1 } }
        END { line = "G/s"; for (i = 1; i <= k; i++) { v = order[i]; m = s[v] / n[v]
              line = line sprintf(" %s=%.3f~%.1f", v, m / 1e9, m > 0 ? 100 * (hi[v] - lo[v]) / m : 0) }
              print line }' | tr -d '\n'
    [ -n "$prof" ] && printf ' P%s' "$prof"
    echo
    exit 1
    ;;
*)
    exit 2
    ;;
esac

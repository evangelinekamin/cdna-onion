#!/usr/bin/env python3
"""A stand-in for an entry's development benchmark, for sweep tests.

    mock_bench.py [--rate R] [--liar] [--prof] [--crash] --bench SECONDS TARGETS CHARS

Prints what onion-walk's --bench prints: with --prof a {"prof": [...]} line,
then one JSON line with the self-timed rate and the hit counts. An honest build
finds the hits its claimed work predicts; --liar finds half of them.
"""

import argparse
import json
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rate", type=float, default=1e9)
    parser.add_argument("--liar", action="store_true")
    parser.add_argument("--prof", action="store_true")
    parser.add_argument("--crash", action="store_true")
    parser.add_argument("--bench", nargs=3, type=float, required=True,
                        metavar=("SECONDS", "TARGETS", "CHARS"))
    args = parser.parse_args()
    seconds, targets, chars = args.bench
    if args.crash:
        print("simulated GPU fault", file=sys.stderr)
        return 3
    time.sleep(min(seconds, 0.05))
    candidates = int(args.rate * seconds)
    expected = candidates * targets / 32**chars
    verified = round(expected / 2 if args.liar else expected)
    if args.prof:
        print(json.dumps({"prof": [1500, 0.9, seconds, seconds, 1, 2, 3, 4, 5, 6]}))
    print(json.dumps({"rate": args.rate, "seconds": seconds, "candidates": candidates,
                      "flagged": verified, "verified": verified, "expected": expected,
                      "degenerate": 0}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

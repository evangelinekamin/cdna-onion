#!/usr/bin/env python3
"""A misbehaving entry for rig tests: `mock_entry.py --behavior NAME <contract flags>`.

`honest` is the reference entry. Every other behavior breaks one rule (or
attacks the rig), and the rig must catch each one.
"""

import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "entries" / "python-ref"))

import entry as ref  # noqa: E402  (the reference entry, reused for honest grinding)

ORPHAN_SECONDS = "3017"  # a distinctive sleep duration, so tests can find leaked processes


def send(message: dict) -> None:
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def raw(data: bytes) -> None:
    sys.stdout.buffer.write(data)
    sys.stdout.buffer.flush()


def read_start() -> dict:
    return json.loads(sys.stdin.readline())


def spawn_orphan() -> None:
    """A child that tries to outlive the entry by escaping its session."""
    subprocess.Popen(["sleep", ORPHAN_SECONDS], start_new_session=True,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def honest_hits(start: dict, count: int, budget_s: float = 1.0) -> tuple[list[str], int]:
    """Up to `count` genuine hits found within the time budget, and the k range tested."""
    a0 = ref.E.decode(bytes.fromhex(start["a0"]))
    matcher = ref.Matcher(ref.onion.normalize_targets(start["targets"]))
    hits, deadline, tested = [], time.monotonic() + budget_s, 0
    for k, hit in ref.grind_range(a0, matcher, 0, 1 << 30):
        tested = k + 1
        if hit:
            hits.append(str(k))
        if len(hits) == count or (k % 256 == 0 and time.monotonic() > deadline):
            break
    return hits, tested


def info(name: str) -> int:
    if name == "info_orphan":
        spawn_orphan()
    if name == "loud_info":
        raw(b"x" * (2 << 20))
        return 0
    label = "\x1b]52;c;cGF3bmVk\x07evil\x1b[8m" if name == "escape_info" else "mock"
    send({"contract": "onionrig-grind/1", "name": label, "version": "1"})
    return 0


def attack(name: str) -> None:
    """Behaviors that send something hostile after start, then hang."""
    if name == "nonutf8":
        raw(b'{"event": "hits", "k": ["\xff\xfe"]}\n')
    elif name == "deepjson":
        raw(b"[" * 200_000 + b"]" * 200_000 + b"\n")
    elif name == "hugeint":
        raw(b'{"event": "hits", "k": [' + b"7" * 5000 + b"]}\n")
    elif name == "flood":
        for chunk in range(40):
            send({"event": "hits", "k": [str(chunk * 5000 + i) for i in range(5000)]})
    elif name == "stdout_hostage":  # a grandchild keeps stdout open after the entry exits
        subprocess.Popen(["sleep", ORPHAN_SECONDS])
        sys.exit(0)
    else:
        return
    time.sleep(3600)


def behave(name: str) -> int:
    if name == "silent":
        time.sleep(3600)
    send({"event": "ready"})
    if name == "garbage":
        print("this is not json", flush=True)
        time.sleep(60)
        return 0
    start = read_start()
    if name == "crash":
        return 3
    if name == "error":
        send({"event": "error", "message": "deliberate failure"})
        return 2
    if name == "overrun":
        time.sleep(3600)
    attack(name)
    if name == "orphan":
        spawn_orphan()
    hits, candidates = honest_hits(start, 40)
    if name == "sometimes_liar":  # honest at first, forges one key on its 3rd session
        state = Path(os.environ["MOCK_STATE"])
        sessions = int(state.read_text() or 0) + 1 if state.exists() else 1
        state.write_text(str(sessions))
        if sessions == 3:
            hits.append(str(random.randrange(1 << 40)))
    if name == "liar":
        # One forgery hidden among the genuine hits.
        hits.insert(len(hits) // 2, str(random.randrange(1 << 40)))
    if name == "duplicator":
        hits = hits + hits[:10]
    send({"event": "hits", "k": hits})
    time.sleep(start.get("seconds") or 0.5)
    if name == "inflater":
        candidates *= 100
    send({"event": "done", "candidates": str(candidates)})
    if name == "linger":
        time.sleep(3600)
    return 5 if name == "bad_exit" else 0


def main() -> int:
    args = sys.argv[1:]
    behavior = args[args.index("--behavior") + 1]
    if "--info" in args and behavior in ("info_orphan", "loud_info", "escape_info"):
        return info(behavior)
    if behavior == "honest" or "--grind" not in args:
        sys.argv = [sys.argv[0]] + [a for a in args if a not in ("--behavior", behavior)]
        return ref.main()
    return behave(behavior)


if __name__ == "__main__":
    sys.exit(main())

"""onionrig: conformance, benchmarking, and split-key search for vanity .onion entries.

An ENTRY is a command line, e.g. "./grind" or "python3 entries/python-ref/entry.py".
"""

import argparse
import contextlib
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

from . import keys, onion, vectors
from .rig import bench as bench_mod
from .rig import check as check_mod
from .rig import isolation, results, runner
from .rig import search as search_mod
from .rig import telemetry as tm
from .rig.protocol import sanitize


def parse_cpus(text: str) -> tuple[int, ...]:
    """'0-3,8' -> (0, 1, 2, 3, 8)"""
    cpus: list[int] = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        cpus += range(int(lo), int(hi or lo) + 1)
    return tuple(cpus)


def _cpus(text: str) -> tuple[int, ...] | None:
    """'auto' pins to the first 2 allowed CPUs, 'all' disables pinning."""
    if text == "all":
        return None
    return isolation.default_cpus() if text == "auto" else parse_cpus(text)


def _even_pairs(text: str) -> int:
    value = int(text)
    if value < 2 or value % 2:
        raise argparse.ArgumentTypeError("pairs must be even and >= 2 (whole ABBA blocks)")
    return value


def _device(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("device index must be >= 0")
    return value


def entry_spec(command: str, args: argparse.Namespace) -> runner.EntrySpec:
    argv = tuple(shlex.split(command))
    if not argv:
        raise SystemExit("empty ENTRY command")
    return runner.EntrySpec(argv, device=args.device, cpus=_cpus(args.cpus),
                            isolate=not args.no_isolation, log_dir=Path(args.out) / "logs")


def _session_file(args: argparse.Namespace, mode: str) -> results.JsonlWriter:
    return results.JsonlWriter(Path(args.out) / f"{mode}-{time.strftime('%Y%m%dT%H%M%S')}.jsonl")


def _telemetry(args: argparse.Namespace) -> tm.Telemetry | None:
    if args.no_telemetry:
        return None
    backend = tm.detect(os.environ, args.device)
    if backend is None:
        print("warning: no GPU telemetry available; runs will be flagged no-telemetry",
              file=sys.stderr)
    return backend


def _bench_config(args: argparse.Namespace) -> bench_mod.BenchConfig:
    return bench_mod.BenchConfig(
        seconds=args.seconds, calibrate_seconds=args.calibrate_seconds,
        hits_per_second=args.hits_per_second, min_runs=args.min_runs, max_runs=args.max_runs,
        target_rel_sem=args.rel_sem, strict_thermal=args.strict_thermal,
        verify_workers=args.verify_workers)


def cmd_vectors(args: argparse.Namespace) -> int:
    out = Path(args.out)
    if args.check:
        problems = vectors.check(out)
        print("\n".join(problems) or f"{out}: all vector files match a fresh regeneration")
        return 1 if problems else 0
    digests = vectors.write_all(out)
    print(f"wrote {len(digests)} files to {out} (manifest: {out / 'MANIFEST.json'})")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    spec = entry_spec(args.entry, args)

    def show(r: check_mod.CheckResult) -> None:
        status = "PASS" if r.ok else ("FAIL" if r.required else "WARN")
        print(f"  {status}  {r.name:<16} {r.detail}", flush=True)

    print(f"conformance: {args.entry}")
    checks = check_mod.run_checks(spec, Path(args.vectors), report=show)
    ok = check_mod.passed(checks)
    print("conforms to onionrig-grind/1" if ok else "DOES NOT CONFORM")
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"entry": args.entry, "conforms": ok, "checks": [asdict(c) for c in checks]},
            indent=1) + "\n")
    return 0 if ok else 1


def cmd_bench(args: argparse.Namespace) -> int:
    writer = _session_file(args, "bench")
    with bench_mod.Bench(_bench_config(args), writer, _telemetry(args)) as b:
        summary = bench_mod.bench(entry_spec(args.entry, args), b, args.gpu)
    if summary.failure:
        print(f"{sanitize(summary.entry)} FAILED: {sanitize(summary.failure)}")
        print(f"  an invalid run disqualifies the entry; details in {writer.path}")
        return 1
    if summary.summary is None:
        print(f"{sanitize(summary.entry)}: no countable runs (see {writer.path})")
        return 1
    s = summary.summary
    print(f"{sanitize(summary.entry)} on {sanitize(summary.gpu)}: {results.human_rate(s.mean)} "
          f"(95% CI {results.human_rate(s.ci_low)} - {results.human_rate(s.ci_high)}, "
          f"{summary.runs} runs)")
    print(f"  median time to a 10-char prefix: "
          f"{results.human_duration(results.expected_seconds(s.mean, 10))}")
    print(f"  median time to a 13-char prefix: "
          f"{results.human_duration(results.expected_seconds(s.mean, 13))}")
    print(f"  log: {writer.path}")
    return 0


def cmd_duel(args: argparse.Namespace) -> int:
    writer = _session_file(args, "duel")
    with bench_mod.Bench(_bench_config(args), writer, _telemetry(args)) as b:
        d = bench_mod.duel(entry_spec(args.entry_a, args), entry_spec(args.entry_b, args), b,
                           args.pairs, args.gpu)
    if d.failure:
        print(f"duel FAILED: {sanitize(d.failure)} (see {writer.path})")
        return 1
    if d.ratio is None:
        print(f"no complete ABBA blocks counted (see {writer.path})")
        return 1
    print(f"{sanitize(d.b)} vs {sanitize(d.a)}: {d.ratio.ratio:.4f}x "
          f"(95% CI {d.ratio.ci_low:.4f} - {d.ratio.ci_high:.4f}, {d.pairs} pairs) "
          f"-> {d.ratio.verdict}")
    return 0


def cmd_board(args: argparse.Namespace) -> int:
    files = [Path(f) for f in args.files] or sorted(Path(args.out).glob("*.jsonl"))
    print(results.format_leaderboard(results.leaderboard(results.read_jsonl(files))))
    return 0


def cmd_keygen(args: argparse.Namespace) -> int:
    public = keys.keygen(Path(args.dir), args.count)
    print(f"wrote {args.count} start points to {public}")
    print(f"copy ONLY that file to the GPU host; {Path(args.dir) / keys.SECRET_NAME} stays here")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    targets = onion.normalize_targets(args.prefix)
    devices = parse_cpus(args.devices)
    pool = search_mod.PointPool(keys.load_points(Path(args.points)),
                                Path(args.out) / "used-points.json")
    config = search_mod.SearchConfig(targets, devices, args.max_hits)
    writer = results.JsonlWriter(Path(args.out) / "search.jsonl")
    search = search_mod.Search(entry_spec(args.entry, args), pool, config, writer,
                               progress=lambda line: print(line, flush=True))

    def interrupt(signum: int, _frame: object) -> None:
        if search.stop.is_set() or signum == signal.SIGTERM:
            runner.kill_all()  # entries live in their own sessions; reap them
            raise KeyboardInterrupt
        print("stopping after in-flight work; press Ctrl-C again to abort", flush=True)
        search.stop.set()

    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    os.umask(0o077)  # results tie an address to this account; keep them private
    print(f"searching {', '.join(t.text for t in targets)} on GPU(s) {devices}; "
          f"{pool.remaining} start points available; hits -> {writer.path}", flush=True)
    found = search.run()
    print(f"{found} verified hit(s) recorded in {writer.path}")
    return 0


def cmd_combine(args: argparse.Namespace) -> int:
    key = keys.combine(Path(args.secrets), args.point, args.k, Path(args.out), args.address)
    print(f"wrote {key.address}.onion to {args.out} (HiddenServiceDir, mode 0700)")
    return 0


def cmd_telemetry(args: argparse.Namespace) -> int:
    backend = tm.detect(os.environ, args.device)
    if backend is None:
        print("no telemetry backend answered (tried amd-smi, nvidia-smi)")
        return 1
    if args.follow is None:
        print(f"{backend.name}: {backend.sample()}")
        return 0
    _follow(backend, results.JsonlWriter(Path(args.follow)), args.interval, args.count)
    return 0


def _follow(backend: tm.Telemetry, out: results.JsonlWriter, interval: float,
            count: int | None) -> None:
    """One JSON line per sample, stamped with epoch time to line up with other logs,
    until `count` samples or the process is stopped. A failed sample is skipped."""
    taken = 0
    while count is None or taken < count:
        started = time.monotonic()
        with contextlib.suppress(OSError, ValueError, subprocess.SubprocessError):
            sample = asdict(backend.sample())
            del sample["t"]  # the sample's monotonic clock means nothing outside this process
            out.write({"t": time.time(), **sample})
        taken += 1
        time.sleep(max(0.0, interval - (time.monotonic() - started)))


def _entry_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("--device", type=_device, default=0,
                   help="GPU index among those visible to the rig (default 0)")
    p.add_argument("--cpus", default="auto",
                   help="pin the entry: 'auto' = first 2 allowed CPUs, 'all' = no pinning, "
                        "or a list like 0-3")
    p.add_argument("--no-isolation", action="store_true",
                   help="run entries without network/PID namespaces (recorded in results)")
    p.add_argument("--out", default="bench", help="results directory (default: bench)")


def _bench_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("--seconds", type=float, default=60.0)
    p.add_argument("--calibrate-seconds", type=float, default=10.0)
    p.add_argument("--hits-per-second", type=float, default=2_000.0,
                   help="target hit rate; low keeps per-hit work out of the measurement")
    p.add_argument("--min-runs", type=int, default=3)
    p.add_argument("--max-runs", type=int, default=10)
    p.add_argument("--rel-sem", type=float, default=0.005, help="stop when SEM/mean <= this")
    p.add_argument("--gpu", help="label for the leaderboard (default: entry-reported device)")
    p.add_argument("--strict-thermal", action="store_true",
                   help="exclude runs with missing telemetry or persistent throttling")
    p.add_argument("--no-telemetry", action="store_true")
    p.add_argument("--verify-workers", type=int, default=isolation.default_verify_workers())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="onionrig", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("vectors", help="write or check the shared correctness vectors")
    p.add_argument("--out", default="vectors")
    p.add_argument("--check", action="store_true")
    p.set_defaults(func=cmd_vectors)

    p = sub.add_parser("check", help="conformance suite for one entry")
    p.add_argument("entry")
    p.add_argument("--vectors", default="vectors")
    p.add_argument("--json", metavar="PATH", help="also write the results as JSON to PATH")
    _entry_options(p)
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("bench", help="measured throughput of one entry")
    p.add_argument("entry")
    _entry_options(p)
    _bench_options(p)
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("duel", help="paired ABBA comparison: B's speed relative to A")
    p.add_argument("entry_a")
    p.add_argument("entry_b")
    p.add_argument("--pairs", type=_even_pairs, default=4,
                   help="paired runs, even and >= 2 (ABBA blocks of two pairs)")
    _entry_options(p)
    _bench_options(p)
    p.set_defaults(func=cmd_duel)

    p = sub.add_parser("board", help="leaderboard from result files")
    p.add_argument("files", nargs="*")
    p.add_argument("--out", default="bench")
    p.set_defaults(func=cmd_board)

    p = sub.add_parser("keygen", help="make split-key start points (run on a trusted machine)")
    p.add_argument("dir")
    p.add_argument("--count", type=int, default=64)
    p.set_defaults(func=cmd_keygen)

    p = sub.add_parser("search", help="real vanity search on the GPU host (no secrets needed)")
    p.add_argument("entry")
    p.add_argument("--points", required=True, help="start-points.json from keygen")
    p.add_argument("--prefix", action="append", required=True, help="repeatable")
    p.add_argument("--devices", default="0", help="GPUs to use, e.g. 0-7")
    p.add_argument("--max-hits", type=int)
    _entry_options(p)
    p.set_defaults(func=cmd_search, out="search")

    p = sub.add_parser("combine", help="turn a verified hit into Tor key files (trusted machine)")
    p.add_argument("--secrets", required=True, help="start-scalars.secret from keygen")
    p.add_argument("--point", type=int, required=True)
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--address", required=True,
                   help="the hit's address (from search.jsonl): the key must reproduce it")
    p.add_argument("--out", required=True, help="new HiddenServiceDir to create")
    p.set_defaults(func=cmd_combine)

    p = sub.add_parser("telemetry", help="probe GPU telemetry (validate on new hardware)")
    p.add_argument("--device", type=_device, default=0)
    p.add_argument("--follow", metavar="PATH",
                   help="keep sampling, appending JSON lines to PATH until stopped")
    p.add_argument("--interval", type=float, default=0.25, help="seconds between samples")
    p.add_argument("--count", type=int, help="with --follow: stop after this many samples")
    p.set_defaults(func=cmd_telemetry)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

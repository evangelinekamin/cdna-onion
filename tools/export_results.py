#!/usr/bin/env python3
"""Export scrubbed raw JSONL benchmark, sweep, microbenchmark, and telemetry logs.

Packages all numerical observations underlying the preprint tables into a single
release tarball (`dist/cdna-onion-v0.7.1-raw-results.tar.gz`) while stripping
local user paths, private IPv4 addresses, and split-key public start points /
offsets from live search logs.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import tarfile
from pathlib import Path

HOME_PATH_RE = re.compile(r"/home/[A-Za-z0-9._-]+")
SCRATCH_PATH_RE = re.compile(r"/scratch/[A-Za-z0-9._-]+")
IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
SAFE_IPS = {"127.0.0.1", "0.0.0.0"}


def scrub_text(text: str) -> str:
    text = HOME_PATH_RE.sub("/home/user", text)
    text = SCRATCH_PATH_RE.sub("/scratch/user", text)
    return IPV4_RE.sub(lambda m: m.group(0) if m.group(0) in SAFE_IPS else "<REDACTED_IP>", text)


def scrub_search_jsonl(text: str) -> str:
    """Keep timestamps, devices, and verified addresses; drop start-point / offset metadata."""
    out_lines: list[str] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        rec.pop("a0_pub", None)
        rec.pop("k", None)
        out_lines.append(json.dumps(rec, sort_keys=True))
    return "\n".join(out_lines) + "\n"


def add_bytes(tf: tarfile.TarFile, arcname: str, data: bytes) -> None:
    info = tarfile.TarInfo(name=arcname)
    info.size = len(data)
    info.mode = 0o644
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    tf.addfile(info, io.BytesIO(data))


def collect_files(results_root: Path) -> list[tuple[Path, str]]:
    items: list[tuple[Path, str]] = []

    def add_glob(sub: str, pattern: str) -> None:
        base = results_root / sub
        if not base.exists():
            return
        for p in sorted(base.glob(pattern)):
            if p.is_file():
                items.append((p, str(p.relative_to(results_root))))

    # MI210 baseline + microbench
    add_glob("mi210-65794131", "*.jsonl")
    add_glob("mi210-65800414", "*.jsonl")

    # Arena ranked benchmark JSONLs + ledger
    add_glob("arena-round1-mi210/arena", "ledger.jsonl")
    judge_dir = results_root / "arena-round1-mi210/judge"
    if judge_dir.exists():
        for p in sorted(judge_dir.rglob("bench-*.jsonl")):
            items.append((p, str(p.relative_to(results_root))))

    # MI300X & MI350X rental tuning suites
    for sub in ("rental-mi300x-20261102/tune-mi300x", "rental-mi350x-20261017/tune-mi350x"):
        base = results_root / sub
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if not p.is_file():
                continue
            if p.suffix == ".jsonl" or p.name == "prebuilt-info.txt":
                items.append((p, str(p.relative_to(results_root))))

    # 8x MI355X live search + telemetry
    mi355_base = results_root / "mi355x-8x-20261017/evangeline"
    if mi355_base.exists():
        for p in sorted(mi355_base.rglob("*")):
            if not p.is_file():
                continue
            if (
                p.suffix == ".jsonl"
                or p.name in ("prebuilt-info.txt", "search.log")
            ):
                items.append((p, str(p.relative_to(results_root))))

    return items


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Path to local results directory",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("dist/cdna-onion-v0.7.1-raw-results.tar.gz"),
        help="Output archive path",
    )
    args = parser.parse_args()

    items = collect_files(args.results_dir)
    if not items:
        raise SystemExit(f"No result files found under {args.results_dir}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    with tarfile.open(args.out, "w:gz") as tf:
        for path, rel in items:
            raw = path.read_text(encoding="utf-8", errors="replace")
            cleaned = scrub_search_jsonl(raw) if path.name == "search.jsonl" else scrub_text(raw)
            encoded = cleaned.encode("utf-8")
            add_bytes(tf, f"raw-results/{rel}", encoded)
            manifest.append({"path": rel, "bytes": len(encoded)})
        readme = (
            b"# cdna-onion v0.7.1 Raw Measurement Logs\n\n"
            b"Scrubbed JSONL benchmark, sweep, microbenchmark, binary-hash (`build.jsonl`),\n"
            b"and GPU telemetry logs underlying all tables in the preprint.\n"
            b"Local user home/scratch paths and split-key start-point/offset fields in\n"
            b"`search.jsonl` have been stripped.\n"
        )
        add_bytes(tf, "raw-results/README.md", readme)
        add_bytes(
            tf,
            "raw-results/manifest.json",
            (json.dumps(manifest, indent=2) + "\n").encode("utf-8"),
        )

    print(f"wrote {args.out} ({len(items)} files, {args.out.stat().st_size / 1024:.1f} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

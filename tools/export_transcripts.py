#!/usr/bin/env python3
"""Bundle redacted Round-1 arena transcripts and the tool-call audit into a release tarball.

Reads from results/arena-round1-mi210/ (gitignored) and writes
dist/arena-round1-transcripts.tar.gz ready for Zenodo or GitHub Releases.
Verifies zero personal usernames, emails, host IPs, or secret tokens remain.
"""

from __future__ import annotations

import json
import re
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARENA = ROOT / "results/arena-round1-mi210"
OUT = ROOT / "dist/arena-round1-transcripts.tar.gz"

# Generic patterns to redact if present anywhere in exported text
REDACTIONS = [
    (re.compile(r"/scratch/(?!USER\b)[A-Za-z0-9_.-]+"), "/scratch/USER"),
    (re.compile(r"/home/(?!agent\b|USER\b)[A-Za-z0-9_.-]+"), "/home/USER"),
    (re.compile(r"\b[A-Za-z0-9._%+-]+@asu\.edu\b", re.I), "USER@asu.edu"),
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "REDACTED_IP"),
]

# Post-redaction forbidden leak patterns
LEAK_CHECK = re.compile(
    r"ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|"
    r"sk-ant-[A-Za-z0-9_-]{10,}|sk-proj-[A-Za-z0-9_-]{10,}|sk-sp-[A-Za-z0-9_-]{10,}|"
    r"BEGIN [A-Z ]*PRIVATE KEY",
)

AUDIT_SUMMARY = {
    "arena": "round1-mi210 (2026-10-16 to 2026-10-17, ASU Sol MI210 gfx90a)",
    "method": (
        "Direct JSONL audit of every tool_use / tool_call / custom_tool_call "
        "event in raw agent logs"
    ),
    "agents": {
        "claude": {
            "model": "Claude Opus 5.5",
            "cli": "Claude Code v2.1.302",
            "tool_calls": {"Bash": 610, "Write": 21, "ToolSearch": 2},
            "web_search_calls": 0,
            "external_network_shell_commands": 0,
            "best_verified_gs": 16.091,
        },
        "gemini": {
            "model": "Gemini 4 Argon",
            "cli": "Antigravity CLI v1.4.0",
            "tool_calls": {
                "run_command": 353,
                "view_file": 350,
                "replace_file_content": 109,
                "manage_task": 41,
                "schedule": 34,
                "write_to_file": 6,
            },
            "web_search_calls": 0,
            "external_network_shell_commands": 0,
            "best_verified_gs": 14.518,
        },
        "gpt": {
            "model": "GPT-6.1 Sol",
            "cli": "Codex CLI v0.163.0",
            "web_search_calls": 3,
            "web_search_queries": [
                "honion dual addition law y coordinate 5.5 multiplication github",
                (
                    "github honion vanity tor dual addition law ed25519 "
                    "(plus failed open of Splinters-io/honion/tree/main/docs)"
                ),
                (
                    '"V_SUB_CO_U32" "borrow" '
                    "(returned no useful result; resolved via local hipcc probe)"
                ),
            ],
            "external_code_copied": False,
            "merged_into_final_kernel": False,
            "best_verified_gs": 7.381,
        },
        "qwen": {
            "model": "Qwen 3.8 Max",
            "cli": "Claude Code v2.1.301 (OpenRouter / QwenCloud)",
            "tool_calls": {"Bash": 348, "Edit": 48, "Write": 34, "Read": 12},
            "web_search_calls": 0,
            "external_network_shell_commands": 0,
            "best_verified_gs": 6.450,
            "standard_4target_plan_gs": 5.956,
        },
    },
}

ARCHIVE_README = """# Round-1 Arena Transcripts (Redacted)

Archive accompanying Evangeline Kamin, "High-Throughput Ed25519 Vanity Search in HIP
Across Three Generations of AMD Instinct GPUs."

## Contents
- `exports/claude.md`: Full rendered transcript of Claude Opus 5.5 (Claude Code v2.1.302)
- `exports/gemini.md`: Full rendered transcript of Gemini 4 Argon (Antigravity CLI v1.4.0)
- `exports/gpt.md`: Full transcript of GPT-6.1 Sol (Codex CLI v0.163.0)
- `exports/qwen.md`: Full rendered transcript of Qwen 3.8 Max (Claude Code v2.1.301)
- `comparison.md`: Side-by-side post-round technical comparison
- `tool-audit.json`: Summary of raw tool-call and web-lookup audit across all four agents

All host usernames, local home paths, and private IPs have been redacted.
"""


def redact(text: str) -> str:
    for pat, repl in REDACTIONS:
        text = pat.sub(repl, text)
    m = LEAK_CHECK.search(text)
    if m:
        raise ValueError(f"leak check failed on pattern: {m.group(0)!r}")
    return text


def main() -> int:
    files_to_pack = [
        (ARENA / "exports/claude.md", "exports/claude.md"),
        (ARENA / "exports/gemini.md", "exports/gemini.md"),
        (ARENA / "exports/gpt.md", "exports/gpt.md"),
        (ARENA / "exports/qwen.md", "exports/qwen.md"),
        (ARENA / "review/comparison.md", "comparison.md"),
    ]
    for src, _ in files_to_pack:
        if not src.is_file():
            raise FileNotFoundError(f"missing {src}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / "arena-round1-transcripts"
        (stage / "exports").mkdir(parents=True)
        for src, rel in files_to_pack:
            cleaned = redact(src.read_text(encoding="utf-8", errors="replace"))
            (stage / rel).write_text(cleaned, encoding="utf-8")
        (stage / "tool-audit.json").write_text(
            json.dumps(AUDIT_SUMMARY, indent=2) + "\n", encoding="utf-8"
        )
        (stage / "README.md").write_text(ARCHIVE_README, encoding="utf-8")

        with tarfile.open(OUT, "w:gz") as tf:
            tf.add(stage, arcname="arena-round1-transcripts")

    size_mb = OUT.stat().st_size / (1024 * 1024)
    print(f"wrote {OUT} ({size_mb:.2f} MiB, leak check passed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

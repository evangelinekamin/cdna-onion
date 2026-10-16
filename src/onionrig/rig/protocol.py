"""The onionrig-grind/1 wire protocol: newline-delimited JSON between rig and entry.

Parsing is strict. An entry that sends something malformed has a bug, and the
rig reports it instead of guessing what was meant.
"""

import json
import re
from dataclasses import dataclass

CONTRACT = "onionrig-grind/1"
MAX_ABS_OFFSET = 2**63  # |k| must fit a signed 64-bit integer
MAX_TARGETS = 256

# ASCII only: str.isdigit() would also accept digits like "١٢٣" that int() parses.
_SIGNED_DECIMAL = re.compile(r"-?[0-9]+")
_UNSIGNED_DECIMAL = re.compile(r"[0-9]+")


class ProtocolError(ValueError):
    """The entry broke the contract."""


@dataclass(frozen=True)
class Ready:
    pass


@dataclass(frozen=True)
class Hits:
    offsets: tuple[int, ...]


@dataclass(frozen=True)
class Done:
    candidates: int


@dataclass(frozen=True)
class EntryError:
    message: str


@dataclass(frozen=True)
class Progress:
    """Optional, display only: candidates tested so far. Never used for scoring."""

    candidates: int


Event = Ready | Hits | Done | EntryError | Progress


def start_command(a0: bytes, targets: tuple[str, ...], seconds: float | None) -> str:
    if len(a0) != 32:
        raise ValueError("a0 must be a 32-byte point encoding")
    if not 1 <= len(targets) <= MAX_TARGETS:
        raise ValueError(f"need 1..{MAX_TARGETS} targets, got {len(targets)}")
    if seconds is not None and seconds <= 0:
        raise ValueError("seconds must be positive or None")
    return json.dumps({"cmd": "start", "a0": a0.hex(), "targets": list(targets),
                       "seconds": seconds}) + "\n"


STOP_COMMAND = json.dumps({"cmd": "stop"}) + "\n"


def sanitize(text: object, limit: int = 200) -> str:
    """Entry-controlled text made safe to print: control characters, escape
    sequences, and lone surrogates become '?', and the length is bounded."""
    return "".join(c if c.isprintable() else "?" for c in str(text)[:limit])


def _json(text: str) -> object:
    try:
        return json.loads(text)
    except (ValueError, RecursionError) as exc:  # JSONDecodeError is a ValueError
        raise ProtocolError(f"not JSON: {sanitize(text, 120)!r}") from exc


def _is_decimal(text: object, pattern: re.Pattern[str], max_len: int) -> bool:
    return isinstance(text, str) and len(text) <= max_len and pattern.fullmatch(text) is not None


def parse_offset(text: object) -> int:
    if not _is_decimal(text, _SIGNED_DECIMAL, 21):
        raise ProtocolError(f"offset must be a signed decimal string, got {text!r:.40}")
    k = int(text)  # type: ignore[arg-type]  # checked by _is_decimal
    if abs(k) >= MAX_ABS_OFFSET:
        raise ProtocolError(f"offset {k} does not fit in a signed 64-bit integer")
    return k


def parse_event(line: str) -> Event:
    message = _json(line)
    if not isinstance(message, dict):
        raise ProtocolError(f"expected a JSON object, got {sanitize(line, 120)!r}")
    kind = message.get("event")
    if kind == "ready":
        return Ready()
    if kind == "hits":
        offsets = message.get("k")
        if not isinstance(offsets, list):
            raise ProtocolError("hits event needs a list 'k'")
        return Hits(tuple(parse_offset(k) for k in offsets))
    if kind == "done":
        candidates = message.get("candidates")
        if not _is_decimal(candidates, _UNSIGNED_DECIMAL, 24):
            raise ProtocolError(f"done needs decimal-string 'candidates', got {candidates!r:.40}")
        return Done(int(candidates))
    if kind == "progress":
        candidates = message.get("candidates")
        if not _is_decimal(candidates, _UNSIGNED_DECIMAL, 24):
            raise ProtocolError(f"progress needs decimal 'candidates', got {candidates!r:.40}")
        return Progress(int(candidates))
    if kind == "error":
        return EntryError(sanitize(message.get("message", "")))
    raise ProtocolError(f"unknown event {sanitize(kind, 40)!r}")


_INFO_FIELDS = {"name": True, "version": True, "model": False, "device": False}
_INFO_MAX_LEN = 96


@dataclass(frozen=True)
class Info:
    name: str
    version: str
    model: str
    device: str


def _info_field(raw: dict, field: str, required: bool) -> str:
    value = raw.get(field, "")
    if not isinstance(value, str):
        raise ProtocolError(f"--info field {field!r} must be a string")
    if required and not value:
        raise ProtocolError(f"--info is missing {field!r}")
    if len(value) > _INFO_MAX_LEN or not all(" " <= c <= "~" for c in value):
        raise ProtocolError(f"--info field {field!r} must be printable ASCII, "
                            f"at most {_INFO_MAX_LEN} characters")
    return value


def parse_info(text: str) -> Info:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) != 1:
        raise ProtocolError(f"--info must print exactly one line, got {len(lines)}")
    raw = _json(lines[0])
    if not isinstance(raw, dict):
        raise ProtocolError("--info must print a JSON object")
    if raw.get("contract") != CONTRACT:
        raise ProtocolError(f"entry speaks {sanitize(raw.get('contract'), 40)!r}, "
                            f"rig expects {CONTRACT!r}")
    fields = {f: _info_field(raw, f, required) for f, required in _INFO_FIELDS.items()}
    return Info(**fields)

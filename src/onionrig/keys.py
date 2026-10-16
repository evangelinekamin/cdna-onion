"""Local key material for split-key searches: scalars stay here, only points travel.

keygen writes two files into a directory:
  start-scalars.secret  mode 0600, one JSON object per line {"id", "a0"}. Never copy it anywhere.
  start-points.json     the public start points; copy this to the GPU host.
combine turns a verified hit (point id, offset k) into a Tor HiddenServiceDir.
"""

import json
import os
import re
import resource
from pathlib import Path

from . import onion, splitkey, torkeys
from .ed25519 import point as E

# The source checkout (None when installed as a package): it gets copied to GPU hosts.
_CHECKOUT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = _CHECKOUT if (_CHECKOUT / "pyproject.toml").exists() else None

SECRET_NAME = "start-scalars.secret"
PUBLIC_NAME = "start-points.json"


def _no_core_dumps() -> None:
    """Secrets live in this process's memory; never let a crash write them to disk."""
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def _refuse_project_tree(directory: Path) -> None:
    """The project tree gets copied to the GPU host; start scalars must not be in it."""
    target = directory.resolve()
    if PROJECT_ROOT is not None and (target == PROJECT_ROOT or PROJECT_ROOT in target.parents):
        raise ValueError(f"{directory} is inside the onionrig project, which gets copied to "
                         "the GPU host. Keep start scalars elsewhere, e.g. ~/onion-keys")


def _write_new(path: Path, text: str) -> None:
    """Create a new public file (never following or replacing an existing one)."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def keygen(directory: Path, count: int) -> Path:
    """Draw `count` start scalars; returns the public points file to ship."""
    if count < 1:
        raise ValueError("count must be at least 1")
    _no_core_dumps()
    _refuse_project_tree(directory)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    secret_path, public_path = directory / SECRET_NAME, directory / PUBLIC_NAME
    for path in (secret_path, public_path):
        if path.exists():
            raise FileExistsError(f"{path} exists; refusing to replace key material")
    scalars = [splitkey.random_start_scalar() for _ in range(count)]
    secret = "".join(json.dumps({"id": i, "a0": f"{a0:064x}"}) + "\n"
                     for i, a0 in enumerate(scalars))
    torkeys.write_private(secret_path, secret.encode())
    points = [{"id": i, "a0_pub": splitkey.start_point(a0).hex()} for i, a0 in enumerate(scalars)]
    _write_new(public_path, json.dumps({"points": points}, indent=1) + "\n")
    return public_path


def load_points(public_path: Path) -> list[tuple[int, bytes]]:
    document = json.loads(public_path.read_text())
    points = [(int(p["id"]), bytes.fromhex(p["a0_pub"])) for p in document["points"]]
    for point_id, a0 in points:
        try:
            E.decode(a0)
        except ValueError as exc:
            raise ValueError(f"start point {point_id} is not a curve point") from exc
    return points


_SCALAR_HEX = re.compile(r"[0-9a-f]{64}")


def _parse_record(line: str) -> tuple[int, str] | None:
    try:
        record = json.loads(line)
    except (ValueError, RecursionError):
        return None
    if not isinstance(record, dict):
        return None
    point_id, scalar = record.get("id"), record.get("a0")
    if type(point_id) is not int or not isinstance(scalar, str) \
            or not _SCALAR_HEX.fullmatch(scalar):
        return None
    return point_id, scalar


def _load_scalar(secret_path: Path, point_id: int) -> int:
    for line in secret_path.read_text().splitlines():
        parsed = _parse_record(line)
        # Raised outside any except block, so no exception that quotes the secret
        # (a parse error's message or its .doc) is ever attached as context.
        if parsed is None:
            raise ValueError(f"{secret_path} has a malformed record")
        if parsed[0] == point_id:
            return int(parsed[1], 16)
    raise KeyError(f"no start scalar with id {point_id} in {secret_path}")


def combine(secret_path: Path, point_id: int, k: int, out_dir: Path,
            expected_address: str | None = None) -> torkeys.OnionKey:
    """Build the key for hit (point_id, k), check it against the public walk, and write it."""
    _no_core_dumps()
    a0 = _load_scalar(secret_path, point_id)
    key = torkeys.OnionKey.from_scalar(splitkey.combine(a0, k))
    walked = E.encode(splitkey.point_at(E.decode(splitkey.start_point(a0)), k))
    if key.pubkey != walked:
        raise RuntimeError("secret arithmetic disagrees with the public walk; refusing to write")
    if expected_address is not None:
        wanted = expected_address.lower().removesuffix(".onion")
        if key.address != wanted:
            raise ValueError(f"hit is {wanted}, but point {point_id} offset {k} gives "
                             f"{key.address}: wrong point id or offset?")
    onion.pubkey_from_address(key.address)  # final self-consistency check
    torkeys.write_service_dir(out_dir, key)
    return key

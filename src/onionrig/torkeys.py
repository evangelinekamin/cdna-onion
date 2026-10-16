"""Tor v3 onion service keys, in the hs_ed25519_* files Tor reads from HiddenServiceDir.

Tor stores the *expanded* ed25519 secret: a 32-byte clamped scalar followed by a
32-byte nonce prefix for deterministic signing. A vanity key has no seed, so the
nonce prefix is fresh randomness.
"""

import hashlib
import os
import secrets
import stat
from dataclasses import dataclass
from pathlib import Path

from . import onion, splitkey
from .ed25519 import point as E

PUBLIC_HEADER = b"== ed25519v1-public: type0 ==\x00\x00\x00"
SECRET_HEADER = b"== ed25519v1-secret: type0 ==\x00\x00\x00"


@dataclass(frozen=True, repr=False)
class OnionKey:
    """A usable onion service key. repr/str show only the public address."""

    scalar: int
    nonce: bytes
    pubkey: bytes
    address: str

    @classmethod
    def from_scalar(cls, scalar: int, nonce: bytes | None = None) -> "OnionKey":
        if not splitkey.is_clamped(scalar):
            raise ValueError("scalar is not clamped; Tor would re-clamp it into a different key")
        nonce = secrets.token_bytes(32) if nonce is None else nonce
        if len(nonce) != 32:
            raise ValueError(f"nonce prefix must be 32 bytes, got {len(nonce)}")
        pubkey = splitkey.public_point(scalar)  # constant-time: the scalar is secret
        return cls(scalar, nonce, pubkey, onion.address_from_pubkey(pubkey))

    @classmethod
    def from_secret_key_file(cls, data: bytes) -> "OnionKey":
        if len(data) != 96 or data[:32] != SECRET_HEADER:
            raise ValueError("not a Tor hs_ed25519_secret_key file")
        return cls.from_scalar(int.from_bytes(data[32:64], "little"), data[64:96])

    def __repr__(self) -> str:
        return f"OnionKey({self.address}.onion)"

    __str__ = __repr__


def secret_key_file(key: OnionKey) -> bytes:
    return SECRET_HEADER + key.scalar.to_bytes(32, "little") + key.nonce


def public_key_file(key: OnionKey) -> bytes:
    return PUBLIC_HEADER + key.pubkey


def hostname_file(key: OnionKey) -> bytes:
    return f"{key.address}.onion\n".encode("ascii")


def sign(key: OnionKey, message: bytes) -> bytes:
    """RFC 8032 Ed25519 signature from the expanded key.

    This exists to prove a generated key is usable (Tor does the real signing).
    """
    r = int.from_bytes(hashlib.sha512(key.nonce + message).digest(), "little") % E.L
    big_r = splitkey.public_point(r)
    h = int.from_bytes(hashlib.sha512(big_r + key.pubkey + message).digest(), "little") % E.L
    return big_r + ((r + h * key.scalar) % E.L).to_bytes(32, "little")


def write_service_dir(path: Path | str, key: OnionKey, *, include_hostname: bool = True) -> None:
    """Create a Tor HiddenServiceDir (mode 0700, files 0600) holding the key.

    The directory must be new or empty: an existing key is never overwritten.
    Files are fsynced, and a partially written set is removed on failure.
    """
    path = Path(path)
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise FileExistsError(f"{path} is not empty; refusing to overwrite an existing key")
    path.chmod(0o700)
    files = {
        "hs_ed25519_secret_key": secret_key_file(key),
        "hs_ed25519_public_key": public_key_file(key),
    }
    if include_hostname:
        files["hostname"] = hostname_file(key)
    written: list[Path] = []
    try:
        for name, data in files.items():
            write_private(path / name, data)
            written.append(path / name)
        _fsync(path)
    except BaseException:
        for done in written:
            done.unlink(missing_ok=True)
        raise


def write_private(path: Path, data: bytes) -> None:
    """Create a new file readable only by its owner; refuses to overwrite, fsyncs.

    Refuses filesystems that ignore chmod (DrvFs, FAT, some network mounts), and
    removes the partial file if anything goes wrong mid-write.
    """
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.fchmod(fd, 0o600)  # the umask must not loosen it
        info = os.fstat(fd)
        if stat.S_IMODE(info.st_mode) != 0o600 or info.st_uid != os.getuid():
            raise PermissionError(f"{path}: this filesystem does not enforce owner-only "
                                  "access; refusing to write a secret here")
        view = memoryview(data)
        while view:
            view = view[os.write(fd, view):]
        os.fsync(fd)
    except BaseException:
        os.close(fd)
        path.unlink(missing_ok=True)  # O_EXCL proves we created it
        raise
    os.close(fd)


def _fsync(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)

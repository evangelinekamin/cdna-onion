#!/usr/bin/env python3
"""Offline Ed25519 proof-of-possession signer for Tor v3 HiddenServiceDir keys.

Run this locally on the trusted machine that holds your hs_ed25519_secret_key
files. It writes ONLY public fields (onion address, public key hex, signed
message, and RFC 8032 Ed25519 signature hex) to the output JSON file.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import resource
import sys
from pathlib import Path

from nacl.signing import VerifyKey

from onionrig import torkeys

DEFAULT_MESSAGE = (
    "Proof of possession for Evangeline Kamin's preprint "
    "'High-Throughput Ed25519 Vanity Search in HIP Across Three Generations "
    "of AMD Instinct GPUs' (8x MI355X run, 2026-10-17)."
)


def disable_core_dumps() -> None:
    with contextlib.suppress(OSError, ValueError):
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def sign_service_dir(hs_dir: Path, message: str) -> dict[str, str]:
    secret_path = hs_dir / "hs_ed25519_secret_key"
    if not secret_path.is_file():
        raise FileNotFoundError(f"missing {secret_path}")

    key = torkeys.OnionKey.from_secret_key_file(secret_path.read_bytes())

    pub_path = hs_dir / "hs_ed25519_public_key"
    if pub_path.is_file():
        expected_pub = torkeys.public_key_file(key)
        if pub_path.read_bytes() != expected_pub:
            raise ValueError(f"{pub_path} does not match derived public key")

    host_path = hs_dir / "hostname"
    if host_path.is_file():
        recorded = host_path.read_text(encoding="ascii").strip().lower()
        if recorded != f"{key.address}.onion":
            raise ValueError(f"{host_path} ({recorded}) does not match derived address")

    msg_bytes = message.encode("utf-8")
    sig = torkeys.sign(key, msg_bytes)

    # Cross-check with libsodium (PyNaCl) before emitting anything.
    VerifyKey(key.pubkey).verify(msg_bytes, sig)

    return {
        "onion": f"{key.address}.onion",
        "pubkey_hex": key.pubkey.hex(),
        "message": message,
        "signature_hex": sig.hex(),
    }


def main(argv: list[str] | None = None) -> int:
    disable_core_dumps()
    parser = argparse.ArgumentParser(
        description=(
            "Sign a proof-of-possession statement offline using Tor v3 HiddenServiceDir keys."
        )
    )
    parser.add_argument(
        "dirs",
        nargs="+",
        type=Path,
        help="One or more Tor HiddenServiceDir directories containing hs_ed25519_secret_key",
    )
    parser.add_argument(
        "--message",
        default=DEFAULT_MESSAGE,
        help="Exact UTF-8 statement to sign with each key",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output path for the public proof JSON (e.g. proofs/evangeline-proofs.json)",
    )
    args = parser.parse_args(argv)

    proofs = [sign_service_dir(d, args.message) for d in args.dirs]
    proofs.sort(key=lambda item: item["onion"])

    payload = {
        "format": "cdna-onion-pop/1",
        "algorithm": "Ed25519 (RFC 8032) + Tor v3 .onion checksum (SHA3-256)",
        "proofs": proofs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    for p in proofs:
        print(f"signed  {p['onion']}")
    print(f"wrote   {args.out} ({len(proofs)} public proof(s); zero secret material written)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

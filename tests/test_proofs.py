"""End-to-end tests for offline Ed25519 proof-of-possession signing and verification."""

import json
import subprocess
import sys
from pathlib import Path

from onionrig import splitkey, torkeys

ROOT = Path(__file__).resolve().parents[1]
SIGN_SCRIPT = ROOT / "tools/sign_proof.py"
VERIFY_SCRIPT = ROOT / "proofs/verify.py"


def _make_throwaway_hs_dir(base: Path, name: str) -> tuple[Path, torkeys.OnionKey]:
    scalar = splitkey.random_start_scalar()
    key = torkeys.OnionKey.from_scalar(scalar)
    hs_dir = base / name
    torkeys.write_service_dir(hs_dir, key)
    return hs_dir, key


def test_sign_and_verify_roundtrip(tmp_path: Path) -> None:
    d1, k1 = _make_throwaway_hs_dir(tmp_path, "hs1")
    d2, k2 = _make_throwaway_hs_dir(tmp_path, "hs2")
    out_json = tmp_path / "proofs.json"

    res = subprocess.run(
        [sys.executable, str(SIGN_SCRIPT), "--out", str(out_json), str(d1), str(d2)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert f"{k1.address}.onion" in res.stdout
    assert f"{k2.address}.onion" in res.stdout

    # Ensure zero secret bytes leaked into the JSON output.
    raw_text = out_json.read_text(encoding="utf-8")
    assert k1.scalar.to_bytes(32, "little").hex() not in raw_text
    assert k1.nonce.hex() not in raw_text

    # Verify with the standalone verifier (under current env AND bare stdlib Python).
    for py in (sys.executable, "/usr/bin/python3"):
        if not Path(py).exists():
            continue
        vres = subprocess.run(
            [py, str(VERIFY_SCRIPT), str(out_json)],
            capture_output=True,
            text=True,
        )
        assert vres.returncode == 0, vres.stderr
        assert "All 2 proof(s) verified cleanly." in vres.stdout

    # Tampering 1 bit of the signature must fail verification.
    data = json.loads(raw_text)
    sig = bytearray.fromhex(data["proofs"][0]["signature_hex"])
    sig[0] ^= 0x01
    data["proofs"][0]["signature_hex"] = sig.hex()
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(data), encoding="utf-8")
    tres = subprocess.run(
        [sys.executable, str(VERIFY_SCRIPT), str(tampered)],
        capture_output=True,
        text=True,
    )
    assert tres.returncode != 0
    assert "FAIL" in tres.stderr

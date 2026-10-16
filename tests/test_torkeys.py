"""Tor hidden-service key files, signing, and a round trip through real Tor."""

import os
import shutil
import stat
import subprocess
import time

import pytest
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from onionrig import onion, splitkey, torkeys


def make_key() -> torkeys.OnionKey:
    return torkeys.OnionKey.from_scalar(splitkey.random_start_scalar())


def test_key_file_layouts():
    key = make_key()
    secret = torkeys.secret_key_file(key)
    public = torkeys.public_key_file(key)
    assert len(secret) == 96 and secret[:32] == torkeys.SECRET_HEADER
    assert len(public) == 64 and public[:32] == torkeys.PUBLIC_HEADER
    assert secret[32:64] == key.scalar.to_bytes(32, "little")
    assert public[32:] == key.pubkey
    assert torkeys.hostname_file(key) == f"{key.address}.onion\n".encode()


def test_repr_never_shows_the_secret():
    key = make_key()
    text = repr(key) + str(key)
    assert key.address in text
    assert str(key.scalar) not in text and key.nonce.hex() not in text


def test_secret_file_round_trip():
    key = make_key()
    loaded = torkeys.OnionKey.from_secret_key_file(torkeys.secret_key_file(key))
    assert loaded.pubkey == key.pubkey and loaded.nonce == key.nonce


def test_from_secret_key_file_rejects_garbage():
    with pytest.raises(ValueError):
        torkeys.OnionKey.from_secret_key_file(b"\x00" * 96)
    with pytest.raises(ValueError):
        torkeys.OnionKey.from_secret_key_file(torkeys.SECRET_HEADER + b"\x00" * 63)


def test_unclamped_scalar_is_rejected():
    with pytest.raises(ValueError, match="clamp"):
        torkeys.OnionKey.from_scalar(2**254 + 4)


def test_signatures_verify_with_an_independent_library():
    for _ in range(5):
        key = make_key()
        message = os.urandom(64)
        signature = torkeys.sign(key, message)
        Ed25519PublicKey.from_public_bytes(key.pubkey).verify(signature, message)
        with pytest.raises(InvalidSignature):
            Ed25519PublicKey.from_public_bytes(key.pubkey).verify(signature, message + b"x")


def test_write_service_dir_permissions_and_contents(tmp_path):
    key = make_key()
    service = tmp_path / "hs"
    torkeys.write_service_dir(service, key)
    assert stat.S_IMODE(service.stat().st_mode) == 0o700
    for name in ("hs_ed25519_secret_key", "hs_ed25519_public_key", "hostname"):
        assert stat.S_IMODE((service / name).stat().st_mode) == 0o600
    assert (service / "hs_ed25519_secret_key").read_bytes() == torkeys.secret_key_file(key)
    assert (service / "hostname").read_text().strip() == key.address + ".onion"


def test_write_service_dir_never_overwrites(tmp_path):
    service = tmp_path / "hs"
    torkeys.write_service_dir(service, make_key())
    with pytest.raises(FileExistsError):
        torkeys.write_service_dir(service, make_key())


@pytest.mark.integration
@pytest.mark.skipif(shutil.which("tor") is None, reason="tor not installed")
def test_real_tor_accepts_the_key(tmp_path):
    """Tor loads our secret key and derives exactly our address (no network needed)."""
    key = torkeys.OnionKey.from_scalar(splitkey.combine(splitkey.random_start_scalar(), 12345))
    service = tmp_path / "hs"
    torkeys.write_service_dir(service, key, include_hostname=False)
    data = tmp_path / "data"
    data.mkdir(mode=0o700)
    torrc = tmp_path / "torrc"
    torrc.write_text(
        f"DataDirectory {data}\nDisableNetwork 1\nSocksPort 0\nLog notice stdout\n"
        f"HiddenServiceDir {service}\nHiddenServicePort 80 127.0.0.1:9\n")
    proc = subprocess.Popen(["tor", "-f", str(torrc)], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    hostname = service / "hostname"
    try:
        deadline = time.monotonic() + 30
        while not hostname.exists() and time.monotonic() < deadline and proc.poll() is None:
            time.sleep(0.1)
    finally:
        proc.terminate()
        output = proc.communicate(timeout=10)[0]
    assert hostname.exists(), f"tor wrote no hostname:\n{output}"
    assert hostname.read_text().strip() == key.address + ".onion"
    # Tor must have loaded our key, not generated its own.
    assert (service / "hs_ed25519_secret_key").read_bytes() == torkeys.secret_key_file(key)
    assert onion.pubkey_from_address(key.address) == key.pubkey

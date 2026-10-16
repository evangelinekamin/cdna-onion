"""How an entry is confined: one GPU, pinned CPUs, a scrubbed environment, no
network, and a PID namespace that dies with it.

This is what the rig can impose without root. With isolation on, an entry runs
inside unprivileged user, network, PID, and mount namespaces: it cannot open
network connections, and every process it starts (setsid escapees included) is
killed when it exits. It still sees the host filesystem as the invoking user, so
entries must never run on a machine or account that holds start scalars or Tor
keys (see the README's security notes).
"""

import functools
import os
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path

ISOLATE = ("unshare", "--user", "--map-root-user", "--net", "--pid", "--fork",
           "--kill-child", "--mount-proc")

# Variables an entry may inherit; everything else (credentials, agent sockets,
# tokens) is dropped. HOME and TMPDIR are replaced by a private directory.
_ENV_EXACT = frozenset({"PATH", "LANG", "LANGUAGE", "TZ", "LD_LIBRARY_PATH"})
_ENV_PREFIXES = ("LC_", "HSA_", "HIP_", "ROCR_", "ROCM_", "AMD_", "GPU_", "CUDA_", "NVIDIA_")


class IsolationUnavailable(RuntimeError):
    """Namespaces cannot be created here; the operator must opt out explicitly."""


@functools.cache
def isolation_prefix() -> tuple[str, ...] | None:
    """The namespace command that works on this host, or None if none does.

    --kill-child ends an entry if the rig dies. util-linux 2.39 implements it with
    pidfd_open, which older kernels (RHEL 8) lack, so it is dropped where it fails;
    the PID namespace and the rig's process-group kill still end every process.
    """
    if shutil.which("unshare") is None:
        return None
    for prefix in (ISOLATE, tuple(a for a in ISOLATE if a != "--kill-child")):
        try:
            probe = subprocess.run([*prefix, "true"], capture_output=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            return None
        if probe.returncode == 0:
            return prefix
    return None


def isolation_available() -> bool:
    """Whether unprivileged user/net/pid/mount namespaces work on this host."""
    return isolation_prefix() is not None


def gpu_vendor() -> str:
    """'amd' or 'nvidia'; ONIONRIG_GPU_VENDOR overrides detection."""
    override = os.environ.get("ONIONRIG_GPU_VENDOR", "").lower()
    if override in ("amd", "nvidia"):
        return override
    return "amd" if Path("/dev/kfd").exists() or shutil.which("rocminfo") else "nvidia"


def _listing(env: Mapping[str, str], var: str) -> list[str] | None:
    """The devices a visibility variable lists: None if unset, [] if set but empty."""
    if var not in env:
        return None
    return [x.strip() for x in env[var].split(",") if x.strip()]


def _pick(listing: list[str] | None, index: int, var: str) -> str:
    if listing is None:
        return str(index)
    if not listing:
        raise ValueError(f"no GPUs visible: {var} is set but empty")
    if index >= len(listing):
        raise ValueError(f"device {index} is not visible: {var}={','.join(listing)}")
    return listing[index]


def select_gpu(env: Mapping[str, str], device: int, vendor: str) -> tuple[dict[str, str], str]:
    """Narrow visibility to the `device`-th GPU the rig itself can see.

    Returns the child's environment and the GPU's physical id (index or UUID),
    which is what nvidia-smi / amd-smi need, since they ignore *_VISIBLE_DEVICES.

    On AMD, ROCR_VISIBLE_DEVICES filters at the runtime level and HIP (with its
    CUDA_VISIBLE_DEVICES alias) then indexes into what is left. So the chosen
    physical id goes to ROCR alone and HIP is pointed at the single survivor;
    setting both to the same index would leave no GPU at all for device >= 1.
    """
    if device < 0:
        raise ValueError("device index must be >= 0")
    out = dict(env)
    if vendor == "nvidia":
        physical = _pick(_listing(env, "CUDA_VISIBLE_DEVICES"), device, "CUDA_VISIBLE_DEVICES")
        out["CUDA_VISIBLE_DEVICES"] = physical
        return out, physical
    hip_var = next((v for v in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES") if v in env), None)
    chosen = _pick(_listing(env, hip_var), device, hip_var) if hip_var else str(device)
    if chosen.isdigit():  # an index into ROCR's list (or a physical index if ROCR is unset)
        chosen = _pick(_listing(env, "ROCR_VISIBLE_DEVICES"), int(chosen), "ROCR_VISIBLE_DEVICES")
    out["ROCR_VISIBLE_DEVICES"] = chosen
    out["HIP_VISIBLE_DEVICES"] = "0"
    for alias in ("CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL"):
        out.pop(alias, None)
    return out, chosen


def scrubbed_environment(env: Mapping[str, str], home: Path) -> dict[str, str]:
    """Only runtime-relevant variables survive; HOME and TMPDIR become `home`."""
    kept = {k: v for k, v in env.items() if k in _ENV_EXACT or k.startswith(_ENV_PREFIXES)}
    kept["HOME"] = kept["TMPDIR"] = str(home)
    return kept


ENTRY_CPUS = 2  # CPUs an entry is pinned to by default


def default_cpus(count: int = ENTRY_CPUS) -> tuple[int, ...]:
    """The first `count` CPUs this process may use (Slurm cpusets included)."""
    return tuple(sorted(os.sched_getaffinity(0))[:count])


def default_verify_workers(cap: int = 8) -> int:
    """Hit-verification processes: the allowed CPUs the entry is not pinned to.

    Uses the affinity mask, not os.cpu_count(), which on a cluster node counts
    every core rather than the job's allocation.
    """
    return max(1, min(cap, len(os.sched_getaffinity(0)) - ENTRY_CPUS))


def wrap_command(argv: list[str], cpus: tuple[int, ...] | None, isolate: bool) -> list[str]:
    """Prefix argv with CPU pinning and namespace isolation as requested."""
    if cpus is not None:
        taskset = shutil.which("taskset")
        if taskset is None:
            raise RuntimeError("CPU pinning requested but taskset is not installed")
        argv = [taskset, "--cpu-list", ",".join(map(str, cpus)), *argv]
    if isolate:
        prefix = isolation_prefix()
        if prefix is None:
            raise IsolationUnavailable(
                "cannot create user/net/pid namespaces on this host (unprivileged user "
                "namespaces disabled?). Rerun with --no-isolation to run entries without "
                "network/process confinement; results record that choice.")
        argv = [*prefix, *argv]
    return argv

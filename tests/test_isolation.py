"""Entry confinement: GPU visibility per vendor, environment scrubbing, namespaces."""

import subprocess
import sys

import pytest

from onionrig.rig import isolation


def nvidia(env, device):
    return isolation.select_gpu(env, device, "nvidia")


def amd(env, device):
    return isolation.select_gpu(env, device, "amd")


def test_nvidia_visibility():
    assert nvidia({}, 1)[0]["CUDA_VISIBLE_DEVICES"] == "1"
    env, physical = nvidia({"CUDA_VISIBLE_DEVICES": "4,6"}, 1)
    assert env["CUDA_VISIBLE_DEVICES"] == physical == "6"
    assert nvidia({"CUDA_VISIBLE_DEVICES": "GPU-abc"}, 0)[1] == "GPU-abc"


def test_amd_gives_rocr_the_physical_gpu_and_hip_the_survivor():
    # ROCR filters first and HIP indexes what is left: setting both to "1" would
    # leave no GPU at all, which is the bug this guards against.
    env, physical = amd({}, 1)
    assert (env["ROCR_VISIBLE_DEVICES"], env["HIP_VISIBLE_DEVICES"]) == ("1", "0")
    assert physical == "1" and "CUDA_VISIBLE_DEVICES" not in env


@pytest.mark.parametrize("given,device,rocr", [
    ({"ROCR_VISIBLE_DEVICES": "2,3"}, 1, "3"),                              # Slurm sets ROCR
    ({"HIP_VISIBLE_DEVICES": "4,5"}, 0, "4"),                               # site sets HIP only
    ({"ROCR_VISIBLE_DEVICES": "6,7", "HIP_VISIBLE_DEVICES": "1"}, 0, "7"),  # HIP indexes ROCR
    ({"CUDA_VISIBLE_DEVICES": "2,3"}, 1, "3"),                              # the CUDA alias
    ({"HIP_VISIBLE_DEVICES": "GPU-9f"}, 0, "GPU-9f"),                       # UUIDs pass through
])
def test_amd_honors_scheduler_visibility(given, device, rocr):
    env, physical = amd({**given, "GPU_DEVICE_ORDINAL": "5"}, device)
    assert env["ROCR_VISIBLE_DEVICES"] == physical == rocr
    assert env["HIP_VISIBLE_DEVICES"] == "0"
    assert "CUDA_VISIBLE_DEVICES" not in env and "GPU_DEVICE_ORDINAL" not in env


@pytest.mark.parametrize("pick,env,device", [
    (nvidia, {"CUDA_VISIBLE_DEVICES": ""}, 0),       # set but empty: no GPUs, not "all"
    (amd, {"ROCR_VISIBLE_DEVICES": ""}, 0),
    (nvidia, {"CUDA_VISIBLE_DEVICES": "0,1"}, 2),
    (amd, {"ROCR_VISIBLE_DEVICES": "3"}, 1),
    (nvidia, {}, -1),
])
def test_invisible_devices_are_refused(pick, env, device):
    with pytest.raises(ValueError):
        pick(env, device)


def test_vendor_override(monkeypatch):
    monkeypatch.setenv("ONIONRIG_GPU_VENDOR", "amd")
    assert isolation.gpu_vendor() == "amd"
    monkeypatch.setenv("ONIONRIG_GPU_VENDOR", "nvidia")
    assert isolation.gpu_vendor() == "nvidia"


def test_environment_keeps_runtime_variables_only(tmp_path):
    env = {"PATH": "/usr/bin", "LC_ALL": "C", "HIP_FOO": "1", "CUDA_CACHE_DISABLE": "1",
           "HOME": "/home/eve", "SSH_AUTH_SOCK": "/tmp/agent", "AWS_SECRET_ACCESS_KEY": "x",
           "HF_TOKEN": "hf_x", "GITHUB_TOKEN": "y"}
    kept = isolation.scrubbed_environment(env, tmp_path)
    assert {"PATH", "LC_ALL", "HIP_FOO", "CUDA_CACHE_DISABLE"} <= kept.keys()
    assert not {"SSH_AUTH_SOCK", "AWS_SECRET_ACCESS_KEY", "HF_TOKEN", "GITHUB_TOKEN"} & kept.keys()
    assert kept["HOME"] == kept["TMPDIR"] == str(tmp_path)


def test_wrap_command(monkeypatch):
    monkeypatch.setattr(isolation, "isolation_available", lambda: True)
    wrapped = isolation.wrap_command(["entry"], (0, 2), True)
    assert wrapped[:len(isolation.ISOLATE)] == list(isolation.ISOLATE)
    assert wrapped[len(isolation.ISOLATE) + 1:] == ["--cpu-list", "0,2", "entry"]
    assert isolation.wrap_command(["entry"], None, False) == ["entry"]


def test_isolation_fails_closed(monkeypatch):
    monkeypatch.setattr(isolation, "isolation_available", lambda: False)
    with pytest.raises(isolation.IsolationUnavailable, match="--no-isolation"):
        isolation.wrap_command(["entry"], None, True)


def test_default_cpus_come_from_the_allowed_set():
    cpus = isolation.default_cpus()
    assert 1 <= len(cpus) <= 2 and set(cpus) <= set(isolation.os.sched_getaffinity(0))


@pytest.mark.integration
@pytest.mark.skipif(not isolation.isolation_available(), reason="needs unprivileged namespaces")
def test_isolated_processes_have_no_network():
    probe = ("import socket\ntry:\n    socket.create_connection(('1.1.1.1', 443), timeout=2)\n"
             "    print('reachable')\nexcept OSError:\n    print('blocked')\n")
    argv = isolation.wrap_command([sys.executable, "-c", probe], None, True)
    assert subprocess.run(argv, capture_output=True, text=True).stdout.strip() == "blocked"

"""Telemetry parsing (fixtures from real tool output and amd-smi's own source) and gating."""

import json
import shutil

import pytest

from onionrig.rig import telemetry as tm

# Captured from `nvidia-smi` on the dev box's RTX 4060 Ti (idle).
NVIDIA_IDLE = "35, 20.05, 210, 0x0000000000000001"


def amd_document(hotspot=55, power=300, gfx=(1700, 1690), status="UNTHROTTLED"):
    """Shaped like ROCm 7.2 `amd-smi metric -p -c -t --json` (see amdsmi_commands.py)."""
    unit = lambda v, u: {"value": v, "unit": u} if v is not None else "N/A"  # noqa: E731
    clocks = {f"gfx_{i}": {"clk": unit(c, "MHz"), "min_clk": unit(500, "MHz"),
                           "max_clk": unit(1700, "MHz"), "clk_locked": "N/A",
                           "deep_sleep": "DISABLED"} for i, c in enumerate(gfx)}
    clocks["mem_0"] = {"clk": unit(1600, "MHz")}
    return [{"gpu": 0,
             "temperature": {"edge": "N/A", "hotspot": unit(hotspot, "C"), "mem": unit(40, "C")},
             "power": {"socket_power": unit(power, "W"), "throttle_status": status,
                       "gfx_voltage": "N/A"},
             "clock": clocks}]


def test_parse_nvidia_idle_row():
    s = tm.parse_nvidia(NVIDIA_IDLE, 1.0)
    assert (s.temp_c, s.power_w, s.sclk_mhz) == (35.0, 20.05, 210.0)
    assert s.thermal_throttle is False and s.power_capped is False


@pytest.mark.parametrize("mask,thermal,capped", [
    (0x40, True, False), (0x20, True, False), (0x08, True, False),
    (0x04, False, True), (0x05, False, True), (0x44, True, True),
])
def test_power_cap_is_not_thermal_throttling(mask, thermal, capped):
    s = tm.parse_nvidia(f"80, 160.0, 2500, 0x{mask:016x}", 0.0)
    assert (s.thermal_throttle, s.power_capped) == (thermal, capped)


def test_parse_nvidia_unsupported_fields():
    s = tm.parse_nvidia("[N/A], [Not Supported], 1500, [N/A]", 0.0)
    assert s.temp_c is None and s.power_w is None and s.thermal_throttle is None
    with pytest.raises(ValueError):
        tm.parse_nvidia("1, 2", 0.0)


def test_parse_amd():
    s = tm.parse_amd(amd_document(), 0.0, hot_c=100)
    assert (s.temp_c, s.power_w, s.sclk_mhz) == (55.0, 300.0, 1695.0)
    assert s.thermal_throttle is False and s.power_capped is False
    hot = tm.parse_amd(amd_document(hotspot=101, status="THROTTLED"), 0.0, hot_c=100)
    assert hot.thermal_throttle is True and hot.power_capped is True


def test_parse_amd_missing_values():
    s = tm.parse_amd(amd_document(hotspot=None, power=None, gfx=(None,), status="N/A")[0],
                     0.0, hot_c=100)
    assert s.temp_c is None and s.power_w is None and s.sclk_mhz is None
    assert s.thermal_throttle is None and s.power_capped is None
    with pytest.raises(ValueError):
        tm.parse_amd("garbage", 0.0, hot_c=100)


def test_backends_build_the_right_commands():
    seen: list[list[str]] = []

    def nvidia(argv):
        seen.append(argv)
        return NVIDIA_IDLE + "\n"

    def amd(argv):
        seen.append(argv)
        return json.dumps(amd_document())

    assert tm.NvidiaSmi("3", run=nvidia).sample().temp_c == 35.0
    assert tm.AmdSmi("6", run=amd).sample().sclk_mhz == 1695.0
    assert seen[0][:3] == ["nvidia-smi", "-i", "3"]
    assert seen[1][:4] == ["amd-smi", "metric", "-g", "6"]


class FakeTelemetry:
    name = "fake"

    def __init__(self, temps):
        self.temps = list(temps)

    def sample(self):
        temp = self.temps.pop(0) if len(self.temps) > 1 else self.temps[0]
        return tm.Sample(0.0, temp, 100.0, 1500.0, False, False)


def test_wait_until_cool():
    assert tm.wait_until_cool(FakeTelemetry([80, 70, 60, 55]), 60, 30, sleep=lambda _: None) == 60
    assert tm.wait_until_cool(FakeTelemetry([90]), 60, 0.0, sleep=lambda _: None) is None
    assert tm.wait_until_cool(FakeTelemetry([None]), 60, 30, sleep=lambda _: None) is None


def test_sampler_collects_in_background():
    with tm.Sampler(FakeTelemetry([50]), interval=0.01) as sampler:
        import time
        time.sleep(0.15)
    assert len(sampler.samples) >= 5


def test_assess():
    samples = [tm.Sample(0, 60, 200, 1700, False, True), tm.Sample(1, 70, 220, 1600, True, True),
               tm.Sample(2, None, None, None, None, None)]
    report = tm.assess(samples)
    assert report.max_temp_c == 70 and report.min_sclk_mhz == 1600
    assert report.mean_power_w == pytest.approx(210)
    assert report.thermal_throttle_fraction == pytest.approx(0.5)
    assert report.flags == ("thermal-throttle",)
    assert tm.assess([]).flags == ("no-telemetry",)


@pytest.mark.integration
@pytest.mark.skipif(shutil.which("nvidia-smi") is None, reason="no NVIDIA GPU")
def test_detects_the_local_gpu():
    backend = tm.detect({}, 0)
    assert backend is not None and backend.sample().temp_c is not None

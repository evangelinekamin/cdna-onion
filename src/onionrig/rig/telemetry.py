"""GPU telemetry for the thermal gate: temperature, power, clocks, and throttle state.

Two traps shape this module:
- Hitting the power cap under full load is normal. NVIDIA's SwPowerCap bit and
  AMD's THROTTLED status both fire then, so only unambiguous thermal or
  hardware slowdown invalidates a run. Power capping is recorded, not punished.
- nvidia-smi and amd-smi enumerate physical GPUs and ignore *_VISIBLE_DEVICES,
  so the caller passes the physical id the scheduler mapped (see physical_id).
"""

import contextlib
import json
import shutil
import subprocess
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol

from . import isolation

# NVIDIA clocks_event_reasons bits that mean the GPU was forced to slow down.
NV_HW_SLOWDOWN, NV_SW_THERMAL, NV_HW_THERMAL, NV_HW_POWER_BRAKE = 0x8, 0x20, 0x40, 0x80
NV_SW_POWER_CAP = 0x4
NV_FORCED_SLOWDOWN = NV_HW_SLOWDOWN | NV_SW_THERMAL | NV_HW_THERMAL | NV_HW_POWER_BRAKE


@dataclass(frozen=True)
class Sample:
    t: float
    temp_c: float | None
    power_w: float | None
    sclk_mhz: float | None
    thermal_throttle: bool | None   # unambiguous thermal/hardware slowdown
    power_capped: bool | None       # informational


class Telemetry(Protocol):
    name: str

    def sample(self) -> Sample: ...


def _number(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def parse_nvidia(line: str, t: float) -> Sample:
    """One row of: temperature.gpu, power.draw, clocks.sm, clocks_event_reasons.active."""
    fields = [f.strip() for f in line.split(",")]
    if len(fields) != 4:
        raise ValueError(f"unexpected nvidia-smi row: {line!r}")
    temp, power, clock, reasons = fields
    mask = int(reasons, 16) if reasons.startswith("0x") else None
    return Sample(t, _number(temp), _number(power), _number(clock),
                  None if mask is None else bool(mask & NV_FORCED_SLOWDOWN),
                  None if mask is None else bool(mask & NV_SW_POWER_CAP))


def _amd_value(field: object) -> float | None:
    """amd-smi JSON values are {"value": x, "unit": u} or "N/A"."""
    if isinstance(field, dict):
        field = field.get("value")
    return float(field) if isinstance(field, int | float) else None


def parse_amd(document: object, t: float, hot_c: float) -> Sample:
    """`amd-smi metric -p -c -t --json` output (ROCm 7.x schema) for one GPU."""
    gpu = document[0] if isinstance(document, list) and document else document
    if not isinstance(gpu, dict):
        raise ValueError("unexpected amd-smi JSON")
    temps = gpu.get("temperature") or {}
    temp = _amd_value(temps.get("hotspot")) or _amd_value(temps.get("edge"))
    power = gpu.get("power") or {}
    clocks = gpu.get("clock") or {}
    gfx = [_amd_value(v.get("clk")) for k, v in clocks.items()
           if k.startswith("gfx_") and isinstance(v, dict)]
    gfx = [c for c in gfx if c is not None]
    status = power.get("throttle_status")
    return Sample(t, temp, _amd_value(power.get("socket_power")),
                  sum(gfx) / len(gfx) if gfx else None,
                  None if temp is None else temp >= hot_c,
                  None if status not in ("THROTTLED", "UNTHROTTLED") else status == "THROTTLED")


Runner = Callable[[list[str]], str]


def _run(argv: list[str]) -> str:
    return subprocess.run(argv, capture_output=True, text=True, check=True, timeout=10).stdout


@dataclass
class NvidiaSmi:
    gpu: str
    run: Runner = _run
    name: str = "nvidia-smi"

    def sample(self) -> Sample:
        out = self.run(["nvidia-smi", "-i", self.gpu, "--format=csv,noheader,nounits",
                        "--query-gpu=temperature.gpu,power.draw,clocks.sm,"
                        "clocks_event_reasons.active"])
        return parse_nvidia(out.strip().splitlines()[0], time.monotonic())


@dataclass
class AmdSmi:
    gpu: str
    hot_c: float = 100.0  # hotspot temperature treated as thermal throttling
    run: Runner = _run
    name: str = "amd-smi"

    def sample(self) -> Sample:
        out = self.run(["amd-smi", "metric", "-g", self.gpu, "-p", "-c", "-t", "--json"])
        return parse_amd(json.loads(out), time.monotonic(), self.hot_c)


def detect(env: Mapping[str, str], device: int) -> Telemetry | None:
    """Telemetry for exactly the GPU an entry gets as `device`, or None.

    Uses the same visibility mapping as the runner (isolation.select_gpu), since
    nvidia-smi and amd-smi address physical GPUs and ignore *_VISIBLE_DEVICES.
    """
    vendor = isolation.gpu_vendor()
    try:
        _, physical = isolation.select_gpu(env, device, vendor)
    except ValueError:
        return None
    tool = "amd-smi" if vendor == "amd" else "nvidia-smi"
    if shutil.which(tool) is None:
        return None
    backend: Telemetry = AmdSmi(physical) if vendor == "amd" else NvidiaSmi(physical)
    try:
        return backend if backend.sample().temp_c is not None else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


class Sampler:
    """Background sampling at a fixed interval while a run is in flight."""

    def __init__(self, telemetry: Telemetry, interval: float = 1.0):
        self.telemetry, self.interval = telemetry, interval
        self.samples: list[Sample] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self._stop.is_set():
            # A missed sample shows up as a gap in coverage, not as an error.
            with contextlib.suppress(OSError, ValueError, subprocess.SubprocessError):
                self.samples.append(self.telemetry.sample())
            self._stop.wait(self.interval)

    def __enter__(self) -> "Sampler":
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=15)


def wait_until_cool(telemetry: Telemetry, max_temp_c: float, timeout_s: float,
                    poll_s: float = 2.0,
                    sleep: Callable[[float], None] = time.sleep) -> float | None:
    """Block until the GPU is at or below max_temp_c. Returns the last temperature
    seen, or None if it never cooled in time (or temperature is unreadable)."""
    deadline = time.monotonic() + timeout_s
    while True:
        temp = telemetry.sample().temp_c
        if temp is not None and temp <= max_temp_c:
            return temp
        if temp is None or time.monotonic() >= deadline:
            return None
        sleep(poll_s)


@dataclass(frozen=True)
class ThermalReport:
    samples: int
    max_temp_c: float | None
    mean_power_w: float | None
    mean_sclk_mhz: float | None
    min_sclk_mhz: float | None
    thermal_throttle_fraction: float | None
    power_capped_fraction: float | None

    @property
    def flags(self) -> tuple[str, ...]:
        flags = []
        if self.samples == 0:
            flags.append("no-telemetry")
        if self.thermal_throttle_fraction:
            flags.append("thermal-throttle")
        return tuple(flags)


def assess(samples: list[Sample]) -> ThermalReport:
    def values(attr: str) -> list[float]:
        return [v for s in samples if (v := getattr(s, attr)) is not None]

    def fraction(attr: str) -> float | None:
        known = [bool(v) for s in samples if (v := getattr(s, attr)) is not None]
        return sum(known) / len(known) if known else None

    def mean(xs: list[float]) -> float | None:
        return sum(xs) / len(xs) if xs else None

    temps, clocks = values("temp_c"), values("sclk_mhz")
    return ThermalReport(len(samples), max(temps) if temps else None, mean(values("power_w")),
                         mean(clocks), min(clocks) if clocks else None,
                         fraction("thermal_throttle"), fraction("power_capped"))

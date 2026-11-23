"""CPU-OPT-1 monitor equivalence: control and treatment `VramMonitor` must record
the same thing; only the wait after `stop()` may differ.

Two parts, as registered in `CPU-OPT-1_PROTOCOL.md`:

1. Deterministic. The sampler runs synchronously under a scripted clock and a
   scripted `get_gpu_memory`, so timing-derived fields (`max_sample_gap_s`,
   `monitor_overhead_s`) are compared under a fake clock, never real time.
2. Real threads. Once the sampler is in its interruptible wait, the treatment's
   `stop()` returns within 50 ms and the control's waits out the sleep. The
   bound is not asserted while a GPU query is in flight.
"""

from __future__ import annotations

import importlib.util
import sys
import threading
import time
import types
from typing import Any

INTERVAL_S = 0.5
QUERY_S = 0.08  # scripted duration of one GPU query
# One metrics dict per sample; peaks rise so a skipped sample changes them.
METRICS = [
    {"device_vram_used_mib": 3000 + 500 * i, "ollama_vram_used_mib": 2500 + 400 * i}
    for i in range(8)
]
STOP_BOUND_S = 0.05
# Stop request times (from the monitor's start): mid-sleep, mid-query, later.
SCENARIOS = {"mid-sleep": 1.2, "mid-query": 0.04, "late": 2.3}


def load(path: str, name: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses look the module up by name
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def patch(module: types.ModuleType, **names: Any) -> None:
    """Replace names in the telemetry module (it is loaded fresh for each arm)."""
    names["check_gpu_telemetry"] = lambda: "available"
    for key, value in names.items():
        setattr(module, key, value)


class Clock:
    """A scripted clock. A stop request at `stop_at` sets the event when time
    passes it, whether or not the sampler is the one advancing time."""

    def __init__(self, stop_at: float):
        self.now = 1000.0
        self.stop_at = self.now + stop_at
        self.event: FakeEvent | None = None

    def advance(self, dt: float) -> None:
        self.now += dt
        self._arm()

    def _arm(self) -> None:
        if self.event is not None and self.now >= self.stop_at:
            self.event.flag = True

    def time(self) -> float:
        return self.now

    def sleep(self, dt: float) -> None:  # unconditional: nothing interrupts it
        self.advance(dt)


class FakeEvent:
    def __init__(self, clock: Clock):
        self.flag = False
        self.clock = clock
        clock.event = self

    def is_set(self) -> bool:
        return self.flag

    def set(self) -> None:
        self.flag = True

    def clear(self) -> None:
        self.flag = False

    def wait(self, timeout: float | None = None) -> bool:
        if self.flag:
            return True
        wake = self.clock.now + (timeout or 0.0)
        if self.clock.now <= self.clock.stop_at <= wake:  # interrupted
            self.clock.now = self.clock.stop_at
            self.flag = True
            return True
        self.clock.advance(timeout or 0.0)
        return self.flag


def run_scripted(module: types.ModuleType, stop_at: float) -> dict[str, Any]:
    """Run one monitor's sampler to completion under the scripted clock."""
    clock = Clock(stop_at)
    calls: list[float] = []
    last_query_end = [clock.now]

    def get_gpu_memory() -> dict[str, Any]:
        calls.append(clock.now)
        clock.advance(QUERY_S)
        last_query_end[0] = clock.now
        return METRICS[min(len(calls), len(METRICS)) - 1]

    clock_ns = types.SimpleNamespace(
        time=clock.time, perf_counter=clock.time, sleep=clock.sleep
    )
    patch(module, time=clock_ns, get_gpu_memory=get_gpu_memory)
    monitor = module.VramMonitor(INTERVAL_S)
    monitor.stop_event = FakeEvent(clock)
    monitor._sample()  # synchronous: no thread, no real time
    record = monitor.stop()
    # a query already in flight is allowed to finish; only the sleep is "extra"
    extra = clock.now - max(clock.stop_at, last_query_end[0])
    return {"record": record, "exit_after_stop_s": extra}


def scripted_problems(control: types.ModuleType, treatment: types.ModuleType):
    problems = []
    for name, stop_at in SCENARIOS.items():
        c = run_scripted(control, stop_at)
        t = run_scripted(treatment, stop_at)
        if c["record"] != t["record"]:
            problems.append(f"{name}: records differ: {c['record']} vs {t['record']}")
        if c["record"]["monitor_sample_count"] < 1:
            problems.append(f"{name}: the scripted monitor never sampled")
        if t["exit_after_stop_s"] > 1e-9:
            problems.append(f"{name}: treatment kept waiting after the stop request")
    mid = SCENARIOS["mid-sleep"]
    if run_scripted(control, mid)["exit_after_stop_s"] <= 0:
        problems.append("the control did not wait out its sleep: test is not testing")
    return problems


def real_stop_latency(module: types.ModuleType) -> float:
    """Seconds `stop()` takes once the sampler has finished a query and is about
    to wait: real threads, a fast scripted query."""
    sampled = threading.Event()

    def get_gpu_memory() -> dict[str, Any]:
        sampled.set()
        return METRICS[0]

    patch(module, time=time, get_gpu_memory=get_gpu_memory)
    monitor = module.VramMonitor(INTERVAL_S)
    monitor.start()
    if not sampled.wait(5.0):
        raise RuntimeError("the sampler never ran")
    t0 = time.perf_counter()
    monitor.stop()
    return time.perf_counter() - t0


def threaded_problems(control: types.ModuleType, treatment: types.ModuleType):
    problems = []
    fast = real_stop_latency(treatment)
    slow = real_stop_latency(control)
    if fast > STOP_BOUND_S:
        problems.append(f"treatment stop() took {fast:.3f}s (> {STOP_BOUND_S}s)")
    if slow < 5 * STOP_BOUND_S:
        problems.append(f"control stop() took only {slow:.3f}s: test is not testing")
    return problems


def equivalence_problems(control_path: str, treatment_path: str) -> list[str]:
    """Empty when the two implementations are equivalent as registered."""
    control = load(control_path, "opt1_control_telemetry")
    treatment = load(treatment_path, "opt1_treatment_telemetry")
    try:
        return scripted_problems(control, treatment) + threaded_problems(
            control, treatment
        )
    finally:
        sys.modules.pop("opt1_control_telemetry", None)
        sys.modules.pop("opt1_treatment_telemetry", None)
